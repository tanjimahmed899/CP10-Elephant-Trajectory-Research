#!/usr/bin/env python3
"""Non-destructive EXP3 array-level verification and EXP4 exploratory group-held-out audit.

Usage: python run_audit.py --exp3-dir <extracted EXP3 ZIP directory> --out <output directory>
Only saved arrays are used; no checkpoint, original video or warning-event validation.
Source data and historical EXP3 outputs are never modified.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def q_conformal(values, coverage=0.90):
    """Finite-sample conformal quantile (higher rank / +1), finite nonnegative scores."""
    a = np.asarray(values, dtype=np.float64)
    if not len(a) or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError('Scores must be finite, nonnegative, and nonempty')
    k = math.ceil((len(a) + 1) * coverage)
    if k > len(a):
        return float('inf')
    return float(np.partition(a, k - 1)[k - 1])


def audit(exp3, output):
    output.mkdir(parents=True, exist_ok=True)
    results = exp3 / 'results'
    assert results.exists(), f'Missing {results}'
    expected_metrics = pd.read_csv(results / 'fresh_holdout_model_metrics.csv')
    summaries = pd.read_csv(results / 'fresh_holdout_dataset_summary.csv')
    recalculated = []
    folds = []
    by_video_rows = []
    integrity = []
    for task in ('2s_to_2s', '2s_to_5s'):
        pred_path = results / f'{task}_exp3_frozen_ensemble_predictions.npz'
        dataset_path = results / f'{task}_fresh_dataset.npz'
        meta_path = results / f'{task}_fresh_metadata.csv'
        for p in (pred_path, dataset_path, meta_path):
            integrity.append(dict(relative_path=p.relative_to(exp3).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        with np.load(pred_path,allow_pickle=False) as z:
            predicted=np.asarray(z['ensemble_prediction'],dtype=np.float64)
            future=np.asarray(z['future_abs'],dtype=np.float64)
            history=np.asarray(z['history_abs'],dtype=np.float64)
        with np.load(dataset_path,allow_pickle=False) as z:
            dataset_future=np.asarray(z['future_abs'],dtype=np.float64)
            pathscale=np.asarray(z['scale'],dtype=np.float64)
        meta=pd.read_csv(meta_path)
        n=len(meta)
        assert predicted.shape==future.shape==(n,20 if task=='2s_to_2s' else 50,2)
        assert history.shape==(n,20,2)
        assert np.all(np.isfinite(predicted)) and np.all(np.isfinite(future)) and np.all(np.isfinite(history))
        assert np.max(abs(dataset_future-future))<1e-5, 'Saved dataset and prediction GT mismatch'
        assert meta.sample_id.is_unique and set(meta.task)=={task}
        assert set(meta.elephant_group)=={'elep1','elep2','elep3'}
        assert meta.source_group.nunique()==6
        assert np.all(np.isfinite(pathscale)) and np.all(pathscale>0)
        err=np.linalg.norm(predicted-future,axis=-1)
        staterr=np.linalg.norm(history[:,-1:, :]-future,axis=-1)
        vals={('Transformer_3Seed_Ensemble','ADE'):float(np.mean(err)),
              ('Transformer_3Seed_Ensemble','FDE'):float(np.mean(err[:,-1])),
              ('Stationary','ADE'):float(np.mean(staterr)),
              ('Stationary','FDE'):float(np.mean(staterr[:,-1]))}
        for (model, metric), v in vals.items():
            e=expected_metrics[(expected_metrics.task==task)&(expected_metrics.model==model)]
            assert len(e)==1
            published=float(e.iloc[0][metric]);delta=v-published
            assert abs(delta)<1e-6, (task,model,metric,delta)
            recalculated.append(dict(task=task,model=model,metric=metric,computed=v,published=published,absolute_difference=abs(delta),status='PASS'))
        for group in sorted(meta.source_group.unique()):
            mask=(meta.source_group==group).to_numpy()
            by_video_rows.append(dict(task=task,video=group,capture_group=meta.loc[mask,'elephant_group'].iloc[0],windows=int(mask.sum()),ensemble_ADE=float(err[mask].mean()),ensemble_FDE=float(err[mask,-1].mean()),stationary_ADE=float(staterr[mask].mean()),stationary_FDE=float(staterr[mask,-1].mean())))
        # Explo­ratory candidate: raw max radial error over all 20/50 future steps.
        # Groups are held out at the elep1/elep2/elep3 capture-group level.
        scores=np.max(err,axis=1)
        hpath=meta.history_path.to_numpy(dtype=float)
        assert np.all(np.isfinite(hpath)) and np.all(hpath>=0)
        for testgroup in sorted(meta.elephant_group.unique()):
            te=(meta.elephant_group==testgroup).to_numpy()
            cal=~te
            threshold=float(np.median(hpath[cal])) # only calibrating groups used to set regime boundary
            calreg=(hpath[cal]<=threshold)
            tereg=(hpath[te]<=threshold)
            assert calreg.sum()>0 and (~calreg).sum()>0
            qpool=q_conformal(scores[cal])
            qlow=q_conformal(scores[cal][calreg])
            qhigh=q_conformal(scores[cal][~calreg])
            for method, radii in (
                ('Pooled raw-error split quantile',np.full(te.sum(),qpool)),
                ('Two-regime Mondrian raw-error quantile',np.where(tereg,qlow,qhigh)),
            ):
                test_cov=(scores[te]<=radii)
                folds.append(dict(task=task,held_out_capture_group=testgroup,method=method,calibration_windows=int(cal.sum()),test_windows=int(te.sum()),calibration_videos=int(meta.loc[cal,'source_group'].nunique()),test_videos=int(meta.loc[te,'source_group'].nunique()),regime_boundary_history_path=threshold,low_regime_calibration_windows=int(calreg.sum()),high_regime_calibration_windows=int((~calreg).sum()),test_low_regime_windows=int(tereg.sum()),q_pooled=qpool,q_low=qlow if 'Mondrian' in method else float('nan'),q_high=qhigh if 'Mondrian' in method else float('nan'),empirical_pathwise_coverage=float(test_cov.mean()),mean_prediction_radius=float(np.mean(radii)),low_test_pathwise_coverage=float(test_cov[tereg].mean()) if tereg.sum() else float('nan'),high_test_pathwise_coverage=float(test_cov[~tereg].mean()) if (~tereg).sum() else float('nan')))
    rec=pd.DataFrame(recalculated)
    rec.to_csv(output/'EXP3_independent_array_recalculation.csv',index=False)
    pd.DataFrame(by_video_rows).to_csv(output/'EXP3_independent_per_video.csv',index=False)
    fold_df=pd.DataFrame(folds)
    fold_df.to_csv(output/'EXP4_EXPLORATORY_leave_capture_group_out.csv',index=False)
    (output/'EXP3_file_hashes.json').write_text(json.dumps(integrity,indent=2)+'\n')
    # Important: group-count is 3; no inferential CI or distribution-free coverage claim.
    for task in ('2s_to_2s','2s_to_5s'):
        print('\n',task)
        print(rec[rec.task==task][['model','metric','computed','published','absolute_difference']].to_string(index=False))
        print(fold_df[fold_df.task==task][['held_out_capture_group','method','test_windows','empirical_pathwise_coverage','mean_prediction_radius']].to_string(index=False))
    print('\nPASS: saved-array reproducibility checks. Exploratory rows are not confirmatory evidence.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--exp3-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    audit(a.exp3_dir,a.out)

if __name__=='__main__':main()
