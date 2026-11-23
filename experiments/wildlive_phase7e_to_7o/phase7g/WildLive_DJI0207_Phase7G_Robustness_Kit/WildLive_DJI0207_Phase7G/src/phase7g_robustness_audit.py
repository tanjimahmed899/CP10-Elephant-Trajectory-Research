#!/usr/bin/env python3
"""Phase 7G: frozen-inference descriptive robustness audit, no fitted parameters.

Usage: python phase7g_robustness_audit.py --phase7f-zip PATH --phase7e-zip PATH --output DIR
Only summary-derived outputs are written. Do not treat overlapping windows as independent events.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=('stationary','cv20','cv_ols','transformer_ensemble')
METRICS=('ade_norm','fde_norm')
TASKS=(20,50)
IDS=(0,1,2)

def read_zip_member(z, suffix):
    names=[n for n in z.namelist() if n.endswith(suffix)]
    if len(names)!=1:raise ValueError(f'expected one {suffix}, got {names}')
    return z.read(names[0])

def assert_close(a,b, tol=1e-7, context=''):
    if not np.allclose(a,b,atol=tol,rtol=0,equal_nan=False):
        raise AssertionError(f'{context} mismatch; max discrepancy={np.max(np.abs(np.asarray(a)-np.asarray(b)))}')

def save_records(rows,filename):
    pd.DataFrame(rows).to_csv(filename,index=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase7f-zip',type=Path,required=True)
    ap.add_argument('--phase7e-zip',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    out=args.output; figs=out/'figures'; results=out/'results';figs.mkdir(parents=True,exist_ok=True);results.mkdir(parents=True,exist_ok=True)

    with zipfile.ZipFile(args.phase7f_zip) as fz, zipfile.ZipFile(args.phase7e_zip) as ez:
        if fz.testzip() or ez.testzip():raise ValueError('ZIP CRC failure')
        manifest=json.loads(read_zip_member(fz,'manifest.json'))
        fsum=pd.read_csv(io.BytesIO(read_zip_member(fz,'ALL_MODELS_comparison_phase7f.csv')))
        fdetail=pd.read_csv(io.BytesIO(read_zip_member(fz,'ALL_MODELS_per_window_phase7f.csv')))
        fbdetail=pd.read_csv(io.BytesIO(read_zip_member(fz,'baseline_per_window_phase7f.csv')))
        fcan=pd.read_csv(io.BytesIO(read_zip_member(fz,'canonicalization_statistics.csv')))
        predictions=np.load(io.BytesIO(read_zip_member(fz,'frozen_transformer_predictions_exploratory.npz')),allow_pickle=False)
        e_raw=read_zip_member(ez,'frame_based_windows_and_baselines.npz')
        e_hash=hashlib.sha256(e_raw).hexdigest()
        e=np.load(io.BytesIO(e_raw),allow_pickle=False)

        assert manifest['input_sha256']==e_hash,(manifest['input_sha256'],e_hash)
        assert manifest['status']=='FROZEN_INFERENCE_COMPLETED_EXPLORATORY'
        assert len(manifest['model_checkpoints'])==6
        assert all(v is not None and len(v['sha256'])==64 for v in manifest['model_checkpoints'].values())
        assert len(fsum)==64 and len(fdetail)==4884 and len(fbdetail)==4884 and len(fcan)==6
        assert not fdetail.duplicated(['horizon_frames','track_id','first_observed_frame']).any()
        assert set(fdetail.horizon_frames)==set(TASKS) and set(fdetail.track_id)==set(IDS)
        assert set(fdetail.qa_flagged_window)=={0,1}
        assert fdetail.select_dtypes(include='number').notna().all().all()
        max_error_diff=0.0
        max_7e_pred_diff=0.0
        step_rows=[]
        for h in TASKS:
            for tid in IDS:
                key=f'id{tid}_h{h}'
                mask=(fdetail.horizon_frames==h)&(fdetail.track_id==tid)
                dat=fdetail.loc[mask].sort_values('first_observed_frame')
                n=868-20-h+1
                assert len(dat)==n
                assert np.array_equal(dat.first_observed_frame.to_numpy(),np.arange(n))
                assert np.array_equal(dat.qa_flagged_window.to_numpy().astype(bool),e[f'{key}_qa_flagged'])
                assert np.array_equal(e[f'{key}_start_frame'],predictions[f'{key}_start_frame'])
                assert np.array_equal(e[f'{key}_qa_flagged'],predictions[f'{key}_qa_flagged'])
                hist=e[f'{key}_history']
                future=e[f'{key}_future_gt']
                model_predictions={
                    'stationary':e[f'{key}_stationary_pred'],
                    'cv20':e[f'{key}_cv20_pred'],
                    'transformer_ensemble':predictions[f'{key}_transformer_ensemble_pred'],
                }
                cv=e[f'{key}_cv20_pred']
                assert len(cv)==n
                for mod,pred in model_predictions.items():
                    assert pred.shape==future.shape
                    err=np.linalg.norm(pred.astype('float64')-future,axis=-1)
                    ade=err.mean(axis=1);fde=err[:,-1]
                    source_a=dat[f'{mod}_ade_norm'].to_numpy()
                    source_f=dat[f'{mod}_fde_norm'].to_numpy()
                    max_error_diff=max(max_error_diff,float(np.max(np.abs(ade-source_a))),float(np.max(np.abs(fde-source_f))))
                    assert_close(ade,source_a,context=f'{key} {mod} ADE')
                    assert_close(fde,source_f,context=f'{key} {mod} FDE')
                    for t in range(h):
                        step_rows.append({'horizon_frames':h,'track_id':tid,'forecast_step':t+1,
                            'model':mod,'mean_l2_norm':float(err[:,t].mean())})
                # Phase 7E original stationary/CV predictions are verified against frozen metrics.
        # Validate 64 pooled and track-specific summary rows.
        max_summary_discrepancy=0.0
        for rec in fsum.to_dict(orient='records'):
            group=fdetail[fdetail.horizon_frames==rec['horizon_frames']]
            if rec['track_id']!='pooled':group=group[group.track_id==int(rec['track_id'])]
            if rec['subset']=='QA_unflagged':group=group[group.qa_flagged_window==0]
            assert len(group)==rec['windows']
            for metr in METRICS:
                val=group[f'{rec["model"]}_{metr}'].mean()
                max_summary_discrepancy=max(max_summary_discrepancy,abs(float(val)-float(rec[metr])))
                assert_close(val,rec[metr],tol=1e-12,context=f'{rec} summary {metr}')

        # Check baseline detail duplicates exact.
        merged=fdetail.merge(fbdetail,on=['horizon_frames','track_id','first_observed_frame','last_observed_frame','last_target_frame','qa_flagged_window'],suffixes=('_7f','_baseline'),validate='one_to_one')
        assert len(merged)==4884
        for mod in MODELS[:3]:
            for metr in METRICS:
                assert_close(merged[f'{mod}_{metr}_7f'],merged[f'{mod}_{metr}_baseline'],tol=1e-12,context=f'{mod} baseline')

    pooled=[];per_track=[];qa=[];temporal=[];offsets=[]
    for h in TASKS:
        d=fdetail[fdetail.horizon_frames==h].copy()
        step=20+h
        for name,g in (('all',d),('QA_unflagged',d[d.qa_flagged_window==0]),('QA_flagged',d[d.qa_flagged_window==1])):
            for base in ('stationary','cv20','cv_ols'):
                delta_ade=(g.transformer_ensemble_ade_norm-g[f'{base}_ade_norm']).to_numpy()
                delta_fde=(g.transformer_ensemble_fde_norm-g[f'{base}_fde_norm']).to_numpy()
                t_ade=float(g.transformer_ensemble_ade_norm.mean());b_ade=float(g[f'{base}_ade_norm'].mean())
                pooled.append({'horizon_frames':h,'subset':name,'baseline':base,'windows':len(g),'T_ADE':t_ade,'baseline_ADE':b_ade,
                    'T_FDE':float(g.transformer_ensemble_fde_norm.mean()),'baseline_FDE':float(g[f'{base}_fde_norm'].mean()),
                    'delta_ADE_T_minus_baseline':float(delta_ade.mean()),'delta_FDE_T_minus_baseline':float(delta_fde.mean()),
                    'relative_ADE_improvement_percent':float(100*(b_ade-t_ade)/b_ade),
                    'fraction_windows_T_lower_ADE':float(np.mean(delta_ade<0))})
        for tid,g in d.groupby('track_id'):
            s=g.transformer_ensemble_ade_norm.mean();b=g.stationary_ade_norm.mean()
            per_track.append({'horizon_frames':h,'track_id':int(tid),'windows':len(g),'T_ADE':float(s),'stationary_ADE':float(b),'ADE_delta_T_minus_S':float(s-b),'T_win_fraction':float(np.mean(g.transformer_ensemble_ade_norm<g.stationary_ade_norm)),
                'T_FDE':float(g.transformer_ensemble_fde_norm.mean()),'stationary_FDE':float(g.stationary_fde_norm.mean()),
                'FDE_delta_T_minus_S':float(g.transformer_ensemble_fde_norm.mean()-g.stationary_fde_norm.mean())})
        # Three equal partitions in start-frame index (not independent temporal groups).
        maxstart=int(d.first_observed_frame.max())
        for i in range(3):
            # integer boundaries avoid floating labels and ensure no gaps.
            lo=(maxstart+1)*i//3
            hi=(maxstart+1)*(i+1)//3-1
            g=d[(d.first_observed_frame>=lo)&(d.first_observed_frame<=hi)]
            assert len(g)>0
            temporal.append({'horizon_frames':h,'segment':i+1,'start_frame_min':lo,'start_frame_max':hi,'windows':len(g),
                'T_ADE':float(g.transformer_ensemble_ade_norm.mean()),'stationary_ADE':float(g.stationary_ade_norm.mean()),
                'ADE_delta_T_minus_S':float((g.transformer_ensemble_ade_norm-g.stationary_ade_norm).mean()),
                'T_win_fraction':float((g.transformer_ensemble_ade_norm<g.stationary_ade_norm).mean())})
        # Each phase offset yields windows separated by obs+future length within elephant ID.
        for offset in range(step):
            g=d[d.first_observed_frame%step==offset]
            assert len(g)>0
            offsets.append({'horizon_frames':h,'stride_frames':step,'offset':offset,'windows':len(g),
                'T_ADE':float(g.transformer_ensemble_ade_norm.mean()),'stationary_ADE':float(g.stationary_ade_norm.mean()),
                'ADE_delta_T_minus_S':float((g.transformer_ensemble_ade_norm-g.stationary_ade_norm).mean()),
                'T_better_than_S':int(g.transformer_ensemble_ade_norm.mean()<g.stationary_ade_norm.mean())})

    save_records(pooled,results/'pooled_and_QA_model_comparisons.csv')
    save_records(per_track,results/'per_elephant_ade_fde.csv')
    save_records(qa,results/'qa_detail.csv') if qa else None
    save_records(temporal,results/'temporal_thirds_sensitivity.csv')
    save_records(offsets,results/'nonoverlapping_phase_offset_sensitivity.csv')
    save_records(step_rows,results/'per_forecast_step_errors.csv')

    # Plots: matplotlib default palette (illustrative/descriptive only).
    pa=pd.DataFrame(pooled);tr=pd.DataFrame(per_track);te=pd.DataFrame(temporal);ph=pd.DataFrame(step_rows);off=pd.DataFrame(offsets)
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    xs=np.array([0,1])
    w=.35
    station=[pa[(pa.horizon_frames==h)&(pa.subset=='all')&(pa.baseline=='stationary')].iloc[0].baseline_ADE for h in TASKS]
    trans=[pa[(pa.horizon_frames==h)&(pa.subset=='all')&(pa.baseline=='stationary')].iloc[0].T_ADE for h in TASKS]
    ax.bar(xs-w/2,station,w,label='Stationary')
    ax.bar(xs+w/2,trans,w,label='Frozen Transformer ensemble')
    ax.set_xticks(xs,['20-frame future','50-frame future']);ax.set_ylabel('Normalized image-plane ADE (lower is better)');ax.set_title('DJI_0207 | Frozen transfer: pooled ADE');ax.legend();fig.tight_layout()
    fig.savefig(figs/'01_pooled_ADE_comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    for h in TASKS:
        g=tr[tr.horizon_frames==h].sort_values('track_id')
        ax.plot(g.track_id,g.ADE_delta_T_minus_S,marker='o',label=f'{h}-frame future')
    ax.axhline(0,color='black',linewidth=1)
    ax.set_xticks([0,1,2]);ax.set_xlabel('Annotated elephant ID');ax.set_ylabel('Transformer minus Stationary ADE');ax.set_title('Per-elephant direction of model advantage');ax.legend();fig.tight_layout()
    fig.savefig(figs/'02_per_elephant_delta_ADE.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    for h in TASKS:
        g=te[te.horizon_frames==h].sort_values('segment')
        ax.plot(g.segment,g.ADE_delta_T_minus_S,marker='o',label=f'{h}-frame future')
    ax.axhline(0,color='black',linewidth=1);ax.set_xticks([1,2,3],['Early third','Middle third','Late third'])
    ax.set_ylabel('Transformer minus Stationary ADE');ax.set_title('Temporal nonstationarity (descriptive only)');ax.legend();fig.tight_layout()
    fig.savefig(figs/'03_temporal_thirds.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    for h in TASKS:
        for m in ('stationary','transformer_ensemble'):
            g=ph[(ph.horizon_frames==h)&(ph.model==m)].groupby('forecast_step',as_index=False).mean(numeric_only=True)
            ax.plot(g.forecast_step,g.mean_l2_norm,label=f'{m} ({h} frames)',linewidth=1.8)
    ax.set_xlabel('Forecast step (frame index; not seconds)');ax.set_ylabel('Mean per-step L2 normalized error');ax.set_title('Error accumulation across frame horizons');ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(figs/'04_forecast_step_error_curves.png',dpi=170);plt.close(fig)

    audit={'phase':'7G','data':'WildLive DJI_0207; one capture group; unknown frame timing; uncorrected image-plane positions',
        'input_phase7f_zip_sha256':hashlib.sha256(args.phase7f_zip.read_bytes()).hexdigest(),
        'input_phase7e_zip_sha256':hashlib.sha256(args.phase7e_zip.read_bytes()).hexdigest(),
        'phase7e_npz_sha256_matches_phase7f_manifest':e_hash==manifest['input_sha256'],
        'phase7e_npz_sha256':e_hash,'frozen_checkpoint_hashes_recorded_not_independently_rehashed':True,
        'ZIP_CRC': 'PASS','phase7f_summary_rows_recomputed':64,'input_window_rows':len(fdetail),
        'max_summary_absolute_difference':max_summary_discrepancy,
        'max_prediction_npz_vs_csv_absolute_difference':max_error_diff,
        'QA_flagged_window_counts': {str(h):int(((fdetail.horizon_frames==h)&(fdetail.qa_flagged_window==1)).sum()) for h in TASKS},
        'nonoverlap_phase_offsets':{str(h): {'offsets':20+h,
            'T_better_fraction_across_offsets':float(off[off.horizon_frames==h].T_better_than_S.mean()),
            'min_delta_ADE':float(off[off.horizon_frames==h].ADE_delta_T_minus_S.min()),
            'max_delta_ADE':float(off[off.horizon_frames==h].ADE_delta_T_minus_S.max())} for h in TASKS},
        'invalid_claims':['seconds-based ADE','physical displacement error','independent recording inference','valid population CI from overlapping windows','image-plane motion equals elephant ground movement'],
        'statistical_inference':'Not performed: one capture group, strongly overlapping windows, correlated tracks and camera motion.'}
    (results/'audit_verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2));print('\nPOOLED VS STATIONARY');print(pa[(pa.subset=='all')&(pa.baseline=='stationary')].to_string(index=False))
    print('\nPER TRACK');print(tr.to_string(index=False))

if __name__=='__main__':main()
