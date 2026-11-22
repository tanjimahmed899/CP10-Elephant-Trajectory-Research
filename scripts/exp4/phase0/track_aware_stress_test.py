#!/usr/bin/env python3
"""Exploratory TRACK-unit conformal feasibility stress test (NOT new independent validation).
Primary score: per-track maximum over all windows AND all future steps.
Regime feature: history_path of the *first available window per track*, a past-only input at the start of that track.
Calibrate only on two capture groups and test the third; repeated exposure of these groups makes this exploratory.
"""
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd

def q90(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0 or not np.all(np.isfinite(vals)):
        raise ValueError('Invalid conformal scores')
    rank = math.ceil((len(vals) + 1) * 0.9)
    return float('inf') if rank > len(vals) else float(np.partition(vals, rank - 1)[rank - 1])

def run(base, out):
    rows=[]
    for task in ['2s_to_2s','2s_to_5s']:
        with np.load(base/f'{task}_exp3_frozen_ensemble_predictions.npz',allow_pickle=False) as z:
            err=np.linalg.norm(z['ensemble_prediction'].astype(float)-z['future_abs'].astype(float),axis=2)
        m=pd.read_csv(base/f'{task}_fresh_metadata.csv')
        m['window_max_error']=np.max(err,axis=1)
        m['history_path']=pd.to_numeric(m['history_path'],errors='raise')
        # Original task is 10 Hz and consecutive predictions overlap.
        tr=m.groupby('track_key',sort=False).agg(capture_group=('elephant_group','first'),video=('source_group','first'),track_score=('window_max_error','max'),windows=('sample_id','count')).reset_index()
        first=m.sort_values(['track_key','anchor_frame_index','sample_id']).groupby('track_key',sort=False).first().reset_index()
        tr=tr.merge(first[['track_key','history_path']],on='track_key',validate='one_to_one')
        assert len(tr)==m.track_key.nunique() and len(tr) in (59,50)
        for testgroup in sorted(tr.capture_group.unique()):
            cal=tr[tr.capture_group!=testgroup]
            test=tr[tr.capture_group==testgroup]
            threshold=float(cal.history_path.median())
            low=cal[cal.history_path<=threshold]
            high=cal[cal.history_path>threshold]
            test_low=test.history_path<=threshold
            for label, radius in [('Pooled, track-unit',np.repeat(q90(cal.track_score),len(test))),('Mondrian, track-unit',np.where(test_low,q90(low.track_score),q90(high.track_score)))]:
                cov=np.asarray(test.track_score<=radius,dtype=bool)
                rows.append(dict(task=task,held_out_group=testgroup,method=label,calibration_tracks=len(cal),test_tracks=len(test),calibration_low_tracks=len(low),calibration_high_tracks=len(high),test_low_tracks=int(test_low.sum()),test_high_tracks=int((~test_low).sum()),empirical_whole_track_coverage=float(cov.mean()),mean_radius=float(np.mean(radius)),finite_radius=bool(np.all(np.isfinite(radius))),median_first_window_history_path_threshold=threshold))
    df=pd.DataFrame(rows);df.to_csv(out,index=False)
    print(df[['task','held_out_group','method','calibration_tracks','test_tracks','empirical_whole_track_coverage','mean_radius','finite_radius']].to_string(index=False))
    print('Exploratory; all three capture groups previously inspected; not 90% guarantee under shift.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exp3-results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.exp3_results,args.out)
