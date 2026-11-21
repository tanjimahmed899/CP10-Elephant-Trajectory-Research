#!/usr/bin/env python3
"""Pure-function smoke tests; NOT a real tracker experiment."""
import numpy as np,pandas as pd
import stage7d_offline_replay as work
import r13e_metrics_core as metric

x=pd.DataFrame([{'model':'test','sequence':'s','track_id':1,'frame':i} for i in range(215)])
r=work.count_fcast(x)
assert len(r)==1 and int(r.windows_2s.iloc[0])==4 and int(r.windows_5s.iloc[0])==1
x=pd.DataFrame([{'model':'test','sequence':'s','track_id':1,'frame':i} for i in list(range(40))+list(range(50,100))])
r=work.count_fcast(x)
assert len(r)==2 and r.windows_2s.sum()==0
s={'gt_ids':[np.array([0]),np.array([0])], 'tracker_ids':[np.array([0]),np.array([0])],
 'similarity_scores':[np.array([[0.90]]),np.array([[0.90]])],
 'num_tracker_dets':2,'num_gt_dets':2,'num_gt_ids':1,'num_tracker_ids':1,'num_timesteps':2}
assert metric.identity_eval(s)['IDF1']==1.0
assert metric.clear_eval(s)['IDSW']==0
assert np.mean(metric.hota_eval(s)['HOTA'])>0.90
print('PASS: trajectory window rules, gap split, R13E IDF1/HOTA functions (synthetic only)')
