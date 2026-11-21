#!/usr/bin/env python3
"""Account for EVERY locked source, including zero-window videos. Never impute missing ADE/FDE as 0.
Produces honest descriptive source/elephant-group summaries; no unsupported confidence intervals.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);a=ap.parse_args()
 lock=pd.read_csv(a.lock); assert len(lock)==6 and lock.repo_path.is_unique
 expected=set(lock.basename.str.removesuffix('.zip'))
 path=a.results/'fresh_holdout_per_source.csv'
 if not path.exists():raise FileNotFoundError(f'No actual frozen-model source output: {path}')
 actual=pd.read_csv(path)
 required={'task','source_group','samples','ADE','FDE'}
 if not required<=set(actual):raise ValueError(f'Missing columns {required-set(actual)}')
 assert len(actual)==len(actual[['task','source_group']].drop_duplicates())
 if not set(actual.source_group)<=expected:raise RuntimeError('Unexpected source video in output')
 if not set(actual.task)<=set(['2s_to_2s','2s_to_5s']):raise RuntimeError('Unexpected task')
 rows=[]
 for horizon in ('2s_to_2s','2s_to_5s'):
  for row in lock.itertuples(index=False):
   source=row.basename.removesuffix('.zip'); x=actual[(actual.task==horizon)&(actual.source_group==source)]
   if len(x)==1:
    z=x.iloc[0]; n=int(z.samples)
    if n<0:raise ValueError('Negative window count')
    ade=float(z.ADE) if n>0 else float('nan');fde=float(z.FDE) if n>0 else float('nan')
    if n>0 and (not np.isfinite([ade,fde]).all() or min(ade,fde)<0):raise ValueError(f'Invalid per-video errors {source}')
    label='FORECASTABLE' if n>0 else 'ZERO_WINDOWS'
   else:n=0;ade=float('nan');fde=float('nan');label='SOURCE_MISSING_FROM_EVALUATOR'
   rows.append({'task':horizon,'video':source,'elephant_group':row.elephant_group,'n_windows':n,'ADE':ade,'FDE':fde,'status':label})
 table=pd.DataFrame(rows);a.results.mkdir(parents=True,exist_ok=True);table.to_csv(a.results/'EXP3_ALL_LOCKED_VIDEO_ACCOUNTING.csv',index=False)
 agg=[]
 for horizon,g in table.groupby('task',sort=False):
  use=g[g.n_windows>0]
  for metric in ('ADE','FDE'):
   agg.append({'task':horizon,'metric':metric,'n_locked_videos':6,'n_forecastable_videos':len(use),'n_zero_or_missing':6-len(use),'n_capture_groups':3,'n_forecastable_groups':use.elephant_group.nunique(),'n_windows':int(use.n_windows.sum()),'window_pooled':float(np.average(use[metric],weights=use.n_windows)) if len(use) else float('nan'),'video_macro_conditional':float(use[metric].mean()) if len(use) else float('nan'),'status':'ALL_SIX_HAVE_WINDOWS' if len(use)==6 else 'INCOMPLETE_WINDOW_COVERAGE_NO_UNCONDITIONAL_CLAIM'})
 pd.DataFrame(agg).to_csv(a.results/'EXP3_aggregate_metrics_by_source.csv',index=False)
 all_ok=all(r['status']=='FORECASTABLE' for r in rows)
 report={'all_six_videos_with_windows_per_horizon':all_ok,'new_video_evaluation_ran':True,'method':'frozen_forecast conditional on forecastable annotated trajectory windows','n_video_horizon_cells':12,'n_available_video_horizon_cells':sum(r['status']=='FORECASTABLE' for r in rows),'caveats':['Same capture groups as original holdout','Source-macro conditional on positive sample count','No original Step7H warning evaluation performed by this postprocessor','Any zero-window source explicitly retained as missing, never zero ADE']}
 (a.results/'EXP3_SOURCE_COMPLETENESS_STATUS.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Processed',len(rows),'source×horizon cells; forecastable:',report['n_available_video_horizon_cells'])
 print('WARNING: No new Step7H warning metrics produced. Inspect all outputs before publication.')
if __name__=='__main__':main()
