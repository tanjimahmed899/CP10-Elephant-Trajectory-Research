#!/usr/bin/env python3
"""Check whether successive prediction windows share a fixed 2-D coordinate frame.
Future GT at physically overlapping 10Hz indices would agree if reference were fixed.
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from stage6_warning_audit import load_from_zip

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--exp3',type=Path,default=Path('/mnt/data/EXP3_Output_Review.zip'));ap.add_argument('--stage5',type=Path,default=Path('/mnt/data/CP10_Stage5_Frozen_Results_Review.zip'));ap.add_argument('--out',type=Path,default=Path('/mnt/data/CP10_Stage6_Warning_Evidence'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for source,z in [('EXP3',args.exp3),('Stage5',args.stage5)]:
  for h in ['2s','5s']:
   m,p,t,his,scale,role=load_from_zip(z,h)
   for key,g in m.groupby('track_key'):
    o=g.sort_values('anchor_frame_index');indices=o.index.to_numpy()
    for i,j in zip(indices[:-1],indices[1:]):
     step=int(m.anchor_frame_index[j]-m.anchor_frame_index[i])
     if not 0<step<len(t[i]) or step!=10:continue
     diff=np.linalg.norm(t[i,step:]-t[j,:-step],axis=1)
     rows.append({'source':source,'horizon':h,'video':m.video[i], 'track_key':key,
       'role':role[i], 'anchor_a':int(m.anchor_frame_index[i]),'anchor_b':int(m.anchor_frame_index[j]),
       'overlap_steps':len(diff),'mean_same_time_coordinate_displacement':float(diff.mean()),
       'max_same_time_coordinate_displacement':float(diff.max()),
       'median_scale':float(np.median([scale[i],scale[j]]))})
 df=pd.DataFrame(rows);df.to_csv(args.out/'Stage6_Adjacent_Window_Coordinate_Shift_Audit.csv',index=False)
 agg=[]
 for (source,h,role),g in df.groupby(['source','horizon','role']):
  a=g.mean_same_time_coordinate_displacement.to_numpy()
  agg.append({'source':source,'horizon':h,'role':role,'n_adjacent_pairs':len(g),
   'median_pair_mean_displacement':float(np.median(a)),'p90_pair_mean_displacement':float(np.quantile(a,.9)),
   'maximum_pair_displacement':float(g.max_same_time_coordinate_displacement.max()),
   'pct_pairs_gt_0p001':float((a>.001).mean()*100)})
 summary=pd.DataFrame(agg);summary.to_csv(args.out/'Stage6_Coordinate_Shift_Summary.csv',index=False)
 print(summary.to_string(index=False))
if __name__=='__main__':main()
