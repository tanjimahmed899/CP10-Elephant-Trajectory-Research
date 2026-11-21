#!/usr/bin/env python3
"""Ground-truth-only numerical consistency test across observation anchor changes.
Uses future 3D depth from annotations ONLY to reproject held-out truth,
never to reproject predicted paths or define a deployable decision.
"""
import importlib.util,sys
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('stage6geometry', HERE/'stage6_fixed_reference_study.py')
a=importlib.util.module_from_spec(sp);sys.modules['stage6geometry']=a;sp.loader.exec_module(a)
records=[];summary=[]
for source,p in a.SOURCES:
 for h in ['2s','5s']:
  m,pr,truth,hist,sc=a.load(source,p,h)
  caches={}
  for i,r in m.iterrows():
    d=a.metadata[(str(r.video),str(r.segment))]
    t=d['tracks'][str(r.track_id)];pos=t['find'][int(r.anchor_frame_index)]
    fs=t['frames'][pos+np.arange(20+len(truth[i]))]
    coords=t['xyz'][pos+np.arange(20+len(truth[i]))]
    Ka,Ra,ta,W,H=d['cam'][d['idx'][int(r.anchor_frame_index)]]
    Kr,Rr,tr,_,_=d['cam'][0]
    worlds=[]
    for f,x in zip(fs,coords):
     _,R,tc,_,_=d['cam'][d['idx'][int(f)]]
     worlds.append((x-tc)@R)
    worlds=np.asarray(worlds)
    depths=(worlds@Ra.T+ta)[:,2]
    gt_anchor=np.concatenate([hist[i],truth[i]])
    uv=gt_anchor*np.array([W,H]);rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(Ka).T
    cam=rays*depths[:,None]; world_o=(cam-ta)@Ra
    projected=a.project(world_o,Kr,Rr,tr,W,H)
    direct=a.project(worlds,Kr,Rr,tr,W,H)
    # true GT numeric reproject equivalence, not model projection
    err=float(np.max(np.abs(projected-direct)))
    caches[(str(r.video),str(r.segment),str(r.track_id),int(r.anchor_frame_index))]=dict(frame_to_point=dict(zip(fs,projected)),replay_err=err)
  prevs=[]
  for (v,seg,tid,anchor),r in caches.items():
    prior=caches.get((v,seg,tid,anchor-10))
    if prior is None: continue
    common=set(r['frame_to_point']) & set(prior['frame_to_point'])
    if not common:continue
    dis=np.array([np.linalg.norm(r['frame_to_point'][frame]-prior['frame_to_point'][frame]) for frame in common])
    prevs.append({'source':source,'horizon':h,'video':v,'segment':seg,'track_id':tid,'anchor':anchor,'n_overlap':len(common),
      'overlap_max_difference':float(dis.max()),'overlap_mean_difference':float(dis.mean())})
  record=pd.DataFrame(prevs);records.append(record)
  print(source,h,'pairs',len(record),'max adjacent GT overlap diff',record.overlap_max_difference.max(),
       'max GT replay proj error',max(c['replay_err'] for c in caches.values()))
  summary.append({'source':source,'horizon':h,'matched_adjacent_pairs':len(record),
                  'median_pair_mean_fixed_GT_displacement':record.overlap_mean_difference.median(),
                  'maximum_fixed_GT_pair_displacement':record.overlap_max_difference.max(),
                  'max_single_sample_GT_reprojection_error':max(c['replay_err'] for c in caches.values())})
pd.concat(records,ignore_index=True).to_csv(HERE/'Stage6_FixedReference_AdjacentAnchor_GT_Consistency.csv',index=False)
pd.DataFrame(summary).to_csv(HERE/'Stage6_FixedReference_AdjacentAnchor_GT_Consistency_Summary.csv',index=False)
