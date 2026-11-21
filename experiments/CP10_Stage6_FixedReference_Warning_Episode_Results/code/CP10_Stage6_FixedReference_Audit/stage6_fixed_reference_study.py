#!/usr/bin/env python3
"""Retrospective, read-only fixed-reference geometry and exploratory geofence simulation.

Projection of predicted 2-D paths into reference image requires unknown future depth.
Uses only last observed 3-D center depth, holds it constant. Ground-truth uses
future 3-D tracking annotations ONLY FOR SCORING, never to build predictions.
No surveyed world coordinates; no independent validation; no field lead time.
"""
import io, zipfile, json, hashlib, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from itertools import product
BASE=Path(__file__).resolve().parent; OUT=BASE; OUT.mkdir(exist_ok=True)
S=zipfile.ZipFile(BASE/'inputs/CP10_Stage6_StaticFence_Metadata.zip')
SOURCES=[('EXP3',BASE/'inputs/EXP3_Output_Review.zip'),('Stage5',BASE/'inputs/CP10_Stage5_Frozen_Results_Review.zip')]
BOUND=[('x',0,.25),('x',0,.50),('x',0,.75),('y',1,.25),('y',1,.50),('y',1,.75)]
Q={'2s':2.158768,'5s':1.975130}
metadata={};names=S.namelist()
for name in names:
    if name.endswith('/cameras.json'):
        parts=name.split('/');group,video,segment=parts[-4:-1]
        cams=json.loads(S.read(name))['cameras']; data=json.loads(S.read(name.replace('cameras.json','tracking_summary.json')))['tracks']
        idx={int(c['frame_index']):i for i,c in enumerate(cams)}
        rec=[]
        for c in cams:
            R=np.array(c['extrinsic'],float)[:,:3];t=np.array(c['extrinsic'],float)[:,3]
            rec.append((np.array(c['intrinsic'],float),R,t,int(c['image_width']),int(c['image_height'])))
        tracks={}
        for tid,tr in data.items():
            fs=np.asarray(tr['frames'],dtype=int); xyz=np.asarray(tr['centers'],dtype=float)
            if len(fs)!=len(xyz):continue
            tracks[str(tid)]={'frames':fs,'xyz':xyz,'find':{int(f):k for k,f in enumerate(fs)}}
        metadata[(video,segment)]={'cam':rec,'tracks':tracks,'idx':idx,'group':group}

def project(X,K,R,t,W,H):
    c=X @ R.T + t
    y=c @ K.T
    return y[...,:2]/y[...,2:3]/np.array([W,H])

def pred_to_ref(p_anchor, z_last,K_anchor,R_anchor,t_anchor,K_ref,R_ref,t_ref,W,H):
    # image normalized -> pixel rays, backproject constant last OBSERVED depth
    uv=np.asarray(p_anchor,float)*np.array([W,H])
    ray=np.column_stack([uv,np.ones(len(uv))])@np.linalg.inv(K_anchor).T
    cam=ray*z_last
    world=(cam-t_anchor)@R_anchor
    return project(world,K_ref,R_ref,t_ref,W,H)

def per_sample(source,h,meta,pred,truth,hist,scale):
    recs=[]; projected=[]
    for i,r in meta.iterrows():
        key=(str(r.video),str(r.segment));data=metadata.get(key)
        if not data:recs.append({'source':source,'horizon':h,'sample_index':i,'status':'no_segment'});continue
        t=data['tracks'].get(str(r.track_id))
        if not t:recs.append({'source':source,'horizon':h,'sample_index':i,'status':'no_track'});continue
        pos=t['find'].get(int(r.anchor_frame_index))
        if pos is None:recs.append({'source':source,'horizon':h,'sample_index':i,'status':'no_anchor'});continue
        T=pred.shape[1];indices=pos+np.arange(20+T)
        if indices.max()>=len(t['frames']):recs.append({'source':source,'horizon':h,'sample_index':i,'status':'short_track'});continue
        fs=t['frames'][indices]; xx=t['xyz'][indices]
        if not np.all(np.diff(fs)==1):recs.append({'source':source,'horizon':h,'sample_index':i,'status':'nonconsecutive_track'});continue
        if any(int(f) not in data['idx'] for f in fs):recs.append({'source':source,'horizon':h,'sample_index':i,'status':'missing_camera'});continue
        ic=data['idx'][int(r.anchor_frame_index)]; ic_ref=0
        Ka,Ra,ta,W,H=data['cam'][ic];Kr,Rr,tr,Wr,Hr=data['cam'][ic_ref]
        if (W,H)!=(Wr,Hr):recs.append({'source':source,'horizon':h,'sample_index':i,'status':'different_resolution'});continue
        # repro saved GT: 3-D centers expressed in their OWN frame camera coordinates
        xy=[]
        for f,xc in zip(fs,xx):
            curr=data['cam'][data['idx'][int(f)]]; Kc,Rc,tc,_,_=curr
            X=(xc-tc)@Rc
            xy.append(project(X,Ka,Ra,ta,W,H))
        xy=np.array(xy)
        source_gt=np.concatenate([hist[i],truth[i]])
        diff=np.max(np.abs(xy-source_gt))
        # Anchor projected 3-D observed at last obs frame; use its depth not future
        cobs_idx=int(fs[19]); _,Ro,to,_,_=data['cam'][data['idx'][cobs_idx]]
        X_obs=(xx[19]-to)@Ro
        z_last=(X_obs@Ra.T+ta)[2]
        if z_last<=0:recs.append({'source':source,'horizon':h,'sample_index':i,'status':'invalid_last_depth'});continue
        # Fixed-ref project of world-registered full GT 3D trajectory
        allX=[]
        for f,x in zip(fs,xx):
            _,Rc,tc,_,_=data['cam'][data['idx'][int(f)]]
            allX.append((x-tc)@Rc)
        allX=np.array(allX)
        gtref=project(allX,Kr,Rr,tr,W,H)
        # Project model pred to reference using last observed depth only
        predref=pred_to_ref(pred[i],z_last,Ka,Ra,ta,Kr,Rr,tr,W,H)
        # Identical last-observed-depth conversion on true future anchor positions
        truth_const=pred_to_ref(truth[i],z_last,Ka,Ra,ta,Kr,Rr,tr,W,H)
        mapping_error=np.linalg.norm(truth_const-gtref[20:],axis=1)
        # Estimate uncertainty scale's max one-axis derivative using local +/- epsilon
        # Additive isotropic q*scale in anchor space; derivative w.r.t reference x/y on each predicted point
        eps=0.001;dpx=pred[i].copy();dpx[:,0]+=eps;dpy=pred[i].copy();dpy[:,1]+=eps
        dx=(pred_to_ref(dpx,z_last,Ka,Ra,ta,Kr,Rr,tr,W,H)-predref)/eps
        dy=(pred_to_ref(dpy,z_last,Ka,Ra,ta,Kr,Rr,tr,W,H)-predref)/eps
        # norm of row of 2x2 Jacobian per future step; units reference axis / anchor isotropic radius
        rad=np.sqrt(dx**2+dy**2)*Q[h]*scale[i]
        projected.append({'source':source,'horizon':h,'sample_index':i,'meta':r,'gtref':gtref,'predref':predref,'radius':rad,
                          'center_world':allX,'future_frame_indices':fs[20:], 'obs_frame_index':int(fs[19]),'mapping_error':mapping_error})
        recs.append({'source':source,'horizon':h,'sample_index':i,'status':'matched',
           'group':data['group'],'video':r.video,'segment':r.segment,'track_id':r.track_id,'role':r.get('analysis_role','EXP3_six_video'),
           'anchor_frame_index':int(r.anchor_frame_index),'obs_frame_index':int(fs[19]),
           'gt_replay_max_abs':float(diff),'gt_ref_mapping_error_mean':float(mapping_error.mean()),
           'gt_ref_mapping_error_max':float(mapping_error.max()),'gt_ref_mapping_error_endpoint':float(mapping_error[-1]),
           'observed_last_depth':float(z_last),'zfuture_ratio_max':float(np.max(np.abs((allX[20:]@Ra.T+ta)[:,2]/z_last-1))),
           'fixed_ref_gt_range_x':float(np.ptp(gtref[:,0])),'fixed_ref_gt_range_y':float(np.ptp(gtref[:,1]))})
    return pd.DataFrame(recs),projected

def load(source,p,h):
    with zipfile.ZipFile(p) as z:
        if source=='EXP3':
            pref='results\\2s_to_'+h
            a=np.load(io.BytesIO(z.read(pref+'_exp3_frozen_ensemble_predictions.npz')))
            pred=a['ensemble_prediction'].astype(float);gt=a['future_abs'].astype(float); hist=a['history_abs'].astype(float)
            meta=pd.read_csv(io.BytesIO(z.read(pref+'_fresh_metadata.csv')))
            scale=np.load(io.BytesIO(z.read(pref+'_fresh_dataset.npz')))['scale'].astype(float)
        else:
            pref='2s_to_'+h
            a=np.load(io.BytesIO(z.read(pref+'_FROZEN_PREDICTIONS_FOR_CALIBRATION.npz')))
            pred=a['ensemble_prediction_abs'].astype(float);gt=a['future_abs'].astype(float);hist=a['history_abs'].astype(float)
            scale=a['scale'].astype(float)
            meta=pd.read_csv(io.BytesIO(z.read(pref+'_fresh_metadata.csv')))
    return meta,pred,gt,hist,scale

if __name__=='__main__':
  recs=[]; projected=[]
  for source,p in SOURCES:
    for h in ['2s','5s']:
        m,pr,gt,hi,sc=load(source,p,h)
        r,points=per_sample(source,h,m,pr,gt,hi,sc)
        recs.append(r);projected.extend(points)
        print(source,h,'matched',int(r.status.eq('matched').sum()),'/',len(r), 'max replay error',r.gt_replay_max_abs.max(),
              'median mapping mean',r.gt_ref_mapping_error_mean.median(), flush=True)
  allr=pd.concat(recs,ignore_index=True);allr.to_csv(OUT/'Stage6_FixedReference_Geometry_Replay.csv',index=False)
  valid=allr[allr.status=='matched']
  valid.groupby(['source','horizon'],sort=False).agg(N=('sample_index','size'),max_saved_gt_replay_error=('gt_replay_max_abs','max'),
      median_mean_constant_depth_projection_error=('gt_ref_mapping_error_mean','median'),p90_mean_constant_depth_projection_error=('gt_ref_mapping_error_mean',lambda x:x.quantile(.9)),
      median_max_constant_depth_projection_error=('gt_ref_mapping_error_max','median'),p90_max_constant_depth_projection_error=('gt_ref_mapping_error_max',lambda x:x.quantile(.9)),
      p90_endpoint_constant_depth_projection_error=('gt_ref_mapping_error_endpoint',lambda x:x.quantile(.9))).to_csv(OUT/'Stage6_FixedReference_Projection_Error_Summary.csv')
  # GT crossing potential across original fixed-reference footage in each segment, 6 fences.
  fulltruth=[]
  for (video,segment),dat in metadata.items():
    cams=dat['cam'];Kr,Rr,tr,W,H=cams[0];
    for tid,t in dat['tracks'].items():
      world=[];f=[]
      for fi,cent in zip(t['frames'],t['xyz']):
        if int(fi) not in dat['idx']:continue
        _,R,tc,_,_=cams[dat['idx'][int(fi)]]
        world.append((cent-tc)@R);f.append(fi)
      if not world:continue
      xy=project(np.asarray(world),Kr,Rr,tr,W,H)
      for axis,j,b in BOUND:
        crossed=np.flatnonzero((xy[:-1,j]-b)*(xy[1:,j]-b)<0)
        crossing=[(int(f[k]),int(f[k+1])) for k in crossed if f[k+1]-f[k]==1]
        fulltruth.append({'video':video,'segment':segment,'track_id':tid,'group':dat['group'],'axis':axis,'boundary':b,'true_crossing_event_count':len(crossing),
                          'available_frames':len(f),'first_frame':int(f[0]),'last_frame':int(f[-1]),'crossings':str(crossing)})
  fulltruth=pd.DataFrame(fulltruth);fulltruth.to_csv(OUT/'Stage6_FixedReference_AllTracks_TrueCrossing_Events.csv',index=False)
  print('True crossings count GT all videos',fulltruth.true_crossing_event_count.sum(),'by-group',fulltruth.groupby('group').true_crossing_event_count.sum().to_dict())
  # fixed-reference per forecast boundary cases (current-side orientation) and window labels.
  rows=[]
  for p in projected:
    gt=p['gtref'];prd=p['predref'];rad=p['radius'];r=p['meta'];
    for ax,j,b in BOUND:
       if abs(gt[19,j]-b)<1e-10:continue
       s=1 if gt[19,j]<b else -1
       tr=(gt[20:,j]-b)*s;pp=(prd[:,j]-b)*s
       label=bool(tr.max()>=0);nom=bool(pp.max()>=0)
       # A discrete actual crossing event in GT happens if prior frame and future frame straddle fence.
       # When current observation past the fence, event is in predicted horizon only if true crosses that side.
       crossing=np.flatnonzero(tr>=0)
       rows.append({'source':p['source'],'horizon':p['horizon'],'sample_index':p['sample_index'],
         'role':r.get('analysis_role','EXP3_six_video'),'group':r.elephant_group,'video':r.video,'segment':r.segment,
         'track_id':int(r.track_id),'track_key':r.track_key,'anchor':int(r.anchor_frame_index),
         'observation_end':p['obs_frame_index'],'axis':ax,'boundary':b,'direction':'high' if s==1 else 'low',
         'true_window_cross':label,'nominal':nom,'risk_score':float(pp.max()),
         'intersects_frozen_q90':bool(np.any(pp+rad[:,j]>=0)),
         'first_future_cross_frame':int(p['future_frame_indices'][crossing[0]]) if len(crossing) else None,
         'first_future_cross_seconds':float((crossing[0]+1)/10) if len(crossing) else None,
         'projection_error_mean':float(p['mapping_error'].mean())})
  df=pd.DataFrame(rows);df.to_csv(OUT/'Stage6_FixedReference_Projected_Warning_Cases.csv',index=False)
  print('crossing cases',df.groupby(['source','horizon']).true_window_cross.agg(['sum','count']))
  # N2/N3 gap-aware policies: original rule uses inter previous regardless of gap; use exact 10 for safer.
  keys=['source','horizon','video','segment','track_id','axis','boundary','direction'];df=df.sort_values(keys+['anchor']).reset_index(drop=True)
  gb=df.groupby(keys,sort=False);prev=df.anchor-gb.anchor.shift(1);prev2=df.anchor-gb.anchor.shift(2)
  i0=df.intersects_frozen_q90.to_numpy(bool);i1=df.groupby(keys,sort=False).intersects_frozen_q90.shift(1,fill_value=False).to_numpy(bool);i2=df.groupby(keys,sort=False).intersects_frozen_q90.shift(2,fill_value=False).to_numpy(bool)
  n0=df.nominal.to_numpy(bool)
  df['fixed_N1']=n0|i0;df['fixed_N2_gapaware']=n0|(i0&i1&(prev==10).to_numpy());df['fixed_N3_gapaware']=n0|(i0&i1&i2&(prev==10).to_numpy()&(prev2==20).to_numpy())
  df.to_csv(OUT/'Stage6_FixedReference_Projected_Warning_Cases.csv',index=False)
  measures=[]
  for (source,h,role),g in df.groupby(['source','horizon','role']):
    for pol in ['nominal','fixed_N1','fixed_N2_gapaware','fixed_N3_gapaware']:
      t=g.true_window_cross.to_numpy(bool);a=g[pol].to_numpy(bool)
      TP=int(np.sum(t&a));FN=int(np.sum(t&~a));FP=int(np.sum(~t&a));TN=int(np.sum(~t&~a))
      measures.append({'source':source,'horizon':h,'role':role,'policy':pol,'TP':TP,'FN':FN,'FP':FP,'TN':TN,
                       'recall':TP/(TP+FN) if TP+FN else None,'FWR':FP/(FP+TN) if FP+TN else None,
                       'total_cases':len(g)})
  result=pd.DataFrame(measures);result.to_csv(OUT/'Stage6_FixedReference_Window_Simulation_Summary.csv',index=False)
  print(result.to_string(index=False))
  print('outputs',OUT)
