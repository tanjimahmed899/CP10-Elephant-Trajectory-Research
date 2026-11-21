#!/usr/bin/env python3
"""Rebuild and audit the released BIRDSAI affine-correction pipeline on matched raw windows.
Read-only inputs. Uses existing raw windows, repaired ego-motion, saved corrected NPZ.
Generates exact same-window raw/corrected arrays, future mapping matrices, and baseline metrics.
No learned-model performance claim is made by this script.
"""
from pathlib import Path
import json, numpy as np, pandas as pd
from collections import defaultdict
import argparse
try:
    from numba import njit
except ImportError:
    def njit(f): return f

@njit
def transform_points(pts, mv, W, H):
    L=len(pts); pred=np.empty((L,2),dtype=np.float64); q=np.empty((L,2,3),dtype=np.float64)
    t00=1.;t01=0.;t02=0.;t10=0.;t11=1.;t12=0.
    for k in range(L):
        x=pts[k,0];y=pts[k,1]
        pred[k,0]=(t00*x+t01*y+t02)/W
        pred[k,1]=(t10*x+t11*y+t12)/H
        q[k,0,0]=t00;q[k,0,1]=t01*H/W;q[k,0,2]=t02/W
        q[k,1,0]=t10*W/H;q[k,1,1]=t11;q[k,1,2]=t12/H
        if k<len(mv):
            theta=mv[k,0]*np.pi/180.;sc=mv[k,1];dx=mv[k,2];dy=mv[k,3];cs=np.cos(theta);sn=np.sin(theta)
            m00=sc*cs;m01=-sc*sn;m10=sc*sn;m11=sc*cs
            n00=t00*m00+t01*m10;n01=t00*m01+t01*m11;n02=t00*dx+t01*dy+t02
            n10=t10*m00+t11*m10;n11=t10*m01+t11*m11;n12=t10*dx+t11*dy+t12
            t00=n00;t01=n01;t02=n02;t10=n10;t11=n11;t12=n12
    return pred,q


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--r8-root',type=Path,required=True,help='Path containing the R8_inputs folder')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    r=pd.read_csv(a.raw,dtype={'sequence':str,'segment_id':str}); print('Loaded raw',len(r),flush=True)
    d=a.r8_root/'R8_inputs'/'Main Result'; croot=d/'corrected_trajectory_dataset'; nroot=d/'final_neural_trajectory_dataset'
    mo=pd.read_csv(croot/'repaired_ego_motion.csv',dtype={'sequence':str});
    sizes={k:(int(round(np.nanmedian(g.cx/g.cx_n))),int(round(np.nanmedian(g.cy/g.cy_n)))) for k,g in r.groupby(['split','sequence'])}
    tracks={k:group.set_index('frame').sort_index() for k,group in r.groupby(['split','sequence','segment_id'],sort=False)}
    motions={k:group.set_index('pair').sort_index() for k,group in mo.groupby(['split','sequence'],sort=False)}
    audit=[]; all_metrics=[]; samples_data={}; source_details=[]
    for sec in (2,5):
        task=f'2s_to_{sec}s'; z=np.load(croot/f'corrected_{task}.npz'); corr_hist=z['history']; corr_future=z['future'];
        for partition in ('trainval','test'):
            mz=pd.read_csv(nroot/f'{task}_birdsai_{partition}_metadata.csv',dtype={'sequence':str,'segment_id':str});
            nz=np.load(nroot/f'{task}_birdsai_{partition}_canonical.npz')
            n=len(mz); steps=20+sec*10
            hraw=np.empty((n,20,2),np.float32);fraw=np.empty((n,sec*10,2),np.float32)
            Qfuture=np.empty((n,sec*10,2,3),np.float32)
            Qhist=np.empty((n,20,2,3),np.float32)
            maxdiff=[]; failures=[]
            for i,meta in enumerate(mz.itertuples(index=False)):
                if i % 100 == 0: print("  process",task,partition,i,"/",n,flush=True)
                key=(meta.split,meta.sequence,meta.segment_id)
                start=int(meta.obs_start_frame);finish=int(meta.future_end_frame)
                tr=tracks[key]; mm=motions[(meta.split,meta.sequence)]
                pts=tr.loc[start:finish,['cx','cy']].to_numpy(np.float64)
                raw=tr.loc[start:finish,['cx_n','cy_n']].to_numpy(np.float64)
                mv=mm.loc[start:finish-1,['rotation_deg','scale','dx_pixels','dy_pixels','usable_motion']].to_numpy(np.float64)
                if len(raw)!=(60+sec*30) or (mv[:,4]!=1).any(): raise ValueError(('Missing/invalid',key,start,finish))
                W,H=sizes[(meta.split,meta.sequence)]
                pred,q=transform_points(pts,mv,W,H)
                target=np.concatenate([corr_hist[meta.sample_id,:,:2],corr_future[meta.sample_id,:,:2]],axis=0)
                diff=float(np.max(np.abs(pred-target)));maxdiff.append(diff)
                if diff>1e-5: failures.append((i,diff))
                hraw[i]=raw[:60:3]; fraw[i]=raw[60::3];Qhist[i]=q[:60:3];Qfuture[i]=q[60::3]
                if (not np.allclose(nz['history_abs'][i],target[:60:3],atol=2e-7) or not np.allclose(nz['future_abs'][i],target[60::3],atol=2e-7)):
                    raise ValueError(f'Corrected vs neural mismatch {task} {partition} idx {i}')
                audit.append({'task':task,'partition':partition,'row':i,'source_index':int(meta.source_index),'sample_id':int(meta.sample_id),'source_group':meta.sequence,'track_key':meta.track_key,'start_frame':start,'max_rebuild_discrepancy':diff})
            raw_rel=np.max(np.abs(np.einsum('ntij,ntj->nti',Qfuture,np.concatenate([fraw,np.ones((*fraw.shape[:2],1),np.float32)],axis=2))-nz['future_abs']),axis=(1,2))
            print(f'{task} {partition}: n={n}, max error corrected-rebuild={max(maxdiff):.4g}, max future raw-transform={max(raw_rel):.4g}',flush=True)
            if max(maxdiff)>1e-5 or max(raw_rel)>1e-5:raise ValueError('Geometric invariance fails')
            # Raw baseline forecasts at future camera coordinates; apply target-frame affines for SAME target metric.
            hr=hraw.astype(np.float64); hc=nz['history_abs'].astype(np.float64); yc=nz['future_abs'].astype(np.float64)
            future_n=yc.shape[1]
            times=np.arange(20,dtype=np.float64);tc=times-times.mean();den=(tc**2).sum()
            def CV(x):
                vel=np.sum((x-x.mean(axis=1,keepdims=True))*tc[None,:,None],axis=1)/den
                return x[:,-1:, :]+np.arange(1,future_n+1)[None,:,None]*vel[:,None,:]
            for model in ['Stationary','ConstantVelocity']:
                pred_raw=hr[:,-1:, :].repeat(future_n,axis=1) if model=='Stationary' else CV(hr)
                pred_cor=hc[:,-1:, :].repeat(future_n,axis=1) if model=='Stationary' else CV(hc)
                p=np.einsum('ntij,ntj->nti',Qfuture.astype(float),np.concatenate([pred_raw,np.ones((*pred_raw.shape[:2],1))],axis=-1))
                er=np.linalg.norm(p-yc,axis=-1);ec=np.linalg.norm(pred_cor-yc,axis=-1)
                metric=mz[['source_index','source_group','track_key']].copy()
                metric['task']=task;metric['partition']=partition;metric['model']=model;metric['ADE_raw_projected']=er.mean(axis=1);metric['FDE_raw_projected']=er[:,-1];metric['ADE_corrected']=ec.mean(axis=1);metric['FDE_corrected']=ec[:,-1]
                all_metrics.append(metric)
            data={'history_raw':hraw,'future_raw':fraw,'Q_history':Qhist,'Q_future':Qfuture,'history_corrected':nz['history_abs'],'future_corrected':nz['future_abs'],'train_indices':nz['train_indices'] if partition=='trainval' else np.array([],int),'validation_indices':nz['validation_indices'] if partition=='trainval' else np.array([],int),'train_sample_weights':nz['train_sample_weights'] if partition=='trainval' else np.array([],np.float32)}
            np.savez_compressed(a.out/f'R8_Paired_{task}_{partition}.npz',**data)
            mz.to_csv(a.out/f'R8_Paired_{task}_{partition}_metadata.csv',index=False)
            source_details.append(dict(task=task,partition=partition,n=n,rebuild_max=max(maxdiff),raw_to_corrected_max=float(max(raw_rel)),sources=int(mz.source_group.nunique()),tracks=int(mz.track_key.nunique())))
    pd.DataFrame(audit).to_csv(a.out/'R8_Reconstruction_2792_Sample_Audit.csv',index=False)
    pd.concat(all_metrics,ignore_index=True).to_csv(a.out/'R8_Common_Target_Stationary_CV_Sample_Metrics.csv',index=False)
    mm=pd.concat(all_metrics,ignore_index=True)
    group_summary=mm[mm.partition=='test'].groupby(['task','model','source_group']).agg(n=('source_index','size'),raw_ADE=('ADE_raw_projected','mean'),corrected_ADE=('ADE_corrected','mean'),raw_FDE=('FDE_raw_projected','mean'),corrected_FDE=('FDE_corrected','mean')).reset_index();group_summary.to_csv(a.out/'R8_Common_Target_Test_Group_Metrics.csv',index=False)
    tab=mm[mm.partition=='test'].groupby(['task','model']).agg(n=('source_index','size'),raw_ADE=('ADE_raw_projected','mean'),corrected_ADE=('ADE_corrected','mean'),raw_FDE=('FDE_raw_projected','mean'),corrected_FDE=('FDE_corrected','mean')).reset_index();tab['ADE_pct_reduction']=(1-tab.corrected_ADE/tab.raw_ADE)*100;tab['FDE_pct_reduction']=(1-tab.corrected_FDE/tab.raw_FDE)*100
    tab.to_csv(a.out/'R8_Common_Target_Test_Summary.csv',index=False)
    with open(a.out/'R8_Pairing_Provenance.json','w') as f:json.dump({'method':'Original window-start affine correction; raw future predictions projected to same corrected target with per-future-frame maps; evaluation only','pairs':source_details,'width_height_by_sequence':{'/'.join(k):v for k,v in sizes.items()},'future_transform_use':'For retrospective evaluation only; raw model does not receive future camera transforms','note':'This is an affine-motion model reconstructed from originally saved estimates, not exact 3D camera-pose geometry.'},f,indent=2)
    print('\nTEST SUMMARY\n',tab.to_string(index=False),flush=True)

if __name__=='__main__':main()
