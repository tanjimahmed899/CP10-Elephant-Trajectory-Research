#!/usr/bin/env python3
"""Stage7F: frozen Transformer predictions from *tracker* histories, with GT-only audit/targets.

Strictly read-only. Stage7E labels determine retrospective inclusion; only Stage7D box
centers enter the tracker input arm. Model weights remain frozen, no test tuning.
"""
import argparse, hashlib, io, json, math, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from frozen_transformer import (canonicalize_window, load_transformer, predict,
                                inverse_transform_batch, MODEL_ROOT)

FPS=10
SEEDS=(42,52,62)
SCALE_FLOORS={2:0.06820264416877736,5:0.06732638773715147}
SEQ='0000000353_0000000000'
EXPECTED={2:187,5:124}


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def idcanon(v):
    if pd.isna(v):return ''
    x=float(v)
    if x.is_integer():return str(int(x))
    return str(v).strip()


def zip_csv(z,n,**kwargs):
    if n not in z.namelist():raise ValueError(f'Missing {n} in {z.filename}')
    return pd.read_csv(io.BytesIO(z.read(n)),**kwargs)


def must_same_hash(p,expected,label):
    s=sha(p)
    if s!=expected:raise ValueError(f'{label} SHA mismatch: {s} != {expected}')
    return s


def load_sources(args):
    if not MODEL_ROOT.is_dir(): raise FileNotFoundError(f'Original frozen checkpoint directory missing: {MODEL_ROOT}')
    with zipfile.ZipFile(args.stage7e) as ze,zipfile.ZipFile(args.stage7d) as zd:
        for z in (ze,zd):
            fail=z.testzip()
            if fail:raise ValueError(f'Corrupt ZIP: {z.filename} / {fail}')
        prov=json.loads(ze.read('Stage7E_Provenance.json'))
        if prov['status']!='GT_LINKAGE_AUDIT_COMPLETE' or prov['actual_transformer_predictions_run']:
            raise RuntimeError('Unexpected Stage7E provenance: no verified audit lock')
        must_same_hash(args.stage7d,prov['stage7d_zip_sha256'],'Stage7D reference')
        must_same_hash(args.gt,prov['source_gt_sha256'],'Original GT provenance')
        windows=zip_csv(ze,'Stage7E_All_Tracked_Window_GT_Linkage.csv',dtype={'sequence':str})
        assoc=zip_csv(ze,'Stage7E_Oracle_Frame_Associations.csv',dtype={'sequence':str})
        boxes=zip_csv(zd,'Stage7D_ByteTrack_All_Boxes.csv',dtype={'sequence':str})
        dprov=json.loads(zd.read('Stage7D_Provenance.json'))
        if dprov.get('status')!='TRACKER_METRICS_COMPLETE':raise RuntimeError('Invalid Stage7D status')
    strict=windows.loc[windows.gt_linked_strict_20of20.astype(bool)].copy()
    if set(strict.model.unique())!={'adapted'} or set(strict.sequence.unique())!={SEQ}:
        raise ValueError('Unexpected strict-window model/sequence distribution')
    for sec,expected in EXPECTED.items():
        got=int((strict.horizon==f'{sec}s').sum())
        if got!=expected:raise ValueError(f'Strict {sec}s count {got} != {expected}')
    raw=pd.read_csv(args.gt,dtype={'sequence':str},low_memory=False)
    if not {'split','sequence','object_id','frame','x','y','w','h','cx','cy','cx_n','cy_n'}.issubset(raw.columns):
        raise ValueError('Original raw GT lacks mandatory columns')
    raw=raw.loc[(raw['sequence']==SEQ)&(raw['split'].astype(str).str.lower()=='test_real')].copy()
    if 'interpolated' in raw.columns:
        raw=raw.loc[pd.to_numeric(raw.interpolated,errors='coerce').fillna(0)==0]
    raw['gt_id']=raw.object_id.map(idcanon);raw['frame']=raw.frame.astype(int)
    raw['x']=pd.to_numeric(raw.x);raw['y']=pd.to_numeric(raw.y)
    raw['w']=pd.to_numeric(raw.w);raw['h']=pd.to_numeric(raw.h)
    if raw.duplicated(['gt_id','frame']).any():raise RuntimeError('Duplicate true elephant/frame rows')
    raw['center_x']=raw.x+raw.w/2;raw['center_y']=raw.y+raw.h/2
    goodx=raw.cx_n.abs()>1e-7;goody=raw.cy_n.abs()>1e-7
    W=int(round(np.median(raw.loc[goodx,'cx']/raw.loc[goodx,'cx_n'])))
    H=int(round(np.median(raw.loc[goody,'cy']/raw.loc[goody,'cy_n'])))
    if not (W>100 and H>100):raise ValueError(f'Implausible source size {W}x{H}')
    if max(np.abs(raw.center_x-raw.cx))>1e-5 or max(np.abs(raw.center_y-raw.cy))>1e-5:
        raise ValueError('GT center field inconsistent with GT boxes')
    if max(np.abs(raw.cx_n*W-raw.cx))>1e-4 or max(np.abs(raw.cy_n*H-raw.cy))>1e-4:
        raise ValueError('Cannot confirm GT original pixel dimensions')
    motions=pd.read_csv(args.motion,dtype={'sequence':str})
    motions=motions.loc[(motions.sequence==SEQ)&(motions.split=='test_real')].copy()
    if len(motions)<3000:raise ValueError(f'Motion rows too few {len(motions)}')
    motions['pair']=motions.pair.astype(int)
    if motions.pair.duplicated().any():raise ValueError('Duplicate motion pair indices')
    motion_lookup=motions.set_index('pair')[['rotation_deg','scale','dx_pixels','dy_pixels','usable_motion']].sort_index()
    b=boxes[(boxes.model=='adapted')&(boxes.sequence==SEQ)].copy()
    b['frame']=b.frame.astype(int);b['track_id']=b.track_id.astype(int)
    if b.duplicated(['track_id','frame']).any():raise RuntimeError('Duplicate tracker box per frame')
    b['cx']=(b.x1+b.x2)/2;b['cy']=(b.y1+b.y2)/2
    gt_lookup={(s,int(t)):row for (s,t),row in raw.set_index(['gt_id','frame']).iterrows()}
    box_lookup={(int(t),int(f)):row for (t,f),row in b.set_index(['track_id','frame']).iterrows()}
    links=assoc[(assoc.model=='adapted')&(assoc.sequence==SEQ)&(assoc.oracle_matched.astype(bool))]
    assoc_lookup={(int(row.track_id),int(row.frame)):idcanon(row.oracle_GT_id) for row in links.itertuples(index=False)}
    return strict,gt_lookup,box_lookup,assoc_lookup,motion_lookup,W,H,prov,dprov


def correction_matrices(start,end,mo):
    """Exact R8 window-start affine composition, without future GT in model input."""
    needed=np.arange(start,end,dtype=int)
    if not np.isin(needed,mo.index).all():
        miss=needed[~np.isin(needed,mo.index)];raise RuntimeError(f'Missing affine frame pairs {miss[:4]}')
    m=mo.loc[needed].to_numpy(dtype=float)
    if not np.isfinite(m).all() or not np.all(m[:,4]==1):
        raise RuntimeError('Original R8 camera transform flagged unusable for this window')
    T=np.eye(3,dtype=np.float64);maps={}
    for k,f in enumerate(range(start,end+1)):
        maps[f]=T.copy()
        if f==end:break
        theta,scale,dx,dy,_=m[k];theta=math.radians(theta);c=math.cos(theta);s=math.sin(theta)
        M=np.array([[scale*c,-scale*s,dx],[scale*s,scale*c,dy],[0.,0.,1.]])
        T=T@M
    return maps


def warp(p,T,W,H):
    out=T@np.array([p[0],p[1],1.],dtype=float)
    return np.array([out[0]/W,out[1]/H],dtype=np.float64)


def prep_windows(strict,gt,boxes,assoc,mo,W,H):
    results=[]; arrays={h:{'tracker_hist':[],'oracle_hist':[],'future':[],'tracker_can':[],'oracle_can':[],
                          'tracker_origin':[],'oracle_origin':[],'tracker_angle':[],'oracle_angle':[],
                          'tracker_scale':[],'oracle_scale':[]} for h in (2,5)}
    for row in strict.itertuples(index=False):
        sec=int(row.horizon[:-1]);g=idcanon(row.GT_id_ORACLE_DIAGNOSTIC);tid=int(row.track_id);start=int(row.window_start_30hz)
        hist=np.arange(start,start+60,3,dtype=int)
        fut=np.arange(start+60,start+60+sec*30,3,dtype=int)
        assert len(hist)==20 and len(fut)==sec*10 and int(row.anchor_frame_30hz)==int(hist[-1])
        assert int(row.future_end_30hz)==int(fut[-1])
        mm=correction_matrices(int(start),int(fut[-1]),mo)
        tracker_hist=[];oracle_hist=[];future=[]
        for f in hist:
            key=(tid,int(f))
            if key not in boxes:raise RuntimeError(f'Missing tracker history observation {key}')
            if key not in assoc or assoc[key]!=g:raise RuntimeError(f'GT audit mismatch at {key}: {assoc.get(key)} vs {g}')
            r=boxes[key];qt=gt.get((g,int(f)))
            if qt is None:raise RuntimeError(f'No GT obs at {g} frame {f}')
            tracker_hist.append(warp([r.cx,r.cy],mm[int(f)],W,H))
            oracle_hist.append(warp([qt.center_x,qt.center_y],mm[int(f)],W,H))
        for f in fut:
            q=gt.get((g,int(f)))
            if q is None:raise RuntimeError(f'Missing future GT for {g} frame {f}')
            future.append(warp([q.center_x,q.center_y],mm[int(f)],W,H))
        htr=np.stack(tracker_hist);hgt=np.stack(oracle_hist);yg=np.stack(future)
        # Future target is deliberately NOT passed to canonical input preprocessing.
        placeholder_future=np.zeros_like(yg)
        c_tr=canonicalize_window(htr,placeholder_future,SCALE_FLOORS[sec])
        c_gt=canonicalize_window(hgt,placeholder_future,SCALE_FLOORS[sec])
        arr=arrays[sec];arr['tracker_hist'].append(htr);arr['oracle_hist'].append(hgt);arr['future'].append(yg)
        for arm,c in [('tracker',c_tr),('oracle',c_gt)]:
            arr[arm+'_can'].append(c['history_canonical']);arr[arm+'_origin'].append(c['origin']);
            arr[arm+'_angle'].append(c['heading_angle']);arr[arm+'_scale'].append(c['scale'])
        noise=np.linalg.norm(htr-hgt,axis=-1)
        results.append({'sequence':SEQ,'horizon':row.horizon,'track_id':tid,'GT_id_OFFLINE_AUDIT':g,
                        'start_frame':start,'anchor_frame':int(hist[-1]),'future_end_frame':int(fut[-1]),
                        'history_tracker_vs_GT_mean_dist':float(noise.mean()),
                        'history_tracker_vs_GT_final_dist':float(noise[-1]),
                        'GT_USED_TO_SELECT_STRICT_WINDOWS':True,'GT_FUTURE_USED_FOR_METRICS_ONLY':True})
    for sec,a in arrays.items():
        if len(a['future'])!=EXPECTED[sec]:raise RuntimeError(f'Processed {sec}: {len(a["future"])} != {EXPECTED[sec]}')
        for k,v in a.items():a[k]=np.stack(v) if k not in ('tracker_angle','oracle_angle','tracker_scale','oracle_scale') else np.asarray(v)
    return pd.DataFrame(results),arrays


def summarize(rows):
    d=pd.DataFrame(rows)
    groups=d.groupby(['horizon','input_arm','predictor'],sort=True)
    tab=groups.agg(n=('ADE','size'),ADE_mean=('ADE','mean'),ADE_median=('ADE','median'),
                   FDE_mean=('FDE','mean'),FDE_median=('FDE','median'),source_groups=('sequence','nunique'),
                   GT_identities=('GT_id_OFFLINE_AUDIT','nunique'),tracker_ids=('track_id','nunique')).reset_index()
    return d,tab


def error_table(pred,y,met,arm,predictor):
    if pred.shape!=y.shape:raise RuntimeError(f'prediction/GT shape mismatch {pred.shape} vs {y.shape}')
    err=np.linalg.norm(pred-y,axis=-1)
    z=met.copy();z['input_arm']=arm;z['predictor']=predictor
    z['ADE']=err.mean(axis=1);z['FDE']=err[:,-1];z['max_AE']=err.max(axis=1)
    return z


def run(args):
    out=Path(args.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    strict,gt,boxes,assoc,mo,W,H,sp,dp=load_sources(args)
    print(f'PROVENANCE OK: original GT SHA matches Stage7E, Stage7D ZIP hash matches.')
    print(f'TestReal sequence {SEQ} original frame: {W}x{H}; strict samples n={len(strict)}')
    meta,arrs=prep_windows(strict,gt,boxes,assoc,mo,W,H)
    print('Reconstruction & GT linking complete: ',meta.groupby('horizon').size().to_dict())
    meta.to_csv(out/'Stage7F_Verified_Evaluation_Windows.csv',index=False)
    metric=[];ck_hash={}
    for sec in (2,5):
        task=f'2s_to_{sec}s';a=arrs[sec];m=meta[meta.horizon==f'{sec}s'].reset_index(drop=True)
        y=a['future'];preds={'tracker':[],'oracle':[]}
        for seed in SEEDS:
            ck=MODEL_ROOT/f'{task}_Transformer_seed{seed}.pt'
            ck_hash[ck.name]=sha(ck)
            model=load_transformer(task,seed,sec*10)
            for arm in ('tracker','oracle'):
                pcan=predict(model,a[arm+'_can'],batch_size=256)
                p=inverse_transform_batch(pcan,a[arm+'_origin'],a[arm+'_angle'],a[arm+'_scale'])
                preds[arm].append(p)
                metric.append(error_table(p,y,m,arm,f'Transformer_seed{seed}'))
            print('  evaluated',task,'seed',seed,flush=True)
        saved={'gt_future':y,'tracker_history':a['tracker_hist'],'oracle_GT_history':a['oracle_hist']}
        for arm in ('tracker','oracle'):
            p=np.stack(preds[arm]).mean(axis=0)
            saved[f'{arm}_ensemble_pred_abs']=p
            metric.append(error_table(p,y,m,arm,'Transformer_ensemble_3seed'))
            stationary=np.repeat(a[arm+'_hist'][:,-1:, :],sec*10,axis=1)
            metric.append(error_table(stationary,y,m,arm,'Stationary'))
        np.savez_compressed(out/f'Stage7F_{task}_Predictions.npz',**saved)
    samples,tab=summarize(metric)
    samples.to_csv(out/'Stage7F_Sample_Level_ADE_FDE.csv',index=False)
    tab.to_csv(out/'Stage7F_Frozen_ThreeSeed_Summary.csv',index=False)
    pergt=samples.groupby(['horizon','input_arm','predictor','GT_id_OFFLINE_AUDIT'],as_index=False).agg(
         n_windows=('ADE','size'),ADE_mean=('ADE','mean'),FDE_mean=('FDE','mean'))
    pergt.to_csv(out/'Stage7F_PerElephant_Error_Clustering.csv',index=False)
    prov={'status':'FROZEN_TRACKER_TRANSFORMER_COMPLETE','stage7d_sha256':sha(args.stage7d),
          'stage7e_sha256':sha(args.stage7e),'source_gt_sha256':sha(args.gt),
          'motion_csv_sha256':sha(args.motion),'input_checkpoint_sha256':ck_hash,
          'source_sequence':SEQ,'source_resolution':[W,H],
          'input_tracker':'Stage7D tracked bounding-box centers, unaltered',
          'model':'Original Main Result/transformer_results frozen three-seed Transformer',
          'GT_identity':'Used only to define retrospective strict population and evaluate oracle-history diagnostic',
          'GT_future':'Used only as evaluation target, not inference input',
          'image_motion':'Original R8 repaired 2D affine transforms; window-start reference frame',
          'selection':'Strict 20/20 GT-associated history, full future GT; not online-available oracle gate',
          'no_new_model_training':True,'no_test_tuning':True,'not_independent_holdout':True,
          'limits':['One TestReal sequence only','Overlapping 10Hz windows clustered by GT identity',
                    'This is annotation-conditioned retrospective downstream forecasting, not a deployable full end-to-end score',
                    'Model paths & affine geometry locked to original verified checkpoint/data conventions'],
          'n_horizon':{f'{sec}s':EXPECTED[sec] for sec in (2,5)}}
    (out/'Stage7F_Provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    report='Stage7F real frozen forecasting, conditional on GT-verified TRACKER histories\n'
    report+='No retraining or test-set model selection. GT future not used for model input.\n'
    report+='This is NOT a fully GT-blind pipeline metric: GT eligibility gate applied offline.\n\n'
    report+=tab.to_string(index=False)+'\n'
    (out/'Stage7F_Readout.txt').write_text(report)
    print(report,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage7d',type=Path,required=True)
    p.add_argument('--stage7e',type=Path,required=True)
    p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
