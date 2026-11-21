#!/usr/bin/env python3
"""R8 matched validation, retrospective BIRDSAI TrainReal/TestReal trajectories.

Run after restoring original per-frame raw tracks and camera-corrected arrays.
Usage:
 python exp1_r8_matched_validation.py \
  --raw-points /path/trajectory_dataset/cleaned_track_points.csv \
  --raw-windows-dir /path/trajectory_dataset \
  --corrected-dir /path/corrected_trajectory_dataset \
  --output-dir /path/R8_matched_NEW --boot 2000

Research restrictions:
- Only frozen CORE TestReal windows. No selection/tuning on TestReal.
- Both forecasts evaluated against EXACTLY THE SAME corrected future target.
- Raw baseline's predicted displacement is reanchored at corrected last observation;
  this is a last-point translation, NOT an oracle future camera-motion transform.
- Raw-native ADE/FDE is provided separately as a diagnostic, NOT a causal estimate.
- Future ego-motion-corrected truth is used retrospectively only for scoring.
- Track/source clustered CIs are descriptive with 5-6 sources.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['split','sequence','object_id','segment_id','obs_start_frame','obs_end_frame','future_end_frame','horizon_sec']
TASKS = {'2s_to_2s':(20,1053),'2s_to_5s':(50,826)}
MODELS = ('Stationary','ConstantVelocity')
BOUNDS = (0.25,0.50,0.75)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def read(p):
    return pd.read_csv(p,dtype={'split':str,'sequence':str,'segment_id':str})

def core_table(raw, corr, task, expected):
    if len(raw)!=len(corr): raise ValueError(f'{task}: different number of rows')
    match=raw.merge(corr,on=KEY,how='outer',indicator=True,validate='one_to_one')
    if (match['_merge']!='both').any():raise ValueError(f'{task}: mismatched window keys')
    m=corr.loc[(corr.split=='test_real')&(corr.accepted==1)&(corr.quality_tier.astype(str).str.lower()=='core')].copy()
    if len(m)!=expected: raise ValueError(f'{task}: expected {expected} core TestReal, found {len(m)}')
    if (m.sample_id<0).any() or m.sample_id.duplicated().any():raise ValueError('Invalid accepted sample_ids')
    return m.sort_values('sample_id').reset_index(drop=True)

def load_track_lookup(p):
    df=read(p)
    required={'split','sequence','segment_id','frame','cx_n','cy_n'}
    if not required.issubset(df.columns):raise ValueError('Missing track columns: '+repr(required-set(df.columns)))
    result={}
    for k,g in df.groupby(['split','sequence','segment_id'],sort=False):
        g=g.sort_values('frame')
        if g.frame.duplicated().any():raise ValueError(f'Duplicate frames: {k}')
        result[k]=g.set_index('frame')[['cx_n','cy_n']]
    return result

def raw_sample(tracks, row, horizon):
    k=(str(row['split']),str(row['sequence']),str(row['segment_id']))
    if k not in tracks:raise ValueError(f'Missing raw track: {k}')
    a=int(row['obs_start_frame']); b=int(row['future_end_frame'])
    need=np.arange(a,b+1,dtype=int)
    t=tracks[k].reindex(need)
    if t.isna().any().any():raise ValueError(f'Missing raw frames: {k}, {a}..{b}')
    pos=t.to_numpy(float)
    if pos.shape!=(60+3*horizon,2):raise ValueError(f'Raw shape {pos.shape} for {k}')
    return pos[:60:3], pos[60::3]

def corr_sample(npz, ix, horizon):
    hs=np.asarray(npz['history'][ix,:,:2],float)
    fs=np.asarray(npz['future'][ix,:,:2],float)
    if len(hs)==60:hs=hs[::3]
    if len(fs)==3*horizon:fs=fs[::3]
    if hs.shape!=(20,2) or fs.shape!=(horizon,2):raise ValueError(f'Unexpected corrected shape: {hs.shape},{fs.shape}')
    if not np.isfinite(hs).all() or not np.isfinite(fs).all():raise ValueError('NaNs or inf in corrected arrays')
    return hs,fs

def predict(x,h,model):
    if model=='Stationary':return np.repeat(x[-1:,:],h,axis=0)
    t=np.arange(len(x),dtype=float)/10.0
    velocity=((t-t.mean())[:,None]*(x-x.mean(0))).sum(axis=0)/np.sum((t-t.mean())**2)
    return x[-1][None,:]+(np.arange(1,h+1)/10.0)[:,None]*velocity[None,:]

def error(p,y):
    e=np.linalg.norm(p-y,axis=1)
    return float(e.mean()),float(e[-1])

def warning_instances(row,model,pred_raw,pred_corr,truth,current):
    result=[]
    for ax,axis in [('x',0),('y',1)]:
        for b in BOUNDS:
            if abs(current[axis]-b)<=1e-12:continue
            sign=1.0 if current[axis]<b else -1.0
            truth_margin=sign*(truth[:,axis]-b)
            positive=int(np.max(truth_margin)>=0.)
            item={'task':row['task'],'model':model,'sequence':row['sequence'], 'track_key':row['track_key'],
                  'sample_id':row['sample_id'],'axis':ax,'boundary':b,'truth':positive}
            for name,p in [('raw_aligned',pred_raw),('corrected',pred_corr)]:
                item[f'{name}_alert']=int(np.max(sign*(p[:,axis]-b))>=0.)
            result.append(item)
    return result

def aggregate_forecasts(samples):
    rows=[]
    for (task,model),g in samples.groupby(['task','model'],sort=True):
        line={'task':task,'model':model,'windows':len(g),'tracklets':g.track_key.nunique(), 'sources':g.sequence.nunique()}
        for measure in ('ADE','FDE'):
            for name in ('raw_native','corrected_native','raw_aligned','corrected_common'):
                line[f'{name}_{measure}']=float(g[f'{name}_{measure}'].mean())
            delta=g[f'corrected_common_{measure}']-g[f'raw_aligned_{measure}']
            line[f'paired_delta_{measure}_corrected_minus_raw']=float(delta.mean())
            line[f'track_macro_paired_delta_{measure}']=float(g.groupby('track_key').apply(lambda v:(v[f'corrected_common_{measure}']-v[f'raw_aligned_{measure}']).mean(),include_groups=False).mean())
            line[f'source_macro_paired_delta_{measure}']=float(g.groupby('sequence').apply(lambda v:(v[f'corrected_common_{measure}']-v[f'raw_aligned_{measure}']).mean(),include_groups=False).mean())
        rows.append(line)
    return pd.DataFrame(rows)

def calc_alert(g,kind):
    p=g[f'{kind}_alert'].to_numpy(int);y=g.truth.to_numpy(int)
    tp=int(np.sum((p==1)&(y==1)));fp=int(np.sum((p==1)&(y==0)))
    fn=int(np.sum((p==0)&(y==1)));tn=int(np.sum((p==0)&(y==0)))
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,
            'precision':tp/(tp+fp) if tp+fp else np.nan,
            'recall':tp/(tp+fn) if tp+fn else np.nan,
            'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
            'FWR':fp/(fp+tn) if fp+tn else np.nan}

def aggregate_warnings(instances):
    rows=[]
    for (task,model),g in instances.groupby(['task','model'],sort=True):
        line={'task':task,'model':model,'instances':len(g),'positive':int(g.truth.sum())}
        for kind in ('raw_aligned','corrected'):
            line.update({f'{kind}_{m}':v for m,v in calc_alert(g,kind).items()})
        line['paired_F1_corrected_minus_raw']=line['corrected_F1']-line['raw_aligned_F1']
        line['paired_FWR_corrected_minus_raw']=line['corrected_FWR']-line['raw_aligned_FWR']
        rows.append(line)
    return pd.DataFrame(rows)

def bootstrap_ci(samples,group_col,values,nboot,seed):
    rng=np.random.default_rng(seed)
    clusters=[a.to_numpy(float) for _,a in samples.groupby(group_col)[values]]
    n=len(clusters)
    if n<2:return (np.nan,np.nan)
    boot=np.empty(nboot)
    for b in range(nboot):
        selected=rng.integers(0,n,size=n)
        y=np.concatenate([clusters[i] for i in selected])
        boot[b]=float(np.mean(y))
    return np.quantile(boot,[.025,.975]).tolist()

def make_ci(samples,boot):
    results=[]
    for (task,model),g in samples.groupby(['task','model'],sort=True):
        for metric in ('ADE','FDE'):
            delta=g[f'corrected_common_{metric}']-g[f'raw_aligned_{metric}']
            t=g[['sequence','track_key']].copy();t['delta']=delta
            for unit,field in [('tracklet','track_key'),('source','sequence')]:
                lo,hi=bootstrap_ci(t,field,'delta',boot,42+len(g)+(metric=='FDE'))
                results.append({'task':task,'model':model,'metric':metric,'cluster_unit':unit,
                                'clusters':int(g[field].nunique()),'n_boot':boot,'mean_paired_delta':float(delta.mean()),
                                'ci95_low':lo,'ci95_high':hi, 'negative_delta_favors':'corrected'})
    return pd.DataFrame(results)

def audit(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    audits=[];cores={}
    for task,(horizon,expected) in TASKS.items():
        raw_p=Path(args.raw_windows_dir)/f'windows_{task}.csv'
        corr_p=Path(args.corrected_dir)/f'corrected_windows_{task}.csv'
        if not raw_p.is_file() or not corr_p.is_file():raise FileNotFoundError(f'Need metadata: {raw_p}, {corr_p}')
        raw=read(raw_p);corr=read(corr_p)
        core=core_table(raw,corr,task,expected);cores[task]=core
        audits.append({'task':task,'raw_windows':len(raw),'corrected_metadata_windows':len(corr),
                       'matched_identity_windows':len(raw),'matched_core_test_windows':len(core),
                       'core_test_tracklets':core.segment_id.nunique(),'core_test_sources':core.sequence.nunique(),
                       'valid_sample_ids':bool(core.sample_id.is_unique and (core.sample_id>=0).all()),
                       'arrays_present':(Path(args.corrected_dir)/f'corrected_{task}.npz').exists(),
                       'raw_points_present':Path(args.raw_points).exists()})
    aud=pd.DataFrame(audits);aud.to_csv(out/'R8_pair_registry_audit.csv',index=False)
    return cores,aud

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-points',required=True)
    ap.add_argument('--raw-windows-dir',required=True)
    ap.add_argument('--corrected-dir',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--boot',type=int,default=2000)
    ap.add_argument('--audit-only',action='store_true')
    args=ap.parse_args()
    if args.boot<100:ap.error('--boot must be >=100')
    cores,aud=audit(args)
    file_paths=[Path(args.raw_points)]
    for task in TASKS:
        file_paths.extend([Path(args.raw_windows_dir)/f'windows_{task}.csv',
                           Path(args.corrected_dir)/f'corrected_windows_{task}.csv',
                           Path(args.corrected_dir)/f'corrected_{task}.npz'])
    missing=[str(f) for f in file_paths if not f.is_file()]
    status='AUDIT_ONLY' if args.audit_only else 'NOT_RUN_MISSING_INPUTS' if missing else 'COMPLETE'
    manifest={'status':status,'protocol':'R8 common-target translation-aligned comparison v1',
              'raw_frame_rate':30,'evaluation_frame_rate':10,'bootstrap_iterations':args.boot,
              'truth':'camera-corrected future trajectory, held fixed for both models',
              'raw_prediction_mapping':'p_raw_aligned(t)=p_raw_native(t)-raw_last+corrected_last',
              'known_limitations':['Corrected labels are not world ground truth',
                                   'Input CORE selection may depend on future correction feasibility',
                                   'Only 5-6 independent TestReal sources',
                                   'Translation alignment does not model future camera pose or rotation/scale',
                                   'No live detector/tracker inputs'],
              'missing_required_files':missing,
              'input_sha256':{str(f):digest(f) for f in file_paths if f.is_file()}}
    output=Path(args.output_dir)
    if args.audit_only or missing:
        (output/'R8_protocol_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
        print(aud.to_string(index=False));print('Status:',status)
        if missing: print('Missing files:',*missing,sep='\n  ')
        if missing and not args.audit_only: sys.exit(2)
        return
    tracks=load_track_lookup(Path(args.raw_points))
    rows=[];warnings=[]
    for task,(horizon,_) in TASKS.items():
        z=np.load(Path(args.corrected_dir)/f'corrected_{task}.npz',allow_pickle=False)
        if not {'history','future'}.issubset(z.files):raise ValueError(f'{task} NPZ must have history/future keys')
        m=cores[task]
        if int(m.sample_id.max())>=len(z['history']) or int(m.sample_id.max())>=len(z['future']):
            raise ValueError(f'{task} sample_id outside corrected array bounds')
        for _,r in m.iterrows():
            rh,rf=raw_sample(tracks,r,horizon)
            ch,cf=corr_sample(z,int(r.sample_id),horizon)
            shift=ch[-1]-rh[-1]
            for model in MODELS:
                rp=predict(rh,horizon,model)
                cp=predict(ch,horizon,model)
                aligned=rp+shift[None,:]
                rn=error(rp,rf);cn=error(cp,cf);rc=error(aligned,cf)
                if model=='Stationary' and not np.allclose(aligned,cp,rtol=0,atol=1e-10):
                    raise RuntimeError('Stationary alignment invariant failed')
                common=error(cp,cf)
                record={'task':task,'model':model,'sample_id':int(r.sample_id),'sequence':str(r.sequence),
                        'track_key':str(r.segment_id),'obs_start_frame':int(r.obs_start_frame),
                        'raw_native_ADE':rn[0],'raw_native_FDE':rn[1],
                        'corrected_native_ADE':cn[0],'corrected_native_FDE':cn[1],
                        'raw_aligned_ADE':rc[0],'raw_aligned_FDE':rc[1],
                        'corrected_common_ADE':common[0],'corrected_common_FDE':common[1],
                        'delta_ADE_corrected_minus_raw':common[0]-rc[0],
                        'delta_FDE_corrected_minus_raw':common[1]-rc[1],
                        'last_point_alignment_shift':float(np.linalg.norm(shift))}
                rows.append(record)
                warnings.extend(warning_instances(record,model,aligned,cp,cf,ch[-1]))
        z.close()
    samp=pd.DataFrame(rows);instance=pd.DataFrame(warnings)
    samp.to_csv(output/'R8_same_sample_predictions_metrics.csv',index=False)
    instance.to_csv(output/'R8_same_truth_geofence_instances.csv',index=False)
    aggregate_forecasts(samp).to_csv(output/'R8_matched_metrics.csv',index=False)
    aggregate_warnings(instance).to_csv(output/'R8_matched_warning_metrics.csv',index=False)
    make_ci(samp,args.boot).to_csv(output/'R8_paired_cluster_ci.csv',index=False)
    samp.groupby(['task','model','sequence'],as_index=False).agg(
        n_windows=('sample_id','size'),
        raw_aligned_ade=('raw_aligned_ADE','mean'),corrected_ade=('corrected_common_ADE','mean'),
        delta_ade=('delta_ADE_corrected_minus_raw','mean'),
        raw_aligned_fde=('raw_aligned_FDE','mean'),corrected_fde=('corrected_common_FDE','mean'),
        delta_fde=('delta_FDE_corrected_minus_raw','mean')).to_csv(output/'R8_by_source.csv',index=False)
    manifest['status']='COMPLETE'; manifest['n_sample_model_rows']=len(samp);manifest['n_boundary_instances']=len(instance)
    (output/'R8_protocol_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('COMPLETE:',len(samp),'model-window rows;',len(instance),'warning instances')

if __name__=='__main__':main()
