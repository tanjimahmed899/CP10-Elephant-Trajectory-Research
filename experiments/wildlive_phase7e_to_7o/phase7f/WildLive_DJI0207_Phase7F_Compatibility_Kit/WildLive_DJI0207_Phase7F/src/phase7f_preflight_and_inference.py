#!/usr/bin/env python3
"""Phase 7F: Frozen Transformer compatibility and optional inference.

User research pipeline, exploratory only. Inputs are Phase 7E original annotation-centre
arrays, not camera-stabilized physical trajectories. No validated FPS. Never retrain,
fit, re-calibrate, or change checkpoint parameters.

Source-preserved: EXP3 fixed canonicalization, frozen model architecture, task scale
floors. Checkpoints are required for numerical Transformer inference and are never
synthesized by this program.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np

TASKS={20:{'name':'2s_to_2s','scale_floor':0.06820264416877736},
       50:{'name':'2s_to_5s','scale_floor':0.06732638773715147}}
SEEDS=(42,52,62)
IDS=(0,1,2)
OBS=20
HEADING_FRAMES=10
EPS=1e-10


def sha256_file(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): d.update(chunk)
    return d.hexdigest()


def read_data(path):
    tracks={}
    with np.load(path,allow_pickle=False) as z:
        for horizon in TASKS:
            for tid in IDS:
                stem=f'id{tid}_h{horizon}'
                sub={name:z[f'{stem}_{name}'].copy() for name in (
                    'history','future_gt','stationary_pred','cv20_pred','start_frame','qa_flagged')}
                n=868-OBS-horizon+1
                expected={'history':(n,OBS,2),'future_gt':(n,horizon,2),
                          'stationary_pred':(n,horizon,2),'cv20_pred':(n,horizon,2),
                          'start_frame':(n,),'qa_flagged':(n,)}
                for k,shape in expected.items():
                    assert sub[k].shape==shape,(stem,k,sub[k].shape,shape)
                assert np.array_equal(sub['start_frame'],np.arange(n)),stem
                assert sub['qa_flagged'].dtype==bool,stem
                for k in ('history','future_gt','stationary_pred','cv20_pred'):
                    assert np.isfinite(sub[k]).all(),(stem,k)
                assert np.allclose(sub['stationary_pred'],sub['history'][:,-1:, :],atol=1e-12),stem
                cv=stationary(sub['history'],horizon)+np.arange(1,horizon+1)[None,:,None]*(
                    (sub['history'][:,-1,:]-sub['history'][:,0,:])/19.0)[:,None,:]
                assert np.allclose(sub['cv20_pred'],cv,atol=1e-12),stem
                tracks[(horizon,tid)]=sub
    return tracks


def stationary(x,horizon):
    return np.repeat(x[:,-1:, :],horizon,axis=1)


def cv_ols(x,horizon):
    # EXP3 historical baseline, different from Phase 7E endpoint CV20.
    t=np.arange(x.shape[1],dtype=np.float64)
    tc=t-t.mean()
    vx=np.sum((x-x.mean(axis=1,keepdims=True))*tc[None,:,None],axis=1)/np.sum(tc*tc)
    return x[:,-1:, :] + np.arange(1,horizon+1,dtype=np.float64)[None,:,None]*vx[:,None,:]


def heading_from_history(x):
    """Exact EXP3 last-10-position least-squares heading; x [N,20,2]."""
    recent=x[:,-min(HEADING_FRAMES,x.shape[1]):,:]
    t=np.arange(recent.shape[1],dtype=np.float64)
    tc=t-t.mean()
    vel=np.sum((recent-recent.mean(axis=1,keepdims=True))*tc[None,:,None],axis=1)/np.sum(tc**2)
    norm=np.hypot(vel[:,0],vel[:,1]); ang=np.arctan2(vel[:,1],vel[:,0]);ang[norm<=EPS]=0.0
    return ang


def canonicalize(x,floor):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(20,2): raise ValueError('history must be [N,20,2]')
    origin=x[:,-1,:].copy(); angle=heading_from_history(x)
    path=np.linalg.norm(np.diff(x,axis=1),axis=-1).sum(axis=1)
    scale=np.maximum(path,float(floor))
    c=np.cos(angle);s=np.sin(angle)
    # R = [[cos,sin],[-sin,cos]], matching original EXP3 code exactly.
    R=np.stack((np.stack((c,s),axis=1),np.stack((-s,c),axis=1)),axis=1)
    can=np.einsum('nfi,nij->nfj',(x-origin[:,None,:]),R.transpose(0,2,1))/scale[:,None,None]
    return {'history_canonical':can.astype('float32'),'origin':origin,
            'heading':angle,'scale':scale,'path':path,'R':R}


def reconstruct(pred_can,meta):
    return np.einsum('nfi,nij->nfj',np.asarray(pred_can,dtype=np.float64),meta['R'])*meta['scale'][:,None,None]+meta['origin'][:,None,:]


def ade_fde(p,y):
    err=np.linalg.norm(np.asarray(p)-y,axis=-1)
    return err.mean(axis=1),err[:,-1]


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def model_factory(future_length,params):
    import torch
    import torch.nn as nn
    d_model=int(params.get('d_model',64));nhead=int(params.get('nhead',4))
    layers=int(params.get('num_layers',2));ff=int(params.get('dim_feedforward',128))
    dropout=float(params.get('dropout',0.10))
    if not(1<=d_model<=2048 and 1<=layers<=32 and 1<=ff<=8192 and 1<=nhead<=64 and
           d_model % nhead==0 and 0<=dropout<1):
        raise ValueError('Invalid checkpoint architecture metadata')
    class PositionalEncoding(nn.Module):
        def __init__(self):
            super().__init__()
            pe=torch.zeros(32,d_model)
            pos=torch.arange(32,dtype=torch.float32).unsqueeze(1)
            div=torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*(-math.log(10000.0)/d_model))
            pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div)
            self.register_buffer('pe',pe.unsqueeze(0))
        def forward(self,x):return x+self.pe[:,:x.shape[1]]
    class TrajectoryTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.future_length=future_length
            self.input_projection=nn.Linear(2,d_model)
            self.position=PositionalEncoding()
            layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,
                dim_feedforward=ff,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,num_layers=layers)
            self.norm=nn.LayerNorm(d_model)
            self.head=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),
                                    nn.Dropout(dropout),nn.Linear(d_model,future_length*2))
        def forward(self,x):
            x=self.encoder(self.position(self.input_projection(x)))
            return self.head(self.norm(x.mean(dim=1))).view(-1,self.future_length,2)
    return TrajectoryTransformer()


def load_checkpoint(path,task,seed,future_length):
    import torch
    # weights_only prevents arbitrary pickle objects; user should still use trusted checkpoints.
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(ck,dict):raise TypeError('Checkpoint must be a dict')
    if 'task' in ck and ck['task']!=task:raise ValueError(f'checkpoint task mismatch {path}')
    if 'seed' in ck and int(ck['seed'])!=seed:raise ValueError(f'checkpoint seed mismatch {path}')
    sd=ck.get('state_dict',ck)
    if not isinstance(sd,dict):raise TypeError('checkpoint state_dict missing')
    model=model_factory(future_length,ck)
    # PE buffer is deterministic and can vary in stored length; exactly historical EXP3 loader.
    sd={k:v for k,v in sd.items() if k!='position.pe'}
    result=model.load_state_dict(sd,strict=False)
    missing=[k for k in result.missing_keys if k!='position.pe']
    if missing or result.unexpected_keys:
        raise ValueError(f'checkpoint incompatible: missing={missing} extra={result.unexpected_keys}')
    model.eval()
    return model


def predict(model,can,batch=128):
    import torch
    y=[]
    with torch.inference_mode():
        for i in range(0,len(can),batch):
            p=model(torch.as_tensor(can[i:i+batch],dtype=torch.float32)).numpy()
            if not np.isfinite(p).all():raise ValueError('Non-finite model predictions')
            y.append(p)
    return np.concatenate(y,axis=0)


def summarize_metrics(pools,models):
    summaries=[];detail=[]
    for h in TASKS:
        for subset in ('all','QA_unflagged'):
            for ident in ('pooled',*IDS):
                for model in models:
                    selections=[];weights=[]
                    for tid in IDS:
                        if ident!='pooled' and tid!=ident:continue
                        group=pools[(h,tid)];selected=np.ones(len(group['history']),dtype=bool)
                        if subset=='QA_unflagged': selected=~group['qa_flagged']
                        if not np.any(selected):continue
                        a,f=ade_fde(group[f'{model}_pred'],group['future_gt'])
                        selections.extend((a[selected],f[selected])); weights.append((int(selected.sum()),float(a[selected].sum()),float(f[selected].sum())))
                    n=sum(q[0] for q in weights)
                    summaries.append({'horizon_frames':h,'subset':subset,'track_id':ident,
                                      'model':model,'windows':n,'ade_norm':sum(q[1] for q in weights)/n,
                                      'fde_norm':sum(q[2] for q in weights)/n})
        for tid in IDS:
            grp=pools[(h,tid)]
            errors={m:ade_fde(grp[f'{m}_pred'],grp['future_gt']) for m in models}
            for i,start in enumerate(grp['start_frame']):
                row={'horizon_frames':h,'track_id':tid,'first_observed_frame':int(start),
                     'last_observed_frame':int(start)+19,'last_target_frame':int(start)+19+h,
                     'qa_flagged_window':int(grp['qa_flagged'][i])}
                for m in models:
                    row[f'{m}_ade_norm']=float(errors[m][0][i]);row[f'{m}_fde_norm']=float(errors[m][1][i])
                detail.append(row)
    return summaries,detail


def preflight(npz,out):
    pools=read_data(npz)
    stats=[];max_roundtrip=0
    for (h,tid),group in pools.items():
        meta=canonicalize(group['history'],TASKS[h]['scale_floor'])
        recovered=reconstruct(meta['history_canonical'],meta)
        err=float(np.max(np.abs(recovered-group['history'])))
        max_roundtrip=max(max_roundtrip,err)
        if err>1e-6:raise AssertionError(f'roundtrip mismatch {h},{tid}: {err}')
        group.update({'canonical':meta['history_canonical'],'meta':meta,
                      'stationary_pred':group['stationary_pred'],
                      'cv20_pred':group['cv20_pred'],
                      'cv_ols_pred':cv_ols(group['history'],h)})
        stats.append({'horizon_frames':h,'track_id':tid,'windows':len(group['history']),
                      'scale_floor':TASKS[h]['scale_floor'],
                      'floor_applied_count':int(np.sum(meta['path']<TASKS[h]['scale_floor'])),
                      'floor_applied_percent':float(100*np.mean(meta['path']<TASKS[h]['scale_floor'])),
                      'path_median_norm':float(np.median(meta['path'])),
                      'path_p90_norm':float(np.percentile(meta['path'],90)),
                      'canonical_roundtrip_max_abs':err})
    write_csv(out/'canonicalization_statistics.csv',stats)
    return pools,stats,max_roundtrip


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase7e-npz',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,default=None,
                   help='optional directory with 6 untouched .pt models; absence = compatibility preflight only')
    p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args()
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    pools,stats,roundtrip=preflight(args.phase7e_npz,args.output)
    files={f'{TASKS[h]["name"]}_Transformer_seed{seed}.pt':None for h in TASKS for seed in SEEDS}
    if args.checkpoint_dir:
        for name in files:
            candidate=args.checkpoint_dir/name
            if candidate.is_file():files[name]={'sha256':sha256_file(candidate),'size_bytes':candidate.stat().st_size}
    missing=[name for name,v in files.items() if v is None]
    manifest={'phase':'7F','dataset':'WildLive DJI_0207','input_sha256':sha256_file(args.phase7e_npz),
        'input_path':str(args.phase7e_npz),'timing_verified':False,
        'data_origin':'WildLive bbox centres; no object world coordinates; no camera stabilization',
        'interpretation':'exploratory frame-index-only; one capture group; overlapping windows',
        'training_timing_context':'frozen legacy 10-Hz model; WildLive released frame cadence UNVERIFIED',
        'model_architecture':'Compact Temporal Transformer (original archived Python implementation)',
        'model_checkpoints':files,'missing_checkpoints':missing,
        'canonical_roundtrip_max_abs':roundtrip,'canonicalization_exact_task_floors':{TASKS[h]['name']:TASKS[h]['scale_floor'] for h in TASKS},
        'baseline_difference':'Phase7E CV20 uses endpoint slope; historical EXP3 CV uses OLS slope',
        'status':'BLOCKED_MISSING_CHECKPOINTS' if missing else 'READY_FOR_FROZEN_INFERENCE'}
    # Re-evaluate phase7e original baselines and separate EXP3 OLS CV as descriptive comparator.
    summary,detail=summarize_metrics(pools,('stationary','cv20','cv_ols'))
    write_csv(args.output/'baseline_comparison_phase7f.csv',summary)
    write_csv(args.output/'baseline_per_window_phase7f.csv',detail)
    if missing:
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
        print('PHASE7F: PREPROCESSING/BASELINE PREFLIGHT PASSED; FROZEN INFERENCE BLOCKED')
        print('MISSING CHECKPOINTS:',*missing,sep='\n  ')
        print('No Transformer metrics were computed.')
        return
    predictions={}
    import torch
    for h in TASKS:
        task=TASKS[h]['name'];models=[]
        for seed in SEEDS:
            name=f'{task}_Transformer_seed{seed}.pt'
            model=load_checkpoint(args.checkpoint_dir/name,task,seed,h)
            models.append(model)
        for tid in IDS:
            grp=pools[(h,tid)]
            pred_abs=[]
            for model in models:
                pred_can=predict(model,grp['canonical'],args.batch_size)
                pred_abs.append(reconstruct(pred_can,grp['meta']))
            ensemble=np.mean(np.stack(pred_abs,axis=0),axis=0)
            grp['transformer_ensemble_pred']=ensemble
            stem=f'id{tid}_h{h}'
            predictions[f'{stem}_transformer_ensemble_pred']=ensemble.astype('float32')
            predictions[f'{stem}_qa_flagged']=grp['qa_flagged']
            predictions[f'{stem}_start_frame']=grp['start_frame']
    model_names=('stationary','cv20','cv_ols','transformer_ensemble')
    summaries,details=summarize_metrics(pools,model_names)
    write_csv(args.output/'ALL_MODELS_comparison_phase7f.csv',summaries)
    write_csv(args.output/'ALL_MODELS_per_window_phase7f.csv',details)
    np.savez_compressed(args.output/'frozen_transformer_predictions_exploratory.npz',**predictions)
    manifest['status']='FROZEN_INFERENCE_COMPLETED_EXPLORATORY'
    manifest['model_evaluation_performed']=True
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('PHASE7F: FROZEN INFERENCE COMPLETED (exploratory only, timing not verified)')
    for row in summaries:
        if row['track_id']=='pooled' and row['subset']=='all':print(row)


if __name__=='__main__':main()
