#!/usr/bin/env python3
"""Phase 7J: recording-blocked, retrospective adaptive forecasting selector.

Purposefully narrow design: a fixed ridge expected-log-ADE estimator selects from
Stationary, CV-OLS, and precomputed frozen Transformer forecasts. Its inputs are
STRICTLY history-derived Phase 7I features; each training label is a future ADE.
No model checkpoints are loaded or tuned. The target of the method is ADE only.
NO untouched confirmation is claimed with these two previously inspected sequences.
"""
from __future__ import annotations
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

EXPERTS = ('stationary','cv_ols','transformer_ensemble')
FEATURES = ('mean_step_norm_per_frame','straightness','ols_line_residual_norm',
            'acceleration_rms_norm_per_frame2','mean_abs_turn_rad',
            'ols_velocity_norm_per_frame')
TRANSFORM = ('log','plain','log','log','plain','log')
KEYS = ('sequence','horizon_frames','track_id','first_observed_frame')
RIDGE_ALPHA = 25.0
Y_EPS=1e-6
X_EPS=1e-6


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as fp:
        for data in iter(lambda:fp.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def load_phase7i(path):
    path=Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            bad=z.testzip()
            if bad:raise ValueError('Bad zip member '+bad)
            candidates=[name for name in z.namelist() if name.endswith('per_window_observation_motion_and_model_errors.csv')]
            if len(candidates)!=1:raise ValueError('Expected exactly one Phase 7I per-window CSV')
            df=pd.read_csv(io.BytesIO(z.read(candidates[0])))
    else:
        df=pd.read_csv(path)
    validate(df)
    return df


def validate(df):
    required=set(KEYS+FEATURES+tuple(f'{x}_ade_norm' for x in EXPERTS)
                 +tuple(f'{x}_fde_norm' for x in EXPERTS))
    missing=sorted(required-set(df.columns))
    if missing:raise ValueError('Missing required columns: '+repr(missing))
    if df.duplicated(list(KEYS)).any():raise ValueError('Duplicate windows')
    if df[list(KEYS)].isna().any().any():raise ValueError('Missing IDs')
    if not set(df['horizon_frames'].unique()) <= {20,50}:raise ValueError('Unapproved horizon')
    for m in EXPERTS:
        for metric in ('ade_norm','fde_norm'):
            vals=df[f'{m}_{metric}'].to_numpy(float)
            if not np.isfinite(vals).all() or (vals<0).any():raise ValueError('Invalid metric '+m+' '+metric)
    for name in FEATURES:
        values=df[name].to_numpy(float)
        if np.isinf(values).any():raise ValueError('Infinite motion feature: '+name)
    if df['straightness'].dropna().gt(1.000001).any() or df['straightness'].dropna().lt(-1e-8).any():
        raise ValueError('Straightness outside [0,1]')
    if df['sequence'].nunique()<2:raise ValueError('Require >1 recording for blocked diagnostic')


def build_raw_features(frame, fill=None):
    """Only FEATURES consulted. No future targets or sequence identity enters X."""
    if fill is None:fill={}
    cols=[];missing_cols=[];out_fill={}
    for feat, transform in zip(FEATURES,TRANSFORM):
        v=frame[feat].to_numpy(dtype=float)
        bad=~np.isfinite(v)
        if feat in fill: replacement=float(fill[feat])
        else:
            finite=v[~bad]
            replacement=float(np.median(finite)) if len(finite) else 0.
        v=np.where(bad,replacement,v)
        if transform=='log':v=np.log10(np.maximum(v,0)+X_EPS)
        cols.append(v)
        missing_cols.append(bad.astype(float))
        out_fill[feat]=replacement
    return np.column_stack(cols+missing_cols),out_fill


def design_train(frame):
    x,fill=build_raw_features(frame)
    mean=x.mean(0)
    std=x.std(0)
    std=np.where(std>1e-9,std,1.)
    xx=np.clip((x-mean)/std,-4,4)
    return xx, {'fill':fill,'feature_mean':mean.tolist(),'feature_std':std.tolist()}


def design_predict(frame, model):
    x,_=build_raw_features(frame,model['fill'])
    mean=np.array(model['feature_mean'],float);std=np.array(model['feature_std'],float)
    return np.clip((x-mean)/std,-4,4)


def nonoverlap_train_subset(frame,horizon):
    """Within each track: [history+future] blocks do not overlap.

    Starts 0, 40, 80,.. for h20; 0,70,140,.. for h50. This only reduces
    within-track duplication; different animals in same recording still co-move.
    """
    length=20+horizon
    return frame.loc[frame['first_observed_frame'].astype(int)%length==0].copy()


def fit_selector(frame, h, sequences, alpha=RIDGE_ALPHA):
    train=frame[(frame['horizon_frames']==h)&(frame['sequence'].isin(sequences))]
    if train.empty:raise ValueError('Empty training set')
    train=nonoverlap_train_subset(train,h)
    if train.empty:raise ValueError('No nonoverlapping training windows')
    xx,normalizer=design_train(train)
    yy=np.log(np.maximum(train[[f'{m}_ade_norm' for m in EXPERTS]].to_numpy(float),0)+Y_EPS)
    A=np.column_stack([np.ones(len(xx)),xx])
    regularizer=np.eye(A.shape[1]); regularizer[0,0]=0
    weights=np.linalg.solve(A.T@A+alpha*regularizer,A.T@yy)
    best_idx=int(np.argmin(train[[f'{m}_ade_norm' for m in EXPERTS]].mean().to_numpy()))
    model=dict(horizon_frames=int(h),history_frames=20,experts=list(EXPERTS),features=list(FEATURES),
               feature_transforms=list(TRANSFORM),alpha=float(alpha),target='log(ADE+1e-6)',
               selection_criterion='smallest_predicted_log_ADE',
               train_sequences=sorted(set(sequences)),n_nonoverlap_train_windows=len(train),
               training_tracks=len(train[['sequence','track_id']].drop_duplicates()),
               global_best_train_expert=EXPERTS[best_idx],feature_clip_z=4,weights=weights.tolist(),**normalizer)
    return model


def select(frame,model):
    if not (frame['horizon_frames']==model['horizon_frames']).all():raise ValueError('Horizon mismatch')
    xx=design_predict(frame,model)
    A=np.column_stack([np.ones(len(xx)),xx])
    pred=A@np.asarray(model['weights'],float)
    idx=pred.argmin(axis=1)
    return np.asarray(model['experts'],object)[idx]


def score_eval(held,model):
    pred=select(held,model)
    out=held[list(KEYS)].copy()
    out['selected_expert']=pred
    out['train_best_fixed_expert']=model['global_best_train_expert']
    for expert in EXPERTS:
        for suffix in ('ade_norm','fde_norm'):
            out[f'{expert}_{suffix}']=held[f'{expert}_{suffix}'].to_numpy(float)
    def choose(suffix, labels):
        matrix=held[[f'{m}_{suffix}' for m in EXPERTS]].to_numpy(float)
        idx=np.array([EXPERTS.index(m) for m in labels])
        return matrix[np.arange(len(matrix)),idx]
    out['adaptive_ade_norm']=choose('ade_norm',pred)
    out['adaptive_fde_norm']=choose('fde_norm',pred)
    out['train_best_fixed_ade_norm']=choose('ade_norm',np.repeat(model['global_best_train_expert'],len(held)))
    out['train_best_fixed_fde_norm']=choose('fde_norm',np.repeat(model['global_best_train_expert'],len(held)))
    out['hindsight_oracle_ade_norm']=held[[f'{m}_ade_norm' for m in EXPERTS]].min(axis=1).to_numpy()
    # Infeasible lower bound; future target needed to select this on each test window.
    return out


def summarize(out):
    scored=[]
    models=['stationary','cv_ols','transformer_ensemble','adaptive','train_best_fixed','hindsight_oracle']
    for keys,g in out.groupby(['train_sequences','heldout_sequence','horizon_frames'],sort=True):
        row=dict(zip(['train_sequences','heldout_sequence','horizon_frames'],keys))
        row['windows']=len(g)
        row['n_tracks']=int(g['track_id'].nunique())
        for m in models:
            row[f'{m}_ADE']=float(g[f'{m}_ade_norm'].mean())
            if m!='hindsight_oracle':row[f'{m}_FDE']=float(g[f'{m}_fde_norm'].mean())
        for m in EXPERTS:
            row['selected_'+m+'_fraction']=float((g['selected_expert']==m).mean())
        row['adaptive_minus_best_expert_ADE']=row['adaptive_ADE']-min(row[f'{m}_ADE'] for m in EXPERTS)
        row['adaptive_minus_train_best_ADE']=row['adaptive_ADE']-row['train_best_fixed_ADE']
        scored.append(row)
    return pd.DataFrame(scored)


def run(frame,output,source_file):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'locked_models').mkdir(exist_ok=True)
    seqs=sorted(frame['sequence'].unique())
    if len(seqs)!=2:raise ValueError('Phase 7J exploratory two-recording protocol requires exactly 2 recordings')
    results=[];model_records={}
    for heldout in seqs:
        train=[s for s in seqs if s != heldout]
        for h in (20,50):
            model=fit_selector(frame,h,train)
            # Group-disjoint; also prevent scene id from acting as a feature.
            assert heldout not in model['train_sequences']
            target=frame[(frame['sequence']==heldout)&(frame['horizon_frames']==h)].copy()
            out=score_eval(target,model)
            out.insert(0,'train_sequences',';'.join(train));out.insert(1,'heldout_sequence',heldout)
            results.append(out)
            model_records[f'{heldout}_h{h}']=dict(train=sorted(train),train_windows=model['n_nonoverlap_train_windows'],
                                                 n_tracks=model['training_tracks'],train_fixed=model['global_best_train_expert'])
    predictions=pd.concat(results,ignore_index=True).sort_values(['heldout_sequence','horizon_frames','track_id','first_observed_frame'])
    predictions.to_csv(output/'blocked_stress_per_window.csv',index=False,float_format='%.12g')
    summary=summarize(predictions)
    summary.to_csv(output/'blocked_stress_summary.csv',index=False,float_format='%.12g')
    selected=[]
    for keys,g in predictions.groupby(['heldout_sequence','horizon_frames','track_id']):
        row=dict(zip(['sequence','horizon_frames','track_id'],keys),windows=len(g))
        for m in ('stationary','cv_ols','transformer_ensemble','adaptive','train_best_fixed','hindsight_oracle'):
            row[f'{m}_ADE']=float(g[f'{m}_ade_norm'].mean())
        selected.append(row)
    pd.DataFrame(selected).to_csv(output/'blocked_stress_per_track.csv',index=False,float_format='%.12g')
    # FINAL candidate model refit on BOTH PRE-INSPECTED recordings (development only).
    # This is NOT evaluated as independently validated on these same recordings.
    model_files={}
    for h in (20,50):
        final=fit_selector(frame,h,seqs)
        final.update(status='DEVELOPMENT_ONLY_NOT_INDEPENDENTLY_VALIDATED',
                     trained_using='DJI_0207 and DJI_0204_video2, previously inspected',
                     source_data_sha256=sha256_file(source_file))
        dest=output/'locked_models'/f'phase7j_dev_gate_h{h}.json'
        dest.write_text(json.dumps(final,indent=2,sort_keys=True)+'\n')
        model_files[dest.name]=sha256_file(dest)
    manifest={
        'phase':'7J','design':'two-fold leave-one-recording-out retrospective stress test; fixed ridge alpha 25',
        'source_phase7i_zip':str(source_file),'source_sha256':sha256_file(source_file),
        'n_window_records':len(frame),'sequences':seqs,
        'heldout_recordings_untouched':False,
        'capture_group_independence_verified':False,
        'fps_verified':False,'training_targets_future_ADE_used_only_in_fitting':True,
        'model_input_strictly_observation_features':list(FEATURES),
        'nonoverlap_train_stride':'history 20 + horizon (40 or 70) indexed start positions',
        'training_group_disjoint_in_blocked_test':True,
        'adaptive_model_performs_real_time_inference':False,
        'locked_models_trained_on_all_development_sequences':model_files,
        'results_not_for_statistical_significance':True,
        'checkpoint_weights_not_retrained':True,
        'note':'Forecast metric selection only; no conformal/alert evaluation, no physical-time mapping.'}
    (output/'phase7j_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    draw(summary,output)
    return summary


def draw(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=['stationary','cv_ols','transformer_ensemble','adaptive','train_best_fixed']
    fig, ax=plt.subplots(figsize=(10,5))
    indices=np.arange(len(summary));width=0.15
    for i,m in enumerate(metrics):
        ax.bar(indices+(i-(len(metrics)-1)/2)*width,summary[f'{m}_ADE'],width,label=m)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"{r.heldout_sequence}\n{r.horizon_frames}-frame future" for r in summary.itertuples()],fontsize=9)
    ax.set_ylabel('Mean ADE (normalized image-plane; lower is better)')
    ax.set_title('Phase 7J: retrospective blocked transfer; no untouched confirmation')
    ax.legend(ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(out/'phase7j_blocked_comparison.png',dpi=170);plt.close(fig)


def run_predict(model_path,feature_path,output_path):
    model=json.loads(Path(model_path).read_text())
    frame=pd.read_csv(feature_path)
    missing=set(FEATURES+('horizon_frames',))-set(frame.columns)
    if missing:raise ValueError('Missing inputs '+str(missing))
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    out=frame[[c for c in KEYS if c in frame.columns]].copy()
    out['selected_expert']=select(frame,model)
    out.to_csv(output_path,index=False)
    return len(out)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('develop',help='blocked stress test and fit final development-only gates')
    a.add_argument('--phase7i',type=Path,required=True,help='WildLive_Phase7I_Reproduced_Results.zip')
    a.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('predict',help='apply one locked gate to observation-only feature CSV')
    b.add_argument('--model',type=Path,required=True);b.add_argument('--features',type=Path,required=True)
    b.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='develop':
        df=load_phase7i(args.phase7i)
        result=run(df,args.output,args.phase7i)
        print('PHASE7J RETROSPECTIVE BLOCKED STRESS TEST COMPLETED; NO INDEPENDENT CONFIRMATION')
        print(result[['heldout_sequence','horizon_frames','windows','stationary_ADE',
                      'cv_ols_ADE','transformer_ensemble_ADE','adaptive_ADE',
                      'train_best_fixed_ADE','hindsight_oracle_ADE']].to_string(index=False))
    elif args.command=='predict':
        n=run_predict(args.model,args.features,args.output)
        print('PREDICTIONS',n,': label only, no ADE/FDE computed')

if __name__=='__main__':main()
