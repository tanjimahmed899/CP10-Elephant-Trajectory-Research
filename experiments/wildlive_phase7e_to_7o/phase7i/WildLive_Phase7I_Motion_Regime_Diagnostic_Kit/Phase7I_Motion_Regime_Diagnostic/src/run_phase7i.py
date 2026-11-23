#!/usr/bin/env python3
"""CP1.0 Phase 7I: descriptive, observation-only motion-regime diagnosis.

Runs using archived user inputs. No FPS assumption, independent-sample inference,
checkpoint training, model selection, or camera-motion correction is made.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, re, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ('stationary', 'cv20', 'cv_ols', 'transformer_ensemble')
HIST = 20


def sha256(path):
    hash_obj = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            hash_obj.update(block)
    return hash_obj.hexdigest()


def get_member(z, suffix):
    files=[f for f in z.namelist() if f.endswith(suffix)]
    if len(files)!=1: raise ValueError(f'{suffix}: expected exactly one member, found {files}')
    return files[0]


def load_zip_csv(path,suffix):
    with zipfile.ZipFile(path) as z:
        broken=z.testzip()
        if broken: raise ValueError(f'Bad ZIP member: {broken}')
        return pd.read_csv(io.BytesIO(z.read(get_member(z,suffix))))


def load_zip_json(path,suffix):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read(get_member(z,suffix)))


def read_0204_centres(zip_path):
    frames=defaultdict(dict)
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip():raise ValueError('Invalid DJI_0204 annotation ZIP')
        for name in z.namelist():
            match=re.search(r'(?:^|/)frame_(\d+)\.json$',name)
            if match is None:continue
            f=int(match.group(1)); j=json.loads(z.read(name));w=j['imageWidth'];h=j['imageHeight']
            for shape in j['shapes']:
                if str(shape.get('label','')).lower() !='elephant':continue
                tid=int(shape['group_id']);pts=np.asarray(shape['points'],dtype=float)
                low=pts.min(axis=0);high=pts.max(axis=0)
                if np.any(low<0) or high[0]>w or high[1]>h:raise ValueError('Out of bounds box')
                if tid in frames[f]:raise ValueError(f'Duplicate track {tid} in frame {f}')
                frames[f][tid] = (low+high)/[2*w,2*h]
    track_map=defaultdict(dict)
    for f,trks in frames.items():
        for tid,xy in trks.items():track_map[tid][f]=xy
    return track_map


def read_0207_centres(zip_path):
    df=load_zip_csv(zip_path,'original_bbox_centres.csv')
    track_map=defaultdict(dict)
    for r in df.itertuples(index=False):
        track_map[int(r.track_id)][int(r.frame_index)]=np.array([r.x_center_normalized,r.y_center_normalized],float)
    return track_map


def motion_features(hist):
    """Input: (N,20,2) from observation window only. Units normalized image/fr."""
    if hist.ndim!=3 or hist.shape[1:]!=(20,2):raise ValueError('Require N×20×2')
    v=np.diff(hist,axis=1)
    step=np.linalg.norm(v,axis=2)
    path=step.sum(axis=1)
    disp=np.linalg.norm(hist[:,-1]-hist[:,0],axis=1)
    straightness=np.divide(disp,path,out=np.full_like(disp,np.nan),where=path>1e-12)
    accel=np.diff(v,axis=1)
    accel_rms=np.sqrt(np.mean(np.sum(accel**2,axis=2),axis=1))
    t=np.arange(20,dtype=float);tc=t-t.mean();sxx=np.sum(tc**2)
    velocity=np.einsum('t,ntc->nc',tc,hist)/sxx
    line=hist.mean(axis=1)[:,None,:]+tc[None,:,None]*velocity[:,None,:]
    residual_rms=np.sqrt(np.mean(np.sum((hist-line)**2,axis=-1),axis=1))
    speed=path/19
    # Turning is unstable at tiny steps; only compare adjacent displacements with
    # both magnitudes >= max(5e-5, 0.25*track median step).
    eps=np.maximum(5e-5,0.25*np.median(step,axis=1))
    valid=(step[:,:-1]>=eps[:,None])&(step[:,1:]>=eps[:,None])
    cross=v[:,:-1,0]*v[:,1:,1]-v[:,:-1,1]*v[:,1:,0]
    dot=np.sum(v[:,:-1,:]*v[:,1:,:],axis=-1)
    theta=np.abs(np.arctan2(cross,dot))
    cnt=valid.sum(axis=1)
    turning=np.divide((theta*valid).sum(axis=1),cnt,out=np.full(len(hist),np.nan),where=cnt>0)
    return {'path_norm':path,'mean_step_norm_per_frame':speed,'net_displacement_norm':disp,
            'straightness':straightness,'ols_line_residual_norm':residual_rms,
            'acceleration_rms_norm_per_frame2':accel_rms,'mean_abs_turn_rad':turning,
            'turn_pairs_used':cnt, 'ols_velocity_norm_per_frame':np.linalg.norm(velocity,axis=1)}


def input_rows_0207(metrics_path,npz_path):
    metric=load_zip_csv(metrics_path,'ALL_MODELS_per_window_phase7f.csv')
    rows=[]
    with zipfile.ZipFile(npz_path) as z:
        with np.load(io.BytesIO(z.read(get_member(z,'frame_based_windows_and_baselines.npz'))),allow_pickle=False) as npz:
            for h in [20,50]:
                for tid in [0,1,2]:
                    hist=npz[f'id{tid}_h{h}_history']
                    starts=npz[f'id{tid}_h{h}_start_frame']
                    feat=motion_features(hist)
                    for i,start in enumerate(starts):
                        row={'sequence':'DJI_0207','horizon_frames':h,'track_id':tid,
                             'first_observed_frame':int(start)}
                        row.update({k:float(val[i]) if np.issubdtype(val.dtype,np.floating) else int(val[i]) for k,val in feat.items()})
                        rows.append(row)
    frame=pd.DataFrame(rows)
    merge_cols=['horizon_frames','track_id','first_observed_frame']
    metric=metric.drop_duplicates(subset=merge_cols)
    assert len(metric)==4884
    joined=frame.merge(metric,how='inner',on=merge_cols,validate='one_to_one')
    if len(joined)!=4884:raise ValueError('DJI_0207 key mismatch')
    return joined


def input_rows_0204(metrics_path,original_zip):
    metric=load_zip_csv(metrics_path,'locked_model_per_window.csv')
    tracks=read_0204_centres(original_zip)
    rows=[]
    for h in [20,50]:
        for tid,timeline in sorted(tracks.items()):
            for start in range(0,357-20-h+1):
                keys=list(range(start,start+20))
                if not all(k in timeline for k in keys):continue
                hist=np.array([[timeline[k] for k in keys]])
                feat=motion_features(hist)
                row={'sequence':'DJI_0204_video2','horizon_frames':h,'track_id':tid,'first_observed_frame':start}
                row.update({k:float(val[0]) if np.issubdtype(val.dtype,np.floating) else int(val[0]) for k,val in feat.items()})
                rows.append(row)
    frame=pd.DataFrame(rows)
    merge_cols=['horizon_frames','track_id','first_observed_frame']
    metric=metric.drop_duplicates(subset=merge_cols)
    assert len(metric)==2424
    joined=frame.merge(metric,how='inner',on=merge_cols,validate='one_to_one')
    if len(joined)!=2424:raise ValueError('DJI_0204 key mismatch')
    return joined,tracks


def compute_common_mode(tracks):
    # An animal-only common-mode motion *proxy*; may reflect herd co-motion,
    # camera motion, or both. NOT background/camera stabilization.
    frames=sorted(set.intersection(*[set(d) for d in tracks.values()]))
    if len(frames)<2:return {}
    ids=sorted(tracks)
    loc=np.stack([[tracks[t][f] for t in ids] for f in frames],axis=0)
    use=np.diff(np.asarray(frames))==1
    delta=np.diff(loc,axis=0)[use]
    if len(delta)==0:return {}
    common=np.median(delta,axis=1)
    norm=np.linalg.norm(delta,axis=-1)
    residual=np.linalg.norm(delta-common[:,None,:],axis=-1)
    common_mags=np.linalg.norm(common,axis=-1)
    # Fraction of per-animal velocity energy attributed to common-mode proxy,
    # descriptive only; ratio not guaranteed 0..1.
    return {'frames_with_all_tracks':len(frames),'adjacent_frame_pairs_used':int(len(delta)),
            'n_elephants':len(ids),'mean_per_track_displacement_norm':float(norm.mean()),
            'median_per_track_displacement_norm':float(np.median(norm)),
            'mean_common_mode_step_norm':float(common_mags.mean()),
            'mean_residual_step_norm':float(residual.mean()),
            'common_mode_over_animal_step_ratio':float(common_mags.mean()/max(norm.mean(),1e-15))}


def quantile_bins(df,feature):
    out=[]
    for (seq,h),group in df.groupby(['sequence','horizon_frames']):
        arr=group[feature].to_numpy()
        q1,q2=np.quantile(arr[np.isfinite(arr)],[1/3,2/3])
        if q1==q2:
            cat=np.array(['tie/undetermined']*len(group),dtype=object)
        else:cat=np.where(arr<=q1,'low',np.where(arr<=q2,'mid','high'))
        out.append(pd.Series(cat,index=group.index))
    return pd.concat(out).sort_index()


def compare_summary(df):
    summary=[]
    for (seq,h),g in df.groupby(['sequence','horizon_frames'],sort=True):
        row={'sequence':seq,'horizon_frames':h,'windows':len(g),
             'elephant_tracks':g['track_id'].nunique()}
        for m in METRICS:
            row[f'{m}_ADE']=g[f'{m}_ade_norm'].mean()
            row[f'{m}_FDE']=g[f'{m}_fde_norm'].mean()
        row['trans_minus_stationary_ADE']=row['transformer_ensemble_ADE']-row['stationary_ADE']
        row['trans_minus_cv_ols_ADE']=row['transformer_ensemble_ADE']-row['cv_ols_ADE']
        for feat in ['mean_step_norm_per_frame','straightness','ols_line_residual_norm','acceleration_rms_norm_per_frame2']:
            row[f'median_{feat}']=g[feat].median()
        summary.append(row)
    return pd.DataFrame(summary)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase7f',type=Path,required=True,help='Phase7F_Frozen_Results.zip')
    p.add_argument('--phase7e',type=Path,required=True,help='WildLive_Phase7E_Reproduced_Results.zip')
    p.add_argument('--phase7h',type=Path,required=True,help='Phase7H_DJI0204_Frozen_Evaluation.zip')
    p.add_argument('--dj0204',type=Path,required=True,help='DJI_0204_video2_boxid.zip')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    out=a.output;out.mkdir(exist_ok=True,parents=True)
    (out/'figures').mkdir(exist_ok=True)
    paths={n:getattr(a,n) for n in ['phase7f','phase7e','phase7h','dj0204']}
    for n,path in paths.items():
        if not path.exists():raise FileNotFoundError(f'{n}: {path}')
    m7f=load_zip_json(a.phase7f,'manifest.json')
    m7h=load_zip_json(a.phase7h,'evaluation_manifest.json')
    if m7h['file_sha256'] != sha256(a.dj0204):raise ValueError('DJI_0204 source SHA256 mismatch')
    with zipfile.ZipFile(a.phase7e) as z:
        npzbytes=z.read(get_member(z,'frame_based_windows_and_baselines.npz'))
    if hashlib.sha256(npzbytes).hexdigest() != m7f['input_sha256']:
        raise ValueError('DJI_0207 Phase7E NPZ SHA256 mismatch')
    for ck,entry in m7f['model_checkpoints'].items():
        if entry['sha256']!=m7h['locked_model_checkpoint_hashes_verified'][ck]['sha256']:
            raise ValueError(f'Checkpoint manifest mismatch: {ck}')
    d7f=input_rows_0207(a.phase7f,a.phase7e)
    d7h,tracks7h=input_rows_0204(a.phase7h,a.dj0204)
    joined=pd.concat([d7f,d7h],ignore_index=True)
    assert len(joined)==7308
    # explicit paired window-based comparisons; observed movement only
    joined['trans_minus_stationary_ADE']=joined['transformer_ensemble_ade_norm']-joined['stationary_ade_norm']
    joined['trans_minus_cv_ols_ADE']=joined['transformer_ensemble_ade_norm']-joined['cv_ols_ade_norm']
    joined['best_model_hindsight_descriptive']=joined[[m+'_ade_norm' for m in METRICS]].idxmin(axis=1).str.replace('_ade_norm','',regex=False)
    joined['speed_tercile']=quantile_bins(joined,'mean_step_norm_per_frame')
    joined['straightness_tercile']=quantile_bins(joined,'straightness')
    joined.to_csv(out/'per_window_observation_motion_and_model_errors.csv',index=False,float_format='%.12g')
    summ=compare_summary(joined)
    # Reconcile all original source summary rows to source per-window rows.
    original_7f=load_zip_csv(a.phase7f,'ALL_MODELS_comparison_phase7f.csv')
    original_7h=load_zip_csv(a.phase7h,'locked_model_per_track_and_pooled.csv')
    summary_verified=0
    for rec in original_7f.itertuples(index=False):
        d=joined[(joined.sequence=='DJI_0207')&(joined.horizon_frames==int(rec.horizon_frames))]
        if rec.track_id!='pooled':d=d[d.track_id==int(rec.track_id)]
        if rec.subset=='QA_unflagged':d=d[d.qa_flagged_window==0]
        if not len(d) or int(rec.windows)!=len(d):raise ValueError('Original 7F summary window mismatch')
        for k in ['ade','fde']:
            if not np.isclose(d[f'{rec.model}_{k}_norm'].mean(),getattr(rec,f'{k}_norm'),atol=1e-10,rtol=0):
                raise ValueError('Original 7F summary numerical mismatch')
        summary_verified+=1
    for rec in original_7h.itertuples(index=False):
        d=joined[(joined.sequence=='DJI_0204_video2')&(joined.horizon_frames==int(rec.horizon_frames))]
        if rec.track_id!='pooled':d=d[d.track_id==int(rec.track_id)]
        if not len(d) or int(rec.windows)!=len(d):raise ValueError('Original 7H summary window mismatch')
        for k in ['ade','fde']:
            if not np.isclose(d[f'{rec.model}_{k}_norm'].mean(),getattr(rec,f'{k}_norm'),atol=1e-10,rtol=0):
                raise ValueError('Original 7H summary numerical mismatch')
        summary_verified+=1
    summ.to_csv(out/'per_recording_horizon_summary.csv',index=False,float_format='%.12g')

    bytrack=(joined.groupby(['sequence','horizon_frames','track_id']).agg(
      windows=('track_id','size'),speed_median=('mean_step_norm_per_frame','median'),
      straightness_median=('straightness','median'),ols_residual_median=('ols_line_residual_norm','median'),
      stationary_ade=('stationary_ade_norm','mean'),cv_ols_ade=('cv_ols_ade_norm','mean'),
      transformer_ade=('transformer_ensemble_ade_norm','mean'),
      transformer_minus_cv_ols_ade=('trans_minus_cv_ols_ADE','mean'),
      transformer_minus_stationary_ade=('trans_minus_stationary_ADE','mean')).reset_index())
    bytrack.to_csv(out/'per_track_motion_and_model_errors.csv',index=False,float_format='%.12g')
    # Within-track tertiles reduce (but do not remove) track-mix confounding.
    joined['within_track_speed_tercile']=pd.concat([
      pd.Series(np.where(g.mean_step_norm_per_frame<=g.mean_step_norm_per_frame.quantile(1/3),'low',
                np.where(g.mean_step_norm_per_frame<=g.mean_step_norm_per_frame.quantile(2/3),'mid','high')),index=g.index)
      for _,g in joined.groupby(['sequence','horizon_frames','track_id'])]).sort_index()
    within_track=(joined.groupby(['sequence','horizon_frames','track_id','within_track_speed_tercile']).agg(
        windows=('track_id','size'),median_speed=('mean_step_norm_per_frame','median'),
        transformer_minus_cv_ols_ade=('trans_minus_cv_ols_ADE','mean'),
        transformer_minus_stationary_ade=('trans_minus_stationary_ADE','mean')).reset_index())
    within_track.to_csv(out/'within_elephant_speed_regime_deltas.csv',index=False,float_format='%.12g')
    # QA sensitivity only exists for DJI_0207. Do not assume DJI_0204 has no bad labels.
    qa=joined[joined.sequence=='DJI_0207']
    qa_sensitivity=[]
    for h,g in qa.groupby('horizon_frames'):
        for subset,rows in [('all',g),('QA_unflagged',g[g.qa_flagged_window==0])]:
            qa_sensitivity.append({'sequence':'DJI_0207','horizon_frames':h,'subset':subset,
             'windows':len(rows),'median_observed_speed':rows.mean_step_norm_per_frame.median(),
             'transformer_ade':rows.transformer_ensemble_ade_norm.mean(),
             'stationary_ade':rows.stationary_ade_norm.mean(),
             'cv_ols_ade':rows.cv_ols_ade_norm.mean()})
    pd.DataFrame(qa_sensitivity).to_csv(out/'DJI0207_QA_subset_motion_results.csv',index=False,float_format='%.12g')

    regime=(joined.groupby(['sequence','horizon_frames','speed_tercile']).agg(
      windows=('speed_tercile','size'),median_speed=('mean_step_norm_per_frame','median'),
      median_straightness=('straightness','median'),
      stationary_ade=('stationary_ade_norm','mean'),cv_ols_ade=('cv_ols_ade_norm','mean'),
      transformer_ade=('transformer_ensemble_ade_norm','mean'),
      transformer_minus_cv_ols_ade=('trans_minus_cv_ols_ADE','mean'),
      transformer_minus_stationary_ade=('trans_minus_stationary_ADE','mean'),
      transformer_win_cv_ols_fraction=('trans_minus_cv_ols_ADE',lambda x:float(np.mean(x<0)))).reset_index())
    regime['speed_tercile']=pd.Categorical(regime['speed_tercile'],categories=['low','mid','high'],ordered=True)
    regime.sort_values(['sequence','horizon_frames','speed_tercile']).to_csv(out/'speed_tercile_error_table.csv',index=False,float_format='%.12g')
    straight=(joined.groupby(['sequence','horizon_frames','straightness_tercile']).agg(
      windows=('straightness_tercile','size'),median_straightness=('straightness','median'),
      stationary_ade=('stationary_ade_norm','mean'),cv_ols_ade=('cv_ols_ade_norm','mean'),
      transformer_ade=('transformer_ensemble_ade_norm','mean'),
      trans_minus_cv_ols=('trans_minus_cv_ols_ADE','mean')).reset_index())
    straight.to_csv(out/'straightness_tercile_error_table.csv',index=False,float_format='%.12g')
    # cross-sequence common motion: DIY herd displacement proxy, not camera motion
    tracks7f=read_0207_centres(a.phase7e)
    common=pd.DataFrame([{'sequence':'DJI_0207',**compute_common_mode(tracks7f)},
                         {'sequence':'DJI_0204_video2',**compute_common_mode(tracks7h)}])
    common.to_csv(out/'multi_animal_common_motion_proxy.csv',index=False,float_format='%.12g')
    # figures (static matplotlib default palette)
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),sharey=False)
    for ax,h in zip(axs,[20,50]):
        temp=summ[summ.horizon_frames==h].sort_values('sequence')
        x=np.arange(len(temp));w=0.24
        for i,m in enumerate(['stationary','cv_ols','transformer_ensemble']):
            ax.bar(x+(i-1)*w,temp[f'{m}_ADE'],w,label=m)
        ax.set_xticks(x);ax.set_xticklabels(temp.sequence,rotation=8)
        ax.set_title(f'Future {h} frames');ax.set_ylabel('ADE (normalized image-plane)')
        ax.grid(axis='y',alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle('Two-sequence descriptive ADE comparison (not independent validation)')
    fig.tight_layout();fig.savefig(out/'figures/01_sequence_model_ADE.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=False)
    for i,seq in enumerate(['DJI_0207','DJI_0204_video2']):
        for j,h in enumerate([20,50]):
            ax=axs[i,j]
            g=regime[(regime.sequence==seq)&(regime.horizon_frames==h)].set_index('speed_tercile').reindex(['low','mid','high'])
            x=np.arange(3);w=.26
            for k,model in enumerate(['stationary','cv_ols','transformer']):
                ax.bar(x+(k-1)*w,g[model+'_ade'],w,label=model)
            ax.set_xticks(x);ax.set_xticklabels(['Low','Middle','High'])
            ax.set_title(f'{seq}: future {h} frames');ax.set_ylabel('ADE (normalized)')
            if i==0 and j==0:ax.legend(fontsize=8)
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('Observation-only speed tertiles; thresholds within each sequence/horizon')
    fig.tight_layout();fig.savefig(out/'figures/02_observed_speed_regime_ADE.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for ax,h in zip(axs,[20,50]):
        for seq in ['DJI_0207','DJI_0204_video2']:
            g=joined[(joined.sequence==seq)&(joined.horizon_frames==h)]
            ax.scatter(g.mean_step_norm_per_frame,g.trans_minus_cv_ols_ADE,s=3,alpha=.15,label=seq,rasterized=True)
        ax.axhline(0,color='black',lw=1)
        ax.set_title(f'{h}-frame future');ax.set_xlabel('Mean observed step / frame (normalized)')
        ax.set_ylabel('Transformer ADE - CV-OLS ADE');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle('Observation motion versus model gap (overlapping windows; descriptive)')
    fig.tight_layout();fig.savefig(out/'figures/03_observed_speed_vs_error_gap.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    common.set_index('sequence')[['mean_per_track_displacement_norm','mean_common_mode_step_norm','mean_residual_step_norm']].plot.bar(ax=ax)
    ax.set_ylabel('Normalized image-plane displacement per adjacent index')
    ax.set_title('Animal-only common-motion proxy: NOT camera correction')
    ax.tick_params(axis='x',rotation=0);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'figures/04_animal_common_motion_proxy.png',dpi=160);plt.close(fig)
    # descriptive associations no inferential statistics
    corr=[]
    for (seq,h),g in joined.groupby(['sequence','horizon_frames']):
        for feat in ['mean_step_norm_per_frame','straightness','ols_line_residual_norm','acceleration_rms_norm_per_frame2']:
            x=g[feat];y=g['trans_minus_cv_ols_ADE'];valid=x.notna()&y.notna()
            corr.append({'sequence':seq,'horizon_frames':h,'feature':feat,'n_windows':int(valid.sum()),
                         'spearman_rho_descriptive':float(x[valid].rank().corr(y[valid].rank()))})
    pd.DataFrame(corr).to_csv(out/'descriptive_feature_error_correlations.csv',index=False,float_format='%.12g')
    manifests={'phase':'7I','status':'OBSERVATION_ONLY_DESCRIPTIVE_DIAGNOSIS_COMPLETE',
        'inputs':{k:{'filename':p.name,'sha256':sha256(p)} for k,p in paths.items()},
        'checkpoints':{k:d['sha256'] for k,d in m7f['model_checkpoints'].items()},
        'source_summary_rows_reconciled':summary_verified, 'n_overlapping_windows':int(len(joined)),'recordings':['DJI_0207','DJI_0204_video2'],
        'capture_event_independence_verified':False,'fps_verified':False,
        'no_model_training':True,'no_learned_gate_or_threshold':True,
        'all_regime_features_from_observation_only':True,
        'speed_bins': 'within-sequence and within-horizon empirical tertiles; post-hoc descriptive, not deployment thresholds',
        'common_motion': 'animal-only same-frame median displacement; camera motion vs herd co-motion unidentifiable without static background or camera poses',
        'statistical_note':'One unverified capture event per filename; overlapping windows; no P values, no cluster confidence intervals, no independence claim.'}
    (out/'analysis_manifest.json').write_text(json.dumps(manifests,indent=2))
    print('--- PHASE7I AUDIT ---')
    print(f'7308 observation-only paired windows across 2 source folders; checkpoint hashes matched; {summary_verified} archived summary rows reconciled')
    print(summ.to_string(index=False,float_format=lambda v:f'{v:.8f}'))
    print('\nCOMMON MODE\n',common.to_string(index=False,float_format=lambda v:f'{v:.8f}'))
    print('\nSPEED TERCILES\n',regime[['sequence','horizon_frames','speed_tercile','windows','median_speed','transformer_minus_cv_ols_ade','transformer_minus_stationary_ade']].to_string(index=False,float_format=lambda v:f'{v:.8f}'))

if __name__=='__main__': main()
