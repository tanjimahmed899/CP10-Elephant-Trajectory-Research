#!/usr/bin/env python3
"""Read-only Stage 8 retrospective Stage7F identity-clustered audit.
Usage: python audit_stage8.py --input /path/to/CP10_Stage7F_Tracker_Forecast_Review.zip --out /path/to/output --replicates 20000
No training, no inference, no GT-blind claim. Requires numpy, pandas, matplotlib.
"""
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def zip_csv(z, name):
    return pd.read_csv(io.BytesIO(z.read(name)))


def summarize(data, horizon, rng, repeats):
    d = data[(data.horizon == horizon) & (data.input_arm == 'tracker') & data.predictor.isin(['Stationary', 'Transformer_ensemble_3seed'])]
    keys = ['sequence', 'GT_id_OFFLINE_AUDIT', 'track_id', 'anchor_frame']
    wide = d.pivot(index=keys, columns='predictor', values=['ADE', 'FDE'])
    if len(wide) != (187 if horizon == '2s' else 124):
        raise AssertionError('Unexpected window count: '+ str((horizon,len(wide))))
    wide.columns=['_'.join(c) for c in wide.columns]
    wide=wide.reset_index()
    wide['ADE_improvement']=wide['ADE_Stationary']-wide['ADE_Transformer_ensemble_3seed']
    wide['FDE_improvement']=wide['FDE_Stationary']-wide['FDE_Transformer_ensemble_3seed']
    wide['horizon']=horizon
    gt_group=wide.groupby('GT_id_OFFLINE_AUDIT',sort=True)
    by_gt=gt_group.agg(n_windows=('ADE_improvement','size'), ADE_improvement=('ADE_improvement','mean'), FDE_improvement=('FDE_improvement','mean'), ADE_stationary=('ADE_Stationary','mean'), ADE_transformer=('ADE_Transformer_ensemble_3seed','mean'), FDE_stationary=('FDE_Stationary','mean'), FDE_transformer=('FDE_Transformer_ensemble_3seed','mean')).reset_index()
    by_gt['horizon']=horizon
    groups=[v for _,v in gt_group]
    n=len(groups)
    idx=rng.integers(0,n,size=(repeats,n))
    sums_stat=np.array([g.ADE_Stationary.sum() for g in groups]); sums_pred=np.array([g.ADE_Transformer_ensemble_3seed.sum() for g in groups]); counts=np.array([len(g) for g in groups]); sums_fdstat=np.array([g.FDE_Stationary.sum() for g in groups]); sums_fdpred=np.array([g.FDE_Transformer_ensemble_3seed.sum() for g in groups]);
    bs_n=counts[idx].sum(axis=1)
    bs_ade_sta=sums_stat[idx].sum(axis=1)/bs_n
    bs_ade_pre=sums_pred[idx].sum(axis=1)/bs_n
    bs_fde_sta=sums_fdstat[idx].sum(axis=1)/bs_n
    bs_fde_pre=sums_fdpred[idx].sum(axis=1)/bs_n
    boot_gain=bs_ade_sta-bs_ade_pre
    boot_reduction=100*boot_gain/bs_ade_sta
    boot_fd_gain=bs_fde_sta-bs_fde_pre
    mean_stat=wide.ADE_Stationary.mean(); mean_pred=wide.ADE_Transformer_ensemble_3seed.mean()
    mean_fde_sta=wide.FDE_Stationary.mean(); mean_fde_pre=wide.FDE_Transformer_ensemble_3seed.mean()
    macro_stat=by_gt.ADE_stationary.mean(); macro_pred=by_gt.ADE_transformer.mean()
    met={
        'horizon':horizon,
        'sequence_count':int(wide.sequence.nunique()),
        'n_windows':int(len(wide)),
        'n_gt_identities':int(n),
        'n_tracker_ids':int(wide.track_id.nunique()),
        'ADE_stationary_window_mean':float(mean_stat),
        'ADE_transformer_window_mean':float(mean_pred),
        'ADE_gain_window_mean':float(mean_stat-mean_pred),
        'ADE_reduction_window_pct':float((mean_stat-mean_pred)/mean_stat*100),
        'ADE_gain_gt_macro_mean':float(macro_stat-macro_pred),
        'ADE_reduction_gt_macro_pct':float((macro_stat-macro_pred)/macro_stat*100),
        'FDE_stationary_window_mean':float(mean_fde_sta),
        'FDE_transformer_window_mean':float(mean_fde_pre),
        'FDE_gain_window_mean':float(mean_fde_sta-mean_fde_pre),
        'FDE_reduction_window_pct':float((mean_fde_sta-mean_fde_pre)/mean_fde_sta*100),
        'GT_identities_with_positive_ADE_gain':int((by_gt.ADE_improvement>0).sum()),
        'GT_identities_with_positive_FDE_gain':int((by_gt.FDE_improvement>0).sum()),
        'windows_with_positive_ADE_gain':int((wide.ADE_improvement>0).sum()),
        'bootstrap_2p5_ADE_gain':float(np.quantile(boot_gain,.025)),
        'bootstrap_97p5_ADE_gain':float(np.quantile(boot_gain,.975)),
        'bootstrap_2p5_ADE_reduction_pct':float(np.quantile(boot_reduction,.025)),
        'bootstrap_97p5_ADE_reduction_pct':float(np.quantile(boot_reduction,.975)),
        'bootstrap_2p5_FDE_gain':float(np.quantile(boot_fd_gain,.025)),
        'bootstrap_97p5_FDE_gain':float(np.quantile(boot_fd_gain,.975)),
        'bootstrap_fraction_ADE_gain_positive':float(np.mean(boot_gain>0)),
    }
    return wide, by_gt, met


def run(inputzip, out, repeats):
    out.mkdir(parents=True,exist_ok=True)
    input_hash=hashlib.sha256(inputzip.read_bytes()).hexdigest()
    with zipfile.ZipFile(inputzip) as z:
        if z.testzip() is not None:
            raise AssertionError('Source ZIP failed CRC test')
        prov=json.loads(z.read('Stage7F_Provenance.json'))
        if prov.get('status')!='FROZEN_TRACKER_TRANSFORMER_COMPLETE':
            raise AssertionError('Runner has not completed')
        src=zip_csv(z,'Stage7F_Sample_Level_ADE_FDE.csv')
        win=zip_csv(z,'Stage7F_Verified_Evaluation_Windows.csv')
        reported=zip_csv(z,'Stage7F_Frozen_ThreeSeed_Summary.csv')
        if len(src)!=3110 or len(win)!=311:
            raise AssertionError('Missing sample metric or eligibility rows')
        checks=[]
        for horizon,filename in [('2s','Stage7F_2s_to_2s_Predictions.npz'),('5s','Stage7F_2s_to_5s_Predictions.npz')]:
            pred=np.load(io.BytesIO(z.read(filename)),allow_pickle=False)
            for arm, pkey, hkey in [('tracker','tracker_ensemble_pred_abs','tracker_history'),('oracle','oracle_ensemble_pred_abs','oracle_GT_history')]:
                errs=np.linalg.norm(pred[pkey]-pred['gt_future'],axis=-1)
                base=np.linalg.norm(pred[hkey][:,-1:,:]-pred['gt_future'],axis=-1)
                for predname, vals in [('Transformer_ensemble_3seed',errs),('Stationary',base)]:
                    rows=src[(src.horizon==horizon)&(src.input_arm==arm)&(src.predictor==predname)]
                    if len(rows)!=len(vals):raise AssertionError('Missing per-sample rows')
                    adediff=float(np.max(np.abs(rows.ADE.to_numpy()-vals.mean(axis=1))))
                    fdediff=float(np.max(np.abs(rows.FDE.to_numpy()-vals[:,-1])))
                    if max(adediff,fdediff)>1e-10:raise AssertionError('NPZ to CSV mismatch')
                    checks.append({'horizon':horizon,'input_arm':arm,'predictor':predname,'n':len(vals),'max_abs_ADE_difference':adediff,'max_abs_FDE_difference':fdediff})
        pd.DataFrame(checks).to_csv(out/'Stage8_Prediction_Array_to_CSV_Verification.csv',index=False)
        actual=src.groupby(['horizon','input_arm','predictor'],as_index=False)[['ADE','FDE']].mean()
        reported_red=reported.merge(actual,on=['horizon','input_arm','predictor'],how='outer',validate='one_to_one',suffixes=('_reported','_recomputed'),indicator=True)
        if (reported_red._merge!='both').any() or max((reported_red.ADE_mean-reported_red.ADE).abs().max(),(reported_red.FDE_mean-reported_red.FDE).abs().max())>1e-12:
            raise AssertionError('Final reported means differ')
        reported_red.to_csv(out/'Stage8_Report_to_Samples_Consistency.csv',index=False)
    rng=np.random.default_rng(20261121)
    all_w=[];all_g=[];all_s=[]
    for horizon in ['2s','5s']:
        w,g,s=summarize(src,horizon,rng,repeats)
        all_w.append(w);all_g.append(g);all_s.append(s)
    table=pd.DataFrame(all_s)
    table.to_csv(out/'Stage8_GT_Clustered_Summary.csv',index=False)
    pergt=pd.concat(all_g,ignore_index=True)
    pergt.to_csv(out/'Stage8_Per_GT_Identity_Differences.csv',index=False)
    pd.concat(all_w,ignore_index=True).to_csv(out/'Stage8_Paired_Window_Differences.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for ax,hz in zip(axs,['2s','5s']):
        rows=pergt[pergt.horizon==hz].sort_values('GT_id_OFFLINE_AUDIT')
        x=np.arange(len(rows))
        ax.bar(x,rows.ADE_improvement)
        ax.axhline(0,linestyle='--',linewidth=1)
        ax.set_xticks(x,rows.GT_id_OFFLINE_AUDIT.astype(str).tolist())
        ax.set_ylabel('ADE gain (Stationary minus Transformer)')
        ax.set_xlabel('GT elephant identity (offline)')
        ax.set_title(hz+' • '+str((rows.ADE_improvement>0).sum())+'/'+str(len(rows))+' identities improve')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Stage 7F — within-sequence, GT-identity clustered ADE differences')
    fig.tight_layout()
    fig.savefig(out/'Figure_Stage8_PerElephant_ADE_Differences.png',dpi=190,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.7,4.4))
    y=np.arange(2)
    for i,s in enumerate(all_s):
        gain=s['ADE_gain_window_mean']
        lo=s['bootstrap_2p5_ADE_gain'];hi=s['bootstrap_97p5_ADE_gain']
        ax.errorbar(gain,i,xerr=[[gain-lo],[hi-gain]],fmt='o',capsize=5,label=s['horizon'])
    ax.set_yticks(y,['2-second','5-second'])
    ax.axvline(0,linewidth=1,linestyle='--')
    ax.set_xlim(left=min(0,min(s['bootstrap_2p5_ADE_gain'] for s in all_s)-.001))
    ax.set_xlabel('ADE reduction: stationary minus Transformer (normalized units)')
    ax.set_title('GT-identity resampling intervals (descriptive, one video)')
    ax.grid(axis='x',alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'Figure_Stage8_Identity_Bootstrap_ADE_Gain.png',dpi=190,bbox_inches='tight')
    plt.close(fig)
    audit={
        'input':inputzip.name,'input_sha256':input_hash,'source_provenance':prov,
        'algorithm':'Nonparametric bootstrap: draw GT identities with replacement and include all of each identity windows. Recompute pooled paired ADE and FDE differences per resample; 2.5/97.5 percentiles.',
        'bootstrap_seed':20261121,'bootstrap_replicates':repeats,
        'scope':'Within a single BIRDSAI TestReal sequence; not a cross-recording confidence interval or a confirmatory p value.',
        'non_independence':'GT identities may correlate via interactions or shared camera scene. Overlapping windows are not independent. Recording-group count=1.',
        'array_checks_passed':True,'report_checks_passed':True,'statistics':all_s,
    }
    (out/'Stage8_Audit_Manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    return table,pergt,audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--replicates',type=int,default=20000)
    a=p.parse_args()
    tab,_,_=run(a.input,a.out,a.replicates)
    print(tab.to_string(index=False))
