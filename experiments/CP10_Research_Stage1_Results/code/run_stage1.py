#!/usr/bin/env python3
"""CP10 real-EXP3 retrospective research audit / development evaluation.
Only numpy,pandas,matplotlib. Results not independent confirmatory evidence.
"""
import argparse, hashlib, io, json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q_TRAIN = {"2s": 2.158768, "5s": 1.975130}
REF_METRICS = {"2s": {"coverage": .9655, "n2_recall": .8662, "n2_fwr": .4762,
                       "nominal_recall": .3312, "nominal_fwr": .0169},
               "5s": {"coverage": .6976, "n2_recall": .7951, "n2_fwr": .4093,
                       "nominal_recall": .1947, "nominal_fwr": .0112}}


def q90(x):
    # Empirical 0.90 quantile; this is a RETROSPECTIVE heuristic under dependent windows.
    return float(np.quantile(np.asarray(x,float),.90))


def wquantile(values, weights, alpha=.9):
    order=np.argsort(values)
    v=np.asarray(values,dtype=float)[order]
    w=np.asarray(weights,dtype=float)[order]
    if np.any(w<0) or not np.isfinite(w).all() or w.sum() <= 0: raise ValueError('Invalid weights')
    return float(v[np.searchsorted(np.cumsum(w)/w.sum(), alpha, side='left')])


def load_h(zipfile_handle, h):
    tag='2s_to_'+h; prefix='results\\'
    d=np.load(io.BytesIO(zipfile_handle.read(prefix+tag+'_exp3_frozen_ensemble_predictions.npz')))
    a=np.load(io.BytesIO(zipfile_handle.read(prefix+tag+'_fresh_dataset.npz')))
    meta=pd.read_csv(io.BytesIO(zipfile_handle.read(prefix+tag+'_fresh_metadata.csv')))
    pred=np.asarray(d['ensemble_prediction'],float)
    true=np.asarray(d['future_abs'],float)
    hist=np.asarray(d['history_abs'],float)
    scale=np.asarray(a['scale'],float)
    if not (len(meta)==len(pred)==len(true)==len(hist)==len(scale)): raise RuntimeError('Shape mismatch')
    if not (np.allclose(a['future_abs'],true, atol=1e-6) and np.allclose(meta['scale'],scale,atol=1e-7)):
        raise RuntimeError('Source consistency failed')
    scores=np.linalg.norm(pred-true,axis=2).max(axis=1)/scale
    speeds=np.linalg.norm(np.diff(hist,axis=1),axis=2).mean(axis=1)
    return dict(h=h,pred=pred,true=true,hist=hist,scale=scale,score=scores,speed=speeds,meta=meta)


def estimate_thresholds(train_idx,d,method):
    scores=d['score'][train_idx]
    meta=d['meta'].iloc[train_idx]
    if method=='target_window_p90': return q90(scores),None
    if method=='video_equal_weight_p90':
        n=meta['video'].map(meta['video'].value_counts()).to_numpy(float)
        return wquantile(scores,1/n),None
    if method=='track_equal_weight_p90':
        n=meta['track_key'].map(meta['track_key'].value_counts()).to_numpy(float)
        return wquantile(scores,1/n),None
    if method=='group_worst_p90':
        return max(q90(scores[meta['elephant_group'].to_numpy()==g]) for g in meta['elephant_group'].unique()),None
    if method=='speed_2bin_p90':
        speed=d['speed'][train_idx]
        split=float(np.median(speed))
        if min((speed<=split).sum(),(speed>split).sum())<25:
            raise RuntimeError('Not enough calibration windows in speed bin')
        qlo=q90(scores[speed<=split]); qhi=q90(scores[speed>split])
        return (qlo,qhi),split
    raise ValueError(method)


def exp4(d):
    groups=sorted(d['meta']['elephant_group'].unique())
    baseline=d['score']<=Q_TRAIN[d['h']]
    methods=['frozen_trainreal_q90','target_window_p90','video_equal_weight_p90',
             'track_equal_weight_p90','group_worst_p90','speed_2bin_p90']
    out=[]; details=[]
    for method in methods:
        for held in groups:
            test=d['meta']['elephant_group'].to_numpy()==held
            train=~test
            if method=='frozen_trainreal_q90': qt=Q_TRAIN[d['h']]; split=None
            else: qt,split=estimate_thresholds(train,d,method)
            n=test.sum()
            if isinstance(qt,tuple): qs=np.where(d['speed'][test]<=split,qt[0],qt[1]); qstr=json.dumps({'low':qt[0],'high':qt[1],'median_speed':split})
            else: qs=np.repeat(qt,n);qstr=str(qt)
            cover=d['score'][test]<=qs
            radius=qs*d['scale'][test]
            for v in np.unique(d['meta'].loc[test,'video']):
                m=d['meta']['video'].to_numpy()==v
                m=m[test]
                details.append({'horizon':d['h'],'method':method,'held_out_group':held,'video':v,
                                'n_windows':int(m.sum()),'coverage':float(cover[m].mean()),
                                'median_radius':float(np.median(radius[m]))})
            out.append({'horizon':d['h'],'method':method,'held_out_group':held,
                        'train_groups':','.join([g for g in groups if g!=held]),'n_windows':int(n),
                        'n_cal_windows':int(train.sum()),'n_cal_tracklets':int(d['meta'].loc[train,'track_key'].nunique()),
                        'coverage':float(cover.mean()),'median_radius':float(np.median(radius)),
                        'mean_radius':float(np.mean(radius)),
                        'median_q':float(np.median(qs)),
                        'q_setting':qstr,
                        'count_covered':int(cover.sum())})
    per_group=pd.DataFrame(out)
    per_video=pd.DataFrame(details)
    summary=[]
    for method in methods:
        rows=per_group[per_group.method==method]
        ns=rows.n_windows.to_numpy();n=ns.sum()
        summary.append({'horizon':d['h'],'method':method,'n_windows':int(n),
                        'pooled_coverage':float(np.dot(rows.coverage,ns)/n),
                        'group_macro_coverage':float(rows.coverage.mean()),
                        'worst_group_coverage':float(rows.coverage.min()),
                        'weighted_mean_radius':float(np.dot(rows.mean_radius,ns)/n),
                        'relative_region_expansion_vs_frozen':np.nan})
    summ=pd.DataFrame(summary)
    base=float(summ.loc[summ.method=='frozen_trainreal_q90','weighted_mean_radius'].iloc[0])
    summ['relative_region_expansion_vs_frozen']=summ.weighted_mean_radius/base
    return per_group,per_video,summ


def exp5_replay(d, multipliers):
    """Reconstruct Step18b six directional synthetic boundaries and Step7H N=1,N=2 alert logic.

    The true/nominal geometry comes from step18b_evaluate_locked_geofence.py.
    The persistence groups/order follow step7h_final_response_aware_with_locked_persistence.py.
    """
    meta=d['meta']
    chunks=[]
    start=d['hist'][:,-1,:]
    for ax,idx in [('x',0),('y',1)]:
        for bound in [.25,.50,.75]:
            pos=start[:,idx]
            valid=np.abs(pos-bound)>1e-10
            directions=np.where(pos<bound,'high','low')
            sign=np.where(pos<bound,1.,-1.)
            truth_sign=sign[:,None]*(d['true'][:,:,idx]-bound)
            pred_sign=sign[:,None]*(d['pred'][:,:,idx]-bound)
            t_cross=truth_sign.max(axis=1)>=0
            p_cross=pred_sign.max(axis=1)>=0
            near=pred_sign.max(axis=1)
            entries={'sample_index':np.arange(len(meta)), 'video':meta.video.to_numpy(),
                     'source_group':meta.source_group.to_numpy(), 'elephant_group':meta.elephant_group.to_numpy(),
                     'track_key':meta.track_key.to_numpy(), 'anchor_frame_index':meta.anchor_frame_index.to_numpy(),
                     'axis':ax,'boundary':bound,'direction':directions,
                     'risk_score':near, 'true_cross':t_cross,'pred_cross':p_cross,
                     'scale':d['scale']}
            chunks.append(pd.DataFrame(entries)[valid])
    df=pd.concat(chunks,ignore_index=True)
    # Match Step7H: rolling over sorted sample_index by source_group,track_key,axis,boundary,direction,boundary_id
    df=df.sort_values(['source_group','track_key','axis','boundary','direction','sample_index']).reset_index(drop=True)
    groupcols=['source_group','track_key','axis','boundary','direction']
    rows=[]
    for m in multipliers:
        inter=(df.risk_score.to_numpy()+m*Q_TRAIN[d['h']]*df.scale.to_numpy())>=0
        prev=pd.Series(inter,index=df.index).groupby([df[c] for c in groupcols],sort=False).shift(1, fill_value=False).to_numpy(bool)
        for n in [1,2]:
            alarm=df.pred_cross.to_numpy(bool) | (inter if n==1 else (inter & prev))
            true=df.true_cross.to_numpy(bool)
            TP=int((alarm & true).sum());FN=int((~alarm & true).sum());FP=int((alarm & ~true).sum());TN=int((~alarm & ~true).sum())
            row={'horizon':d['h'],'q_multiplier':float(m),'N':n,'n_boundary_cases':len(df),
                 'TP':TP,'FN':FN,'FP':FP,'TN':TN,
                 'recall':TP/max(1,TP+FN),'fwr':FP/max(1,FP+TN),
                 'precision':TP/max(1,TP+FP),'f1':2*TP/max(1,2*TP+FP+FN)}
            # Deduplication along valid consecutive forecast WINDOWS in same track/boundary/direction.
            # Frame-index gaps of > stride 10 indicate distinct alert episodes.
            a=pd.Series(alarm)
            previous=a.groupby([df[c] for c in groupcols],sort=False).shift(1,fill_value=False).to_numpy(bool)
            last_anchor=df.anchor_frame_index.groupby([df[c] for c in groupcols],sort=False).shift(1).to_numpy()
            starts=alarm & (~previous | ((df.anchor_frame_index.to_numpy()-last_anchor)>10))
            row['alert_episodes']=int(starts.sum())
            row['episodes_per_100_boundary_windows']=float(starts.sum()/len(df)*100)
            rows.append(row)
    return pd.DataFrame(rows),df


def exp5_budget(d, df, budgets=(.10,.20)):
    # Pre-specified candidate multipliers with fixed N=2; group-disjoint calibration/evaluation.
    groupcols=['source_group','track_key','axis','boundary','direction']
    groups=sorted(df.elephant_group.unique())
    xs=[0.,.10,.20,.30,.40,.50,.60,.75,1.,1.25,1.50,2.]
    results=[]; base=[]
    true=df.true_cross.to_numpy(bool)
    for m in xs:
        inter=(df.risk_score.to_numpy()+m*Q_TRAIN[d['h']]*df.scale.to_numpy())>=0
        prev=pd.Series(inter,index=df.index).groupby([df[c] for c in groupcols],sort=False).shift(1, fill_value=False).to_numpy(bool)
        alarm=df.pred_cross.to_numpy(bool)|(inter&prev)
        for g in groups:
            ix=df.elephant_group.to_numpy()==g
            tp=int((alarm & true & ix).sum()); fn=int((~alarm & true & ix).sum())
            fp=int((alarm & ~true & ix).sum()); tn=int((~alarm & ~true & ix).sum())
            base.append({'horizon':d['h'],'group':g,'q_multiplier':m,'TP':tp,'FN':fn,'FP':fp,'TN':tn,
                         'recall':tp/max(1,tp+fn),'fwr':fp/max(1,fp+tn),
                         'precision':tp/max(1,tp+fp)})
    table=pd.DataFrame(base)
    for budget in budgets:
        for held in groups:
            cal=table[table.group!=held]
            by=cal.groupby('q_multiplier').agg(worst_cal_fwr=('fwr','max'),cal_macro_recall=('recall','mean')).reset_index()
            allowed=by[by.worst_cal_fwr<=budget+1e-12]
            # highest macro recall; deterministic tie break favors less intrusive q.
            if allowed.empty: raise RuntimeError('Nominal rule should pass declared budget')
            pick=allowed.sort_values(['cal_macro_recall','q_multiplier'],ascending=[False,True]).iloc[0]
            heldrow=table[(table.group==held)&(table.q_multiplier==pick.q_multiplier)].iloc[0]
            nominal=table[(table.group==held)&(table.q_multiplier==0.)].iloc[0]
            frozen=table[(table.group==held)&(table.q_multiplier==1.)].iloc[0]
            results.append({'horizon':d['h'],'budget':budget,'held_out_group':held,
                            'selected_q_multiplier':float(pick.q_multiplier),
                            'cal_max_fwr':float(pick.worst_cal_fwr),
                            'cal_macro_recall':float(pick.cal_macro_recall),
                            'test_recall':float(heldrow.recall),'test_fwr':float(heldrow.fwr),
                            'nominal_recall':float(nominal.recall),'nominal_fwr':float(nominal.fwr),
                            'frozen_N2_recall':float(frozen.recall),'frozen_N2_fwr':float(frozen.fwr),
                            'test_TP':int(heldrow.TP),'test_FN':int(heldrow.FN),
                            'test_FP':int(heldrow.FP),'test_TN':int(heldrow.TN),
                            'budget_respected_on_heldout':bool(heldrow.fwr <= budget + 1e-12)})
    return table,pd.DataFrame(results)


def charts(summaries, warnings, out):
    s=pd.concat(summaries)
    # Scientifically neutral comparison with coverage versus region width
    fig,axs=plt.subplots(1,2,figsize=(12.8,5.4),dpi=165)
    labels=['frozen_trainreal_q90','target_window_p90','video_equal_weight_p90',
            'track_equal_weight_p90','group_worst_p90','speed_2bin_p90']
    short={'frozen_trainreal_q90':'Frozen TrainReal','target_window_p90':'Window quantile',
           'video_equal_weight_p90':'Video-weighted','track_equal_weight_p90':'Track-weighted',
           'group_worst_p90':'Worst-group quantile','speed_2bin_p90':'Two speed bins'}
    # Numeric markers, explicit legend below: overlapping method points remain distinguishable.
    from matplotlib.lines import Line2D
    for j,h in enumerate(['2s','5s']):
        part=s[s.horizon==h].set_index('method')
        for k,method in enumerate(labels):
            r=part.loc[method]
            axs[j].scatter(r.relative_region_expansion_vs_frozen,r.pooled_coverage*100,s=185,facecolor='white',edgecolor='black',linewidth=.8,zorder=3)
            axs[j].text(r.relative_region_expansion_vs_frozen,r.pooled_coverage*100,str(k+1),
                        ha='center',va='center',fontsize=8,zorder=4)
        axs[j].axhline(90,linestyle='--',linewidth=.9,color='grey')
        axs[j].set_xlabel('Mean region-radius ratio to frozen q90')
        axs[j].set_ylabel('Pathwise coverage (%)')
        axs[j].set_title(h+' horizon | leave-one-recording-group-out')
        axs[j].grid(alpha=.17)
        axs[j].set_xlim(left=0)
        axs[j].set_ylim(55,105)
    fig.suptitle('EXP4: Calibration versus region inflation (retrospective)',fontsize=13)
    legend='     '.join(f'{i+1}. {short[n]}' for i,n in enumerate(labels[:3]))+'\n'+ \
           '     '.join(f'{i+4}. {short[n]}' for i,n in enumerate(labels[3:]))
    fig.text(.5,.012,legend,ha='center',va='bottom',fontsize=9)
    fig.tight_layout(rect=[0,.085,1,.945])
    fig.savefig(out/'EXP4_Coverage_vs_RegionInflation.png',bbox_inches='tight')
    plt.close(fig)
    w=pd.concat(warnings)
    fig,axs=plt.subplots(1,2,figsize=(11.3,4.7),dpi=165)
    for j,h in enumerate(['2s','5s']):
        part=w[w.horizon==h]
        for n in [1,2]:
            a=part[part.N==n].sort_values('q_multiplier')
            axs[j].plot(a.fwr*100,a.recall*100,marker='.',label=f'N={n}')
        nom=part[(part.N==2)&(part.q_multiplier==0)].iloc[0]
        axs[j].scatter(nom.fwr*100,nom.recall*100,marker='x',s=75,color='black',label='Nominal')
        axs[j].set_xlabel('False-warning rate (%)')
        axs[j].set_ylabel('Crossing recall (%)')
        axs[j].set_title(h+' horizon | six synthetic boundaries')
        axs[j].grid(alpha=.17);axs[j].legend()
    fig.suptitle('EXP5: Frozen-q multiplier sweep (retrospective; do not select on test)',fontsize=13)
    fig.tight_layout();fig.savefig(out/'EXP5_Recall_FWR_Curves.png',bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,default=Path('/mnt/data/EXP3_Output_Review.zip'))
    p.add_argument('--out',type=Path,default=Path('/mnt/data/CP10_Research_Stage1'))
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(args.zip.read_bytes()).hexdigest()
    gs=[];vs=[];ss=[];ww=[]; audits=[]
    with zipfile.ZipFile(args.zip) as z:
        for h in ['2s','5s']:
            d=load_h(z,h)
            grp,vid,summ=exp4(d)
            grp.to_csv(out/f'EXP4_{h}_per_group.csv',index=False)
            vid.to_csv(out/f'EXP4_{h}_per_video.csv',index=False)
            summ.to_csv(out/f'EXP4_{h}_summary.csv',index=False)
            gs.append(grp);vs.append(vid);ss.append(summ)
            grid=[0.,.25,.5,.75,1.,1.25,1.5,2.,2.5,3.,4.]
            w, inst=exp5_replay(d,grid)
            w.to_csv(out/f'EXP5_{h}_operating_curve.csv',index=False)
            ww.append(w)
            budget_sweep, budget_holdout=exp5_budget(d,inst)
            budget_sweep.to_csv(out/f'EXP5_{h}_group_sweep.csv',index=False)
            budget_holdout.to_csv(out/f'EXP5_{h}_budget_holdout.csv',index=False)
            original=w[(w.q_multiplier==1.)&(w.N==2)].iloc[0]
            nominal=w[(w.q_multiplier==0.)&(w.N==2)].iloc[0]
            scorecov=float(np.mean(d['score']<=Q_TRAIN[h]))
            audit={'horizon':h,'n_samples':len(d['scale']),
                   'n_videos':int(d['meta']['video'].nunique()),'n_recording_groups':int(d['meta']['elephant_group'].nunique()),
                   'n_track_keys':int(d['meta']['track_key'].nunique()),
                   'baseline_q90_coverage':scorecov,
                   'prior_reported_coverage_rounded':REF_METRICS[h]['coverage'],
                   'original_rule_N2':{'recall':float(original.recall),'fwr':float(original.fwr)},
                   'previous_paper_N2':{'recall':REF_METRICS[h]['n2_recall'],'fwr':REF_METRICS[h]['n2_fwr']},
                   'nominal_replay':{'recall':float(nominal.recall),'fwr':float(nominal.fwr)},
                   'prior_nominal':{'recall':REF_METRICS[h]['nominal_recall'],'fwr':REF_METRICS[h]['nominal_fwr']}}
            audits.append(audit)
    charts(ss,ww,out)
    (out/'EXP3_Reproduction_Audit.json').write_text(json.dumps({'input_sha256':sha,'source_zip':args.zip.name,'audits':audits},indent=2))
    print(json.dumps(audits,indent=2))
    print(pd.concat(ss).to_string(index=False))

if __name__=='__main__':main()
