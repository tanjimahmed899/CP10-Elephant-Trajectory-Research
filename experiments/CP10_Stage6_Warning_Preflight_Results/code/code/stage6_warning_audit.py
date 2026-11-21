#!/usr/bin/env python3
"""CP10 Stage 6: read-only, reproducible retrospective warning policy audit.

Uncertainty-intersection rule, boundary geometry and persistence inspired by
original Step18b/Step7H as independently replayed by Stage1 script.
These boundaries are reset to the observation-start camera reference per sample.
Therefore consecutive boundary cases DO NOT represent a fixed geographic fence.
"""
import argparse, io, json, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q_TRAIN={'2s':2.158768, '5s':1.975130}
BOUNDARIES=[('x',0,.25),('x',0,.50),('x',0,.75),('y',1,.25),('y',1,.50),('y',1,.75)]
G=['source_group','track_key','axis','boundary','direction']
POLICIES=['nominal', 'frozen_N1', 'frozen_N2_original', 'frozen_N2_gapaware', 'frozen_N3_gapaware']

def load_from_zip(path,h):
    with zipfile.ZipFile(path) as z:
        if 'EXP3_Output' in path.name:
            prefix='results\\2s_to_'+h
            with np.load(io.BytesIO(z.read(prefix+'_exp3_frozen_ensemble_predictions.npz'))) as arr:
                pred=arr['ensemble_prediction'].astype(np.float64); truth=arr['future_abs'].astype(np.float64); hist=arr['history_abs'].astype(np.float64)
            meta=pd.read_csv(io.BytesIO(z.read(prefix+'_fresh_metadata.csv')))
            with np.load(io.BytesIO(z.read(prefix+'_fresh_dataset.npz'))) as arr: scale=arr['scale'].astype(np.float64)
            role=np.repeat('EXP3_six_video',len(meta))
        else:
            prefix='2s_to_'+h
            with np.load(io.BytesIO(z.read(prefix+'_FROZEN_PREDICTIONS_FOR_CALIBRATION.npz'))) as arr:
                pred=arr['ensemble_prediction_abs'].astype(np.float64); truth=arr['future_abs'].astype(np.float64); hist=arr['history_abs'].astype(np.float64); scale=arr['scale'].astype(np.float64)
            meta=pd.read_csv(io.BytesIO(z.read(prefix+'_fresh_metadata.csv')))
            role=meta.analysis_role.astype(str).to_numpy()
    assert pred.shape==truth.shape and hist.shape[0]==len(meta)==len(scale)
    assert hist.shape[1]==20 and pred.shape[1] in [20,50]
    assert np.isfinite(pred).all() and np.isfinite(truth).all() and np.isfinite(scale).all()
    assert (scale>0).all()
    return meta,pred,truth,hist,scale,role

def build_cases(meta,pred,truth,hist,scale,role):
    chunks=[]
    for axis,j,bound in BOUNDARIES:
        anchor=hist[:,-1,j]
        valid=np.abs(anchor-bound)>1e-10
        direction=np.where(anchor<bound,'high','low')
        s=np.where(anchor<bound,1.,-1.)
        tr=s[:,None]*(truth[:,:,j]-bound);pr=s[:,None]*(pred[:,:,j]-bound)
        true=tr.max(axis=1)>=0; nominal=pr.max(axis=1)>=0
        first_cross=np.where(tr>=0,np.arange(tr.shape[1])[None,:],999).min(axis=1)
        d=pd.DataFrame({'sample_index':np.arange(len(meta)), 'source_group':meta.source_group.astype(str).to_numpy(),
            'video':meta.video.astype(str).to_numpy(), 'elephant_group':meta.elephant_group.astype(str).to_numpy(),
            'track_key':meta.track_key.astype(str).to_numpy(), 'anchor_frame_index':meta.anchor_frame_index.to_numpy(int),
            'role':role, 'axis':axis,'boundary':bound,'direction':direction,
            'true_cross':true,'nominal':nominal,'risk_score':pr.max(axis=1),
            'scale':scale, 'first_cross_step':np.where(true,first_cross,np.nan)})
        chunks.append(d.loc[valid].copy())
    df=pd.concat(chunks,ignore_index=True)
    df=df.sort_values(G+['anchor_frame_index','sample_index']).reset_index(drop=True)
    assert not df.duplicated(G+['anchor_frame_index']).any(), 'duplicate timestamps for boundary/direction'
    grp=df.groupby(G,sort=False)
    prev_anchor=grp.anchor_frame_index.shift(1)
    gap=df.anchor_frame_index-prev_anchor
    inter=df.risk_score.to_numpy()+Q_TRAIN['2s' if pred.shape[1]==20 else '5s']*df.scale.to_numpy()>=0
    prev1=pd.Series(inter,index=df.index).groupby([df[c] for c in G],sort=False).shift(1,fill_value=False).to_numpy(bool)
    prev2=pd.Series(inter,index=df.index).groupby([df[c] for c in G],sort=False).shift(2,fill_value=False).to_numpy(bool)
    prevprev_anchor=grp.anchor_frame_index.shift(2)
    contiguous1=gap.to_numpy()==10
    contiguous2=(df.anchor_frame_index.to_numpy()-prevprev_anchor.to_numpy()==20)
    nominal=df.nominal.to_numpy(bool)
    df['intersects_q90']=inter
    df['nominal']=nominal
    df['frozen_N1']=nominal|inter
    df['frozen_N2_original']=nominal|(inter&prev1)
    df['frozen_N2_gapaware']=nominal|(inter&prev1&contiguous1)
    df['frozen_N3_gapaware']=nominal|(inter&prev1&prev2&contiguous1&contiguous2)
    df['valid_prev_stride10']=contiguous1
    return df

def confusion(d,method):
    t=d.true_cross.to_numpy(bool);a=d[method].to_numpy(bool)
    tp=int(np.sum(t&a)); fn=int(np.sum(t&~a)); fp=int(np.sum(~t&a));tn=int(np.sum(~t&~a))
    return {'TP':tp,'FN':fn,'FP':fp,'TN':tn,
      'recall':tp/(tp+fn) if tp+fn else np.nan,
      'false_warning_rate':fp/(fp+tn) if fp+tn else np.nan,
      'precision':tp/(tp+fp) if tp+fp else np.nan,
      'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
      'n_boundary_windows':len(d),'true_windows':int(t.sum()),'alert_windows':int(a.sum())}

def alert_runs(df,policy):
    """Purely algorithmic consecutive alert runs, not fixed-boundary real episodes."""
    records=[]; truths=[]
    for key,grp in df.groupby(G,sort=False):
        g=grp.sort_values('anchor_frame_index')
        anchors=g.anchor_frame_index.to_numpy(int)
        truth=g.true_cross.to_numpy(bool)
        alarm=g[policy].to_numpy(bool)
        # True positive windows are joined only when anchors are 10 indices apart.
        prior_t=False; prior_a=False
        gtstart=[]; aruns=[]
        for i,(a,t) in enumerate(zip(alarm,truth)):
            adjacent=i>0 and anchors[i]-anchors[i-1]==10
            if t and (not prior_t or not adjacent):gtstart.append([i,i])
            elif t:gtstart[-1][1]=i
            if a and (not prior_a or not adjacent): aruns.append([i,i])
            elif a:aruns[-1][1]=i
            prior_t=t;prior_a=a
        for lo,hi in gtstart:
            detected=bool(alarm[lo:hi+1].any())
            # All measured lead times are within a single forecast window,
            # NOT physical first-alert-to-fixed-fence crossing time.
            first_confirm=np.flatnonzero(alarm[lo:hi+1])
            leading=float((g.first_cross_step.iloc[lo+first_confirm[0]]+1)/10) if len(first_confirm) else np.nan
            truths.append({**dict(zip(G,key)), 'video':g.video.iloc[lo], 'role':g.role.iloc[lo],
                'truth_start_anchor':int(anchors[lo]),'truth_end_anchor':int(anchors[hi]),'truth_window_count':hi-lo+1,
                'detected_within_truth_run':detected,'first_hit_window_future_cross_seconds':leading})
        for lo,hi in aruns:
            has_pos=bool(truth[lo:hi+1].any())
            records.append({**dict(zip(G,key)),'video':g.video.iloc[lo], 'role':g.role.iloc[lo],
                'alarm_start_anchor':int(anchors[lo]),'alarm_end_anchor':int(anchors[hi]),
                'alarm_window_count':hi-lo+1,'overlaps_true_cross_window':has_pos,
                'onset_is_true_cross_window':bool(truth[lo]),
                'first_anchor_future_cross_seconds':float((g.first_cross_step.iloc[lo]+1)/10) if truth[lo] else np.nan})
    return pd.DataFrame(records),pd.DataFrame(truths)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--exp3',type=Path,default=Path('/mnt/data/EXP3_Output_Review.zip'))
    ap.add_argument('--stage5',type=Path,default=Path('/mnt/data/CP10_Stage5_Frozen_Results_Review.zip'))
    ap.add_argument('--out',type=Path,default=Path('/mnt/data/CP10_Stage6_Warning_Evidence'))
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    summary=[]; per_video=[]; per_group=[];run_summary=[];allruns=[];alltruths=[];baseline=[];wpath=[]
    for data_name,zfile in [('EXP3',args.exp3),('Stage5',args.stage5)]:
        for h in ('2s','5s'):
            meta,pred,true,hist,scale,role=load_from_zip(zfile,h)
            df=build_cases(meta,pred,true,hist,scale,role)
            # Drop Stage5 development for retrospective transfer evidence, but keep its results separate.
            for part,select in [('all',np.ones(len(df),bool))] if data_name=='EXP3' else [('development',df.role.eq('adaptation').to_numpy()),('retrospective_evaluation',df.role.ne('adaptation').to_numpy())]:
                sub=df.loc[select].copy()
                if len(sub)==0:continue
                for policy in POLICIES:
                    cm=confusion(sub,policy)
                    summary.append({'source':data_name, 'partition':part, 'horizon':h,'policy':policy,**cm})
                    runs,gt=alert_runs(sub,policy)
                    n_gt=len(gt);n_runs=len(runs)
                    run_summary.append({'source':data_name, 'partition':part, 'horizon':h,'policy':policy,
                        'truth_positive_window_runs':n_gt,'truth_run_detected':int(gt.detected_within_truth_run.sum()) if n_gt else 0,
                        'truth_run_hit_fraction':float(gt.detected_within_truth_run.mean()) if n_gt else np.nan,
                        'alert_runs':n_runs,
                        'alert_runs_without_positive_window':int((~runs.overlaps_true_cross_window).sum()) if n_runs else 0,
                        'alert_run_overlap_fraction':float(runs.overlaps_true_cross_window.mean()) if n_runs else np.nan,
                        'alert_run_onset_positive_fraction':float(runs.onset_is_true_cross_window.mean()) if n_runs else np.nan,
                        'alert_runs_per_100_boundary_windows':n_runs*100/len(sub),
                        'median_TTE_seconds_in_true_positive_alert_windows':float(np.median((sub.loc[sub.true_cross&sub[policy],'first_cross_step']+1)/10)) if cm['TP'] else np.nan})
                    for _,g in sub.groupby('video'):
                        per_video.append({'source':data_name,'partition':part,'horizon':h,'policy':policy,
                            'video':g.video.iloc[0], **confusion(g,policy)})
                    for _,g in sub.groupby('elephant_group'):
                        per_group.append({'source':data_name,'partition':part,'horizon':h,'policy':policy,
                            'elephant_group':g.elephant_group.iloc[0], **confusion(g,policy)})
                    if policy in ['nominal','frozen_N2_original','frozen_N2_gapaware','frozen_N3_gapaware']:
                        runs['source']=data_name;runs['partition']=part;runs['horizon']=h;runs['policy']=policy
                        gt['source']=data_name;gt['partition']=part;gt['horizon']=h;gt['policy']=policy
                        allruns.append(runs);alltruths.append(gt)
            df.to_csv(out/f'Stage6_{data_name}_{h}_Boundary_Window_Decisions.csv',index=False)
            print('computed',data_name,h,len(df))
    s=pd.DataFrame(summary);v=pd.DataFrame(per_video);g=pd.DataFrame(per_group);r=pd.DataFrame(run_summary)
    for name,t in [('Window_Level_Policy_Summary',s),('Per_Video_Window_Metrics',v),('Per_Group_Window_Metrics',g),('Alert_Run_Proxy_Diagnostics',r)]:
        t.to_csv(out/f'Stage6_{name}.csv',index=False)
    pd.concat(allruns,ignore_index=True).to_csv(out/'Stage6_Alert_Runs_Proxy_Detail.csv',index=False)
    pd.concat(alltruths,ignore_index=True).to_csv(out/'Stage6_True_Crossing_Window_Runs_Proxy_Detail.csv',index=False)
    # Exact reproduction check against original Step7H/Stage1 headline.
    for h,expected in [('2s',(272,42,2005,2205)),('5s',(388,100,1205,1739))]:
        z=s[(s.source=='EXP3')&(s.horizon==h)&(s.policy=='frozen_N2_original')].iloc[0]
        assert tuple(int(z[k]) for k in ['TP','FN','FP','TN'])==expected,(h,z)
    # 0.5 and 0.75 percentile decisions are pre-specified diagnostics only; no tuning on test.
    col={'nominal':'#4477AA','frozen_N1':'#EE7733','frozen_N2_original':'#CC3311','frozen_N2_gapaware':'#228833','frozen_N3_gapaware':'#AA4499'}
    for source,part in [('EXP3','all'),('Stage5','retrospective_evaluation')]:
        fig,ax=plt.subplots(figsize=(9,6),dpi=165)
        for h,marker in [('2s','o'),('5s','s')]:
            d=s[(s.source==source)&(s.partition==part)&(s.horizon==h)]
            for _,row in d.iterrows():
                ax.scatter(row.false_warning_rate*100,row.recall*100,c=col[row.policy],marker=marker,s=95)
                ax.annotate(row.policy.replace('frozen_','')+(' 2s' if h=='2s' else ' 5s'),(row.false_warning_rate*100,row.recall*100),xytext=(6,2),textcoords='offset points',fontsize=7)
        ax.grid(alpha=.18);ax.set_xlabel('False-warning rate, % of negative boundary-windows');ax.set_ylabel('Positive crossing-window recall, %')
        ax.set_title(f'{source} | {part} | synthetic boundary warning trade-off')
        ax.set_xlim(left=0);ax.set_ylim(bottom=0,top=103)
        fig.tight_layout();fig.savefig(out/f'Figure_Stage6_{source}_Warning_Tradeoff.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.5,5.2),dpi=165)
    d=r[(r.source=='EXP3')&(r.partition=='all')&(r.policy.isin(['nominal','frozen_N2_original','frozen_N3_gapaware']))]
    x=np.arange(2);w=.24
    for i,pol in enumerate(['nominal','frozen_N2_original','frozen_N3_gapaware']):
        t=d[d.policy==pol].set_index('horizon').reindex(['2s','5s'])
        ax.bar(x+(i-1)*w,t.alert_runs_without_positive_window,width=w,label=pol)
    ax.set_xticks(x,['2-second','5-second']);ax.set_ylabel('Alert runs with no positive truth window');ax.legend(fontsize=8);ax.set_title('EXP3: repeated-alert-run proxy (NOT physical geofence episodes)')
    fig.tight_layout();fig.savefig(out/'Figure_Stage6_AlertRunProxy_Comparison.png');plt.close(fig)
    (out/'Stage6_Method_Provenance.json').write_text(json.dumps({
        'input_sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.exp3,args.stage5)},
        'fixed_trainreal_q90':Q_TRAIN,'fps_assumed_for_window_TTE':10,
        'boundary_positions':[0.25,0.50,0.75],'boundary_axes':['x','y'],
        'critical_limitation':'Boundary values are defined in each window observation-start camera coordinate system. A repeated alert run is NOT an episode at one physical geofence. Ground-truth future availability is a retrospective label, not deployment evidence.',
        'not_done':['R6 per-window classifier reconstruction','real fixed-geofence episode matching','operational warning lead time','independent recording-group testing','prospective risk control'],
        'R6_limitation':'Only published aggregate metrics available from R6, no per-window classifier decisions in the selected artifact.'
    },indent=2),encoding='utf-8')
    print('\nEXP3 window-level')
    print(s[(s.source=='EXP3')&(s.policy.isin(['nominal','frozen_N2_original','frozen_N3_gapaware']))][['horizon','policy','TP','FN','FP','TN','recall','false_warning_rate','precision']].to_string(index=False))
    print('\nEXP3 run-level proxy')
    print(r[(r.source=='EXP3')&(r.policy.isin(['nominal','frozen_N2_original','frozen_N3_gapaware']))][['horizon','policy','truth_positive_window_runs','truth_run_hit_fraction','alert_runs','alert_runs_without_positive_window','alert_run_onset_positive_fraction']].to_string(index=False))
    print('\nStage5 eval window-level')
    print(s[(s.source=='Stage5')&(s.partition=='retrospective_evaluation')&(s.policy.isin(['nominal','frozen_N2_original','frozen_N3_gapaware']))][['horizon','policy','TP','FN','FP','TN','recall','false_warning_rate','precision']].to_string(index=False))

if __name__=='__main__':main()
