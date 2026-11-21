#!/usr/bin/env python3
"""Stage 5 retrospective pathwise conformal transfer audit.
Inputs: untouched WSL frozen inference result zip. Does NOT use evaluation labels to choose thresholds.
Requires numpy, pandas, matplotlib. Run:
  python analyze_stage5.py --input CP10_Stage5_Frozen_Results_Review.zip --out stage5_audit
"""
import argparse, csv, hashlib, io, json, math, os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q_FROZEN={'2s_to_2s': 2.158768, '2s_to_5s': 1.975130}


def conformal_q(scores, alpha=.1):
    v = np.sort(np.array(scores, dtype=float))
    k=int(math.ceil((len(v)+1)*(1-alpha)))
    # If k>n, split-conformal theoretical upper quantile is +infinity.
    return float(v[k-1]) if k <= len(v) else math.inf, k


def run(zipfile_path, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    p=Path(zipfile_path)
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    summ=[]; pervideo=[]; pertrack=[]; samples=[]; diagnostics=[]
    with zipfile.ZipFile(p) as z:
      for horizon in ('2s_to_2s', '2s_to_5s'):
        zz=np.load(io.BytesIO(z.read(horizon+'_FROZEN_PREDICTIONS_FOR_CALIBRATION.npz')))
        md=pd.read_csv(io.BytesIO(z.read(horizon+'_fresh_metadata.csv')))
        assert len(md)==len(zz['ensemble_prediction_abs'])==len(zz['future_abs'])==len(zz['scale'])
        assert np.all(np.isfinite(zz['ensemble_prediction_abs']))
        assert np.all(zz['scale']>0)
        err=np.linalg.norm(zz['ensemble_prediction_abs']-zz['future_abs'],axis=2)
        sc=zz['scale'].astype(float)
        score=err.max(axis=1)/sc
        md=md.copy()
        md['score']=score;md['mean_point_error']=err.mean(axis=1)
        md['final_point_error']=err[:,-1];md['scale_from_predictions']=sc
        md['floor_scale_flag']=np.isclose(sc,sc.min(),rtol=1e-10,atol=1e-12)
        d=md[md.analysis_role=='adaptation']; v=md[md.analysis_role=='evaluation']
        assert set(d.video).isdisjoint(set(v.video)), 'Video leakage between adaptation and evaluation'
        assert len(set(d.video)) == 1 and len(set(v.video)) == 2
        qdev,order=conformal_q(d.score)
        original=Q_FROZEN[horizon]
        thresholds={
          'Frozen TrainReal q90':original,
          'Development-only split-conformal q90':qdev,
          'Development maximum score (diagnostic)':float(d.score.max()),
          'No-shrink safeguard max(q_frozen,q_dev)':max(original,qdev),
        }
        for mode,frame in [('development',d),('evaluation',v)]:
          for meth,q in thresholds.items():
            idx=frame.index.to_numpy()
            s=frame.score.to_numpy()
            hits=(s<=q+1e-12)
            point=(err[idx] <= q*sc[idx,None]+1e-12)
            summ.append(dict(horizon=horizon,split=mode,method=meth,threshold_q=q,covered=int(hits.sum()),n=int(len(idx)),pathwise_coverage=float(hits.mean()),pointwise_coverage=float(point.mean()),mean_absolute_radius=float(np.mean(q*sc[idx])),relative_radius_to_original=float(q/original),q_calibration_order=order,nominal_target=0.9))
        for video,frame in md.groupby('video',sort=True):
          for meth,q in thresholds.items():
            x=frame.score.to_numpy(); ind=frame.index.to_numpy()
            pervideo.append(dict(horizon=horizon,role=frame.analysis_role.iloc[0],video=video,method=meth,threshold_q=q,n=len(frame),covered=int((x<=q+1e-12).sum()),pathwise_coverage=float(np.mean(x<=q+1e-12)),ADE=float(np.mean(err[ind])),FDE=float(np.mean(err[ind,-1])),mean_scale=float(sc[ind].mean()),floor_scale_count=int(np.sum(frame.floor_scale_flag))))
        for (role,video,seg,track),frame in md.groupby(['analysis_role','video','segment','track_id'],sort=True):
          x=frame.score.to_numpy()
          pertrack.append(dict(horizon=horizon,role=role,video=video,segment=seg,track_id=str(track),windows=len(frame),score_median=float(np.median(x)),score_max=float(x.max()),score_p90=float(np.quantile(x,.9)),frozen_coverage=float(np.mean(x<=original)),dev_q_coverage=float(np.mean(x<=qdev)),scale_mean=float(frame.scale_from_predictions.mean())))
        for i,r in md.iterrows():
          samples.append(dict(horizon=horizon,role=r.analysis_role,video=r.video,segment=r.segment,track_id=str(r.track_id),sample_id=int(r.sample_id),score=float(r.score),scale=float(r.scale_from_predictions),ADE=float(r.mean_point_error),FDE=float(r.final_point_error),covered_frozen=bool(r.score<=original+1e-12),covered_dev=bool(r.score<=qdev+1e-12)))
        diagnostics.append(dict(horizon=horizon,n_dev=len(d),n_eval=len(v),n_dev_videos=d.video.nunique(),n_eval_videos=v.video.nunique(),n_dev_tracks=d.track_key.nunique(),n_eval_tracks=v.track_key.nunique(),dev_q=qdev,frozen_q=original,dev_q_ratio_to_frozen=qdev/original,dev_max_score=float(d.score.max()),eval_min_score=float(v.score.min()),floor_scale=float(sc.min()),floor_count_dev=int(d.floor_scale_flag.sum()),floor_count_eval=int(v.floor_scale_flag.sum()),scale_dev_range=[float(d.scale_from_predictions.min()),float(d.scale_from_predictions.max())],scale_eval_range=[float(v.scale_from_predictions.min()),float(v.scale_from_predictions.max())],dev_score_p90=float(np.quantile(d.score,.9)),eval_score_p90=float(np.quantile(v.score,.9)),mean_ADE_dev=float(d.mean_point_error.mean()),mean_ADE_eval=float(v.mean_point_error.mean())))
    pd.DataFrame(summ).to_csv(output/'Stage5_Calibration_Coverage_Summary.csv',index=False)
    pd.DataFrame(pervideo).to_csv(output/'Stage5_Calibration_By_Video.csv',index=False)
    pd.DataFrame(pertrack).to_csv(output/'Stage5_Calibration_By_Track.csv',index=False)
    pd.DataFrame(samples).to_csv(output/'Stage5_Individual_Window_Diagnostics.csv',index=False)
    (output/'Stage5_Provenance_And_Diagnostics.json').write_text(json.dumps(dict(source_zip=p.name,sha256=sha,threshold_source='Stage1 reproduced EXP3 TrainReal q90',original_q=Q_FROZEN,method='per-window pathwise maximum Euclidean error / scale; q_dev=sorted(scores)[ceil((n+1)*0.9)-1]',data_roles='development 0006_V only; evaluation two 0004_V videos only',limitations=['previously inspected same recording groups','window overlap; nominal 90% guarantee not established across shift','sample eligibility uses GT matching','no confirmatory independent holdout','segment counts are not independent samples'],diagnostics=diagnostics),indent=2))
    s=pd.DataFrame(summ);pv=pd.DataFrame(pervideo);x=pd.DataFrame(samples)
    # Figure 1: grouped coverage for evaluated sample set.
    e=s[(s['split']=='evaluation') & (s['method'].isin(['Frozen TrainReal q90','Development-only split-conformal q90']))]
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),sharey=True)
    for ax,h,title in zip(axes,['2s_to_2s','2s_to_5s'],['2-second prediction','5-second prediction']):
      a=e[e.horizon==h].set_index('method')
      methods=['Frozen TrainReal q90','Development-only split-conformal q90']
      vals=[float(a.loc[k,'pathwise_coverage'])*100 for k in methods]
      bars=ax.bar([0,1],vals,width=.55)
      ax.set_xticks([0,1],['Frozen\nTrainReal q90','Development-only\nq90'])
      ax.axhline(90,linestyle='--',linewidth=1,label='Nominal 90%')
      ax.set_ylim(0,110);ax.set_title(title,fontsize=13)
      for bar,k in zip(bars,methods):
        v=a.loc[k];ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()-6 if bar.get_height()>50 else bar.get_height()+2, f"{int(v['covered'])}/{int(v['n'])}  ({v['pathwise_coverage']*100:.1f}%)",ha='center',fontsize=10,color='white' if bar.get_height()>50 else 'black')
      ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Full-path simultaneous coverage (%)')
    fig.suptitle('Retrospective evaluation: calibration transfer failure',fontsize=15,y=1.02)
    fig.tight_layout()
    fig.savefig(output/'Figure_Stage5_Retrospective_Coverage.png',dpi=240,bbox_inches='tight');plt.close(fig)
    # Figure 2: score distribution and thresholds
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),sharey=False)
    for ax,h,title in zip(axs,['2s_to_2s','2s_to_5s'],['2-second prediction','5-second prediction']):
      y=x[x.horizon==h]; dev=y[y.role=='adaptation'].score.values;ev=y[y.role=='evaluation'].score.values
      ax.boxplot([dev,ev],positions=[1,2],widths=.45,showfliers=True)
      j=np.random.default_rng(23).uniform(-.10,.10,size=len(dev));ax.scatter(np.ones(len(dev))+j,dev,alpha=.48,s=14)
      j=np.random.default_rng(25).uniform(-.10,.10,size=len(ev));ax.scatter(np.ones(len(ev))*2+j,ev,alpha=.48,s=14)
      d=next(z for z in diagnostics if z['horizon']==h)
      ax.axhline(d['dev_q'],linestyle='--',linewidth=1.3,label=f"Dev-only q90 = {d['dev_q']:.3f}")
      ax.axhline(d['frozen_q'],linestyle=':',linewidth=1.5,label=f"Frozen q90 = {d['frozen_q']:.3f}")
      ax.set_xticks([1,2],['Adaptation video','Evaluation videos']);ax.set_ylabel('Normalized max-path residual')
      ax.set_title(title);ax.legend(fontsize=9,loc='upper left');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Residual-score shift between development and evaluation videos',fontsize=14,y=1.02)
    fig.tight_layout();fig.savefig(output/'Figure_Stage5_Score_Distribution_Shift.png',dpi=240,bbox_inches='tight');plt.close(fig)
    # Figure 3: source-specific coverage
    e=pv[(pv.role=='evaluation') & pv.method.isin(['Frozen TrainReal q90','Development-only split-conformal q90'])]
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,h,title in zip(axs,['2s_to_2s','2s_to_5s'],['2-second','5-second']):
      t=e[e.horizon==h]
      vs=sorted(t.video.unique());short=[v.replace('DJI_','').replace('_V','')[-12:] for v in vs]
      w=.36;xx=np.arange(len(vs))
      for off,met,label in [(-w/2,'Frozen TrainReal q90','Frozen q90'),(w/2,'Development-only split-conformal q90','Development-only q90')]:
        y=[100*float(t[(t.video==v)&(t.method==met)].pathwise_coverage.iloc[0]) for v in vs]
        bars=ax.bar(xx+off,y,w,label=label)
        for bar in bars:
          ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{bar.get_height():.1f}',ha='center',fontsize=8)
      ax.set_xticks(xx,[v.split('DJI_')[-1].replace('_V','')[-12:] for v in vs])
      ax.set_title(title);ax.axhline(90,linestyle='--',linewidth=1);ax.grid(axis='y',alpha=.15)
    axs[0].set_ylabel('Full-path coverage (%)');axs[0].set_ylim(0,112)
    axs[1].legend(loc='upper right',fontsize=8)
    fig.suptitle('Per-video variability (only two evaluation videos)',fontsize=14,y=1.02)
    fig.tight_layout();fig.savefig(output/'Figure_Stage5_PerVideo_Coverage.png',dpi=240,bbox_inches='tight');plt.close(fig)
    return summ, diagnostics, pv

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--input',required=True)
    a.add_argument('--out',required=True)
    args=a.parse_args()
    summ,diag,pv=run(args.input,args.out)
    for d in diag:
        print(d['horizon'], 'n_dev',d['n_dev'],'n_eval',d['n_eval'],'dev_q',d['dev_q'],'frozen',d['frozen_q'],'floor dev/eval',d['floor_count_dev'],d['floor_count_eval'])
    print(pd.DataFrame(summ).query("split=='evaluation'")[['horizon','method','covered','n','pathwise_coverage','relative_radius_to_original']].to_string(index=False))
