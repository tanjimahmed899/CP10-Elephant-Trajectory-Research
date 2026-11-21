#!/usr/bin/env python3
"""Independent Stage7F post-hoc metric audit from saved frozen prediction arrays.

No model inference, no GT-label selection, no checkpoint reconstruction. The sole
input is the completed-or-partial Stage7F review ZIP containing NPZ predictions.
Outputs are window-level and within-sequence GT-identity macro diagnostics.
"""
import argparse, io, json, hashlib, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def stats_error(pred,y):
    err=np.linalg.norm(pred-y,axis=-1)
    if not np.isfinite(err).all(): raise ValueError('Nonfinite trajectory error')
    return err.mean(axis=1),err[:,-1],err.max(axis=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True,help='Stage7F_Tracker_Forecast_Review.zip')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    original=args.input.read_bytes(); source_sha=hashlib.sha256(original).hexdigest()
    tables=[]; summary=[]; ids=[]; input_diag=[]; pairs=[]; audit=[]
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        if z.testzip() is not None: raise ValueError('Bad ZIP CRC')
        m=pd.read_csv(io.BytesIO(z.read('Stage7F_Verified_Evaluation_Windows.csv')),dtype={'sequence':str})
        if len(m)!=311 or set(m.horizon)!={'2s','5s'} or m.sequence.nunique()!=1:raise ValueError('Unexpected source population')
        audit.append(f'Original ZIP sha256 {source_sha}')
        for horizon,tag,n in [('2s','2s_to_2s',187),('5s','2s_to_5s',124)]:
            met=m[m.horizon==horizon].reset_index(drop=True).copy()
            if len(met)!=n: raise ValueError(f'{horizon} unexpected window count')
            with np.load(io.BytesIO(z.read(f'Stage7F_{tag}_Predictions.npz')),allow_pickle=False) as a:
                want={'gt_future','tracker_history','oracle_GT_history','tracker_ensemble_pred_abs','oracle_ensemble_pred_abs'}
                if set(a.files)!=want:raise ValueError(f'NPZ schema differs: {set(a.files)}')
                y=a['gt_future'].copy(); ht=a['tracker_history'].copy(); hg=a['oracle_GT_history'].copy(); pt=a['tracker_ensemble_pred_abs'].copy();pg=a['oracle_ensemble_pred_abs'].copy()
            T=20 if horizon=='2s' else 50
            if y.shape!=(n,T,2) or ht.shape!=(n,20,2) or hg.shape!=(n,20,2):raise ValueError(f'{horizon} shape mismatch')
            if not all(np.isfinite(v).all() for v in (y,ht,hg,pt,pg)):raise ValueError('Nonfinite inputs or predictions')
            methods={
                'Tracker history / Transformer ensemble':pt,
                'GT history / Transformer ensemble (oracle diagnostic)':pg,
                'Tracker history / Stationary':np.broadcast_to(ht[:,-1:,:],y.shape),
                'GT history / Stationary (oracle diagnostic)':np.broadcast_to(hg[:,-1:,:],y.shape),
            }
            dists={}
            for label,pred in methods.items():
                ade,fde,maxe=stats_error(pred,y)
                dists[label]=ade
                t=met[['sequence','horizon','track_id','GT_id_OFFLINE_AUDIT','start_frame','anchor_frame','future_end_frame']].copy()
                t['input_and_method']=label;t['ADE']=ade;t['FDE']=fde;t['max_AE']=maxe
                tables.append(t)
                summary.append({'horizon':horizon,'input_and_method':label,'n_windows':n,
                                'n_GT_elephants':met.GT_id_OFFLINE_AUDIT.nunique(),'n_tracked_fragments':met.track_id.nunique(),
                                'ADE_mean':float(ade.mean()),'FDE_mean':float(fde.mean()),
                                'ADE_median':float(np.median(ade)), 'FDE_median':float(np.median(fde)),
                                'ADE_p95':float(np.percentile(ade,95))})
            delta=dists['Tracker history / Transformer ensemble']-dists['Tracker history / Stationary']
            delta_oracle=dists['Tracker history / Transformer ensemble']-dists['GT history / Transformer ensemble (oracle diagnostic)']
            p=met[['sequence','horizon','track_id','GT_id_OFFLINE_AUDIT','start_frame','anchor_frame']].copy()
            p['ADE_tracker_ensemble']=dists['Tracker history / Transformer ensemble']
            p['ADE_oracle_ensemble']=dists['GT history / Transformer ensemble (oracle diagnostic)']
            p['ADE_tracker_stationary']=dists['Tracker history / Stationary']
            p['ADE_oracle_stationary']=dists['GT history / Stationary (oracle diagnostic)']
            p['ADE_delta_tracker_ensemble_minus_tracker_stationary']=delta
            p['ADE_delta_tracker_ensemble_minus_oracle_ensemble']=delta_oracle
            p['history_mean_tracker_GT_distance']=np.linalg.norm(ht-hg,axis=-1).mean(axis=1)
            pairs.append(p)
            for g,block in p.groupby('GT_id_OFFLINE_AUDIT'):
                ids.append({'horizon':horizon,'GT_id':g,'n_windows':len(block),
                            'tracker_transformer_ADE_macro_component':block.ADE_tracker_ensemble.mean(),
                            'GT_oracle_transformer_ADE_macro_component':block.ADE_oracle_ensemble.mean(),
                            'tracker_stationary_ADE_macro_component':block.ADE_tracker_stationary.mean(),
                            'tracker_minus_stationary_ADE':block.ADE_delta_tracker_ensemble_minus_tracker_stationary.mean(),
                            'tracker_minus_GT_oracle_ADE':block.ADE_delta_tracker_ensemble_minus_oracle_ensemble.mean()})
            input_diag.append({'horizon':horizon,'n_windows':n,'mean_tracker_vs_GT_history_dist':float(np.linalg.norm(ht-hg,axis=-1).mean()),
                               'p95_sample_history_mean_dist':float(np.percentile(p.history_mean_tracker_GT_distance,95)),
                               'max_history_point_dist':float(np.linalg.norm(ht-hg,axis=-1).max()),
                               'tracker_ensemble_better_than_stationary_windows':int(np.sum(delta<0)),
                               'tracker_ensemble_better_than_oracle_windows':int(np.sum(delta_oracle<0))})
        console=z.read('Stage7F_Console.log').decode('utf-8',errors='replace')
        audit.append('Console reported provenance gate passed: '+str('PROVENANCE OK' in console))
        audit.append('Console reported all 6 seed+horizon evaluations: '+str(all(f'evaluated {tag} seed {seed}' in console for tag in ('2s_to_2s','2s_to_5s') for seed in (42,52,62))))
        audit.append('Console reported final pandas DataFrame(rows) error: '+str('ValueError: setting an array element with a sequence' in console))
    samples=pd.concat(tables,ignore_index=True); s=pd.DataFrame(summary); by_gt=pd.DataFrame(ids); diagnostics=pd.DataFrame(input_diag); paired=pd.concat(pairs,ignore_index=True)
    s.to_csv(args.out/'Stage7F_Independent_Ensemble_Summary.csv',index=False)
    samples.to_csv(args.out/'Stage7F_Independent_Sample_Level.csv',index=False)
    by_gt.to_csv(args.out/'Stage7F_Independent_Per_GT_Identity.csv',index=False)
    diagnostics.to_csv(args.out/'Stage7F_Independent_Input_Diagnostics.csv',index=False)
    paired.to_csv(args.out/'Stage7F_Independent_Paired_Window_Differences.csv',index=False)
    # Macro-level descriptive stats prevent large overlapping-window clusters dominating one estimate.
    rows=[]
    for h,b in by_gt.groupby('horizon'):
        rows.append({'horizon':h,'n_GT_identities':len(b),
          'macro_tracker_ensemble_ADE':b.tracker_transformer_ADE_macro_component.mean(),
          'macro_GT_oracle_ensemble_ADE':b.GT_oracle_transformer_ADE_macro_component.mean(),
          'macro_tracker_stationary_ADE':b.tracker_stationary_ADE_macro_component.mean(),
          'GT_identities_tracker_ensemble_better_than_tracker_stationary':int((b.tracker_minus_stationary_ADE<0).sum()),
          'GT_identities_tracker_ensemble_better_than_GT_oracle':int((b.tracker_minus_GT_oracle_ADE<0).sum())})
    macro=pd.DataFrame(rows);macro.to_csv(args.out/'Stage7F_Independent_GT_Macro_Results.csv',index=False)
    manifest={'source_file':args.input.name,'source_sha256':source_sha,'input_n_windows':311,'source_sequence':m.sequence.unique().tolist(),
      'predictions_frozen_seed_ensemble':True,'per_seed_outputs_available_in_uploaded_zip':False,
      'metric_protocol':'ADE=mean_t Euclidean(pred_future-GT_future); FDE=Euclidean final prediction-GT',
      'coordinate_units':'normalized image-coordinate units in observation-start affine-reference plane',
      'cautions':['Only GT-linked strict windows, NOT blind end-to-end','All windows in one test sequence',
       'Overlapping windows correlated by GT identity','GT-history is an offline oracle diagnostic',
       'No direct original-detector forecasting comparison (0 strict windows)','Models not retrained in this report'],
      'log_checks':audit}
    (args.out/'Stage7F_Independent_Audit_Manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # plotting optional to keep core analysis portable
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9.3,5.4))
        ix=np.arange(2);labels=['2 seconds','5 seconds']; w=.24
        names=['Tracker history / Transformer ensemble','GT history / Transformer ensemble (oracle diagnostic)','Tracker history / Stationary']
        for i,name in enumerate(names):
            y=[s[(s.horizon==h)&(s.input_and_method==name)]['ADE_mean'].iloc[0] for h in ('2s','5s')]
            ax.bar(ix+(i-1)*w,y,width=w,label=name.replace(' (oracle diagnostic)',''))
        ax.set_xticks(ix);ax.set_xticklabels(labels);ax.set_ylabel('Mean ADE (normalized image coordinates)')
        ax.set_title('Stage 7F: frozen Transformer vs stationary, strict GT-linked windows')
        ax.legend(loc='upper left',fontsize=8);ax.grid(axis='y',alpha=.25)
        ax.text(.5,-.2,'One test sequence; 187 / 124 overlapping windows; GT selected offline',transform=ax.transAxes,ha='center',fontsize=9)
        fig.tight_layout();fig.savefig(args.out/'Figure_Stage7F_Ensemble_vs_Stationary.png',dpi=190,bbox_inches='tight');plt.close(fig)
        fig,ax=plt.subplots(figsize=(9.3,5.2))
        for h in ('2s','5s'):
            a=paired[paired.horizon==h]
            delta=a.ADE_delta_tracker_ensemble_minus_tracker_stationary.to_numpy()
            ax.hist(delta,bins=30,alpha=.55,label=f'{h} (n={len(a)})')
        ax.axvline(0,color='black',linestyle='--',linewidth=1)
        ax.set_xlabel('Window ADE: Tracker Transformer minus tracker stationary (negative = improvement)')
        ax.set_ylabel('Number of overlapping windows');ax.set_title('Stage 7F: within-window gains and failures')
        ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
        fig.savefig(args.out/'Figure_Stage7F_Paired_Improvement_Distribution.png',dpi=190,bbox_inches='tight');plt.close(fig)
    except ImportError:pass
    print('Wrote',args.out,'\n',s.to_string(index=False),'\n',macro.to_string(index=False))

if __name__=='__main__':main()
