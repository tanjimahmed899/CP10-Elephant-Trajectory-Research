from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
base=Path(__file__).resolve().parent
parts=[p for p in [base/'learned_pilot',base/'learned_additional_seeds'] if (p/'R8_Learned_Test_Summary.csv').exists()]
df=pd.concat([pd.read_csv(p/'R8_Learned_Test_Summary.csv') for p in parts],ignore_index=True)
df=df.drop_duplicates(['task','branch','seed'],keep='last').sort_values(['task','seed','branch'])
df.to_csv(base/'R8_ThreeSeed_Learned_Results.csv',index=False)
print('Branch results\n',df[['task','seed','branch','test_ADE','test_FDE','test_track_macro_ADE']].to_string(index=False))
assert len(df)==12, f'Expected 2 tasks * 3 seeds * 2 branches =12, got {len(df)}'
sum_rows=[]
for t,g in df.groupby('task'):
 r=g[g.branch=='raw'].set_index('seed');c=g[g.branch=='corrected'].set_index('seed')
 for metric in ['test_ADE','test_FDE','test_track_macro_ADE','test_track_macro_FDE']:
  red=(1-c[metric]/r[metric])*100
  sum_rows.append({'task':t,'metric':metric,'raw_mean':r[metric].mean(),'raw_sd':r[metric].std(ddof=1),'corrected_mean':c[metric].mean(),'corrected_sd':c[metric].std(ddof=1),'reduction_pct_mean':red.mean(),'reduction_pct_sd':red.std(ddof=1),'min_reduction_pct':red.min(),'max_reduction_pct':red.max()})
pd.DataFrame(sum_rows).to_csv(base/'R8_ThreeSeed_Summary.csv',index=False)
# Pair samples by exact source-index AND seed, track level source view
pairs=[]
for t in ['2s_to_2s','2s_to_5s']:
 for seed in (42,52,62):
  frames={}
  for br in ('raw','corrected'):
   location=next((p/f'R8_Learned_{t}_{br}_seed{seed}_samples.csv' for p in parts if (p/f'R8_Learned_{t}_{br}_seed{seed}_samples.csv').exists()),None)
   if location is None:raise RuntimeError(f'Missing sample CSV {t} {seed} {br}')
   frames[br]=pd.read_csv(location,dtype={'source_group':str,'track_key':str})
  z=frames['raw'].merge(frames['corrected'],on=['source_index','source_group','track_key','task','seed'],suffixes=('_raw','_corrected'),validate='one_to_one')
  assert len(z)==len(frames['raw'])==len(frames['corrected'])
  pairedroot=base/f'R8_Paired_{t}_test.npz'
  Q=np.load(pairedroot)['Q_future'].astype(float)
  drift=np.sqrt(np.sum((np.einsum('ntij,j->nti',Q,np.array([.5,.5,1.]))-.5)**2,axis=-1)).mean(axis=1)
  z['camera_drift_norm']=drift
  pairs.append(z)
samp=pd.concat(pairs,ignore_index=True)
samp.to_csv(base/'R8_Learned_ThreeSeed_Paired_Sample_Metrics.csv',index=False)
# quartile per horizon, matched common source frame, report deviations by motion magnitude
rows=[]
for task,group in samp.groupby('task'):
 cuts=np.quantile(group.camera_drift_norm,[0,.25,.5,.75,1]);cuts=np.unique(cuts)
 for i in range(len(cuts)-1):
  s=group[(group.camera_drift_norm>=cuts[i]) & (group.camera_drift_norm < cuts[i+1] if i<len(cuts)-2 else group.camera_drift_norm<=cuts[i+1])]
  rows.append({'task':task,'drift_bin':i+1,'n_rows_all_seeds':len(s),'motion_drift_min':s.camera_drift_norm.min(),'motion_drift_max':s.camera_drift_norm.max(),'raw_ADE':s.ADE_raw.mean(),'corrected_ADE':s.ADE_corrected.mean(),'ADE_pct_reduction':(1-s.ADE_corrected.mean()/s.ADE_raw.mean())*100})
qdf=pd.DataFrame(rows);qdf.to_csv(base/'R8_Learned_Improvement_by_CameraMotion.csv',index=False)
# Fig1: paired 3seed
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,(task,label) in zip(axes,[('2s_to_2s','2-second horizon'),('2s_to_5s','5-second horizon')]):
 x=[0,1];
 for i,branch in enumerate(['raw','corrected']):
  v=df[(df.task==task)&(df.branch==branch)].test_ADE.to_numpy()
  ax.bar(i,v.mean(),yerr=v.std(ddof=1),width=.52,capsize=4,color=['#99a6b3','#1b7f83'][i]);ax.text(i,v.mean()+v.std(ddof=1)+.003,f'{v.mean():.4f}',ha='center',fontsize=10)
 ax.set_xticks(x,['Raw-history +\nraw-target','Corrected-history +\ncorrected-target']);ax.set_title(label,fontweight='bold');ax.set_ylabel('TestReal ADE (normalized image coordinates)');ax.set_ylim(0,max(df[df.task==task].test_ADE)*1.28);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Same-architecture Transformer: matched common-target evaluation',fontweight='bold',fontsize=13)
fig.text(.5,.90,'Means and standard deviations across 3 training seeds; retrospective BIRDSAI TestReal',ha='center',fontsize=9)
fig.tight_layout(rect=[0,0,1,.88]);fig.savefig(base/'Figure_R8_ThreeSeed_Learned_CommonTarget.png',dpi=220,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
for ax,(task,label) in zip(axes,[('2s_to_2s','2-second'),('2s_to_5s','5-second')]):
 d=qdf[qdf.task==task];ax.plot(d.drift_bin,d.raw_ADE,'o-',label='Raw forecast (registered)',color='#8697a6',lw=2);ax.plot(d.drift_bin,d.corrected_ADE,'o-',label='Corrected forecast',color='#16857b',lw=2);ax.set_xticks([1,2,3,4]);ax.set_xlabel('Future camera-motion proxy quartile (low → high)');ax.set_title(label);ax.grid(alpha=.2);ax.set_ylabel('TestReal ADE');ax.set_ylim(bottom=0)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.045));fig.suptitle('Sensitivity to estimated frame-to-anchor affine displacement',fontsize=13,fontweight='bold');fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(base/'Figure_R8_ADE_by_EgoMotionProxy.png',dpi=220,bbox_inches='tight');plt.close(fig)
print('\nSummary\n',pd.DataFrame(sum_rows).to_string(index=False));print('\nDrift quartiles\n',qdf.to_string(index=False))
