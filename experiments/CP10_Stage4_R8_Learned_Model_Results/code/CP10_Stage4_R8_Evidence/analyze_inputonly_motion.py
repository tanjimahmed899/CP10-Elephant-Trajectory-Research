from pathlib import Path
import pandas as pd,numpy as np,matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
rawfolders=[p/'learned_input_only',p/'learned_input_only_extra'];corfolders=[p/'learned_pilot',p/'learned_additional_seeds']
rows=[]
for t in ['2s_to_2s','2s_to_5s']:
 z=np.load(p/f'R8_Paired_{t}_test.npz'); q=z['Q_future'].astype(float); drift=np.linalg.norm(np.einsum('ntij,j->nti',q,np.array([.5,.5,1]))-.5,axis=-1).mean(1)
 for seed in [42,52,62]:
  def find(folders,br):
   return next(folder/f'R8_Learned_{t}_{br}_seed{seed}_samples.csv' for folder in folders if (folder/f'R8_Learned_{t}_{br}_seed{seed}_samples.csv').exists())
  r=pd.read_csv(find(rawfolders,'raw'),dtype={'source_group':str,'track_key':str})
  c=pd.read_csv(find(corfolders,'corrected'),dtype={'source_group':str,'track_key':str})
  z2=r.merge(c,on=['source_index','source_group','track_key','task','seed'],suffixes=('_raw','_corrected'),validate='one_to_one')
  assert len(z2)==len(drift)
  z2['camera_motion_proxy']=drift
  rows.append(z2)
all=pd.concat(rows,ignore_index=True)
# save with only input-only physical targets
all.to_csv(p/'R8_InputOnly_PairedSample_Drift.csv',index=False)
qrows=[];srows=[]
for task,x in all.groupby('task'):
 boundaries=np.quantile(x.camera_motion_proxy,[0,.25,.5,.75,1]);boundaries=np.unique(boundaries)
 for k in range(len(boundaries)-1):
  v=x[(x.camera_motion_proxy>=boundaries[k]) & (x.camera_motion_proxy<=boundaries[k+1] if k==len(boundaries)-2 else x.camera_motion_proxy<boundaries[k+1])]
  qrows.append({'task':task,'quartile':k+1,'unique_windows':v.source_index.nunique(),'motion_proxy_min':v.camera_motion_proxy.min(),'motion_proxy_max':v.camera_motion_proxy.max(),'raw_ADE':v.ADE_raw.mean(),'corrected_ADE':v.ADE_corrected.mean(),'reduction_pct':(1-v.ADE_corrected.mean()/v.ADE_raw.mean())*100})
 for source,v in x.groupby('source_group'):
  srows.append({'task':task,'source_group':source,'unique_windows':v.source_index.nunique(),'raw_ADE':v.ADE_raw.mean(),'corrected_ADE':v.ADE_corrected.mean(),'reduction_pct':(1-v.ADE_corrected.mean()/v.ADE_raw.mean())*100})
quartile=pd.DataFrame(qrows);source=pd.DataFrame(srows)
quartile.to_csv(p/'R8_InputOnly_ADEReduction_by_CameraMotion.csv',index=False)
source.to_csv(p/'R8_InputOnly_ADEReduction_by_SourceGroup.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=False)
for ax,task in zip(axes,['2s_to_2s','2s_to_5s']):
 d=quartile[quartile.task==task];ax.plot(d.quartile,d.raw_ADE,marker='o',lw=2,color='#7e96ab',label='Raw-history');ax.plot(d.quartile,d.corrected_ADE,marker='o',lw=2,color='#1a8778',label='Corrected-history');ax.set_xticks([1,2,3,4]);ax.set_xlabel('Future frame-to-anchor camera displacement (quartile)');ax.set_ylabel('Common-target ADE');ax.set_title(task.replace('2s_to_','')+' horizon');ax.grid(alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.045));fig.suptitle('Input-only Transformer ablation by camera-motion proxy',fontsize=13,fontweight='bold');fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(p/'Figure_R8_InputOnly_CameraMotion_Sensitivity.png',dpi=220,bbox_inches='tight');plt.close(fig)
print('source\n',source.to_string(index=False));print('quartile\n',quartile.to_string(index=False))
