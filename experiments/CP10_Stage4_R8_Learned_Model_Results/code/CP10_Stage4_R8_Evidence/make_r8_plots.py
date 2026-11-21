from pathlib import Path
import pandas as pd,numpy as np,matplotlib.pyplot as plt
r=Path(__file__).resolve().parent
m=pd.read_csv(r/'R8_Common_Target_Test_Summary.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=False)
for i,(task,h) in enumerate([('2s_to_2s','2-second horizon'),('2s_to_5s','5-second horizon')]):
 d=m[m.task==task].set_index('model').loc[['Stationary','ConstantVelocity']];x=np.arange(2);w=.33;ax=axes[i]
 ax.bar(x-w/2,d['raw_ADE'],width=w,label='Raw-coordinate prediction\nregistered to common target',color='#8f9cad')
 ax.bar(x+w/2,d['corrected_ADE'],width=w,label='Camera-motion-corrected prediction',color='#1b7f83')
 for j,(raw,corr) in enumerate(zip(d.raw_ADE,d.corrected_ADE)):
  ax.text(j-w/2,raw+raw*.02,f'{raw:.3f}',ha='center',va='bottom',fontsize=9)
  ax.text(j+w/2,corr+corr*.02,f'{corr:.3f}',ha='center',va='bottom',fontsize=9)
 ax.set_xticks(x,['Stationary','Constant velocity']);ax.set_ylabel('ADE (normalized image coordinates)');ax.set_title(h,weight='bold');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 ax.set_ylim(0,max(d.raw_ADE)*1.23)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False)
fig.suptitle('R8 paired common-target baseline comparison (BIRDSAI TestReal)',fontsize=13,weight='bold')
fig.text(.5,.91,'Raw forecasts are registered using recorded future affine transforms for retrospective evaluation only',ha='center',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.09,1,.87]);fig.savefig(r/'Figure_R8_Common_Target_CV_Stationary.png',dpi=220,bbox_inches='tight');plt.close(fig)

a=pd.read_csv(r/'R8_Reconstruction_2792_Sample_Audit.csv')
summary=a.groupby(['task','partition'],as_index=False).agg(samples=('source_index','size'),max_error=('max_rebuild_discrepancy','max'),p95_error=('max_rebuild_discrepancy',lambda x:np.quantile(x,.95)))
summary.to_csv(r/'R8_Exact_Reconstruction_Summary.csv',index=False)
