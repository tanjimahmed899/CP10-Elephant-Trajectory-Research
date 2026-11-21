from pathlib import Path
import pandas as pd,numpy as np,matplotlib.pyplot as plt
base=Path(__file__).resolve().parent
paths=[base/'learned_input_only',base/'learned_input_only_extra']
allrows=[]
for p in paths:
 if (p/'R8_Learned_Test_Summary.csv').exists():allrows.append(pd.read_csv(p/'R8_Learned_Test_Summary.csv'))
a=pd.concat(allrows,ignore_index=True).drop_duplicates(['task','seed'],keep='last').sort_values(['task','seed'])
a.to_csv(base/'R8_InputOnly_Results.csv',index=False)
assert len(a)==6, 'Need seed 42,52,62 for each 2s,5s'
c=pd.read_csv(base/'R8_ThreeSeed_Learned_Results.csv');c=c[c.branch=='corrected'].set_index(['task','seed'])
print('Same physical target input-only contrast:')
rows=[]
for x in a.itertuples(index=False):
 d=c.loc[(x.task,x.seed)];rows.append({'task':x.task,'seed':x.seed,'raw_input_ADE':x.test_ADE,'corrected_input_ADE':d.test_ADE,'raw_input_FDE':x.test_FDE,'corrected_input_FDE':d.test_FDE,'ADE_reduction_pct':100*(1-d.test_ADE/x.test_ADE),'FDE_reduction_pct':100*(1-d.test_FDE/x.test_FDE)})
b=pd.DataFrame(rows);b.to_csv(base/'R8_SameTarget_InputOnly_Seed_Comparisons.csv',index=False)
print(b.to_string(index=False))
print(b.groupby('task').mean(numeric_only=True).to_string())
fig,axes=plt.subplots(1,2,figsize=(11.7,4.8))
for ax,(task,title) in zip(axes,[('2s_to_2s','2-second horizon'),('2s_to_5s','5-second horizon')]):
 q=b[b.task==task];vals=[q.raw_input_ADE,q.corrected_input_ADE]
 for i,v in enumerate(vals):
  ax.bar(i,v.mean(),yerr=v.std(ddof=1),width=.55,capsize=4,color=['#829caf','#1d8f7c'][i]);ax.text(i,v.mean()+v.std(ddof=1)+.004,f'{v.mean():.4f}',ha='center',fontsize=10)
 ax.set_xticks([0,1],['Raw history','Corrected history']);ax.set_title(title,fontweight='bold');ax.set_ylabel('TestReal ADE (same corrected future target)');ax.set_ylim(0,max(q.raw_input_ADE)*1.25);ax.grid(alpha=.2,axis='y');ax.set_axisbelow(True)
fig.suptitle('Input-only ablation: identical corrected-frame future targets',fontweight='bold',fontsize=13)
fig.text(.5,.9,'Same Transformer architecture and three matched seeds; raw input receives no camera-transform features',ha='center',fontsize=9)
fig.tight_layout(rect=[0,0,1,.87]);fig.savefig(base/'Figure_R8_InputOnly_SameTarget_ThreeSeed.png',dpi=220,bbox_inches='tight');plt.close(fig)
