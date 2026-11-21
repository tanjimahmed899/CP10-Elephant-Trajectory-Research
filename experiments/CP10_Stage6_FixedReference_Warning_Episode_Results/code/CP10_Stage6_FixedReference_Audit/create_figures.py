from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
prior=pd.read_csv(D/'inputs/Stage6_Coordinate_Shift_Summary.csv')
fixed=pd.read_csv(D/'Stage6_FixedReference_AdjacentAnchor_GT_Consistency_Summary.csv')
prior=prior[(prior.source=='EXP3')|(prior.role=='evaluation')]
prior=prior.merge(fixed,on=['source','horizon'])
prior['set']=prior['source']+' '+prior['horizon']
fig,ax=plt.subplots(figsize=(8.4,4.8),dpi=185)
x=np.arange(len(prior));w=.38
ax.bar(x-w/2,prior.median_pair_mean_displacement,width=w,label='Moving observation-start frame')
ax.bar(x+w/2,prior.median_pair_mean_fixed_GT_displacement,width=w,label='Fixed segment reference (GT-only replay)')
ax.set_yscale('log');ax.set_xticks(x,prior['set']);ax.set_ylim(1e-9,0.2)
ax.set_ylabel('Median overlapping GT-point discrepancy (normalized, log scale)')
ax.set_title('Static reference eliminates artificial window-to-window GT coordinate shift')
ax.legend(fontsize=8)
fig.text(.015,.015,'Fixed-reference GT replay uses annotated future 3-D centers; NOT a deployable model operation.',fontsize=8)
fig.tight_layout(rect=(0,.055,1,1));fig.savefig(D/'Figure_Stage6_Moving_vs_Fixed_Reference_Consistency.png');plt.close(fig)
# Event tradeoff: event recall, false alert-runs in separate visual axes; show only K3 EXP3 and Stage5 eval
m=pd.read_csv(D/'Stage6_FixedReference_Exploratory_Episode_Metrics.csv')
m=m[(m.persistence_k==3)&(m.role!='adaptation')]
selpol=['nominal','fixed_N2_gapaware','fixed_N3_gapaware']
fig,ax=plt.subplots(figsize=(9,5.5),dpi=190)
for (source,h),g in m.groupby(['source','horizon']):
    for pol in selpol:
        r=g[g.policy==pol].iloc[0]
        ax.scatter(r.event_unassociated_alert_runs,r.event_recall*100,s=90,marker='o' if h=='2s' else 's')
        ax.annotate(f'{source} {h} {pol.replace("fixed_", "")}',(r.event_unassociated_alert_runs,r.event_recall*100),fontsize=7,xytext=(5,4),textcoords='offset points')
ax.set_xlim(left=-8);ax.set_ylim(0,101);ax.set_xlabel('Alert runs not associated with a stable GT crossing (COUNT, not operational false alarms)')
ax.set_ylabel('Eligible annotated crossing episodes detected (%)')
ax.set_title('Fixed-segment-reference synthetic geofence: event-recall vs alert-run burden')
ax.grid(alpha=.2);fig.tight_layout();fig.savefig(D/'Figure_Stage6_Conditional_Event_Tradeoff.png');plt.close(fig)
