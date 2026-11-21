"""Generate paired 2s/5s forecast visualization on the exact same observation-start RGB frame."""
import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
p=Path('/mnt/data/CP10_Stage3_Verified_Forecast_Overlays/build_verified_overlays.py')
spec=importlib.util.spec_from_file_location('v',p);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
for vid,two,five in [('elep1',0,0),('elep2',247,187),('elep3',643,482)]:
 a=v.get_row('2s_to_2s',two);b=v.get_row('2s_to_5s',five)
 assert a['src']==b['src'] and a['row'].track_id==b['row'].track_id
 bounds=v.crop_bounds(b,pad=100,min_w=600,min_h=460)
 fig,axs=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
 for ax,d,h in zip(axs,[a,b],['(a) Two-second future','(b) Five-second future']):
  v.draw_case(ax,d,h,crop=False)
  x0,x1,y0,y1=bounds;ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
 handles=[Line2D([0],[0],color=v.COLORS['history'],lw=3,label='Observed history (2 s)'),
          Line2D([0],[0],color=v.COLORS['actual'],lw=3,label='Annotated future'),
          Line2D([0],[0],color=v.COLORS['pred'],lw=3,ls='--',label='Frozen Transformer prediction')]
 fig.suptitle(f'Same RGB anchor frame and tracked elephant: {vid} | 2 s versus 5 s forecasting',fontsize=14,weight='bold',y=1.04)
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.045),ncol=3,frameon=False)
 fig.savefig(v.OUT/f'Figure_Stage3_{vid}_Matched_2s_vs_5s.png',dpi=240,bbox_inches='tight',facecolor='white')
 plt.close(fig)
