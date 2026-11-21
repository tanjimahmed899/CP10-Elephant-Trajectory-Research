from pathlib import Path
import zipfile,io, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser()
ap.add_argument('--stage7d',default='/mnt/c/Users/HP/Downloads/CP10_Stage7D_Tracker_Review.zip')
args=ap.parse_args()
p=Path(__file__).parent
z=zipfile.ZipFile(args.stage7d)
read=lambda n:pd.read_csv(io.BytesIO(z.read(n)),dtype={'sequence':str})
agg=read('Stage7D_Aggregate_Tracking.csv');fc=read('Stage7D_Forecastable_Segments.csv');seq=read('Stage7D_PerSequence_Tracking.csv')
original=agg[agg.tracker=='original'].iloc[0];adapted=agg[agg.tracker=='adapted'].iloc[0]
metrics=['HOTA','IDF1','MOTA']; x=np.arange(len(metrics));width=0.35
fig,ax=plt.subplots(figsize=(8.6,4.8))
ax.bar(x-width/2,[original[m] for m in metrics],width,label='Original YOLO + ByteTrack')
ax.bar(x+width/2,[adapted[m] for m in metrics],width,label='Fine-tuned YOLO + ByteTrack')
ax.set_xticks(x,metrics);ax.set_ylim(0,0.24);ax.set_ylabel('Metric score (0–1)')
ax.set_title('Stage 7D | Frozen ByteTrack on BIRDSAI TestReal')
ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=0.18)
fig.tight_layout();fig.savefig(p/'Figure_Stage7D_Tracking_Metrics.png',dpi=180);plt.close(fig)
# source concentration
rows=[]
for model in ['original','adapted']:
 for seqno in sorted(fc.sequence.unique()):
  sg=fc[(fc.model==model)&(fc.sequence==seqno)]
  rows.append(dict(model=model,sequence=seqno,window_2s=int(sg.windows_2s.sum()),window_5s=int(sg.windows_5s.sum()),
                   forecastable_segments_2s=int((sg.windows_2s>0).sum()),forecastable_segments_5s=int((sg.windows_5s>0).sum())))
pd.DataFrame(rows).to_csv(p/'Stage7D_Window_Concentration.csv',index=False)
latest=pd.DataFrame(rows);latest=latest[latest.model=='adapted']
x=np.arange(len(latest));width=0.36
fig,ax=plt.subplots(figsize=(9,4.5))
ax.bar(x-width/2,latest.window_2s,width,label='2s-to-2s potential')
ax.bar(x+width/2,latest.window_5s,width,label='2s-to-5s potential')
ax.set_xticks(x,[str(s).split('_')[0][-4:] for s in latest.sequence]);ax.set_ylabel('Potential windows (not GT-verified)')
ax.set_xlabel('BIRDSAI TestReal source sequence (last 4 digits)')
ax.set_title('Stage 7D | Potential window yield is highly source-concentrated')
ax.legend();ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(p/'Figure_Stage7D_Window_Concentration.png',dpi=180);plt.close(fig)
# direct clean subset summary for humans
cols=['tracker','HOTA','IDF1','IDSW','Frag','MOTA','num_tracker_dets','CLR_TP','CLR_FP','num_tracker_ids']
agg[cols].to_csv(p/'Stage7D_Verified_Tracking_Headline.csv',index=False)
print('Figures and audit CSV created')
