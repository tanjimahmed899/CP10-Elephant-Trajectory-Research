#!/usr/bin/env python3
"""Create publication figures from the user-provided EXP3 frozen model predictions and the source-frame sample ZIP.
Sources are read only; does NOT run detector or create forecast predictions. Requires numpy, pandas, pillow, matplotlib.
"""
from __future__ import annotations
import io, json, zipfile, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

BASE=Path('/mnt/data')
OUT=BASE/'CP10_Stage3_Verified_Forecast_Overlays';OUT.mkdir(exist_ok=True)
S=zipfile.ZipFile(BASE/'CP10_Stage3_EXP3_Source_Samples.zip')
P=zipfile.ZipFile(BASE/'EXP3_Output_Review.zip')
NAMES=set(S.namelist())

T={}
for t in ['2s_to_2s','2s_to_5s']:
    m=pd.read_csv(io.BytesIO(P.read(f'results\\{t}_fresh_metadata.csv')))
    pr=np.load(io.BytesIO(P.read(f'results\\{t}_exp3_frozen_ensemble_predictions.npz')),allow_pickle=False)
    T[t]=(m,{k:pr[k] for k in pr.files})
CAM={};TRACK={}
for n in S.namelist():
    if n.endswith('/cameras.json'):
        folder=n.rsplit('/',1)[0]+'/'
        data=json.loads(S.read(n))
        CAM[folder]={int(c['frame_index']):c for c in data['cameras']}
        TRACK[folder]=json.loads(S.read(folder+'tracking_summary.json'))['tracks']

def get_path(row):
    return f'EXP3_samples/{row.elephant_group}/{row.video}/{row.segment}/'

def get_row(task,idx):
    m,arr=T[task];r=m.iloc[idx]
    folder=get_path(r); cam=CAM[folder][int(r.anchor_frame_index)]
    source=folder+cam['image_name']
    if source not in NAMES: raise FileNotFoundError(source)
    txt=folder+'kitti_labels/'+cam['image_name'].replace('.jpg','.txt')
    boxes=[]
    for line in S.read(txt).decode().splitlines():
        q=line.split()
        if q[0].lower()=='elephant':boxes.append(np.array([float(t) for t in q[4:8]]))
    W,H=int(cam['image_width']),int(cam['image_height']);scale=np.array([W,H])
    img=Image.open(io.BytesIO(S.read(source))).convert('RGB')
    return dict(index=idx, row=r, folder=folder,cam=cam,src=source,image=img, boxes=boxes,
       hist=arr['history_abs'][idx]*scale,future=arr['future_abs'][idx]*scale,pred=arr['ensemble_prediction'][idx]*scale,
       W=W,H=H)

def closest_box(dat):
    h=dat['hist'][0]
    if not dat['boxes']:return -1, np.nan
    dist=[np.linalg.norm((b[:2]+b[2:])/2-h) for b in dat['boxes']]
    return int(np.argmin(dist)),float(np.min(dist))

def verify_geometry(dat,task):
    # reconstruct from the original adapter's 3-D tracking centers and the exact per-frame camera transforms.
    r=dat['row'];frm=int(r.anchor_frame_index)
    allcams=CAM[dat['folder']];track=TRACK[dat['folder']][str(r.track_id)]
    frames=np.asarray(track['frames'],dtype=int);coords=np.asarray(track['centers'],dtype=np.float64)
    anchor_idx=np.where(frames==frm)[0]
    if len(anchor_idx)!=1:return {'status':'missing_anchor_in_track'}
    idx=int(anchor_idx[0]);n=40 if task=='2s_to_2s' else 70
    if idx+n>len(frames):return {'status':'short_track'}
    anchor=dat['cam'];K=np.array(anchor['intrinsic'],dtype=np.float64)
    E0=np.array(anchor['extrinsic'],dtype=np.float64); R0,t0=E0[:,:3], E0[:,3]
    norm=[]
    for f,x in zip(frames[idx:idx+n],coords[idx:idx+n]):
        if f not in allcams: return {'status':'missing_camera'}
        current=allcams[int(f)];E=np.array(current['extrinsic'],dtype=np.float64);R,t=E[:,:3],E[:,3]
        xworld=R.T@(x-t);xanchor=R0@xworld+t0
        if xanchor[2]<=0:return {'status':'invalid_depth'}
        projected=K@xanchor
        norm.append([projected[0]/projected[2],projected[1]/projected[2]])
    pix=np.array(norm)
    saved=np.vstack([dat['hist'],dat['future']])
    return {'status':'matched', 'max_abs_pixel_difference':float(np.abs(pix-saved).max()),
        'mean_abs_pixel_difference':float(np.abs(pix-saved).mean()),
        'anchor_frame_index':frm,'track_id':int(r.track_id)}

def crop_bounds(dat,pad=85,min_w=530,min_h=420):
    bbidx,_=closest_box(dat)
    a=np.concatenate([dat['hist'],dat['future'],dat['pred']])
    if bbidx>=0:
        b=dat['boxes'][bbidx];a=np.vstack([a,b[:2],b[2:]])
    lo=a.min(axis=0)-pad;hi=a.max(axis=0)+pad
    cx,cy=((hi+lo)/2); span=np.maximum(hi-lo,[min_w,min_h]);aspect=1.55
    if span[0]/span[1]>aspect:span[1]=span[0]/aspect
    else:span[0]=span[1]*aspect
    x0,x1=cx-span[0]/2,cx+span[0]/2;y0,y1=cy-span[1]/2,cy+span[1]/2
    if x0<0:x1-=x0;x0=0
    if y0<0:y1-=y0;y0=0
    if x1>dat['W']:x0-=x1-dat['W'];x1=dat['W']
    if y1>dat['H']:y0-=y1-dat['H'];y1=dat['H']
    return max(0,x0),min(dat['W'],x1),max(0,y0),min(dat['H'],y1)

COLORS={'history':'#1565C0','actual':'#19A15F','pred':'#FB493B'}

def draw_case(ax,dat,title,crop=True,gt=True):
    ax.imshow(dat['image'])
    if gt:
        for b in dat['boxes']:
            ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,
                                   edgecolor='#F3E84C',linewidth=2.2,linestyle=':'))
    # Chronology: anchor image is the FIRST observed frame, so its animal position is observed-history start.
    h,f,p=dat['hist'],dat['future'],dat['pred']
    ax.plot(h[:,0],h[:,1],lw=3.2,color=COLORS['history'],solid_capstyle='round',alpha=.92,zorder=7)
    ax.plot(np.r_[h[-1,0],f[:,0]],np.r_[h[-1,1],f[:,1]],lw=3.0,color=COLORS['actual'],alpha=.95,zorder=8)
    ax.plot(np.r_[h[-1,0],p[:,0]],np.r_[h[-1,1],p[:,1]],lw=3.0,color=COLORS['pred'],ls='--',alpha=.95,zorder=9)
    ax.scatter([h[0,0]],[h[0,1]],c='white',edgecolors=COLORS['history'],s=70,linewidths=2,zorder=11)
    ax.scatter([h[-1,0]],[h[-1,1]],marker='o',c=COLORS['history'],edgecolors='white',s=85,linewidths=1.5,zorder=12)
    ax.scatter([f[-1,0]],[f[-1,1]],marker='X',c=COLORS['actual'],edgecolors='white',s=94,linewidths=.7,zorder=12)
    ax.scatter([p[-1,0]],[p[-1,1]],marker='X',c=COLORS['pred'],edgecolors='white',s=94,linewidths=.7,zorder=12)
    ax.set_title(title,fontsize=12,pad=8,loc='left',weight='bold')
    if crop:
        x0,x1,y0,y1=crop_bounds(dat)
        ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
    ax.axis('off')
    for spine in ax.spines.values():spine.set_visible(False)

def plt_main():
    chosen=[(0,'(a) elep1 | two elephants in source frame'),(187,'(b) elep2 | isolated elephant'),
            (482,'(c) elep3 | herd scene'),(52,'(d) elep2 | high-error example')]
    fig,axes=plt.subplots(2,2,figsize=(14,9.8),layout='constrained')
    audit=[]
    for ax,(idx,title) in zip(axes.flat,chosen):
        d=get_row('2s_to_5s',idx);draw_case(ax,d,title)
        report=verify_geometry(d,'2s_to_5s');bindex,center_dist=closest_box(d)
        row={**report,'sample_index':idx,'video':d['row'].video,'group':d['row'].elephant_group,'segment':d['row'].segment,
             'anchor_image':d['cam']['image_name'],'gt_elephants':len(d['boxes']),
             'matched_annotation_center_distance_px':center_dist,
             'trajectory_ADE_normalized':float(np.mean(np.linalg.norm((d['pred']-d['future'])/np.array([d['W'],d['H']]),axis=1)))}
        audit.append(row)
    handles=[Line2D([0],[0],color=COLORS['history'],lw=3,label='Observed trajectory (2 s)'),
             Line2D([0],[0],color=COLORS['actual'],lw=3,label='Annotated future (5 s)'),
             Line2D([0],[0],color=COLORS['pred'],lw=3,linestyle='--',label='Frozen Transformer forecast (5 s)'),
             Line2D([0],[0],color='#F3E84C',lw=2,linestyle=':',label='Dataset elephant box at anchor frame')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.047),ncol=2,fontsize=10,frameon=False)
    fig.suptitle('Five-second elephant forecasts shown in the observation-start camera reference frame',fontsize=15,y=1.015,weight='bold')
    fig.savefig(OUT/'Figure_Stage3_RealDrone_5s_Forecast_Examples.png',dpi=245,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    pd.DataFrame(audit).to_csv(OUT/'Selected_Forecast_Examples_Geometry_Validation.csv',index=False)

def plt_three_elephants():
    # Three GT tracks in the same frame, each has an independent saved 5-second ensemble forecast.
    dat=[get_row('2s_to_5s',i) for i in [482,496,510]]
    assert len(set(x['src'] for x in dat))==1
    fig,ax=plt.subplots(figsize=(14,8),layout='constrained')
    ax.imshow(dat[0]['image']);colors=['#ef476f','#ffae00','#00b4d8']
    for b in dat[0]['boxes']:
        ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='white',linewidth=2.1,linestyle=':'))
    for d,col in zip(dat,colors):
        h,p,f=d['hist'],d['pred'],d['future']
        ax.plot(h[:,0],h[:,1],color=col,lw=3,ls=':',alpha=.9)
        ax.plot(np.r_[h[-1,0],f[:,0]],np.r_[h[-1,1],f[:,1]],color=col,lw=3,alpha=.9)
        ax.plot(np.r_[h[-1,0],p[:,0]],np.r_[h[-1,1],p[:,1]],color=col,lw=3,ls='--',alpha=1)
        ax.scatter(h[0,0],h[0,1],s=85,color=col,edgecolors='white',linewidths=2,zorder=10)
        ax.scatter(p[-1,0],p[-1,1],s=85,color=col,marker='X',edgecolors='black',linewidths=.5,zorder=10)
        ax.text(h[0,0]+12,h[0,1]-10,'Track '+str(d['row'].track_id),color='white',fontsize=12,weight='bold',
           bbox={'boxstyle':'round,pad=0.2','facecolor':col,'alpha':.9,'edgecolor':'none'})
    ax.set_xlim(185,1640);ax.set_ylim(1000,200);ax.axis('off')
    fig.suptitle('Three annotated elephants with separately evaluated 5-second forecasts (elep3)',fontsize=16,weight='bold')
    legend=[Line2D([0],[0],color='black',lw=3,ls=':',label='Observed 2-second history'),
            Line2D([0],[0],color='black',lw=3,ls='-',label='Annotated 5-second future'),
            Line2D([0],[0],color='black',lw=3,ls='--',label='Frozen model prediction')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,-.025),ncol=3,frameon=False,fontsize=10)
    fig.savefig(OUT/'Figure_Stage3_ThreeElephants_IndividualForecasts.png',dpi=250,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def plt_gt_comparison():
    dat=get_row('2s_to_5s',482)
    fig,axs=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    for ax in axs:ax.imshow(dat['image']);ax.axis('off')
    for j,b in enumerate(dat['boxes']):
        axs[1].add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,linewidth=3,edgecolor='#00e5ff'))
        axs[1].text(b[0],max(30,b[1]-8),f'Elephant GT {j+1}',color='black',fontsize=11,
                    bbox={'facecolor':'#00e5ff','edgecolor':'none','pad':2})
    axs[0].set_title('Original RGB image (WildBox)',loc='left',fontsize=13,weight='bold')
    axs[1].set_title('Dataset-provided ground-truth annotations',loc='left',fontsize=13,weight='bold')
    fig.savefig(OUT/'Figure_Stage3_Original_vs_ThreeElephant_GT.png',dpi=245,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def all_geometry():
    results=[]
    for task in ['2s_to_2s','2s_to_5s']:
        meta,_=T[task]
        for idx,row in meta.iterrows():
            folder=get_path(row)
            if folder not in CAM:continue
            camera=CAM[folder].get(int(row.anchor_frame_index))
            if camera is None or folder+camera['image_name'] not in NAMES:continue
            try:
                d=get_row(task,idx)
                v=verify_geometry(d,task)
                results.append({'task':task,'sample_index':idx,'group':row.elephant_group,'video':row.video,
                   'segment':row.segment,'track_id':int(row.track_id),'anchor_frame_index':int(row.anchor_frame_index),
                   'source_image':d['src'],**v})
            except Exception as exc:
                results.append({'task':task,'sample_index':idx,'group':row.elephant_group,'video':row.video,
                  'segment':row.segment,'track_id':int(row.track_id),'status':'error','message':str(exc)})
    pd.DataFrame(results).to_csv(OUT/'EXP3_Anchor_Frame_Geometry_Audit.csv',index=False)
    return results

if __name__=='__main__':
    allrows=all_geometry();print('Anchor geometry checks',len(allrows),'matched',sum(r['status']=='matched' for r in allrows))
    print('Maximum pixel difference',max((r['max_abs_pixel_difference'] for r in allrows if r['status']=='matched'),default=None))
    plt_main();plt_three_elephants();plt_gt_comparison()
    print('Created',*[str(x.name) for x in OUT.glob('Figure*.png')],sep='\n')
