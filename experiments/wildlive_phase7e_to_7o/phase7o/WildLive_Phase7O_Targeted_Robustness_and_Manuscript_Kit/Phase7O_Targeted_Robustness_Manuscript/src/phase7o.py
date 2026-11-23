#!/usr/bin/env python3
"""Phase 7O: retrospective BG-registration model sensitivity and observation/target QA exposure.

Uses previously identified RGB pairs; this does NOT correct full trajectories or estimate new model ADE.
No model retraining, no new independent recording, and no prediction-coordinate inference.
"""
import argparse, csv, hashlib, io, json, sys, zipfile
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAIRS = {
    'DJI_0207': [('WildLive_DJI0207_priority_frames.zip', 'WildLive_DJI_0207_boxid.zip',[(389,390),(400,401),(401,402),(858,859),(859,860),(860,861)])],
    'DJI_0204_video2': [('DJI_0204_video2_visual_QA_frames.zip','DJI_0204_video2_boxid.zip',[(33,34),(303,304)])]
}
RESULTS={
  'DJI_0207':('Phase7F_Frozen_Results.zip','Phase7F_Frozen_Results/ALL_MODELS_per_window_phase7f.csv'),
  'DJI_0204_video2':('Phase7H_DJI0204_Frozen_Evaluation.zip','results/DJI_0204_video2_locked/locked_model_per_window.csv'),
}
SCORE_MODELS=['stationary','cv20','cv_ols','transformer_ensemble']

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def zip_member(z, filename):
    names=[n for n in z.namelist() if n.split('/')[-1] == filename]
    if len(names)!=1:raise ValueError(f'expected exactly one {filename}, got {len(names)}')
    return z.read(names[0])

def boxes(z,frame):
    ob=json.loads(zip_member(z,f'frame_{frame}.json'))
    out={}
    for sh in ob['shapes']:
        if sh.get('label','').casefold()!='elephant':continue
        p=np.array(sh['points'],float);id_=int(sh['group_id'])
        if id_ in out:raise ValueError('duplicate track ID in frame')
        out[id_]=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
    return out

def image(z,frame):
    a=cv2.imdecode(np.frombuffer(zip_member(z,f'frame_{frame}.jpg'),np.uint8),cv2.IMREAD_COLOR)
    if a is None:raise ValueError(f'cannot decode {frame}')
    return a

def mask_for(shape,bb,W,H,inflate=1.4):
    h,w=shape;mask=np.full((h,w),255,np.uint8);sx=w/W;sy=h/H
    for x1,y1,x2,y2 in bb.values():
        cx=(x1+x2)*sx/2;cy=(y1+y2)*sy/2
        rx=(x2-x1)*sx*inflate/2;ry=(y2-y1)*sy*inflate/2
        cv2.rectangle(mask,(max(0,int(cx-rx)),max(0,int(cy-ry))),
                      (min(w-1,int(cx+rx)),min(h-1,int(cy+ry))),0,-1)
    return mask

def pair_features(im0,im1,b0,b1,w=1536,h=864):
    if im0.shape!=im1.shape:raise ValueError('image shapes differ')
    H,W=im0.shape[:2]
    gray=[cv2.cvtColor(cv2.resize(im,(w,h)),cv2.COLOR_BGR2GRAY) for im in [im0,im1]]
    orb=cv2.ORB_create(nfeatures=6000,fastThreshold=10)
    k0,d0=orb.detectAndCompute(gray[0],mask_for((h,w),b0,W,H))
    k1,d1=orb.detectAndCompute(gray[1],mask_for((h,w),b1,W,H))
    if d0 is None or d1 is None:raise ValueError('no ORB descriptors')
    knn=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0,d1,k=2)
    match=[v[0] for v in knn if len(v)==2 and v[0].distance < .75*v[1].distance]
    if len(match)<160:raise ValueError('insufficient feature matches')
    src=np.array([k0[m.queryIdx].pt for m in match],np.float32)
    dst=np.array([k1[m.trainIdx].pt for m in match],np.float32)
    return src,dst,(W,H)

def spatial_split(src,seed=720,grid=4,holdout_fraction=0.25,w=1536,h=864):
    """Stratify in 4x4 source-image cells; split before model fitting."""
    rng=np.random.RandomState(seed)
    cells=np.clip(np.floor(src[:,0]*grid/w).astype(int),0,grid-1) + grid*np.clip(np.floor(src[:,1]*grid/h).astype(int),0,grid-1)
    hold=[]
    for cell in np.unique(cells):
        idx=np.flatnonzero(cells==cell)
        if len(idx)>1:
            n=max(1,int(round(len(idx)*holdout_fraction)));n=min(n,len(idx)-1)
            hold.extend(rng.choice(idx,n,replace=False).tolist())
    val=np.zeros(len(src),bool);val[hold]=True
    if np.count_nonzero(~val)<80 or val.sum()<20:raise ValueError('insufficient spatially held-out points')
    return ~val,val

def matrices(src,dst,seed=720):
    """Return 3x3 transforms in resized-image coordinates."""
    t=np.median(dst-src,axis=0)
    trans=np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]],dtype=float)
    cv2.setRNGSeed(seed)
    aff,aff_in=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3.0,maxIters=2000,confidence=.99,refineIters=10)
    cv2.setRNGSeed(seed)
    homo,h_in=cv2.findHomography(src,dst,cv2.RANSAC,3.0,maxIters=2000,confidence=.99)
    if aff is None or homo is None:raise ValueError('model estimation failure')
    sim=np.vstack([aff,[0,0,1]])
    return {'translation':(trans,None),'similarity':(sim,aff_in),'homography':(homo,h_in)}

def project(H,xy):
    xy=np.asarray(xy,float).reshape(-1,2)
    q=cv2.perspectiveTransform(xy.astype(np.float64).reshape(-1,1,2),np.asarray(H,float)).reshape(-1,2)
    return q

def full_res_matrix(M,W,H,small_w=1536,small_h=864):
    S=np.diag([small_w/W,small_h/H,1.])
    return np.linalg.inv(S)@M@S

def evaluate_pair(seq,frm,to,im0,im1,b0,b1,seed=720):
    p,q,(W,H)=pair_features(im0,im1,b0,b1)
    train,test=spatial_split(p,seed=seed)
    models=matrices(p[train],q[train],seed=seed)
    reg=[]; animal=[]
    for method,(M,inmask) in models.items():
        residual=np.linalg.norm(project(M,p[test])-q[test],axis=1)
        trainres=np.linalg.norm(project(M,p[train])-q[train],axis=1)
        # check independent keypoint spread and fit validity
        rows=dict(sequence=seq,frame_source=frm,frame_target=to,method=method,
                 n_matches=len(p),n_fit=int(train.sum()),n_heldout=int(test.sum()),
                 median_heldout_error_resized_px=float(np.median(residual)),
                 p90_heldout_error_resized_px=float(np.percentile(residual,90)),
                 median_fit_error_resized_px=float(np.median(trainres)),
                 estimated_inlier_fraction=float(np.mean(inmask)) if inmask is not None else np.nan,
                 notes='held-out ORB matches; same match set and split for all transforms')
        reg.append(rows)
        Mf=full_res_matrix(M,W,H)
        for id_ in sorted(b0.keys() & b1.keys()):
            x0,y0,x1,y1=b0[id_];a=np.array([(x0+x1)/2,(y0+y1)/2])
            x0,y0,x1,y1=b1[id_];b=np.array([(x0+x1)/2,(y0+y1)/2])
            c=project(Mf,a)[0]
            animal.append(dict(sequence=seq,frame_source=frm,frame_target=to,track_id=id_,method=method,
                          raw_centre_step_px=float(np.linalg.norm(b-a)),
                          background_model_step_px=float(np.linalg.norm(c-a)),
                          registered_residual_px=float(np.linalg.norm(b-c)),
                          predicted_B_x=float(c[0]),predicted_B_y=float(c[1])))
    return reg,animal

def load_windows(root):
    out=[]
    for seq,(fn,member) in RESULTS.items():
        with zipfile.ZipFile(root/fn) as z:
            if z.testzip() is not None:raise ValueError('bad prediction archive')
            df=pd.read_csv(io.BytesIO(z.read(member)))
        if df.duplicated(['track_id','horizon_frames','first_observed_frame']).any():raise ValueError('duplicate prediction windows')
        # Phase 7H archives omit last_observed_frame; history length is fixed at 20.
        if 'last_observed_frame' not in df:
            df['last_observed_frame']=df['first_observed_frame'].astype(int)+19
        expected_end=df['last_observed_frame'].astype(int)+df['horizon_frames'].astype(int)
        if not np.array_equal(df['last_target_frame'].astype(int).values,expected_end.values):
            raise ValueError(f'{seq}: target frame index inconsistent with 20-observation + future horizon')
        df['sequence']=seq
        for mod in SCORE_MODELS:
            if f'{mod}_ade_norm' not in df:raise ValueError('missing model ADE '+mod)
        out.append(df)
    return pd.concat(out,ignore_index=True)

def event_window_analysis(windows,animal,threshold_px=30.0):
    """Flag 7N-derived high image-register residuals, not confirmed annotation errors.

    First/last target are inclusive. One event is adjacent frames (a,b). Prediction window
    exposure if a and b both fall within history+future; history-only when b<=last_observed.
    """
    registered=animal[animal.method=='homography'].copy()
    events=registered[registered.registered_residual_px>=threshold_px].copy()
    rec=[]
    keys=set((r.sequence,int(r.track_id),int(r.frame_source),int(r.frame_target)) for r in events.itertuples())
    for w in windows.itertuples():
        obs=False;target=False;event_count=0
        for seq,tid,a,b in keys:
            if seq!=w.sequence or tid!=int(w.track_id):continue
            if a>=int(w.first_observed_frame) and b<=int(w.last_target_frame):
                event_count+=1
                if b<=int(w.last_observed_frame):obs=True
                else:target=True
        d=dict(sequence=w.sequence,track_id=int(w.track_id),horizon_frames=int(w.horizon_frames),
               first_observed_frame=int(w.first_observed_frame),last_observed_frame=int(w.last_observed_frame),
               last_target_frame=int(w.last_target_frame),event_count=event_count,
               exposure='history' if obs and not target else ('future' if target and not obs else ('both' if obs and target else 'none')),
               any_event=bool(event_count),
               pre_existing_qa_flagged=int(w.qa_flagged_window) if hasattr(w,'qa_flagged_window') and pd.notna(w.qa_flagged_window) else -1)
        for mod in SCORE_MODELS:
            d[f'{mod}_ade_norm']=float(getattr(w,mod+'_ade_norm'))
            d[f'{mod}_fde_norm']=float(getattr(w,mod+'_fde_norm'))
        rec.append(d)
    perwin=pd.DataFrame(rec)
    group=[]
    for (seq,h,exposed),df in perwin.groupby(['sequence','horizon_frames','any_event']):
        row=dict(sequence=seq,horizon_frames=int(h),any_high_residual_pair=bool(exposed),
                 n_windows=len(df),n_tracks=int(df.track_id.nunique()),
                 scope='retrospective diagnostic: high registration residual is not a confirmed annotation error')
        for mod in SCORE_MODELS:
            row[mod+'_ADE']=float(df[f'{mod}_ade_norm'].mean())
        row['best_expert_by_ADE']=min(SCORE_MODELS,key=lambda a:row[a+'_ADE'])
        group.append(row)
    return events,perwin,pd.DataFrame(group)

def charts(reg,perwin,groups,out):
    figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    rank={'translation':0,'similarity':1,'homography':2}
    pairs=reg[['sequence','frame_source','frame_target']].drop_duplicates()
    names=[f"{x.sequence.replace('DJI_','')}:{x.frame_source}-{x.frame_target}" for x in pairs.itertuples()]
    fig,ax=plt.subplots(figsize=(11.5,5.5))
    xs=np.arange(len(names))
    for m,offset,c in [('translation',-.25,'#8A9AA8'),('similarity',0,'#37A68E'),('homography',.25,'#2B517C')]:
        r=reg[reg.method==m].sort_values(['sequence','frame_source','frame_target'])
        # sorted using same stable key order as pairs
        order={tuple(v):i for i,v in enumerate(pairs.itertuples(index=False,name=None))}
        yy=np.zeros(len(names))
        for x in r.itertuples():yy[order[(x.sequence,x.frame_source,x.frame_target)]]=x.median_heldout_error_resized_px
        ax.bar(xs+offset,yy,width=.24,label=m.title(),color=c)
    ax.set_xticks(xs,names,rotation=32,ha='right',fontsize=8)
    ax.set_ylabel('Median held-out background-feature error (resized-image px)')
    ax.set_title('Phase 7O | Out-of-fit geometric registration error')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);fig.tight_layout()
    fig.savefig(figdir/'01_heldout_registration_model_comparison.png',dpi=180);plt.close(fig)
    # Unfiltered vs exposed vs non-exposed, keep diagnostic cohort sizes annotated
    a=groups[groups.sequence=='DJI_0207']
    if not a.empty:
        fig,axs=plt.subplots(1,2,figsize=(11.8,4.7),sharey=False)
        for ax,h in zip(axs,[20,50]):
            sub=a[a.horizon_frames==h].set_index('any_high_residual_pair')
            if not {False,True}.issubset(sub.index):continue
            modes=['stationary','cv_ols','transformer_ensemble'];x=np.arange(3)
            for offset,sel,color,label in [(-.19,False,'#8AA6BD','Not exposed'),(.19,True,'#E5A65B','Exposed')]:
                ax.bar(x+offset,[sub.loc[sel,m+'_ADE'] for m in modes],width=.36,color=color,label=label)
            ax.set_xticks(x,['Stationary','CV-OLS','Transformer']);ax.set_title(f'DJI_0207, {h} future frames');ax.grid(axis='y',alpha=.15)
            ax.set_ylabel('Mean normalized ADE (descriptive)')
            ax.legend(frameon=False,fontsize=8)
        fig.suptitle('Phase 7O | Window exposure to high-residual annotation pairs (not corrected ADE)')
        fig.tight_layout();fig.savefig(figdir/'02_QA_exposure_sensitivity.png',dpi=180);plt.close(fig)

def run(root,out,threshold_px=30.):
    out.mkdir(parents=True,exist_ok=True)
    source=[]
    for seq,lst in PAIRS.items():
        for fn,ann,_ in lst:
            for n in [fn,ann]:
                source.append(dict(filename=n,sha256=sha256(root/n),bytes=(root/n).stat().st_size))
    for seq,(fn,_) in RESULTS.items():
        source.append(dict(filename=fn,sha256=sha256(root/fn),bytes=(root/fn).stat().st_size))
    source=pd.DataFrame(source).drop_duplicates('filename').sort_values('filename')
    lockpath=Path(__file__).resolve().parent.parent/'protocol'/'INPUT_SHA256_LOCK.json'
    if lockpath.exists():
        expected=json.loads(lockpath.read_text())
        for row in source.itertuples():
            if expected.get(row.filename)!=row.sha256:
                raise ValueError(f'source file hash mismatch: {row.filename}; do not substitute input archive')
    source.to_csv(out/'source_file_hashes.csv',index=False)
    registration=[];animal=[]
    for seq,lst in PAIRS.items():
        for imfn,annfn,pairs in lst:
            with zipfile.ZipFile(root/imfn) as iz, zipfile.ZipFile(root/annfn) as az:
                if iz.testzip() is not None or az.testzip() is not None:raise ValueError('bad ZIP CRC')
                for f0,f1 in pairs:
                    r,a=evaluate_pair(seq,f0,f1,image(iz,f0),image(iz,f1),boxes(az,f0),boxes(az,f1),seed=720+f0)
                    registration.extend(r);animal.extend(a)
    r=pd.DataFrame(registration); a=pd.DataFrame(animal)
    r.to_csv(out/'heldout_registration_models.csv',index=False)
    a.to_csv(out/'per_animal_registration_model_sensitivity.csv',index=False)
    windows=load_windows(root)
    events,perwin,groups=event_window_analysis(windows,a,threshold_px)
    events.to_csv(out/'high_residual_QA_events_NOT_confirmed_errors.csv',index=False)
    perwin.to_csv(out/'per_window_QA_exposure.csv',index=False)
    groups.to_csv(out/'QA_exposure_ADE_sensitivity.csv',index=False)
    # preserve exact Phase7N comparison data as separate source output; no new model inference
    ref=pd.read_csv(Path(__file__).resolve().parent.parent/'protocol'/'PHASE7N_original_recording_level_ADE_FDE.csv')
    ref.to_csv(out/'archived_four_recording_model_ADE_FDE.csv',index=False)
    charts(r,perwin,groups,out)
    manifest=dict(phase='7O',evaluated_pairs=8,image_resized_dimensions=[1536,864],
      pair_selection='previously targeted Phase7N pairs; nonrandom',threshold_registered_residual_px=threshold_px,
      registered_heldout_matches='25% within 4x4 spatial cells; fit solely on remaining 75%',
      records_per_method=len(a)//3,locked_predictor_retraining=False,new_forecasting_inference=False,
      new_camera_corrected_forecasting_ADE=False,new_independent_capture_events=False,
      scope='Retrospective sampled-frame geometry model sensitivity and flagged-window descriptive error stratification',
      source_files=source.to_dict('records'))
    (out/'phase7o_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Evaluation pairs:',len(r)//3,' registration rows:',len(r),' animal rows:',len(a),' high-residual events:',len(events))
    print('Model median held-out registration error, resized px:')
    print(r.groupby('method').median_heldout_error_resized_px.agg(['median','max']).to_string())
    print('High-residual event count by recording:',events.groupby('sequence').size().to_dict())
    print('Windows (with and without events):',groups[['sequence','horizon_frames','any_high_residual_pair','n_windows','best_expert_by_ADE']].to_string(index=False))
    return r,a,events,perwin,groups

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True,help='Existing original ZIPs and Phase 7N extracted directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qa-residual-threshold-px',type=float,default=30.)
    x=p.parse_args();run(x.input_dir,x.output,x.qa_residual_threshold_px)
