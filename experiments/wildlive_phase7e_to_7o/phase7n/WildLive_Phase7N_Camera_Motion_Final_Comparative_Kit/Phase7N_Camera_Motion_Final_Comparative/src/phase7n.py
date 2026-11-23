#!/usr/bin/env python3
"""CP 1.0 Phase 7N: descriptive locked-model synthesis + sparse image-based camera motion audit.

Recomputes metrics exclusively from archived per-window errors; does not rerun model inference.
Camera alignment is only for the eight available *consecutive* RGB frame pairs.
This DOES NOT produce a camera-corrected forecasting evaluation, ground motion, or flight independence.
"""
import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILES={
    'DJI_0207':('Phase7F_Frozen_Results.zip','Phase7F_Frozen_Results/ALL_MODELS_per_window_phase7f.csv', 'WildLive_DJI_0207_boxid.zip', 'DEV', 'UNRESOLVED_0207'),
    'DJI_0204_video2':('Phase7H_DJI0204_Frozen_Evaluation.zip', 'results/DJI_0204_video2_locked/locked_model_per_window.csv', 'DJI_0204_video2_boxid.zip', 'DEV', 'SCENE_0204'),
    'DJI_0204_video1':('Phase7L_DJI0204_video1_Exploratory_Results.zip','results/DJI_0204_video1_RELATED_SCENE_EXPLORATORY/all_experts_and_gate_per_window.csv','DJI_0204_video1_boxid.zip','RELATED_SCENE_EXPLORATORY','SCENE_0204'),
    'DJI_0161_video1':('Phase7K_DJI0161_Locked_Evaluation.zip','results/DJI_0161_video1_locked/all_experts_and_gate_per_window.csv','DJI_0161_video1_boxid.zip','PRIORLY_EVALUATED_TEST_CANDIDATE','UNRESOLVED_0161'),
}
IMG_ARCHIVES={
    'DJI_0207':('WildLive_DJI0207_priority_frames.zip',[(389,390),(400,401),(401,402),(858,859),(859,860),(860,861)]),
    'DJI_0204_video2':('DJI_0204_video2_visual_QA_frames.zip',[(33,34),(303,304)]),
}
MODELS=['stationary','cv20','cv_ols','transformer_ensemble','adaptive']


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def load_zip_csv(path, member):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, f'Bad ZIP CRC: {path}'
        df=pd.read_csv(io.BytesIO(z.read(member)))
    return df


def get_zipped_file(z, name):
    found=[n for n in z.namelist() if n.split('/')[-1]==name]
    if len(found)!=1: raise ValueError(f'Expected one {name} in archive; got {len(found)}')
    return z.read(found[0])


def get_annotation_boxes(az, frame):
    source=json.loads(get_zipped_file(az,f'frame_{frame}.json'))
    boxes={}
    for obj in source['shapes']:
        if obj.get('label','').lower()!='elephant': continue
        coords=np.asarray(obj['points'],dtype=float)
        lo=np.min(coords,axis=0);hi=np.max(coords,axis=0)
        track=int(obj['group_id'])
        if track in boxes: raise ValueError(f'duplicate ID {track} frame {frame}')
        boxes[track]=[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])]
    return boxes


def get_jpg(z, n):
    im=cv2.imdecode(np.frombuffer(get_zipped_file(z,f'frame_{n}.jpg'),np.uint8),cv2.IMREAD_COLOR)
    if im is None: raise ValueError(f'Could not decode frame_{n}.jpg')
    return im


def exclusion_mask(smallshape, bounding_boxes, original_w, original_h, inflate=1.4):
    sh,sw=smallshape[:2]
    mask=np.full((sh,sw),255,dtype=np.uint8)
    sx=sw/original_w;sy=sh/original_h
    for xmin,ymin,xmax,ymax in bounding_boxes.values():
        cx=(xmin+xmax)*sx/2;cy=(ymin+ymax)*sy/2
        radx=(xmax-xmin)*sx*inflate/2
        rady=(ymax-ymin)*sy*inflate/2
        x1=max(0,int(cx-radx));y1=max(0,int(cy-rady))
        x2=min(sw-1,int(cx+radx));y2=min(sh-1,int(cy+rady))
        cv2.rectangle(mask,(x1,y1),(x2,y2),0,-1)
    return mask


def apply_h(H,p):
    out=H @ np.array([p[0],p[1],1.0])
    if abs(out[2]) < 1e-10:raise ValueError('ill-conditioned homography')
    return out[:2]/out[2]


def frame_registration(im0, im1, box0, box1, width=1536, height=864):
    if im0.shape[:2]!=im1.shape[:2]: raise ValueError('source frames differ in image dimensions')
    H0,W0=im0.shape[:2]
    sm0=cv2.cvtColor(cv2.resize(im0,(width,height)),cv2.COLOR_BGR2GRAY)
    sm1=cv2.cvtColor(cv2.resize(im1,(width,height)),cv2.COLOR_BGR2GRAY)
    orb=cv2.ORB_create(nfeatures=6000,fastThreshold=10)
    k0,d0=orb.detectAndCompute(sm0,exclusion_mask(sm0.shape,box0,W0,H0))
    k1,d1=orb.detectAndCompute(sm1,exclusion_mask(sm1.shape,box1,W0,H0))
    if d0 is None or d1 is None:raise ValueError('no descriptors')
    candidates=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0,d1,k=2)
    good=[a for a,b in candidates if a.distance < 0.75*b.distance]
    if len(good)<4:raise ValueError('insufficient descriptor matches')
    pts0=np.float32([k0[m.queryIdx].pt for m in good]);pts1=np.float32([k1[m.trainIdx].pt for m in good])
    # Seed OpenCV RNG for reproducibility, but note library/version/hardware variation.
    cv2.setRNGSeed(202607)
    H,inlier_mask=cv2.findHomography(pts0,pts1,cv2.RANSAC,3.0)
    if H is None:raise ValueError('homography failed')
    goodmask=inlier_mask.ravel().astype(bool)
    in0=pts0[goodmask];in1=pts1[goodmask]
    pred=cv2.perspectiveTransform(in0.reshape(-1,1,2),H).reshape(-1,2)
    residual=np.linalg.norm(pred-in1,axis=1)
    hull=cv2.convexHull(in0).reshape(-1,2)
    coverage=float(cv2.contourArea(hull.astype('float32'))/(width*height))
    grid={(min(3,int(max(0,p[0]*4/width))),min(3,int(max(0,p[1]*4/height)))) for p in in0}
    inlier_rate=float(len(in0)/len(good))
    median_residual=float(np.median(residual))
    # Independent status: 2D BG homography is an estimator, not ground-truth ego-motion.
    accepted=(len(in0)>=80 and inlier_rate>=0.55 and coverage>=0.1 and len(grid)>=8 and median_residual<=2)
    # Mapping to original-pixel coordinates: S^-1 * H_small * S
    S=np.diag([width/W0,height/H0,1.0]);fullH=np.linalg.inv(S)@H@S
    return fullH,dict(keypoints0=len(k0),keypoints1=len(k1),ratio_matches=len(good),inliers=len(in0),inlier_fraction=inlier_rate,inlier_hull_coverage=coverage,spatial_4x4_cells=len(grid),median_reprojection_error_small_px=median_residual,bg_registration_qc_pass=bool(accepted))


def cam_pair_rows(seq, src, dst, im0, im1, b0, b1):
    H,info=frame_registration(im0,im1,b0,b1)
    measurements=[]
    for track in sorted(b0.keys() & b1.keys()):
        x1,y1,x2,y2=b0[track];s=np.array([(x1+x2)/2,(y1+y2)/2])
        x1,y1,x2,y2=b1[track];t=np.array([(x1+x2)/2,(y1+y2)/2])
        prediction=apply_h(H,s)
        measurements.append(dict(sequence=seq,frame_source=src,frame_target=dst,track_id=track,
                   raw_bbox_center_step_px=float(np.linalg.norm(t-s)),
                   estimated_local_background_step_px=float(np.linalg.norm(prediction-s)),
                   residual_after_BG_alignment_px=float(np.linalg.norm(t-prediction)),
                   raw_dx_px=float(t[0]-s[0]),raw_dy_px=float(t[1]-s[1]),
                   residual_dx_px=float(t[0]-prediction[0]),residual_dy_px=float(t[1]-prediction[1]),
                   **info))
    return measurements


def load_and_validate(root):
    all_frames=[]; manifest=[]
    for seq,(file,member,annot,role,group) in DATA_FILES.items():
        path=root/file
        if not path.exists():raise FileNotFoundError(f'Missing source {path}')
        df=load_zip_csv(path,member)
        for col in ['horizon_frames','track_id','first_observed_frame']:
            if col not in df:raise ValueError(f'{seq}: Missing {col}')
        if df[['horizon_frames','track_id','first_observed_frame']].isna().any().any():raise ValueError(f'{seq}: missing window key')
        if df.duplicated(['horizon_frames','track_id','first_observed_frame']).any():raise ValueError(f'{seq}: duplicate prediction window keys')
        if not set(df.horizon_frames.unique())=={20,50}:raise ValueError(f'{seq}: unexpected forecasting horizons')
        for model in MODELS:
            for metric in ['ade','fde']:
                col=f'{model}_{metric}_norm'
                if col in df:
                    if not np.isfinite(df[col].to_numpy()).all() or (df[col]<0).any():raise ValueError(f'{seq}: invalid {col}')
        if 'adaptive_ade_norm' in df:
            allowed={'stationary','cv_ols','transformer_ensemble'}
            if not set(df.selected_expert).issubset(allowed):raise ValueError(f'{seq}: unsupported selection')
            for metric in ['ade','fde']:
                expected=np.array([row[f'{sel}_{metric}_norm'] for sel,row in zip(df.selected_expert,df.to_dict('records'))])
                if not np.allclose(df[f'adaptive_{metric}_norm'].values,expected,rtol=0,atol=1e-11):
                    raise ValueError(f'{seq}: adaptive mismatch vs selected expert {metric}')
        df['sequence']=seq;df['role']=role;df['scene_component']=group
        if not (root/annot).is_file():raise FileNotFoundError(f'Missing annotation archive {annot}')
        # Phase 7F manifest's 'input_sha256' is a different intermediate .npz file, not boxid.zip.
        # Phase 7H/7K/7L explicitly report boxid ZIP hashes; verify those here.
        with zipfile.ZipFile(path) as z:
            names=[name for name in z.namelist() if name.endswith('evaluation_manifest.json')]
            if names:
                mf=json.loads(z.read(names[0]))
                if mf.get('file_sha256') != sha256(root/annot):
                    raise ValueError(f'{seq}: stored evaluation source hash does not match original annotation ZIP')
        for h in [20,50]:
            sub=df[df.horizon_frames==h]
            manifest.append({'sequence':seq,'horizon_frames':h,'n_windows':len(sub),'n_tracks':sub.track_id.nunique(),
                'source_zip':file,'source_sha256':sha256(path),'annotation_zip':annot,'annotation_sha256':sha256(root/annot),
                'role':role,'scene_component':group})
        all_frames.append(df)
    full=pd.concat(all_frames,ignore_index=True)
    return full,pd.DataFrame(manifest)


def score_tables(full):
    by=[]
    for (seq,h),df in full.groupby(['sequence','horizon_frames'],sort=False):
        for model in MODELS:
            if f'{model}_ade_norm' not in df or df[f'{model}_ade_norm'].isna().all():continue
            by.append(dict(sequence=seq,horizon_frames=int(h),model=model,n_windows=len(df),
                           n_tracks=int(df.track_id.nunique()),role=df.role.iloc[0],scene_component=df.scene_component.iloc[0],
                           ADE=float(df[f'{model}_ade_norm'].mean()), FDE=float(df[f'{model}_fde_norm'].mean())))
    scores=pd.DataFrame(by)
    best=[]
    for (seq,h),sub in scores.groupby(['sequence','horizon_frames']):
        eligible=sub[sub.model!='adaptive']
        bestmodel=eligible.loc[eligible.ADE.idxmin()]
        best.append(dict(sequence=seq,horizon_frames=h,best_expert=bestmodel.model,best_ADE=bestmodel.ADE))
    winners=pd.DataFrame(best)
    adaptive=[]
    for (seq,h),s in scores[scores.model=='adaptive'].groupby(['sequence','horizon_frames']):
        cv=scores[(scores.sequence==seq)&(scores.horizon_frames==h)&(scores.model=='cv_ols')].iloc[0]
        ad=s.iloc[0];win=winners[(winners.sequence==seq)&(winners.horizon_frames==h)].iloc[0]
        adaptive.append(dict(sequence=seq,horizon_frames=h,adaptive_ADE=ad.ADE,cv_ols_ADE=cv.ADE,
                             adaptive_minus_cv_ols_ADE=ad.ADE-cv.ADE,
                             adaptive_improvement_pct_vs_cv_ols=100*(cv.ADE-ad.ADE)/cv.ADE,
                             best_individual_model=win.best_expert,best_individual_ADE=win.best_ADE,
                             adaptive_vs_best_ADE_penalty_pct=100*(ad.ADE-win.best_ADE)/win.best_ADE,
                             inference_unit='one recording, not n_windows independent observations'))
    adapt=pd.DataFrame(adaptive)
    # scene-level descriptive mean: first average all tracks/windows within each recording,
    # then equal-weight each recording in the provisional scene group.
    # Adaptive was not evaluated in development recordings; OMIT it from scene group aggregation
    # to prevent a misleading cross-model comparison using different recording subsets.
    groups=scores[scores.model!='adaptive'].groupby(['scene_component','horizon_frames','model'],as_index=False).agg(
        unweighted_recording_mean_ADE=('ADE','mean'),unweighted_recording_mean_FDE=('FDE','mean'),
        n_recordings=('sequence','nunique'))
    groups['author_verified_independent_flight']=False
    return scores,winners,adapt,groups


def scale_floor_audit(root):
    paths={
      'DJI_0207':('Phase7F_Frozen_Results.zip','canonicalization_statistics.csv'),
      'DJI_0204_video2':('Phase7H_DJI0204_Frozen_Evaluation.zip','canonicalization_scale_floor_rates.csv'),
      'DJI_0204_video1':('Phase7L_DJI0204_video1_Exploratory_Results.zip','scale_floor_rates.csv'),
      'DJI_0161_video1':('Phase7K_DJI0161_Locked_Evaluation.zip','scale_floor_rates.csv'),
    }
    out=[]
    for seq,(fn,suffix) in paths.items():
        with zipfile.ZipFile(root/fn) as z:
            member=[name for name in z.namelist() if name.endswith(suffix)]
            if len(member)!=1:raise ValueError('cannot identify scale-floor summary')
            df=pd.read_csv(io.BytesIO(z.read(member[0])))
        for h,one in df.groupby('horizon_frames'):
            if seq=='DJI_0207':
                floor=float(one.floor_applied_count.sum()/one.windows.sum())
            else:
                floor=float((one.scale_floor_active_fraction*one.windows).sum()/one.windows.sum())
            out.append({'sequence':seq,'horizon_frames':int(h),'total_windows':int(one.windows.sum()),
                        'scale_floor_active_fraction':floor,
                        'provenance':'archived preprocessing statistics, not camera-motion correction'})
    return pd.DataFrame(out)


def camera_analysis(root):
    pair_rows=[];track_rows=[]
    for seq,(jpg,pairs) in IMG_ARCHIVES.items():
        with zipfile.ZipFile(root/jpg) as jpgz,zipfile.ZipFile(root/DATA_FILES[seq][2]) as ann:
            if jpgz.testzip() is not None or ann.testzip() is not None:raise ValueError('corrupt image or annotation zip')
            for a,b in pairs:
                if b != a+1: raise AssertionError('Not consecutive frames')
                im0=get_jpg(jpgz,a);im1=get_jpg(jpgz,b)
                boxes0=get_annotation_boxes(ann,a);boxes1=get_annotation_boxes(ann,b)
                records=cam_pair_rows(seq,a,b,im0,im1,boxes0,boxes1)
                track_rows.extend(records)
                q=records[0]
                pair_rows.append(dict(sequence=seq,frame_source=a,frame_target=b,
                           n_annotated_elephants=len(records),
                           median_raw_bbox_center_step_px=np.median([r['raw_bbox_center_step_px'] for r in records]),
                           median_estimated_local_background_step_px=np.median([r['estimated_local_background_step_px'] for r in records]),
                           median_residual_after_BG_alignment_px=np.median([r['residual_after_BG_alignment_px'] for r in records]),
                           **{k:q[k] for k in ['ratio_matches','inliers','inlier_fraction','inlier_hull_coverage','spatial_4x4_cells','median_reprojection_error_small_px','bg_registration_qc_pass']}))
    return pd.DataFrame(pair_rows), pd.DataFrame(track_rows)


def make_figures(scores, cam, details, output):
    figdir=output/'figures';figdir.mkdir(exist_ok=True,parents=True)
    seqorder=['DJI_0207','DJI_0204_video2','DJI_0204_video1','DJI_0161_video1']
    for h in [20,50]:
        plt.figure(figsize=(11,5.8))
        for model in MODELS:
            cur=scores[(scores.horizon_frames==h)&(scores.model==model)].set_index('sequence').reindex(seqorder)
            plt.plot(range(len(seqorder)),cur.ADE,marker='o',linewidth=2,label=model.replace('_',' '))
        plt.xticks(range(len(seqorder)),['0207\nDev','0204 video2\nDev','0204 video1\nRelated scene','0161 video1\nPrior test'],fontsize=9)
        plt.ylabel('Mean ADE (normalized image-plane units)')
        plt.title(f'Phase 7N | 20 observed → {h} future frames\nModels were NOT retrained; adaptive available on only 2 recordings')
        plt.legend(ncol=3,fontsize=8);plt.grid(axis='y',alpha=.25);plt.tight_layout()
        plt.savefig(figdir/f'01_locked_recording_ADE_h{h}.png',dpi=175);plt.close()
    c=cam.copy(); c['pair']=c['sequence'].str.replace('DJI_','',regex=False)+'\n'+c.frame_source.astype(str)+'→'+c.frame_target.astype(str)
    x=np.arange(len(c));wid=.27
    plt.figure(figsize=(13,6))
    for j,(col,label) in enumerate([('median_raw_bbox_center_step_px','Raw bounding-box centre'),('median_estimated_local_background_step_px','Background-induced proxy'),('median_residual_after_BG_alignment_px','After 2D registration')]):
        plt.bar(x+(j-1)*wid,c[col],wid,label=label)
    plt.xticks(x,c['pair'],fontsize=8);plt.ylabel('Median image-plane displacement (pixels)')
    plt.title('Sparse consecutive-frame camera registration | not corrected forecast ADE')
    plt.legend(fontsize=8);plt.grid(axis='y',alpha=.2);plt.tight_layout()
    plt.savefig(figdir/'02_sparse_background_motion_AUDIT.png',dpi=175);plt.close()
    # log-scale per-animal residual highlights focal annotation anomalies vs group medians.
    d=details.copy(); d['pair']=d['sequence'].str.replace('DJI_','',regex=False)+' '+d.frame_source.astype(str)+'→'+d.frame_target.astype(str)
    pairs=list(d['pair'].drop_duplicates());
    fig,ax=plt.subplots(figsize=(12,5.3))
    for pid,t in d.groupby('track_id'):
        vals=t.set_index('pair').reindex(pairs)['residual_after_BG_alignment_px']
        ax.plot(range(len(pairs)),vals,marker='.',linewidth=1.2,label=f'Track {pid}')
    ax.set_xticks(range(len(pairs)),pairs,rotation=20,ha='right',fontsize=8)
    ax.set_ylabel('Post-registration bbox-centre residual (pixels)')
    ax.set_title('Track-specific residuals after estimated background alignment')
    ax.grid(axis='y',alpha=.25);ax.legend(ncol=6,fontsize=8);fig.tight_layout()
    fig.savefig(figdir/'03_track_residuals_spot_QA.png',dpi=175);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir',required=True,type=Path,help='Directory with original named ZIPs')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    output=args.output;output.mkdir(exist_ok=True,parents=True)
    full,inputs=load_and_validate(args.inputs_dir)
    scores,winners,adaptive,groups=score_tables(full)
    pairs,tracks=camera_analysis(args.inputs_dir)
    floor=scale_floor_audit(args.inputs_dir)
    inputs.to_csv(output/'input_provenance_sha256.csv',index=False)
    floor.to_csv(output/'scale_floor_regime_summary.csv',index=False,float_format='%.12g')
    scores.to_csv(output/'recording_level_ADE_FDE.csv',index=False,float_format='%.12g')
    winners.to_csv(output/'best_expert_per_recording.csv',index=False,float_format='%.12g')
    adaptive.to_csv(output/'locked_adaptive_vs_cv_ols.csv',index=False,float_format='%.12g')
    groups.to_csv(output/'provisional_scene_component_summary.csv',index=False,float_format='%.12g')
    pairs.to_csv(output/'consecutive_frame_camera_registration.csv',index=False,float_format='%.12g')
    tracks.to_csv(output/'per_track_raw_vs_registered_motion.csv',index=False,float_format='%.12g')
    make_figures(scores,pairs,tracks,output)
    check={
        'phase':'7N','scope':'descriptive/retrospective; no independent flight IDs or seconds-based forecasts',
        'total_windows':int(len(full)),'windows_by_recording_and_horizon':inputs[['sequence','horizon_frames','n_windows']].to_dict('records'),
        'n_consecutive_image_pairs':int(len(pairs)),
        'n_animal_frame_pairs':int(len(tracks)),
        'camera_bg_registration_qc_pass_pairs':int(pairs.bg_registration_qc_pass.sum()),
        'two_d_registration_not_ground_truth':True,
        'n_verified_independent_capture_events':0,
        'prediction_coordinates_in_test_result_archives':False,
        'camera_corrected_ADE_FDE_available':False,
        'gate_retrained_in_7N':False,
        'transformer_rerun_in_7N':False,
        'claim':'no camera-corrected trajectory forecasting accuracy estimate from these sparse frames'
    }
    (output/'phase7n_audit_manifest.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(check,indent=2))
    print('\nAdaptive comparison:\n',adaptive.to_string(index=False))
    print('\nCamera pairs:\n',pairs[['sequence','frame_source','frame_target','median_raw_bbox_center_step_px','median_estimated_local_background_step_px','median_residual_after_BG_alignment_px','inliers','inlier_hull_coverage']].to_string(index=False))

if __name__=='__main__':main()
