#!/usr/bin/env python3
"""WildLive DJI_0207 descriptive, frame-index-only 20->20/50 forecasting.

NO temporal (seconds) interpretation, 10-Hz assumption, camera correction, or
independent confirmatory dataset claim. Uses original annotation bbox centres.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, zipfile
from pathlib import Path
import numpy as np

W=3840
H=2160
OBS=20
HORIZONS=(20,50)
QA_FLAGS={0:{860}, 1:set(), 2:{389,401}}


def hash_file(path):
    sha=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): sha.update(chunk)
    return sha.hexdigest()


def read_annotations(zip_path):
    observed={}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            match=re.search(r'(?:^|/)frame_(\d+)\.json$',name)
            if not match: continue
            frame=int(match.group(1))
            if frame in observed: raise ValueError(f'duplicate frame={frame}')
            data=json.loads(z.read(name))
            if (data.get('imageWidth'),data.get('imageHeight')) != (W,H):
                raise ValueError(f'annotation dimensions changed at frame {frame}')
            shapes={}
            for s in data['shapes']:
                if s.get('label','').strip().lower()!='elephant': continue
                ident=int(s['group_id'])
                if ident in shapes: raise ValueError(f'duplicate ID={ident} at frame={frame}')
                pts=np.asarray(s['points'],dtype=np.float64)
                if pts.ndim!=2 or pts.shape[1]!=2 or pts.shape[0]<2 or not np.isfinite(pts).all():
                    raise ValueError(f'invalid bbox: frame={frame}, id={ident}')
                x1,y1=np.min(pts,axis=0);x2,y2=np.max(pts,axis=0)
                if x1<0 or y1<0 or x2>W or y2>H or x2<=x1 or y2<=y1:
                    raise ValueError(f'out-of-bounds bbox frame={frame} id={ident}')
                shapes[ident]={'xy':((x1+x2)/(2*W),(y1+y2)/(2*H)),
                               'box_native':(x1,y1,x2,y2),
                               'score':s.get('score',None)}
            observed[frame]=shapes
    if sorted(observed) != list(range(868)):
        raise ValueError(f'expected exact 0..867 indices; got {len(observed)} frames')
    ids=sorted({i for group in observed.values() for i in group})
    if ids!=[0,1,2]: raise ValueError(f'expected IDs 0,1,2; got {ids}')
    for frame,shapes in observed.items():
        if sorted(shapes)!=ids: raise ValueError(f'ID missing at frame={frame}: {sorted(shapes)}')
    tracks={i:np.asarray([observed[t][i]['xy'] for t in range(868)],dtype=np.float64) for i in ids}
    assert all(np.isfinite(x).all() for x in tracks.values())
    return tracks,observed


def predict_stationary(history,horizon):
    return np.broadcast_to(history[-1],(horizon,2)).copy()


def predict_cv20(history,horizon):
    velocity=(history[-1]-history[0])/(len(history)-1)
    return history[-1]+np.arange(1,horizon+1,dtype=np.float64)[:,None]*velocity


def ade_fde(pred,truth):
    distance=np.linalg.norm(pred-truth,axis=-1)
    return float(distance.mean()),float(distance[-1])


def make_windows(track,horizon,flags):
    assert track.ndim==2 and track.shape[1]==2
    for start in range(len(track)-OBS-horizon+1):
        stop=start+OBS+horizon
        bad=any(start<=f<stop for f in flags)
        yield start,track[start:start+OBS],track[start+OBS:stop],bad


def evaluate(tracks):
    details=[]; summary=[]; arr={}
    for horizon in HORIZONS:
        for ident,track in tracks.items():
            xs=[]; ys=[]; preds_stationary=[];preds_cv=[];starts=[];flags=[]
            for start,x,y,bad in make_windows(track,horizon,QA_FLAGS[ident]):
                ps=predict_stationary(x,horizon)
                pc=predict_cv20(x,horizon)
                ade_s,fde_s=ade_fde(ps,y); ade_c,fde_c=ade_fde(pc,y)
                details.append({'horizon_frames':horizon,'track_id':ident,'first_observed_frame':start,
                                'last_observed_frame':start+OBS-1,'last_target_frame':start+OBS+horizon-1,
                                'qa_flagged_window':int(bad),
                                'stationary_ade_norm':ade_s,'stationary_fde_norm':fde_s,
                                'cv20_ade_norm':ade_c,'cv20_fde_norm':fde_c})
                xs.append(x); ys.append(y); preds_stationary.append(ps); preds_cv.append(pc);starts.append(start);flags.append(bad)
            key=f'id{ident}_h{horizon}'
            arr[f'{key}_history']=np.stack(xs)
            arr[f'{key}_future_gt']=np.stack(ys)
            arr[f'{key}_stationary_pred']=np.stack(preds_stationary)
            arr[f'{key}_cv20_pred']=np.stack(preds_cv)
            arr[f'{key}_start_frame']=np.asarray(starts,dtype='int32')
            arr[f'{key}_qa_flagged']=np.asarray(flags,dtype=bool)
    for horizon in HORIZONS:
        for subset in ('all','QA_unflagged'):
            for ident in ('pooled',0,1,2):
                rows=[r for r in details if r['horizon_frames']==horizon and (ident=='pooled' or r['track_id']==ident)
                      and (subset=='all' or not r['qa_flagged_window'])]
                if not rows:continue
                for model,prefix in [('stationary','stationary'),('CV20','cv20')]:
                    summary.append({'horizon_frames':horizon,'subset':subset,'track_id':ident,
                                    'baseline':model,'windows':len(rows),
                                    'ade_norm':np.mean([r[f'{prefix}_ade_norm'] for r in rows]),
                                    'fde_norm':np.mean([r[f'{prefix}_fde_norm'] for r in rows])})
    return details,summary,arr


def write_csv(path,rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def plot_results(tracks,summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6))
    for ident,coords in tracks.items():
        ax.plot(coords[:,0],coords[:,1],lw=1.3,label=f'Elephant ID {ident}')
        ax.scatter(coords[0,0],coords[0,1],marker='o',s=24)
    ax.set_title('DJI_0207: observed annotation-centre trajectories (frame order)')
    ax.set_xlabel('Bounding-box centre x / image width [normalized]')
    ax.set_ylabel('Bounding-box centre y / image height [normalized]')
    ax.invert_yaxis();ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/'DJI0207_raw_annotation_centre_tracks.png',dpi=165);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    keys=[(h,b) for h in HORIZONS for b in ('stationary','CV20')]
    vals=[next(float(r['ade_norm']) for r in summary if r['horizon_frames']==h and r['subset']=='all' and r['track_id']=='pooled' and r['baseline']==b) for h,b in keys]
    ax.bar(range(len(vals)),vals)
    ax.set_xticks(range(len(vals)),[f'{h}-frame\n{b}' for h,b in keys]);ax.set_ylabel('ADE [normalized image-plane units]')
    ax.set_title('DJI_0207: descriptive baselines on overlapping windows')
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(out/'DJI0207_frame_based_ADE_baselines.png',dpi=165);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zip',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    tracks,obs=read_annotations(args.zip)
    details,summary,arr=evaluate(tracks)
    raw_rows=[]
    for frame in range(868):
        for ident in (0,1,2):
            box=obs[frame][ident]
            x1,y1,x2,y2=box['box_native']
            raw_rows.append({'frame_index':frame,'track_id':ident,'x_center_normalized':box['xy'][0],
                             'y_center_normalized':box['xy'][1],
                             'bbox_x1_native_px':x1,'bbox_y1_native_px':y1,
                             'bbox_x2_native_px':x2,'bbox_y2_native_px':y2,
                             'score_original':box['score'],'qa_review_flag':int(frame in QA_FLAGS[ident])})
    write_csv(args.output/'original_bbox_centres.csv',raw_rows)
    write_csv(args.output/'per_window_baseline_metrics.csv',details)
    write_csv(args.output/'baseline_summary.csv',summary)
    np.savez_compressed(args.output/'frame_based_windows_and_baselines.npz',**arr)
    plot_results(tracks,summary,args.output)
    metadata={'input_annotation_zip':str(args.zip),'input_sha256':hash_file(args.zip),
              'dataset':'WildLive DJI_0207','source_annotation_resolution_px':[W,H],
              'frames_per_track':868,'track_ids':[0,1,2],
              'observation_frames':OBS,'future_frames':list(HORIZONS),'window_stride_frames':1,
              'prediction_units':'normalized original image-plane (not metres; no camera correction)',
              'no_fps':True,'no_verified_timestamps':True,'no_transformer_evaluation':True,
              'not_independent_confirmation':True,'number_independent_capture_groups':1,
              'qa_flags_from_prior_image_review':{str(i):sorted(v) for i,v in QA_FLAGS.items()},
              'qa_exclusion_criterion':'drop window if flagged frame lies anywhere in history OR future of that track',
              'total_windows':len(details),'overlapping_windows':True,
              'missing_annotation_scores':sum(int(o['score'] is None) for v in obs.values() for o in v.values())}
    (args.output/'audit_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))
    print('SUMMARY:')
    for s in summary:
        if s['track_id']=='pooled':print(s)

if __name__=='__main__':main()
