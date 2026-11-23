#!/usr/bin/env python3
"""Reproducible, annotation-only WildLive DJI_0204_video2 intake audit.

No image-based verification, timestamps, camera-motion correction, or model inference.
Usage: python audit_dji0204.py --zip /path/to/DJI_0204_video2_boxid.zip --output ./audit_outputs
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse,csv,hashlib,json,math,re,zipfile
import numpy as np

FRAME = re.compile(r'(?:^|/)frame_(\d+)\.json$',re.I)

def save_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

def bbox_iou(a,b):
    x0,y0,x1,y1=a;X0,Y0,X1,Y1=b
    w=max(0,min(x1,X1)-max(x0,X0));h=max(0,min(y1,Y1)-max(y0,Y0))
    inter=w*h; union=(x1-x0)*(y1-y0)+(X1-X0)*(Y1-Y0)-inter
    return float(inter/union) if union>0 else float('nan')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    src=args.zip.read_bytes();sha=hashlib.sha256(src).hexdigest()
    frames={};labels=Counter();missing_scores=0;pt_format=Counter();dimensions=set()
    with zipfile.ZipFile(args.zip) as z:
        bad=z.testzip()
        if bad:raise ValueError('Corrupted ZIP member '+bad)
        for name in z.namelist():
            m=FRAME.search(name)
            if not m:continue
            frame=int(m.group(1))
            if frame in frames:raise ValueError('Duplicate frame index '+str(frame))
            obj=json.loads(z.read(name));W,H=obj['imageWidth'],obj['imageHeight'];dimensions.add((W,H))
            candidates={}
            for shape in obj['shapes']:
                label=str(shape.get('label','')).strip().lower();labels[label]+=1
                if label!='elephant':continue
                tid=shape.get('group_id')
                if tid is None or str(tid) in candidates:raise ValueError('Invalid/duplicate elephant group_id')
                pts=np.asarray(shape['points'],float);pt_format[len(pts)]+=1
                if pts.ndim!=2 or pts.shape[1]!=2 or not np.isfinite(pts).all() or len(pts)<2:
                    raise ValueError('Nonfinite/invalid points')
                lo=pts.min(axis=0); hi=pts.max(axis=0)
                if (lo<0).any() or (hi>[W,H]).any() or (hi<=lo).any():
                    raise ValueError('Invalid bounds')
                if shape.get('score') is None:missing_scores+=1
                candidates[str(tid)]=(float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1]))
            frames[frame]=candidates
    if not frames:raise ValueError('No frame JSON')
    if len(dimensions)!=1:raise ValueError('Mixed frame dimensions')
    W,H=dimensions.pop();ordered=sorted(frames)
    tracks=defaultdict(list); bboxrows=[];pairrows=[]
    for frame in ordered:
        for tid,b in frames[frame].items():
            x0,y0,x1,y1=b; cx=(x0+x1)/2;cy=(y0+y1)/2
            tracks[tid].append((frame,np.array([cx/W,cy/H]),b))
            bboxrows.append(dict(frame_index=frame,track_id=tid,label='elephant',center_x_norm=cx/W,center_y_norm=cy/H,xmin_px=x0,ymin_px=y0,xmax_px=x1,ymax_px=y1,width_px=x1-x0,height_px=y1-y0))
        tids=sorted(frames[frame],key=lambda x:int(x))
        for i,ti in enumerate(tids):
            for tj in tids[i+1:]:
                boxa,boxb=frames[frame][ti],frames[frame][tj]
                dist=math.hypot((boxa[0]+boxa[2]-boxb[0]-boxb[2])/2,(boxa[1]+boxa[3]-boxb[1]-boxb[3])/2)
                pairrows.append(dict(frame_index=frame,track_a=ti,track_b=tj,iou=bbox_iou(boxa,boxb),center_distance_px=dist))
    rows=[];step_rows=[]
    for tid,t in sorted(tracks.items(),key=lambda x:int(x[0])):
        ids=np.array([v[0] for v in t]);centres=np.stack([v[1] for v in t]);normdist=np.linalg.norm(np.diff(centres,axis=0),axis=1)
        contiguous=np.where(np.diff(ids)!=1)[0]
        run_start=0;runs=[]
        for cut in [*contiguous,len(t)-1]:
            runs.append(len(t[run_start:cut+1]));run_start=cut+1
        rows.append(dict(track_id=tid,n_annotations=len(t),first_frame=int(ids[0]),last_frame=int(ids[-1]),n_continuous_runs=len(runs),longest_run_frames=max(runs),eligible_windows_20=sum(max(0,n-39) for n in runs),eligible_windows_50=sum(max(0,n-69) for n in runs),median_adjacent_centre_jump_norm=float(np.median(normdist)),max_adjacent_centre_jump_norm=float(np.max(normdist))))
        for k,d in enumerate(normdist):
            if ids[k+1]!=ids[k]+1:continue
            step_rows.append(dict(track_id=tid,frame_from=int(ids[k]),frame_to=int(ids[k+1]),centre_jump_norm=float(d),centre_jump_px=float(math.hypot((centres[k+1,0]-centres[k,0])*W,(centres[k+1,1]-centres[k,1])*H))))
    save_csv(args.output/'track_availability.csv',rows)
    save_csv(args.output/'elephant_bbox_centres.csv',bboxrows)
    save_csv(args.output/'pairwise_bbox_overlap.csv',pairrows)
    save_csv(args.output/'adjacent_frame_centre_jumps.csv',step_rows)
    top=sorted(step_rows,key=lambda r:r['centre_jump_norm'],reverse=True)[:20]
    save_csv(args.output/'top_20_motion_QA_pairs.csv',top)
    pair_summary=[]
    pairs=sorted({(r['track_a'],r['track_b']) for r in pairrows})
    for a,b in pairs:
        chosen=[r for r in pairrows if (r['track_a'],r['track_b'])==(a,b)]
        ious=np.array([r['iou'] for r in chosen]); dists=np.array([r['center_distance_px'] for r in chosen])
        pair_summary.append(dict(track_a=a,track_b=b,frames_compared=len(chosen),median_iou=float(np.median(ious)),max_iou=float(np.max(ious)),median_center_distance_px=float(np.median(dists)),n_overlap_frames=int(np.sum(ious>0))))
    save_csv(args.output/'pairwise_overlap_summary.csv',pair_summary)
    manifest={'sequence':'DJI_0204_video2','input_file':args.zip.name,'input_sha256':sha,'input_bytes':len(src),'n_frames':len(ordered),'frame_index_min':min(ordered),'frame_index_max':max(ordered),'contiguous_frame_indices':len(ordered)==(max(ordered)-min(ordered)+1),'labels':dict(labels),'n_elephant_boxes':labels['elephant'],'n_tracks':len(tracks),'image_width_px':W,'image_height_px':H,'missing_elephant_scores':missing_scores,'points_per_box':dict(pt_format),'eligible_windows_20':sum(r['eligible_windows_20'] for r in rows),'eligible_windows_50':sum(r['eligible_windows_50'] for r in rows),'fps_verified':False,'source_independence_verified':False,'visual_species_verified':False,'visual_ID_verified':False,'model_evaluation_performed':False,'notes':['Annotations have label=elephant; no RGB sampled frame inspected','Overlap between IDs 1 and 3 needs visual identity check','Overlapping windows are not independent capture units','No timestamp or FPS evidence in annotation ZIP']}
    (args.output/'annotation_intake_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
