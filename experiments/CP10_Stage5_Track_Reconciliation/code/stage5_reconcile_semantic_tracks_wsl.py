#!/usr/bin/env python3
"""Read-only KITTI elephant vs generic 3D tracker 2D-box reconciliation.

All source media remain untouched. Sample eligibility is a *geometry/annotation*
feasibility check, NOT a forecast or held-out evaluation.
Requires only Python 3 stdlib.
"""
import argparse, csv, json, re, zipfile, math, os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

VIDEOS = ('DJI_20240119152552_0006_V', 'DJI_20240626143832_0004_V', 'DJI_20260227084907_0004_V')
RECORDING_GROUP = {VIDEOS[0]:'elep1', VIDEOS[1]:'elep2', VIDEOS[2]:'elep3'}

def iou(a,b):
    if not a or not b: return 0.
    try:
        a=[float(x) for x in a];b=[float(x) for x in b]
        if len(a)!=4 or len(b)!=4: return 0.
        area=lambda q:max(0.,q[2]-q[0])*max(0.,q[3]-q[1])
        ix=max(0.,min(a[2],b[2])-max(a[0],b[0]))
        iy=max(0.,min(a[3],b[3])-max(a[1],b[1]))
        overlap=ix*iy
        return overlap/max(area(a)+area(b)-overlap,1e-15)
    except (ValueError,TypeError): return 0.

def read_json(z,n):return json.loads(z.read(n))

def safe_csv(path,rows,columns):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def get_kitti_boxes(text):
    ret=[]
    for l in text.splitlines():
        parts=l.strip().split()
        if len(parts)<8: continue
        try:
            xyxy=[float(v) for v in parts[4:8]]
            if xyxy[2]<=xyxy[0] or xyxy[3]<=xyxy[1]: continue
            ret.append((parts[0].lower(),xyxy))
        except ValueError: continue
    return ret

def longest_run(values):
    current=best=0; prev=None
    for v in sorted(set(values)):
        current=current+1 if prev is not None and v==prev+1 else 1
        prev=v;best=max(best,current)
    return best

def windows(indices,total):
    ss=set(indices); return sum(all(i+j in ss for j in range(total)) for i in ss)

def find_archive(root,video,preferred=None):
    if preferred and Path(preferred).exists():return Path(preferred)
    locations=[root/'Main Result'/f'{video}.zip',root/'ZIP 4.0'/f'{video}.zip']
    for group in ('elep1','elep2','elep3'):
        locations.append(root/'Q1_Research_Completion_Sprint'/'new_data'/group/f'{video}.zip')
    for loc in locations:
        if loc.is_file():return loc
    return None

def discover_segment_dirs(z):
    result=defaultdict(dict)
    for name in z.namelist():
        m=re.search(r'(^|/)(seg\d+)/(tracking_summary\.json|cameras\.json)$',name)
        if not m:continue
        base=name.rsplit('/',1)[0]
        result[base][m.group(3)]=name
    return {base:x for base,x in result.items() if 'cameras.json' in x and 'tracking_summary.json' in x}

def format_bbox(b,iw,ih,sw,sh):
    return [float(b[0])*iw/sw,float(b[1])*ih/sh,float(b[2])*iw/sw,float(b[3])*ih/sh]

def load_segment(z,base,files,video,threshold,semantic_fraction):
    cams=read_json(z,files['cameras.json'])['cameras']
    tracks=read_json(z,files['tracking_summary.json'])['tracks']
    members=set(z.namelist())
    tracker_by_frame=defaultdict(list)
    for tid,t in tracks.items():
        frames=t.get('frames',[]); boxes=t.get('bbox_2d',[])
        for fi,b in zip(frames,boxes):
            if b and len(b)==4:tracker_by_frame[int(fi)].append((str(tid),b))
    records=[]
    for cam in cams:
        idx=int(cam.get('frame_index',-1));im=cam.get('image_name','');
        if not im:continue
        ln=base+'/kitti_labels/'+Path(im).with_suffix('.txt').name
        if ln not in members:continue
        gt=get_kitti_boxes(z.read(ln).decode('utf-8',errors='replace'))
        records.append((idx,im,int(cam.get('image_width',1920)),int(cam.get('image_height',1080)),gt,tracker_by_frame.get(idx,[])))
    # Choose detector's implicit input size on *calibration/sample geometry*,
    # not on predicted forecast accuracy. A separate audit should inspect this.
    sizes=((512,288),(640,360),(960,540),(1280,720),(1920,1080))
    candidates=[]
    for sw,sh in sizes:
        matches=tot=0; ious=[]
        for idx,im,iw,ih,gt,pts in records:
            eles=[b for cl,b in gt if cl=='elephant']
            if not eles:continue
            for tid,b in pts:
                score=max((iou(format_bbox(b,iw,ih,sw,sh),q) for q in eles),default=0.)
                ious.append(score)
                matches+=score>=threshold; tot+=1
        candidates.append((matches, sum(ious)/(len(ious) or 1), sw,sh,tot))
    matches,mean_iou,sw,sh,tot=max(candidates)
    votes=defaultdict(lambda:Counter())
    frames_by_id=defaultdict(set); matched_by_id=defaultdict(set); ious_by_id=defaultdict(list)
    gt_total=gt_elephants=gt_matched=0
    ambiguous=0
    for idx,im,iw,ih,gt,pts in records:
        gt_total+=len(gt);gt_elephants+=sum(cl=='elephant' for cl,b in gt)
        for tid,b in pts:frames_by_id[tid].add(idx)
        # One-to-one match for elephant GT (non elephant labels remain potential controls)
        edges=[]
        for ti,(tid,b) in enumerate(pts):
            b2=format_bbox(b,iw,ih,sw,sh)
            for gi,(cl,q) in enumerate(gt):
                val=iou(b2,q)
                if val>=threshold:edges.append((val,ti,gi))
        edges.sort(reverse=True)
        used_t=set();used_g=set()
        for val,ti,gi in edges:
            if ti in used_t or gi in used_g:continue
            used_t.add(ti);used_g.add(gi)
            tid=pts[ti][0]; cl=gt[gi][0]
            votes[tid][cl]+=1;ious_by_id[tid].append(val)
            if cl=='elephant':matched_by_id[tid].add(idx);gt_matched+=1
        ambiguous+=len(pts)-len(used_t)
    track_rows=[]
    for tid,t in tracks.items():
        tid=str(tid); n=len(frames_by_id[tid]); m=len(matched_by_id[tid]);
        gt_fraction=round(m/n,4) if n else 0.
        resolved=n>=10 and gt_fraction>=semantic_fraction
        present=frames_by_id[tid]; verified=matched_by_id[tid]
        track_rows.append({'video':video,'segment':base.rsplit('/',1)[-1], 'track_id':tid,
          'tracking_class':t.get('class_name'),'observed_frames':n,'gt_elephant_matched_frames':m,
          'elephant_match_fraction':gt_fraction,'matched_gt_classes':json.dumps(dict(votes[tid]),sort_keys=True),
          'semantic_elephant':bool(resolved),'longest_tracking_run':longest_run(present),
          'longest_verified_elephant_run':longest_run(verified),
          'observed_2s_windows':windows(present,40),'observed_5s_windows':windows(present,70),
          'strict_elephant_2s_windows':windows(verified,40) if resolved else 0,
          'strict_elephant_5s_windows':windows(verified,70) if resolved else 0,
          'mean_matched_iou':round(sum(ious_by_id[tid])/len(ious_by_id[tid]),5) if ious_by_id[tid] else 0})
    segment={'video':video,'segment':base.rsplit('/',1)[-1], 'paired_frame_labels':len(records),
      'GT_elephant_boxes':gt_elephants,'GT_elephant_matched':gt_matched,
      'track_count':len(tracks),'resolved_elephant_tracks':sum(x['semantic_elephant'] for x in track_rows),
      'assumed_tracker_input_width':sw,'assumed_tracker_input_height':sh,
      'matched_box_pairs_at_iou_threshold':matches,
      'mean_max_iou_over_tracker_frames':round(mean_iou,5),'unmatched_tracking_boxes':ambiguous,
      'possible_2s_windows_resolved':sum(x['observed_2s_windows'] for x in track_rows if x['semantic_elephant']),
      'possible_5s_windows_resolved':sum(x['observed_5s_windows'] for x in track_rows if x['semantic_elephant']),
      'strict_GT_matched_2s_windows':sum(x['strict_elephant_2s_windows'] for x in track_rows),
      'strict_GT_matched_5s_windows':sum(x['strict_elephant_5s_windows'] for x in track_rows)}
    return segment,track_rows

def collect_mentions(root,video_ids,max_bytes=4_000_000):
    results=[]
    candidate_dirs=[root/'Main Result 3.0'/'paper_strengthening_revision',root/'Q1_Research_Completion_Sprint']
    extensions={'.csv','.json','.txt','.md','.log','.yaml','.yml'}
    for base in candidate_dirs:
        if not base.is_dir():continue
        for dirpath,folders,files in os.walk(base):
            # skip huge source/download/model directories and image collections
            folders[:]=[f for f in folders if f.lower() not in ('images','new_data','.cache','checkpoints','models','train2017','val2017','raw_images')]
            for fname in files:
                f=Path(dirpath)/fname
                if f.suffix.lower() not in extensions:continue
                try:
                    if f.stat().st_size>max_bytes:continue
                    txt=f.read_text(encoding='utf-8',errors='replace')
                except (OSError,UnicodeError):continue
                for vid in video_ids:
                    if vid in txt or vid in fname:
                        results.append({'video':vid,'file':str(f),'note':'mention only: not proof of model evaluation or holdout leakage'})
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('/mnt/e/AIUB/Capstone Project/Journal Based Results'))
    ap.add_argument('--out',type=Path,default=Path.home()/'CP10_Stage5_Semantic_Track_Audit')
    ap.add_argument('--audit-json',type=Path,default=Path(__file__).with_name('CP10_Stage5_Candidate_Audit.json'))
    ap.add_argument('--iou',type=float,default=.5)
    ap.add_argument('--min-track-fraction',type=float,default=.8)
    ap.add_argument('--no-provenance-scan',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    meta=json.loads(a.audit_json.read_text(encoding='utf-8'))
    segments=[]; tracks=[];errors=[]
    for cand in meta['candidate_archives']:
        video=cand['video']; src=find_archive(a.root,video,cand.get('source_zip'))
        if src is None:
            errors.append({'video':video,'reason':'Archive not found'});continue
        try:
            with zipfile.ZipFile(src) as z:
                for base,files in discover_segment_dirs(z).items():
                    seg,track=load_segment(z,base,files,video,a.iou,a.min_track_fraction)
                    seg['group']=RECORDING_GROUP.get(video,'unknown')
                    seg['source_zip']=str(src)
                    segments.append(seg);tracks.extend(track)
        except Exception as e:
            errors.append({'video':video,'reason':f'{type(e).__name__}: {str(e)}'})
    mentions=[] if a.no_provenance_scan else collect_mentions(a.root,VIDEOS)
    seg_cols=['group','video','segment','paired_frame_labels','GT_elephant_boxes','GT_elephant_matched','track_count','resolved_elephant_tracks','assumed_tracker_input_width','assumed_tracker_input_height','matched_box_pairs_at_iou_threshold','mean_max_iou_over_tracker_frames','unmatched_tracking_boxes','possible_2s_windows_resolved','possible_5s_windows_resolved','strict_GT_matched_2s_windows','strict_GT_matched_5s_windows','source_zip']
    track_cols=['video','segment','track_id','tracking_class','observed_frames','gt_elephant_matched_frames','elephant_match_fraction','matched_gt_classes','semantic_elephant','longest_tracking_run','longest_verified_elephant_run','observed_2s_windows','observed_5s_windows','strict_elephant_2s_windows','strict_elephant_5s_windows','mean_matched_iou']
    safe_csv(a.out/'CP10_Stage5_Semantic_Segments.csv',segments,seg_cols)
    safe_csv(a.out/'CP10_Stage5_Semantic_Tracks.csv',tracks,track_cols)
    safe_csv(a.out/'CP10_Stage5_Prior_Usage_Mentions.csv',mentions,['video','file','note'])
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'method':'per-frame one-to-one KITTI GT vs tracker bbox_2d IoU matching','iou_threshold':a.iou,'min_track_elephant_match_fraction':a.min_track_fraction,'tracker_input_size_inferred_from_candidate_sizes':True,
      'segments':segments,'tracks':tracks,'prior_use_mentions':mentions,'errors':errors,
      'caution':['Generic tracking class object is reconciled to KITTI elephant only by per-frame box matching; semantic labels remain inferred, not re-annotated.',
      'GT bounding boxes are not detector predictions; these are annotation-conditioned potential forecast windows.',
      'The source camera metadata and tracking_summary must be used to reconstruct actual trajectories before inference.',
      'All candidate videos share previously inspected elep1/elep2/elep3 recording groups. No independent group validation claim.',
      'Previously inspected qualitative video 0006 is development-only. Video name mentions in other files are evidence to review, not conclusive prior evaluation.',
      'Do not use all windows as independent; many overlap and share tracks and scenes.']}
    (a.out/'CP10_Stage5_Semantic_Audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=[f'Stage 5 semantic track-box reconciliation | {len(segments)} candidate segments',f'IoU threshold={a.iou}; minimum elephant matching fraction={a.min_track_fraction}',f'Prior-use mentions found: {len(mentions)} (mentions do not prove previous evaluation)']
    for seg in segments:
        lines.append(f"{seg['group']} {seg['video']} {seg['segment']}: {seg['GT_elephant_matched']}/{seg['GT_elephant_boxes']} GT elephant boxes matched, resolved tracks={seg['resolved_elephant_tracks']}/{seg['track_count']}; observed potential windows 2s/5s={seg['possible_2s_windows_resolved']}/{seg['possible_5s_windows_resolved']}; strict GT verified={seg['strict_GT_matched_2s_windows']}/{seg['strict_GT_matched_5s_windows']}; internal bbox size={seg['assumed_tracker_input_width']}x{seg['assumed_tracker_input_height']}")
    for e in errors:lines.append('ERROR '+str(e))
    lines.extend(['\nInterpretation: preliminary sample feasibility only, NOT independent confirmatory calibration.', 'Next: audit source overlap and retain candidate 0004 videos as retrospective same-group exploratory material; confirmatory results require a new recording group.'])
    (a.out/'CP10_Stage5_Semantic_Decision.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    zpath=a.out/'CP10_Stage5_Semantic_Audit_Results.zip'
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED) as zout:
        for path in sorted(a.out.iterdir()):
            if path.is_file() and path!=zpath: zout.write(path,path.name)
    print('\n'.join(lines));print('Output ZIP:',zpath)

if __name__=='__main__':main()
