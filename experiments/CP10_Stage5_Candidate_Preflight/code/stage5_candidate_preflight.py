#!/usr/bin/env python3
"""Read-only audit of three candidate WildBox video archives for Stage 5.
No image/model predictions are generated. Python standard library only.
"""
import argparse, csv, json, re, sys, zipfile, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

CANDIDATES = [
    ('elep1','DJI_20240119152552_0006_V','previously_inspected_qualitative_frames'),
    ('elep2','DJI_20240626143832_0004_V','prior_use_not_yet_verified'),
    ('elep3','DJI_20260227084907_0004_V','prior_use_not_yet_verified'),
]
EXISTING_EXP3 = [
    'DJI_20240119151048_0001_V','DJI_20240119151437_0002_V',
    'DJI_20240626143248_0002_V','DJI_20240626143443_0003_V',
    'DJI_20260227082707_0003_V','DJI_20260227084724_0003_V',
]
R10_REUSED = [
    'DJI_20240119152335_0005_V','DJI_20240626142859_0001_V',
    'DJI_20260227084135_0002_V',
]
FRAME_RE = re.compile(r'(?:^|/)(seg\d+)/frame_(\d+)\.(?:jpg|jpeg|png)$', re.I)
LABEL_RE = re.compile(r'(?:^|/)(seg\d+)/kitti_labels/frame_(\d+)\.txt$', re.I)
META_RE = re.compile(r'(?:^|/)(seg\d+)/(cameras|tracking_summary)\.json$',re.I)

def longest_consecutive(nums):
    if not nums: return 0
    s=sorted(set(int(x) for x in nums));best=run=1
    for a,b in zip(s,s[1:]):
        run=run+1 if b==a+1 else 1;best=max(best,run)
    return best

def resolve_path(root,group,video):
    paths = [
        root/'Q1_Research_Completion_Sprint'/'new_data'/group/(video+'.zip'),
        root/'ZIP 4.0'/(video+'.zip'),
        root/'Main Result'/(video+'.zip'),
    ]
    for p in paths:
        if p.is_file(): return p
    return None

def safe_json(zf,name):
    try:
        if zf.getinfo(name).file_size > 15_000_000:return None
        return json.loads(zf.read(name))
    except Exception: return None

def analyze(zpath,group,video,history_frames=20,future_2s=20,future_5s=50,check_crc=False):
    results=[]
    with zipfile.ZipFile(zpath) as z:
        if check_crc:
            bad=z.testzip()  # Optional full-archive decompression; can take considerable time
            if bad: raise ValueError(f'ZIP CRC failure at member: {bad}')
        frames=defaultdict(set); labels=defaultdict(dict); meta=defaultdict(dict)
        for name in z.namelist():
            m=FRAME_RE.search(name)
            if m: frames[m.group(1)].add(int(m.group(2)));continue
            m=LABEL_RE.search(name)
            if m: labels[m.group(1)][int(m.group(2))]=name;continue
            m=META_RE.search(name)
            if m: meta[m.group(1)][m.group(2).lower()]=name
        for seg in sorted(set(frames)|set(labels)):
            ids=sorted(frames[seg]); paired=sorted(set(ids)&set(labels[seg])); hist=Counter(); gt_ele_frames=0; gt_multi_frames=0; nonempty=0
            for fid in paired:
                raw=z.read(labels[seg][fid]).decode('utf8',errors='replace')
                classes=[line.split()[0].lower() for line in raw.splitlines() if line.split()]
                hist.update(classes); ele=sum(c in {'elephant','elephants'} for c in classes)
                gt_ele_frames += ele>0;gt_multi_frames += ele>=2;nonempty += bool(classes)
            ts=safe_json(z,meta[seg]['tracking_summary']) if 'tracking_summary' in meta[seg] else None
            cams=safe_json(z,meta[seg]['cameras']) if 'cameras' in meta[seg] else None
            all_tracks=[]
            if isinstance(ts,dict):
                td=ts.get('tracks',{})
                it=td.items() if isinstance(td,dict) else enumerate(td) if isinstance(td,list) else []
                for tid,track in it:
                    if not isinstance(track,dict):continue
                    cls=str(track.get('class_name','unknown')).lower()
                    idxs=track.get('frames',[])
                    L=longest_consecutive(idxs) if isinstance(idxs,list) else 0
                    all_tracks.append({'track':str(tid),'class':cls,'longest_consecutive':L,
                                       'possible_2s_windows':max(0,L-(history_frames+future_2s)+1),
                                       'possible_5s_windows':max(0,L-(history_frames+future_5s)+1)})
            elephant_tracks=[x for x in all_tracks if x['class'] in {'elephant','elephants'}]
            ambiguous_tracks=[x for x in all_tracks if x['class']=='object']
            results.append({
                'group':group,'video':video,'segment':seg,'frames':len(ids), 'labels':len(labels[seg]),
                'paired_frames':len(paired),'labels_with_any_object':nonempty,
                'elephant_gt_frames':gt_ele_frames,'multi_elephant_gt_frames':gt_multi_frames,
                'class_counts':dict(hist), 'camera_records':len(cams.get('cameras',[])) if isinstance(cams,dict) and isinstance(cams.get('cameras'),list) else None,
                'has_cameras_json':'cameras' in meta[seg],
                'has_tracking_summary':'tracking_summary' in meta[seg],
                'tracks':len(all_tracks),'elephant_tracks':len(elephant_tracks),
                'ambiguous_object_tracks':len(ambiguous_tracks),
                'longest_elephant_run':max((x['longest_consecutive'] for x in elephant_tracks),default=0),
                'potential_2s_elephant_windows':sum(x['possible_2s_windows'] for x in elephant_tracks),
                'potential_5s_elephant_windows':sum(x['possible_5s_windows'] for x in elephant_tracks),
                'potential_2s_ambiguous_windows':sum(x['possible_2s_windows'] for x in ambiguous_tracks),
                'potential_5s_ambiguous_windows':sum(x['possible_5s_windows'] for x in ambiguous_tracks),
                'track_details':all_tracks,
            })
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--check-crc',action='store_true',help='Fully decompress ZIP archives to check member CRC; considerably slower')
    ap.add_argument('--root',default='/mnt/e/AIUB/Capstone Project/Journal Based Results')
    ap.add_argument('--out',default=str(Path.home()/'CP10_Stage5_Candidate_Audit'))
    args=ap.parse_args();root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    details=[]; manifest=[]
    for group,video,prior in CANDIDATES:
        path=resolve_path(root,group,video)
        rec={'group':group,'video':video,'prior_evidence_status':prior,'source_zip':str(path) if path else None,
             'source_bytes':path.stat().st_size if path else None, 'status':'NOT_FOUND' if not path else 'FOUND'}
        if path:
            try:
                data=analyze(path,group,video,check_crc=args.check_crc)
                rec['status']='INSPECTED';rec['segments']=len(data)
                details+=data
            except Exception as exc:
                rec['status']='FAILED';rec['error']=str(exc)
        manifest.append(rec);print(f'{video}: {rec["status"]}',flush=True)
    obj={'generated_utc':datetime.now(timezone.utc).isoformat(), 'root':str(root), 'purpose':'candidate archive feasibility only, not forecasting inference or validation',
         'previous_EXP3_videos':EXISTING_EXP3,'previous_R10_fresh_videos':R10_REUSED,
         'candidate_archives':manifest,'segments':details,
         'limitations':[
           'All three candidate archives are from the already-inspected elep1/elep2/elep3 recording groups.',
           '0006_V was used for qualitative annotation and appears in previous source inventories.',
           'No candidate should be called a locked independent holdout without checking all earlier analyses and ensuring non-overlap.',
           'An object-class KITTI label is not independently established as elephant.',
           'Potential window counts assume 10-Hz contiguous track indices; they are not actual tested forecast samples.',
           'ZIP files were inspected without modifying the source tree; no new model was run.',
         ]}
    (out/'CP10_Stage5_Candidate_Audit.json').write_text(json.dumps(obj,indent=2),encoding='utf8')
    fields=['group','video','segment','frames','labels','paired_frames','elephant_gt_frames','multi_elephant_gt_frames','class_counts','camera_records','has_cameras_json','has_tracking_summary','tracks','elephant_tracks','ambiguous_object_tracks','longest_elephant_run','potential_2s_elephant_windows','potential_5s_elephant_windows','potential_2s_ambiguous_windows','potential_5s_ambiguous_windows']
    with (out/'CP10_Stage5_Segment_Summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for d in details:
            w.writerow({k:json.dumps(d[k],sort_keys=True) if isinstance(d.get(k),dict) else d.get(k) for k in fields})
    lines=['STAGE 5 — Candidate archive feasibility audit','',
           'Existing: six EXP3 videos; additionally three R10 fresh videos have already been evaluated.',
           'The remaining 3 candidates are from the same three recording groups. Avoid claims of new-group external validation.','']
    for rec in manifest:
        lines+= [f"{rec['group']} / {rec['video']}",f"  ZIP: {rec['status']} — {rec['source_zip']}",f"  Prior usage: {rec['prior_evidence_status']}"]
        for d in details:
            if d['video']==rec['video']:
                lines += [f"  {d['segment']}: {d['paired_frames']} frame-label pairs; GT elephant frames={d['elephant_gt_frames']}; GT multi-elephant frames={d['multi_elephant_gt_frames']}; classes={d['class_counts']}; elephant track potential windows 2s/5s={d['potential_2s_elephant_windows']}/{d['potential_5s_elephant_windows']}; class='object' ambiguous windows={d['potential_5s_ambiguous_windows']}"]
        lines.append('')
    lines += ['RESEARCH DECISION:',
              '1. Inspect provenance, sample overlap, and the full evaluation history of any proposed adaptation/holdout data.',
              '2. Treat results on previously inspected recording groups as retrospective within-group transfer, not unseen-group validation.',
              '3. Only use labeled elephant tracks with sufficient contiguous history and future for calibration.',
              '4. Run frozen-model inference before fitting any calibration threshold; retain original TrainReal threshold as baseline.',
              '5. For a genuinely independent confirmatory test, acquire and lock video(s) from additional recording groups before evaluation.']
    (out/'CP10_Stage5_Research_Decision.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Written:',out,flush=True)
    for line in lines[:12]:print(line)

if __name__=='__main__':main()
