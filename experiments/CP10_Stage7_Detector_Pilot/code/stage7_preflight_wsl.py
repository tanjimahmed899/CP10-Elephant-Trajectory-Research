#!/usr/bin/env python3
"""Read-only TrainReal development-set inventory, locked R13 comparator checkpoint.
This does NOT access TestReal images except counting the known root for provenance.
Dependencies: Python standard library only.
"""
import argparse,csv,hashlib,json,random,re,sys
from pathlib import Path
from collections import defaultdict,Counter

LOCKED_SHA='1d2eb19d22da78e82a92fcfb3a44ab87b78c5690d7d9caef6e0f55fba26d1055'
IMAGE_EXT={'.jpg','.jpeg','.png','.bmp','.tiff','.tif'}

def nframe(s):
    nums=re.findall(r'(\d+)',Path(s).stem)
    return int(nums[-1]) if nums else None

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2*1024*1024),b''):h.update(b)
    return h.hexdigest()

def split_name(row):
    return str(row.get('split','')).lower().replace('-','_').strip()

def check_ckpt(root,explicit):
    cap=root.parent
    rel=Path('Final 1.0')/'Elephant Detection.v2-resized640.yolov8 (2)'/'Elephant Detection.v2-resized640.yolov8'/'runs'/'detect'/'elephant_final_clean'/'weights'/'best.pt'
    opts=[Path(explicit)] if explicit else [cap/rel]
    for p in opts:
        if p.is_file():
            v=sha(p)
            return {'path':str(p),'bytes':p.stat().st_size,'sha256':v,'original_r13_checkpoint_match':v==LOCKED_SHA}
    return {'path':str(opts[0]),'original_r13_checkpoint_match':False,'error':'checkpoint missing'}

def find_train_roots(root):
    # Most likely layouts first; avoid scanning full E drive.
    mr=root/'Main Result'
    candidates=[
        mr/'conservation_drones_train_real'/'TrainReal'/'images',
        mr/'conservation_drones_train_real'/'TrainReal',
        mr/'conservation_drones_train_real'/'images',
        mr/'conservation_drones'/'TrainReal'/'images',
        mr/'conservation_drones'/'TrainReal',
        mr/'conservation_drones_train_real',
        mr/'TrainReal'/'images',
        mr/'TrainReal',
    ]
    if mr.exists():
        # Pruned discovery at depth <= 4; never walk into image directories.
        frontier=[mr]
        for depth in range(4):
            following=[]
            for p in frontier:
                try:ds=[q for q in p.iterdir() if q.is_dir()]
                except OSError:continue
                for q in ds:
                    low=str(q).lower()
                    if ('trainreal' in low or 'train_real' in low or 'conservation_drones' in low) and q not in candidates:
                        candidates.append(q)
                    if q.name.lower()=='images' and ('trainreal' in low or 'train_real' in low):
                        candidates.append(q)
                    if q.name.lower()!='images' and len(following)<800 and (depth==0 or ('train' in low or 'conservation' in low)):
                        following.append(q)
            frontier=following
    return [p for p in dict.fromkeys(candidates) if p.is_dir()]

def locate_images(candidates,seq):
    for b in candidates:
        for p in [b/seq,b/'images'/seq,b/'TrainReal'/'images'/seq,b/'TrainReal'/seq]:
            if p.is_dir():
                try:items=[q for q in p.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_EXT]
                except OSError:continue
                if items:return p,items
    return None,[]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--out',default=str(Path.home()/'CP10_Stage7_Dev_Pilot'))
    ap.add_argument('--checkpoint',default='')
    ap.add_argument('--max-sequences',type=int,default=6)
    ap.add_argument('--frames-per-sequence',type=int,default=24)
    args=ap.parse_args()
    root=Path(args.root).expanduser().resolve();out=Path(args.out).expanduser();out.mkdir(parents=True,exist_ok=True)
    gt_file=root/'Main Result'/'trajectory_dataset'/'cleaned_track_points.csv'
    test_root=root/'Main Result'/'conservation_drones_test_real'/'TestReal'/'images'
    cp=check_ckpt(root,args.checkpoint)
    gt=defaultdict(lambda:defaultdict(list));splits=defaultdict(set);sizes=Counter()
    if gt_file.is_file():
        with gt_file.open(newline='',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                sp=split_name(r);seq=str(r.get('sequence','')).strip();fr=nframe(r.get('frame',''))
                if not seq or fr is None:continue
                splits[seq].add(sp);sizes[sp]+=1
                if sp!='train_real':continue
                if str(r.get('interpolated','0')).strip().lower() in {'1','1.0','true'}:continue
                # R13's cleaned BIRDSAI GT has species 1.0 = elephant (class 0.0).
                # Never silently relabel another encoded species as elephant.
                if str(r.get('species','')).strip() not in {'','1','1.0','Elephant','elephant'}:continue
                try: x,y,w,h=[float(r[k]) for k in ['x','y','w','h']]
                except (ValueError,KeyError,TypeError):continue
                if w>0 and h>0:gt[seq][fr].append((x,y,x+w,y+h,str(r.get('object_id',''))))
    candidates=find_train_roots(root)
    seqs=sorted(gt, key=lambda s:(-sum(len(v) for v in gt[s].values()),s))
    random.Random(20261121).shuffle(seqs)  # fixed dev selection, does not examine TestReal scores
    samples=[];boxes=[];availability=[];num_selected=0
    for seq in seqs:
        if num_selected>=max(1,args.max_sequences):break
        dirp,items=locate_images(candidates,seq)
        imgs={nframe(p.name):p for p in items if nframe(p.name) is not None}
        valid=sorted(set(imgs)&set(gt[seq]))
        availability.append({'sequence':seq,'raw_gt_frames':len(gt[seq]),'image_dir':str(dirp or ''),'image_frames':len(imgs),'paired_annotated_frames':len(valid)})
        if not valid:continue
        num_selected+=1
        n=min(args.frames_per_sequence,len(valid))
        # spread uniformly through sequence; at most one image per index.
        indices=sorted(set(round(i*(len(valid)-1)/(n-1)) for i in range(n))) if n>1 else [len(valid)//2]
        for j in indices:
            frame=valid[j];path=imgs[frame]
            samples.append({'sequence':seq,'frame':frame,'image_path':str(path),'gt_count':len(gt[seq][frame]),'split':'train_real'})
            for k,(x1,y1,x2,y2,obj) in enumerate(gt[seq][frame]):
                boxes.append({'sequence':seq,'frame':frame,'gt_index':k,'object_id':obj,'x1':x1,'y1':y1,'x2':x2,'y2':y2})
    def write_csv(name,rows,cols):
        with (out/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,cols);w.writeheader();w.writerows(rows)
    write_csv('Stage7_Dev_Image_Manifest.csv',samples,['sequence','frame','image_path','gt_count','split'])
    write_csv('Stage7_Dev_GT_Boxes.csv',boxes,['sequence','frame','gt_index','object_id','x1','y1','x2','y2'])
    write_csv('Stage7_TrainReal_Availability.csv',availability,['sequence','raw_gt_frames','image_dir','image_frames','paired_annotated_frames'])
    # Explicitly fail closed rather than use any TestReal evaluation images for tuning.
    status=('READY_TRAINREAL_DEV_ONLY' if cp.get('original_r13_checkpoint_match') and len(samples)>=20 and len({r['sequence'] for r in samples})>=2 else 'BLOCKED_REQUIRE_TRAINREAL_FRAMES_OR_LOCKED_CHECKPOINT')
    report={'status':status,'root':str(root),'gt_csv':str(gt_file),'gt_exists':gt_file.is_file(),
            'checkpoint':cp,'testreal_root_for_provenance_ONLY':str(test_root),'testreal_exists':test_root.is_dir(),
            'trainreal_candidate_roots':[str(x) for x in candidates],
            'source_split_counts':dict(sizes),'GT_trainreal_sequence_count':len(gt),'image_availability':availability,
            'dev_sequences':sorted(set(x['sequence'] for x in samples)),'dev_frames':len(samples),
            'dev_GT_boxes':len(boxes),'max_sequences':args.max_sequences,'frames_per_sequence':args.frames_per_sequence,
            'selection_seed':20261121,'no_testreal_tuning':True,
            'original_files_modified':False}
    (out/'Stage7_Preflight_Manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    t=['STAGE 7 — Frozen detector development-only preflight',f'Status: {status}',
       f'Original R13 checkpoint match: {cp.get("original_r13_checkpoint_match")}',
       f'GT raw TrainReal sequences: {len(gt)}; matching TrainReal frames: {len(samples)}; GT boxes: {len(boxes)}',
       f'TestReal images located: {test_root.is_dir()} (PROVENANCE ONLY; not read for tuning)',
       'TrainReal candidates:']+[f'  {p}' for p in candidates]+['','Matched TrainReal sequences:']+[f'  {x["sequence"]}: {x["paired_annotated_frames"]} matched GT frames' for x in availability]+[
       '', 'If blocked: obtain/locate actual TrainReal image sequence directories and rerun preflight.',
       'Do not substitute R13 TestReal for development/tuning. Pilot makes no prediction yet.']
    (out/'Stage7_Readout.txt').write_text('\n'.join(t)+'\n',encoding='utf8')
    print('\n'.join(t))
    print('Output:',out)
    return 0

if __name__=='__main__':sys.exit(main())
