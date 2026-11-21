#!/usr/bin/env python3
"""Prepare GT-grounded BIRDSAI TrainReal YOLO labels. TestReal is not read.
Sequence-disjoint train/validation; positive-image-only exploratory dataset.
"""
import argparse, csv, hashlib, json, random, re, shutil
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

TRAIN=('0000000367_0000000000','0000000351_0000000000','0000000056_0000000000')
VAL=('0000000010_0000000000','0000000054_0000000000')
LOCK_SHA='1d2eb19d22da78e82a92fcfb3a44ab87b78c5690d7d9caef6e0f55fba26d1055'
ROOT_CKPT=Path('Final 1.0')/'Elephant Detection.v2-resized640.yolov8 (2)'/'Elephant Detection.v2-resized640.yolov8'/'runs'/'detect'/'elephant_final_clean'/'weights'/'best.pt'
IMG_EXT={'.jpg','.jpeg','.png'}

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def frame_num(p):
 a=re.findall(r'\d+',Path(p).stem);return int(a[-1]) if a else None

def sample_even(items,cap):
 if cap<=0 or len(items)<=cap:return items
 return [items[round(k*(len(items)-1)/(cap-1))] for k in range(cap)] if cap>1 else [items[len(items)//2]]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
 p.add_argument('--max-train-per-sequence',type=int,default=1200);p.add_argument('--max-val-per-sequence',type=int,default=150)
 a=p.parse_args(); root=Path(a.root).expanduser().resolve(); out=Path(a.out).expanduser().resolve()
 gtfile=root/'Main Result'/'trajectory_dataset'/'cleaned_track_points.csv'
 ckpt=root.parent/ROOT_CKPT
 if not gtfile.is_file() or not ckpt.is_file():raise SystemExit('Missing original TrainReal GT CSV or original checkpoint')
 if digest(ckpt)!=LOCK_SHA:raise SystemExit('Original R13 checkpoint SHA-256 mismatch; stop')
 if out == root or root in out.parents:raise SystemExit('Output cannot be inside original research source tree')
 if any(x.exists() for x in [out/'Stage7B_Frozen_Preflight.json',out/'dataset'/'data.yaml']):
  raise SystemExit('Output already prepared. Use a fresh --out path; do not overwrite experimental data.')
 imgsbase=root/'Main Result'/'conservation_drones_train_real'/'TrainReal'/'images'
 if not imgsbase.is_dir():raise SystemExit('TrainReal images root not found: '+str(imgsbase))
 seqs=set(TRAIN+VAL); raw=defaultdict(list); excluded=Counter()
 with gtfile.open('r',encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   seq=str(r.get('sequence','')).strip()
   if seq not in seqs:continue
   if str(r.get('split','')).strip().lower().replace('-','_')!='train_real':
    excluded['non_train_real']+=1;continue
   if str(r.get('species','')).strip() not in ('1','1.0','elephant','Elephant'):
    excluded['not_explicit_elephant']+=1;continue
   if str(r.get('interpolated','0')).strip().lower() in ('1','1.0','true'):
    excluded['interpolated']+=1;continue
   try:
    fr=frame_num(r['frame']);x,y,w,h=(float(r[z]) for z in ('x','y','w','h'))
    if fr is None or w<=0 or h<=0:raise ValueError('invalid box')
   except (KeyError,ValueError,TypeError):excluded['invalid_raw_box']+=1;continue
   raw[(seq,fr)].append((x,y,x+w,y+h))
 if not raw:raise SystemExit('No explicitly labeled elephant TrainReal GT matches')
 selected={}; allfiles=[]
 for seq in TRAIN+VAL:
  dire=imgsbase/seq
  if not dire.is_dir():raise SystemExit('Missing sequence '+str(dire))
  files={frame_num(f):f for f in dire.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT and frame_num(f) is not None}
  valid=sorted(fr for fr in files if (seq,fr) in raw)
  if len(valid)<10:raise SystemExit(f'Insufficient annotated frames for {seq}: {len(valid)}')
  role='train' if seq in TRAIN else 'val'
  chosen=sample_even(valid,a.max_train_per_sequence if role=='train' else a.max_val_per_sequence)
  selected[seq]={'role':role,'available_annotated_images':len(valid),'sampled_images':len(chosen)}
  allfiles.extend((role,seq,fr,files[fr]) for fr in chosen)
 ds=out/'dataset'
 for role in ('train','val'):
  (ds/'images'/role).mkdir(parents=True,exist_ok=True);(ds/'labels'/role).mkdir(parents=True,exist_ok=True)
 manifest=[];drops=Counter(); included=Counter()
 for role,seq,fr,src in allfiles:
  try:
   with Image.open(src) as im:width,height=im.size
  except Exception as e:raise SystemExit('Cannot read source image '+str(src)+': '+str(e))
  if width<16 or height<16:raise SystemExit('Bad image resolution for '+str(src))
  labels=[]
  for bx1,by1,bx2,by2 in raw[(seq,fr)]:
   x1=max(0,min(float(width),bx1));x2=max(0,min(float(width),bx2))
   y1=max(0,min(float(height),by1));y2=max(0,min(float(height),by2))
   if x2<=x1 or y2<=y1:drops['out_of_frame']+=1;continue
   if (x1,y1,x2,y2)!=(bx1,by1,bx2,by2):drops['clipped_to_image']+=1
   cx=(x1+x2)/(2*width);cy=(y1+y2)/(2*height)
   bw=(x2-x1)/width;bh=(y2-y1)/height
   labels.append(f'0 {cx:.9f} {cy:.9f} {bw:.9f} {bh:.9f}')
  if not labels:drops['no_valid_gt_in_frame']+=1;continue
  key=f'{seq}__frame_{fr:010d}'
  link=ds/'images'/role/(key+src.suffix.lower()); txt=ds/'labels'/role/(key+'.txt')
  if link.exists() or link.is_symlink():raise SystemExit('Duplicate output '+str(link))
  link.symlink_to(src)
  txt.write_text('\n'.join(labels)+'\n',encoding='utf8')
  included[(role,seq)]+=1
  manifest.append({'role':role,'sequence':seq,'frame':fr,'image':str(link),'source_image':str(src),
                   'width':width,'height':height,'gt_elephants':len(labels)})
 yaml=ds/'data.yaml';yaml.write_text('path: '+str(ds)+'\ntrain: images/train\nval: images/val\nnames:\n  0: elephant\nnc: 1\n',encoding='utf8')
 for seq in TRAIN+VAL:
  if included[('train' if seq in TRAIN else 'val',seq)]<10:raise SystemExit('Output contains insufficient samples for '+seq)
 columns=list(manifest[0]);
 with (out/'Stage7B_Dataset_Manifest.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(manifest)
 report={'status':'READY_FOR_TRAINREAL_DOMAIN_ADAPTATION','source_gt':str(gtfile),'original_checkpoint':str(ckpt),
  'original_checkpoint_sha256':digest(ckpt),'train_sequences':TRAIN,'validation_sequences':VAL,
  'sequence_overlap':sorted(set(TRAIN)&set(VAL)), 'selected':selected,'included_images_by_sequence':{f'{a}/{b}':v for (a,b),v in included.items()},
  'images_total':len(manifest),'elephant_gt_total':sum(x['gt_elephants'] for x in manifest),'excluded_raw_rows':dict(excluded),
  'box_adjustments':dict(drops),'data_yaml':str(yaml),'trainreal_only':True,'testreal_accessed':False,
  'no_negative_only_images':'This is a positive-frame-only pilot; FP metrics do not estimate true negative-image false alarm rates',
  'note':'This is an exploratory pilot. Both source and target domains were previously analyzed; TestReal evaluation would be retrospective.'}
 (out/'Stage7B_Frozen_Preflight.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print('READY: dataset',yaml, 'images',len(manifest),'elephant boxes',report['elephant_gt_total'])
 print('Sequence-disjoint TrainReal only. No TestReal accessed.')

if __name__=='__main__':main()
