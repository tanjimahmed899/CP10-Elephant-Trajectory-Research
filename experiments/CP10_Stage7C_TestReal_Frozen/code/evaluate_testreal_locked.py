#!/usr/bin/env python3
"""Stage 7C: frozen, retrospective R13 TestReal detector comparison.
No train/tune/threshold sweep. Same conf=0.10, imgsz=640 for original/adapted.
TestReal previously analyzed in R13, NOT independent or prospective.
"""
import argparse, collections, csv, hashlib, json, math, os, re, sys, time
from pathlib import Path

SEQS=(
 '0000000055_0000000000', '0000000058_0000000000',
 '0000000349_0000000000', '0000000352_0000000000',
 '0000000353_0000000000', '0000000357_0000000000')
FRAME_COUNTS=(268,684,1456,3209,3193,275)
GT_COUNTS=(264,2666,3292,19343,21467,1023)
ORIGINAL_SHA='1d2eb19d22da78e82a92fcfb3a44ab87b78c5690d7d9caef6e0f55fba26d1055'
ADAPTED_SHA='71671b4175509e96bb6e4cd7e803b82c9c4b606061438a009d98d9f8b8e8d3d1'
R13_BASELINE={'GT':48055,'pred_boxes':4288,'TP':2431,'FP':1857,'FN':45624,'recall':2431/48055}


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def to_num(s):
 try:return float(s)
 except (ValueError,TypeError):return float('nan')

def parse_frame(x):
 t=str(x).rsplit('/',1)[-1].rsplit('\\',1)[-1]
 m=re.findall(r'\d+',Path(t).stem)
 return int(m[-1]) if m else None

def read_gt(path):
 gt=collections.defaultdict(list);excluded=collections.Counter()
 with open(path,encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   seq=str(r.get('sequence','')).strip()
   if seq not in SEQS:continue
   if r.get('split','').strip().lower().replace('-','_')!='test_real': excluded['not_testreal']+=1;continue
   if r.get('species','').strip().lower() not in ('1','1.0','elephant'):excluded['not_elephant']+=1;continue
   if str(r.get('interpolated','0')).strip().lower() in ('1','1.0','true'):excluded['interpolated']+=1;continue
   try:
    n=parse_frame(r['frame']);x,y,w,h=[float(r[k]) for k in ('x','y','w','h')]
    if n is None or not all(map(math.isfinite,[x,y,w,h])) or w<=0 or h<=0:raise ValueError()
   except (KeyError,ValueError,TypeError):excluded['invalid_box']+=1;continue
   gt[(seq,n)].append((x,y,x+w,y+h))
 return gt,dict(excluded)

def bb_iou(a,b):
 ix=max(0.,min(a[2],b[2])-max(a[0],b[0]));iy=max(0.,min(a[3],b[3])-max(a[1],b[1]));inters=ix*iy
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);ba=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inters/max(1.e-12,aa+ba-inters)

def match_boxes(gt,pred,threshold=.5):
 candidates=[]
 for i,a in enumerate(gt):
  for j,b in enumerate(pred):
   v=bb_iou(a,b)
   if v>=threshold:candidates.append((v,i,j))
 usedg=set();usedp=set()
 for v,i,j in sorted(candidates,reverse=True):
  if i not in usedg and j not in usedp:usedg.add(i);usedp.add(j)
 return usedg,usedp

def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')

def writer(file,header):
 f=open(file,'w',newline='',encoding='utf8');w=csv.DictWriter(f,fieldnames=header);w.writeheader();return f,w

def rates(t,f,n):
 p=t/(t+f) if t+f else 0.;r=t/(t+n) if t+n else 0.;f1=2*p*r/(p+r) if p+r else 0.
 return p,r,f1

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--device',default='cpu')
 args=ap.parse_args();root=Path(args.root).expanduser().resolve();out=Path(args.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
 if out==root or root in out.parents:raise SystemExit('Must write outside source research tree')
 if (out/'Stage7C_Overall_IoU50.csv').exists():raise SystemExit('Already evaluated: refusing to overwrite frozen results')
 base_ck=root.parent/'Final 1.0'/'Elephant Detection.v2-resized640.yolov8 (2)'/'Elephant Detection.v2-resized640.yolov8'/'runs'/'detect'/'elephant_final_clean'/'weights'/'best.pt'
 adapted_ck=Path(__file__).resolve().parent/'best_Stage7B_trainreal.pt'
 for model,path,digest in [('original',base_ck,ORIGINAL_SHA),('adapted',adapted_ck,ADAPTED_SHA)]:
  if not path.is_file():raise SystemExit('Missing '+model+' checkpoint: '+str(path))
  if sha(path)!=digest:raise SystemExit('BLOCK: '+model+' checkpoint SHA mismatch: '+str(path))
 img_base=root/'Main Result'/'conservation_drones_test_real'/'TestReal'/'images'
 raw_gt=root/'Main Result'/'trajectory_dataset'/'cleaned_track_points.csv'
 if not raw_gt.is_file():raise SystemExit('Missing GT '+str(raw_gt))
 gt,excluded=read_gt(raw_gt)
 seq_files={};observed={}
 for seq,fr_expected,gt_expected in zip(SEQS,FRAME_COUNTS,GT_COUNTS):
  d=img_base/seq
  if not d.is_dir():raise SystemExit('Missing TestReal sequence '+str(d))
  files={}
  for p in d.iterdir():
   if p.suffix.lower() in ('.jpg','.jpeg','.png'):
    f=parse_frame(p.name)
    if f is None or f in files:raise SystemExit('Nonunique frame id in '+str(d))
    files[f]=p
  count_gt=sum(len(v) for (s,n),v in gt.items() if s==seq)
  observed[seq]={'images':len(files),'GT':count_gt,'expected_images':fr_expected,'expected_GT':gt_expected}
  if len(files)!=fr_expected or count_gt!=gt_expected:
   save_json(out/'Stage7C_Blocked_Preflight.json',{'status':'BLOCK_GT_OR_IMAGE_COUNT_MISMATCH','by_sequence':observed,'excluded':excluded})
   raise SystemExit(f'BLOCK: TestReal sample mismatch in {seq}: images {len(files)}/{fr_expected}, GT {count_gt}/{gt_expected}; upload Blocked_Preflight.json')
  if any((s==seq and n not in files) for (s,n) in gt):raise SystemExit('GT frame missing image for '+seq)
  seq_files[seq]=sorted(files.items())
 proof={'status':'TESTREAL_MATCHED_R13_PROTOCOL','source_root':str(root),'original_sha256':ORIGINAL_SHA,'adapted_sha256':ADAPTED_SHA,
        'source_GT':str(raw_gt),'sequences':observed,'GT_total':sum(q['GT'] for q in observed.values()),'images_total':sum(q['images'] for q in observed.values()),
        'excluded_GT_rows':excluded,'comparison':{'imgsz':640,'conf':0.10,'iou_nms':0.7,'max_det':300,'augment':False,'agnostic_nms':False},
        'test_is_retrospective':True,'training_or_test_tuning':False,'tracker_run':False,'experiment_note':'IoU50 greedy one-to-one detected bounding boxes; counts may differ from historical R13 due to library/environment revisions'}
 save_json(out/'Stage7C_Preflight.json',proof)
 print('TESTREAL INPUTS VALID:',proof['images_total'],'images;',proof['GT_total'],'elephant boxes',flush=True)
 from ultralytics import YOLO
 import torch,ultralytics
 per_cols=['model','sequence','frame','GT','TP','FP','FN','pred_boxes','small_GT','small_TP','wall_sec','image_width','image_height']
 det_cols=['model','sequence','frame','conf','x1','y1','x2','y2']
 fimg,wimg=writer(out/'Stage7C_PerImage_IoU50.csv',per_cols)
 fdet,wdet=writer(out/'Stage7C_Detections.csv',det_cols)
 done=[];summary_rows=[];seq_rows=[]
 try:
  for model_name,weights in [('original',base_ck),('adapted',adapted_ck)]:
   net=YOLO(str(weights)); print('START',model_name,'class names:',net.names,flush=True)
   if len(net.names)!=1:raise SystemExit('Checkpoint does not have exactly one detector class: '+str(net.names))
   tot=collections.Counter()
   for seq in SEQS:
    ctr=collections.Counter();files=seq_files[seq]; t0=time.perf_counter()
    for k,(fr,path) in enumerate(files,1):
     start=time.perf_counter()
     pred=net.predict(source=str(path),device=args.device,imgsz=640,conf=.10,iou=.70,max_det=300,augment=False,agnostic_nms=False,verbose=False)[0]
     runtime=time.perf_counter()-start
     H,W=pred.orig_shape
     b=pred.boxes
     arr=b.xyxy.cpu().numpy() if len(b)>0 else []
     conf=b.conf.cpu().numpy() if len(b)>0 else []
     classes=b.cls.cpu().numpy() if len(b)>0 else []
     if any(int(c)!=0 for c in classes):raise SystemExit('Non-elephant model class found in '+model_name)
     boxes=[tuple(map(float,a)) for a in arr]
     g=gt.get((seq,fr),[]);mg,mp=match_boxes(g,boxes)
     small={i for i,bb in enumerate(g) if (bb[2]-bb[0])*(bb[3]-bb[1])/(W*H)<.005}
     row={'model':model_name,'sequence':seq,'frame':fr,'GT':len(g),'TP':len(mg),'FP':len(boxes)-len(mp),'FN':len(g)-len(mg),'pred_boxes':len(boxes),'small_GT':len(small),'small_TP':len(small&mg),'wall_sec':round(runtime,6),'image_width':W,'image_height':H}
     wimg.writerow(row)
     for bb,score in zip(boxes,conf):
      wdet.writerow({'model':model_name,'sequence':seq,'frame':fr,'conf':round(float(score),7),'x1':bb[0],'y1':bb[1],'x2':bb[2],'y2':bb[3]})
     for name in ['GT','TP','FP','FN','pred_boxes','small_GT','small_TP']:
      ctr[name]+=row[name];tot[name]+=row[name]
     ctr['wall_sec']+=runtime;tot['wall_sec']+=runtime
     if k%200==0 or k==len(files):print(f'  {model_name} {seq}: {k}/{len(files)}; detected={ctr["pred_boxes"]}; matched={ctr["TP"]}',flush=True)
    p,r,f1=rates(ctr['TP'],ctr['FP'],ctr['FN'])
    seq_rows.append({'model':model_name,'sequence':seq,'frames':len(files),**{name:ctr[name] for name in ['GT','TP','FP','FN','pred_boxes','small_GT','small_TP']},'precision':p,'recall':r,'F1':f1,'false_boxes_per_frame':ctr['FP']/len(files),'mean_wall_seconds':ctr['wall_sec']/len(files)})
    print('DONE',model_name,seq,'recall',round(r,5),'precision',round(p,5),'seconds',round(time.perf_counter()-t0,3),flush=True)
    fimg.flush();fdet.flush()
   p,r,f1=rates(tot['TP'],tot['FP'],tot['FN']);nfiles=sum(len(v) for v in seq_files.values())
   summary_rows.append({'model':model_name,'frames':nfiles,**{name:tot[name] for name in ['GT','TP','FP','FN','pred_boxes','small_GT','small_TP']},'precision':p,'recall':r,'F1':f1,'false_boxes_per_frame':tot['FP']/nfiles,'mean_wall_seconds':tot['wall_sec']/nfiles})
   del net
 finally:fimg.close();fdet.close()
 cols=list(summary_rows[0]) if summary_rows else []
 if len(summary_rows)!=2:raise SystemExit('Model evaluation incomplete: expected original and adapted')
 for fname,rs in [('Stage7C_PerSequence_IoU50.csv',seq_rows),('Stage7C_Overall_IoU50.csv',summary_rows)]:
  with open(out/fname,'w',newline='',encoding='utf8') as f:
   wr=csv.DictWriter(f,fieldnames=list(rs[0]));wr.writeheader();wr.writerows(rs)
 observed_base=summary_rows[0]
 gate={'historical_R13_baseline':R13_BASELINE,'replayed_original':{k:observed_base[k] for k in ['GT','pred_boxes','TP','FP','FN','recall','precision']},
       'exact_count_parity':all(observed_base[k]==R13_BASELINE[k] for k in ['GT','pred_boxes','TP','FP','FN']),
       'relative_pred_boxes_difference':(observed_base['pred_boxes']-R13_BASELINE['pred_boxes'])/R13_BASELINE['pred_boxes'],
       'relative_TP_difference':(observed_base['TP']-R13_BASELINE['TP'])/R13_BASELINE['TP'],
       'note':'No retuning permitted. Any discrepancy must be explained (e.g. Ultralytics version, image inputs, GT evaluator) before calling this an exact R13 replication.'}
 save_json(out/'Stage7C_R13_Baseline_Parity.json',gate)
 save_json(out/'Stage7C_Run_Provenance.json',{'status':'TWO_CHECKPOINTS_COMPLETE','torch':torch.__version__,'ultralytics':ultralytics.__version__,
   'model_order':['original','adapted'],'fixed_settings':proof['comparison'],'historical_baseline_exact_match':gate['exact_count_parity'],
   'adapted_checkpoint_hash':ADAPTED_SHA,'original_checkpoint_hash':ORIGINAL_SHA,'total_images':sum(len(v) for v in seq_files.values()),'no_testreal_tuning':True})
 print('\nFINAL RESULTS:')
 for r in summary_rows:print(r)
 print('Historical R13 exact count parity:',gate['exact_count_parity']);print('Save results at',out)

if __name__=='__main__':main()
