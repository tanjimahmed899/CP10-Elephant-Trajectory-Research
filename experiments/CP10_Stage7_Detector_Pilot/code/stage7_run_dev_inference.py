#!/usr/bin/env python3
"""Stage7 frozen detector pilot on preselected TrainReal frames ONLY.
No training, no TestReal metrics, no test-set configuration tuning.
Requires ultralytics + PyTorch + numpy, optional tiled mode requires cv2.
"""
import argparse,csv,hashlib,json,time,sys
from pathlib import Path

CONFIGS={
 'baseline_640_conf010':{'imgsz':640,'conf':0.10,'tile':False},
 'highres_960_conf010':{'imgsz':960,'conf':0.10,'tile':False},
 'lowconf_640_conf003':{'imgsz':640,'conf':0.03,'tile':False},
 'highres_960_conf003':{'imgsz':960,'conf':0.03,'tile':False},
 'tiled_640_conf010':{'imgsz':640,'conf':0.10,'tile':True},
}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for buf in iter(lambda:f.read(2*1024*1024),b''):h.update(buf)
 return h.hexdigest()

def iou(a,b):
 ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=ix*iy
 if inter<=0:return 0.
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
 return inter/max(1e-12,aa+bb-inter)

def nms(dets,th=.70):
 """Single-class NMS, keeps higher confidence, de-duplicates cross-tile boxes."""
 out=[]
 for d in sorted(dets,key=lambda x:-x['confidence']):
  rect=[d[k] for k in ('x1','y1','x2','y2')]
  if all(iou(rect,[o[k] for k in ('x1','y1','x2','y2')])<=th for o in out):out.append(d)
 return out[:300]

def im_boxes(res,img_w,img_h,offx=0,offy=0):
 out=[]
 if res.boxes is None:return out
 for xy,conf,cl in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.conf.cpu().numpy(),res.boxes.cls.cpu().numpy()):
  name=str(res.names[int(cl)] if isinstance(res.names,dict) else res.names[int(cl)])
  if name.strip().lower()!='elephant':continue
  x1,y1,x2,y2=[float(x) for x in xy]
  out.append({'class_name':name,'confidence':float(conf),
   'x1':max(0.,x1+offx),'y1':max(0.,y1+offy),
   'x2':min(float(img_w),x2+offx),'y2':min(float(img_h),y2+offy)})
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(Path.home()/'CP10_Stage7_Dev_Pilot'))
 ap.add_argument('--configs',nargs='+',default=['baseline_640_conf010','highres_960_conf010','lowconf_640_conf003','highres_960_conf003'])
 ap.add_argument('--device',default='cpu');ap.add_argument('--limit',type=int,default=0)
 args=ap.parse_args();out=Path(args.out).expanduser().resolve()
 pre=json.loads((out/'Stage7_Preflight_Manifest.json').read_text())
 if pre['status']!='READY_TRAINREAL_DEV_ONLY':raise SystemExit('BLOCKED: Stage7 preflight must be READY_TRAINREAL_DEV_ONLY; never tune on TestReal')
 ck=Path(pre['checkpoint']['path'])
 if sha(ck)!=pre['checkpoint']['sha256'] or not pre['checkpoint']['original_r13_checkpoint_match']:
  raise SystemExit('Blocked: original R13 checkpoint hash differs')
 if any(c not in CONFIGS for c in args.configs):raise SystemExit('Unknown config. Allowed: '+str(list(CONFIGS)))
 try:
  import numpy as np
  from ultralytics import YOLO
 except Exception as ex:raise SystemExit('Requires ultralytics+torch+numpy in current WSL venv: '+str(ex))
 rows=list(csv.DictReader((out/'Stage7_Dev_Image_Manifest.csv').open(encoding='utf8',newline='')))
 if args.limit>0:rows=rows[:args.limit]
 if len(rows)<1:raise SystemExit('No matched TrainReal frames')
 for r in rows:
  if r['split']!='train_real' or not Path(r['image_path']).exists() or 'TestReal' in r['image_path'] or 'test_real' in r['image_path'].lower():
   raise SystemExit('Blocked: nonexistent/non-TrainReal dev frame '+r['image_path'])
 model=YOLO(str(ck));out_boxes=out/'Stage7_Dev_Detections.csv';out_frames=out/'Stage7_Dev_Inference_Times.csv'
 f1=out_boxes.open('w',newline='',encoding='utf8');f2=out_frames.open('w',newline='',encoding='utf8')
 bw=csv.DictWriter(f1,fieldnames=['config','sequence','frame','image_path','image_width','image_height','class_name','confidence','x1','y1','x2','y2']);bw.writeheader()
 fw=csv.DictWriter(f2,fieldnames=['config','sequence','frame','image_path','num_elephant_boxes','wall_sec']);fw.writeheader()
 completed=0
 try:
  for cfg in args.configs:
   settings=CONFIGS[cfg]
   for k,r in enumerate(rows,1):
    path=r['image_path'];start=time.perf_counter()
    if settings['tile']:
     try:import cv2
     except ImportError:raise SystemExit('tiled mode needs opencv-python')
     im=cv2.imread(path)
     if im is None:raise RuntimeError('Could not read image '+path)
     h,w=im.shape[:2];cw=int(w*.575);ch=int(h*.575)
     origins=[(0,0),(w-cw,0),(0,h-ch),(w-cw,h-ch)]
     boxes=[]
     for xx,yy in origins:
      patch=im[yy:yy+ch,xx:xx+cw]
      pred=model.predict(source=patch,imgsz=settings['imgsz'],conf=settings['conf'],iou=.7,
                         max_det=300,augment=False,agnostic_nms=False,device=args.device,verbose=False)[0]
      boxes.extend(im_boxes(pred,w,h,xx,yy))
     boxes=nms(boxes)
    else:
     pred=model.predict(source=path,imgsz=settings['imgsz'],conf=settings['conf'],iou=.7,
                        max_det=300,augment=False,agnostic_nms=False,device=args.device,verbose=False)[0]
     h,w=pred.orig_shape
     boxes=im_boxes(pred,w,h)
    spent=time.perf_counter()-start
    for b in boxes:
     bw.writerow({'config':cfg,'sequence':r['sequence'],'frame':r['frame'],'image_path':path,
       'image_width':w,'image_height':h,**b})
    fw.writerow({'config':cfg,'sequence':r['sequence'],'frame':r['frame'],'image_path':path,
      'num_elephant_boxes':len(boxes),'wall_sec':round(spent,6)})
    completed+=1
    if k%10==0 or k==len(rows):print(f'  {cfg}: {k}/{len(rows)} frames; last boxes={len(boxes)}; {spent:.2f}s',flush=True)
   f1.flush();f2.flush()
 finally:f1.close();f2.close()
 meta={'role':'TRAINREAL_DEVELOPMENT_ONLY','checkpoint_sha256':pre['checkpoint']['sha256'],
       'configurations':{c:CONFIGS[c] for c in args.configs},'frames_per_configuration':len(rows),
       'total_frame_config_runs':completed,'testreal_used':False,'model_training':False,'pilot_complete':completed==len(rows)*len(args.configs)}
 (out/'Stage7_Dev_Run_Provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
 print('Development inference complete. Score with stage7_score_dev.py --out',out)

if __name__=='__main__':main()
