#!/usr/bin/env python3
"""One fixed TrainReal-only fine-tuning pilot, and original-vs-adapted validation.
No TestReal data are accessed. Hyperparameters predetermined, do not tune after TestReal.
"""
import argparse, hashlib, json, os, platform, sys
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def normal(m):
 # Ultralytics DetMetrics box maps: P,R,mAP50,mAP50-95
 r=m.box
 return {'precision':float(r.mp),'recall':float(r.mr),'mAP50':float(r.map50),'mAP50_95':float(r.map)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--device',default='cpu');a=ap.parse_args()
 out=Path(a.out).expanduser().resolve(); manifest=out/'Stage7B_Frozen_Preflight.json'
 if not manifest.exists():raise SystemExit('Run stage7b_prepare.py first')
 m=json.loads(manifest.read_text())
 assert m['status']=='READY_FOR_TRAINREAL_DOMAIN_ADAPTATION' and m['trainreal_only'] is True and m['testreal_accessed'] is False
 import torch,ultralytics
 from ultralytics import YOLO
 orig=Path(m['original_checkpoint']);yaml=m['data_yaml']
 assert sha(orig)==m['original_checkpoint_sha256'],'Original checkpoint changed, abort'
 runs=out/'training_runs'
 if (runs/'domain_adapt'/'weights'/'best.pt').exists():
  raise SystemExit('Trained weights already exist. Do not overwrite evidence; inspect outputs instead.')
 params={'epochs':8,'imgsz':640,'batch':8,'patience':8,'seed':20261121,'deterministic':True,
         'optimizer':'AdamW','lr0':0.0001,'lrf':0.1,'weight_decay':0.0005,
         'workers':0,'device':a.device,'amp':False,'project':str(runs),'name':'domain_adapt',
         'exist_ok':False,'plots':False,'verbose':True,'save':True,'cache':False,'pretrained':True,
         'close_mosaic':0,'mosaic':0.0,'mixup':0.0,'copy_paste':0.0,'fliplr':0.0,'flipud':0.0}
 (out/'Stage7B_Training_Protocol_LOCKED.json').write_text(json.dumps({'params':params,'baseline_checkpoint_sha256':m['original_checkpoint_sha256'],
         'split':{'train':m['train_sequences'],'val':m['validation_sequences']},
         'selection':'Ultralytics default validation-fitness best.pt from exactly eight epochs; no TestReal','comparison_conf':0.10,
         'eval_scope':'TrainReal validation sequences only; TestReal not touched'},indent=2))
 baseline=YOLO(str(orig))
 b=baseline.val(data=yaml,split='val',imgsz=640,conf=0.001,iou=0.7,device=a.device,plots=False,
               project=str(runs),name='baseline_val',exist_ok=False,verbose=False)
 orig_metrics=normal(b)
 del baseline
 model=YOLO(str(orig));model.train(data=yaml,**params)
 best=runs/'domain_adapt'/'weights'/'best.pt'
 if not best.is_file():raise SystemExit('Fine-tuning failed to create best.pt')
 adapted=YOLO(str(best))
 r=adapted.val(data=yaml,split='val',imgsz=640,conf=0.001,iou=0.7,device=a.device,plots=False,
               project=str(runs),name='adapted_val',exist_ok=False,verbose=False)
 improved=normal(r)
 report={'status':'TRAINREAL_VALIDATION_ONLY_COMPLETE','testreal_accessed':False,
         'baseline_checkpoint':str(orig),'baseline_sha256':sha(orig),
         'adapted_checkpoint':str(best),'adapted_sha256':sha(best),
         'validation_model_metrics':{'original':orig_metrics,'adapted':improved},
         'torch':torch.__version__,'ultralytics':ultralytics.__version__,
         'note':'Validation was repeatedly used during training for best.pt selection. Not independent holdout performance. No tracker results.'}
 (out/'Stage7B_TrainReal_Validation_Readout.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
