#!/usr/bin/env python3
"""Stage 7D: offline frozen ByteTrack replay of saved Stage 7C detections.
No detector inference, training, or TestReal tuning. See README for protocol caveats.
"""
import argparse, csv, hashlib, io, json, os, sys, time, traceback, zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = Path('/mnt/e/AIUB/Capstone Project/Journal Based Results')
EXPECTED_YAML_HASH = 'eaefad899964d7a12a13090d5da32bcdae8f0109968ffca982fdba2c60ad3c38'
SEQS = ['0000000055_0000000000','0000000058_0000000000','0000000349_0000000000','0000000352_0000000000','0000000353_0000000000','0000000357_0000000000']
ORIGINAL_CKPT = '1d2eb19d22da78e82a92fcfb3a44ab87b78c5690d7d9caef6e0f55fba26d1055'
ADAPTED_CKPT = '71671b4175509e96bb6e4cd7e803b82c9c4b606061438a009d98d9f8b8e8d3d1'

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def jsonout(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,default=str)+'\n')

def find_yaml(override=None):
 paths=[]
 if override:paths.append(Path(override))
 base=ROOT
 rel='R13_end_to_end_pipeline/R13C_frozen_end_to_end_protocol_lock/bytetrack_FROZEN.yaml'
 paths.extend([base/'Main Result 3.0/paper_strengthening_revision'/rel,base/'Main Result 3.0'/rel,
     Path('/mnt/c/Users/HP/Downloads/paper_strengthening_revision')/rel,
     Path.home()/'Downloads/paper_strengthening_revision'/rel,
     Path.home()/'paper_strengthening_revision'/rel,
     base/'paper_strengthening_revision'/rel])
 for p in paths:
  if p.is_file():
   if sha256(p)==EXPECTED_YAML_HASH:return p
   print('SKIP: ByteTrack YAML hash mismatch',p,sha256(p),flush=True)
 return None

def count_fcast(tracks):
 rows=[]
 for (model,sequence,tid),g in tracks.groupby(['model','sequence','track_id'],sort=True):
  fs=np.unique(g.frame.to_numpy(dtype=int)); blocks=np.split(fs,np.where(np.diff(fs)>4)[0]+1)
  for j,arr in enumerate(blocks,1):
   if len(arr)==0:continue
   dense=int(arr[-1]-arr[0]+1)
   n10=(dense+2)//3
   w2=max(0, (n10-40)//10+1) if n10>=40 else 0
   w5=max(0, (n10-70)//10+1) if n10>=70 else 0
   rows.append(dict(model=model,sequence=sequence,track_id=int(tid),segment=j,frames_observed=int(len(arr)),dense_30hz_frames=dense,
         interpolated_30hz_frames=dense-len(arr),n10=n10,windows_2s=int(w2),windows_5s=int(w5),start_frame=int(arr[0]),end_frame=int(arr[-1])))
 return pd.DataFrame(rows,columns=['model','sequence','track_id','segment','frames_observed','dense_30hz_frames','interpolated_30hz_frames','n10','windows_2s','windows_5s','start_frame','end_frame'])

def run(args):
 out=Path(args.out).expanduser();out.mkdir(parents=True,exist_ok=True)
 try:
  from ultralytics import __version__ as ultra_version
  from ultralytics.trackers.byte_tracker import BYTETracker
  from ultralytics.engine.results import Boxes
  import yaml
 except Exception as e:
  raise RuntimeError('Missing Ultralytics/PyTorch runtime in active virtual environment: '+str(e))
 import r13e_metrics_core as ev
 import scipy
 if str(ultra_version)!='8.4.216':
  raise RuntimeError('Expected Ultralytics 8.4.216 to match Stage7C. Got '+str(ultra_version))
 inzip=Path(args.input).expanduser()
 if not inzip.is_file():raise FileNotFoundError('Missing Stage7C results ZIP: '+str(inzip))
 with zipfile.ZipFile(inzip) as z:
  if z.testzip():raise RuntimeError('Corrupted Stage7C ZIP')
  need=['Stage7C_Detections.csv','Stage7C_PerImage_IoU50.csv','Stage7C_Run_Provenance.json','Stage7C_Overall_IoU50.csv']
  for n in need:
   if n not in z.namelist():raise RuntimeError('Missing Stage7C item '+n)
  provenance=json.loads(z.read('Stage7C_Run_Provenance.json'))
  if provenance['original_checkpoint_hash']!=ORIGINAL_CKPT or provenance['adapted_checkpoint_hash']!=ADAPTED_CKPT:
   raise RuntimeError('Unexpected model checkpoints')
  if provenance['total_images'] !=9085 or provenance['status']!='TWO_CHECKPOINTS_COMPLETE':
   raise RuntimeError('Stage7C inference incomplete')
  det=pd.read_csv(io.BytesIO(z.read('Stage7C_Detections.csv')),dtype={'sequence':'str'})
  perim=pd.read_csv(io.BytesIO(z.read('Stage7C_PerImage_IoU50.csv')),dtype={'sequence':'str'})
 for df in (det,perim):
  df['sequence']=df.sequence.astype(str).str.strip()
 if len(perim)!=18170 or sorted(perim.sequence.unique().tolist())!=sorted(SEQS):
  raise RuntimeError(f'Unexpected Stage7C per-frame coverage {len(perim)} rows')
 for (model,seq),g in perim.groupby(['model','sequence']):
  fr=np.sort(g.frame.to_numpy(dtype=int));
  if len(fr)!=len(np.unique(fr)) or not np.array_equal(fr,np.arange(len(fr))):
   raise RuntimeError(f'Incomplete or noncontiguous frame series: {model}/{seq}')
 observed=(det.groupby(['model','sequence','frame']).size().rename('observed').reset_index())
 counts=perim.merge(observed,on=['model','sequence','frame'],how='left')
 counts['observed']=counts['observed'].fillna(0).astype(int)
 bad=counts[counts['observed']!=counts['pred_boxes']]
 if len(bad):raise RuntimeError(f'Detection count mismatch, example: {bad.iloc[0].to_dict()}')
 yamlpath=find_yaml(args.yaml)
 if yamlpath is None:
  msg='Frozen ByteTrack YAML missing or hash mismatch; expected SHA-256 '+EXPECTED_YAML_HASH+\
      '. Supply --yaml PATH to original R13C bytetrack_FROZEN.yaml. No tracking run attempted.'
  jsonout(out/'Stage7D_Registry.json',{'status':'BLOCKED_YAML_NOT_VERIFIED','reason':msg,'candidate_hint':'R13_end_to_end_pipeline/R13C_frozen_end_to_end_protocol_lock/bytetrack_FROZEN.yaml'})
  raise RuntimeError(msg)
 cfg=yaml.safe_load(yamlpath.read_text()); print('Verified R13 ByteTrack YAML:',yamlpath,flush=True)
 print('Configuration:',cfg,flush=True)
 needed={'track_high_thresh','track_low_thresh','new_track_thresh','track_buffer','match_thresh','fuse_score'}
 if not needed.issubset(cfg):raise RuntimeError('Frozen YAML keys missing: '+str(needed-set(cfg)))
 gtpath=ROOT/'Main Result/trajectory_dataset/cleaned_track_points.csv'
 if not gtpath.is_file():raise FileNotFoundError('Expected observed GT file '+str(gtpath))
 rows=[];start=time.monotonic()
 for model in ['original','adapted']:
  for seq in SEQS:
   fr=perim[(perim.model==model)&(perim.sequence==seq)].sort_values('frame')
   ds=det[(det.model==model)&(det.sequence==seq)].sort_values('frame')
   fdict={int(f):d for f,d in ds.groupby('frame',sort=False)}
   tracker=BYTETracker(args=SimpleNamespace(**cfg),frame_rate=30)
   try:tracker.reset_id()
   except AttributeError:pass
   trackrows=0
   for _,r in fr.iterrows():
    f=int(r.frame)
    d=fdict.get(f)
    if d is None: input_boxes=np.empty((0,6),dtype=np.float32)
    else:
     input_boxes=d[['x1','y1','x2','y2','conf']].to_numpy(dtype=np.float32)
     input_boxes=np.column_stack([input_boxes,np.zeros((len(input_boxes),),dtype=np.float32)])
    boxes=Boxes(input_boxes,orig_shape=(int(r.image_height),int(r.image_width))).numpy()
    try: result=tracker.update(boxes,img=None)
    except Exception as e:
     raise RuntimeError(f'ByteTrack.update failed on {model}/{seq}/frame={f}: {e}') from e
    for obj in result:
     if len(obj)<8:raise RuntimeError('Unexpected tracker output shape. Expected xyxy, id, score, cls, det_idx.')
     rows.append({'model':model,'sequence':seq,'frame':f,'track_id':int(round(obj[4])),
       'class_name':'elephant','x1':float(obj[0]),'y1':float(obj[1]),'x2':float(obj[2]),'y2':float(obj[3]),'confidence':float(obj[5]),
       'class_id':int(round(obj[6])),'detection_index':int(round(obj[7]))})
     trackrows+=1
   print(f'DONE {model} {seq}: frames={len(fr)} det={len(ds)} tracked={trackrows}; elapsed_sec={time.monotonic()-start:.1f}',flush=True)
 trackcols=['model','sequence','frame','track_id','class_name','x1','y1','x2','y2','confidence','class_id','detection_index']
 tracks=pd.DataFrame(rows,columns=trackcols)
 tracks.to_csv(out/'Stage7D_ByteTrack_All_Boxes.csv',index=False)
 perim=perim.rename(columns={'model':'model'});frames=perim[['model','sequence','frame']].copy()
 frames['num_boxes']=perim['pred_boxes'];frames['num_assigned_track_ids']=0
 for model in ['original','adapted']:
  t=tracks[tracks.model==model].copy()
  f=frames[frames.model==model].copy()
  a=t.drop(columns=['model']); b=f.drop(columns=['model']); a['sequence']=a.sequence.astype(str); b['sequence']=b.sequence.astype(str)
  a.to_csv(out/f'Stage7D_{model}_Tracked_Boxes.csv',index=False)
  b.to_csv(out/f'Stage7D_{model}_Frames.csv',index=False)
 # Original R13E algorithm ported directly from the supplied original code.
 ev.CLEANED=gtpath
 protocol={'data':{'sequences':SEQS}}
 gt=ev.load_gt(protocol)
 if len(gt)!=48055:raise RuntimeError('GT count differs from Stage7C: '+str(len(gt)))
 preds=[]; aggs=[];alpha_rows=[]
 for model in ['original','adapted']:
  p,a,alpha,audit=ev.evaluate_tracker(model,out/f'Stage7D_{model}_Tracked_Boxes.csv',out/f'Stage7D_{model}_Frames.csv',gt,SEQS)
  preds.append(p);aggs.append(a);alpha_rows.append(alpha)
  print(f'{model} tracker HOTA={a.HOTA.iloc[0]:.6f} IDF1={a.IDF1.iloc[0]:.6f}',flush=True)
 pd.concat(preds,ignore_index=True).to_csv(out/'Stage7D_PerSequence_Tracking.csv',index=False)
 pd.concat(aggs,ignore_index=True).to_csv(out/'Stage7D_Aggregate_Tracking.csv',index=False)
 pd.concat(alpha_rows,ignore_index=True).to_csv(out/'Stage7D_HOTA_Alpha_Curve.csv',index=False)
 fv=count_fcast(tracks)
 fv.to_csv(out/'Stage7D_Forecastable_Segments.csv',index=False)
 sumrows=[]
 for model in ['original','adapted']:
  g=fv[fv.model==model]
  sumrows.append({'model':model,'tracked_box_rows':int((tracks.model==model).sum()),'total_segments':len(g),
     'windows_2s':int(g.windows_2s.sum()),'windows_5s':int(g.windows_5s.sum()),
     'forecastable_segments_2s':int((g.windows_2s>0).sum()),'forecastable_segments_5s':int((g.windows_5s>0).sum()),
     'window_yield_vs_GT_2s':float(g.windows_2s.sum()/1053),'window_yield_vs_GT_5s':float(g.windows_5s.sum()/826)})
 pd.DataFrame(sumrows).to_csv(out/'Stage7D_Forecastability_Summary.csv',index=False)
 prov={'status':'TRACKER_METRICS_COMPLETE','source_stage7c_sha256':sha256(inzip),'tracker_yaml_sha256':sha256(yamlpath),
   'original_checkpoint_sha256':ORIGINAL_CKPT,'adapted_checkpoint_sha256':ADAPTED_CKPT,
   'ultralytics_version':str(ultra_version),'tracker_api':'offline direct BYTETracker.update(Boxes) replay',
   'frozen_tracker_from_historical_R13':True,'tracker_frame_rate':30,'source_frames':9085,'source_GT_rows':len(gt),
   'old_R13_original_tracker_boxes':1863,'historical_R13_expected_IDF1':0.02460034456508674,
   'historical_R13_expected_HOTA':0.029722219051547955,'TestReal_used_to_tune':False,
   'not_exact_original_R13_replication':'Ultralytics version 8.4.216 vs historical 8.3.75; original Stage7C TP differs 22',
   'downstream_forecast_windows_are_construction_feasibility_only':True,
   'elapsed_wall_sec':time.monotonic()-start}
 jsonout(out/'Stage7D_Provenance.json',prov)
 report='Stage 7D — Offline frozen ByteTrack replay of saved Stage7C detections\n'
 report+='NO new detector inference. Same six BIRDSAI TestReal sequences.\n'
 report+='Historical R13 parity NOT guaranteed (Ultralytics 8.3.75 vs current 8.4.216).\n'
 report+='The forecastable-window counts measure trajectory availability, NOT actual forecasting performance.\n\n'
 report+=pd.concat(aggs,ignore_index=True)[['tracker','HOTA','IDF1','IDSW','Frag','MOTA','num_tracker_dets']].to_string(index=False)+'\n\n'
 report+=pd.DataFrame(sumrows).to_string(index=False)+'\n'
 (out/'Stage7D_Readout.txt').write_text(report)
 print(report,flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);ap.add_argument('--yaml');args=ap.parse_args()
 out=Path(args.out).expanduser();out.mkdir(parents=True,exist_ok=True)
 try:run(args)
 except BaseException as e:
  (out/'Stage7D_Failure.txt').write_text(traceback.format_exc())
  print(traceback.format_exc(),file=sys.stderr)
  sys.exit(2)

if __name__=='__main__':main()
