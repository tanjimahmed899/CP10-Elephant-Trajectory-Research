#!/usr/bin/env python3
"""CP 1.0 Phase 7M. Provenance evidence assessment, NOT flight authentication.

Processes locally held WildLive annotations and sparse RGB images without
releasing any original video/images or any wildlife GPS metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

THIS = Path(__file__).resolve().parent.parent
LOCK_PATH = THIS / 'protocol' / 'PHASE7K_ORIGINAL_LOCK_DO_NOT_EDIT.json'
EXPECTED_LOCK_SHA256 = '408022abb8923eb61afdf1eb665db8ba7bb02ce4258b21ae4a3fd89dab4abb5c'

SOURCE_MAP = {
  'DJI_0207': ('WildLive_DJI_0207_boxid.zip','WildLive_DJI0207_priority_frames.zip'),
  'DJI_0204_video2': ('DJI_0204_video2_boxid.zip','DJI_0204_video2_visual_QA_frames.zip'),
  'DJI_0204_video1': ('DJI_0204_video1_boxid.zip','DJI_0204_video1_visual_qa_frames.zip'),
  'DJI_0161_video1': ('DJI_0161_video1_boxid.zip','DJI_0161_video1_visual_qa_frames.zip'),
}

EVAL_MAP = {
  'DJI_0161_video1':'Phase7K_DJI0161_Locked_Evaluation.zip',
  'DJI_0204_video1':'Phase7L_DJI0204_video1_Exploratory_Results.zip',
}

# Every pair here defines sampled images; *no* lack of overlap can certify independence.
IMAGE_PAIRS = (
 ('DJI_0204_video1',200,'DJI_0204_video2',200),
 ('DJI_0204_video1',200,'DJI_0161_video1',200),
 ('DJI_0204_video2',200,'DJI_0161_video1',200),
 ('DJI_0207',389,'DJI_0204_video1',200),
 ('DJI_0207',389,'DJI_0204_video2',200),
 ('DJI_0207',389,'DJI_0161_video1',200),
)

PROVISIONAL_GROUP = {
  'DJI_0204_video1':'SCENE_FAMILY_0204_POTENTIALLY_LINKED',
  'DJI_0204_video2':'SCENE_FAMILY_0204_POTENTIALLY_LINKED',
  'DJI_0207':'UNRESOLVED_DJI0207',
  'DJI_0161_video1':'UNRESOLVED_DJI0161',
}

PROJECT_ROLE = {
  'DJI_0207':'DEVELOPMENT',
  'DJI_0204_video2':'DEVELOPMENT',
  'DJI_0204_video1':'RELATED_SCENE_EXPLORATORY',
  'DJI_0161_video1':'EVALUATED_BUT_FLIGHT_INDEPENDENCE_UNVERIFIED',
}

def sha256(path:Path) -> str:
  h=hashlib.sha256()
  with Path(path).open('rb') as f:
    for buf in iter(lambda:f.read(1024*1024), b''): h.update(buf)
  return h.hexdigest()

def check_lock() -> dict:
  if not LOCK_PATH.is_file() or sha256(LOCK_PATH)!=EXPECTED_LOCK_SHA256:
    raise RuntimeError('Original Phase 7K lock changed or absent. STOP.')
  obj=json.loads(LOCK_PATH.read_text(encoding='utf-8'))
  if obj.get('primary_comparator')!='cv_ols' or obj.get('history_frames')!=20 or obj.get('future_horizons_in_frames')!=[20,50]:
    raise RuntimeError('Unexpected lock protocol content. STOP.')
  return obj

def write_csv(path:Path, rows:list[dict], fields:list[str]):
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
    w.writeheader();w.writerows(rows)

def safe_json_members(z:zipfile.ZipFile):
  frame_re=re.compile(r'(?:^|/)frame_(\d+)\.json$',re.IGNORECASE)
  matches=[]
  for member in z.namelist():
    m=frame_re.search(member)
    if m:matches.append((int(m.group(1)),member))
  return sorted(matches)

def audit_annotations(path:Path, seq:str) -> dict:
  with zipfile.ZipFile(path) as z:
    bad=z.testzip()
    if bad:raise ValueError(f'ZIP CRC failure {path.name}: {bad}')
    members=safe_json_members(z)
    if not members:raise ValueError('No frame_N.json annotations in source')
    frames=[]; label_counts=Counter(); id_to_frames=defaultdict(set)
    score_missing=0; invalid_boxes=0; timestamps=set(); dimensions=set()
    frame_metadata_fields=set()
    for fi, member in members:
      frames.append(fi)
      obj=json.loads(z.read(member))
      frame_metadata_fields.update(obj.keys())
      dimensions.add((obj.get('imageWidth'),obj.get('imageHeight')))
      for k in ('timestamp','datetime','capture_time','flight_id','video_id','fps','frame_time','gps','camera_pose'):
        if k in obj and obj.get(k) not in (None,''):
          timestamps.add(k)
      for s in obj.get('shapes',[]):
        label=str(s.get('label','')).lower().strip()
        label_counts[label]+=1
        if label!='elephant':continue
        ident=s.get('group_id')
        if ident is None:raise ValueError(f'Missing group_id in {member}')
        id_to_frames[str(ident)].add(fi)
        if s.get('score') is None: score_missing+=1
        pts=s.get('points') or []
        if len(pts)<2:invalid_boxes+=1
        else:
          xs=[pt[0] for pt in pts]; ys=[pt[1] for pt in pts]
          if min(xs)>=max(xs) or min(ys)>=max(ys):invalid_boxes+=1
    if len(frames)!=len(set(frames)): raise ValueError('Duplicate frame index in ZIP')
    def windows(n):
      count=0
      for flist in id_to_frames.values():
        ss=sorted(flist)
        for i in range(len(ss)):
          if i+20+n-1<len(ss) and ss[i+20+n-1]-ss[i]==20+n-1:
            count+=1
      return count
    maxend=frames[-1];minstart=frames[0]
    return dict(sequence_id=seq,annotation_zip_name=path.name,annotation_sha256=sha256(path),
      annotated_frames=len(frames),first_frame=minstart,last_frame=maxend,
      missing_frame_indices=(maxend-minstart+1-len(frames)),
      elephant_boxes=label_counts.get('elephant',0),other_species_boxes=sum(v for k,v in label_counts.items() if k!='elephant'),
      elephant_tracks=len(id_to_frames),missing_elephant_scores=score_missing,
      invalid_elephant_geometry=invalid_boxes,
      eligible_h20_windows=windows(20),eligible_h50_windows=windows(50),
      annotation_dimensions=str(sorted(dimensions)),
      frame_top_level_keys=';'.join(sorted(frame_metadata_fields)),
      authoritative_capture_fields_populated=';'.join(sorted(timestamps)) or 'NONE',
      provenance_note='frame index and annotation imagePath are not capture timestamps')

def sample_image_meta(path:Path,seq:str):
  from PIL import Image
  rows=[]
  with zipfile.ZipFile(path) as z:
    bad=z.testzip()
    if bad:raise ValueError(f'Bad visual ZIP member: {bad}')
    for n in sorted(z.namelist()):
      m=re.search(r'frame_(\d+)\.(?:jpg|jpeg)$',n,re.I)
      if not m:continue
      with Image.open(io.BytesIO(z.read(n))) as im:
        exif=im.getexif()
        # Do not export source-value EXIF, especially potential GPS coordinates.
        # Export only whether a capture-identity/time field is present.
        exif_keys=set(exif.keys())
        has_capture_datetime=any(k in exif_keys for k in (306,36867,36868))
        has_gps=34853 in exif_keys
        rows.append(dict(sequence_id=seq,image_zip_name=path.name,frame_index=int(m.group(1)),
          image_width=im.width,image_height=im.height,
          exif_tag_count=len(exif_keys),exif_capture_datetime_present=has_capture_datetime,
          exif_gps_present=has_gps,
          usable_authoritative_flight_id_present=False))
  return rows

def image_bytes(zip_path:Path,index:int)->bytes:
  with zipfile.ZipFile(zip_path) as z:
    selected=[n for n in z.namelist() if n.endswith(f'frame_{index}.jpg')]
    if len(selected)!=1:raise ValueError(f'Expected 1 frame_{index}.jpg in {zip_path}, got {len(selected)}')
    return z.read(selected[0])

def scene_compare(image_a:bytes,image_b:bytes) -> dict:
  import cv2
  import numpy as np
  a=cv2.imdecode(np.frombuffer(image_a,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
  b=cv2.imdecode(np.frombuffer(image_b,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
  if a is None or b is None:raise ValueError('Failed to decode image')
  a=cv2.resize(a,(1280,720));b=cv2.resize(b,(1280,720))
  orb=cv2.ORB_create(nfeatures=5000,fastThreshold=20)
  k1,d1=orb.detectAndCompute(a,None); k2,d2=orb.detectAndCompute(b,None)
  good=[]
  if d1 is not None and d2 is not None and len(d1)>1 and len(d2)>1:
    for cand in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2):
      if len(cand)==2 and cand[0].distance<0.75*cand[1].distance:good.append(cand[0])
  inliers=0
  if len(good)>=4:
    u=np.float32([k1[g.queryIdx].pt for g in good]).reshape(-1,1,2)
    v=np.float32([k2[g.trainIdx].pt for g in good]).reshape(-1,1,2)
    h,mask=cv2.findHomography(u,v,cv2.RANSAC,4.0)
    if h is not None and mask is not None:inliers=int(mask.sum())
  ratio=round(inliers/len(good),5) if good else 0
  is_strong = inliers>=80 and ratio>=0.50
  return dict(ratio_filtered_matches=len(good),homography_inliers=inliers,
       inlier_fraction=ratio,scene_overlap_evidence=('STRONG' if is_strong else 'INCONCLUSIVE_SAMPLE'),
       independence_verified='NO',
       interpretation=('matched background/scene; exact flight relationship UNKNOWN' if is_strong else
                       'absence of a strong sample match does NOT prove independent flights'))

def verify_evaluation(path:Path,sequence:str,annotations_hash:str,lock:dict)->dict:
  with zipfile.ZipFile(path) as z:
    if z.testzip():raise ValueError('Corrupted evaluation ZIP')
    ms=[f for f in z.namelist() if f.endswith('evaluation_manifest.json')]
    if len(ms)!=1:raise ValueError('No unique evaluation manifest')
    mf=json.loads(z.read(ms[0]))
    if mf.get('sequence_id')!=sequence:raise ValueError('Evaluation sequence mismatch')
    if mf.get('file_sha256')!=annotations_hash:raise ValueError('Evaluation annotation sha mismatch')
    for h in (20,50):
      k=f'phase7j_dev_gate_h{h}.json'
      if mf.get('gate_sha256',{}).get(k)!=lock['release_files_sha256'][f'locked_models/{k}']:
        raise ValueError(f'Gate SHA mismatch: {k}')
    for k,v in lock['model_checkpoint_sha256'].items():
      rec=mf.get('checkpoint_hashes_verified',{}).get(k)
      if not isinstance(rec,dict) or rec.get('sha256')!=v['sha256']:
        raise ValueError(f'Checkpoint hash mismatch in manifest: {k}')
    predicted=mf.get('n_scored_windows',{})
    return dict(sequence_id=sequence,evaluation_zip=path.name,evaluation_zip_sha256=sha256(path),
       manifest_matches_source_sha256='YES',gate_sha256_in_manifest_matches_lock='YES',
       checkpoint_sha256_in_manifest_matches_lock='YES',
       scored_windows_from_manifest=str(predicted),
       evaluation_scope='manifest verification; NOT forward-pass rerun; NOT flight independence proof')

def decide_groups(row:dict)->dict:
  seq=row['sequence_id']
  return dict(sequence_id=seq,project_role=PROJECT_ROLE[seq],
    provisional_group=PROVISIONAL_GROUP[seq],scene_overlap_with_development=(
      'YES_WITH_DJI_0204_video2' if seq=='DJI_0204_video1' else
      'YES_WITH_DJI_0204_video1' if seq=='DJI_0204_video2' else 'UNKNOWN_NOT_DISPROVEN'),
    flight_id_verified='NO',original_video_time_verified='NO',verified_fps='NO',
    independent_of_development_capture_event='NOT_ESTABLISHED',
    acceptable_independent_external_confirmation='NO',
    rationale=('Same physical background scene documented; conservative grouping with development DJI_0204_video2' if seq=='DJI_0204_video1' else
      'Used for method development; cannot be untouched validation' if seq in ('DJI_0204_video2','DJI_0207') else
      'No flight ID, verified capture timestamps, or author event mapping; results exploratory until provenance confirmed'))

def run(inputs:Path,output:Path,with_images:bool=True)->dict:
  lock=check_lock();output.mkdir(parents=True,exist_ok=True)
  allinputs=[];seq=[];meta=[];g=[];evals=[];scenes=[]
  for name,(a,i) in SOURCE_MAP.items():
    ap=inputs/a;ip=inputs/i
    if not ap.is_file() or (with_images and not ip.is_file()):
      raise FileNotFoundError(f'Missing local original: {ap if not ap.is_file() else ip}')
    ar=audit_annotations(ap,name);seq.append(ar)
    allinputs.append(dict(file_name=a,sha256=sha256(ap),type='ANNOTATION',owner_sequence=name))
    if with_images:
      meta+=sample_image_meta(ip,name)
      allinputs.append(dict(file_name=i,sha256=sha256(ip),type='SPARSE_RGB_SAMPLE',owner_sequence=name))
    g.append(decide_groups(ar))
    if name in EVAL_MAP and (inputs/EVAL_MAP[name]).is_file():
      evp=inputs/EVAL_MAP[name]
      evals.append(verify_evaluation(evp,name,ar['annotation_sha256'],lock))
      allinputs.append(dict(file_name=evp.name,sha256=sha256(evp),type='LOCKED_EVALUATION_OUTPUT',owner_sequence=name))
  if with_images:
    for a,ai,b,bi in IMAGE_PAIRS:
      m=scene_compare(image_bytes(inputs/SOURCE_MAP[a][1],ai),image_bytes(inputs/SOURCE_MAP[b][1],bi))
      scenes.append(dict(sequence_a=a,frame_a=ai,sequence_b=b,frame_b=bi,**m))
    scene_0204=next(x for x in scenes if {x['sequence_a'],x['sequence_b']}=={'DJI_0204_video1','DJI_0204_video2'})
    if scene_0204['scene_overlap_evidence']!='STRONG':
      raise RuntimeError('Expected 0204 common-scene diagnostic is missing; requires manual review')
  output_fields={
   'annotation_inventory.csv':seq,
   'image_exif_presence_only.csv':meta,
   'scene_sample_matching.csv':scenes,
   'provisional_capture_group_decisions.csv':g,
   'locked_evaluation_manifest_checks.csv':evals,
   'source_sha256_inventory.csv':allinputs,
  }
  for filename,rows in output_fields.items():
    if rows:write_csv(output/filename,rows,list(rows[0]))
  assert all(r['acceptable_independent_external_confirmation']=='NO' for r in g)
  report={'phase':'7M','evidence_date':'2026-11-22','n_sequences_full_annotation_audited':len(seq),
    'n_annotated_frames_sum_NOT_INDEPENDENT':sum(r['annotated_frames'] for r in seq),
    'n_elephant_boxes_sum_NOT_INDEPENDENT':sum(r['elephant_boxes'] for r in seq),
    'n_eligible_windows_sum_NOT_INDEPENDENT':sum(r['eligible_h20_windows']+r['eligible_h50_windows'] for r in seq),
    'n_distinct_folders':len(seq),'n_author_verified_independent_capture_events':0,
    'n_provisional_groups_NOT_INDEPENDENCE_PROOF':len({r['provisional_group'] for r in g}),
    'n_image_samples_checked':len(meta),
    'n_scene_pairs_checked':len(scenes),
    'n_strong_scene_pairs':sum(x['scene_overlap_evidence']=='STRONG' for x in scenes),
    'phase7k_protocol_sha256':EXPECTED_LOCK_SHA256,
    'report_limitations':[
      'Official dataset does not supply a verified per-sequence flight/clip lineage in provided README.',
      'Image samples are sparse; negative image matches cannot prove different capture events.',
      'No camera FPS, genuine UTC capture timestamps or original video flight metadata verified.',
      'Only archived lock manifests verified, original neural checkpoint binaries not rerun.',
      'All currently assessed forecasting evidence remains exploratory or method development.',
    ]}
  (output/'audit_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
  return report

def verify_installed_lock(phase7k_kit:Path)->dict:
  lock=check_lock()
  missing=[];bad=[];pass_count=0
  for rel,expected in lock['release_files_sha256'].items():
    f=phase7k_kit/rel
    if not f.is_file():missing.append(rel)
    elif sha256(f)!=expected:bad.append(rel)
    else:pass_count+=1
  return {'files_checked':len(lock['release_files_sha256']),'matches':pass_count,
          'missing':missing,'different':bad,'passed':not missing and not bad}

def main():
  p=argparse.ArgumentParser(description=__doc__)
  sub=p.add_subparsers(dest='command',required=True)
  a=sub.add_parser('audit');a.add_argument('--inputs-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
  a.add_argument('--annotations-only',action='store_true')
  b=sub.add_parser('verify-release');b.add_argument('--phase7k-kit',type=Path,required=True)
  sub.add_parser('verify-lock')
  opts=p.parse_args()
  if opts.command=='audit':result=run(opts.inputs_dir,opts.output,not opts.annotations_only)
  elif opts.command=='verify-release':
    result=verify_installed_lock(opts.phase7k_kit)
    if not result['passed']:
      print(json.dumps(result,indent=2));sys.exit(2)
  else:
    result={'phase7k_protocol_sha256':EXPECTED_LOCK_SHA256,'status':'PASS' if check_lock() else 'FAIL'}
  print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
