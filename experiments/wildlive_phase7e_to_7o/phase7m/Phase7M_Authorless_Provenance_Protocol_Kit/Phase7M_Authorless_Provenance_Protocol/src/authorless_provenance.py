#!/usr/bin/env python3
"""Privacy-preserving, author-independent *scene relation* audit (NOT flight-ID verification).

Dependencies: numpy, OpenCV (cv2), Pillow. Python 3.10+.
Copies NO RGB pixels, geolocation or raw annotation into output tables.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, re, zipfile
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_IMAGES = {
 'DJI_0207': 'WildLive_DJI0207_priority_frames.zip',
 'DJI_0204_video2': 'DJI_0204_video2_visual_QA_frames.zip',
 'DJI_0204_video1': 'DJI_0204_video1_visual_qa_frames.zip',
 'DJI_0161_video1': 'DJI_0161_video1_visual_qa_frames.zip',
}
ROLES = {'DJI_0207':'DEVELOPMENT', 'DJI_0204_video2':'DEVELOPMENT',
         'DJI_0204_video1':'RELATED_SCENE_EXPLORATORY',
         'DJI_0161_video1':'PREVIOUSLY_EVALUATED_INDEPENDENCE_UNKNOWN'}


def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
def sha256_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda: f.read(2**20), b''):h.update(b)
 return h.hexdigest()

def csv_save(path, rows, fields):
 path.parent.mkdir(parents=True, exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
  w.writeheader();w.writerows(rows)


def detect_features(jpg, width=1024, height=576):
 import cv2, numpy as np
 im=cv2.imdecode(np.frombuffer(jpg,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
 if im is None:raise RuntimeError('Invalid JPEG')
 im=cv2.resize(im,(width,height),interpolation=cv2.INTER_AREA)
 orb=cv2.ORB_create(nfeatures=3200,fastThreshold=18)
 key,desc=orb.detectAndCompute(im,None)
 return key,desc


def compare_features(a,b):
 import cv2, numpy as np
 k1,d1=a;k2,d2=b
 good=[]
 if d1 is not None and d2 is not None and len(d1)>2 and len(d2)>2:
  for matches in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2):
   if len(matches)==2 and matches[0].distance < .75*matches[1].distance:
    good.append(matches[0])
 inliers=0; coverage=0.
 if len(good)>=5:
  pts1=np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
  pts2=np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
  H,mask=cv2.findHomography(pts1,pts2,cv2.RANSAC,4.0)
  if H is not None and mask is not None:
   flag=mask.reshape(-1).astype(bool)
   inliers=int(flag.sum())
   if inliers>=4:
    # convex-hull area fraction of the first image (not geographic area)
    p=pts1.reshape(-1,2)[flag]
    coverage=float(cv2.contourArea(cv2.convexHull(p)))/(1024*576)
 ratio=inliers/len(good) if good else 0.
 # Heuristic for *positive* scene overlap; negative matches never establish independence.
 strong = (inliers>=80 and ratio >= .55 and coverage >= .015)
 return dict(filtered_matches=len(good),ransac_inliers=inliers,
            inlier_fraction=round(ratio,4),inlier_coverage_frac=round(coverage,4),
            evidence='STRONG_SHARED_SCENE' if strong else 'INCONCLUSIVE',
            independent_flight_proven='NO')


def read_inputs(input_dir):
 from PIL import Image
 records=[];feat={}; hashes={}
 for seq, fn in SOURCE_IMAGES.items():
  p=input_dir / fn
  if not p.exists():raise FileNotFoundError(f'Missing original JPEG ZIP: {p}')
  hashes[fn]=sha256_file(p)
  with zipfile.ZipFile(p) as z:
   bad=z.testzip()
   if bad:raise RuntimeError(f'ZIP CRC failure: {fn}: {bad}')
   seen=set()
   for name in z.namelist():
    m=re.search(r'(?:^|/)frame_(\d+)\.(?:jpg|jpeg)$',name,re.I)
    if not m:continue
    frame=int(m.group(1))
    if frame in seen:raise RuntimeError(f'Duplicate frame {seq}:{frame}')
    seen.add(frame)
    jpg=z.read(name)
    with Image.open(io.BytesIO(jpg)) as im:
     tags=set(im.getexif().keys())
     records.append(dict(sequence=seq,frame_index=frame,
                         image_width=im.width,image_height=im.height,
                         jpeg_sha256=sha256_bytes(jpg),
                         capture_datetime_exif_present=any(k in tags for k in (306,36867,36868)),
                         gps_exif_present=34853 in tags))
    feat[(seq,frame)]=detect_features(jpg)
 return sorted(records,key=lambda x:(x['sequence'],x['frame_index'])),feat,hashes


def audit(input_dir:Path,out:Path):
 images,feat,hashes=read_inputs(input_dir)
 pairs=[]
 # all across distinct recording folders, never within a folder.
 for a,b in combinations(images,2):
  if a['sequence']==b['sequence']:continue
  result=compare_features(feat[(a['sequence'],a['frame_index'])],
                          feat[(b['sequence'],b['frame_index'])])
  pairs.append(dict(sequence_a=a['sequence'],frame_a=a['frame_index'],
                    sequence_b=b['sequence'],frame_b=b['frame_index'],
                    exact_jpeg_equal=a['jpeg_sha256']==b['jpeg_sha256'],**result))
 fields=list(images[0]);csv_save(out/'sample_image_metadata_presence.csv',images,fields)
 pair_fields=list(pairs[0]);csv_save(out/'all_cross_recording_image_pairs.csv',pairs,pair_fields)
 grouped=defaultdict(list)
 for r in pairs:
  grouped[tuple(sorted((r['sequence_a'],r['sequence_b'])))].append(r)
 summaries=[];links=[]
 for (a,b),val in sorted(grouped.items()):
  best=max(val,key=lambda x:(x['ransac_inliers'],x['inlier_fraction']))
  strong=[v for v in val if v['evidence']=='STRONG_SHARED_SCENE']
  if strong:links.append((a,b))
  summaries.append(dict(sequence_a=a,sequence_b=b,examined_frame_pairs=len(val),
                        strong_shared_scene_pairs=len(strong),
                        maximum_homography_inliers=best['ransac_inliers'],
                        max_inlier_fraction=best['inlier_fraction'],
                        best_frame_a=best['frame_a'] if best['sequence_a']==a else best['frame_b'],
                        best_frame_b=best['frame_b'] if best['sequence_b']==b else best['frame_a'],
                        evidence=('SHARED_SCENE_OBSERVED' if strong else 'NO_POSITIVE_MATCH_IN_SAMPLES'),
                        independent_flight_proven='NO'))
 csv_save(out/'recording_pair_summary.csv',summaries,list(summaries[0]))
 # Components from positive links only. Unlinked nodes are unknown, not confirmed independent.
 parent={seq:seq for seq in SOURCE_IMAGES}
 def root(x):
  while parent[x]!=x:
   x=parent[x]
  return x
 for a,b in links:
  p,q=root(a),root(b)
  if p!=q: parent[max(p,q)]=min(p,q)
 groups=defaultdict(list)
 for seq in SOURCE_IMAGES:groups[root(seq)].append(seq)
 labels={rootseq:('RELATED_SCENE_GROUP_'+str(i+1) if len(nodes)>1 else 'UNRESOLVED_SINGLETON_'+str(i+1))
         for i,(rootseq,nodes) in enumerate(sorted(groups.items()))}
 register=[]
 for seq in sorted(SOURCE_IMAGES):
  comp=sorted(groups[root(seq)])
  register.append(dict(sequence=seq,project_role=ROLES[seq],
              evidence_based_scene_component=labels[root(seq)],
              connected_scene_members=';'.join(comp),
              scene_connection_detected=len(comp)>1,
              verified_flight_or_session_id='NOT_AVAILABLE',
              verified_independence_of_development='NO',
              suitable_as_independent_confirmatory_test='NO',
              permitted_claim='sequence-level retrospective evidence with unverified capture-event provenance'))
 csv_save(out/'provisional_no_author_group_register.csv',register,list(register[0]))
 status=dict(sampled_rgb_files=len(images),cross_recording_pairs=len(pairs),
       recording_pair_tests=len(summaries),strong_shared_scene_image_pairs=sum(x['evidence']=='STRONG_SHARED_SCENE' for x in pairs),
       positive_scene_links=links,
       scene_connected_components=[v for v in groups.values()],
       confirmed_independent_capture_events=0,
       note='No match is not proof of independence; no source video/FPS/authoritative flight ID in these samples',
       source_zip_sha256=hashes,
       parameters=dict(orb_features=3200,downsample=[1024,576],ratio=0.75,ransac_pixels=4.0,
                       minimum_inliers=80,minimum_fraction=.55,minimum_coverage=.015))
 (out/'summary.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
 print(json.dumps({k:status[k] for k in ('sampled_rgb_files','cross_recording_pairs','recording_pair_tests','strong_shared_scene_image_pairs','scene_connected_components','confirmed_independent_capture_events')},indent=2))
 return status


def census_zip(path):
 """Full annotation JSON species census, not necessarily a full video species census."""
 counts=Counter();frames=set();bad_json=0
 with zipfile.ZipFile(path) as z:
  if z.testzip():raise RuntimeError('Bad ZIP')
  for filename in z.namelist():
   m=re.search(r'frame_(\d+)\.json$',filename)
   if not m:continue
   f=int(m.group(1));frames.add(f)
   try:
    obj=json.loads(z.read(filename))
    for s in obj.get('shapes',[]):counts[str(s.get('label','')).lower().strip()]+=1
   except Exception:bad_json+=1
 return dict(archive=path.name,frames_with_json=len(frames),
             min_frame=min(frames) if frames else None,max_frame=max(frames) if frames else None,
             elephant_annotations=counts.get('elephant',0),
             other_species_counts=json.dumps({k:v for k,v in sorted(counts.items()) if k!='elephant'}),
             unreadable_json=bad_json,
             species_scope='full available annotation ZIP, not original video')


def cmd_census(root,out):
 files=sorted(root.glob('**/*_boxid.zip'))
 if not files:raise ValueError(f'No *_boxid.zip files under {root}')
 rows=[census_zip(p) for p in files]
 csv_save(out,rows,list(rows[0]));print(f'Wrote {len(rows)} annotation-archive records to {out}')


def main():
 p=argparse.ArgumentParser()
 sub=p.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('audit');a.add_argument('--inputs-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
 c=sub.add_parser('census');c.add_argument('--zip-dir',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
 arg=p.parse_args()
 if arg.cmd=='audit':audit(arg.inputs_dir,arg.output)
 else:cmd_census(arg.zip_dir,arg.output)
if __name__=='__main__':main()
