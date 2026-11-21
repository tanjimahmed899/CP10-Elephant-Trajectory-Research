#!/usr/bin/env python3
"""Non-metric WildBox camera metadata preflight; NOT a corrected forecast experiment."""
import argparse,zipfile,json,math,hashlib
import numpy as np
import pandas as pd
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,default=Path('/mnt/data/CP10_real_frames_labels_sample.zip'));p.add_argument('--out',type=Path,default=Path('/mnt/data/CP10_Research_Stage1'));a=p.parse_args()
 a.out.mkdir(exist_ok=True,parents=True)
 with zipfile.ZipFile(a.zip) as z:
  files=z.namelist(); cam=next(x for x in files if x.endswith('/cameras.json'))
  obj=json.loads(z.read(cam));cams={q['image_name']:q for q in obj['cameras']}
  frames=sorted([x for x in files if x.endswith('.jpg')])
  first=cams[Path(frames[0]).name];r0=np.asarray(first['extrinsic'])[:,:3];t0=np.asarray(first['extrinsic'])[:,3]
  records=[]
  for f in frames:
   bn=Path(f).name
   q=cams.get(bn)
   if q is None: raise RuntimeError('Frame without camera record: '+bn)
   x=np.asarray(q['extrinsic'],float);R=x[:,:3];T=x[:,3];K=np.asarray(q['intrinsic'],float)
   theta=np.degrees(np.arccos(np.clip((np.trace(R@r0.T)-1)/2,-1,1)))
   ortho=np.linalg.norm(R@R.T-np.eye(3))
   records.append({'image_name':bn,'frame_index':q['frame_index'],
      'rotation_deg_relative_to_first':float(theta),
      'translation_distance_in_source_units':float(np.linalg.norm(T-t0)),
      'fx_px':float(K[0,0]),'fy_px':float(K[1,1]),'rotation_orthogonality_residual':float(ortho),
      'frame_width':q['image_width'],'frame_height':q['image_height'],
      'has_kitti_label':any(x.endswith('/kitti_labels/'+bn.replace('.jpg','.txt')) for x in files)})
 df=pd.DataFrame(records);df.to_csv(a.out/'R8_WildBox_Camera_Metadata_Preflight.csv',index=False)
 report={'source_zip':a.zip.name,'sha256':hashlib.sha256(a.zip.read_bytes()).hexdigest(),
         'frames_checked':len(frames),'camera_records_in_full_segment':len(cams),
         'matched_image_camera_count':len(df),
         'matching_kitti_label_count':int(df.has_kitti_label.sum()),
         'max_relative_rotation_deg':float(df.rotation_deg_relative_to_first.max()),
         'max_relative_translation_in_source_units':float(df.translation_distance_in_source_units.max()),
         'fx_min_px':float(df.fx_px.min()),'fx_max_px':float(df.fx_px.max()),
         'max_rotation_orthogonality_error':float(df.rotation_orthogonality_residual.max()),
         'limits':['Relative camera pose and intrinsic metadata found for supplied frames; coordinate conventions, depth and metric scale remain unverified.',
                   'No full world-coordinate transformation, learned raw-versus-corrected evaluation, landmark reprojection error, or metre-valued claim is established.']}
 (a.out/'R8_WildBox_Camera_Preflight.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
