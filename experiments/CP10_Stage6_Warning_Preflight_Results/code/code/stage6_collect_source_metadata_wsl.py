#!/usr/bin/env python3
"""Stage 6 read-only metadata collector for a fixed-camera-reference fence audit.

Copies only cameras.json and tracking_summary.json from 9 original WildBox ZIPs.
Does not modify source files, perform training or claim physical-world scale.
No non-standard dependencies.
"""
import argparse, hashlib, io, json, zipfile
from pathlib import Path
from datetime import datetime, timezone

VIDEO_GROUPS={
 'elep1':['DJI_20240119151048_0001_V','DJI_20240119151437_0002_V','DJI_20240119152552_0006_V'],
 'elep2':['DJI_20240626143248_0002_V','DJI_20240626143443_0003_V','DJI_20240626143832_0004_V'],
 'elep3':['DJI_20260227082707_0003_V','DJI_20260227084724_0003_V','DJI_20260227084907_0004_V']}
STAGE5={'DJI_20240119152552_0006_V','DJI_20240626143832_0004_V','DJI_20260227084907_0004_V'}

def locate(root,group,video):
 choices=[root/'Q1_Research_Completion_Sprint'/'new_data'/group/(video+'.zip'),
          root/'Main Result'/(video+'.zip'),root/'ZIP 4.0'/(video+'.zip')]
 for p in choices:
  if p.is_file():return p
 # Deliberately do not descend into the complete E: drive; notify user about exact missing source.
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path)
 ap.add_argument('--out',type=Path,default=Path.home()/'CP10_Stage6_StaticFence_Metadata.zip')
 args=ap.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
 manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'metadata-only geometrical reference-frame feasibility','videos':[],
 'limitations':['Cameras and tracked 3D centers are dataset-provided estimates, not independently measured world poses.',
 'Source metadata is retrospective ground truth and cannot be used as future information in an online algorithm.',
 'Metadata collection alone does not establish a stable calibrated physical geofence or metre scale.']}
 success=0;errors=[];k=0
 with zipfile.ZipFile(args.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as dest:
  for group,vids in VIDEO_GROUPS.items():
   for video in vids:
    src=locate(args.root,group,video)
    entry={'video':video,'elephant_group':group,'role':'Stage5_candidate' if video in STAGE5 else 'EXP3_source','zip_source':str(src) if src else None,
       'status':'MISSING','segments':[]}
    if src is None:
     errors.append(f'Missing archive: {video}.zip');manifest['videos'].append(entry);continue
    try:
     with zipfile.ZipFile(src) as z:
      members={}
      for name in z.namelist():
       q=name.strip('/').split('/')
       if len(q)>=3 and q[-2].startswith('seg') and q[-1] in {'cameras.json','tracking_summary.json'}:
        segment=q[-2];members.setdefault(segment,{})[q[-1]]=name
      for seg,files in sorted(members.items()):
       if not all(x in files for x in ('cameras.json','tracking_summary.json')):
        errors.append(f'Missing cameras or tracks: {video}/{seg}');continue
       for kind in ('cameras.json','tracking_summary.json'):
        payload=z.read(files[kind])
        destname=f'Original_Source_Metadata/{group}/{video}/{seg}/{kind}'
        dest.writestr(destname,payload)
        k+=1
        if kind=='cameras.json':
         obj=json.loads(payload)
         cameras=int(len(obj['cameras']))
        else:
         obj=json.loads(payload)
         tracks=int(len(obj['tracks']))
       entry['segments'].append({'name':seg,'camera_records':cameras,'track_keys':tracks})
      if entry['segments']:
       entry['status']='COLLECTED';success+=1
      else:errors.append(f'No pairs of camera/tracking metadata: {video}')
    except Exception as e:
     entry['status']='ERROR';entry['error']=repr(e);errors.append(f'{video}: {repr(e)}')
    manifest['videos'].append(entry)
  manifest['summary']={'videos_collected':success,'videos_expected':9,'json_files_collected':k,'errors':errors}
  dest.writestr('CP10_Stage6_Metadata_Manifest.json',json.dumps(manifest,indent=2))
 print(f'Output: {args.out}')
 print(f'Collected: {success}/9 source video archives; JSON files: {k}')
 print('Segment counts:',[(e['video'],len(e['segments'])) for e in manifest['videos']])
 if errors: print('Errors:',*errors,sep='\n  ')
 print('Original source tree unchanged.')
 if success!=9:raise SystemExit('INCOMPLETE: upload zip anyway; inspect manifest for missing videos')
if __name__=='__main__':main()
