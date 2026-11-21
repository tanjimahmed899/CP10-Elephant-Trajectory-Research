#!/usr/bin/env python3
"""WSL: collect minimally sufficient original raw BIRDSAI tracks for matched R8 experiments, without modifying the original source tree.
Requires Python stdlib only. Script outputs a ZIP for upload, with raw 30Hz cx_n/cy_n + matched window metadata.
"""
from pathlib import Path
import argparse, csv, json, zipfile, collections, datetime

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--root',type=Path,default=Path('/mnt/e/AIUB/Capstone Project/Journal Based Results'))
 ap.add_argument('--out',type=Path,default=Path.home()/'CP10_Stage4_R8_Raw_Pairing')
 args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
 base=args.root/'Main Result';raw=base/'trajectory_dataset'/'cleaned_track_points.csv'
 corr=base/'corrected_trajectory_dataset'
 if not raw.exists():raise SystemExit('MISSING: '+str(raw))
 ranges=collections.defaultdict(list); meta={}
 for task in ('2s_to_2s','2s_to_5s'):
  path=corr/f'corrected_windows_{task}.csv'
  if not path.exists():raise SystemExit('MISSING: '+str(path))
  with path.open(newline='',encoding='utf-8-sig') as f:
   records=[]
   for r in csv.DictReader(f):
    if str(r['accepted'])!='1' or r['quality_tier'].lower()!='core':continue
    key=(r['split'],r['sequence'],r['segment_id'])
    ranges[key].append((int(r['obs_start_frame']),int(r['future_end_frame'])))
    records.append(r)
   meta[task]=records
 if not ranges:raise SystemExit('No accepted core metadata windows')
 outcsv=out/'R8_Matched_Original_Raw_30Hz_Tracks.csv'
 saved=collections.defaultdict(set);total=0;write=0
 with raw.open('r',encoding='utf-8-sig',newline='') as src, outcsv.open('w',encoding='utf8',newline='') as dst:
  reader=csv.DictReader(src);cols=list(reader.fieldnames or [])
  must={'split','sequence','segment_id','frame','cx_n','cy_n'}
  if not must.issubset(cols): raise SystemExit('Raw CSV missing columns: '+str(must-set(cols)))
  writer=csv.DictWriter(dst,fieldnames=cols);writer.writeheader()
  # Reduce large file copy by limiting to frames needed by accepted matching windows.
  for row in reader:
   total+=1;key=(row['split'],row['sequence'],row['segment_id'])
   runs=ranges.get(key)
   if not runs:continue
   frame=int(row['frame'])
   if any(lo<=frame<=hi for lo,hi in runs):
    writer.writerow(row);write+=1;saved[key].add(frame)
 checks=[]
 for task,records in meta.items():
  for row in records:
   key=(row['split'],row['sequence'],row['segment_id'])
   lo,hi=int(row['obs_start_frame']),int(row['future_end_frame'])
   have=saved.get(key,set());missing=sum(f not in have for f in range(lo,hi+1))
   checks.append(dict(task=task,source_index=row['source_index'],split=row['split'],sequence=row['sequence'],
    segment_id=row['segment_id'],sample_id=row['sample_id'],obs_start_frame=lo,future_end_frame=hi,
    missing_raw_frames=missing,valid=missing==0))
 with (out/'R8_Matched_Raw_Sample_Completeness.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
 summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw_input':str(raw),'raw_size':raw.stat().st_size,
          'source_rows_examined':total,'raw_rows_collected':write,'unique_tracks':len(saved),'core_windows':len(checks),
          'complete_windows':sum(r['valid'] for r in checks),'incomplete_windows':sum(not r['valid'] for r in checks),
          'notes':'Original raw positions (cx_n/cy_n) in each source image frame; MUST be explicitly registered to corrected reference frame before comparing prediction errors.'}
 (out/'R8_Stage4_Raw_Completeness_Summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
 zpath=out/'CP10_Stage4_R8_Raw_Pairing.zip'
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for q in [outcsv,out/'R8_Matched_Raw_Sample_Completeness.csv',out/'R8_Stage4_Raw_Completeness_Summary.json']:
   z.write(q,q.name)
 print('Output:',zpath);print(json.dumps(summary,indent=2))
 print('Source tree unchanged.')

if __name__=='__main__':main()
