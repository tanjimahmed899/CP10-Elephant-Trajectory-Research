#!/usr/bin/env python3
"""Filename-based manuscript evidence presence check; NEVER interprets files as scientifically verified."""
import argparse,csv,os
from pathlib import Path
CHECKS=[
('Original YOLO baseline','yolov8','elephant_detection','detection_baseline'),
('BIRDSAI / WildBox preprocessing','build_unified_10hz','dataset_audit','corrected_trajectory'),
('Fair forecasting R2','R2_','R2','lstm_gru'),
('Conformal R4','R4_','conformal','Step6'),
('R5-R7 warning','R6_','warning','geofence'),
('R8 & Stage 4 camera ablation','Stage4','R8_','ego_motion'),
('EXP3 benchmark','EXP3','Stage6','wildbox'),
('Stage 5 calibration failure','Stage5','CALIBRATION','calibration'),
('Stage 6 fixed reference events','Stage6','FixedReference','fixed_reference'),
('Stage 7A/B development and training','Stage7B','Stage7A','stage7b'),
('Stage 7C detection','Stage7C','stage7c','Detector'),
('Stage 7D ByteTrack','Stage7D','stage7d','Tracker'),
('Stage 7E strict GT linkage','Stage7E','stage7e','GT_Linkage'),
('Stage 7F frozen forecasting','Stage7F','stage7f','Tracker_Forecast'),
('Stage 8 statistical audit','Stage8','stage8','Statistical'),
('R9-R19 revision history','Revision_Reports','paper_strengthening','R19'),
('BAMBI cross-domain','bambi','K3','step6d'),
('Final manuscript / faculty evidence','docs/papers/','Final_Demo','Faculty')]

def main():
 a=argparse.ArgumentParser();a.add_argument('--repo',default='.');a.add_argument('--out',default='PAPER_EVIDENCE_PRESENCE_CHECK.csv');args=a.parse_args();root=Path(args.repo).resolve()
 paths=[]
 for base,dirs,files in os.walk(root):
  dirs[:]=[x for x in dirs if x not in ('.git','.venv','venv','__pycache__','node_modules')]
  paths.extend((Path(base)/f).relative_to(root).as_posix() for f in files)
 out=Path(args.out)
 with out.open('w',encoding='utf8',newline='') as f:
  writer=csv.writer(f);writer.writerow(['claim_family','filename_match_count','example_paths','status'])
  for row in CHECKS:
   title,*tokens=row
   matches=[p for p in paths if any(token.lower() in p.lower() for token in tokens)]
   writer.writerow([title,len(matches),' | '.join(matches[:3]),'FILE_NAME_MATCH_ONLY' if matches else 'NOT_FOUND_BY_NAME'])
 print('File name coverage check written:',out)
 print('IMPORTANT: A filename match does NOT prove reproducible code, scientific validity or correct source/checkpoint.')
if __name__=='__main__':main()
