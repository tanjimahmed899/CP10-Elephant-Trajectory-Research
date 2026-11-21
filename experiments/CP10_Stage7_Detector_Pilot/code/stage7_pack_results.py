#!/usr/bin/env python3
"""Package only compact auditable Stage7 data, no photos or checkpoints."""
import argparse,zipfile,json
from pathlib import Path

FILES=['Stage7_Preflight_Manifest.json','Stage7_Readout.txt','Stage7_TrainReal_Availability.csv',
 'Stage7_Dev_Image_Manifest.csv','Stage7_Dev_GT_Boxes.csv',
 'Stage7_Dev_Run_Provenance.json','Stage7_Dev_Detections.csv','Stage7_Dev_Inference_Times.csv',
 'Stage7_Dev_PerImage_IoU50.csv','Stage7_Dev_PerSequence_IoU50.csv',
 'Stage7_Dev_Overall_IoU50.csv','Stage7_Dev_Selected_Configuration_PRETEST.json',
 'Stage7_Dev_Decision.txt']

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default=str(Path.home()/'CP10_Stage7_Dev_Pilot'));p.add_argument('--zip',default=str(Path.home()/'CP10_Stage7_Dev_Review.zip'))
 a=p.parse_args();root=Path(a.out).expanduser();zpath=Path(a.zip).expanduser();n=0
 with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
  for x in FILES:
   f=root/x
   if f.exists():z.write(f,arcname=x);n+=1
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None
 print(f'Collected {n} evidence files. Original data and checkpoint excluded.\nOutput: {zpath}')

if __name__=='__main__':main()
