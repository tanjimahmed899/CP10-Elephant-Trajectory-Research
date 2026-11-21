#!/usr/bin/env python3
"""Package compact evidence and model hash. No source frames."""
import argparse,json,zipfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--zip',default='');p.add_argument('--include-best-pt',action='store_true');a=p.parse_args()
 out=Path(a.out).expanduser().resolve();dest=Path(a.zip).expanduser().resolve() if a.zip else Path.home()/'CP10_Stage7B_TrainReal_Review.zip'
 if not (out/'Stage7B_Frozen_Preflight.json').exists():raise SystemExit('Missing preflight evidence')
 names=['Stage7B_Frozen_Preflight.json','Stage7B_Dataset_Manifest.csv','Stage7B_Training_Protocol_LOCKED.json','Stage7B_TrainReal_Validation_Readout.json']
 for x in names[2:]:
  if not (out/x).exists():print('WARNING, missing:',x)
 sub=out/'training_runs'/'domain_adapt'
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for x in names:
   f=out/x
   if f.is_file():z.write(f,x)
  for x in ('args.yaml','results.csv'):
   f=sub/x
   if f.is_file():z.write(f,'training_runs/domain_adapt/'+x)
  if a.include_best_pt:
   f=sub/'weights'/'best.pt'
   if f.is_file():z.write(f,'training_runs/domain_adapt/weights/best.pt')
  for f in [out/'CP10_Stage7B_Train.log',out/'CP10_Stage7B_Prepare.log']:
   if f.is_file():z.write(f,f.name)
 print('Review ZIP:',dest,'bytes=',dest.stat().st_size)

if __name__=='__main__':main()
