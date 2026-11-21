#!/usr/bin/env python3
"""Package Stage7C outputs for review, whether evaluation succeeded or failed."""
import argparse,os,zipfile,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--dest',required=True)
a=p.parse_args();out=Path(a.out).expanduser().resolve();dest=Path(a.dest).expanduser().resolve();dest.parent.mkdir(parents=True,exist_ok=True)
selected=[]
if out.is_dir():
 for f in sorted(out.iterdir()):
  if f.is_file() and f.suffix in ('.json','.csv','.txt','.log'):
   selected.append((f,f.name))
for f in [Path.home()/'CP10_Stage7C_Run.log']:
 if f.is_file():selected.append((f,f.name))
selected.append((Path(__file__).resolve().parent/'README.md','README.md'))
selected.append((Path(__file__).resolve().parent/'STAGE7C_LOCK.json','STAGE7C_LOCK.json'))
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f,arc in selected:z.write(f,arc)
with zipfile.ZipFile(dest) as z:bad=z.testzip()
if bad:raise SystemExit('Bad zip member '+bad)
print('Created',dest,'with',len(selected),'files; completed?', (out/'Stage7C_Overall_IoU50.csv').is_file())
