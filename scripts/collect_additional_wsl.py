#!/usr/bin/env python3
"""Optional extra WSL code inventory and COPY (never move) from E: drive.
By default DRY RUN. Supply --apply to copy; optional --include-results for non-code files.
"""
import argparse,os,hashlib,csv
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--root',default='/mnt/e/AIUB/Capstone Project/Journal Based Results')
p.add_argument('--apply',action='store_true')
p.add_argument('--include-results',action='store_true')
p.add_argument('--max-mb',type=int,default=10)
a=p.parse_args();src=Path(a.root); dest=Path(__file__).resolve().parents[1]/'additional_wsl'
if not src.is_dir():raise SystemExit('Source root not found: '+str(src))
ignore={'venv','.venv','site-packages','node_modules','__pycache__','.git','images','imgs','frames','labels','video','videos','dataset','datasets','checkpoints','weights','runs','raw','media','masks'}
code={'.py','.ipynb','.sh','.ps1','.r','.m'}
results={'.csv','.md','.json','.txt','.yaml','.yml'}
found=0;copied=0
for base,dirs,files in os.walk(src):
 dirs[:]=[d for d in dirs if d.lower() not in ignore and not d.startswith('.')]
 for name in files:
  f=Path(base)/name;ext=f.suffix.lower()
  if ext not in code and not(a.include_results and ext in results):continue
  if f.stat().st_size>a.max_mb*1024*1024:continue
  if any(x in name.lower() for x in ('camera','tracking_summary','credential','secret','private_key','gps','latitude','longitude','geoloc')):continue
  rel=f.relative_to(src)
  if any(part in {'..'} for part in rel.parts):continue
  found+=1
  if a.apply:
   out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True)
   out.write_bytes(f.read_bytes());copied+=1
print(f'Found {found} eligible local files. Copied {copied}.')
print('Mode:', 'COPY COMPLETE' if a.apply else 'DRY RUN: add --apply to copy.')
print('Review additional_wsl/ for sensitive paths, data rights and duplication BEFORE git commit.')
