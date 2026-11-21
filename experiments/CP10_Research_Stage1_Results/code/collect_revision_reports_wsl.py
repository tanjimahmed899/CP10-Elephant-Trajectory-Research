#!/usr/bin/env python3
"""Read-only, size-bounded collection of existing R1-R19 revision reports from Windows E: drive via WSL."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse, hashlib, json, re

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--root',type=Path,default=Path('/mnt/e/AIUB/Capstone Project/Journal Based Results/Main Result 3.0/paper_strengthening_revision'))
 p.add_argument('--out',type=Path,default=Path.home()/'CP10_Research_Stage2_Revision_Reports.zip')
 p.add_argument('--max-mb',type=float,default=25.)
 a=p.parse_args()
 if not a.root.is_dir():p.error('Revision directory not found: '+str(a.root))
 suffixes={'.csv','.json','.md','.txt'}
 include=re.compile(r'(claim|report|interpret|summary|metric|conclusion|bootstrap|protocol|lock|audit|coverage|warning|source|reconcil|analysis|log|result|table|feasib|preflight|frozen)',re.I)
 exclude=re.compile(r'(raw_prediction|raw_instances|all_instances|source_image|manifest_with_all|full_bbox|all_boxes|large_array|checkpoint|\.npz)',re.I)
 items=[];used=0;limit=int(a.max_mb*1024*1024)
 for f in sorted(a.root.rglob('*')):
  if not f.is_file() or f.suffix.lower() not in suffixes or f.stat().st_size>3_000_000: continue
  if not include.search(f.name) or exclude.search(f.name): continue
  size=f.stat().st_size
  if size+used>limit: continue
  items.append(f); used+=size
 summary={'root':str(a.root),'included_count':len(items),'included_bytes':used,'files':[{'path':str(x.relative_to(a.root)),'size':x.stat().st_size} for x in items]}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with ZipFile(a.out,'w',ZIP_DEFLATED,compresslevel=6) as z:
  for f in items: z.write(f,arcname=str(f.relative_to(a.root)))
  z.writestr('COLLECTION_MANIFEST.json',json.dumps(summary,indent=2))
 print('Created:',a.out,'files:',len(items),'input bytes:',used,'ZIP bytes:',a.out.stat().st_size)
 print('No files changed in source tree.')
if __name__=='__main__':main()
