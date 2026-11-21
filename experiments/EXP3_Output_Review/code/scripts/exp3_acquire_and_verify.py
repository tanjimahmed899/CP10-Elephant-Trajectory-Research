#!/usr/bin/env python3
"""Acquire locked new WildBox elephant ZIP files on a networked machine.

Dry-run is default. Full-content downloads require --download.
Never substitutes a video or modifies EXP3_new_video_holdout_LOCKED.csv.
"""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path
import pandas as pd
REPO='wildbox-anon-2026/wildbox-review'

def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--lock',type=Path,default=Path(__file__).resolve().parent/'EXP3_new_video_holdout_LOCKED.csv')
 p.add_argument('--dest',type=Path,required=True)
 p.add_argument('--download',action='store_true',help='download absent files; can require multiple GB')
 p.add_argument('--verify-full-zip',action='store_true',help='decompress and CRC-check each downloaded archive; slower')
 args=p.parse_args()
 lock=pd.read_csv(args.lock)
 assert len(lock)==6 and lock.repo_path.is_unique and lock.elephant_group.value_counts().to_dict()=={'elep1':2,'elep2':2,'elep3':2}
 if args.download:
  from huggingface_hub import hf_hub_download
 args.dest.mkdir(parents=True,exist_ok=True)
 out=[]
 for r in lock.itertuples(index=False):
  f=args.dest/r.repo_path
  state='MISSING'
  if not f.is_file() and args.download:
   print(f'Downloading {r.repo_path} (potentially large)...',flush=True)
   got=hf_hub_download(repo_id=REPO,repo_type='dataset',filename=r.repo_path,local_dir=str(args.dest))
   assert Path(got).resolve()==f.resolve()
  if f.is_file():
   state='ZIP_CENTRAL_DIRECTORY_OK' if zipfile.is_zipfile(f) else 'NOT_A_ZIP'
   if state=='ZIP_CENTRAL_DIRECTORY_OK' and args.verify_full_zip:
    with zipfile.ZipFile(f) as z:
     bad=z.testzip()
     state='FULL_ZIP_CRC_OK' if bad is None else 'BAD_CRC:'+bad
   d={'holdout_id':r.holdout_id,'repo_path':r.repo_path,'status':state,'size_bytes':f.stat().st_size,'content_sha256':sha_file(f) if state.startswith(('ZIP','FULL')) else ''}
  else:
   d={'holdout_id':r.holdout_id,'repo_path':r.repo_path,'status':'NOT_AVAILABLE','size_bytes':None,'content_sha256':''}
  out.append(d)
  print(f'{r.holdout_id}: {r.repo_path} — {d["status"]}')
 df=pd.DataFrame(out)
 df.to_csv(args.dest/'EXP3_local_source_integrity_receipt.csv',index=False)
 ready=int(df.status.str.startswith(('ZIP','FULL')).sum())
 print(f'Available: {ready}/6; receipt: {args.dest / "EXP3_local_source_integrity_receipt.csv"}')
 if args.download and ready!=6:raise SystemExit('Acquisition incomplete: all six locked files required; do not replace them based on QC outcomes.')

if __name__=='__main__':main()
