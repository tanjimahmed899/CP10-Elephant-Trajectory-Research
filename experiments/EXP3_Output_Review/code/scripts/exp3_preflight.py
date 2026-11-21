#!/usr/bin/env python3
"""Validate that all six locked new source archives and six frozen model checkpoints exist.

This does not run predictions or certify new source independence at a site level.
"""
from __future__ import annotations
import argparse,hashlib,zipfile,json
from pathlib import Path
import pandas as pd
TASKS=['2s_to_2s','2s_to_5s'];SEEDS=[42,52,62]
def sha_file(f):
 h=hashlib.sha256()
 with f.open('rb') as fin:
  for b in iter(lambda:fin.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 here=Path(__file__).resolve().parent
 ap.add_argument('--lock',type=Path,default=here/'EXP3_new_video_holdout_LOCKED.csv')
 ap.add_argument('--data-root',type=Path,required=True)
 ap.add_argument('--model-root',type=Path,required=True)
 ap.add_argument('--out',type=Path,default=here/'EXP3_local_preflight.json')
 args=ap.parse_args()
 lock=pd.read_csv(args.lock)
 assert len(lock)==6
 files=[]
 for row in lock.itertuples(index=False):
  f=args.data_root/row.repo_path
  files.append({'kind':'source','id':row.holdout_id,'file':str(f),'exists':f.is_file(),'valid_zip':f.is_file() and zipfile.is_zipfile(f),'sha256':sha_file(f) if f.is_file() and zipfile.is_zipfile(f) else ''})
 for task in TASKS:
  for seed in SEEDS:
   f=args.model_root/f'{task}_Transformer_seed{seed}.pt'
   files.append({'kind':'checkpoint','id':f'{task}-seed{seed}','file':str(f),'exists':f.is_file(),'valid_zip':None,'sha256':sha_file(f) if f.is_file() else ''})
 ok=all(x['exists'] and (x['valid_zip'] if x['kind']=='source' else True) for x in files)
 res={'status':'READY_TO_RUN' if ok else 'BLOCKED_MISSING_INPUTS', 'data_and_model_files':files,'model_checkpoint_hashes_must_match_original_historical_checkpoint_manifest':True,'no_new_results_generated':True}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(res,indent=2)+'\n')
 print(res['status']);print(f'{sum(x["exists"] for x in files)}/{len(files)} files present');print('Preflight receipt:',args.out)
 return 0 if ok else 2
if __name__=='__main__':raise SystemExit(main())
