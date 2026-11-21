#!/usr/bin/env python3
"""Safely fix the one-line DataFrame list aggregation bug in existing Stage7F code.
Does not change model, data preprocessing, inference, seed, checkpoint, or metrics.
"""
from pathlib import Path
import argparse,shutil,hashlib
ap=argparse.ArgumentParser();ap.add_argument('--script',type=Path,default=Path.home()/'cp10_stage7f_tools/CP10_Stage7F_Frozen_Tracker_Forecasting/stage7f_run.py');args=ap.parse_args()
p=args.script.expanduser().resolve();s=p.read_text()
old='    d=pd.DataFrame(rows)\n';new='    d=pd.concat(rows,ignore_index=True)\n'
if old not in s:
 if new in s:print('Already patched',p);raise SystemExit(0)
 raise RuntimeError('Expected exact unpatched code not found; no change made.')
if s.count(old)!=1:raise RuntimeError('Expected exactly one bug site, no changes made.')
bak=p.with_suffix('.py.before_stage7f_reporting_fix')
if bak.exists():raise RuntimeError(f'Backup already exists, refusing overwrite: {bak}')
shutil.copy2(p,bak)
p.write_text(s.replace(old,new))
compile(p.read_text(),str(p),'exec')
print('PATCHED',p,'\nBACKUP',bak,'\nORIGINAL SHA256',hashlib.sha256(bak.read_bytes()).hexdigest(),'\nPATCHED SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
