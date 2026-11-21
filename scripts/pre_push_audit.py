#!/usr/bin/env python3
"""Preflight before committing/pushing a private research archive. Read-only."""
from pathlib import Path
import re,sys
root=Path(__file__).resolve().parents[1]
ignore_dirs={'.git','.venv','venv','__pycache__','node_modules','.ipynb_checkpoints'}
secret_re=[re.compile(x,re.I) for x in (r'ghp_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{30,}',r'AKIA[0-9A-Z]{16}',r'-----BEGIN [A-Z ]*PRIVATE KEY-----',r'sk-[A-Za-z0-9]{25,}')]
fail=[];warnings=[];count=0;total=0
for p in root.rglob('*'):
 if not p.is_file() or any(part in ignore_dirs for part in p.relative_to(root).parts):continue
 rel=p.relative_to(root).as_posix();size=p.stat().st_size; count+=1;total+=size
 if size>49*1024*1024:fail.append(f'>49 MiB: {rel}')
 if p.suffix.lower() in {'.pt','.pth','.ckpt','.mp4','.mov','.avi','.key','.pem'}:fail.append(f'Forbidden sensitive/binary type: {rel}')
 if p.name.startswith('.env'):fail.append(f'Environment secret file: {rel}')
 if p.suffix.lower() in {'.py','.sh','.md','.txt','.json','.csv','.ps1','.yaml','.yml'} and size<25_000_000:
  data=p.read_text('utf-8','replace')
  for pat in secret_re:
   if pat.search(data):fail.append(f'Possible secret pattern: {rel}');break
print(f'Checked {count} files, {total/1024**2:.1f} MiB.')
if warnings:print('Warnings:',*warnings,sep='\n- ')
if fail:
 print('BLOCKED: manual review required:',*fail,sep='\n- ')
 sys.exit(2)
print('Automated file-size and high-confidence secret checks PASS. This is NOT a substitute for license/location/privacy review.')
