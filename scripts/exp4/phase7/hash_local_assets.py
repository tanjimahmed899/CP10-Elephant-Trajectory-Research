#!/usr/bin/env python3
"""Local-source hash inventory; no remote access; no frame decoding; no outcomes."""
from pathlib import Path
import argparse,csv,hashlib

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('source_dir');a.add_argument('--out',required=True)
    args=a.parse_args()
    source=Path(args.source_dir).resolve()
    if not source.is_dir():raise SystemExit('source_dir not found')
    out=Path(args.out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['relative_path','size_bytes','sha256'])
        n=0
        for p in sorted(source.rglob('*')):
            if not p.is_file() or p.resolve()==out:continue
            w.writerow([str(p.relative_to(source)),p.stat().st_size,sha256(p)]);n+=1
    print(f'Hashed {n} local file(s). This does NOT prove independent capture-group provenance.')
if __name__=='__main__':main()
