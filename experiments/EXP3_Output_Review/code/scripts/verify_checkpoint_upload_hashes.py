#!/usr/bin/env python3
"""Verify byte identity of six uploaded checkpoints against the upload manifest.
CAUTION: this does not verify historical training-time hashes.
"""
import argparse,csv,hashlib,pathlib

def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser();here=pathlib.Path(__file__).resolve().parent.parent
    a.add_argument('--model-dir',type=pathlib.Path,default=here/'checkpoints');a.add_argument('--manifest',type=pathlib.Path,default=here/'data'/'EXP3_uploaded_checkpoint_hashes.csv')
    q=a.parse_args()
    rows=list(csv.DictReader(q.manifest.open(encoding='utf-8-sig')))
    if len(rows)!=6: raise SystemExit('Expected 6 checkpoint hashes')
    print('This only verifies current uploaded checkpoint byte identity, NOT historic provenance')
    ok=0
    for r in rows:
        fname=r.get('filename','')
        cp=q.model_dir/fname
        sha=r.get('sha256','')
        passed=cp.is_file() and hashfile(cp)==sha
        print('PASS' if passed else 'FAIL',fname)
        ok+=int(passed)
    print(f'{ok}/6 matched')
    if ok!=6:raise SystemExit(2)

if __name__=='__main__':main()
