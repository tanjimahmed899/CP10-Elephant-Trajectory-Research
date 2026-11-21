#!/usr/bin/env python3
"""Locate original detector training provenance. Read-only; paths/logs remain LOCAL.
Does not copy any image, checkpoint, dataset or private config into the Git repo.
"""
import argparse,csv,hashlib,json,os,re
from pathlib import Path
from collections import Counter

CANDIDATE=re.compile(r'(?i)(?:^(?:args|data|hyp|training|config|dataset|train|results|metrics|split|best|last|model)[-_.]|(?:yolo|elephant|detect|weights|checkpoint|train|val|test))')
EXT={'.yaml','.yml','.json','.csv','.txt','.py','.pt','.pth'}
SKIP={'__pycache__','.git','.venv','venv','node_modules','site-packages','dataset-main','images','labels','frames','video'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('/mnt/e/AIUB/Capstone Project/Journal Based Results'))
    ap.add_argument('--out',type=Path,default=Path.home()/'CP10_Stage12_Review')
    ap.add_argument('--hash-weights',action='store_true',help='Read large model files locally to compute SHA256; does NOT upload weights')
    args=ap.parse_args();root=args.root.expanduser().resolve();out=args.out.expanduser().resolve()
    if not root.is_dir():raise SystemExit('E drive source directory missing')
    rows=[]
    for cur,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d.lower() not in SKIP and not d.startswith('.')]
        for fname in files:
            p=Path(cur)/fname
            if p.suffix.lower() not in EXT:continue
            relative=p.relative_to(root)
            if not CANDIDATE.search(fname) and not any(re.search(r'(?i)elephant|yolo|detector|ultralytics|training',a) for a in relative.parts):continue
            size=p.stat().st_size; ext=p.suffix.lower()
            if ext in {'.pt','.pth'}:typ='CHECKPOINT_HASH_ONLY'
            elif fname.lower() in {'args.yaml','results.csv','data.yaml'}:typ='HIGH_PRIORITY_TRAINING_PROVENANCE'
            elif 'train' in fname.lower() or 'split' in fname.lower():typ='TRAINING_OR_SPLIT_CANDIDATE'
            elif ext in {'.yml','.yaml'}:typ='CONFIG_CANDIDATE'
            else:typ='OTHER_PROVENANCE_CANDIDATE'
            rows.append({'local_relative_path':str(relative),'type':typ,'size_bytes':size,
                         'sha256':sha(p) if ext not in {'.pt','.pth'} or args.hash_weights else '',
                         'release_status':'LOCAL_REVIEW_ONLY'})
    out.mkdir(parents=True,exist_ok=True)
    with (out/'STAGE12_ORIGINAL_YOLO_PROVENANCE_CANDIDATES.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['local_relative_path','type','size_bytes','sha256','release_status']);w.writeheader();w.writerows(rows)
    summary={'files_located':len(rows),'categories':dict(Counter(r['type'] for r in rows)),
        'note':'Filename search, NOT verification that original YOLO training records are present. Keep private until rights/contents examined.'}
    (out/'STAGE12_YOLO_PROVENANCE_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
