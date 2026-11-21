#!/usr/bin/env python3
"""Read-only public repo audit. Writes safe summaries OUTSIDE the Git repository.
Heuristic checks: not a guarantee of privacy, license compliance, or security.
"""
import argparse, csv, hashlib, json, re, subprocess
from collections import Counter
from pathlib import Path

CREDENTIALS=[
  ('GitHub token',re.compile(rb'(?:ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}')),
  ('AWS key',re.compile(rb'AKIA[0-9A-Z]{16}')),
  ('private key',re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
  ('OpenAI-like key',re.compile(rb'sk-[A-Za-z0-9]{30,}')),
  ('literal secret assignment',re.compile(rb'(?im)^\s*(?:API_KEY|ACCESS_TOKEN|PASSWORD|CLIENT_SECRET)\s*=\s*[\x22\x27][^\x22\x27\n]{10,}[\x22\x27]')),
]
HEADERS={'gps','latitude','longitude','lat_deg','lon_deg','easting','northing','utm_x','utm_y','world_x','world_y','geolocation','gps_lat','gps_lon'}
RISK_PATH=re.compile(r'(?i)(?:gps|latitude|longitude|geo(?:reference|location)|camera_?(?:pose|extrinsic)|flight_?metadata|world_?(?:trajectory|coordinate)|precise_?location|cameras\.json)')
EXTERNAL_PATH=re.compile(rb'(?i)/(?:mnt/[ce]/|home/student/)|[A-Za-z]:\\(?:Users|AIUB)')
DOCUMENT_EXT={'.pdf','.docx','.doc','.pptx','.xlsx'}
UNINSPECTED_EXT={'.npz','.npy','.pt','.pth','.ckpt','.onnx','.zip','.mp4','.avi','.mov'}
TEXT_EXT={'.py','.sh','.ps1','.json','.csv','.txt','.md','.yaml','.yml','.toml','.ini','.xml','.gitattributes','.gitignore'}

def repo_paths(repo):
  if (repo/'.git').is_dir():
    p=subprocess.run(['git','-C',str(repo),'ls-files','-z'],capture_output=True,check=True)
    return [Path(x.decode('utf-8','surrogateescape')) for x in p.stdout.split(b'\0') if x]
  return [p.relative_to(repo) for p in repo.rglob('*') if p.is_file() and '.git' not in p.parts]

def audit(repo,out):
  rows=[]; maxsize=0
  for rel in repo_paths(repo):
    path=repo/rel
    if not path.is_file():continue
    size=path.stat().st_size;maxsize=max(size,maxsize)
    flags=[];ext=path.suffix.lower();raw=None
    if size>100*1024**2:flags.append('GIT_SIZE_GT_100M')
    if ext in DOCUMENT_EXT:flags.append('DOCUMENT_MANUAL_RIGHTS_AND_PERSONAL_DATA_REVIEW')
    if ext in UNINSPECTED_EXT:flags.append('BINARY_CONTENT_NOT_INSPECTED')
    if RISK_PATH.search(str(rel)):flags.append('POSSIBLE_LOCATION_METADATA_FILENAME')
    if ext in TEXT_EXT and size<=20*1024**2:
      raw=path.read_bytes()
      for name,pat in CREDENTIALS:
        if pat.search(raw):flags.append('POTENTIAL_SECRET_'+name.replace(' ','_').upper())
      if ext=='.csv':
        head=raw.splitlines()[0].decode('utf-8','replace') if raw.splitlines() else ''
        try:
          columns=[x.strip().strip('"\'').lower().replace(' ','_') for x in next(csv.reader([head]))]
        except Exception:columns=[]
        if set(columns)&HEADERS:flags.append('POSSIBLE_GEOLOCATION_CSV_FIELD')
      if EXTERNAL_PATH.search(raw):flags.append('LOCAL_ABSOLUTE_PATH_IN_TEXT')
    elif ext in TEXT_EXT:flags.append('LARGE_TEXT_NOT_CONTENT_SCANNED')
    rows.append({'path':str(rel).replace('\\','/'),'size_bytes':size,'findings':';'.join(sorted(set(flags))),
                 'needs_review':bool(flags)})
  counts=Counter(flag for r in rows for flag in r['findings'].split(';') if flag)
  current_readme=(repo/'README.md').read_text(encoding='utf-8',errors='replace') if (repo/'README.md').exists() else ''
  current_license=(repo/'LICENSE-STATUS.md').read_text(encoding='utf-8',errors='replace') if (repo/'LICENSE-STATUS.md').exists() else ''
  doc_findings=[]
  if 'intended to be **private' in current_readme:doc_findings.append('OUTDATED_README_PRIVATE_STATUS')
  if 'Keep this repository private' in current_license:doc_findings.append('OUTDATED_LICENSE_PRIVATE_STATUS')
  if 'No code license has been selected' in current_license or 'No general open-source license has been selected' in current_license:doc_findings.append('NO_OPEN_SOURCE_LICENSE_SELECTED')
  summary={'files_inspected':len(rows),'total_bytes':sum(x['size_bytes'] for x in rows),
     'flagged_files':sum(x['needs_review'] for x in rows), 'max_file_size_bytes':maxsize,
     'finding_counts':dict(counts),'documentation_findings':doc_findings,
     'potential_secret_file_count':sum('POTENTIAL_SECRET_' in x['findings'] for x in rows),
     'audit_note':'Read-only heuristic. No automated finding proves a leak; no clean result proves safe public redistribution.'}
  out.mkdir(parents=True,exist_ok=True)
  (out/'STAGE12_PUBLIC_RELEASE_AUDIT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
  with (out/'STAGE12_MANUAL_FILE_REVIEW.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','size_bytes','findings','needs_review']);w.writeheader();w.writerows([x for x in rows if x['needs_review']])
  print(json.dumps(summary,indent=2))
  return summary

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,default=Path.home()/'CP10_Stage12_Review');args=p.parse_args()
  audit(args.repo.expanduser().resolve(),args.out.expanduser().resolve())
