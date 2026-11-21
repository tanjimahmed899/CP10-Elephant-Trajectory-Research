#!/usr/bin/env python3
"""CP10 Stage 10: local, non-destructive research-evidence inventory and cautious copy.

Default: inventory ONLY into ~/CP10_Evidence_Review (outside the Git repository).
--copy: stage safe, bounded evidence copies OUTSIDE the Git repository. Never push.
--include-figures: stage plots/images for MANUAL copyright/privacy review.
--include-arrays: stage small NPZ/NPY arrays for MANUAL privacy/size review.
--extract-archive-evidence: inspect and, with --copy, stage selected files inside CP10 ZIPs.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,io,os,re,sys,zipfile,json
from pathlib import Path, PurePosixPath
from datetime import datetime,timezone

DEFAULT_ROOT='/mnt/e/AIUB/Capstone Project/Journal Based Results'
DEFAULT_DOWNLOADS='/mnt/c/Users/HP/Downloads'
CODE={'.py','.ipynb','.sh','.ps1','.r','.m','.toml','.ini','.yaml','.yml','.cfg'}
METRICS={'.csv','.tsv','.json','.jsonl','.md','.txt','.log'}
ARRAYS={'.npz','.npy','.parquet','.feather'}
FIGURES={'.png','.jpg','.jpeg','.webp','.svg','.pdf','.docx'}
BANNED={'.pt','.pth','.ckpt','.onnx','.engine','.mp4','.mov','.avi','.mkv','.tif','.tiff','.jp2','.sqlite','.db','.pkl','.pickle','.zip','.7z','.rar','.tar','.gz','.pem','.key','.env'}
PRUNE={'__pycache__','.git','.venv','venv','env','node_modules','.cache','site-packages','.ipynb_checkpoints','lost+found','train2017','val2017','test2017','images','imgs','frames','jpegimages','raw_images','raw_videos','masks','labels','videos','source_videos'}
PRIVATE_DIR={'private_metadata','secrets','credentials','personal','raw_gps','source_video','source_videos'}
PRIVATE_RE=re.compile(r'(?i)(?:^|[_\-\.])(password|secret|token|credential|private[_-]?key|access[_-]?key|gps|geolocation|latitude|longitude|geocoord|exact[_-]?location)(?:[_\-\.]|$)')
LOCATION_RE=re.compile(r'(?i)(?:^|[_\-.])(camera[s]?|poses?|intrinsics?|extrinsics?|tracking_summary|georef|georeference|kitti|utm|world_coordinates)(?:[_\-.]|$)')
VISUAL_RE=re.compile(r'(?i)(figure|fig[_\d-]|plot|chart|curve|graph|histogram|ablation|heatmap|report|experimental|research|paper|results|summary)')
SECRETS=[re.compile(s,re.I) for s in [r'ghp_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{30,}',r'AKIA[0-9A-Z]{16}',r'-----BEGIN [A-Z ]*PRIVATE KEY-----',r'sk-[A-Za-z0-9]{25,}',r'(?<!\w)password\s*[=:]\s*["\']?\S{6,}',r'(?:auth|access)[_-]?token\s*[=:]\s*["\']?\S{12,}']]
MAX_EXCLUDED_SAMPLES=1500


def digest_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def digest_bytes(b):return hashlib.sha256(b).hexdigest()

def safe_name(name):
 s=name.replace('\\','/')
 parts=PurePosixPath(s).parts
 if not parts or s.startswith('/') or any(p in ('..','.') for p in parts) or ':' in parts[0]:return None
 return '/'.join(parts)

def classify(rel,size,include_figures=False,include_arrays=False,max_bytes=12*1024*1024):
 p=PurePosixPath(rel); suffix=p.suffix.lower(); name=p.name; parts=[x.lower() for x in p.parts]
 if size > max_bytes:return 'EXCLUDE_TOO_LARGE'
 if any(x in PRIVATE_DIR for x in parts):return 'EXCLUDE_PRIVATE_DIRECTORY'
 if suffix in BANNED:return 'EXCLUDE_BINARY_OR_MODEL_OR_VIDEO'
 if name.lower().startswith('.env') or PRIVATE_RE.search(name):return 'EXCLUDE_POSSIBLE_SECRET'
 if LOCATION_RE.search(name) and suffix not in CODE:return 'EXCLUDE_CAMERA_LOCATION_METADATA'
 if suffix in CODE:return 'CANDIDATE_CODE'
 if suffix in METRICS:
  if any(x in {'train2017','val2017','test2017','frames','labels','images'} for x in parts):return 'EXCLUDE_RAW_LABELS'
  return 'CANDIDATE_METRIC_OR_REPORT'
 if suffix in ARRAYS:return 'CANDIDATE_ARRAY_REVIEW' if include_arrays else 'REVIEW_ARRAY_NOT_DEFAULT'
 if suffix in FIGURES:
  if not VISUAL_RE.search(name):return 'EXCLUDE_UNRELATED_VISUAL'
  return 'CANDIDATE_VISUAL_REVIEW' if include_figures else 'REVIEW_FIGURE_NOT_DEFAULT'
 return 'EXCLUDE_TYPE'

def secret_scan_file(p):
 if p.stat().st_size > 2_000_000:return False
 try:
  with open(p,'rb') as f: b=f.read(2_000_000)
  if b'\0' in b[:1024]:return False
  t=b.decode('utf8','replace')
  return any(x.search(t) for x in SECRETS)
 except OSError:return True

def secret_scan_bytes(b):
 if len(b)>2_000_000 or b'\0' in b[:1024]:return False
 t=b.decode('utf8','replace')
 return any(x.search(t) for x in SECRETS)

def repo_hashes(repo):
 """Deduplicate against existing tracked-like files by digest (skips bulky files)."""
 hashes=set()
 if not repo.is_dir():return hashes
 for base,dirs,files in os.walk(repo):
  dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','.venv','venv','node_modules'}]
  for n in files:
   p=Path(base)/n
   try:
    if p.stat().st_size <= 12*1024*1024 and p.suffix.lower() in CODE|METRICS|ARRAYS|FIGURES:
     hashes.add(digest_file(p))
   except OSError:pass
 return hashes

def scan_files(root,tag,cb,totals):
 if not root.is_dir():return
 for base,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[d for d in dirs if d.lower() not in PRUNE and not d.startswith('.') and not (Path(base)/d).is_symlink()]
  for n in files:
   f=Path(base)/n
   if f.is_symlink():continue
   try: size=f.stat().st_size
   except OSError:continue
   rel=f.relative_to(root).as_posix()
   totals['visited_files']+=1
   cb(tag,f,rel,size)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',default=DEFAULT_ROOT)
 ap.add_argument('--downloads',default=DEFAULT_DOWNLOADS)
 ap.add_argument('--repo',default=str(Path.home()/'cp10_github'/'CP10-Elephant-Trajectory-Research'))
 ap.add_argument('--out',default=str(Path.home()/'CP10_Evidence_Review'))
 ap.add_argument('--copy',action='store_true',help='copy vetted candidates to OUTSIDE-repo staging; never push')
 ap.add_argument('--include-figures',action='store_true',help='include plots/figures; manual redistribution review needed')
 ap.add_argument('--include-arrays',action='store_true',help='include bounded numerical NPZ/NPY evidence; manually inspect')
 ap.add_argument('--extract-archive-evidence',action='store_true',help='scan CP10/EXP3 ZIP contents in Downloads, and stage eligible members if --copy')
 ap.add_argument('--max-mb',type=float,default=12.0)
 ap.add_argument('--max-copies',type=int,default=15000)
 a=ap.parse_args()
 root=Path(a.root);downloads=Path(a.downloads);repo=Path(a.repo);out=Path(a.out).expanduser().resolve();max_bytes=int(a.max_mb*1024*1024)
 if not root.is_dir():raise SystemExit('E: drive root unavailable: '+str(root))
 if out == repo.resolve() or repo.resolve() in out.parents:raise SystemExit('ERROR: staging MUST remain outside git repo')
 out.mkdir(parents=True,exist_ok=True)
 candidates=open(out/'CANDIDATES_REVIEW.csv','w',encoding='utf-8',newline=''); excluded=open(out/'EXCLUDED_SUMMARY_SAMPLE.csv','w',encoding='utf-8',newline='')
 fields=['origin','relative_path','bytes','category','sha256','duplicate_in_repo','staged_path','note']
 cw=csv.DictWriter(candidates,fieldnames=fields);cw.writeheader();ew=csv.DictWriter(excluded,fieldnames=fields);ew.writeheader()
 existing=repo_hashes(repo)
 totals=collections.Counter(); categories=collections.Counter(); origins=collections.Counter(); generated=[];copy_bytes=0
 print('Repo hashes loaded:',len(existing),'Read-only E drive:',root,'\nStaging:',out,'\n',flush=True)
 def record(origin,rel,size,category,sha='',duplicate='',staged='',note=''):
  item=dict(origin=origin,relative_path=rel,bytes=size,category=category,sha256=sha,duplicate_in_repo=duplicate,staged_path=staged,note=note)
  categories[category]+=1;origins[origin]+=1
  if category.startswith('CANDIDATE') or category=='DUPLICATE_EXISTING':cw.writerow(item)
  elif totals['excluded_sample']<MAX_EXCLUDED_SAMPLES:ew.writerow(item);totals['excluded_sample']+=1
 def maybe_copy(origin,p,rel,size,z=None,member=None):
  nonlocal copy_bytes
  category=classify(rel,size,a.include_figures,a.include_arrays,max_bytes)
  if not category.startswith('CANDIDATE'):
   record(origin,rel,size,category);return
  try:
   if z is None:
    if secret_scan_file(p):record(origin,rel,size,'EXCLUDE_POSSIBLE_SECRET_CONTENT',note='manual check');return
    sha=digest_file(p)
   else:
    # ZIP members are only extracted if copy; inventory uses no decompression/hashes.
    if not a.copy:
     record(origin,rel,size,category,note='archive member not yet hashed');return
    blob=z.read(member)
    if len(blob)!=size:record(origin,rel,size,'EXCLUDE_ZIP_SIZE_MISMATCH');return
    if secret_scan_bytes(blob):record(origin,rel,size,'EXCLUDE_POSSIBLE_SECRET_CONTENT',note='manual check');return
    sha=digest_bytes(blob)
   if sha in existing:
    record(origin,rel,size,'DUPLICATE_EXISTING',sha,'yes',note='same SHA-256 already in repo or previously staged');return
   dst='staged_evidence/'+origin+'/'+rel
   if a.copy:
    if totals['copied']>=a.max_copies:record(origin,rel,size,'EXCLUDE_COPY_LIMIT');return
    target=out/dst
    target.parent.mkdir(parents=True,exist_ok=True)
    if z is None:
     with p.open('rb') as src,target.open('wb') as dest:
      for chunk in iter(lambda:src.read(1024*1024),b''):dest.write(chunk)
    else:target.write_bytes(blob)
    if digest_file(target)!=sha:raise RuntimeError('Copy checksum mismatch: '+str(target))
    totals['copied']+=1;copy_bytes+=size;existing.add(sha)
    record(origin,rel,size,category,sha,'no',dst,'staged only; MUST review before git')
   else:record(origin,rel,size,category,sha,'no','','eligible; rerun with --copy after review')
  except (OSError,zipfile.BadZipFile,RuntimeError,EOFError) as e:
   record(origin,rel,size,'EXCLUDE_READ_ERROR',note=str(e)[:100])
 def local_zip(p,rel,size):
  totals['local_archives_seen']+=1
  # This is optional: never automatically extract massive dataset ZIPs.
  if not a.extract_archive_evidence or size > 2*1024**3:return
  n=p.name.lower()
  if not any(term in n for term in ('all python code','main result','exp3','cp10','research','zip 4.0','copy1','output_review')):return
  try:
   with zipfile.ZipFile(p) as z:
    if len(z.infolist()) > 100_000:
     totals['oversized_member_lists']+=1;return
    for m in z.infolist():
     if m.is_dir() or m.file_size>max_bytes:continue
     if ((m.external_attr>>16)&0o170000)==0o120000:continue
     member_path=safe_name(m.filename)
     if member_path is None:totals['unsafe_archive_members']+=1;continue
     if any(x.lower() in PRUNE|PRIVATE_DIR for x in PurePosixPath(member_path).parts):continue
     maybe_copy('E_DRIVE_ZIPS/'+p.stem,p,member_path,m.file_size,z,m)
  except (zipfile.BadZipFile,OSError,EOFError) as e:
   totals['broken_local_zips']+=1;print('Archive warning',p.name,str(e)[:100])
 def file_callback(tag,p,rel,size):
  if p.suffix.lower()=='.zip':
   if tag=='E_DRIVE':local_zip(p,rel,size)
   return
  if p.suffix.lower() in {'.7z','.rar','.tar','.gz'}:return
  maybe_copy(tag,p,rel,size)
 scan_files(root,'E_DRIVE',file_callback,totals)
 # Avoid scanning user's entire Downloads: only relevant reports and stage tools, ZIP list.
 if downloads.is_dir():
  for p in sorted(downloads.iterdir()):
   if not p.is_file() or p.is_symlink():continue
   low=p.name.lower()
   if not (low.startswith('cp10_') or low.startswith('exp3_') or 'experimental' in low or 'demo paper' in low):continue
   if p.suffix.lower()=='.zip':
    totals['zip_archives_seen']+=1
    if not a.extract_archive_evidence:continue
    try:
     with zipfile.ZipFile(p) as z:
      for m in z.infolist():
       if m.is_dir() or m.file_size>max_bytes:continue
       # skip symlinks and suspicious archive paths
       if ((m.external_attr>>16)&0o170000)==0o120000:continue
       member_path=safe_name(m.filename)
       if member_path is None:totals['unsafe_archive_members']+=1;continue
       if any(x.lower() in PRUNE|PRIVATE_DIR for x in PurePosixPath(member_path).parts):continue
       maybe_copy('DOWNLOAD_ZIPS/'+p.stem,p,member_path,m.file_size,z,m)
    except (zipfile.BadZipFile, OSError,EOFError) as e:totals['broken_zips']+=1;print('Archive warning',p.name,str(e)[:100])
   else:
    try:maybe_copy('DOWNLOAD_FILES',p,p.name,p.stat().st_size)
    except OSError:pass
 candidates.close();excluded.close()
 # Summaries written outside repository and include no source location-sensitive raw rows beyond names.
 summary={'created_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'downloads':str(downloads),'repo':str(repo),'mode':'COPY_TO_EXTERNAL_STAGING_ONLY' if a.copy else 'INVENTORY_ONLY','options':{'include_figures':a.include_figures,'include_arrays':a.include_arrays,'extract_archive_evidence':a.extract_archive_evidence,'max_mb':a.max_mb},'totals':dict(totals),'categories':dict(sorted(categories.items())),'origins':dict(origins),'copied_mb':round(copy_bytes/1024**2,2)}
 (out/'EVIDENCE_INVENTORY_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 with (out/'REVIEW_BEFORE_GITHUB.md').open('w',encoding='utf8') as f:
  f.write('# CP10 local research evidence: staging review\n\n')
  f.write('No file has been pushed to GitHub. This report is local-only.\n\n')
  f.write('Root: '+str(root)+'\n\n')
  for k,v in categories.most_common():f.write(f'- {k}: {v}\n')
  f.write(f'\nNew copies staged outside repo: {totals["copied"]}, {copy_bytes/1024**2:.2f} MiB.\n\n')
  f.write('## Review instructions\n\n1. Inspect CANDIDATES_REVIEW.csv.\n2. Do NOT commit EXCLUDED_SUMMARY_SAMPLE.csv or this local review folder, since paths can contain private metadata.\n3. Check source license, copied code author ownership, CSV fields (GPS/person/location), figure copyrights, accidental credentials, and video frames. Automated checks are not perfect.\n4. If safe, copy ONLY approved subdirectories from staged_evidence/ into the research repo under additional_wsl_evidence/.\n5. Run repository pre_push_audit.py and inspect `git status --short` plus `git diff --cached --stat` before any push.\n6. Missing checkpoint weights or restricted video files should be listed in a local manifest by filename/hash, not publicly committed.\n\n')
  f.write('Source code historical variants are provenance evidence. Prefer labeled FINAL/FROZEN scripts for official reproduction; preserve test group separation.\n')
 print('\nCOMPLETE:',json.dumps({'categories':dict(categories),'staged':totals['copied'],'staged_MB':round(copy_bytes/1024**2,2)},indent=2))
 print('OPEN:',out/'REVIEW_BEFORE_GITHUB.md')

if __name__=='__main__':main()
