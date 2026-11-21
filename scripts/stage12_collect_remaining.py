#!/usr/bin/env python3
"""Inventory-backed, hash-verified local-only staging for candidate aggregate research evidence.
DOES NOT copy into repo, commit, push, license, or authorize publishing.
"""
import argparse,csv,hashlib,json,re,shutil,subprocess
from pathlib import Path
from collections import Counter

PAT=re.compile(r'(?i)(overall|summary|aggregate|model_comparison|metric|coverage|bootstrap|result|readout|frozen|protocol|report|manifest|claim|decision|seed_level)')
RESTRICT=re.compile(r'(?i)(/Dataset-main/|/flight_metadata/|/source_videos/|/raw/|/bambi_downloads/|/private_metadata/|/wildlive_api_discovery/|camera[_-]?pose|extrinsics|world[_-]?(?:track|trajectory|coordinate)|gps|georefer|geolocat|latitude|longitude|metadata.*location)')
SENSITIVE_HEADERS={'gps','lat','lon','latitude','longitude','lat_deg','lon_deg','world_x','world_y','utm_x','utm_y','easting','northing'}
TOKEN_PAT=[re.compile(x,re.I) for x in [rb'github_pat_[a-z0-9_]{30,}',rb'ghp_[a-z0-9]{25,}',rb'AKIA[0-9A-Z]{16}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']]


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def repo_hashes(repo):
    h=set()
    mf=repo/'manifests/repository_file_sha256.csv'
    if mf.is_file():
        for r in csv.DictReader(mf.open(encoding='utf-8-sig')):
            if r.get('sha256'):h.add(r['sha256'].lower())
    if (repo/'.git').is_dir():
        a=subprocess.run(['git','-C',str(repo),'ls-files','-z'],capture_output=True,check=True).stdout
        for bs in a.split(b'\0'):
            if bs:
                p=repo/bs.decode('utf-8','surrogateescape')
                if p.is_file():h.add(digest(p))
    return h

def run(args):
    repo=args.repo.expanduser().resolve();root=args.root.expanduser().resolve();out=args.out.expanduser().resolve()
    if not root.is_dir() or not repo.is_dir():raise SystemExit('Missing source root or repository')
    if out==repo or repo in out.parents or out==root or root in out.parents:
        raise SystemExit('Output must be OUTSIDE repository and source root')
    inventory=args.inventory.expanduser().resolve()
    if not inventory.is_file():raise SystemExit(f'Missing inventory: {inventory}')
    known=repo_hashes(repo);seen=set();data=[]
    for r in csv.DictReader(inventory.open(encoding='utf-8-sig',newline='')):
        if r.get('origin')!='E_DRIVE' or r.get('category')!='CANDIDATE_METRIC_OR_REPORT':continue
        rel=r.get('relative_path','');sha=r.get('sha256','').lower();cat=r.get('category','')
        if not sha or not re.fullmatch('[0-9a-f]{64}',sha):continue
        path=(root/rel).resolve(); size=int(r.get('bytes') or 0)
        state='REVIEW_ONLY'; reason='Not automatically chosen as a small aggregate'
        if sha in known:state='IN_REPOSITORY';reason='SHA256 already included'
        elif sha in seen:state='DUPLICATE_CANDIDATE';reason='Identical content already listed'
        elif path==root or root not in path.parents:state='BLOCKED_PATH';reason='Unsafe source path'
        elif RESTRICT.search('/'+rel+'/'):state='HOLD_SENSITIVE_SOURCE';reason='Potentially sensitive source/location path'
        elif path.suffix.lower() not in {'.csv','.json','.md','.txt','.yml','.yaml'}:state='REVIEW_UNSUPPORTED';reason='Non-text or non-aggregate format'
        elif size>2*1024*1024:state='REVIEW_LARGE';reason='Exceeds 2MiB summary policy'
        elif PAT.search(Path(rel).name):state='ELIGIBLE_LOCAL_STAGE';reason='Summary filename only; content still requires review'
        data.append({'relative_path':rel,'sha256':sha,'bytes':size,'decision':state,'reason':reason})
        seen.add(sha)
    if args.stage_aggregates:
        for r in data:
            if r['decision']!='ELIGIBLE_LOCAL_STAGE':continue
            src=(root/r['relative_path']).resolve()
            if not src.is_file():r['decision']='BLOCKED_MISSING';r['reason']='Missing source';continue
            if digest(src)!=r['sha256']:r['decision']='BLOCKED_CHANGED';r['reason']='SHA256 mismatch';continue
            content=src.read_bytes()
            if b'\x00' in content[:65536] or any(p.search(content) for p in TOKEN_PAT):
                r['decision']='BLOCKED_CONTENT';r['reason']='Binary/possible credential';continue
            if src.suffix.lower()=='.csv':
                header=content.splitlines()[0].decode('utf-8','replace').lower() if content.splitlines() else ''
                try: fields={x.strip().strip('"\'').replace(' ','_') for x in next(csv.reader([header]))}
                except:fields=set()
                if fields&SENSITIVE_HEADERS:
                    r['decision']='HOLD_GEOLOCATION_COLUMNS';r['reason']='CSV has potentially sensitive coordinates';continue
            dest=out/'staged_aggregates'/r['relative_path']
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and digest(dest)!=r['sha256']:
                r['decision']='BLOCKED_COLLISION';r['reason']='Existing destination has different hash';continue
            shutil.copy2(src,dest)
            r['decision']='STAGED_OUTSIDE_REPO';r['reason']='Hash verified; still requires rights/privacy review before public Git'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'STAGE12_REMAINING_EVIDENCE_DECISIONS.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['relative_path','sha256','bytes','decision','reason']);w.writeheader();w.writerows(data)
    counters=dict(Counter(r['decision'] for r in data))
    result={'mode':'LOCAL_STAGE' if args.stage_aggregates else 'INVENTORY_ONLY','source':'E_DRIVE candidate metrics/reports only','candidates':len(data),'decisions':counters,'note':'Not approved for public release. No git commands executed. No originals modified.'}
    (out/'STAGE12_LOCAL_EVIDENCE_SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/mnt/e/AIUB/Capstone Project/Journal Based Results'))
    p.add_argument('--inventory',type=Path,default=Path.home()/'CP10_Evidence_Review/CANDIDATES_REVIEW.csv')
    p.add_argument('--out',type=Path,default=Path.home()/'CP10_Stage12_Review')
    p.add_argument('--stage-aggregates',action='store_true',help='Copy safe-looking aggregate candidates OUTSIDE repository after hash/header checks')
    run(p.parse_args())
