#!/usr/bin/env python3
"""Acquire six predetermined WildBox archives, pinning one Hugging Face revision.

Does not inspect, reselect, or tune against footage. Runs on a PC with Internet.
No files are downloaded without --download. Result receipts document the
repository revision, SHA-256 of archive bytes and ZIP integrity.
"""
import argparse, csv, hashlib, json, pathlib, zipfile
REPO='wildbox-anon-2026/wildbox-review'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    here=pathlib.Path(__file__).resolve().parent.parent
    p.add_argument('--lock',type=pathlib.Path,default=here/'data'/'EXP3_new_video_holdout_LOCKED.csv')
    p.add_argument('--data-root',type=pathlib.Path,default=here/'new_data')
    p.add_argument('--revision',help='Optional fixed full Hugging Face repo commit hash')
    p.add_argument('--download',action='store_true')
    p.add_argument('--full-crc',action='store_true')
    a=p.parse_args()
    lock=list(csv.DictReader(a.lock.open(encoding='utf-8-sig')))
    assert len(lock)==6 and len({r['repo_path'] for r in lock})==6
    assert {r['elephant_group'] for r in lock}=={'elep1','elep2','elep3'}
    if a.download:
        from huggingface_hub import HfApi,hf_hub_download
        revision=a.revision or HfApi().repo_info(repo_id=REPO,repo_type='dataset').sha
    else: revision=a.revision or 'NOT_RESOLVED_OFFLINE'
    a.data_root.mkdir(parents=True,exist_ok=True)
    receipt=[]
    for row in lock:
        rel=pathlib.PurePosixPath(row['repo_path'])
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts)!=2 or rel.parts[0] not in {'elep1','elep2','elep3'}: raise ValueError('Unsafe path')
        path=a.data_root.joinpath(*rel.parts)
        status='MISSING'
        if not path.exists() and a.download:
            got=hf_hub_download(repo_id=REPO,repo_type='dataset',filename=rel.as_posix(),revision=revision,local_dir=str(a.data_root))
            if pathlib.Path(got).resolve()!=path.resolve():raise RuntimeError('Unexpected download path')
        if path.is_file():
            status='ZIP_DIRECTORY_PASS' if zipfile.is_zipfile(path) else 'INVALID_ZIP'
            if a.full_crc and status=='ZIP_DIRECTORY_PASS':
                with zipfile.ZipFile(path) as z:
                    first=z.testzip()
                status='FULL_ZIP_CRC_PASS' if first is None else f'ZIP_CRC_FAIL:{first}'
        receipt.append({**row,'repo_id':REPO,'download_revision':revision,'archive_bytes':path.stat().st_size if path.is_file() else '',
                        'archive_sha256':sha256(path) if path.is_file() and zipfile.is_zipfile(path) else '', 'zip_status':status})
        print(rel.as_posix(),status,flush=True)
    out=a.data_root/'EXP3_SOURCE_DOWNLOAD_RECEIPT.json'
    out.write_text(json.dumps({'repo':REPO,'revision_used':revision,'records':receipt,'new_metrics_produced':False},indent=2)+'\n',encoding='utf8')
    good=sum(x['zip_status'] in {'ZIP_DIRECTORY_PASS','FULL_ZIP_CRC_PASS'} for x in receipt)
    print(f'{good}/6 archives available; receipt: {out}')
    if a.download and good!=6:raise SystemExit(2)

if __name__=='__main__':main()
