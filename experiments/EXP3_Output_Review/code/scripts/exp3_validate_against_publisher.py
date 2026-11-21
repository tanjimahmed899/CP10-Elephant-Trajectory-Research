#!/usr/bin/env python3
"""Validate exact six locked files against publisher Croissant metadata (sizes + SHA256).
Only verifies downloaded bytes; does not claim test results or independence.
"""
import csv, hashlib, json, pathlib, argparse, zipfile

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 home=pathlib.Path(__file__).resolve().parent.parent
 ap=argparse.ArgumentParser()
 ap.add_argument('--data-root',type=pathlib.Path,default=home/'new_data')
 ap.add_argument('--expected',type=pathlib.Path,default=home/'data'/'EXP3_OFFICIAL_expected_SHA256.csv')
 ap.add_argument('--out',type=pathlib.Path,default=home/'reports'/'EXP3_publisher_checksum_verification.json')
 a=ap.parse_args()
 records=[]
 for r in csv.DictReader(a.expected.open(encoding='utf8')):
  dest=a.data_root/r['repo_path']
  state={'id':r['holdout_id'],'repo_path':r['repo_path'],'exists':dest.is_file(), 'expected_sha256':r['expected_sha256']}
  if dest.is_file():
   state['actual_size_bytes']=dest.stat().st_size
   state['size_match']=dest.stat().st_size==int(r['expected_size_bytes'])
   state['actual_sha256']=sha(dest)
   state['hash_match']=state['actual_sha256']==r['expected_sha256']
   state['is_zip']=zipfile.is_zipfile(dest)
  else:state['size_match']=False;state['hash_match']=False;state['is_zip']=False
  state['pass']=state['exists'] and state['size_match'] and state['hash_match'] and state['is_zip']
  records.append(state)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 summary={'pass_count':sum(x['pass'] for x in records),'expected_count':len(records),'integrity_pass':all(x['pass'] for x in records),'records':records}
 a.out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
 for r in records: print(('PASS' if r['pass'] else 'MISSING/FAIL'),r['repo_path'])
 print(summary['pass_count'],'/',len(records),'official publisher hashes verified. Report:',a.out)
 return 0 if summary['integrity_pass'] else 2
if __name__=='__main__':raise SystemExit(main())
