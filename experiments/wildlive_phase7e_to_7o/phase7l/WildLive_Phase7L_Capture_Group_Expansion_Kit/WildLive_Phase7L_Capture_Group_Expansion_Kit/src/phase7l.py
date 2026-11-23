#!/usr/bin/env python3
"""CP1.0 Phase 7L: observational provenance and locked-output verification.

This tool does NOT train, retune, or run new Transformer inference. It audits
stored Phase 7K results and records uncertain recording provenance separately.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, re, sys, zipfile
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
LOCK=ROOT/'protocol/PHASE7K_ORIGINAL_LOCK_DO_NOT_EDIT.json'
LOCK_SHA256='408022abb8923eb61afdf1eb665db8ba7bb02ce4258b21ae4a3fd89dab4abb5c'
FEATURE_MODELS=('stationary','cv20','cv_ols','transformer_ensemble')
CHOICES=('stationary','cv_ols','transformer_ensemble')
KEYS=('sequence','horizon_frames','track_id','segment_id','first_observed_frame')
BASE='https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69'
KNOWN_SEQUENCE_IDS=[
 'DJI_0117_video1','DJI_0117_video2','DJI_0117_video3','DJI_0117_video4',
 'DJI_0119','DJI_0133_video3','DJI_0142_video1','DJI_0161_video1',
 'DJI_0204_video1','DJI_0204_video2','DJI_0207',
 'DJI_0601_video2','DJI_0601_video3','DJI_0601_video4',
 'DJI_0601_video5','DJI_0601_video6',
 'DJI_20230719145427_0002_V_video2',
 'DJI_20230719145427_0002_V_video3',
 'DJI_20230719145427_0002_V_video4',
 'DJI_20230719145427_0002_V_video5',
 'DJI_20230719145816_0003_V_video2',
 'DJI_vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V']
ALREADY_SEEN={'DJI_0207','DJI_0204_video2','DJI_0161_video1'}
DEVELOPMENT={'DJI_0207','DJI_0204_video2'}
SAMPLE_LABELS={
 'DJI_0161_video1':'elephant; audited/evaluated',
 'DJI_0204_video1':'elephant; sampled, same 0204 family',
 'DJI_0204_video2':'elephant; development',
 'DJI_0207':'elephant; development',
 'DJI_0601_video2':'zebra; full source audit',
 'DJI_0117_video3':'zebra in sampled frames only',
 'DJI_0601_video6':'zebra in sampled frames only',
 'DJI_20230719145427_0002_V_video5':'zebra in sampled frames only',
 'DJI_0119':'zebra in sampled frame only',
 'DJI_0133_video3':'giraffe in sampled frame only'}


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def lock():
 if digest(LOCK)!=LOCK_SHA256:
  raise ValueError('Locked Phase 7K protocol copy modified — STOP')
 obj=json.loads(LOCK.read_text(encoding='utf-8'))
 if obj['primary_comparator']!='cv_ols':
  raise ValueError('Unexpected comparator — STOP')
 return obj

def family(name):
 # Convenience grouping heuristic ONLY; not verified flight-event identity.
 m=re.match(r'^(DJI_\d{4})_video\d+$',name)
 if m:return m.group(1)
 m=re.match(r'^(DJI_\d{14}_\d{4}_V)_video\d+$',name)
 if m:return m.group(1)
 return name

def make_register(output):
 lock()
 dev_family={family(x) for x in DEVELOPMENT}
 rows=[]
 for seq in KNOWN_SEQUENCE_IDS:
  same_dev=family(seq) in dev_family
  state=('already examined' if seq in ALREADY_SEEN else 'untouched by CP 1.0 as far as archived record indicates')
  rows.append(dict(sequence_id=seq,filename_family_HEURISTIC=family(seq),
   current_project_status=state,sample_label_evidence=SAMPLE_LABELS.get(seq,'unknown (not yet screened)'),
   same_filename_family_as_development=same_dev,
   candidate_priority=('STOP: development' if seq in DEVELOPMENT else
      'already evaluated; not new' if seq in ALREADY_SEEN else
      'related to development, provenance review first' if same_dev else
      'species screening first'),
   independent_capture_event_verified='NO',source_group_identity='UNKNOWN',
   official_annotation_url=f'{BASE}/{seq}/boxid/',
   note='Distinct folder or filename family is not evidence of independent flight/herd'))
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(output,index=False)
 return rows

def find_member(z,suffix):
 paths=[n for n in z.namelist() if n.endswith(suffix) and not n.startswith('__MACOSX/')]
 if len(paths)!=1:raise ValueError(f'Expected exactly one {suffix}; found {len(paths)}')
 return paths[0]

def load_archive(path,source_zip=None):
 lk=lock()
 with zipfile.ZipFile(path) as z:
  if z.testzip() is not None:raise ValueError('Corrupt ZIP member')
  manifest=json.loads(z.read(find_member(z,'evaluation_manifest.json')))
  df=pd.read_csv(io.BytesIO(z.read(find_member(z,'all_experts_and_gate_per_window.csv'))))
  summ=pd.read_csv(io.BytesIO(z.read(find_member(z,'per_track_and_pooled_descriptive.csv'))))
 if manifest.get('phase')!='7K':raise ValueError('Only original Phase 7K evaluation archives supported')
 seq=manifest['sequence_id']
 if seq in DEVELOPMENT:raise ValueError('Development sequence must not enter new-evidence report')
 for horizon in (20,50):
  k=f'phase7j_dev_gate_h{horizon}.json'
  if manifest.get('gate_sha256',{}).get(k)!=lk['release_files_sha256'][f'locked_models/{k}']:
   raise ValueError(f'Gate hash mismatch {k} — STOP')
 for fn,expected in lk['model_checkpoint_sha256'].items():
  entry=manifest.get('checkpoint_hashes_verified',{}).get(fn)
  if not isinstance(entry,dict) or entry.get('sha256')!=expected['sha256']:
   raise ValueError(f'Checkpoint provenance mismatch {fn} — STOP')
 if source_zip and digest(source_zip)!=manifest.get('file_sha256'):
  raise ValueError('Annotation source hash mismatch — STOP')
 if df.empty or df[list(KEYS)].isna().any().any() or df.duplicated(list(KEYS)).any():
  raise ValueError('Empty, null or duplicated prediction window keys — STOP')
 if not set(df['horizon_frames'])=={20,50}:raise ValueError('Expected both horizons')
 if not (df['sequence']==seq).all():raise ValueError('Mismatched sequence IDs')
 if not df['selected_expert'].isin(CHOICES).all():raise ValueError('Unknown gate choice')
 if len(df)!=int(manifest.get('n_scored_windows',-1)):raise ValueError('Unexpected count')
 for model in (*FEATURE_MODELS,'adaptive'):
  for metric in ('ade','fde'):
   col=f'{model}_{metric}_norm'
   if col not in df or not np.isfinite(df[col].to_numpy()).all() or (df[col]<0).any():
    raise ValueError(f'Invalid errors {col}')
 for metric in ('ade','fde'):
  selected=np.array([df.at[i,f'{m}_{metric}_norm'] for i,m in zip(df.index,df['selected_expert'])])
  if not np.allclose(selected,df[f'adaptive_{metric}_norm'],rtol=1e-9,atol=1e-10):
   raise ValueError(f'Future-label or adaptive-choice mismatch {metric}')
 if not np.allclose(df['adaptive_ade_norm']-df['cv_ols_ade_norm'],df['adaptive_minus_cv_ols_ADE'],rtol=1e-8,atol=1e-10):
  raise ValueError('Inconsistent ADE differences')
 rows=[]
 for h in (20,50):
  part=df[df.horizon_frames.eq(h)]
  pool=summ[(summ.horizon_frames.eq(h))&(summ.track_id.astype(str)=='pooled')]
  if len(pool)!=1:raise ValueError('Expected one pooled summary row per horizon')
  pool=pool.iloc[0]
  if int(pool['windows'])!=len(part):raise ValueError('Summary window count mismatch')
  for m in (*FEATURE_MODELS,'adaptive'):
   for metric in ('ade','fde'):
    col=f'{m}_{metric}_norm'
    if not np.isclose(part[col].mean(),pool[col],atol=1e-10,rtol=1e-8):
     raise ValueError(f'Mean recomputation mismatch: {h} {col}')
  for metric in ('ade','fde'):
   a=float(part[f'adaptive_{metric}_norm'].mean());c=float(part[f'cv_ols_{metric}_norm'].mean())
   rows.append(dict(sequence_id=seq,horizon_frames=int(h),metric=metric.upper(),
                    n_overlapping_windows=len(part),adaptive=a,cv_ols=c,
                    adaptive_minus_cv_ols=a-c,
                    relative_improvement_over_cv_ols_pct=100*(c-a)/c if c>0 else float('nan'),
                    adaptive_better_than_cv_ols=bool(a<c),
                    source_sha256=manifest['file_sha256'],
                    gate_hashes_checked=True,checkpoint_hashes_checked=True,
                    capture_event_verified=False,evidence_unit='recording folder, not confirmed independent capture'))
 return {'sequence':seq,'rows':rows,'n_windows':len(df),'source_sha256':manifest['file_sha256']}

def audit_eval(archives,output,source_zip=None):
 lock()
 if len(archives)>1 and source_zip is not None:
  raise ValueError('Single source ZIP can only be matched to one evaluation archive')
 records=[load_archive(p,source_zip) for p in archives]
 seqs=[r['sequence'] for r in records]
 hashes=[r['source_sha256'] for r in records]
 if len(set(seqs))!=len(seqs) or len(set(hashes))!=len(hashes):
  raise ValueError('Repeated sequence or annotation-source hash')
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 table=pd.DataFrame([r for record in records for r in record['rows']])
 table.to_csv(out/'phase7l_verified_recording_comparisons.csv',index=False,float_format='%.12g')
 status=dict(phase='7L',evaluated_recording_folders=len(records),
             verified_independent_capture_groups=0,
             group_independence='NOT ESTABLISHED',
             primary_comparator='CV-OLS',
             no_model_retraining=True,
             interpretation='Descriptive across selected recording folders only, not independent capture event inference.',
             sequence_ids=seqs)
 (out/'phase7l_audit_status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
 return table,status

def provenance_review(evidence_csv,output):
 # Data-entry completeness checks only; never claims independent verification.
 with Path(evidence_csv).open(newline='',encoding='utf-8') as f:
  records=list(csv.DictReader(f))
 fields=('sequence_id','original_video_or_flight_log','capture_datetime_utc_with_source',
         'camera_device_or_flight_id','overlap_review_against_development',
         'evidence_location','reviewer','review_date','reviewer_assessment')
 if records and any(k not in records[0] for k in fields):
  raise ValueError('Missing mandatory provenance template columns')
 outcome=[]
 for rec in records:
  complete=all(str(rec.get(x,'')).strip() and str(rec.get(x,'')).lower() not in ('unknown','pending','n/a')
               for x in fields[1:-1])
  reviewer_says=rec.get('reviewer_assessment','').strip().lower()
  status=('REVIEW_REQUIRES_HUMAN_SOURCE_VERIFICATION' if complete and reviewer_says=='independent' else
          'NOT_CONFIRMED_INDEPENDENT')
  outcome.append({'sequence_id':rec['sequence_id'],'evidence_fields_present':complete,
                  'reviewer_assessment':reviewer_says,'software_status':status,
                  'independent_capture_event_verified_by_software':False})
 out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(outcome).to_csv(out,index=False)
 return outcome

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 sub=p.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('register');a.add_argument('--output',type=Path,required=True)
 a=sub.add_parser('audit-eval');a.add_argument('--archive',type=Path,action='append',required=True)
 a.add_argument('--source-zip',type=Path);a.add_argument('--output',type=Path,required=True)
 a=sub.add_parser('provenance-review');a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
 a=sub.add_parser('verify-lock')
 x=p.parse_args(argv)
 if x.cmd=='verify-lock':print('Phase7K source lock copy verified:',digest(LOCK))
 elif x.cmd=='register':
  rows=make_register(x.output);print('Official root folder register rows:',len(rows))
 elif x.cmd=='audit-eval':
  df,status=audit_eval(x.archive,x.output,x.source_zip);print(df.to_string(index=False));print(json.dumps(status,indent=2))
 elif x.cmd=='provenance-review':
  r=provenance_review(x.input,x.output);print('Provenance reviewed as documentation only:',len(r))

if __name__=='__main__':main()
