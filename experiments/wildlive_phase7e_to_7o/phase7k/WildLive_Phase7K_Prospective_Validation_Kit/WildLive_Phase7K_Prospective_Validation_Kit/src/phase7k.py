#!/usr/bin/env python3
"""Phase 7K lock/check/evaluate: frame-index-only validation candidate.

NEVER treat a video filename as a verified capture group.
All actual candidate results remain EXPLORATORY until independent provenance documented.
The two inspected development sequences are NEVER new confirmation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, sys, urllib.error, urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from phase7h import inventory, evaluate, make_windows, write_csv, OBS, TASKS, ALL_MODELS
from phase7i_features import motion_features
from phase7j import select, FEATURES, EXPERTS

ROOT=Path(__file__).resolve().parent.parent
DEVELOPMENT={'DJI_0207','DJI_0204_video2'}
PRIMARY_COMPARATOR='cv_ols'
SAMPLES=(0,100,200,400,600)
BASE='https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69'
CANDIDATES=(
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
 'DJI_vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V')


def digest(p):
    hh=hashlib.sha256()
    with Path(p).open('rb') as fd:
        for block in iter(lambda:fd.read(1<<20), b''):hh.update(block)
    return hh.hexdigest()


def get_lock():
    return json.loads((ROOT/'prospective_lock.json').read_text())


def verify_assets(checkpoints=None):
    lock=get_lock()
    assert lock['primary_comparator']=='cv_ols'
    for path,expected in lock['release_files_sha256'].items():
        actual=digest(ROOT/path)
        if actual!=expected:
            raise ValueError(f'RELEASE LOCK FAILED: {path}: {actual} != {expected}')
    if checkpoints is not None:
        ck=Path(checkpoints)
        for name,item in lock['model_checkpoint_sha256'].items():
            if not (ck/name).is_file():raise FileNotFoundError(f'MISSING CHECKPOINT {ck/name}')
            if digest(ck/name)!=item['sha256']:
                raise ValueError(f'CHECKPOINT LOCK FAILED: {name}')
            if (ck/name).stat().st_size!=item['size_bytes']:
                raise ValueError(f'CHECKPOINT SIZE FAILED: {name}')
    return lock


def build_observation_features(segments, seq):
    rows=[]
    for h in (20,50):
        for seg in segments:
            win=make_windows(seg,h)
            if win is None:continue
            hist, _, starts=win
            f=motion_features(hist)
            for i, start in enumerate(starts):
                row={'sequence':seq,'horizon_frames':h,'track_id':str(seg['track_id']),
                     'segment_id':int(seg['segment_id']),'first_observed_frame':int(start)}
                for key in FEATURES:
                    value=float(f[key][i]);row[key]=value if np.isfinite(value) else np.nan
                rows.append(row)
    result=pd.DataFrame(rows)
    if result.empty: raise ValueError('No eligible forecast windows')
    keys=['sequence','horizon_frames','track_id','segment_id','first_observed_frame']
    if result.duplicated(keys).any():raise ValueError('Duplicate history keys')
    return result


def predict_gate_from_features(features):
    # Copy observation-only features into predictor. Future predictions/errors are
    # deliberately not passed to the gate at all.
    result=features[['sequence','horizon_frames','track_id','segment_id','first_observed_frame']].copy()
    result['selected_expert']=''
    for h in (20,50):
        model=json.loads((ROOT/'locked_models'/f'phase7j_dev_gate_h{h}.json').read_text())
        if set(model['train_sequences']) != DEVELOPMENT:
            raise ValueError('Gate training-source lock failed')
        q=features['horizon_frames'].eq(h)
        if q.any():result.loc[q,'selected_expert']=select(features.loc[q],model)
    if not result.selected_expert.isin(EXPERTS).all():raise ValueError('Unexpected expert label')
    return result


def score_with_frozen_choices(features, choices, detail_rows):
    keys=['horizon_frames','track_id','segment_id','first_observed_frame']
    metrics=pd.DataFrame(detail_rows).copy()
    for key in ['track_id']:
        metrics[key]=metrics[key].astype(str)
    for k in ('horizon_frames','segment_id','first_observed_frame'):
        metrics[k]=metrics[k].astype(int)
    if metrics.duplicated(keys).any():raise ValueError('Duplicate frozen window scores')
    if choices.duplicated(keys).any():raise ValueError('Duplicate gate window choices')
    # One-to-one join, with exact-length guard; choices were computed WITHOUT errors.
    out=choices.merge(metrics,on=keys,how='inner',validate='one_to_one')
    if len(out)!=len(choices) or len(out)!=len(metrics):
        raise ValueError('Gate/forecast window mismatch; abort')
    if not out.selected_expert.isin(EXPERTS).all():raise ValueError('Unknown expert')
    out['adaptive_ade_norm']=[float(getattr(row,f'{row.selected_expert}_ade_norm')) for row in out.itertuples()]
    out['adaptive_fde_norm']=[float(getattr(row,f'{row.selected_expert}_fde_norm')) for row in out.itertuples()]
    if not np.isfinite(out[['adaptive_ade_norm','adaptive_fde_norm']]).all().all():
        raise ValueError('Nonfinite adaptive errors')
    out['adaptive_minus_cv_ols_ADE']=out['adaptive_ade_norm']-out['cv_ols_ade_norm']
    return out


def summarize_scored(df):
    rows=[]
    for h in (20,50):
        for tid in ('pooled', *sorted(df.loc[df.horizon_frames==h,'track_id'].unique())):
            part=df[df.horizon_frames==h]
            if tid!='pooled':part=part[part.track_id==tid]
            if part.empty:continue
            rec={'horizon_frames':h,'track_id':tid,'windows':len(part)}
            for expert in ('adaptive','stationary','cv20','cv_ols','transformer_ensemble'):
                for metric in ('ade_norm','fde_norm'):
                    rec[f'{expert}_{metric}']=float(part[f'{expert}_{metric}'].mean())
            rec['adaptive_minus_cv_ols_ADE']=rec['adaptive_ade_norm']-rec['cv_ols_ade_norm']
            for expert in EXPERTS: rec[f'selected_{expert}_fraction']=float(part.selected_expert.eq(expert).mean())
            rows.append(rec)
    return rows


def run_eval(zip_path, sequence, checkpoint_dir, output, allow_dev=False):
    if sequence in DEVELOPMENT and not allow_dev:
        raise ValueError('Development recording cannot be claimed as new; --allow-development-diagnostic for local regression only')
    if sequence=='DJI_0204_video1':
        print('CAUTION: same DJI_0204 family as development recording. Flight/herd independence NOT verified.')
    lock=verify_assets(checkpoint_dir)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    inv,tracks,segments=inventory(Path(zip_path),sequence)
    if not inv['elephant_track_ids'] or inv['n_20_frame_windows']==0:
        raise ValueError('No eligible elephant forecasting windows')
    # Prediction gate is frozen and feature-only, computed before reference labels are used for scoring.
    obs=build_observation_features(segments,sequence)
    choices=predict_gate_from_features(obs)
    choices.to_csv(out/'gate_choices_OBSERVATION_ONLY.csv',index=False)
    # Actual expert ADE/FDE uses future labels and locked pretrained checkpoints.
    expert_summary,per_window,scales,ck=evaluate(segments,Path(checkpoint_dir),
                                               json.loads((ROOT/'checkpoint_lock.json').read_text()),128)
    scored=score_with_frozen_choices(obs,choices,per_window)
    scored.to_csv(out/'all_experts_and_gate_per_window.csv',index=False,float_format='%.12g')
    rows=summarize_scored(scored)
    pd.DataFrame(rows).to_csv(out/'per_track_and_pooled_descriptive.csv',index=False,float_format='%.12g')
    write_csv(out/'scale_floor_rates.csv',scales)
    write_csv(out/'track_availability.csv',tracks)
    manifest={**inv,
     'phase':'7K', 'fixed_primary_comparison':'adaptive - CV-OLS, group-level ADE, horizons separately',
     'gate_fitted_on_untouched_test_data':False,
     'models':'stationary, cv20, CV-OLS, frozen transformer 3-seed ensemble, locked 7J adaptive',
     'candidate_output_status':('DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATORY' if sequence in DEVELOPMENT
         else 'NEW_SEQUENCE_EXPLORATORY_CAPTURE_EVENT_INDEPENDENCE_UNVERIFIED'),
     'independent_capture_groups_verified':0,
     'source_video_fps_verified':False,'method_effect_on_warnings_or_conformal_coverage':'NOT TESTED',
     'gate_sha256':{f'phase7j_dev_gate_h{h}.json':digest(ROOT/'locked_models'/f'phase7j_dev_gate_h{h}.json') for h in (20,50)},
     'checkpoint_hashes_verified':ck,
     'new_sequence_qa':'Sampled visual species and track review REQUIRED separately',
     'n_scored_windows':len(scored),
     'summary':'Descriptive scores only; no population inference or independent claims'}
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(rows).query("track_id == 'pooled'").to_string(index=False))
    print('CAUTION: NOT independently confirmed; filename does not prove distinct flight/herd.')
    return manifest


def screen_remote(out, names=None):
    rows=[]
    for seq in names or CANDIDATES:
        found=set(); success=0; errors=[]
        for fr in SAMPLES:
            url=f'{BASE}/{seq}/boxid/frame_{fr}.json'
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'CP10-Phase7K/1.0 label-only-screen'})
                with urllib.request.urlopen(req,timeout=15) as response:
                    doc=json.loads(response.read(2_000_000))
                labels={str(s.get('label','')).lower() for s in doc.get('shapes',[])}
                success+=1;found.update(labels)
            except (urllib.error.URLError,TimeoutError,ValueError,json.JSONDecodeError) as e:
                errors.append(f'{fr}:{type(e).__name__}')
        status='ELEPHANT_IN_SAMPLES' if 'elephant' in found else ('OTHER_LABELS_ONLY_IN_SAMPLES' if success else 'UNRESOLVED')
        row={'sequence_id':seq,'sampled_frames':','.join(str(i) for i in SAMPLES),
             'readable_samples':success,'labels_in_samples':';'.join(sorted(found)),
             'status':status,'errors':';'.join(errors),
             'full_recording_species_verified':False,'flight_independence_verified':False}
        rows.append(row)
        print(seq, status, 'sample labels:',row['labels_in_samples'])
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    return rows


def aggregate(dirs, out):
    seen_seq=set();seen_sha=set();rows=[]
    for dr in dirs:
        dr=Path(dr)
        m=json.loads((dr/'evaluation_manifest.json').read_text())
        seq=m['sequence_id']
        if m.get('phase')!='7K':raise ValueError('Only Phase7K evaluation packages may be aggregated')
        locked=verify_assets()
        expected={f'phase7j_dev_gate_h{h}.json':
                  locked['release_files_sha256'][f'locked_models/phase7j_dev_gate_h{h}.json']
                  for h in (20,50)}
        if m.get('gate_sha256')!=expected:
            raise ValueError(f'Gate hash mismatch for {seq}')
        for name,item in locked['model_checkpoint_sha256'].items():
            if m.get('checkpoint_hashes_verified',{}).get(name,{}).get('sha256')!=item['sha256']:
                raise ValueError(f'Checkpoint hash mismatch for {seq}: {name}')
        if seq in DEVELOPMENT:raise ValueError('Development recording found in new candidate aggregation')
        if seq in seen_seq or m['file_sha256'] in seen_sha:raise ValueError('Repeated sequence/ZIP')
        seen_seq.add(seq);seen_sha.add(m['file_sha256'])
        for h in (20,50):
            f=pd.read_csv(dr/'per_track_and_pooled_descriptive.csv')
            match=f[(f.track_id.astype(str)=='pooled') & (f.horizon_frames==h)]
            if len(match)!=1:continue
            r=match.iloc[0]
            rows.append({'sequence_id':seq,'horizon_frames':h,
                'n_overlapping_windows':int(r.windows),
                'adaptive_ADE':float(r.adaptive_ade_norm),
                'cv_ols_ADE':float(r.cv_ols_ade_norm),
                'adaptive_minus_cv_ols_ADE':float(r.adaptive_minus_cv_ols_ADE),
                'descriptive_only':True,'independence':'UNVERIFIED'})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'candidate_sequence_descriptive.csv',index=False)
    (out/'summary_status.json').write_text(json.dumps({
        'unique_sequence_folders':len(seen_seq),'independent_capture_groups_verified':0,
        'predeclared_primary_comparator':'CV-OLS',
        'data_peeking_adjustment':'No tuning on new candidate data permitted',
        'note':'No p-values, CIs or confirmed advantage without verified capture-event units and adequate groups.'},indent=2)+'\n')
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('verify-lock');c.add_argument('--checkpoint-dir',type=Path)
    c=sub.add_parser('screen');c.add_argument('--output',type=Path,required=True)
    c.add_argument('--sequence',action='append',dest='names')
    c=sub.add_parser('intake');c.add_argument('--zip',type=Path,required=True)
    c.add_argument('--sequence',required=True);c.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('evaluate');c.add_argument('--zip',type=Path,required=True)
    c.add_argument('--sequence',required=True);c.add_argument('--checkpoint-dir',type=Path,required=True)
    c.add_argument('--output',type=Path,required=True)
    c.add_argument('--allow-development-diagnostic',action='store_true')
    c=sub.add_parser('aggregate');c.add_argument('--inputs',type=Path,nargs='+',required=True)
    c.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(argv)
    if a.cmd=='verify-lock':
        lock=verify_assets(a.checkpoint_dir)
        print('LOCK VERIFIED:',len(lock['release_files_sha256']),'release files and',
              len(lock['model_checkpoint_sha256']) if a.checkpoint_dir else 'unchecked', 'checkpoints')
    elif a.cmd=='screen':screen_remote(a.output,a.names)
    elif a.cmd=='intake':
        verify_assets();inv,track,seg=inventory(a.zip,a.sequence)
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'intake_manifest.json').write_text(json.dumps(inv,indent=2)+'\n')
        write_csv(a.output/'track_availability.csv',track)
        print(json.dumps(inv,indent=2))
    elif a.cmd=='evaluate':run_eval(a.zip,a.sequence,a.checkpoint_dir,a.output,a.allow_development_diagnostic)
    elif a.cmd=='aggregate':aggregate(a.inputs,a.output)

if __name__=='__main__':main()
