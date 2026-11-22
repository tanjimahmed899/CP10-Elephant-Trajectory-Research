#!/usr/bin/env python3
"""Preflight for independent recording-group elephant validation.

Pure audit: never converts an unapproved prospective design into confirmed evidence.
Requires no external packages.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

MANIFEST_REQ = ('recording_id','collection_group_id','split','capture_event_id',
                'sensor_modality','raw_file_sha256','previously_inspected',
                'annotation_complete','geometry_type','geometry_validated',
                'source_independence_evidence','monitoring_duration_s')
INSTANCE_REQ = ('recording_id','collection_group_id','task','sample_index',
                'forecast_timestamp_utc','track_key','axis','boundary_id','direction',
                'true_crossing_bool','predicted_crossing_bool','uncertainty_intersects_geofence',
                'oriented_risk_score','absolute_conformal_radius','reference_geometry',
                'uncertainty_transport_method','uncertainty_transport_validated','ground_truth_label_source','eligible')
SPLITS={'development','calibration','confirmation'}
TASKS={'2s_to_2s','2s_to_5s'}
YES={'true','1','yes'}; NO={'false','0','no'}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def boolean(v):
    v=str(v).strip().lower()
    if v in YES:return True
    if v in NO:return False
    raise ValueError(f'invalid boolean {v!r}')

def reader(path,required):
    with open(path,'r',encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        if rd.fieldnames is None:raise ValueError(f'{path}: empty CSV')
        missing=[k for k in required if k not in rd.fieldnames]
        if missing:raise ValueError(f'{path}: missing columns {missing}')
        rows=list(rd)
    if not rows:raise ValueError(f'{path}: CSV has no data rows')
    return rows

def timestamp(v):
    # Reject naive local timestamps: streams need a defensible clock.
    s=v.strip().replace('Z','+00:00')
    z=dt.datetime.fromisoformat(s)
    if z.utcoffset() is None:raise ValueError('timestamp lacks UTC offset')
    return z.astimezone(dt.timezone.utc)

def audit(manifest_path,instances_path=None):
    m=reader(manifest_path,MANIFEST_REQ)
    errors=[]; warnings=[]; splits=collections.defaultdict(set)
    events=collections.defaultdict(set); subjects=collections.defaultdict(set)
    raw_hashes=collections.defaultdict(list)
    by_id={}
    for i,r in enumerate(m,2):
        rec=r['recording_id'].strip(); group=r['collection_group_id'].strip()
        split=r['split'].strip().lower()
        if not rec or not group:errors.append(f'manifest line {i}: blank recording/group ID')
        if rec in by_id:errors.append(f'manifest line {i}: duplicated recording ID {rec}')
        by_id[rec]=r
        if split not in SPLITS:errors.append(f'manifest line {i}: invalid split {split}')
        splits[group].add(split)
        event=r['capture_event_id'].strip()
        if event:events[event].add(split)
        subject=r.get('subject_group_id','').strip()
        if subject:subjects[subject].add(split)
        raw=r['raw_file_sha256'].strip().lower()
        if re.fullmatch(r'[0-9a-f]{64}',raw):raw_hashes[raw].append(rec)
        else:errors.append(f'manifest line {i}: invalid/missing SHA256 for {rec}')
        try:
            past=boolean(r['previously_inspected'])
            complete=boolean(r['annotation_complete'])
            valid=boolean(r['geometry_validated'])
        except ValueError as exc:
            errors.append(f'manifest line {i}: {exc}'); continue
        if split=='confirmation':
            if past:errors.append(f'{rec}: previously inspected cannot enter confirmation')
            if not complete:errors.append(f'{rec}: incomplete annotation cannot certify full confirmatory evaluation')
            if not valid:errors.append(f'{rec}: geometry not validated')
            if r['geometry_type'].strip() not in {'moving_anchor','segment_fixed_reference','both'}:
                errors.append(f'{rec}: unrecognized geometry_type')
            if not r['source_independence_evidence'].strip():
                errors.append(f'{rec}: source-group independence evidence missing')
            if group.lower() in {'elep1','elep2','elep3'}:
                errors.append(f'{rec}: known historical WildBox group {group}')
        try:
            dur=float(r['monitoring_duration_s'])
            if dur<=0:raise ValueError('nonpositive')
        except ValueError:
            errors.append(f'{rec}: nonpositive or invalid monitoring_duration_s')
        if r['sensor_modality'].strip().upper() not in {'RGB','THERMAL','INFRARED'}:
            errors.append(f'{rec}: unsupported modality label')
    for group,ss in splits.items():
        if len(ss)>1:errors.append(f'group leakage: {group} occurs in {sorted(ss)}')
    for event,ss in events.items():
        if len(ss)>1:errors.append(f'capture-event leakage: {event} occurs in {sorted(ss)}')
    for subject,ss in subjects.items():
        if len(ss)>1:errors.append(f'known-subject leakage: {subject} occurs in {sorted(ss)}')
    for hx,rr in raw_hashes.items():
        if len(rr)>1:errors.append(f'byte-identical raw media across recording IDs: {rr}')
    conf=[r for r in m if r['split'].strip().lower()=='confirmation']
    cal=[r for r in m if r['split'].strip().lower()=='calibration']
    conf_groups={r['collection_group_id'] for r in conf}
    cal_groups={r['collection_group_id'] for r in cal}
    if len(conf_groups)<8:warnings.append(f'Only {len(conf_groups)} confirmation groups vs proposed target of 8; underpowered/limited-domain interpretation may be necessary')
    if len(cal_groups)<3:warnings.append('Below proposed 3 calibration groups (needed only for separately evaluated adaptive-calibration arm)')
    if {r['sensor_modality'].strip().upper() for r in conf}!={'RGB'}:
        warnings.append('Main RGB elephant-confirmation scope mixes modalities; stratify and do not pool')
    for r in m:
        if r['split'].strip().lower()=='confirmation':
            for flag in ['wildlife_clearance','airspace_clearance']:
                if r.get(flag,'').strip().lower() not in {'approved','not_applicable_public_source'}:
                    warnings.append(f"{r['recording_id']}: {flag} is not documented as approved/not_applicable_public_source")
    info={"status":"FAIL" if errors else "SCHEMA_PASS_NOT_SCIENTIFIC_CONFIRMATION",
          "manifest_sha256":sha256(manifest_path),"recordings":len(m),
          "confirmation_groups":len(conf_groups),"calibration_groups":len(cal_groups),
          "errors":errors,"warnings":warnings}
    if instances_path:
        instances=reader(instances_path,INSTANCE_REQ)
        info['instances_sha256']=sha256(instances_path)
        info['instance_count']=len(instances)
        keys=set(); instances_by_rec=collections.Counter(); valid_by_rec=collections.Counter()
        for i,r in enumerate(instances,2):
            rec=r['recording_id'].strip(); group=r['collection_group_id'].strip()
            if rec not in by_id:
                errors.append(f'instances line {i}: recording {rec} absent from manifest')
                continue
            if group!=by_id[rec]['collection_group_id'].strip():
                errors.append(f'instances line {i}: group mismatch for {rec}')
            if by_id[rec]['split'].strip().lower()!='confirmation':
                errors.append(f'instances line {i}: non-confirmation prediction rows in confirmatory input')
            if r['task'].strip() not in TASKS:errors.append(f'instances line {i}: invalid task')
            if r['reference_geometry'].strip() not in {'moving_anchor','segment_fixed_reference'}:
                errors.append(f'instances line {i}: unsupported reference geometry')
            if by_id[rec]['geometry_type'].strip() not in {r['reference_geometry'].strip(),'both'}:
                errors.append(f'instances line {i}: geometry not declared in manifest')
            try:
                transport_ok=boolean(r['uncertainty_transport_validated'])
            except ValueError:
                transport_ok=False
            if not transport_ok or not r['uncertainty_transport_method'].strip():
                errors.append(f'instances line {i}: missing/invalid uncertainty transport provenance')
            if not r['ground_truth_label_source'].strip():
                errors.append(f'instances line {i}: missing truth provenance')
            try:
                j=int(r['sample_index']); timestamp(r['forecast_timestamp_utc'])
                if j<0:raise ValueError('negative sample_index')
                for field in ['true_crossing_bool','predicted_crossing_bool','uncertainty_intersects_geofence','eligible']:
                    boolean(r[field])
                z=float(r['oriented_risk_score']); rad=float(r['absolute_conformal_radius'])
                import math
                if not math.isfinite(z) or not math.isfinite(rad) or rad<=0:raise ValueError('bad risk or radius')
            except (ValueError,OverflowError) as exc:
                errors.append(f'instances line {i}: {exc}')
            k=(rec,r['reference_geometry'],r['task'],r['sample_index'],r['track_key'],r['axis'],r['boundary_id'],r['direction'])
            if k in keys:errors.append(f'instances line {i}: duplicate scenario key {k}')
            keys.add(k)
            instances_by_rec[rec]+=1
            if r['eligible'].strip().lower() in YES:valid_by_rec[rec]+=1
        for r in conf:
            rec=r['recording_id']
            if valid_by_rec[rec]==0:errors.append(f'{rec}: no valid confirmation instance rows')
        info['eligible_instance_count']=sum(valid_by_rec.values())
        info['per_recording_rows']={k:{'total':v,'eligible':valid_by_rec[k]} for k,v in sorted(instances_by_rec.items())}
        info['status']='FAIL' if errors else 'SCHEMA_PASS_NOT_SCIENTIFIC_CONFIRMATION'
    return info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--instances')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    r=audit(a.manifest,a.instances)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(r,indent=2),encoding='utf8')
    print(json.dumps({'status':r['status'],'errors':len(r['errors']),'warnings':len(r['warnings'])},indent=2))
    raise SystemExit(1 if r['errors'] else 0)

if __name__=='__main__': main()
