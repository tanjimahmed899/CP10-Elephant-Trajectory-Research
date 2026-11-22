#!/usr/bin/env python3
"""Phase 7: conservative metadata-only gate. Never certifies independence by itself.

Standard library only. No model fitting or outcome computation. This program
checks entered evidence metadata; authenticity requires human/source verification.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

REQ = ('candidate_id','recording_id','collection_group_id','capture_event_id',
       'subject_group_id','sensor_modality','species','raw_video_path',
       'raw_file_sha256','annotations_path','annotation_sha256',
       'track_ids_present','camera_pose_present','temporal_timestamps_present',
       'source_rights_review','overlap_review','overlap_evidence_path',
       'preregistration_reference','split')
EXPECTED_SPLITS = {'development','calibration','confirmation'}
YES = {'yes','true','1'}
NO = {'no','false','0'}
HEX = re.compile(r'^[0-9a-f]{64}$')
EXCLUDED = {'elep1','elep2','elep3'}

def hashfile(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def read_csv(path, columns):
    with open(path, encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f'{path}: empty or malformed CSV')
        missing=set(columns)-set(r.fieldnames)
        if missing:raise ValueError(f'{path}: missing required columns {sorted(missing)}')
        return list(r)

def parse_bool(value):
    v=value.strip().lower()
    if v in YES:return True
    if v in NO:return False
    raise ValueError(f'not yes/no: {value!r}')

def intake(path, history_path=None):
    rows=read_csv(path,REQ)
    errors=[];warnings=[];groups=collections.defaultdict(set)
    events=collections.defaultdict(set);subjects=collections.defaultdict(set)
    hashes=collections.defaultdict(list); rec_ids=[]
    history_groups=set(EXCLUDED); hist_events=set(); hist_ids=set(); hist_hashes=set()
    if history_path:
        prior=read_csv(history_path,('source_label','recording_id','collection_group_id','capture_event_id','raw_file_sha256'))
        for r in prior:
            for column, bucket in [('collection_group_id',history_groups),('capture_event_id',hist_events),('recording_id',hist_ids),('raw_file_sha256',hist_hashes)]:
                v=r[column].strip().lower()
                if v and v!='unknown':bucket.add(v)
    if not rows:
        return {'status':'NO_INPUT_RECORDINGS','message':'No data; template only','errors':[], 'warnings':[], 'row_count':0}
    for line,r in enumerate(rows,2):
        rec=r['recording_id'].strip(); group=r['collection_group_id'].strip()
        event=r['capture_event_id'].strip(); subj=r['subject_group_id'].strip()
        split=r['split'].strip().lower(); raw=r['raw_file_sha256'].strip().lower()
        if not rec or not group or not event:errors.append(f'line {line}: missing recording/group/capture event')
        if not r['candidate_id'].strip():errors.append(f'line {line}: candidate_id is required')
        if rec in rec_ids:errors.append(f'line {line}: duplicate recording_id {rec}')
        rec_ids.append(rec)
        if split not in EXPECTED_SPLITS:errors.append(f'{rec}: invalid split {split!r}')
        groups[group.lower()].add(split); events[event.lower()].add(group.lower())
        if subj:subjects[subj.lower()].add(group.lower())
        if not HEX.fullmatch(raw):errors.append(f'{rec}: raw_file_sha256 absent or invalid')
        else:hashes[raw].append(rec)
        an=r['annotation_sha256'].strip().lower()
        if not HEX.fullmatch(an):errors.append(f'{rec}: annotation_sha256 absent or invalid')
        if group.lower() in history_groups or event.lower() in hist_events or rec.lower() in hist_ids or raw in hist_hashes:
            errors.append(f'{rec}: direct conflict with historical source manifest: cannot mark fresh')
        if split=='confirmation':
            if r['sensor_modality'].strip().upper()!='RGB':errors.append(f'{rec}: confirmation main arm must be RGB; separately stratify cross-modal data')
            if r['species'].strip().lower() not in {'elephant','african bush elephant','asian elephant'}:
                errors.append(f'{rec}: non-elephant species not valid for elephant confirmation')
            for field in ('track_ids_present','temporal_timestamps_present'):
                try:ok=parse_bool(r[field])
                except ValueError:ok=False
                if not ok:errors.append(f'{rec}: {field} not verified yes')
            # Required attestation is metadata, not independent proof that the attestations are true.
            if r['overlap_review'].strip().lower()!='verified_no_overlap':
                errors.append(f'{rec}: source overlap unresolved; requires verified_no_overlap after manual provenance check')
            if not r['overlap_evidence_path'].strip():errors.append(f'{rec}: missing provenance-review evidence reference')
            if r['source_rights_review'].strip().lower()!='approved_for_this_research':
                errors.append(f'{rec}: media/data rights not approved for this research')
            if not r['preregistration_reference'].strip():errors.append(f'{rec}: supervisor-approved registration reference missing')
            if not r['raw_video_path'].strip() or not r['annotations_path'].strip():
                errors.append(f'{rec}: local video and annotation paths required')
            # Camera pose is a separate gate: moving-anchor experiment can run without it.
            try:pose=parse_bool(r['camera_pose_present'])
            except ValueError:pose=False
            if not pose:warnings.append(f'{rec}: no verified camera pose: fixed-reference arm NOT eligible')
        if r['raw_video_path'].strip():
            p=Path(r['raw_video_path'])
            if p.exists() and p.is_file():
                if HEX.fullmatch(raw) and hashfile(p)!=raw:errors.append(f'{rec}: raw video SHA256 mismatch')
            elif split=='confirmation':errors.append(f'{rec}: raw media path not accessible for SHA256 verification')
        if r['annotations_path'].strip():
            p=Path(r['annotations_path'])
            if p.exists() and p.is_file():
                if HEX.fullmatch(an) and hashfile(p)!=an:errors.append(f'{rec}: annotation SHA256 mismatch')
            elif split=='confirmation':errors.append(f'{rec}: annotation path inaccessible for SHA256 verification')
        if not r.get('source_date_utc','').strip():warnings.append(f'{rec}: no documented capture date')
    for group,sp in groups.items():
        if len(sp)>1:errors.append(f'group split leakage: {group} in {sorted(sp)}')
    for event,gg in events.items():
        if len(gg)>1:errors.append(f'capture-event {event} split into multiple group IDs: {sorted(gg)}')
    for subj,gg in subjects.items():
        if len(gg)>1:errors.append(f'known elephant subject group {subj} split into multiple groups: {sorted(gg)}')
    for h,rr in hashes.items():
        if len(rr)>1:errors.append(f'byte-identical raw media disguised as independent videos: {rr}')
    conf={r['collection_group_id'].lower() for r in rows if r['split'].lower()=='confirmation'}
    calib={r['collection_group_id'].lower() for r in rows if r['split'].lower()=='calibration'}
    if len(conf)<8:warnings.append(f'Only {len(conf)} confirmation groups; 8 is an earlier design target, NOT power-justified')
    if len(calib)<3:warnings.append(f'Only {len(calib)} calibration groups; 3 is a design target for separate adaptation arm')
    return {
        'status':'BLOCKED' if errors else 'METADATA_CHECKS_PASSED_HUMAN_REVIEW_REQUIRED',
        'not_proof_of_independence':True,
        'manifest_sha256':hashfile(path),
        'row_count':len(rows),'confirmation_groups_entered':len(conf),
        'calibration_groups_entered':len(calib),
        'errors':errors,'warnings':warnings,
        'interpretation':'No outcome metrics calculated; media authenticity and actual provenance must be human-reviewed.'
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--intake',required=True)
    ap.add_argument('--historical',help='historical source labels, known hashes/groups')
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    result=intake(a.intake,a.historical)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'errors':len(result['errors']),'warnings':len(result['warnings'])},indent=2))
    raise SystemExit(1 if result['status']=='BLOCKED' else 0)

if __name__=='__main__':main()
