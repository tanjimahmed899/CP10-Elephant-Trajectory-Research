#!/usr/bin/env python3
"""Reproducible *instance-level* prospective Phase-6 policy replay.

Cannot assess calibrated coverage, physical fences, or matched crossing episodes.
For those outcomes use true future paths and fixed-reference event annotations in a
separate source-exact evaluator. No model training, tuning, or label generation here.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
from phase6_preflight import audit,reader,boolean,timestamp,INSTANCE_REQ,sha256

POLICIES=('baseline_N2_available','phase5_locked_N2_available',
          'baseline_N2_clock_1p1s','phase5_locked_N2_clock_1p1s')
RHO={'2s_to_2s':0.75,'2s_to_5s':1.0}

def choose_candidate(row,rho):
    # Truth labels are never touched by the alarm computation.
    return row['uncertainty_intersects_geofence'] and (
        row['oriented_risk_score'] >= (-rho * row['absolute_conformal_radius'] - 1e-12))

def score(rows):
    a=collections.Counter()
    for truth,alarm in rows:
        a['tp' if truth and alarm else 'fn' if truth else 'fp' if alarm else 'tn']+=1
    tp,fp,fn,tn=(a[k] for k in ('tp','fp','fn','tn'))
    safe=lambda x,y:x/y if y else None
    return dict(n=tp+fp+fn+tn,positive=tp+fn,tp=tp,fp=fp,fn=fn,tn=tn,
                recall=safe(tp,tp+fn),false_alert_rate=safe(fp,fp+tn),
                precision=safe(tp,tp+fp),f1=safe(2*tp,2*tp+fp+fn))

def analyze(records, geometry='moving_anchor'):
    groups=collections.defaultdict(list)
    for r in records:
        if r['reference_geometry'].strip()!=geometry:continue
        if not boolean(r['eligible']):continue
        row={k:r[k] for k in r}
        row['sample_index']=int(r['sample_index'])
        row['t']=timestamp(r['forecast_timestamp_utc'])
        row['predicted_crossing_bool']=boolean(r['predicted_crossing_bool'])
        row['uncertainty_intersects_geofence']=boolean(r['uncertainty_intersects_geofence'])
        row['true_crossing_bool']=boolean(r['true_crossing_bool'])
        row['oriented_risk_score']=float(r['oriented_risk_score'])
        row['absolute_conformal_radius']=float(r['absolute_conformal_radius'])
        key=tuple(row[x] for x in ('recording_id','task','track_key','axis','boundary_id','direction'))
        groups[key].append(row)
    agg=collections.defaultdict(list)
    missing_gaps=0
    for key,rr in groups.items():
        rr.sort(key=lambda z:(z['sample_index'],z['t']))
        for j,r in enumerate(rr):
            prior=rr[j-1] if j else None
            if prior and (r['sample_index']<=prior['sample_index'] or r['t']<=prior['t']):
                raise ValueError(f'non-monotonic clock/index sequence at {key}')
            if prior and r['sample_index']>prior['sample_index']+1:missing_gaps+=1
            for mode in POLICIES:
                rho=1.0 if mode.startswith('baseline_') else RHO[r['task']]
                enough=bool(prior and choose_candidate(prior,rho) and choose_candidate(r,rho))
                if mode.endswith('_clock_1p1s'):
                    enough=bool(enough and (0 < (r['t']-prior['t']).total_seconds() <= 1.1))
                alarm=r['predicted_crossing_bool'] or enough
                agg[(r['task'],r['collection_group_id'],mode)].append((r['true_crossing_bool'],alarm))
    pergroup=[]
    for (task,group,mode),rr in sorted(agg.items()):
        pergroup.append(dict(task=task,collection_group_id=group,policy=mode,**score(rr)))
    pooled=[]
    for task in sorted({r['task'] for r in records}):
        for mode in POLICIES:
            rr=[]
            for (t,group,m),vals in agg.items():
                if t==task and m==mode:rr.extend(vals)
            if rr:pooled.append(dict(task=task,policy=mode,**score(rr)))
    deltas=[]
    by={(r['task'],r['collection_group_id'],r['policy']):r for r in pergroup}
    for task,group in sorted({(r['task'],r['collection_group_id']) for r in pergroup}):
        for suffix in ('available','clock_1p1s'):
            b=by[(task,group,'baseline_N2_'+suffix)]
            c=by[(task,group,'phase5_locked_N2_'+suffix)]
            diff=lambda key:None if b[key] is None or c[key] is None else c[key]-b[key]
            deltas.append(dict(task=task,collection_group_id=group,mode=suffix,
                               false_alert_rate_delta=diff('false_alert_rate'),
                               recall_delta=diff('recall'),precision_delta=diff('precision'),
                               tp_delta=c['tp']-b['tp'],fp_delta=c['fp']-b['fp'],
                               interpretive_status='descriptive_not_confirmatory_significance'))
    return dict(pergroup=pergroup,pooled=pooled,deltas=deltas,scenario_streams=len(groups),
                skipped_sample_index_intervals=missing_gaps)

def write(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--instances',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--geometry',choices=['moving_anchor','segment_fixed_reference'],default='moving_anchor')
    a=p.parse_args()
    check=audit(a.manifest,a.instances)
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    (out/'preflight.json').write_text(json.dumps(check,indent=2),encoding='utf8')
    if check['errors']:
        raise SystemExit(f'Preflight FAILED; see {out}/preflight.json')
    records=reader(a.instances,INSTANCE_REQ)
    r=analyze(records,geometry=a.geometry)
    if not r['pergroup']:
        raise SystemExit(f'No eligible {a.geometry} input rows; no analysis run')
    write(out/'phase6_instance_per_group.csv',r['pergroup'])
    write(out/'phase6_instance_pooled_descriptive.csv',r['pooled'])
    write(out/'phase6_paired_group_deltas.csv',r['deltas'])
    summary={k:v for k,v in r.items() if k not in {'pergroup','pooled','deltas'}}
    summary.update({'geometry_scope':a.geometry,'manifest_sha256':sha256(a.manifest),'instances_sha256':sha256(a.instances),
        'outcome':'retrospective computation on supplied input; independent pre-registration requires separate evidence',
        'excluded':'No trajectory coverage computation, no source-exact fixed-reference event matching; no observed field performance',
        'confirmatory_approval':'NOT AUTOMATICALLY ASSERTED'})
    (out/'phase6_replay_receipt.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
