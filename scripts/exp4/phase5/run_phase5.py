#!/usr/bin/env python3
"""Phase 5 exploratory policy ablation from authentic Step7B observations.

No new footage, model inference, recalibration, or field validation.
Preserve Step7G frozen q90, N2; compare thresholded NOMINAL FORECAST CLEARANCE.
Requires no 3rd-party dependencies; does not read ground truth for any decisions.
"""
import argparse, collections, csv, hashlib, json, math
from pathlib import Path

KEY=('dataset','task','sample_index','source_group','track_key','axis','boundary','direction','boundary_id','alpha')
GROUP=('dataset','task','source_group','track_key','axis','boundary','direction','boundary_id')
FIELDS=('dataset','task','sample_index','source_group','track_key','axis','boundary','direction','boundary_id','alpha','response_time_sec','true_crossing_bool','predicted_crossing_bool','uncertainty_intersects_geofence','absolute_conformal_radius','oriented_risk_score','uncertainty_risk_score')
RATIOS=(0.25,0.5,0.75,1.0)
TRUE={'true','1','yes'}

def b(v):
    z=v.strip().lower()
    if z in TRUE:return True
    if z in {'false','0','no'}:return False
    raise ValueError(f'invalid boolean {v!r}')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def div(n,d):return n/d if d else None

def writecsv(p,rows):
    rows=list(rows)
    if not rows:raise ValueError(f'no rows for {p}')
    with open(p,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def score_dict(c):
    tp,fp,fn,tn=(c[k] for k in ('tp','fp','fn','tn'))
    return dict(instances=tp+fp+fn+tn,positive=tp+fn,tp=tp,fp=fp,fn=fn,tn=tn,
                recall=div(tp,tp+fn),false_alert_rate=div(fp,fp+tn),precision=div(tp,tp+fp),
                f1=div(2*tp,2*tp+fp+fn),false_alerts_per_true_alert=div(fp,tp),
                issued_alerts=tp+fp)

def metrics(rows):
    c=collections.Counter()
    for y, a in rows:
        c['tp' if y and a else 'fn' if y else 'fp' if a else 'tn']+=1
    return score_dict(c)

def experiment(path,out,baseline_path):
    out.mkdir(parents=True,exist_ok=True)
    unique={};invalid_repeat=0;raw_count=0;q90_rows=0
    with open(path,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            raw_count+=1
            if abs(float(r['alpha'])-0.1)>1e-9:continue
            q90_rows+=1
            entry={k:r[k] for k in FIELDS}
            entry['sample_index']=int(entry['sample_index'])
            entry['response_time_sec']=float(entry['response_time_sec'])
            for k in ('true_crossing_bool','predicted_crossing_bool','uncertainty_intersects_geofence'):
                entry[k]=b(entry[k])
            for k in ('absolute_conformal_radius','oriented_risk_score','uncertainty_risk_score'):
                entry[k]=float(entry[k])
            key=tuple(entry[k] for k in KEY)
            if key in unique:
                old=unique[key]
                for k in ('true_crossing_bool','predicted_crossing_bool','uncertainty_intersects_geofence','absolute_conformal_radius','oriented_risk_score'):
                    if entry[k]!=old[k]:invalid_repeat+=1;break
                if entry['response_time_sec']<old['response_time_sec']:
                    unique[key]=entry
            else:unique[key]=entry
    if invalid_repeat:raise AssertionError(f'duplicated labels differ: {invalid_repeat}')
    groups=collections.defaultdict(list)
    for row in unique.values():groups[tuple(row[x] for x in GROUP)].append(row)
    # Compute all policies without using true crossing label; use truth only in metric accounting.
    pols=[('original_N2_available',None,'available'),('original_N2_adjacent',None,'adjacent')]
    for method in ['available','adjacent']:
        for rho in RATIOS:pols.append((f'clearance_rho_{rho:g}_{method}',rho,method))
    acc=collections.defaultdict(list);by_source=collections.defaultdict(list);diagnostics=collections.Counter()
    for gkey,items in groups.items():
        items.sort(key=lambda r:r['sample_index'])
        for j,r in enumerate(items):
            if j and r['sample_index']<=items[j-1]['sample_index']:raise AssertionError('nonincreasing sample_index')
            if j and r['sample_index']>items[j-1]['sample_index']+1:diagnostics['groups_with_gaps']=1 # per-group counted separately below
            q=r['absolute_conformal_radius'];ris=r['oriented_risk_score']
            if not math.isfinite(q) or q<=0 or not math.isfinite(ris):raise ValueError('invalid radius or risk score')
            for k in ('uncertainty_risk_score',):
                if not math.isclose(r[k],ris+q,rel_tol=1e-9,abs_tol=1e-10):raise AssertionError('risk inconsistency')
            for pname,rho,mode in pols:
                def eligible(x):
                    if not x['uncertainty_intersects_geofence']:return False
                    if rho is None:return True
                    # signed maximum predicted penetration is below zero for nominal negative.
                    # D_pred = -max directional predicted penetration. Apply only if D_pred <= rho * frozen radius.
                    return x['oriented_risk_score']>=(-rho*x['absolute_conformal_radius'] - 1e-12)
                preceding=items[j-1] if j else None
                two=bool(preceding is not None and eligible(preceding) and eligible(r))
                if mode=='adjacent':two=two and r['sample_index']-preceding['sample_index']==1 if preceding else False
                alarm=r['predicted_crossing_bool'] or two
                outcome=(r['true_crossing_bool'],bool(alarm))
                acc[(r['dataset'],r['task'],pname)].append(outcome)
                by_source[(r['dataset'],r['task'],r['source_group'],pname)].append(outcome)
    rows=[]
    for (ds,task,pname),outcomes in sorted(acc.items()):
        base=metrics(outcomes)
        rows.append(dict(dataset=ds,task=task,policy=pname,**base))
    sourcerows=[]
    for (ds,task,source,pname),outcomes in sorted(by_source.items()):
        sourcerows.append(dict(dataset=ds,task=task,source_group=source,policy=pname,**metrics(outcomes)))
    # compare source-exact Stage7F N=2 baseline, including all absolute cell counts.
    ref={}
    for r in csv.DictReader(open(baseline_path,newline='',encoding='utf8')):
        if int(r['persistence_windows'])==2:ref[(r['dataset'],r['task'])]=r
    checks=[]
    for r in rows:
        if r['policy']!='original_N2_available':continue
        p=ref[(r['dataset'],r['task'])]
        for k in ('instances','tp','fp','fn','tn'):
            checks.append(dict(dataset=r['dataset'],task=r['task'],field=k,independent=r[k],original=int(p[k]),exact_match=r[k]==int(p[k])))
    if not all(x['exact_match'] for x in checks):raise AssertionError('Stage7F benchmark regression mismatch')
    # A post-hoc exploratory selection using already-studied BIRDSAI only.
    # One shared rho and adjacency choice for both horizons, choose min macro FPR with >=0.8 recall at BOTH horizons.
    # If none satisfies report no feasible candidate; do not silently relax.
    dev=[r for r in rows if r['dataset']=='BIRDSAI_TestReal' and r['policy'].startswith('clearance_')]
    candidates=[]
    for pname in sorted({r['policy'] for r in dev}):
        items=[r for r in dev if r['policy']==pname]
        candidates.append(dict(policy=pname,recall_2s=next(r['recall'] for r in items if r['task']=='2s_to_2s'),recall_5s=next(r['recall'] for r in items if r['task']=='2s_to_5s'),mean_fpr=sum(r['false_alert_rate'] for r in items)/2,mean_precision=sum(r['precision'] for r in items)/2,eligible=all(r['recall']>=0.8 for r in items)))
    qualified=[r for r in candidates if r['eligible']]
    pick=min(qualified,key=lambda r:(r['mean_fpr'],-r['mean_precision'],r['policy'])) if qualified else None
    writecsv(out/'phase5_policy_grid.csv',rows)
    writecsv(out/'phase5_policy_by_source.csv',sourcerows)
    writecsv(out/'phase5_baseline_regression_checks.csv',checks)
    writecsv(out/'phase5_exploratory_selection.csv',candidates)
    status={
      'experiment':'exploratory decision layer only: frozen q90, do not infer calibrated coverage from warning metrics',
      'source_csv_sha256':sha(path),'source_original_script':'step7f_temporal_persistence_hysteresis_diagnostic.py',
      'raw_rows':raw_count,'q90_rows_before_response_time_dedup':q90_rows,
      'unique_q90_instances':len(unique),'scenario_groups':len(groups),
      'baseline_regression_checks':len(checks),'baseline_regression_mismatches':sum(not x['exact_match'] for x in checks),
      'selection_criterion':'BIRDSAI TestReal post hoc exploratory: a single rho/mode with >=80% recall at both horizons; minimize macro mean FPR across horizons; WildBox Fresh descriptive replay only',
      'selected_exploratory_policy':pick,
      'no_test_separation':'All four populations have previously been inspected, so no independent confirmatory inference is possible.',
      'no_field_claim':'All labels are virtual image-plane boundaries, not physical/georeferenced conflict events.',
      'no_time_claim':'Adjacent sample indices do not prove actual seconds of contiguous monitoring.',
      'no_coverage_claim':'Thresholding uncertainty warning decisions does NOT modify or improve conformal pathwise coverage.',
      'policy_basis':'No future labels, future GT, or event outcomes enter policy decisions; GT used for evaluation and post-hoc selection only.'}
    (out/'phase5_machine_audit.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in status.items() if k not in ('source_csv_sha256','no_test_separation','no_field_claim','no_time_claim','no_coverage_claim','policy_basis')},indent=2))
    for r in rows:
        if r['policy']=='original_N2_available' or (pick and r['policy']==pick['policy']):
            print(r['dataset'],r['task'],r['policy'], 'P=',round(r['precision'],4),'R=',round(r['recall'],4),'FPR=',round(r['false_alert_rate'],4),'FP=',r['fp'],'TP=',r['tp'])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,default=Path('/mnt/data/step7b_v2_all_geofence_instance_alarms.csv'));a.add_argument('--output-dir',type=Path,default=Path('/mnt/data/EXP4_Phase5_Method'));a.add_argument('--baseline',type=Path,default=Path('/mnt/data/EXP4_Phase2_Report/step7f_independent_replay.csv'))
    x=a.parse_args();experiment(x.source,x.output_dir,x.baseline)
