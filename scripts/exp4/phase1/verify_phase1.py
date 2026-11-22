#!/usr/bin/env python3
"""EXP4 Phase 1 reproducible source CSV / operating-lock / arithmetic audit.

Reads authentic uploaded CSV/JSON; does not recompute experimental model outputs.
Only report/table arithmetic for values explicitly recorded in source PDFs.
"""
from pathlib import Path
import argparse, csv, json, hashlib, math, numpy as np, pandas as pd

EXPECTED_Q = {
    ('2s_to_2s', .10): 2.158767888280567,
    ('2s_to_2s', .05): 2.560772554862189,
    ('2s_to_5s', .10): 1.975129668255982,
    ('2s_to_5s', .05): 2.950976253259077,
}
EXPECTED_TRACKS = {'2s_to_2s': 62, '2s_to_5s': 34}
EXPECTED_WINDOWS = {'2s_to_2s': 531, '2s_to_5s': 382}
REQUIRED = {'track_key','source_group','outer_fold','windows','track_max_canonical_error','track_mean_ADE_absolute','track_mean_FDE_absolute','task'}

def filehash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()

def quantile_finite(scores, alpha):
    s=np.asarray(scores,dtype=float)
    if not len(s) or not np.all(np.isfinite(s)):
        raise ValueError('Scores must be nonempty and finite')
    n=len(s); k=math.ceil((n+1)*(1-alpha))
    # Only a finite-sample split conformal upper quantile; +infinity is correct
    # if k > n, rather than silently clipping to the last score.
    return n,k,(float(np.sort(s)[k-1]) if k<=n else math.inf)

def audit(csv_path,lock_path,output):
    output.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(csv_path, dtype={'track_key':str,'source_group':str,'task':str})
    assert REQUIRED.issubset(d.columns), f'Missing fields {REQUIRED-set(d.columns)}'
    assert len(d)==96,f'Expected 96 calibration tracks; got {len(d)}'
    assert not d[list(REQUIRED)].isna().any().any(),'Nulls in required columns'
    assert not d.duplicated(['task','track_key']).any(),'Duplicate task/track_key'
    assert np.isfinite(d[['track_max_canonical_error','track_mean_ADE_absolute','track_mean_FDE_absolute']].to_numpy()).all()
    assert (d['windows']>=1).all()
    assert set(d['task'])==set(EXPECTED_TRACKS)
    assert d['outer_fold'].isin([0,1,2]).all()
    qrows=[]; grows=[]
    for task,g in d.groupby('task',sort=True):
        assert len(g)==EXPECTED_TRACKS[task]
        assert int(g['windows'].sum())==EXPECTED_WINDOWS[task]
        for group,sub in g.groupby('source_group'):
            grows.append({'task':task,'source_group':group,'tracks':len(sub),'windows':int(sub.windows.sum()),'share_of_tracks':len(sub)/len(g),'share_of_windows':sub.windows.sum()/g.windows.sum(), 'max_score':sub.track_max_canonical_error.max(), 'median_score':sub.track_max_canonical_error.median()})
        for alpha in [.10,.05]:
            n,k,q=quantile_finite(g.track_max_canonical_error,alpha)
            report_q=EXPECTED_Q[(task,alpha)]
            assert abs(q-report_q)<1e-12, f'Mismatch {task=} {alpha=} {q=} {report_q=}'
            qrows.append({'task':task,'alpha':alpha,'target_coverage':1-alpha,'n_calibration_tracks':n,'n_calibration_windows':int(g.windows.sum()),'rank_k':k,'recomputed_q':q,'source_pdf_q':report_q,'abs_difference':abs(q-report_q),'calibration_score_fraction_le_q':float((g.track_max_canonical_error<=q).mean()),'source_groups':g.source_group.nunique(),'outer_folds':g.outer_fold.nunique()})
    qdf=pd.DataFrame(qrows); gdf=pd.DataFrame(grows)
    qdf.to_csv(output/'conformal_quantile_verification.csv',index=False)
    gdf.to_csv(output/'calibration_source_concentration.csv',index=False)
    lock=json.loads(lock_path.read_text(encoding='utf8'))
    assert lock['selected_persistence_windows']==2
    assert lock['primary_conformal_level']==.9
    assert lock['selection_is_post_hoc_exploratory'] is True
    assert lock['selection_dataset']=='BIRDSAI_TestReal'
    assert lock['selection_rule']['candidate_set']==[2,3,5]
    assert lock['selection_rule']['minimum_sensitivity']==.8
    assert lock['selection_rule']['minimum_relative_false_alert_reduction_vs_N1']==.2
    # From Experimental Results (Part-4).pdf pp 10-11; source-table only.
    crossing=[]
    for horizon,cases,changed,events,matched_n2,alert_runs,unassociated in [
        ('2s',4524,299,133,116,463,348),('5s',3432,327,124,109,347,237)]:
        crossing.append({'horizon':horizon,'original_boundary_cases':cases,'reported_changed_labels':changed,'recomputed_changed_pct':100*changed/cases, 'reported_eligible_events':events,'reported_matched_events_N2':matched_n2,'recomputed_event_recall_pct':100*matched_n2/events,'reported_alert_runs_N2':alert_runs,'reported_unassociated_alert_runs_N2':unassociated,'recomputed_unassociated_run_pct':100*unassociated/alert_runs,'status':'ARITHMETIC_OF_PDF_TABLE_ONLY_NOT_RAW_LABEL_REPLAY'})
    pd.DataFrame(crossing).to_csv(output/'stage6_pdf_arithmetic_checks.csv',index=False)
    # External BAMBI q90 in two different representation setups: should NEVER be merged.
    bambi=[]
    for representation,horizon,rows,coverage in [('original_metre','2s',1855,1.0),('original_metre','5s',623,.011236),('compatible_normalized','2s',1855,1.0),('compatible_normalized','5s',623,.025682)]:
        bambi.append({'representation':representation,'horizon':horizon,'windows':rows,'coverage_reported_rounded':coverage,'approx_covered_count_from_rounded_coverage':round(rows*coverage),'status':'SOURCE_PDF_REPORTED_NOT_INDEPENDENTLY_RECOMPUTED'})
    pd.DataFrame(bambi).to_csv(output/'bambi_representation_separation.csv',index=False)
    meta={'csv_rows':len(d),'task_counts':d.task.value_counts().to_dict(),'lock_status':lock['status'],'sha256':{'score_csv':filehash(csv_path),'lock_json':filehash(lock_path)},'missing_to_verify_N2':['TABLE_step7f_persistence_tradeoff.csv','step7b_v2_all_geofence_instance_alarms.csv'],'missing_to_verify_fixed_reference':['per-case raw crossing-labels and anchor reference transforms','GT 3-D centers, camera geometry/depth evidence'], 'limitation':'Quantiles recomputed from source CSV, no model re-training or untouched test performed.'}
    (output/'machine_readable_audit.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print(qdf.to_string(index=False));print('\n',gdf.to_string(index=False));print('\nAudit saved:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,default=Path('/mnt/data/crossfit_track_calibration_scores.csv'));p.add_argument('--lock',type=Path,default=Path('/mnt/data/STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json'));p.add_argument('--out',type=Path,default=Path('/mnt/data/EXP4_Phase1_Report'))
    a=p.parse_args();audit(a.scores,a.lock,a.out)
