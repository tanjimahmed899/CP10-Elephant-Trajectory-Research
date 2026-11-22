"""Independent stdlib replay of Step-7F, Step-7G, and Step-7H on saved CSVs.

No model retraining, online inference, new elephant video evaluation, or field tests.
Run: python verify_phase2.py --input-dir /mnt/data --output-dir /mnt/data/EXP4_Phase2_Report
"""
import argparse, csv, hashlib, json, math, collections, os
from pathlib import Path
from statistics import median

KEYS=('dataset','task','sample_index','source_group','track_key','axis','boundary','direction','boundary_id','alpha')
SEQ=('dataset','task','source_group','track_key','axis','boundary','direction','boundary_id')
FIELDS=('dataset','task','sample_index','source_group','track_key','axis','boundary','direction','boundary_id','alpha','response_time_sec','true_crossing_bool','predicted_crossing_bool','uncertainty_intersects_geofence','true_tte_sec','predicted_tte_sec','distance_to_boundary','alarm_state')
METRICS=('instances','actual_crossings','nominal_sensitivity','nominal_false_alert_rate','nominal_precision','persistent_alert_sensitivity','persistent_false_alert_rate','persistent_alert_precision','nominal_missed_true_crossings','rescued_true_crossings','rescue_rate_among_nominal_misses','incremental_alerts_total','incremental_alerts_true','incremental_alerts_false','incremental_alert_precision','false_alerts_per_rescued_crossing')
TRUE={'true','1','yes','y','t'}; FALSE={'false','0','no','n','f'}

def b(v):
    t=str(v).strip().lower()
    if t not in TRUE|FALSE: raise ValueError(f'Invalid boolean {v!r}')
    return t in TRUE

def num(s):
    return float(s) if s.strip() else math.nan

def safe(a,d): return a/d if d else math.nan

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for blob in iter(lambda:f.read(1024*1024),b''):h.update(blob)
    return h.hexdigest()

def write_csv(path,records,header=None):
    records=list(records)
    if header is None: header=list(records[0]) if records else []
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=header,extrasaction='ignore')
        w.writeheader();w.writerows(records)

def cm(y,alarm):
    tp=sum(1 for a,z in zip(alarm,y) if a and z)
    fp=sum(1 for a,z in zip(alarm,y) if a and not z)
    fn=sum(1 for a,z in zip(alarm,y) if not a and z)
    tn=sum(1 for a,z in zip(alarm,y) if not a and not z)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,sensitivity=safe(tp,tp+fn),false_alert_rate=safe(fp,fp+tn),precision=safe(tp,tp+fp))

def main(indir,outdir):
    indir=Path(indir); outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    original_path=indir/'step7b_v2_all_geofence_instance_alarms.csv'
    table_path=indir/'TABLE_step7f_persistence_tradeoff.csv'
    lock_path=indir/'STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json'
    lock=json.load(open(lock_path,encoding='utf8'))
    files={x.name:sha256(x) for x in [original_path,table_path,lock_path]}
    checksum_pass=files[table_path.name]==lock['source_step7f_sha256']

    table={}
    with open(table_path,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['dataset'],r['task'],int(r['persistence_windows']))
            if key in table: raise AssertionError(f'duplicate original table key: {key}')
            table[key]=r

    # Stage-7F response-time duplicates and Step-7H records
    dedup={}
    first_min_response={}
    all_h=[]
    population_rows=collections.Counter()
    alpha_counts=collections.Counter()
    duplicate_mismatch=collections.Counter()
    duplicated_inconsistent=0
    # Group is simple because all rows include the same exact sample index and 6 boundaries.
    with open(original_path,newline='',encoding='utf-8-sig') as f:
        r=csv.reader(f);header=next(r)
        pos={k:i for i,k in enumerate(header)}
        required=set(FIELDS)
        if not required.issubset(pos):raise RuntimeError(f'Missing CSV columns: {required-set(pos)}')
        for row in r:
            if len(row)!=len(header):raise RuntimeError('Unexpected field count')
            dataset,task,alpha,resp=(row[pos[x]] for x in ('dataset','task','alpha','response_time_sec'))
            population_rows[(dataset,task)]+=1
            alpha_counts[alpha]+=1
            if abs(float(alpha)-.1)>1e-12:continue
            d={k:row[pos[k]] for k in FIELDS}
            d['sample_index']=int(d['sample_index'])
            d['response_time_sec']=float(d['response_time_sec'])
            for x in ('true_crossing_bool','predicted_crossing_bool','uncertainty_intersects_geofence'):
                d[x]=b(d[x])
            d['true_tte_sec']=num(d['true_tte_sec'])
            d['predicted_tte_sec']=num(d['predicted_tte_sec'])
            d['distance_to_boundary']=num(d['distance_to_boundary'])
            all_h.append(d)
            key=tuple(d[k] for k in KEYS)
            prev=dedup.get(key)
            if prev is None or d['response_time_sec']<prev['response_time_sec']:
                dedup[key]=d
            if prev is not None:
                if any(prev[k]!=d[k] for k in ('true_crossing_bool','predicted_crossing_bool','uncertainty_intersects_geofence')):
                    duplicated_inconsistent+=1

    d=list(dedup.values())
    d.sort(key=lambda x:tuple(x[k] for k in SEQ)+(x['sample_index'],))
    groups=collections.defaultdict(list)
    for v in d:groups[tuple(v[k] for k in SEQ)].append(v)
    duplicated_index_groups=0
    gap_rows=[]
    group_gap_with_holes=0
    n2_after_gap=0
    n2_total_positive=0
    stage7f_rec=collections.defaultdict(list)
    stage7f_by_source=collections.defaultdict(list)
    # Avoid recomputing states for all response variants; duplicates were removed.
    for key,grp in groups.items():
        grp.sort(key=lambda x:x['sample_index'])
        idx=[x['sample_index'] for x in grp]
        if len(idx)!=len(set(idx)):duplicated_index_groups+=1
        gaps=[idx[j]-idx[j-1] for j in range(1,len(idx))]
        n_gaps=sum(1 for gap in gaps if gap>1)
        if n_gaps:group_gap_with_holes+=1
        if any(gap<1 for gap in gaps):raise RuntimeError(f'unsorted/duplicate indices {key}')
        gap_rows.append({**dict(zip(SEQ,key)),'n_windows':len(grp),'max_index_gap':max(gaps) if gaps else None,'gaps_gt_1':n_gaps})
        for n in (1,2,3,5):
            for j,v in enumerate(grp):
                persistent=(j+1>=n and all(grp[z]['uncertainty_intersects_geofence'] for z in range(j-n+1,j+1)))
                nominal=v['predicted_crossing_bool']; active=nominal or persistent
                stage7f_rec[(v['dataset'],v['task'],n)].append((v['true_crossing_bool'],nominal,active))
                stage7f_by_source[(v['dataset'],v['task'],v['source_group'],n)].append((v['true_crossing_bool'],nominal,active))
                if n==2 and persistent:
                    n2_total_positive+=1
                    if j>0 and idx[j]-idx[j-1]>1:n2_after_gap+=1
    output=[]
    comparisons=[]
    failed=[]
    for key,rows in sorted(stage7f_rec.items()):
        dataset,task,n=key
        truth=[q[0] for q in rows]; nominal=[q[1] for q in rows]; active=[q[2] for q in rows]
        m0=cm(truth,nominal); m=cm(truth,active)
        missed=[bool(y and not a) for y,a in zip(truth,nominal)]
        rescued=sum(1 for y,a,z in rows if y and not a and z)
        extra_true=sum(1 for y,a,z in rows if y and not a and z)
        extra_false=sum(1 for y,a,z in rows if not y and not a and z)
        row=dict(dataset=dataset,task=task,persistence_windows=n,instances=len(rows),actual_crossings=sum(truth),
             nominal_sensitivity=m0['sensitivity'],nominal_false_alert_rate=m0['false_alert_rate'],nominal_precision=m0['precision'],
             persistent_alert_sensitivity=m['sensitivity'],persistent_false_alert_rate=m['false_alert_rate'],persistent_alert_precision=m['precision'],
             nominal_missed_true_crossings=sum(missed),rescued_true_crossings=rescued,rescue_rate_among_nominal_misses=safe(rescued,sum(missed)),
             incremental_alerts_total=extra_true+extra_false,incremental_alerts_true=extra_true,incremental_alerts_false=extra_false,
             incremental_alert_precision=safe(extra_true,extra_true+extra_false),false_alerts_per_rescued_crossing=safe(extra_false,rescued),
             tp=m['tp'],fp=m['fp'],fn=m['fn'],tn=m['tn'],nominal_tp=m0['tp'],nominal_fp=m0['fp'],nominal_fn=m0['fn'],nominal_tn=m0['tn'])
        output.append(row)
        original=table.get(key)
        if not original:failed.append(('absent key',key));continue
        for column in METRICS:
            expected=float(original[column]);got=float(row[column])
            if not (math.isclose(expected,got,abs_tol=1e-12,rel_tol=1e-11) or (math.isnan(expected) and math.isnan(got))):
                failed.append((key,column,expected,got))
            comparisons.append(dict(dataset=dataset,task=task,persistence_windows=n,metric=column,original=expected,computed=got,match=math.isclose(expected,got,abs_tol=1e-12,rel_tol=1e-11)))

    selection=[]
    selected=None
    for n in (2,3,5):
        passed=True
        for task in ('2s_to_2s','2s_to_5s'):
            base=next(x for x in output if (x['dataset'],x['task'],x['persistence_windows'])==('BIRDSAI_TestReal',task,1))
            cand=next(x for x in output if (x['dataset'],x['task'],x['persistence_windows'])==('BIRDSAI_TestReal',task,n))
            reduction=1-cand['persistent_false_alert_rate']/base['persistent_false_alert_rate']
            p1=reduction>=lock['selection_rule']['minimum_relative_false_alert_reduction_vs_N1']
            p2=cand['persistent_alert_sensitivity']>=lock['selection_rule']['minimum_sensitivity']
            passed&=(p1 and p2)
            selection.append(dict(task=task,persistence_windows=n,N1_far=base['persistent_false_alert_rate'],N_far=cand['persistent_false_alert_rate'],relative_far_reduction=reduction,sensitivity=cand['persistent_alert_sensitivity'],passes_reduction=p1,passes_sensitivity=p2,passes_both=p1 and p2))
        if passed:
            selected=n
            break

    # Step-7H replay - no regrouping across different response thresholds.
    h_groups=collections.defaultdict(list)
    for v in all_h:
        hkey=tuple(v[k] for k in SEQ)+(v['response_time_sec'],)
        h_groups[hkey].append(v)
    h_stats=collections.defaultdict(lambda: collections.Counter())
    h_sources=collections.defaultdict(lambda:collections.Counter())
    h_unique_invalid=0
    h_consistency=0
    for key,grp in h_groups.items():
        grp.sort(key=lambda v:v['sample_index'])
        if len(set(z['sample_index'] for z in grp))!=len(grp):h_unique_invalid+=1
        for j,v in enumerate(grp):
            pers=j>=1 and grp[j]['uncertainty_intersects_geofence'] and grp[j-1]['uncertainty_intersects_geofence']
            nom=v['predicted_crossing_bool']
            red=(nom and math.isfinite(v['predicted_tte_sec']) and v['predicted_tte_sec']<=v['response_time_sec']) or (nom and not math.isfinite(v['predicted_tte_sec']) and math.isfinite(v['distance_to_boundary']) and v['distance_to_boundary']<=0)
            yellow=not red and (nom or pers)
            alarm=('RED' if red else ('YELLOW' if yellow else 'GREEN'))
            target=v['true_crossing_bool']
            urg=target and math.isfinite(v['true_tte_sec']) and v['true_tte_sec']<=v['response_time_sec']
            hkey=(v['dataset'],v['task'],v['response_time_sec'])
            c=h_stats[hkey]; sc=h_sources[(v['dataset'],v['task'],v['source_group'],v['response_time_sec'])]
            for ctr in (c,sc):
                ctr['instances']+=1;ctr['actual_crossings']+=int(target);ctr['urgent_actual_crossings']+=int(urg)
                ctr[alarm.lower()]+=1
                ctr['tp']+=int((red or yellow) and target)
                ctr['fp']+=int((red or yellow) and not target)
                ctr['fn']+=int(alarm=='GREEN' and target)
                ctr['tn']+=int(alarm=='GREEN' and not target)
                ctr['red_tp_urgent']+=int(red and urg)
                ctr['red_fp_nonurgent']+=int(red and not urg)
                ctr['red_total']+=int(red)
            h_consistency+=int((alarm in {'YELLOW','RED'}) != (nom or pers))

    final_rows=[]
    for key,c in sorted(h_stats.items()):
        n=c['instances'];actual=c['actual_crossings'];urgent=c['urgent_actual_crossings'];numred=c['red_total']
        final_rows.append(dict(dataset=key[0],task=key[1],response_time_sec=key[2],persistence_windows=2,conformal_level=.90,
            instances=n,actual_crossings=actual,urgent_actual_crossings=urgent,
            green_rate=safe(c['green'],n),yellow_rate=safe(c['yellow'],n),red_rate=safe(c['red'],n),
            alert_sensitivity_any_crossing=safe(c['tp'],actual),red_sensitivity_urgent_crossing=safe(c['red_tp_urgent'],urgent),
            false_alert_rate_non_crossing=safe(c['fp'],n-actual),false_red_rate_non_urgent=safe(c['red_fp_nonurgent'],n-urgent),
            alert_precision_any_crossing=safe(c['tp'],c['tp']+c['fp']),red_precision_urgent_crossing=safe(c['red_tp_urgent'],numred),
            tp=c['tp'],fp=c['fp'],fn=c['fn'],tn=c['tn'],red_tp_urgent=c['red_tp_urgent'],red_fp_nonurgent=c['red_fp_nonurgent'],red_total=numred))
    source_rows=[]
    for key,c in sorted(h_sources.items()):
        source_rows.append(dict(dataset=key[0],task=key[1],source_group=key[2],response_time_sec=key[3],instances=c['instances'],true_crossings=c['actual_crossings'],tp=c['tp'],fp=c['fp'],fn=c['fn'],tn=c['tn'],precision=safe(c['tp'],c['tp']+c['fp']),recall=safe(c['tp'],c['actual_crossings']),false_alert_rate=safe(c['fp'],c['instances']-c['actual_crossings'])))
    source_f=[]
    for key,rows in sorted(stage7f_by_source.items()):
        truth=[x[0] for x in rows]; active=[x[2] for x in rows]
        m=cm(truth,active)
        source_f.append(dict(dataset=key[0],task=key[1],source_group=key[2],persistence_windows=key[3],instances=len(rows),actual_crossings=sum(truth),tp=m['tp'],fp=m['fp'],fn=m['fn'],tn=m['tn'],sensitivity=m['sensitivity'],precision=m['precision'],false_alert_rate=m['false_alert_rate']))

    write_csv(outdir/'step7f_independent_replay.csv',output)
    write_csv(outdir/'step7f_metric_cell_comparison.csv',comparisons)
    write_csv(outdir/'step7g_selection_replay.csv',selection)
    write_csv(outdir/'step7f_per_source_replay.csv',source_f)
    write_csv(outdir/'step7h_response_aware_independent_replay.csv',final_rows)
    write_csv(outdir/'step7h_per_source_replay.csv',source_rows)
    write_csv(outdir/'forecast_index_gap_audit.csv',gap_rows)
    record={
      'input_sha256':files,
      'step7f_table_matches_json_hash':checksum_pass,
      'step7f_cell_checks':len(comparisons),'step7f_mismatches':failed,
      'selection_according_to_code':selected,'selection_in_uploaded_lock':lock['selected_persistence_windows'],
      'step7g_selection_match':selected==lock['selected_persistence_windows'],
      'full_csv_rows':sum(population_rows.values()),'primary_q90_rows_before_response_dedup':len(all_h),
      'primary_q90_unique_instances':len(dedup),'response_time_duplicate_inconsistent_labels':duplicated_inconsistent,
      'stage7f_group_count':len(groups),'stage7f_groups_duplicate_sample_idx':duplicated_index_groups,
      'stage7f_groups_with_gaps':group_gap_with_holes,
      'stage7f_n2_persistent_intersection_after_index_gap':n2_after_gap,'stage7f_n2_persistent_intersection_total':n2_total_positive,
      'step7h_group_count':len(h_groups),'step7h_groups_duplicate_index':h_unique_invalid,
      'step7h_any_alert_disagreement_with_step7f_active_logic':h_consistency,
      'original_rows_by_population':{f'{d}/{t}':n for (d,t),n in population_rows.items()},
      'source_code_methodology':'step7f_temporal_persistence_hysteresis_diagnostic.py and step7h_final_response_aware_with_locked_persistence.py (from uploaded All Python Code.zip)',
      'limitations':['Retrospective test-set threshold selection','Window-level synthetic geofence metrics','No independent untouched elephant capture group','No online detector-to-warning end-to-end replay']}
    (outdir/'phase2_machine_audit.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in record.items() if k not in ('input_sha256','original_rows_by_population','source_code_methodology','limitations')},indent=2))
    print('SELECT',selection)
    print('STAGE7F N2',*[x for x in output if x['persistence_windows']==2],sep='\n')
    print('STAGE7H PRIMARY',*[x for x in final_rows if x['response_time_sec']==1.0],sep='\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',default='/mnt/data');p.add_argument('--output-dir',default='/mnt/data/EXP4_Phase2_Report')
    a=p.parse_args();main(a.input_dir,a.output_dir)
