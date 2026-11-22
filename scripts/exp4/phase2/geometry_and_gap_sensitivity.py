"""Stage-7B algebraic row-consistency and nonconsecutive sample-index sensitivity checks.
Diagnostic only; does not establish real-time physical spacing, field event validity, or novelty.
"""
import argparse,csv,collections,math,json
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--input-dir',default='/mnt/data')
parser.add_argument('--output-dir',default='/mnt/data/EXP4_Phase2_Report')
args=parser.parse_args()
p=Path(args.input_dir)/'step7b_v2_all_geofence_instance_alarms.csv'
out=Path(args.output_dir)
out.mkdir(parents=True,exist_ok=True)
q={('2s_to_2s','0.1'):2.158768,('2s_to_2s','0.05'):2.560773,('2s_to_5s','0.1'):1.975130,('2s_to_5s','0.05'):2.950976}
issues=collections.Counter();counts=collections.Counter();independent=collections.defaultdict(collections.Counter);unique={}
seqcols=('dataset','task','source_group','track_key','axis','boundary','direction','boundary_id')
keycols=('dataset','task','sample_index','source_group','track_key','axis','boundary','direction','boundary_id')
with p.open(newline='',encoding='utf8') as f:
 for r in csv.DictReader(f):
  t=r['task'];a=r['alpha'];ds=r['dataset']
  expected=float(r['scale'])*q[t,a]
  stored=float(r['absolute_conformal_radius'])
  if not math.isclose(expected,stored,abs_tol=1e-12,rel_tol=1e-10): issues['radius']+=1
  risk=float(r['risk_score']);oris=float(r['oriented_risk_score']);uris=float(r['uncertainty_risk_score'])
  if not (math.isclose(oris,risk,abs_tol=1e-12,rel_tol=1e-10) or math.isclose(oris,-risk,abs_tol=1e-12,rel_tol=1e-10)):issues['oriented_score']+=1
  if not math.isclose(uris,oris+stored,abs_tol=1e-12,rel_tol=1e-10):issues['uncertainty_score']+=1
  intersects=(uris>=0)
  if intersects!=(r['uncertainty_intersects_geofence']=='True'):issues['uncertainty_intersection']+=1
  pc=r['predicted_crossing_bool']=='True';resp=float(r['response_time_sec']);d=float(r['distance_to_boundary']);tte=float(r['predicted_tte_sec']) if r['predicted_tte_sec'] else math.nan
  red=(pc and math.isfinite(tte) and tte<=resp) or (pc and not math.isfinite(tte) and math.isfinite(d) and d<=0)
  yellow=(not red) and (intersects or pc)
  alarm='RED' if red else ('YELLOW' if yellow else 'GREEN')
  if alarm!=r['alarm_state']:issues['alarm_state']+=1
  counts[(ds,t,a)]+=1
  if a=='0.1':
   k=tuple(r[x] for x in keycols)
   if k not in unique:unique[k]=(r['true_crossing_bool']=='True',pc,intersects,int(r['sample_index']),tuple(r[z] for z in seqcols))
# Compare two successive available windows vs strict sample_index adjacency only.
groups=collections.defaultdict(list)
for v in unique.values():groups[v[4]].append(v)
for group,rows in groups.items():
 rows.sort(key=lambda x:x[3]);ds,task=group[:2]
 for j,(truth,nom,u,idx,g) in enumerate(rows):
  prev=(j>0 and rows[j-1][2]);pers=u and prev
  adj=pers and idx-rows[j-1][3]==1
  key=(ds,task)
  for name,alert in [('original',nom or pers),('strict_index_adjacency',nom or adj)]:
   c=independent[key+(name,)];c['n']+=1;c['pos']+=truth;c['tp']+=bool(alert and truth);c['fp']+=bool(alert and not truth);c['fn']+=bool(not alert and truth);c['tn']+=bool(not alert and not truth)
  if pers and not adj:independent[key+('removed_after_gap',)]['count']+=1;independent[key+('removed_after_gap',)]['true_count']+=int(truth);independent[key+('removed_after_gap',)]['nominal_true_count']+=int(nom)
rows=[]
for (dataset,task,approach),c in sorted(independent.items()):
 if approach=='removed_after_gap':
  rows.append(dict(dataset=dataset,task=task,approach=approach,affected_windows=c['count'],true_crossing_windows=c['true_count'],already_nominal_alert_windows=c['nominal_true_count']))
 else:
  rows.append(dict(dataset=dataset,task=task,approach=approach,instances=c['n'],actual_crossings=c['pos'],tp=c['tp'],fp=c['fp'],fn=c['fn'],tn=c['tn'],sensitivity=c['tp']/c['pos'] if c['pos'] else '',far=c['fp']/(c['n']-c['pos']) if c['n']!=c['pos'] else '',precision=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else ''))
with (out/'index_gap_counterfactual.csv').open('w',newline='') as f:
 keys=list(dict.fromkeys(k for row in rows for k in row.keys()));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
result={'checked_stage7b_rows':sum(counts.values()),'row_level_formula_mismatches':dict(issues),'q_levels_used':{','.join(k):v for k,v in q.items()},'q_note':'Step-7B table uses q values rounded to six decimal places; original Step-6C calibration stores higher precision.','unique_q90_instances':len(unique),'counterfactual':rows}
(out/'stage7b_formula_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
