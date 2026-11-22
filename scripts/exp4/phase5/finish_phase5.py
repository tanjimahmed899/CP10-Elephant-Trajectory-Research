#!/usr/bin/env python3
"""Secondary report generation from Phase 5 policy-grid rows. No retraining."""
from pathlib import Path
import csv,json,math,hashlib,zipfile,datetime
D=Path(__file__).parent

def load(n):return list(csv.DictReader((D/n).open(newline='',encoding='utf8')))
def outcsv(name,rows):
 with (D/name).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def pct(v):return f'{float(v)*100:.2f}%'
pol=load('phase5_policy_grid.csv');sources=load('phase5_policy_by_source.csv')
# Explicit post-hoc horizon-specific criterion: least FPR among available-window policies having >=0.80 recall on BIRDSAI TestReal.
choices=[]
for task in ['2s_to_2s','2s_to_5s']:
 a=[r for r in pol if r['dataset']=='BIRDSAI_TestReal' and r['task']==task and r['policy'].startswith('clearance_') and r['policy'].endswith('_available') and float(r['recall'])>=0.8]
 if not a:raise AssertionError(f'No feasible exploratory choice for {task}')
 win=min(a,key=lambda r:float(r['false_alert_rate']))
 choices.append(dict(task=task,policy=win['policy'],development_recall=win['recall'],development_fpr=win['false_alert_rate']))
rows=[];source_rows=[]
for ds in ['BIRDSAI_TestReal','WildBox_Fresh']:
 for task in ['2s_to_2s','2s_to_5s']:
  chosen=next(x['policy'] for x in choices if x['task']==task)
  for method in ['original_N2_available',chosen,'original_N2_adjacent']:
   r=next(x for x in pol if x['dataset']==ds and x['task']==task and x['policy']==method)
   rows.append(dict(dataset=ds,task=task,policy=method,windows=r['instances'],actual_crossings=r['positive'],tp=r['tp'],fp=r['fp'],fn=r['fn'],tn=r['tn'],recall=r['recall'],false_alert_rate=r['false_alert_rate'],precision=r['precision'],issued_alerts=r['issued_alerts'],f1=r['f1']))
   source_rows.extend(dict(dataset=r['dataset'],task=r['task'],source_group=r['source_group'],policy=r['policy'],n=r['instances'],positive=r['positive'],tp=r['tp'],fp=r['fp'],recall=r['recall'],false_alert_rate=r['false_alert_rate'],precision=r['precision']) for r in sources if r['dataset']==ds and r['task']==task and r['policy']==method)
outcsv('phase5_horizon_specific_selection.csv',choices)
outcsv('phase5_selected_comparison.csv',rows)
outcsv('phase5_selected_by_source.csv',source_rows)
# Check no policy exceeds nominal baseline on sensitivity (should be nested alarm set). Verified counts.
for ds in ['BIRDSAI_TestReal','WildBox_Fresh']:
 for task in ['2s_to_2s','2s_to_5s']:
  a=next(x for x in rows if x['dataset']==ds and x['task']==task and x['policy']=='original_N2_available')
  chosen=next(x['policy'] for x in choices if x['task']==task)
  b=next(x for x in rows if x['dataset']==ds and x['task']==task and x['policy']==chosen)
  for k in ['tp','fp']:
   assert int(b[k])<=int(a[k])
report='''# EXP4 Phase 5 — Exploratory Clearance-Gated Virtual-Boundary Warnings

**Status:** Retrospective *decision-layer feasibility study*, not independently validated research novelty. Original datasets, model predictions, Step6C q90 calibration and N=2 historical baseline are unmodified. Date: 22 November 2026.\n\n## Experimental question

Can a simple *predicted clearance to uncertainty radius* gate lower the virtual-boundary warning burden while keeping predefined recall on the already studied BIRDSAI TestReal set? This experiment does **not** solve 5-second conformal undercoverage.

## Method

For each forecast at 90% nominal conformal level, let `z` be the existing signed maximum penetration of the **nominal predicted path** towards the directional synthetic boundary. Positive `z` signals a nominal crossing. Let `r = q90 * history_scale` be the original frozen uncertainty radius. Define `u = 1[z+r >= 0]` (the stored `uncertainty_intersects_geofence` must also be true). A clearance-gated uncertainty candidate is `u_rho = u * 1[z >= -rho*r]`. For one track/boundary/direction ordered by `sample_index`, N=2 flags an uncertainty warning only if `u_rho` is true for two consecutive *available forecast rows*. The final binary alert is `nominal_crossing OR persistent(u_rho)`. Existing nominal crossing alerts are never suppressed. We examine rho = 0.25, 0.50, 0.75, and 1.00 (1.00 is original N=2), plus index-adjacency sensitivity (`sample_index` difference must equal one) as a separate counterfactual. All gates are based on stored predictions and uncertainty radii, not future labels. The truth labels appear only in evaluation and retrospective policy selection.

**Evidence units:** synthetic geofence-instance rows, *not* independent elephant incidents or physical notification episodes. One 90% branch with response-time duplicates removed. The four dataset × horizon evaluation populations have 6,318 / 4,956 / 5,190 / 3,822 instances respectively.

## Reproducibility and baseline check

* Source: `step7b_v2_all_geofence_instance_alarms.csv`, q90 branch.
* Input rows = 214,956; q90 rows before dedup = 107,478; unique q90 instances = 20,286; ordered track-boundary sequences = 2,123.
* Source script verified against original `step7f_temporal_persistence_hysteresis_diagnostic.py`.
* All 20 baseline confusion-count regression checks exactly reproduce the previously audited original Step7F table (0 mismatches).

## Primary fixed-grid observations

| Population | Horizon | Policy | Recall | False-alert rate | Precision | TP | FP |
|---|---|---|---:|---:|---:|---:|---:|
'''
for ds in ['BIRDSAI_TestReal','WildBox_Fresh']:
 for task in ['2s_to_2s','2s_to_5s']:
  for method in ['original_N2_available','clearance_rho_0.75_available','clearance_rho_1_available']:
   if method=='clearance_rho_1_available':continue
   r=next(x for x in pol if x['dataset']==ds and x['task']==task and x['policy']==method)
   report+=f"| {ds} | {'2 s' if task=='2s_to_2s' else '5 s'} | {method} | {pct(r['recall'])} | {pct(r['false_alert_rate'])} | {pct(r['precision'])} | {r['tp']} | {r['fp']} |\n"
report+='''\n## Horizon-specific exploratory selection and already-inspected transfer

At both horizons, independently choose the smallest false-alert rate among available-window clearance policies meeting a minimum 80% recall *on BIRDSAI TestReal*. This is **post-hoc** model selection on a previously studied test partition; it cannot be treated as pristine development/testing, nor does WildBox Fresh become a clean confirmatory test by replaying the chosen setting.\n\n'''
for c in choices:report+=f"* {('2 s' if c['task']=='2s_to_2s' else '5 s')}: `{c['policy']}` selected; BIRDSAI recall {pct(c['development_recall'])}, FPR {pct(c['development_fpr'])}.\n"
report+='''\n| Population | Horizon | Original N=2 (Recall, FPR, Precision) | Selected (Recall, FPR, Precision) | Change in FP | Change in TP |
|---|---|---|---|---:|---:|
'''
for ds in ['BIRDSAI_TestReal','WildBox_Fresh']:
 for task in ['2s_to_2s','2s_to_5s']:
  original=next(r for r in rows if r['dataset']==ds and r['task']==task and r['policy']=='original_N2_available')
  method=next(c['policy'] for c in choices if c['task']==task)
  new=next(r for r in rows if r['dataset']==ds and r['task']==task and r['policy']==method)
  trip=lambda r:', '.join(pct(r[k]) for k in ['recall','false_alert_rate','precision'])
  report+=f"| {ds} | {'2 s' if task=='2s_to_2s' else '5 s'} | {trip(original)} | {trip(new)} | {int(new['fp'])-int(original['fp']):+} | {int(new['tp'])-int(original['tp']):+} |\n"
report+='''\n**Interpretation:** With the stated >=80% criterion on previously studied BIRDSAI data, the 2-second gate trades some true crossing windows for fewer false warning windows. At the 5-second horizon the only eligible grid configuration is the original rho=1.00, so **no warning improvement** is demonstrated there. The observed 2-second trade-off transfers descriptively to already studied WildBox Fresh, with a recall loss; these are highly correlated image-plane boundary instances, not independent field alarms. Index-adjacent counterfactuals are included for timing sensitivity only; `sample_index` adjacency does not establish measured elapsed-time continuity.\n\n## Major limitations\n
1. No source-disjoint, previously untouched elephant recording group. TestReal was used historically for operating point selection and cannot be reused for confirmatory significance.
2. Each trajectory can contribute several overlapping forecast windows and six synthetic line cases; report counts and clustered/source-unit comparisons, not independent-row p-values.
3. This is a thresholded warning **decision**, not a new conformal calibration; existing 5-second 69.76% EXP3 pathwise coverage and BAMBI undercoverage remain unchanged.
4. Virtual boundary and labels are in camera-relative image-plane coordinates (Step7B), not the fixed segment-reference Stage6 events. A saved Stage6 fixed-reference **per-instance** truth/prediction table is still needed for source-exact fixed-reference warning testing.
5. Response-time-aware RED/YELLOW/ GREEN Step7H thresholds were not changed; the candidate alters the combined alert indicator only.
6. New data collection and a frozen protocol are required before an operational or Q1-methodological success claim.

## Next gates

1. **Method gate:** separate calibration scores from test groups; group-robust or regime-aware calibration should be evaluated on the same original track-scale score, with explicit finite-sample and source-shift caveats.
2. **Geometry gate:** recover per-instance fixed-reference case table and per-forecast camera transforms to recompute event-level warning burden with a physically meaningful temporal basis.
3. **Confirmation gate:** freeze one horizon-specific rule and thresholds before collecting/viewing independent elephant recording groups; log detection→tracking→forecastability yield and capture-group denominators; then evaluate coverage, radius, recall, alerts per monitored hour and latency under the new recordings.

**Novelty decision:** the tested clearance rule is an elementary risk-threshold heuristic; the data do not yet establish a novel Q1-quality method, a conformal coverage guarantee, or reliable field alerting.
'''
(D/'EXP4_PHASE5_EXPLORATORY_REPORT.md').write_text(report,encoding='utf8')
(D/'README.md').write_text('''# Phase 5 Research Package\n\nSource-exact exploratory study, without modifying the original project.\n\n## Run\n\n```bash\npython run_phase5.py --source path/to/step7b_v2_all_geofence_instance_alarms.csv --output-dir ./outputs\n```\n\nThis script expects `/mnt/data/EXP4_Phase2_Report/step7f_independent_replay.csv` for exact prior benchmark regression (currently hard-coded to the provided Project audit location). Copy that baseline or update this path when transferring to Windows.\n\n`finish_phase5.py` generates post hoc horizon-specific comparison and textual report. It must run from a directory with `phase5_policy_grid.csv` and `phase5_policy_by_source.csv`.\n\nThis package omits raw video, trained checkpoints, large source CSV, and prior scripts; original data were read-only. No novel validated method claims.\n''',encoding='utf8')
print(report[report.index('## Horizon-specific exploratory selection'):report.index('## Major limitations')])
