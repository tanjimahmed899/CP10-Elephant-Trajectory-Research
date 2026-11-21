#!/usr/bin/env python3
"""Recompute prior Stage7A development acceptance gates from uploaded CSV."""
import csv,json,zipfile,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--out',required=True);a=p.parse_args()
with zipfile.ZipFile(a.review) as z:
 vals=list(csv.DictReader(z.read('Stage7_Dev_Overall_IoU50.csv').decode().splitlines()))
 prov=json.loads(z.read('Stage7_Dev_Run_Provenance.json'))
 pre=json.loads(z.read('Stage7_Preflight_Manifest.json'))
 assert prov['pilot_complete'] and not prov['testreal_used'] and pre['checkpoint']['original_r13_checkpoint_match']
 base=next(r for r in vals if r['config']=='baseline_640_conf010')
 for v in vals:
  tp=int(v['TP']);fp=int(v['FP']);fn=int(v['FN']);
  assert tp+fn==521 and int(v['images'])==120
  assert abs(tp/(tp+fn)-float(v['recall']))<1e-6
  assert abs(tp/(tp+fp)-float(v['precision']))<1e-6
  assert abs(fp/120-float(v['false_boxes_per_annotated_image']))<1e-6
 for v in vals:
  v['delta_recall_pp']=round(100*(float(v['recall'])-float(base['recall'])),3)
  v['delta_fp_per_image']=round(float(v['false_boxes_per_annotated_image'])-float(base['false_boxes_per_annotated_image']),3)
  v['passes_recall_gain']=float(v['recall'])>=float(base['recall'])+.03 if v is not base else None
  v['passes_precision_gate']=float(v['precision'])>=max(.10,float(base['precision'])-.10) if v is not base else None
  v['passes_fp_gate']=float(v['false_boxes_per_annotated_image'])<=float(base['false_boxes_per_annotated_image'])+.5 if v is not base else None
  v['passes_all_predeclared_gates']=all((v['passes_recall_gain'],v['passes_precision_gate'],v['passes_fp_gate'])) if v is not base else None
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/'Stage7A_Independent_Recalculation.json').write_text(json.dumps({'source_review':str(a.review),'verified_480_config_frame_runs':prov['total_frame_config_runs'],
   'original_r13_checkpoint_sha256':pre['checkpoint']['sha256'],'testreal_tuning':False,'configurations':vals},indent=2))
 print('Independent Stage7A aggregate check PASS. New configs passing all gates:',sum(bool(v['passes_all_predeclared_gates']) for v in vals))
