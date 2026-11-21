#!/usr/bin/env python3
"""One-to-one frame-level IoU50 assessment on annotated TrainReal frames.
Stdlib only. Explicitly excludes any assessment of unlabeled frames.
"""
import argparse,csv,collections,json,math
from pathlib import Path

def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y
 a1=max(0,a[2]-a[0])*max(0,a[3]-a[1]);b1=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(1e-12,a1+b1-inter)

def rect(r):return [float(r[z]) for z in ('x1','y1','x2','y2')]

def one_to_one(gt,pr):
 pairs=[]
 for i,g in enumerate(gt):
  for j,p in enumerate(pr):
   v=iou(rect(g),rect(p))
   if v>=.50:pairs.append((v,i,j))
 mgt=set();mpr=set()
 for v,i,j in sorted(pairs,reverse=True):
  if i not in mgt and j not in mpr:mgt.add(i);mpr.add(j)
 return mgt,mpr

def metrics(rows):
 t=sum(r['TP'] for r in rows);f=sum(r['FP'] for r in rows);n=sum(r['FN'] for r in rows)
 p=t/(t+f) if t+f else 0.;r=t/(t+n) if t+n else 0.;f1=2*p*r/(p+r) if p+r else 0.
 return dict(images=len(rows),GT=t+n,TP=t,FP=f,FN=n,precision=round(p,6),recall=round(r,6),F1=round(f1,6),
    false_boxes_per_annotated_image=round(f/len(rows),6) if rows else None,
    small_GT=sum(q['small_gt'] for q in rows),small_matched=sum(q['small_matched'] for q in rows),
    small_recall=round(sum(q['small_matched'] for q in rows)/sum(q['small_gt'] for q in rows),6) if sum(q['small_gt'] for q in rows)>0 else None,
    mean_wall_sec_per_annotated_image=round(sum(q['wall_sec'] for q in rows)/len(rows),5) if rows else None)

def write(out,name,rows):
 if not rows:return
 cols=list(rows[0].keys())
 with (out/name).open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(Path.home()/'CP10_Stage7_Dev_Pilot'))
 args=ap.parse_args();out=Path(args.out).expanduser().resolve()
 pre=json.loads((out/'Stage7_Preflight_Manifest.json').read_text());prov=json.loads((out/'Stage7_Dev_Run_Provenance.json').read_text())
 assert pre['status']=='READY_TRAINREAL_DEV_ONLY' and prov['role']=='TRAINREAL_DEVELOPMENT_ONLY' and prov['testreal_used'] is False
 assert prov['pilot_complete'],'Incomplete model run: do not evaluate partial configurations'
 manifest=read(out/'Stage7_Dev_Image_Manifest.csv');gt=collections.defaultdict(list);det=collections.defaultdict(list);times={}
 for r in read(out/'Stage7_Dev_GT_Boxes.csv'):gt[(r['sequence'],r['frame'])].append(r)
 for r in read(out/'Stage7_Dev_Detections.csv'):det[(r['config'],r['sequence'],r['frame'])].append(r)
 for r in read(out/'Stage7_Dev_Inference_Times.csv'):times[(r['config'],r['sequence'],r['frame'])]=r
 configs=list(prov['configurations']);total=len(manifest)
 for c in configs:
  assert sum((c,r['sequence'],r['frame']) in times for r in manifest)==total, 'Missing config images '+c
 images=[]
 for c in configs:
  for m in manifest:
   seq,fr=m['sequence'],m['frame'];g=gt[(seq,fr)];p=det[(c,seq,fr)]
   assigned,matched_pred=one_to_one(g,p)
   # Image area recovered from detection if present; otherwise read image dims by PIL.
   # Predeclared small-box cutoff is <0.5% of source-image area.
   if p:
    w=float(p[0]['image_width']);h=float(p[0]['image_height'])
   else:
    try:
     from PIL import Image
     with Image.open(m['image_path']) as im:w,h=im.size
    except Exception:
     w=h=None
   small=[i for i,one in enumerate(g) if w and h and (float(one['x2'])-float(one['x1']))*(float(one['y2'])-float(one['y1']))/(w*h)<0.005]
   images.append({'config':c,'sequence':seq,'frame':fr,'TP':len(assigned),'FP':len(p)-len(matched_pred),'FN':len(g)-len(assigned),
    'gt':len(g),'small_gt':len(small),'small_matched':len(set(small)&assigned),
    'wall_sec':float(times[(c,seq,fr)]['wall_sec'])})
 overall=[];perseq=[]
 for c in configs:
  m=metrics([r for r in images if r['config']==c]);overall.append({'config':c,**m})
  for seq in sorted(set(r['sequence'] for r in images)):
   z=[r for r in images if r['config']==c and r['sequence']==seq]
   if z:perseq.append({'config':c,'sequence':seq,**metrics(z)})
 write(out,'Stage7_Dev_PerImage_IoU50.csv',images)
 write(out,'Stage7_Dev_PerSequence_IoU50.csv',perseq)
 write(out,'Stage7_Dev_Overall_IoU50.csv',overall)
 baseline=next((r for r in overall if r['config']=='baseline_640_conf010'),None)
 if baseline is None: raise SystemExit('baseline_640_conf010 mandatory for paired comparison')
 viable=[]
 for r in overall:
  if r['config']=='baseline_640_conf010':continue
  if (r['recall']>=baseline['recall']+.03 and
      r['precision']>=max(0.10,baseline['precision']-.10) and
      r['false_boxes_per_annotated_image']<=baseline['false_boxes_per_annotated_image']+.5):
   viable.append(r)
 # Policy is fixed before looking at Dev outcomes. Pick only if all gates met.
 if viable:
  best=sorted(viable,key=lambda z:(-z['recall'],z['false_boxes_per_annotated_image'],z['mean_wall_sec_per_annotated_image']))[0]
  decision='DEV_CANDIDATE_PASSED_PREDECLARED_GATES_NOT_TESTED'
 else:
  best=baseline;decision='NO_DEV_CANDIDATE_PASSED_GATES_RETAIN_BASELINE'
 chosen={'decision':decision,'selected_config':best['config'],'selected_settings':prov['configurations'][best['config']],
  'scope':'development-only / annotated TrainReal frames, not TestReal',
  'selection_rule':'Recall gain >=3 percentage points; precision >=max(0.10,baseline_precision-0.10); false boxes/annotated image <= baseline+0.5; maximize recall, tie FP then latency.',
  'testreal_tuning':False,'testreal_evaluation_complete':False,
  'experimental_claim':'No end-to-end benefit established from dev-only inference'}
 (out/'Stage7_Dev_Selected_Configuration_PRETEST.json').write_text(json.dumps(chosen,indent=2),encoding='utf8')
 lines=['STAGE 7 — TRAINREAL DEVELOPMENT-ONLY DETECTOR PILOT',
  'This is NOT a BIRDSAI TestReal performance result and not a tracker/forecast evaluation.','',
  f'Annotated images: {total}; original R13 checkpoint hash verified.',
  'Small elephant defined as GT bounding-box area <0.5% of full image area.',
  'Precision/false boxes are CONDITIONAL ON annotated positive frames; these are not fully negative-image alarms.','']
 for r in overall:lines.append(f"{r['config']}: recall={r['recall']:.4f}, precision={r['precision']:.4f}, F1={r['F1']:.4f}, FP/annotated image={r['false_boxes_per_annotated_image']:.4f}, wall={r['mean_wall_sec_per_annotated_image']:.3f}s")
 lines+=['',f'Decision: {decision}; candidate={best["config"]}',
  'Next: freeze this decision; run on original R13 TestReal ONCE; measure detection, tracking IDF1/HOTA, 2s/5s track-derived windows.',
  'Do not claim improvement from pilot alone.']
 (out/'Stage7_Dev_Decision.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print('\n'.join(lines))

if __name__=='__main__':main()
