
from pathlib import Path
import pandas as pd, json, hashlib
from datetime import datetime, timezone

D=Path(r"C:\Users\HP\Downloads")
C=D/"paper_strengthening_revision"/"R6_direct_crossing_classifier"/"R6C_oof_threshold_lock_and_frozen_evaluation"
B=D/"paper_strengthening_revision"/"R6_direct_crossing_classifier"/"R6D_paired_track_cluster_bootstrap"
O=D/"paper_strengthening_revision"/"R6_direct_crossing_classifier"/"R6E_interpretation_claim_lock"
O.mkdir(parents=True,exist_ok=True)

CM=C/"R6C_MANIFEST.json"
CC=C/"R6C_DIRECT_CLASSIFIER_VS_NOMINAL_BASELINE.csv"
BM=B/"R6D_MANIFEST.json"
BR=B/"R6D_TRACK_CLUSTER_BOOTSTRAP_RESULTS.csv"

def h(p):
    x=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): x.update(b)
    return x.hexdigest()

for p in [CM,CC,BM,BR]:
    if not p.exists(): raise FileNotFoundError(p)

cm=json.loads(CM.read_text(encoding="utf-8"))
bm=json.loads(BM.read_text(encoding="utf-8"))
if cm.get("status")!="R6C_VALIDATION_ONLY_THRESHOLD_LOCK_AND_FROZEN_DIRECT_CLASSIFIER_EVALUATION_COMPLETE_READY_FOR_R6D_INTERPRETATION":
    raise RuntimeError("Unexpected R6C status")
if bm.get("status")!="R6D_PAIRED_TRACK_CLUSTER_BOOTSTRAP_COMPLETE_READY_FOR_R6E_CLAIM_LOCK":
    raise RuntimeError("Unexpected R6D status")
if bm.get("results_sha256") and h(BR)!=bm["results_sha256"]:
    raise RuntimeError("R6D result hash mismatch")

comp=pd.read_csv(CC)
boot=pd.read_csv(BR)

def brow(ds,task,metric):
    q=boot[(boot.dataset==ds)&(boot.task==task)&(boot.metric==metric)]
    if len(q)!=1: raise RuntimeError(f"Missing bootstrap row: {ds} {task} {metric}")
    return q.iloc[0]

rows=[]
for _,r in comp.iterrows():
    ds,task=str(r.dataset),str(r.task)
    ba=brow(ds,task,"balanced_accuracy_delta")
    se=brow(ds,task,"sensitivity_delta")
    fp=brow(ds,task,"false_positive_rate_delta")
    pr=brow(ds,task,"precision_delta")
    f1=brow(ds,task,"F1_delta")
    rows.append({
        "dataset":ds,"task":task,"rows":int(r["rows"]),
        "balanced_accuracy_delta":float(r.balanced_accuracy_delta),
        "balanced_accuracy_ci95_low":float(ba.ci95_low),
        "balanced_accuracy_ci95_high":float(ba.ci95_high),
        "sensitivity_delta":float(r.sensitivity_delta),
        "sensitivity_ci95_low":float(se.ci95_low),
        "sensitivity_ci95_high":float(se.ci95_high),
        "false_positive_rate_delta":float(r.false_positive_rate_delta),
        "false_positive_rate_ci95_low":float(fp.ci95_low),
        "false_positive_rate_ci95_high":float(fp.ci95_high),
        "precision_delta":float(r.direct_precision-r.baseline_precision),
        "precision_ci95_low":float(pr.ci95_low),
        "precision_ci95_high":float(pr.ci95_high),
        "F1_delta":float(r.direct_F1-r.baseline_F1),
        "F1_ci95_low":float(f1.ci95_low),
        "F1_ci95_high":float(f1.ci95_high),
        "direct_AUROC":float(r.direct_AUROC),
        "direct_AUPRC":float(r.direct_AUPRC),
    })

summary=pd.DataFrame(rows)

claims=pd.DataFrame([
["R6E_CLAIM_01","SUPPORTED","Direct-classifier thresholds were selected only from track-disjoint validation OOF predictions; frozen evaluation was not used for threshold reselection.","R6B3/R6C protocol lock."],
["R6E_CLAIM_02","SUPPORTED","The direct classifier improves balanced accuracy across TestReal and Fresh WildBox at both horizons.","All four track-cluster 95% CIs for balanced-accuracy delta are above zero."],
["R6E_CLAIM_03","SUPPORTED","The direct classifier substantially improves crossing sensitivity across all four evaluation conditions.","All four sensitivity-delta 95% CIs are above zero."],
["R6E_CLAIM_04","SUPPORTED","The sensitivity gain has a false-positive cost: false-positive rate increases across all four conditions.","All four FPR-delta 95% CIs are above zero."],
["R6E_CLAIM_05","SUPPORTED","The direct classifier is less precise than the nominal crossing rule in all four conditions.","All four precision-delta 95% CIs are below zero."],
["R6E_CLAIM_06","SUPPORTED_WITH_LIMITATION","F1 improves on both TestReal horizons and Fresh WildBox 5s, but Fresh WildBox 2s F1 improvement is uncertain.","Fresh 2s F1 CI crosses zero; the other three exclude zero."],
["R6E_CLAIM_07","SUPPORTED_WITH_LIMITATION","The direct classifier provides a useful higher-sensitivity operating point for virtual-geofence crossing detection.","Balanced accuracy/sensitivity rise, but FPR rises and precision falls."],
["R6E_CLAIM_08","NOT_SUPPORTED","The direct classifier is universally better than the nominal crossing rule.","Trade-offs remain: higher FPR and lower precision in every condition."],
["R6E_CLAIM_09","NOT_SUPPORTED","The direct classifier reduces false alarms relative to the nominal rule.","Observed FPR deltas are positive with CIs above zero."],
["R6E_CLAIM_10","METHOD_NOTE","Report the direct classifier as a secondary decision-layer benchmark/ablation, not a replacement for primary Step7H RED/YELLOW/GREEN logic.","Step7H remains the locked primary operational protocol."],
],columns=["claim_id","status","claim","basis"])

sp=O/"R6E_condition_interpretation.csv"
cp=O/"R6E_CLAIM_LOCK.csv"
mp=O/"R6E_manuscript_direct_classifier_table.csv"
summary.to_csv(sp,index=False)
claims.to_csv(cp,index=False)
summary.to_csv(mp,index=False)

status="R6E_DIRECT_CROSSING_CLASSIFIER_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R7_WARNING_HIERARCHY"
manifest={
"step":"R6E_DIRECT_CROSSING_CLASSIFIER_INTERPRETATION_CLAIM_LOCK",
"created_utc":datetime.now(timezone.utc).isoformat(),
"status":status,
"retraining":False,"threshold_reselection":False,"new_statistical_test":False,
"source_r6c":str(CC),"source_r6c_sha256":h(CC),
"source_r6d":str(BR),"source_r6d_sha256":h(BR),
"claim_lock_csv":str(cp),"claim_lock_sha256":h(cp),
"manuscript_table_csv":str(mp)
}
(O/"R6E_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

print("\nR6-E — DIRECT CROSSING CLASSIFIER INTERPRETATION + CLAIM LOCK")
print("\nCONDITION-LEVEL INTERPRETATION:")
print(summary.to_string(index=False))
print("\nCLAIM LOCK:")
print(claims.to_string(index=False))
print("\nSTATUS:")
print(status)
print("\nR6-E COMPLETE")
print("Outputs:",O)
