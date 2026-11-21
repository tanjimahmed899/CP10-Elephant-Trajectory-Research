from pathlib import Path
import hashlib, json
from datetime import datetime, timezone
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
BASE = DOWNLOADS / "step6d_uncertainty_holdout" / "bambi_flight6" / "step6dk_georef" / "k3_operational_warning"
OUT = DOWNLOADS / "paper_strengthening_revision" / "R1_protocol_lock"
OUT.mkdir(parents=True, exist_ok=True)

FROZEN = {
    "K3C10_prelock": (BASE/"step7_external_prelock"/"STEP6D_K3C10_BAMBI_STEP7_PRELOCK.json", "8cc687c3f91ecbfb2a3c0f491427d91ccfdca959659a52e0836a98889c890bd1"),
    "K3C13_canonical": (BASE/"anchor_normalized_canonical_lock_v1"/"STEP6D_K3C13_CANONICAL_DATASET_LOCK.json", "71d83c774233538342f4b410a789750943e61a8513fe983249c8f8f5a2aee605"),
    "K3C15_predictions": (BASE/"secondary_frozen_inference_v2"/"STEP6D_K3C15_SECONDARY_FROZEN_INFERENCE_LOCK.json", "06a863cf9d86d1cf9faa3416d7fbfe41f0eadcb3d9ffba25845e7cf801a65c33"),
    "K3C16_conformal": (BASE/"secondary_fixed_conformal_audit_v3"/"STEP6D_K3C16_SUMMARY.json", "21685938c4995476cd6a45acca525c800a30ff12913690b059e92f16506e3104"),
    "K3C17_step7": (BASE/"frozen_step7_external_evaluation_v1"/"STEP6D_K3C17_FROZEN_STEP7_EXTERNAL_EVALUATION_MANIFEST.json", "5f89576a3d27114b7a8d599c6a2c3a7f88574777a0d2dae810420b7a862d22ab"),
    "K3C18_final": (BASE/"final_external_pipeline_consolidation"/"STEP6D_K3C18_FINAL_EXTERNAL_PIPELINE_SUMMARY.json", "b2c7ff95095c19dbe564b239c20b8631820b962321fc84cc6d28c8277fb41050"),
}

def sha256_file(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def verify_frozen():
    rows=[]
    for name,(path,expected) in FROZEN.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual=sha256_file(path)
        ok=actual==expected
        rows.append({"artifact":name,"path":str(path),"expected_sha256":expected,"actual_sha256":actual,"match":ok})
        if not ok:
            raise RuntimeError(f"Frozen integrity failed: {name}\nExpected {expected}\nActual {actual}")
    return pd.DataFrame(rows)

PROTOCOL = {
  "protocol":"CP1.0 Paper Strengthening Revision Protocol",
  "version":"R1",
  "created_utc":datetime.now(timezone.utc).isoformat(),
  "purpose":"Close reviewer-facing weaknesses without modifying the K3C18 frozen snapshot.",
  "immutable_rules":[
    "Never overwrite K3C10-K3C18 artifacts.",
    "Never tune using BIRDSAI TestReal, Fresh WildBox, or BAMBI outcomes.",
    "Keep negative results.",
    "Any change after R1 must be a new prelocked amendment before test results are inspected."
  ],
  "paper_framing":"Rigorous moving-UAV elephant trajectory benchmark: simple baselines may be strong; learned gains are task-dependent; short-horizon transfer is more reliable; uncertainty and long-horizon transfer may fail.",
  "warning_hierarchy":{
    "primary":"Step7H GREEN/YELLOW/RED response-aware warning",
    "secondary":"Step18 binary crossing benchmark / ablation",
    "RED":"actionable/selective warning",
    "YELLOW":"watch / uncertainty flag",
    "GREEN":"no warning"
  },
  "R2_fair_forecast_benchmark":{
    "datasets":{
      "BIRDSAI_TestReal":{"2s_to_2s":{"samples":1053,"tracklets":89,"sources":6},"2s_to_5s":{"samples":826,"tracklets":63,"sources":5}},
      "Fresh_WildBox":{
        "locked_sources":[
          "fresh_01 | elep1 | DJI_20240119152335_0005_V.zip",
          "fresh_02 | elep2 | DJI_20240626142859_0001_V.zip",
          "fresh_03 | elep3 | DJI_20260227084135_0002_V.zip"],
        "2s_to_2s":{"samples":865,"tracklets":78,"sources":3},
        "2s_to_5s":{"samples":637,"tracklets":71,"sources":3}
      }
    },
    "models":["Stationary","Constant Velocity","Kalman CV","GRU","LSTM","Transformer seed42","Transformer seed52","Transformer seed62","Transformer 3-seed ensemble"],
    "metrics":["ADE","FDE","track-macro ADE/FDE","source-macro ADE/FDE"]
  },
  "R3_clustered_bootstrap":{
    "replicates":5000,"seed":20261029,
    "primary":"paired track-clustered 95% percentile CI",
    "secondary":"paired source-clustered 95% CI / sensitivity analysis",
    "paired_reference":"Stationary",
    "extra_pairs":["LSTM vs Transformer ensemble","GRU vs Transformer ensemble","CV vs Transformer ensemble"],
    "ranking_rule":"No superiority claim if paired 95% CI for metric difference includes zero."
  },
  "R4_fixed_conformal":{
    "no_recalibration":True,
    "q":{"2s_to_2s":{"q90":2.158767888280567,"q95":2.560772554862189},"2s_to_5s":{"q90":1.975129668255982,"q95":2.950976253259077}},
    "evaluate_on":["BIRDSAI TestReal","Fresh WildBox","retain BAMBI as frozen cross-domain stress test"],
    "report":["window coverage","track-macro coverage","fully-covered-track fraction","stepwise coverage","endpoint coverage","radius distribution"]
  },
  "R5_R6_geofence_baselines":{
    "models":["distance-only current-position score","Stationary future","CV extrapolation","direct crossing classifier from 20-frame history","Transformer ensemble"],
    "continuous_metrics":["AUROC","AUPRC"],
    "validation_only_operating_points":{
      "Balanced":"maximize F1 on validation",
      "Safety":"threshold achieving >=90% validation recall if feasible; otherwise max recall",
      "Conservative":"threshold achieving >=90% validation precision if feasible; otherwise max precision"},
    "freeze_rule":"Apply selected thresholds unchanged to TestReal and Fresh WildBox.",
    "ci":"track-clustered 95% bootstrap; source-clustered sensitivity"
  },
  "R8_R10_ablations":{
    "seeds":[42,52,62],
    "ego_motion":["raw","camera-corrected"],
    "canonicalization":["none","translation-only","translation+heading+scale"],
    "scale":["no path-length scale","locked TrainReal path-length scale"],
    "normalization":["existing x/W,y/H anisotropic","isotropic x/S,y/S with S=max(anchor width,height)"],
    "ensemble":["seed42","seed52","seed62","3-seed mean"],
    "test_rule":"Fit/select on TrainReal development only; TestReal/Fresh remain evaluation-only."
  },
  "R11_noise_floor":{
    "status":"exploratory",
    "method":"TrainReal-only low-motion windows: summarize high-frequency centroid residual / second-difference magnitude.",
    "caveat":"Call this a noise-floor proxy, not true annotation noise without repeated independent annotations."
  },
  "R12_track_accounting":{"fields":["source","original track ID","segment ID","derived tracklet ID","split reason","window count"]},
  "R13_R16_end_to_end":{
    "pipeline":"video -> frozen elephant detector -> fixed tracker -> elephant tracks -> camera correction -> forecast -> warning",
    "tracker_rule":"Lock tracker choice before test metrics are inspected.",
    "metrics":["elephant detection P/R/AP","HOTA","IDF1","ID switches","fragmentation","forecast ADE/FDE delta GT vs detected-tracked","warning metric delta","latency mean/median/P95 by stage and full chain"],
    "source_rule":"Choose WildBox subset by raw-media/annotation availability only, never observed performance."
  },
  "R17_mixed_species_elephant_filtering":{
    "question":"Can the pipeline preserve elephant-focused detection/tracking/forecast/warning when elephant and non-elephant animals/people appear together?",
    "systems":{
      "A":"current elephant-only detector baseline",
      "B":"elephant vs other-animal vs human detector, then elephant filter",
      "C":"class-agnostic tracking + track-level class voting, then elephant-track filter (primary candidate)"},
    "distractors":["human","deer","cattle/buffalo if available","wild boar if available","other large animals in chosen public data","lion/tiger/bear only as generality stress test"],
    "BAMBI_negative_test":"deer+human frames: expected elephant count zero; every elephant detection is a false positive.",
    "invariance_test":"Add non-elephant distractors to elephant scenes; measure change in elephant confidence, track ID, ADE/FDE, and warning state.",
    "metrics":["elephant P/R/AP","elephant-vs-other confusion","false-elephant detections per 1000 non-elephant frames","HOTA/IDF1/ID switches","forecast ADE/FDE","elephant warning sensitivity","non-elephant false alarms per sequence/hour","distractor-induced warning changes"],
    "error_decomposition":"FA_total = FA_elephant_forecast + FA_distractor",
    "domain_rule":"Report RGB and thermal detector results separately."
  },
  "R18_meter_space":{"goal":"WildBox meter-space ADE/FDE where geometry is reliable; never compare normalized-image ADE across datasets as if units were equivalent."},
  "execution_order":[
    "R1 Protocol lock",
    "R2 Fair all-baseline benchmark",
    "R3 Clustered bootstrap CI",
    "R4 In-domain/fresh fixed conformal coverage",
    "R5 Geofence trivial baselines",
    "R6 Direct crossing classifier and validation-only operating points",
    "R7 Warning hierarchy cleanup",
    "R8 Ego-motion ablation",
    "R9 Canonicalization/scale/isotropic ablations",
    "R10 Ensemble ablation",
    "R11 Noise-floor proxy",
    "R12 Track accounting",
    "R13 End-to-end chain",
    "R14 Tracking metrics",
    "R15 Error propagation",
    "R16 Latency",
    "R17 Mixed-species elephant filtering",
    "R18 Meter-space WildBox",
    "R19 Final claim matrix and paper rewrite"
  ],
  "global_nonclaims":[
    "No Transformer superiority without paired clustered CI support.",
    "No end-to-end system validation before detector/tracker/error-propagation/latency evidence.",
    "BAMBI is not elephant-specific validation.",
    "BAMBI 5s q90 is not calibrated.",
    "Normalized virtual boundaries are not real village/farm geofences.",
    "No field-effectiveness claim without deployment."
  ]
}

def main():
    print("\nR1 — PAPER-STRENGTHENING REVISION PROTOCOL LOCK")
    print("NO new experiment is run in R1. The revision plan is frozen before new results are inspected.\n")
    integrity=verify_frozen()
    integrity_path=OUT/"R1_existing_frozen_snapshot_integrity.csv"
    integrity.to_csv(integrity_path,index=False)
    protocol_path=OUT/"R1_REVISION_PROTOCOL_LOCK.json"
    protocol_path.write_text(json.dumps(PROTOCOL,indent=2,ensure_ascii=False),encoding='utf-8')
    protocol_hash=sha256_file(protocol_path)
    pd.DataFrame([{"order":i,"revision_step":s} for i,s in enumerate(PROTOCOL["execution_order"],1)]).to_csv(OUT/"R1_revision_execution_order.csv",index=False)
    receipt={"status":"R1_REVISION_PROTOCOL_LOCKED_READY_FOR_R2","protocol_path":str(protocol_path),"protocol_sha256":protocol_hash,"existing_frozen_snapshot_verified":True}
    (OUT/"R1_REVISION_PROTOCOL_RECEIPT.json").write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print("EXISTING FROZEN SNAPSHOT INTEGRITY:")
    print(integrity[["artifact","match","actual_sha256"]].to_string(index=False))
    print("\nPROTOCOL:",protocol_path)
    print("PROTOCOL SHA256:",protocol_hash)
    print("\nSTATUS:\nR1_REVISION_PROTOCOL_LOCKED_READY_FOR_R2")
    print("\nR1 COMPLETE\nOutputs:",OUT)

if __name__=='__main__':
    main()
