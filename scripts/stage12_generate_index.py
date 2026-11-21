#!/usr/bin/env python3
"""Generate a checked link index of EXISTING artifacts only; no inference/reproduction implied."""
from pathlib import Path
from urllib.parse import quote
import argparse

STUDIES = [
 ('R2','Fair model benchmarks', 'LSTM/GRU/Transformer and baselines; Transformer not universally best',
  [('metrics','experiments/CP10_Research_Stage2_Revision_Reports/results/R2_fair_forecasting/R2E_six_model_fair_benchmark/R2E_six_model_fair_summary.csv'),
   ('paired rows','experiments/CP10_Research_Stage2_Revision_Reports/results/R2_fair_forecasting/R2E_six_model_fair_benchmark/R2E_six_model_same_sample_metrics.csv'),
   ('code','src/original/R2E_V3_six_model_fair_same_sample_benchmark_METADATA_ROW_FIX.py')],
  'Different source cohorts; no universal model winner.'),
 ('R3-R4','Clustered uncertainty and earlier fixed-q calibration', 'Source-wise coverage varies; earlier Fresh is not EXP3',
  [('metric','experiments/CP10_Research_Stage2_Revision_Reports/results/R4_conformal_coverage/R4B_method_lock_and_existing_coverage/R4B_EXISTING_TESTREAL_FRESH_FIXED_COVERAGE.csv'),
   ('CI','experiments/CP10_Research_Stage2_Revision_Reports/results/R3_clustered_bootstrap/R3B_paired_clustered_bootstrap_CI/R3B_paired_cluster_bootstrap_results.csv'),
   ('code','src/original/R4B_step6c_method_lock_and_existing_TestReal_Fresh_coverage.py')],
  'Earlier 3-video Fresh, 6-video EXP3 and BIRDSAI are distinct populations.'),
 ('R6','Crossing classifier trade-off', 'Improved sensitivity, increased false-positive rate',
  [('metrics','experiments/CP10_Research_Stage2_Revision_Reports/results/R6_direct_crossing_classifier/R6C_oof_threshold_lock_and_frozen_evaluation/R6C_FROZEN_EVALUATION_METRICS.csv'),
   ('threshold','experiments/CP10_Research_Stage2_Revision_Reports/results/R6_direct_crossing_classifier/R6C_oof_threshold_lock_and_frozen_evaluation/R6C_LOCKED_VALIDATION_ONLY_THRESHOLDS.csv'),
   ('code','src/original/R6C_validation_only_OOF_classifier_threshold_lock_and_frozen_evaluation.py')],
  'Classifier cannot be characterized as precision/FPR improvement.'),
 ('EXP3','Six-video WildBox frozen ensemble and warnings', '2s/5s forecasting, frozen q90 coverage, synthetic warning rules',
  [('forecast','experiments/EXP3_Output_Review/results/results/fresh_holdout_model_metrics.csv'),
   ('uncertainty','experiments/EXP3_Output_Review/results/results/fresh_holdout_uncertainty.csv'),
   ('source counts','experiments/EXP3_Output_Review/results/results/EXP3_ALL_LOCKED_VIDEO_ACCOUNTING.csv'),
   ('code','experiments/EXP3_Output_Review/code/scripts/exp3_postprocess.py')],
  'Six-video retrospective population; warning FWR is per negative window.'),
 ('Stage 4','Same-target input-only camera correction', 'Mean within-seed ADE reductions 66.51%/53.62% at 2s/5s',
  [('seed comparisons','experiments/CP10_Stage4_R8_Learned_Model_Results/results/CP10_Stage4_R8_Evidence/R8_SameTarget_InputOnly_Seed_Comparisons.csv'),
   ('sample audit','experiments/CP10_Stage4_R8_Learned_Model_Results/results/CP10_Stage4_R8_Evidence/R8_Reconstruction_2792_Sample_Audit.csv'),
   ('code','experiments/CP10_Stage4_R8_Learned_Model_Results/code/CP10_Stage4_R8_Evidence/analyze_input_only.py'),
   ('figure','experiments/CP10_Stage4_R8_Learned_Model_Results/results/CP10_Stage4_R8_Evidence/Figure_R8_InputOnly_SameTarget_ThreeSeed.png')],
  'Seed-wise percentage reduction average, not ensemble error reduction. Image-plane coordinates.'),
 ('Stage 5','Calibration transfer failure', 'Development-only calibration: retrospective 5.66%/7.14% coverage',
  [('coverage','experiments/CP10_Stage5_Calibration_Audit_Results/results/Stage5_Calibration_Coverage_Summary.csv'),
   ('video','experiments/CP10_Stage5_Calibration_Audit_Results/results/Stage5_Calibration_By_Video.csv'),
   ('code','experiments/CP10_Stage5_Calibration_Audit_Results/code/analyze_stage5.py'),
   ('figure','experiments/CP10_Stage5_Calibration_Audit_Results/results/Figure_Stage5_Retrospective_Coverage.png')],
  'Negative result. No conformal cross-domain coverage guarantee.'),
 ('Stage 6','Fixed-reference virtual-line event simulation', 'N=2 event detection 116/133 and 109/124, with high alert-run burden',
  [('events','experiments/CP10_Stage6_FixedReference_Warning_Episode_Results/results/CP10_Stage6_FixedReference_Audit/Stage6_FixedReference_Exploratory_Episode_Metrics.csv'),
   ('label changes','experiments/CP10_Stage6_FixedReference_Warning_Episode_Results/results/CP10_Stage6_FixedReference_Audit/Stage6_Moving_vs_FixedReference_Label_Changes.csv'),
   ('depth sensitivity','experiments/CP10_Stage6_FixedReference_Warning_Episode_Results/results/CP10_Stage6_FixedReference_Audit/Stage6_FixedReference_Depth_Sensitivity_Summary.csv'),
   ('code','experiments/CP10_Stage6_FixedReference_Warning_Episode_Results/code/CP10_Stage6_FixedReference_Audit/stage6_fixed_reference_study.py')],
  'Simulated image-plane line, depth assumption; not field geofence or real alerts.'),
 ('R13','Original detector/tracker bottleneck', 'Very low BIRDSAI cross-domain detection recall and nearly no forecastable tracks',
  [('detection','experiments/CP10_Research_Stage2_Revision_Reports/results/R13_end_to_end_pipeline/R13F_A_detector_tracker_bottleneck_audit/R13F_A_AGGREGATE_STAGE_DETECTION_METRICS.csv'),
   ('tracking','experiments/CP10_Research_Stage2_Revision_Reports/results/R13_end_to_end_pipeline/R13E_tracking_metrics/R13E_AGGREGATE_TRACKING_METRICS.csv'),
   ('code','src/original/R13D_frozen_detector_tracker_execution.py')],
  'Historical Ultralytics and later Stage7D software versions differ.'),
 ('Stage 7A','Detector confidence/resolution pilot', '120 annotated development images; recall versus FP trade-off',
  [('metrics','experiments/CP10_Stage7_Dev_RealInference_Review/results/Stage7_Dev_Overall_IoU50.csv'),
   ('decision','experiments/CP10_Stage7_Dev_RealInference_Review/results/Stage7_Dev_Decision.txt'),
   ('code','experiments/CP10_Stage7_Detector_Pilot/code/stage7_run_dev_inference.py')],
  'TrainReal development results; no test tuning.'),
 ('Stage 7B','TrainReal 8-epoch domain adaptation', 'Sequence-disjoint training/validation; validation recall 3.78%',
  [('validation','experiments/CP10_Stage7B_TrainReal_Review/results/Stage7B_TrainReal_Validation_Readout.json'),
   ('protocol','experiments/CP10_Stage7B_TrainReal_Review/results/Stage7B_Training_Protocol_LOCKED.json'),
   ('code','experiments/CP10_Stage7B_TrainReal_Adaptation/code/stage7b_train_validate.py')],
  'Confidence-swept validation and fixed-conf TestReal are not directly comparable.'),
 ('Stage 7C','Frozen detector comparison on TestReal', 'Recall 5.10% to 56.98%, FP boxes 1,835 to 52,684',
  [('aggregate','experiments/CP10_Stage7C_TestReal_Review/results/Stage7C_Overall_IoU50.csv'),
   ('sequences','experiments/CP10_Stage7C_TestReal_Review/results/Stage7C_PerSequence_IoU50.csv'),
   ('baseline parity','experiments/CP10_Stage7C_TestReal_Review/results/Stage7C_R13_Baseline_Parity.json'),
   ('code','experiments/CP10_Stage7C_TestReal_Frozen/code/evaluate_testreal_locked.py')],
  'Precision and FP burden worsen; retrospective previously analyzed source group.'),
 ('Stage 7D','Frozen ByteTrack evaluation', 'HOTA 0.0252 to 0.2041; IDSW increases markedly',
  [('aggregate','experiments/CP10_Stage7D_Tracker_Review/results/Stage7D_Aggregate_Tracking.csv'),
   ('sequences','experiments/CP10_Stage7D_Tracker_Review/results/Stage7D_PerSequence_Tracking.csv'),
   ('code','experiments/CP10_Stage7D_Offline_Tracker_Replay/code/stage7d_offline_replay.py'),
   ('frozen config','experiments/r13_tracking/config/bytetrack_FROZEN.yaml')],
  'HOTA improved but identity switches/fragmentations increased. No tuning on TestReal.'),
 ('Stage 7E','GT-linked track window eligibility', '187 strict 2s and 124 strict 5s windows from only sequence 0353',
  [('summary','experiments/CP10_Stage7E_GT_Linkage_Review/results/Stage7E_GT_Linked_Window_Summary.csv'),
   ('per-sequence','experiments/CP10_Stage7E_GT_Linkage_Review/results/Stage7E_PerSequence_GT_Linked_Windows.csv'),
   ('code','experiments/CP10_Stage7E_GT_Linkage_Audit/code/stage7e_gt_linkage.py')],
  'Offline GT eligibility; not fully GT-blind operational tracking.'),
 ('Stage 7F','Frozen tracker-history Transformer forecasting', 'ADE 0.010664 (2s), 0.023050 (5s); conditional selected sample',
  [('ensemble','experiments/CP10_Stage7F_Tracker_Forecast_Review/results/Stage7F_Frozen_ThreeSeed_Summary.csv'),
   ('sample metrics','experiments/CP10_Stage7F_Tracker_Forecast_Review/results/Stage7F_Sample_Level_ADE_FDE.csv'),
   ('code','experiments/CP10_Stage7F_Frozen_Tracker_Forecasting/code/stage7f_run.py'),
   ('figure','experiments/CP10_Stage7F_Independent_Audit/results/Figure_Stage7F_Ensemble_vs_Stationary.png')],
  'One video, overlapping selected windows, offline GT identity gate; units normalized.'),
 ('Stage 8','Within-video identity clustered bootstrap', '13/13 and 9/10 identities improved; descriptive bootstrap intervals',
  [('summary','experiments/CP10_Stage8_Statistical_Evidence_Audit_Results/results/Stage8_GT_Clustered_Summary.csv'),
   ('identity detail','experiments/CP10_Stage8_Statistical_Evidence_Audit_Results/results/Stage8_Per_GT_Identity_Differences.csv'),
   ('code','experiments/CP10_Stage8_Statistical_Evidence_Audit_Results/code/audit_stage8.py'),
   ('figure','experiments/CP10_Stage8_Statistical_Evidence_Audit_Results/results/Figure_Stage8_PerElephant_ADE_Differences.png')],
  'Within-video descriptive intervals do NOT establish cross-video significance.'),
 ('R16','CPU computational latency', 'Frozen original throughput under 10 FPS',
  [('timing','experiments/CP10_Research_Stage2_Revision_Reports/results/R16_latency/R16A_computational_latency_audit/R16A_AGGREGATE_WALLCLOCK_LATENCY.csv'),
   ('code','src/original/R16A_computational_latency_audit.py')],
  'Hardware/runtime-specific evidence, not field safety timing.'),
 ('R17','Mixed species four-class detector', 'Negative-image alarms reduced but elephant recall fell',
  [('metrics','experiments/CP10_Research_Stage2_Revision_Reports/results/R17_mixed_species_robustness/R17D_locked_metric_evaluation/R17D_PRIMARY_CONF025_METRICS.csv'),
   ('protocol','experiments/CP10_Research_Stage2_Revision_Reports/results/R17_mixed_species_robustness/R17B_frozen_robustness_protocol_lock/R17B_PROTOCOL_LOCK.json'),
   ('code','src/original/R17D_locked_metric_evaluation_PANDASFIX.py')],
  'Illustrative COCO photo cases cannot establish operational wildlife recall.'),
 ('R18','Metric-scale readiness', 'No verified physical scale anchor for inspected WildBox scenes',
  [('audit','experiments/CP10_Research_Stage2_Revision_Reports/results/R18_meter_space_wildbox/R18A2_wildbox_only_metric_scale_audit/R18A2_SCALE_READINESS_SUMMARY.csv'),
   ('code','src/original/R18A2_strict_wildbox_metric_scale_audit.py')],
  'No validated metres or GPS-based WildBox geofence.'),
 ('Original YOLO','Original elephant-only detection training provenance', 'Reported separate-dataset detector training benchmark',
  [('training manifest','experiments/Original_YOLO/Training_Dataset_Split_Manifest.csv'),
   ('training log','experiments/Original_YOLO/Training_Log.csv'),
   ('config','experiments/Original_YOLO/Training_Config.yaml')],
  'No verified training manifest/log in current published archive; must retrieve, not invent.'),
]


def run(repo: Path, destination: Path):
    lines=[
      '# Stage 12 — Published manuscript-to-evidence index',
      '',
      '**Scope:** file-presence verification against the current local checkout of the published repository. Presence is not independent scientific reproduction, licensing clearance or a claim of cross-domain generalization.',
      '',
      'This index distinguishes direct published outputs from missing files; it does **not** aggregate different evaluation populations.',
      '',
      '| Evidence group | Main finding | Located artifacts | Status |',
      '|---|---|---|---|'
    ]; found=0; missing=0; complete=0
    for code,title,finding,artifacts,caveat in STUDIES:
      present=[(tag,p) for tag,p in artifacts if (repo/p).is_file()]
      absent=[(tag,p) for tag,p in artifacts if not (repo/p).is_file()]
      found+=len(present);missing+=len(absent)
      status=('PRESENT' if not absent else ('PARTIAL' if present else 'MISSING'))
      if status=='PRESENT':complete+=1
      link=', '.join(f'[{tag}]({quote(p,safe="/._-")})' for tag,p in present[:3])
      # Markdown file inside docs/analysis: paths relative to repo root need ../../ prefix
      link=', '.join(f'[{tag}](../../{quote(p,safe="/._-")})' for tag,p in present[:3])
      if len(present)>3:link+=f' (+{len(present)-3} additional)'
      lines.append(f'| **{code} — {title}** | {finding} | {link or "—"} | **{status}** |')
    lines.extend(['','## Scientific interpretation boundaries',''])
    for code,title,finding,artifacts,caveat in STUDIES: lines.append(f'- **{code}:** {caveat}')
    lines += ['','## Limitations of this index','',
      '- `PRESENT` means named local repository files exist. It does not establish clean execution from scratch or original-model provenance.',
      '- Several intermediate CSVs contain selected source IDs and tracked image coordinates; evaluate data-sharing rights before distributing or mirroring.',
      '- Original training images, source videos, checkpoints and precise camera geolocation are not published here; maintain protected storage where legally appropriate.',
      '- Repository lacks an independently validated GT-blind full end-to-end field evaluation and a physical geofence ground-scale reference.',
      '- For a journal submission, freeze a release tag and cite exact commit hash; archive approved supplementary outputs separately.',
      '',
      f'**Generated evidence check:** {len(STUDIES)} claim groups, {complete} with all listed files present, {found} named files present, {missing} named files missing.',
      ''
    ]
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(lines),encoding='utf-8')
    return dict(claim_groups=len(STUDIES),complete_groups=complete,found_artifacts=found,missing_artifacts=missing)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path);args=p.parse_args()
    repo=args.repo.expanduser().resolve(); target=args.out or repo/'docs/analysis/STAGE12_PUBLISHED_EVIDENCE_INDEX.md'
    print(run(repo,target))
