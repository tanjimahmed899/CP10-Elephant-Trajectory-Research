# CP 1.0 — Stage 2 Revision Evidence Audit

**Basis:** Uploaded `CP10_Research_Stage2_Revision_Reports.zip`; 326 collected report files plus one collection manifest. Values below are extracted from saved outputs, **not newly trained or independently reproduced model inferences**. Examined on 2026-11-20.

## Executive decision

Do **not** restart R2–R19. Existing artifacts already include fair model comparisons, clustered uncertainty, a direct crossing classifier, a tracking bottleneck audit, a geometry ablation, a mixed-species detector stress test, and limitation/claim locks. The remaining promising method contributions are (1) group-independent cross-domain conformal calibration with economically useful region width; (2) warning utility/false-alarm control at episode level; (3) exact-common-frame learned-model camera correction, while addressing the severe end-to-end detector domain-transfer bottleneck. Main submission novelty is still not independently proven by the contents of this ZIP.

## Verified facts from saved outputs

### Conformal coverage (R4)

The frozen 5-second q90 regions have window-level coverage of **94.07% on BIRDSAI TestReal** but **86.03% on the earlier three-video Fresh WildBox**. The Fresh track-macro value is **79.17%**. The distinct six-video EXP3 result previously reported in the paper is **69.76%**, which must not be merged with or substituted for Fresh. R4 does not test a corrected cross-domain calibration method. `R4_conformal_coverage/R4C_interpretation_claim_lock/R4C_manuscript_uncertainty_table.csv`

### Direct crossing classifier (R6)

The existing `DIRECT_LOGISTIC_CLASSIFIER` is **forecast-derived decision scoring**, using distance to boundary, nominal predicted crossing and predicted time-to-event transformed into four numeric features; it is not a new learned trajectory architecture. Its thresholds were selected from track-disjoint, development-only 5-fold OOF scores. The small development inventory has **13 tracks (2 s)** and **7 tracks (5 s)**; source-disjoint CV was not consistently feasible. `R6B3_LOCKED_DIRECT_CLASSIFIER_PROTOCOL.json`, `R6B2V3_track_group_cv_feasibility.csv`

| dataset | horizon | nominal_recall_% | classifier_recall_% | nominal_FPR_% | classifier_FPR_% | nominal_F1 | classifier_F1 |
|---|---|---|---|---|---|---|---|
| BIRDSAI_TestReal | 2 s | 30.2 | 77.25 | 0.76 | 5.05 | 0.407 | 0.52 |
| BIRDSAI_TestReal | 5 s | 24.74 | 60.68 | 0.68 | 7.39 | 0.373 | 0.488 |
| WildBox_Fresh | 2 s | 28.47 | 64.24 | 0.51 | 6.83 | 0.415 | 0.458 |
| WildBox_Fresh | 5 s | 19.36 | 52.52 | 0.29 | 9.46 | 0.317 | 0.44 |

The classifier raises recall but **also raises FPR and reduces precision on all four datasets/horizons**. Fresh 2-s F1 improvement's 95% track-cluster CI includes zero. It is a valuable secondary comparison and **must not be described as an across-the-board improved warning policy**. `R6C_FROZEN_EVALUATION_METRICS.csv`, `R6D_TRACK_CLUSTER_BOOTSTRAP_RESULTS.csv`.

### Camera motion correction (R8)

The revision `R8B_raw_vs_corrected_forecasting_results.csv` contains **native-frame** raw-versus-corrected comparisons, e.g. 2-s constant velocity ADE 0.076936 versus 0.020148. The current manuscript's stricter **common-target translation-aligned** comparison reports a *different* raw ADE (0.053522 versus 0.020148). These are **different estimands** and must never be blended. The native-frame stationary comparison suggests improvement, whereas its properly last-point-matched stationary control has **zero prediction difference** by construction. This is strong reason to prioritize exact reference-frame verification and same-architecture raw/corrected learned forecasting.

For stationary background residuals, displayed TestReal medians are 0.705481 and 0.203296 px. Their simple ratio implies a **71.18% reduction**, while the revision reports **46.85%** under a different aggregation. This is not automatically an error, but aggregation needs an explicit formula and source-level table before publication. `R8B_stationary_landmark_residual_results.csv`, `R8C_CLAIM_LOCK.csv`.

### End-to-end detector/track availability (R13)

**9,085 frames**, **48,055 annotated instances**, **4,288 raw detector boxes**. At IoU=0.50 the raw detector recall was **5.06%**. ByteTrack HOTA/IDF1 = **0.0297/0.0246**; BoT-SORT HOTA/IDF1 = **0.0337/0.0292**. Both yielded only **2 forecastable 2-s windows and 0 forecastable 5-s windows**. The observed primary bottleneck is detection domain transfer, not merely tracker thresholds. `R13F_C_MANUSCRIPT_READY_CONCLUSION.txt`, `R13E_AGGREGATE_TRACKING_METRICS.csv`, `R13F_B_CONCLUSION.csv`.

### Mixed-species 4-class detector (R17)

The selectively trained 4-class model reduced image-level false-elephant alarms on elephant-negative test images from **50.99% to 14.57%**, but its elephant recall fell from **62.05% to 12.86%** and F1 from **0.644 to 0.204**. R17 images selected from COCO `train2017` demonstrate qualitative behavior, **not fresh aerial holdout performance**. It is **not a justified detector replacement**. `R17E_KEY_RESULT_LOCK.csv`.

### Other established limitations

- R9 matched isotropic-vs-anisotropic retraining: **no uniform superiority** across GRU/LSTM/Transformer and horizons. `R9I_E2C_REPRESENTATION_DECISION_TABLE.csv`.
- R10 ensemble yields small consistent benefits versus mean individual seed: about **0.9–2.5%**, not universal superiority versus other models. `R10D_MANUSCRIPT_READY_ENSEMBLE_CONCLUSION.txt`.
- R11 post-correction geometric residual floor is **below absolute forecast error**, while small ensemble gains over average seeds are **below the conservative F95 residual envelope**. `R11E_FINAL_CLAIM_MATRIX.csv`.
- R12: 45 original BIRDSAI TestReal tracks; 37 (2 s) / 31 (5 s) represented in 89 / 63 forecastable segments. `R12F_CLAIM_MATRIX.csv`.
- R16 measured detector+ByteTrack CPU throughput ≈ **8.73 FPS**, BoT-SORT ≈ **7.76 FPS**, both below 10 Hz. `R16A_AGGREGATE_WALLCLOCK_LATENCY.csv`.
- R18 found camera intrinsics/extrinsics for **3/3** inspected WildBox sources, metric anchors for **0/3**; metre-valued WildBox reporting is not supported. `R18A2_SCALE_READINESS_SUMMARY.csv`.

## Recommended new experiment sequence

1. **Stage 3A — target-independent calibration**: On newly acquired, untouched video-group holdouts (rather than reusing already tested EXP3 clips), compare frozen TrainReal q90 with pre-locked target-global, track-weighted, group-conservative and low-count motion-conditioned calibration. Evaluate pathwise 2-s/5-s coverage, per-group reliability, region inflation, and bootstrap sensitivity. Earlier Stage 1 leave-one-group-out results are exploratory model-selection evidence only.
2. **Stage 3B — decision policy**: Use identical held-out groups to compare nominal, existing N=2 rule, R6 direct classifier and threshold-controlled alternative. Compute source-level and deduplicated episode-level recall, FWR, false alerts per annotated-animal-hour and lead time under specified frame timestamps. If uncertainty envelopes inflate, explicitly test the warning impact.
3. **Stage 3C — exact-geometry R8**: Identify the underlying BIRDSAI raw/corrected observation/target arrays and per-frame camera matrices. First validate true common-frame mapping and residual error; then pair frozen same-model raw/corrected forecasts under matched samples and video-level ego-motion strata. Without pose/depth or ground-plane compatibility, report an honest approximation, not “exact geometry.”
4. **Stage 3D — end-to-end recovery**: Validate domain-adapted elephant detection on new training/development data, keep a source-disjoint test, then measure detection recall before tracker IDF1/HOTA and forecastable-window yield. Avoid tracker-only tuning on existing TestReal labels.

## Immediate WSL next-step: find raw R8 data and additional WildBox sources

Use `find` and inspect candidate size/path before packaging large arrays. The already reported source folders are in `/mnt/e/AIUB/Capstone Project/Journal Based Results/Main Result`, `Main Result 2.0`, `Main Result 3.0`. The original checkpoints exist under `Main Result/transformer_results`. The original EXP3 six video archives were **not** found in the initial restricted inventory; search the wider E: drive before concluding they are absent.

## Submission boundaries

- Do not describe R17 COCO visual examples as drone frames or representative COCO test accuracy.
- Do not describe R6 as a new elephant forecasting neural architecture.
- Do not treat old native-frame R8 and matched-common-target R8 as the same metric.
- Do not use synthetic boundary FWR as operational alerts/hour.
- Do not claim numerical q90 transfer results are a conformal validity guarantee under arbitrary distribution shift.
- Do not claim detector-driven 5-s forecasting, WildBox metre-scale distances, field conflict mitigation or 10 Hz operational throughput.

**Audit type:** evidence review of uploaded saved artifacts. No original checkpoint was executed, no dataset was trained and no claim of new prospective confirmation is made here.
