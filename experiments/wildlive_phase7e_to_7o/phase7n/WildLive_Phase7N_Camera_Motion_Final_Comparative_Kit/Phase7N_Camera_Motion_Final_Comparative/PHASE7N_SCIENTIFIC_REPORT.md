# CP 1.0 — Phase 7N: Camera-Motion Sensitivity and Final Exploratory Model Comparison

**Date:** 23 November 2026  
**Study:** WildLive aerial-elephant bounding-box-centre forecasting  
**Decision:** Numerical-output synthesis **PASS**; sparse background-registration feasibility **PASS on 8 sampled consecutive-frame pairs**; camera-corrected forecasting ADE/FDE **NOT evaluated**; adaptive-model superiority **NOT demonstrated**; independent-flight generalization **NOT established**.

## 1. Scope and evidence boundaries

This report reanalyses all available archived per-window model errors from four elephant-labelled video sequences and performs a *new* image registration audit of eight consecutive RGB-frame pairs. The original frozen Transformer checkpoint binaries and per-window prediction-coordinate arrays are **not** in the four archived evaluation ZIPs; thus no new Transformer forward passes were run. The phase recomputes ADE/FDE means from previously generated per-window scores. Existing models, thresholds, and the two released Phase 7J gate JSONs were not refitted.

All numbers are **normalized 3840 × 2160 image-plane distances** for 20-frame observed histories forecasting 20 or 50 **future frames**, except the sparse camera-audit distances, which are in native-image **pixels**. FPS, world coordinates, absolute ground displacement, and independent capture-event identities are not established. A video filename is not a flight/session identifier.

## 2. Input audit and methodological safeguards

- Independently read `Phase7F_Frozen_Results.zip`, `Phase7H_DJI0204_Frozen_Evaluation.zip`, `Phase7K_DJI0161_Locked_Evaluation.zip`, and `Phase7L_DJI0204_video1_Exploratory_Results.zip`.
- Validated archive CRC, unique `(horizon, track ID, first observed frame)` keys per recording, nonnegative finite ADE/FDE, and saved adaptive error equal to its selected expert error for every gate-evaluated window.
- Verified original annotation ZIP SHA-256 against Phase 7H/7K/7L evaluation manifests. **Important:** Phase 7F `input_sha256` refers to an intermediate `.npz`, **not** the annotation ZIP; that field was therefore **not** incorrectly compared to `WildLive_DJI_0207_boxid.zip`.
- Original checkpoint binary hashes were **not independently recomputed** in this container. Prior evaluation manifests report locked hashes, but those reports are not evidence of new forward-inference replication in Phase 7N.
- Total 12,948 *overlapping* windows: 4,884 for `DJI_0207`, 2,424 for `DJI_0204_video2`, 2,748 for `DJI_0204_video1`, and 2,892 for `DJI_0161_video1`. These are **not 12,948 independent samples**.
- The original `DJI_0207` and `DJI_0204_video2` recordings are development material. `DJI_0204_video1` has previously identified overlapping landscape with `DJI_0204_video2`; classify as related-scene exploratory. `DJI_0161_video1` is previously evaluated and its flight independence is unverified. This is a retrospective comparison, not a new preregistered test.

## 3. Final locked-expert mean ADE by recording

Lower is better. **No adaptive result exists for either development recording from the released *locked final gate*.** Phase 7J leave-one-recording-out development stress-test scores are intentionally not mixed into this final-lock table.

| Sequence and role | Future frames | Stationary | CV20 | CV-OLS | Frozen Transformer | Locked Adaptive |
|---|---:|---:|---:|---:|---:|---:|
| DJI_0207 (development) | 20 | 0.004827 | 0.004875 | 0.004911 | **0.004522** | N/A |
| DJI_0207 (development) | 50 | **0.008416** | 0.009764 | 0.009928 | 0.009539 | N/A |
| DJI_0204_video2 (development) | 20 | 0.004056 | 0.001279 | **0.001246** | 0.002752 | N/A |
| DJI_0204_video2 (development) | 50 | 0.009768 | 0.002342 | **0.002271** | 0.005248 | N/A |
| DJI_0204_video1 (related scene) | 20 | 0.002217 | 0.001861 | **0.001818** | 0.001930 | 0.001829 |
| DJI_0204_video1 (related scene) | 50 | **0.004621** | 0.005221 | 0.005173 | 0.006708 | 0.005175 |
| DJI_0161_video1 (previously evaluated) | 20 | 0.011163 | 0.012417 | 0.012940 | **0.010934** | 0.011195 |
| DJI_0161_video1 (previously evaluated) | 50 | **0.020406** | 0.026922 | 0.028270 | 0.020745 | 0.021524 |

**Finding:** The identity of the best fixed expert varies by recording and horizon: Transformer leads both short-horizon `0207`/`0161` evaluations; Stationary leads their 50-frame evaluations; CV-OLS leads both `0204_video2` evaluations and the `0204_video1` 20-frame evaluation. A single universal winner is not observed.

**Predeclared primary adaptive comparator:** CV-OLS (not the hindsight-best expert). Locked adaptive model on `DJI_0161_video1`: 13.4901% smaller ADE at 20 frames and 23.8636% smaller ADE at 50 frames. On related-scene `DJI_0204_video1`: 0.6064% *larger* ADE at 20 frames and 0.05263% *larger* at 50 frames. Adaptive is **worse than the best single expert** for all four available sequence–horizon comparisons. These are descriptive within-recording comparisons, not an independent-flight generalization estimate.

FDE is preserved in `results/recording_level_ADE_FDE.csv` and should be consulted independently: the FDE winner can differ from the ADE winner (e.g., `DJI_0204_video1`, 20 future frames, Transformer has lower FDE despite CV-OLS having the lowest ADE). No new conformal-calibration, geofence, or field-warning evaluation was conducted.

## 4. Conservative scene-component grouping

Provisional groups used for descriptive sensitivity: `SCENE_0204` = `{DJI_0204_video1, DJI_0204_video2}`; `UNRESOLVED_0207` and `UNRESOLVED_0161` are single unlinked scene-evidence components. These are **not three verified independent original flights**. For the 0204 component, first compute each video's mean ADE and then equally weight the two videos; do not elevate any model's apparent per-window sample size. At 20 frames, this gives CV-OLS `0.00153182` vs Transformer `0.00234139`; at 50 frames CV-OLS `0.00372200` vs Transformer `0.00597794`.

The adaptive method is **not aggregated at the 0204 group level** because it was not evaluated in `0204_video2` under the same released final gate: comparing its single-video mean to a two-video expert mean would be misleading. The machine-readable `provisional_scene_component_summary.csv` omits adaptive for all scene-component aggregations for this reason.

With at most three provisional scene components, none independently flight-verified, no independent capture-group variance estimate, p-value, bootstrapped group-level confidence interval, or statistical superiority claim is justified. Offsets and animal IDs within a shared recording are not independent capture events.

## 5. Sparse original-frame camera registration: new Phase 7N empirical audit

**Method:** For consecutive available original 3840×2160 RGB image pairs, resize to 1536×864 for feature matching. Exclude 1.4× expanded elephant bounding-box rectangles in *both* images, detect up to 6,000 ORB features in the remainder, match via Hamming BF with Lowe ratio <0.75, and fit a 2D homography with RANSAC (3.0 resized-image-pixel reprojection threshold; fixed RNG seed). Quality gate: >=80 geometric inliers, >=55% match inlier fraction, >=10% convex-hull coverage, >=8 of 16 spatial grid cells covered, and <=2 resized-image-pixel median reprojection error. Map homography back to native image size before computing pixel motion.

For each animal with the same annotation ID in both frames, compute:

- `raw_step`: Euclidean pixel displacement between frame-to-frame bounding-box centres;
- `background_step_proxy`: distance from frame-A animal-centre position to its image-B location predicted by the fitted **static-background homography**;
- `registered_residual`: distance from the *actual* frame-B annotation centre to the homography-predicted location.

All **eight** consecutive-frame pairs met the prespecified image-registration *numerical quality gate*. This measures a strong **2D background alignment**, not proof of exact drone pose recovery, world-ground movement or full-scene static rigidity. A homography can be biased by parallax, rolling-shutter geometry, moving foliage, ground relief or mismatched depth planes.

| Sequence | Consecutive frames | Median raw animal step (px) | Median background step proxy (px) | Median post-registration residual (px) |
|---|---|---:|---:|---:|
| DJI_0207 | 389→390 | 6.21 | 1.55 | 7.15 |
| DJI_0207 | 400→401 | 2.85 | 0.67 | 3.42 |
| DJI_0207 | 401→402 | 8.36 | 3.33 | 5.05 |
| DJI_0207 | 858→859 | **24.35** | **24.26** | **3.03** |
| DJI_0207 | 859→860 | **27.06** | **24.27** | **4.53** |
| DJI_0207 | 860→861 | **27.30** | **22.15** | **5.15** |
| DJI_0204_video2 | 33→34 | 2.11 | 0.32 | 1.49 |
| DJI_0204_video2 | 303→304 | 3.90 | 0.48 | 3.90 |

In the three late `DJI_0207` pairs (858–861), a substantial component of the *observed image-plane displacement* is consistent with background transformation, not necessarily autonomous animal motion. At the same time, high residuals are present for specific individuals: `DJI_0207` track 2, frame pairs 389→390, 400→401, 401→402 have roughly 59, 66, and 68-pixel residuals; track 0 around 859→861 has >34-pixel residuals. These cases deserve **manual identity/annotation/pose-change review**, not post hoc exclusion. Other pairs show small or even increased residuals after alignment. Results are neither universally improved nor a reliable source of corrected full-trajectory kinematics.

**Key scientific distinction:** A camera-motion proxy observed for eight isolated frame-pairs does **not** establish how camera motion affected aggregate 20/50-future-frame ADE. In particular, transforming *both* a forecast and its ground truth by the same translation or rotation leaves Euclidean error unchanged; obtaining new forecasting performance requires precisely defining a stabilized-history forecasting method, transforming all available necessary frames/predictions, and re-evaluating the fixed method without test-data tuning. Sparse pairs and absent complete prediction-coordinate arrays cannot achieve that. Thus **no camera-corrected forecasting ADE/FDE or causal improvement is claimed here**.

## 6. Canonicalization scale-floor regime audit

Weighted fractions derived from archived canonicalization statistics:

| Recording | 20-frame scale-floor active | 50-frame active |
|---|---:|---:|
| DJI_0207 | 97.51% | 97.29% |
| DJI_0204_video2 | 100.00% | 100.00% |
| DJI_0204_video1 | 100.00% | 100.00% |
| DJI_0161_video1 | 86.65% | 84.98% |

Scale-floor use is frequent across all four recordings. This supports **preprocessing-regime sensitivity as a plausible diagnostic hypothesis**, not causal attribution of model errors to the floor. The inference parameters are locked and unchanged. Testing a new normalization method would require a newly defined development/evaluation protocol and new untouched confirmation material.

## 7. Publication-safe conclusions and manuscript-ready wording

**Suggested result paragraph** (appropriate for an exploratory robustness paper):

> Across four aerial-elephant annotation sequences, we reanalysed 12,948 overlapping forecasting windows at 20- and 50-frame prediction horizons using archived frozen-model outputs. No forecasting expert was uniformly best. A locked observation-only adaptive selector improved mean ADE over the prespecified constant-velocity comparator on one previously evaluated sequence but not on a related-scene sequence; it did not outperform the best single expert on either sequence. A conservative visual audit identified strong shared-scene evidence between the two DJI_0204 videos, which were grouped during descriptive comparisons. Background-feature alignment of eight sampled consecutive-frame pairs revealed substantial camera-associated image-plane displacement in selected DJI_0207 transitions, demonstrating a confounder for naive animal-motion interpretations. The available evidence does not establish flight-independent generalization, metre/second kinematics, camera-corrected forecasting gains, or operational geofence effectiveness.

**Contribution that is supported:** transparent cross-video failure-case taxonomy, source-overlap control, locked-gate negative-result analysis, and sparse background-motion feasibility diagnostics. This is a well-scoped **audit/diagnostic contribution**, not a validated novel superior adaptive architecture. Journal quartile/acceptance cannot be inferred from the audit.

## 8. Next empirical work, ranked by value

1. Preserve exact locked checkpoint/gate versions and image/annotation provenance. Do not treat `DJI_0161_video1` or either `0204` clip as an untouched new test after these analyses.
2. Perform targeted annotation QA on `DJI_0207` tracks 2 at 389–402 and track 0 at 859–861. Check image coordinates and apparent switches/box-jitter; publish both raw and flagged-window sensitivity, never silently remove.
3. For **actual camera-motion-corrected forecasting**, obtain *consecutive RGB frames* covering complete selected 20-history-plus-20/50-future windows **and raw prediction-coordinate arrays**; define a reference-image coordinate system and stabilized-input forecasting protocol. Do **not** fit adjustments based on held-out forecast errors.
4. If operational goals remain in scope, future metrics must include geofence warning false alarms, missed alerts, calibration coverage and latency **under a fully specified cadence**, not ADE alone.
5. Without documented independent capture events or new elephant observations, position the next journal manuscript as exploratory, reproducible, and limitations-forward rather than claiming universal or statistically verified transfer gains.

## 9. Deliverables and reproducibility

`src/phase7n.py` uses original named ZIPs from a specified directory to recreate descriptive tables and the eight-pair image registration, while `src/test_phase7n.py` contains offline tests. The `results/` directory contains:

- `input_provenance_sha256.csv` — source hashes / roles;
- `recording_level_ADE_FDE.csv`, `best_expert_per_recording.csv` — model outcome tables;
- `locked_adaptive_vs_cv_ols.csv` — original locked-gate primary comparison;
- `provisional_scene_component_summary.csv` — expert-only component-level descriptive mean (no mixed adaptive coverage);
- `scale_floor_regime_summary.csv` — weighted archived preprocessing frequency;
- `consecutive_frame_camera_registration.csv` and `per_track_raw_vs_registered_motion.csv` — eight-pair image registration and 26 track-specific displacements;
- `phase7n_audit_manifest.json`, `automated_tests.log`, and three figures.

No original WildLive photos, elephant GPS information, annotation payloads, model checkpoint binaries or source clips are redistributed in the package. This is consistent with keeping source material separate under the dataset's **Non-Commercial Government Licence**; users must independently check license conditions before redistribution.

**Dataset reference:** Nguyen Ngoc Dat et al., *WildLive* (University of Bristol dataset, 2026), DOI: `10.5523/bris.ewnuoroebuae20vei0cu6zu69`. Its official README notes 22 UAV video sequences and the absence of a released prescribed data split; this does not resolve independent flight membership. See `https://data.bris.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69` and `https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/Readme.txt`.
