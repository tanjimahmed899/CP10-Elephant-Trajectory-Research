# Phase 7I — Observation-Only Motion-Regime and Forecasting Failure Diagnosis

**Project:** CP 1.0 aerial elephant forecasting. **Date:** 22 November 2026. **Status:** descriptive computational diagnosis completed; **no new independent validation or adaptive method established**.

## Executive finding

The frozen Transformer and simple forecasting baselines have different relative rankings in the two WildLive aerial-elephant source folders. The observed bounding-box-centre motion differs as well. On the previously examined `DJI_0207` sequence, the Transformer is the best of the four evaluated models at a 20-frame horizon, while Stationary is best at 50 frames. On `DJI_0204_video2`, CV-OLS is best at both horizons. This pattern persists when descriptively separating history windows by observed motion speed. It provides a *hypothesis* for method development, not causal evidence of why the model fails.

**Computational audit:** 7,308 overlapping per-window records; two named recording folders; seven elephant IDs (three plus four); original source SHA-256 and six matching checkpoint hashes checked through their archived manifests; **104/104 original summary rows reconciled** with per-window results; nine software unit tests passed. No checkpoint binaries were included in the analyzed result archives, so a fresh neural-network forward pass was **not** independently performed in this phase.

## Input provenance

The following files supplied previously by the user were used directly:

- `Phase7F_Frozen_Results.zip` — DJI_0207 model per-window errors and checkpoint manifest.
- `WildLive_Phase7E_Reproduced_Results.zip` — DJI_0207 normalized 20-frame observation arrays and bbox-centre trajectory CSV.
- `Phase7H_DJI0204_Frozen_Evaluation.zip` — DJI_0204_video2 frozen model per-window errors and checkpoint/annotation provenance manifest.
- `DJI_0204_video2_boxid.zip` — DJI_0204_video2 original `elephant` bounding-box annotations, parsed as bbox centres.

Frame indices, annotation IDs and source labels are treated exactly as present in these files; neither a capture timestamp nor physical coordinates have been added. The script compares per-window errors on keys `(sequence, horizon_frames, track_id, first_observed_frame)`.

## Core numerical outcomes

ADE/FDE are computed in **normalized image-plane coordinates**, not metres. Lower is better. A 'window' is a highly overlapping observation/future segment, not an independent sample.

| Source sequence | Future frames | Windows | Stationary ADE | CV-OLS ADE | Transformer ADE | Best among these |
|---|---:|---:|---:|---:|---:|---|
| DJI_0207 | 20 | 2,487 | 0.004827 | 0.004911 | **0.004522** | Transformer |
| DJI_0207 | 50 | 2,397 | **0.008416** | 0.009928 | 0.009539 | Stationary |
| DJI_0204_video2 | 20 | 1,272 | 0.004056 | **0.001246** | 0.002752 | CV-OLS |
| DJI_0204_video2 | 50 | 1,152 | 0.009768 | **0.002271** | 0.005248 | CV-OLS |

**Important nuance:** at 50 frames in DJI_0207, the Transformer remains modestly better than CV-OLS, but it is **worse than Stationary**. The failure is baseline- and horizon-specific.

## Observation-only feature definitions

All motion features are calculated from the 20 observed bbox-centre coordinates `(x_t,y_t)` *before the forecast starts*; no future ground-truth positions enter the feature computation:

- **Mean apparent step per frame:** sum of 19 consecutive Euclidean displacements, divided by 19, in normalized image-plane coordinates. It is not metres/second.
- **Net displacement:** Euclidean distance between first and last observed coordinates.
- **Straightness / directional persistence:** net displacement divided by the sum of step lengths; values range from 0 to 1 when the path is nonzero. Near-zero path -> undefined.
- **OLS line residual:** root mean square distance to the least-squares position-versus-frame straight-line fit.
- **Acceleration proxy:** RMS first difference of adjacent image-plane displacement vectors.
- **Turn-angle proxy:** average absolute direction change for step pairs above a conservative minimum step threshold. Small-step turning is omitted to limit instability caused by annotation noise.

Because the original frame cadence is unverified, these quantities are **per frame index**, not physical animal speed or acceleration. The normalized coordinate norm also inherits x/y image aspect-ratio anisotropy.

### Observed motion differences

| Feature — medians over 20-frame horizon windows | DJI_0207 | DJI_0204_video2 |
|---|---:|---:|
| Mean observed step/frame | 0.00103586 | 0.00048199 |
| Straightness | 0.31624 | 0.63629 |
| OLS line-fit residual | 0.00125575 | 0.00037355 |
| RMS adjacent-step change | 0.00182658 | 0.00073163 |

These summaries are **not independent measurements**: windows overlap strongly and distributions can also reflect individual track differences, recording geometry, annotations, and drone motion.

## Stratified motion-regime findings

For exploratory description only, each source sequence and forecast horizon was split into **three equal-frequency bins by observed mean step size** (low/mid/high). The boundaries were determined from each recording's own observation windows, so the bins are not interchangeable across videos. Crucially, these are **post-hoc descriptions**, not deployable selection thresholds.

- In `DJI_0204_video2`, CV-OLS has lower mean ADE than the Transformer in **every** low/mid/high speed bin at **both** horizons. The Transformer remains better than Stationary in those bins. Its excess ADE over CV-OLS increases toward the high-speed bin, especially for the 50-frame horizon.
- In `DJI_0207`, the Transformer beats Stationary and CV-OLS in all three speed bins at 20 frames. At 50 frames, Stationary beats Transformer in all three speed bins; the Transformer still beats CV-OLS in the middle and high bins but not the low bin.
- A supplementary within-elephant split avoids directly mixing tracks to form tertiles: `DJI_0204_video2` yields **0/12** track × speed groups with lower Transformer ADE than CV-OLS per horizon, **0/24** across both. In `DJI_0207`, Transformer outperforms CV-OLS in **8/9** groups at 20 frames and **6/9** groups at 50 frames. These overlapping groups are not independent trials.

The source-specific ranking is therefore not explained merely by pooled speed-bin composition, but it **does not prove motion causes the Transformer to underperform**.

## Observation noise, score coverage, and diagnostic checks

`DJI_0207`'s Phase 7C visual review previously flagged uncertain bounding-box annotations at ID 2 frames 389, 401 and ID 0 frame 860. Excluding *windows containing those frames* does not reverse the short/long horizon ranking:

| DJI_0207 subset | Future frames | Windows | Transformer ADE | Stationary ADE | CV-OLS ADE |
|---|---:|---:|---:|---:|---:|
| All windows | 20 | 2,487 | 0.004522 | 0.004827 | 0.004911 |
| Pre-flagged windows excluded | 20 | 2,427 | 0.004384 | 0.004701 | 0.004768 |
| All windows | 50 | 2,397 | 0.009539 | 0.008416 | 0.009928 |
| Pre-flagged windows excluded | 50 | 2,307 | 0.009409 | 0.008361 | 0.009783 |

`DJI_0204_video2` had only sampled visual QA; **no equally complete per-frame artifact-label audit exists**. Do not treat every unflagged window there as certified error-free.

Training canonicalization scale floors were active in approximately 97% of `DJI_0207` windows and **100%** of `DJI_0204_video2` windows according to the earlier Phase 7F/H audits. This is a domain-transfer warning. Floor activation by itself does **not** prove a causal mechanism for model ranking.

## Camera-motion confounding: an explicit limitation

Using the available elephant centroid sequences, this phase computed an **animal-only common-displacement proxy** at each frame index: the component-wise median of all simultaneously tracked elephants' adjacent-frame motions. The goal was to quantify correlated image-plane movement, **not** to remove or identify camera motion.

| Metric (average per adjacent frame index) | DJI_0207 | DJI_0204_video2 |
|---|---:|---:|
| Mean per-elephant displacement norm | 0.00144927 | 0.00054631 |
| Median shared vector's norm (mean across frames) | 0.00089089 | 0.00038080 |
| Mean per-elephant residual norm after shared-vector subtraction | 0.00101528 | 0.00039748 |

The common vector may arise from drone translation/rotation, shared herd walking, bounding-box jitter, or combinations of these. These signals **cannot be separated from elephant boxes alone**. True camera-motion analysis requires original consecutive scene frames and static-background feature registration (plus homography quality controls) or camera pose/IMU telemetry; physical elephant displacement needs an appropriate camera/world-coordinate mapping. The current recordings' native FPS and extracted-frame timestamps are also unverified.

**No camera-stabilized ADE/FDE, real-world velocity, estimated FPS, conformal coverage, geofence validity, or flight-independence result was produced in Phase 7I.**

## Statistical and methodological limitations

1. There are only two named source folders with unverified flight/herd/session independence, not two confirmed independent capture events.
2. All 7,308 forecast windows are heavily overlapping; they cannot be treated as 7,308 independent replications.
3. The speed-regime bin cutoffs were computed after examining both sequences. They must **not** be reused as though preregistered or validated hyperparameters.
4. Differences in camera geometry, frame rate, footage extraction and annotation extent may affect both motion features and apparent forecasting error.
5. This phase reanalyses archived results; checkpoint hashes were reconciled between prior output manifests, but original checkpoint bytes and new forward predictions were not generated here.
6. Any adaptive gating method trained or selected on these sources would make these recordings development data, never untouched holdout confirmation data.

## Next experiment design (Phase 7J/7K, not executed)

**Phase 7J — development-only model selector:** Define a small, interpretable, strictly observation-only rule or trained gating model that chooses among Stationary / CV-OLS / frozen Transformer. Set architecture, hyperparameters, thresholds and evaluation criteria using designated development capture groups only, and compare against each individual baseline. Use sequence/group-aware partitions; avoid random-window splits and target leakage. Before building a selector, inspect whether any available source shows consistent *within-group* gains rather than only post-hoc oracle gains. Do not label a hindsight-per-window best-model selector as a deployable algorithm.

**Phase 7K — prospective frozen test:** Obtain additional elephant recordings with documented nonoverlap and capture-event identities, plus FPS if seconds are claimed. Lock checkpoints, preprocessing, all gating rules, inclusion rules, and statistical protocol *before* revealing the new sequence results. Report all intended baselines per capture event; only do group-level uncertainty intervals when the number and independence of verified events permit it. Test impact on a properly specified alert outcome separately; forecasting ADE improvements do not establish boundary-warning utility.

## Reproducibility and deliverables

`src/run_phase7i.py` operates on the four existing ZIPs and writes: 7,308-row derived feature/error table, per-sequence/horizon scores, track-level summaries, speed/straightness regime tables, QA subset comparisons, within-track checks, animal-only common-displacement proxy, descriptive rank correlations, four figures and SHA-256 manifest. `src/test_phase7i.py` contains nine unit tests.

See `README_UBUNTU.md` for exact Windows/Ubuntu commands. Raw WildLive annotations, videos and checkpoint binaries are **not bundled** in the Phase 7I kit.
