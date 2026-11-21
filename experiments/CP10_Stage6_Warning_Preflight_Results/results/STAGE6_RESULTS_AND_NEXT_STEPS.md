# CP10 — Stage 6 Warning Decisions: Retrospective Baseline and Geometry Preflight

**Evidence type:** verified replay plus new observational warning-policy diagnostics. **Not** a confirmed field deployment, not a risk-control guarantee, and not a fixed-geofence episode evaluation. All coordinates and virtual boundary values are **normalized image-plane units, not metres**.

## Inputs and reproducibility

- `inputs/EXP3_Output_Review.zip`: 754 two-second and 572 five-second frozen three-seed Transformer predictions across six WildBox videos. The **same samples contribute six directional virtual-boundary cases each** (x and y at 0.25/0.50/0.75).
- `inputs/CP10_Stage5_Frozen_Results_Review.zip`: 89 two-second and 56 five-second new frozen forecasts across three candidate videos; analysis explicitly separates development `0006_V` from retrospective evaluation `0004_V` videos.
- Thresholds are **fixed original BIRDSAI TrainReal q90**: 2s = 2.158768; 5s = 1.975130. No Stage 6 threshold was fitted on the evaluation samples.
- The source R6 classifier is included as **aggregate-only comparison**; individual classified windows were not available. Do not claim it has an episode-level outcome.

## Policy and label definitions

At each forecast window, for each axis at each boundary b ∈ {0.25,0.50,0.75}, define crossing direction from the observed endpoint. A true-crossing label is positive when **any annotated future point in this window** is beyond boundary b on the selected side. Nominal alert is positive if any frozen ensemble predicted future point crosses. Uncertainty intersection is positive when the maximum signed predicted proximity plus `q90 × trajectory_scale` is non-negative. N=1 alerts on nominal **or** immediate uncertainty intersection. The original N=2 rule alerts on nominal **or** two consecutive uncertainty intersections in the same track/direction/boundary key. Our additional N=2 and N=3 *gap-aware* diagnostics demand exact 10-index anchor spacing for each required prior alert candidate.

All policies are **retrospective candidates**; gap-aware N3 was not a held-out, pre-registered winning method. Candidate policies trade sensitivity for false alarms. Results on six correlated boundary cases per trajectory window must not be treated as six independent observations.

## EXACT baseline reproduction — EXP3

| Horizon | Policy | True positives | False positives | Recall | False-warning rate | Precision |
|---|---|---:|---:|---:|---:|---:|
| 2s | nominal | 104 | 71 | 33.12% | 1.69% | 59.43% |
| 2s | frozen_N2_original | 272 | 2005 | 86.62% | 47.62% | 11.95% |
| 2s | frozen_N2_gapaware | 258 | 1954 | 82.17% | 46.41% | 11.66% |
| 2s | frozen_N3_gapaware | 232 | 1540 | 73.89% | 36.58% | 13.09% |
| 5s | nominal | 95 | 33 | 19.47% | 1.12% | 74.22% |
| 5s | frozen_N2_original | 388 | 1205 | 79.51% | 40.93% | 24.36% |
| 5s | frozen_N2_gapaware | 374 | 1177 | 76.64% | 39.98% | 24.11% |
| 5s | frozen_N3_gapaware | 317 | 923 | 64.96% | 31.35% | 25.56% |

Replayed original N=2 numerators and denominators were asserted **exactly**: 2s `TP=272,FN=42,FP=2005,TN=2205`; 5s `TP=388,FN=100,FP=1205,TN=1739`. This is equivalent to previously reported 86.62% recall/47.62% FWR and 79.51% recall/40.93% FWR.

## Additional retrospective Stage 5 video comparison

Only the two `0004_V` videos contribute evaluation rows; the development `0006_V` predictions are excluded from evaluation totals. Data are same recording groups as previous work and not independent external video-group confirmation.

| Horizon | Policy | TP | FP | Recall | FWR |
|---|---|---:|---:|---:|---:|
| 2s | nominal | 14 | 3 | 43.75% | 1.05% |
| 2s | frozen_N2_original | 26 | 166 | 81.25% | 58.04% |
| 2s | frozen_N3_gapaware | 20 | 125 | 62.50% | 43.71% |
| 5s | nominal | 11 | 2 | 42.31% | 1.41% |
| 5s | frozen_N2_original | 22 | 82 | 84.62% | 57.75% |
| 5s | frozen_N3_gapaware | 20 | 64 | 76.92% | 45.07% |

For any study targeting operational deployment, a >30% false-warning rate on the negative boundary-window pool is not satisfactory without a real-world cost/benefit specification. Lower false-warning rates from N3 come with substantial lost positive-window recall; a method improvement is NOT established.

## Alert-run proxy—not true fixed-fence episodes

Algorithmic `alert_run` means consecutive forecast-window decisions for the same `(source_group,track_key,axis,boundary,direction)` with anchor separation 10. `truth_run` means consecutive positive future-crossing **window labels** on the same key. Both are **diagnostic run counts**, not true geographical crossing events or first intervention times, because the coordinate system changes between windows.

| Horizon | Policy | Algorithmic alert runs | Alert runs with no positive GT crossing-window | Positive GT window-run hit fraction |
|---|---|---:|---:|---:|
| 2s | nominal | 123 | 41 | 42.33% |
| 2s | frozen_N2_original | 489 | 338 | 85.71% |
| 2s | frozen_N3_gapaware | 390 | 260 | 73.02% |
| 5s | nominal | 104 | 26 | 38.33% |
| 5s | frozen_N2_original | 349 | 208 | 84.44% |
| 5s | frozen_N3_gapaware | 277 | 155 | 71.67% |

`first_cross_step` in saved per-window details can give a **simulated within-window future crossing offset** at 10 Hz, but it cannot be interpreted as operational lead time for a fixed physical boundary. No alert-based intervention, delay, or action was recorded.

## Key geometry discovery—window coordinates do NOT share a common physical geofence

EXP3 future GT points refer to the camera reference at each window's **observation start**, not a fixed location frame. To test this directly, for adjacent windows on a single track separated by 10 sampled indices, the points at the **same overlapping future time** were compared. If both forecasts lived in a fixed reference plane, those truth coordinates would be nearly identical. Instead:

| Source | Horizon | Adjacent pairs | Median of pair-mean displacement | 90th percentile | Maximum individual discrepancy |
|---|---|---:|---:|---:|---:|
| EXP3 | 2s | 693 | 0.02551 | 0.09755 | 0.35517 |
| EXP3 | 5s | 521 | 0.02607 | 0.09954 | 0.36003 |
| Stage5 | 2s | 44 | 0.06881 | 0.15682 | 0.29962 |
| Stage5 | 5s | 22 | 0.08270 | 0.15516 | 0.20075 |

This numerical comparison strongly corroborates **moving reference-frame geometry**. Persisting an `x=0.50` alarm for consecutive windows does not demonstrate persistence against a single fixed fence. Merely deduplicating these flags into episodes **would be misleading**. This limitation applies to our new alert-run proxy tables and to earlier synthetic-boundary persistence claims.

## What must happen next for a defensible Stage 6 experiment

1. Use the supplied WSL collector to gather each source segment's `cameras.json` and `tracking_summary.json` for the six EXP3 plus three Stage 5 videos. This is metadata only; no multi-GB raw image transfer.
2. Lock a **single spatial reference per video/scene**, and define synthetic boundary coordinates in that fixed reference **before** measuring new outcomes. Camera poses are dataset estimates, so do not call it surveyed world metres.
3. Project the forecast geometry into the fixed reference using **only information available at inference time**; never use future GT 3D positions/depth to backproject a model prediction. Use dataset future GT only as evaluation truth.
4. Rebuild chronological cases, deduplicate **real fixed-reference boundary crossings** into episodes, and report episodes hit, false alert episodes per track-hour, onset-to-crossing lead-time using 10 Hz timestamps, and horizon-specific miss rates. Confirm the underlying camera-frame geometry at multiple overlapping anchors.
5. Compare nominal, frozen N=2, and one validation-locked alert rule. Report video-/track-level uncertainty, false-warning cost, and R6 classifier only after collecting its **per-window predictions**. Avoid selecting the operating threshold on evaluation videos.
6. These sources still belong to previously examined groups; genuinely external validation needs a newly acquired, held-out recording group.

## Commands for metadata collection in WSL

Copy **this package** to `C:\Users\HP\Downloads`, then:

```bash
mkdir -p ~/cp10_stage6
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage6_Warning_Preflight_Results.zip" ~/cp10_stage6
python3 ~/cp10_stage6/code/stage6_collect_source_metadata_wsl.py \
 --root "/mnt/e/AIUB/Capstone Project/Journal Based Results" \
 --out ~/CP10_Stage6_StaticFence_Metadata.zip
cp ~/CP10_Stage6_StaticFence_Metadata.zip "/mnt/c/Users/HP/Downloads/"
```

Upload **`CP10_Stage6_StaticFence_Metadata.zip`** for the fixed-reference experiment. Collector is read-only and relies solely on Python's standard library.

## Reproduce baseline metrics

The package contains both original input ZIPs, plus:

```bash
python3 -m pip install numpy pandas matplotlib
python3 code/stage6_warning_audit.py --exp3 inputs/EXP3_Output_Review.zip \
 --stage5 inputs/CP10_Stage5_Frozen_Results_Review.zip --out reproducibility_output
python3 code/stage6_coordinate_consistency.py --exp3 inputs/EXP3_Output_Review.zip \
 --stage5 inputs/CP10_Stage5_Frozen_Results_Review.zip --out reproducibility_output
```

No model fitting, checkpoint updates, or new predictions occur in these scripts.
