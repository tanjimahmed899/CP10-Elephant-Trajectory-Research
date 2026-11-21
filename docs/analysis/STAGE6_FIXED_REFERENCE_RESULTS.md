# Stage 6 — Fixed-Segment-Reference Virtual-Line Warning Study

**Status: numerical geometry replay and retrospective synthetic virtual-line warning-event audit complete.** This is **not** a surveyed ground geofence, an independent-group test, an operational deployment, an observed wildlife intervention, or proof of statistical risk control.

## Executive findings

1. Source data: metadata from **nine WildBox videos / 28 segments**; archived EXP3 three-seed predictions for **754 2-second + 572 5-second** windows and Stage 5 frozen three-seed predictions for **89 + 56** windows. Total **1,471** model forecast windows. Every window was linked to its original camera poses and source 3-D track centers, with **0 missing pairs**.
2. Reconstructed original normalized observation-start GT positions across all 1,471 windows agree with saved histories and futures to at most **1.19e-7** coordinate units. The first camera in each **segment** is used as the fixed reference. Adjacent-anchor overlapping GT positions, replayed using their **annotated future 3-D depth for truth checking only**, agree to at most **6.23e-7**. This contrasts with the previously reported moving-frame median discrepancies ~0.0255/0.0261 on EXP3 2s/5s.
3. A static virtual line at x or y = 0.25/0.50/0.75 in the FIRST camera's normalized image plane is fixed **relative to the estimated segment coordinate frame**. It is an image ray-plane, not a surveyed geographical/geodesic fence. Moving observation-start boundaries are not the same physical plane.
4. In EXP3, **299 / 4,524** (2s) and **327 / 3,432** (5s) synthetic crossing-window labels change when moving-frame boundaries are replaced with static segment-reference lines. Such retrospective window-level metrics are different evaluation problems and must not be compared as if identical.
5. For predicted 2-D future coordinates, **future depth is not supplied by the Transformer**, hence projection into the first-camera reference is not uniquely determined. The primary simulation assumes **constant depth equal to the last observed annotated 3-D track position** (known at the forecast time in annotation-conditioned evaluation). Original frozen q90 radii are approximately transported via the 2-D map Jacobian. This is a **conditional projection method** with no claimed calibrated coverage guarantee after the warp. An oracle future-depth projection is computed separately as *diagnostic only* and must never be presented as deployable inference.
6. Synthetic warning trade-off remains severe. The frozen uncertainty N=2 policy detects more *annotated stable crossing episodes* than nominal-only prediction, but generates many more alert runs without an associated crossing event. No candidate was validated as a superior field decision policy.

## Geometry and label audit

| Prediction population | Horizon | Windows | Max saved GT re-projection error | Median mean per-window constant-depth GT projection error | 90th percentile mean projection error |
|---|---|---:|---:|---:|---:|
| EXP3 six videos | 2s | 754 | 1.06e-7 | 0.000422 | 0.00607 |
| EXP3 six videos | 5s | 572 | 1.19e-7 | 0.000548 | 0.00896 |
| Stage5 three candidate videos | 2s | 89 | 0 | 0.0000360 | 0.00984 |
| Stage5 three candidate videos | 5s | 56 | 0 | 0.00000132 | 0.01236 |

All positional values are **normalized image-coordinate** units, not metres. Some cases are substantially worse: the **maximum per-window mean** constant-depth error across EXP3 5s is **0.12390**, and the maximum individual future-point error in that case is **0.31514**. Thus low pooled error does not imply the assumption is reliable near a critical line.

At adjacent anchors, pairwise GT consistency in a fixed scene reference (ground-truth-only, using known future depths) is:

| Source | Horizon | Adjacent pairs | Median pair-mean discrepancy | Maximum individual discrepancy |
|---|---|---:|---:|---:|
| EXP3 | 2s | 693 | 3.41e-8 | 5.14e-7 |
| EXP3 | 5s | 521 | 3.29e-8 | 6.23e-7 |
| Stage 5 (including development) | 2s | 77 | 1.21e-8 | 1.53e-7 |
| Stage 5 (including development) | 5s | 48 | 4.71e-9 | 1.94e-7 |

The estimated camera pose system is assumed internally coherent. This test **establishes numerical replay consistency**; it does not independently establish absolute pose accuracy or world scale.

Moving- vs fixed-reference crossing labels: EXP3 2s 314 original positive windows vs 279 fixed reference; EXP3 5s 488 vs 469; Stage5 2s 32 vs 22; Stage5 5s 26 vs 16. These labels answer different reference-frame crossing questions.

## Fixed-reference stable crossing events (retrospective, exploratory)

A true crossing event is a **directional sign change** at one of six fixed image-plane lines, with at least **three successive 10-Hz tracker positions on the pre-crossing side and three on the post-crossing side**, with no frame gaps. A crossing is **evaluable** at a forecast horizon if at least one saved window observed the animal before crossing and forecast far enough to include its annotated crossing time. Multiple forecast windows linked to the same annotated crossing **count once**. N=2/N=3 requires exact 10-frame separation between relevant uncertainty intersections; a direct nominal predicted crossing is always permitted.

All reported event counts are defined by annotated 3-D reconstructed track positions; all model decisions rely on the frozen Transformer forecasts and a **last-observed-depth projection**, not future GT. The following are the most informative **K=3** values:

| Population | Horizon | Eligible events | Nominal events hit | Frozen N=2 events hit | Frozen N=3 events hit |
|---|---|---:|---:|---:|---:|
| EXP3 six videos | 2s | 133 | 73 (54.9%) | 116 (87.2%) | 108 (81.2%) |
| EXP3 six videos | 5s | 124 | 59 (47.6%) | 109 (87.9%) | 101 (81.5%) |
| Stage5 two evaluation videos | 2s | 14 | 8 (57.1%) | 11 (78.6%) | 8 (57.1%) |
| Stage5 two evaluation videos | 5s | 10 | 4 (40.0%) | 7 (70.0%) | 7 (70.0%) |

**Alert-run burden** (consecutive same-key alert windows, each key = video / segment / track / axis / line / crossing direction; no physical action observed):

| Population | Horizon | Nominal unassociated runs / total | N=2 unassociated runs / total | N=3 unassociated runs / total |
|---|---|---:|---:|---:|
| EXP3 | 2s | 44 / 117 | 348 / 463 | 271 / 379 |
| EXP3 | 5s | 27 / 85 | 237 / 347 | 183 / 286 |
| Stage5 evaluation | 2s | 1 / 8 | 29 / 38 | 24 / 31 |
| Stage5 evaluation | 5s | 2 / 5 | 15 / 19 | 13 / 17 |

**Important:** “Unassociated” means the alert run did **not** temporally overlap a three-frame-stable annotated crossing of the same synthetic reference line, within its covered forecast horizon. It does not prove an environmental false alarm. All denominators are short, overlapping and clustered by source recording, tracked animal and boundary line. Consequently the high per-track-hour rate in the CSV is **not a field false-alarm rate** and is omitted from the main claims.

The calculated median seconds before the future annotated crossing in an alarm-matched event (EXP3 N=2) is **1.3s for the 2s horizon** and **3.2s for 5s**. These are **simulated video-time offsets** based on assumed 10 Hz and cannot be called operational lead times. No person acted on these alerts.

## Depth-sensitivity (non-deployable diagnostic)

For a sensitivity bound, model-predicted 2-D trajectories were **also** reprojected using *oracle future depth from tracked 3-D GT*. THIS IS NOT A VALID INFERENCE POLICY; the future depths were supplied solely to quantify approximation sensitivity. Nominal warning decisions changed between the last-observed-depth and oracle-depth versions in **7/4,524 EXP3 2s cases**, **15/3,432 EXP3 5s cases**, and **1/168 Stage5 evaluation 5s cases** (0/318 at Stage5 evaluation 2s). A low fraction of flips at these six arbitrary lines does **not** certify accurate projection: some locations have large normalized errors, particularly near the line.

## Research interpretation and manuscript-safe wording

**Defensible:** “A camera-pose-based fixed-segment-reference replay was constructed using the dataset's estimated extrinsics and intrinsics. The reference made ground-truth positions from adjacent observation windows numerically consistent and revealed substantial changes to virtual-boundary labels. Retrospective event-level simulation showed a sensitivity–alert-burden trade-off for uncertainty-intersection warning rules.”

**Not defensible:** “Our system provides georeferenced real-world boundary alerts”, “early-warning lead time in the field”, “independently confirmed risk control”, “exact metric global trajectories”, “the frozen N=2 method safely detects all crossings”, “all elephants were detected/tracked by YOLO”, or “future GT depth was available at inference time”.

**Next experimentally required:** (i) independently surveyed or manually scene-anchored fence with a known physical semantic, (ii) camera-depth estimation deployable without future GT for each forecast point or a conservative projection uncertainty bound, (iii) threshold/confirmation rule frozen before test on *new recording groups*, (iv) source- and event-disjoint statistical uncertainty and alarm cost, (v) obtain R6 per-window classifier outputs for the same static-fence definition, (vi) verify detector→tracker recall and latency before any deployment claim.

## Reproduce

All inputs are included under `inputs/` and the reference code is self-contained apart from Python packages `numpy`, `pandas`, and `matplotlib`.

```bash
python -m pip install numpy pandas matplotlib
python stage6_fixed_reference_study.py
python stage6_episode_sensitivity.py
python stage6_fixed_overlap_verify.py
python create_figures.py
```

Run scripts **in the directory containing this report**. Files write to the same folder. No network access, model training, or mutation of original research files. The R6 per-window classifier was not available and cannot be part of the event-level comparison. The evaluation is retrospective, not independent: Stage5's two evaluation videos are in the same overarching recording groups as prior video development.
