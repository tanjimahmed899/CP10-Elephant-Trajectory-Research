# EXP4 Phase 5 — Exploratory Clearance-Gated Virtual-Boundary Warnings

**Status:** Retrospective *decision-layer feasibility study*, not independently validated research novelty. Original datasets, model predictions, Step6C q90 calibration and N=2 historical baseline are unmodified. Date: 22 November 2026.

## Experimental question

Can a simple *predicted clearance to uncertainty radius* gate lower the virtual-boundary warning burden while keeping predefined recall on the already studied BIRDSAI TestReal set? This experiment does **not** solve 5-second conformal undercoverage.

## Method

For each forecast at 90% nominal conformal level, let `z` be the existing signed maximum penetration of the **nominal predicted path** towards the directional synthetic boundary. Positive `z` signals a nominal crossing. Let `r = q90 * history_scale` be the original frozen uncertainty radius. Define `u = 1[z+r >= 0]` (the stored `uncertainty_intersects_geofence` must also be true). A clearance-gated uncertainty candidate is `u_rho = u * 1[z >= -rho*r]`. For one track/boundary/direction ordered by `sample_index`, N=2 flags an uncertainty warning only if `u_rho` is true for two consecutive *available forecast rows*. The final binary alert is `nominal_crossing OR persistent(u_rho)`. Existing nominal crossing alerts are never suppressed. We examine rho = 0.25, 0.50, 0.75, and 1.00 (1.00 is original N=2), plus index-adjacency sensitivity (`sample_index` difference must equal one) as a separate counterfactual. All gates are based on stored predictions and uncertainty radii, not future labels. The truth labels appear only in evaluation and retrospective policy selection.

**Evidence units:** synthetic geofence-instance rows, *not* independent elephant incidents or physical notification episodes. One 90% branch with response-time duplicates removed. The four dataset × horizon evaluation populations have 6,318 / 4,956 / 5,190 / 3,822 instances respectively.

## Reproducibility and baseline check

* Source: `step7b_v2_all_geofence_instance_alarms.csv`, q90 branch.
* Input rows = 214,956; q90 rows before dedup = 107,478; unique q90 instances = 20,286; ordered track-boundary sequences = 2,123.
* Source script verified against original `step7f_temporal_persistence_hysteresis_diagnostic.py`.
* All 20 baseline confusion-count regression checks exactly reproduce the previously audited original Step7F table (0 mismatches).

## Primary fixed-grid observations

| Population | Horizon | Policy | Recall | False-alert rate | Precision | TP | FP |
|---|---|---|---:|---:|---:|---:|---:|
| BIRDSAI_TestReal | 2 s | original_N2_available | 82.35% | 35.07% | 8.99% | 210 | 2126 |
| BIRDSAI_TestReal | 2 s | clearance_rho_0.75_available | 80.00% | 26.31% | 11.34% | 204 | 1595 |
| BIRDSAI_TestReal | 5 s | original_N2_available | 83.33% | 30.56% | 18.64% | 320 | 1397 |
| BIRDSAI_TestReal | 5 s | clearance_rho_0.75_available | 77.60% | 22.44% | 22.51% | 298 | 1026 |
| WildBox_Fresh | 2 s | original_N2_available | 86.81% | 37.11% | 12.08% | 250 | 1819 |
| WildBox_Fresh | 2 s | clearance_rho_0.75_available | 82.64% | 28.03% | 14.76% | 238 | 1374 |
| WildBox_Fresh | 5 s | original_N2_available | 77.19% | 30.97% | 21.43% | 291 | 1067 |
| WildBox_Fresh | 5 s | clearance_rho_0.75_available | 68.97% | 23.13% | 24.60% | 260 | 797 |

## Horizon-specific exploratory selection and already-inspected transfer

At both horizons, independently choose the smallest false-alert rate among available-window clearance policies meeting a minimum 80% recall *on BIRDSAI TestReal*. This is **post-hoc** model selection on a previously studied test partition; it cannot be treated as pristine development/testing, nor does WildBox Fresh become a clean confirmatory test by replaying the chosen setting.

* 2 s: `clearance_rho_0.75_available` selected; BIRDSAI recall 80.00%, FPR 26.31%.
* 5 s: `clearance_rho_1_available` selected; BIRDSAI recall 83.33%, FPR 30.56%.

| Population | Horizon | Original N=2 (Recall, FPR, Precision) | Selected (Recall, FPR, Precision) | Change in FP | Change in TP |
|---|---|---|---|---:|---:|
| BIRDSAI_TestReal | 2 s | 82.35%, 35.07%, 8.99% | 80.00%, 26.31%, 11.34% | -531 | -6 |
| BIRDSAI_TestReal | 5 s | 83.33%, 30.56%, 18.64% | 83.33%, 30.56%, 18.64% | +0 | +0 |
| WildBox_Fresh | 2 s | 86.81%, 37.11%, 12.08% | 82.64%, 28.03%, 14.76% | -445 | -12 |
| WildBox_Fresh | 5 s | 77.19%, 30.97%, 21.43% | 77.19%, 30.97%, 21.43% | +0 | +0 |

**Interpretation:** With the stated >=80% criterion on previously studied BIRDSAI data, the 2-second gate trades some true crossing windows for fewer false warning windows. At the 5-second horizon the only eligible grid configuration is the original rho=1.00, so **no warning improvement** is demonstrated there. The observed 2-second trade-off transfers descriptively to already studied WildBox Fresh, with a recall loss; these are highly correlated image-plane boundary instances, not independent field alarms. Index-adjacent counterfactuals are included for timing sensitivity only; `sample_index` adjacency does not establish measured elapsed-time continuity.

## Major limitations

1. No source-disjoint, previously untouched elephant recording group. TestReal was used historically for operating point selection and cannot be reused for confirmatory significance.
2. Each trajectory can contribute several overlapping forecast windows and six synthetic line cases; report counts and clustered/source-unit comparisons, not independent-row p-values.
3. This is a thresholded warning **decision**, not a new conformal calibration; existing 5-second 69.76% EXP3 pathwise coverage and BAMBI undercoverage remain unchanged.
4. Virtual boundary and labels are in camera-relative image-plane coordinates (Step7B), not the fixed segment-reference Stage6 events. A saved Stage6 fixed-reference **per-instance** truth/prediction table is still needed for source-exact fixed-reference warning testing.
5. Response-time-aware RED/YELLOW/ GREEN Step7H thresholds were not changed; the candidate alters the combined alert indicator only.
6. New data collection and a frozen protocol are required before an operational or Q1-methodological success claim.

## Next gates

1. **Method gate:** separate calibration scores from test groups; group-robust or regime-aware calibration should be evaluated on the same original track-scale score, with explicit finite-sample and source-shift caveats.
2. **Geometry gate:** recover per-instance fixed-reference case table and per-forecast camera transforms to recompute event-level warning burden with a physically meaningful temporal basis.
3. **Confirmation gate:** freeze one horizon-specific rule and thresholds before collecting/viewing independent elephant recording groups; log detection→tracking→forecastability yield and capture-group denominators; then evaluate coverage, radius, recall, alerts per monitored hour and latency under the new recordings.

**Novelty decision:** the tested clearance rule is an elementary risk-threshold heuristic; the data do not yet establish a novel Q1-quality method, a conformal coverage guarantee, or reliable field alerting.
