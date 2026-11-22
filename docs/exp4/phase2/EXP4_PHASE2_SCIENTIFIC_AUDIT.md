# EXP4 Phase 2 — Source-exact persistence, warning-rule and timing-sensitivity audit

**Audit date:** 22 November 2026  
**Study:** Aerial-elephant trajectory forecasting and retrospective virtual-geofence warning evaluation  
**Status:** **Completed source-level CSV replay; not an independent field or confirmatory test**

## 1. Scope and exact evidence

This phase uses two newly supplied authentic machine-readable exports:

- `step7b_v2_all_geofence_instance_alarms.csv` — **214,956 rows**, **35 columns**. It contains response-time and calibration-level variants of virtual-geofence instances, not 214,956 independent forecast windows.
- `TABLE_step7f_persistence_tradeoff.csv` — **16 rows**, **19 columns** of already-observed persistence trade-offs for N=1,2,3,5.

Cross-checked with the previously supplied:

- `STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json` — source-table hash, exact selection requirements, interpretation and provisional status.
- `All Python Code.zip` — `step7b_v2_response_aware_green_yellow_red.py`, `step7f_temporal_persistence_hysteresis_diagnostic.py`, `step7g_pareto_and_provisional_operating_rule_lock.py`, and `step7h_final_response_aware_with_locked_persistence.py`.
- Previously uploaded experimental PDFs and Phase 1 calibration-score audit were used solely to understand the historical research scope and the distinct Stage 6 fixed-reference analysis. We did **not** recompute the Stage 6 camera geometry in this phase.

All replays were independently implemented with Python's standard-library `csv` reader and basic arithmetic; original model inference and original scripts' Windows output paths were not executed. The two original uploaded CSVs and JSON lock were **not edited**.

## 2. File integrity and protocol selection — VERIFIED

- SHA-256 of the supplied Step-7F table: `3d8ef15f3899cfb023d3b3fa53033c7fbc4c1bb5ec7b64e72ab8fe503c81470f`.
- The table digest **exactly equals** `source_step7f_sha256` in the user's Step-7G JSON lock.
- For 90% conformal, the raw Step-7B table has **107,478** rows across response-time settings, representing **20,286** unique instance keys after response-setting deduplication. Repeated response-time copies showed **zero contradictory labels** for truth, nominal crossing or uncertainty intersection.
- The Step-7F persistence algorithm was independently reconstructed for N={1,2,3,5} within `(dataset,task,source_group,track_key,axis,boundary,direction,boundary_id)`, sorted by `sample_index`. Its **16 rows × 16 independently checked numeric fields = 256 / 256 matches** with the uploaded reference table (no mismatches within floating-point tolerance).
- The Step-7G rule was independently applied: choose the **smallest** N in {2,3,5} that reduces the false-alert rate by at least 20% relative to N=1 **on both BIRDSAI TestReal forecast horizons**, while retaining sensitivity at least 80% on both. N=2 passes:

| BIRDSAI TestReal | N=1 false-alert rate | N=2 false-alert rate | Relative reduction | N=2 sensitivity | Criterion |
|---|---:|---:|---:|---:|---|
| 2 s | 46.05% | 35.07% | 23.85% | 82.35% | PASS |
| 5 s | 40.31% | 30.56% | 24.20% | 83.33% | PASS |

**Interpretation:** The provisional **selection procedure and source hash are reproducible**. Selection is expressly **post-hoc exploratory on already-studied TestReal**, not a valid confirmation of superior generalization.

## 3. N=2 event-label warning metrics — independently RECOMPUTED

The original Step-7F **RED means any nominal predicted crossing**. YELLOW means uncertainty intersects the geofence in N consecutive **available** forecast windows (unless already RED); GREEN means neither. This is a **window–virtual-boundary instance** analysis, not operational alert episodes per hour.

The denominator for sensitivity is all positive instances (`TP+FN`). The denominator for false-alert rate is all negative instances (`FP+TN`). The denominator for precision is all issued RED/YELLOW instances (`TP+FP`).

| Evaluation population | Instances | Positive instances | TP | FP | FN | TN | Sensitivity | False-alert rate | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BIRDSAI TestReal, 2 s | 6,318 | 255 | 210 | 2,126 | 45 | 3,937 | 82.35% | 35.07% | **8.99%** |
| BIRDSAI TestReal, 5 s | 4,956 | 384 | 320 | 1,397 | 64 | 3,175 | 83.33% | 30.56% | **18.64%** |
| WildBox Fresh, 2 s | 5,190 | 288 | 250 | 1,819 | 38 | 3,083 | 86.81% | 37.11% | **12.08%** |
| WildBox Fresh, 5 s | 3,822 | 377 | 291 | 1,067 | 86 | 2,378 | 77.19% | 30.97% | **21.43%** |

The comparison against **nominal crossing alone** shows the fundamental trade-off:

| Population | Horizon | Nominal TP/FP | Nominal precision | N=2 TP/FP | N=2 precision |
|---|---|---|---:|---|---:|
| BIRDSAI TestReal | 2 s | 77 / 46 | 62.60% | 210 / 2,126 | 8.99% |
| BIRDSAI TestReal | 5 s | 95 / 31 | 75.40% | 320 / 1,397 | 18.64% |
| WildBox Fresh | 2 s | 82 / 25 | 76.64% | 250 / 1,819 | 12.08% |
| WildBox Fresh | 5 s | 73 / 10 | 87.95% | 291 / 1,067 | 21.43% |

These findings **do not support** a claim that N=2 creates an efficient, field-ready warning system. It improves the sensitivity/false-alert trade-off compared with N=1, but still has substantial unnecessary virtual-warning burden compared with nominal crossing alone. Report all three metrics; do not use recall alone.

## 4. Step-7B source-rule consistency and Step-7H response-aware replay — VERIFIED/REPLAYED

### Step-7B row audit

For **all 214,956 Step-7B rows** the stored confidence radius, oriented score, uncertainty-intersection flag and original response-aware `alarm_state` are consistent with the original Step-7B equations and the **six-decimal rounded q values** embedded in its protocol:

- 2 s q90 = 2.158768; q95 = 2.560773.
- 5 s q90 = 1.975130; q95 = 2.950976.
- `absolute_conformal_radius = scale × rounded_q`.
- `uncertainty_risk_score = oriented_risk_score + absolute_conformal_radius`.
- `uncertainty_intersects_geofence = (uncertainty_risk_score ≥ 0)`.
- RED if nominal crossing occurs within the response threshold (or specified already-inside fallback); otherwise YELLOW if nominal crossing later or uncertainty intersects; GREEN otherwise.

**No row-level formula mismatches were detected.** Note: The original Step-6C calibration CSV stores **more decimal precision** than Step-7B's embedded thresholds. Recomputing with the exact Step-6C q rather than the rounded downstream q introduces very small radius differences; this is a rounding/provenance detail, not a newly observed failure.

### Step-7H final response-aware rule, N=2

Using the original `step7h_final_response_aware_with_locked_persistence.py` logic, **22** dataset × horizon × response-threshold evaluation rows were independently reconstructed from the raw Step-7B CSV. The response-aware RED urgency rule is **not the same RED definition** as the Step-7F diagnostic. The **any-RED-or-YELLOW** decisions agree with Step-7F N=2 for all response-time copies; the classification of *RED versus YELLOW* varies with response threshold.

Example for a **1.0-second response-threshold assumption** (illustrative operating point from the supplied grid, not an independently tested real response time):

| Population / horizon | N=2 any-alert sensitivity | N=2 any-alert precision | RED urgent-crossing sensitivity | RED urgent-crossing precision |
|---|---:|---:|---:|---:|
| TestReal 2 s | 82.35% | 8.99% | 28.37% | 62.50% |
| TestReal 5 s | 83.33% | 18.64% | 26.36% | 58.00% |
| Fresh WildBox 2 s | 86.81% | 12.08% | 25.00% | 77.36% |
| Fresh WildBox 5 s | 77.19% | 21.43% | 24.39% | 74.07% |

RED-specific metrics have a **different positive-event target** (urgent true crossing by the response threshold). They must not be compared with overall any-crossing metrics as though their denominators and tasks were the same. The current audit is source-code replay, **not** a comparison to an independently uploaded original Step-7H summary.

## 5. New exploratory diagnostic: persistence across index gaps

The original Step-7F code deliberately uses N consecutive **available** windows and does **not** require adjacent `sample_index` values or prove a fixed elapsed duration. Across **2,123** grouped track–boundary scenarios, **208** contain sample-index gaps greater than one. Of **7,384** N=2 persistent-intersection decisions, **266** follow such an index gap.

We tested a **counterfactual index-adjacency restriction** requiring `sample_index[t] - sample_index[t-1] = 1` for N=2. It is **not** adopted as the correct rule, because sample-index continuity is not the same thing as a verified elapsed-time guarantee.

| Population / horizon | Original N=2 sensitivity / FAR | Index-adjacent-only sensitivity / FAR |
|---|---|---|
| BIRDSAI TestReal 2 s | 82.35% / 35.07% | 72.55% / 33.83% |
| BIRDSAI TestReal 5 s | 83.33% / 30.56% | 77.86% / 29.44% |
| WildBox Fresh 2 s | 86.81% / 37.11% | 79.86% / 36.82% |
| WildBox Fresh 5 s | 77.19% / 30.97% | 74.27% / 30.68% |

**Scientific consequence:** Operational persistence is materially dependent on how skipped forecasting windows are handled. If claims refer to seconds, use explicit timestamps or verified frame-time mapping and define an allowable maximum gap. Do not silently reinterpret N=2 as 2 seconds.

## 6. What the evidence does not establish

- The supplied Step-7B CSV is **not** the source-level Stage 6 moving-reference versus fixed-reference event replay. It lacks per-frame camera transforms, depth-informed fixed-reference 3-D annotated trajectories and paired crossing-label case IDs; the reported 299/4,524 and 327/3,432 changed labels from `Experimental Results (Part-4).pdf` remain report-level checks in this phase.
- The evaluated virtual boundaries are synthetic normalized image-plane boundaries. Neither physical village/farm geofences nor field-conflict outcomes are established.
- Instance rows are correlated: the same trajectory may be evaluated against multiple lines/directions and response settings. Naive iid uncertainty intervals or claims of tens of thousands of independent observations are incorrect.
- The original rule and the Step-7H replay were evaluated on datasets already used for iterative research. Selection based on BIRDSAI TestReal is post hoc; WildBox Fresh is a **retrospective transfer comparison**, not untouched confirmatory evidence for the newly selected rule.
- The Step-7B data are based on annotated/curated forecasts and do not establish unrestricted detector→tracker→forecast→warning performance. Previously documented Stage 7E strict GT-eligible windows were concentrated in one TestReal sequence.
- No calibration fix, new model, real-time field performance, or independent elephant-group validation was demonstrated in Phase 2.

## 7. Priority decisions for EXP4 Phase 3

**P0 — Critical, before novelty claims:** Define a primary outcome at **event or alert-run level**, with fixed-reference labels and provenance. Obtain original Stage 6 per-event labels/transforms/GT 3-D positions if exact repro is desired; do not use future annotated depth for real-time decisions. Establish independent, prelocked elephant recording-group confirmation.

**P1 — Major:** Address **precision/alert burden** rather than maximizing sensitivity alone. Predefine a utility trade-off and compare nominal-only, N=1, N=2, fixed-time gap-aware persistence, and calibrated decision thresholds under equal data conditions. Keep Step-7F versus Step-7H RED semantics distinct.

**P1 — Major:** Investigate the **time-gap definition** using real timestamps and valid source-specific frame spacing. The counterfactual results demonstrate a measurable sensitivity to this modeling choice but do not show what interval should be chosen.

**P1 — Major:** For any proposed regime-aware conformal method, evaluate coverage/radius/false warnings jointly and separate exploratory recordings from an untouched confirmatory evaluation. Do not assert Mondrian or adaptation is novel without literature and mechanism analysis.

**P2 — Manuscript:** Obtain the current `Demo Journal Paper.docx` and reconcile reported Stage 6/7 and BAMBI experiments, alarm denominators and RED hierarchy with the present evidence.

## 8. Deliverables and reproducibility

- `verify_phase2.py`: stdlib replay of 16 Step-7F rows, all 256 checked fields, Step-7G selection, 22 Step-7H response-time conditions, per-source metrics and index-gap inventory.
- `geometry_and_gap_sensitivity.py`: independent Stage-7B formula checks on 214,956 rows and retrospective index-gap sensitivity comparison.
- `phase2_machine_audit.json`, `stage7b_formula_check.json`: machine-readable provenance and validation assertions.
- `step7f_independent_replay.csv`, `step7f_metric_cell_comparison.csv`, `step7g_selection_replay.csv`, `step7h_response_aware_independent_replay.csv`, source-wise metric tables and `index_gap_counterfactual.csv`.
- `forecast_index_gap_audit.csv`: per-scenario gap inventory.

To rerun, place the two original CSV files and the Step-7G JSON in one folder and run:

```bash
python verify_phase2.py --input-dir "/path/to/input" --output-dir "/path/to/output"
python geometry_and_gap_sensitivity.py --input-dir "/path/to/input" --output-dir "/path/to/output"
```

The audit scripts are independent reimplementations meant for verification; they do not replace the original research code, which should remain frozen. Neither the raw ~122 MB CSV nor the previous ZIPs are duplicated in this compact report package.
