# EXP4 — Phase 1: Source-Level Calibration and Warning-Protocol Audit

**Date:** 22 November 2026  
**Evidence status:** A reproducible numerical audit of the user-supplied `crossfit_track_calibration_scores.csv` and the provisional JSON lock; arithmetic checks of *reported* tables in four supplied PDFs. This is **not** a full execution of the detection-to-warning system and does **not** validate generalization or new methodological novelty.

## 1. Audited source set and scope

**Primary machine-readable evidence**
- `crossfit_track_calibration_scores.csv` (96 calibration-track rows; 8 fields)
- `STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json`
- `All Python Code.zip`: original `step6c_crossfit_conformal.py`, `step7f_temporal_persistence_hysteresis_diagnostic.py`, `step7g_pareto_and_provisional_operating_rule_lock.py`, and `step7b_v2_response_aware_green_yellow_red.py`

**Supporting reports**
- `Experiment Dataset (Part-2).pdf`, pp. 2–6 (calibration and external BAMBI data)
- `Experimental Results (Part-4).pdf`, pp. 2, 4–5, 10–18 (EXP3, exploratory calibration, fixed-reference warning, and detector-tracker-forecast linkage)
- `Experimental Data (Part-03).pdf`, pp. 5–9 (frozen comparative results, warning hierarchy, claim lock)
- `Experimental Results (Part-1).pdf`, pp. 1–7 (earlier trajectory protocol and model results)

The code in the ZIP is inspected as a source artifact rather than executed end-to-end. Prior EXP3 saved prediction arrays had been audited in Phase 0. The present additions do not include individual Step-7F persistence table or Step-7B v2 geofence-instance warning rows.

## 2. Verified: track-level conformal quantiles

The original `step6c_crossfit_conformal.py` defines the calibration unit as **track**; each track's primary score is the maximum canonical Euclidean forecast error across all its future steps and overlapping windows. The finite-sample rank is `ceil((n+1)(1-alpha))`. The CSV contains the exact aggregated scores needed to replay that quantile operation.

| Task | Tracks | Windows represented | Target | Rank | Recomputed q | Source-PDF q | Calibration-score fraction <= q |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2s→2s | 62 | 531 | 90% | 57 | 2.158767888280567 | 2.158767888280567 | 91.935% |
| 2s→2s | 62 | 531 | 95% | 60 | 2.560772554862189 | 2.560772554862189 | 96.774% |
| 2s→5s | 34 | 382 | 90% | 32 | 1.975129668255982 | 1.975129668255982 | 94.118% |
| 2s→5s | 34 | 382 | 95% | 34 | 2.950976253259077 | 2.950976253259077 | 100.000% |

All four values match `Experiment Dataset (Part-2).pdf`, p. 3, exactly to the stored double-precision values. These numbers alone are **not** external coverage validation; they are self-consistency of the finite sample calibration-score quantiles.

### Severe source concentration

| Task | Source | Tracks | Fraction of task's tracks | Windows | Fraction of task's windows |
|---|---|---:|---:|---:|---:|
| 2s→2s | 0367 | 52/62 | 83.87% | 515/531 | 96.99% |
| 2s→5s | 0367 | 32/34 | 94.12% | 380/382 | 99.48% |

This concentration is directly verified from the CSV. It is a **major generalization limitation**: track-level cross-fitting and track-disjoint outer folds do not produce independent capture groups. Report both the track and source-group denominators. A guarantee across arbitrary unseen recording domains does **not** follow from these data.

**Further technical caveat:** The original script's helper clamps `ceil((n+1)(1-alpha))` to `n`. This has no effect for the two current primary calibration pools at alpha 0.10 and 0.05, where ranks are within 1…n. If reused for small Mondrian strata, however, standard finite-sample conformal theory requires treating rank > n as an infinite quantile, not silently returning the maximum observed score. For a 90% target a 2-track stratum is too small for a finite conformal quantile under that convention. Any proposed stratified extension must explicitly handle this case.

**Separate theoretical caveat:** Scores are **out-of-fold** from multiple cross-fitted model fits, whereas frozen external inference may use a separately trained final model. Therefore, avoid presenting this as an automatically valid ordinary split-conformal exchangeability guarantee for the frozen ensemble. Formal guarantee requires a theorem matching the actual cross-fit algorithm and sampling protocol, or appropriately qualified empirical evaluation.

## 3. Verified: interpretation of the provisional operating-rule lock

`STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json` explicitly sets N=2 consecutive **available forecast windows**, q=90%, selection on BIRDSAI TestReal, and `selection_is_post_hoc_exploratory=true`. Original `step7g_pareto_and_provisional_operating_rule_lock.py` selects the smallest of N={2,3,5} satisfying BOTH on both horizons: at least 20% relative false-alert-rate reduction against N=1 and sensitivity >=80%.

- **Verified:** JSON schema values agree with the selection code, including post-hoc status and the intended dataset.
- **Not independently verified:** N=2 *actually* satisfies those numerical conditions, because `TABLE_step7f_persistence_tradeoff.csv` is absent. The SHA-256 in the lock refers to that absent result table, not to the code or JSON file.
- **Do not interpret N=2 as two seconds:** `step7f_temporal_persistence_hysteresis_diagnostic.py` orders available windows and uses a rolling count; it does not require a fixed maximal physical-time gap. Its 'consecutive available' is not a guarantee of consecutive, evenly spaced timestamps.

### Important: two distinct RED definitions

- **Step-7G persistence diagnostic:** RED if *any* nominal predicted geofence crossing, regardless of time-to-event; YELLOW for persistence-qualified uncertainty intersection otherwise.
- **Step-7B v2 / response-aware Step-7H:** RED only if predicted crossing occurs at/before a response-time threshold (or an already-at/inside special case); nominal crossing after that threshold is YELLOW, as is appropriate uncertainty intersection.

These algorithms are deliberately different; they are **not interchangeable operational rules**. The PDF labeled `Experiment Dataset (Part-2).pdf`, p. 3, uses the response-aware definitions, whereas the uploaded provisional JSON is the simpler Step-7G rule. In the manuscript, use exact step names and avoid associating RED-specific metrics with the wrong rule.

## 4. Source-supported, not independently rerun: fixed-reference warning evaluation

`Experimental Results (Part-4).pdf`, pp. 10–11, reports the following simulated-boundary comparison. Ratios were independently recalculated from its tabulated counts:

| Reported quantity | 2 s | 5 s |
|---|---:|---:|
| Boundary cases | 4,524 | 3,432 |
| Changed labels after reference correction | 299 / 4,524 = **6.61%** | 327 / 3,432 = **9.53%** |
| Stable annotated events | 133 | 124 |
| N=2 matched stable events | 116 / 133 = **87.22%** | 109 / 124 = **87.90%** |
| Unassociated N=2 alert runs | 348 / 463 = **75.16%** | 237 / 347 = **68.30%** |

**Do not label the last line as clinical/field false-alarm precision.** It is a run-association fraction inside a synthetic fixed-reference, retrospective event simulation, not false alarms per hour or real conflict-event precision.

The fixed-reference ground-truth geometry uses **known annotated future 3-D centers/depth** (explicit in Fig. 10, p. 11). This is useful as a diagnostic *oracle/reference replay*, not a deployable online camera-only geofencing method. To propose a runtime method, produce an independently validated way to estimate the relevant reference mapping and metric scale using only causally available observations.

**Missing for exact replication:** per-case original/replayed boundary labels, line identities, GT 3-D center/depth tracks, and camera transforms; plus the `step7b_v2_all_geofence_instance_alarms.csv` with individual warning decisions.

## 5. Newer results constrain novelty and 'end-to-end' claims

`Experimental Results (Part-4).pdf` adds post-EXP3 results which must be reflected in any later manuscript:

- **Stage 7C detector shift (p. 13):** original IoU 0.50 recall **5.10%** versus adapted **56.98%** on previously used BIRDSAI TestReal. Precision falls from **57.21%** to **34.20%**; false boxes/frame rise from **0.202** to **5.799**. This is a significant recall/false-positive trade-off, not a uniformly better detector.
- **Stage 7D tracking (p. 14):** ByteTrack HOTA **0.025220→0.204086**, IDF1 **0.021390→0.193310**, but identity switches **43→1,307** and fragmentations **145→3,597**. Increased coverage is accompanied by strong association instability.
- **Stage 7E strict eligibility (pp. 15–16):** adapted potential windows **371/191** at 2/5 s become **187/124** strict GT-linked windows (50.4% / 64.9% retention). *All* strict windows come from sequence **0353**, not six independent sequence-level successful forecasts.
- **Stage 7F conditional forecasting (p. 16):** on those strict, one-sequence GT-linked windows, Transformer ADE **0.010664 vs Stationary 0.013623** (2 s) and **0.023050 vs 0.028285** (5 s). These are positive *conditional*, retrospectively selected, one-sequence results; not representative unrestricted detector-to-warning effectiveness.
- **BAMBI Flight 6 (Experiment Dataset Part-2, pp. 4–7):** red-deer/human, not elephant, cross-species stress evaluation. Frozen 5 s q90 pathwise coverage is **1.1236%** in one original metre-space evaluation and **2.5682%** in the separately reconstructed compatible normalized evaluation. These are different representations/protocol stages; neither supports external 90% coverage. The 2 s coverage is reported as 100% with very conservative regions.

The new material strengthens the scientific case for quantifying the **forecastability bottleneck**, but no prospective elephant-specific end-to-end demonstration is established.

## 6. Reported percentage ambiguity requiring manuscript cleanup

`Experiment Dataset (Part-2).pdf`, p. 2, gives TestReal landmark residual median **0.705481 px → 0.203296 px** alongside a *reported* 46.846% reduction. The simple reduction of the two displayed medians is **71.18%**, not 46.846%. `Experimental Results (Part-4).pdf`, p. 4, already warns that different aggregation methods are involved. **Do not silently change the published statistic**: either provide the underlying paired per-point reduction formula supporting 46.846%, or explicitly name it as a differently aggregated statistic. The same caution applies to the TrainReal row (1.200805→0.753953 px, displayed 7.480% versus 37.21% reduction of displayed medians).

## 7. Ranked research issues and concrete remedies

| Priority | Severity | Issue | Scientific consequence | Required correction |
|---|---|---|---|---|
| P0 | **Critical** | No untouched source-group elephant confirmatory evaluation for the newly proposed calibration/warning method | Adaptation is retrospective; Q1 methodological claims cannot be confirmed | Acquire/lock independent elephant recording groups before method tuning; evaluate frozen algorithm once |
| P0 | **Major** | Stage-7G vs Step-7H RED hierarchy differs | Misattributed warning metrics | Label both protocols explicitly and select one primary rule before any new test |
| P0 | **Major** | Fixed-reference GT uses future annotated depth | Offline oracle can leak future information into an operational claim | Treat as GT-only evaluation; build causally available runtime coordinate mapping separately |
| P1 | **Major** | Calibration source concentration 0367 at 94.12% of 5s tracks | Fragile cross-source calibration | Show source-wise score analysis and sample counts; do not claim guaranteed domain-robust Mondrian calibration |
| P1 | **Major** | Detector precision/track fragmentation collapse under recall adaptation | End-to-end alert burden and coverage cannot be deduced from oracle forecasts | Measure end-to-end failure/yield and event-level alerts for every source |
| P1 | **Major** | Missing Step-7F and Step-7B instance tables | N=2 selection and detailed alerts not independently rerunnable | Supply exact tables and source hashes; run locked replay |
| P2 | **Minor/Major depending on claim** | Residual-motion percentage uses opaque aggregation | Confusing or seemingly contradictory numerical claims | Publish exact reduction definitions and paired values |

## 8. Phase 2 experiment plan (do not call completed)

1. **Source-exact Step-7F/7G replay:** supply raw `step7b_v2_all_geofence_instance_alarms.csv` and `TABLE_step7f_persistence_tradeoff.csv`; verify JSON `source_step7f_sha256`; independently compute TP/FP/FN/TN at each horizon and N, with explicit denominators. Retain original JSON unchanged.
2. **Fixed-reference diagnostic replay:** get per-case GT coordinates/transforms/labels; rerun moving-anchor vs fixed-reference crossing with exact same scenario IDs; verify 299/4524 and 327/3432 from individual rows. Run depth-perturbation sensitivity tests and ensure online claims do not use future GT.
3. **Exploratory calibration:** use already-inspected recordings only for hypothesis development. Compare frozen TrainReal q90, pooled/group-equal recalibration, and *only feasible* pre-specified regime bins. Report nominal target, actual coverage, radius inflation, worst-group coverage and group denominators. Handle small-stratum rank>n properly; no invented guarantee.
4. **Group-level independent assessment:** lock calibration method, warning policy, training checkpoint, primary metrics and acceptance criteria; then evaluate on genuinely untouched, sufficiently varied **elephant** recording groups. Treat BAMBI as cross-species stress only.
5. **Manuscript source-to-claim pass:** once `Demo Journal Paper.docx` is provided, reconcile claims against the above evidence, mark historical observations separately from new proposed methods, and review references.

## 9. Files produced in this audit

- `verify_phase1.py` — rerunnable numerical and schema checks (Python 3, NumPy, pandas).
- `conformal_quantile_verification.csv` — four track-level quantile checks.
- `calibration_source_concentration.csv` — task/source-track and window count audit.
- `stage6_pdf_arithmetic_checks.csv` — derived ratios *from the report tables only*.
- `bambi_representation_separation.csv` — explicit metre vs normalized stress-test metadata.
- `machine_readable_audit.json` — hashes and unresolved evidence requirements.

**No new elephant experiment, retraining, online warning run, or prospective validation was performed in Phase 1.**
