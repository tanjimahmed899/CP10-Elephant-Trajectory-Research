# EXP4 Phase 0 — Independent evidence audit and exploratory calibration feasibility

**Audit date:** 22 November 2026. **Scope:** Two uploaded archives, `All Python Code.zip` and `EXP3_Output_Review.zip`. No original videos, checkpoints, earlier experimental PDFs or current journal manuscript were provided in this session. This report distinguishes independent recomputation from claims contained only in saved reports. **No model training or new checkpoint inference was performed.**

## 1. Input inventory and source integrity

- `All Python Code.zip`: 269 ZIP entries including 267 `*.py` scripts (and a Python backup file); compressed 1.4 MB, uncompressed about 6.3 MB. ZIP CRC check passed, and all 267 Python scripts passed AST syntax parsing. **This does not establish runtime correctness or successful pipeline integration.**
- `EXP3_Output_Review.zip`: 36 ZIP entries; ZIP CRC check passed. Includes two frozen-prediction NPZ archives, two corresponding trajectory datasets, metadata CSVs, result tables, execution scripts, source-completeness summaries, a publisher-checksum verification report, and historical notes.
- Six source video ZIPs and model checkpoint `.pt` bytes were **not** present in the uploaded archives. The file `reports/EXP3_publisher_checksum_verification.json` reports six prior successful video hash verifications. Those original video bytes were not available for an independent checksum replay here.
- Older `reports/EXP3_current_input_preflight.json` indicates absent videos at an earlier state. Later saved `results/SUMMARY.txt`, NPZ arrays, `fresh_holdout_per_source.csv` and `EXP3_SOURCE_COMPLETENESS_STATUS.json` show an actual six-video result set. Treat the earlier missing-video preflight as historic, not as the current numerical state. A full end-to-end rerun still requires the original assets.

## 2. Saved-array numerical verification (independent recomputation)

Using `results/{task}_exp3_frozen_ensemble_predictions.npz`, per-window errors were recomputed as `||ensemble_prediction - future_abs||_2` at each future step, with **ADE** the mean over windows and future steps and **FDE** the mean final-step error. The **Stationary** comparator uses the last observed point as the prediction at every future step. All values are in the saved **normalized image-plane coordinate system**, not metres.

| Horizon | Forecastable windows | Videos | Capture groups | Transformer ensemble ADE / FDE | Stationary ADE / FDE | Ensemble ADE improvement over Stationary |
|---|---:|---:|---:|---:|---:|---:|
| 2 s | 754 | 6 | 3 | 0.0504879 / 0.0953141 | 0.0581736 / 0.1037781 | 13.21% |
| 5 s | 572 | 6 | 3 | 0.1127979 / 0.2065770 | 0.1171431 / 0.2100885 | 3.71% |

The eight recalculated primary values match the published summary to within `1e-6` absolute error. Source: `results/fresh_holdout_model_metrics.csv` and the two saved ensemble-prediction NPZ files. The code and full-precision comparison are in `run_audit.py` and `EXP3_independent_array_recalculation.csv`.

**Interpretation:** EXP3 contains genuine numeric evidence consistent with its saved arrays. This is not independent confirmation of the checkpoint provenance, original preprocessing, ground-truth annotation correctness, or identity of the six original videos. The reported frozen inference and publisher SHA verification remain historical provenance statements.

**Source heterogeneity:** At 5 s the ensemble per-video ADE ranges from 0.0551548 to 0.2114199. The latter is the `DJI_20240626143443_0003_V` video (`elep2`). This supports investigating group-specific failure, without asserting the cause is known.

## 3. Repeated-window dependence and evaluation units

| Horizon | elep1 windows / tracks | elep2 windows / tracks | elep3 windows / tracks | Total windows / distinct tracks |
|---|---:|---:|---:|---:|
| 2 s | 73 / 7 | 242 / 17 | 439 / 35 | 754 / 59 |
| 5 s | 52 / 7 | 191 / 16 | 329 / 27 | 572 / 50 |

Metadata show consecutive within-track forecast anchors are generally separated by **10 10-Hz frames**, while the prediction spans **20 or 50 steps**, meaning future target windows overlap substantially, particularly at 5 s. Consequently, `N=572` is *not* 572 independent trajectories. The stronger cluster counts are 50 tracked segments across six videos and three capture groups; independence of those 50 tracks across videos is also not established.

## 4. EXP4 exploratory calibration stress test — deliberately NOT a confirmatory result

**Method A, window-level descriptive stress test.** Define a maximum Euclidean forecast error across all future steps for each saved forecast window. For each horizon, leave out an entire `elep1`/`elep2`/`elep3` capture group; calibrate on the other two groups. Compare (i) pooled 90% higher-rank split-conformal empirical quantile and (ii) two-regime quantiles, where the regime is determined by `history_path` and the threshold is the median from calibration windows **only**. This deliberately does **not** replicate original Step 6C, which uses track-level canonical scores. The test uses previously seen groups and overlapping windows, so no coverage guarantee follows.

| Held-out group | Pooled 2-s window coverage | Regime 2-s | Pooled 5-s window coverage | Regime 5-s |
|---|---:|---:|---:|---:|
| elep1 | 95.89% | 94.52% | 92.31% | 100.00% |
| elep2 | 71.90% | 76.03% | 57.59% | 51.31% |
| elep3 | 97.27% | 92.71% | 100.00% | 91.79% |

**Method B, track-unit feasibility stress test.** Primary score is the maximum radial forecast error across all windows and future steps within a `track_key`, more closely matching the track-as-unit structure of historical Step 6C (but not its exact canonical score or scale-adaptive construction). Regime is determined from the first available window's observed `history_path`, ensuring no *subsequent* future window is needed for the regime label. Again, leave out each capture group in turn and calibrate on the remaining two.

| Held-out group | Pooled 2-s whole-track coverage | Regime 2-s | Pooled 5-s whole-track coverage | Regime 5-s |
|---|---:|---:|---:|---:|
| elep1 | 100.00% | 100.00% | 100.00% | 100.00% |
| elep2 | 47.06% | 52.94% | 37.50% | 43.75% |
| elep3 | 100.00% | 100.00% | 100.00% | 100.00% |

For `elep2` at 5 s, the average *track-level* radius increases from 0.364967 (pooled) to 0.550408 (regime), while coverage improves only from 37.5% to 43.75%: a large width cost for still-poor coverage. On `elep3` the track-level radius also grows (1.027867 to 1.490835) without a coverage gain (both 100%). This **does not support claiming that this simple regime-aware method fixes cross-group coverage**.

**Methodological caveats:** All groups were used in existing EXP3 analyses. These are retrospective exploratory calculations, **not** an untouched external test; conditioning on regime within the test group does not restore coverage guarantees under distribution shift. Group count is three. No clustered confidence intervals or statistical significance tests are reported. This experiment evaluates raw normalized image-plane Euclidean radii, *not* the original learned model's canonical conformal radii or the Stage 7H warning system. Neither the window-level nor the track-level method should be quoted as a validated new paper contribution.

Source: `EXP4_EXPLORATORY_leave_capture_group_out.csv` and `EXP4_EXPLORATORY_track_unit.csv`. Execution: `run_audit.py` and `track_aware_stress_test.py`.

## 5. Code/protocol inspection — issues, severity, and actions

| Severity | Location | Observation and consequence | Required action |
|---|---|---|---|
| **CRITICAL for claims** | `results/` and `scripts/`, EXP3 | No per-sample Step 7H virtual-geofence label/alert table is contained in EXP3 package. It is not possible to verify the reported 6.61%/9.53% crossing-label changes or an improved warning rule from this input. | Obtain `step7b_v2_all_geofence_instance_alarms.csv`, `STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json`, Stage 6 original geometry/canonical transforms, and the relevant output evidence before rewriting claims. |
| **MAJOR** | `step6c_crossfit_conformal.py` | Historical conformal primary unit is **track**, with primary score maximum canonical Euclidean error across all windows and steps in a track. Direct window-wise conformal calculations are methodologically non-equivalent and risk overstating precision. | Reconstruct exact TrainReal calibration track scores and frozen q, then compare identical operational units in EXP4. |
| **MAJOR** | Metadata in both saved tasks | Many overlapping windows, only three video capture groups. Naive window-wise standard errors and nominal independent-sample conformal assumptions are inappropriate. | Keep capture-group separation, track-level scores, and evaluate on genuinely uninspected new groups. |
| **MAJOR** | `results/fresh_holdout_model_metrics.csv` | Improvement over Stationary is modest at 5 s (3.71% ADE, 1.67% FDE) and source-dependent. | Preserve fair same-sample baselines, report source/track-macro outcomes, and test whether any change exceeds trivial alternatives. |
| **MAJOR** | `step7h_final_response_aware_with_locked_persistence.py`, lines 17–36 and 100–175 | The final warning script requires a pre-existing label/alert CSV and a locked `N=2` operating-rule JSON; neither was in the EXP3 upload. The persistence implementation uses rolling adjacent retained sample rows, without independently validating temporal gap continuity. This is a code-level risk, **not evidence** that the original output was wrong. | Audit sample-index/time continuity and re-evaluate fixed-frame labels from the original scenario input. |
| **MAJOR** | `results/EXP3_SOURCE_COMPLETENESS_STATUS.json` and `exp3_postprocess.py` | `new_video_evaluation_ran: true` is written from the existence of per-source output CSVs; by itself this flag is not direct proof that the original forecaster was rerun. NPZ output is strong supportive evidence, but source/checkpoint hashes and process logs are needed for full provenance. | Add a source-to-checkpoint-to-prediction manifest and retain executable receipts. |
| **MINOR (conditional runtime bug)** | `R19_final_claim_matrix_and_paper_rewrite_lock.py`, lines 83–105 | Code accepts a top-level JSON list to collect statuses but then unconditionally calls `data.get('step', '')`. If such a list contains status-bearing objects, execution raises `AttributeError`. | Use `data.get('step', '') if isinstance(data, dict) else ''` or define an explicit list-step extraction policy. |

**Verified and not verified:** ZIP CRCs, AST syntax, saved-array forecasting errors and the exploratory retrospective quantiles were directly checked here. Full training, video ingestion, detector performance, Stage 6 canonical calibration, geofence ground truth, warning precision, Raspberry Pi latency, and scholarly novelty remain unverified by these two uploads.

## 6. Next execution gate: EXP4 rather than paper-writing

1. **Freeze source evidence.** Retain both original ZIPs unchanged, with SHA-256; preserve the original EXP3 results separately from all new calculations. The new scripts read original arrays without writing back.
2. **Recover source-exact conformal artifacts.** The primary required inputs are original `crossfit_track_calibration_scores.csv` / task-level q calibration objects, the exact checkpoint and canonicalization definitions, and historical test per-track score outputs. If unavailable, rebuild from genuinely available TrainReal residuals after provenance verification.
3. **Recover virtual-geofence evidence.** Provide original Step 7B/7G/7H CSV/JSON outputs and ground-truth/frozen-reference coordinate transforms. Verify identical boundary frame and event definition for truth and forecast. Do not re-label past experiment as an independent test.
4. **Pre-register one method comparison.** Compare original frozen conformal, pooled recalibration (if legitimately permitted), simple Mondrian, and a justified candidate group-robust method under the *same* score definition and track-group unit. Prespecify coverage, radius budget and warning burden jointly; record model-selection data separately from an unseen test.
5. **Independent evaluation.** Obtain different, genuinely uninspected recording/capture groups before claiming confirmatory calibration transfer, operational warning reliability or generalization. New RGB footage cannot be treated as in-domain thermal data without a separate domain-shift protocol.
6. **Manuscript update only after validation.** Figures and tables must cite exact source artifacts; do not add invented Stage 7 numbers or convert image-plane distances to metres.

**Current conclusion:** The EXP3 saved array-level forecasting numbers are reproducible. A naive two-regime conformal fix is not consistently effective, especially on `elep2`. The highest-value next task is a source-exact calibration-and-warning audit with original Stage 6/7 files, not immediate manuscript rewriting or claiming a new calibrated method.
