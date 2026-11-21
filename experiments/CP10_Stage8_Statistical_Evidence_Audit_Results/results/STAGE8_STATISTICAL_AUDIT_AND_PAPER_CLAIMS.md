# CP10 — Stage 8 Statistical Evidence Audit (21 November 2026)

## Status

The corrected Stage 7F review ZIP completed without the previous `summarize()` error. All six frozen Transformer checkpoints (three seeds, two horizons) were evaluated. The final archive contains two prediction NPZ files, 3,110 per-predictor metric rows, 311 unique eligible windows, provenance records, and per-elephant summaries. Numerical replay from the NPZ arrays reproduced the reported sample ADE/FDE and stationary errors to floating-point precision (<1e-12). The source ZIP passed CRC verification.

**Scope:** these are retrospective, GT-eligibility-conditioned forecasts from one BIRDSAI TestReal sequence (`0000000353_0000000000`). Window timestamps overlap, and the source recording was used in earlier research; this is not a new external holdout or a GT-blind production score.

## Stage 7F paired ADE results

| Measure | 2 s | 5 s |
|---|---:|---:|
| GT-linked strict evaluation windows | 187 | 124 |
| Distinct GT elephants | 13 | 10 |
| Tracker-history Transformer ensemble ADE | 0.01066 | 0.02305 |
| Tracker-history stationary ADE | 0.01362 | 0.02828 |
| Window-weighted ADE reduction | 21.72% | 18.51% |
| Window-weighted absolute ADE gain | 0.00296 | 0.00523 |
| GT-identity macro ADE reduction | 16.37% | 17.62% |
| GT identities with positive mean ADE gain | 13/13 | 9/10 |
| GT-identity bootstrap 2.5% to 97.5% interval for absolute ADE gain | [0.00187, 0.00390] | [0.00212, 0.00720] |
| GT-identity bootstrap interval for relative ADE reduction | [14.5%, 27.1%] | [9.3%, 22.8%] |
| Transformer FDE | 0.01661 | 0.04288 |
| Stationary FDE | 0.02269 | 0.05098 |

### Statistical design

We calculate **paired differences** for each identical anchor window: `stationary ADE – frozen Transformer ADE`. Instead of pretending 187/124 overlapping windows are independent, we resample GT elephants **with replacement** (20,000 replicates, seed 20261121), carrying each selected elephant's complete set of windows and recomputing the pooled paired means each replicate. This returns within-video *identity-resampling descriptive intervals*. The intervals DO NOT reflect between-video shift or between-recording variability: there is one sequence and elephant motion/recording effects can correlate across identities. Do not present these intervals as cross-domain guarantees or an independent-significance test.

Window-weighted and identity-macro reductions differ (especially 2s) because elephants contribute unequal counts of overlapping windows. The manuscript should report **both**. Three model seeds are not three independent datasets; ensemble metrics and bootstrap samples should not be confused with independent replication.

### Eligibility and distribution caution

Offline ground truth was used to admit only histories with 20/20 GT-associated tracked positions and complete future annotations. Neither future annotations nor GT IDs were inputs to the frozen Transformer, but the evaluation population was selected with oracle information. Other TestReal sequences contributed **zero strict windows**. Therefore neither detector-level improvement nor HOTA enhancement implies a reliable online warning system. In particular, a lower tracker-history error than the GT-history diagnostic on this selected population does **not** prove more accurate elephant localization than ground truth.

## Cross-stage evidence synthesis

| Experiment | Result that is supported | What is **not** established |
|---|---|---|
| R8 / Stage 4 | Corrected-history Transformers outperform raw-history alternatives under same-target retrospective TestReal ablation (2s ~66.5% ADE reduction, 5s ~53.6%) | An independent dataset, exact 3D camera-movement correction, or metre accuracy |
| Stage 5 | Development-only conformal q90 transfers poorly (2s 3/53, 5s 2/28 pathwise coverage on selected evaluation videos), while frozen TrainReal q90 obtains 53/53 and 24/28 | Restored 90% conditional coverage, a universal calibration law, or independent-group generalization |
| Stage 6 | Fixed-reference GT crossing-event simulation detects reference inconsistencies and quantifies recall/false-run trade-offs; EXP3 N2 detects 116/133 (2s) and 109/124 (5s) stable events | GPS-anchored ground fence, verified future-depth projection, or deployment safety |
| Stage 7C | Fixed-protocol TestReal detector recall rises 5.10%→56.98%, with much higher false boxes/frame | Precision-preserving detector replacement or cross-dataset independence |
| Stage 7D | Same-protocol tracker replay HOTA rises 0.0252→0.2041; IDF1 rises 0.0214→0.1933 | Stable operational identities; ID switches also rise sharply |
| Stage 7E | Adapted tracker yields 187 strict 2s and 124 strict 5s GT-linked evaluation windows | GT-blind track eligibility or multi-sequence strict forecast availability |
| Stage 7F | Frozen Transformer improves conditional ADE over stationary predictions on one sequence | Deployment-grade end-to-end forecasting or generalizable risk alerts |

## Manuscript-ready conservative wording

"On a retrospective BIRDSAI TestReal sequence, the frozen three-seed Transformer ensemble achieved mean ADE of **0.01066** and **0.02305** normalized image-coordinate units for 2-s and 5-s horizons, respectively, when supplied with detector–ByteTrack histories passing a strict offline ground-truth association gate. These correspond to **21.7%** and **18.5%** lower mean ADE than stationary forecasts using the same tracked histories. Across 13 and 10 annotated elephant identities, respectively, the per-identity mean ADE advantage was positive for 13/13 and 9/10 animals. The analysis remains limited to one previously studied recording, overlapping forecast windows, and ground-truth-conditional eligibility; it does not establish fully automated field performance."

## Decision and next experiments

1. **Lock the positive results** (R8, Stage7C/7D, conditional Stage7F), retaining exact data and evaluation protocol definitions; do not generalize across distinct TestReal/Fresh/EXP3 populations.
2. **Report the failed or weak aspects** (Stage5 domain-shift calibration, Stage6 high false-run burden and assumed depth, Stage7C high FP/frame, only one sequence supporting strict downstream forecasts). These are scientifically necessary, not presentation errors.
3. A genuine next major confirmation requires **new, preregistered recording groups** and a fixed physical boundary/reference protocol. Thresholds, detector models and inclusion criteria must be locked beforehand. Without additional independent scenes, no statistical device can manufacture an external holdout.
4. For manuscript finalization, include tables clearly marked *annotation-conditioned retrospective*, *fixed-protocol previously studied TestReal*, and *exploratory geometry simulation*. Do not claim clinical/field/operational readiness.

## Files and reproducibility

- `Stage8_Prediction_Array_to_CSV_Verification.csv` — numerical NPZ-to-CSV verification for both arms, stationary and ensemble.
- `Stage8_Report_to_Samples_Consistency.csv` — corrected reporter outputs vs independently recomputed sample aggregates.
- `Stage8_GT_Clustered_Summary.csv` — paired means, identity-macro effects and bootstrap intervals.
- `Stage8_Per_GT_Identity_Differences.csv` and `Stage8_Paired_Window_Differences.csv` — audit and replot data.
- `Stage8_Cross_Experiment_Claim_Ledger.csv` — study-level claim status and source archives.
- `Stage8_Audit_Manifest.json` / `Stage8_Upstream_Archive_Hashes.json` — source provenance.
- `audit_stage8.py` — standalone rerun script: `python audit_stage8.py --input CP10_Stage7F_Tracker_Forecast_Review.zip --out ./Stage8_Rerun --replicates 20000`.
