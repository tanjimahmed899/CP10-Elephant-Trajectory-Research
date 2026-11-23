# Phase 7G — WildLive DJI_0207 Frozen Transformer Robustness Audit

**Date:** 22 November 2026  
**Status:** Numerical source-consistency and descriptive robustness audit **PASS**; independent scientific validation **NOT established**.

## 1. Inputs, integrity, and scope

- User-uploaded `Phase7F_Frozen_Results.zip` (ZIP CRC **PASS**), containing complete model-comparison summaries (64 rows), per-window metrics (4,884 rows), inference predictions, baseline-only records, model-checkpoint SHA-256 manifest, and canonicalization stats.
- User's previous `WildLive_Phase7E_Reproduced_Results.zip` was reopened to compare original input arrays against the frozen outputs. SHA-256 of its NPZ `35b2ae5a4b28d3fbfc6de0b6d657d177dbdf5c543461fb2b70f55e612512fa8e` **equals** the 7F recorded input SHA-256.
- All 64 aggregate model metric rows were independently recomputed from the corresponding CSV window records: maximum absolute discrepancy **5.21e-17**.
- All model ADE/FDE values available as prediction arrays were independently computed from frozen predictions and Phase 7E ground-truth arrays and matched the per-window CSV within **4.01e-8** (float32 prediction serialization).
- Checkpoint hashes are present in the 7F manifest, **but actual original `.pt` files are not in the user-uploaded ZIP**. Their bytes were not available to independently recompute SHA-256. This audit validates saved predictions and reported arithmetic, not the original training history or the physical capture provenance.

### Data/evidence unit

One DJI_0207 capture sequence (3 annotated elephant IDs, 868 indexed frames per ID). The 2,487 windows at 20-frame horizon and 2,397 windows at 50-frame horizon strongly overlap. They are **not** independent experimental groups; the three animals also share a camera and capture event. Published frame-time mapping, native FPS, camera stabilization/poses, physical-scale calibration, and certainty of WildBox non-overlap remain unverified. Model was trained on legacy 10-Hz trajectories, but the WildLive released-frame sampling cadence is unknown. Results are image-plane normalized-position error, **not meters or seconds**. x and y have been separately normalized by image dimensions; interpret magnitudes within this coordinate convention.

## 2. Pooled comparative results

The values below are means over overlapping prediction windows, *not* generalization population estimates.

| Task | Model | Windows | ADE (normalized) | FDE (normalized) |
|---|---|---:|---:|---:|
| 20 observed → 20 future frames | Stationary | 2,487 | 0.004827 | 0.008356 |
| 20 observed → 20 future frames | Endpoint CV20 | 2,487 | 0.004875 | 0.008539 |
| 20 observed → 20 future frames | EXP3 OLS CV | 2,487 | 0.004911 | 0.008605 |
| 20 observed → 20 future frames | Frozen transformer ensemble | 2,487 | **0.004522** | **0.007564** |
| 20 observed → 50 future frames | Stationary | 2,397 | **0.008416** | **0.015408** |
| 20 observed → 50 future frames | Endpoint CV20 | 2,397 | 0.009764 | 0.019471 |
| 20 observed → 50 future frames | EXP3 OLS CV | 2,397 | 0.009928 | 0.019861 |
| 20 observed → 50 future frames | Frozen transformer ensemble | 2,397 | 0.009539 | 0.017461 |

At a 20-frame future, the Transformer shows **6.31% lower ADE** than Stationary; at a 50-frame future, **13.33% higher ADE**. Percent changes use Stationary as denominator. T outperforms Stationary on **60.19%** of individual overlapping short-horizon windows and **41.09%** of long-horizon windows. These fractions are descriptive and are not independent-trial success rates.

## 3. Per-elephant robustness (stationary comparator)

| Forecast | Track ID | Transformer ADE | Stationary ADE | T minus S ADE | Better |
|---|---:|---:|---:|---:|---|
| 20 frames | 0 | 0.004582 | 0.004758 | -0.000176 | Transformer |
| 20 frames | 1 | 0.004133 | 0.004626 | -0.000493 | Transformer |
| 20 frames | 2 | 0.004851 | 0.005096 | -0.000245 | Transformer |
| 50 frames | 0 | 0.009624 | 0.007942 | +0.001682 | Stationary |
| 50 frames | 1 | 0.009035 | 0.008623 | +0.000412 | Stationary |
| 50 frames | 2 | 0.009957 | 0.008685 | +0.001272 | Stationary |

All three annotated IDs show the same pooled horizon-wise directional pattern; however, three co-observed elephants do not constitute three independent flights or herds. Model superiority does not necessarily extend to future captures.

## 4. Pre-identified annotation-jitter sensitivity

QA flags are taken **unchanged** from earlier annotated-image inspection (ID 0, frame 860; ID 2, frames 389 and 401); a window is flagged if these frames lie in the history or target. This selection is a *different subset*, not correction of the original ground truth.

| Task | Subset | Windows | Transformer ADE | Stationary ADE | T relative improvement |
|---|---|---:|---:|---:|---:|
| 20 frames | All | 2,487 | 0.004522 | 0.004827 | +6.31% |
| 20 frames | QA unflagged | 2,427 | 0.004384 | 0.004701 | +6.73% |
| 20 frames | QA flagged | 60 | 0.010084 | 0.009910 | -1.76% |
| 50 frames | All | 2,397 | 0.009539 | 0.008416 | -13.33% |
| 50 frames | QA unflagged | 2,307 | 0.009409 | 0.008361 | -12.54% |
| 50 frames | QA flagged | 90 | 0.012854 | 0.009842 | -30.60% |

A substantial error difference in the flagged subset is consistent with image annotation variability affecting trajectory calculations, but these data **cannot** isolate error causality (e.g. label jitter versus inherently difficult motion), and no replacement labels have been validated.

## 5. Temporal heterogeneity

Sequence positions were partitioned by starting frame into three approximately equal thirds, without random splitting or refitting:

| Task | Early third (T−S ADE) | Middle third | Late third |
|---|---:|---:|---:|
| 20 frames | −0.000037 | −0.000228 | −0.000648 |
| 50 frames | +0.002195 | +0.001377 | **−0.000200** |

Negative means Transformer lower error. At 50 future frames the late sequence has a small local reversal favoring Transformer, despite the **pooled** result favoring Stationary. This illustrates nonstationary segment effects but does not warrant causal attribution.

## 6. Nonoverlapping-within-ID offset sensitivity

For each task, a series of phase-offset subsets was formed by selecting start-frame indices `start % (20 + horizon) == offset`, yielding no shared frames between consecutive windows **for a given elephant ID** at a given offset. It does *not* eliminate between-elephant capture dependence or camera correlation, and phase-offset series themselves share source frames across different offsets.

| Task | Phase offsets | Fraction of offsets T better than S | Mean ADE-difference across offsets: range |
|---|---:|---:|---|
| 20 future frames | 40 | 36/40 (90.0%) | −0.000777 to +0.000038 |
| 50 future frames | 70 | 2/70 (2.9%) | −0.000182 to +0.003253 |

These are **sensitivity checks**, not bootstrap estimates, confidence intervals, p-values, or independent replication evidence.

## 7. Methodological implications for the journal paper

1. **Defensible positive result:** A frozen original three-seed Transformer ensemble executes on a distinct named RGB aerial annotation dataset and yields lower observed image-plane ADE/FDE than Stationary for a 20-frame future. The earlier weight file hashes and training code provenance are documented, without any new tuning on DJI_0207.
2. **Equally important negative result:** The same model underperforms Stationary at the 50-frame future. This is not obscured by global averaging or selective reporting. Per-track direction and the QA-unflagged subset are consistent with the pooled directions.
3. **What is not established:** Frame-to-second equivalence; physical world displacement; actual geofence crossing or real-time warning; conformal coverage transfer; domain-generalization performance; no overlap with historical WildBox footage; independence of observations. In particular, no Q1-level external confirmation can be inferred from 4,884 overlapping windows in one recording.
4. **Training-preprocessing caution:** Original scale floors activate in approximately 97% of WildLive windows (earlier Phase 7F preflight). The data-domain mismatch and uncalibrated camera motion may be consequential, but neither is demonstrated as the cause of the model's horizon failure.
5. **Ethical/legal:** Original dataset licensing should be respected. This summary-only package does **not** redistribute original frames, bounding boxes, checkpoint weights, or all per-window model predictions. Do not put restricted source assets or precise sensitive wildlife locations in a public repository.

### Publication-safe language

> In a strictly exploratory, frame-index-based evaluation on one externally sourced DJI_0207 aerial elephant sequence (three annotated tracks), a frozen three-seed temporal Transformer reduced pooled normalized image-plane ADE relative to a stationary baseline for a 20-frame forecast (0.004522 vs 0.004827), but increased error for a 50-frame forecast (0.009539 vs 0.008416). These overlapping windows do not constitute independent capture groups. Native frame timing and inter-dataset recording independence remain unverified; therefore no seconds-based performance, physical tracking accuracy, or external generalization claim is made.

## 8. Next scientific actions, in priority order

- Find additional genuinely independent elephant UAV sequences with native frame timestamps or explicit extraction cadence; conduct provenance checks with metadata and, where permitted, visual near-duplicate comparisons to the earlier WildBox source. A single new sequence does not resolve the inference-unit problem.
- Retain DJI_0207 as a prospective exploratory stress test and avoid changing checkpoint, preprocessing, or model selection based on its outcome if planning subsequent confirmatory evaluation.
- If pursuing annotation correction, commission/perform a **blinded, documented** reannotation of identified frames, keep old and new coordinate versions, and rerun the same locked evaluation as a sensitivity analysis.
- Evaluate 20/50-frame error in an additional independent recording before discussing model-specific domain effects; obtain verified cadence before calling them 2/5 seconds.
- For conformal or early-warning claims, explicitly evaluate warning labels and coverage under valid calibration/test separation. Those metrics have **not** been computed here.

## 9. Reproducibility

`src/phase7g_robustness_audit.py` reads the original user Phase 7E and 7F ZIPs, validates their SHA-256/CRC, recalculates all 64 CSV summary rows, independently recomputes per-window ADE/FDE from saved arrays, and reproduces the included charts and summaries. The complete source ZIP inputs must remain in a private working directory.
