# EXP4 Research Protocol — preregistration draft (NOT executed)

Status: Proposed, pending source-code and data audit. Date: 2026-11-22.

## 0. Evidence boundary

The user-provided Claude assessment discusses Stage 4, Stage 6 and EXP3, reports approximately 69.76% 5-second coverage and weak warning precision. These are **third-party review claims supplied in chat**, not independently confirmed in this session. This document does not assert their numerical correctness.

## 1. Research questions

- **RQ1 / geometry:** Does a common, verified reference frame materially change crossing labels and warning scores versus the current pipeline? Primary unit: crossing event within independent recording groups; include matched label disagreements and edge cases.
- **RQ2 / uncertainty:** Does a prespecified, observation-only motion-regime split improve coverage/radius balance versus horizon-specific pooled calibration, given sufficient calibration samples? Primary units: held-out recording groups and horizon/regime strata.
- **RQ3 / decisions:** Can a prespecified decision rule improve event-level alert precision at a matched recall level, without altering ground-truth labels during tuning? Primary units: independent events, false alerts per recording-hour when duration metadata permit, and lead time only for valid matched positives.

## 2. Work packages and gates

**Gate A: Input verification.** Read original Python files, configs, raw predictions, track IDs, timestamps and video metadata. Make a source-to-claim register with filenames, equations, figures and run settings. No experiments before Gate A.

**Gate B: Freeze historical EXP3.** Cryptographic SHA-256 inventory of originals; create working copies; register Python environment, package versions, seed, CLI commands, and all parameters. Do not overwrite.

**Gate C: Coordinate audit.** Define the intended boundary (image-plane, stabilized local reference, or ground plane). Confirm both trajectory forecast and ground-truth future observations are expressed in the same frame, with a defensible transform. Test stationary-animal/moving-camera and boundary-tangency cases. Quantify label agreement by recording and horizon; classify unmatched/undefined events explicitly.

**Gate D: Exploratory calibration.** First reproduce frozen and pooled horizon-specific split conformal baselines with exact EXP3 score. Only then try small prespecified regime families selected from past-observable variables (e.g. motion magnitude or scale-floor indicator). Avoid dozens of strata, target leakages, and unreported data-dependent bin selection. Compare observed coverage, radius, and per-group variation. Do not claim conditional guarantees under arbitrary domain shift.

**Gate E: Warning analysis.** With reference label frozen, compare a baseline warning rule to a candidate decision-aware rule using identical event-matching criteria. Preselect thresholds from development/calibration data; disclose matched-recall comparisons and false-alert burden. No retrospective tuning on final groups.

**Gate F: EXP5.** Obtain new untouched, provenance-documented recording groups with relevant elephant motion and ground-truth paths. If training/calibration used thermal BIRDSAI and new footage is RGB drone imagery, treat this as an explicit modality/domain transfer study, not same-distribution validation. Lock code and thresholds before opening test labels.

## 3. Preliminary acceptance criteria (must be locked before independent evaluation)

- Source-to-manuscript reproduction of all central numerical tables, or a transparent discrepancy report.
- No known split-level recording overlap; review near-duplicates and previously inspected test groups.
- Crossing labels either geometrically supported or described explicitly as image-relative.
- Report per-horizon empirical coverage and radius, including confidence/uncertainty with group-aware logic and group size; do NOT declare `>=88%` equal to nominal 90%.
- For warning claims, use TP, FP, FN and clearly defined denominators, plus recall/precision trade-off at matched operating points.
- Consider superiority only when appropriate held-out validation supports it. No publication claim solely from lower errors on already repeatedly inspected data.

## 4. Evidence register (empty until originals are reviewed)

| Claim ID | Claimed result | Original file | Code/function or page/table | Unit, n and split | Status | Correction |
|---|---|---|---|---|---|---|
| E01 | EXP3 trajectory baseline comparison | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | UNVERIFIED | Require raw forecast/error tables |
| E02 | 5-second conformal coverage | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | UNVERIFIED | Require calibration scores and test residuals |
| E03 | Stage 6 crossing-label differences | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | UNVERIFIED | Require per-window labels, coordinate transforms |
| E04 | N=2 alert precision | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | UNVERIFIED | Require alert/event matches and denominators |
| E05 | Forecastability yield | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | UNVERIFIED | Require tracker windows and GT match records |

## 5. Data restrictions

Do not use generic aerial footage as proof of elephant forecasting. Preserve file provenance and licenses. Neither a sample file nor a toy test is an empirical study. Distinguish development data, calibration data, exploratory evaluation data, and genuinely untouched prospective validation.

## 6. Draft paper contribution ladder

Level 1 (possible now, pending audit): reproducible failure analysis + system implementation, carefully limited.

Level 2 (requires verified geometry audit): reference-frame-consistent evaluation protocol + measured consequences.

Level 3 (requires actual experiments and new data): regime-aware + decision-aware uncertainty/warning method with fair baselines and independent evidence.

No claim of Q1-level novelty or field readiness follows automatically from any level.
