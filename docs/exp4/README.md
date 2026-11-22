# EXP4 research extension — Phases 0–7

This directory is an **additional research archive**, intended to sit alongside the pre-existing CP10 project files. It does not replace or revise the historical code, saved predictions, or original manuscript.

## Evidence status

- **Phases 0–2:** Retrospective numerical replay and source-to-claim auditing on previously supplied experimental evidence.
- **Phase 4:** Editorial consistency updates based on those sources; revised unpublished manuscript is intentionally **not** part of this public add-on.
- **Phase 5:** **Exploratory** threshold/clearance-gate warning-policy ablation on previously examined cohorts. No new recording-group confirmation, no novel conformal validity theorem and no improvement at the 5-second horizon under the chosen criterion.
- **Phase 6:** Preregistration and independent-validation **design only**; no new confirmatory recordings were available.
- **Phase 7:** Dataset sourcing and preflight **preparation only**; no new confirmatory results were produced.

Read phase-specific source notes under `docs/exp4/`, compact result tables under `experiments/exp4/`, code under `scripts/exp4/`, and input schemas under `templates/exp4/`.

## Important distinctions

These files use different **evaluation populations and counting units**: trajectory windows; track-level conformal calibration scores; moving-anchor virtual-boundary instances; fixed-reference crossing events; and distinct source/video/capture groups. Their performance numbers are not interchangeable or poolable.

Forecast errors reported in normalized image coordinates are **not** physical metres. Camera-reference and cross-species stress tests are **not** evidence of deployed elephant-conflict reduction. Some retrospective fixed-reference diagnostics use annotated depth/GT for evaluation only.

## Running scripts

Phase 0 scripts expect original saved EXP3 arrays and metadata. Phase 1 needs the original calibration scores and lock file. Phases 2/5 need the historical Step7B instance-level source CSV (~122 MB) and Step7F/Step7G artifacts. Source files are **not** included here because they may already be hosted in the repository or subject to access, sensitivity, or GitHub size limits. Scripts contain their own CLI argument/help descriptions; do not assume a universal one-command pipeline. New groups must pass Phase 6/7 intake checks before any claims of external validation.

## Public repository release gate

Do **not** add third-party videos, images, dataset archive copies, raw GPS/camera metadata, unpublished manuscript drafts, passwords, API tokens, or third-party model weights without explicit rights review. Check `.gitignore`, file sizes, terms of use, and paper submission requirements before committing. Do not upload the raw `step7b_v2_all_geofence_instance_alarms.csv` (~122 MB) with GitHub's normal browser uploader.

**Replication caveat:** The small result tables in this add-on document prior analysis, but full reproduction still depends on original code, input arrays, checkpoints, compatible dependencies, and dataset permissions.
