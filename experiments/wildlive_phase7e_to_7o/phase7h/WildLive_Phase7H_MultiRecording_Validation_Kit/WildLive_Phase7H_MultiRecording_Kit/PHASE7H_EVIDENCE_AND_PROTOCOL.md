# Phase 7H — New-recording intake and locked frozen-model evaluation

**Research date:** 22 November 2026. **Phase 7H status:** CODE/INTAKE REGRESSION VALIDATED; **ZERO new WildLive sequence ZIPs provided; ZERO new independent capture groups verified; no new model ADE/FDE.**

## Purpose and frozen experiment

Carry the prior Phase 7F model and 7G descriptive comparisons into additional unseen aerial-elephant sequences **without** refitting weights, changing preprocessing scales or selecting sequences/models based on test results. Observation=20 **released frame indices**; prediction horizons=20/50 **released frame indices**, not 2/5 seconds. Output=image-plane normalized bounding-box centre (not physical motion). Frozen models: six original seed 42/52/62 checkpoints (two horizons) from `Main Result/transformer_results`. `checkpoint_lock.json` stores their **user-supplied Phase 7F run** SHA-256 hashes, and refuses a mismatching set (including R9 isotropic-ablation files). Original 7F heading rotation, task-specific scale floors and reconstruction are preserved in `src/phase7f_locked_reference.py`; evaluator never calls optimizer/train.

## Source evidence

- WildLive official Bristol portal: https://data.bristol.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69 — 22 UAV-acquired sequences, 19,139 total frames, 291 cross-species tracklets. Portal lists `boxid` sections for `DJI_0117_video2`, `DJI_0117_video3`, `DJI_0204_video2`, `DJI_0601_video2`, `DJI_0601_video6`, and `DJI_20230719145427_0002_V_video5`. **Whether any of these sequences contains elephants is UNVERIFIED.** Also, 22 files do not imply 22 independent camera flights.
- WildLive project: https://dat-nguyenvn.github.io/WildLive/; author-reported realtime processing frame rates ARE NOT verified input recording FPS.
- WildLive data are distributed with the portal's **Non-Commercial Government Licence**, not an assumed blanket redistribution permission. Preserve originals privately and review actual license terms before publishing data or derived annotation trajectories.
- Historical EXP3 `fresh_holdout_download_manifest.csv` in user-provided `EXP3_Output_Review.zip` contains six WildBox video basenames; these are embedded for **exact ID collision screening** in `src/phase7h.py`. A miss does **not** prove recording, session, flight, herd or individual independence. Two datasets can include the same source recording with different filenames/crops.

## Eligibility gates before confirmatory inference

1. Acquire *only* a new candidate's `boxid` JSON archive; check `imageWidth`, `imageHeight`, track ID, species label, duplicates and contiguous 20+20 or 20+50 released frame index runs. If no 40-frame continuous elephant run, 20-frame task ineligible; if no 70-frame run, 50-frame task ineligible.
2. Acquire 3–5 geographically non-sensitive sample JPEGs from that same candidate, align actual boxes, check that elephants are visible, identity remains consistent, annotation quality. Record any uncertainty. Never repair annotations silently.
3. Determine **capture-group independence** from acquisition event identifiers, flight/session date/time, camera trajectory, contextual footage and similarity to historical WildBox. Sequence names or separate ZIP files alone are insufficient. Make a source-level grouping decision *before* computing inference results. If unavailable, run exploratory analyses only.
4. Verify released frame timestamps/extraction cadence before assigning a result to elapsed seconds, even if another paper mentions 30 FPS.
5. Keep original checkpoints, six SHA hashes, canonicalization, baseline formulas, and both horizons locked. Predefined models: Stationary, endpoint CV20, historical OLS CV and frozen 3-seed Transformer ensemble. Do not select winning model/horizon post hoc.
6. Use **capture groups** (not windows, IDs or clips sharing a flight) as inferential units. Report every group, per track and per window as descriptive. Group-level uncertainty requires an adequate number of independent capture groups and prespecified analysis; this package intentionally computes **no p-values or confidence intervals**.
7. If no independent confirmation or timestamps, use exactly 'exploratory frame-index-based evaluation'; do not call the result 2-second/5-second accuracy, field warning, geofence prediction, conformal coverage transfer, or generalizable validation.

## Validation already performed

- Reopened the user's uploaded original `WildLive_DJI_0207_boxid.zip` as **a software-regression fixture only**. Its inventory: 868 annotated frame indices, 3 elephant tracks, 2,604 elephant boxes, 101 absent optional `score` fields, 2,487 windows for horizon 20 and 2,397 windows for horizon 50. Those counts exactly match Phase 7E, despite generalized frame/track handling.
- The Phase 7H evaluator does **not** run DJI_0207 as a *new* confirmation group; the aggregation script explicitly rejects `DJI_0207`. No new recordings supplied in this turn.
- Frozen-checkpoint SHA hashes are pinned to the **recorded** six bytes hashes in the user's Phase 7F manifest; physical checkpoint files are on the user's E: drive, not in the kit, and cannot be rehashed here. A successful Windows/WSL evaluation will rehash all six locally before model inference.
- Independent unit tests: see `src/test_phase7h.py` and test log below.

## Required outputs per candidate

`intake_manifest.json`, `track_availability.csv` (without model) → `evaluation_manifest.json`, `locked_model_per_window.csv`, `locked_model_per_track_and_pooled.csv`, `canonicalization_scale_floor_rates.csv` (when six checkpoint hashes match). `aggregate` produces descriptive **per-sequence** summary CSV only and explicitly reports **0 confirmed independent groups** until externally verified provenance is collected.

## Statistical interpretation

Frame-indexed box centres can move when drones move, box extents jitter, occlusions change, and apparent motion differs from world-plane motion. No physical scale or camera pose is asserted. Overlapping windows are highly dependent. Aggregate dataset mean computed from all windows would overweight long sequences; the aggregate command reports per-sequence model scores rather than pooling into a misleadingly precise N=thousands statistic.

**Next actual-data dependency:** at least one new candidate's `boxid` annotation ZIP, preferably multiple potentially independent groups with source provenance. Sample JPEG for visual species QA; no original 204.6 GiB download is necessary to screen candidate JSONs.
