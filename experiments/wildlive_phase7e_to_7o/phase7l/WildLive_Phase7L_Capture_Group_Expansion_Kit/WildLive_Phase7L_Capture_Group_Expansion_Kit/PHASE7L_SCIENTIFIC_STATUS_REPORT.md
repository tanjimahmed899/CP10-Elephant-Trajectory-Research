# CP 1.0 — Phase 7L: Additional recording screening & capture-event provenance

Date: 22 November 2026. **Status: reproducible candidate/verification kit implemented; fresh elephant-positive candidates and independent capture events NOT yet verified.**

## Evidence already established

- Inspected the exact original Phase 7K prospective lock and locked-model source hashes. Copied its bytes unchanged into `protocol/PHASE7K_ORIGINAL_LOCK_DO_NOT_EDIT.json`; a pinned SHA-256 is checked in code. Primary endpoint remains recording/capture-event-level (Adaptive − CV-OLS) ADE at frame horizons 20 and 50.
- Independently reopened the uploaded `Phase7K_DJI0161_Locked_Evaluation.zip` and `DJI_0161_video1_boxid.zip`, verified evaluation-manifest gates, original checkpoint hashes as reported by the locked evaluation, annotation SHA-256, 2,892 unique scored windows, exact adaptive-to-selected-expert joins, and summary means. This is **output-level** re-verification; checkpoint binaries and predicted coordinates are not in the ZIP, so no Transformer forward rerun.
- Recomputed for DJI_0161_video1: 20-frame adaptive ADE 0.0111948076362 versus CV-OLS 0.0129404936255 (13.4901% lower); 50-frame adaptive ADE 0.0215236900759 versus CV-OLS 0.0282699089145 (23.8636% lower). This is **not a fresh external recording** and should not be counted twice. It does not beat the best individual expert on either horizon.
- Official University of Bristol WildLive root currently lists **22 sequence folders**, which are not evidence of 22 independent flights. A matching 22-row candidate register and conservative filename-family screening flag are included. Directory checks establish `boxid` folders exist for `DJI_0142_video1`, `DJI_0117_video1`, `DJI_0117_video4`, `DJI_0601_video4`, and `DJI_0601_video5`, but **not their elephant labels**.
- Current selected species screen candidates: `DJI_0142_video1`, `DJI_0117_video1`, `DJI_0117_video4`, `DJI_0601_video4`, `DJI_0601_video5`, `DJI_20230719145816_0003_V_video2`.
- The sibling `DJI_0204_video1` has a prior sampled elephant label but belongs to the same *name family* as examined development `DJI_0204_video2`. Prefer independent-family candidates first, while remembering different filename families are also not proof of independence.

## Evidence missing

1. Complete new elephant-labelled annotation ZIP(s) from an unexamined candidate.
2. Ground-truth visual identity QA of sampled images from that candidate.
3. Original flight/herd provenance, source time, flight telemetry, comparison for overlap and continuity, supervisor review.
4. Original recording FPS/time metadata, image stabilization/camera-motion correction, conversion to world-coordinate metres.
5. Several truly independent capture-event evaluation units before any formal inferential/statistical generalization.

## Scientific controls

- **No gate training, optimizer tuning, expert swaps, source exclusion or threshold changes based on outcomes from DJI_0161 or future test recordings.**
- Candidate species preview is **label-only** and source-path specific; never claim negative species on sparse samples.
- Same-name-family sequences are related-collection **risk flags**, not automatically the same flight.
- Raw frame pixels and source annotations remain private. The public dataset licence and conservation ethics govern redistribution.
- This is not an externally timestamped preregistration and has no confirmatory p-values.
- Retain explicit failure to outperform best individual expert, not only wins relative to the predeclared CV-OLS comparator.

## Next actionable task

Run `phase7l_screen_selected.sh` from Ubuntu (uses original Phase 7K screening code). Upload `selected_candidate_species_samples.csv`; choose a new elephant-positive source only after inspecting its labels. Download complete boxid and representative JPEGs, upload them for QA, then run the locked Phase 7K evaluation unchanged.

## Primary sources

- Official dataset root: https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/
- Official dataset portal and licence: https://data.bristol.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69
- Dataset README: https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/Readme.txt
