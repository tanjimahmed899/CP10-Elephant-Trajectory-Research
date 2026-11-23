# CP 1.0 — Phase 7K, pre-outcome prospective validation preparation

Date: 22 November 2026. **Status: locked preparation kit developed and regression-tested; NO fresh evaluation results; independent confirmation NOT established.**

## Scope actually completed

1. Reopened original Phase 7H multi-recording prediction code, Phase 7I observation-only motion feature function, Phase 7J adaptive gate source, and both Phase 7J released development gate JSONs from prior project evidence.
2. Copied **exact original** development gate JSONs, 20-frame SHA256 `2988bf39a996016b1bd67026c7fe97ffa2756576114e007dcc5f2f047a5dd334` and 50-frame SHA256 `208d21c1d65734c5a96611eac4af7ea14bfd78d67f5f43f06c30d1bd2159e763`. Checkpoint lock holds all six original `.pt` SHA256 and sizes, not checkpoint binaries.
3. Created `prospective_lock.json`: history=20 frames, horizon=20/50 frames, normalized image-plane coordinates, original checkpoint ensemble, fixed released 7J ridge models, primary comparator **CV-OLS**, primary unit **verified capture event** (not individual window). A smaller adaptive ADE than CV-OLS is favorable. This is a candidate lock file and protocol DRAFT, **not a claim of external preregistration**.
4. Created executable `phase7k.py` with SHA256 checks; annotation intake; label-only remote preview; observation-only gate choice; original three-seed frozen-model inference via imported Phase7H code; exact one-to-one score join; descriptive per-track summaries; guard against promoting previously inspected sequences to new confirmation; descriptive multi-sequence aggregation with checkpoint/gate manifest consistency.
5. Ran **16 regression/structural tests successfully** in authoring environment, including exact Phase7I motion-feature comparison against archived DJI_0204 development outputs, gate-choice/ground-truth matching, unique keys, leak checks and rejection of prior development recording as new confirmation. No original six `.pt` binaries were available within the authoring container. Accordingly, actual 7K forward inference on new footage was **not** executed or independently re-verified here.
6. Compared reproduced Phase7J gates with originally released files. They were **not byte-identical**: 20-frame reproduced SHA256 `b7ce2d11a6615aeab984adc2f2f3bd416fd581bc217be126dafeef26ebbad3bc`; 50-frame reproduced SHA256 `235ed70b5d5dfc6010da5f66b4fe70c02b7a2711e81cfeedc22d9938486ae0b9`. Numerical regression weights differed at <=~1.9e-15 and <=~5.9e-16 respectively, consistent with cross-runtime floating point roundoff. This package explicitly locks the ORIGINAL released JSON bytes; it is wrong to label the reproduced gate files as hash-identical.

## External source screening (not entire new ZIP intake)

University of Bristol source listing confirms 22 UAV sequence folders. On 2026-11-22, directly inspected `boxid/frame_0.json` for new candidates:
- `DJI_0161_video1`: contains an **elephant** label (at least one). Prioritized for full JSON intake and sampled visual QA. Counts/continuity/independence **unverified**.
- `DJI_0204_video1`: contains several **elephant** labels. Treated as **development-related/high risk** due shared `DJI_0204` stem with previously inspected `DJI_0204_video2` until flight/herd metadata/content proven distinct.
- `DJI_0133_video3`: `giraffe` in frame 0, not evidence of elephant presence.
- `DJI_0119`: `zebra` in frame 0, not evidence of elephant presence.

Prior user sample-screen results already flagged other zebra-only samples. **Never infer whole-sequence absence from a few selected frames.**

Official sources:
- Dataset DOI: https://doi.org/10.5523/bris.ewnuoroebuae20vei0cu6zu69
- Root listing: https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/
- README: https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/Readme.txt
- Licence portal: https://data.bristol.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69

## Critical scientific caveats

- No verified FPS, timestamps, camera motion compensation or metric-scale locations. Frame-index ADE/FDE only, no seconds/metres/geofence utility.
- `DJI_0207` and `DJI_0204_video2` are pre-examined development sources. Their earlier 7J adaptive results were not favorable versus the best fixed experts on either examined recording. No positive method claim follows from the new code.
- New filename or distinct frame hashes cannot establish independent flight/herd groups. Source-overlap, recording continuity and visual species checks remain necessary.
- WildLive is governed by a Non-Commercial Government Licence listed on the official University of Bristol portal. Review licensing before dissemination of the images/annotations or derived artifacts.
- Results from a new recording must not be used to change the ridge gate/primary comparison and then be called confirmation. If a method changes after viewing it, label that recording development and seek additional untouched data.

## Next empirical task

Run the ZIP's `README_UBUNTU.md` commands for **DJI_0161_video1**: remote label-only preview, full annotation ZIP, intake, five sample RGB frames, then user review of identity/overlap. Only after QA use unchanged original six checkpoints to generate results, and preserve provenance metadata for meaningful group-level interpretation. No email to authors is required to perform this frame-based exploratory intake.
