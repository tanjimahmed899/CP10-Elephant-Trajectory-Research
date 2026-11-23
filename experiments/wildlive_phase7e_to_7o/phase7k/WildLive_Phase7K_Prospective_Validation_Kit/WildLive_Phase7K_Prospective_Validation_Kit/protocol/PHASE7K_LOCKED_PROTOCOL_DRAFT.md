# CP 1.0 — Phase 7K Locked Candidate-Evaluation Protocol (DRAFT)

**Prepared:** 22 November 2026. **Status:** prepared locally, NOT externally timestamped/preregistered, NOT proof of independent confirmation.  No new 7K model scores were inspected when preparing this document.

## Scientific question
Can the original development-fitted, observation-only ridge selection gate improve elephant image-plane trajectory forecasts relative to the fixed OLS constant-velocity baseline on previously unexamined elephant recording groups, at 20- and 50-frame horizons separately?

## Design lock (do not revise after inspecting new scores)

- Input: exactly 20 consecutive observed annotated elephant bbox centres with unique within-frame `group_id`; forecast targets 20 or 50 subsequent consecutive annotated frames. Use native frame indices and normalized 3840×2160 image-plane coordinates. Never equate one frame to 0.1 seconds or results to real-world metres.
- Candidate experts: stationary / CV-OLS / original three-seed frozen Transformer. The adaptive ridge gate selects one expert *using six observation-only features and their missing-value indicators*, with frozen gate coefficients and normalization. CV20 is an additional descriptive baseline but not a gate expert.
- Model binary lock: six original 2s_to_2s/2s_to_5s seeds (42,52,62) in `Main Result/transformer_results`; SHA-256 and sizes in `prospective_lock.json` and `checkpoint_lock.json`. R9 isotropic-ablation and other similarly named checkpoints are not interchangeable.
- Gate binary lock: use exact released gate files in `locked_models`; never overwrite them with cross-environment re-fits even if differences are numerically tiny. Both gates were fitted on previously inspected `DJI_0207` and `DJI_0204_video2`; these are development, not new confirmation.
- Primary estimand: for each independently verified capture event and each horizon, mean per-window **adaptive ADE minus CV-OLS ADE**, then summarize capture-event differences with equal weight per *independent event* if event identity can be established. Negative means adaptive is better.
- Key secondary outcomes: ADE/FDE of Stationary, CV20, CV-OLS, frozen Transformer; per-capture-event and per-track changes; tail/large-error cases; model selection frequencies; scale-floor rates; predefined annotation-QA sensitivity. Explicitly report failures.
- Repeated windows share future labels and are correlated. Report window totals descriptively, never treat 1 window as an independent flight/herd/event. Do not compute a p-value or group confidence interval unless a sufficient number of verified independent events and an appropriate prespecified statistical analysis exist.
- Eligible tracks: any elephant `group_id` with a contiguous run of at least history+horizon annotations in a single video. Reject invalid/missing bboxes by prespecified parser checks, not test outcome. Visual species and track-identity QA must be performed before confirmatory status and documented. If frames are subsampled, do not assert physical time without extraction metadata.
- No tuning/selection of gate on test outcomes, no post-hoc primary-baseline switch, no checkpoint changes, no exclusion based on relative model errors. If any candidate's results influence a design change, that candidate becomes development evidence and fresh, untouched data are required for confirmation.
- Capture/flight/herd independence: metadata or reliable source mapping plus content checks against WildBox and all WildLive sources are required. `DJI_0204_video1` shares the `DJI_0204` stem with examined development video2; do not mark independent until resolved. Two folder names do not prove two flights. Even `DJI_0161_video1` is still independence-unverified.
- Ethical and legal: WildLive is licensed under the **Non-Commercial Government Licence for public sector information** per University of Bristol portal; obey terms, cite paper/dataset, do not publicly republish imagery, annotation ZIPs, nonlicensed checkpoints or precise sensitive wildlife locations. Seek supervisor review for research governance; no need to contact authors for the current frame-based exploratory workflow.

## Chronology & status gates

1. **Before performance access:** Record candidate source, species-preview-only evidence, ZIP SHA, sampling frame list, and whether event overlaps development and WildBox. Record acceptance and exclusion reasons independent of model outputs.
2. **Intake:** Run full annotation continuity/species inventory; flag invalid or short tracks. Sample images for confirmation of elephant labels and bounding-box/identity alignment; do not rely on JSON alone for final species correctness.
3. **Frozen evaluation:** Run Phase7K `evaluate` only with full ZIP, exact original checkpoints and exact development gate JSONs. The program computes gate choices from observed history *before* error scoring and refuses missing/duplicate joins.
4. **Human provenance review:** Record same-flight/same-herd uncertainty. Until this review is passed, outputs have `NEW_SEQUENCE_EXPLORATORY_CAPTURE_EVENT_INDEPENDENCE_UNVERIFIED` status regardless of numeric improvement.
5. **Group analysis:** If independent captures can actually be verified, present event-level paired ADE differences separately by horizon. A formal inference plan and number of events require further supervisor/power-method review. No arbitrary pseudo-independent window-level confidence intervals.

## Interpretation of a result

Even a strong decrease in normalized frame-index ADE cannot establish metre-based performance, improved conformal coverage, earlier elephant-human conflict warnings, valid geofence behavior or deployed safety. Those require distinct evidence and audited physical geometry.

## References

WildLive, University of Bristol dataset DOI 10.5523/bris.ewnuoroebuae20vei0cu6zu69, source index and README.
Historical CP10 Phase7H frozen evaluation and Phase7I/7J development evidence.  Full source filenames and checksum locks are in `prospective_lock.json`.
