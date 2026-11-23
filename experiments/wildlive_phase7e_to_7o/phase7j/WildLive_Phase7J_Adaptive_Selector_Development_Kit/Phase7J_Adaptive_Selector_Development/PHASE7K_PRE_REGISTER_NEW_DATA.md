# Phase 7K — prospective protocol starter (DRAFT, not preregistered)

**Research question:** Can a completely frozen observation-only ridge gate trained on two retrospective elephant recording folders improve normalized image-plane forecast ADE over (a) stationary, (b) OLS constant velocity, and (c) the frozen Transformer ensemble for 20-frame and 50-frame future horizons in *new* eligible aerial elephant recording groups?

**Crucial restriction:** As currently configured, the gate has *not* beaten the best fixed expert on retrospective transfer; a positive effect must not be presumed. Do not claim a pre-registration exists until it has been dated/locked and agreed before future recordings or labels are inspected.

Primary outcome (proposal): capture-group-level paired ADE difference, adaptive minus strongest prespecified comparator, separately at the two frame horizons. Since selecting the lowest of three models *based on test outcomes* is itself a retrospective criterion, a fully prospective design must declare the primary comparator before inspecting future outcomes (e.g., CV-OLS as strong fixed development baseline), and report other experts descriptively. Include FDE, bad-case tail errors and model selection rate as secondary. Explain whether any exclusion criterion is prospective, and apply identically to all experts.

Unit of generalization: *capture event*, not frame, window, ID or filename. Never count two clips from same uninterrupted flight/herd encounter as two independent units without provenance.

No use of new confirmation data for gate fitting, feature threshold adjustment, checkpoint replacement, preprocessing choice, inclusion changes, or repeated model selection. When this is broken, relabel that group as development data and acquire fresh confirmation data.

Require metadata on source, species, flight/time/camera where available, rights of use and source overlap. If FPS not verified, restrict to frame-index horizons and normalized image-plane metrics. Never interpret moving-camera image-plane displacement as ground speed without camera pose/registration evidence.

Lock hashes of: two Phase 7J gate JSON files, the Phase 7J inference code, six original frozen Transformer checkpoints, and original forecasting canonicalization. Maintain a chain-of-custody table per source recording.

Only calculate intervals or infer a population effect if sufficient **verified independent capture groups** are available; determine sample size/power with domain supervision before choosing a target count. Report all eligible groups including negative results.

No evidence of improved warning-system recall, conformal coverage or geofence safety follows from lower ADE alone; evaluate warnings separately on valid evidence units and audited boundaries.
