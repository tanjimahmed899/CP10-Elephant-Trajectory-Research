# How to adjudicate flight lineage after an author reply

The initial Phase 7M verdict is **NOT ESTABLISHED**, not independent. Future new information must be documented in a new dated version of the group register. Never edit or replace the Phase 7K frozen protocol/selector.

## Evidence requirements per sequence

- [ ] Author/official provenance source, accessible evidence and dated response archived.
- [ ] Originating source video ID and segment offset confirmed (or a written negative statement that no parent segmentation exists).
- [ ] Source flight/session ID confirmed without wildlife-sensitive coordinates.
- [ ] Capture time or approximate time interval sufficient to distinguish or link capture events; verify time zone if supplied.
- [ ] Confirmation of any multiple recordings from the same flight/herd/physical scene, and of any temporal overlap.
- [ ] Actual image capture FPS and sampled annotation cadence confirmed separately. Onboard tracking throughput is NOT camera capture FPS.
- [ ] Training/development group membership checked for overlap before treating a clip as held out.
- [ ] Any candidate newly included must pass full elephant-label inventory and sampled visual QA before scoring.
- [ ] Author/recording lineage uncertainties are recorded explicitly as unknown, not assumed false.

## Decision rules

1. **Development capture or source overlap**: retain as DEVELOPMENT or RELATED_SCENE_EXPLORATORY. Never relabel as an independent test just because the clip filename differs.
2. **Flight ID unknown or contradictory**: remain UNVERIFIED. Absence of a visual feature match is not evidence of independence.
3. **Verified distinct event but already inspected/tuned by research team**: may support retrospective study, not untouched preregistered confirmation.
4. **New event verified distinct from development and lock unchanged, result not inspected beforehand**: candidate for documented locked external validation; group-level analysis still needed.
5. **Clip shares same flight but occurs at a separate time interval**: not an automatically independent flight; define conservative shared group until study protocol predefines a defensible session split.

Do not make deployment, geofence-safety, seconds/physical-metre, or statistical-significance claims from frame-based ground-truth centers alone.
