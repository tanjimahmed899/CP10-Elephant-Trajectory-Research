# Q1 Research Continuation — Status and claim audit (16 November 2026)

This dossier uses the provided Results PDFs, original code ZIP, additional trajectory files, previous Experiment 1/2 reports and six verified checkpoint files. It does NOT claim a new external evaluation, Raspberry Pi validation, or peer-reviewed novelty.

## Research workstreams
- **Historical detection experiment:** Reported (YOLOv8n mAP@50 94.0%; independent frozen metric audit still advised)
- **R8 matched CV forecasting:** Completed (2s ADE 0.053522 -> 0.020148; 5s 0.123675 -> 0.040174; same corrected target, translation-aligned raw)
- **R8 full geometry mapping:** BLOCKED: no frame-specific raw-to-corrected transform matrices or original images in uploaded artifacts
- **Step7H retrospective ablation:** Completed and source-code rechecked against 20,286 cases (prior report); N=2 selection was post-hoc on TestReal
- **Independent 6-video WildBox experiment:** BLOCKED: six new video ZIP files missing; cannot generate real predictions
- **Six seed Transformer checkpoints:** Uploaded and successfully forward-pass tested; independent historical SHA match pending
- **Raspberry Pi real-time demo:** NOT VALIDATED; not necessary if paper claims only offline forecasting analysis
- **Q1 novelty validation:** Preliminary literature overlap matrix completed; targeted systematic review and peer review needed
- **Journal manuscript:** Evidence-bounded first working draft, not submission-ready

## Mandatory open items before an empirical Q1 submission

1. **Independent holdout:** acquire the six *locked* WildBox elephant ZIP archives and preserve all failures and source identities. Run frozen ensemble and Step7H on newly constructed tracks without changing thresholds or selecting videos by difficulty/outcomes. Ensure video-level independence; do not assert independent ecological sites.
2. **R8 geometry:** retrieve actual frame-by-frame transform matrices and camera geometry or underlying original video frames. Current last-observation translation alignment is not an exact geometric transform and cannot support unique causal attribution to ego-motion.
3. **Data/protocol leakage:** Step7G explicitly states N=2 was selected post-hoc on BIRDSAI TestReal. Treat those as development diagnostics; new unseen holdout is needed for confirmation.
4. **Model provenance:** current six hashes prove file integrity after upload only. Confirm against pre-evaluation frozen checkpoint manifest.
5. **Warning metrics:** separate any RED+YELLOW watch alerts from actionable RED urgent alerts. Report event-based false alarms/hour only if actual fps and event aggregation are validated; boundary-instance FPR is not alarms/hour.
6. **Model claim:** Transformer ensemble is not the best model for every horizon and split; compare against LSTM/GRU and stationary/CV fairly.
7. **External validity:** WildBox normalized image coordinates are not metres; virtual boundaries are not surveyed village/farm fences; BAMBI is red deer/human stress testing, not elephant validation.
8. **No deployment inference:** R13–R16 indicate severe detector-to-tracker coverage and failure to meet 10 Hz CPU. Do not state real-time or field effectiveness.

## Priority order

A. Download six video ZIPs with frozen repo SHA and hashes (data gate).
B. Run source-disjoint frozen forecaster / coverage / Step7H evaluation, retaining zero-window sources (analysis gate).
C. Retrieve and validate per-frame ego-motion transforms (geometry gate).
D. Finish reviewer-level novelty matrix, manuscript results and limitation statements (writing gate).
E. Add Pi latency only if the claim will include hardware/real-time field feasibility (conditional gate).

## Explicitly prohibited extrapolations

- 'We show a real-world reduction of human–elephant conflict' — untested.
- 'Q90 guarantees 90% coverage on every external domain' — BAMBI 5-second severe undercoverage.
- 'Camera motion is the dominant error source in all models' — not demonstrated.
- 'The new six WildBox videos represent entirely new ecological sites' — shared elep1/elep2/elep3 campaign groups.
- 'Validated Raspberry Pi end-to-end pipeline' — not tested.
- 'Q1 acceptance assured' — unknown and subject to external peer review.
