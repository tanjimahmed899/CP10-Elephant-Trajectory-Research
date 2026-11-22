# EXP4 Phase 4 — Manuscript Revision & Scientific Change Log

**Date:** 22 November 2026  
**Source file:** `Final Demo paper.docx` (20-page retrospective draft)  
**Corrected manuscript:** `Elephant_Trajectory_Phase4_Scientifically_Corrected.docx` (24 rendered pages)  
**Status:** Supervisor-review scientifically corrected draft — **not** a claimed new Q1-ready method or field validated system.

## 1. Changes performed

1. **Preserved scientific history.** Retained all ten original figures, the original nine quantitative/comparative tables, the main Stage 4 and Stage 7 experiments, and the existing forecasts/limitations without manufacturing model outputs. One new verified table was added, and the old Appendix table was renumbered Table 11.
2. **Abstract, contributions and conclusion.** Condensed the Abstract to 278 words while retaining original headline experimental findings; inserted verified N=2 TestReal instance-level precision and limitations, distinguishing it from Stage 6 events and new-method confirmation.
3. **§4.5 and §6.2 calibration.** Added original finite-sample calibration ranks (57/62 at 2 s and 32/34 at 5 s), the concentration of 32/34 5-s tracks in BIRDSAI sequence 0367, and the distinction between correct quantile arithmetic and external-domain coverage.
4. **§4.6 virtual-boundary methods.** Identified the three distinct historical evaluation units: moving-anchor hypothetical cases; Stage 6 fixed-reference annotated crossing events and alert runs; Step 7B–7H synthetic-line instances. Clarified RED versus YELLOW labeling differences between persistence diagnostics and response-aware urgency.
5. **§5.4 Table 6.** Replaced ambiguous unassociated alert-run numerators with counts including their run denominators (e.g., 348/463 and 237/347). These are not operational alarms.
6. **§5.8 BAMBI.** Distinguished original **metre-space** 5-s q90 empirical coverage **1.1236% (7/623)** from the separately evaluated compatible **anchor-normalized** result **2.5682% (16/623)**. BAMBI is red-deer/human cross-species stress testing, not independent elephant validation.
7. **New §5.9 Table 10.** Added the authentic N=2 instance counts (TP/FP/FN/TN), sensitivity, false-alert rate and precision on prior TestReal and Fresh WildBox cohorts; included Step 7G retrospective N-selection conditions, Step 7H urgent-only scope, and index-gap counterfactual caveats.
8. **Statistical limitations.** Stated that response-threshold copies and correlated windows do not provide independent sample replication, while index adjacency does not prove elapsed-time persistence.
9. **Related Work.** Added two independently verified publications: Gibbs and Candès (NeurIPS 2021) and Huang, He, Miao (AAAI 2025). The revised paper does **not** claim implementation of their methods.
10. **Document QA.** Updated revision date and footer, preserved original images, ensured new Table 10 stays readable, and moved Appendix A onto a fresh page to avoid orphaned table fragments.

The exact list of **40 revisions** is available in `revisions.json`.

## 2. Key source references

- `Final Demo paper.docx`: authored manuscript and figures (original preserved).
- `Experimental Results (Part-4).pdf`: Stage 6 fixed-reference label and event/run summaries; Stage 7C–F component results.
- `crossfit_track_calibration_scores.csv`: Step 6C frozen track-level thresholds and source concentration.
- `Experiment Dataset (Part-2).pdf`: original and compatible BAMBI representation-specific coverage.
- `step7b_v2_all_geofence_instance_alarms.csv`: source Step 7B virtual-line instances (214,956 total scenario/threshold rows).
- `TABLE_step7f_persistence_tradeoff.csv`: Step 7F persistence comparison; 256/256 replay checks from the prior Phase 2 audit.
- `STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json`: provisional N=2 post-hoc TestReal selection, checked source SHA-256.
- `EXP4_Phase2_Report/EXP4_PHASE2_SCIENTIFIC_AUDIT.md`: independent source-exact replay and index-gap diagnostic.
- Primary verified literature: <https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html>; <https://doi.org/10.1609/aaai.v39i16.33915>.

## 3. Unresolved scientific issues — cannot be corrected by text edits

**Critical:** No method improvement has yet been evaluated on genuinely independent unseen elephant recording groups; the revised paper should not be presented as a validated novel conformal / warning method.

**Major:** Stage 6 fixed-reference event replay still relies on 3-D annotated truth for reference consistency. No GPS-surveyed physical virtual fence or camera-only operational validation exists.

**Major:** N=2 precision remains 8.99%/18.64% on TestReal 2 s/5 s synthetic instances despite recall gains. Event-level and operational false-alert frequency are not established by these instance statistics.

**Major:** The detector-to-tracker-to-forecaster strict subset remains one BIRDSAI sequence and requires additional GT-blind validation across new sources.

**Major:** Conformal q90 undercoverage at 5 s persists after distribution shift. All proposed regime-aware/adaptive corrections remain exploratory or untested on untouched groups.

**Publication readiness:** A complete bibliography verification and journal-specific numbered citation formatting pass remain outstanding; only the two new publications were explicitly verified for this revision. Dataset-license/shareability, per-source original artifact availability, and live hardware latency also require later validation.

## 4. Reproducibility and QA

- Original manuscript SHA-256: `85010bd97aafe2068869efc8a9c698a40addc21f5f97ba333ee26ae201553ad2`
- Corrected manuscript SHA-256: `2a59751324139261820aaff1ad5ebe71fae2bb60322f0bd8c893a8cebea37a06`
- Original figures retained: **10/10**
- Original table content preserved except for qualified denominator additions to Table 6 and Appendix Table 11.
- Final table count: **11** (10 original tables, 1 inserted new evidence table).
- Final rendered page count: **24**. Word pagination may vary by application/fonts; content source is the DOCX.
- New Table 10 percentages independently recomputed from counts within 0.01 percentage point.

## 5. Next experimental phase

EXP4 Phase 5 should specify one reproducible methodological question and compare baselines against a candidate correction on **development groups only**. Freeze model/policy choices, then recruit new, independent elephant recording groups for confirmatory coverage–radius–warning precision evaluation. The manuscript must not include future experimental performance until authentic outputs exist.
