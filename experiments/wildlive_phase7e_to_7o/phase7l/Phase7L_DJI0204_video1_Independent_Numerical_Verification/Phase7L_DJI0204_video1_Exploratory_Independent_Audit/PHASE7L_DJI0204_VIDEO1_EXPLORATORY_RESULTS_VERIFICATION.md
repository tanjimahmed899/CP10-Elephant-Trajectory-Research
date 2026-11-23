# Phase 7L: DJI_0204_video1 related-scene exploratory locked evaluation

**Status (22 November 2026): numerical-output verification PASS; adaptive superiority NOT demonstrated; independent capture-event validation NOT established.**

## Evidence examined

- User-supplied `Phase7L_DJI0204_video1_Exploratory_Results.zip`: window-level expert and adaptive errors, gate decisions, summary, scale-floor rates, tracking availability and evaluation manifest.
- User-supplied `DJI_0204_video1_boxid.zip`: raw original track annotations, used to regenerate the frozen observation-only gate decisions.
- Released `WildLive_Phase7K_Prospective_Validation_Kit`: frozen gate JSON, evaluation code, checkpoint SHA-256 release manifest.
- Earlier `DJI_0204_video2` Phase 7H evaluation and `DJI_0161_video1` Phase 7K locked evaluation for descriptive comparison only.
- Previous Phase 7L image-based provenance audit: substantial static-background overlap between DJI_0204_video1 and DJI_0204_video2; biological/flight independence unverified.

## 1. Dataset/task audit

The evaluated recording has 283 consecutive annotation frames (0–282), six elephant tracks with 283 frames each and 1,698 elephant boxes. It yields 1,464 overlapping 20-observed-to-20-future windows and 1,284 overlapping 20-observed-to-50-future windows, or 2,748 total. Metrics are normalized 3840 × 2160 image-plane box-centre distances; frame rate, ground-plane coordinates, metres/sec and independent flight membership are not established. Overlapping windows and tracks are not independent capture events.

## 2. Verified pooled ADE / FDE

| Forecast model | 20-frame ADE | 20-frame FDE | 50-frame ADE | 50-frame FDE |
|---|---:|---:|---:|---:|
| Stationary | 0.002217133 | 0.003872702 | **0.004620653** | **0.008082589** |
| CV20 | 0.001860731 | 0.003475039 | 0.005221120 | 0.011274705 |
| CV-OLS (predeclared primary comparator) | **0.001817579** | 0.003410802 | 0.005172604 | 0.011248575 |
| Frozen Transformer ensemble | 0.001930449 | **0.002986256** | 0.006708351 | 0.011908046 |
| Adaptive selector | 0.001828601 | 0.003416710 | 0.005175326 | 0.011223306 |

**Predeclared primary ADE difference, adaptive minus CV-OLS:**

- 20-frame: +0.000011022; adaptive ADE **0.6064% higher** (worse).
- 50-frame: +0.000002722; adaptive ADE **0.05263% higher** (worse).

Thus there is no ADE improvement against the predeclared comparator at either horizon. At 50 frames, Stationary is the best pooled ADE model and is about 10.72% lower in ADE than Adaptive. FDE ranking is not identical to ADE: for 20 frames, Transformer has smallest FDE; for 50 frames, Stationary has smallest FDE. At 50 frames Adaptive FDE is marginally lower than CV-OLS despite slightly higher ADE. This does not override the predeclared ADE criterion.

## 3. Frozen gate-selection behaviour

| Horizon | Stationary | CV-OLS | Transformer |
|---|---:|---:|---:|
| 20 frames (n=1464) | 141 (9.63%) | **1,307 (89.28%)** | 16 (1.09%) |
| 50 frames (n=1284) | 3 (0.23%) | **1,273 (99.14%)** | 8 (0.62%) |

There is no evidence that a broadly useful dynamic selector has been established by this recording. Most windows simply reuse the CV-OLS prediction. Per-elephant 20-frame adaptive-minus-CV-OLS ADE: 2 tracks improve, 2 deteriorate, 2 equal (within reported numerical precision). At 50 frames: 1 improves, 1 deteriorates, 4 equal.

At 20 frames, 89.28% of windows have exactly identical adaptive and CV-OLS errors (because CV-OLS was selected). At 50 frames, 99.14% do. There were 83/1464 20-frame windows with lower adaptive ADE, and 74/1464 with higher adaptive ADE; for 50 frames, 3/1284 lower and 8/1284 higher.

## 4. Offset sensitivity: descriptive, NOT independent tests

To avoid treating strongly overlapping windows as independent, windows were partitioned by start modulo window length (40 or 70 frames). Each offset still represents the same single recording and six correlated elephant tracks.

- Horizon 20 frames: adaptive mean ADE beats CV-OLS in **17 of 40** start-offset configurations. The remaining 23 offsets have a higher mean adaptive error. Delta range: -9.48828e-05 to +1.02281e-04.
- Horizon 50 frames: adaptive mean ADE beats CV-OLS in **3 of 70** offsets. The other 67 offsets are worse. Delta range: -1.22190e-04 to +2.02845e-04.

There are no valid population p-values or capture-group confidence intervals based on these counts.

## 5. Canonicalization / scale floor

The inference manifest reports the canonicalization scale-floor condition on **100%** of all 2,748 history windows. This is a measured regime property, not proof that the floor *caused* degradation. The result should motivate camera-motion and feature-scale sensitivity analysis using properly split development material, not post-hoc retuning on this candidate.

## 6. Programmatic audit results and limits

**PASS**: ZIP CRC; raw annotation ZIP SHA-256 and byte size; both released gate JSON SHA-256 digests; all six checkpoint digest/byte-size values consistent with the locked checkpoint manifest; unique window keys and exact contiguous start ranges across six IDs; per-window nonnegative finite errors; selected-expert ADE and FDE exactly correspond to stored adaptive errors; all 14 summary rows recomputed from per-window files; horizon-specific counts; 2,748 of 2,748 locked gate choices independently regenerated from original annotation histories (no future data for gate reconstruction).

**Scope restriction:** Six model checkpoint **binaries** and predicted trajectory coordinates were not included in the uploaded ZIP. Checkpoint hashes are verified against the file's manifest, **not against the binaries in this container**; frozen Transformer forward inference was not independently rerun. The manifest may accurately reflect WSL's verified binaries but this audit does not prove it independently. Likewise, correct derived metrics do not prove that original neural forecasts were computed correctly beyond the consistency checks above.

## 7. Relationship to earlier recordings

- DJI_0161_video1 (new but still provenance-unverified recording): Adaptive reduced ADE vs CV-OLS by **13.4901%** (20) and **23.8636%** (50). That recording did not show adaptive superiority over the best model.
- DJI_0204_video1 (this recording): Adaptive increased ADE vs CV-OLS by **0.6064%** (20) and **0.05263%** (50).
- DJI_0204_video2 (Phase 7J development recording): Frozen Transformer ADE substantially exceeded CV-OLS in both horizons. DJI_0204_video1 is in the same landscape/filename family, but has different model rankings and magnitude of errors. This is related-scene heterogeneity, **not another independent validation replicate**.

The two recordings are descriptive comparisons. Do not pool DJI_0204_video1 with DJI_0204_video2 as independent validation units, and do not claim robust transfer, method superiority, a statistically established percentage improvement, or validated seconds-based prediction.

## 8. Research decision and next tasks

1. Preserve Phase 7K release gate SHA-256 files and original Transformer checkpoint versions; no new threshold choice or adaptive fit based on candidate results.
2. Label DJI_0204_video1 **DJI_0204 related-scene exploratory** in all tables. Keep DJI_0204_video2 under **development**.
3. Request author/curator source video lineage, original flight ID, capture timestamp and segment mapping for DJI_0204_video1, DJI_0204_video2, DJI_0161_video1, DJI_0207. Shared scenery is insufficient to decide whether flights/animals/time intervals are the same.
4. Obtain further truly separate elephant flight recordings or a defensible independent new-flight capture. Score only with locked methods and predeclared horizon/comparator.
5. If independent data cannot be obtained, report the current study as method-development plus descriptive cross-recording robustness, with this negative result and the original methodological limitations. A Q1/Q2 acceptance or new method superiority claim is not scientifically guaranteed.

**Final judgment: computational outputs reproducible; primary adaptive ADE advantage absent; capture-group independence unverified; confirmatory conclusion not supported.**
