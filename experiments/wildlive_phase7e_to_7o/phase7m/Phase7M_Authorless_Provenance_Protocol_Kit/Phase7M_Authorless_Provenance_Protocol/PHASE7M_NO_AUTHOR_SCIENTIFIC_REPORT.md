# CP 1.0 — Phase 7M-A: Author-independent scene grouping and conservative validation

**Date:** 22 November 2026  
**Decision:** Author correspondence is **optional** for a transparent **sequence-level exploratory evaluation**, but is NOT replaceable by an image algorithm for verifying distinct original UAV flights.

## Why this alternative is legitimate, and what it does not prove

The WildLive official README describes **22 UAV-acquired videos**, notes **no prescribed dataset splits**, and does not provide an authoritative per-video capture-flight ID in the consulted metadata. These conditions do not force research to stop; they force scope-limited claims. We can (1) compare image content, (2) group clips sharing scene evidence conservatively, (3) report results per sequence and per provisional scene component, (4) conduct leakage-aware *descriptive* robustness analysis, and (5) source additional annotation-only recordings if necessary. We cannot infer independent recording sessions or exact time horizons from filenames or nonmatching sample images.

## New empirical all-pairs RGB audit (performed in this package)

- Input: **25 native source JPEGs** already supplied in four ZIP archives (`DJI_0207`: 9; `DJI_0204_video2`: 8; `DJI_0204_video1`: 3; `DJI_0161_video1`: 5).
- 223 cross-recording sample-image pairs, covering six recording-folder pairs. In every pair: ORB descriptors (3,200 maximum), 1024×576 grayscale, Lowe ratio <0.75, RANSAC homography, minimum 80 inliers / 55% inlier fraction / 1.5% source-image convex-hull coverage to mark *strong shared-scene evidence*.
- **24/24** sampled `DJI_0204_video1` × `DJI_0204_video2` image pairs show a strong common scene. Maximum **1,084 RANSAC inliers** in this parameterization; sampled best pair: frame 200 vs frame 200. This is **shared physical-background evidence**, not proof of the same flight or contiguous footage.
- In the other **199 cross-recording image pairs**, no strong shared-scene match was found. **All 199 are INCONCLUSIVE for flight independence**, not negatives establishing separation.
- **0 confirmed independent capture-event IDs** available from current source files. Existing annotations have frame indices, not verified time information.
- No JPEG pixels, endangered wildlife GPS data, or full annotations are bundled in this release.

Consult `results/recording_pair_summary.csv`, `results/all_cross_recording_image_pairs.csv`, `results/provisional_no_author_group_register.csv`, and `results/summary.json`. These are reproducible calculations, not ground truth flight labels.

## Conservative grouping decision

| Provisional group | Recording(s) | Dataset role | Confirmatory independence? |
|---|---|---|---|
| SCENE_0204 | `DJI_0204_video1`, `DJI_0204_video2` | Development-related/previously inspected; group as one **scene component** | **NO** |
| UNRESOLVED_0207 | `DJI_0207` | Development | **NO** |
| UNRESOLVED_0161 | `DJI_0161_video1` | Previously scored before this audit | **NO** |

These are **three provisional scene-evidence components**, not three verified independent drone sorties. The unlinked folders might still be related through capture event, day, operator, drone mission or ecological setting. Conversely, the two 0204 clips could have come from separate flights at the same location. Hence treat them as related for a *conservative grouping rule*, without asserting exact original provenance.

## Past forecasting results: retain their limitations

- DJI_0161_video1 adaptive ADE versus CV-OLS: **13.49% lower at 20 future frames**, **23.86% lower at 50**. The single track and overlapping windows do not yield group-level statistical significance; neither horizon beats the best individual expert.
- DJI_0204_video1 adaptive ADE versus CV-OLS: **0.61% higher at 20 frames**, **~0.053% higher at 50**. Related scene to a development recording, hence not independent external validation.
- DJI_0207 and DJI_0204_video2 influenced model development. They cannot become untouched test data retrospectively.
- No additional Transformer checkpoint training or gate tuning performed in this alternative audit. Original Phase 7K lock is included for reference and must not be changed.

## Publication-safe evaluation route without contacting authors

1. Replace claims of **flight-independent external validation** with **multiple-video/sequence-level exploratory cross-recording stress tests** and *known scene-overlap control*.
2. Make full-annotation species census of not-yet-fully-examined WildLive clips before treating zebra/giraffe sampled recordings as definitively elephant-free. Use optional `phase7m_download_json_census.sh` (downloads `boxid` JSON only) and inspect HTTP failures/holes before deeming absence meaningful. This script is prepared but **not executed** because it needs the user's WSL network.
3. Predefine additional new candidate sequences and their evaluation protocol **before** inspecting their forecast losses. Preserve Phase 7K frozen gates and checkpoints; do not optimize the selector using DJI_0161_video1 or DJI_0204_video1 outcomes.
4. Compare expert methods at the **recording or conservatively linked scene-component level**, not a window-level t-test / binomial test. Report all successful/failed horizons transparently, including negative results and wide uncertainty. With three scene components and no certified flight IDs, refrain from generalization p-values.
5. Image-camera-motion compensation can be studied as a separate analysis, but sparse RGB samples do not support per-frame motion-corrected trajectories; frames/second are **unknown** absent reliable original cadence metadata.
6. If an external dataset with truly documented independent capture sessions and elephant tracks becomes available, use it for stronger confirmation. Otherwise explicitly label the study exploratory and make the limitation central.

## Recommended claim language

> We performed frozen-model cross-sequence exploratory evaluations on aerial elephant bounding-box tracks. A content-based audit found substantial background overlap between `DJI_0204_video1` and `DJI_0204_video2`, which were therefore grouped conservatively to reduce evident scene-level leakage. Original flight/session IDs and capture cadence were unavailable, and independence of the remaining sequences could not be verified. The findings describe sequence-level model robustness, rather than proving independence across drone missions or superior out-of-domain generalization.

## Input reproducibility

`src/authorless_provenance.py audit --inputs-dir <original JPEG ZIP folder> --output <results folder>` regenerates the audit. `python src/test_authorless.py` validates structural logic and matching safety semantics. `census --zip-dir ... --output ...` counts labels from locally supplied full annotation ZIPs. Outputs include ZIP SHA256 inventory for traceability, and no source imagery. Threshold is a **heuristic for positive shared-scene evidence**, not a calibrated independence test; audit image matches visually if extending the dataset.

## Sources

- University of Bristol WildLive official dataset: https://data.bris.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69
- Official README: https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/Readme.txt
- WildLive project: https://dat-nguyenvn.github.io/WildLive/

**Licence:** WildLive is distributed under a noncommercial public-sector-information licence. Keep original images/annotations private unless permitted by the applicable terms. Do not publish wildlife GPS details.
