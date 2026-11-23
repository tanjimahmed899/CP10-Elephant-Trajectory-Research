# CP 1.0 — Phase 7O: targeted image-registration model robustness, annotation-QA exposure, and manuscript integration

**Date:** 23 November 2026  
**Status:** Empirical retrospective analysis completed on existing source archives; independent-flight validation unresolved; no new forecasting model inference.

## Purpose and relation to the main manuscript

The currently provided main manuscript (`Final Demo paper.docx`) primarily describes BIRDSAI, WildBox, a same-target camera-correction Transformer ablation, conformal calibration and synthetic geofence warning studies. The WildLive 7E–7N series is a **different, separately labelled dataset/experiment**. This Phase 7O supplement can be integrated without pooling WildLive 20/50-**frame** metrics with the main paper's 2/5-**second** metrics, or interpreting both studies' camera compensation as the same tested method.

The purpose of Phase 7O is to test whether Phase 7N's sparse background-registration observations are sensitive to the choice of global 2D transformation and whether large post-registration centre residuals could materially affect descriptive model-score summaries. This is not a newly optimized forecasting architecture.

## Input and independence boundaries

- Reviewed original Phase 7N source archives and 7N numerical tables. No video/annotation/model binaries are redistributed in this kit.
- Two selected recordings have usable consecutive RGB image pairs: six pairs in `DJI_0207`, two pairs in `DJI_0204_video2`; eight in total. They were **selected for targeted review**, not sampled randomly from the population of frames.
- All images use 3840 × 2160 native pixels, resized to 1536 × 864 for keypoint matching. Output errors and image registration residuals have distinct units.
- Four-recording archived model comparison from 7N contains 12,948 **overlapping** windows at 20/50 future frames; no window is counted as an independent capture event. The final adaptive gate is available for only two recordings. `DJI_0204_video1` and `DJI_0204_video2` have proven scene overlap. Flight independence has not been verified.
- This experiment is exploratory and retrospective. Source pairs and the 30-native-pixel QA threshold were chosen in light of earlier audits; it is **not preregistered**.

## A. Held-out static-background image alignment sensitivity

Animal bounding boxes are masked using 1.4× their dimensions in each frame. ORB descriptors (up to 6,000 per image; Hamming ratio test < 0.75) are matched on the masked background at 1536 × 864 resolution. For each pair, approximately 75% of matches in each 4 × 4 spatial cell are reserved for estimating transforms; the remaining ~25% are used only to evaluate the geometric residual of those transforms. The same matched points and held-out split are used for all methods.

Three transformations are estimated from the training matches: (1) robust median translation; (2) partial affine similarity, estimated by RANSAC; and (3) projective homography, estimated by RANSAC. A fixed random seed makes the split reproducible; OpenCV numeric variation across versions/hardware remains possible. We report per-pair **median held-out point reprojection error** in resized-image pixels. These matched feature points are still spatially and temporally correlated and should not be interpreted as independent trials. Furthermore, the algorithm only assumes matched *background-like* features are static: vegetation/parallax can violate that assumption.

| Method | Median across eight per-pair held-out medians (1536-wide px) | Maximum per-pair median |
|---|---:|---:|
| Translation | 1.262585 | 1.465417 |
| Similarity | 0.950101 | 1.489395 |
| **Homography** | **0.752017** | **0.933211** |

Homography has the smallest held-out median on **all eight selected image pairs**; the result compares *background feature alignment*, not camera pose error, elephant world motion, or target forecasting ADE. In the original 3840-pixel image, one resized pixel spans 2.5 image pixels horizontally and vertically; these tabulated model-fit values are **not native-pixel** errors.

## B. Targeted high-residual track transitions

For each retained match-based transform, we apply its 3×3 image-plane mapping to the animal-box centre in the earlier frame, then measure distance to the annotation centre in the next frame. This is a **post-registration centre residual**. A high residual alone cannot discriminate true movement, box-jitter/occlusion, track identity problems, and spatial parallax at the elephant's ground depth. We flag transitions with at least **30 native-pixel homography residual** for manual QA, not for automatic data deletion.

| Recording | ID | Frames | High post-registration residual (native px, approximate) |
|---|---:|---|---:|
| DJI_0207 | 2 | 389→390 | 59.04 |
| DJI_0207 | 2 | 400→401 | 65.89 |
| DJI_0207 | 2 | 401→402 | 67.86 |
| DJI_0207 | 0 | 859→860 | 39.33 |
| DJI_0207 | 0 | 860→861 | 34.19 |

These five events correspond to previously identified high-residual regions. There were no >30-pixel residual events within the eight-pair sampled subset of `DJI_0204_video2`. This does **not** show that that recording is anomaly-free across all frames.

## C. Descriptive forecast-window exposure sensitivity

For the two recordings where original archived per-window expert errors are available, we flagged a window as *exposed* if one or more high-residual target transitions for the **same track** lies within its 20-frame input plus its 20-/50-frame prediction target interval. The original ADE/FDE values were **not recomputed from new predictions**; we only stratified saved errors. Target-event overlap must be viewed as retrospective diagnostic exposure, not causal intervention. The full, unfiltered model scores remain primary.

| Recording | Forecast horizon (frames) | Exposed windows | Other windows | Best model among exposed | Best model among other |
|---|---:|---:|---:|---|---|
| DJI_0207 | 20 | 59 | 2,428 | Stationary | Transformer |
| DJI_0207 | 50 | 89 | 2,308 | Stationary | Stationary |
| DJI_0204_video2 | 20 | 0 | 1,272 | N/A | CV-OLS |
| DJI_0204_video2 | 50 | 0 | 1,152 | N/A | CV-OLS |

For the DJI_0207 short-horizon exposed subset, Stationary ADE is 0.010026 and Transformer ADE 0.010205. The unexposed subset favours Transformer (0.004384 vs Stationary 0.004700). These event-defined groups overlap heavily; there are just two relevant tracks and one source recording. **No statistical test, independence inference, causal label-noise effect, or new method advantage is justified.**

## D. What changes and what does not

**Supported additional evidence:** held-out feature registration shows that the global geometric transform assumption matters in eight selected pairs; targeted post-registration high-residual events intersect some forecasting windows, and reported subgroup best-model ranking can differ from the pooled ranking. These are useful quality-assurance and failure-interpretation diagnostics.

**Not established:** homography superiority for every WildLive frame; actual drone egomotion recovery; physical elephant velocity; annotation-error causality; correction of complete 20/50-frame histories and future trajectories; change in neural forecasting ADE/FDE; superior locked adaptive selector; independent-flight generalization; any field-tested warning performance.

A translation/rotation applied identically to both prediction and ground-truth coordinates need not change Euclidean ADE at all. A genuine stabilized-input forecasting claim would require full consecutive imagery and a specified inference-space transformation, as well as rerunning the model and evaluating predicted coordinates. This study does not provide such a result.

## E. Manuscript integration safety

`manuscript/Final_Demo_paper_Phase7O_integrated.docx` retains the supplied main manuscript's original analyses and adds a **separately identified WildLive supplementary analysis** under Methods, Results, Discussion and Limitations, plus sources [18] and [19]. It is a supervisor-review draft, not submission-ready: full manuscript references, claims and tables outside the added WildLive sections were not independently re-audited here. A recheck of overall word limit, journal template, references and figure licensing is required before submission.

### Suggested concluding paragraph

> A supplementary WildLive analysis across four elephant-labelled annotation recordings revealed substantial between-sequence and between-horizon variability in the rankings of frozen Stationary, constant-velocity and Transformer experts. For eight targeted consecutive RGB-frame pairs, projective homography achieved lower held-out background-feature reprojection error than translation-only or similarity transforms. Five high post-registration animal-centre residual transitions were flagged for annotation review, but their origin remains unresolved. These supplementary experiments neither establish improvements in full-trajectory camera-corrected forecasting nor demonstrate independent-flight generalization or adaptive-selector superiority.

## Reproduction and privacy

Run the included `src/test_phase7o.py` and then `src/phase7o.py --input-dir PATH_TO_SOURCE_ZIPS --output PATH_TO_RESULTS`. The CLI verifies source ZIP SHA-256 against `protocol/INPUT_SHA256_LOCK.json`, decodes only the preselected sample frames, recomputes image-registration and QA-exposure tables, and saves descriptive charts/manifest. It does not load PyTorch or run new model inference. Preserve the original source ZIPs privately and review the dataset's Non-Commercial Government Licence before redistribution. Do not expose precise wildlife geolocations.

### References added to the supervisor draft

[18] Nguyen Ngoc, D., Richardson, T., Watson, M., et al. (2026). *WildLive*. University of Bristol. DOI: 10.5523/bris.ewnuoroebuae20vei0cu6zu69. https://data.bristol.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69

[19] Ljubičić, R., Strelnikova, D., Perks, M. T., et al. (2021). A comparison of tools and techniques for stabilising unmanned aerial system (UAS) imagery for surface flow observations. *Hydrology and Earth System Sciences*, 25, 5105–5132. DOI: 10.5194/hess-25-5105-2021.
