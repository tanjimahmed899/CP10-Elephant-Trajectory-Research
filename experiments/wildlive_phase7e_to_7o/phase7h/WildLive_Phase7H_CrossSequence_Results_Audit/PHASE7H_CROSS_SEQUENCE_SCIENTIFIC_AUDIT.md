# Phase 7H: DJI_0204_video2 frozen evaluation verification and cross-sequence comparison

## What was examined

- User-uploaded `Phase7H_DJI0204_Frozen_Evaluation.zip`: ZIP integrity passed, five derived output files, including 2,424 per-window metric rows and 40 summary rows.
- Original uploaded `DJI_0204_video2_boxid.zip`: SHA-256 **e196356fb231aa520b63e709c9698b497bb0fa22edbbd94aacbe37b3a3ecd689**, exactly matching the Phase 7H evaluation manifest; file size **245,018 bytes**.
- Previously uploaded `Phase7F_Frozen_Results.zip` from `DJI_0207`: six checkpoint SHA-256 hashes **and byte sizes** match the six Phase 7H manifest entries (6/6). The actual checkpoint binaries were not supplied in either evaluation ZIP for independent byte-level verification.
- All 40 summary rows re-aggregated from per-window data and matched to numerical precision (<1e-12). No duplicate `(horizon,track,segment,start)` rows, missing metrics or nonfinite values found.

## Core pooled results (ADE / FDE; normalized image-plane units)

| Sequence | Future frames | Model | ADE | FDE | Windows |
|---|---:|---|---:|---:|---:|
| DJI_0204_video2 | 20 | cv20 | 0.001279 | 0.002229 | 1,272 |
| DJI_0204_video2 | 20 | cv_ols | 0.001246 | 0.002184 | 1,272 |
| DJI_0204_video2 | 20 | stationary | 0.004056 | 0.007601 | 1,272 |
| DJI_0204_video2 | 20 | transformer_ensemble | 0.002752 | 0.004447 | 1,272 |
| DJI_0204_video2 | 50 | cv20 | 0.002342 | 0.004818 | 1,152 |
| DJI_0204_video2 | 50 | cv_ols | 0.002271 | 0.004708 | 1,152 |
| DJI_0204_video2 | 50 | stationary | 0.009768 | 0.019103 | 1,152 |
| DJI_0204_video2 | 50 | transformer_ensemble | 0.005248 | 0.012450 | 1,152 |
| DJI_0207 | 20 | cv20 | 0.004875 | 0.008539 | 2,487 |
| DJI_0207 | 20 | cv_ols | 0.004911 | 0.008605 | 2,487 |
| DJI_0207 | 20 | stationary | 0.004827 | 0.008356 | 2,487 |
| DJI_0207 | 20 | transformer_ensemble | 0.004522 | 0.007564 | 2,487 |
| DJI_0207 | 50 | cv20 | 0.009764 | 0.019471 | 2,397 |
| DJI_0207 | 50 | cv_ols | 0.009928 | 0.019861 | 2,397 |
| DJI_0207 | 50 | stationary | 0.008416 | 0.015408 | 2,397 |
| DJI_0207 | 50 | transformer_ensemble | 0.009539 | 0.017461 | 2,397 |

## Central finding

In **DJI_0204_video2**, CV-OLS is the lowest-ADE baseline for **both** forecast horizons and **all four** sampled elephant tracks:

- Future 20 frames: CV-OLS ADE 0.001246; Transformer ADE 0.002752 (**120.9% higher**); Transformer FDE **103.6% higher** than CV-OLS. Transformer improves over Stationary by **32.1%** in ADE. In 10.1% of overlapping windows Transformer has lower ADE than CV-OLS.
- Future 50 frames: CV-OLS ADE 0.002271; Transformer ADE 0.005248 (**131.0% higher**); Transformer FDE **164.5% higher** than CV-OLS. Transformer improves over Stationary by **46.3%** in ADE. In 10.3% of overlapping windows Transformer has lower ADE than CV-OLS.

This contrasts with `DJI_0207`, where the Transformer was the best of these four models at the 20-frame horizon, while Stationary had the lowest ADE at the 50-frame horizon. Thus **the model ranking is not stable across these two recordings**. Report each recording separately. Do not combine overlapping windows from both recordings and describe the result as independent external validation.

## Robustness and interpretation

- Non-overlap-within-track offsets for the 20-frame horizon: Transformer beat CV-OLS in **0/40** offset configurations. Offsets share recording events and are not independent replications.
- Non-overlap-within-track offsets for the 50-frame horizon: Transformer beat CV-OLS in **0/70** offset configurations. Offsets share recording events and are not independent replications.
- The original training canonicalization's **scale floor activates for 100% of windows in this recording** (all eight track×horizon rows). This is a major input-distribution warning; a successful inference run does not demonstrate calibrated or robust generalization.
- Bounding-box centers are normalized image-plane locations. They are affected by camera motion, and native FPS/frame timestamps are unverified. Therefore the results are **20/50-frame exploratory forecasts**, not 2/5-second or world-coordinate forecasting.
- The two WildLive folders have not been proven to be distinct flight/herd events or independent of the earlier WildBox data. Visual QA was sampled, not exhaustive.
- Per-window superiority percentages and non-overlapping offset counts describe this video only. They are **not p-values or confidence intervals**, and there is no defensible cluster-based significance inference from one unverified capture group per recording.
- The submitted output ZIP contains per-window **error metrics**, but not the underlying per-step predicted trajectories or model weights. It is possible to recalculate all reported ADE/FDE aggregates, **not** to independently rerun frozen-model forward passes from this output ZIP alone.

## Recommendation for manuscript

Present this as **an exploratory two-sequence, frame-index-based transfer stress test**. Emphasize that stationary and linear extrapolation baselines identify different apparent motion regimes across sequences, and that frozen learned forecasts can underperform low-complexity baselines. No method novelty, model improvement, significance, conformal coverage, or independent validation is established by these results alone.

## Next scientifically valid work

1. Lock the original six checkpoints and preprocessing; retain both recordings with hashes.
2. Acquire **additional genuinely independent elephant recordings**, verify species and sampled identities, and where possible verify recording cadence and source relationships before evaluation.
3. Report per-recording, per-horizon error tables first; use a capture-event-level analysis only after independence is documented. Do not tune against either of these assessed recordings.
4. If needed, investigate whether camera motion compensation or documented annotation-jitter handling changes the relative ranking as a **new, explicitly exploratory method experiment**, not a retroactive correction of frozen results.

## Files included

- `cross_sequence_pooled_ADE_FDE.csv`
- `DJI0204_per_track_ADE.csv`
- `DJI0204_pairwise_model_comparison.csv`
- `DJI0204_nonoverlap_offset_sensitivity.csv`
- `cross_sequence_ADE_comparison.png`
