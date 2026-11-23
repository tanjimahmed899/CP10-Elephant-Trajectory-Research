# Phase 7E — WildLive DJI_0207 frame-index forecasting pilot

**Evidence status:** Reproducible **retrospective exploratory baseline evaluation** based on user-supplied `WildLive_DJI_0207_boxid.zip`; not a new independent capture-group confirmation, not a 2-second/5-second experiment, not an end-to-end detector/Transformer evaluation.

## 1. Evidence and processing

- Input: 868 `frame_0.json`–`frame_867.json` WildLive `boxid` annotations, 3840 × 2160 native annotation coordinates.
- Three labelled elephant IDs (0,1,2) present in all 868 frame indices, with 2,604 source bounding boxes. Source metadata lacks verified FPS, timestamps, camera intrinsics/poses and source independence checks.
- Bounding-box annotations are sometimes rectangles represented by **two diagonal points** and sometimes by **four corners**. The parsing routine takes the minimum/maximum of all valid point coordinates without altering the source JSON.
- For each frame, a 2-D target is the centre of the published bounding box, normalized as `(x / 3840, y / 2160)`. This is **annotation-centre** motion in the original *moving camera image plane*; not a physical path in metres, not camera-stabilized trajectories, and not manually verified anatomical centres.
- Input is 20 **frame indices**; prediction targets are next 20 or 50 **frame indices**, stride 1. Windows are only assumed contiguous *by annotation index*, not proven equally spaced in real time.
- Forecasting baselines were specified in code and evaluated directly, with no training, tuning or calibration on this sequence:
  - **Stationary:** all future positions equal the last observed normalized centre.
  - **CV20:** estimated per-index velocity `(last_history_centre - first_history_centre)/19`, linearly extrapolated for each future frame index.
- `ADE = mean_k ||p_hat_k - p_k||_2` and `FDE = ||p_hat_H - p_H||_2` in normalized 2-D image-plane units. No perspective-depth correction or camera-motion correction is applied.

## 2. Real-data exploratory results

| Horizon | Baseline | Window count | Mean ADE | Mean FDE |
|---|---|---:|---:|---:|
| 20 future frames | Stationary | 2,487 | 0.004826607 | 0.008356044 |
| 20 future frames | CV20 | 2,487 | 0.004875328 | 0.008539015 |
| 50 future frames | Stationary | 2,397 | 0.008416475 | 0.015407623 |
| 50 future frames | CV20 | 2,397 | 0.009763880 | 0.019471345 |

The stationary baseline achieves lower **pooled descriptive** error at both frame horizons. This finding cannot be generalized to an unseen elephant population. At the **20-frame** horizon, CV20 is *better than stationary* on **ID 1** (ADE 0.00415370 versus 0.00462563), showing the pooled result does not hold for every track. No independent confidence interval or significance test is defensible using 2,487/2,397 heavily overlapping windows from one capture group as if they were independent subjects.

These frame-based metrics **cannot be compared directly** with the previous EXP3 corrected-coordinate, fixed-FPS 2 s/5 s Transformer results because the domain, target representation, horizon definition, camera-motion correction and evaluation populations differ.

## 3. Previously identified annotation jitter — pre-specified sensitivity

Phase 7C had visually flagged the following source bounding-box extents **before** this baseline evaluation:

- Track 2 (calf): frame indices **389** and **401**, unusually wide left-extending boxes.
- Track 0 (adult): frame index **860**, unusually tall box.

**Primary result** uses ALL available annotations, including flagged boxes. A supplementary **QA-unflagged** analysis excludes a window only if its 20-frame history OR future target contains a previously flagged frame for that track. No source labels are edited and no windows are filtered based on observed prediction error.

| Horizon | Windows retained | Stationary ADE | CV20 ADE | Interpretation |
|---|---:|---:|---:|---|
| 20 frames | 2,427 / 2,487 | 0.004700942 | 0.004724703 | Descriptive selection sensitivity |
| 50 frames | 2,307 / 2,397 | 0.008360860 | 0.009600564 | Descriptive selection sensitivity |

The 20-frame filtering removes **60** windows, and 50-frame filtering **90**. Lower ADE after excluding flagged windows is **not proof that corrected trajectories or a new model improve performance**; it also changes the evaluated population. Full per-window results and predictions are saved so a reviewer can inspect exactly which samples were included.

## 4. Experimental dependencies and scientific limits

1. **Timing:** no verified FPS or frame-to-time mapping. Do not call either horizon 2 or 5 seconds; never assume 10 Hz for this release.
2. **Camera:** predictions are raw image-plane annotation-centre paths; camera ego-motion, perspective scaling and depth are unknown. No physical metres or geofence safety claims.
3. **Annotation noise:** bounding-box extent jitter may alter the centre without corresponding animal movement. These are source annotations, not independently adjudicated point trajectories.
4. **Independence:** 3 tracks in one DJI_0207 sequence are correlated, not 3 independent capture groups. WildBox recording overlap and capture provenance remain unresolved. DJI_0207 has already been inspected repeatedly and is no longer prospectively untouched.
5. **Methods tested:** only deterministic Stationary and CV20. No trained Transformer, detector, tracker or conformal calibration was executed. No precision, recall, 90% coverage, warning performance, or deployment latency result was produced.
6. **Data rights:** WildLive source annotation licence and any publication/redistribution permissions must be reviewed; retain source and derived trajectory arrays privately. Code/method descriptions may be shared separately.

## 5. Reproducibility

The source ZIP is not repackaged in this bundle; retain it privately and verify its SHA-256 against `outputs/audit_manifest.json`. The code validates 868 exact indexed frames, elephant identities, source image size and bounds. It reproduces the numbers above from the original JSONs and outputs:

- `outputs/original_bbox_centres.csv` — 2,604 original normalized centres and native boxes (derived data; private).
- `outputs/frame_based_windows_and_baselines.npz` — each track/horizon's 20-frame histories, future GT, predicted arrays, window start and QA indicator (derived data; private).
- `outputs/per_window_baseline_metrics.csv` — every horizon/track/start's baseline errors and QA status.
- `outputs/baseline_summary.csv` — pooled and track-wise summaries, all/QA-unflagged.
- `outputs/DJI0207_raw_annotation_centre_tracks.png` — descriptive plot (x/y normalized image axes, no physical coordinates).
- `outputs/DJI0207_frame_based_ADE_baselines.png` — actual numerical baseline ADE; plot is not a model superiority claim.

Software checks: 6/6 unit tests passed; additionally checked that all windows had the expected sizes and 2-D prediction arrays, that frame-index starts were consecutive, that predictions used only observed histories, and that independently vectorized ADE recomputation matched recorded values.

## 6. Next scientifically justified task

Do a **camera-motion and annotation-extent sensitivity analysis** only after documenting its assumptions, then investigate whether a frozen trained Transformer can ingest this frame-index/raw-image-coordinate representation. A Transformer trained on 10 Hz corrected coordinates may not be compatible without a valid temporal and geometric mapping. The next new-group validation should be performed on *different*, properly documented video groups and not re-tuned on DJI_0207.

**Final classification:** Real source-supported *exploratory descriptive evidence*, **not independent confirmatory performance**, **not 2 s or 5 s forecasting**, **not evidence of new method novelty**.
