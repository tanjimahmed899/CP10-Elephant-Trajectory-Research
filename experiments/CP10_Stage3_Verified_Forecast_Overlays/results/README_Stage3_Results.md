# CP10 Stage 3 — Verified real RGB forecasting overlays and R8 data lineage

## Scope and source integrity

All displayed forecasts originate from the user's uploaded `EXP3_Output_Review.zip` (frozen 3-seed Transformer ensemble saved predictions, plus 2-second observed history and annotated future). All background images and dotted elephant boxes originate from the six archived WildBox source videos, provided as `CP10_Stage3_EXP3_Source_Samples.zip`. The exact per-frame `cameras.json` and `tracking_summary.json` entries were used to validate source-frame alignment. **No new detector inference or forecasting has been run and no new test set has been collected.** The visualizations show *annotation-conditioned offline forecasting*, not a detector-to-tracker-to-warning demonstration.

The displayed image is the **first observed frame** (the 2-second observation-start camera reference). Future and observed coordinates are projected into that fixed reference. It is NOT a time-varying real-time overlay on future camera images. Do not claim that future contours align directly to the later moving camera frame. All metrics are relative normalized-image coordinates; do not describe them as metres.

## Verified geometry

Reconstructed 3D tracking centers using camera convention `X_camera = R X_world + t`. For each selected window, source-frame 3D centers are converted into the world frame, reprojected into the observation-start anchor camera, and normalized by image width and height. This precisely reproduces historical adapter `build_anchor_window` in `EXP3_Output_Review.zip`.

- 65 candidate sample/frame pairs had sampled RGB anchors available and complete matching tracking/camera data: **65/65 matched** (33 two-second; 32 five-second).
- Maximum absolute difference, across *all sampled projected coordinates*: **0.0001128 pixel** (float32 save precision). These are reconstruction differences, not model forecast errors.
- The individual selected examples and exact sample indices are in `Selected_Forecast_Examples_Geometry_Validation.csv`.
- The geometry-audit table for every matched anchor sample is `EXP3_Anchor_Frame_Geometry_Audit.csv`.

## R8 data matching (not a new camera-correction model result)

Verified the existing 30Hz corrected trajectory arrays against the existing 10Hz model input and evaluation data by `source_index`. For all 4 canonical cohorts, the coordinates `corrected.history[:, ::3, :2]` and `corrected.future[:, ::3, :2]` are **exactly equal** to the corresponding `history_abs` and `future_abs` (maximum absolute difference = 0):

| Horizon | Cohort | Canonical matched | Missing | Max abs difference |
|---|---|---:|---:|---:|
| 2s | TrainReal train/val | 531 | 0 | 0 |
| 2s | TestReal test | 1053 | 0 | 0 |
| 5s | TrainReal train/val | 382 | 0 | 0 |
| 5s | TestReal test | 826 | 0 | 0 |

Total: **2,792 matched model samples**. This shows verified dataset lineage and correct time sampling; it does not establish exact physical camera correction or benefits from training a learned model on raw vs corrected histories. Original 30Hz source-image trajectory points are required for that paired comparison.

## Files and suggested manuscript placement

`Figure_Stage3_RealDrone_5s_Forecast_Examples.png` — four 5s forecast examples from three source groups, selected to show multiple elephants in the field of view, one elephant, a group scene, and a high-error example. Use in forecasting qualitative evaluation.

`Figure_Stage3_ThreeElephants_IndividualForecasts.png` — 3 separate saved forecasts for 3 annotated elephants visible in one RGB frame. Use in annotated multi-instance trajectory evaluation (NOT detector-based multi-object tracking).

`Figure_Stage3_Original_vs_ThreeElephant_GT.png` — original RGB image and matching KITTI elephant annotations. Use as dataset example.

`Figure_Stage3_elep{1,2,3}_Matched_2s_vs_5s.png` — same video frame and same target for both forecasting horizons, for each group. Use one in the main paper, others as supplements.

### Main-paper caption — qualitative prediction examples

> **Figure X. Source-frame visualization of frozen Transformer elephant trajectory forecasts on WildBox RGB footage.** Blue trajectories denote 2 s of observed annotated history, green trajectories denote the annotated future, and red dashed trajectories denote the model's 5 s prediction. Dotted yellow boxes denote dataset ground-truth elephant annotations in the observation-start frame. All trajectories are represented in the fixed camera coordinate system at the start of the observation period; future positions are not projections into later moving-camera images. The displayed examples were retrospectively selected and do not represent independent new evaluation.

### Optional multi-elephant caption

> **Figure Y. Three simultaneously annotated elephants and their corresponding offline 5 s trajectory forecasts.** Distinct colors identify source track IDs. The observed history is dotted, the annotated future is solid, and frozen ensemble predictions are dashed. Boxes are dataset-supplied annotations, not new YOLO detections. Each trajectory is expressed in the initial observation camera frame.

## Next experiment: collect raw 30Hz BIRDSAI positions

The existing `step2b_raw_vs_corrected_forecasting.py` in `All Python Code.zip` locates original raw positions at:
`/mnt/e/AIUB/Capstone Project/Journal Based Results/Main Result/trajectory_dataset/cleaned_track_points.csv`.

To avoid uploading an entire raw dataset, use `stage4_collect_raw_wsl.py` (stdlib-only) to select ONLY frames belonging to the paired accepted core train/test windows.

1. Place `CP10_Stage3_Verified_Forecast_Overlays.zip` in `C:\Users\HP\Downloads`.
2. WSL commands:

```bash
mkdir -p ~/cp10_stage3_verified
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage3_Verified_Forecast_Overlays.zip" ~/cp10_stage3_verified
python3 ~/cp10_stage3_verified/CP10_Stage3_Verified_Forecast_Overlays/stage4_collect_raw_wsl.py \
  --root "/mnt/e/AIUB/Capstone Project/Journal Based Results" \
  --out ~/CP10_Stage4_R8_Raw_Pairing
cat ~/CP10_Stage4_R8_Raw_Pairing/R8_Stage4_Raw_Completeness_Summary.json
explorer.exe "$(wslpath -w ~/CP10_Stage4_R8_Raw_Pairing)"
```

Upload `CP10_Stage4_R8_Raw_Pairing.zip` if raw matched data is needed for the next stage. It is NOT sufficient to compare raw prediction coordinates directly to corrected target coordinates. Fair learned-model experiments require consistent coordinate conventions, identical train/validation/test samples, equal architecture/training budgets, validated reference-frame mapping, paired errors and source/track clustered uncertainties.

## Reproducibility

Included `build_verified_overlays.py`, `compare_horizons.py`, `audit_r8_pairing.py` and `stage4_collect_raw_wsl.py`. The figure scripts assume the two uploaded source ZIPs are present at `/mnt/data`. They never write into source ZIPs. `stage4_collect_raw_wsl.py` has been unit-tested on a small temporary source tree, not yet executed on the user's E: drive.
