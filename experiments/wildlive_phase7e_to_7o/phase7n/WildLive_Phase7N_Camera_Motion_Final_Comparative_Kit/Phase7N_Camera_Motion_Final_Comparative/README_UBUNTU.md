# Phase 7N — Ubuntu WSL Reproduction Guide

This package already contains the computed Phase 7N report, CSV files, and figures. You do **not** need to redownload the 204.6-GiB WildLive dataset or retrain any model.

## Step 1: Extract and test (no source data needed for tests)

Download `WildLive_Phase7N_Camera_Motion_Final_Comparative_Kit.zip` to Windows Downloads. In Ubuntu WSL:

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7N
unzip -o /mnt/c/Users/HP/Downloads/WildLive_Phase7N_Camera_Motion_Final_Comparative_Kit.zip \
  -d ~/CP10_new_data_private/WildLive/Phase7N
cd ~/CP10_new_data_private/WildLive/Phase7N/Phase7N_Camera_Motion_Final_Comparative
source ~/cp10-pilot-venv/bin/activate
python -m pip install numpy pandas matplotlib opencv-python-headless
python src/test_phase7n.py
```

Expected: `Ran 11 tests` and `OK`. These offline structural/numerical tests cannot replace audit of the real data.

## Step 2: Optional FULL independent recalculation from original ZIPs

To reproduce ALL outcomes, collect the following ten files in **one** folder. These are the original project ZIPs, not files inside this release:

- `Phase7F_Frozen_Results.zip`
- `Phase7H_DJI0204_Frozen_Evaluation.zip`
- `Phase7K_DJI0161_Locked_Evaluation.zip`
- `Phase7L_DJI0204_video1_Exploratory_Results.zip`
- `WildLive_DJI_0207_boxid.zip`
- `DJI_0204_video2_boxid.zip`
- `DJI_0204_video1_boxid.zip`
- `DJI_0161_video1_boxid.zip`
- `WildLive_DJI0207_priority_frames.zip`
- `DJI_0204_video2_visual_QA_frames.zip`

Example if all ten are present in Windows Downloads:

```bash
INPUTS=/mnt/c/Users/HP/Downloads
python src/phase7n.py --inputs-dir "$INPUTS" --output ./my_recomputed_7n
```

**Do not assume every file is in Downloads**. You may need to copy the original files from your prior project folders into a single dedicated directory. The program aborts on a missing source and will **not fabricate** camera-motion data. It only reads files and writes results to `--output`.

Compare the reported CSVs to the ones in `results/` (minor image registration numeric variation across OpenCV releases is possible). Package was computed with OpenCV 4.13.0 in the review environment. No original JPEGs or annotations are included in this kit due to data licensing and research-sensitivity safeguards.

## Step 3: Submission-ready items

Read `PHASE7N_SCIENTIFIC_REPORT.md`, use `results/recording_level_ADE_FDE.csv` for complete ADE/FDE, `results/locked_adaptive_vs_cv_ols.csv` for the locked adaptive claims, and `results/consecutive_frame_camera_registration.csv` for *sampled* registration. Figures are in `results/figures/`.

**Never state that camera-corrected forecasting was validated.** The sparse RGB snapshots do not cover the complete forecasting windows and prediction coordinates for most evaluations are not archived.
