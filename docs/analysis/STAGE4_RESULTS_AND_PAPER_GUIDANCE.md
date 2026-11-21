# CP 1.0 — Stage 4 R8 geometric replay and learned forecasting comparison

**Status:** Executed with the user's uploaded BIRDSAI raw-window ZIP, the previously uploaded corrected/core dataset, and the original model architecture reproduced from `All Python Code.zip`. All reported learned results are **new retrospective experimental evaluations**, not original-paper results or independent confirmation. Date: 2026-11-20.

## 1. What was verified

- Raw 30-Hz centroid observations: 63,750 rows (collected from 91,419 source rows), representing 151 distinct split/sequence/segment tracks.
- Complete core windows: **2,792 / 2,792**. The correct matching units are 531 TrainReal + 1,053 TestReal 2-second windows, and 382 TrainReal + 826 TestReal 5-second windows.
- Using the originally saved `repaired_ego_motion.csv`, each per-frame 2D similarity transform was composed in the original `build_corrected_trajectories.py` order to the observation-start frame. Frame dimensions were recovered from raw pixel versus normalized coordinates, checked against the saved correction pipeline.
- Maximum absolute coordinate discrepancy against saved corrected NPZ for all 2,792 replayed windows: **5.96e-8**. This is an **exact numerical replay** of the existing *estimated affine* correction, **not** a proof of true 3D homography, ground-plane registration, or camera-pose accuracy.
- Matched normalized raw and corrected history/future arrays, and per-future-frame mappings to a common evaluation frame, are saved in `R8_Paired_*.npz`.

## 2. Baseline evaluation — matched common-target ADE

All predictions below are evaluated against the SAME observation-start corrected-frame ground truth. For raw-image predictions, the recorded future camera-motion transforms are applied **only to the predictions during retrospective evaluation**. They are **not model features** and cannot be assumed available as future information in deployment.

| TestReal horizon | Baseline | Raw prediction ADE | Corrected prediction ADE |
|---|---|---:|---:|
| 2 s | Stationary | 0.051772 | 0.020033 |
| 2 s | Constant velocity | 0.077796 | 0.020148 |
| 5 s | Stationary | 0.081493 | 0.039685 |
| 5 s | Constant velocity | 0.159073 | 0.040174 |

These raw values are **not the same as earlier last-point translation-aligned R8 metrics**. Do not substitute them into the original manuscript without updating the protocol.

## 3. Same architecture, different representation — three matched seeds

Identical 2-layer, 64-width, 4-head Transformer architecture; seeds 42/52/62; same TrainReal train/validation split, same BIRDSAI TestReal windows; early stopping by track-macro validation ADE in the corrected evaluation frame. TrainReal per-arm 5th-percentile scale floor is computed using *training indices only*. Both arms are retrained; corrected arm is not just an existing frozen checkpoint.

### A. Full representation change: raw input→raw future vs corrected input→corrected future

| Horizon | Raw TestReal ADE mean (SD) | Corrected ADE mean (SD) | Mean within-seed ADE reduction |
|---|---:|---:|---:|
| 2 s | 0.057697 (0.000705) | 0.019500 (0.000399) | **66.20%** |
| 5 s | 0.107395 (0.004500) | 0.036340 (0.000852) | **66.14%** |

Both raw and corrected outputs are evaluated in the same reference frame; nevertheless this changes both the input and output training representation. For that reason, it alone does not isolate the effect of the history representation.

### B. Stronger input-only ablation: raw history→corrected future vs corrected history→same corrected future

| Horizon | Raw-history ADE (three-seed mean) | Corrected-history ADE (same seeds) | Mean paired within-seed ADE reduction |
|---|---:|---:|---:|
| 2 s | **0.058217** | **0.019500** | **66.51%** |
| 5 s | **0.078495** | **0.036340** | **53.62%** |

This is the more interpretable comparison for the manuscript because BOTH branches predict the **same physical coordinate target** (corrected future in observation-start frame); only their observed histories differ. The canonicalization of each input and its target is still computed relative to that branch's observed history, as required by the original Transformer normalization design, and should be disclosed.

- Matching train/validation sizes: 2 s = 434 / 97, 5 s = 294 / 88. TestReal samples: 2 s = 1,053; 5 s = 826. Overlapping windows are not independent observations.
- Same-target input-only FDE means: 2 s raw 0.071927 vs corrected 0.033630; 5 s raw 0.107932 vs corrected 0.067905.
- The input-only reduction was positive for every evaluated TestReal source sequence, but there are only **six** sequences with 2-second windows and **five** with 5-second windows. One 5-second source has only **two** windows. Source-level replication is limited.

### C. Estimated camera-motion stratification

The figure `Figure_R8_InputOnly_CameraMotion_Sensitivity.png` stratifies by the average displacement of the normalized image center under the future frame-to-start affine mapping. This is a **camera-motion proxy**, not independently measured UAV pose/velocity. In the 5-second input-only ablation, ADE reductions range from approximately **42.7% (lowest motion quartile)** to **62.2% (highest quartile)**. The 2-second quartiles are not strictly monotonic. Correlation is descriptive, not causal evidence; source/scene effects may contribute.

## 4. Critical limitations / submission conditions

1. **Not exact 3D geometry:** The transforms are estimated 2D similarity/affine motion, reconstructed numerically. To claim exact geometry, require calibrated camera intrinsics/extrinsics, consistent source/target projection (and a ground plane/depth assumption where appropriate), plus stationary-background reprojection residuals.
2. **Retrospective evaluation:** TestReal is pre-existing, not an untouched prospective holdout; the training is new but target coverage or cross-camera generalization is not newly established.
3. **No real metre-scale:** Errors are normalized image-coordinate values; no conversion to metres or warning lead-time distances.
4. **No end-to-end result:** The experiment uses annotated trajectory histories. It does not repair the detector-to-tracker failure or prove a deployable YOLO→tracking→forecasting pipeline.
5. **Source-dependence:** Samples overlap. Seed-to-seed SD is not a confidence interval for independent capture sites. Do not claim statistical significance using thousands of overlapping windows as independent data.
6. **Future transforms:** Only the original raw-output arm uses recorded future transformations for *evaluation mapping*. The stronger raw-input→corrected-target ablation does not use future transforms in the model or prediction mapping.

## 5. Suggested manuscript placement and caption

Place `Figure_R8_InputOnly_SameTarget_ThreeSeed.png` in the **Camera Motion Correction Ablation** Results subsection, followed by `Figure_R8_InputOnly_CameraMotion_Sensitivity.png`. The full-representation control and detailed baselines can be Supplementary Figures.

**Figure X. Matched learned-model ablation for motion-corrected elephant trajectory forecasting.** A two-layer Transformer was trained from raw or camera-motion-corrected observation histories using identical training partitions, hyperparameters, and three random seeds. Both models predicted the same corrected-frame future positions and were evaluated against identical BIRDSAI TestReal targets. Bars show mean ADE and standard deviation across training seeds at 2-second and 5-second horizons. Predictions and ground truth use normalized image coordinates.

**Suggested careful result statement:** "In a matched retrospective TestReal ablation, correction of the observed trajectory history reduced mean ADE from 0.0582 to 0.0195 at 2 s and from 0.0785 to 0.0363 at 5 s across three Transformer seeds. These findings support the predictive utility of the estimated affine camera-motion correction on the evaluated BIRDSAI sequences; they do not establish exact 3D geometric alignment or independent cross-camera deployment performance."

## 6. Reproducibility

The ZIP contains paired normalized raw/corrected arrays, per-future-frame transforms, metadata and train/validation index arrays, sample-wise evaluation CSVs, all seed checkpoints and training logs, source-group and ego-motion stratification, code and figure files.

With Python 3 and NumPy, Pandas, PyTorch, Numba and Matplotlib in a clean environment, from inside the extracted package:

```bash
python3 stage5_matched_transformer_pilot.py --paired-root . --out reproduced_full --seeds 42 52 62
python3 stage5_raw_input_only_pilot.py --paired-root . --out reproduced_input_only --seeds 42 52 62
```

The raw-input-only script runs the raw branch, and the corrected comparison checkpoints/results are available in `learned_pilot` and `learned_additional_seeds` (or `reproduced_full`). For an end-to-end *raw→corrected affine reconstruction* repeat, run `stage4_verify_and_pair.py` with the original Stage 4 and Stage 3 ZIP files extracted as described by its CLI help.

**Do not yet write "exact-geometry correction" or "generalizes to unseen drones" in the abstract.** The substantive new result is a strong **affine-correction learned-model ablation on pre-existing BIRDSAI TestReal**.
