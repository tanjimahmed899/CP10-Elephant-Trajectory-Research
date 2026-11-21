# CP10 Stage 7D — Offline frozen ByteTrack replay

## Scope

This is a **retrospective** experiment using the already generated Stage7C `Stage7C_Detections.csv` as input, NOT new detector inference, retraining, or TestReal hyperparameter selection. The original and adapted detectors are compared on the same six BIRDSAI TestReal sequences and frame list, and track state resets for every sequence. The tracking YAML must match the original R13C SHA256 `eaefad899964d7a12a13090d5da32bcdae8f0109968ffca982fdba2c60ad3c38`. No implicit default-YAML substitution is made.

The original R13 baseline was run under Ultralytics **8.3.75**; Stage7C used **8.4.216**. This tracker replay uses **8.4.216** with the old YAML and is an updated-environment matched comparison, not exact R13 reproduction. Stage7C's original detector matched 2453 TP versus historical 2431. Report both.

## WSL commands

1. Download `CP10_Stage7D_Offline_Tracker_Replay.zip` to `C:\Users\HP\Downloads`.
2. Run:

```bash
mkdir -p ~/cp10_stage7d_tools
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage7D_Offline_Tracker_Replay.zip" ~/cp10_stage7d_tools
bash ~/cp10_stage7d_tools/CP10_Stage7D_Offline_Tracker_Replay/run_stage7d_wsl.sh
```

The runner activates `~/cp10_detection_venv`, reads `C:\Users\HP\Downloads\CP10_Stage7C_TestReal_Review.zip` and the original unmodified TestReal GT at `/mnt/e/AIUB/Capstone Project/Journal Based Results/Main Result/trajectory_dataset/cleaned_track_points.csv`. It looks for original R13C `bytetrack_FROZEN.yaml` in known research directories. For a YAML elsewhere, run `export CP10_R13_YAML='/full/wsl/path/to/bytetrack_FROZEN.yaml'` first. Any file with the wrong hash will be rejected. **Never bypass the YAML check or adjust tracker thresholds after inspecting TestReal.**

The review ZIP `CP10_Stage7D_Tracker_Review.zip` will be placed in Windows Downloads even if the experiment is blocked. Upload it for results audit.

## Outputs (if complete)

* `Stage7D_Aggregate_Tracking.csv` and `Stage7D_PerSequence_Tracking.csv`: HOTA (19 thresholds from 0.05 to 0.95), IDF1, IDSW, fragmentation, MOTA, plus underlying counts, evaluated using the functions of the original user-provided `R13E_tracking_metrics_FRESH.py` (not a new independent evaluator).
* `Stage7D_ByteTrack_All_Boxes.csv`: matched offline tracker output with exact frame/sequence/ID.
* `Stage7D_Forecastable_Segments.csv` and `Stage7D_Forecastability_Summary.csv`: R13F_B-compatible construction with 30→10 Hz decimation, up to 3 missing 30 Hz frames interpolated, 20 observed-history steps, 20/50 future steps, stride 10 at 10 Hz. **These count only possible trajectory windows; they are not evaluated Transformer prediction windows.**
* `Stage7D_Provenance.json`: version, hashes, reproducibility limitations.

## Interpretation policy

- Stage7C fine-tuning increased detector recall from 5.10% to 56.98% but false boxes/frame increased from 0.20 to 5.80. The tracker experiment must measure whether increased detections actually improve identity consistency and forecasting window yield.
- The original baseline precision and HOTA may not match historical R13 due to version differences; this is not grounds to tune tracker settings on TestReal.
- Even if more windows are constructed, the final paper must not claim improved **end-to-end Transformer ADE/FDE**, real wildlife alerts, or independent external validation without those separate evaluations.
- BIRDSAI TestReal was previously analyzed (R13); the present evaluation is retrospective.
