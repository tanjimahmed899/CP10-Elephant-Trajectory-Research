# CP10 Stage 7 — WSL quick run

## Purpose

Development-only frozen YOLOv8 inference: original R13 checkpoint, in-domain BIRDSAI `train_real` images, predeclared sensitivity settings. **Not** retraining and **not** an unseen-TestReal paper result yet. Source tree is read-only.

## Setup

Download `CP10_Stage7_Detector_Pilot.zip` into `C:\Users\HP\Downloads`.

```bash
mkdir -p ~/cp10_stage7_tools
python3 -m zipfile -e \
"/mnt/c/Users/HP/Downloads/CP10_Stage7_Detector_Pilot.zip" \
~/cp10_stage7_tools

# In WSL, activate the Python environment that already contains ultralytics.
# If the current (cp10_detection_venv) environment is active, keep it.
python3 -c 'import ultralytics, torch, numpy; print("Detector environment ready")'

bash ~/cp10_stage7_tools/CP10_Stage7_Detector_Pilot/run_stage7_wsl.sh \
  2>&1 | tee ~/CP10_Stage7_Console.log

cat ~/CP10_Stage7_Dev_Pilot/Stage7_Readout.txt
# If the pilot ran, also display:
if [ -f ~/CP10_Stage7_Dev_Pilot/Stage7_Dev_Decision.txt ]; then
  cat ~/CP10_Stage7_Dev_Pilot/Stage7_Dev_Decision.txt
fi

cp ~/CP10_Stage7_Dev_Review.zip "/mnt/c/Users/HP/Downloads/"
```

Upload `CP10_Stage7_Dev_Review.zip` here. If preflight says **BLOCKED**, upload that same ZIP; it contains the exact missing path/status information. You do not need to upload RGB frames or model checkpoint to ChatGPT.

### Notes

- The runner first confirms the original checkpoint file's locked SHA-256 and requires at least two TrainReal sequences with >=20 total paired annotated images. If not, it refuses to run detector inference rather than tune the previously examined TestReal set.
- The runner's default configurations are original confidence 0.10 at 640px, 0.10 at 960px, 0.03 at 640px, 0.03 at 960px. Images are sequential and CPU is the default. For CUDA-enabled WSL, set `CP10_STAGE7_DEVICE=0` only if `torch.cuda.is_available()` returns True.
- IoU >= 0.50 matching is one-to-one and frame-level. FP/image is measured only on the **annotated TrainReal positive images selected by the pilot**. Do not quote it as a no-elephant false-alarm rate.
- The source R13 HOTA/IDF1 cannot be reused as if new pilot inference improved tracking. Following this run, freeze chosen configuration and run a newly measured tracker/test evaluation.
- Pilot settings and selected images are selected before any model predictions. Once TestReal is evaluated, do not tune settings using those test outcomes.
- All output files are written under `~/CP10_Stage7_Dev_Pilot`, and a compact review ZIP under `~/CP10_Stage7_Dev_Review.zip`.
