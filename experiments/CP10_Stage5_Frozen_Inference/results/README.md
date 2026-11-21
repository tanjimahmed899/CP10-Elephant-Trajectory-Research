# Stage 5 — retrospective frozen Transformer inference (candidate videos)

This package uses the *original* `evaluate_fresh_wildbox_holdout.py` forecasting/geometry/model architecture, adapted to run on the three local archives (`0006_V` for development and two `0004_V` for retrospective evaluation) **without an internet download** and without retraining. Sources are only read. Model checkpoints must be the original frozen Transformer seeds from `Main Result/transformer_results`.

**Important:** Even though the original script has `fresh_holdout_*` output filenames, the generated results are **not independent fresh holdout evidence**. These candidate videos are from already inspected recording groups and prior mention exists. The previously inspected `0006_V` archive is explicitly **adaptation/development** only; the other two videos are **retrospective evaluation**. None is an unseen-group holdout. Use only the terms *retrospective same-group exploratory evaluation*.

An eligibility filter performs **per-frame one-to-one** IoU ≥ 0.5 matching between each tracked `object` bounding box (512×288 representation scaled to the original frame) and KITTI `elephant` GT boxes. Windows require all 20+T frames to be matched. This intentionally avoids treating unknown-object tracks as elephants. The matching is used only to select/evaluate samples, not to manufacture detector predictions.

Run from WSL Ubuntu:

```bash
mkdir -p ~/cp10_stage5_frozen
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage5_Frozen_Inference.zip" ~/cp10_stage5_frozen
cd ~/cp10_stage5_frozen/CP10_Stage5_Frozen_Inference
python3 preflight.py
```

If dependencies missing, use a dedicated CPU environment:

```bash
sudo apt install python3-venv -y  # only if venv is missing
python3 -m venv ~/cp10_stage5_venv
source ~/cp10_stage5_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python preflight.py
```

Then:

```bash
python -u run_frozen_candidates.py 2>&1 | tee ~/CP10_Stage5_Frozen_Inference.log
```

Results in `~/CP10_Stage5_Frozen_Inference_Results`. The most important are `2s_to_2s_FROZEN_PREDICTIONS_FOR_CALIBRATION.npz`, `2s_to_5s_FROZEN_PREDICTIONS_FOR_CALIBRATION.npz`, matching `*_fresh_metadata.csv` (file name inherited from original script), `fresh_holdout_model_metrics.csv`, `STAGE5_RETROSPECTIVE_PROTOCOL.txt`, and logs. Do not fit thresholds on evaluation windows before designing a separate adaptation/test protocol. For efficient upload, run `python pack_results.py` in the kit folder and upload `~/CP10_Stage5_Frozen_Results_Review.zip`.

**Cautions:** The number of windows can be lower than the track feasibility report because anchor projection and stride=10 filtering are additional constraints. Windows overlap and should not be treated as statistically independent. No inference has been run by ChatGPT on these inaccessible E: archives; numerical forecasting results must come from your WSL run.
