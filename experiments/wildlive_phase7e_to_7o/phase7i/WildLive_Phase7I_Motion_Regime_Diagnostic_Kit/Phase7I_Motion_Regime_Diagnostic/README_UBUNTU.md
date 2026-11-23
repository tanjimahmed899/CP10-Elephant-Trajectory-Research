# Phase 7I — Ubuntu / WSL reproduction

This kit **does not retrain** the Transformer, alter frozen checkpoints, estimate FPS, or demonstrate independent validation. It computes descriptive motion features using only 20 observed coordinates and aligns them with the previously archived per-window model errors.

## Input files (must already be in Windows Downloads)

- `Phase7F_Frozen_Results.zip`
- `WildLive_Phase7E_Reproduced_Results.zip`
- `Phase7H_DJI0204_Frozen_Evaluation.zip`
- `DJI_0204_video2_boxid.zip`

Download `WildLive_Phase7I_Motion_Regime_Diagnostic_Kit.zip` into your Windows Downloads folder and run:

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7I
unzip -o /mnt/c/Users/HP/Downloads/WildLive_Phase7I_Motion_Regime_Diagnostic_Kit.zip \
  -d ~/CP10_new_data_private/WildLive/Phase7I
cd ~/CP10_new_data_private/WildLive/Phase7I/Phase7I_Motion_Regime_Diagnostic

source ~/cp10-pilot-venv/bin/activate
pip install numpy pandas matplotlib
python src/test_phase7i.py

python src/run_phase7i.py \
  --phase7f /mnt/c/Users/HP/Downloads/Phase7F_Frozen_Results.zip \
  --phase7e /mnt/c/Users/HP/Downloads/WildLive_Phase7E_Reproduced_Results.zip \
  --phase7h /mnt/c/Users/HP/Downloads/Phase7H_DJI0204_Frozen_Evaluation.zip \
  --dj0204 /mnt/c/Users/HP/Downloads/DJI_0204_video2_boxid.zip \
  --output ./recomputed_results
```

Expected console statement: `7308 observation-only paired windows...; checkpoint hashes matched; 104 archived summary rows reconciled`. All **9 tests** should be `OK`. The code automatically rejects mismatches in source SHA-256 or original checkpoint manifest records.

Inspect generated figures:

```bash
explorer.exe ./recomputed_results/figures
```

Copy everything into Windows Downloads as an archive:

```bash
zip -qr "$HOME/WildLive_Phase7I_Reproduced_Results.zip" recomputed_results
cp "$HOME/WildLive_Phase7I_Reproduced_Results.zip" /mnt/c/Users/HP/Downloads/
```

## What the output means

- `per_window_observation_motion_and_model_errors.csv`: joined 7,308 derived observation features / original forecast errors. Highly **overlapping** rows.
- `speed_tercile_error_table.csv`: LOW/MID/HIGH bins computed separately within *each* source and horizon. **Descriptive only**.
- `within_elephant_speed_regime_deltas.csv`: track-stratified descriptive differences.
- `multi_animal_common_motion_proxy.csv`: elephant-only common displacement; **not true camera motion**.
- `analysis_manifest.json`: input hashes and explicit nonconfirmation declarations.

No new animal-coordinate data or model checkpoints are included in this ZIP. Respect WildLive licensing and avoid committing privately derived/raw annotation data to a public GitHub repository without authorization.
