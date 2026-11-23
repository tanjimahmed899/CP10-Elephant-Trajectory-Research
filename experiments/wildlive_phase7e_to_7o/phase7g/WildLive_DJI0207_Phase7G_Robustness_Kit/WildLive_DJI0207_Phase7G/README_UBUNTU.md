# Phase 7G — Ubuntu / WSL reproducibility

The audit source, summary CSVs, verified figures, and scientific interpretation are included. This package intentionally excludes source JPEGs, original boxid JSON, trained model checkpoints and underlying prediction arrays.

## 1. Download and extract this Phase 7G ZIP to Windows Downloads

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7G
unzip -o /mnt/c/Users/HP/Downloads/WildLive_DJI0207_Phase7G_Robustness_Kit.zip -d ~/CP10_new_data_private/WildLive/Phase7G
cd ~/CP10_new_data_private/WildLive/Phase7G/WildLive_DJI0207_Phase7G
source ~/cp10-pilot-venv/bin/activate
python -m pip install numpy pandas matplotlib
```

## 2. Re-run using your original private files

Assumes the two previous phase result ZIPs were downloaded in Windows Downloads. If they are elsewhere, change the filenames below to the actual paths.

```bash
python src/phase7g_robustness_audit.py \
  --phase7f-zip /mnt/c/Users/HP/Downloads/Phase7F_Frozen_Results.zip \
  --phase7e-zip /mnt/c/Users/HP/Downloads/WildLive_Phase7E_Reproduced_Results.zip \
  --output ./local_reproduced_phase7g

explorer.exe ./local_reproduced_phase7g
```

Expected audit: `ZIP_CRC: PASS`, 64 recomputed summary rows, 4,884 input window rows, and NPZ SHA-256 matching the Phase 7F manifest. If the SHA-256 does not match, do not accept the evaluation as reproducible until the source file mismatch is resolved.

**Scientific restrictions:** Unknown released-frame cadence and possible source overlap; all results describe one capture group and overlapping image-plane windows only. No significance testing is licensed by these data. Neither nominal prediction horizon equals verified seconds.
