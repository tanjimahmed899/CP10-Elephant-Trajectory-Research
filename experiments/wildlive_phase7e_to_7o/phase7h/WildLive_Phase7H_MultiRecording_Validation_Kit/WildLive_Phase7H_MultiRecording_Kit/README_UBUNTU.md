# Phase 7H — Ubuntu/WSL commands (Windows user HP)

## Setup
Download `WildLive_Phase7H_MultiRecording_Kit.zip` to Windows Downloads.

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7H
unzip -o /mnt/c/Users/HP/Downloads/WildLive_Phase7H_MultiRecording_Kit.zip \
  -d ~/CP10_new_data_private/WildLive/Phase7H
cd ~/CP10_new_data_private/WildLive/Phase7H/WildLive_Phase7H_MultiRecording_Kit
source ~/cp10-pilot-venv/bin/activate
python src/test_phase7h.py
```

## Regression check (do NOT count DJI_0207 as new dataset)

```bash
python src/phase7h.py intake \
  --zip ~/WildLive_DJI_0207_boxid.zip \
  --sequence DJI_0207 \
  --output ./results/DJI_0207_regression_intake
```

Expected: 868 annotated indices, 2604 elephant annotations, 3 tracks, windows=2487/2397.

## Download candidates' `boxid` JSON only (no MP4, no 4K frames)

Use the official portal https://data.bristol.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69 to navigate to candidate's `boxid` section. The following uses the same folder URL pattern that previously worked for DJI_0207, but **accessibility of other directories is not guaranteed**. Some directory trees include an extra `__boxid` subfolder.

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7H/candidates
BASE="https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69"
for name in DJI_0117_video2 DJI_0117_video3 DJI_0204_video2 DJI_0601_video2 DJI_0601_video6 DJI_20230719145427_0002_V_video5; do
  target="$HOME/CP10_new_data_private/WildLive/Phase7H/candidates/$name"
  mkdir -p "$target/boxid"
  wget -r -l 3 -np -nd -A '*.json' -P "$target/boxid" "$BASE/$name/boxid/"
  count=$(find "$target/boxid" -type f -name 'frame_*.json' | wc -l)
  echo "$name : $count frame annotations"
  if [ "$count" -gt 0 ]; then
    (cd "$target" && zip -q -r "../${name}_boxid.zip" boxid)
  fi
done
```

If the server refuses listing or count=0, manually navigate that candidate's official dataset `boxid` page; **do not manufacture empty ZIPs or assert dataset absent**. It is acceptable to test only one candidate first if downloading all is inconvenient. If a ZIP includes `frame_N.json` with duplicate indices from distinct nested sequences, separate those sequences first.

## Audit one candidate locally, without fitting a model

```bash
python src/phase7h.py intake \
  --zip ./candidates/DJI_0601_video2_boxid.zip \
  --sequence DJI_0601_video2 \
  --output ./results/DJI_0601_video2_intake
```

Check `labels` contains `elephant`, and number of 40/70-frame continuous windows >0. Then review original sample JPEG+boxes before accepting IDs.

## Run checkpoint-locked exploratory inference only AFTER visual/species audit

```bash
CHECKPOINT_DIR="/mnt/e/AIUB/Capstone Project/Journal Based Results/Main Result/transformer_results"
python src/phase7h.py evaluate \
  --zip ./candidates/DJI_0601_video2_boxid.zip \
  --sequence DJI_0601_video2 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output ./results/DJI_0601_video2_locked
```

The script computes **NO results** when checkpoint SHA hashes differ from the original Phase 7F run. Model files in the R9 ablation directory are intentionally incompatible with the lock. This is index-based, source independence UNVERIFIED.

## Aggregate several **new** sequences as descriptive, NOT independent groups

```bash
python src/phase7h.py aggregate \
  --inputs ./results/DJI_0601_video2_locked ./results/DJI_0117_video3_locked \
  --output ./results/new_sequence_descriptive_summary
```

Only include actual results folders that exist. `DJI_0207` is intentionally rejected from the new-recording aggregate. No automatic confidence intervals or p-values are provided.

## Copy outputs to Windows Downloads and share for verification

```bash
cd ~/CP10_new_data_private/WildLive/Phase7H/WildLive_Phase7H_MultiRecording_Kit
zip -r ~/Phase7H_Local_Results.zip results -x '*.pt' '*.jpg'
cp ~/Phase7H_Local_Results.zip /mnt/c/Users/HP/Downloads/
explorer.exe /mnt/c/Users/HP/Downloads/
```

**Important:** Do not redistribute dataset annotation ZIPs, downloaded photos or model `.pt` files in public GitHub. Do not assume independent sessions simply because filenames differ.
