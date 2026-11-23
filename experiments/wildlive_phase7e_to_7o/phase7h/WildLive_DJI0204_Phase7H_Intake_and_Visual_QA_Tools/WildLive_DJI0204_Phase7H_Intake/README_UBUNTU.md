# Phase 7H — DJI_0204_video2 annotation intake and visual QA (Ubuntu WSL)

## 1. Download this package and unzip (optional — all intake outputs already supplied)

Save `WildLive_DJI0204_Phase7H_Intake_and_Visual_QA_Tools.zip` to Windows Downloads.

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7H/DJI_0204_QA
unzip -o /mnt/c/Users/HP/Downloads/WildLive_DJI0204_Phase7H_Intake_and_Visual_QA_Tools.zip \
  -d ~/CP10_new_data_private/WildLive/Phase7H/DJI_0204_QA
cd ~/CP10_new_data_private/WildLive/Phase7H/DJI_0204_QA/WildLive_DJI0204_Phase7H_Intake
source ~/cp10-pilot-venv/bin/activate
python scripts/test_intake_scripts.py
python scripts/audit_dji0204.py \
  --zip ~/CP10_new_data_private/WildLive/Phase7H/candidates/DJI_0204_video2_boxid.zip \
  --output ./local_intake_recomputed
```

## 2. Download eight original images for visual check

```bash
NAME="DJI_0204_video2"
ROOT="$HOME/CP10_new_data_private/WildLive/Phase7H/candidates/$NAME"
mkdir -p "$ROOT/sample_frames"
BASE="https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/$NAME/frames"
for n in 0 33 34 100 200 303 304 356; do
  wget -c -P "$ROOT/sample_frames" "$BASE/frame_${n}.jpg" || echo "FAILED frame_${n}.jpg"
done
file "$ROOT/sample_frames"/frame_*.jpg
```

If a JPG is unavailable under this exact URL, inspect the official WildLive portal's `frames` directory for that sequence before changing URLs. Do not fabricate missing files.

## 3. Overlay original annotation boxes (no editing of source JPEGs/JSON)

From the extracted package directory:

```bash
python scripts/make_visual_overlays.py \
  --zip "$HOME/CP10_new_data_private/WildLive/Phase7H/candidates/DJI_0204_video2_boxid.zip" \
  --frames-dir "$HOME/CP10_new_data_private/WildLive/Phase7H/candidates/DJI_0204_video2/sample_frames" \
  --output ./visual_overlays
```

Check that ID 1 and ID 3 correspond to distinct elephants rather than two boxes on the same elephant, check ID 0 jump around 303/304 and ID 2 around 33/34. A small sample cannot prove all 357 frames are correct. A full 357-frame overlay movie/annotation inspection may be needed if sampled evidence is ambiguous.

## 4. Copy source frames ZIP and overlays to Windows Downloads for upload

```bash
ROOT="$HOME/CP10_new_data_private/WildLive/Phase7H/candidates/DJI_0204_video2"
(cd "$ROOT/sample_frames" && zip -q -r "$HOME/DJI_0204_video2_visual_QA_frames.zip" frame_*.jpg)
cp "$HOME/DJI_0204_video2_visual_QA_frames.zip" /mnt/c/Users/HP/Downloads/
# Optional: overlays ZIP
zip -q -r "$HOME/DJI_0204_video2_visual_overlays.zip" visual_overlays
cp "$HOME/DJI_0204_video2_visual_overlays.zip" /mnt/c/Users/HP/Downloads/
explorer.exe /mnt/c/Users/HP/Downloads/
```

Upload the **visual QA frames ZIP** and/or overlays here for checking. Use original images from the WildLive portal. Do not upload these images to public GitHub without checking the dataset license.

## 5. Do NOT run frozen evaluation until ID QA is checked

Once approved for exploratory evaluation, use the already extracted Phase 7H kit, with locked checkpoints from **Main Result**, not R9 ablation weights:

```bash
cd ~/CP10_new_data_private/WildLive/Phase7H/WildLive_Phase7H_MultiRecording_Kit
CHECKPOINT_DIR="/mnt/e/AIUB/Capstone Project/Journal Based Results/Main Result/transformer_results"
python src/phase7h.py evaluate \
  --zip "$HOME/CP10_new_data_private/WildLive/Phase7H/candidates/DJI_0204_video2_boxid.zip" \
  --sequence DJI_0204_video2 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output ./results/DJI_0204_video2_locked
```

This would be **exploratory** frame-index-only testing, not a confirmed independent capture-group validation. Do not use observed results to retune models, thresholds or QA selections.
