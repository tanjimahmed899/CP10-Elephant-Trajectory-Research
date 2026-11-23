# Phase 7K Ubuntu WSL — reproducible preparation and new elephant recording intake

**Purpose:** validate a *candidate* adaptive forecaster against CV-OLS on new WildLive elephant recordings, without model or gate tuning. Verified independent capture events = 0 at this point.

## Step 0 — extract

1. Save `WildLive_Phase7K_Prospective_Validation_Kit.zip` in Windows Downloads.
2. Run:

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7K
unzip -o /mnt/c/Users/HP/Downloads/WildLive_Phase7K_Prospective_Validation_Kit.zip \
  -d ~/CP10_new_data_private/WildLive/Phase7K
cd ~/CP10_new_data_private/WildLive/Phase7K/WildLive_Phase7K_Prospective_Validation_Kit
source ~/cp10-pilot-venv/bin/activate
python src/test_phase7k.py
```

Tests use previous files in Windows Downloads when present (16 pass when all development fixtures available); otherwise 11 fixture-dependent tests may skip while 5 always-run tests execute.

## Step 1 — original-checkpoint and gate hashes

```bash
CHECKPOINT_DIR="/mnt/e/AIUB/Capstone Project/Journal Based Results/Main Result/transformer_results"
python src/phase7k.py verify-lock --checkpoint-dir "$CHECKPOINT_DIR"
```

**Stop if any hash mismatch.** Never use R9 ablation or `Q1_Research_Completion_Sprint` checkpoints as an automatic replacement.

## Step 2 — limited species probe, not full confirmation

```bash
python src/phase7k.py screen \
 --sequence DJI_0161_video1 \
 --output ./results/DJI_0161_species_preview.csv
```

The script only requests `frame_0.json`, `frame_100.json`, etc. A negative or inaccessible sample is **not** evidence of species absence. The official `DJI_0161_video1/boxid/frame_0.json` had one label `elephant` when inspected on 2026-11-22. `DJI_0204_video1` also had elephant in frame 0 but is higher risk of collection overlap with prior development data.

## Step 3 — get original `boxid` annotation JSON for the *first* candidate

```bash
NAME="DJI_0161_video1"
ROOT="$HOME/CP10_new_data_private/WildLive/Phase7K/candidates"
TARGET="$ROOT/$NAME"
mkdir -p "$TARGET/boxid"
wget -r -l 3 -np -nd -A '*.json' -P "$TARGET/boxid" \
  "https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/$NAME/boxid/"
find "$TARGET/boxid" -type f -name 'frame_*.json' | wc -l
(cd "$TARGET" && zip -q -r "$ROOT/${NAME}_boxid.zip" boxid)
unzip -t "$ROOT/${NAME}_boxid.zip" | tail -n 2

python src/phase7k.py intake \
  --zip "$ROOT/${NAME}_boxid.zip" \
  --sequence "$NAME" \
  --output ./results/${NAME}_intake
```

If zero JSONs arrive, inspect wget listing/report rather than concluding the sequence is absent. Full result may have many missing frames/tracks; intake only reports eligibility.

## Step 4 — minimal visual QA before any inference

```bash
mkdir -p "$TARGET/visual_qa_frames"
for n in 0 100 200 400 600; do
  wget -q -c -P "$TARGET/visual_qa_frames" \
    "https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/$NAME/frames/frame_${n}.jpg" \
    || echo "Frame $n not available"
done
(cd "$TARGET" && zip -q -r "$ROOT/${NAME}_visual_qa_frames.zip" visual_qa_frames)
cp "$ROOT/${NAME}_boxid.zip" "$ROOT/${NAME}_visual_qa_frames.zip" /mnt/c/Users/HP/Downloads/
explorer.exe /mnt/c/Users/HP/Downloads/
```

Upload annotation and sample-frame ZIPs for identity/bbox QA. Verify the footage is an eligible distinct elephant collection event **before making any independent-confirmation claim**.

## Step 5 — locked frozen evaluation (ONLY after visual QA)

```bash
python src/phase7k.py evaluate \
  --zip "$ROOT/${NAME}_boxid.zip" \
  --sequence "$NAME" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output ./results/${NAME}_locked
```

The released `phase7k.py` checks every pinned SHA (2 gate JSONs, checkpoint/source code + 6 `.pt` files), applies observation-only selectors and computes per-window and per-track results. New sequence remains **independence unverified** by default. For original development sequences, this command deliberately aborts unless a special local regression flag is used (the flag DOES NOT create new validation).

To send outputs:

```bash
(cd results && zip -q -r "$ROOT/${NAME}_Phase7K_Results.zip" "${NAME}_locked")
cp "$ROOT/${NAME}_Phase7K_Results.zip" /mnt/c/Users/HP/Downloads/
```

## Step 6 — multi-sequence descriptive aggregation (when additional candidates exist)

```bash
python src/phase7k.py aggregate \
  --inputs ./results/SEQUENCE_A_locked ./results/SEQUENCE_B_locked \
  --output ./results/multi_recording_descriptive
```

Replace placeholders with actual completed candidate folders. This is **descriptive sequence aggregation**, never proof of independent capture-group count or a confirmatory significance test.

## Data protection

Keep raw frames, annotation ZIPs, model checkpoint binaries and sensitive wildlife geolocations private, obey official noncommercial licence; `prospective_lock.json` and code may be reviewed separately with supervisor. Do not silently turn a test-set review into method development.
