# CP 1.0 — Phase 7M-A (No-author alternative) Ubuntu WSL steps

This package needs no dataset-author email. It checks **shared-scene evidence**, not verified flight independence.

## 1. Install and test

Copy `Phase7M_Authorless_Provenance_Protocol_Kit.zip` to Windows Downloads, then:

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7M_A
unzip -o /mnt/c/Users/HP/Downloads/Phase7M_Authorless_Provenance_Protocol_Kit.zip \
  -d ~/CP10_new_data_private/WildLive/Phase7M_A
cd ~/CP10_new_data_private/WildLive/Phase7M_A/Phase7M_Authorless_Provenance_Protocol
source ~/cp10-pilot-venv/bin/activate
pip install numpy opencv-python Pillow
python src/test_authorless.py
```

## 2. Optional: recompute the 223 visual comparisons

Copy the four original **already downloaded** visual-QA ZIPs to a private directory (from prior WildLive candidate folders or Windows Downloads). This released package does not redistribute original JPEGs.

Expected exact filenames:

```
WildLive_DJI0207_priority_frames.zip
DJI_0204_video2_visual_QA_frames.zip
DJI_0204_video1_visual_qa_frames.zip
DJI_0161_video1_visual_qa_frames.zip
```

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7M_A/source_images
INPUT="$HOME/CP10_new_data_private/WildLive/Phase7M_A/source_images"
for NAME in WildLive_DJI0207_priority_frames.zip DJI_0204_video2_visual_QA_frames.zip \
            DJI_0204_video1_visual_qa_frames.zip DJI_0161_video1_visual_qa_frames.zip; do
  if [ -f "/mnt/c/Users/HP/Downloads/$NAME" ]; then
    cp "/mnt/c/Users/HP/Downloads/$NAME" "$INPUT/"
  else
    echo "Find this existing ZIP in your earlier private project folders: $NAME"
  fi
done
python src/authorless_provenance.py audit --inputs-dir "$INPUT" --output ./local_results
```

If a ZIP isn't present, copy it from its original private folder first. The full audit expects **25 images, 223 cross-recording image pairs, 24 strong shared-scene pairs, 0 confirmed independent flights** (exact inlier counts may vary by OpenCV build).

## 3. Optional: census remaining WildLive folders without downloading frames/videos

The included networked helper downloads *only* public JSON `boxid` annotations to your private Linux directory. It deliberately skips four already-full elephant recordings and known full-zebra `DJI_0601_video2`. **This operation has not run in this chat** and can involve many downloads; use it only when you want to search for additional elephant-positive clips.

```bash
bash ./phase7m_download_json_census.sh
```

Then inspect:

```bash
ROOT="$HOME/CP10_new_data_private/WildLive/Phase7M/authorless_json_census"
cat "$ROOT/download_status.csv"
cat "$ROOT/new_recording_species_census.csv"
cp "$ROOT/new_recording_species_census.csv" /mnt/c/Users/HP/Downloads/
explorer.exe /mnt/c/Users/HP/Downloads/
```

Upload only `new_recording_species_census.csv` if you want help identifying next elephant recording candidates; do **not** upload original wildlife media or coordinates unnecessarily.

## Ethics / interpretation

Do not claim each video comes from an independent flight, do not convert frames to seconds without verified capture FPS, and do not run a per-window significance test on highly overlapping samples. Original Phase 7K gates and checkpoints must remain frozen.
