# Phase 7M — Ubuntu WSL reproduction

This package does **not** retrain or alter the original Phase 7K gates/checkpoints. `phase7m.py` requires Python 3.10+, `numpy`, `opencv-python` and `Pillow` to reproduce image-content matching; `unittest` and basic annotation checks use the standard library. No networking is required for the audit.

## 1 — Unzip and check code/protocol

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7M
unzip -o \
 /mnt/c/Users/HP/Downloads/WildLive_Phase7M_Capture_Provenance_Audit_Kit.zip \
 -d ~/CP10_new_data_private/WildLive/Phase7M

cd ~/CP10_new_data_private/WildLive/Phase7M/WildLive_Phase7M_Capture_Provenance_Audit
source ~/cp10-pilot-venv/bin/activate
pip install numpy opencv-python Pillow
python src/test_phase7m.py
python src/phase7m.py verify-lock
```

Optional: compare the still-installed Phase 7K release against its exact original 8 file hashes:

```bash
python src/phase7m.py verify-release \
 --phase7k-kit "$HOME/CP10_new_data_private/WildLive/Phase7K/WildLive_Phase7K_Prospective_Validation_Kit"
```

## 2 — Supply local source ZIPs

The audit does not download the original imagery. Place these **exact existing files** in a private local folder, e.g. `~/CP10_new_data_private/WildLive/Phase7M/inputs`. Some may already exist in your Windows Downloads, and other copies may be under earlier Phase7 candidate folders.

Annotation archives:

```text
WildLive_DJI_0207_boxid.zip
DJI_0204_video2_boxid.zip
DJI_0204_video1_boxid.zip
DJI_0161_video1_boxid.zip
```

Sparse RGB sample archives:

```text
WildLive_DJI0207_priority_frames.zip
DJI_0204_video2_visual_QA_frames.zip
DJI_0204_video1_visual_qa_frames.zip
DJI_0161_video1_visual_qa_frames.zip
```

Optional prior evaluation archives, verified when present:

```text
Phase7K_DJI0161_Locked_Evaluation.zip
Phase7L_DJI0204_video1_Exploratory_Results.zip
```

Example copy for files **already** in Windows Downloads:

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7M/inputs
INPUTS="$HOME/CP10_new_data_private/WildLive/Phase7M/inputs"
for NAME in \
 WildLive_DJI_0207_boxid.zip \
 DJI_0204_video2_boxid.zip \
 DJI_0204_video1_boxid.zip \
 DJI_0161_video1_boxid.zip \
 WildLive_DJI0207_priority_frames.zip \
 DJI_0204_video2_visual_QA_frames.zip \
 DJI_0204_video1_visual_qa_frames.zip \
 DJI_0161_video1_visual_qa_frames.zip \
 Phase7K_DJI0161_Locked_Evaluation.zip \
 Phase7L_DJI0204_video1_Exploratory_Results.zip; do
  SRC="/mnt/c/Users/HP/Downloads/$NAME"
  if [ -f "$SRC" ]; then cp "$SRC" "$INPUTS/"; else echo "NOT IN DOWNLOADS: $NAME"; fi
done
```

If any required source archive is not in Downloads, copy that archive from the private prior Phase7 folder. Missing originals cannot be reconstructed from the 7M package.

## 3 — Recompute audit

```bash
cd ~/CP10_new_data_private/WildLive/Phase7M/WildLive_Phase7M_Capture_Provenance_Audit
python src/phase7m.py audit \
 --inputs-dir "$HOME/CP10_new_data_private/WildLive/Phase7M/inputs" \
 --output ./local_recomputed
```

For only the 4 annotation ZIPs, when sample JPEGs aren't locally available:

```bash
python src/phase7m.py audit \
 --inputs-dir "$HOME/CP10_new_data_private/WildLive/Phase7M/inputs" \
 --output ./local_annotation_only --annotations-only
```

The full audit should report **4 full sequences; 3,008 annotated frames; 7,230 elephant boxes; 12,948 windows; 25 JPEG samples; 1 strong matching image pair; 0 author-verified independent capture events**. Exact ORB matching counts can change slightly between OpenCV versions.

## 4 — Export small outputs only

```bash
zip -qr "$HOME/Phase7M_Local_Recomputed_Audit.zip" local_recomputed
cp "$HOME/Phase7M_Local_Recomputed_Audit.zip" /mnt/c/Users/HP/Downloads/
explorer.exe /mnt/c/Users/HP/Downloads/
```

Upload **only** the generated audit ZIP for numerical verification; retain the original source JPEGs and annotations privately. The most useful new evidence is an author-provided, anonymized flight/clip lineage table — see `templates/DATASET_AUTHOR_METADATA_REQUEST.md`.

## Research restrictions

- Flight independence is **not** proved by different folder names, different apparent backgrounds, or camera file creation/modification time.
- Frame horizons are not seconds without verified FPS and annotation stride.
- Never adjust the frozen selector or checkpoints based on held-out outcomes.
- Do not publish animal locations, original licensed imagery or raw checkpoints without explicit authorisation and licence review.
