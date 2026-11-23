# Phase 7L — Multi-recording species screen and capture-event evidence

**Scientific scope:** Expand the elephant dataset without changing the released 7J gate, Phase7K lock, primary ADE comparator, checkpoints or evaluation windows. The new scripts perform metadata/candidate tracking, source-hash verification and numerical audit. They do **not** provide new trained forecasting models or any new unseen recording scores.

## 1 — Install this Phase 7L kit

1. Place `WildLive_Phase7L_Capture_Group_Expansion_Kit.zip` in `C:\Users\HP\Downloads`.
2. In Ubuntu WSL run:

```bash
mkdir -p "$HOME/CP10_new_data_private/WildLive/Phase7L"
unzip -o "/mnt/c/Users/HP/Downloads/WildLive_Phase7L_Capture_Group_Expansion_Kit.zip" \
  -d "$HOME/CP10_new_data_private/WildLive/Phase7L"
cd "$HOME/CP10_new_data_private/WildLive/Phase7L/WildLive_Phase7L_Capture_Group_Expansion_Kit"
source "$HOME/cp10-pilot-venv/bin/activate"
python -m pip install numpy pandas
python src/test_phase7l.py
python src/phase7l.py verify-lock
python src/phase7l.py register --output ./results/my_candidate_register.csv
```

Fixture-based tests should pass when the archived Phase7K zip is present at `/mnt/data/` in the authoring environment. On **your** WSL laptop, fixture-dependent tests are SKIPPED unless you set:

```bash
export PHASE7L_FIXTURE="/mnt/c/Users/HP/Downloads/Phase7K_DJI0161_Locked_Evaluation.zip"
export PHASE7L_ANN="/mnt/c/Users/HP/Downloads/DJI_0161_video1_boxid.zip"
python src/test_phase7l.py
```

The gate and checkpoint weights remain in **the original Phase7K kit**; do not copy substituted/reproduced 7J gates into it.

## 2 — Label-only screening first (minimum bandwidth)

```bash
KIT7K="$HOME/CP10_new_data_private/WildLive/Phase7K/WildLive_Phase7K_Prospective_Validation_Kit"
bash ./phase7l_screen_selected.sh "$KIT7K"
cat ./phase7l_screening/selected_candidate_species_samples.csv
cp ./phase7l_screening/selected_candidate_species_samples.csv /mnt/c/Users/HP/Downloads/
```

This probes official per-frame `boxid` JSON at sample frames 0, 100, 200, 400, 600 for **six** candidate folders, without inspecting prediction errors, and writes a CSV. A hit means *elephant present in samples*, not track continuity or independent origin. A nonhit means only *none of the inspected readable frames contained elephant* and does not prove species absence. `readable_samples=0` means network/download unresolved; do not classify as zebra/elephant. `DJI_0142_video1` is a practical first screen, not an elephant-verified recommendation.

## 3 — Download a confirmed elephant-positive candidate

Choose **one** from the new *sample-positive* CSV, preferring a filename family not previously examined (not `DJI_0204_video1` if an alternative exists). For example, **only if** the CSV actually shows elephant for `DJI_0142_video1`:

```bash
bash ./phase7l_download_candidate.sh DJI_0142_video1
```

The script gets whole `boxid` annotations plus five sparse original JPEG samples and creates two ZIPs at `C:\Users\HP\Downloads`. Upload these two files to this conversation:

- `<NAME>_boxid.zip`
- `<NAME>_visual_qa_frames.zip`

**Do not run locked evaluation until full species/track continuity and sampled visual QA have been checked.** The script makes no independence claim.

## 4 — Evaluate a QA-approved candidate with ORIGINAL Phase 7K kit

After annotation and visual QA:

```bash
KIT7K="$HOME/CP10_new_data_private/WildLive/Phase7K/WildLive_Phase7K_Prospective_Validation_Kit"
NAME="YOUR_APPROVED_NEW_NAME"
CK="/mnt/e/AIUB/Capstone Project/Journal Based Results/Main Result/transformer_results"
ZIP="$HOME/CP10_new_data_private/WildLive/Phase7L/candidates/$NAME"_boxid.zip

cd "$KIT7K"
python src/phase7k.py verify-lock --checkpoint-dir "$CK"
python src/phase7k.py intake --zip "$ZIP" --sequence "$NAME" --output "./results/${NAME}_intake"
python src/phase7k.py evaluate --zip "$ZIP" --sequence "$NAME" \
  --checkpoint-dir "$CK" --output "./results/${NAME}_locked"
```

The above result is descriptive until capture-event provenance is established. No tuning on this result is permitted. Upload the resulting evaluation ZIP for independent auditing.

## 5 — Independently audit the output ZIP

From the Phase7L directory:

```bash
python src/phase7l.py audit-eval \
  --archive /mnt/c/Users/HP/Downloads/Phase7K_DJI0161_Locked_Evaluation.zip \
  --source-zip /mnt/c/Users/HP/Downloads/DJI_0161_video1_boxid.zip \
  --output ./results/reverified_DJI0161
```

For **additional fresh candidate** archives, repeat with `--archive <fresh_evaluation.zip> --source-zip <fresh_annotations.zip>`. You may pass multiple `--archive` arguments, but omit `--source-zip` when aggregating multiple distinct archives. Group-independence remains **unverified** regardless of number of archives.

## 6 — Capture-event evidence

`protocol/capture_event_provenance_template.csv` has fields for: original video/flight log; camera or flight identifier; capture datetime with an authoritative source; visual/temporal overlap review against development footage; evidence location; reviewer; review date; assessment. Fill only from authentic source records. Do not infer capture time from JPEG **file modification time**.

```bash
python src/phase7l.py provenance-review \
  --input ./protocol/capture_event_provenance_template.csv \
  --output ./results/provenance_review_status.csv
```

Even fully filled rows need **human source-verification**; this tool does not automatically certify independence. Related filename-family flags are screening heuristics only. Supervisor-approved documented grouping is necessary before capture-group inferential analysis. FPS and geographic scale still unavailable from this evidence.

## Data licence and ethics

WildLive DOI: <https://doi.org/10.5523/bris.ewnuoroebuae20vei0cu6zu69>. The portal indicates a **Non-Commercial Government Licence**. Avoid publishing raw images/annotations, checkpoints, or precise endangered-animal locations without evaluating licence permissions and conservation implications. Citation is required by the dataset README. Avoid claims of seconds/metres, independent flights or safety utility without the requisite evidence.
