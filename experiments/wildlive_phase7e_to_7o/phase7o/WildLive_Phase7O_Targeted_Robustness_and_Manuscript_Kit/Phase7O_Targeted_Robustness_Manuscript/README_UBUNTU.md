# Phase 7O — Ubuntu WSL reproduction

The released ZIP includes code, output CSVs and figures, the main-manuscript supervisor draft, SHA-256 lock and report. **It excludes original WildLive images, annotation ZIPs and checkpoint binaries**.

1. Copy `WildLive_Phase7O_Targeted_Robustness_and_Manuscript_Kit.zip` to Windows Downloads; extract in WSL:

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7O
unzip -o /mnt/c/Users/HP/Downloads/WildLive_Phase7O_Targeted_Robustness_and_Manuscript_Kit.zip \
  -d ~/CP10_new_data_private/WildLive/Phase7O
cd ~/CP10_new_data_private/WildLive/Phase7O/Phase7O_Targeted_Robustness_Manuscript
source ~/cp10-pilot-venv/bin/activate
python -m pip install numpy pandas matplotlib opencv-python-headless
python src/test_phase7o.py
```

2. To fully recompute the image results, place **these exact original archived files** in one private directory (for example `$HOME/CP10_new_data_private/WildLive/Phase7O/source_zips`):

```
WildLive_DJI0207_priority_frames.zip
WildLive_DJI_0207_boxid.zip
Phase7F_Frozen_Results.zip
DJI_0204_video2_visual_QA_frames.zip
DJI_0204_video2_boxid.zip
Phase7H_DJI0204_Frozen_Evaluation.zip
```

Original source hashes are required; the program will fail safely on replacement or corruption. To inspect which are present:

```bash
SOURCE="$HOME/CP10_new_data_private/WildLive/Phase7O/source_zips"
mkdir -p "$SOURCE"
ls -lh "$SOURCE"
```

3. When all six original ZIPs are present, run:

```bash
python src/phase7o.py --input-dir "$SOURCE" --output ./my_recomputed_results
```

4. Save the results:

```bash
zip -qr "$HOME/Phase7O_Reproduced_Results.zip" my_recomputed_results
cp "$HOME/Phase7O_Reproduced_Results.zip" /mnt/c/Users/HP/Downloads/
explorer.exe /mnt/c/Users/HP/Downloads/
```

No source re-download is necessary if exact originals still exist in your WSL or Windows directories. If not, re-download only the necessary original archives, and compare their hashes to `protocol/INPUT_SHA256_LOCK.json` before treating them as the same experimental inputs. **Do not change the model checkpoints, adaptive gate, or thresholds to improve this retrospective result.**

Review `manuscript/Final_Demo_paper_Phase7O_integrated.docx` as a supervisor draft. The existing manuscript's other chapters and historical citations were not newly audited here.
