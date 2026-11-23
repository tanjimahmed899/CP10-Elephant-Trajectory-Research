# Reproduce independent numerical-output verification (Ubuntu WSL)

Requires Python 3 with numpy, pandas and the Phase 7K kit dependencies installed in your `~/cp10-pilot-venv`.

Place these three inputs where the script can access them:

- `Phase7L_DJI0204_video1_Exploratory_Results.zip`
- `DJI_0204_video1_boxid.zip`
- The original **extracted** `WildLive_Phase7K_Prospective_Validation_Kit` directory, containing `prospective_lock.json`, `src/phase7k.py`, `locked_models/*.json`.

Example, adapt `KIT` if your home directory differs:

```bash
source ~/cp10-pilot-venv/bin/activate
KIT="$HOME/CP10_new_data_private/WildLive/Phase7K/WildLive_Phase7K_Prospective_Validation_Kit"
python verify_phase7l.py \
  --evaluation /mnt/c/Users/HP/Downloads/Phase7L_DJI0204_video1_Exploratory_Results.zip \
  --annotations /mnt/c/Users/HP/Downloads/DJI_0204_video1_boxid.zip \
  --kit "$KIT" \
  --out ./independently_reproduced
```

Expected output: a `PASS` status, 2,748 unique windows and 2,748 matching regenerated gate decisions. The script does not re-run neural inference because the exported results ZIP contains neither the checkpoint binaries nor forecast coordinate arrays.

This verification tests integrity, consistent metrics, frozen gate choices, and reported lock manifest. It does not certify independent flights, physical motion, annotation accuracy outside the earlier sparse image QA, or confirmation-level statistical evidence.
