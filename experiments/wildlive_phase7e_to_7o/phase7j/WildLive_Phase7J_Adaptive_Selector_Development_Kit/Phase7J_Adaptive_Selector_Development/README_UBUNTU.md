# Phase 7J — run on Ubuntu 24.04 / WSL

1. Download `WildLive_Phase7J_Adaptive_Selector_Development_Kit.zip` into `C:\Users\HP\Downloads`.
2. Run:

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7J
unzip -o /mnt/c/Users/HP/Downloads/WildLive_Phase7J_Adaptive_Selector_Development_Kit.zip \
  -d ~/CP10_new_data_private/WildLive/Phase7J
cd ~/CP10_new_data_private/WildLive/Phase7J/Phase7J_Adaptive_Selector_Development
source ~/cp10-pilot-venv/bin/activate
python -c 'import numpy, pandas, matplotlib; print("Dependencies OK")'
python src/test_phase7j.py
python src/phase7j.py develop \
  --phase7i /mnt/c/Users/HP/Downloads/WildLive_Phase7I_Reproduced_Results.zip \
  --output ./local_phase7j_results
```

The `--phase7i` file is the ZIP you previously reproduced on your machine. Do not use the Phase 7I kit ZIP as a substitute for the reproduced-results ZIP without checking it holds the matching data.

3. Compare your output:

```bash
cat ./local_phase7j_results/blocked_stress_summary.csv
sha256sum /mnt/c/Users/HP/Downloads/WildLive_Phase7I_Reproduced_Results.zip
```

4. Copy to Windows Downloads and (optionally) upload here:

```bash
cd ~/CP10_new_data_private/WildLive/Phase7J/Phase7J_Adaptive_Selector_Development
zip -q -r ~/WildLive_Phase7J_Reproduced_Results.zip local_phase7j_results
cp ~/WildLive_Phase7J_Reproduced_Results.zip /mnt/c/Users/HP/Downloads/
explorer.exe /mnt/c/Users/HP/Downloads/
```

5. Applying *one* refitted development-only gate to a feature CSV (all features must be computed using only the 20 observed positions):

```bash
python src/phase7j.py predict \
  --model ./local_phase7j_results/locked_models/phase7j_dev_gate_h20.json \
  --features ./examples/example_observation_features_h20.csv \
  --output ./local_phase7j_results/example_gate_choices_h20.csv
```

The `predict` step outputs selected model **names only**, not future coordinates, and **not** evaluation ADE/FDE. The example CSV derives from already-inspected development footage and has **no** future error labels.

## Strict scientific interpretation

The gate is a research prototype. It was retrospectively stressed on two already-inspected WildLive folders and *did not outperform the strongest individual expert in any of four tests*. The two JSONs are fitted on both development folders and have no untouched validation. Native FPS, camera correction, and independent capture-event identity remain unknown. No seconds-based or field-warning claim is authorized.
