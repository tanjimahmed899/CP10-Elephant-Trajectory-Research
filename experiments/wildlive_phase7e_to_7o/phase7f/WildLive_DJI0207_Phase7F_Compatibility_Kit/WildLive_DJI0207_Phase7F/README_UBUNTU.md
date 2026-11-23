# Phase 7F — run on user's Ubuntu 24.04 WSL

### 1. Download the zip from ChatGPT into `C:\Users\HP\Downloads`

```bash
mkdir -p ~/CP10_new_data_private/WildLive/Phase7F
unzip -o /mnt/c/Users/HP/Downloads/WildLive_DJI0207_Phase7F_Compatibility_Kit.zip \
  -d ~/CP10_new_data_private/WildLive/Phase7F
cd ~/CP10_new_data_private/WildLive/Phase7F/WildLive_DJI0207_Phase7F
source ~/cp10-pilot-venv/bin/activate
```

### 2. Inspect and reproduce the compatibility preflight (NumPy only)

```bash
export PHASE7F_TEST_NPZ="$HOME/CP10_new_data_private/WildLive/Phase7E/WildLive_DJI0207_Phase7E/local_recomputed_outputs/frame_based_windows_and_baselines.npz"
python src/test_phase7f.py
python src/phase7f_preflight_and_inference.py \
  --phase7e-npz "$PHASE7F_TEST_NPZ" \
  --output ./local_preflight
cat ./local_preflight/manifest.json
```

`BLOCKED_MISSING_CHECKPOINTS` is the expected status until actual trained `.pt` files are found; numerical canonicalization, QA and three nonlearned baselines still run. The architecture test is skipped if PyTorch is not installed.

### 3. Search for genuine original checkpoints (do not recreate or retrain)

```bash
find /mnt/c/Users/HP/Downloads /mnt/c/Users/HP/Documents "$HOME" \
  -type f \( -name '2s_to_2s_Transformer_seed*.pt' -o -name '2s_to_5s_Transformer_seed*.pt' \) \
  -print 2>/dev/null
```

Expected six filenames (must all be original/unchanged):

```
2s_to_2s_Transformer_seed42.pt
2s_to_2s_Transformer_seed52.pt
2s_to_2s_Transformer_seed62.pt
2s_to_5s_Transformer_seed42.pt
2s_to_5s_Transformer_seed52.pt
2s_to_5s_Transformer_seed62.pt
```

The old Windows training code's default output folder was `C:\Users\HP\Downloads\transformer_results`. If those exact files exist there, use `/mnt/c/Users/HP/Downloads/transformer_results` as the `--checkpoint-dir` without copying them.

### 4. Only after all six originals are present: frozen Transformer evaluation

```bash
python -c 'import torch; print(torch.__version__)'
```

If `torch` is missing, install CPU PyTorch into your activated virtualenv using the official PyTorch instructions, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then run:

```bash
python src/phase7f_preflight_and_inference.py \
  --phase7e-npz "$PHASE7F_TEST_NPZ" \
  --checkpoint-dir /mnt/c/Users/HP/Downloads/transformer_results \
  --output ./local_frozen_comparison
```

Output includes per-window metrics, same-window all-model comparisons, frozen checkpoint SHA-256, and optional predictions. **These are exploratory frame-only results, not seconds-based results.** Never upload original model weights or licensed WildLive data to a public GitHub repo without confirming permissions.

### 5. Copy private results to Windows Downloads when desired

```bash
cp -r ./local_preflight /mnt/c/Users/HP/Downloads/Phase7F_local_preflight
explorer.exe /mnt/c/Users/HP/Downloads/
```
