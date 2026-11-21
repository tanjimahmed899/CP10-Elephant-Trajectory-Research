#!/usr/bin/env bash
# Read-only checks to locate original matched R8 arrays and fresh WildBox sources.
set -euo pipefail
ROOT="/mnt/e/AIUB/Capstone Project/Journal Based Results"
OUT="$HOME/CP10_Stage3_Input_Locator.txt"
{
 echo "=== R8 candidate files (.npy/.npz/.csv/.json) ==="
 for d in "$ROOT/Main Result/ego_motion_full" "$ROOT/Main Result/corrected_trajectory_dataset" "$ROOT/Main Result/final_neural_trajectory_dataset" "$ROOT/Main Result 2.0/step2b_raw_vs_corrected_baseline" "$ROOT/Main Result 2.0/step2_ego_motion_validation"; do
  if [ -d "$d" ]; then
    echo "--- $d"; find "$d" -type f \( -iname '*.npy' -o -iname '*.npz' -o -iname '*.csv' -o -iname '*camera*.json' \) -printf '%s bytes | %p\n' | head -n 70 || true
  fi
 done
 echo "=== Exact EXP3 source filenames in Journal Based Results ==="
 for fragment in 20240119151048_0001 20240119151437_0002 20240626143248_0002 20240626143443_0003 20260227082707_0003 20260227084724_0003; do
  echo "--- $fragment"; find "$ROOT" -iname "*$fragment*" -printf '%p\n' | head -n 10 || true
 done
} > "$OUT"
echo "Created $OUT"; echo "Open with: explorer.exe $(wslpath -w "$OUT")"
