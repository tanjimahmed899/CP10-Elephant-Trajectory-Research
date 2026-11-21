#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/e/AIUB/Capstone Project/Journal Based Results"
OUT="$HOME/CP10_Stage7B_TrainReal_Adaptation"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/cp10_detection_venv/bin/python3"
if [[ ! -x "$PY" ]]; then echo 'Missing ~/cp10_detection_venv Python';exit 2;fi
"$PY" -c 'import torch,ultralytics,numpy,PIL; print("Dependencies OK",torch.__version__,ultralytics.__version__)'
mkdir -p "$OUT"
"$PY" "$DIR/stage7b_prepare.py" --root "$ROOT" --out "$OUT" 2>&1 | tee "$OUT/CP10_Stage7B_Prepare.log"
"$PY" -u "$DIR/stage7b_train_validate.py" --out "$OUT" --device cpu 2>&1 | tee "$OUT/CP10_Stage7B_Train.log"
"$PY" "$DIR/stage7b_pack.py" --out "$OUT" --include-best-pt
if [[ -d /mnt/c/Users/HP/Downloads ]]; then cp "$HOME/CP10_Stage7B_TrainReal_Review.zip" /mnt/c/Users/HP/Downloads/;fi
printf '\nResults: %s\n' "$HOME/CP10_Stage7B_TrainReal_Review.zip"
