#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HOME/cp10_detection_venv/bin/activate" ]]; then
  source "$HOME/cp10_detection_venv/bin/activate"
fi
ROOT="/mnt/e/AIUB/Capstone Project/Journal Based Results"
OUT="$HOME/CP10_Stage7C_TestReal_Results"
LOG="$HOME/CP10_Stage7C_Run.log"
DEST="$HOME/CP10_Stage7C_TestReal_Review.zip"
python3 -u "$HERE/evaluate_testreal_locked.py" --root "$ROOT" --out "$OUT" --device cpu 2>&1 | tee "$LOG"
CODE=${PIPESTATUS[0]}
python3 "$HERE/pack_stage7c.py" --out "$OUT" --dest "$DEST"
PACK=$?
echo "Inference exit code: $CODE | Pack exit code: $PACK"
echo "Review ZIP: $DEST"
if [[ -d /mnt/c/Users/HP/Downloads && -f "$DEST" ]]; then
  cp -f "$DEST" "/mnt/c/Users/HP/Downloads/CP10_Stage7C_TestReal_Review.zip"
  echo "Copied to Windows Downloads"
fi
if [[ $CODE -ne 0 ]]; then exit "$CODE";fi
exit "$PACK"
