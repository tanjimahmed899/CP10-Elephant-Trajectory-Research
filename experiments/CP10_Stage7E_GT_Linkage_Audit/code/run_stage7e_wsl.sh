#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/cp10_detection_venv/bin/activate"
DATA="/mnt/c/Users/HP/Downloads/CP10_Stage7D_Tracker_Review.zip"
ROOT="/mnt/e/AIUB/Capstone Project/Journal Based Results"
OUT="$HOME/CP10_Stage7E_GT_Linkage_Results"
ZIP="$HOME/CP10_Stage7E_GT_Linkage_Review.zip"
mkdir -p "$OUT"
if [ ! -f "$VENV" ]; then
  echo "ERROR: detector virtual environment is missing: $VENV" | tee "$OUT/Stage7E_Console.log"
  RC=2
elif [ ! -f "$DATA" ]; then
  echo "ERROR: Stage7D results ZIP not found in Windows Downloads: $DATA" | tee "$OUT/Stage7E_Console.log"
  RC=2
else
  source "$VENV"
  python3 -u "$SCRIPT_DIR/stage7e_gt_linkage.py" \
    --stage7d "$DATA" \
    --gt "$ROOT/Main Result/trajectory_dataset/cleaned_track_points.csv" \
    --out "$OUT" 2>&1 | tee "$OUT/Stage7E_Console.log"
  RC=${PIPESTATUS[0]}
fi
# Package logs as well as successful artifacts. Do not overwrite original research inputs.
python3 - "$OUT" "$ZIP" <<'PY'
import sys,zipfile
from pathlib import Path
out=Path(sys.argv[1]);p=Path(sys.argv[2]);
with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(out.glob('*')):
  if f.is_file():z.write(f,f.name)
print('REVIEW ZIP:',p)
PY
cp "$ZIP" /mnt/c/Users/HP/Downloads/ || echo "Could not copy ZIP to Downloads; WSL ZIP still exists"
echo "Stage7E exit code: $RC"
exit "$RC"
