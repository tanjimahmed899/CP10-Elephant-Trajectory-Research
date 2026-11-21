#!/usr/bin/env bash
# Start from any working directory. No new training or detector inference.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/CP10_Stage7D_Offline_Tracker_Results"
mkdir -p "$OUT"
LOG="$OUT/Stage7D_Console.log"
# Explicitly use the same detector environment as successful Stage7C.
if [[ -f "$HOME/cp10_detection_venv/bin/activate" ]]; then
    source "$HOME/cp10_detection_venv/bin/activate"
fi
SOURCE_ZIP="/mnt/c/Users/HP/Downloads/CP10_Stage7C_TestReal_Review.zip"
if [[ ! -f "$SOURCE_ZIP" && -f "$HOME/CP10_Stage7C_TestReal_Review.zip" ]]; then
    SOURCE_ZIP="$HOME/CP10_Stage7C_TestReal_Review.zip"
fi
YAML_ARGS=()
if [[ -n "${CP10_R13_YAML:-}" ]]; then
    YAML_ARGS=(--yaml "$CP10_R13_YAML")
fi
set +e
python3 -u "$SCRIPT_DIR/stage7d_offline_replay.py" --input "$SOURCE_ZIP" --out "$OUT" "${YAML_ARGS[@]}" 2>&1 | tee "$LOG"
RESULT=${PIPESTATUS[0]}
set -e
export STAGE7D_OUT="$OUT"
python3 - <<'PY'
import os,zipfile
from pathlib import Path
p=Path(os.environ['STAGE7D_OUT']);zipp=Path.home()/'CP10_Stage7D_Tracker_Review.zip'
with zipfile.ZipFile(zipp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(p.iterdir()):
  if f.is_file():z.write(f,arcname=f.name)
print('Review package:',zipp)
PY
if [[ -d /mnt/c/Users/HP/Downloads ]]; then
 cp "$HOME/CP10_Stage7D_Tracker_Review.zip" /mnt/c/Users/HP/Downloads/
 echo 'Copied results to C:\Users\HP\Downloads\CP10_Stage7D_Tracker_Review.zip'
fi
if [[ "$RESULT" -ne 0 ]];then
 echo "Stage7D tracking blocked/failed, exit=$RESULT. Upload CP10_Stage7D_Tracker_Review.zip; it contains diagnostic log." >&2
fi
exit "$RESULT"
