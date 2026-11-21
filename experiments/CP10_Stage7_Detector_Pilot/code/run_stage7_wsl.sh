#!/usr/bin/env bash
set -euo pipefail
ROOT="${CP10_ROOT:-/mnt/e/AIUB/Capstone Project/Journal Based Results}"
OUT="${CP10_STAGE7_OUT:-$HOME/CP10_Stage7_Dev_Pilot}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
python3 "$HERE/stage7_preflight_wsl.py" --root "$ROOT" --out "$OUT"
STATUS="$(python3 - "$OUT" <<'PY'
import json,sys,pathlib
print(json.loads((pathlib.Path(sys.argv[1])/'Stage7_Preflight_Manifest.json').read_text())['status'])
PY
)"
if [ "$STATUS" != 'READY_TRAINREAL_DEV_ONLY' ]; then
  echo 'Stage 7 model inference NOT run: TrainReal frames or original checkpoint unavailable.'
  python3 "$HERE/stage7_pack_results.py" --out "$OUT"
  exit 0
fi
if ! python3 -c 'import ultralytics, torch, numpy' >/dev/null 2>&1; then
  echo 'Stage 7 ready, but ultralytics/PyTorch/numpy missing from current Python environment.'
  echo 'Activate your existing cp10_detection_venv. Do not install without checking current env.'
  python3 "$HERE/stage7_pack_results.py" --out "$OUT"
  exit 0
fi
python3 -u "$HERE/stage7_run_dev_inference.py" --out "$OUT" --device "${CP10_STAGE7_DEVICE:-cpu}"
python3 "$HERE/stage7_score_dev.py" --out "$OUT"
python3 "$HERE/stage7_pack_results.py" --out "$OUT"
