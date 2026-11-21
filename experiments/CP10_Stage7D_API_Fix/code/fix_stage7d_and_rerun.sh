#!/usr/bin/env bash
# Stage 7D API compatibility repair, no tuning/training/inference
set -uo pipefail
ROOT="$HOME/cp10_stage7d_tools/CP10_Stage7D_Offline_Tracker_Replay"
SCRIPT="$ROOT/stage7d_offline_replay.py"
RUNNER="$ROOT/run_stage7d_wsl.sh"
if [[ ! -f "$SCRIPT" || ! -f "$RUNNER" ]]; then
  echo "ERROR: original Stage7D extracted scripts not found under $ROOT" >&2
  exit 2
fi
python3 - "$SCRIPT" <<'PY'
import hashlib
from pathlib import Path
import py_compile
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = 'tracker=BYTETracker(args=SimpleNamespace(**cfg),frame_rate=30)'
new = 'tracker=BYTETracker(args=SimpleNamespace(**cfg))'
if old in s:
    if s.count(old) != 1:
        sys.exit('ERROR: Expected exactly one old API call; no change made.')
    backup = p.with_name('stage7d_offline_replay.before_API_fix.py')
    if not backup.exists():
        backup.write_text(s)
    newtext = s.replace(old, new, 1)
    p.write_text(newtext)
    print('PATCHED:', p)
elif new in s:
    print('ALREADY PATCHED:', p)
else:
    sys.exit('ERROR: Unexpected script contents. No change made.')
py_compile.compile(str(p), doraise=True)
print('Syntax OK; patched script SHA256:', hashlib.sha256(p.read_bytes()).hexdigest())
PY
status=$?
if [[ $status -ne 0 ]]; then exit $status; fi
# Verify the actual active tracker API under the existing detector environment.
VENV_PY="$HOME/cp10_detection_venv/bin/python3"
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: Detector Python environment missing: $VENV_PY" >&2
  exit 2
fi
"$VENV_PY" - <<'PY'
import inspect
import ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker
print('Ultralytics version:', ultralytics.__version__)
print('BYTETracker signature:', inspect.signature(BYTETracker))
if ultralytics.__version__ != '8.4.216':
    raise SystemExit('ERROR: Runner was tested against 8.4.216; version drift detected. Do not silently change tracker versions.')
s = inspect.signature(BYTETracker)
if 'args' not in s.parameters or 'frame_rate' in s.parameters:
    raise SystemExit('ERROR: Different tracker signature found; stop for audit.')
print('API compatibility check passed')
PY
status=$?
if [[ $status -ne 0 ]]; then exit $status; fi
# Move the previous traceback out of the result directory, so a new successful ZIP
# cannot accidentally contain stale failure evidence.
OLD_FAIL="$HOME/CP10_Stage7D_Offline_Tracker_Results/Stage7D_Failure.txt"
if [[ -f "$OLD_FAIL" ]]; then
  cp "$OLD_FAIL" "$HOME/CP10_Stage7D_Previous_API_Failure.txt"
  rm "$OLD_FAIL"
fi
set +e
bash "$RUNNER" 2>&1 | tee "$HOME/CP10_Stage7D_After_API_Fix.log"
rc=${PIPESTATUS[0]}
set -e
if [[ -f "$HOME/CP10_Stage7D_Tracker_Review.zip" ]]; then
  echo "RESULT: $HOME/CP10_Stage7D_Tracker_Review.zip"
  echo 'WINDOWS: C:\Users\HP\Downloads\CP10_Stage7D_Tracker_Review.zip'
fi
if [[ "$rc" -ne 0 ]]; then
  echo 'WARNING: Tracker replay still failed. Upload the NEW review ZIP, which contains the latest log.' >&2
fi
exit "$rc"
