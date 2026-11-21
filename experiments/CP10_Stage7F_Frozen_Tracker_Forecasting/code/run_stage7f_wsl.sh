#!/usr/bin/env bash
# Reads original frozen model/data and *saved* stage7D/E results; no retraining.
set -Eeuo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/mnt/e/AIUB/Capstone Project/Journal Based Results"
WIN="/mnt/c/Users/HP/Downloads"
OUT="$HOME/CP10_Stage7F_Frozen_Tracker_Results"
REVIEW="$HOME/CP10_Stage7F_Tracker_Forecast_Review.zip"
mkdir -p "$OUT"
# Prior review archives must be present; diagnostic errors can be accidentally uploaded as if completed.
for FILE in CP10_Stage7D_Tracker_Review.zip CP10_Stage7E_GT_Linkage_Review.zip; do
 if [[ -f "$WIN/$FILE" ]]; then :
 elif [[ -f "$HOME/$FILE" ]]; then cp -n "$HOME/$FILE" "$WIN/$FILE"
 else
  echo "Missing $FILE. Copy your COMPLETED Stage7D/Stage7E review ZIP to $WIN" | tee -a "$OUT/Stage7F_Console.log"
  python3 "$TOOL/pack_stage7f.py" --out "$OUT" --zip "$REVIEW" || true
  exit 3
 fi
done
# Use a virtualenv that actually has Torch. The original stage5 venv is also valid.
if [[ -f "$HOME/cp10_detection_venv/bin/activate" ]]; then
 source "$HOME/cp10_detection_venv/bin/activate"
elif [[ -f "$HOME/cp10_stage5_venv/bin/activate" ]]; then
 source "$HOME/cp10_stage5_venv/bin/activate"
fi
export CP10_JOURNAL_ROOT="$ROOT"
set +e
{
 echo '=== Stage7F initial checks ==='
 python3 -c "import torch,pandas,numpy;print('Torch',torch.__version__,'Pandas',pandas.__version__)"
 python3 -u "$TOOL/stage7f_run.py" \
  --stage7d "$WIN/CP10_Stage7D_Tracker_Review.zip" \
  --stage7e "$WIN/CP10_Stage7E_GT_Linkage_Review.zip" \
  --gt "$ROOT/Main Result/trajectory_dataset/cleaned_track_points.csv" \
  --motion "$ROOT/Main Result/corrected_trajectory_dataset/repaired_ego_motion.csv" \
  --out "$OUT"
} 2>&1 | tee "$OUT/Stage7F_Console.log"
status=${PIPESTATUS[0]}
set -e
python3 "$TOOL/pack_stage7f.py" --out "$OUT" --zip "$REVIEW"
cp -f "$REVIEW" "$WIN/"
if (( status == 0 )); then
  echo "SUCCESS: $REVIEW (also in Windows Downloads)"
else
  echo "STAGE7F FAILED with exit=$status. Upload ZIP containing full diagnostics: $REVIEW"
fi
exit "$status"
