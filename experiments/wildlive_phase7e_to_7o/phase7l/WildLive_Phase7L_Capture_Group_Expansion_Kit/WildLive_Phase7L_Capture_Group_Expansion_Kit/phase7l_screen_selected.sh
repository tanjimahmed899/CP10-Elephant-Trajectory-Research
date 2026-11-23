#!/usr/bin/env bash
set -euo pipefail
# Usage: bash phase7l_screen_selected.sh /full/path/to/Phase7K_Kit
KIT="${1:?Usage: bash phase7l_screen_selected.sh /path/to/WildLive_Phase7K_Prospective_Validation_Kit}"
test -f "$KIT/src/phase7k.py" || { echo "Missing Phase7K src/phase7k.py" >&2; exit 2; }
mkdir -p ./phase7l_screening
# Label-only sampling: sample matches do NOT verify presence/absence in whole recording.
python "$KIT/src/phase7k.py" screen \
  --sequence DJI_0142_video1 \
  --sequence DJI_0117_video1 \
  --sequence DJI_0117_video4 \
  --sequence DJI_0601_video4 \
  --sequence DJI_0601_video5 \
  --sequence DJI_20230719145816_0003_V_video2 \
  --output ./phase7l_screening/selected_candidate_species_samples.csv
printf '\nDONE: ./phase7l_screening/selected_candidate_species_samples.csv\n'
