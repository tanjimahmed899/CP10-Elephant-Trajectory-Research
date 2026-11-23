#!/usr/bin/env bash
set -euo pipefail
KIT7K="$HOME/CP10_new_data_private/WildLive/Phase7K/WildLive_Phase7K_Prospective_Validation_Kit"
OUT="$HOME/CP10_new_data_private/WildLive/Phase7L/remaining_six_candidate_species_samples.csv"
test -f "$KIT7K/src/phase7k.py" || { echo "Missing Phase 7K kit: $KIT7K" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
python "$KIT7K/src/phase7k.py" screen --sequence DJI_0117_video2 --sequence DJI_0601_video3 --sequence DJI_20230719145427_0002_V_video2 --sequence DJI_20230719145427_0002_V_video3 --sequence DJI_20230719145427_0002_V_video4 --sequence DJI_vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V --output "$OUT"
printf "\nScreening summary:\n"
cat "$OUT"
cp "$OUT" /mnt/c/Users/HP/Downloads/
echo "Copied CSV to Windows Downloads: remaining_six_candidate_species_samples.csv"
