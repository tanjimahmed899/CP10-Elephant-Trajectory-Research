#!/usr/bin/env bash
set -uo pipefail
# Optional, NETWORKED, annotation-ONLY species census. Does not download frames or videos.
# Run from this kit folder on user's Ubuntu WSL with wget and Python 3.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HOME/CP10_new_data_private/WildLive/Phase7M/authorless_json_census}"
mkdir -p "$DEST" "$DEST/archives" "$DEST/logs" "$DEST/annotations"
STATUS="$DEST/download_status.csv"
echo 'sequence,download_exit_code,json_files,contiguous_index_check,archive_path' > "$STATUS"
while IFS= read -r SEQ; do
  [ -n "$SEQ" ] || continue
  case "$SEQ" in
    DJI_0207|DJI_0204_video1|DJI_0204_video2|DJI_0161_video1|DJI_0601_video2)
      echo "SKIP already available full annotation ZIP: $SEQ"
      continue;;
  esac
  DIR="$DEST/annotations/$SEQ/boxid"
  mkdir -p "$DIR"
  echo "Downloading only JSON annotations: $SEQ"
  RC=0
  wget -r -l 3 -np -nd -c -A '*.json' --timeout=30 --tries=2 \
    -P "$DIR" \
    "https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/$SEQ/boxid/" \
    >"$DEST/logs/${SEQ}.log" 2>&1 || RC=$?
  COUNT="$(find "$DIR" -maxdepth 1 -type f -name 'frame_*.json' | wc -l | tr -d ' ')"
  ARCHIVE=''
  CONTIG='UNKNOWN'
  if [ "$COUNT" -gt 0 ]; then
    ARCHIVE="$DEST/archives/${SEQ}_boxid.zip"
    (cd "$DEST/annotations/$SEQ" && zip -qr "$ARCHIVE" boxid)
    CONTIG="$(python - "$DIR" <<'PY'
import pathlib,re,sys
nums=[]
for p in pathlib.Path(sys.argv[1]).glob('frame_*.json'):
 m=re.fullmatch(r'frame_(\d+)\.json',p.name)
 if m: nums.append(int(m[1]))
print('CONTIGUOUS' if nums and len(nums)==len(set(nums)) and max(nums)-min(nums)+1==len(nums) else 'GAPS_OR_DUPLICATES')
PY
)"
  fi
  echo "$SEQ,$RC,$COUNT,$CONTIG,$ARCHIVE" >> "$STATUS"
done < <(python - "$PROJECT/protocol/wildlive_22_folders_prior_status.csv" <<'PY'
import csv,sys
with open(sys.argv[1],newline='',encoding='utf-8') as f:
 for row in csv.DictReader(f):
  print(row['sequence_id'])
PY
)
python "$PROJECT/src/authorless_provenance.py" census \
 --zip-dir "$DEST/archives" --output "$DEST/new_recording_species_census.csv" \
 || echo 'No ZIPs collected; inspect download logs.'
echo "DONE. Review: $STATUS and $DEST/new_recording_species_census.csv"
echo 'Do not treat HTTP errors or partial downloads as proof a species is absent.'
