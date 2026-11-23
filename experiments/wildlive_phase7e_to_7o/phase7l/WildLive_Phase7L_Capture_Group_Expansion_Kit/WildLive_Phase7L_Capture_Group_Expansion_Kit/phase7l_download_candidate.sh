#!/usr/bin/env bash
set -euo pipefail
# Usage: bash phase7l_download_candidate.sh DJI_0142_video1
NAME="${1:?Usage: bash phase7l_download_candidate.sh DJI_0142_video1}"
case "$NAME" in
  ''|*[!a-zA-Z0-9_-]*) echo "Unsafe name" >&2; exit 2;;
esac
BASE="https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69"
ROOT="$HOME/CP10_new_data_private/WildLive/Phase7L/candidates"
TARGET="$ROOT/$NAME"
mkdir -p "$TARGET/boxid" "$TARGET/visual_qa_frames"
wget -r -l 3 -np -nd -A '*.json' -P "$TARGET/boxid" "$BASE/$NAME/boxid/"
N=$(find "$TARGET/boxid" -maxdepth 1 -type f -name 'frame_*.json' | wc -l)
printf 'JSON files downloaded: %s\n' "$N"
if (( N < 70 ));then echo 'Insufficient annotation frames for 20+50-frame evaluation — STOP' >&2; exit 3;fi
(cd "$TARGET" && zip -q -r "$ROOT/${NAME}_boxid.zip" boxid)
unzip -t "$ROOT/${NAME}_boxid.zip" | tail -n 2
# Images are requested for sparse visual QA; don't claim full visual verification.
for n in 0 100 200 400 600; do
 if [ -e "$TARGET/boxid/frame_${n}.json" ]; then
   wget -q -c -P "$TARGET/visual_qa_frames" "$BASE/$NAME/frames/frame_${n}.jpg" || echo "JPEG frame_$n unavailable"
 fi
done
for i in "$TARGET"/visual_qa_frames/*.jpg;do
 [ -f "$i" ] || { echo 'No JPEG sample images downloaded' >&2; exit 4; }
done
(cd "$TARGET" && zip -q -r "$ROOT/${NAME}_visual_qa_frames.zip" visual_qa_frames)
cp "$ROOT/${NAME}_boxid.zip" "$ROOT/${NAME}_visual_qa_frames.zip" /mnt/c/Users/HP/Downloads/
printf '\nReady to upload to ChatGPT:\n%s\n%s\n' "$ROOT/${NAME}_boxid.zip" "$ROOT/${NAME}_visual_qa_frames.zip"
