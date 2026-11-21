# Stage 5 — WildBox candidate archives preflight

This script performs a **read-only** inspection of the three candidate video ZIPs not included in EXP3 or R10. It requires **Python 3.9+ standard library only** (no torch, NumPy, CUDA, cv2 or unzip).

From WSL Ubuntu, place `CP10_Stage5_Candidate_Preflight.zip` in `C:\Users\HP\Downloads`, then run:

```bash
mkdir -p ~/cp10_stage5_tools
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage5_Candidate_Preflight.zip" ~/cp10_stage5_tools
python3 ~/cp10_stage5_tools/CP10_Stage5_Candidate_Preflight/stage5_candidate_preflight.py \
  --root "/mnt/e/AIUB/Capstone Project/Journal Based Results" \
  --out ~/CP10_Stage5_Candidate_Audit
cat ~/CP10_Stage5_Candidate_Audit/CP10_Stage5_Research_Decision.txt
explorer.exe "$(wslpath -w ~/CP10_Stage5_Candidate_Audit)"
```

**Send back** `CP10_Stage5_Candidate_Audit.json` and `CP10_Stage5_Segment_Summary.csv` (both small). The script reads the video ZIPs in place and does not extract JPGs or alter the research directories. The default scan does not decompress all images; use `--check-crc` only if a full integrity check is separately needed. Window estimates are **feasibility only**, not a validation metric. Do not use the outcome to claim unseen-group conformal coverage.
