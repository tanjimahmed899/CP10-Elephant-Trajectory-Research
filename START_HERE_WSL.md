# CP10 — Complete the GitHub evidence archive from local E: drive

The earlier `CP10_GitHub_Research_Repository.zip` contains a substantial **starting** evidence archive. It is **not** proof that all files on `E:` and all newer Stage ZIPs were included. The accompanying collector scans your Windows drive through Ubuntu WSL. It **does not** modify source files or push anything to GitHub.

## 0. Arrange both packages

- Keep `CP10_GitHub_Research_Repository.zip` from the previous delivery in Windows Downloads.
- Keep `CP10_GitHub_Evidence_Completion_Tools.zip` in the same folder.
- Existing GitHub account is connected, but the research repo must be created under your desired account separately; do not commit into an unrelated repository.

## 1. Extract the original research repository if not done yet

```bash
mkdir -p ~/cp10_github
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_GitHub_Research_Repository.zip" ~/cp10_github
```

## 2. Extract these supplementary tools into that repository

```bash
cd ~/cp10_github/CP10-Elephant-Trajectory-Research
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_GitHub_Evidence_Completion_Tools.zip" .
```

## 3. First run: INVENTORY ONLY (NO COPYING)

```bash
python3 scripts/stage10_evidence_inventory_wsl.py \
  --root "/mnt/e/AIUB/Capstone Project/Journal Based Results" \
  --downloads "/mnt/c/Users/HP/Downloads" \
  --repo "$PWD" \
  --out "$HOME/CP10_Evidence_Review"
```

Open output using:

```bash
explorer.exe "$(wslpath -w ~/CP10_Evidence_Review)"
```

Inspect `EVIDENCE_INVENTORY_SUMMARY.json`, `CANDIDATES_REVIEW.csv` and the `REVIEW_BEFORE_GITHUB.md` before copying any new content.

## 4. Second run: COPY SELECTED EVIDENCE to OUTSIDE-repo staging

```bash
python3 scripts/stage10_evidence_inventory_wsl.py \
  --root "/mnt/e/AIUB/Capstone Project/Journal Based Results" \
  --downloads "/mnt/c/Users/HP/Downloads" \
  --repo "$PWD" \
  --out "$HOME/CP10_Evidence_Review" \
  --copy --extract-archive-evidence
```

Optional switches: `--include-figures` (figure copyright/privacy review), `--include-arrays` (NPZ/NPY coordinates review), and `--max-mb 20` (only if reviewed). **Never copy full wildlife raw videos or original dataset ZIPs into a public repo without permission.**

**Important:** This second run does NOT add files to Git, commit, push, or modify E: drive. New copies are in `~/CP10_Evidence_Review/staged_evidence/` with source folders preserved.

## 5. Manually review, then copy approved files

Read the claim mapping at `docs/analysis/PAPER_CLAIM_EVIDENCE_MAP.md`. Look for filenames absent from the old repo. After reviewing for source licenses, privacy, GPS, secrets and duplication, copy only the approved subdirectories you choose:

```bash
mkdir -p additional_wsl_evidence
# Example: only after review, choose a specific folder:
# cp -a ~/CP10_Evidence_Review/staged_evidence/E_DRIVE/'Main Result 3.0'/paper_strengthening_revision additional_wsl_evidence/
```

Avoid doing `cp -a ~/CP10_Evidence_Review/staged_evidence/. additional_wsl_evidence/` before review. Never commit `CANDIDATES_REVIEW.csv` or `EXCLUDED_SUMMARY_SAMPLE.csv` as-is if their filenames or paths disclose private information.

## 6. Run checks before a PRIVATE repo push

```bash
python3 scripts/pre_push_audit.py
git status --short
# After selecting, and before any push:
# git add .
# git diff --cached --stat
```

Use the earlier `GITHUB_SETUP_WSL.md` for private repository creation and push. `git status` will fail if you have not initialized Git; that is normal during staging. There is no default public upload.

## Scientific review note

Save all historical scripts for completeness, but identify a **locked/frozen protocol and executable entry point** for each manuscript claim. New Stage 4 data are not the same experimental population as old R8; Stage 7F is conditional one-sequence evidence; Stage 5 is an adverse calibration outcome. Avoid combining their pooled metrics.

## 7. Check which paper claim families have evidence files in the repo

```bash
python3 scripts/stage10_repo_claim_coverage.py \
  --repo "$PWD" \
  --out "$HOME/CP10_Evidence_Review/PAPER_EVIDENCE_PRESENCE_CHECK.csv"
```

This checks **filename presence only**, not reproducibility or scientific correctness. Manually inspect missing claims and verify the scientific numerical sources. Share the resulting `EVIDENCE_INVENTORY_SUMMARY.json` plus `PAPER_EVIDENCE_PRESENCE_CHECK.csv` for help deciding what to include next; do **not** upload local `CANDIDATES_REVIEW.csv` without checking whether local filenames contain restricted wildlife location or private information.
