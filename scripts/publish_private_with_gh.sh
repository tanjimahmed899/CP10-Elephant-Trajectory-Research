#!/usr/bin/env bash
# Safely create a NEW PRIVATE GitHub repository from the unpacked folder.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/pre_push_audit.py
if ! command -v git >/dev/null 2>&1; then echo 'git not found. Install: sudo apt install git'; exit 1; fi
if ! command -v gh >/dev/null 2>&1; then echo 'GitHub CLI (gh) not found. Install GitHub CLI and run gh auth login.'; exit 1; fi
gh auth status
NAME="${1:-CP10-Elephant-Trajectory-Research}"
if [[ -n "$(git status --porcelain 2>/dev/null || true)" && -d .git ]]; then echo 'Existing Git repo detected: review changes before pushing'; fi
if [[ ! -d .git ]]; then git init -b main; fi
git add --all
printf '\nStaged summary (verify BEFORE upload):\n'
git diff --cached --stat | tail -n 30
printf '\nAbout to create PRIVATE GitHub repository %s under your authenticated account.\n' "$NAME"
read -r -p 'Type UPLOAD PRIVATE to continue: ' CONFIRM
if [[ "$CONFIRM" != 'UPLOAD PRIVATE' ]]; then echo 'Cancelled; no upload.'; exit 1; fi
if git diff --cached --quiet; then echo 'No new changes to commit'; else git commit -m 'Initial audited CP10 research code and experimental evidence'; fi
if git remote get-url origin >/dev/null 2>&1; then echo 'Remote origin exists. Refusing automatic overwrite.'; exit 1; fi
gh repo create "$NAME" --private --source=. --remote=origin --push
printf 'Private repository created and pushed. Review visibility before sharing.\n'
