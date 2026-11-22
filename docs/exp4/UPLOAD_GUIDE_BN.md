# GitHub upload guide (Bangla)

Target public repository: `tanjimahmed899/CP10-Elephant-Trajectory-Research`

## Recommended: GitHub Desktop on Windows

1. GitHub Desktop install/sign in করো। `File > Clone repository` দিয়ে existing public repository clone করো (আগে clone থাকলে `File > Add local repository`).
2. `CP10_GitHub_Addon_Ready.zip` **extract** করো, ZIP-টির contents cloned repository-র root-এ copy করো। Existing folders `docs/`, `experiments/`, `scripts/` এবং নতুন `templates/` merge হবে।
3. GitHub Desktop-এ `Changes` দেখে verify করো; শুধু `docs/exp4/`, `experiments/exp4/`, `scripts/exp4/`, `templates/exp4/` এবং কোনো unintended files নেই।
4. Commit message: `Add EXP4 exploratory audits and validation protocols (Phases 0–7)`.
5. `Commit to main` তারপর `Push origin`; অতিরিক্ত caution চাইলে dedicated branch + PR ব্যবহার করো।

## Git CLI alternative

After cloning once and extracting ZIP **into the repository root**:

```bash
git status
git add docs/exp4/ experiments/exp4/ scripts/exp4/ templates/exp4/
git diff --cached --stat
git diff --cached --check
git commit -m "Add EXP4 exploratory audits and validation protocols (Phases 0-7)"
git push origin main
```

If main is protected, create a feature branch and open a pull request instead.

## What should remain outside public GitHub

Raw third-party MP4/videos or frames, proprietary images/datasets, wildlife GPS/camera metadata, private credentials/tokens, checkpoints, unpublished manuscript drafts pending supervisor approval, and large historical output CSVs requiring provenance/licensing checks. The public repository `.gitignore` already ignores `*.zip`, `*.mp4`, model checkpoint extensions and selected raw directories, but **manual review is still necessary**.

These are retrospective or exploratory **research artifacts**; they are not a verified deployed elephant-warning system. Phase 6–7 are readiness/preflight only.
