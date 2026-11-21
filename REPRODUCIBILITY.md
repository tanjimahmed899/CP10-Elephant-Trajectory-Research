# Reproducibility

This is the original evolving research history, not a single polished Python package. Reproduce one experiment at a time and recover its frozen data split, checkpoints, calibration threshold, source sequence, software versions and evaluation protocol from the per-stage files and manifests. Do not compare different cohorts as if they shared sample identities.

### Typical WSL project root

`/mnt/e/AIUB/Capstone Project/Journal Based Results`

Scripts in `src/original` preserve historical path assumptions and bug-fix versions. Before rerunning them, identify the final locked revision and verify checkpoint SHA-256; do not blindly execute all versions. The isolated stage runners under `experiments` are accompanied by outputs for traceability.

### Computational environments

- Primary work was done on Windows + Ubuntu WSL.
- Ultralytics, PyTorch and image-processing package versions changed between historical R13 and later Stage 7D; historical tracker results do not exactly reproduce byte-for-byte.
- Stage 7F results are selected using an **offline GT identity gate**; GT future is used for evaluation only.
- Model weights and original data are not included.

### Confirm source integrity

`python3 scripts/pre_push_audit.py`

Look up any file in `manifests/repository_file_sha256.csv` to recover the exact source archive and member. See `manifests/root_source_archive_inventory.csv` for excluded original archives and sizes.
