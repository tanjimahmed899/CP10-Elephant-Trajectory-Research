# Stage 12 — Public repository release and reproducibility status

The project is already **public**. This status is not a statement that all artifacts have redistribution rights or that experiments are independently reproducible without restricted input data and pretrained checkpoints.

## What is documented

1. Main experimental tables and figures are linked in [the published-evidence index](STAGE12_PUBLISHED_EVIDENCE_INDEX.md).
2. Code and frozen-result CSV/JSON/NPZ files are preserved under `src/original/` and `experiments/` with source-archive provenance.
3. Scientific caution statements are preserved for Stage 4 seed averaging, Stage 5 calibration transfer, Stage 6 virtual-line warning simulation, and Stage 7E/F GT-conditioned single-sequence evaluation.
4. No operational 2s/5s warning lead time, physically surveyed WildBox fence, cross-video Stage 7F statistical guarantee, or universally superior Transformer is claimed.

## Remaining reproducibility dependencies

- Original YOLO training split, configuration, logs and independent evaluation provenance need a verified primary source. Existing R13 and Stage 7A–C artifacts do not substitute for the original detector training records.
- Original model `.pt` checkpoint files are intentionally not included. Record exact hashes and retrieval conditions without copying protected weights into this public repository.
- Dataset redistribution rights, upstream code attribution and source URL/license notes remain subject to review.
- `requirements/` contains a starting point, not a complete frozen cross-platform environment. Historical experiments used different Ultralytics versions.
- Some experimental CSVs contain sample IDs/tracks and local filesystem paths. They may not be appropriate for unrestricted third-party distribution without additional review.

## Public release audit

On the user's own WSL checkout, run the read-only audit. Keep the output folder OUTSIDE this public repo:

```bash
python3 scripts/stage12_public_audit.py --repo . --out ~/CP10_Stage12_Review
```

Inspect `STAGE12_PUBLIC_RELEASE_AUDIT.json` and `STAGE12_MANUAL_FILE_REVIEW.csv`. Flags are risk indicators, **not proof of sensitive disclosure**. Absence of flags is not proof that the repository is safe to redistribute.

## Adding further evidence

Use `scripts/stage12_collect_remaining.py` to compare E:-drive candidate summary files against current tracked SHA256 hashes and stage eligible aggregates **outside the repository**. Neither that script nor the audit performs Git commits or pushes. Manually approve the content, origin, license and study role before any release.

## Freeze for a manuscript

Once scientific and release checks are complete, create a new Git tag for the paper evidence version and use the full commit SHA in the manuscript's Code Availability section. Do not describe the repository as an independent reproduction benchmark unless all dependencies and frozen protocols are available.
