# Camera-Motion-Corrected Elephant Trajectory Forecasting (CP 1.0)

**Status:** audited retrospective research repository; **not** an operational wildlife-alert system. Prepared for faculty review and reproducible research. This repository is intended to be **private until dataset redistribution rights, software licenses, and sensitive wildlife-location information are reviewed**.

## Research questions

1. Does estimated image-plane camera correction improve learned elephant trajectory forecasting when the future target is held fixed?
2. How do frozen conformal regions transfer across video-source and motion-regime shifts?
3. What changes when synthetic virtual boundaries are evaluated in a camera-consistent fixed-segment reference?
4. Does domain adaptation of the elephant detector improve tracking and provide usable histories for frozen Transformer forecasting?

## High-level audited findings

| Analysis | Dataset & horizon | Outcome | Interpretation |
|---|---|---|---|
| Stage 4 input-only matched-target Transformer ablation | BIRDSAI TestReal, 2s/5s | 66.51% / 53.62% mean within-seed ADE reduction | Corrected **observed history** improves ADE against the **same corrected future target**; 3 model seeds, reused source sequences |
| EXP3 frozen q90 pathwise coverage | WildBox 6 videos, 5s | 69.76% | Below nominal under source shift; overlapping windows |
| Stage 5 development-only calibration transfer | WildBox 2 retrospective evaluation videos, 5s | 2/28 = 7.14% covered | Negative transfer finding; do **not** claim a new coverage guarantee |
| Stage 6 fixed-reference synthetic crossing alerts | EXP3, 2s/5s | 116/133 and 109/124 events detected by N=2 | High event sensitivity, but 348/237 unassociated alert runs; not field alarms |
| Stage 7C original to TrainReal-adapted detector | BIRDSAI TestReal, 9,085 frames | Recall 5.10% -> 56.98%; precision 57.21% -> 34.20% | Major recall gain with substantial FP increase (1,835 -> 52,684) |
| Stage 7D frozen ByteTrack | Same TestReal sequences | HOTA 0.0252 -> 0.2041 | Tracking improves; 1,307 ID switches and 3,597 fragmentations remain |
| Stage 7E strict GT-linked windows | TestReal | Adapted 187 (2s), 124 (5s) | **All** from one sequence, GT used offline for eligibility |
| Stage 7F frozen 3-seed forecasting on selected tracks | TestReal sequence 0353 | ADE 0.010664 (2s), 0.023050 (5s) | 21.72% / 18.51% below matched stationary ADE, **conditional**, not GT-blind end-to-end |

All ADE/FDE values above are normalized image-coordinate units unless the source explicitly states otherwise. **The 66.51% and 53.62% are the mean of three within-seed percentage reductions, not a claim about ensemble prediction error.** Dataset cohorts and experimental rules differ across stages; no universal pooled performance estimate is warranted.

## Repository map

- [`src/original/`](src/original/) — all **267** original uploaded Python scripts, including iterative/fixed variants. They are preserved for provenance, **not automatically guaranteed to run as-is**.
- [`experiments/`](experiments/) — extracted experiment scripts and stage-specific CSV/JSON/NPZ metrics and plots from reviewed ZIP archives. Each subdirectory retains the original package name.
- [`docs/research_reports/`](docs/research_reports/) — four historical experimental-result PDFs.
- [`docs/papers/`](docs/papers/) — latest journal drafts, faculty results and review notes.
- [`docs/analysis/`](docs/analysis/) — scientific audit findings and claim ledgers.
- [`figures/`](figures/) — selected statistical plots (not copyrighted source video frames).
- [`manifests/`](manifests/) — file-by-file SHA-256 manifest; source package inventory; exclusion reasons.
- [`scripts/`](scripts/) — secret/file-size checks, local WSL supplement collection, private repository publishing.

## Datasets and models

This repository **does not redistribute** original BIRDSAI/WildBox/BAMBI/COCO images, videos, protected wildlife camera metadata, GPS/geocoordinates, or `.pt` checkpoints. See [`data/README.md`](data/README.md) and [`models/README.md`](models/README.md) for the provenance checklist. Historical scripts may require datasets/checkpoints in their original expected E: drive paths and old library versions.

## Reproducibility and evidence classification

The experiment archive manifests identify sources, included files and excluded content. This is a **research archive**, not a push-button single-script pipeline. Use each stage's historical protocol and original checkpoint hashes before re-running. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

**Evidence guardrails:** Do not describe the original detector-only forecasting benchmarks as GT-blind perception-to-warning performance, a synthetic image-plane line as a GPS fence, 2s/5s horizon as measured field response time, a Stage 5 negative transfer result as successful recalibration, or within-one-video clustered bootstrap as proof of generalization to new groups.

## Publication and licensing

The source code is provided for supervisor/private reproducibility review. **No general open-source license is granted yet.** Review rights to redistribute copied/adapted scripts, imported data and research photos before making public. Paper and figures may have separate copyright considerations.
