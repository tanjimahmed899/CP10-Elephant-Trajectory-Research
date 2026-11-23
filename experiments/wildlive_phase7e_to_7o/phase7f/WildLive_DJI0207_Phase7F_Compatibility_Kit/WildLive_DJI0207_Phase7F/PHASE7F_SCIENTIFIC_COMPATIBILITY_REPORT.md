# Phase 7F — Frozen Transformer Compatibility Audit (DJI_0207)

**Status:** `PREPROCESSING COMPATIBLE; FROZEN INFERENCE BLOCKED — CHECKPOINTS NOT PROVIDED`.

## Sources inspected

- User-provided `All Python Code.zip` (267 `.py` programs), especially `train_transformer.py` and `evaluate_transformer_ensemble.py`.
- User-provided `EXP3_Output_Review.zip`, notably `scripts/exp3_frozen_forecaster_adapter.py`.
- `WildLive_Phase7E_Reproduced_Results.zip` and the exact `frame_based_windows_and_baselines.npz` produced from the original `WildLive_DJI_0207_boxid.zip`.
- Earlier annotation/overlay QA reports for tracks 0/1/2.

**Checkpoint file inventory:** No `.pt`/`.pth`/`.ckpt` files in either original uploaded code archive or EXP3 review archive. No original six model weights are presently supplied. Do not report Transformer ADE/FDE until genuine original weights have been located and passed strict loader checks.

## Model-specific compatibility

| Requirement | Source-grounded status |
| --- | --- |
| Expected input shape | `[N,20,2]`, matches Phase 7E raw-history array shape but **not** native input coordinate representation |
| Expected output horizon | 20 (`2s_to_2s`) and 50 (`2s_to_5s`), matches array dimensions |
| Training model architecture | Compact Temporal Transformer, 64-d latent, 4 heads, 2 encoder layers, FFN 128, dropout 0.1 |
| Model seeds | 42, 52, 62 for each task (six total) |
| Checkpoint naming | `{task}_Transformer_seed{seed}.pt` |
| Frozen inference controls | `eval()`, `torch.inference_mode()`, no optimizer/training |
| Input canonicalization | Last observed point as origin; last 10 frame OLS heading; `sum(norm(diff(history)))` scale with task-specific frozen floor; angle rotation |
| Frozen scale floor, 20 | 0.06820264416877736 |
| Frozen scale floor, 50 | 0.06732638773715147 |
| Native source of observations | Original model's projected normalized image-plane target positions; WildLive uses bbox centres normalized by 3840×2160. These are *not* identical measurements of physical movement. |
| Temporal resolution | Legacy model trained at 10 Hz. WildLive frame timestamps/FPS unverified. Thus a possible inference can be an **exploratory temporal-domain stress test only**. |
| Independent recording status | One DJI_0207 group, cannot establish independent replication or nonoverlap with WildBox. |

The original task identifiers `2s_to_2s` and `2s_to_5s` are checkpoint filenames. **They must not be used to label seconds for these WildLive frames.** Report horizons as 20 and 50 released *frames* only.

## Audited phase 7E windows

- 3 elephant tracks × 829 windows at 20-frame horizon = 2,487.
- 3 elephant tracks × 799 windows at 50-frame horizon = 2,397.
- Total overlapping windows = **4,884**, all from one DJI_0207 capture group.
- Input NPZ SHA-256 = `35b2ae5a4b28d3fbfc6de0b6d657d177dbdf5c543461fb2b70f55e612512fa8e`.
- Canonicalization + inverse-transform max absolute round-trip error, across all histories: **9.9634e-10**.
- Original source preservation: no annotation value altered; flagged windows retained and separately evaluated by subset.

### Frozen scale-floor activation — warning

| Horizon | Floor-applied windows | Fraction | Median path, representative track IDs 0 / 1 / 2 |
| --- | ---: | ---: | --- |
| 20 frames | 2,425 / 2,487 | **97.51%** | 0.0240 / 0.0148 / 0.0331 |
| 50 frames | 2,332 / 2,397 | **97.29%** | 0.0231 / 0.0147 / 0.0327 |

The scale floor is not a tuned correction: it is **locked to historical EXP3**. A very high activation rate indicates that this geometry differs substantially from the training/preprocessing scale distribution and strengthens the out-of-domain interpretation. Do not tune the floor on test errors. A broader source-comparative statistical conclusion needs original training distribution evidence.

### Same-window descriptive baselines (all overlapping windows)

| Horizon (frames) | Baseline | ADE | FDE |
| --- | --- | ---: | ---: |
| 20 | Stationary | 0.004826607 | 0.008356044 |
| 20 | Phase7E endpoint CV20 | 0.004875328 | 0.008539015 |
| 20 | EXP3-style OLS CV | 0.004911255 | 0.008604828 |
| 50 | Stationary | 0.008416475 | 0.015407623 |
| 50 | Phase7E endpoint CV20 | 0.009763880 | 0.019471345 |
| 50 | EXP3-style OLS CV | 0.009928268 | 0.019860759 |

Important: EXP3's constant-velocity fit uses *all 20 observed positions via OLS*. Phase7E's earlier CV20 uses the endpoint-to-endpoint slope. These are different, and neither is a learned model. They must be labeled separately.

## Tests performed

1. 9/9 unit tests passed in the audit environment (including independent scalar restatement of source canonicalization).
2. Synthetic-weights loader/forward smoke test passed on both model architecture dimensions. **The synthetic random weights are not model evidence, are not in the deliverable, and were not used to calculate any Transformer score.**
3. Original Phase7E Stationary and endpoint-CV metrics reproduced to numerical precision.
4. Source NPZ shape, start indices, QA flags and numeric finiteness were verified.

## Requirements to unblock model inference

Find **all six** original unmodified checkpoints on the user's Windows/WSL machine:

`2s_to_2s_Transformer_seed42.pt`, `...52.pt`, `...62.pt`, `2s_to_5s_Transformer_seed42.pt`, `...52.pt`, `...62.pt`.

Run the provided script with `--checkpoint-dir`, which audits SHA-256 and uses `torch.load(..., weights_only=True)` and strict state-dict compatibility. The script never trains or modifies original checkpoints. It does not perform inference unless all six are present.

## Evidentiary limits for a journal

- No confirmed timing or capture-date metadata, camera stabilization, ground-plane coordinates, or cross-dataset recording deduplication.
- No conformal interval or virtual-boundary inference can be declared calibrated on this untimed frame-based dataset.
- A model-vs-baselines result, if completed, is a **single-recording exploratory domain stress test**, not a confirmatory 2s/5s forecasting experiment.
- All 4,884 windows overlap; recording group, not windows, is the independent unit. No reliable cross-group CI can be estimated from N=1 group.
- WildLive licensing must be checked before publishing any images, annotations, derivative arrays, or model data to a public repository.

**Decision:** Phase 7F compatibility/preflight **PASS**; frozen Transformer empirical comparison **BLOCKED until original six checkpoint files are supplied**.
