# Stage 5 semantic track reconciliation — data-derived interpretation

All numbers below are derived from the user's `CP10_Stage5_Semantic_Audit_Results.zip` (IoU ≥ 0.50, one-to-one matching). They are **eligibility/preflight counts**, not Transformer inference, calibrated coverage, or independent model performance.

| Group | Analysis role | Elephant GT matched | Verified track IDs | Strict 2s candidate starts | Strict 5s candidate starts |
|---|---|---:|---:|---:|---:|
| elep1 | adaptation/development | 459/459 | 3 | 342 | 262 |
| elep2 | retrospective evaluation | 174/178 | 2 | 78 | 18 |
| elep3 | retrospective evaluation | 729/742 | 6 | 412 | 239 |
| **Total** | | **1362/1379** | **11** | **832** | **519** |

Strict start counts use every possible contiguous start. The original forecaster's `STRIDE=10` plus camera projection can reduce these substantially. No numbers from smoke testing are research evidence.

## Experimental protocol

- Reuse original frozen 3-seed Transformer and 10 Hz canonicalization; no re-training.
- Adaptation/development: `DJI_20240119152552_0006_V` (previously inspected qualitatively).
- Retrospective evaluation: `DJI_20240626143832_0004_V` and `DJI_20260227084907_0004_V`.
- No evaluation video may influence fitted calibration quantiles/threshold choices; do not use a retrospective comparison as an unseen-group guarantee.
- These videos are from the same broader recording groups previously studied. Source mentions in earlier R12 and other audits mean prior-use status is not a clean blind lock.
- Detector output is NOT being tested: `tracking_summary` is annotation conditioned. The labels are used for semantic validation and evaluation only.
- For formal conformal claims, address dependent overlapping windows, calibration track counts and prior data inspection. Save per-video/per-track sample counts and region sizes.
