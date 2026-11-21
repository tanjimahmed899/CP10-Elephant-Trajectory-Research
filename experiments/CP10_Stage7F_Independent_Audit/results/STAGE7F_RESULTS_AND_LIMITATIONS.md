# CP 1.0 — Stage 7F independent audit of frozen tracker-history forecasting

**Source:** `CP10_Stage7F_Tracker_Forecast_Review.zip` uploaded on 2026-11-21. The archive contains `Stage7F_Verified_Evaluation_Windows.csv`, two NPZ prediction arrays, and a console log. It does **not** contain the final run provenance JSON, official metrics, or individual-seed predictions because reporting failed after frozen inference.

## Status and verified execution

- The log reports successful GT/source SHA and previous Stage7D archive verification before inference; 311 strict, GT-linked windows were reconstructed (187 at 2 s, 124 at 5 s).
- All six horizon/seed checkpoint evaluations (seeds 42, 52, and 62 for both horizons) completed, according to console log.
- The pipeline then failed during final reporting in `summarize(rows)`, where `rows` is a list of pandas DataFrames. `pd.DataFrame(rows)` raises an inhomogeneous-shape `ValueError`. The correct operation is `pd.concat(rows, ignore_index=True)`. This does not alter model predictions.
- The five float arrays in each saved NPZ have correct shapes and are finite. Independent calculations of Euclidean ADE/FDE from arrays are reproduced in the included CSVs.

## Window-weighted quantitative results

All errors below are measured in **normalized source-image coordinates in a common observation-start affine-corrected reference**, not metres. These are conditional *retrospective* results, not GT-blind operational measures.

| Horizon | Strict windows | Input | Method | Mean ADE | Mean FDE |
|---|---:|---|---|---:|---:|
| 2 s | 187 | Tracker centers | Frozen 3-seed Transformer | **0.010664** | **0.016606** |
| 2 s | 187 | Tracker centers | Stationary, last-observed point | 0.013623 | 0.022686 |
| 2 s | 187 | GT centers (oracle diagnostic) | Frozen 3-seed Transformer | 0.011099 | 0.017786 |
| 5 s | 124 | Tracker centers | Frozen 3-seed Transformer | **0.023050** | **0.042877** |
| 5 s | 124 | Tracker centers | Stationary, last-observed point | 0.028285 | 0.050985 |
| 5 s | 124 | GT centers (oracle diagnostic) | Frozen 3-seed Transformer | 0.024602 | 0.044849 |

Compared against the **same tracker-history stationary baseline**, the frozen Transformer yields **21.72% lower 2-s ADE and 18.51% lower 5-s ADE** (FDE reductions **26.80%** and **15.90%**, respectively). This baseline is a genuine matched-input baseline but not the same as original-detector Transformer forecasting: the original detector produced zero strict eligible windows.

The selected-window GT-center/oracle arm does *not* have lower aggregate error than the tracker-center arm (2s: 0.011099 vs. 0.010664; 5s: 0.024602 vs. 0.023050). This does **not** mean tracking is more accurate than GT. The history inputs differ, and the frozen model can react non-monotonically to localization errors and normalization. The evaluated conditional sample is narrow and overlapping. Average tracker-to-GT history-coordinate discrepancies are 0.006632 (2 s) and 0.006753 (5 s).

## Cluster-aware descriptive check

Because many windows overlap, aggregate windows are not statistically independent. In the selected one-sequence subset there are 13 unique GT elephant identities at 2 s (30 tracked fragment IDs) and 10 unique GT identities at 5 s (19 tracked fragment IDs).

Averaging each GT elephant's ADE first, then averaging those per-elephant means gives:

| Horizon | Tracker/Transformer macro ADE | Tracker/stationary macro ADE | GT identities where Transformer wins |
|---|---:|---:|---:|
| 2 s | 0.012732 | 0.015224 | 13 / 13 |
| 5 s | 0.023807 | 0.028900 | 9 / 10 |

These within-one-sequence checks do not provide an independent-group confidence interval. Given selection and prior testing, avoid significance claims across independent videos or independent elephants in different environments.

## Required limitations for manuscript

1. **Not an overall end-to-end error:** strict sample inclusion uses offline GT associations for the observed 20/20 tracking history and requires full future annotations. Operational model does not have this eligibility oracle.
2. **Single TestReal sequence (`0000000353_0000000000`):** all 311 windows came from this sequence; others contribute no strict window here.
3. **Overlapping forecasts:** 187 and 124 windows should not be described as 311 independent elephant trajectories. There are 13 and 10 unique GT identities by horizon.
4. **No paired original-detector ADE/FDE:** original detector generated zero strict matched windows. Availability improvements established at Stage7E cannot substitute for a direct paired forecasting-error comparison.
5. **2D affine motion correction:** reconstruction uses previous estimated affine transforms in a normalized image coordinate frame, not true surveyed ground-plane trajectory or metre-scale localization.
6. **Reporting did not finish:** saved three-seed **ensemble predictions** are sufficient for this independent ADE/FDE audit; individual-seed metrics and final checkpoint-hash manifest need the corrected WSL report rerun.
7. **Retrospective, previously inspected TestReal:** this is not untouched independent confirmation and does not validate field-ready alerting.

## Suggested manuscript wording

> On the subset of 187 two-second and 124 five-second forecasting windows satisfying strict retrospective elephant-identity linkage in one BIRDSAI TestReal sequence, three-seed frozen Transformer forecasts driven by detector/ByteTrack observations achieved mean ADE of 0.01066 and 0.02305 normalized image-coordinate units, respectively. Corresponding stationary baselines using the same tracked histories yielded ADE of 0.01362 and 0.02828 (21.7% and 18.5% reductions). These results establish conditional downstream forecasting feasibility after detector adaptation, not GT-blind end-to-end forecasting accuracy. The evaluation subset was selected using offline ground-truth track associations; sequence-level generalization and end-to-end yield remain unresolved.

## Full reporting repair

The included `fix_stage7f_reporting_wsl.py` performs one guarded change in the original WSL script, with a backup:

```python
# Old, incorrect (list of DataFrames):
d = pd.DataFrame(rows)
# Correct:
d = pd.concat(rows, ignore_index=True)
```

To obtain original per-seed output and a completed experiment provenance file, patch and rerun the original Stage7F script in WSL (no training, but frozen inference is repeated):

```bash
python3 ~/cp10_stage7f_audit/fix_stage7f_reporting_wsl.py
bash ~/cp10_stage7f_tools/CP10_Stage7F_Frozen_Tracker_Forecasting/run_stage7f_wsl.sh
```

The existing saved NPZs already support the ensemble analysis and can be reported with the above limitations even before the rerun.

## Independent reproducibility

Run `analyze_stage7f.py --input CP10_Stage7F_Tracker_Forecast_Review.zip --out RESULTS` with NumPy and pandas to regenerate all summary CSVs and PNG figures. Original source ZIP is included for reproducibility. No new predictions are generated by this audit.
