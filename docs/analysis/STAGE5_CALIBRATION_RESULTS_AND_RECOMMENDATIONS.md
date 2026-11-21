# CP1.0 — Stage 5 retrospective calibration transfer analysis

**Analyzed input:** `CP10_Stage5_Frozen_Results_Review.zip` (the supplied WSL frozen Transformer outputs). Results were recomputed from `ensemble_prediction_abs`, `future_abs`, `scale` and per-window `analysis_role`. No model training or new inference was performed here.

## 1. Source data and forecasting evidence

- Horizon 2 s: **89** full forecast windows, 11 distinct tracklets; **36 development** windows in `DJI_20240119152552_0006_V`, **53 retrospective evaluation** windows in `DJI_20240626143832_0004_V` (8) and `DJI_20260227084907_0004_V` (45).
- Horizon 5 s: **56** full forecast windows, 8 tracklets; **28 development** windows in `0006_V`, **28 evaluation** windows across `20240626143832_0004_V` (2) and `20260227084907_0004_V` (26).
- Recomputed three-seed ensemble ADE/FDE: **0.04946/0.08520** (2 s) and **0.09187/0.16670** (5 s). These agree with the attached `fresh_holdout_model_metrics.csv`.
- These are annotation-eligible, frozen forecasts, **not detector-driven tracking results**. All three videos belong to previously inspected elep1/elep2/elep3 recording groups.

## 2. Correct conformal score and comparator thresholds

The per-window *simultaneous pathwise* nonconformity score is

\[r_i=\frac{\max_t\lVert\hat p_{i,t}-p_{i,t}\rVert_2}{s_i}.\]

For a candidate threshold \(q\), a window is covered only when **every predicted future point** is within the scale-adjusted region. We used the original externally frozen TrainReal thresholds already reproduced in Stage 1: \(q_{2s}=2.158768\), \(q_{5s}=1.975130\).

The development-only 90% empirical split-conformal order statistic uses \(k=\lceil 0.9(n+1)\rceil\) and the \(k\)-th smallest development score; this rule is *not* claimed to enjoy a cross-video guarantee under distribution shift.

| Horizon | Development windows | Development-only q90 | Frozen TrainReal q90 | q_dev/q_frozen |
|---|---:|---:|---:|---:|
| 2 s | 36 | 0.193358 | 2.158768 | 0.0896 |
| 5 s | 28 | 0.335485 | 1.975130 | 0.1699 |

## 3. Primary retrospective evaluation results

| Horizon | Method | Covered / evaluated windows | Pathwise coverage | Mean region radius relative to frozen |
|---|---|---:|---:|---:|
| 2 s | **Frozen TrainReal q90** | **53 / 53** | **100.0%** | 1.000× |
| 2 s | Development-only q90 | 3 / 53 | 5.66% | 0.0896× |
| 5 s | **Frozen TrainReal q90** | **24 / 28** | **85.71%** | 1.000× |
| 5 s | Development-only q90 | 2 / 28 | 7.14% | 0.1699× |

The *development-only* threshold is much smaller because this chosen development video is unusually easy. Applying it to the other two videos catastrophically reduces coverage. A `max(q_frozen, q_dev)` **no-shrink diagnostic** exactly reproduces the frozen threshold, avoiding this particular collapse but **not fixing** the 5-second 85.71% undercoverage.

Even the largest score *observed* on the development video has low evaluation coverage: 4/53 (7.55%) at 2 s and 3/28 (10.71%) at 5 s. This illustrates lack of support in that development source; it is not a new calibration method.

### Per-evaluation video

| Horizon | Evaluation video | Windows | Frozen q90 covered | Development-only q90 covered |
|---|---|---:|---:|---:|
| 2 s | `DJI_20240626143832_0004_V` | 8 | 8/8 | 0/8 |
| 2 s | `DJI_20260227084907_0004_V` | 45 | 45/45 | 3/45 |
| 5 s | `DJI_20240626143832_0004_V` | 2 | 1/2 | 0/2 |
| 5 s | `DJI_20260227084907_0004_V` | 26 | 23/26 | 2/26 |

## 4. Critical observed distribution mismatch

For each horizon, **100% of the development windows use the minimum allowed residual scale**, while **0% of the evaluation windows use the minimum scale**:

- 2 s: 36/36 development windows have scale 0.06820264; evaluation scale ranges from 0.11111 to 1.78819.
- 5 s: 28/28 development windows have scale 0.06732639; evaluation scale ranges from 0.13433 to 1.78819.

The normalized max-residual score distribution is also markedly shifted: at 5 s, the development score maximum is 0.43753 while the evaluation empirical 90th percentile is about 1.998. These specific two-domain datasets have almost no calibrated residual support overlap. This suggests **motion/scale-regime shift** as a plausible mechanism, but does not by itself establish a causal explanation.

## 5. Scientific interpretation and limitations

1. **Do not claim the proposed Stage 5 target-only recalibration restored 90% coverage.** It failed badly on the two retrospective evaluation videos. The previous Stage 1 leave-one-group-out pooled 90.21% result used a different, previously inspected group-split protocol, and does not override this result.
2. **Original frozen TrainReal q90** achieves 100% coverage at 2 s but only 85.71% at 5 s in the evaluated windows. This is an important horizon-specific failure; 5-second sample size is just 28 overlapping windows.
3. The **two evaluation videos are not two independent new recording groups**. Windows overlap within tracked elephants, and source-level counts are severely imbalanced (only 2 five-second windows in the elep2 video). Avoid naive binomial confidence intervals treating windows as independent. Data reuse and earlier qualitative inspection also prohibit describing these as an untouched confirmatory set.
4. Sample eligibility is based on matching elephant ground truth to tracker boxes. This **does not** demonstrate detector-to-tracker deployability or field warning coverage.
5. No weighted conformal, ACI, Mondrian, or conformal risk-control guarantee has been established by this stage. Post-hoc selecting a better threshold on evaluation scores would be leakage.

## 6. Recommended next experiment

**Decision:** Keep Stage 4 R8 input-only ablation as the strongest current method comparison. For Stage 5, the most valuable next step is to obtain an independent recording group and a representative adaptation partition with both low and high motion/scale regimes. Pre-register the original frozen q90, a no-shrink safe comparator, and a video-level/track-level calibrated method; use video-disjoint calibration and evaluation, and report coverage together with region radius, effective number of independent tracks and recording groups.

In parallel, Stage 6 can move forward **without new footage**, using existing Stage 1/Stage 2 decisions to evaluate episode-level warning metrics. It must label synthetic boundaries as simulations and not claim real-world lead time or a conformal risk guarantee.

## Outputs

- `Stage5_Calibration_Coverage_Summary.csv` — all threshold comparisons including development and evaluation.
- `Stage5_Calibration_By_Video.csv` and `Stage5_Calibration_By_Track.csv` — video/track stratification and warnings about unequal counts.
- `Stage5_Individual_Window_Diagnostics.csv` — nonconformity, scale, error and coverage per sample.
- `Stage5_Provenance_And_Diagnostics.json` — original ZIP SHA-256, sample counts, code formula and limitations.
- Three `Figure_Stage5_*.png` plots — manuscript-quality but retrospective evidence.
- `analyze_stage5.py` — reruns numerical analysis and chart creation from the uploaded input ZIP; no new training/inference.

### Reproduce on WSL

```bash
source ~/cp10_stage5_venv/bin/activate
python analyze_stage5.py --input CP10_Stage5_Frozen_Results_Review.zip --out stage5_audit
```

Input ZIP and script paths should be adjusted for your local folders.
