# CP 1.0 — Phase 7J: Development-only adaptive model selection for frame-based elephant forecasting

**Audit date:** 2026-11-22  
**Status:** COMPUTATION AND CODE TESTS PASS; ADAPTIVE-GENERALIZATION HYPOTHESIS NOT SUPPORTED BY THIS RETROSPECTIVE TEST  
**Statistical unit:** one recording/capture event only when independently verified; capture-event independence is *currently unverified*.

## Purpose and input evidence

The Phase 7J implementation uses the uploaded **WildLive_Phase7I_Reproduced_Results.zip**, containing 7,308 motion-feature and ground-truth forecast-error rows from the two previously examined WildLive recording folders:

- `DJI_0207`: 2,487 forecast windows at 20-frame future, 2,397 at 50-frame future; 3 elephant IDs, 868 annotation indices.
- `DJI_0204_video2`: 1,272 forecast windows at 20-frame future, 1,152 at 50-frame future; 4 elephant IDs, 357 annotation indices.

The historical scores for three unchanged prediction methods are passed through from their archived frozen-model evaluations: (1) stationary, (2) OLS constant velocity, and (3) the original three-seed Transformer ensemble. Phase 7J **does not rerun the frozen checkpoints** and **does not retrain the Transformer**. The Phase 7I input contains 20-frame observation-derived features and the forecasted trajectory ADE/FDE scores; the latter are used exclusively as **supervision** during selector fitting and for evaluation, never as selector inputs.

There is no verified native FPS or per-frame timestamp, no camera-motion-corrected trajectory, and no verified geographic-scale trajectory. All ADE and FDE results are in normalized image-plane coordinate units and **frame horizons only**.

## Selector architecture (fully specified)

For each horizon separately (20-frame, 50-frame), a deterministic ridge model predicts each expert's log ADE based solely on its 20-frame observation window. Six pre-chosen input features are:

1. `mean_step_norm_per_frame` (log10 with 1e-6 offset)
2. `straightness` (raw)
3. `ols_line_residual_norm` (log10 with 1e-6 offset)
4. `acceleration_rms_norm_per_frame2` (log10 with 1e-6 offset)
5. `mean_abs_turn_rad` (raw)
6. `ols_velocity_norm_per_frame` (log10 with 1e-6 offset)

Six missingness indicators accompany them, keeping undefined turning features distinguishable from zero turning. Imputation uses the *training fold's* feature medians only, followed by *training-only* mean/std normalization (std floor 1), clipping z-scores to [-4, 4]. Responses are `log(ADE + 1e-6)` for stationary/CV-OLS/Transformer. Ridge coefficient penalty is fixed at 25.0, with an unpenalized intercept. At inference, the selector picks the expert with the smallest predicted log ADE. There is no direct access to a frame's future target or its future ADE in the inference API. `sequence`, `track_id`, `first_observed_frame`, and `horizon_frames` are not numerical model features; horizon is only used to pick the separately fitted expert head.

For development fitting, each animal's windows begin at indices divisible by `20 + horizon` (40 or 70), preventing **within-track overlap of full observation+future intervals** in the fitting subset. This does not make the tracklets or the recording independent, and in a single shared scene animals may be correlated. Models are not tuned on a holdout folder in a given fold; there is no random-window train/test partition.

The fixed ridge design and its hyperparameters were **chosen after viewing the two recordings' earlier performance**, so this is exploratory method construction—not preregistered validation.

## Retrospective leave-one-recording-out stress test

For each horizon, one entire named recording is excluded from selector fitting and evaluated, then the direction is reversed. The held-out recording's ADE/FDE never enters that fold's fitting algorithm. However, both source recordings had already been inspected and reported in Phases 7F–7I, so neither is an **untouched** independent external test set. Differing folder names do not prove distinct flights/herds/capture events.

| Evaluation folder | Future frames | N overlapping eval windows | Stationary ADE | CV-OLS ADE | Transformer ADE | Adaptive selector ADE | Fixed best-on-training ADE | Hindsight per-window oracle ADE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `DJI_0207` | 20 | 2,487 | 0.004827 | 0.004911 | **0.004522** | 0.004911 | 0.004911 | 0.003760 |
| `DJI_0207` | 50 | 2,397 | **0.008416** | 0.009928 | 0.009539 | 0.009800 | 0.009928 | 0.006376 |
| `DJI_0204_video2` | 20 | 1,272 | 0.004056 | **0.001246** | 0.002752 | 0.002736 | 0.002752 | 0.001180 |
| `DJI_0204_video2` | 50 | 1,152 | 0.009768 | **0.002271** | 0.005248 | 0.009766 | 0.009768 | 0.002130 |

The adaptive selector **does not outperform the best individual forecasting expert in any of the four recording-by-horizon comparisons**. It often nearly reproduces the training recording's best fixed expert, which is precisely the problematic cross-recording transfer behaviour this diagnostic was designed to expose. The dynamic gate did improve slightly over *the expert selected as globally best on its one training recording* in some folds, but this improvement is not robust and does not overcome the best actual expert on the evaluation recording.

The hindsight oracle is **not a method**. It uses future target ADE to choose the best expert on each window and is included only as an unattainable lower bound on *per-window ADE*. No oracle FDE selection metric is reported.

### Which models did the gate select?

| Evaluation folder | Horizon | % Stationary | % CV-OLS | % Transformer |
|---|---:|---:|---:|---:|
| `DJI_0207` | 20 | 0.00 | 99.88 | 0.12 |
| `DJI_0207` | 50 | 0.00 | 74.13 | 25.87 |
| `DJI_0204_video2` | 20 | 7.31 | 5.50 | 87.19 |
| `DJI_0204_video2` | 50 | 99.39 | 0.52 | 0.09 |

This behaviour documents **large sequence-dependent gate failure**. It may reflect large camera/annotation/frame-cadence shifts, strong differences in training outcomes, limited development recordings, insufficiently transferable features, model misspecification, or combinations. The retrospective analysis **cannot identify the cause**.

## Final development-only model candidates

`results/locked_models/phase7j_dev_gate_h20.json` and `..._h50.json` refit exactly the same fixed ridge rule on both previously inspected **development** recordings. The 20-frame head uses 95 nonoverlapping-within-track training windows; the 50-frame head uses 56. These weights are frozen reproducible **candidate gates** for a new prospective protocol; **no independent performance estimate for the two-source-refit candidate exists**. Do not present training performance or the two-fold retrospective stress test as independent validation of this refit.

`src/phase7j.py predict --model ... --features ...` consumes only observation feature columns plus horizon; it does not need or consult `*_ade_norm` or `*_fde_norm` target columns. To calculate actual forecast errors later, new camera-frame-based prediction trajectories and future elephant annotations are required. Inference output here is only the selected expert name, **not a newly generated trajectory**.

## Implications for the paper

- **Method implementation**: PASS — reproducible candidate selector exists; no future-ground-truth leakage in prediction inputs; missing turning angles handled explicitly; group-blocked retrospective diagnostic available.
- **Claim of improved forecasting**: NOT SUPPORTED — gate loses to the strongest fixed baseline in all four tested conditions.
- **Claim of cross-recording/capture-event generalization**: NOT ESTABLISHED — only two pre-examined sequence folders with unverified independence.
- **Claim of improved conformal calibration, virtual geofence alert utility, or physical real-time safety**: NOT TESTED.

A Q1/Q2 contribution could arise from a revised hypothesis and a demonstrably successful locked method on **fresh, independently documented** recordings; it cannot be asserted from the present negative adaptive-selector result. Development of other rules is permissible but must be labeled method development, and every new tuning/choice must be frozen before an untouched test.

## Next locked-protocol steps (Phase 7K)

1. Preserve the two source recordings as **previously inspected development material**; do not retroactively relabel either as untouched confirmation.
2. Record the final selector JSON hashes, original Transformer six-checkpoint SHA-256, preprocessing version, fixed baselines, feature definitions, and eligibility rules *before* examining candidate test outcomes.
3. Obtain additional elephant-labelled sequences with reliable source/session identifiers. Confirm absence of flight/herd duplication with WildBox and development sources as far as evidence permits. Frame-based tests may proceed if FPS is unknown; seconds-based claims cannot.
4. Run the original stationary, CV-OLS, and frozen Transformer predictions on the **same valid windows** and apply the gate using observation-only features. Report results separately for each verified capture group and each forecast horizon, including selection counts, ADE/FDE and adverse failures. Do not pool overlapping windows as independent statistical units.
5. Do not deploy the candidate gate or advertise an accuracy gain merely because the software tests pass. If the prospective gate fails, report the failure rather than tune on the test set.

## Reproducibility

From the extracted kit root, install `numpy pandas matplotlib`, then run:

```bash
python src/test_phase7j.py
python src/phase7j.py develop \
  --phase7i /mnt/c/Users/HP/Downloads/WildLive_Phase7I_Reproduced_Results.zip \
  --output ./local_phase7j_results
```

Outputs: per-window out-of-recording scores, per-track summaries, horizon/sequence summaries, selection fractions, a comparison figure, SHA-256 manifest, and **two explicitly development-only** refitted gate JSONs. Source data were not rewritten. The package does not contain WildLive raw imagery or original model checkpoints.
