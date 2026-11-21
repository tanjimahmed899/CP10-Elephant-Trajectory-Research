# CP1.0 Research Strengthening — Stage 1 (20 November 2026)

**Scope:** Direct analysis of user-supplied `EXP3_Output_Review.zip`, `All Python Code.zip`, and `CP10_real_frames_labels_sample.zip`.  The output is retrospective methods development, **not new independent external confirmation**. No architecture was retrained, and no YOLO checkpoint was executed.

## EXP3 frozen-baseline reproduction

The exact frozen 2 s / 5 s prediction arrays were loaded and matched with ground truth, normalized residual scales and source metadata. Reproduced simultaneous q90 pathwise coverage:

| Horizon | Windows | q90 | Reproduced coverage | Paper coverage |
|---|---:|---:|---:|---:|
| 2 s | 754 | 2.158768 | 96.5517% | 96.55% |
| 5 s | 572 | 1.975130 | 69.7552% | 69.76% |

The warning replay reproduces the frozen Step18/Step7H combined N=2 rule, using directional virtual boundaries x/y=0.25, 0.50, 0.75 and persistence grouped by source, track, axis, boundary and direction, ordered by sample index. Values:

| Horizon | Nominal recall/FWR | N=2 recall/FWR | Reported paper values |
|---|---|---|---|
| 2 s | 33.1210% / 1.6865% | 86.6242% / 47.6247% | 86.62% / 47.62% |
| 5 s | 19.4672% / 1.1209% | 79.5082% / 40.9307% | 79.51% / 40.93% |

This explicitly resolves the prior alternative N=2 replay mismatch. **Do not confuse this reproduction with the novelty contribution.**

## EXP4: 2 s and 5 s leave-one-recording-group-out calibration

For each held-out elephant recording group (elep1/elep2/elep3), fitting uses only windows from the other two groups. Threshold candidates are prespecified in the code: frozen TrainReal q90, pooled-window empirical 90th percentile, video-equal-weight empirical percentile, track-equal-weight percentile, maximum of per-calibration-group empirical percentile, and 2-bin observed-history speed-conditional percentiles. Results are pooled over the three held-out groups; only **three independent recording groups** are available, and some videos were previously inspected.

| Method | 2 s coverage | 2 s mean-radius ratio | 5 s coverage | 5 s mean-radius ratio |
|---|---:|---:|---:|---:|
| Original frozen TrainReal | 96.55% | 1.00x | 69.76% | 1.00x |
| Target-window quantile | 88.20% | 0.60x | 88.99% | 1.77x |
| Video-equal weight | 88.99% | 0.61x | 88.81% | 1.76x |
| Track-equal weight | 88.46% | 0.59x | 89.34% | 1.78x |
| Group-worst empirical quantile | 88.59% | 0.61x | 90.21% | 1.86x |
| Two speed bins | 88.73% | 0.56x | 84.27% | 1.33x |

The radius ratio is computed using the **same weighted mean of per-window absolute region radii** for every method, unlike an earlier median-radius comparison. Coverage is the fraction of full future paths covered. Group-worst 5 s coverage above 90% occurs at 1.86x the mean region radius, with worst held-out group coverage only 86.63%. Neither target-quantile method nor speed-bin method establishes any cross-domain finite-sample coverage guarantee under overlapping dependent windows. Not all methods improve both horizons. The speed-conditioned method does **not** improve over simple global recalibration here. Do not add a claim of novel adaptive conformal theory or solved undercoverage.

## EXP5: exact frozen-rule replay and retrospective operating curves

The original N=2 rule is reconstructed exactly at multiplier=1, and candidate q multipliers are swept. These curves are *post-hoc diagnostic*, not optimized test-set results.

For example, at multiplier 0.5 with N=2:

| Horizon | Recall | FWR | F1 | Deduplicated synthetic alert episodes |
|---|---:|---:|---:|---:|
| 2 s | 77.71% | 24.23% | 0.309 | 370 |
| 5 s | 61.89% | 19.29% | 0.445 | 263 |

The alert episodes use **per-track, per-boundary, per-direction runs of consecutive eligible windows**. This rate is not alerts per hour or real incident frequency, and cannot substitute for physical warning episodes.

A stricter exploratory group-held-out decision experiment selects a **fixed N=2** multiplier on the other two elephant groups, maximizing calibration group-macro recall subject to the worst calibration-group FWR <= 10% or <=20%. At the 20% budget:

| Horizon | Held-out groups satisfying 20% budget |
|---|---|
| 2 s | 2/3 |
| 5 s | 3/3 |

Even this empirical budget is **not conformal risk control**: test-group violations occur, the groups are not an exchangeable population validated for guarantees, and operator/field costs are unknown. Neither a guaranteed false-warning target nor guaranteed lead time is established.

## R8 geometry input preflight

The separate supplied WildBox `seg1` sample contains 75 RGB images, all 75 with matching KITTI labels and camera records, plus 200 camera poses in its full `cameras.json`. Intrinsic and extrinsic matrices are structurally present and have near-orthogonal rotations. Source translation has unverified scale, and camera coordinate semantics/ground depth remain to be checked. This is not the exact-geometry BIRDSAI R8 ablation, and the 75 images are *not* from the six EXP3 videos.

## Files and provenance

- `run_stage1.py` — reproduces frozen coverage and exact N=2 recall/FWR; explores group-held-out calibration and constrained warning selection
- `EXP3_Reproduction_Audit.json` — baseline checks, input ZIP SHA-256 and sample counts
- `EXP4_*` — 2s/5s group and video metrics, plus summary and plot
- `EXP5_*` — 2s/5s operating curves, group-specific sweeps and held-out budgets, plus plot
- `R8_geometry_preflight.py`, `R8_*` — original WildBox camera/annotation structural audit
- `Source_Method_Script_Hashes.json` — hashes of archived original Step18 and Step7H method code
- `collect_revision_reports_wsl.py` — optional size-bounded collector for R1–R19 finished result files on the user's own E: drive

To reproduce Stage 1 in a Python environment with `numpy`, `pandas`, `matplotlib`:

```bash
python run_stage1.py --zip /path/to/EXP3_Output_Review.zip --out /path/to/stage1_results
python R8_geometry_preflight.py --zip /path/to/CP10_real_frames_labels_sample.zip --out /path/to/stage1_results
```

## Next mandatory steps before manuscript claims

1. **Audit existing R1–R19 revised results** before rerunning any older R4/R6/R8/R13 experiments. Run `python3 collect_revision_reports_wsl.py` in WSL (no third-party dependencies) and upload the resulting ZIP. This avoids duplicate work and checks whether results already exist.
2. **EXP4**: design a *fresh* adaptation/confirmatory split using independently acquired recording groups. Compare the empirical window, track, video and conditional procedures, and include region size and group-specific coverage. New calibration needs enough independent source groups, not merely many overlapping windows.
3. **EXP5**: use genuine decision-focused calibration with a defensible risk definition and assumptions, episode-level events and response lead times. No alert policy is certified by the present synthetic six-boundary sweep.
4. **R8**: verify pose conventions, ground-plane/depth assumptions, complete camera projective equivalence and source files; train/evaluate learned raw-history and corrected-history predictors using the same target. Cannot complete this using only the 75 WildBox sample images and saved EXP3 outputs.
5. **R13**: obtain the relevant detector checkpoint and its source-domain images and tracking logs for a genuine detector→tracker→forecast experiment. A small annotated image sample alone does not establish end-to-end throughput.

**Do not silently revise the submitted paper's abstract** until independent follow-up results are generated and checked.
