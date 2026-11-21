# CP10 Stage 7F — Original frozen Transformer on GT-verified tracker histories

## What is evaluated

This is the **first actual downstream forecasting experiment from the Stage7D adapted detector / ByteTrack tracker**. Unlike Stage7E (feasibility only), Stage7F executes the original 3-seed frozen Transformer checkpoints. New training, test threshold optimization, YOLO inference and ByteTrack inference are **not** performed.

Strict Stage7E eligibility is fixed before predictions are computed: 187 2-second and 124 5-second forecast windows (all sequence `0000000353_0000000000`). This selection is **annotation-assisted retrospective analysis**; real deployments cannot use future GT and oracle track identities to select safe windows. No original detector windows passed the strict gate, so there is **no paired original-detector forecasting ADE/FDE comparison**.

Each history input is built **only from Stage7D tracker box centers**, not GT centers. Original BIRDSAI image frame coordinates are transformed with the saved **R8 2D affine camera-motion estimates** to the beginning-of-observation reference frame; the original model canonicalization is then used (heading based on 10-history-point slope, path-length scale floor 0.06820264416877736 for 2s or 0.06732638773715147 for 5s). The ground-truth future is extracted separately and used only to calculate ADE/FDE in the same observation-start frame. An optional **oracle-GT-history diagnostic arm** uses the same frozen model on perfect observed GT positions to quantify history quality loss; this is never a deployed capability.

For every horizon, three pre-existing checkpoint seeds (42, 52, 62) are used. Results include seed-wise and three-seed ensemble ADE/FDE plus a stationary baseline, tracker-to-GT input discrepancy, elephant-cluster aggregation, and saved prediction arrays. Error units are **normalized image coordinates**, never metres.

## WSL run

1. Download `CP10_Stage7F_Frozen_Tracker_Forecasting.zip` to `C:\Users\HP\Downloads`.
2. Make sure the **successful** `CP10_Stage7D_Tracker_Review.zip` and `CP10_Stage7E_GT_Linkage_Review.zip` are in that Downloads folder. The runner verifies SHA-256 against the Stage7E manifest and fails on a stale archive.
3. Run:

```bash
mkdir -p ~/cp10_stage7f_tools
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage7F_Frozen_Tracker_Forecasting.zip" ~/cp10_stage7f_tools
bash ~/cp10_stage7f_tools/CP10_Stage7F_Frozen_Tracker_Forecasting/run_stage7f_wsl.sh
```

Outputs: `~/CP10_Stage7F_Tracker_Forecast_Review.zip` and a copy in Windows Downloads. **Upload this ZIP**, even if failure occurs; diagnostics will be included.

## Verifications

- The underlying Stage7D ZIP hash and raw GT dataset hash **must match** Stage7E provenance.
- Strict sample counts **must equal 187 and 124**, from the locked source sequence; program terminates on changes.
- Every 10 Hz tracker box must have the matching previously audited GT association, but GT is never substituted into the tracker input.
- Window future GT must be complete. Missing/invalid R8 motion estimates result in hard failure instead of silently substituting identity transforms.
- Source resolution and GT center coordinates are checked; original frozen checkpoint filenames/hashes are included in the results.
- End-to-end inference here is **conditional forecasting from precomputed detector/tracker histories**, not the full end-to-end yield or a field-deployable result.
- Results are retrospective on an earlier-used BIRDSAI TestReal sequence; overlapping windows are not independent samples and must not be used as 311 independent subjects in significance tests.

## What passed locally before shipping

- Stage7E result/provenance CRC and corresponding Stage7D hash verified.
- 308/311 strict samples could be rebuilt from the **partial** earlier-collected Stage4 raw GT set (3 lacked raw rows in that partial subset); all 308 passed coordinate and association checks. The full original GT file in WSL is required for all 311.
- A selected original BIRDSAI R8 `history_abs` / `future_abs` trajectory rebuilt from saved affines matched numerical ground truth to within 3e-8 normalized coordinates. Canonicalized history matched within 8e-7, under both horizons.
- The shipped Torch checkpoint-loader/evaluator is based on the original completed Stage5 frozen inference implementation; local code does not include the user's checkpoint files, so **real 3-seed model evaluation remains to be executed in WSL**.
