from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# STEP 4: CLUSTERED BOOTSTRAP + PAIRED SIGNIFICANCE
# All NEW outputs -> Downloads only
# Existing Main Result files are READ ONLY
# ============================================================

ROOT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
FRESH_DIR = ROOT / "wildbox_fresh_confirmatory_results"
TRANSFORMER_DIR = ROOT / "transformer_results"
STEP3_DIR = Path(r"C:\Users\HP\Downloads\step3_fresh_wildbox_full_comparison")
OUT = Path(r"C:\Users\HP\Downloads\step4_clustered_bootstrap")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
SEEDS = [42, 52, 62]
N_BOOT = 10000
N_PERM = 50000
RNG_SEED = 20261026
METRIC_MATCH_TOL = 2e-4

TASKS = {
    "2s_to_2s": {"future_len": 20, "expected_n": 865},
    "2s_to_5s": {"future_len": 50, "expected_n": 637},
}

# ============================================================
# HELPERS
# ============================================================

def ade_fde(pred, truth):
    e = np.linalg.norm(pred - truth, axis=2)
    return e.mean(axis=1), e[:, -1]

def cv_predict(history, future_len, fps=10.0):
    n = history.shape[1]
    t = np.arange(n, dtype=float) / fps
    tc = t - t.mean()
    denom = np.sum(tc**2)
    out = []
    ft = np.arange(1, future_len + 1, dtype=float) / fps
    for h in history:
        pc = h - h.mean(axis=0, keepdims=True)
        v = (tc[:, None] * pc).sum(axis=0) / max(denom, 1e-12)
        out.append(h[-1][None, :] + ft[:, None] * v[None, :])
    return np.asarray(out)

def stationary_predict(history, future_len):
    return np.repeat(history[:, -1:, :], future_len, axis=1)

def inverse_transform_batch(canonical, origin, angle, scale):
    output = np.empty_like(canonical, dtype=np.float64)
    for i in range(len(canonical)):
        a = float(angle[i])
        c, s = np.cos(a), np.sin(a)
        # row-vector rotation back to absolute frame
        R = np.asarray([[c, s], [-s, c]], dtype=np.float64)
        output[i] = canonical[i] * float(scale[i]) @ R + origin[i]
    return output

# ============================================================
# TRANSFORMER RECONSTRUCTION
# We search a small set of plausible forward variants and
# REQUIRE numerical agreement with the already-saved Fresh
# WildBox metrics before continuing.
# ============================================================

class SavedPosition(nn.Module):
    def __init__(self, pe_shape):
        super().__init__()
        self.register_buffer("pe", torch.zeros(pe_shape, dtype=torch.float32))

    def forward(self, x):
        # x is always [B,T,D]
        T = x.shape[1]
        pe = self.pe

        if pe.ndim == 3:
            if pe.shape[0] == 1 and pe.shape[1] >= T:
                return x + pe[:, :T, :]
            if pe.shape[1] == 1 and pe.shape[0] >= T:
                return x + pe[:T, :, :].transpose(0, 1)
            if pe.shape[0] >= T and pe.shape[1] == x.shape[2]:
                return x + pe[:T, :].unsqueeze(0)

        if pe.ndim == 2 and pe.shape[0] >= T:
            return x + pe[:T, :].unsqueeze(0)

        raise RuntimeError(f"Unsupported positional buffer shape: {tuple(pe.shape)}")

class RebuiltTransformer(nn.Module):
    def __init__(self, ckpt, future_len, pooling="last",
                 norm_first=False, encoder_activation="relu",
                 head_activation="relu"):
        super().__init__()

        sd = ckpt["state_dict"]
        d_model = int(ckpt["d_model"])
        nhead = int(ckpt["nhead"])
        num_layers = int(ckpt["num_layers"])
        dim_ff = int(ckpt["dim_feedforward"])
        dropout = float(ckpt["dropout"])

        input_in = int(sd["input_projection.weight"].shape[1])
        self.input_projection = nn.Linear(input_in, d_model)

        pe_shape = tuple(sd["position.pe"].shape)
        self.position = SavedPosition(pe_shape)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=encoder_activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        head_w0 = sd["head.0.weight"]
        head_w3 = sd["head.3.weight"]
        head_in = int(head_w0.shape[1])
        head_hidden = int(head_w0.shape[0])
        head_out = int(head_w3.shape[0])

        if head_out != future_len * 2:
            raise RuntimeError(
                f"Checkpoint output {head_out} != expected future_len*2={future_len*2}"
            )

        if head_activation == "relu":
            act = nn.ReLU()
        elif head_activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(head_activation)

        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),   # head.0
            act,                               # head.1
            nn.Dropout(dropout),               # head.2
            nn.Linear(head_hidden, head_out),  # head.3
        )

        self.pooling = pooling
        self.d_model = d_model
        self.head_in = head_in
        self.future_len = future_len

    def forward(self, x):
        x = self.input_projection(x)
        x = self.position(x)
        x = self.encoder(x)
        x = self.norm(x)

        if self.head_in == self.d_model:
            if self.pooling == "last":
                z = x[:, -1, :]
            elif self.pooling == "mean":
                z = x.mean(dim=1)
            elif self.pooling == "first":
                z = x[:, 0, :]
            else:
                raise ValueError(self.pooling)
        elif self.head_in == self.d_model * x.shape[1]:
            z = x.reshape(x.shape[0], -1)
        else:
            raise RuntimeError(
                f"Cannot infer pooling: head_in={self.head_in}, "
                f"d_model={self.d_model}, T={x.shape[1]}"
            )

        return self.head(z).view(-1, self.future_len, 2)

def predict_torch(model, X, batch_size=256):
    parts = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.tensor(X[s:s+batch_size], dtype=torch.float32, device=DEVICE)
            parts.append(model(xb).cpu().numpy())
    return np.concatenate(parts, axis=0)

def load_expected_transformer_metrics(task):
    p = FRESH_DIR / "fresh_holdout_model_metrics.csv"
    df = pd.read_csv(p)
    return df[df["task"] == task].copy()

def choose_transformer_variant(task, arrays, metadata):
    expected = load_expected_transformer_metrics(task)
    future_len = TASKS[task]["future_len"]

    X = np.asarray(arrays["history_canonical"], dtype=np.float32)
    truth_abs = np.asarray(arrays["future_abs"], dtype=np.float64)
    origin = np.asarray(arrays["origin"], dtype=np.float64)
    angle = np.asarray(arrays["heading_angle"], dtype=np.float64)
    scale = np.asarray(arrays["scale"], dtype=np.float64)

    candidates = []
    for pooling in ["last", "mean", "first"]:
        for norm_first in [False, True]:
            for enc_act in ["relu", "gelu"]:
                for head_act in ["relu", "gelu"]:
                    candidates.append((pooling, norm_first, enc_act, head_act))

    best = None

    for cand in candidates:
        pooling, norm_first, enc_act, head_act = cand
        total_abs_error = 0.0
        seed_predictions = {}
        failed = False

        for seed in SEEDS:
            ckpt_path = TRANSFORMER_DIR / f"{task}_Transformer_seed{seed}.pt"
            ckpt = torch.load(ckpt_path, map_location=DEVICE)

            try:
                model = RebuiltTransformer(
                    ckpt,
                    future_len,
                    pooling=pooling,
                    norm_first=norm_first,
                    encoder_activation=enc_act,
                    head_activation=head_act,
                ).to(DEVICE)
                model.load_state_dict(ckpt["state_dict"], strict=True)
                pred_c = predict_torch(model, X)
                pred_abs = inverse_transform_batch(pred_c, origin, angle, scale)
                a, f = ade_fde(pred_abs, truth_abs)
            except Exception:
                failed = True
                break

            seed_predictions[seed] = pred_abs

            row = expected[expected["model"] == f"Transformer_seed{seed}"]
            if len(row) != 1:
                raise RuntimeError(f"Missing expected metric row for {task} seed {seed}")

            exp_ade = float(row.iloc[0]["ADE"])
            exp_fde = float(row.iloc[0]["FDE"])
            total_abs_error += abs(a.mean() - exp_ade) + abs(f.mean() - exp_fde)

        if failed:
            continue

        if best is None or total_abs_error < best["score"]:
            best = {
                "candidate": cand,
                "score": total_abs_error,
                "seed_predictions": seed_predictions,
            }

    if best is None:
        raise RuntimeError(f"Could not reconstruct Transformer for {task}")

    pooling, norm_first, enc_act, head_act = best["candidate"]
    print(
        f"{task}: selected Transformer variant "
        f"pool={pooling}, norm_first={norm_first}, "
        f"enc_act={enc_act}, head_act={head_act}, "
        f"match_score={best['score']:.10f}"
    )

    # Final strict numerical self-check
    check_rows = []
    for seed in SEEDS:
        pred = best["seed_predictions"][seed]
        a, f = ade_fde(pred, truth_abs)

        row = expected[expected["model"] == f"Transformer_seed{seed}"].iloc[0]
        ade_diff = abs(a.mean() - float(row["ADE"]))
        fde_diff = abs(f.mean() - float(row["FDE"]))

        check_rows.append({
            "task": task,
            "model": f"Transformer_seed{seed}",
            "regenerated_ADE": a.mean(),
            "expected_ADE": float(row["ADE"]),
            "ADE_abs_diff": ade_diff,
            "regenerated_FDE": f.mean(),
            "expected_FDE": float(row["FDE"]),
            "FDE_abs_diff": fde_diff,
        })

        if ade_diff > METRIC_MATCH_TOL or fde_diff > METRIC_MATCH_TOL:
            raise RuntimeError(
                f"Transformer self-check FAILED for {task} seed {seed}: "
                f"ADE diff={ade_diff}, FDE diff={fde_diff}. "
                f"Do not use bootstrap results."
            )

    ensemble_pred = np.mean(
        np.stack([best["seed_predictions"][s] for s in SEEDS], axis=0),
        axis=0,
    )
    ea, ef = ade_fde(ensemble_pred, truth_abs)

    erow = expected[expected["model"] == "Transformer_3Seed_Ensemble"]
    if len(erow) == 1:
        erow = erow.iloc[0]
        ade_diff = abs(ea.mean() - float(erow["ADE"]))
        fde_diff = abs(ef.mean() - float(erow["FDE"]))
        check_rows.append({
            "task": task,
            "model": "Transformer_3Seed_Ensemble",
            "regenerated_ADE": ea.mean(),
            "expected_ADE": float(erow["ADE"]),
            "ADE_abs_diff": ade_diff,
            "regenerated_FDE": ef.mean(),
            "expected_FDE": float(erow["FDE"]),
            "FDE_abs_diff": fde_diff,
        })
        if ade_diff > METRIC_MATCH_TOL or fde_diff > METRIC_MATCH_TOL:
            raise RuntimeError(
                f"Transformer ensemble self-check FAILED for {task}: "
                f"ADE diff={ade_diff}, FDE diff={fde_diff}."
            )

    return best["seed_predictions"], ensemble_pred, pd.DataFrame(check_rows)

# ============================================================
# BUILD ALL PER-SAMPLE MODEL ERRORS
# ============================================================

def aggregate_step3_seed_errors(step3, task, base_model):
    d = step3[
        (step3["task"] == task) &
        (step3["model"] == base_model)
    ].copy()

    if len(d) == 0:
        raise RuntimeError(f"No Step-3 sample rows for {task} {base_model}")

    g = (
        d.groupby(["sample_id", "source_group", "track_key"], as_index=False)
         .agg(ADE=("ADE", "mean"), FDE=("FDE", "mean"))
    )
    g["model"] = f"{base_model}_3Seed_Mean"
    g["task"] = task
    return g

def build_model_error_table(task, arrays, metadata, transformer_ensemble_pred):
    truth = np.asarray(arrays["future_abs"], dtype=np.float64)
    hist = np.asarray(arrays["history_abs"], dtype=np.float64)
    future_len = TASKS[task]["future_len"]

    base = metadata[["sample_id", "source_group", "track_key"]].copy()
    base["sample_id"] = base["sample_id"].astype(int)

    outputs = []

    # Stationary
    pred = stationary_predict(hist, future_len)
    a, f = ade_fde(pred, truth)
    x = base.copy()
    x["task"] = task
    x["model"] = "Stationary"
    x["ADE"] = a
    x["FDE"] = f
    outputs.append(x)

    # Constant Velocity
    pred = cv_predict(hist, future_len)
    a, f = ade_fde(pred, truth)
    x = base.copy()
    x["task"] = task
    x["model"] = "ConstantVelocity"
    x["ADE"] = a
    x["FDE"] = f
    outputs.append(x)

    # Transformer ensemble
    a, f = ade_fde(transformer_ensemble_pred, truth)
    x = base.copy()
    x["task"] = task
    x["model"] = "Transformer_3Seed_Ensemble"
    x["ADE"] = a
    x["FDE"] = f
    outputs.append(x)

    # Step-3 saved per-sample errors
    step3_file = STEP3_DIR / "step3_new_model_sample_metrics.csv"
    step3 = pd.read_csv(
        step3_file,
        dtype={"source_group": str, "track_key": str, "seed": str}
    )

    # Kalman fixed
    k = step3[
        (step3["task"] == task) &
        (step3["model"] == "KalmanCV")
    ][["sample_id", "source_group", "track_key", "ADE", "FDE"]].copy()
    k["task"] = task
    k["model"] = "KalmanCV"
    outputs.append(k)

    # GRU/LSTM 3-seed mean errors
    outputs.append(aggregate_step3_seed_errors(step3, task, "GRU"))
    outputs.append(aggregate_step3_seed_errors(step3, task, "LSTM"))

    out = pd.concat(outputs, ignore_index=True)

    expected_models = [
        "Stationary",
        "ConstantVelocity",
        "KalmanCV",
        "GRU_3Seed_Mean",
        "LSTM_3Seed_Mean",
        "Transformer_3Seed_Ensemble",
    ]

    for model in expected_models:
        d = out[out["model"] == model]
        if len(d) != TASKS[task]["expected_n"]:
            raise RuntimeError(
                f"{task} {model}: sample count {len(d)} "
                f"!= {TASKS[task]['expected_n']}"
            )

    return out

# ============================================================
# CLUSTERED BOOTSTRAP
# Cluster = track_key
# We report both:
# 1) window-weighted mean metric
# 2) track-macro metric
# ============================================================

def prepare_cluster_stats(df, metric):
    # one model only
    g = (
        df.groupby("track_key")[metric]
          .agg(["sum", "count", "mean"])
          .reset_index()
    )
    return g

def bootstrap_model_ci(df, metric, rng, n_boot=N_BOOT):
    g = prepare_cluster_stats(df, metric)
    n_tracks = len(g)

    sums = g["sum"].to_numpy(float)
    counts = g["count"].to_numpy(float)
    means = g["mean"].to_numpy(float)

    boot_window = np.empty(n_boot, dtype=float)
    boot_track = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n_tracks, size=n_tracks)

        boot_window[b] = (
            sums[idx].sum()
            /
            counts[idx].sum()
        )

        boot_track[b] = means[idx].mean()

    observed_window = df[metric].mean()
    observed_track = means.mean()

    return {
        "observed_window": observed_window,
        "window_CI_low": np.quantile(boot_window, 0.025),
        "window_CI_high": np.quantile(boot_window, 0.975),
        "observed_track_macro": observed_track,
        "track_CI_low": np.quantile(boot_track, 0.025),
        "track_CI_high": np.quantile(boot_track, 0.975),
    }

def paired_cluster_bootstrap(all_errors, task, comparator, metric, rng, n_boot=N_BOOT):
    ens_name = "Transformer_3Seed_Ensemble"

    e = all_errors[
        (all_errors["task"] == task) &
        (all_errors["model"] == ens_name)
    ][["sample_id", "track_key", metric]].rename(columns={metric: "ens"})

    c = all_errors[
        (all_errors["task"] == task) &
        (all_errors["model"] == comparator)
    ][["sample_id", "track_key", metric]].rename(columns={metric: "cmp"})

    m = e.merge(
        c[["sample_id", "cmp"]],
        on="sample_id",
        how="inner",
        validate="one_to_one"
    )

    if len(m) != TASKS[task]["expected_n"]:
        raise RuntimeError(
            f"Paired sample mismatch {task} {comparator}: {len(m)}"
        )

    # per cluster: sums and counts for both models, plus mean difference
    cluster = (
        m.groupby("track_key")
         .agg(
             ens_sum=("ens", "sum"),
             cmp_sum=("cmp", "sum"),
             count=("ens", "count"),
             ens_mean=("ens", "mean"),
             cmp_mean=("cmp", "mean"),
         )
         .reset_index()
    )

    n_tracks = len(cluster)
    ens_sum = cluster["ens_sum"].to_numpy(float)
    cmp_sum = cluster["cmp_sum"].to_numpy(float)
    counts = cluster["count"].to_numpy(float)
    track_diff = (
        cluster["ens_mean"].to_numpy(float)
        -
        cluster["cmp_mean"].to_numpy(float)
    )

    boot_window_delta = np.empty(n_boot, dtype=float)
    boot_track_delta = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n_tracks, size=n_tracks)

        ens_w = ens_sum[idx].sum() / counts[idx].sum()
        cmp_w = cmp_sum[idx].sum() / counts[idx].sum()
        boot_window_delta[b] = ens_w - cmp_w

        boot_track_delta[b] = track_diff[idx].mean()

    observed_window_delta = m["ens"].mean() - m["cmp"].mean()
    observed_track_delta = track_diff.mean()

    # Negative delta => Transformer ensemble lower error (better)
    p_ens_better_window = float(np.mean(boot_window_delta < 0))
    p_ens_better_track = float(np.mean(boot_track_delta < 0))

    approx_bootstrap_p_window = min(
        1.0,
        2.0 * min(
            np.mean(boot_window_delta <= 0),
            np.mean(boot_window_delta >= 0),
        )
    )

    approx_bootstrap_p_track = min(
        1.0,
        2.0 * min(
            np.mean(boot_track_delta <= 0),
            np.mean(boot_track_delta >= 0),
        )
    )

    return {
        "task": task,
        "comparator": comparator,
        "metric": metric,
        "tracks": n_tracks,
        "samples": len(m),

        "ensemble_mean": m["ens"].mean(),
        "comparator_mean": m["cmp"].mean(),
        "window_delta_ensemble_minus_comparator": observed_window_delta,
        "window_delta_CI_low": np.quantile(boot_window_delta, 0.025),
        "window_delta_CI_high": np.quantile(boot_window_delta, 0.975),
        "P_ensemble_better_window": p_ens_better_window,
        "approx_two_sided_bootstrap_p_window": approx_bootstrap_p_window,

        "ensemble_track_macro": cluster["ens_mean"].mean(),
        "comparator_track_macro": cluster["cmp_mean"].mean(),
        "track_delta_ensemble_minus_comparator": observed_track_delta,
        "track_delta_CI_low": np.quantile(boot_track_delta, 0.025),
        "track_delta_CI_high": np.quantile(boot_track_delta, 0.975),
        "P_ensemble_better_track": p_ens_better_track,
        "approx_two_sided_bootstrap_p_track": approx_bootstrap_p_track,
    }

# ============================================================
# PAIRED TRACK-LEVEL SIGN-FLIP TEST
# More defensible inferential p-value for track-macro difference.
# ============================================================

def signflip_pvalue(track_differences, rng, n_perm=N_PERM):
    d = np.asarray(track_differences, dtype=float)
    observed = abs(d.mean())

    count = 0
    batch = 5000
    done = 0

    while done < n_perm:
        m = min(batch, n_perm - done)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(m, len(d))
        )
        perm_means = np.abs((signs * d[None, :]).mean(axis=1))
        count += int(np.sum(perm_means >= observed))
        done += m

    return (count + 1) / (n_perm + 1)

def add_signflip_pvalues(pairwise, all_errors, rng):
    pvals = []

    for _, row in pairwise.iterrows():
        task = row["task"]
        comparator = row["comparator"]
        metric = row["metric"]

        e = all_errors[
            (all_errors["task"] == task) &
            (all_errors["model"] == "Transformer_3Seed_Ensemble")
        ][["sample_id", "track_key", metric]].rename(columns={metric: "ens"})

        c = all_errors[
            (all_errors["task"] == task) &
            (all_errors["model"] == comparator)
        ][["sample_id", metric]].rename(columns={metric: "cmp"})

        m = e.merge(c, on="sample_id", how="inner", validate="one_to_one")
        t = (
            m.groupby("track_key")
             .agg(ens=("ens", "mean"), cmp=("cmp", "mean"))
        )
        diff = (t["ens"] - t["cmp"]).to_numpy(float)

        pvals.append(signflip_pvalue(diff, rng))

    pairwise = pairwise.copy()
    pairwise["paired_track_signflip_p"] = pvals
    return pairwise

# ============================================================
# MAIN
# ============================================================

def main():
    rng = np.random.default_rng(RNG_SEED)

    all_error_tables = []
    self_checks = []

    # --------------------------------------------------------
    # Regenerate and verify Transformer predictions
    # --------------------------------------------------------
    for task, cfg in TASKS.items():
        print("\n" + "=" * 78)
        print("TASK:", task)
        print("=" * 78)

        dataset_file = FRESH_DIR / f"{task}_fresh_dataset.npz"
        metadata_file = FRESH_DIR / f"{task}_fresh_metadata.csv"

        arrays = np.load(dataset_file)
        metadata = pd.read_csv(
            metadata_file,
            dtype={"source_group": str, "track_key": str}
        )

        if len(metadata) != cfg["expected_n"]:
            raise RuntimeError(
                f"{task}: metadata rows {len(metadata)} != {cfg['expected_n']}"
            )

        print("Reconstructing and self-checking frozen Transformer...")
        seed_preds, ensemble_pred, check = choose_transformer_variant(
            task, arrays, metadata
        )
        self_checks.append(check)

        print("Transformer self-check PASSED.")

        # Save regenerated transformer per-sample metrics
        truth = np.asarray(arrays["future_abs"], dtype=np.float64)
        tr_rows = []

        for seed in SEEDS:
            a, f = ade_fde(seed_preds[seed], truth)
            for i in range(len(metadata)):
                tr_rows.append({
                    "task": task,
                    "sample_id": int(metadata.iloc[i]["sample_id"]),
                    "source_group": metadata.iloc[i]["source_group"],
                    "track_key": metadata.iloc[i]["track_key"],
                    "model": f"Transformer_seed{seed}",
                    "ADE": float(a[i]),
                    "FDE": float(f[i]),
                })

        a, f = ade_fde(ensemble_pred, truth)
        for i in range(len(metadata)):
            tr_rows.append({
                "task": task,
                "sample_id": int(metadata.iloc[i]["sample_id"]),
                "source_group": metadata.iloc[i]["source_group"],
                "track_key": metadata.iloc[i]["track_key"],
                "model": "Transformer_3Seed_Ensemble",
                "ADE": float(a[i]),
                "FDE": float(f[i]),
            })

        pd.DataFrame(tr_rows).to_csv(
            OUT / f"{task}_regenerated_transformer_sample_metrics.csv",
            index=False
        )

        # Build all comparison models
        all_error_tables.append(
            build_model_error_table(
                task, arrays, metadata, ensemble_pred
            )
        )

    self_check_df = pd.concat(self_checks, ignore_index=True)
    self_check_df.to_csv(
        OUT / "transformer_regeneration_self_check.csv",
        index=False
    )

    all_errors = pd.concat(all_error_tables, ignore_index=True)
    all_errors.to_csv(
        OUT / "all_model_sample_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Per-model clustered 95% CI
    # --------------------------------------------------------
    ci_rows = []

    for (task, model), d in all_errors.groupby(["task", "model"], sort=False):
        for metric in ["ADE", "FDE"]:
            res = bootstrap_model_ci(d, metric, rng)
            ci_rows.append({
                "task": task,
                "model": model,
                "metric": metric,
                "samples": len(d),
                "tracks": d["track_key"].nunique(),
                **res,
            })

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(
        OUT / "model_clustered_bootstrap_95CI.csv",
        index=False
    )

    # --------------------------------------------------------
    # Paired ensemble-vs-baseline comparisons
    # --------------------------------------------------------
    comparators = [
        "LSTM_3Seed_Mean",
        "GRU_3Seed_Mean",
        "ConstantVelocity",
        "KalmanCV",
        "Stationary",
    ]

    pair_rows = []

    for task in TASKS:
        for comparator in comparators:
            for metric in ["ADE", "FDE"]:
                pair_rows.append(
                    paired_cluster_bootstrap(
                        all_errors,
                        task,
                        comparator,
                        metric,
                        rng,
                    )
                )

    pairwise = pd.DataFrame(pair_rows)
    pairwise = add_signflip_pvalues(pairwise, all_errors, rng)
    pairwise.to_csv(
        OUT / "pairwise_vs_transformer_ensemble.csv",
        index=False
    )

    # --------------------------------------------------------
    # Compact ADE table for terminal
    # --------------------------------------------------------
    ade_ci = ci_df[ci_df["metric"] == "ADE"].copy()
    ade_ci = ade_ci.sort_values(
        ["task", "observed_window"],
        ascending=[True, True]
    )

    # --------------------------------------------------------
    # Summary text
    # --------------------------------------------------------
    with open(OUT / "SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("STEP 4: CLUSTERED BOOTSTRAP + PAIRED SIGNIFICANCE\n")
        f.write("=" * 78 + "\n\n")
        f.write(
            "Protocol:\n"
            "- Same locked Fresh WildBox holdout.\n"
            "- Cluster = track_key; all windows from a track are resampled together.\n"
            "- 10,000 clustered bootstrap replicates.\n"
            "- 95% percentile confidence intervals.\n"
            "- Pairwise delta = Transformer ensemble - comparator; negative is better.\n"
            "- Paired track-level sign-flip test provides the inferential p-value.\n"
            "- Bootstrap tail probabilities are also reported descriptively.\n\n"
        )

        f.write("TRANSFORMER REGENERATION SELF-CHECK\n")
        f.write("-----------------------------------\n")
        f.write(self_check_df.to_string(index=False))
        f.write("\n\nMODEL ADE 95% CI\n")
        f.write("----------------\n")
        f.write(ade_ci.to_string(index=False))
        f.write("\n\nPAIRWISE COMPARISONS\n")
        f.write("--------------------\n")
        f.write(pairwise.to_string(index=False))
        f.write(
            "\n\nInterpretation caution:\n"
            "A non-significant difference does not prove equivalence; it means the "
            "locked holdout does not provide strong evidence of a difference under "
            "this paired track-level analysis.\n"
        )

    print("\nSTEP 4 COMPLETE")
    print("Outputs:", OUT)

    print("\nTRANSFORMER SELF-CHECK:")
    print(self_check_df.to_string(index=False))

    print("\nADE CLUSTERED 95% CI:")
    print(
        ade_ci[
            [
                "task",
                "model",
                "observed_window",
                "window_CI_low",
                "window_CI_high",
                "observed_track_macro",
                "track_CI_low",
                "track_CI_high",
            ]
        ].to_string(index=False)
    )

    print("\nPAIRWISE ADE VS TRANSFORMER ENSEMBLE:")
    print(
        pairwise[pairwise["metric"] == "ADE"][
            [
                "task",
                "comparator",
                "window_delta_ensemble_minus_comparator",
                "window_delta_CI_low",
                "window_delta_CI_high",
                "P_ensemble_better_window",
                "track_delta_ensemble_minus_comparator",
                "track_delta_CI_low",
                "track_delta_CI_high",
                "P_ensemble_better_track",
                "paired_track_signflip_p",
            ]
        ].to_string(index=False)
    )

if __name__ == "__main__":
    main()
