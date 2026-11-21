from pathlib import Path
import importlib.util
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# STEP 4 v2
# Exact original Transformer class + clustered bootstrap
# ============================================================

ROOT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
FRESH_DIR = ROOT / "wildbox_fresh_confirmatory_results"
TRANSFORMER_DIR = ROOT / "transformer_results"
ORIGINAL_EVAL = ROOT / "evaluate_fresh_wildbox_holdout.py"
STEP3_DIR = Path(r"C:\Users\HP\Downloads\step3_fresh_wildbox_full_comparison")

OUT = Path(r"C:\Users\HP\Downloads\step4_clustered_bootstrap")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
SEEDS = [42, 52, 62]

N_BOOT = 10000
N_PERM = 50000
RNG_SEED = 20261026

# Existing saved metrics are printed to 6 decimals, so this is strict enough
# to verify that we regenerated the same model behavior.
MATCH_TOL = 5e-6

TASKS = {
    "2s_to_2s": {"future_len": 20, "expected_n": 865},
    "2s_to_5s": {"future_len": 50, "expected_n": 637},
}


# ============================================================
# EXTRACT ORIGINAL MODEL DEFINITIONS WITHOUT EXECUTING
# TOP-LEVEL HOLDOUT / PATH CHECKS
# ============================================================

def import_original_module():
    import ast
    import types
    import math

    if not ORIGINAL_EVAL.exists():
        raise FileNotFoundError(ORIGINAL_EVAL)

    source = ORIGINAL_EVAL.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(ORIGINAL_EVAL)
    )

    required_names = {
        "PositionalEncoding",
        "TrajectoryTransformer",
        "inverse_transform_batch",
    }

    selected_nodes = []

    # Keep only exact class/function definitions we need.
    for node in tree.body:
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef)
        ) and node.name in required_names:
            selected_nodes.append(node)

    found = {
        node.name
        for node in selected_nodes
    }

    missing = (
        required_names
        -
        found
    )

    if missing:
        raise RuntimeError(
            "Could not extract required original definitions: "
            + ", ".join(sorted(missing))
        )

    # Preserve source order.
    selected_nodes.sort(
        key=lambda n: n.lineno
    )

    module_ast = ast.Module(
        body=selected_nodes,
        type_ignores=[]
    )

    ast.fix_missing_locations(
        module_ast
    )

    # Namespace contains common globals used by the original
    # definitions, but DOES NOT execute the original script's
    # top-level code.
    ns = {
        "__name__": "fresh_eval_original_extracted",
        "np": np,
        "pd": pd,
        "torch": torch,
        "nn": nn,
        "math": math,
        "Path": Path,
    }

    exec(
        compile(
            module_ast,
            filename=str(ORIGINAL_EVAL),
            mode="exec"
        ),
        ns,
        ns
    )

    module = types.SimpleNamespace(
        PositionalEncoding=
            ns["PositionalEncoding"],

        TrajectoryTransformer=
            ns["TrajectoryTransformer"],

        inverse_transform_batch=
            ns["inverse_transform_batch"],
    )

    return module


ORIG = import_original_module()


# ============================================================
# BASIC METRICS
# ============================================================

def ade_fde(pred, truth):
    err = np.linalg.norm(pred - truth, axis=2)
    return err.mean(axis=1), err[:, -1]


def stationary_predict(history, future_len):
    return np.repeat(
        history[:, -1:, :],
        future_len,
        axis=1
    )


def constant_velocity_predict(history, future_len, fps=10.0):
    n = history.shape[1]

    t = np.arange(
        n,
        dtype=np.float64
    ) / fps

    tc = t - t.mean()
    denom = np.sum(tc ** 2)

    future_t = np.arange(
        1,
        future_len + 1,
        dtype=np.float64
    ) / fps

    outputs = []

    for h in history:
        pc = h - h.mean(axis=0, keepdims=True)

        v = (
            tc[:, None] * pc
        ).sum(axis=0) / max(denom, 1e-12)

        pred = (
            h[-1][None, :]
            +
            future_t[:, None] * v[None, :]
        )

        outputs.append(pred)

    return np.asarray(outputs)


# ============================================================
# ORIGINAL MODEL CONSTRUCTOR
# ============================================================

def build_original_transformer(task, seed):
    future_len = TASKS[task]["future_len"]

    ckpt_path = (
        TRANSFORMER_DIR
        /
        f"{task}_Transformer_seed{seed}.pt"
    )

    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE
    )

    cls = ORIG.TrajectoryTransformer

    sig = inspect.signature(cls.__init__)

    values = {
        # future horizon synonyms
        "future_len": future_len,
        "future_length": future_len,
        "pred_len": future_len,
        "prediction_length": future_len,
        "output_len": future_len,
        "horizon": future_len,

        # history/input length synonyms
        "history_len": 20,
        "history_length": 20,
        "obs_len": 20,
        "observation_length": 20,
        "input_len": 20,
        "seq_len": 20,

        # checkpoint architecture values
        "d_model": int(ckpt["d_model"]),
        "nhead": int(ckpt["nhead"]),
        "num_layers": int(ckpt["num_layers"]),
        "dim_feedforward": int(ckpt["dim_feedforward"]),
        "dropout": float(ckpt["dropout"]),

        # common I/O dimensions
        "input_dim": 2,
        "output_dim": 2,
        "in_features": 2,
        "coord_dim": 2,
    }

    kwargs = {}

    for name, p in sig.parameters.items():
        if name == "self":
            continue

        if name in values:
            kwargs[name] = values[name]
        elif p.default is inspect._empty:
            raise RuntimeError(
                "\nCannot automatically instantiate original Transformer.\n"
                f"Unknown required constructor argument: {name}\n"
                f"Constructor signature: {sig}\n"
            )

    model = cls(**kwargs).to(DEVICE)

    model.load_state_dict(
        ckpt["state_dict"],
        strict=True
    )

    model.eval()

    return model, ckpt, kwargs


# ============================================================
# ORIGINAL / EXACT INVERSE TRANSFORM
# ============================================================

def call_original_inverse(pred_canonical, arrays):
    fn = ORIG.inverse_transform_batch
    sig = inspect.signature(fn)

    available = {
        "canonical": pred_canonical,
        "canonical_xy": pred_canonical,
        "pred_canonical": pred_canonical,
        "prediction_canonical": pred_canonical,
        "predictions_canonical": pred_canonical,
        "pred": pred_canonical,
        "prediction": pred_canonical,
        "predictions": pred_canonical,

        "origin": np.asarray(arrays["origin"]),
        "origins": np.asarray(arrays["origin"]),

        "angle": np.asarray(arrays["heading_angle"]),
        "angles": np.asarray(arrays["heading_angle"]),
        "heading_angle": np.asarray(arrays["heading_angle"]),
        "heading_angles": np.asarray(arrays["heading_angle"]),

        "scale": np.asarray(arrays["scale"]),
        "scales": np.asarray(arrays["scale"]),
    }

    kwargs = {}

    for name, p in sig.parameters.items():
        if name in available:
            kwargs[name] = available[name]
        elif p.default is inspect._empty:
            raise RuntimeError(
                "\nCannot call original inverse_transform_batch.\n"
                f"Unknown required argument: {name}\n"
                f"Signature: {sig}\n"
            )

    out = fn(**kwargs)

    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()

    return np.asarray(
        out,
        dtype=np.float64
    )


# ============================================================
# PREDICT ORIGINAL TRANSFORMER
# ============================================================

def predict_original_model(model, history_canonical, batch_size=256):
    pieces = []

    with torch.no_grad():
        for start in range(
            0,
            len(history_canonical),
            batch_size
        ):
            x = torch.tensor(
                history_canonical[
                    start:
                    start + batch_size
                ],
                dtype=torch.float32,
                device=DEVICE
            )

            y = model(x)

            if isinstance(y, (tuple, list)):
                y = y[0]

            pieces.append(
                y.detach()
                .cpu()
                .numpy()
            )

    return np.concatenate(
        pieces,
        axis=0
    )


# ============================================================
# REGENERATE + STRICT SELF-CHECK
# ============================================================

def regenerate_transformers(task, arrays):
    expected = pd.read_csv(
        FRESH_DIR /
        "fresh_holdout_model_metrics.csv"
    )

    expected = expected[
        expected["task"] == task
    ].copy()

    X = np.asarray(
        arrays["history_canonical"],
        dtype=np.float32
    )

    truth = np.asarray(
        arrays["future_abs"],
        dtype=np.float64
    )

    seed_predictions = {}
    check_rows = []

    for seed in SEEDS:
        model, ckpt, kwargs = build_original_transformer(
            task,
            seed
        )

        pred_canonical = predict_original_model(
            model,
            X
        )

        pred_abs = call_original_inverse(
            pred_canonical,
            arrays
        )

        a, f = ade_fde(
            pred_abs,
            truth
        )

        row = expected[
            expected["model"]
            ==
            f"Transformer_seed{seed}"
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Missing saved metric for {task} seed {seed}"
            )

        row = row.iloc[0]

        ade_diff = abs(
            float(a.mean())
            -
            float(row["ADE"])
        )

        fde_diff = abs(
            float(f.mean())
            -
            float(row["FDE"])
        )

        check_rows.append({
            "task": task,
            "model": f"Transformer_seed{seed}",
            "constructor_kwargs": str(kwargs),
            "regenerated_ADE": float(a.mean()),
            "saved_ADE": float(row["ADE"]),
            "ADE_abs_diff": ade_diff,
            "regenerated_FDE": float(f.mean()),
            "saved_FDE": float(row["FDE"]),
            "FDE_abs_diff": fde_diff,
        })

        if (
            ade_diff > MATCH_TOL
            or
            fde_diff > MATCH_TOL
        ):
            raise RuntimeError(
                "\nEXACT TRANSFORMER SELF-CHECK FAILED.\n"
                f"{task} seed {seed}\n"
                f"ADE diff={ade_diff:.10f}\n"
                f"FDE diff={fde_diff:.10f}\n"
                "Bootstrap has NOT been run.\n"
            )

        seed_predictions[seed] = pred_abs

    ensemble = np.mean(
        np.stack(
            [seed_predictions[s] for s in SEEDS],
            axis=0
        ),
        axis=0
    )

    ea, ef = ade_fde(
        ensemble,
        truth
    )

    erow = expected[
        expected["model"]
        ==
        "Transformer_3Seed_Ensemble"
    ]

    if len(erow) != 1:
        raise RuntimeError(
            f"Missing ensemble saved metric for {task}"
        )

    erow = erow.iloc[0]

    ade_diff = abs(
        float(ea.mean())
        -
        float(erow["ADE"])
    )

    fde_diff = abs(
        float(ef.mean())
        -
        float(erow["FDE"])
    )

    check_rows.append({
        "task": task,
        "model": "Transformer_3Seed_Ensemble",
        "constructor_kwargs": "mean of frozen seeds 42/52/62",
        "regenerated_ADE": float(ea.mean()),
        "saved_ADE": float(erow["ADE"]),
        "ADE_abs_diff": ade_diff,
        "regenerated_FDE": float(ef.mean()),
        "saved_FDE": float(erow["FDE"]),
        "FDE_abs_diff": fde_diff,
    })

    if (
        ade_diff > MATCH_TOL
        or
        fde_diff > MATCH_TOL
    ):
        raise RuntimeError(
            "\nENSEMBLE SELF-CHECK FAILED.\n"
            f"{task}\n"
            f"ADE diff={ade_diff:.10f}\n"
            f"FDE diff={fde_diff:.10f}\n"
            "Bootstrap has NOT been run.\n"
        )

    return (
        seed_predictions,
        ensemble,
        pd.DataFrame(check_rows)
    )


# ============================================================
# BUILD MODEL SAMPLE-ERROR TABLE
# ============================================================

def aggregate_recurrent_seed_errors(
    step3,
    task,
    base_model
):
    d = step3[
        (step3["task"] == task)
        &
        (step3["model"] == base_model)
    ].copy()

    if len(d) == 0:
        raise RuntimeError(
            f"No Step-3 rows for {task} {base_model}"
        )

    out = (
        d.groupby(
            [
                "sample_id",
                "source_group",
                "track_key"
            ],
            as_index=False
        )
        .agg(
            ADE=("ADE", "mean"),
            FDE=("FDE", "mean")
        )
    )

    out["task"] = task
    out["model"] = (
        f"{base_model}_3Seed_Mean"
    )

    return out


def build_all_errors(
    task,
    arrays,
    metadata,
    ensemble_pred
):
    truth = np.asarray(
        arrays["future_abs"],
        dtype=np.float64
    )

    history = np.asarray(
        arrays["history_abs"],
        dtype=np.float64
    )

    future_len = TASKS[task][
        "future_len"
    ]

    base = metadata[
        [
            "sample_id",
            "source_group",
            "track_key"
        ]
    ].copy()

    base["sample_id"] = (
        base["sample_id"]
        .astype(int)
    )

    tables = []

    # Stationary
    pred = stationary_predict(
        history,
        future_len
    )

    a, f = ade_fde(
        pred,
        truth
    )

    x = base.copy()
    x["task"] = task
    x["model"] = "Stationary"
    x["ADE"] = a
    x["FDE"] = f
    tables.append(x)

    # Constant Velocity
    pred = constant_velocity_predict(
        history,
        future_len
    )

    a, f = ade_fde(
        pred,
        truth
    )

    x = base.copy()
    x["task"] = task
    x["model"] = "ConstantVelocity"
    x["ADE"] = a
    x["FDE"] = f
    tables.append(x)

    # Transformer Ensemble
    a, f = ade_fde(
        ensemble_pred,
        truth
    )

    x = base.copy()
    x["task"] = task
    x["model"] = (
        "Transformer_3Seed_Ensemble"
    )
    x["ADE"] = a
    x["FDE"] = f
    tables.append(x)

    # Step 3 saved sample metrics
    step3 = pd.read_csv(
        STEP3_DIR /
        "step3_new_model_sample_metrics.csv",
        dtype={
            "source_group": str,
            "track_key": str,
            "seed": str
        }
    )

    # Kalman
    k = step3[
        (step3["task"] == task)
        &
        (step3["model"] == "KalmanCV")
    ][
        [
            "sample_id",
            "source_group",
            "track_key",
            "ADE",
            "FDE"
        ]
    ].copy()

    k["task"] = task
    k["model"] = "KalmanCV"
    tables.append(k)

    # GRU/LSTM means across frozen seeds
    tables.append(
        aggregate_recurrent_seed_errors(
            step3,
            task,
            "GRU"
        )
    )

    tables.append(
        aggregate_recurrent_seed_errors(
            step3,
            task,
            "LSTM"
        )
    )

    out = pd.concat(
        tables,
        ignore_index=True
    )

    expected_models = [
        "Stationary",
        "ConstantVelocity",
        "KalmanCV",
        "GRU_3Seed_Mean",
        "LSTM_3Seed_Mean",
        "Transformer_3Seed_Ensemble"
    ]

    for model in expected_models:
        n = len(
            out[
                out["model"] == model
            ]
        )

        if n != TASKS[task]["expected_n"]:
            raise RuntimeError(
                f"{task} {model}: "
                f"{n} samples != "
                f"{TASKS[task]['expected_n']}"
            )

    return out


# ============================================================
# CLUSTERED BOOTSTRAP CI
# ============================================================

def model_clustered_ci(
    df,
    metric,
    rng
):
    g = (
        df.groupby("track_key")[metric]
        .agg(
            total="sum",
            count="count",
            mean="mean"
        )
        .reset_index()
    )

    totals = g["total"].to_numpy(
        dtype=float
    )

    counts = g["count"].to_numpy(
        dtype=float
    )

    means = g["mean"].to_numpy(
        dtype=float
    )

    n_tracks = len(g)

    boot_window = np.empty(
        N_BOOT,
        dtype=float
    )

    boot_track = np.empty(
        N_BOOT,
        dtype=float
    )

    for b in range(N_BOOT):
        idx = rng.integers(
            0,
            n_tracks,
            size=n_tracks
        )

        boot_window[b] = (
            totals[idx].sum()
            /
            counts[idx].sum()
        )

        boot_track[b] = (
            means[idx].mean()
        )

    return {
        "observed_window": float(
            df[metric].mean()
        ),

        "window_CI_low": float(
            np.quantile(
                boot_window,
                0.025
            )
        ),

        "window_CI_high": float(
            np.quantile(
                boot_window,
                0.975
            )
        ),

        "observed_track_macro": float(
            means.mean()
        ),

        "track_CI_low": float(
            np.quantile(
                boot_track,
                0.025
            )
        ),

        "track_CI_high": float(
            np.quantile(
                boot_track,
                0.975
            )
        ),
    }


# ============================================================
# PAIRED CLUSTERED BOOTSTRAP
# delta = ensemble - comparator
# negative => ensemble lower error / better
# ============================================================

def paired_cluster_bootstrap(
    all_errors,
    task,
    comparator,
    metric,
    rng
):
    ens = all_errors[
        (all_errors["task"] == task)
        &
        (
            all_errors["model"]
            ==
            "Transformer_3Seed_Ensemble"
        )
    ][
        [
            "sample_id",
            "track_key",
            metric
        ]
    ].rename(
        columns={metric: "ens"}
    )

    cmp = all_errors[
        (all_errors["task"] == task)
        &
        (
            all_errors["model"]
            ==
            comparator
        )
    ][
        [
            "sample_id",
            metric
        ]
    ].rename(
        columns={metric: "cmp"}
    )

    m = ens.merge(
        cmp,
        on="sample_id",
        validate="one_to_one"
    )

    if len(m) != TASKS[task]["expected_n"]:
        raise RuntimeError(
            f"Pair mismatch: "
            f"{task} {comparator}"
        )

    g = (
        m.groupby("track_key")
        .agg(
            ens_sum=("ens", "sum"),
            cmp_sum=("cmp", "sum"),
            count=("ens", "count"),
            ens_mean=("ens", "mean"),
            cmp_mean=("cmp", "mean")
        )
        .reset_index()
    )

    ens_sum = g["ens_sum"].to_numpy(float)
    cmp_sum = g["cmp_sum"].to_numpy(float)
    counts = g["count"].to_numpy(float)

    track_diff = (
        g["ens_mean"].to_numpy(float)
        -
        g["cmp_mean"].to_numpy(float)
    )

    n_tracks = len(g)

    boot_window = np.empty(
        N_BOOT,
        dtype=float
    )

    boot_track = np.empty(
        N_BOOT,
        dtype=float
    )

    for b in range(N_BOOT):
        idx = rng.integers(
            0,
            n_tracks,
            size=n_tracks
        )

        ens_w = (
            ens_sum[idx].sum()
            /
            counts[idx].sum()
        )

        cmp_w = (
            cmp_sum[idx].sum()
            /
            counts[idx].sum()
        )

        boot_window[b] = (
            ens_w - cmp_w
        )

        boot_track[b] = (
            track_diff[idx].mean()
        )

    return {
        "task": task,
        "comparator": comparator,
        "metric": metric,
        "samples": len(m),
        "tracks": n_tracks,

        "ensemble_mean": float(
            m["ens"].mean()
        ),

        "comparator_mean": float(
            m["cmp"].mean()
        ),

        "window_delta_ensemble_minus_comparator":
            float(
                m["ens"].mean()
                -
                m["cmp"].mean()
            ),

        "window_delta_CI_low":
            float(
                np.quantile(
                    boot_window,
                    0.025
                )
            ),

        "window_delta_CI_high":
            float(
                np.quantile(
                    boot_window,
                    0.975
                )
            ),

        "P_ensemble_better_window":
            float(
                np.mean(
                    boot_window < 0
                )
            ),

        "ensemble_track_macro":
            float(
                g["ens_mean"].mean()
            ),

        "comparator_track_macro":
            float(
                g["cmp_mean"].mean()
            ),

        "track_delta_ensemble_minus_comparator":
            float(
                track_diff.mean()
            ),

        "track_delta_CI_low":
            float(
                np.quantile(
                    boot_track,
                    0.025
                )
            ),

        "track_delta_CI_high":
            float(
                np.quantile(
                    boot_track,
                    0.975
                )
            ),

        "P_ensemble_better_track":
            float(
                np.mean(
                    boot_track < 0
                )
            ),
    }


# ============================================================
# PAIRED TRACK SIGN-FLIP TEST
# ============================================================

def signflip_pvalue(
    differences,
    rng
):
    d = np.asarray(
        differences,
        dtype=float
    )

    observed = abs(
        d.mean()
    )

    count = 0
    done = 0
    batch = 5000

    while done < N_PERM:
        n_now = min(
            batch,
            N_PERM - done
        )

        signs = rng.choice(
            np.array(
                [-1.0, 1.0]
            ),
            size=(
                n_now,
                len(d)
            )
        )

        perm = np.abs(
            (
                signs
                *
                d[None, :]
            ).mean(axis=1)
        )

        count += int(
            np.sum(
                perm >= observed
            )
        )

        done += n_now

    return (
        count + 1
    ) / (
        N_PERM + 1
    )


def add_signflip_pvalues(
    pairwise,
    all_errors,
    rng
):
    values = []

    for _, row in pairwise.iterrows():
        task = row["task"]
        comparator = row["comparator"]
        metric = row["metric"]

        e = all_errors[
            (all_errors["task"] == task)
            &
            (
                all_errors["model"]
                ==
                "Transformer_3Seed_Ensemble"
            )
        ][
            [
                "sample_id",
                "track_key",
                metric
            ]
        ].rename(
            columns={metric: "ens"}
        )

        c = all_errors[
            (all_errors["task"] == task)
            &
            (
                all_errors["model"]
                ==
                comparator
            )
        ][
            [
                "sample_id",
                metric
            ]
        ].rename(
            columns={metric: "cmp"}
        )

        m = e.merge(
            c,
            on="sample_id",
            validate="one_to_one"
        )

        t = (
            m.groupby("track_key")
            .agg(
                ens=("ens", "mean"),
                cmp=("cmp", "mean")
            )
        )

        d = (
            t["ens"]
            -
            t["cmp"]
        ).to_numpy(float)

        values.append(
            signflip_pvalue(
                d,
                rng
            )
        )

    out = pairwise.copy()

    out[
        "paired_track_signflip_p"
    ] = values

    return out


# ============================================================
# HOLM CORRECTION
# Applied across all planned pairwise tests
# ============================================================

def holm_adjust(pvalues):
    p = np.asarray(
        pvalues,
        dtype=float
    )

    m = len(p)

    order = np.argsort(p)

    adjusted = np.empty(
        m,
        dtype=float
    )

    running = 0.0

    for rank, idx in enumerate(order):
        value = (
            (m - rank)
            *
            p[idx]
        )

        running = max(
            running,
            value
        )

        adjusted[idx] = min(
            running,
            1.0
        )

    return adjusted


# ============================================================
# MAIN
# ============================================================

def main():
    rng = np.random.default_rng(
        RNG_SEED
    )

    all_checks = []
    all_error_tables = []

    for task, cfg in TASKS.items():
        print(
            "\n"
            +
            "=" * 78
        )

        print(
            "TASK:",
            task
        )

        print(
            "=" * 78
        )

        arrays = np.load(
            FRESH_DIR /
            f"{task}_fresh_dataset.npz"
        )

        metadata = pd.read_csv(
            FRESH_DIR /
            f"{task}_fresh_metadata.csv",
            dtype={
                "source_group": str,
                "track_key": str
            }
        )

        if len(metadata) != cfg["expected_n"]:
            raise RuntimeError(
                f"{task}: unexpected sample count"
            )

        print(
            "Using ORIGINAL "
            "evaluate_fresh_wildbox_holdout.py "
            "TrajectoryTransformer..."
        )

        (
            seed_preds,
            ensemble_pred,
            check_df
        ) = regenerate_transformers(
            task,
            arrays
        )

        print(
            "Transformer exact self-check PASSED."
        )

        all_checks.append(
            check_df
        )

        # save regenerated Transformer sample metrics
        truth = np.asarray(
            arrays["future_abs"],
            dtype=np.float64
        )

        rows = []

        for seed in SEEDS:
            a, f = ade_fde(
                seed_preds[seed],
                truth
            )

            for i in range(
                len(metadata)
            ):
                rows.append({
                    "task": task,
                    "sample_id":
                        int(
                            metadata.iloc[i][
                                "sample_id"
                            ]
                        ),
                    "source_group":
                        metadata.iloc[i][
                            "source_group"
                        ],
                    "track_key":
                        metadata.iloc[i][
                            "track_key"
                        ],
                    "model":
                        f"Transformer_seed{seed}",
                    "ADE":
                        float(a[i]),
                    "FDE":
                        float(f[i]),
                })

        ea, ef = ade_fde(
            ensemble_pred,
            truth
        )

        for i in range(
            len(metadata)
        ):
            rows.append({
                "task": task,
                "sample_id":
                    int(
                        metadata.iloc[i][
                            "sample_id"
                        ]
                    ),
                "source_group":
                    metadata.iloc[i][
                        "source_group"
                    ],
                "track_key":
                    metadata.iloc[i][
                        "track_key"
                    ],
                "model":
                    "Transformer_3Seed_Ensemble",
                "ADE":
                    float(ea[i]),
                "FDE":
                    float(ef[i]),
            })

        pd.DataFrame(
            rows
        ).to_csv(
            OUT /
            f"{task}_transformer_sample_metrics.csv",
            index=False
        )

        all_error_tables.append(
            build_all_errors(
                task,
                arrays,
                metadata,
                ensemble_pred
            )
        )

    checks = pd.concat(
        all_checks,
        ignore_index=True
    )

    checks.to_csv(
        OUT /
        "transformer_exact_self_check.csv",
        index=False
    )

    all_errors = pd.concat(
        all_error_tables,
        ignore_index=True
    )

    all_errors.to_csv(
        OUT /
        "all_model_sample_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Model-level clustered CIs
    # --------------------------------------------------------

    ci_rows = []

    for (
        task,
        model
    ), d in all_errors.groupby(
        [
            "task",
            "model"
        ],
        sort=False
    ):
        for metric in [
            "ADE",
            "FDE"
        ]:
            ci = model_clustered_ci(
                d,
                metric,
                rng
            )

            ci_rows.append({
                "task": task,
                "model": model,
                "metric": metric,
                "samples": len(d),
                "tracks":
                    d[
                        "track_key"
                    ].nunique(),
                **ci,
            })

    ci_df = pd.DataFrame(
        ci_rows
    )

    ci_df.to_csv(
        OUT /
        "model_clustered_bootstrap_95CI.csv",
        index=False
    )

    # --------------------------------------------------------
    # Pairwise vs Transformer ensemble
    # --------------------------------------------------------

    comparators = [
        "LSTM_3Seed_Mean",
        "GRU_3Seed_Mean",
        "ConstantVelocity",
        "KalmanCV",
        "Stationary"
    ]

    pair_rows = []

    for task in TASKS:
        for comparator in comparators:
            for metric in [
                "ADE",
                "FDE"
            ]:
                pair_rows.append(
                    paired_cluster_bootstrap(
                        all_errors,
                        task,
                        comparator,
                        metric,
                        rng
                    )
                )

    pairwise = pd.DataFrame(
        pair_rows
    )

    pairwise = add_signflip_pvalues(
        pairwise,
        all_errors,
        rng
    )

    pairwise[
        "holm_adjusted_p"
    ] = holm_adjust(
        pairwise[
            "paired_track_signflip_p"
        ].to_numpy()
    )

    pairwise[
        "significant_after_holm_0.05"
    ] = (
        pairwise[
            "holm_adjusted_p"
        ]
        <
        0.05
    )

    pairwise.to_csv(
        OUT /
        "pairwise_vs_transformer_ensemble.csv",
        index=False
    )

    # --------------------------------------------------------
    # Terminal views
    # --------------------------------------------------------

    ade_ci = ci_df[
        ci_df["metric"] == "ADE"
    ].copy()

    ade_ci = ade_ci.sort_values(
        [
            "task",
            "observed_window"
        ]
    )

    pair_ade = pairwise[
        pairwise["metric"] == "ADE"
    ].copy()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    with open(
        OUT / "SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 4: CLUSTERED BOOTSTRAP "
            "+ PAIRED SIGNIFICANCE\n"
        )

        f.write(
            "=" * 78
            +
            "\n\n"
        )

        f.write(
            "Transformer regeneration uses the "
            "ORIGINAL TrajectoryTransformer class "
            "from evaluate_fresh_wildbox_holdout.py.\n"
        )

        f.write(
            "Analysis is aborted unless regenerated "
            "Fresh WildBox metrics match the saved "
            "frozen evaluation metrics.\n\n"
        )

        f.write(
            "Bootstrap protocol:\n"
            "- cluster = track_key\n"
            "- 10,000 bootstrap replicates\n"
            "- 95% percentile confidence intervals\n"
            "- all windows belonging to a track are "
            "resampled together\n"
            "- paired delta = Transformer ensemble "
            "- comparator; negative means ensemble "
            "has lower error\n"
            "- paired track-level sign-flip test "
            "used for inferential p-values\n"
            "- Holm correction applied across the "
            "planned pairwise tests\n\n"
        )

        f.write(
            "TRANSFORMER EXACT SELF-CHECK\n"
        )

        f.write(
            "----------------------------\n"
        )

        f.write(
            checks.to_string(
                index=False
            )
        )

        f.write(
            "\n\nADE CLUSTERED 95% CI\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            ade_ci.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPAIRWISE TESTS\n"
        )

        f.write(
            "--------------\n"
        )

        f.write(
            pairwise.to_string(
                index=False
            )
        )

        f.write(
            "\n\nCaution: non-significance does "
            "not prove equivalence.\n"
        )

    print(
        "\nSTEP 4 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )

    print(
        "\nTRANSFORMER EXACT SELF-CHECK:\n"
    )

    print(
        checks[
            [
                "task",
                "model",
                "regenerated_ADE",
                "saved_ADE",
                "ADE_abs_diff",
                "regenerated_FDE",
                "saved_FDE",
                "FDE_abs_diff"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nADE CLUSTERED 95% CI:\n"
    )

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
                "track_CI_high"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nPAIRWISE ADE VS TRANSFORMER ENSEMBLE:\n"
    )

    print(
        pair_ade[
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
                "holm_adjusted_p",
                "significant_after_holm_0.05"
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
