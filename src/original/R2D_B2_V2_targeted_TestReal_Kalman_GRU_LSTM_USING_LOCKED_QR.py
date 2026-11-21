from pathlib import Path
import ast
import hashlib
import inspect
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

TRAIN_SCRIPT = MAIN_RESULT / "train_lstm_gru.py"
KALMAN_SCRIPT = MAIN_RESULT / "run_kalman_baseline.py"

B1_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B1_stationary_cv_transformer"
    / "R2D_B1_MANIFEST.json"
)

B2A_SUMMARY = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2_preflight"
    / "R2D_B2A_SUMMARY.json"
)

B2A_CHECKPOINTS = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2_preflight"
    / "R2D_B2A_lstm_gru_checkpoint_inventory.csv"
)


B2C_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2C_exact_kalman_param_lock"
    / "R2D_B2C_MANIFEST.json"
)

B2C_LOCKED_PARAMS = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2C_exact_kalman_param_lock"
    / "R2D_B2C_LOCKED_KALMAN_PARAMS.csv"
)

R2D_A3_RESOLVED = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_targeted_recompute_preflight_v3"
    / "R2D_A3_resolved_testreal_npz.csv"
)

R2D_A_METADATA = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_targeted_recompute_preflight"
    / "R2D_A_metadata_candidates.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2_testreal_kalman_gru_lstm"
)

OUT.mkdir(parents=True, exist_ok=True)

TASK_CFG = {
    "2s_to_2s": {
        "samples": 1053,
        "future_len": 20,
        "reference": {
            "KalmanCV": {"ADE": 0.023802, "FDE": 0.042645},
            "GRU": {"ADE": 0.017546, "FDE": 0.030032},
            "LSTM": {"ADE": 0.017395, "FDE": 0.029637},
        },
    },
    "2s_to_5s": {
        "samples": 826,
        "future_len": 50,
        "reference": {
            "KalmanCV": {"ADE": 0.040086, "FDE": 0.077245},
            "GRU": {"ADE": 0.036220, "FDE": 0.067864},
            "LSTM": {"ADE": 0.036407, "FDE": 0.068380},
        },
    },
}

SEEDS = [42, 52, 62]
MODELS = ["GRU", "LSTM"]


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_prerequisites():
    for p in [
        TRAIN_SCRIPT,
        KALMAN_SCRIPT,
        B1_MANIFEST,
        B2A_SUMMARY,
        B2A_CHECKPOINTS,
        B2C_MANIFEST,
        B2C_LOCKED_PARAMS,
        R2D_A3_RESOLVED,
        R2D_A_METADATA,
    ]:
        require(p)

    b1 = json.loads(B1_MANIFEST.read_text(encoding="utf-8"))
    if b1.get("status") != "R2D_B1_TARGETED_RECOMPUTATION_COMPLETE_REFERENCE_SANITY_PASS":
        raise RuntimeError(f"Unexpected B1 status: {b1.get('status')}")

    b2a = json.loads(B2A_SUMMARY.read_text(encoding="utf-8"))
    if (
        b2a.get("status")
        !=
        "R2D_B2A_PREFLIGHT_COMPLETE_READY_FOR_TESTREAL_KALMAN_GRU_LSTM_RECOMPUTATION"
    ):
        raise RuntimeError(f"Unexpected B2A status: {b2a.get('status')}")

    b2c = json.loads(B2C_MANIFEST.read_text(encoding="utf-8"))
    if (
        b2c.get("status")
        !=
        "R2D_B2C_FROZEN_KALMAN_PARAMS_EXACTLY_RECOVERED_AND_LOCKED_READY_FOR_B2"
    ):
        raise RuntimeError(f"Unexpected B2C status: {b2c.get('status')}")

    if (
        b2c.get("locked_params_sha256")
        and
        sha256_file(B2C_LOCKED_PARAMS) != b2c.get("locked_params_sha256")
    ):
        raise RuntimeError(
            "B2C locked Kalman parameter CSV hash does not match its manifest."
        )

    # Verify the frozen checkpoint files still match the B2A audit hashes.
    inv = pd.read_csv(B2A_CHECKPOINTS)
    if len(inv) != 12:
        raise RuntimeError(f"Expected 12 B2A checkpoint rows, found {len(inv)}")

    for _, row in inv.iterrows():
        p = Path(row["path"])
        require(p)
        actual = sha256_file(p)
        expected = str(row["sha256"])
        if actual != expected:
            raise RuntimeError(
                f"Frozen checkpoint changed since B2A.\nPath: {p}\n"
                f"Expected: {expected}\nActual:   {actual}"
            )

    return inv


def bound_names(node):
    names = set()
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
    return names


def loaded_names(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            out.add(sub.id)
    return out


def safe_extract_namespace(source_path, targets):
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text)

    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    funcs = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assignments = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in bound_names(node):
                assignments[name] = node

    selected_funcs = {x for x in targets if x in funcs}
    selected_classes = {x for x in targets if x in classes}
    selected_assignments = {x for x in targets if x in assignments}

    missing = [
        x
        for x in targets
        if x not in funcs and x not in classes and x not in assignments
    ]
    if missing:
        raise RuntimeError(f"Missing source targets in {source_path}: {missing}")

    pending = set()
    for name in selected_funcs:
        pending.update(loaded_names(funcs[name]))
    for name in selected_classes:
        pending.update(loaded_names(classes[name]))
    for name in selected_assignments:
        pending.update(loaded_names(assignments[name]))

    processed = set()

    while True:
        new = pending - processed
        if not new:
            break

        for name in list(new):
            processed.add(name)

            if name in funcs and name not in selected_funcs:
                selected_funcs.add(name)
                pending.update(loaded_names(funcs[name]))

            elif name in classes and name not in selected_classes:
                selected_classes.add(name)
                pending.update(loaded_names(classes[name]))

            elif name in assignments and name not in selected_assignments:
                selected_assignments.add(name)
                pending.update(loaded_names(assignments[name]))

    nodes = list(imports)

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if bound_names(node) & selected_assignments:
                nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            if node.name in selected_classes:
                nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in selected_funcs:
                nodes.append(node)

    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)

    ns = {
        "__file__": str(source_path),
        "__name__": "_r2d_b2_safe_source_",
    }

    sys.path.insert(0, str(source_path.parent))
    exec(compile(module, str(source_path), "exec"), ns)

    return ns, {
        "selected_functions": sorted(selected_funcs),
        "selected_classes": sorted(selected_classes),
        "selected_assignments": sorted(selected_assignments),
        "top_level_script_executed": False,
    }


def get_testreal_paths():
    resolved = pd.read_csv(R2D_A3_RESOLVED)
    metadata = pd.read_csv(R2D_A_METADATA)

    out = {}

    for task, cfg in TASK_CFG.items():
        hit = resolved[resolved["task"].astype(str) == task]
        if len(hit) != 1:
            raise RuntimeError(f"Expected one resolved TestReal NPZ for {task}, got {len(hit)}")

        npz_path = Path(hit.iloc[0]["selected_path"])
        require(npz_path)

        meta = metadata[
            (metadata["dataset"] == "BIRDSAI_TestReal")
            & (metadata["task"].astype(str) == task)
            & (metadata["rows"] == cfg["samples"])
        ].copy()

        if len(meta) == 0:
            raise RuntimeError(f"No exact TestReal metadata for {task}")

        meta = meta.sort_values(["score", "path"], ascending=[False, True])
        meta_path = Path(meta.iloc[0]["path"])
        require(meta_path)

        out[task] = {
            "npz": npz_path,
            "metadata": meta_path,
        }

    return out


def normalize_task_value(v):
    s = str(v).strip().lower().replace(" ", "")
    if any(x in s for x in ["2s_to_2s", "2s-to-2s", "2to2", "2s2s"]):
        return "2s_to_2s"
    if any(x in s for x in ["2s_to_5s", "2s-to-5s", "2to5", "2s5s"]):
        return "2s_to_5s"
    return None


def find_col(df, names):
    lookup = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


def boolish(v):
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "selected", "best", "winner"}


def load_locked_kalman_params():
    """
    Load exact frozen q/r values previously recovered and locked by R2D-B2C.
    No tuning, ranking, or parameter search occurs in this step.
    """
    df = pd.read_csv(B2C_LOCKED_PARAMS)

    required = {
        "task",
        "q",
        "r",
        "crosscheck_pass",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"B2C locked parameter CSV missing columns: {sorted(missing)}"
        )

    expected_tasks = set(TASK_CFG.keys())
    actual_tasks = set(df["task"].astype(str))

    if actual_tasks != expected_tasks:
        raise RuntimeError(
            f"B2C locked parameter tasks mismatch: {actual_tasks} != {expected_tasks}"
        )

    if len(df) != 2:
        raise RuntimeError(
            f"Expected exactly two locked Kalman parameter rows, found {len(df)}"
        )

    if not df["crosscheck_pass"].astype(bool).all():
        raise RuntimeError(
            "At least one B2C locked Kalman parameter row failed cross-check."
        )

    # Exact values independently established by B2C.
    expected_values = {
        "2s_to_2s": (0.1, 0.0001),
        "2s_to_5s": (0.1, 0.01),
    }

    for _, row in df.iterrows():
        task = str(row["task"])
        q = float(row["q"])
        r = float(row["r"])

        exp_q, exp_r = expected_values[task]

        if q != exp_q or r != exp_r:
            raise RuntimeError(
                f"Locked Kalman parameters changed for {task}: "
                f"got q={q}, r={r}; expected q={exp_q}, r={exp_r}"
            )

    return df[["task", "q", "r"]].copy()

def instantiate_recurrent(cls, checkpoint, future_len):
    sig = inspect.signature(cls.__init__)
    kwargs = {}

    meta = {
        "model_type": checkpoint["model_type"],
        "future_length": future_len,
        "future_len": future_len,
        "output_length": future_len,
        "pred_len": future_len,
        "horizon": future_len,
        "hidden_size": checkpoint["hidden_size"],
        "num_layers": checkpoint["num_layers"],
        "dropout": checkpoint["dropout"],
    }

    required_unknown = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        if name in meta:
            kwargs[name] = meta[name]
        elif param.default is inspect._empty:
            required_unknown.append(name)

    if required_unknown:
        raise RuntimeError(
            f"Cannot safely instantiate RecurrentForecaster; unknown required "
            f"constructor args: {required_unknown}; signature={sig}"
        )

    model = cls(**kwargs)
    return model, str(sig), kwargs


def load_testreal_data(npz_path, metadata_path, expected_samples, future_len):
    z = np.load(npz_path, allow_pickle=False)
    meta = pd.read_csv(metadata_path, dtype={"source_group": str, "track_key": str})

    required = [
        "history_canonical",
        "future_abs",
        "origin",
        "heading_angle",
        "scale",
        "history_abs",
    ]

    missing = [k for k in required if k not in z.files]
    if missing:
        raise RuntimeError(f"Missing arrays {missing} in {npz_path}; keys={z.files}")

    history_can = np.asarray(z["history_canonical"], dtype=np.float32)
    history_abs = np.asarray(z["history_abs"], dtype=np.float64)
    truth_abs = np.asarray(z["future_abs"], dtype=np.float64)
    origin = np.asarray(z["origin"], dtype=np.float32)
    angle = np.asarray(z["heading_angle"], dtype=np.float32)
    scale = np.asarray(z["scale"], dtype=np.float32)

    if history_can.shape != (expected_samples, 20, 2):
        raise RuntimeError(f"history_canonical shape mismatch: {history_can.shape}")

    if history_abs.shape != (expected_samples, 20, 2):
        raise RuntimeError(f"history_abs shape mismatch: {history_abs.shape}")

    if truth_abs.shape != (expected_samples, future_len, 2):
        raise RuntimeError(f"future_abs shape mismatch: {truth_abs.shape}")

    if len(meta) != expected_samples:
        raise RuntimeError(f"metadata count mismatch: {len(meta)} != {expected_samples}")

    return {
        "history_can": history_can,
        "history_abs": history_abs,
        "truth_abs": truth_abs,
        "origin": origin,
        "angle": angle,
        "scale": scale,
        "metadata": meta,
    }


def sample_metric_arrays(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    if pred.shape != truth.shape:
        raise RuntimeError(f"Prediction/truth shape mismatch: {pred.shape} vs {truth.shape}")

    err = np.linalg.norm(pred - truth, axis=2)
    return err.mean(axis=1), err[:, -1]


def build_sample_df(task, model, metadata, ade, fde):
    rows = pd.DataFrame(
        {
            "dataset": "BIRDSAI_TestReal",
            "task": task,
            "model": model,
            "sample_id": np.arange(len(metadata)).astype(str),
            "track_id": metadata["track_key"].astype(str).values
            if "track_key" in metadata.columns
            else None,
            "source_id": metadata["source_group"].astype(str).values
            if "source_group" in metadata.columns
            else None,
            "ade": ade.astype(float),
            "fde": fde.astype(float),
        }
    )
    return rows


def per_sample_kalman(kalman_forecast, history_abs, future_len, q, r):
    preds = []
    for i in range(len(history_abs)):
        p = kalman_forecast(
            history_abs[i],
            future_len,
            q,
            r,
        )
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (future_len, 2):
            raise RuntimeError(
                f"Kalman sample {i} shape {p.shape}, expected {(future_len, 2)}"
            )
        preds.append(p)
    return np.stack(preds, axis=0)


def aggregate_summary(sample_df):
    return (
        sample_df.groupby(["dataset", "task", "model"], as_index=False)
        .agg(
            samples=("ade", "size"),
            ADE=("ade", "mean"),
            FDE=("fde", "mean"),
        )
    )


def sanity_check(summary):
    rows = []
    for _, row in summary.iterrows():
        task = str(row["task"])
        model = str(row["model"])
        ref = TASK_CFG[task]["reference"][model]

        ade_diff = abs(float(row["ADE"]) - float(ref["ADE"]))
        fde_diff = abs(float(row["FDE"]) - float(ref["FDE"]))

        passed = ade_diff <= 2.5e-6 and fde_diff <= 2.5e-6

        rows.append(
            {
                "task": task,
                "model": model,
                "computed_ADE": float(row["ADE"]),
                "computed_FDE": float(row["FDE"]),
                "reference_ADE": float(ref["ADE"]),
                "reference_FDE": float(ref["FDE"]),
                "abs_ADE_diff": ade_diff,
                "abs_FDE_diff": fde_diff,
                "pass": bool(passed),
            }
        )
    return pd.DataFrame(rows)


def main():
    print(
        "\nR2D-B2-V2 — TARGETED TESTREAL SAMPLE-LEVEL RECOMPUTATION: "
        "KALMAN + GRU + LSTM"
    )
    print(
        "NO retraining. NO Fresh recomputation. GRU/LSTM are aggregated as "
        "per-sample mean metrics across the three frozen seeds, NOT as a trajectory ensemble. "
        "Kalman q/r is read only from the exact R2D-B2C locked parameter artifact.\n"
    )

    checkpoint_inventory = verify_prerequisites()
    paths = get_testreal_paths()

    recurrent_ns, recurrent_audit = safe_extract_namespace(
        TRAIN_SCRIPT,
        [
            "RecurrentForecaster",
            "predict_model",
            "inverse_transform_batch",
        ],
    )

    kalman_ns, kalman_audit = safe_extract_namespace(
        KALMAN_SCRIPT,
        [
            "kalman_forecast",
        ],
    )

    # Force CPU-only inference for reproducibility and to match user environment.
    recurrent_ns["DEVICE"] = torch.device("cpu")

    kalman_params = load_locked_kalman_params()

    kalman_params_path = OUT / "R2D_B2_locked_frozen_kalman_params_used.csv"
    kalman_params.to_csv(kalman_params_path, index=False)

    all_samples = []
    seed_samples = []
    model_loading_rows = []
    input_rows = []

    ckpt_lookup = {
        (str(row["task"]), str(row["model"]), int(row["seed"])): Path(row["path"])
        for _, row in checkpoint_inventory.iterrows()
    }

    for task, cfg in TASK_CFG.items():
        data = load_testreal_data(
            paths[task]["npz"],
            paths[task]["metadata"],
            cfg["samples"],
            cfg["future_len"],
        )

        input_rows.append(
            {
                "task": task,
                "npz": str(paths[task]["npz"]),
                "npz_sha256": sha256_file(paths[task]["npz"]),
                "metadata": str(paths[task]["metadata"]),
                "metadata_sha256": sha256_file(paths[task]["metadata"]),
                "samples": cfg["samples"],
                "future_len": cfg["future_len"],
            }
        )

        # ------------------------------------------------------------
        # Kalman: exact frozen q/r recovered from prior artifact.
        # ------------------------------------------------------------
        kp = kalman_params[kalman_params["task"] == task]
        if len(kp) != 1:
            raise RuntimeError(f"Expected one recovered Kalman parameter row for {task}")

        q = float(kp.iloc[0]["q"])
        r = float(kp.iloc[0]["r"])

        kalman_pred = per_sample_kalman(
            kalman_ns["kalman_forecast"],
            data["history_abs"],
            cfg["future_len"],
            q,
            r,
        )

        k_ade, k_fde = sample_metric_arrays(
            kalman_pred,
            data["truth_abs"],
        )

        all_samples.append(
            build_sample_df(
                task,
                "KalmanCV",
                data["metadata"],
                k_ade,
                k_fde,
            )
        )

        # ------------------------------------------------------------
        # GRU / LSTM: exact 3 frozen checkpoints each.
        # Aggregate sample metrics across seeds, not trajectories.
        # ------------------------------------------------------------
        for model_type in MODELS:
            seed_ade_arrays = []
            seed_fde_arrays = []

            for seed in SEEDS:
                ckpt_path = ckpt_lookup[(task, model_type, seed)]
                checkpoint = torch.load(
                    ckpt_path,
                    map_location="cpu",
                    weights_only=False,
                )

                required_meta = [
                    "model_type",
                    "hidden_size",
                    "num_layers",
                    "dropout",
                    "state_dict",
                    "task",
                    "seed",
                ]

                missing = [k for k in required_meta if k not in checkpoint]
                if missing:
                    raise RuntimeError(
                        f"{ckpt_path}: missing checkpoint metadata {missing}"
                    )

                if str(checkpoint["model_type"]).upper() != model_type:
                    raise RuntimeError(
                        f"{ckpt_path}: model_type mismatch "
                        f"{checkpoint['model_type']} != {model_type}"
                    )

                if str(checkpoint["task"]) != task:
                    raise RuntimeError(
                        f"{ckpt_path}: task mismatch {checkpoint['task']} != {task}"
                    )

                if int(checkpoint["seed"]) != seed:
                    raise RuntimeError(
                        f"{ckpt_path}: seed mismatch {checkpoint['seed']} != {seed}"
                    )

                model, constructor_signature, constructor_kwargs = instantiate_recurrent(
                    recurrent_ns["RecurrentForecaster"],
                    checkpoint,
                    cfg["future_len"],
                )

                model.load_state_dict(
                    checkpoint["state_dict"],
                    strict=True,
                )
                model.to(torch.device("cpu"))
                model.eval()

                pred_can = recurrent_ns["predict_model"](
                    model,
                    data["history_can"],
                )

                pred_can = np.asarray(pred_can, dtype=np.float32)

                if pred_can.shape != (
                    cfg["samples"],
                    cfg["future_len"],
                    2,
                ):
                    raise RuntimeError(
                        f"{task} {model_type} seed{seed}: prediction shape "
                        f"{pred_can.shape} unexpected"
                    )

                pred_abs = recurrent_ns["inverse_transform_batch"](
                    pred_can,
                    data["origin"],
                    data["angle"],
                    data["scale"],
                )

                pred_abs = np.asarray(pred_abs, dtype=np.float64)

                ade, fde = sample_metric_arrays(
                    pred_abs,
                    data["truth_abs"],
                )

                seed_ade_arrays.append(ade)
                seed_fde_arrays.append(fde)

                seed_df = build_sample_df(
                    task,
                    model_type,
                    data["metadata"],
                    ade,
                    fde,
                )
                seed_df["seed"] = seed
                seed_samples.append(seed_df)

                model_loading_rows.append(
                    {
                        "task": task,
                        "model": model_type,
                        "seed": seed,
                        "checkpoint": str(ckpt_path),
                        "checkpoint_sha256": sha256_file(ckpt_path),
                        "constructor_signature": constructor_signature,
                        "constructor_kwargs_json": json.dumps(
                            constructor_kwargs,
                            default=str,
                        ),
                        "best_epoch": checkpoint.get("best_epoch"),
                    }
                )

            mean_ade = np.mean(
                np.stack(seed_ade_arrays, axis=0),
                axis=0,
            )
            mean_fde = np.mean(
                np.stack(seed_fde_arrays, axis=0),
                axis=0,
            )

            all_samples.append(
                build_sample_df(
                    task,
                    model_type,
                    data["metadata"],
                    mean_ade,
                    mean_fde,
                )
            )

        print(
            f"COMPLETE: BIRDSAI_TestReal {task} "
            f"({cfg['samples']} samples; Kalman + GRU + LSTM)"
        )

    sample_df = pd.concat(all_samples, ignore_index=True)
    seed_df = pd.concat(seed_samples, ignore_index=True)
    loading_df = pd.DataFrame(model_loading_rows)
    input_df = pd.DataFrame(input_rows)

    summary_df = aggregate_summary(sample_df)
    sanity_df = sanity_check(summary_df)

    sample_path = OUT / "R2D_B2_sample_metrics.csv"
    seed_path = OUT / "R2D_B2_GRU_LSTM_seed_level_sample_metrics.csv"
    summary_path = OUT / "R2D_B2_summary_metrics.csv"
    sanity_path = OUT / "R2D_B2_reference_sanity_check.csv"
    loading_path = OUT / "R2D_B2_model_loading_audit.csv"
    input_path = OUT / "R2D_B2_input_provenance.csv"

    sample_df.to_csv(sample_path, index=False)
    seed_df.to_csv(seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sanity_df.to_csv(sanity_path, index=False)
    loading_df.to_csv(loading_path, index=False)
    input_df.to_csv(input_path, index=False)

    failed = sanity_df[sanity_df["pass"] == False]

    status = (
        "R2D_B2_TESTREAL_KALMAN_GRU_LSTM_SAMPLE_METRICS_COMPLETE_REFERENCE_SANITY_PASS"
        if len(failed) == 0
        else
        "R2D_B2_RECOMPUTATION_COMPLETE_REFERENCE_SANITY_MISMATCH_REVIEW_REQUIRED"
    )

    manifest = {
        "step": "R2D_B2_TESTREAL_KALMAN_GRU_LSTM_SAMPLE_LEVEL_RECOMPUTATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "retraining": False,
        "fresh_recomputed": False,
        "kalman_post_test_tuning": False,
        "kalman_param_source": str(B2C_LOCKED_PARAMS),
        "kalman_param_source_sha256": sha256_file(B2C_LOCKED_PARAMS),
        "gru_lstm_aggregation": "per-sample mean ADE/FDE across frozen seeds 42/52/62; not trajectory ensemble",
        "train_script": str(TRAIN_SCRIPT),
        "train_script_sha256": sha256_file(TRAIN_SCRIPT),
        "kalman_script": str(KALMAN_SCRIPT),
        "kalman_script_sha256": sha256_file(KALMAN_SCRIPT),
        "recurrent_safe_extraction": recurrent_audit,
        "kalman_safe_extraction": kalman_audit,
        "sample_metrics_csv": str(sample_path),
        "sample_metrics_sha256": sha256_file(sample_path),
        "summary_metrics_csv": str(summary_path),
        "reference_sanity_csv": str(sanity_path),
        "seed_level_metrics_csv": str(seed_path),
        "model_loading_audit_csv": str(loading_path),
        "input_provenance_csv": str(input_path),
        "kalman_params_csv": str(kalman_params_path),
    }

    manifest_path = OUT / "R2D_B2_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nRECOVERED FROZEN KALMAN PARAMETERS:")
    print(kalman_params.to_string(index=False))

    print("\nSUMMARY METRICS:")
    print(summary_df.to_string(index=False))

    print("\nREFERENCE SANITY CHECK:")
    print(sanity_df.to_string(index=False))

    print("\nSTATUS:")
    print(status)

    print("\nR2D-B2 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
