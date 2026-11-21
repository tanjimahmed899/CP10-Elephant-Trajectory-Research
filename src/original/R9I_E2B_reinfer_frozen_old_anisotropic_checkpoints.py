
from pathlib import Path
import ast
import contextlib
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

TRANSFORMER_SRC = MAIN1 / "train_transformer.py"
RECURRENT_SRC = MAIN1 / "train_lstm_gru.py"

BASE = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
)

E2A_DIR = BASE / "R9I_E2A_old_anisotropic_checkpoint_and_testreal_lock"
R9I_E2A_MANIFEST = E2A_DIR / "R9I_E2A_MANIFEST.json"
E2A_CHECKPOINT_CSV = E2A_DIR / "R9I_E2A_OLD_ANISOTROPIC_CHECKPOINT_LOCK.csv"
E2A_PREDICTION_CSV = E2A_DIR / "R9I_E2A_EXISTING_OLD_PREDICTION_INVENTORY.csv"

TESTREAL_DIR = MAIN1 / "final_neural_trajectory_dataset"
GEOMETRY_CSV = (
    BASE
    / "R9G_V3B_corrected_isotropic_rebuild"
    / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"
)

OUT = BASE / "R9I_E2B_frozen_old_anisotropic_reinference"
TRANSFORMER_OUT = OUT / "transformer"
RECURRENT_OUT = OUT / "lstm_gru"
SAMPLE_OUT = OUT / "sample_metrics"
LOG_OUT = OUT / "logs"

for p in [OUT, TRANSFORMER_OUT, RECURRENT_OUT, SAMPLE_OUT, LOG_OUT]:
    p.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {"expected_samples": 1053, "future_length": 20},
    "2s_to_5s": {"expected_samples": 826, "future_length": 50},
}
SEEDS = [42, 52, 62]
MODELS = ["Transformer", "LSTM", "GRU"]
DATASET_NAME = "BIRDSAI_TestReal_Anisotropic_R9Fresh"
ROLE = "matched_testreal_old_anisotropic"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def verify_upstream():
    for p in [
        R9I_E2A_MANIFEST,
        E2A_CHECKPOINT_CSV,
        E2A_PREDICTION_CSV,
        TRANSFORMER_SRC,
        RECURRENT_SRC,
        GEOMETRY_CSV,
    ]:
        require(p)

    m = json.loads(R9I_E2A_MANIFEST.read_text(encoding="utf-8"))
    expected = (
        "R9I_E2A_ALL_18_OLD_ANISOTROPIC_CHECKPOINTS_AND_TESTREAL_LOCKED_READY_FOR_R9I_E2B_FROZEN_ANISOTROPIC_REINFERENCE"
    )
    if m.get("status") != expected:
        raise RuntimeError(f"Unexpected R9-I-E2A status: {m.get('status')}")
    if int(m.get("checkpoints_resolved", -1)) != 18:
        raise RuntimeError("R9-I-E2A did not resolve all 18 old checkpoints.")


def make_definition_only_module(source_path, module_name):
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text, filename=str(source_path))

    definition_nodes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    first_definition_line = min(
        (
            int(node.lineno)
            for node in tree.body
            if isinstance(node, definition_nodes)
        ),
        default=10**9,
    )

    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and int(node.lineno) < first_definition_line
        ):
            body.append(node)
        elif isinstance(node, definition_nodes):
            body.append(node)

    clean_tree = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(clean_tree)

    ns = {
        "__name__": module_name,
        "__file__": str(source_path),
    }
    exec(compile(clean_tree, str(source_path), "exec"), ns)
    return ns


def load_geometry():
    df = pd.read_csv(GEOMETRY_CSV, dtype={"sequence": str})
    needed = {"sequence", "width_px", "height_px"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Geometry lock missing columns: {sorted(missing)}")

    return {
        str(row["sequence"]): {
            "W": float(row["width_px"]),
            "H": float(row["height_px"]),
        }
        for _, row in df.iterrows()
    }


def load_checkpoint_lock():
    df = pd.read_csv(
        E2A_CHECKPOINT_CSV,
        dtype={
            "task": str,
            "model": str,
            "resolved_path": str,
            "resolved_sha256": str,
        },
    )

    if len(df) != 18:
        raise RuntimeError(f"Checkpoint lock has {len(df)} rows, expected 18.")

    mapping = {}
    for _, row in df.iterrows():
        key = (str(row["task"]), str(row["model"]), int(row["seed"]))
        p = Path(str(row["resolved_path"]))
        require(p)
        current_hash = sha256_file(p)
        expected_hash = str(row["resolved_sha256"])
        if current_hash != expected_hash:
            raise RuntimeError(f"Old checkpoint hash changed: {p}")
        mapping[key] = {"path": p, "sha256": current_hash}

    if len(mapping) != 18:
        raise RuntimeError("Checkpoint identity map is not exactly 18.")
    return mapping


def load_existing_birdsai_prediction_inventory():
    df = pd.read_csv(
        E2A_PREDICTION_CSV,
        dtype={"task": str, "model": str, "path": str, "sha256": str},
    )

    df = df[
        df["path"]
        .astype(str)
        .str.contains("BIRDSAI_test", case=False, regex=False)
    ].copy()

    mapping = {}
    for _, row in df.iterrows():
        key = (str(row["task"]), str(row["model"]), int(row["seed"]))
        p = Path(str(row["path"]))
        require(p)
        current_hash = sha256_file(p)
        if current_hash != str(row["sha256"]):
            raise RuntimeError(f"Existing old prediction hash changed: {p}")
        mapping[key] = {"path": p, "sha256": current_hash}

    if len(mapping) != 18:
        raise RuntimeError(
            f"Expected 18 existing BIRDSAI TestReal predictions, found {len(mapping)}."
        )
    return mapping


def testreal_paths(task):
    npz_path = TESTREAL_DIR / f"{task}_birdsai_test_canonical.npz"
    meta_path = TESTREAL_DIR / f"{task}_birdsai_test_metadata.csv"
    require(npz_path)
    require(meta_path)
    return npz_path, meta_path


def load_checkpoint(path):
    import torch

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def construct_model(
    task,
    model_name,
    checkpoint,
    transformer_ns,
    recurrent_ns,
):
    future_length = TASKS[task]["future_length"]

    if model_name == "Transformer":
        model = transformer_ns["TrajectoryTransformer"](future_length)
    else:
        model = recurrent_ns["RecurrentForecaster"](
            model_name,
            future_length,
        )

    model.load_state_dict(checkpoint["state_dict"])
    return model


def fresh_prediction_path(task, model, seed):
    if model == "Transformer":
        return (
            TRANSFORMER_OUT
            / f"{task}_{DATASET_NAME}_Transformer_seed{seed}_predictions.npz"
        )
    return (
        RECURRENT_OUT
        / f"{task}_{DATASET_NAME}_{model}_seed{seed}_predictions.npz"
    )


def compare_prediction_arrays(fresh_path, old_path):
    fresh = np.load(fresh_path, allow_pickle=False)
    old = np.load(old_path, allow_pickle=False)

    keys = {
        "truth_abs",
        "prediction_abs",
        "prediction_canonical",
    }

    if not keys.issubset(set(fresh.files)):
        raise RuntimeError(f"Fresh prediction missing required keys: {fresh_path}")
    if not keys.issubset(set(old.files)):
        raise RuntimeError(f"Old prediction missing required keys: {old_path}")

    rows = []
    all_exact = True
    max_any = 0.0

    for key in sorted(keys):
        a = np.asarray(fresh[key])
        b = np.asarray(old[key])

        if a.shape != b.shape:
            raise RuntimeError(
                f"{key}: fresh shape={a.shape}, old shape={b.shape}"
            )

        exact = np.array_equal(a, b)
        diff = float(
            np.max(
                np.abs(
                    a.astype(np.float64)
                    -
                    b.astype(np.float64)
                )
            )
        )

        all_exact = all_exact and exact
        max_any = max(max_any, diff)

        rows.append(
            {
                "array": key,
                "exact_array_equal": bool(exact),
                "max_abs_difference": diff,
                "shape": str(a.shape),
                "fresh_dtype": str(a.dtype),
                "old_dtype": str(b.dtype),
            }
        )

    return all_exact, max_any, rows


def sample_pixel_scales(meta, geometry):
    if "sequence" in meta.columns:
        seqs = meta["sequence"].astype(str)
    elif "source_group" in meta.columns:
        seqs = meta["source_group"].astype(str)
    else:
        raise RuntimeError("Metadata has neither sequence nor source_group.")

    missing = sorted(set(seqs) - set(geometry))
    if missing:
        raise RuntimeError(f"Missing geometry for sequences: {missing}")

    W = np.asarray(
        [geometry[seq]["W"] for seq in seqs],
        dtype=np.float64,
    )
    H = np.asarray(
        [geometry[seq]["H"] for seq in seqs],
        dtype=np.float64,
    )
    return W, H


def compute_independent_metrics(
    task,
    model,
    seed,
    prediction_npz,
    metadata_path,
    geometry,
):
    arrays = np.load(prediction_npz, allow_pickle=False)

    truth_abs = np.asarray(arrays["truth_abs"], dtype=np.float64)
    pred_abs = np.asarray(arrays["prediction_abs"], dtype=np.float64)

    cfg = TASKS[task]
    expected_shape = (
        cfg["expected_samples"],
        cfg["future_length"],
        2,
    )

    if truth_abs.shape != expected_shape:
        raise RuntimeError(
            f"{prediction_npz.name}: truth shape={truth_abs.shape}, expected={expected_shape}"
        )
    if pred_abs.shape != expected_shape:
        raise RuntimeError(
            f"{prediction_npz.name}: pred shape={pred_abs.shape}, expected={expected_shape}"
        )

    meta = pd.read_csv(
        metadata_path,
        dtype={
            "sample_id": str,
            "sequence": str,
            "source_group": str,
            "segment_id": str,
            "track_key": str,
        },
    )

    if len(meta) != cfg["expected_samples"]:
        raise RuntimeError(
            f"{task}: metadata rows={len(meta)}, expected={cfg['expected_samples']}"
        )

    delta = pred_abs - truth_abs

    err_aniso = np.linalg.norm(delta, axis=2)

    W, H = sample_pixel_scales(meta, geometry)

    dx_px = delta[:, :, 0] * W[:, None]
    dy_px = delta[:, :, 1] * H[:, None]
    err_px = np.sqrt(dx_px ** 2 + dy_px ** 2)

    sample_ade_aniso = err_aniso.mean(axis=1)
    sample_fde_aniso = err_aniso[:, -1]
    sample_ade_px = err_px.mean(axis=1)
    sample_fde_px = err_px[:, -1]

    sample_df = meta.copy()
    sample_df["task"] = task
    sample_df["model"] = model
    sample_df["seed"] = seed
    sample_df["ADE_aniso"] = sample_ade_aniso
    sample_df["FDE_aniso"] = sample_fde_aniso
    sample_df["ADE_px"] = sample_ade_px
    sample_df["FDE_px"] = sample_fde_px
    sample_df["width_px"] = W
    sample_df["height_px"] = H

    sample_path = (
        SAMPLE_OUT
        / f"{task}_{model}_seed{seed}_sample_metrics.csv"
    )
    sample_df.to_csv(sample_path, index=False)

    summary = {
        "task": task,
        "model": model,
        "seed": seed,
        "samples": int(len(sample_df)),
        "ADE_aniso_window_mean": float(sample_ade_aniso.mean()),
        "FDE_aniso_window_mean": float(sample_fde_aniso.mean()),
        "ADE_px_window_mean": float(sample_ade_px.mean()),
        "FDE_px_window_mean": float(sample_fde_px.mean()),
        "sample_metrics_csv": str(sample_path),
        "sample_metrics_sha256": sha256_file(sample_path),
        "prediction_npz": str(prediction_npz),
        "prediction_npz_sha256": sha256_file(prediction_npz),
    }

    if "track_key" in sample_df.columns:
        track_df = (
            sample_df
            .groupby("track_key", dropna=False)[
                ["ADE_aniso", "FDE_aniso", "ADE_px", "FDE_px"]
            ]
            .mean()
        )
        summary["tracks"] = int(len(track_df))
        summary["ADE_px_track_macro"] = float(track_df["ADE_px"].mean())
        summary["FDE_px_track_macro"] = float(track_df["FDE_px"].mean())

    source_col = (
        "source_group"
        if "source_group" in sample_df.columns
        else (
            "sequence"
            if "sequence" in sample_df.columns
            else None
        )
    )

    if source_col is not None:
        source_df = (
            sample_df
            .groupby(source_col, dropna=False)[
                ["ADE_aniso", "FDE_aniso", "ADE_px", "FDE_px"]
            ]
            .mean()
        )
        summary["sequences"] = int(len(source_df))
        summary["ADE_px_sequence_macro"] = float(
            source_df["ADE_px"].mean()
        )
        summary["FDE_px_sequence_macro"] = float(
            source_df["FDE_px"].mean()
        )

    return summary


def run_one(
    task,
    model_name,
    seed,
    transformer_ns,
    recurrent_ns,
    geometry,
    checkpoint_lock,
    old_prediction_lock,
):
    import torch

    key = (task, model_name, seed)
    ckpt_info = checkpoint_lock[key]
    old_pred_info = old_prediction_lock[key]

    checkpoint = load_checkpoint(ckpt_info["path"])

    if str(checkpoint.get("task")) != task:
        raise RuntimeError(f"{ckpt_info['path'].name}: task mismatch.")
    if int(checkpoint.get("seed")) != seed:
        raise RuntimeError(f"{ckpt_info['path'].name}: seed mismatch.")

    if model_name in ["LSTM", "GRU"]:
        if str(checkpoint.get("model_type")) != model_name:
            raise RuntimeError(
                f"{ckpt_info['path'].name}: model_type mismatch."
            )

    npz_path, meta_path = testreal_paths(task)

    if model_name == "Transformer":
        ns = transformer_ns
        ns["OUT"] = TRANSFORMER_OUT
    else:
        ns = recurrent_ns
        ns["OUT"] = RECURRENT_OUT

    model = construct_model(
        task,
        model_name,
        checkpoint,
        transformer_ns,
        recurrent_ns,
    )

    device = ns.get("DEVICE", "cpu")
    model = model.to(device)
    model.eval()

    best_epoch = int(checkpoint["best_epoch"])
    stem = f"{task}_{model_name}_seed{seed}"
    log_path = LOG_OUT / f"{stem}.log"

    print("\n" + "=" * 100)
    print(
        f"REINFERENCE START | task={task} | model={model_name} | "
        f"seed={seed} | best_epoch={best_epoch}"
    )
    print("=" * 100)

    with open(log_path, "w", encoding="utf-8") as log_f:
        tee = Tee(sys.stdout, log_f)

        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            with torch.no_grad():
                if model_name == "Transformer":
                    ns["evaluate"](
                        model,
                        task,
                        DATASET_NAME,
                        ROLE,
                        npz_path,
                        meta_path,
                        seed,
                        best_epoch,
                    )
                else:
                    ns["evaluate_dataset"](
                        model,
                        task,
                        DATASET_NAME,
                        ROLE,
                        npz_path,
                        meta_path,
                        model_name,
                        seed,
                        best_epoch,
                    )

    fresh_pred = fresh_prediction_path(task, model_name, seed)
    require(fresh_pred)

    arrays_exact, max_array_diff, array_rows = compare_prediction_arrays(
        fresh_pred,
        old_pred_info["path"],
    )

    summary = compute_independent_metrics(
        task,
        model_name,
        seed,
        fresh_pred,
        meta_path,
        geometry,
    )

    summary.update(
        {
            "best_epoch": best_epoch,
            "best_validation_track_macro_ADE": float(
                checkpoint["best_validation_track_macro_ADE"]
            ),
            "checkpoint": str(ckpt_info["path"]),
            "checkpoint_sha256": ckpt_info["sha256"],
            "existing_old_prediction": str(old_pred_info["path"]),
            "existing_old_prediction_sha256": old_pred_info["sha256"],
            "fresh_vs_existing_arrays_exact": bool(arrays_exact),
            "fresh_vs_existing_max_abs_difference": float(max_array_diff),
        }
    )

    array_df = pd.DataFrame(array_rows)
    array_csv = OUT / f"{stem}_fresh_vs_existing_array_check.csv"
    array_df.to_csv(array_csv, index=False)

    summary["fresh_vs_existing_array_check_csv"] = str(array_csv)
    summary["fresh_vs_existing_array_check_sha256"] = sha256_file(
        array_csv
    )

    print(
        f"[COMPLETE] {task} | {model_name} | seed={seed} | "
        f"ADE_px={summary['ADE_px_window_mean']:.6f} | "
        f"FDE_px={summary['FDE_px_window_mean']:.6f} | "
        f"fresh_vs_existing_exact={arrays_exact} | "
        f"maxdiff={max_array_diff:.3e}"
    )

    return summary


def main():
    print(
        "\nR9-I-E2B — FROZEN OLD ANISOTROPIC CHECKPOINT REINFERENCE"
    )
    print(
        "NO training, NO tuning, NO recalibration. "
        "All 18 original anisotropic checkpoints are loaded frozen and re-evaluated "
        "on the original anisotropic BIRDSAI TestReal dataset using the exact original "
        "evaluate/evaluate_dataset functions. Fresh prediction files are written only "
        "under Downloads. Each fresh prediction is compared array-by-array with the "
        "already-existing old BIRDSAI TestReal prediction. Pixel error is computed as "
        "sqrt((dx_norm*W)^2 + (dy_norm*H)^2).\n"
    )

    verify_upstream()
    checkpoint_lock = load_checkpoint_lock()
    old_prediction_lock = load_existing_birdsai_prediction_inventory()
    geometry = load_geometry()

    transformer_ns = make_definition_only_module(
        TRANSFORMER_SRC,
        "r9i_e2b_transformer_defs",
    )
    recurrent_ns = make_definition_only_module(
        RECURRENT_SRC,
        "r9i_e2b_recurrent_defs",
    )

    if "evaluate" not in transformer_ns:
        raise RuntimeError("Transformer evaluate function missing.")
    if "evaluate_dataset" not in recurrent_ns:
        raise RuntimeError("Recurrent evaluate_dataset function missing.")

    results = []

    try:
        for task in TASKS:
            for model_name in MODELS:
                for seed in SEEDS:
                    results.append(
                        run_one(
                            task,
                            model_name,
                            seed,
                            transformer_ns,
                            recurrent_ns,
                            geometry,
                            checkpoint_lock,
                            old_prediction_lock,
                        )
                    )
    except Exception:
        progress = {
            "step": "R9I_E2B_FROZEN_OLD_ANISOTROPIC_REINFERENCE",
            "updated_utc": utc_now(),
            "status": "R9I_E2B_PARTIAL_OR_FAILED_NO_TRAINING_SAFE_TO_RESUME_AFTER_PATCH",
            "completed_runs": results,
            "training": False,
            "tuning": False,
            "checkpoint_modification": False,
            "traceback": traceback.format_exc(),
        }
        (
            OUT / "R9I_E2B_PROGRESS.json"
        ).write_text(
            json.dumps(progress, indent=2),
            encoding="utf-8"
        )
        raise

    result_df = pd.DataFrame(results)
    seed_path = (
        OUT / "R9I_E2B_SEED_LEVEL_OLD_ANISOTROPIC_TESTREAL_METRICS.csv"
    )
    result_df.to_csv(seed_path, index=False)

    aggregate_rows = []
    metric_cols = [
        "ADE_px_window_mean",
        "FDE_px_window_mean",
        "ADE_px_track_macro",
        "FDE_px_track_macro",
        "ADE_px_sequence_macro",
        "FDE_px_sequence_macro",
    ]

    for (task, model), group in result_df.groupby(
        ["task", "model"],
        sort=False,
    ):
        row = {
            "task": task,
            "model": model,
            "n_seeds": int(len(group)),
        }

        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1))

        aggregate_rows.append(row)

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_path = (
        OUT / "R9I_E2B_3SEED_OLD_ANISOTROPIC_MODEL_SUMMARY.csv"
    )
    aggregate_df.to_csv(aggregate_path, index=False)

    exact_count = int(
        result_df["fresh_vs_existing_arrays_exact"].sum()
    )
    max_diff = float(
        result_df["fresh_vs_existing_max_abs_difference"].max()
    )

    if exact_count == 18:
        reproducibility_status = (
            "ALL_18_FRESH_PREDICTIONS_ARRAY_EXACT_MATCH_EXISTING_OLD_PREDICTIONS"
        )
    else:
        reproducibility_status = (
            "NOT_ALL_FRESH_PREDICTIONS_ARRAY_EXACT_CHECK_MAX_DIFF_AND_NUMERICAL_TOLERANCE"
        )

    status = (
        "R9I_E2B_ALL_18_OLD_ANISOTROPIC_CHECKPOINTS_REINFERRED_READY_FOR_R9I_E2C_PAIRED_SEGMENT_CLUSTER_BOOTSTRAP_COMPARISON"
    )

    manifest = {
        "step": "R9I_E2B_FROZEN_OLD_ANISOTROPIC_REINFERENCE",
        "created_utc": utc_now(),
        "status": status,
        "runs": 18,
        "training": False,
        "tuning": False,
        "recalibration": False,
        "checkpoint_modification": False,
        "old_E_drive_results_modified": False,
        "fresh_vs_existing_exact_run_count": exact_count,
        "fresh_vs_existing_max_abs_difference_over_all_runs": max_diff,
        "reproducibility_status": reproducibility_status,
        "primary_fair_metric_for_next_step":
            "pixel-space ADE/FDE on exact matched TestReal samples",
        "anisotropic_pixel_error_formula":
            "sqrt((dx_norm*W)^2 + (dy_norm*H)^2)",
        "seed_level_metrics_csv": str(seed_path),
        "seed_level_metrics_sha256": sha256_file(seed_path),
        "three_seed_summary_csv": str(aggregate_path),
        "three_seed_summary_sha256": sha256_file(aggregate_path),
        "runs_detail": results,
    }

    (
        OUT / "R9I_E2B_MANIFEST.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\nSEED-LEVEL OLD ANISOTROPIC TESTREAL METRICS:")
    cols = [
        "task",
        "model",
        "seed",
        "ADE_px_window_mean",
        "FDE_px_window_mean",
        "ADE_px_track_macro",
        "FDE_px_track_macro",
        "fresh_vs_existing_arrays_exact",
        "fresh_vs_existing_max_abs_difference",
        "best_epoch",
    ]
    print(
        result_df[
            [c for c in cols if c in result_df.columns]
        ].to_string(index=False)
    )

    print("\n3-SEED OLD ANISOTROPIC MODEL SUMMARY:")
    summary_cols = [
        "task",
        "model",
        "ADE_px_window_mean_mean",
        "ADE_px_window_mean_std",
        "FDE_px_window_mean_mean",
        "FDE_px_window_mean_std",
        "ADE_px_track_macro_mean",
        "ADE_px_track_macro_std",
        "FDE_px_track_macro_mean",
        "FDE_px_track_macro_std",
    ]
    print(
        aggregate_df[
            [c for c in summary_cols if c in aggregate_df.columns]
        ].to_string(index=False)
    )

    print("\nREPRODUCIBILITY CHECK:")
    print(f"  exact array matches: {exact_count}/18")
    print(
        f"  max absolute array difference across all runs: {max_diff:.3e}"
    )
    print(f"  {reproducibility_status}")

    print("\nSTATUS:")
    print(status)

    print("\nR9-I-E2B COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
