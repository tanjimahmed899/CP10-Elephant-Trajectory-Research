
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


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R10_ROOT = REVISION_ROOT / "R10_ensemble_analysis"

R10A_DIR = R10_ROOT / "R10A_frozen_transformer_ensemble_evidence_audit"
R10A_MANIFEST = R10A_DIR / "R10A_MANIFEST.json"
R10A_PRED_INVENTORY = (
    R10A_DIR
    / "R10A_FROZEN_TRANSFORMER_PREDICTION_INVENTORY.csv"
)

R10C0B_DIR = (
    R10_ROOT
    / "R10C0B_fresh_wildbox_schema_and_loader_audit"
)
R10C0B_MANIFEST = R10C0B_DIR / "R10C0B_MANIFEST.json"

TRANSFORMER_SRC = MAIN1 / "train_transformer.py"

FRESH_DIR = (
    MAIN1
    / "wildbox_fresh_confirmatory_results"
)

R2_B1_SAMPLE_METRICS = (
    REVISION_ROOT
    / "R2_fair_forecasting"
    / "R2D_B1_stationary_cv_transformer"
    / "R2D_B1_sample_metrics.csv"
)

OUT = (
    R10_ROOT
    / "R10C1_fresh_wildbox_frozen_transformer_reinference"
)

PRED_OUT = OUT / "transformer"
SAMPLE_OUT = OUT / "sample_metrics"
LOG_OUT = OUT / "logs"

for p in [
    OUT,
    PRED_OUT,
    SAMPLE_OUT,
    LOG_OUT,
]:
    p.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "samples": 865,
        "future_length": 20,
        "npz": FRESH_DIR / "2s_to_2s_fresh_dataset.npz",
        "metadata": FRESH_DIR / "2s_to_2s_fresh_metadata.csv",
        "npz_sha256": "87f7b4cea2e2433b83421e2c82f06f67867e40997df8e06b53a7ea6ba57c092b",
        "metadata_sha256": "2b686775122441883a95cb25280d4a9c35363f29a697b409fae871ca9d09ee5d",
    },
    "2s_to_5s": {
        "samples": 637,
        "future_length": 50,
        "npz": FRESH_DIR / "2s_to_5s_fresh_dataset.npz",
        "metadata": FRESH_DIR / "2s_to_5s_fresh_metadata.csv",
        "npz_sha256": "1cd237f4eb5d950c0c1480add27874206fc07ddc797bf03dc9cd7b9b693d7f56",
        "metadata_sha256": "7a97cf8b1add962273f423f63b12c145278621da60e87f1227f32ba9ae3c8a2d",
    },
}

SEEDS = [42, 52, 62]

DATASET_NAME = "Fresh_WildBox_R10C1"
ROLE = "fresh_external_matched"


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
        R10A_MANIFEST,
        R10A_PRED_INVENTORY,
        R10C0B_MANIFEST,
        TRANSFORMER_SRC,
        R2_B1_SAMPLE_METRICS,
    ]:
        require(p)

    r10a = json.loads(
        R10A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    r10c0b = json.loads(
        R10C0B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_a = (
        "R10A_PRIMARY_TESTREAL_READY_EXISTING_WILDBOX_NOT_FRESH_READY_FOR_R10B_TESTREAL_AND_R10C_FRESH_WILDBOX_REINFERENCE"
    )

    expected_c0b = (
        "R10C0B_FRESH_WILDBOX_SCHEMA_AUDITED_LOADER_EVIDENCE_FOUND_READY_FOR_R10C1_ADAPTER_OR_DIRECT_INFERENCE_BUILD"
    )

    if r10a.get("status") != expected_a:
        raise RuntimeError(
            f"Unexpected R10-A status: {r10a.get('status')}"
        )

    if r10c0b.get("status") != expected_c0b:
        raise RuntimeError(
            f"Unexpected R10-C0B status: {r10c0b.get('status')}"
        )

    return r10a, r10c0b


def verify_fresh_inputs():
    rows = []

    for task, cfg in TASKS.items():
        require(cfg["npz"])
        require(cfg["metadata"])

        if sha256_file(cfg["npz"]) != cfg["npz_sha256"]:
            raise RuntimeError(
                f"{task}: fresh NPZ hash mismatch."
            )

        if sha256_file(cfg["metadata"]) != cfg["metadata_sha256"]:
            raise RuntimeError(
                f"{task}: fresh metadata hash mismatch."
            )

        arrays = np.load(
            cfg["npz"],
            allow_pickle=False
        )

        required_keys = {
            "history_canonical",
            "future_canonical",
            "history_abs",
            "future_abs",
            "origin",
            "heading_angle",
            "scale",
        }

        missing = required_keys - set(
            arrays.files
        )

        if missing:
            raise RuntimeError(
                f"{task}: fresh NPZ missing keys {sorted(missing)}"
            )

        if tuple(
            arrays["history_canonical"].shape
        ) != (
            cfg["samples"],
            20,
            2,
        ):
            raise RuntimeError(
                f"{task}: unexpected history_canonical shape."
            )

        if tuple(
            arrays["future_canonical"].shape
        ) != (
            cfg["samples"],
            cfg["future_length"],
            2,
        ):
            raise RuntimeError(
                f"{task}: unexpected future_canonical shape."
            )

        meta = pd.read_csv(
            cfg["metadata"],
            dtype={
                "sample_id": str,
                "source_group": str,
                "track_key": str,
                "segment": str,
            }
        )

        if len(meta) != cfg["samples"]:
            raise RuntimeError(
                f"{task}: metadata rows={len(meta)}, expected={cfg['samples']}"
            )

        if (
            "sample_id" not in meta.columns
            or
            meta["sample_id"].nunique() != len(meta)
        ):
            raise RuntimeError(
                f"{task}: metadata sample_id is not unique."
            )

        rows.append(
            {
                "task": task,
                "samples": cfg["samples"],
                "future_length": cfg["future_length"],
                "npz": str(cfg["npz"]),
                "npz_sha256": cfg["npz_sha256"],
                "metadata": str(cfg["metadata"]),
                "metadata_sha256": cfg["metadata_sha256"],
                "tracks": (
                    int(meta["track_key"].nunique())
                    if "track_key" in meta.columns
                    else None
                ),
                "sources": (
                    int(meta["source_group"].nunique())
                    if "source_group" in meta.columns
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def make_definition_only_module(source_path, module_name):
    text = source_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text,
        filename=str(source_path)
    )

    definition_nodes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    first_definition_line = min(
        (
            int(node.lineno)
            for node in tree.body
            if isinstance(
                node,
                definition_nodes
            )
        ),
        default=10**9,
    )

    body = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
            )
        ):
            body.append(node)

        elif (
            isinstance(
                node,
                (
                    ast.Assign,
                    ast.AnnAssign,
                )
            )
            and
            int(node.lineno) < first_definition_line
        ):
            body.append(node)

        elif isinstance(
            node,
            definition_nodes
        ):
            body.append(node)

    clean_tree = ast.Module(
        body=body,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        clean_tree
    )

    ns = {
        "__name__": module_name,
        "__file__": str(source_path),
    }

    exec(
        compile(
            clean_tree,
            str(source_path),
            "exec"
        ),
        ns
    )

    return ns


def load_checkpoint_inventory():
    df = pd.read_csv(
        R10A_PRED_INVENTORY,
        dtype={
            "task": str,
            "dataset_label": str,
            "checkpoint_path": str,
            "checkpoint_sha256": str,
        }
    )

    df = df[
        df["dataset_label"] == "BIRDSAI_test"
    ].copy()

    mapping = {}

    for _, row in df.iterrows():
        key = (
            str(row["task"]),
            int(row["seed"]),
        )

        if key in mapping:
            continue

        p = Path(
            str(row["checkpoint_path"])
        )

        require(p)

        current_hash = sha256_file(p)

        if current_hash != str(
            row["checkpoint_sha256"]
        ):
            raise RuntimeError(
                f"Checkpoint hash changed: {p}"
            )

        mapping[key] = {
            "path": p,
            "sha256": current_hash,
        }

    if len(mapping) != 6:
        raise RuntimeError(
            f"Expected 6 unique Transformer checkpoints, found {len(mapping)}."
        )

    return mapping


def load_checkpoint(path):
    import torch

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def construct_model(
    task,
    checkpoint,
    transformer_ns,
):
    future_length = TASKS[task]["future_length"]

    model = transformer_ns[
        "TrajectoryTransformer"
    ](
        future_length
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    return model


def prediction_path(task, seed):
    return (
        PRED_OUT
        /
        f"{task}_{DATASET_NAME}_Transformer_seed{seed}_predictions.npz"
    )


def compute_sample_metrics(
    task,
    seed,
    pred_path,
    metadata_path,
):
    arrays = np.load(
        pred_path,
        allow_pickle=False
    )

    truth_abs = np.asarray(
        arrays["truth_abs"],
        dtype=np.float64
    )

    pred_abs = np.asarray(
        arrays["prediction_abs"],
        dtype=np.float64
    )

    cfg = TASKS[task]

    expected_shape = (
        cfg["samples"],
        cfg["future_length"],
        2,
    )

    if truth_abs.shape != expected_shape:
        raise RuntimeError(
            f"{pred_path.name}: truth shape={truth_abs.shape}, expected={expected_shape}"
        )

    if pred_abs.shape != expected_shape:
        raise RuntimeError(
            f"{pred_path.name}: pred shape={pred_abs.shape}, expected={expected_shape}"
        )

    error = np.linalg.norm(
        pred_abs
        -
        truth_abs,
        axis=2
    )

    sample_ade = error.mean(
        axis=1
    )

    sample_fde = error[
        :,
        -1
    ]

    meta = pd.read_csv(
        metadata_path,
        dtype={
            "sample_id": str,
            "source_group": str,
            "track_key": str,
            "segment": str,
        }
    )

    out = meta.copy()

    out["task"] = task
    out["seed"] = seed
    out["ADE"] = sample_ade
    out["FDE"] = sample_fde

    sample_path = (
        SAMPLE_OUT
        /
        f"{task}_Transformer_seed{seed}_sample_metrics.csv"
    )

    out.to_csv(
        sample_path,
        index=False
    )

    summary = {
        "task": task,
        "seed": seed,
        "samples": int(len(out)),
        "tracks": (
            int(out["track_key"].nunique())
            if "track_key" in out.columns
            else None
        ),
        "sources": (
            int(out["source_group"].nunique())
            if "source_group" in out.columns
            else None
        ),
        "ADE_window_mean": float(sample_ade.mean()),
        "FDE_window_mean": float(sample_fde.mean()),
        "sample_metrics_csv": str(sample_path),
        "sample_metrics_sha256": sha256_file(sample_path),
        "prediction_npz": str(pred_path),
        "prediction_npz_sha256": sha256_file(pred_path),
    }

    if "track_key" in out.columns:
        track = (
            out.groupby(
                "track_key",
                dropna=False
            )[
                [
                    "ADE",
                    "FDE",
                ]
            ]
            .mean()
        )

        summary["ADE_track_macro"] = float(
            track["ADE"].mean()
        )

        summary["FDE_track_macro"] = float(
            track["FDE"].mean()
        )

    if "source_group" in out.columns:
        source = (
            out.groupby(
                "source_group",
                dropna=False
            )[
                [
                    "ADE",
                    "FDE",
                ]
            ]
            .mean()
        )

        summary["ADE_source_macro"] = float(
            source["ADE"].mean()
        )

        summary["FDE_source_macro"] = float(
            source["FDE"].mean()
        )

    return summary


def run_one(
    task,
    seed,
    transformer_ns,
    checkpoint_lock,
):
    import torch

    ckpt_info = checkpoint_lock[
        (
            task,
            seed,
        )
    ]

    checkpoint = load_checkpoint(
        ckpt_info["path"]
    )

    if str(
        checkpoint.get("task")
    ) != task:
        raise RuntimeError(
            f"{task}/seed{seed}: checkpoint task mismatch."
        )

    if int(
        checkpoint.get("seed")
    ) != seed:
        raise RuntimeError(
            f"{task}/seed{seed}: checkpoint seed mismatch."
        )

    transformer_ns["OUT"] = PRED_OUT

    model = construct_model(
        task,
        checkpoint,
        transformer_ns,
    )

    device = transformer_ns.get(
        "DEVICE",
        "cpu"
    )

    model = model.to(
        device
    )

    model.eval()

    best_epoch = int(
        checkpoint["best_epoch"]
    )

    log_path = (
        LOG_OUT
        /
        f"{task}_Transformer_seed{seed}.log"
    )

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        f"FRESH WILDBOX INFERENCE START | task={task} | seed={seed} | best_epoch={best_epoch}"
    )

    print(
        "=" * 100
    )

    cfg = TASKS[task]

    with open(
        log_path,
        "w",
        encoding="utf-8"
    ) as log_f:
        tee = Tee(
            sys.stdout,
            log_f,
        )

        with contextlib.redirect_stdout(
            tee
        ), contextlib.redirect_stderr(
            tee
        ):
            with torch.no_grad():
                transformer_ns[
                    "evaluate"
                ](
                    model,
                    task,
                    DATASET_NAME,
                    ROLE,
                    cfg["npz"],
                    cfg["metadata"],
                    seed,
                    best_epoch,
                )

    pred_path = prediction_path(
        task,
        seed,
    )

    require(pred_path)

    summary = compute_sample_metrics(
        task,
        seed,
        pred_path,
        cfg["metadata"],
    )

    summary["best_epoch"] = best_epoch
    summary[
        "best_validation_track_macro_ADE"
    ] = float(
        checkpoint[
            "best_validation_track_macro_ADE"
        ]
    )

    summary["checkpoint"] = str(
        ckpt_info["path"]
    )

    summary["checkpoint_sha256"] = (
        ckpt_info["sha256"]
    )

    print(
        f"[COMPLETE] {task} | seed={seed} | "
        f"ADE={summary['ADE_window_mean']:.6f} | "
        f"FDE={summary['FDE_window_mean']:.6f}"
    )

    return summary


def build_ensemble(seed_results):
    ensemble_rows = []
    sample_outputs = []

    for task, cfg in TASKS.items():
        preds = []
        truths = []

        for seed in SEEDS:
            row = next(
                r
                for r in seed_results
                if r["task"] == task
                and int(r["seed"]) == seed
            )

            arrays = np.load(
                row["prediction_npz"],
                allow_pickle=False
            )

            preds.append(
                np.asarray(
                    arrays["prediction_abs"],
                    dtype=np.float64
                )
            )

            truths.append(
                np.asarray(
                    arrays["truth_abs"],
                    dtype=np.float64
                )
            )

        truth0 = truths[0]

        for truth in truths[1:]:
            if not np.array_equal(
                truth,
                truth0
            ):
                raise RuntimeError(
                    f"{task}: truth differs across fresh WildBox seeds."
                )

        ensemble_pred = np.mean(
            np.stack(
                preds,
                axis=0
            ),
            axis=0
        )

        error = np.linalg.norm(
            ensemble_pred
            -
            truth0,
            axis=2
        )

        ade = error.mean(
            axis=1
        )

        fde = error[
            :,
            -1
        ]

        meta = pd.read_csv(
            cfg["metadata"],
            dtype={
                "sample_id": str,
                "source_group": str,
                "track_key": str,
                "segment": str,
            }
        )

        sample = meta.copy()
        sample["task"] = task
        sample["model"] = "TransformerEnsemble"
        sample["ADE"] = ade
        sample["FDE"] = fde

        sample_path = (
            SAMPLE_OUT
            /
            f"{task}_TransformerEnsemble_sample_metrics.csv"
        )

        sample.to_csv(
            sample_path,
            index=False
        )

        sample_outputs.append(
            sample_path
        )

        row = {
            "task": task,
            "model": "TransformerEnsemble",
            "samples": int(len(sample)),
            "tracks": (
                int(sample["track_key"].nunique())
                if "track_key" in sample.columns
                else None
            ),
            "sources": (
                int(sample["source_group"].nunique())
                if "source_group" in sample.columns
                else None
            ),
            "ADE_window_mean": float(ade.mean()),
            "FDE_window_mean": float(fde.mean()),
            "sample_metrics_csv": str(sample_path),
            "sample_metrics_sha256": sha256_file(sample_path),
        }

        if "track_key" in sample.columns:
            track = (
                sample.groupby(
                    "track_key",
                    dropna=False
                )[
                    [
                        "ADE",
                        "FDE",
                    ]
                ]
                .mean()
            )

            row["ADE_track_macro"] = float(
                track["ADE"].mean()
            )

            row["FDE_track_macro"] = float(
                track["FDE"].mean()
            )

        if "source_group" in sample.columns:
            source = (
                sample.groupby(
                    "source_group",
                    dropna=False
                )[
                    [
                        "ADE",
                        "FDE",
                    ]
                ]
                .mean()
            )

            row["ADE_source_macro"] = float(
                source["ADE"].mean()
            )

            row["FDE_source_macro"] = float(
                source["FDE"].mean()
            )

        ensemble_rows.append(
            row
        )

    return (
        pd.DataFrame(ensemble_rows),
        sample_outputs,
    )


def compare_to_r2_b1(ensemble_df):
    r2 = pd.read_csv(
        R2_B1_SAMPLE_METRICS,
        dtype={
            "dataset": str,
            "task": str,
            "model": str,
            "sample_id": str,
        }
    )

    rows = []

    for task, cfg in TASKS.items():
        fresh_mask = (
            r2["dataset"]
            .astype(str)
            .str.lower()
            .str.contains(
                "wildbox|fresh",
                regex=True
            )
        )

        task_mask = (
            r2["task"]
            .astype(str)
            ==
            task
        )

        model_mask = (
            r2["model"]
            .astype(str)
            .str.lower()
            .str.contains(
                "transformer"
            )
        )

        cand = r2[
            fresh_mask
            &
            task_mask
            &
            model_mask
        ].copy()

        # Expect exactly one Transformer ensemble row per sample.
        if len(cand) != cfg["samples"]:
            rows.append(
                {
                    "task": task,
                    "status": "R2_TRANSFORMER_ROWS_NOT_UNIQUE_OR_COUNT_MISMATCH",
                    "r2_rows": int(len(cand)),
                    "expected_rows": cfg["samples"],
                    "max_abs_ADE_difference": None,
                    "max_abs_FDE_difference": None,
                    "mean_abs_ADE_difference": None,
                    "mean_abs_FDE_difference": None,
                }
            )
            continue

        ours_path = Path(
            ensemble_df[
                ensemble_df["task"] == task
            ].iloc[0][
                "sample_metrics_csv"
            ]
        )

        ours = pd.read_csv(
            ours_path,
            dtype={
                "sample_id": str,
            }
        )

        merged = ours[
            [
                "sample_id",
                "ADE",
                "FDE",
            ]
        ].merge(
            cand[
                [
                    "sample_id",
                    "ade",
                    "fde",
                ]
            ],
            on="sample_id",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        if not (
            merged["_merge"] == "both"
        ).all():
            rows.append(
                {
                    "task": task,
                    "status": "SAMPLE_ID_ALIGNMENT_FAILED",
                    "r2_rows": int(len(cand)),
                    "expected_rows": cfg["samples"],
                    "max_abs_ADE_difference": None,
                    "max_abs_FDE_difference": None,
                    "mean_abs_ADE_difference": None,
                    "mean_abs_FDE_difference": None,
                }
            )
            continue

        d_ade = np.abs(
            merged["ADE"].to_numpy(
                dtype=np.float64
            )
            -
            merged["ade"].to_numpy(
                dtype=np.float64
            )
        )

        d_fde = np.abs(
            merged["FDE"].to_numpy(
                dtype=np.float64
            )
            -
            merged["fde"].to_numpy(
                dtype=np.float64
            )
        )

        rows.append(
            {
                "task": task,
                "status": "R2_ENSEMBLE_SAMPLE_METRICS_ALIGNED",
                "r2_rows": int(len(cand)),
                "expected_rows": cfg["samples"],
                "max_abs_ADE_difference": float(
                    d_ade.max()
                ),
                "max_abs_FDE_difference": float(
                    d_fde.max()
                ),
                "mean_abs_ADE_difference": float(
                    d_ade.mean()
                ),
                "mean_abs_FDE_difference": float(
                    d_fde.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR10-C1 — FROZEN TRANSFORMER REINFERENCE ON FRESH WILDBOX 865/637"
    )

    print(
        "NO training, NO tuning, NO recalibration. "
        "This step uses the exact fresh WildBox canonical NPZ/metadata files locked in R10-C0B "
        "and the original frozen anisotropic Transformer checkpoints for seeds 42/52/62. "
        "The original train_transformer.py::evaluate function is used directly; no schema adapter "
        "or data transformation is introduced. Native anisotropic normalized ADE/FDE are valid here "
        "because all ensemble comparisons remain within the same frozen representation. "
        "The 3-seed prediction ensemble is reconstructed and its sample metrics are checked against "
        "the already-completed R2 fair-comparison Transformer ensemble rows.\n"
    )

    verify_upstream()
    input_df = verify_fresh_inputs()
    checkpoint_lock = load_checkpoint_inventory()

    transformer_ns = make_definition_only_module(
        TRANSFORMER_SRC,
        "r10c1_transformer_defs",
    )

    if "evaluate" not in transformer_ns:
        raise RuntimeError(
            "Original Transformer evaluate function not loaded."
        )

    results = []

    try:
        for task in TASKS:
            for seed in SEEDS:
                results.append(
                    run_one(
                        task,
                        seed,
                        transformer_ns,
                        checkpoint_lock,
                    )
                )

    except Exception:
        progress = {
            "step":
                "R10C1_FRESH_WILDBOX_FROZEN_TRANSFORMER_REINFERENCE",

            "updated_utc":
                utc_now(),

            "status":
                "R10C1_PARTIAL_OR_FAILED_NO_TRAINING_SAFE_TO_PATCH_AND_RESUME",

            "completed_runs":
                results,

            "training":
                False,

            "tuning":
                False,

            "checkpoint_modification":
                False,

            "traceback":
                traceback.format_exc(),
        }

        (
            OUT
            /
            "R10C1_PROGRESS.json"
        ).write_text(
            json.dumps(
                progress,
                indent=2
            ),
            encoding="utf-8"
        )

        raise

    seed_df = pd.DataFrame(
        results
    )

    seed_path = (
        OUT
        /
        "R10C1_FRESH_WILDBOX_SEED_LEVEL_METRICS.csv"
    )

    seed_df.to_csv(
        seed_path,
        index=False
    )

    ensemble_df, _ = build_ensemble(
        results
    )

    ensemble_path = (
        OUT
        /
        "R10C1_FRESH_WILDBOX_TRANSFORMER_ENSEMBLE_METRICS.csv"
    )

    ensemble_df.to_csv(
        ensemble_path,
        index=False
    )

    r2_check_df = compare_to_r2_b1(
        ensemble_df
    )

    r2_check_path = (
        OUT
        /
        "R10C1_R2_ENSEMBLE_REPRODUCIBILITY_CHECK.csv"
    )

    r2_check_df.to_csv(
        r2_check_path,
        index=False
    )

    # Numerical tolerance is deliberately tight but not array-exact because
    # old R2 rows may have been CSV-rounded.
    aligned = (
        len(
            r2_check_df
        )
        ==
        2
        and
        (
            r2_check_df[
                "status"
            ]
            ==
            "R2_ENSEMBLE_SAMPLE_METRICS_ALIGNED"
        ).all()
    )

    if aligned:
        max_ade = float(
            pd.to_numeric(
                r2_check_df[
                    "max_abs_ADE_difference"
                ],
                errors="coerce"
            ).max()
        )

        max_fde = float(
            pd.to_numeric(
                r2_check_df[
                    "max_abs_FDE_difference"
                ],
                errors="coerce"
            ).max()
        )

        reproducible = (
            max_ade <= 1e-9
            and
            max_fde <= 1e-9
        )
    else:
        max_ade = None
        max_fde = None
        reproducible = False

    if aligned and reproducible:
        status = (
            "R10C1_FRESH_WILDBOX_6_FROZEN_TRANSFORMER_RUNS_COMPLETE_ENSEMBLE_REPRODUCED_READY_FOR_R10C2_EXTERNAL_ENSEMBLE_VALUE_BOOTSTRAP"
        )
    elif aligned:
        status = (
            "R10C1_FRESH_WILDBOX_6_FROZEN_TRANSFORMER_RUNS_COMPLETE_R2_ENSEMBLE_ALIGNED_WITH_SMALL_NUMERICAL_DIFFERENCE_READY_FOR_R10C2"
        )
    else:
        status = (
            "R10C1_FRESH_WILDBOX_INFERENCE_COMPLETE_R2_ENSEMBLE_REPRODUCIBILITY_NEEDS_REVIEW_BEFORE_R10C2"
        )

    manifest = {
        "step":
            "R10C1_FRESH_WILDBOX_FROZEN_TRANSFORMER_REINFERENCE",

        "created_utc":
            utc_now(),

        "status":
            status,

        "runs":
            6,

        "training":
            False,

        "tuning":
            False,

        "recalibration":
            False,

        "checkpoint_modification":
            False,

        "representation":
            "original frozen anisotropic canonical representation",

        "metric_space":
            "native anisotropic normalized coordinates; only within-representation comparisons are made",

        "direct_original_evaluator":
            "train_transformer.py::evaluate",

        "schema_adapter_used":
            False,

        "fresh_sample_counts":
            {
                "2s_to_2s":
                    865,

                "2s_to_5s":
                    637,
            },

        "r2_ensemble_sample_metrics_aligned":
            bool(
                aligned
            ),

        "r2_ensemble_exact_within_1e_9":
            bool(
                reproducible
            ),

        "r2_max_abs_ADE_difference":
            max_ade,

        "r2_max_abs_FDE_difference":
            max_fde,

        "outputs": {
            "input_lock":
                str(
                    OUT
                    /
                    "R10C1_FRESH_INPUT_LOCK.csv"
                ),

            "seed_metrics":
                str(
                    seed_path
                ),

            "ensemble_metrics":
                str(
                    ensemble_path
                ),

            "r2_reproducibility_check":
                str(
                    r2_check_path
                ),
        },
    }

    input_lock_path = (
        OUT
        /
        "R10C1_FRESH_INPUT_LOCK.csv"
    )

    input_df.to_csv(
        input_lock_path,
        index=False
    )

    (
        OUT
        /
        "R10C1_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "FRESH WILDBOX INPUT LOCK:"
    )

    print(
        input_df.to_string(
            index=False
        )
    )

    print(
        "\nSEED-LEVEL FRESH WILDBOX TRANSFORMER METRICS:"
    )

    print(
        seed_df[
            [
                "task",
                "seed",
                "samples",
                "tracks",
                "sources",
                "ADE_window_mean",
                "FDE_window_mean",
                "ADE_track_macro",
                "FDE_track_macro",
                "best_epoch",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n3-SEED TRANSFORMER ENSEMBLE METRICS:"
    )

    print(
        ensemble_df.to_string(
            index=False
        )
    )

    print(
        "\nR2 ENSEMBLE REPRODUCIBILITY CHECK:"
    )

    print(
        r2_check_df.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR10-C1 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
