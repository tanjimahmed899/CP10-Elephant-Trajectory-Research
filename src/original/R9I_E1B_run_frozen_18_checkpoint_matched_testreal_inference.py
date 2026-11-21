
from pathlib import Path
import ast
import contextlib
import hashlib
import io
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

R9I_D1_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_D1_matched_isotropic_training_only"
    / "R9I_D1_MANIFEST.json"
)

R9I_E0_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_E0_matched_testreal_isotropic_dataset"
    / "R9I_E0_MANIFEST.json"
)

R9I_E1A_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_E1A_model_reconstruction_and_evaluation_audit"
    / "R9I_E1A_MANIFEST.json"
)

CHECKPOINT_ROOT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_D1_matched_isotropic_training_only"
)

TESTREAL_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_E0_matched_testreal_isotropic_dataset"
    / "dataset"
)

GEOMETRY_CSV = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3B_corrected_isotropic_rebuild"
    / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_E1B_frozen_18_checkpoint_testreal_inference"
)

TRANSFORMER_OUT = OUT / "transformer"
RECURRENT_OUT = OUT / "lstm_gru"
SAMPLE_OUT = OUT / "sample_metrics"
LOG_OUT = OUT / "logs"

for p in [
    OUT,
    TRANSFORMER_OUT,
    RECURRENT_OUT,
    SAMPLE_OUT,
    LOG_OUT,
]:
    p.mkdir(
        parents=True,
        exist_ok=True
    )

TASKS = {
    "2s_to_2s": {
        "expected_samples": 1053,
        "future_length": 20,
    },
    "2s_to_5s": {
        "expected_samples": 826,
        "future_length": 50,
    },
}

SEEDS = [
    42,
    52,
    62,
]

MODELS = [
    "Transformer",
    "LSTM",
    "GRU",
]

DATASET_NAME = "BIRDSAI_TestReal_Isotropic"
ROLE = "matched_testreal"


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


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
    def __init__(
        self,
        *streams,
    ):
        self.streams = streams

    def write(
        self,
        data,
    ):
        for stream in self.streams:
            stream.write(
                data
            )

            stream.flush()

        return len(
            data
        )

    def flush(
        self,
    ):
        for stream in self.streams:
            stream.flush()


def verify_manifests():
    for p in [
        TRANSFORMER_SRC,
        RECURRENT_SRC,
        R9I_D1_MANIFEST,
        R9I_E0_MANIFEST,
        R9I_E1A_MANIFEST,
        GEOMETRY_CSV,
    ]:
        require(p)

    d1 = json.loads(
        R9I_D1_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    e0 = json.loads(
        R9I_E0_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    e1a = json.loads(
        R9I_E1A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        d1.get("status")
        !=
        "R9I_D1_ALL_18_MATCHED_ISOTROPIC_NEURAL_RUNS_TRAINED_READY_FOR_R9I_E_MATCHED_TESTREAL_ISOTROPIC_EVALUATION_BUILD"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-D1 status: {d1.get('status')}"
        )

    if (
        e0.get("status")
        !=
        "R9I_E0_MATCHED_TESTREAL_ISOTROPIC_CANONICAL_DATASETS_BUILT_READY_FOR_R9I_E1_FROZEN_18_CHECKPOINT_TESTREAL_INFERENCE"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-E0 status: {e0.get('status')}"
        )

    if (
        e1a.get("status")
        !=
        "R9I_E1A_MODEL_RECONSTRUCTION_AND_TESTREAL_EVALUATION_PROTOCOL_AUDITED_READY_FOR_R9I_E1B_FROZEN_18_CHECKPOINT_INFERENCE"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-E1A status: {e1a.get('status')}"
        )

    if int(
        e1a.get(
            "checkpoints_verified",
            -1
        )
    ) != 18:
        raise RuntimeError(
            "R9-I-E1A did not verify exactly 18 checkpoints."
        )

    return (
        d1,
        e0,
        e1a,
    )


def make_definition_only_module(
    source_path,
    module_name,
):
    text = source_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text,
        filename=str(
            source_path
        )
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
            body.append(
                node
            )

        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ) and int(
            node.lineno
        ) < first_definition_line:
            body.append(
                node
            )

        elif isinstance(
            node,
            definition_nodes
        ):
            body.append(
                node
            )

    clean_tree = ast.Module(
        body=body,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        clean_tree
    )

    namespace = {
        "__name__":
            module_name,

        "__file__":
            str(
                source_path
            ),
    }

    exec(
        compile(
            clean_tree,
            str(
                source_path
            ),
            "exec"
        ),
        namespace
    )

    return namespace


def load_geometry():
    df = pd.read_csv(
        GEOMETRY_CSV,
        dtype={
            "sequence": str,
        }
    )

    required = {
        "sequence",
        "iso_scale_px",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"Geometry lock missing columns: {sorted(missing)}"
        )

    return {
        str(
            row[
                "sequence"
            ]
        ):
            float(
                row[
                    "iso_scale_px"
                ]
            )
        for _, row in df.iterrows()
    }


def checkpoint_path(
    task,
    model,
    seed,
):
    if model == "Transformer":
        return (
            CHECKPOINT_ROOT
            /
            "transformer"
            /
            f"{task}_Transformer_seed{seed}.pt"
        )

    return (
        CHECKPOINT_ROOT
        /
        "lstm_gru"
        /
        f"{task}_{model}_seed{seed}.pt"
    )


def build_checkpoint_hash_lock(
    d1_manifest,
):
    expected = {}

    for row in d1_manifest[
        "runs"
    ]:
        expected[
            (
                row[
                    "task"
                ],
                row[
                    "model"
                ],
                int(
                    row[
                        "seed"
                    ]
                ),
            )
        ] = row[
            "checkpoint_sha256"
        ]

    return expected


def testreal_paths(
    task,
):
    return (
        TESTREAL_DIR
        /
        f"{task}_birdsai_test_canonical.npz",
        TESTREAL_DIR
        /
        f"{task}_birdsai_test_metadata.csv",
    )


def build_testreal_hash_lock(
    e0_manifest,
):
    out = {}

    for row in e0_manifest[
        "tasks"
    ]:
        out[
            row[
                "task"
            ]
        ] = {
            "npz_sha256":
                row[
                    "output_npz_sha256"
                ],

            "metadata_sha256":
                row[
                    "output_metadata_sha256"
                ],
        }

    return out


def verify_all_hashes(
    d1_manifest,
    e0_manifest,
):
    ckpt_hashes = build_checkpoint_hash_lock(
        d1_manifest
    )

    test_hashes = build_testreal_hash_lock(
        e0_manifest
    )

    for task in TASKS:
        npz_path, meta_path = testreal_paths(
            task
        )

        require(
            npz_path
        )

        require(
            meta_path
        )

        if (
            sha256_file(
                npz_path
            )
            !=
            test_hashes[
                task
            ][
                "npz_sha256"
            ]
        ):
            raise RuntimeError(
                f"{task}: TestReal NPZ hash changed."
            )

        if (
            sha256_file(
                meta_path
            )
            !=
            test_hashes[
                task
            ][
                "metadata_sha256"
            ]
        ):
            raise RuntimeError(
                f"{task}: TestReal metadata hash changed."
            )

        for model in MODELS:
            for seed in SEEDS:
                p = checkpoint_path(
                    task,
                    model,
                    seed,
                )

                require(
                    p
                )

                key = (
                    task,
                    model,
                    seed,
                )

                if key not in ckpt_hashes:
                    raise RuntimeError(
                        f"Checkpoint identity absent from D1 manifest: {key}"
                    )

                if (
                    sha256_file(
                        p
                    )
                    !=
                    ckpt_hashes[
                        key
                    ]
                ):
                    raise RuntimeError(
                        f"Checkpoint hash changed: {p}"
                    )

    return (
        ckpt_hashes,
        test_hashes,
    )


def construct_model(
    task,
    model_name,
    checkpoint,
    transformer_ns,
    recurrent_ns,
):
    future_length = TASKS[
        task
    ][
        "future_length"
    ]

    if model_name == "Transformer":
        model = transformer_ns[
            "TrajectoryTransformer"
        ](
            future_length
        )

    else:
        model = recurrent_ns[
            "RecurrentForecaster"
        ](
            model_name,
            future_length
        )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    return model


def load_checkpoint(
    path,
):
    import torch

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def serialize_return_object(
    obj,
    stem,
):
    """
    Save whatever the original evaluate function returns, without assuming
    its exact structure. This is provenance only; the primary numerical
    summary is independently recomputed from the saved prediction NPZ.
    """
    saved = []

    if obj is None:
        return saved

    def handle(
        value,
        name,
    ):
        if isinstance(
            value,
            pd.DataFrame
        ):
            p = (
                OUT
                /
                f"{stem}_{name}.csv"
            )

            value.to_csv(
                p,
                index=False
            )

            saved.append(
                str(
                    p
                )
            )

            return

        if isinstance(
            value,
            dict
        ):
            safe = {}

            for k, v in value.items():
                if isinstance(
                    v,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    )
                ):
                    safe[
                        str(
                            k
                        )
                    ] = v

                elif isinstance(
                    v,
                    np.generic
                ):
                    safe[
                        str(
                            k
                        )
                    ] = v.item()

                else:
                    safe[
                        str(
                            k
                        )
                    ] = repr(
                        v
                    )

            p = (
                OUT
                /
                f"{stem}_{name}.json"
            )

            p.write_text(
                json.dumps(
                    safe,
                    indent=2
                ),
                encoding="utf-8"
            )

            saved.append(
                str(
                    p
                )
            )

            return

        p = (
            OUT
            /
            f"{stem}_{name}.txt"
        )

        p.write_text(
            repr(
                value
            ),
            encoding="utf-8"
        )

        saved.append(
            str(
                p
            )
        )

    if isinstance(
        obj,
        (
            tuple,
            list,
        )
    ):
        for i, item in enumerate(
            obj
        ):
            handle(
                item,
                f"return_{i}"
            )

    else:
        handle(
            obj,
            "return"
        )

    return saved


def prediction_path(
    task,
    model,
    seed,
):
    if model == "Transformer":
        return (
            TRANSFORMER_OUT
            /
            f"{task}_{DATASET_NAME}_Transformer_seed{seed}_predictions.npz"
        )

    return (
        RECURRENT_OUT
        /
        f"{task}_{DATASET_NAME}_{model}_seed{seed}_predictions.npz"
    )


def sample_scale_vector(
    meta,
    geometry,
):
    if "sequence" in meta.columns:
        seqs = meta[
            "sequence"
        ].astype(str)

    elif "source_group" in meta.columns:
        seqs = meta[
            "source_group"
        ].astype(str)

    else:
        raise RuntimeError(
            "TestReal metadata has neither sequence nor source_group."
        )

    missing = sorted(
        set(
            seqs
        )
        -
        set(
            geometry
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing pixel scale geometry for sequences: {missing}"
        )

    return np.asarray(
        [
            geometry[
                seq
            ]
            for seq in seqs
        ],
        dtype=np.float64
    )


def compute_independent_metrics(
    task,
    model,
    seed,
    prediction_npz,
    metadata_path,
    geometry,
):
    arrays = np.load(
        prediction_npz,
        allow_pickle=False
    )

    required = {
        "truth_abs",
        "prediction_abs",
        "prediction_canonical",
    }

    missing = required - set(
        arrays.files
    )

    if missing:
        raise RuntimeError(
            f"{prediction_npz.name}: missing prediction keys {sorted(missing)}"
        )

    truth_abs = np.asarray(
        arrays[
            "truth_abs"
        ],
        dtype=np.float64
    )

    prediction_abs = np.asarray(
        arrays[
            "prediction_abs"
        ],
        dtype=np.float64
    )

    prediction_can = np.asarray(
        arrays[
            "prediction_canonical"
        ]
    )

    cfg = TASKS[
        task
    ]

    expected_shape = (
        cfg[
            "expected_samples"
        ],
        cfg[
            "future_length"
        ],
        2,
    )

    if truth_abs.shape != expected_shape:
        raise RuntimeError(
            f"{prediction_npz.name}: truth_abs shape={truth_abs.shape}, expected={expected_shape}"
        )

    if prediction_abs.shape != expected_shape:
        raise RuntimeError(
            f"{prediction_npz.name}: prediction_abs shape={prediction_abs.shape}, expected={expected_shape}"
        )

    if prediction_can.shape != expected_shape:
        raise RuntimeError(
            f"{prediction_npz.name}: prediction_canonical shape={prediction_can.shape}, expected={expected_shape}"
        )

    if not np.isfinite(
        truth_abs
    ).all():
        raise RuntimeError(
            f"{prediction_npz.name}: non-finite truth_abs."
        )

    if not np.isfinite(
        prediction_abs
    ).all():
        raise RuntimeError(
            f"{prediction_npz.name}: non-finite prediction_abs."
        )

    meta = pd.read_csv(
        metadata_path,
        dtype={
            "sample_id": str,
            "sequence": str,
            "source_group": str,
            "segment_id": str,
            "track_key": str,
        }
    )

    if len(
        meta
    ) != cfg[
        "expected_samples"
    ]:
        raise RuntimeError(
            f"{task}: metadata rows={len(meta)}, expected={cfg['expected_samples']}"
        )

    err_iso = np.linalg.norm(
        prediction_abs
        -
        truth_abs,
        axis=2
    )

    sample_ade_iso = err_iso.mean(
        axis=1
    )

    sample_fde_iso = err_iso[
        :,
        -1
    ]

    scale_px = sample_scale_vector(
        meta,
        geometry
    )

    err_px = (
        err_iso
        *
        scale_px[
            :,
            None
        ]
    )

    sample_ade_px = err_px.mean(
        axis=1
    )

    sample_fde_px = err_px[
        :,
        -1
    ]

    sample_df = meta.copy()

    sample_df[
        "task"
    ] = task

    sample_df[
        "model"
    ] = model

    sample_df[
        "seed"
    ] = seed

    sample_df[
        "ADE_iso"
    ] = sample_ade_iso

    sample_df[
        "FDE_iso"
    ] = sample_fde_iso

    sample_df[
        "ADE_px"
    ] = sample_ade_px

    sample_df[
        "FDE_px"
    ] = sample_fde_px

    sample_df[
        "iso_scale_px"
    ] = scale_px

    sample_path = (
        SAMPLE_OUT
        /
        f"{task}_{model}_seed{seed}_sample_metrics.csv"
    )

    sample_df.to_csv(
        sample_path,
        index=False
    )

    summary = {
        "task":
            task,

        "model":
            model,

        "seed":
            seed,

        "samples":
            int(
                len(
                    sample_df
                )
            ),

        "ADE_iso_window_mean":
            float(
                sample_ade_iso.mean()
            ),

        "FDE_iso_window_mean":
            float(
                sample_fde_iso.mean()
            ),

        "ADE_px_window_mean":
            float(
                sample_ade_px.mean()
            ),

        "FDE_px_window_mean":
            float(
                sample_fde_px.mean()
            ),

        "prediction_npz":
            str(
                prediction_npz
            ),

        "prediction_npz_sha256":
            sha256_file(
                prediction_npz
            ),

        "sample_metrics_csv":
            str(
                sample_path
            ),

        "sample_metrics_sha256":
            sha256_file(
                sample_path
            ),
    }

    if "track_key" in sample_df.columns:
        track_df = (
            sample_df.groupby(
                "track_key",
                dropna=False
            )[
                [
                    "ADE_iso",
                    "FDE_iso",
                    "ADE_px",
                    "FDE_px",
                ]
            ]
            .mean()
        )

        summary[
            "tracks"
        ] = int(
            len(
                track_df
            )
        )

        summary[
            "ADE_iso_track_macro"
        ] = float(
            track_df[
                "ADE_iso"
            ].mean()
        )

        summary[
            "FDE_iso_track_macro"
        ] = float(
            track_df[
                "FDE_iso"
            ].mean()
        )

        summary[
            "ADE_px_track_macro"
        ] = float(
            track_df[
                "ADE_px"
            ].mean()
        )

        summary[
            "FDE_px_track_macro"
        ] = float(
            track_df[
                "FDE_px"
            ].mean()
        )

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
            sample_df.groupby(
                source_col,
                dropna=False
            )[
                [
                    "ADE_iso",
                    "FDE_iso",
                    "ADE_px",
                    "FDE_px",
                ]
            ]
            .mean()
        )

        summary[
            "sequences"
        ] = int(
            len(
                source_df
            )
        )

        summary[
            "ADE_iso_sequence_macro"
        ] = float(
            source_df[
                "ADE_iso"
            ].mean()
        )

        summary[
            "FDE_iso_sequence_macro"
        ] = float(
            source_df[
                "FDE_iso"
            ].mean()
        )

        summary[
            "ADE_px_sequence_macro"
        ] = float(
            source_df[
                "ADE_px"
            ].mean()
        )

        summary[
            "FDE_px_sequence_macro"
        ] = float(
            source_df[
                "FDE_px"
            ].mean()
        )

    return (
        summary,
        sample_df,
    )


def run_one(
    task,
    model_name,
    seed,
    transformer_ns,
    recurrent_ns,
    geometry,
    checkpoint_hashes,
):
    import torch

    ckpt_path = checkpoint_path(
        task,
        model_name,
        seed,
    )

    npz_path, meta_path = testreal_paths(
        task
    )

    expected_hash = checkpoint_hashes[
        (
            task,
            model_name,
            seed,
        )
    ]

    if sha256_file(
        ckpt_path
    ) != expected_hash:
        raise RuntimeError(
            f"Checkpoint hash changed before inference: {ckpt_path}"
        )

    checkpoint = load_checkpoint(
        ckpt_path
    )

    if str(
        checkpoint.get(
            "task"
        )
    ) != task:
        raise RuntimeError(
            f"{ckpt_path.name}: task mismatch."
        )

    if int(
        checkpoint.get(
            "seed"
        )
    ) != seed:
        raise RuntimeError(
            f"{ckpt_path.name}: seed mismatch."
        )

    if model_name in [
        "LSTM",
        "GRU",
    ]:
        if str(
            checkpoint.get(
                "model_type"
            )
        ) != model_name:
            raise RuntimeError(
                f"{ckpt_path.name}: model_type mismatch."
            )

    if model_name == "Transformer":
        ns = transformer_ns
        out_dir = TRANSFORMER_OUT

    else:
        ns = recurrent_ns
        out_dir = RECURRENT_OUT

    ns[
        "OUT"
    ] = out_dir

    model = construct_model(
        task,
        model_name,
        checkpoint,
        transformer_ns,
        recurrent_ns,
    )

    device = ns.get(
        "DEVICE",
        "cpu"
    )

    model = model.to(
        device
    )

    model.eval()

    best_epoch = int(
        checkpoint[
            "best_epoch"
        ]
    )

    stem = (
        f"{task}_{model_name}_seed{seed}"
    )

    log_path = (
        LOG_OUT
        /
        f"{stem}.log"
    )

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        f"INFERENCE START | task={task} | model={model_name} | seed={seed} | best_epoch={best_epoch}"
    )

    print(
        "=" * 100
    )

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
                if model_name == "Transformer":
                    returned = ns[
                        "evaluate"
                    ](
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
                    returned = ns[
                        "evaluate_dataset"
                    ](
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

    returned_files = serialize_return_object(
        returned,
        stem,
    )

    pred_path = prediction_path(
        task,
        model_name,
        seed,
    )

    require(
        pred_path
    )

    summary, _ = compute_independent_metrics(
        task,
        model_name,
        seed,
        pred_path,
        meta_path,
        geometry,
    )

    summary[
        "best_epoch"
    ] = best_epoch

    summary[
        "best_validation_track_macro_ADE"
    ] = float(
        checkpoint[
            "best_validation_track_macro_ADE"
        ]
    )

    summary[
        "checkpoint"
    ] = str(
        ckpt_path
    )

    summary[
        "checkpoint_sha256"
    ] = expected_hash

    summary[
        "original_evaluate_return_artifacts"
    ] = "|".join(
        returned_files
    )

    print(
        f"[COMPLETE] {task} | {model_name} | seed={seed} | "
        f"ADE_px={summary['ADE_px_window_mean']:.6f} | "
        f"FDE_px={summary['FDE_px_window_mean']:.6f}"
    )

    return summary


def main():
    print(
        "\nR9-I-E1B — FROZEN 18-CHECKPOINT MATCHED BIRDSAI TESTREAL INFERENCE"
    )

    print(
        "NO training, NO tuning, NO recalibration. "
        "All 18 matched isotropic checkpoints are hash-verified and loaded frozen. "
        "The exact original evaluate/evaluate_dataset functions are used on the matched "
        "isotropic BIRDSAI TestReal dataset. Prediction NPZ files are saved in a new "
        "Downloads revision folder. Independent window-, track-, and sequence-level metrics "
        "are recomputed from saved absolute predictions; pixel metrics multiply isotropic "
        "coordinate error by each sequence's locked S=max(W,H).\n"
    )

    (
        d1,
        e0,
        e1a,
    ) = verify_manifests()

    (
        checkpoint_hashes,
        test_hashes,
    ) = verify_all_hashes(
        d1,
        e0,
    )

    geometry = load_geometry()

    transformer_ns = make_definition_only_module(
        TRANSFORMER_SRC,
        "r9i_e1b_transformer_defs",
    )

    recurrent_ns = make_definition_only_module(
        RECURRENT_SRC,
        "r9i_e1b_recurrent_defs",
    )

    if "evaluate" not in transformer_ns:
        raise RuntimeError(
            "Transformer evaluate function not loaded."
        )

    if "evaluate_dataset" not in recurrent_ns:
        raise RuntimeError(
            "Recurrent evaluate_dataset function not loaded."
        )

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
                            checkpoint_hashes,
                        )
                    )

    except Exception:
        progress = {
            "step":
                "R9I_E1B_FROZEN_18_CHECKPOINT_TESTREAL_INFERENCE",

            "updated_utc":
                utc_now(),

            "status":
                "R9I_E1B_PARTIAL_OR_FAILED_NO_TRAINING_SAFE_TO_RESUME_AFTER_PATCH",

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
            "R9I_E1B_PROGRESS.json"
        ).write_text(
            json.dumps(
                progress,
                indent=2
            ),
            encoding="utf-8"
        )

        raise

    result_df = pd.DataFrame(
        results
    )

    result_path = (
        OUT
        /
        "R9I_E1B_SEED_LEVEL_TESTREAL_METRICS.csv"
    )

    result_df.to_csv(
        result_path,
        index=False
    )

    aggregate_rows = []

    metric_cols = [
        "ADE_iso_window_mean",
        "FDE_iso_window_mean",
        "ADE_px_window_mean",
        "FDE_px_window_mean",
        "ADE_iso_track_macro",
        "FDE_iso_track_macro",
        "ADE_px_track_macro",
        "FDE_px_track_macro",
        "ADE_iso_sequence_macro",
        "FDE_iso_sequence_macro",
        "ADE_px_sequence_macro",
        "FDE_px_sequence_macro",
    ]

    for (
        task,
        model_name
    ), group in result_df.groupby(
        [
            "task",
            "model",
        ],
        sort=False
    ):
        row = {
            "task":
                task,

            "model":
                model_name,

            "seeds":
                "|".join(
                    str(
                        int(
                            x
                        )
                    )
                    for x in sorted(
                        group[
                            "seed"
                        ].tolist()
                    )
                ),

            "n_seeds":
                int(
                    len(
                        group
                    )
                ),
        }

        for col in metric_cols:
            if col not in group.columns:
                continue

            values = pd.to_numeric(
                group[
                    col
                ],
                errors="coerce"
            )

            row[
                f"{col}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{col}_std"
            ] = float(
                values.std(
                    ddof=1
                )
            )

        aggregate_rows.append(
            row
        )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )

    aggregate_path = (
        OUT
        /
        "R9I_E1B_3SEED_MODEL_SUMMARY.csv"
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False
    )

    status = (
        "R9I_E1B_ALL_18_FROZEN_ISOTROPIC_CHECKPOINTS_EVALUATED_ON_MATCHED_TESTREAL_READY_FOR_R9I_E2_PAIRED_ANISOTROPIC_VS_ISOTROPIC_COMPARISON"
    )

    manifest = {
        "step":
            "R9I_E1B_FROZEN_18_CHECKPOINT_MATCHED_TESTREAL_INFERENCE",

        "created_utc":
            utc_now(),

        "status":
            status,

        "dataset":
            "BIRDSAI TestReal matched isotropic representation",

        "runs":
            18,

        "training":
            False,

        "tuning":
            False,

        "recalibration":
            False,

        "checkpoint_modification":
            False,

        "original_anisotropic_results_modified":
            False,

        "evaluation_functions":
            {
                "Transformer":
                    "original train_transformer.py::evaluate",

                "LSTM_GRU":
                    "original train_lstm_gru.py::evaluate_dataset",
            },

        "primary_fair_comparison_metric_for_next_step":
            "pixel-space ADE/FDE on exact matched TestReal samples",

        "native_isotropic_metrics_are_representation_specific":
            True,

        "seed_level_metrics_csv":
            str(
                result_path
            ),

        "seed_level_metrics_sha256":
            sha256_file(
                result_path
            ),

        "three_seed_summary_csv":
            str(
                aggregate_path
            ),

        "three_seed_summary_sha256":
            sha256_file(
                aggregate_path
            ),

        "runs_detail":
            results,
    }

    (
        OUT
        /
        "R9I_E1B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nSEED-LEVEL TESTREAL METRICS:"
    )

    cols = [
        "task",
        "model",
        "seed",
        "ADE_iso_window_mean",
        "FDE_iso_window_mean",
        "ADE_px_window_mean",
        "FDE_px_window_mean",
        "ADE_px_track_macro",
        "FDE_px_track_macro",
        "best_epoch",
    ]

    print(
        result_df[
            [
                c
                for c in cols
                if c in result_df.columns
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n3-SEED MODEL SUMMARY:"
    )

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
            [
                c
                for c in summary_cols
                if c in aggregate_df.columns
            ]
        ].to_string(
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
        "\nR9-I-E1B COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
