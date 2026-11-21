
from pathlib import Path
import argparse
import ast
import contextlib
import hashlib
import io
import json
import sys
import traceback
from datetime import datetime, timezone


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

TRANSFORMER_SRC = MAIN1 / "train_transformer.py"
RECURRENT_SRC = MAIN1 / "train_lstm_gru.py"

R9I_B_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_B_exact_model_hyperparameter_and_split_lock"
    / "R9I_B_MANIFEST.json"
)

R9I_C_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_C_V2_matched_isotropic_trainreal_dataset"
    / "R9I_C_V2_MANIFEST.json"
)

R9I_D0_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_D0_training_runtime_dependency_audit"
    / "R9I_D0_MANIFEST.json"
)

ISO_DATASET_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_C_V2_matched_isotropic_trainreal_dataset"
    / "dataset"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_D1_matched_isotropic_training_only"
)

TRANSFORMER_OUT = OUT / "transformer"
RECURRENT_OUT = OUT / "lstm_gru"
LOG_DIR = OUT / "logs"
STATE_DIR = OUT / "state"

TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]

SEEDS = [
    42,
    52,
    62,
]

RECURRENT_MODELS = [
    "LSTM",
    "GRU",
]

EXPECTED_TRANSFORMER = {
    "BATCH_SIZE": 64,
    "D_MODEL": 64,
    "NHEAD": 4,
    "NUM_LAYERS": 2,
    "DIM_FEEDFORWARD": 128,
    "DROPOUT": 0.10,
    "LEARNING_RATE": 5e-4,
    "WEIGHT_DECAY": 1e-4,
    "MAX_EPOCHS": 300,
    "PATIENCE": 30,
    "SEEDS": [42, 52, 62],
}

EXPECTED_RECURRENT = {
    "BATCH_SIZE": 64,
    "HIDDEN_SIZE": 64,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.10,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "MAX_EPOCHS": 300,
    "PATIENCE": 30,
    "SEEDS": [42, 52, 62],
    "MODELS": ["LSTM", "GRU"],
}


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


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


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


def verify_protocol_locks():
    for p in [
        TRANSFORMER_SRC,
        RECURRENT_SRC,
        R9I_B_MANIFEST,
        R9I_C_MANIFEST,
        R9I_D0_MANIFEST,
        ISO_DATASET_DIR,
    ]:
        require(p)

    b = json.loads(
        R9I_B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    c = json.loads(
        R9I_C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    d0 = json.loads(
        R9I_D0_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        b.get("status")
        !=
        "R9I_B_EXACT_MODEL_HYPERPARAMETER_SPLIT_AND_CHECKPOINT_LOCK_COMPLETE_READY_FOR_R9I_C_ISOTROPIC_TRAINREAL_DATASET_BUILD"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-B status: {b.get('status')}"
        )

    if (
        c.get("status")
        !=
        "R9I_C_V2_MATCHED_ISOTROPIC_TRAINREAL_CANONICAL_DATASETS_BUILT_READY_FOR_R9I_D_MATCHED_NEURAL_RETRAINING_SCRIPT"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-C status: {c.get('status')}"
        )

    if (
        d0.get("status")
        !=
        "R9I_D0_TRAINING_RUNTIME_DEPENDENCIES_AUDITED_READY_FOR_R9I_D1_SAFE_TRAINING_ONLY_MATCHED_RUNNER"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-D0 status: {d0.get('status')}"
        )

    locked_t_hash = (
        b[
            "original_training_sources"
        ][
            "Transformer"
        ][
            "sha256"
        ]
    )

    locked_r_hash = (
        b[
            "original_training_sources"
        ][
            "GRU_LSTM"
        ][
            "sha256"
        ]
    )

    current_t_hash = sha256_file(
        TRANSFORMER_SRC
    )

    current_r_hash = sha256_file(
        RECURRENT_SRC
    )

    if current_t_hash != locked_t_hash:
        raise RuntimeError(
            "train_transformer.py hash changed after protocol lock."
        )

    if current_r_hash != locked_r_hash:
        raise RuntimeError(
            "train_lstm_gru.py hash changed after protocol lock."
        )

    dep_flags = d0.get(
        "evaluation_dependency_flags",
        {}
    )

    if dep_flags.get(
        "birdsai_test_canonical_detected"
    ):
        raise RuntimeError(
            "R9-I-D0 reported BIRDSAI TestReal dependency inside audited target functions."
        )

    if dep_flags.get(
        "wildbox_external_canonical_detected"
    ):
        raise RuntimeError(
            "R9-I-D0 reported WildBox dependency inside audited target functions."
        )

    return {
        "r9i_b":
            b,

        "r9i_c":
            c,

        "r9i_d0":
            d0,

        "transformer_source_sha256":
            current_t_hash,

        "recurrent_source_sha256":
            current_r_hash,
    }


def verify_isotropic_inputs():
    inventory = {}

    for task in TASKS:
        npz_path = (
            ISO_DATASET_DIR
            /
            f"{task}_birdsai_trainval_canonical.npz"
        )

        meta_path = (
            ISO_DATASET_DIR
            /
            f"{task}_birdsai_trainval_metadata.csv"
        )

        require(
            npz_path
        )

        require(
            meta_path
        )

        inventory[
            task
        ] = {
            "npz":
                str(
                    npz_path
                ),

            "npz_sha256":
                sha256_file(
                    npz_path
                ),

            "metadata":
                str(
                    meta_path
                ),

            "metadata_sha256":
                sha256_file(
                    meta_path
                ),
        }

    # Cross-check against R9-I-C manifest hashes.
    c = json.loads(
        R9I_C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    task_records = {
        row[
            "task"
        ]:
            row
        for row in c[
            "tasks"
        ]
    }

    for task in TASKS:
        if (
            inventory[
                task
            ][
                "npz_sha256"
            ]
            !=
            task_records[
                task
            ][
                "output_npz_sha256"
            ]
        ):
            raise RuntimeError(
                f"{task}: isotropic NPZ hash mismatch against R9-I-C."
            )

        if (
            inventory[
                task
            ][
                "metadata_sha256"
            ]
            !=
            task_records[
                task
            ][
                "output_metadata_sha256"
            ]
        ):
            raise RuntimeError(
                f"{task}: isotropic metadata hash mismatch against R9-I-C."
            )

    return inventory


def make_definition_only_module(
    source_path,
    module_name,
):
    """
    Load only safe top-level declarations from the original training script:
      - imports
      - assignments / annotated assignments
      - function definitions
      - class definitions

    Explicitly skip top-level operational statements such as loops, evaluate calls,
    result aggregation, printing, and file writes. This prevents the original
    anisotropic evaluation/reporting pipeline from running.
    """
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

    allowed = (
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    body = []

    for node in tree.body:
        if isinstance(
            node,
            allowed
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

    code = compile(
        clean_tree,
        str(
            source_path
        ),
        "exec"
    )

    exec(
        code,
        namespace
    )

    return namespace


def assert_global(
    ns,
    name,
    expected,
    family,
):
    if name not in ns:
        raise RuntimeError(
            f"{family}: global {name} missing."
        )

    actual = ns[
        name
    ]

    if isinstance(
        expected,
        float
    ):
        if abs(
            float(
                actual
            )
            -
            expected
        ) > 1e-15:
            raise RuntimeError(
                f"{family}: {name} mismatch: expected={expected}, actual={actual}"
            )
    else:
        if actual != expected:
            raise RuntimeError(
                f"{family}: {name} mismatch: expected={expected}, actual={actual}"
            )


def verify_loaded_protocol(
    transformer_ns,
    recurrent_ns,
):
    for name, expected in EXPECTED_TRANSFORMER.items():
        assert_global(
            transformer_ns,
            name,
            expected,
            "Transformer"
        )

    for name, expected in EXPECTED_RECURRENT.items():
        assert_global(
            recurrent_ns,
            name,
            expected,
            "GRU/LSTM"
        )

    if "train_one" not in transformer_ns:
        raise RuntimeError(
            "Transformer train_one not loaded."
        )

    if "train_one" not in recurrent_ns:
        raise RuntimeError(
            "Recurrent train_one not loaded."
        )


def prepare_namespaces():
    transformer_ns = make_definition_only_module(
        TRANSFORMER_SRC,
        "r9i_d1_original_transformer_definitions",
    )

    recurrent_ns = make_definition_only_module(
        RECURRENT_SRC,
        "r9i_d1_original_recurrent_definitions",
    )

    verify_loaded_protocol(
        transformer_ns,
        recurrent_ns,
    )

    # Redirect ONLY data root and output root.
    transformer_ns[
        "ROOT"
    ] = ISO_DATASET_DIR

    transformer_ns[
        "OUT"
    ] = TRANSFORMER_OUT

    recurrent_ns[
        "ROOT"
    ] = ISO_DATASET_DIR

    recurrent_ns[
        "OUT"
    ] = RECURRENT_OUT

    return (
        transformer_ns,
        recurrent_ns,
    )


def checkpoint_path(
    family,
    task,
    seed,
    model_type=None,
):
    if family == "Transformer":
        return (
            TRANSFORMER_OUT
            /
            f"{task}_Transformer_seed{seed}.pt"
        )

    return (
        RECURRENT_OUT
        /
        f"{task}_{model_type}_seed{seed}.pt"
    )


def sidecar_path(
    family,
    task,
    seed,
    model_type=None,
):
    label = (
        family
        if family == "Transformer"
        else model_type
    )

    return (
        STATE_DIR
        /
        f"{task}_{label}_seed{seed}.json"
    )


def log_path(
    family,
    task,
    seed,
    model_type=None,
):
    label = (
        family
        if family == "Transformer"
        else model_type
    )

    return (
        LOG_DIR
        /
        f"{task}_{label}_seed{seed}.log"
    )


def checkpoint_identity(
    path,
):
    import torch

    require(
        path
    )

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        obj,
        dict
    ):
        raise RuntimeError(
            f"Checkpoint is not a dict: {path}"
        )

    return obj


def validate_checkpoint(
    family,
    task,
    seed,
    path,
    model_type=None,
):
    ckpt = checkpoint_identity(
        path
    )

    if str(
        ckpt.get(
            "task"
        )
    ) != task:
        raise RuntimeError(
            f"{path.name}: checkpoint task mismatch."
        )

    if int(
        ckpt.get(
            "seed"
        )
    ) != int(
        seed
    ):
        raise RuntimeError(
            f"{path.name}: checkpoint seed mismatch."
        )

    if "state_dict" not in ckpt:
        raise RuntimeError(
            f"{path.name}: missing state_dict."
        )

    if "best_epoch" not in ckpt:
        raise RuntimeError(
            f"{path.name}: missing best_epoch."
        )

    if (
        "best_validation_track_macro_ADE"
        not in ckpt
    ):
        raise RuntimeError(
            f"{path.name}: missing best_validation_track_macro_ADE."
        )

    if family == "Transformer":
        expected = {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.10,
        }

    else:
        if str(
            ckpt.get(
                "model_type"
            )
        ) != model_type:
            raise RuntimeError(
                f"{path.name}: model_type mismatch."
            )

        expected = {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.10,
        }

    for key, value in expected.items():
        if key not in ckpt:
            raise RuntimeError(
                f"{path.name}: missing architecture key {key}."
            )

        actual = ckpt[
            key
        ]

        if isinstance(
            value,
            float
        ):
            if abs(
                float(
                    actual
                )
                -
                value
            ) > 1e-15:
                raise RuntimeError(
                    f"{path.name}: {key} mismatch."
                )

        elif actual != value:
            raise RuntimeError(
                f"{path.name}: {key} mismatch."
            )

    return {
        "best_epoch":
            int(
                ckpt[
                    "best_epoch"
                ]
            ),

        "best_validation_track_macro_ADE":
            float(
                ckpt[
                    "best_validation_track_macro_ADE"
                ]
            ),

        "checkpoint_sha256":
            sha256_file(
                path
            ),
    }


def provenance_payload(
    locks,
    inputs,
):
    return {
        "transformer_source":
            str(
                TRANSFORMER_SRC
            ),

        "transformer_source_sha256":
            locks[
                "transformer_source_sha256"
            ],

        "recurrent_source":
            str(
                RECURRENT_SRC
            ),

        "recurrent_source_sha256":
            locks[
                "recurrent_source_sha256"
            ],

        "isotropic_inputs":
            inputs,

        "dataset_root":
            str(
                ISO_DATASET_DIR
            ),
    }


def existing_run_is_valid(
    family,
    task,
    seed,
    locks,
    inputs,
    model_type=None,
):
    ckpt = checkpoint_path(
        family,
        task,
        seed,
        model_type,
    )

    sidecar = sidecar_path(
        family,
        task,
        seed,
        model_type,
    )

    if not ckpt.exists():
        return (
            False,
            None,
        )

    if not sidecar.exists():
        return (
            False,
            None,
        )

    try:
        state = json.loads(
            sidecar.read_text(
                encoding="utf-8"
            )
        )

        if state.get(
            "status"
        ) != "COMPLETE":
            return (
                False,
                None,
            )

        current_prov = provenance_payload(
            locks,
            inputs,
        )

        if state.get(
            "provenance"
        ) != current_prov:
            return (
                False,
                None,
            )

        validated = validate_checkpoint(
            family,
            task,
            seed,
            ckpt,
            model_type,
        )

        if (
            state.get(
                "checkpoint_sha256"
            )
            !=
            validated[
                "checkpoint_sha256"
            ]
        ):
            return (
                False,
                None,
            )

        return (
            True,
            validated,
        )

    except Exception:
        return (
            False,
            None,
        )


def write_run_state(
    family,
    task,
    seed,
    status,
    locks,
    inputs,
    model_type=None,
    extra=None,
):
    p = sidecar_path(
        family,
        task,
        seed,
        model_type,
    )

    label = (
        family
        if family == "Transformer"
        else model_type
    )

    payload = {
        "status":
            status,

        "updated_utc":
            utc_now(),

        "task":
            task,

        "model":
            label,

        "seed":
            int(
                seed
            ),

        "checkpoint":
            str(
                checkpoint_path(
                    family,
                    task,
                    seed,
                    model_type,
                )
            ),

        "log":
            str(
                log_path(
                    family,
                    task,
                    seed,
                    model_type,
                )
            ),

        "provenance":
            provenance_payload(
                locks,
                inputs,
            ),
    }

    if extra:
        payload.update(
            extra
        )

    p.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )


def run_one(
    family,
    task,
    seed,
    ns,
    locks,
    inputs,
    model_type=None,
):
    label = (
        family
        if family == "Transformer"
        else model_type
    )

    ckpt = checkpoint_path(
        family,
        task,
        seed,
        model_type,
    )

    log = log_path(
        family,
        task,
        seed,
        model_type,
    )

    valid_existing, existing_meta = (
        existing_run_is_valid(
            family,
            task,
            seed,
            locks,
            inputs,
            model_type,
        )
    )

    if valid_existing:
        print(
            f"[SKIP VALID COMPLETE] task={task} model={label} seed={seed} "
            f"best_epoch={existing_meta['best_epoch']} "
            f"best_val_track_macro_ADE={existing_meta['best_validation_track_macro_ADE']:.8f}"
        )

        return {
            "task":
                task,

            "model":
                label,

            "seed":
                int(
                    seed
                ),

            "status":
                "SKIPPED_VALID_COMPLETE",

            **existing_meta,
        }

    if ckpt.exists():
        quarantine = (
            ckpt.with_suffix(
                ckpt.suffix
                +
                ".pre_rerun"
            )
        )

        if quarantine.exists():
            quarantine.unlink()

        ckpt.rename(
            quarantine
        )

        print(
            f"[QUARANTINED UNVERIFIED EXISTING CHECKPOINT] {quarantine}"
        )

    write_run_state(
        family,
        task,
        seed,
        "RUNNING",
        locks,
        inputs,
        model_type,
    )

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        f"TRAIN START | task={task} | model={label} | seed={seed}"
    )

    print(
        f"DATA ROOT: {ISO_DATASET_DIR}"
    )

    print(
        f"OUTPUT ROOT: {ns['OUT']}"
    )

    print(
        "=" * 100
    )

    try:
        with open(
            log,
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
                if family == "Transformer":
                    ns[
                        "train_one"
                    ](
                        task,
                        seed,
                    )

                else:
                    ns[
                        "train_one"
                    ](
                        task,
                        model_type,
                        seed,
                    )

        validated = validate_checkpoint(
            family,
            task,
            seed,
            ckpt,
            model_type,
        )

        write_run_state(
            family,
            task,
            seed,
            "COMPLETE",
            locks,
            inputs,
            model_type,
            extra=validated,
        )

        print(
            f"[COMPLETE] task={task} model={label} seed={seed} "
            f"best_epoch={validated['best_epoch']} "
            f"best_val_track_macro_ADE={validated['best_validation_track_macro_ADE']:.8f}"
        )

        return {
            "task":
                task,

            "model":
                label,

            "seed":
                int(
                    seed
                ),

            "status":
                "COMPLETE",

            **validated,
        }

    except Exception as exc:
        error_text = traceback.format_exc()

        write_run_state(
            family,
            task,
            seed,
            "FAILED",
            locks,
            inputs,
            model_type,
            extra={
                "error":
                    str(
                        exc
                    ),

                "traceback":
                    error_text,
            },
        )

        print(
            error_text
        )

        raise


def selected_runs(
    family_filter,
    task_filter,
):
    runs = []

    tasks = (
        TASKS
        if task_filter == "all"
        else [
            task_filter
        ]
    )

    if family_filter in [
        "all",
        "transformer",
    ]:
        for task in tasks:
            for seed in SEEDS:
                runs.append(
                    (
                        "Transformer",
                        task,
                        seed,
                        None,
                    )
                )

    if family_filter in [
        "all",
        "recurrent",
        "lstm",
    ]:
        for task in tasks:
            for seed in SEEDS:
                runs.append(
                    (
                        "Recurrent",
                        task,
                        seed,
                        "LSTM",
                    )
                )

    if family_filter in [
        "all",
        "recurrent",
        "gru",
    ]:
        for task in tasks:
            for seed in SEEDS:
                runs.append(
                    (
                        "Recurrent",
                        task,
                        seed,
                        "GRU",
                    )
                )

    return runs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "R9-I-D1 safe matched isotropic training-only runner."
        )
    )

    parser.add_argument(
        "--family",
        choices=[
            "all",
            "transformer",
            "recurrent",
            "lstm",
            "gru",
        ],
        default="all",
    )

    parser.add_argument(
        "--task",
        choices=[
            "all",
            "2s_to_2s",
            "2s_to_5s",
        ],
        default="all",
    )

    args = parser.parse_args()

    print(
        "\nR9-I-D1 — SAFE MATCHED ISOTROPIC NEURAL TRAINING-ONLY RUNNER"
    )

    print(
        "This runner uses the ORIGINAL locked model/training definitions but does NOT execute "
        "their top-level evaluation/reporting pipeline. ROOT is redirected to the matched "
        "isotropic TrainReal dataset and OUT is redirected to a new Downloads revision folder. "
        "No BIRDSAI TestReal/WildBox evaluation is performed in this step. Original E:\\ results "
        "and frozen checkpoints are never overwritten.\n"
    )

    for p in [
        OUT,
        TRANSFORMER_OUT,
        RECURRENT_OUT,
        LOG_DIR,
        STATE_DIR,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True
        )

    locks = verify_protocol_locks()
    inputs = verify_isotropic_inputs()

    (
        transformer_ns,
        recurrent_ns,
    ) = prepare_namespaces()

    runs = selected_runs(
        args.family,
        args.task,
    )

    if not runs:
        raise RuntimeError(
            "No runs selected."
        )

    print(
        "RUN PLAN:"
    )

    for family, task, seed, model_type in runs:
        label = (
            family
            if family == "Transformer"
            else model_type
        )

        print(
            f"  {task} | {label} | seed={seed}"
        )

    print(
        f"\nTotal selected runs: {len(runs)}"
    )

    results = []

    try:
        for family, task, seed, model_type in runs:
            ns = (
                transformer_ns
                if family == "Transformer"
                else recurrent_ns
            )

            result = run_one(
                family,
                task,
                seed,
                ns,
                locks,
                inputs,
                model_type,
            )

            results.append(
                result
            )

    except Exception:
        progress = {
            "step":
                "R9I_D1_SAFE_MATCHED_ISOTROPIC_TRAINING_ONLY",

            "updated_utc":
                utc_now(),

            "status":
                "R9I_D1_PARTIAL_OR_FAILED_SAFE_TO_RESUME",

            "selected_family":
                args.family,

            "selected_task":
                args.task,

            "completed_this_invocation":
                results,

            "new_training":
                True,

            "testreal_evaluation":
                False,

            "wildbox_evaluation":
                False,

            "original_frozen_results_modified":
                False,
        }

        (
            OUT
            /
            "R9I_D1_PROGRESS.json"
        ).write_text(
            json.dumps(
                progress,
                indent=2
            ),
            encoding="utf-8"
        )

        raise

    import pandas as pd

    results_df = pd.DataFrame(
        results
    )

    result_csv = (
        OUT
        /
        "R9I_D1_TRAINING_RUN_SUMMARY.csv"
    )

    results_df.to_csv(
        result_csv,
        index=False
    )

    # Determine whether the COMPLETE 18-run matched matrix exists,
    # even if this invocation selected only a subset.
    complete_rows = []

    for task in TASKS:
        for seed in SEEDS:
            for family, model_type in [
                (
                    "Transformer",
                    None
                ),
                (
                    "Recurrent",
                    "LSTM"
                ),
                (
                    "Recurrent",
                    "GRU"
                ),
            ]:
                label = (
                    family
                    if family == "Transformer"
                    else model_type
                )

                valid, meta = existing_run_is_valid(
                    family,
                    task,
                    seed,
                    locks,
                    inputs,
                    model_type,
                )

                complete_rows.append(
                    {
                        "task":
                            task,

                        "model":
                            label,

                        "seed":
                            seed,

                        "complete":
                            bool(
                                valid
                            ),

                        "best_epoch":
                            (
                                meta[
                                    "best_epoch"
                                ]
                                if valid
                                else None
                            ),

                        "best_validation_track_macro_ADE":
                            (
                                meta[
                                    "best_validation_track_macro_ADE"
                                ]
                                if valid
                                else None
                            ),

                        "checkpoint_sha256":
                            (
                                meta[
                                    "checkpoint_sha256"
                                ]
                                if valid
                                else None
                            ),
                    }
                )

    matrix_df = pd.DataFrame(
        complete_rows
    )

    matrix_csv = (
        OUT
        /
        "R9I_D1_COMPLETE_18_RUN_MATRIX.csv"
    )

    matrix_df.to_csv(
        matrix_csv,
        index=False
    )

    all_complete = bool(
        matrix_df[
            "complete"
        ].all()
    )

    if all_complete:
        status = (
            "R9I_D1_ALL_18_MATCHED_ISOTROPIC_NEURAL_RUNS_TRAINED_READY_FOR_R9I_E_MATCHED_TESTREAL_ISOTROPIC_EVALUATION_BUILD"
        )
    else:
        status = (
            "R9I_D1_SELECTED_TRAINING_RUNS_COMPLETE_MORE_MATCHED_RUNS_REMAIN"
        )

    manifest = {
        "step":
            "R9I_D1_SAFE_MATCHED_ISOTROPIC_TRAINING_ONLY",

        "created_utc":
            utc_now(),

        "status":
            status,

        "selected_family":
            args.family,

        "selected_task":
            args.task,

        "definition_loading_method":
            (
                "AST definition-only execution: imports, assignments, functions and classes "
                "from locked original training sources; top-level operational/evaluation "
                "statements are not executed."
            ),

        "dataset_root":
            str(
                ISO_DATASET_DIR
            ),

        "transformer_output":
            str(
                TRANSFORMER_OUT
            ),

        "recurrent_output":
            str(
                RECURRENT_OUT
            ),

        "training_source_hashes":
            {
                "transformer":
                    locks[
                        "transformer_source_sha256"
                    ],

                "recurrent":
                    locks[
                        "recurrent_source_sha256"
                    ],
            },

        "input_hashes":
            inputs,

        "hyperparameters_changed":
            False,

        "seeds_changed":
            False,

        "split_membership_changed":
            False,

        "train_sample_weights_changed":
            False,

        "coordinate_representation_changed":
            True,

        "scale_floor_changed_by_matched_original_rule":
            True,

        "testreal_evaluation":
            False,

        "wildbox_evaluation":
            False,

        "original_frozen_results_modified":
            False,

        "all_18_runs_complete":
            all_complete,

        "selected_run_summary_csv":
            str(
                result_csv
            ),

        "complete_matrix_csv":
            str(
                matrix_csv
            ),

        "runs":
            matrix_df.to_dict(
                orient="records"
            ),
    }

    (
        OUT
        /
        "R9I_D1_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nSELECTED RUN SUMMARY:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nCOMPLETE 18-RUN MATRIX:"
    )

    print(
        matrix_df.to_string(
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
        "\nR9-I-D1 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
