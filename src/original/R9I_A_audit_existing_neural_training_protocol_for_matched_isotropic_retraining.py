
from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

C2_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C2_clustered_ci_and_representation_decision"
)

C2_MANIFEST = C2_DIR / "R9H_C2_MANIFEST.json"

FINAL_NEURAL_DIR = MAIN1 / "final_neural_trajectory_dataset"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_A_matched_neural_retraining_protocol_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]

MODEL_FAMILIES = [
    "GRU",
    "LSTM",
    "Transformer",
]

SEED_HINTS = [
    42,
    52,
    62,
]

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".txt",
    ".md",
}

# Strong file-level evidence for an actual neural training/evaluation source.
STRONG_PATTERNS = [
    "birdsai_trainval_canonical",
    "birdsai_test_canonical",
    "wildbox_external_canonical",
    "torch.optim",
    "optimizer",
    "early_stopping",
    "early stopping",
    "state_dict",
    "save_checkpoint",
    "best.pt",
    "model.eval",
    "gru",
    "lstm",
    "transformer",
]

TRAINING_PATTERNS = [
    "DataLoader",
    "TensorDataset",
    "batch_size",
    "learning_rate",
    "lr=",
    "epochs",
    "patience",
    "weight_decay",
    "dropout",
    "hidden_size",
    "num_layers",
    "nhead",
    "d_model",
    "dim_feedforward",
    "Adam",
    "AdamW",
    "MSELoss",
    "SmoothL1Loss",
    "scheduler",
    "manual_seed",
    "seed",
]


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


def verify_c2():
    require(C2_MANIFEST)

    m = json.loads(
        C2_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R9H_C2_CLUSTERED_CI_CONFIRMS_MATERIAL_GEOMETRY_CHANGE_READY_FOR_R9I_MATCHED_ISOTROPIC_NEURAL_RETRAINING_PROTOCOL"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R9-H-C2 status: {m.get('status')}"
        )

    return m


def inspect_split_metadata():
    rows = []
    split_rows = []

    for task in TASKS:
        p = (
            FINAL_NEURAL_DIR
            / f"{task}_birdsai_trainval_metadata.csv"
        )

        require(p)

        df = pd.read_csv(
            p,
            dtype={
                "sequence": str,
                "track_key": str,
                "segment_id": str,
            }
        )

        if "neural_split" not in df.columns:
            raise RuntimeError(
                f"{p} missing neural_split column."
            )

        split_counts = (
            df[
                "neural_split"
            ]
            .astype(str)
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        track_col = None
        for c in [
            "track_key",
            "segment_id",
        ]:
            if c in df.columns:
                track_col = c
                break

        sequence_col = (
            "source_group"
            if "source_group" in df.columns
            else (
                "sequence"
                if "sequence" in df.columns
                else None
            )
        )

        overlap = None

        if track_col is not None:
            train_tracks = set(
                df.loc[
                    df[
                        "neural_split"
                    ].astype(str)
                    ==
                    "train",
                    track_col,
                ].astype(str)
            )

            val_tracks = set(
                df.loc[
                    df[
                        "neural_split"
                    ].astype(str)
                    ==
                    "validation",
                    track_col,
                ].astype(str)
            )

            overlap = len(
                train_tracks
                &
                val_tracks
            )

        rows.append(
            {
                "task":
                    task,

                "path":
                    str(p),

                "sha256":
                    sha256_file(p),

                "rows":
                    int(
                        len(df)
                    ),

                "split_counts_json":
                    json.dumps(
                        split_counts,
                        sort_keys=True
                    ),

                "track_identity_column":
                    track_col or "",

                "train_val_track_overlap":
                    overlap
                    if overlap is not None
                    else -1,

                "sequence_identity_column":
                    sequence_col or "",
            }
        )

        id_cols = [
            c
            for c in [
                "sample_id",
                "sequence",
                "source_group",
                "track_key",
                "segment_id",
                "neural_split",
            ]
            if c in df.columns
        ]

        temp = df[
            id_cols
        ].copy()

        temp.insert(
            0,
            "task",
            task
        )

        split_rows.append(
            temp
        )

    return (
        pd.DataFrame(
            rows
        ),
        pd.concat(
            split_rows,
            ignore_index=True
        ),
    )


def score_source(path):
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        return None

    low = text.lower()

    strong_hits = {
        term:
            low.count(
                term.lower()
            )
        for term in STRONG_PATTERNS
        if low.count(
            term.lower()
        )
        >
        0
    }

    model_hits = {
        model:
            low.count(
                model.lower()
            )
        for model in MODEL_FAMILIES
        if low.count(
            model.lower()
        )
        >
        0
    }

    task_hits = {
        task:
            low.count(
                task.lower()
            )
        for task in TASKS
        if low.count(
            task.lower()
        )
        >
        0
    }

    if not model_hits:
        return None

    training_hits = {
        term:
            low.count(
                term.lower()
            )
        for term in TRAINING_PATTERNS
        if low.count(
            term.lower()
        )
        >
        0
    }

    # Require actual training/evaluation evidence.
    if (
        len(training_hits) == 0
        and
        not any(
            x in low
            for x in [
                "model.eval",
                "state_dict",
                "checkpoint",
            ]
        )
    ):
        return None

    score = (
        10 * sum(
            model_hits.values()
        )
        +
        5 * sum(
            task_hits.values()
        )
        +
        3 * sum(
            training_hits.values()
        )
        +
        sum(
            strong_hits.values()
        )
    )

    return {
        "text":
            text,

        "score":
            int(score),

        "model_hits":
            model_hits,

        "task_hits":
            task_hits,

        "training_hits":
            training_hits,

        "strong_hits":
            strong_hits,
    }


def scan_training_sources():
    rows = []

    for root in [
        MAIN1,
        MAIN2,
    ]:
        if not root.exists():
            continue

        for p in root.rglob(
            "*.py"
        ):
            if not p.is_file():
                continue

            result = score_source(
                p
            )

            if result is None:
                continue

            rows.append(
                {
                    "filename":
                        p.name,

                    "path":
                        str(p),

                    "root":
                        root.name,

                    "score":
                        result[
                            "score"
                        ],

                    "model_hits_json":
                        json.dumps(
                            result[
                                "model_hits"
                            ],
                            sort_keys=True
                        ),

                    "task_hits_json":
                        json.dumps(
                            result[
                                "task_hits"
                            ],
                            sort_keys=True
                        ),

                    "training_hits_json":
                        json.dumps(
                            result[
                                "training_hits"
                            ],
                            sort_keys=True
                        ),

                    "strong_hits_json":
                        json.dumps(
                            result[
                                "strong_hits"
                            ],
                            sort_keys=True
                        ),

                    "sha256":
                        sha256_file(
                            p
                        ),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "score",
            "filename",
            "path",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort"
    )


def extract_relevant_lines(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    categories = {
        "dataset_input": [
            r"birdsai_trainval_canonical",
            r"birdsai_test_canonical",
            r"wildbox_external_canonical",
            r"np\.load",
        ],

        "seed": [
            r"manual_seed",
            r"\bseed\b",
            r"random\.seed",
            r"np\.random",
        ],

        "optimizer": [
            r"torch\.optim",
            r"AdamW",
            r"\bAdam\b",
            r"weight_decay",
            r"learning_rate",
            r"\blr\b",
        ],

        "training_schedule": [
            r"epochs",
            r"patience",
            r"early[_ ]?stopp",
            r"scheduler",
            r"batch_size",
        ],

        "architecture": [
            r"\bGRU\b",
            r"\bLSTM\b",
            r"Transformer",
            r"hidden_size",
            r"num_layers",
            r"d_model",
            r"nhead",
            r"dim_feedforward",
            r"dropout",
        ],

        "loss": [
            r"MSELoss",
            r"SmoothL1Loss",
            r"\bloss\b",
        ],

        "checkpoint": [
            r"state_dict",
            r"checkpoint",
            r"best",
            r"\.pt",
        ],
    }

    rows = []

    for i, line in enumerate(
        lines,
        start=1
    ):
        for category, pats in categories.items():
            if any(
                re.search(
                    pat,
                    line,
                    flags=re.IGNORECASE
                )
                for pat in pats
            ):
                rows.append(
                    {
                        "file":
                            path.name,

                        "path":
                            str(path),

                        "line_number":
                            i,

                        "category":
                            category,

                        "line":
                            line,
                    }
                )

    return pd.DataFrame(
        rows
    )


def extract_top_level_assignments(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text,
        filename=str(path)
    )

    lines = text.splitlines()

    interesting_names = {
        "SEED",
        "SEEDS",
        "BATCH_SIZE",
        "EPOCHS",
        "MAX_EPOCHS",
        "PATIENCE",
        "LR",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "HIDDEN_SIZE",
        "NUM_LAYERS",
        "DROPOUT",
        "D_MODEL",
        "NHEAD",
        "NUM_HEADS",
        "DIM_FEEDFORWARD",
        "FF_DIM",
        "TASKS",
        "MODELS",
        "DEVICE",
    }

    rows = []

    for node in tree.body:
        target = None

        if (
            isinstance(
                node,
                ast.Assign
            )
            and
            len(
                node.targets
            )
            ==
            1
            and
            isinstance(
                node.targets[
                    0
                ],
                ast.Name
            )
        ):
            target = node.targets[
                0
            ].id

        elif (
            isinstance(
                node,
                ast.AnnAssign
            )
            and
            isinstance(
                node.target,
                ast.Name
            )
        ):
            target = node.target.id

        if target is None:
            continue

        if (
            target.upper()
            not in
            interesting_names
            and
            not any(
                token
                in
                target.upper()
                for token in [
                    "SEED",
                    "BATCH",
                    "EPOCH",
                    "PATIENCE",
                    "LR",
                    "LEARNING",
                    "WEIGHT_DECAY",
                    "HIDDEN",
                    "LAYER",
                    "DROPOUT",
                    "D_MODEL",
                    "NHEAD",
                    "HEAD",
                    "FEEDFORWARD",
                ]
            )
        ):
            continue

        start = int(
            node.lineno
        )

        end = int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        )

        rows.append(
            {
                "file":
                    path.name,

                "path":
                    str(path),

                "name":
                    target,

                "start_line":
                    start,

                "end_line":
                    end,

                "source":
                    "\n".join(
                        f"{i}: {lines[i-1]}"
                        for i in range(
                            start,
                            end + 1
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def find_existing_checkpoints():
    rows = []

    for root in [
        MAIN1,
        MAIN2,
    ]:
        if not root.exists():
            continue

        for suffix in [
            "*.pt",
            "*.pth",
        ]:
            for p in root.rglob(
                suffix
            ):
                if not p.is_file():
                    continue

                low = str(
                    p
                ).lower()

                if not any(
                    model.lower()
                    in
                    low
                    for model in MODEL_FAMILIES
                ):
                    continue

                detected_model = ""

                for model in MODEL_FAMILIES:
                    if model.lower() in low:
                        detected_model = model
                        break

                detected_seed = ""

                for seed in SEED_HINTS:
                    if re.search(
                        rf"(?<!\d){seed}(?!\d)",
                        p.name
                    ):
                        detected_seed = str(
                            seed
                        )
                        break

                detected_task = ""

                for task in TASKS:
                    if task.lower() in low:
                        detected_task = task
                        break

                rows.append(
                    {
                        "model":
                            detected_model,

                        "task":
                            detected_task,

                        "seed":
                            detected_seed,

                        "path":
                            str(p),

                        "filename":
                            p.name,

                        "size_bytes":
                            int(
                                p.stat().st_size
                            ),

                        "sha256":
                            sha256_file(
                                p
                            ),
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "model",
            "task",
            "seed",
            "path",
        ],
        kind="mergesort"
    )


def main():
    print(
        "\nR9-I-A — MATCHED ISOTROPIC NEURAL RETRAINING PROTOCOL LOCK"
    )

    print(
        "DIAGNOSTIC / PROTOCOL-LOCK ONLY. R9-H-C2 established that matched isotropic "
        "neural retraining is warranted. This step does NOT train anything. It locks "
        "the existing train/validation identity, discovers the actual GRU/LSTM/"
        "Transformer training sources, extracts architecture/training hyperparameters, "
        "and inventories frozen checkpoints so the isotropic ablation can be matched "
        "exactly except for representation and the re-derived isotropic TrainReal scale floor.\n"
    )

    verify_c2()

    require(
        FINAL_NEURAL_DIR
    )

    split_summary, split_lock = (
        inspect_split_metadata()
    )

    source_inventory = (
        scan_training_sources()
    )

    if len(
        source_inventory
    ) == 0:
        raise RuntimeError(
            "No neural training source candidates found."
        )

    # Audit strongest sources, but retain full inventory.
    top_sources = source_inventory.head(
        20
    )

    line_frames = []
    assignment_frames = []

    for path_str in top_sources[
        "path"
    ]:
        p = Path(
            path_str
        )

        ldf = extract_relevant_lines(
            p
        )

        adf = extract_top_level_assignments(
            p
        )

        if len(
            ldf
        ):
            line_frames.append(
                ldf
            )

        if len(
            adf
        ):
            assignment_frames.append(
                adf
            )

    relevant_lines = (
        pd.concat(
            line_frames,
            ignore_index=True
        )
        if line_frames
        else
        pd.DataFrame()
    )

    assignments = (
        pd.concat(
            assignment_frames,
            ignore_index=True
        )
        if assignment_frames
        else
        pd.DataFrame()
    )

    checkpoints = (
        find_existing_checkpoints()
    )

    split_summary_path = (
        OUT
        / "R9I_A_SPLIT_SUMMARY.csv"
    )

    split_lock_path = (
        OUT
        / "R9I_A_EXACT_TRAIN_VAL_SPLIT_LOCK.csv"
    )

    source_inventory_path = (
        OUT
        / "R9I_A_NEURAL_TRAINING_SOURCE_INVENTORY.csv"
    )

    relevant_lines_path = (
        OUT
        / "R9I_A_TOP_SOURCE_RELEVANT_LINES.csv"
    )

    assignments_path = (
        OUT
        / "R9I_A_TOP_SOURCE_HYPERPARAMETER_ASSIGNMENTS.csv"
    )

    checkpoint_path = (
        OUT
        / "R9I_A_EXISTING_CHECKPOINT_INVENTORY.csv"
    )

    split_summary.to_csv(
        split_summary_path,
        index=False
    )

    split_lock.to_csv(
        split_lock_path,
        index=False
    )

    source_inventory.to_csv(
        source_inventory_path,
        index=False
    )

    relevant_lines.to_csv(
        relevant_lines_path,
        index=False
    )

    assignments.to_csv(
        assignments_path,
        index=False
    )

    checkpoints.to_csv(
        checkpoint_path,
        index=False
    )

    status = (
        "R9I_A_EXISTING_NEURAL_TRAINING_PROTOCOL_AUDITED_READY_FOR_R9I_B_EXACT_MODEL_HYPERPARAMETER_AND_SPLIT_LOCK"
    )

    manifest = {
        "step":
            "R9I_A_MATCHED_ISOTROPIC_NEURAL_RETRAINING_PROTOCOL_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "model_inference":
            False,

        "frozen_results_modified":
            False,

        "matched_ablation_principle":
            (
                "Keep exact train/validation/test identity, architecture, optimizer, "
                "loss, batch size, epochs/early-stopping policy, seeds and evaluation "
                "protocol fixed. Change only coordinate representation to isotropic "
                "and re-derive the canonical scale floor from isotropic TrainReal."
            ),

        "prohibited_shortcuts":
            [
                "Do not evaluate old anisotropic neural checkpoints on isotropic inputs.",
                "Do not tune hyperparameters using TestReal or WildBox.",
                "Do not change train/validation track membership.",
                "Do not reuse the old anisotropic scale floor for final isotropic training.",
            ],

        "split_summary_csv":
            str(
                split_summary_path
            ),

        "split_lock_csv":
            str(
                split_lock_path
            ),

        "training_source_inventory_csv":
            str(
                source_inventory_path
            ),

        "relevant_lines_csv":
            str(
                relevant_lines_path
            ),

        "hyperparameter_assignments_csv":
            str(
                assignments_path
            ),

        "checkpoint_inventory_csv":
            str(
                checkpoint_path
            ),

        "training_source_candidate_count":
            int(
                len(
                    source_inventory
                )
            ),

        "checkpoint_count":
            int(
                len(
                    checkpoints
                )
            ),
    }

    (
        OUT
        / "R9I_A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "EXACT TRAIN / VALIDATION SPLIT SUMMARY:"
    )

    print(
        split_summary.to_string(
            index=False
        )
    )

    print(
        "\nTOP NEURAL TRAINING SOURCE CANDIDATES:"
    )

    print(
        top_sources[
            [
                "filename",
                "score",
                "model_hits_json",
                "task_hits_json",
                "path",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nTOP-SOURCE HYPERPARAMETER ASSIGNMENTS:"
    )

    if len(
        assignments
    ):
        print(
            assignments.to_string(
                index=False
            )
        )
    else:
        print(
            "No matching top-level hyperparameter assignments found."
        )

    print(
        "\nTOP-SOURCE RELEVANT TRAINING LINES:"
    )

    if len(
        relevant_lines
    ):
        print(
            relevant_lines.head(
                1000
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No relevant training lines found."
        )

    print(
        "\nEXISTING CHECKPOINT INVENTORY:"
    )

    if len(
        checkpoints
    ):
        print(
            checkpoints.to_string(
                index=False
            )
        )
    else:
        print(
            "No matching GRU/LSTM/Transformer checkpoint files found by filename/path."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-I-A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
