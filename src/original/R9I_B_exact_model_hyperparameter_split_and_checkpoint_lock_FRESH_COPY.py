
from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

TRANSFORMER_SRC = MAIN1 / "train_transformer.py"
RECURRENT_SRC = MAIN1 / "train_lstm_gru.py"

FINAL_NEURAL_DIR = MAIN1 / "final_neural_trajectory_dataset"
TRANSFORMER_DIR = MAIN1 / "transformer_results"
RECURRENT_DIR = MAIN1 / "lstm_gru_results"

R9I_A_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_A_matched_neural_retraining_protocol_lock"
)

R9I_A_MANIFEST = R9I_A_DIR / "R9I_A_MANIFEST.json"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_B_exact_model_hyperparameter_and_split_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]

SEEDS = [
    42,
    52,
    62,
]

MODELS = [
    "Transformer",
    "GRU",
    "LSTM",
]

EXPECTED_SPLITS = {
    "2s_to_2s": {
        "train": 434,
        "validation": 97,
    },
    "2s_to_5s": {
        "train": 294,
        "validation": 88,
    },
}

INTERESTING_ASSIGNMENTS = {
    "DEVICE",
    "TASKS",
    "SEEDS",
    "MODELS",
    "BATCH_SIZE",
    "D_MODEL",
    "NHEAD",
    "NUM_LAYERS",
    "DIM_FEEDFORWARD",
    "DROPOUT",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "MAX_EPOCHS",
    "PATIENCE",
    "HIDDEN_SIZE",
}

FUNCTION_KEYWORDS = [
    "seed",
    "weight",
    "loss",
    "train",
    "valid",
    "evaluate",
    "metric",
    "ade",
    "checkpoint",
    "predict",
]

LINE_PATTERNS = {
    "dataset_and_split": [
        r"birdsai_trainval_canonical",
        r"birdsai_trainval_metadata",
        r"neural_split",
        r"train_idx",
        r"val_idx",
        r"track_key",
    ],
    "seeding": [
        r"manual_seed",
        r"random\.seed",
        r"np\.random",
        r"\bseed\b",
    ],
    "architecture": [
        r"TrajectoryTransformer",
        r"TrajectoryRNN",
        r"nn\.GRU",
        r"nn\.LSTM",
        r"d_model",
        r"nhead",
        r"num_layers",
        r"dim_feedforward",
        r"hidden_size",
        r"dropout",
    ],
    "optimizer_scheduler": [
        r"torch\.optim",
        r"AdamW",
        r"\bAdam\b",
        r"ReduceLROnPlateau",
        r"scheduler",
        r"learning_rate",
        r"weight_decay",
        r"\blr\s*=",
    ],
    "loss_and_weighting": [
        r"weighted_smooth_l1",
        r"SmoothL1",
        r"sample_weight",
        r"sample_weights",
        r"track_weight",
        r"weights",
        r"loss\s*=",
    ],
    "validation_selection": [
        r"track_macro",
        r"validation",
        r"best_metric",
        r"best_val",
        r"best_epoch",
        r"early stopping",
        r"no_improvement",
        r"PATIENCE",
    ],
    "checkpoint": [
        r"state_dict",
        r"torch\.save",
        r"checkpoint",
        r"best_validation",
        r"\.pt",
    ],
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


def verify_r9i_a():
    require(R9I_A_MANIFEST)

    m = json.loads(
        R9I_A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R9I_A_EXISTING_NEURAL_TRAINING_PROTOCOL_AUDITED_READY_FOR_R9I_B_EXACT_MODEL_HYPERPARAMETER_AND_SPLIT_LOCK"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R9-I-A status: {m.get('status')}"
        )

    return m


def load_source(path):
    require(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(path)
    )

    return text, lines, tree


def get_top_level_assignments(path):
    text, lines, tree = load_source(path)

    rows = []

    for node in tree.body:
        target = None

        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id

        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        ):
            target = node.target.id

        if target is None:
            continue

        if target not in INTERESTING_ASSIGNMENTS:
            continue

        start = int(node.lineno)
        end = int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        )

        try:
            literal_value = ast.literal_eval(
                node.value
            )
            literal_json = json.dumps(
                literal_value,
                sort_keys=True
            )
        except Exception:
            literal_json = ""

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

                "literal_json":
                    literal_json,

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


def collect_functions_and_classes(path):
    text, lines, tree = load_source(path)

    rows = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            )
        ):
            name = node.name
            low = name.lower()

            keep = (
                isinstance(node, ast.ClassDef)
                or any(
                    k in low
                    for k in FUNCTION_KEYWORDS
                )
            )

            if not keep:
                continue

            start = int(node.lineno)
            end = int(
                getattr(
                    node,
                    "end_lineno",
                    node.lineno
                )
            )

            source = "\n".join(
                lines[
                    start - 1:
                    end
                ]
            )

            numbered = "\n".join(
                f"{i}: {lines[i-1]}"
                for i in range(
                    start,
                    end + 1
                )
            )

            rows.append(
                {
                    "file":
                        path.name,

                    "path":
                        str(path),

                    "kind":
                        (
                            "class"
                            if isinstance(node, ast.ClassDef)
                            else "function"
                        ),

                    "name":
                        name,

                    "start_line":
                        start,

                    "end_line":
                        end,

                    "source_sha256":
                        hashlib.sha256(
                            source.encode(
                                "utf-8"
                            )
                        ).hexdigest(),

                    "source":
                        numbered,
                }
            )

    return pd.DataFrame(
        rows
    )


def extract_protocol_lines(path):
    text, lines, tree = load_source(path)

    rows = []

    for i, line in enumerate(
        lines,
        start=1
    ):
        for category, patterns in LINE_PATTERNS.items():
            if any(
                re.search(
                    p,
                    line,
                    flags=re.IGNORECASE
                )
                for p in patterns
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


def build_context(path, line_df, radius=5):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    selected = set()

    for n in line_df[
        "line_number"
    ].astype(int):
        for j in range(
            max(1, n - radius),
            min(len(lines), n + radius) + 1
        ):
            selected.add(j)

    return pd.DataFrame(
        [
            {
                "file":
                    path.name,

                "path":
                    str(path),

                "line_number":
                    j,

                "line":
                    lines[j - 1],
            }
            for j in sorted(selected)
        ]
    )


def inspect_split_identity():
    summary_rows = []
    detail_frames = []

    for task in TASKS:
        p = (
            FINAL_NEURAL_DIR
            / f"{task}_birdsai_trainval_metadata.csv"
        )

        require(p)

        df = pd.read_csv(
            p,
            dtype={
                "source_group": str,
                "sequence": str,
                "track_key": str,
                "segment_id": str,
                "neural_split": str,
            }
        )

        if "neural_split" not in df.columns:
            raise RuntimeError(
                f"{p} missing neural_split."
            )

        counts = (
            df["neural_split"]
            .value_counts()
            .to_dict()
        )

        for split, expected_n in EXPECTED_SPLITS[task].items():
            if int(counts.get(split, 0)) != expected_n:
                raise RuntimeError(
                    f"{task} {split} count mismatch: "
                    f"expected {expected_n}, got {counts.get(split, 0)}"
                )

        if "track_key" not in df.columns:
            raise RuntimeError(
                f"{p} missing track_key."
            )

        train_tracks = set(
            df.loc[
                df["neural_split"] == "train",
                "track_key",
            ].astype(str)
        )

        val_tracks = set(
            df.loc[
                df["neural_split"] == "validation",
                "track_key",
            ].astype(str)
        )

        overlap = train_tracks & val_tracks

        if overlap:
            raise RuntimeError(
                f"{task}: train/validation track overlap detected: "
                f"{sorted(overlap)[:10]}"
            )

        summary_rows.append(
            {
                "task":
                    task,

                "metadata":
                    str(p),

                "metadata_sha256":
                    sha256_file(p),

                "rows":
                    int(len(df)),

                "train_rows":
                    int(counts.get("train", 0)),

                "validation_rows":
                    int(counts.get("validation", 0)),

                "train_tracks":
                    int(len(train_tracks)),

                "validation_tracks":
                    int(len(val_tracks)),

                "track_overlap":
                    int(len(overlap)),
            }
        )

        cols = [
            c
            for c in [
                "sample_id",
                "source_group",
                "sequence",
                "track_key",
                "segment_id",
                "neural_split",
            ]
            if c in df.columns
        ]

        temp = df[
            cols
        ].copy()

        temp.insert(
            0,
            "task",
            task
        )

        detail_frames.append(
            temp
        )

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.concat(
            detail_frames,
            ignore_index=True
        ),
    )


def classify_checkpoint_from_filename(filename):
    name = filename.lower()

    model = None

    if "_transformer_seed" in name:
        model = "Transformer"

    elif "_lstm_seed" in name:
        model = "LSTM"

    elif "_gru_seed" in name:
        model = "GRU"

    task = None

    for t in TASKS:
        if name.startswith(
            t.lower()
        ):
            task = t
            break

    seed = None

    m = re.search(
        r"_seed(\d+)\.(?:pt|pth)$",
        name
    )

    if m:
        seed = int(
            m.group(1)
        )

    return (
        model,
        task,
        seed,
    )


def exact_checkpoint_inventory():
    rows = []

    roots = [
        TRANSFORMER_DIR,
        RECURRENT_DIR,
    ]

    for root in roots:
        require(root)

        for p in sorted(
            root.glob("*.pt")
        ):
            model, task, seed = (
                classify_checkpoint_from_filename(
                    p.name
                )
            )

            if (
                model is None
                or task is None
                or seed is None
            ):
                continue

            if model not in MODELS:
                continue

            if task not in TASKS:
                continue

            if seed not in SEEDS:
                continue

            rows.append(
                {
                    "model":
                        model,

                    "task":
                        task,

                    "seed":
                        seed,

                    "path":
                        str(p),

                    "filename":
                        p.name,

                    "size_bytes":
                        int(
                            p.stat().st_size
                        ),

                    "sha256":
                        sha256_file(p),
                }
            )

    df = pd.DataFrame(
        rows
    ).sort_values(
        [
            "model",
            "task",
            "seed",
        ],
        kind="mergesort"
    ).reset_index(
        drop=True
    )

    expected = {
        (
            model,
            task,
            seed,
        )
        for model in MODELS
        for task in TASKS
        for seed in SEEDS
    }

    actual = set(
        zip(
            df["model"],
            df["task"],
            df["seed"],
        )
    )

    missing = expected - actual
    extras = actual - expected

    if missing:
        raise RuntimeError(
            f"Missing expected frozen checkpoints: {sorted(missing)}"
        )

    if extras:
        raise RuntimeError(
            f"Unexpected checkpoint identities: {sorted(extras)}"
        )

    if len(df) != 18:
        raise RuntimeError(
            f"Expected exactly 18 checkpoints, found {len(df)}."
        )

    return df


def extract_literal_map(assignments):
    result = {}

    for _, row in assignments.iterrows():
        if not row["literal_json"]:
            continue

        try:
            result[
                row["name"]
            ] = json.loads(
                row["literal_json"]
            )
        except Exception:
            pass

    return result


def build_protocol_table(
    transformer_assignments,
    recurrent_assignments,
):
    t = extract_literal_map(
        transformer_assignments
    )

    r = extract_literal_map(
        recurrent_assignments
    )

    expected_t = {
        "SEEDS": [42, 52, 62],
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
    }

    expected_r = {
        "SEEDS": [42, 52, 62],
        "MODELS": ["LSTM", "GRU"],
        "BATCH_SIZE": 64,
        "HIDDEN_SIZE": 64,
        "NUM_LAYERS": 2,
        "DROPOUT": 0.10,
        "LEARNING_RATE": 1e-3,
        "WEIGHT_DECAY": 1e-4,
        "MAX_EPOCHS": 300,
        "PATIENCE": 30,
    }

    for name, expected in expected_t.items():
        if name not in t:
            raise RuntimeError(
                f"Transformer source missing literal {name}."
            )

        if t[name] != expected:
            raise RuntimeError(
                f"Transformer {name} mismatch: "
                f"expected {expected}, got {t[name]}"
            )

    for name, expected in expected_r.items():
        if name not in r:
            raise RuntimeError(
                f"Recurrent source missing literal {name}."
            )

        if r[name] != expected:
            raise RuntimeError(
                f"Recurrent {name} mismatch: "
                f"expected {expected}, got {r[name]}"
            )

    rows = [
        {
            "model_family":
                "Transformer",

            "source":
                str(TRANSFORMER_SRC),

            "source_sha256":
                sha256_file(
                    TRANSFORMER_SRC
                ),

            "seeds":
                "42|52|62",

            "batch_size":
                64,

            "architecture":
                "d_model=64;nhead=4;num_layers=2;dim_feedforward=128;dropout=0.10",

            "learning_rate":
                5e-4,

            "weight_decay":
                1e-4,

            "max_epochs":
                300,

            "patience":
                30,
        },
        {
            "model_family":
                "GRU/LSTM",

            "source":
                str(RECURRENT_SRC),

            "source_sha256":
                sha256_file(
                    RECURRENT_SRC
                ),

            "seeds":
                "42|52|62",

            "batch_size":
                64,

            "architecture":
                "hidden_size=64;num_layers=2;dropout=0.10",

            "learning_rate":
                1e-3,

            "weight_decay":
                1e-4,

            "max_epochs":
                300,

            "patience":
                30,
        },
    ]

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-I-B — EXACT MODEL / HYPERPARAMETER / SPLIT LOCK"
    )

    print(
        "PROTOCOL LOCK ONLY. Uses only the original training sources "
        "train_transformer.py and train_lstm_gru.py. Broad bootstrap/evaluation "
        "scripts are excluded. The R9-I-A checkpoint-label bug is corrected by "
        "classifying checkpoints from each filename, not the parent directory name. "
        "NO training, NO model inference, NO frozen overwrite.\n"
    )

    verify_r9i_a()

    for p in [
        TRANSFORMER_SRC,
        RECURRENT_SRC,
        FINAL_NEURAL_DIR,
        TRANSFORMER_DIR,
        RECURRENT_DIR,
    ]:
        require(p)

    split_summary, split_lock = (
        inspect_split_identity()
    )

    t_assign = get_top_level_assignments(
        TRANSFORMER_SRC
    )

    r_assign = get_top_level_assignments(
        RECURRENT_SRC
    )

    assignments = pd.concat(
        [
            t_assign,
            r_assign,
        ],
        ignore_index=True
    )

    protocol_table = build_protocol_table(
        t_assign,
        r_assign
    )

    function_frames = []
    line_frames = []
    context_frames = []

    for p in [
        TRANSFORMER_SRC,
        RECURRENT_SRC,
    ]:
        fdf = collect_functions_and_classes(
            p
        )

        ldf = extract_protocol_lines(
            p
        )

        cdf = build_context(
            p,
            ldf,
            radius=5
        )

        function_frames.append(
            fdf
        )

        line_frames.append(
            ldf
        )

        context_frames.append(
            cdf
        )

    funcs = pd.concat(
        function_frames,
        ignore_index=True
    )

    protocol_lines = pd.concat(
        line_frames,
        ignore_index=True
    )

    context = pd.concat(
        context_frames,
        ignore_index=True
    )

    checkpoints = exact_checkpoint_inventory()

    split_summary_path = (
        OUT
        / "R9I_B_SPLIT_SUMMARY_LOCK.csv"
    )

    split_lock_path = (
        OUT
        / "R9I_B_EXACT_SPLIT_IDENTITY.csv"
    )

    protocol_path = (
        OUT
        / "R9I_B_MODEL_PROTOCOL_TABLE.csv"
    )

    assignments_path = (
        OUT
        / "R9I_B_EXACT_TOP_LEVEL_ASSIGNMENTS.csv"
    )

    funcs_path = (
        OUT
        / "R9I_B_RELEVANT_FUNCTIONS_AND_CLASSES.csv"
    )

    lines_path = (
        OUT
        / "R9I_B_PROTOCOL_LINES.csv"
    )

    context_path = (
        OUT
        / "R9I_B_PROTOCOL_CONTEXT.csv"
    )

    checkpoint_path = (
        OUT
        / "R9I_B_CORRECTED_CHECKPOINT_INVENTORY.csv"
    )

    split_summary.to_csv(
        split_summary_path,
        index=False
    )

    split_lock.to_csv(
        split_lock_path,
        index=False
    )

    protocol_table.to_csv(
        protocol_path,
        index=False
    )

    assignments.to_csv(
        assignments_path,
        index=False
    )

    funcs.to_csv(
        funcs_path,
        index=False
    )

    protocol_lines.to_csv(
        lines_path,
        index=False
    )

    context.to_csv(
        context_path,
        index=False
    )

    checkpoints.to_csv(
        checkpoint_path,
        index=False
    )

    source_bundle = (
        OUT
        / "R9I_B_EXACT_TRAINING_SOURCE_BUNDLE.txt"
    )

    with open(
        source_bundle,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "R9-I-B EXACT TRAINING SOURCE BUNDLE\n\n"
        )

        for src in [
            TRANSFORMER_SRC,
            RECURRENT_SRC,
        ]:
            f.write(
                "=" * 110
                +
                "\n"
            )

            f.write(
                f"SOURCE: {src}\n"
            )

            f.write(
                f"SHA256: {sha256_file(src)}\n"
            )

            f.write(
                "=" * 110
                +
                "\n"
            )

            for i, line in enumerate(
                src.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).splitlines(),
                start=1
            ):
                f.write(
                    f"{i}: {line}\n"
                )

            f.write(
                "\n\n"
            )

    status = (
        "R9I_B_EXACT_MODEL_HYPERPARAMETER_SPLIT_AND_CHECKPOINT_LOCK_COMPLETE_READY_FOR_R9I_C_ISOTROPIC_TRAINREAL_DATASET_BUILD"
    )

    manifest = {
        "step":
            "R9I_B_EXACT_MODEL_HYPERPARAMETER_AND_SPLIT_LOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "original_training_sources": {
            "Transformer": {
                "path":
                    str(TRANSFORMER_SRC),

                "sha256":
                    sha256_file(
                        TRANSFORMER_SRC
                    ),
            },

            "GRU_LSTM": {
                "path":
                    str(RECURRENT_SRC),

                "sha256":
                    sha256_file(
                        RECURRENT_SRC
                    ),
            },
        },

        "checkpoint_inventory_bug_fixed":
            (
                "R9-I-A inferred model type from full path and mislabeled "
                "LSTM checkpoints as GRU because parent directory is lstm_gru_results. "
                "R9-I-B classifies model identity from checkpoint filename."
            ),

        "expected_checkpoint_count":
            18,

        "actual_checkpoint_count":
            int(
                len(
                    checkpoints
                )
            ),

        "matched_retraining_policy": [
            "Keep exact train/validation sample and track membership.",
            "Keep seeds 42, 52, 62.",
            "Keep model architecture fixed per original training source.",
            "Keep batch size, optimizer settings, scheduler, loss, sample weighting, early stopping and validation criterion fixed.",
            "Change only coordinate representation to aspect-preserving isotropic coordinates.",
            "Re-derive scale floor from isotropic BIRDSAI TrainReal only.",
            "Do not use TestReal or WildBox for tuning or scale-floor derivation.",
            "Do not evaluate old anisotropic checkpoints on isotropic tensors as a performance comparison.",
        ],

        "new_training":
            False,

        "model_inference":
            False,

        "frozen_results_modified":
            False,

        "outputs": {
            "split_summary":
                str(
                    split_summary_path
                ),

            "split_identity":
                str(
                    split_lock_path
                ),

            "protocol_table":
                str(
                    protocol_path
                ),

            "assignments":
                str(
                    assignments_path
                ),

            "functions_and_classes":
                str(
                    funcs_path
                ),

            "protocol_lines":
                str(
                    lines_path
                ),

            "protocol_context":
                str(
                    context_path
                ),

            "corrected_checkpoint_inventory":
                str(
                    checkpoint_path
                ),

            "source_bundle":
                str(
                    source_bundle
                ),
        },
    }

    (
        OUT
        / "R9I_B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "EXACT SPLIT LOCK:"
    )

    print(
        split_summary.to_string(
            index=False
        )
    )

    print(
        "\nMODEL / HYPERPARAMETER LOCK:"
    )

    print(
        protocol_table.to_string(
            index=False
        )
    )

    print(
        "\nCORRECTED CHECKPOINT INVENTORY:"
    )

    print(
        checkpoints.to_string(
            index=False
        )
    )

    print(
        "\nRELEVANT TRAINING FUNCTIONS / CLASSES:"
    )

    print(
        funcs[
            [
                "file",
                "kind",
                "name",
                "start_line",
                "end_line",
                "source_sha256",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nCRITICAL PROTOCOL LINES:"
    )

    print(
        protocol_lines.to_string(
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
        "\nR9-I-B COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
