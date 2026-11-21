from pathlib import Path
import ast
import hashlib
import json
import re
import sys

import pandas as pd

READ_ONLY_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

PRIMARY_EVAL = (
    READ_ONLY_ROOT
    / "evaluate_birdsai_testreal_frozen_ensemble.py"
)

SECONDARY_EVAL = (
    READ_ONLY_ROOT
    / "evaluate_transformer_ensemble.py"
)

TRAIN_SRC = (
    READ_ONLY_ROOT
    / "train_transformer.py"
)

CHECKPOINT_DIR = (
    READ_ONLY_ROOT
    / "transformer_results"
)

BAMBI_CANONICAL_LOCK = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k2b3_final_canonical"
    / "STEP6D_K2B3_CANONICAL_DATASET_LOCK.json"
)

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_frozen_ensemble"
    / "model_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

LOCK_FILE = (
    OUT
    / "STEP6D_K3A1_FROZEN_TRANSFORMER_MODEL_LOCK.json"
)

EXPECTED_TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]

EXPECTED_SEEDS = [
    42,
    52,
    62,
]

EXPECTED_CHECKPOINTS = {
    "2s_to_2s": {
        42: "2s_to_2s_Transformer_seed42.pt",
        52: "2s_to_2s_Transformer_seed52.pt",
        62: "2s_to_2s_Transformer_seed62.pt",
    },
    "2s_to_5s": {
        42: "2s_to_5s_Transformer_seed42.pt",
        52: "2s_to_5s_Transformer_seed52.pt",
        62: "2s_to_5s_Transformer_seed62.pt",
    },
}


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def get_function_source(path, function_name):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(text)

    matches = [
        n
        for n in ast.walk(tree)
        if isinstance(
            n,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and
        n.name == function_name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{path.name}: expected exactly one "
            f"{function_name}, found {len(matches)}"
        )

    node = matches[0]

    source = ast.get_source_segment(
        text,
        node
    )

    if not source:
        lines = text.splitlines()

        source = "\n".join(
            lines[
                node.lineno - 1:
                getattr(
                    node,
                    "end_lineno",
                    node.lineno
                )
            ]
        )

    return {
        "source": source,
        "line_start": int(node.lineno),
        "line_end": int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        ),
        "sha256": sha256_text(source),
    }


def lexical_evidence(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    rows = []

    patterns = [
        r"seed",
        r"42",
        r"52",
        r"62",
        r"2s_to_2s",
        r"2s_to_5s",
        r"Transformer",
        r"transformer_results",
        r"load_transformer",
        r"ensemble",
    ]

    for i, line in enumerate(
        lines,
        start=1
    ):
        low = line.lower()

        if not any(
            re.search(
                p.lower(),
                low
            )
            for p in patterns
        ):
            continue

        lo = max(
            1,
            i - 2
        )

        hi = min(
            len(lines),
            i + 2
        )

        context = "\n".join(
            f"{j:05d}: {lines[j-1]}"
            for j in range(
                lo,
                hi + 1
            )
        )

        rows.append({
            "file": str(path),
            "line": int(i),
            "text": line.strip(),
            "context": context,
        })

    return pd.DataFrame(
        rows
    )


def ast_literal_inventory(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text
    )

    rows = []

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            continue

        targets = []

        if isinstance(
            node,
            ast.Assign
        ):
            for t in node.targets:
                if isinstance(
                    t,
                    ast.Name
                ):
                    targets.append(
                        t.id
                    )

        elif isinstance(
            node,
            ast.AnnAssign
        ):
            if isinstance(
                node.target,
                ast.Name
            ):
                targets.append(
                    node.target.id
                )

        if not targets:
            continue

        try:
            value = ast.literal_eval(
                node.value
            )
        except Exception:
            continue

        joined = " ".join(
            targets
        ).lower()

        # Keep likely task/seed/model configuration literals.
        if not any(
            key in joined
            for key in [
                "seed",
                "task",
                "config",
                "model",
                "transformer",
                "horizon",
                "future",
            ]
        ):
            continue

        rows.append({
            "file": str(path),
            "line": int(node.lineno),
            "targets": json.dumps(targets),
            "value_repr": repr(value),
        })

    return pd.DataFrame(
        rows
    )


def checkpoint_inventory():
    rows = []

    for task in EXPECTED_TASKS:
        for seed in EXPECTED_SEEDS:
            filename = EXPECTED_CHECKPOINTS[
                task
            ][
                seed
            ]

            path = (
                CHECKPOINT_DIR
                / filename
            )

            require(
                path
            )

            rows.append({
                "task":
                    task,

                "seed":
                    int(seed),

                "checkpoint":
                    str(path),

                "filename":
                    filename,

                "size_bytes":
                    int(
                        path.stat().st_size
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            })

    return pd.DataFrame(
        rows
    )


def source_text_contains_mapping(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    checks = []

    for task in EXPECTED_TASKS:
        task_seen = (
            task
            in
            text
        )

        checks.append({
            "file":
                str(path),

            "kind":
                "task",

            "value":
                task,

            "present":
                bool(
                    task_seen
                ),
        })

    for seed in EXPECTED_SEEDS:
        seed_seen = bool(
            re.search(
                rf"(?<!\d){seed}(?!\d)",
                text
            )
        )

        checks.append({
            "file":
                str(path),

            "kind":
                "seed",

            "value":
                str(seed),

            "present":
                seed_seen,
        })

    transformer_seen = (
        "Transformer"
        in
        text
        or
        "transformer"
        in
        text
    )

    checks.append({
        "file":
            str(path),

        "kind":
            "model_family",

        "value":
            "Transformer",

        "present":
            bool(
                transformer_seen
            ),
    })

    return pd.DataFrame(
        checks
    )


def main():
    print(
        "\nSTEP 6D-K3A1 — FROZEN TRANSFORMER "
        "TASK/SEED MAPPING + PRE-INFERENCE MODEL LOCK"
    )

    print(
        "No BAMBI model prediction is generated in this step.\n"
    )

    for p in [
        PRIMARY_EVAL,
        SECONDARY_EVAL,
        TRAIN_SRC,
        CHECKPOINT_DIR,
        BAMBI_CANONICAL_LOCK,
    ]:
        require(
            p
        )

    canonical_lock = json.loads(
        BAMBI_CANONICAL_LOCK.read_text(
            encoding="utf-8"
        )
    )

    if (
        canonical_lock.get(
            "status"
        )
        !=
        "LOCKED_BEFORE_ANY_BAMBI_MODEL_PREDICTIONS"
    ):
        raise RuntimeError(
            "BAMBI canonical lock is not in the expected "
            "pre-prediction state."
        )

    # ------------------------------------------------------------------
    # 1. Freeze exact six Transformer checkpoints.
    # ------------------------------------------------------------------
    checkpoints = checkpoint_inventory()

    checkpoints.to_csv(
        OUT
        / "TABLE_step6dk3a1_locked_transformer_checkpoints.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # 2. Source-confirm task/seed/model-family evidence.
    # ------------------------------------------------------------------
    mapping_checks = []

    for src in [
        PRIMARY_EVAL,
        SECONDARY_EVAL,
        TRAIN_SRC,
    ]:
        mapping_checks.append(
            source_text_contains_mapping(
                src
            )
        )

    mapping_checks = pd.concat(
        mapping_checks,
        ignore_index=True
    )

    mapping_checks.to_csv(
        OUT
        / "step6dk3a1_source_mapping_presence.csv",
        index=False
    )

    primary_checks = mapping_checks[
        mapping_checks[
            "file"
        ]
        ==
        str(
            PRIMARY_EVAL
        )
    ]

    primary_mapping_complete = bool(
        primary_checks[
            "present"
        ].all()
    )

    # ------------------------------------------------------------------
    # 3. Freeze exact loader source.
    # ------------------------------------------------------------------
    loader_primary = get_function_source(
        PRIMARY_EVAL,
        "load_transformer"
    )

    loader_secondary = get_function_source(
        SECONDARY_EVAL,
        "load_transformer"
    )

    train_inverse = get_function_source(
        TRAIN_SRC,
        "inverse_transform_batch"
    )

    with open(
        OUT
        / "STEP6D_K3A1_LOCKED_SOURCE_FUNCTIONS.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "PRIMARY FROZEN ENSEMBLE LOADER\n"
            +
            "=" * 100
            +
            "\n"
        )

        f.write(
            f"FILE: {PRIMARY_EVAL}\n"
            f"LINES: {loader_primary['line_start']}-"
            f"{loader_primary['line_end']}\n"
            f"SHA256: {loader_primary['sha256']}\n\n"
        )

        f.write(
            loader_primary[
                "source"
            ]
        )

        f.write(
            "\n\nSECONDARY TRANSFORMER LOADER\n"
            +
            "=" * 100
            +
            "\n"
        )

        f.write(
            f"FILE: {SECONDARY_EVAL}\n"
            f"LINES: {loader_secondary['line_start']}-"
            f"{loader_secondary['line_end']}\n"
            f"SHA256: {loader_secondary['sha256']}\n\n"
        )

        f.write(
            loader_secondary[
                "source"
            ]
        )

        f.write(
            "\n\nTRAIN INVERSE TRANSFORM\n"
            +
            "=" * 100
            +
            "\n"
        )

        f.write(
            f"FILE: {TRAIN_SRC}\n"
            f"LINES: {train_inverse['line_start']}-"
            f"{train_inverse['line_end']}\n"
            f"SHA256: {train_inverse['sha256']}\n\n"
        )

        f.write(
            train_inverse[
                "source"
            ]
        )

    # ------------------------------------------------------------------
    # 4. Save lexical/AST evidence for review.
    # ------------------------------------------------------------------
    lexical_parts = []
    literal_parts = []

    for src in [
        PRIMARY_EVAL,
        SECONDARY_EVAL,
        TRAIN_SRC,
    ]:
        lexical_parts.append(
            lexical_evidence(
                src
            )
        )

        literal_parts.append(
            ast_literal_inventory(
                src
            )
        )

    lexical = pd.concat(
        lexical_parts,
        ignore_index=True
    )

    literal_inventory = pd.concat(
        literal_parts,
        ignore_index=True
    )

    lexical.to_csv(
        OUT
        / "step6dk3a1_lexical_mapping_evidence.csv",
        index=False
    )

    literal_inventory.to_csv(
        OUT
        / "step6dk3a1_ast_literal_inventory.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # 5. Confirm there are exactly 6 expected Transformer checkpoints.
    #    Other architectures may exist elsewhere but are not part of this lock.
    # ------------------------------------------------------------------
    actual_transformer_files = sorted(
        p.name
        for p in CHECKPOINT_DIR.glob(
            "*Transformer*.pt"
        )
    )

    expected_files = sorted(
        EXPECTED_CHECKPOINTS[
            task
        ][
            seed
        ]
        for task in EXPECTED_TASKS
        for seed in EXPECTED_SEEDS
    )

    exact_checkpoint_set_match = (
        actual_transformer_files
        ==
        expected_files
    )

    # ------------------------------------------------------------------
    # 6. Create immutable pre-inference model manifest.
    # ------------------------------------------------------------------
    lock = {
        "step":
            "STEP_6D_K3A1_FROZEN_TRANSFORMER_MODEL_LOCK",

        "status":
            "LOCKED_BEFORE_ANY_BAMBI_MODEL_PREDICTIONS",

        "model_family":
            "Transformer",

        "ensemble_rule":
            "mean prediction across 3 frozen seed models",

        "tasks":
            EXPECTED_TASKS,

        "seeds":
            EXPECTED_SEEDS,

        "checkpoint_set":
            checkpoints.to_dict(
                orient="records"
            ),

        "exact_transformer_checkpoint_set_match":
            bool(
                exact_checkpoint_set_match
            ),

        "primary_frozen_evaluator":
            str(
                PRIMARY_EVAL
            ),

        "primary_load_transformer":
            {
                "line_start":
                    loader_primary[
                        "line_start"
                    ],

                "line_end":
                    loader_primary[
                        "line_end"
                    ],

                "source_sha256":
                    loader_primary[
                        "sha256"
                    ],
            },

        "secondary_transformer_evaluator":
            str(
                SECONDARY_EVAL
            ),

        "secondary_load_transformer":
            {
                "line_start":
                    loader_secondary[
                        "line_start"
                    ],

                "line_end":
                    loader_secondary[
                        "line_end"
                    ],

                "source_sha256":
                    loader_secondary[
                        "sha256"
                    ],
            },

        "canonical_dataset_lock":
            str(
                BAMBI_CANONICAL_LOCK
            ),

        "canonical_dataset_lock_sha256":
            sha256_file(
                BAMBI_CANONICAL_LOCK
            ),

        "source_mapping_complete_in_primary_evaluator":
            bool(
                primary_mapping_complete
            ),

        "no_predictions_generated":
            True,

        "no_retraining":
            True,

        "no_recalibration":
            True,

        "important_rules": [
            "Only the six locked Transformer checkpoints may be used for BAMBI confirmatory inference.",
            "Seeds are fixed at 42, 52, and 62 for both 2s->2s and 2s->5s tasks.",
            "The ensemble prediction is the mean of the three frozen seed predictions.",
            "No model selection, retraining, checkpoint substitution, or hyperparameter tuning may use BAMBI outcomes.",
            "BAMBI Flight 6 remains a cross-species external confirmatory robustness/stress test.",
        ],
    }

    if not exact_checkpoint_set_match:
        raise RuntimeError(
            "Transformer checkpoint directory does not contain "
            "exactly the expected six checkpoint filenames. "
            "Do not proceed to BAMBI inference."
        )

    if not primary_mapping_complete:
        raise RuntimeError(
            "Primary frozen evaluator does not contain complete "
            "task/seed/model mapping evidence. "
            "Do not proceed to BAMBI inference."
        )

    if LOCK_FILE.exists():
        old = json.loads(
            LOCK_FILE.read_text(
                encoding="utf-8"
            )
        )

        if old != lock:
            raise RuntimeError(
                "Existing K3A1 model lock differs. "
                "Refusing to overwrite."
            )

        lock_status = (
            "EXISTING_FROZEN_MODEL_LOCK_VERIFIED"
        )

    else:
        LOCK_FILE.write_text(
            json.dumps(
                lock,
                indent=2
            ),
            encoding="utf-8"
        )

        lock_status = (
            "NEW_FROZEN_MODEL_LOCK_CREATED"
        )

    summary = {
        "canonical_lock_verified":
            True,

        "exact_checkpoint_set_match":
            bool(
                exact_checkpoint_set_match
            ),

        "source_mapping_complete_in_primary_evaluator":
            bool(
                primary_mapping_complete
            ),

        "locked_tasks":
            EXPECTED_TASKS,

        "locked_seeds":
            EXPECTED_SEEDS,

        "locked_checkpoint_count":
            int(
                len(
                    checkpoints
                )
            ),

        "lock_status":
            lock_status,

        "status":
            "FROZEN_3SEED_TRANSFORMER_ENSEMBLE_READY_FOR_FIRST_BAMBI_INFERENCE",
    }

    (
        OUT
        / "STEP6D_K3A1_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    with open(
        OUT
        / "STEP6D_K3A1_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K3A1 — FROZEN TRANSFORMER MODEL LOCK\n"
        )

        f.write(
            "=" * 78
            +
            "\n\n"
        )

        f.write(
            "LOCKED CHECKPOINTS\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            checkpoints.to_string(
                index=False
            )
        )

        f.write(
            "\n\nSOURCE MAPPING CHECK\n"
        )

        f.write(
            "--------------------\n"
        )

        f.write(
            primary_checks.to_string(
                index=False
            )
        )

        f.write(
            "\n\nLOCK STATUS\n"
        )

        f.write(
            "-----------\n"
        )

        f.write(
            lock_status
            +
            "\n"
        )

        f.write(
            "\nSTATUS\n"
        )

        f.write(
            "------\n"
        )

        f.write(
            summary[
                "status"
            ]
            +
            "\n"
        )

    print(
        "LOCKED CHECKPOINTS:"
    )

    print(
        checkpoints.to_string(
            index=False
        )
    )

    print(
        "\nPRIMARY SOURCE MAPPING CHECK:"
    )

    print(
        primary_checks.to_string(
            index=False
        )
    )

    print(
        "\nEXACT TRANSFORMER CHECKPOINT SET MATCH:"
    )

    print(
        exact_checkpoint_set_match
    )

    print(
        "\nLOCK STATUS:"
    )

    print(
        lock_status
    )

    print(
        "\nSTATUS:"
    )

    print(
        summary[
            "status"
        ]
    )

    print(
        "\nSTEP 6D-K3A1 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
