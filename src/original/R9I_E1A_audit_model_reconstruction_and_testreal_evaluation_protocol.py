
from pathlib import Path
import ast
import hashlib
import inspect
import json
from datetime import datetime, timezone

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

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_E1A_model_reconstruction_and_evaluation_audit"
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


def verify_manifests():
    for p in [
        R9I_D1_MANIFEST,
        R9I_E0_MANIFEST,
        TRANSFORMER_SRC,
        RECURRENT_SRC,
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

    if not d1.get(
        "all_18_runs_complete",
        False
    ):
        raise RuntimeError(
            "R9-I-D1 does not confirm all 18 checkpoints."
        )

    return (
        d1,
        e0,
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


def function_source_rows(
    source_path,
    function_names,
):
    text = source_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(
            source_path
        )
    )

    rows = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            continue

        if node.name not in function_names:
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
                    source_path.name,

                "function":
                    node.name,

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


def model_instantiation_context(
    source_path,
):
    text = source_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    needles = [
        "TrajectoryTransformer(",
        "RecurrentForecaster(",
        "load_state_dict(",
        "torch.load(",
    ]

    selected = set()

    for i, line in enumerate(
        lines,
        start=1
    ):
        if any(
            needle in line
            for needle in needles
        ):
            for j in range(
                max(
                    1,
                    i - 12
                ),
                min(
                    len(
                        lines
                    ),
                    i + 18
                )
                + 1
            ):
                selected.add(
                    j
                )

    return pd.DataFrame(
        [
            {
                "file":
                    source_path.name,

                "line_number":
                    j,

                "line":
                    lines[
                        j - 1
                    ],
            }
            for j in sorted(
                selected
            )
        ]
    )


def checkpoint_inventory(
    d1_manifest,
):
    expected_by_key = {}

    for row in d1_manifest[
        "runs"
    ]:
        key = (
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

        expected_by_key[
            key
        ] = row

    rows = []

    for task in TASKS:
        for seed in SEEDS:
            for model in [
                "Transformer",
                "LSTM",
                "GRU",
            ]:
                if model == "Transformer":
                    p = (
                        CHECKPOINT_ROOT
                        /
                        "transformer"
                        /
                        f"{task}_Transformer_seed{seed}.pt"
                    )
                else:
                    p = (
                        CHECKPOINT_ROOT
                        /
                        "lstm_gru"
                        /
                        f"{task}_{model}_seed{seed}.pt"
                    )

                require(
                    p
                )

                key = (
                    task,
                    model,
                    seed,
                )

                if key not in expected_by_key:
                    raise RuntimeError(
                        f"Manifest missing checkpoint identity {key}."
                    )

                current_hash = sha256_file(
                    p
                )

                expected_hash = expected_by_key[
                    key
                ][
                    "checkpoint_sha256"
                ]

                if current_hash != expected_hash:
                    raise RuntimeError(
                        f"Checkpoint hash mismatch: {p}"
                    )

                rows.append(
                    {
                        "task":
                            task,

                        "model":
                            model,

                        "seed":
                            seed,

                        "path":
                            str(
                                p
                            ),

                        "sha256":
                            current_hash,

                        "best_epoch":
                            int(
                                expected_by_key[
                                    key
                                ][
                                    "best_epoch"
                                ]
                            ),

                        "best_validation_track_macro_ADE":
                            float(
                                expected_by_key[
                                    key
                                ][
                                    "best_validation_track_macro_ADE"
                                ]
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def testreal_inventory(
    e0_manifest,
):
    rows = []

    by_task = {
        row[
            "task"
        ]:
            row
        for row in e0_manifest[
            "tasks"
        ]
    }

    for task in TASKS:
        npz_path = (
            TESTREAL_DIR
            /
            f"{task}_birdsai_test_canonical.npz"
        )

        meta_path = (
            TESTREAL_DIR
            /
            f"{task}_birdsai_test_metadata.csv"
        )

        require(
            npz_path
        )

        require(
            meta_path
        )

        row = by_task[
            task
        ]

        if (
            sha256_file(
                npz_path
            )
            !=
            row[
                "output_npz_sha256"
            ]
        ):
            raise RuntimeError(
                f"{task}: TestReal NPZ hash mismatch."
            )

        if (
            sha256_file(
                meta_path
            )
            !=
            row[
                "output_metadata_sha256"
            ]
        ):
            raise RuntimeError(
                f"{task}: TestReal metadata hash mismatch."
            )

        rows.append(
            {
                "task":
                    task,

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

                "samples":
                    int(
                        row[
                            "samples"
                        ]
                    ),

                "tracks":
                    int(
                        row[
                            "tracks"
                        ]
                    ),

                "sequences":
                    int(
                        row[
                            "sequences"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def inspect_signatures(
    transformer_ns,
    recurrent_ns,
):
    rows = []

    targets = [
        (
            "train_transformer.py",
            "TrajectoryTransformer",
            transformer_ns[
                "TrajectoryTransformer"
            ],
        ),
        (
            "train_transformer.py",
            "evaluate",
            transformer_ns[
                "evaluate"
            ],
        ),
        (
            "train_transformer.py",
            "predict_model",
            transformer_ns[
                "predict_model"
            ],
        ),
        (
            "train_lstm_gru.py",
            "RecurrentForecaster",
            recurrent_ns[
                "RecurrentForecaster"
            ],
        ),
        (
            "train_lstm_gru.py",
            "evaluate_dataset",
            recurrent_ns[
                "evaluate_dataset"
            ],
        ),
        (
            "train_lstm_gru.py",
            "predict_model",
            recurrent_ns[
                "predict_model"
            ],
        ),
    ]

    for file_name, name, obj in targets:
        try:
            signature = str(
                inspect.signature(
                    obj
                )
            )
        except Exception as exc:
            signature = (
                f"<signature unavailable: {exc}>"
            )

        rows.append(
            {
                "file":
                    file_name,

                "object":
                    name,

                "signature":
                    signature,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-I-E1A — MODEL RECONSTRUCTION / FROZEN TESTREAL EVALUATION AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "The 18 matched isotropic checkpoints and matched BIRDSAI TestReal isotropic "
        "datasets are hash-verified. This step audits exact constructor signatures, "
        "evaluation/prediction function signatures, model-instantiation source context, "
        "and checkpoint identities before frozen inference.\n"
    )

    (
        d1,
        e0,
    ) = verify_manifests()

    transformer_ns = make_definition_only_module(
        TRANSFORMER_SRC,
        "r9i_e1a_transformer_defs",
    )

    recurrent_ns = make_definition_only_module(
        RECURRENT_SRC,
        "r9i_e1a_recurrent_defs",
    )

    sig_df = inspect_signatures(
        transformer_ns,
        recurrent_ns,
    )

    checkpoint_df = checkpoint_inventory(
        d1
    )

    testreal_df = testreal_inventory(
        e0
    )

    function_df = pd.concat(
        [
            function_source_rows(
                TRANSFORMER_SRC,
                [
                    "predict_model",
                    "calculate_metrics",
                    "evaluate",
                ],
            ),
            function_source_rows(
                RECURRENT_SRC,
                [
                    "predict_model",
                    "calculate_metrics",
                    "evaluate_dataset",
                ],
            ),
        ],
        ignore_index=True
    )

    instantiation_df = pd.concat(
        [
            model_instantiation_context(
                TRANSFORMER_SRC
            ),
            model_instantiation_context(
                RECURRENT_SRC
            ),
        ],
        ignore_index=True
    )

    sig_path = (
        OUT
        /
        "R9I_E1A_SIGNATURES.csv"
    )

    checkpoint_path = (
        OUT
        /
        "R9I_E1A_CHECKPOINT_INVENTORY.csv"
    )

    testreal_path = (
        OUT
        /
        "R9I_E1A_TESTREAL_INPUT_INVENTORY.csv"
    )

    function_path = (
        OUT
        /
        "R9I_E1A_EVALUATION_FUNCTION_SOURCES.csv"
    )

    instantiation_path = (
        OUT
        /
        "R9I_E1A_MODEL_INSTANTIATION_CONTEXT.csv"
    )

    sig_df.to_csv(
        sig_path,
        index=False
    )

    checkpoint_df.to_csv(
        checkpoint_path,
        index=False
    )

    testreal_df.to_csv(
        testreal_path,
        index=False
    )

    function_df.to_csv(
        function_path,
        index=False
    )

    instantiation_df.to_csv(
        instantiation_path,
        index=False
    )

    status = (
        "R9I_E1A_MODEL_RECONSTRUCTION_AND_TESTREAL_EVALUATION_PROTOCOL_AUDITED_READY_FOR_R9I_E1B_FROZEN_18_CHECKPOINT_INFERENCE"
    )

    manifest = {
        "step":
            "R9I_E1A_MODEL_RECONSTRUCTION_AND_TESTREAL_EVALUATION_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "checkpoints_verified":
            int(
                len(
                    checkpoint_df
                )
            ),

        "expected_checkpoints":
            18,

        "testreal_tasks_verified":
            int(
                len(
                    testreal_df
                )
            ),

        "training":
            False,

        "model_inference":
            False,

        "checkpoint_modification":
            False,

        "outputs": {
            "signatures":
                str(
                    sig_path
                ),

            "checkpoint_inventory":
                str(
                    checkpoint_path
                ),

            "testreal_inventory":
                str(
                    testreal_path
                ),

            "evaluation_function_sources":
                str(
                    function_path
                ),

            "model_instantiation_context":
                str(
                    instantiation_path
                ),
        },
    }

    (
        OUT
        /
        "R9I_E1A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "MODEL / EVALUATION SIGNATURES:"
    )

    print(
        sig_df.to_string(
            index=False
        )
    )

    print(
        "\nCHECKPOINT INVENTORY:"
    )

    print(
        checkpoint_df.to_string(
            index=False
        )
    )

    print(
        "\nTESTREAL INPUT INVENTORY:"
    )

    print(
        testreal_df.to_string(
            index=False
        )
    )

    print(
        "\nMODEL INSTANTIATION / LOAD CONTEXT:"
    )

    print(
        instantiation_df.to_string(
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
        "\nR9-I-E1A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
