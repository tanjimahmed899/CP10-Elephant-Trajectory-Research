from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd
import torch


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

B1_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B1_stationary_cv_transformer"
    / "R2D_B1_MANIFEST.json"
)

TRAIN_SCRIPT = (
    MAIN_RESULT
    / "train_lstm_gru.py"
)

KALMAN_SCRIPT = (
    MAIN_RESULT
    / "run_kalman_baseline.py"
)

CHECKPOINT_ROOT = (
    MAIN_RESULT
    / "lstm_gru_results"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2_preflight"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]

MODELS = [
    "GRU",
    "LSTM",
]

SEEDS = [
    42,
    52,
    62,
]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def require(path):
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(p)


def verify_b1():
    require(B1_MANIFEST)

    obj = json.loads(
        B1_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        obj.get("status")
        !=
        "R2D_B1_TARGETED_RECOMPUTATION_COMPLETE_REFERENCE_SANITY_PASS"
    ):
        raise RuntimeError(
            f"Unexpected B1 status: {obj.get('status')}"
        )

    return obj


def source_inventory(path):
    require(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(
        text
    )

    functions = []

    classes = []

    assignments = []

    constructors = []

    state_loads = []

    for node in tree.body:
        if isinstance(
            node,
            ast.FunctionDef
        ):
            try:
                args = ast.unparse(
                    node.args
                )
            except Exception:
                args = "<unparse_failed>"

            functions.append({
                "name":
                    node.name,

                "line":
                    int(
                        node.lineno
                    ),

                "args":
                    args,
            })

        elif isinstance(
            node,
            ast.ClassDef
        ):
            classes.append({
                "name":
                    node.name,

                "line":
                    int(
                        node.lineno
                    ),

                "bases":
                    [
                        ast.unparse(
                            base
                        )
                        for base in node.bases
                    ],
            })

        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            src = ast.get_source_segment(
                text,
                node,
            )

            if src is None:
                continue

            low = src.lower()

            if any(
                token in low
                for token in [
                    "q",
                    "r",
                    "kalman",
                    "hidden",
                    "num_layers",
                    "dropout",
                    "batch",
                    "lr",
                    "future",
                    "history",
                    "model_root",
                    "result",
                ]
            ):
                assignments.append({
                    "line":
                        int(
                            getattr(
                                node,
                                "lineno",
                                -1
                            )
                        ),

                    "source":
                        src,
                })

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call
        ):
            continue

        func_name = None

        if isinstance(
            node.func,
            ast.Name
        ):
            func_name = node.func.id

        elif isinstance(
            node.func,
            ast.Attribute
        ):
            func_name = node.func.attr

        if func_name is None:
            continue

        expr = ast.get_source_segment(
            text,
            node,
        )

        low = (
            str(
                expr
            ).lower()
            if expr is not None
            else
            ""
        )

        if any(
            token in low
            for token in [
                "lstm",
                "gru",
                "trajectory",
                "model(",
            ]
        ):
            constructors.append({
                "line":
                    int(
                        getattr(
                            node,
                            "lineno",
                            -1
                        )
                    ),

                "call_name":
                    func_name,

                "expression":
                    expr,
            })

        if func_name == "load_state_dict":
            state_loads.append({
                "line":
                    int(
                        getattr(
                            node,
                            "lineno",
                            -1
                        )
                    ),

                "expression":
                    expr,
            })

    return {
        "path":
            str(
                path
            ),

        "sha256":
            sha256_file(
                path
            ),

        "functions":
            functions,

        "classes":
            classes,

        "assignments":
            assignments,

        "constructors":
            constructors,

        "load_state_dict_calls":
            state_loads,
    }


def extract_function_source(
    path,
    function_names,
):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(
        text
    )

    rows = []

    for node in tree.body:
        if (
            isinstance(
                node,
                ast.FunctionDef
            )
            and
            node.name
            in
            function_names
        ):
            rows.append({
                "function":
                    node.name,

                "line":
                    int(
                        node.lineno
                    ),

                "source":
                    ast.get_source_segment(
                        text,
                        node,
                    ),
            })

    return rows


def checkpoint_inventory():
    rows = []

    for task in TASKS:
        for model in MODELS:
            for seed in SEEDS:
                path = (
                    CHECKPOINT_ROOT
                    /
                    f"{task}_{model}_seed{seed}.pt"
                )

                require(path)

                checkpoint = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )

                if isinstance(
                    checkpoint,
                    dict
                ):
                    keys = sorted(
                        map(
                            str,
                            checkpoint.keys()
                        )
                    )

                    scalar_meta = {}

                    tensor_entries = []

                    for key, value in checkpoint.items():
                        if torch.is_tensor(
                            value
                        ):
                            tensor_entries.append({
                                "key":
                                    str(
                                        key
                                    ),

                                "shape":
                                    list(
                                        value.shape
                                    ),

                                "dtype":
                                    str(
                                        value.dtype
                                    ),
                            })

                        elif isinstance(
                            value,
                            dict
                        ):
                            if value and all(
                                torch.is_tensor(
                                    v
                                )
                                for v in value.values()
                            ):
                                tensor_entries.append({
                                    "key":
                                        str(
                                            key
                                        ),

                                    "state_dict_entries":
                                        int(
                                            len(
                                                value
                                            )
                                        ),

                                    "state_dict_preview":
                                        [
                                            {
                                                "name":
                                                    str(
                                                        k
                                                    ),

                                                "shape":
                                                    list(
                                                        v.shape
                                                    ),

                                                "dtype":
                                                    str(
                                                        v.dtype
                                                    ),
                                            }
                                            for k, v in list(
                                                value.items()
                                            )[
                                                :12
                                            ]
                                        ],
                                })

                            else:
                                scalar_meta[
                                    str(
                                        key
                                    )
                                ] = str(
                                    value
                                )[
                                    :1000
                                ]

                        elif isinstance(
                            value,
                            (
                                str,
                                int,
                                float,
                                bool,
                                type(
                                    None
                                ),
                            )
                        ):
                            scalar_meta[
                                str(
                                    key
                                )
                            ] = value

                        else:
                            scalar_meta[
                                str(
                                    key
                                )
                            ] = str(
                                type(
                                    value
                                )
                            )

                else:
                    keys = [
                        f"<non-dict:{type(checkpoint).__name__}>"
                    ]

                    scalar_meta = {}

                    tensor_entries = []

                rows.append({
                    "task":
                        task,

                    "model":
                        model,

                    "seed":
                        int(
                            seed
                        ),

                    "path":
                        str(
                            path
                        ),

                    "sha256":
                        sha256_file(
                            path
                        ),

                    "checkpoint_type":
                        type(
                            checkpoint
                        ).__name__,

                    "checkpoint_keys":
                        json.dumps(
                            keys
                        ),

                    "scalar_metadata_json":
                        json.dumps(
                            scalar_meta,
                            default=str,
                        ),

                    "tensor_structure_json":
                        json.dumps(
                            tensor_entries,
                            default=str,
                        ),
                })

    return pd.DataFrame(
        rows
    )


def kalman_parameter_evidence(
    inventory,
    function_rows,
):
    rows = []

    for item in inventory[
        "assignments"
    ]:
        low = item[
            "source"
        ].lower()

        if (
            "q"
            in low
            or
            "r"
            in low
            or
            "kalman"
            in low
        ):
            rows.append({
                "kind":
                    "assignment",

                "line":
                    item[
                        "line"
                    ],

                "evidence":
                    item[
                        "source"
                    ],
            })

    for item in function_rows:
        if item[
            "function"
        ] in {
            "kalman_forecast",
            "main",
            "load_dataset",
        }:
            rows.append({
                "kind":
                    f"function::{item['function']}",

                "line":
                    item[
                        "line"
                    ],

                "evidence":
                    item[
                        "source"
                    ],
            })

    return pd.DataFrame(
        rows
    )


def model_constructor_evidence(
    inventory,
    function_rows,
):
    rows = []

    for item in inventory[
        "classes"
    ]:
        rows.append({
            "kind":
                "class",

            "line":
                item[
                    "line"
                ],

            "evidence":
                json.dumps(
                    item
                ),
        })

    for item in inventory[
        "constructors"
    ]:
        rows.append({
            "kind":
                "constructor_call",

            "line":
                item[
                    "line"
                ],

            "evidence":
                item[
                    "expression"
                ],
        })

    for item in inventory[
        "load_state_dict_calls"
    ]:
        rows.append({
            "kind":
                "load_state_dict",

            "line":
                item[
                    "line"
                ],

            "evidence":
                item[
                    "expression"
                ],
        })

    for item in function_rows:
        if item[
            "function"
        ] in {
            "build_model",
            "predict_model",
            "evaluate_dataset",
            "main",
        }:
            rows.append({
                "kind":
                    f"function::{item['function']}",

                "line":
                    item[
                        "line"
                    ],

                "evidence":
                    item[
                        "source"
                    ],
            })

    return pd.DataFrame(
        rows
    )


def readiness(
    checkpoints,
    kalman_evidence,
    model_evidence,
):
    checks = []

    checks.append({
        "check":
            "checkpoint_count",

        "pass":
            bool(
                len(
                    checkpoints
                )
                ==
                12
            ),

        "note":
            f"{len(checkpoints)} GRU/LSTM checkpoints found",
    })

    checks.append({
        "check":
            "kalman_source_evidence",

        "pass":
            bool(
                len(
                    kalman_evidence
                )
                >
                0
            ),

        "note":
            f"{len(kalman_evidence)} evidence rows",
    })

    checks.append({
        "check":
            "lstm_gru_model_construction_evidence",

        "pass":
            bool(
                len(
                    model_evidence
                )
                >
                0
            ),

        "note":
            f"{len(model_evidence)} evidence rows",
    })

    return pd.DataFrame(
        checks
    )


def main():
    print(
        "\nR2D-B2A — TESTREAL KALMAN + GRU/LSTM RECOMPUTATION PREFLIGHT"
    )

    print(
        "NO new TestReal ADE/FDE is generated. "
        "This step only recovers exact Kalman parameters, GRU/LSTM architecture/loading "
        "logic, and all 12 frozen neural checkpoints.\n"
    )

    b1 = verify_b1()

    train_inventory = source_inventory(
        TRAIN_SCRIPT
    )

    kalman_inventory = source_inventory(
        KALMAN_SCRIPT
    )

    train_functions = extract_function_source(
        TRAIN_SCRIPT,
        {
            "build_model",
            "predict_model",
            "evaluate_dataset",
            "main",
            "inverse_transform_batch",
        },
    )

    kalman_functions = extract_function_source(
        KALMAN_SCRIPT,
        {
            "kalman_forecast",
            "load_dataset",
            "main",
        },
    )

    checkpoints = checkpoint_inventory()

    kalman_evidence = kalman_parameter_evidence(
        kalman_inventory,
        kalman_functions,
    )

    model_evidence = model_constructor_evidence(
        train_inventory,
        train_functions,
    )

    readiness_df = readiness(
        checkpoints,
        kalman_evidence,
        model_evidence,
    )

    checkpoint_path = (
        OUT
        /
        "R2D_B2A_lstm_gru_checkpoint_inventory.csv"
    )

    kalman_path = (
        OUT
        /
        "R2D_B2A_kalman_parameter_evidence.csv"
    )

    model_path = (
        OUT
        /
        "R2D_B2A_lstm_gru_model_loading_evidence.csv"
    )

    readiness_path = (
        OUT
        /
        "R2D_B2A_readiness_checks.csv"
    )

    source_inventory_path = (
        OUT
        /
        "R2D_B2A_source_inventory.json"
    )

    checkpoints.to_csv(
        checkpoint_path,
        index=False,
    )

    kalman_evidence.to_csv(
        kalman_path,
        index=False,
    )

    model_evidence.to_csv(
        model_path,
        index=False,
    )

    readiness_df.to_csv(
        readiness_path,
        index=False,
    )

    source_inventory_path.write_text(
        json.dumps(
            {
                "train_lstm_gru":
                    train_inventory,

                "run_kalman_baseline":
                    kalman_inventory,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_pass = bool(
        readiness_df[
            "pass"
        ].all()
    )

    status = (
        "R2D_B2A_PREFLIGHT_COMPLETE_READY_FOR_TESTREAL_KALMAN_GRU_LSTM_RECOMPUTATION"
        if all_pass
        else
        "R2D_B2A_PREFLIGHT_COMPLETE_EXACT_LOADING_PATCH_REQUIRED"
    )

    summary = {
        "step":
            "R2D_B2A_TESTREAL_KALMAN_GRU_LSTM_PREFLIGHT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "b1_status":
            b1.get(
                "status"
            ),

        "new_metrics_generated":
            False,

        "train_lstm_gru_source":
            str(
                TRAIN_SCRIPT
            ),

        "train_lstm_gru_sha256":
            sha256_file(
                TRAIN_SCRIPT
            ),

        "kalman_source":
            str(
                KALMAN_SCRIPT
            ),

        "kalman_source_sha256":
            sha256_file(
                KALMAN_SCRIPT
            ),

        "checkpoint_inventory_csv":
            str(
                checkpoint_path
            ),

        "kalman_evidence_csv":
            str(
                kalman_path
            ),

        "model_loading_evidence_csv":
            str(
                model_path
            ),

        "readiness_csv":
            str(
                readiness_path
            ),
    }

    summary_path = (
        OUT
        /
        "R2D_B2A_SUMMARY.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "FROZEN GRU/LSTM CHECKPOINTS:"
    )

    print(
        checkpoints[
            [
                "task",
                "model",
                "seed",
                "path",
                "sha256",
                "checkpoint_type",
                "checkpoint_keys",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nKALMAN PARAMETER / CALL EVIDENCE:"
    )

    print(
        kalman_evidence.to_string(
            index=False
        )
        if len(
            kalman_evidence
        )
        >
        0
        else
        "No Kalman evidence found."
    )

    print(
        "\nGRU/LSTM MODEL LOADING EVIDENCE:"
    )

    print(
        model_evidence.to_string(
            index=False
        )
        if len(
            model_evidence
        )
        >
        0
        else
        "No GRU/LSTM model-loading evidence found."
    )

    print(
        "\nREADINESS CHECKS:"
    )

    print(
        readiness_df.to_string(
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
        "\nR2D-B2A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
