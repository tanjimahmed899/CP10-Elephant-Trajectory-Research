from pathlib import Path
import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

EVALUATOR = (
    MAIN_RESULT
    / "evaluate_birdsai_testreal_frozen_ensemble.py"
)

MODEL_ROOT = (
    MAIN_RESULT
    / "transformer_results"
)

R2D_A3_SUMMARY = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_targeted_recompute_preflight_v3"
    / "R2D_A3_SUMMARY.json"
)

R2D_A3_RESOLVED = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_targeted_recompute_preflight_v3"
    / "R2D_A3_resolved_testreal_npz.csv"
)

R2D_A_SELECTED = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_targeted_recompute_preflight"
    / "R2D_A_selected_npz.csv"
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
    / "R2D_B1_stationary_cv_transformer"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [
    42,
    52,
    62,
]

EXPECTED = {
    ("BIRDSAI_TestReal", "2s_to_2s"): {
        "samples": 1053,
        "future_len": 20,
        "reference": {
            "Stationary": {
                "ADE": 0.020033,
                "FDE": 0.035129,
            },
            "ConstantVelocity": {
                "ADE": 0.020148,
                "FDE": 0.035185,
            },
            "TransformerEnsemble": {
                "ADE": 0.019076,
                "FDE": 0.032890,
            },
        },
    },

    ("BIRDSAI_TestReal", "2s_to_5s"): {
        "samples": 826,
        "future_len": 50,
        "reference": {
            "Stationary": {
                "ADE": 0.039685,
                "FDE": 0.073737,
            },
            "ConstantVelocity": {
                "ADE": 0.040174,
                "FDE": 0.079019,
            },
            "TransformerEnsemble": {
                "ADE": 0.035590,
                "FDE": 0.067247,
            },
        },
    },

    ("Fresh_WildBox", "2s_to_2s"): {
        "samples": 865,
        "future_len": 20,
        "reference": {
            "TransformerEnsemble": {
                "ADE": 0.028814,
                "FDE": 0.053920,
            },
        },
    },

    ("Fresh_WildBox", "2s_to_5s"): {
        "samples": 637,
        "future_len": 50,
        "reference": {
            "TransformerEnsemble": {
                "ADE": 0.050132,
                "FDE": 0.105339,
            },
        },
    },
}

CHECKPOINT_HASHES = {
    ("2s_to_2s", 42):
        "8244d6dc60be34d5c7f3dcbb1a50036152f8c5e96d69f3b3f5a93ac6e4a7f14f",

    ("2s_to_2s", 52):
        "9264f2d81945601149cc2f6a970a2765942cb9a45e711c4d552434e9feb75eb5",

    ("2s_to_2s", 62):
        "0257eb576a78a6a40da51d8bbf728df30d5d8b77cd6ca7e21be81fa0fd3cded2",

    ("2s_to_5s", 42):
        "846e1398eb8354e871ea80b7a90888f99d02ce10dac56f0787bf3205588bbb0a",

    ("2s_to_5s", 52):
        "e8316f3db5a60e72d1fee28384726b5353ffaf8c68ffeec64abb2326b4098958",

    ("2s_to_5s", 62):
        "44d1d6d0e15af6844667b44e2bf6a4cb574b90884df8250af138e7841ec82b3a",
}


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


def verify_prerequisites():
    require(EVALUATOR)
    require(R2D_A3_SUMMARY)
    require(R2D_A3_RESOLVED)
    require(R2D_A_SELECTED)
    require(R2D_A_METADATA)
    require(MODEL_ROOT)

    summary = json.loads(
        R2D_A3_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    if (
        summary.get("status")
        !=
        "R2D_A3_TESTREAL_NPZ_EXACTLY_RESOLVED_READY_FOR_TARGETED_RECOMPUTATION"
    ):
        raise RuntimeError(
            f"Unexpected R2D-A3 status: {summary.get('status')}"
        )

    for (task, seed), expected_hash in CHECKPOINT_HASHES.items():
        path = (
            MODEL_ROOT
            /
            f"{task}_Transformer_seed{seed}.pt"
        )

        require(path)

        actual = sha256_file(path)

        if actual != expected_hash:
            raise RuntimeError(
                f"Checkpoint hash mismatch for {task} seed {seed}.\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual}\n"
                f"Path: {path}"
            )


def bound_names(node):
    names = set()

    if isinstance(node, ast.Assign):
        targets = node.targets

    elif isinstance(node, ast.AnnAssign):
        targets = [
            node.target
        ]

    else:
        return names

    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.add(
                    sub.id
                )

    return names


def loaded_names(node):
    names = set()

    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Name)
            and
            isinstance(sub.ctx, ast.Load)
        ):
            names.add(
                sub.id
            )

    return names


def safe_extract_namespace(
    source_path,
    target_functions,
):
    text = source_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(
        text
    )

    imports = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
            )
        )
    ]

    funcs = {
        node.name:
            node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        )
    }

    classes = {
        node.name:
            node
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef
        )
    }

    assignments = {}

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            for name in bound_names(
                node
            ):
                assignments[
                    name
                ] = node

    missing = [
        name
        for name in target_functions
        if name not in funcs
    ]

    if missing:
        raise RuntimeError(
            f"Target functions missing from evaluator: {missing}"
        )

    selected_funcs = set(
        target_functions
    )

    selected_classes = set()
    selected_assignments = set()

    pending = set()

    for name in selected_funcs:
        pending.update(
            loaded_names(
                funcs[
                    name
                ]
            )
        )

    processed = set()

    while True:
        new_names = (
            pending
            -
            processed
        )

        if not new_names:
            break

        for name in list(
            new_names
        ):
            processed.add(
                name
            )

            if (
                name in funcs
                and
                name not in selected_funcs
            ):
                selected_funcs.add(
                    name
                )

                pending.update(
                    loaded_names(
                        funcs[
                            name
                        ]
                    )
                )

            elif (
                name in classes
                and
                name not in selected_classes
            ):
                selected_classes.add(
                    name
                )

                pending.update(
                    loaded_names(
                        classes[
                            name
                        ]
                    )
                )

            elif (
                name in assignments
                and
                name not in selected_assignments
            ):
                selected_assignments.add(
                    name
                )

                pending.update(
                    loaded_names(
                        assignments[
                            name
                        ]
                    )
                )

    selected_nodes = []

    selected_nodes.extend(
        imports
    )

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            if (
                bound_names(
                    node
                )
                &
                selected_assignments
            ):
                selected_nodes.append(
                    node
                )

        elif isinstance(
            node,
            ast.ClassDef
        ):
            if node.name in selected_classes:
                selected_nodes.append(
                    node
                )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            if node.name in selected_funcs:
                selected_nodes.append(
                    node
                )

    module_ast = ast.Module(
        body=selected_nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module_ast
    )

    namespace = {
        "__file__":
            str(
                source_path
            ),

        "__name__":
            "_r2d_b1_safe_evaluator_",
    }

    sys.path.insert(
        0,
        str(
            source_path.parent
        ),
    )

    exec(
        compile(
            module_ast,
            str(
                source_path
            ),
            "exec",
        ),
        namespace,
    )

    # Bind proven frozen model root.
    namespace[
        "MODEL_ROOT"
    ] = MODEL_ROOT

    for name in target_functions:
        if name not in namespace:
            raise RuntimeError(
                f"Safe extraction did not create {name}"
            )

    return namespace, {
        "selected_functions":
            sorted(
                selected_funcs
            ),

        "selected_classes":
            sorted(
                selected_classes
            ),

        "selected_assignments":
            sorted(
                selected_assignments
            ),

        "top_level_benchmark_executed":
            False,
    }


def extract_torch_module(
    loaded,
    torch,
):
    if isinstance(
        loaded,
        torch.nn.Module
    ):
        return loaded

    if isinstance(
        loaded,
        (
            tuple,
            list,
        )
    ):
        modules = [
            item
            for item in loaded
            if isinstance(
                item,
                torch.nn.Module
            )
        ]

        if len(
            modules
        ) == 1:
            return modules[
                0
            ]

    if isinstance(
        loaded,
        dict
    ):
        modules = [
            value
            for value in loaded.values()
            if isinstance(
                value,
                torch.nn.Module
            )
        ]

        if len(
            modules
        ) == 1:
            return modules[
                0
            ]

    raise RuntimeError(
        "Could not uniquely extract torch.nn.Module "
        f"from load_transformer return type {type(loaded)}"
    )


def resolve_input_paths():
    testreal = pd.read_csv(
        R2D_A3_RESOLVED
    )

    selected = pd.read_csv(
        R2D_A_SELECTED
    )

    metadata = pd.read_csv(
        R2D_A_METADATA
    )

    paths = {}

    # TestReal canonical NPZs from R2D-A3.
    for _, row in testreal.iterrows():
        task = str(
            row[
                "task"
            ]
        )

        paths[
            (
                "BIRDSAI_TestReal",
                task,
                "npz",
            )
        ] = Path(
            row[
                "selected_path"
            ]
        )

    # Fresh canonical NPZs from unambiguous R2D-A.
    fresh = selected[
        selected[
            "dataset"
        ]
        ==
        "Fresh_WildBox"
    ]

    for _, row in fresh.iterrows():
        task = str(
            row[
                "task"
            ]
        )

        paths[
            (
                "Fresh_WildBox",
                task,
                "npz",
            )
        ] = Path(
            row[
                "selected_path"
            ]
        )

    # Highest-scoring exact-count metadata candidate per dataset/task.
    for dataset, task in EXPECTED.keys():
        subset = metadata[
            (
                metadata[
                    "dataset"
                ]
                ==
                dataset
            )
            &
            (
                metadata[
                    "task"
                ]
                ==
                task
            )
            &
            (
                metadata[
                    "rows"
                ]
                ==
                EXPECTED[
                    (
                        dataset,
                        task,
                    )
                ][
                    "samples"
                ]
            )
        ].copy()

        if len(
            subset
        ) == 0:
            raise RuntimeError(
                f"No exact-count metadata candidate for {dataset} {task}"
            )

        subset = subset.sort_values(
            [
                "score",
                "path",
            ],
            ascending=[
                False,
                True,
            ],
        )

        paths[
            (
                dataset,
                task,
                "metadata",
            )
        ] = Path(
            subset.iloc[
                0
            ][
                "path"
            ]
        )

    for p in paths.values():
        require(
            p
        )

    return paths


def find_array_key(
    z,
    exact_names,
):
    lookup = {
        str(
            key
        ).lower():
            key
        for key in z.files
    }

    for name in exact_names:
        if name.lower() in lookup:
            return lookup[
                name.lower()
            ]

    return None


def load_canonical_dataset(
    npz_path,
    metadata_path,
    expected_samples,
    future_len,
):
    z = np.load(
        npz_path,
        allow_pickle=False,
    )

    meta = pd.read_csv(
        metadata_path
    )

    if len(
        meta
    ) != expected_samples:
        raise RuntimeError(
            f"Metadata count mismatch: {len(meta)} != {expected_samples}"
        )

    history_key = find_array_key(
        z,
        [
            "history_canonical",
            "histories_canonical",
            "X",
            "history",
        ],
    )

    future_key = find_array_key(
        z,
        [
            "future_canonical",
            "futures_canonical",
            "Y",
            "future",
        ],
    )

    origin_key = find_array_key(
        z,
        [
            "origin",
            "origins",
        ],
    )

    angle_key = find_array_key(
        z,
        [
            "angle",
            "angles",
        ],
    )

    scale_key = find_array_key(
        z,
        [
            "scale",
            "scales",
        ],
    )

    required = {
        "history":
            history_key,

        "future":
            future_key,

        "origin":
            origin_key,

        "angle":
            angle_key,

        "scale":
            scale_key,
    }

    missing = [
        name
        for name, key in required.items()
        if key is None
    ]

    if missing:
        raise RuntimeError(
            f"Could not identify canonical arrays {missing} in {npz_path}. "
            f"Available keys={z.files}"
        )

    history = np.asarray(
        z[
            history_key
        ],
        dtype=np.float32,
    )

    future = np.asarray(
        z[
            future_key
        ],
        dtype=np.float32,
    )

    origin = np.asarray(
        z[
            origin_key
        ],
        dtype=np.float32,
    )

    angle = np.asarray(
        z[
            angle_key
        ],
        dtype=np.float32,
    ).reshape(
        -1
    )

    scale = np.asarray(
        z[
            scale_key
        ],
        dtype=np.float32,
    ).reshape(
        -1
    )

    expected_history_shape = (
        expected_samples,
        20,
        2,
    )

    expected_future_shape = (
        expected_samples,
        future_len,
        2,
    )

    if history.shape != expected_history_shape:
        raise RuntimeError(
            f"History shape mismatch in {npz_path}: "
            f"{history.shape} != {expected_history_shape}"
        )

    if future.shape != expected_future_shape:
        raise RuntimeError(
            f"Future shape mismatch in {npz_path}: "
            f"{future.shape} != {expected_future_shape}"
        )

    if origin.shape != (
        expected_samples,
        2,
    ):
        raise RuntimeError(
            f"Origin shape mismatch: {origin.shape}"
        )

    if angle.shape != (
        expected_samples,
    ):
        raise RuntimeError(
            f"Angle shape mismatch: {angle.shape}"
        )

    if scale.shape != (
        expected_samples,
    ):
        raise RuntimeError(
            f"Scale shape mismatch: {scale.shape}"
        )

    return {
        "history_canonical":
            history,

        "future_canonical":
            future,

        "origin":
            origin,

        "angle":
            angle,

        "scale":
            scale,

        "metadata":
            meta,

        "keys": {
            "history":
                history_key,

            "future":
                future_key,

            "origin":
                origin_key,

            "angle":
                angle_key,

            "scale":
                scale_key,
        },
    }


def prediction_to_numpy(
    prediction,
):
    if isinstance(
        prediction,
        np.ndarray
    ):
        return prediction

    try:
        import torch

        if torch.is_tensor(
            prediction
        ):
            return (
                prediction
                .detach()
                .cpu()
                .numpy()
            )

    except Exception:
        pass

    return np.asarray(
        prediction
    )


def metric_rows(
    dataset,
    task,
    model,
    prediction,
    truth,
    metadata,
):
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    if prediction.shape != truth.shape:
        raise RuntimeError(
            f"{dataset} {task} {model}: prediction/truth shape mismatch "
            f"{prediction.shape} vs {truth.shape}"
        )

    err = np.linalg.norm(
        prediction
        -
        truth,
        axis=2,
    )

    ade = err.mean(
        axis=1
    )

    fde = err[
        :,
        -1
    ]

    track_col = next(
        (
            c
            for c in [
                "track_id",
                "tracklet_id",
                "track",
                "derived_track_id",
            ]
            if c in metadata.columns
        ),
        None,
    )

    source_col = next(
        (
            c
            for c in [
                "source",
                "source_id",
                "video",
                "video_id",
                "sequence",
                "sequence_id",
            ]
            if c in metadata.columns
        ),
        None,
    )

    sample_col = next(
        (
            c
            for c in [
                "sample_index",
                "sample_id",
                "window_index",
                "window_id",
            ]
            if c in metadata.columns
        ),
        None,
    )

    rows = []

    for i in range(
        len(
            ade
        )
    ):
        rows.append({
            "dataset":
                dataset,

            "task":
                task,

            "model":
                model,

            "sample_id":
                (
                    str(
                        metadata.iloc[
                            i
                        ][
                            sample_col
                        ]
                    )
                    if sample_col is not None
                    else str(
                        i
                    )
                ),

            "track_id":
                (
                    str(
                        metadata.iloc[
                            i
                        ][
                            track_col
                        ]
                    )
                    if track_col is not None
                    else None
                ),

            "source_id":
                (
                    str(
                        metadata.iloc[
                            i
                        ][
                            source_col
                        ]
                    )
                    if source_col is not None
                    else None
                ),

            "ade":
                float(
                    ade[
                        i
                    ]
                ),

            "fde":
                float(
                    fde[
                        i
                    ]
                ),
        })

    return pd.DataFrame(
        rows
    )


def aggregate_summary(
    sample_df,
):
    rows = []

    for (
        dataset,
        task,
        model,
    ), group in sample_df.groupby(
        [
            "dataset",
            "task",
            "model",
        ],
        sort=False,
    ):
        rows.append({
            "dataset":
                dataset,

            "task":
                task,

            "model":
                model,

            "samples":
                int(
                    len(
                        group
                    )
                ),

            "ADE":
                float(
                    group[
                        "ade"
                    ].mean()
                ),

            "FDE":
                float(
                    group[
                        "fde"
                    ].mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


def sanity_check_reference(
    summary_df,
):
    rows = []

    for _, row in summary_df.iterrows():
        key = (
            str(
                row[
                    "dataset"
                ]
            ),
            str(
                row[
                    "task"
                ]
            ),
        )

        model = str(
            row[
                "model"
            ]
        )

        reference = (
            EXPECTED[
                key
            ][
                "reference"
            ].get(
                model
            )
        )

        if reference is None:
            rows.append({
                "dataset":
                    key[
                        0
                    ],

                "task":
                    key[
                        1
                    ],

                "model":
                    model,

                "reference_available":
                    False,

                "computed_ADE":
                    float(
                        row[
                            "ADE"
                        ]
                    ),

                "computed_FDE":
                    float(
                        row[
                            "FDE"
                        ]
                    ),

                "reference_ADE":
                    np.nan,

                "reference_FDE":
                    np.nan,

                "abs_ADE_diff":
                    np.nan,

                "abs_FDE_diff":
                    np.nan,

                "pass":
                    True,
            })

            continue

        ade_diff = abs(
            float(
                row[
                    "ADE"
                ]
            )
            -
            float(
                reference[
                    "ADE"
                ]
            )
        )

        fde_diff = abs(
            float(
                row[
                    "FDE"
                ]
            )
            -
            float(
                reference[
                    "FDE"
                ]
            )
        )

        # Published/summary values were rounded to 6 decimals.
        passed = (
            ade_diff
            <=
            2.5e-6
            and
            fde_diff
            <=
            2.5e-6
        )

        rows.append({
            "dataset":
                key[
                    0
                ],

            "task":
                key[
                    1
                ],

            "model":
                model,

            "reference_available":
                True,

            "computed_ADE":
                float(
                    row[
                        "ADE"
                    ]
                ),

            "computed_FDE":
                float(
                    row[
                        "FDE"
                    ]
                ),

            "reference_ADE":
                float(
                    reference[
                        "ADE"
                    ]
                ),

            "reference_FDE":
                float(
                    reference[
                        "FDE"
                    ]
                ),

            "abs_ADE_diff":
                ade_diff,

            "abs_FDE_diff":
                fde_diff,

            "pass":
                bool(
                    passed
                ),
        })

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR2D-B1 — TARGETED SAMPLE-LEVEL RECOMPUTATION: "
        "STATIONARY + CV + FROZEN TRANSFORMER ENSEMBLE"
    )

    print(
        "Datasets: BIRDSAI TestReal and preregistered Fresh WildBox. "
        "NO retraining, NO model selection, NO Kalman/GRU/LSTM recomputation in this step.\n"
    )

    verify_prerequisites()

    namespace, extraction_audit = safe_extract_namespace(
        EVALUATOR,
        [
            "load_transformer",
            "predict_model",
            "inverse_transform_batch",
            "stationary_prediction",
            "constant_velocity_prediction",
        ],
    )

    import torch

    paths = resolve_input_paths()

    all_sample_rows = []

    input_audit_rows = []

    for dataset, task in EXPECTED.keys():
        cfg = EXPECTED[
            (
                dataset,
                task,
            )
        ]

        npz_path = paths[
            (
                dataset,
                task,
                "npz",
            )
        ]

        metadata_path = paths[
            (
                dataset,
                task,
                "metadata",
            )
        ]

        data = load_canonical_dataset(
            npz_path,
            metadata_path,
            cfg[
                "samples"
            ],
            cfg[
                "future_len"
            ],
        )

        input_audit_rows.append({
            "dataset":
                dataset,

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
                    metadata_path
                ),

            "metadata_sha256":
                sha256_file(
                    metadata_path
                ),

            "npz_keys":
                json.dumps(
                    data[
                        "keys"
                    ]
                ),
        })

        history_abs = prediction_to_numpy(
            namespace[
                "inverse_transform_batch"
            ](
                data[
                    "history_canonical"
                ],
                data[
                    "origin"
                ],
                data[
                    "angle"
                ],
                data[
                    "scale"
                ],
            )
        )

        truth_abs = prediction_to_numpy(
            namespace[
                "inverse_transform_batch"
            ](
                data[
                    "future_canonical"
                ],
                data[
                    "origin"
                ],
                data[
                    "angle"
                ],
                data[
                    "scale"
                ],
            )
        )

        stationary = prediction_to_numpy(
            namespace[
                "stationary_prediction"
            ](
                history_abs,
                cfg[
                    "future_len"
                ],
            )
        )

        cv = prediction_to_numpy(
            namespace[
                "constant_velocity_prediction"
            ](
                history_abs,
                cfg[
                    "future_len"
                ],
            )
        )

        all_sample_rows.append(
            metric_rows(
                dataset,
                task,
                "Stationary",
                stationary,
                truth_abs,
                data[
                    "metadata"
                ],
            )
        )

        all_sample_rows.append(
            metric_rows(
                dataset,
                task,
                "ConstantVelocity",
                cv,
                truth_abs,
                data[
                    "metadata"
                ],
            )
        )

        seed_predictions_canonical = []

        for seed in SEEDS:
            loaded = namespace[
                "load_transformer"
            ](
                task,
                seed,
                cfg[
                    "future_len"
                ],
            )

            model = extract_torch_module(
                loaded,
                torch,
            )

            model.eval()

            pred_can = prediction_to_numpy(
                namespace[
                    "predict_model"
                ](
                    model,
                    data[
                        "history_canonical"
                    ],
                )
            )

            expected_shape = (
                cfg[
                    "samples"
                ],
                cfg[
                    "future_len"
                ],
                2,
            )

            if pred_can.shape != expected_shape:
                raise RuntimeError(
                    f"{dataset} {task} seed{seed}: Transformer prediction "
                    f"shape {pred_can.shape} != {expected_shape}"
                )

            seed_predictions_canonical.append(
                pred_can.astype(
                    np.float32,
                    copy=False,
                )
            )

        ensemble_can = np.mean(
            np.stack(
                seed_predictions_canonical,
                axis=0,
            ),
            axis=0,
        ).astype(
            np.float32
        )

        ensemble_abs = prediction_to_numpy(
            namespace[
                "inverse_transform_batch"
            ](
                ensemble_can,
                data[
                    "origin"
                ],
                data[
                    "angle"
                ],
                data[
                    "scale"
                ],
            )
        )

        all_sample_rows.append(
            metric_rows(
                dataset,
                task,
                "TransformerEnsemble",
                ensemble_abs,
                truth_abs,
                data[
                    "metadata"
                ],
            )
        )

        print(
            f"COMPLETE: {dataset} {task} "
            f"({cfg['samples']} samples)"
        )

    sample_df = pd.concat(
        all_sample_rows,
        ignore_index=True,
    )

    summary_df = aggregate_summary(
        sample_df
    )

    sanity_df = sanity_check_reference(
        summary_df
    )

    failed_known_reference = sanity_df[
        (
            sanity_df[
                "reference_available"
            ]
            ==
            True
        )
        &
        (
            sanity_df[
                "pass"
            ]
            ==
            False
        )
    ]

    sample_path = (
        OUT
        / "R2D_B1_sample_metrics.csv"
    )

    summary_path = (
        OUT
        / "R2D_B1_summary_metrics.csv"
    )

    sanity_path = (
        OUT
        / "R2D_B1_reference_sanity_check.csv"
    )

    input_path = (
        OUT
        / "R2D_B1_input_provenance.csv"
    )

    sample_df.to_csv(
        sample_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    sanity_df.to_csv(
        sanity_path,
        index=False,
    )

    pd.DataFrame(
        input_audit_rows
    ).to_csv(
        input_path,
        index=False,
    )

    status = (
        "R2D_B1_TARGETED_RECOMPUTATION_COMPLETE_REFERENCE_SANITY_PASS"
        if len(
            failed_known_reference
        )
        ==
        0
        else
        "R2D_B1_RECOMPUTATION_COMPLETE_REFERENCE_SANITY_MISMATCH_REVIEW_REQUIRED"
    )

    manifest = {
        "step":
            "R2D_B1_TARGETED_SAMPLE_LEVEL_RECOMPUTATION",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "models":
            [
                "Stationary",
                "ConstantVelocity",
                "TransformerEnsemble",
            ],

        "datasets":
            [
                "BIRDSAI_TestReal",
                "Fresh_WildBox",
            ],

        "retraining":
            False,

        "model_selection":
            False,

        "fresh_kalman_gru_lstm_recomputed":
            False,

        "evaluator_source":
            str(
                EVALUATOR
            ),

        "evaluator_sha256":
            sha256_file(
                EVALUATOR
            ),

        "safe_extraction":
            extraction_audit,

        "sample_metrics_csv":
            str(
                sample_path
            ),

        "sample_metrics_sha256":
            sha256_file(
                sample_path
            ),

        "summary_csv":
            str(
                summary_path
            ),

        "reference_sanity_csv":
            str(
                sanity_path
            ),

        "input_provenance_csv":
            str(
                input_path
            ),
    }

    manifest_path = (
        OUT
        / "R2D_B1_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSUMMARY METRICS:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nREFERENCE SANITY CHECK:"
    )

    print(
        sanity_df.to_string(
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
        "\nR2D-B1 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
