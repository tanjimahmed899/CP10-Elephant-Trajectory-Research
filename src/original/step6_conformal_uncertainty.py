from pathlib import Path
import ast
import math
import types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# STEP 6: SPLIT-CONFORMAL UNCERTAINTY FOR TRAJECTORY FORECASTS
#
# Calibration:
#   BIRDSAI TrainReal validation split ONLY
#
# Locked evaluation:
#   1) BIRDSAI TestReal
#   2) Fresh WildBox holdout
#
# Primary conformal score:
#   max future Euclidean error within each window
#
# This yields a constant-radius simultaneous trajectory band:
#   ||y_t - yhat_t|| <= q_alpha for ALL future steps t
#
# Secondary:
#   step-wise marginal conformal radii q_alpha(t)
#
# IMPORTANT:
#   Windows overlap within tracks. Therefore textbook i.i.d.
#   exchangeability is imperfect. We report empirical coverage
#   plus track-cluster bootstrap confidence intervals and do NOT
#   claim unconditional real-world coverage guarantees.
#
# Existing Main Result files are READ ONLY.
# All NEW outputs go to Downloads.
# ============================================================

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

NEURAL_DIR = (
    ROOT /
    "final_neural_trajectory_dataset"
)

FRESH_DIR = (
    ROOT /
    "wildbox_fresh_confirmatory_results"
)

TRANSFORMER_DIR = (
    ROOT /
    "transformer_results"
)

ORIGINAL_EVAL = (
    ROOT /
    "evaluate_fresh_wildbox_holdout.py"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step6_conformal_uncertainty"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device("cpu")

SEEDS = [
    42,
    52,
    62
]

ALPHAS = [
    0.10,   # 90%
    0.05    # 95%
]

N_BOOT = 10000

RNG_SEED = 20261026

SELF_CHECK_TOL = 5e-6

TASKS = {

    "2s_to_2s": {
        "future_length": 20,
        "expected_val": 97,
        "expected_test": 1053,
        "expected_fresh": 865
    },

    "2s_to_5s": {
        "future_length": 50,
        "expected_val": 88,
        "expected_test": 826,
        "expected_fresh": 637
    }
}


# ============================================================
# EXTRACT ORIGINAL TRANSFORMER DEFINITIONS WITHOUT EXECUTING
# TOP-LEVEL PATH / HOLDOUT CHECKS
# ============================================================

def load_original_definitions():

    source = ORIGINAL_EVAL.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(
            ORIGINAL_EVAL
        )
    )

    required = {
        "PositionalEncoding",
        "TrajectoryTransformer",
        "inverse_transform_batch"
    }

    selected = []

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef
            )
        ):

            if node.name in required:

                selected.append(
                    node
                )

    found = {
        node.name
        for node in selected
    }

    missing = (
        required
        -
        found
    )

    if missing:

        raise RuntimeError(
            "Missing original definitions: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    selected.sort(
        key=lambda n: n.lineno
    )

    module_ast = ast.Module(
        body=selected,
        type_ignores=[]
    )

    ast.fix_missing_locations(
        module_ast
    )

    ns = {
        "__name__":
            "fresh_eval_original_extracted",

        "np":
            np,

        "pd":
            pd,

        "torch":
            torch,

        "nn":
            nn,

        "math":
            math,

        "Path":
            Path
    }

    exec(
        compile(
            module_ast,
            filename=str(
                ORIGINAL_EVAL
            ),
            mode="exec"
        ),
        ns,
        ns
    )

    return types.SimpleNamespace(

        PositionalEncoding=
            ns[
                "PositionalEncoding"
            ],

        TrajectoryTransformer=
            ns[
                "TrajectoryTransformer"
            ],

        inverse_transform_batch=
            ns[
                "inverse_transform_batch"
            ]
    )


ORIG = load_original_definitions()


# ============================================================
# MODEL LOADING / PREDICTION
# ============================================================

def load_model(
    task,
    seed
):

    checkpoint_path = (

        TRANSFORMER_DIR

        /

        f"{task}_Transformer_seed{seed}.pt"
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=
            DEVICE
    )

    model = ORIG.TrajectoryTransformer(

        TASKS[
            task
        ][
            "future_length"
        ],

        d_model=
            int(
                checkpoint[
                    "d_model"
                ]
            ),

        nhead=
            int(
                checkpoint[
                    "nhead"
                ]
            ),

        num_layers=
            int(
                checkpoint[
                    "num_layers"
                ]
            ),

        dim_feedforward=
            int(
                checkpoint[
                    "dim_feedforward"
                ]
            ),

        dropout=
            float(
                checkpoint[
                    "dropout"
                ]
            )
    ).to(
        DEVICE
    )

    model.load_state_dict(

        checkpoint[
            "state_dict"
        ],

        strict=True
    )

    model.eval()

    return model


def predict_model(
    model,
    history,
    batch_size=256
):

    outputs = []

    with torch.no_grad():

        for start in range(

            0,

            len(
                history
            ),

            batch_size
        ):

            x = torch.tensor(

                history[
                    start:
                    start
                    +
                    batch_size
                ],

                dtype=torch.float32,

                device=DEVICE
            )

            y = model(
                x
            )

            outputs.append(

                y.detach()
                .cpu()
                .numpy()
            )

    return np.concatenate(
        outputs,
        axis=0
    )


# ============================================================
# ORIGINAL INVERSE TRANSFORM
# ============================================================

def inverse_transform(
    canonical_prediction,
    origin,
    angle,
    scale
):

    out = ORIG.inverse_transform_batch(

        canonical_prediction,

        origin,

        angle,

        scale
    )

    if isinstance(
        out,
        torch.Tensor
    ):

        out = (

            out.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        out,
        dtype=np.float64
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_validation(
    task
):

    npz_path = (

        NEURAL_DIR

        /

        f"{task}_birdsai_trainval_canonical.npz"
    )

    metadata_path = (

        NEURAL_DIR

        /

        f"{task}_birdsai_trainval_metadata.csv"
    )

    arrays = np.load(
        npz_path
    )

    metadata = pd.read_csv(

        metadata_path,

        dtype={
            "source_group":
                str,

            "track_key":
                str
        }
    )

    indices = np.asarray(

        arrays[
            "validation_indices"
        ],

        dtype=int
    )

    if len(
        indices
    ) != TASKS[
        task
    ][
        "expected_val"
    ]:

        raise RuntimeError(
            f"{task}: unexpected validation count."
        )

    subset = {

        "history_canonical":
            np.asarray(
                arrays[
                    "history_canonical"
                ][
                    indices
                ],
                dtype=np.float32
            ),

        "future_abs":
            np.asarray(
                arrays[
                    "future_abs"
                ][
                    indices
                ],
                dtype=np.float64
            ),

        "origin":
            np.asarray(
                arrays[
                    "origin"
                ][
                    indices
                ],
                dtype=np.float64
            ),

        "heading_angle":
            np.asarray(
                arrays[
                    "heading_angle"
                ][
                    indices
                ],
                dtype=np.float64
            ),

        "scale":
            np.asarray(
                arrays[
                    "scale"
                ][
                    indices
                ],
                dtype=np.float64
            )
    }

    meta = (

        metadata
        .iloc[
            indices
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    # Ensure sample_id exists and is unique within this view.
    if "sample_id" not in meta.columns:

        meta[
            "sample_id"
        ] = indices

    return subset, meta


def load_testreal(
    task
):

    npz_path = (

        NEURAL_DIR

        /

        f"{task}_birdsai_test_canonical.npz"
    )

    metadata_path = (

        NEURAL_DIR

        /

        f"{task}_birdsai_test_metadata.csv"
    )

    arrays = np.load(
        npz_path
    )

    metadata = pd.read_csv(

        metadata_path,

        dtype={
            "source_group":
                str,

            "track_key":
                str
        }
    )

    if len(
        metadata
    ) != TASKS[
        task
    ][
        "expected_test"
    ]:

        raise RuntimeError(
            f"{task}: unexpected TestReal count."
        )

    return {

        "history_canonical":
            np.asarray(
                arrays[
                    "history_canonical"
                ],
                dtype=np.float32
            ),

        "future_abs":
            np.asarray(
                arrays[
                    "future_abs"
                ],
                dtype=np.float64
            ),

        "origin":
            np.asarray(
                arrays[
                    "origin"
                ],
                dtype=np.float64
            ),

        "heading_angle":
            np.asarray(
                arrays[
                    "heading_angle"
                ],
                dtype=np.float64
            ),

        "scale":
            np.asarray(
                arrays[
                    "scale"
                ],
                dtype=np.float64
            )

    }, metadata


def load_fresh(
    task
):

    npz_path = (

        FRESH_DIR

        /

        f"{task}_fresh_dataset.npz"
    )

    metadata_path = (

        FRESH_DIR

        /

        f"{task}_fresh_metadata.csv"
    )

    arrays = np.load(
        npz_path
    )

    metadata = pd.read_csv(

        metadata_path,

        dtype={
            "source_group":
                str,

            "track_key":
                str
        }
    )

    if len(
        metadata
    ) != TASKS[
        task
    ][
        "expected_fresh"
    ]:

        raise RuntimeError(
            f"{task}: unexpected Fresh WildBox count."
        )

    return {

        "history_canonical":
            np.asarray(
                arrays[
                    "history_canonical"
                ],
                dtype=np.float32
            ),

        "future_abs":
            np.asarray(
                arrays[
                    "future_abs"
                ],
                dtype=np.float64
            ),

        "origin":
            np.asarray(
                arrays[
                    "origin"
                ],
                dtype=np.float64
            ),

        "heading_angle":
            np.asarray(
                arrays[
                    "heading_angle"
                ],
                dtype=np.float64
            ),

        "scale":
            np.asarray(
                arrays[
                    "scale"
                ],
                dtype=np.float64
            )

    }, metadata


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(
    task,
    dataset
):

    seed_predictions = []

    for seed in SEEDS:

        model = load_model(
            task,
            seed
        )

        canonical = predict_model(

            model,

            dataset[
                "history_canonical"
            ]
        )

        absolute = inverse_transform(

            canonical,

            dataset[
                "origin"
            ],

            dataset[
                "heading_angle"
            ],

            dataset[
                "scale"
            ]
        )

        seed_predictions.append(
            absolute
        )

    stacked = np.stack(

        seed_predictions,

        axis=0
    )

    mean_prediction = np.mean(

        stacked,

        axis=0
    )

    disagreement = np.mean(

        np.linalg.norm(

            stacked

            -

            mean_prediction[
                None,
                ...
            ],

            axis=3
        ),

        axis=0
    )

    return (
        stacked,
        mean_prediction,
        disagreement
    )


# ============================================================
# SELF-CHECK ON FRESH HOLDOUT
# ============================================================

def self_check_fresh(
    task,
    seed_predictions,
    mean_prediction,
    truth
):

    saved = pd.read_csv(

        FRESH_DIR

        /

        "fresh_holdout_model_metrics.csv"
    )

    saved = saved[

        saved[
            "task"
        ]

        ==

        task
    ]

    rows = []

    for seed_index, seed in enumerate(
        SEEDS
    ):

        errors = np.linalg.norm(

            seed_predictions[
                seed_index
            ]

            -

            truth,

            axis=2
        )

        ade = float(
            errors.mean(
                axis=1
            ).mean()
        )

        fde = float(
            errors[
                :,
                -1
            ].mean()
        )

        expected = saved[

            saved[
                "model"
            ]

            ==

            f"Transformer_seed{seed}"
        ]

        if len(
            expected
        ) != 1:

            raise RuntimeError(
                "Missing Fresh saved seed metric."
            )

        expected = expected.iloc[
            0
        ]

        ade_diff = abs(

            ade

            -

            float(
                expected[
                    "ADE"
                ]
            )
        )

        fde_diff = abs(

            fde

            -

            float(
                expected[
                    "FDE"
                ]
            )
        )

        rows.append({

            "task":
                task,

            "model":
                f"Transformer_seed{seed}",

            "ADE_diff":
                ade_diff,

            "FDE_diff":
                fde_diff
        })

        if (
            ade_diff > SELF_CHECK_TOL

            or

            fde_diff > SELF_CHECK_TOL
        ):

            raise RuntimeError(
                "Fresh self-check failed."
            )

    errors = np.linalg.norm(

        mean_prediction

        -

        truth,

        axis=2
    )

    ade = float(
        errors.mean(
            axis=1
        ).mean()
    )

    fde = float(
        errors[
            :,
            -1
        ].mean()
    )

    expected = saved[

        saved[
            "model"
        ]

        ==

        "Transformer_3Seed_Ensemble"
    ]

    if len(
        expected
    ) != 1:

        raise RuntimeError(
            "Missing Fresh ensemble metric."
        )

    expected = expected.iloc[
        0
    ]

    ade_diff = abs(

        ade

        -

        float(
            expected[
                "ADE"
            ]
        )
    )

    fde_diff = abs(

        fde

        -

        float(
            expected[
                "FDE"
            ]
        )
    )

    rows.append({

        "task":
            task,

        "model":
            "Transformer_3Seed_Ensemble",

        "ADE_diff":
            ade_diff,

        "FDE_diff":
            fde_diff
    })

    if (
        ade_diff > SELF_CHECK_TOL

        or

        fde_diff > SELF_CHECK_TOL
    ):

        raise RuntimeError(
            "Fresh ensemble self-check failed."
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FINITE-SAMPLE CONFORMAL QUANTILE
# ============================================================

def conformal_quantile(
    scores,
    alpha
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    scores = scores[
        np.isfinite(
            scores
        )
    ]

    n = len(
        scores
    )

    if n == 0:

        raise RuntimeError(
            "No finite calibration scores."
        )

    # Standard split-conformal finite-sample order statistic:
    #
    # k = ceil((n + 1) * (1 - alpha))
    #
    # q = kth smallest score, clipped to n.

    k = int(
        np.ceil(
            (
                n + 1
            )
            *
            (
                1.0
                -
                alpha
            )
        )
    )

    k = min(
        max(
            k,
            1
        ),
        n
    )

    ordered = np.sort(
        scores
    )

    return (
        float(
            ordered[
                k - 1
            ]
        ),

        k,

        n
    )


# ============================================================
# TRACK-CLUSTER BOOTSTRAP COVERAGE CI
# ============================================================

def cluster_bootstrap_coverage(
    covered,
    track_keys,
    rng,
    n_boot=N_BOOT
):

    temp = pd.DataFrame({

        "track_key":
            np.asarray(
                track_keys,
                dtype=str
            ),

        "covered":
            np.asarray(
                covered,
                dtype=float
            )
    })

    g = (

        temp
        .groupby(
            "track_key"
        )[
            "covered"
        ]
        .agg(
            total="sum",
            count="count",
            mean="mean"
        )
        .reset_index()
    )

    totals = g[
        "total"
    ].to_numpy(
        dtype=float
    )

    counts = g[
        "count"
    ].to_numpy(
        dtype=float
    )

    means = g[
        "mean"
    ].to_numpy(
        dtype=float
    )

    n_tracks = len(
        g
    )

    boot_window = np.empty(
        n_boot,
        dtype=float
    )

    boot_track = np.empty(
        n_boot,
        dtype=float
    )

    for b in range(
        n_boot
    ):

        idx = rng.integers(

            0,

            n_tracks,

            size=n_tracks
        )

        boot_window[
            b
        ] = (

            totals[
                idx
            ].sum()

            /

            counts[
                idx
            ].sum()
        )

        boot_track[
            b
        ] = means[
            idx
        ].mean()

    return {

        "window_CI_low":
            float(
                np.quantile(
                    boot_window,
                    0.025
                )
            ),

        "window_CI_high":
            float(
                np.quantile(
                    boot_window,
                    0.975
                )
            ),

        "track_macro_coverage":
            float(
                means.mean()
            ),

        "track_macro_CI_low":
            float(
                np.quantile(
                    boot_track,
                    0.025
                )
            ),

        "track_macro_CI_high":
            float(
                np.quantile(
                    boot_track,
                    0.975
                )
            )
    }


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_task(
    task,
    validation,
    validation_metadata,
    mean_prediction
):

    truth = validation[
        "future_abs"
    ]

    errors = np.linalg.norm(

        mean_prediction

        -

        truth,

        axis=2
    )

    # Primary simultaneous score:
    # maximum future-point error per window.
    max_error = errors.max(
        axis=1
    )

    calibration_rows = []

    step_rows = []

    calibration_objects = {}


    for alpha in ALPHAS:

        target = (
            1.0
            -
            alpha
        )

        q_sim, k, n = conformal_quantile(

            max_error,

            alpha
        )

        # Step-wise marginal radii.
        q_steps = []

        for step in range(
            errors.shape[
                1
            ]
        ):

            q_step, k_step, n_step = conformal_quantile(

                errors[
                    :,
                    step
                ],

                alpha
            )

            q_steps.append(
                q_step
            )

            step_rows.append({

                "task":
                    task,

                "alpha":
                    alpha,

                "target_coverage":
                    target,

                "forecast_step":
                    step + 1,

                "forecast_time_sec":
                    (
                        step + 1
                    )
                    /
                    10.0,

                "stepwise_radius":
                    q_step,

                "order_statistic_k":
                    k_step,

                "calibration_n":
                    n_step
            })

        q_steps = np.asarray(
            q_steps,
            dtype=np.float64
        )

        calibration_objects[
            alpha
        ] = {

            "q_sim":
                q_sim,

            "q_steps":
                q_steps
        }

        simultaneous_covered = (

            max_error

            <=

            q_sim
        )

        pointwise_covered = (

            errors

            <=

            q_sim
        )

        stepwise_covered = (

            errors

            <=

            q_steps[
                None,
                :
            ]
        )

        calibration_rows.append({

            "task":
                task,

            "alpha":
                alpha,

            "target_coverage":
                target,

            "calibration_samples":
                len(
                    errors
                ),

            "calibration_tracks":
                validation_metadata[
                    "track_key"
                ].nunique(),

            "simultaneous_radius":
                q_sim,

            "order_statistic_k":
                k,

            "calibration_simultaneous_coverage":
                float(
                    simultaneous_covered.mean()
                ),

            "calibration_pointwise_coverage_constant_radius":
                float(
                    pointwise_covered.mean()
                ),

            "calibration_stepwise_marginal_coverage":
                float(
                    stepwise_covered.mean()
                ),

            "mean_stepwise_radius":
                float(
                    q_steps.mean()
                ),

            "endpoint_stepwise_radius":
                float(
                    q_steps[
                        -1
                    ]
                )
        })

    return (
        pd.DataFrame(
            calibration_rows
        ),

        pd.DataFrame(
            step_rows
        ),

        calibration_objects,

        errors
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_conformal(
    task,
    dataset_name,
    dataset,
    metadata,
    mean_prediction,
    disagreement,
    calibration_objects,
    rng
):

    truth = dataset[
        "future_abs"
    ]

    errors = np.linalg.norm(

        mean_prediction

        -

        truth,

        axis=2
    )

    max_error = errors.max(
        axis=1
    )

    summary_rows = []

    time_rows = []

    source_rows = []

    failure_rows = []


    for alpha in ALPHAS:

        target = (
            1.0
            -
            alpha
        )

        q_sim = calibration_objects[
            alpha
        ][
            "q_sim"
        ]

        q_steps = calibration_objects[
            alpha
        ][
            "q_steps"
        ]

        simultaneous_covered = (

            max_error

            <=

            q_sim
        )

        pointwise_covered = (

            errors

            <=

            q_sim
        )

        endpoint_covered = (

            errors[
                :,
                -1
            ]

            <=

            q_sim
        )

        stepwise_covered = (

            errors

            <=

            q_steps[
                None,
                :
            ]
        )

        cluster_ci = cluster_bootstrap_coverage(

            simultaneous_covered,

            metadata[
                "track_key"
            ],

            rng
        )

        summary_rows.append({

            "task":
                task,

            "dataset":
                dataset_name,

            "alpha":
                alpha,

            "target_coverage":
                target,

            "samples":
                len(
                    metadata
                ),

            "tracks":
                metadata[
                    "track_key"
                ].nunique(),

            "sources":
                metadata[
                    "source_group"
                ].nunique(),

            "simultaneous_radius":
                q_sim,

            "simultaneous_window_coverage":
                float(
                    simultaneous_covered.mean()
                ),

            "simultaneous_coverage_gap_pp":
                100.0
                *
                (
                    simultaneous_covered.mean()
                    -
                    target
                ),

            "simultaneous_window_CI_low":
                cluster_ci[
                    "window_CI_low"
                ],

            "simultaneous_window_CI_high":
                cluster_ci[
                    "window_CI_high"
                ],

            "simultaneous_track_macro_coverage":
                cluster_ci[
                    "track_macro_coverage"
                ],

            "simultaneous_track_macro_CI_low":
                cluster_ci[
                    "track_macro_CI_low"
                ],

            "simultaneous_track_macro_CI_high":
                cluster_ci[
                    "track_macro_CI_high"
                ],

            "pointwise_coverage_constant_radius":
                float(
                    pointwise_covered.mean()
                ),

            "endpoint_coverage_constant_radius":
                float(
                    endpoint_covered.mean()
                ),

            "stepwise_marginal_coverage":
                float(
                    stepwise_covered.mean()
                ),

            "mean_stepwise_radius":
                float(
                    q_steps.mean()
                ),

            "endpoint_stepwise_radius":
                float(
                    q_steps[
                        -1
                    ]
                ),

            "mean_ensemble_disagreement":
                float(
                    disagreement.mean()
                ),

            "mean_ADE":
                float(
                    errors.mean(
                        axis=1
                    ).mean()
                ),

            "mean_FDE":
                float(
                    errors[
                        :,
                        -1
                    ].mean()
                )
        })


        # ----------------------------------------------------
        # Time-wise coverage
        # ----------------------------------------------------

        for step in range(
            errors.shape[
                1
            ]
        ):

            time_rows.append({

                "task":
                    task,

                "dataset":
                    dataset_name,

                "alpha":
                    alpha,

                "target_coverage":
                    target,

                "forecast_step":
                    step + 1,

                "forecast_time_sec":
                    (
                        step + 1
                    )
                    /
                    10.0,

                "constant_radius":
                    q_sim,

                "stepwise_radius":
                    q_steps[
                        step
                    ],

                "constant_radius_coverage":
                    float(

                        (
                            errors[
                                :,
                                step
                            ]

                            <=

                            q_sim
                        ).mean()
                    ),

                "stepwise_radius_coverage":
                    float(

                        (
                            errors[
                                :,
                                step
                            ]

                            <=

                            q_steps[
                                step
                            ]
                        ).mean()
                    ),

                "mean_error":
                    float(
                        errors[
                            :,
                            step
                        ].mean()
                    ),

                "median_error":
                    float(
                        np.median(
                            errors[
                                :,
                                step
                            ]
                        )
                    ),

                "mean_disagreement":
                    float(
                        disagreement[
                            :,
                            step
                        ].mean()
                    )
            })


        # ----------------------------------------------------
        # Source-wise simultaneous coverage
        # ----------------------------------------------------

        temp = metadata[
            [
                "source_group",
                "track_key"
            ]
        ].copy()

        temp[
            "covered"
        ] = simultaneous_covered.astype(
            int
        )

        temp[
            "max_error"
        ] = max_error

        for source_group, group in temp.groupby(
            "source_group"
        ):

            source_rows.append({

                "task":
                    task,

                "dataset":
                    dataset_name,

                "alpha":
                    alpha,

                "target_coverage":
                    target,

                "source_group":
                    source_group,

                "samples":
                    len(
                        group
                    ),

                "tracks":
                    group[
                        "track_key"
                    ].nunique(),

                "simultaneous_coverage":
                    float(
                        group[
                            "covered"
                        ].mean()
                    ),

                "mean_max_error":
                    float(
                        group[
                            "max_error"
                        ].mean()
                    ),

                "simultaneous_radius":
                    q_sim
            })


        # ----------------------------------------------------
        # Failure cases
        # ----------------------------------------------------

        failures = np.where(
            ~simultaneous_covered
        )[0]

        if len(
            failures
        ) > 0:

            order = failures[
                np.argsort(
                    -max_error[
                        failures
                    ]
                )
            ]

            top = order[
                :min(
                    25,
                    len(
                        order
                    )
                )
            ]

            for idx in top:

                failure_rows.append({

                    "task":
                        task,

                    "dataset":
                        dataset_name,

                    "alpha":
                        alpha,

                    "target_coverage":
                        target,

                    "sample_id":
                        metadata.iloc[
                            idx
                        ].get(
                            "sample_id",
                            idx
                        ),

                    "source_group":
                        metadata.iloc[
                            idx
                        ][
                            "source_group"
                        ],

                    "track_key":
                        metadata.iloc[
                            idx
                        ][
                            "track_key"
                        ],

                    "max_error":
                        float(
                            max_error[
                                idx
                            ]
                        ),

                    "radius":
                        q_sim,

                    "excess_over_radius":
                        float(
                            max_error[
                                idx
                            ]
                            -
                            q_sim
                        ),

                    "ADE":
                        float(
                            errors[
                                idx
                            ].mean()
                        ),

                    "FDE":
                        float(
                            errors[
                                idx,
                                -1
                            ]
                        ),

                    "mean_disagreement":
                        float(
                            disagreement[
                                idx
                            ].mean()
                        )
                })

    return (

        pd.DataFrame(
            summary_rows
        ),

        pd.DataFrame(
            time_rows
        ),

        pd.DataFrame(
            source_rows
        ),

        pd.DataFrame(
            failure_rows
        ),

        errors
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rng = np.random.default_rng(
        RNG_SEED
    )

    all_calibration = []

    all_step_calibration = []

    all_summary = []

    all_time = []

    all_source = []

    all_failures = []

    all_self_checks = []

    all_calibration_scores = []


    for task in TASKS:

        print(
            "\n"
            +
            "=" * 78
        )

        print(
            "TASK:",
            task
        )

        print(
            "=" * 78
        )


        # ====================================================
        # VALIDATION CALIBRATION
        # ====================================================

        validation, val_metadata = load_validation(
            task
        )

        (
            _,
            val_mean,
            val_disagreement
        ) = ensemble_predict(

            task,

            validation
        )

        (
            calibration_df,
            step_calibration_df,
            calibration_objects,
            val_errors
        ) = calibrate_task(

            task,

            validation,

            val_metadata,

            val_mean
        )

        all_calibration.append(
            calibration_df
        )

        all_step_calibration.append(
            step_calibration_df
        )


        max_error = val_errors.max(
            axis=1
        )

        score_table = val_metadata[
            [
                "source_group",
                "track_key"
            ]
        ].copy()

        if "sample_id" in val_metadata.columns:

            score_table[
                "sample_id"
            ] = val_metadata[
                "sample_id"
            ].values

        else:

            score_table[
                "sample_id"
            ] = np.arange(
                len(
                    val_metadata
                )
            )

        score_table[
            "task"
        ] = task

        score_table[
            "max_future_error"
        ] = max_error

        score_table[
            "ADE"
        ] = val_errors.mean(
            axis=1
        )

        score_table[
            "FDE"
        ] = val_errors[
            :,
            -1
        ]

        score_table[
            "mean_disagreement"
        ] = val_disagreement.mean(
            axis=1
        )

        all_calibration_scores.append(
            score_table
        )


        print(
            "Calibration locked from validation only:"
        )

        print(
            calibration_df.to_string(
                index=False
            )
        )


        # ====================================================
        # BIRDSAI TESTREAL
        # ====================================================

        testreal, test_metadata = load_testreal(
            task
        )

        (
            _,
            test_mean,
            test_disagreement
        ) = ensemble_predict(

            task,

            testreal
        )

        (
            summary_df,
            time_df,
            source_df,
            failure_df,
            _
        ) = evaluate_conformal(

            task,

            "BIRDSAI_TESTREAL",

            testreal,

            test_metadata,

            test_mean,

            test_disagreement,

            calibration_objects,

            rng
        )

        all_summary.append(
            summary_df
        )

        all_time.append(
            time_df
        )

        all_source.append(
            source_df
        )

        if len(
            failure_df
        ) > 0:

            all_failures.append(
                failure_df
            )


        # ====================================================
        # FRESH WILDBOX
        # ====================================================

        fresh, fresh_metadata = load_fresh(
            task
        )

        (
            fresh_seed_predictions,
            fresh_mean,
            fresh_disagreement
        ) = ensemble_predict(

            task,

            fresh
        )


        check = self_check_fresh(

            task,

            fresh_seed_predictions,

            fresh_mean,

            fresh[
                "future_abs"
            ]
        )


        all_self_checks.append(
            check
        )


        (
            summary_df,
            time_df,
            source_df,
            failure_df,
            _
        ) = evaluate_conformal(

            task,

            "FRESH_WILDBOX",

            fresh,

            fresh_metadata,

            fresh_mean,

            fresh_disagreement,

            calibration_objects,

            rng
        )

        all_summary.append(
            summary_df
        )

        all_time.append(
            time_df
        )

        all_source.append(
            source_df
        )

        if len(
            failure_df
        ) > 0:

            all_failures.append(
                failure_df
            )


    # ========================================================
    # SAVE
    # ========================================================

    calibration = pd.concat(
        all_calibration,
        ignore_index=True
    )

    step_calibration = pd.concat(
        all_step_calibration,
        ignore_index=True
    )

    summary = pd.concat(
        all_summary,
        ignore_index=True
    )

    timewise = pd.concat(
        all_time,
        ignore_index=True
    )

    sourcewise = pd.concat(
        all_source,
        ignore_index=True
    )

    self_checks = pd.concat(
        all_self_checks,
        ignore_index=True
    )

    calibration_scores = pd.concat(
        all_calibration_scores,
        ignore_index=True
    )


    if all_failures:

        failures = pd.concat(
            all_failures,
            ignore_index=True
        )

    else:

        failures = pd.DataFrame()


    calibration.to_csv(

        OUT /
        "conformal_calibration_summary.csv",

        index=False
    )


    step_calibration.to_csv(

        OUT /
        "conformal_stepwise_radii.csv",

        index=False
    )


    summary.to_csv(

        OUT /
        "conformal_locked_evaluation_summary.csv",

        index=False
    )


    timewise.to_csv(

        OUT /
        "conformal_timewise_coverage.csv",

        index=False
    )


    sourcewise.to_csv(

        OUT /
        "conformal_sourcewise_coverage.csv",

        index=False
    )


    failures.to_csv(

        OUT /
        "conformal_failure_cases.csv",

        index=False
    )


    self_checks.to_csv(

        OUT /
        "fresh_transformer_self_check.csv",

        index=False
    )


    calibration_scores.to_csv(

        OUT /
        "validation_calibration_scores.csv",

        index=False
    )


    # ========================================================
    # TEXT SUMMARY
    # ========================================================

    with open(

        OUT /
        "SUMMARY.txt",

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "STEP 6: SPLIT-CONFORMAL TRAJECTORY UNCERTAINTY\n"
        )

        f.write(
            "=" * 78
            +
            "\n\n"
        )

        f.write(
            "Protocol:\n"
        )

        f.write(
            "- Calibration uses ONLY the fixed BIRDSAI "
            "TrainReal validation split.\n"
        )

        f.write(
            "- TestReal and Fresh WildBox are never used "
            "to choose conformal radii.\n"
        )

        f.write(
            "- Primary score = maximum Euclidean error "
            "over the entire future trajectory window.\n"
        )

        f.write(
            "- Primary radius therefore targets "
            "simultaneous trajectory coverage.\n"
        )

        f.write(
            "- Secondary step-wise radii target marginal "
            "coverage at each forecast time.\n"
        )

        f.write(
            "- 90% and 95% nominal coverage levels.\n"
        )

        f.write(
            "- Empirical simultaneous coverage is "
            "reported with track-cluster bootstrap 95% CIs.\n"
        )

        f.write(
            "- Because sliding windows overlap within "
            "tracks, strict i.i.d. exchangeability is "
            "imperfect; results must be described as "
            "empirical calibrated coverage under an "
            "exchangeability approximation, not an "
            "unconditional field guarantee.\n\n"
        )


        f.write(
            "FRESH TRANSFORMER SELF-CHECK\n"
        )

        f.write(
            "----------------------------\n"
        )

        f.write(
            self_checks.to_string(
                index=False
            )
        )


        f.write(
            "\n\nCALIBRATION SUMMARY\n"
        )

        f.write(
            "-------------------\n"
        )

        f.write(
            calibration.to_string(
                index=False
            )
        )


        f.write(
            "\n\nLOCKED EVALUATION SUMMARY\n"
        )

        f.write(
            "-------------------------\n"
        )

        f.write(
            summary.to_string(
                index=False
            )
        )


    # ========================================================
    # TERMINAL
    # ========================================================

    print(
        "\nSTEP 6 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


    print(
        "\nFRESH TRANSFORMER SELF-CHECK:\n"
    )

    print(
        self_checks.to_string(
            index=False
        )
    )


    print(
        "\nCONFORMAL CALIBRATION:\n"
    )

    print(
        calibration[
            [
                "task",
                "alpha",
                "target_coverage",
                "calibration_samples",
                "calibration_tracks",
                "simultaneous_radius",
                "calibration_simultaneous_coverage",
                "mean_stepwise_radius",
                "endpoint_stepwise_radius"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nLOCKED CONFORMAL EVALUATION:\n"
    )

    print(
        summary[
            [
                "task",
                "dataset",
                "alpha",
                "target_coverage",
                "samples",
                "tracks",
                "simultaneous_radius",
                "simultaneous_window_coverage",
                "simultaneous_coverage_gap_pp",
                "simultaneous_window_CI_low",
                "simultaneous_window_CI_high",
                "simultaneous_track_macro_coverage",
                "stepwise_marginal_coverage",
                "endpoint_stepwise_radius",
                "mean_ADE",
                "mean_FDE"
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()
