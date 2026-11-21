from pathlib import Path
import ast
import math
import types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# STEP 6B: SCALE-ADAPTIVE SPLIT-CONFORMAL UNCERTAINTY
#
# Calibration:
#   BIRDSAI TrainReal validation split ONLY
#
# Locked evaluation:
#   1) BIRDSAI TestReal
#   2) Fresh WildBox holdout
#
# Primary adaptive score:
#   maximum future Euclidean error in canonical normalized space
#
# For test sample i:
#   absolute simultaneous radius_i = q_alpha * scale_i
#
# Because canonical -> absolute is rotation + multiplication
# by scale, Euclidean distance is preserved up to that scale.
#
# Secondary:
#   step-wise normalized conformal radii q_alpha(t) * scale_i
#
# IMPORTANT:
#   Sliding windows overlap within tracks. Therefore strict i.i.d.
#   exchangeability is imperfect. Report empirical calibrated
#   coverage + track-cluster bootstrap CIs, not an unconditional
#   real-world coverage guarantee.
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

STEP6A_DIR = Path(
    r"C:\Users\HP\Downloads\step6_conformal_uncertainty"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step6b_scale_adaptive_conformal"
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
    0.10,
    0.05
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
# LOAD DATA
# ============================================================

def validate_scales(
    scale,
    task,
    dataset_name
):

    scale = np.asarray(
        scale,
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(
            scale
        )
    ):

        raise RuntimeError(
            f"{task} {dataset_name}: non-finite scale."
        )

    if np.any(
        scale <= 0
    ):

        raise RuntimeError(
            f"{task} {dataset_name}: non-positive scale."
        )


def load_validation(
    task
):

    arrays = np.load(

        NEURAL_DIR

        /

        f"{task}_birdsai_trainval_canonical.npz"
    )

    metadata = pd.read_csv(

        NEURAL_DIR

        /

        f"{task}_birdsai_trainval_metadata.csv",

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

        "future_canonical":
            np.asarray(
                arrays[
                    "future_canonical"
                ][
                    indices
                ],
                dtype=np.float64
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

    validate_scales(
        subset[
            "scale"
        ],
        task,
        "VALIDATION"
    )

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

    if "sample_id" not in meta.columns:

        meta[
            "sample_id"
        ] = indices

    return subset, meta


def load_testreal(
    task
):

    arrays = np.load(

        NEURAL_DIR

        /

        f"{task}_birdsai_test_canonical.npz"
    )

    metadata = pd.read_csv(

        NEURAL_DIR

        /

        f"{task}_birdsai_test_metadata.csv",

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

    subset = {

        "history_canonical":
            np.asarray(
                arrays[
                    "history_canonical"
                ],
                dtype=np.float32
            ),

        "future_canonical":
            np.asarray(
                arrays[
                    "future_canonical"
                ],
                dtype=np.float64
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
    }

    validate_scales(
        subset[
            "scale"
        ],
        task,
        "BIRDSAI_TESTREAL"
    )

    return subset, metadata


def load_fresh(
    task
):

    arrays = np.load(

        FRESH_DIR

        /

        f"{task}_fresh_dataset.npz"
    )

    metadata = pd.read_csv(

        FRESH_DIR

        /

        f"{task}_fresh_metadata.csv",

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

    subset = {

        "history_canonical":
            np.asarray(
                arrays[
                    "history_canonical"
                ],
                dtype=np.float32
            ),

        "future_canonical":
            np.asarray(
                arrays[
                    "future_canonical"
                ],
                dtype=np.float64
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
    }

    validate_scales(
        subset[
            "scale"
        ],
        task,
        "FRESH_WILDBOX"
    )

    return subset, metadata


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(
    task,
    dataset
):

    canonical_by_seed = []

    absolute_by_seed = []

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

        canonical_by_seed.append(
            np.asarray(
                canonical,
                dtype=np.float64
            )
        )

        absolute_by_seed.append(
            absolute
        )

    stacked_canonical = np.stack(

        canonical_by_seed,

        axis=0
    )

    stacked_absolute = np.stack(

        absolute_by_seed,

        axis=0
    )

    mean_canonical = np.mean(

        stacked_canonical,

        axis=0
    )

    mean_absolute = np.mean(

        stacked_absolute,

        axis=0
    )

    # Diagnostic disagreement in absolute units.
    disagreement_abs = np.mean(

        np.linalg.norm(

            stacked_absolute

            -

            mean_absolute[
                None,
                ...
            ],

            axis=3
        ),

        axis=0
    )

    return (
        stacked_canonical,
        stacked_absolute,
        mean_canonical,
        mean_absolute,
        disagreement_abs
    )


# ============================================================
# FRESH SELF-CHECK
# ============================================================

def self_check_fresh(
    task,
    stacked_absolute,
    mean_absolute,
    truth_abs
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

    for seed_idx, seed in enumerate(
        SEEDS
    ):

        err = np.linalg.norm(

            stacked_absolute[
                seed_idx
            ]

            -

            truth_abs,

            axis=2
        )

        ade = float(
            err.mean(
                axis=1
            ).mean()
        )

        fde = float(
            err[
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
                "Missing saved Fresh seed metric."
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

    err = np.linalg.norm(

        mean_absolute

        -

        truth_abs,

        axis=2
    )

    ade = float(
        err.mean(
            axis=1
        ).mean()
    )

    fde = float(
        err[
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
            "Missing saved Fresh ensemble metric."
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
# TRACK-CLUSTER BOOTSTRAP
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

    grouped = (

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

    totals = grouped[
        "total"
    ].to_numpy(
        dtype=float
    )

    counts = grouped[
        "count"
    ].to_numpy(
        dtype=float
    )

    means = grouped[
        "mean"
    ].to_numpy(
        dtype=float
    )

    n_tracks = len(
        grouped
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
# SCALE-ADAPTIVE CALIBRATION
# ============================================================

def calibrate_task(
    task,
    validation,
    validation_metadata,
    mean_canonical,
    mean_absolute
):

    truth_canonical = validation[
        "future_canonical"
    ]

    truth_abs = validation[
        "future_abs"
    ]

    scale = validation[
        "scale"
    ]

    err_can = np.linalg.norm(

        mean_canonical

        -

        truth_canonical,

        axis=2
    )

    err_abs = np.linalg.norm(

        mean_absolute

        -

        truth_abs,

        axis=2
    )

    # Exact consistency check:
    # abs error should equal canonical error * scale
    reconstructed_abs_error = (

        err_can

        *

        scale[
            :,
            None
        ]
    )

    max_consistency_diff = float(
        np.max(
            np.abs(
                reconstructed_abs_error

                -

                err_abs
            )
        )
    )

    if max_consistency_diff > 1e-5:

        raise RuntimeError(

            "Canonical/absolute distance consistency "
            f"failed: max diff={max_consistency_diff}"
        )

    max_can = err_can.max(
        axis=1
    )

    calibration_rows = []

    step_rows = []

    objects = {}


    for alpha in ALPHAS:

        target = (
            1.0
            -
            alpha
        )

        q_sim_norm, k, n = conformal_quantile(

            max_can,

            alpha
        )

        q_steps_norm = []

        for step in range(
            err_can.shape[
                1
            ]
        ):

            q_step, k_step, n_step = conformal_quantile(

                err_can[
                    :,
                    step
                ],

                alpha
            )

            q_steps_norm.append(
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

                "normalized_stepwise_radius":
                    q_step,

                "order_statistic_k":
                    k_step,

                "calibration_n":
                    n_step
            })

        q_steps_norm = np.asarray(
            q_steps_norm,
            dtype=np.float64
        )

        radius_abs = (

            q_sim_norm

            *

            scale
        )

        covered_sim = (

            err_abs.max(
                axis=1
            )

            <=

            radius_abs
        )

        step_radius_abs = (

            scale[
                :,
                None
            ]

            *

            q_steps_norm[
                None,
                :
            ]
        )

        covered_step = (

            err_abs

            <=

            step_radius_abs
        )

        objects[
            alpha
        ] = {

            "q_sim_norm":
                q_sim_norm,

            "q_steps_norm":
                q_steps_norm
        }

        calibration_rows.append({

            "task":
                task,

            "alpha":
                alpha,

            "target_coverage":
                target,

            "calibration_samples":
                len(
                    validation_metadata
                ),

            "calibration_tracks":
                validation_metadata[
                    "track_key"
                ].nunique(),

            "normalized_simultaneous_quantile":
                q_sim_norm,

            "order_statistic_k":
                k,

            "calibration_simultaneous_coverage":
                float(
                    covered_sim.mean()
                ),

            "mean_absolute_simultaneous_radius":
                float(
                    radius_abs.mean()
                ),

            "median_absolute_simultaneous_radius":
                float(
                    np.median(
                        radius_abs
                    )
                ),

            "p95_absolute_simultaneous_radius":
                float(
                    np.quantile(
                        radius_abs,
                        0.95
                    )
                ),

            "mean_normalized_stepwise_radius":
                float(
                    q_steps_norm.mean()
                ),

            "endpoint_normalized_stepwise_radius":
                float(
                    q_steps_norm[
                        -1
                    ]
                ),

            "calibration_stepwise_marginal_coverage":
                float(
                    covered_step.mean()
                ),

            "max_abs_vs_scaled_canonical_error_diff":
                max_consistency_diff
        })

    return (
        pd.DataFrame(
            calibration_rows
        ),

        pd.DataFrame(
            step_rows
        ),

        objects,

        err_can,

        err_abs
    )


# ============================================================
# LOCKED EVALUATION
# ============================================================

def evaluate_task(
    task,
    dataset_name,
    dataset,
    metadata,
    mean_canonical,
    mean_absolute,
    disagreement_abs,
    calibration_objects,
    rng
):

    truth_canonical = dataset[
        "future_canonical"
    ]

    truth_abs = dataset[
        "future_abs"
    ]

    scale = dataset[
        "scale"
    ]

    err_can = np.linalg.norm(

        mean_canonical

        -

        truth_canonical,

        axis=2
    )

    err_abs = np.linalg.norm(

        mean_absolute

        -

        truth_abs,

        axis=2
    )

    reconstructed_abs_error = (

        err_can

        *

        scale[
            :,
            None
        ]
    )

    consistency_diff = float(
        np.max(
            np.abs(
                reconstructed_abs_error

                -

                err_abs
            )
        )
    )

    if consistency_diff > 1e-5:

        raise RuntimeError(

            f"{task} {dataset_name}: "
            "canonical/absolute distance mismatch "
            f"{consistency_diff}"
        )

    max_abs = err_abs.max(
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

        q_sim_norm = calibration_objects[
            alpha
        ][
            "q_sim_norm"
        ]

        q_steps_norm = calibration_objects[
            alpha
        ][
            "q_steps_norm"
        ]

        radius_abs = (

            q_sim_norm

            *

            scale
        )

        step_radius_abs = (

            scale[
                :,
                None
            ]

            *

            q_steps_norm[
                None,
                :
            ]
        )

        covered_sim = (

            max_abs

            <=

            radius_abs
        )

        covered_step = (

            err_abs

            <=

            step_radius_abs
        )

        endpoint_covered = (

            err_abs[
                :,
                -1
            ]

            <=

            step_radius_abs[
                :,
                -1
            ]
        )

        cluster_ci = cluster_bootstrap_coverage(

            covered_sim,

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

            "normalized_simultaneous_quantile":
                q_sim_norm,

            "mean_absolute_simultaneous_radius":
                float(
                    radius_abs.mean()
                ),

            "median_absolute_simultaneous_radius":
                float(
                    np.median(
                        radius_abs
                    )
                ),

            "p95_absolute_simultaneous_radius":
                float(
                    np.quantile(
                        radius_abs,
                        0.95
                    )
                ),

            "simultaneous_window_coverage":
                float(
                    covered_sim.mean()
                ),

            "simultaneous_coverage_gap_pp":
                100.0
                *
                (
                    covered_sim.mean()
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

            "stepwise_marginal_coverage":
                float(
                    covered_step.mean()
                ),

            "endpoint_stepwise_coverage":
                float(
                    endpoint_covered.mean()
                ),

            "mean_absolute_endpoint_radius":
                float(
                    step_radius_abs[
                        :,
                        -1
                    ].mean()
                ),

            "median_absolute_endpoint_radius":
                float(
                    np.median(
                        step_radius_abs[
                            :,
                            -1
                        ]
                    )
                ),

            "mean_ADE":
                float(
                    err_abs.mean(
                        axis=1
                    ).mean()
                ),

            "mean_FDE":
                float(
                    err_abs[
                        :,
                        -1
                    ].mean()
                ),

            "mean_ensemble_disagreement":
                float(
                    disagreement_abs.mean()
                ),

            "max_abs_vs_scaled_canonical_error_diff":
                consistency_diff
        })


        # ----------------------------------------------------
        # Time-wise
        # ----------------------------------------------------

        for step in range(
            err_abs.shape[
                1
            ]
        ):

            this_radius = (

                scale

                *

                q_steps_norm[
                    step
                ]
            )

            this_covered = (

                err_abs[
                    :,
                    step
                ]

                <=

                this_radius
            )

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

                "normalized_stepwise_radius":
                    q_steps_norm[
                        step
                    ],

                "mean_absolute_radius":
                    float(
                        this_radius.mean()
                    ),

                "median_absolute_radius":
                    float(
                        np.median(
                            this_radius
                        )
                    ),

                "coverage":
                    float(
                        this_covered.mean()
                    ),

                "coverage_gap_pp":
                    100.0
                    *
                    (
                        this_covered.mean()
                        -
                        target
                    ),

                "mean_error":
                    float(
                        err_abs[
                            :,
                            step
                        ].mean()
                    ),

                "median_error":
                    float(
                        np.median(
                            err_abs[
                                :,
                                step
                            ]
                        )
                    ),

                "mean_disagreement":
                    float(
                        disagreement_abs[
                            :,
                            step
                        ].mean()
                    )
            })


        # ----------------------------------------------------
        # Source-wise
        # ----------------------------------------------------

        temp = metadata[
            [
                "source_group",
                "track_key"
            ]
        ].copy()

        temp[
            "covered"
        ] = covered_sim.astype(
            int
        )

        temp[
            "radius"
        ] = radius_abs

        temp[
            "max_error"
        ] = max_abs

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

                "mean_absolute_radius":
                    float(
                        group[
                            "radius"
                        ].mean()
                    ),

                "mean_max_error":
                    float(
                        group[
                            "max_error"
                        ].mean()
                    )
            })


        # ----------------------------------------------------
        # Failure cases
        # ----------------------------------------------------

        failures = np.where(
            ~covered_sim
        )[0]

        if len(
            failures
        ) > 0:

            excess = (

                max_abs[
                    failures
                ]

                -

                radius_abs[
                    failures
                ]
            )

            order = failures[
                np.argsort(
                    -excess
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

                    "scale":
                        float(
                            scale[
                                idx
                            ]
                        ),

                    "max_error":
                        float(
                            max_abs[
                                idx
                            ]
                        ),

                    "adaptive_radius":
                        float(
                            radius_abs[
                                idx
                            ]
                        ),

                    "excess_over_radius":
                        float(
                            max_abs[
                                idx
                            ]
                            -
                            radius_abs[
                                idx
                            ]
                        ),

                    "ADE":
                        float(
                            err_abs[
                                idx
                            ].mean()
                        ),

                    "FDE":
                        float(
                            err_abs[
                                idx,
                                -1
                            ]
                        ),

                    "mean_disagreement":
                        float(
                            disagreement_abs[
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
        )
    )


# ============================================================
# OPTIONAL COMPARISON AGAINST STEP 6A FIXED-RADIUS RESULTS
# ============================================================

def compare_with_step6a(
    adaptive_summary
):

    old_path = (

        STEP6A_DIR

        /

        "conformal_locked_evaluation_summary.csv"
    )

    if not old_path.exists():

        return pd.DataFrame()

    fixed = pd.read_csv(
        old_path
    )

    fixed = fixed[
        [
            "task",
            "dataset",
            "alpha",
            "target_coverage",
            "simultaneous_window_coverage",
            "simultaneous_coverage_gap_pp",
            "simultaneous_track_macro_coverage"
        ]
    ].copy()

    fixed = fixed.rename(
        columns={
            "simultaneous_window_coverage":
                "fixed_window_coverage",

            "simultaneous_coverage_gap_pp":
                "fixed_coverage_gap_pp",

            "simultaneous_track_macro_coverage":
                "fixed_track_macro_coverage"
        }
    )

    adaptive = adaptive_summary[
        [
            "task",
            "dataset",
            "alpha",
            "target_coverage",
            "simultaneous_window_coverage",
            "simultaneous_coverage_gap_pp",
            "simultaneous_track_macro_coverage",
            "mean_absolute_simultaneous_radius",
            "median_absolute_simultaneous_radius"
        ]
    ].copy()

    adaptive = adaptive.rename(
        columns={
            "simultaneous_window_coverage":
                "adaptive_window_coverage",

            "simultaneous_coverage_gap_pp":
                "adaptive_coverage_gap_pp",

            "simultaneous_track_macro_coverage":
                "adaptive_track_macro_coverage"
        }
    )

    merged = adaptive.merge(

        fixed,

        on=[
            "task",
            "dataset",
            "alpha",
            "target_coverage"
        ],

        how="left"
    )

    merged[
        "adaptive_minus_fixed_window_coverage_pp"
    ] = (

        100.0

        *

        (
            merged[
                "adaptive_window_coverage"
            ]

            -

            merged[
                "fixed_window_coverage"
            ]
        )
    )

    merged[
        "adaptive_minus_fixed_track_macro_coverage_pp"
    ] = (

        100.0

        *

        (
            merged[
                "adaptive_track_macro_coverage"
            ]

            -

            merged[
                "fixed_track_macro_coverage"
            ]
        )
    )

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    rng = np.random.default_rng(
        RNG_SEED
    )

    calibration_all = []

    stepwise_all = []

    summary_all = []

    time_all = []

    source_all = []

    failure_all = []

    self_check_all = []

    calibration_scores_all = []


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
        # VALIDATION-ONLY CALIBRATION
        # ====================================================

        validation, val_meta = load_validation(
            task
        )

        (
            _,
            _,
            val_mean_can,
            val_mean_abs,
            val_disagreement
        ) = ensemble_predict(

            task,

            validation
        )

        (
            calibration_df,
            stepwise_df,
            calibration_objects,
            val_err_can,
            val_err_abs
        ) = calibrate_task(

            task,

            validation,

            val_meta,

            val_mean_can,

            val_mean_abs
        )

        calibration_all.append(
            calibration_df
        )

        stepwise_all.append(
            stepwise_df
        )


        score_df = val_meta[
            [
                "source_group",
                "track_key"
            ]
        ].copy()

        if "sample_id" in val_meta.columns:

            score_df[
                "sample_id"
            ] = val_meta[
                "sample_id"
            ].values

        else:

            score_df[
                "sample_id"
            ] = np.arange(
                len(
                    val_meta
                )
            )

        score_df[
            "task"
        ] = task

        score_df[
            "scale"
        ] = validation[
            "scale"
        ]

        score_df[
            "max_canonical_error"
        ] = val_err_can.max(
            axis=1
        )

        score_df[
            "max_absolute_error"
        ] = val_err_abs.max(
            axis=1
        )

        score_df[
            "ADE"
        ] = val_err_abs.mean(
            axis=1
        )

        score_df[
            "FDE"
        ] = val_err_abs[
            :,
            -1
        ]

        score_df[
            "mean_disagreement"
        ] = val_disagreement.mean(
            axis=1
        )

        calibration_scores_all.append(
            score_df
        )


        print(
            "Scale-adaptive calibration locked from validation only:"
        )

        print(
            calibration_df.to_string(
                index=False
            )
        )


        # ====================================================
        # TESTREAL
        # ====================================================

        testreal, test_meta = load_testreal(
            task
        )

        (
            _,
            _,
            test_mean_can,
            test_mean_abs,
            test_disagreement
        ) = ensemble_predict(

            task,

            testreal
        )

        (
            summary_df,
            time_df,
            source_df,
            failure_df
        ) = evaluate_task(

            task,

            "BIRDSAI_TESTREAL",

            testreal,

            test_meta,

            test_mean_can,

            test_mean_abs,

            test_disagreement,

            calibration_objects,

            rng
        )

        summary_all.append(
            summary_df
        )

        time_all.append(
            time_df
        )

        source_all.append(
            source_df
        )

        if len(
            failure_df
        ) > 0:

            failure_all.append(
                failure_df
            )


        # ====================================================
        # FRESH WILDBOX
        # ====================================================

        fresh, fresh_meta = load_fresh(
            task
        )

        (
            _,
            fresh_stacked_abs,
            fresh_mean_can,
            fresh_mean_abs,
            fresh_disagreement
        ) = ensemble_predict(

            task,

            fresh
        )

        check = self_check_fresh(

            task,

            fresh_stacked_abs,

            fresh_mean_abs,

            fresh[
                "future_abs"
            ]
        )

        self_check_all.append(
            check
        )


        (
            summary_df,
            time_df,
            source_df,
            failure_df
        ) = evaluate_task(

            task,

            "FRESH_WILDBOX",

            fresh,

            fresh_meta,

            fresh_mean_can,

            fresh_mean_abs,

            fresh_disagreement,

            calibration_objects,

            rng
        )

        summary_all.append(
            summary_df
        )

        time_all.append(
            time_df
        )

        source_all.append(
            source_df
        )

        if len(
            failure_df
        ) > 0:

            failure_all.append(
                failure_df
            )


    # ========================================================
    # SAVE
    # ========================================================

    calibration = pd.concat(
        calibration_all,
        ignore_index=True
    )

    stepwise = pd.concat(
        stepwise_all,
        ignore_index=True
    )

    summary = pd.concat(
        summary_all,
        ignore_index=True
    )

    timewise = pd.concat(
        time_all,
        ignore_index=True
    )

    sourcewise = pd.concat(
        source_all,
        ignore_index=True
    )

    self_checks = pd.concat(
        self_check_all,
        ignore_index=True
    )

    calibration_scores = pd.concat(
        calibration_scores_all,
        ignore_index=True
    )

    if failure_all:

        failures = pd.concat(
            failure_all,
            ignore_index=True
        )

    else:

        failures = pd.DataFrame()


    comparison = compare_with_step6a(
        summary
    )


    calibration.to_csv(

        OUT /
        "scale_adaptive_calibration_summary.csv",

        index=False
    )

    stepwise.to_csv(

        OUT /
        "scale_adaptive_stepwise_normalized_radii.csv",

        index=False
    )

    summary.to_csv(

        OUT /
        "scale_adaptive_locked_evaluation_summary.csv",

        index=False
    )

    timewise.to_csv(

        OUT /
        "scale_adaptive_timewise_coverage.csv",

        index=False
    )

    sourcewise.to_csv(

        OUT /
        "scale_adaptive_sourcewise_coverage.csv",

        index=False
    )

    failures.to_csv(

        OUT /
        "scale_adaptive_failure_cases.csv",

        index=False
    )

    self_checks.to_csv(

        OUT /
        "fresh_transformer_self_check.csv",

        index=False
    )

    calibration_scores.to_csv(

        OUT /
        "validation_normalized_calibration_scores.csv",

        index=False
    )

    if len(
        comparison
    ) > 0:

        comparison.to_csv(

            OUT /
            "STEP6A_FIXED_vs_STEP6B_ADAPTIVE.csv",

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
            "STEP 6B: SCALE-ADAPTIVE SPLIT-CONFORMAL UNCERTAINTY\n"
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
            "- Calibration uses ONLY fixed BIRDSAI "
            "TrainReal validation windows.\n"
        )

        f.write(
            "- Score = maximum future Euclidean error "
            "in canonical normalized coordinates.\n"
        )

        f.write(
            "- For sample i, absolute simultaneous "
            "radius = normalized quantile × sample scale_i.\n"
        )

        f.write(
            "- Therefore uncertainty band adapts to "
            "trajectory movement scale without using "
            "TestReal or Fresh WildBox for tuning.\n"
        )

        f.write(
            "- 90% and 95% nominal levels evaluated.\n"
        )

        f.write(
            "- TestReal and Fresh WildBox remain locked "
            "evaluation datasets.\n"
        )

        f.write(
            "- Track-cluster bootstrap 95% CIs are "
            "reported for empirical simultaneous coverage.\n"
        )

        f.write(
            "- Because sliding windows overlap within "
            "tracks, do not describe these results as "
            "an unconditional theoretical field guarantee.\n\n"
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
            "\n\nLOCKED ADAPTIVE EVALUATION\n"
        )

        f.write(
            "--------------------------\n"
        )

        f.write(
            summary.to_string(
                index=False
            )
        )


        if len(
            comparison
        ) > 0:

            f.write(
                "\n\nSTEP 6A FIXED vs STEP 6B ADAPTIVE\n"
            )

            f.write(
                "---------------------------------\n"
            )

            f.write(
                comparison.to_string(
                    index=False
                )
            )


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(
        "\nSTEP 6B COMPLETE"
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
        "\nSCALE-ADAPTIVE CALIBRATION:\n"
    )

    print(
        calibration[
            [
                "task",
                "alpha",
                "target_coverage",
                "calibration_samples",
                "calibration_tracks",
                "normalized_simultaneous_quantile",
                "calibration_simultaneous_coverage",
                "mean_absolute_simultaneous_radius",
                "median_absolute_simultaneous_radius",
                "calibration_stepwise_marginal_coverage"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nLOCKED SCALE-ADAPTIVE EVALUATION:\n"
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
                "normalized_simultaneous_quantile",
                "mean_absolute_simultaneous_radius",
                "simultaneous_window_coverage",
                "simultaneous_coverage_gap_pp",
                "simultaneous_window_CI_low",
                "simultaneous_window_CI_high",
                "simultaneous_track_macro_coverage",
                "stepwise_marginal_coverage",
                "endpoint_stepwise_coverage",
                "mean_ADE",
                "mean_FDE"
            ]
        ].to_string(
            index=False
        )
    )


    if len(
        comparison
    ) > 0:

        print(
            "\nSTEP 6A FIXED vs STEP 6B ADAPTIVE:\n"
        )

        print(
            comparison[
                [
                    "task",
                    "dataset",
                    "alpha",
                    "target_coverage",
                    "fixed_window_coverage",
                    "adaptive_window_coverage",
                    "adaptive_minus_fixed_window_coverage_pp",
                    "fixed_track_macro_coverage",
                    "adaptive_track_macro_coverage",
                    "adaptive_minus_fixed_track_macro_coverage_pp"
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":

    main()
