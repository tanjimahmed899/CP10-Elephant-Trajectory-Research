from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\HP\Downloads\unified_trajectory_10hz"
)

OUT = Path(
    r"C:\Users\HP\Downloads\baseline_constant_velocity"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

FPS = 10


DATASETS = {

    "2s_to_2s": [

        (
            "BIRDSAI_train",
            "training",
            ROOT /
            "birdsai_train_real_core_2s_to_2s_10hz.npz",
            ROOT /
            "birdsai_train_real_core_2s_to_2s_10hz_metadata.csv"
        ),

        (
            "BIRDSAI_test",
            "in_domain_test",
            ROOT /
            "birdsai_test_real_core_2s_to_2s_10hz.npz",
            ROOT /
            "birdsai_test_real_core_2s_to_2s_10hz_metadata.csv"
        ),

        (
            "WildBox_external",
            "external_test",
            ROOT /
            "wildbox_external_2s_to_2s_10hz.npz",
            ROOT /
            "wildbox_external_2s_to_2s_10hz_metadata.csv"
        )
    ],


    "2s_to_5s": [

        (
            "BIRDSAI_train",
            "training",
            ROOT /
            "birdsai_train_real_core_2s_to_5s_10hz.npz",
            ROOT /
            "birdsai_train_real_core_2s_to_5s_10hz_metadata.csv"
        ),

        (
            "BIRDSAI_test",
            "in_domain_test",
            ROOT /
            "birdsai_test_real_core_2s_to_5s_10hz.npz",
            ROOT /
            "birdsai_test_real_core_2s_to_5s_10hz_metadata.csv"
        ),

        (
            "WildBox_external",
            "external_test",
            ROOT /
            "wildbox_external_2s_to_5s_10hz.npz",
            ROOT /
            "wildbox_external_2s_to_5s_10hz_metadata.csv"
        )
    ]
}


# ============================================================
# CONSTANT VELOCITY
# ============================================================

def estimate_velocity_linear_regression(
    history,
    fps
):

    """
    history:
        shape [T, 2]

    Velocity estimated using least-squares
    line fitting across the full observation history.

    Units:
        normalized coordinate / second
    """

    T = len(
        history
    )

    times = np.arange(
        T,
        dtype=np.float64
    ) / fps


    times_centered = (
        times
        -
        times.mean()
    )


    denom = np.sum(
        times_centered ** 2
    )


    if denom <= 1e-12:

        return np.zeros(
            2,
            dtype=np.float64
        )


    positions_centered = (
        history
        -
        history.mean(
            axis=0,
            keepdims=True
        )
    )


    velocity = (

        times_centered[:, None]
        *
        positions_centered

    ).sum(
        axis=0
    ) / denom


    return velocity


def predict_constant_velocity(
    history,
    future_length,
    fps
):

    velocity = (
        estimate_velocity_linear_regression(
            history,
            fps
        )
    )


    future_times = (

        np.arange(
            1,
            future_length + 1,
            dtype=np.float64
        )

        /
        fps
    )


    prediction = (

        history[-1][None, :]

        +

        future_times[:, None]
        *
        velocity[None, :]

    )


    return (
        prediction,
        velocity
    )


def predict_stationary(
    history,
    future_length
):

    return np.repeat(

        history[
            -1:
        ],

        future_length,

        axis=0

    )


# ============================================================
# METRICS
# ============================================================

def trajectory_metrics(
    prediction,
    truth
):

    errors = np.linalg.norm(

        prediction
        -
        truth,

        axis=1

    )


    return {

        "ADE":
            float(
                np.mean(
                    errors
                )
            ),

        "FDE":
            float(
                errors[-1]
            ),

        "median_point_error":
            float(
                np.median(
                    errors
                )
            ),

        "max_point_error":
            float(
                np.max(
                    errors
                )
            )
    }


# ============================================================
# STORAGE
# ============================================================

sample_rows = []

curve_rows = []


# ============================================================
# PROCESS
# ============================================================

for task, configs in DATASETS.items():


    print(
        "\n===================================="
    )

    print(
        "TASK:",
        task
    )

    print(
        "===================================="
    )


    for (
        dataset_name,
        role,
        npz_path,
        metadata_path
    ) in configs:


        print(
            "\nProcessing:",
            dataset_name
        )


        if not npz_path.exists():

            raise FileNotFoundError(
                npz_path
            )


        if not metadata_path.exists():

            raise FileNotFoundError(
                metadata_path
            )


        arrays = np.load(
            npz_path
        )


        history = arrays[
            "history_abs"
        ].astype(
            np.float64
        )


        future = arrays[
            "future_abs"
        ].astype(
            np.float64
        )


        metadata = pd.read_csv(
            metadata_path,
            dtype={
                "source_group": str,
                "track_key": str
            }
        )


        if len(history) != len(
            metadata
        ):

            raise RuntimeError(

                f"Length mismatch: "
                f"{dataset_name}"

            )


        # ==========================================
        # PREDICTION ARRAYS
        # ==========================================

        cv_predictions = []

        stationary_predictions = []


        # ==========================================
        # EACH SAMPLE
        # ==========================================

        for i in range(
            len(history)
        ):


            h = history[i]

            f = future[i]


            # -------------------------------
            # Stationary baseline
            # -------------------------------

            stationary_pred = (
                predict_stationary(
                    h,
                    len(f)
                )
            )


            stationary_metrics = (
                trajectory_metrics(
                    stationary_pred,
                    f
                )
            )


            # -------------------------------
            # Constant Velocity baseline
            # -------------------------------

            (
                cv_pred,
                velocity
            ) = (
                predict_constant_velocity(

                    h,

                    len(f),

                    FPS

                )
            )


            cv_metrics = (
                trajectory_metrics(
                    cv_pred,
                    f
                )
            )


            cv_predictions.append(
                cv_pred.astype(
                    np.float32
                )
            )


            stationary_predictions.append(
                stationary_pred.astype(
                    np.float32
                )
            )


            meta = metadata.iloc[
                i
            ]


            source_group = str(
                meta.get(
                    "source_group",
                    ""
                )
            )


            track_key = str(
                meta.get(
                    "track_key",
                    ""
                )
            )


            # -------------------------------
            # SAMPLE METRICS
            # -------------------------------

            for model_name, result in [

                (
                    "Stationary",
                    stationary_metrics
                ),

                (
                    "ConstantVelocity",
                    cv_metrics
                )

            ]:


                sample_rows.append({

                    "task":
                        task,

                    "dataset":
                        dataset_name,

                    "role":
                        role,

                    "model":
                        model_name,

                    "sample_index":
                        i,

                    "source_group":
                        source_group,

                    "track_key":
                        track_key,

                    "ADE":
                        result[
                            "ADE"
                        ],

                    "FDE":
                        result[
                            "FDE"
                        ],

                    "median_point_error":
                        result[
                            "median_point_error"
                        ],

                    "max_point_error":
                        result[
                            "max_point_error"
                        ],

                    "estimated_vx":
                        (
                            velocity[0]
                            if model_name
                            ==
                            "ConstantVelocity"
                            else 0.0
                        ),

                    "estimated_vy":
                        (
                            velocity[1]
                            if model_name
                            ==
                            "ConstantVelocity"
                            else 0.0
                        ),

                    "estimated_speed":
                        (
                            np.linalg.norm(
                                velocity
                            )
                            if model_name
                            ==
                            "ConstantVelocity"
                            else 0.0
                        )
                })


            # -------------------------------
            # ERROR VS FORECAST TIME
            # -------------------------------

            for model_name, pred in [

                (
                    "Stationary",
                    stationary_pred
                ),

                (
                    "ConstantVelocity",
                    cv_pred
                )

            ]:


                errors = np.linalg.norm(

                    pred
                    -
                    f,

                    axis=1

                )


                for step, error in enumerate(

                    errors,

                    start=1

                ):


                    curve_rows.append({

                        "task":
                            task,

                        "dataset":
                            dataset_name,

                        "role":
                            role,

                        "model":
                            model_name,

                        "source_group":
                            source_group,

                        "track_key":
                            track_key,

                        "sample_index":
                            i,

                        "forecast_step":
                            step,

                        "forecast_time_sec":
                            step
                            /
                            FPS,

                        "error":
                            float(
                                error
                            )
                    })


        # ==========================================
        # SAVE PREDICTIONS
        # ==========================================

        np.savez_compressed(

            OUT /
            f"{task}_{dataset_name}_predictions.npz",

            truth=
                future.astype(
                    np.float32
                ),

            constant_velocity=
                np.stack(
                    cv_predictions
                ),

            stationary=
                np.stack(
                    stationary_predictions
                )
        )


# ============================================================
# SAMPLE METRICS
# ============================================================

samples = pd.DataFrame(
    sample_rows
)


samples.to_csv(

    OUT /
    "sample_metrics.csv",

    index=False
)


# ============================================================
# WINDOW-LEVEL OVERALL METRICS
# ============================================================

overall_rows = []


for (
    task,
    dataset,
    role,
    model
), g in samples.groupby(

    [
        "task",
        "dataset",
        "role",
        "model"
    ]

):


    overall_rows.append({

        "task":
            task,

        "dataset":
            dataset,

        "role":
            role,

        "model":
            model,

        "samples":
            len(g),

        "tracks":
            g[
                "track_key"
            ].nunique(),

        "sources":
            g[
                "source_group"
            ].nunique(),

        "ADE_mean":
            g[
                "ADE"
            ].mean(),

        "ADE_median":
            g[
                "ADE"
            ].median(),

        "ADE_std":
            g[
                "ADE"
            ].std(),

        "FDE_mean":
            g[
                "FDE"
            ].mean(),

        "FDE_median":
            g[
                "FDE"
            ].median(),

        "FDE_std":
            g[
                "FDE"
            ].std()
    })


overall = pd.DataFrame(
    overall_rows
)


overall.to_csv(

    OUT /
    "overall_metrics.csv",

    index=False
)


# ============================================================
# PER TRACK
# ============================================================

track_metrics = (

    samples.groupby(

        [
            "task",
            "dataset",
            "role",
            "model",
            "source_group",
            "track_key"
        ],

        as_index=False

    )

    .agg(

        windows=(
            "ADE",
            "size"
        ),

        ADE_mean=(
            "ADE",
            "mean"
        ),

        FDE_mean=(
            "FDE",
            "mean"
        )

    )

)


track_metrics.to_csv(

    OUT /
    "per_track_metrics.csv",

    index=False
)


# ============================================================
# TRACK-MACRO METRICS
# ============================================================

track_macro = (

    track_metrics.groupby(

        [
            "task",
            "dataset",
            "role",
            "model"
        ],

        as_index=False

    )

    .agg(

        tracks=(
            "track_key",
            "size"
        ),

        ADE_track_macro=(
            "ADE_mean",
            "mean"
        ),

        FDE_track_macro=(
            "FDE_mean",
            "mean"
        )

    )

)


track_macro.to_csv(

    OUT /
    "track_macro_metrics.csv",

    index=False
)


# ============================================================
# PER SOURCE
# ============================================================

source_metrics = (

    samples.groupby(

        [
            "task",
            "dataset",
            "role",
            "model",
            "source_group"
        ],

        as_index=False

    )

    .agg(

        windows=(
            "ADE",
            "size"
        ),

        tracks=(
            "track_key",
            "nunique"
        ),

        ADE_mean=(
            "ADE",
            "mean"
        ),

        FDE_mean=(
            "FDE",
            "mean"
        )

    )

)


source_metrics.to_csv(

    OUT /
    "per_source_metrics.csv",

    index=False
)


# ============================================================
# SOURCE-MACRO METRICS
# ============================================================

source_macro = (

    source_metrics.groupby(

        [
            "task",
            "dataset",
            "role",
            "model"
        ],

        as_index=False

    )

    .agg(

        sources=(
            "source_group",
            "size"
        ),

        ADE_source_macro=(
            "ADE_mean",
            "mean"
        ),

        FDE_source_macro=(
            "FDE_mean",
            "mean"
        )

    )

)


source_macro.to_csv(

    OUT /
    "source_macro_metrics.csv",

    index=False
)


# ============================================================
# ERROR CURVE
# ============================================================

curves = pd.DataFrame(
    curve_rows
)


curve_summary = (

    curves.groupby(

        [
            "task",
            "dataset",
            "role",
            "model",
            "forecast_time_sec"
        ],

        as_index=False

    )

    .agg(

        mean_error=(
            "error",
            "mean"
        ),

        median_error=(
            "error",
            "median"
        )

    )

)


curve_summary.to_csv(

    OUT /
    "forecast_error_curve.csv",

    index=False
)


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = overall.merge(

    track_macro,

    on=[
        "task",
        "dataset",
        "role",
        "model"
    ],

    how="left"

).merge(

    source_macro,

    on=[
        "task",
        "dataset",
        "role",
        "model"
    ],

    how="left"

)


comparison.to_csv(

    OUT /
    "baseline_comparison.csv",

    index=False
)


# ============================================================
# TEXT SUMMARY
# ============================================================

with open(

    OUT /
    "SUMMARY.txt",

    "w",
    encoding="utf-8"

) as f:


    f.write(
        "ELEPHANT TRAJECTORY FORECASTING — "
        "NON-LEARNING BASELINES\n"
    )

    f.write(
        "=" * 78
        +
        "\n\n"
    )


    f.write(
        "Input: camera-corrected normalized x,y "
        "at 10 Hz.\n"
    )

    f.write(
        "Observation history: 2 seconds.\n"
    )

    f.write(
        "Constant Velocity velocity estimator: "
        "least-squares linear regression over "
        "the full 2-second observation history.\n"
    )

    f.write(
        "Predictions are anchored at the last "
        "observed position.\n"
    )

    f.write(
        "Metrics: ADE and FDE in camera-corrected "
        "normalized coordinate units.\n\n"
    )


    for _, row in (
        comparison.sort_values(
            [
                "task",
                "dataset",
                "model"
            ]
        ).iterrows()
    ):


        f.write(

            f"{row['task']} | "
            f"{row['dataset']} | "
            f"{row['role']} | "
            f"{row['model']}\n"

        )


        f.write(

            f"Samples: "
            f"{int(row['samples'])}\n"

        )


        f.write(

            f"ADE mean: "
            f"{row['ADE_mean']:.6f}\n"

        )


        f.write(

            f"FDE mean: "
            f"{row['FDE_mean']:.6f}\n"

        )


        f.write(

            f"Track-macro ADE: "
            f"{row['ADE_track_macro']:.6f}\n"

        )


        f.write(

            f"Track-macro FDE: "
            f"{row['FDE_track_macro']:.6f}\n"

        )


        f.write(

            f"Source-macro ADE: "
            f"{row['ADE_source_macro']:.6f}\n"

        )


        f.write(

            f"Source-macro FDE: "
            f"{row['FDE_source_macro']:.6f}\n\n"

        )


print(
    "\n============================================="
)

print(
    "CONSTANT VELOCITY BASELINE COMPLETE"
)

print(
    "============================================="
)

print(
    "\nMain comparison:"
)

print(
    comparison[
        [
            "task",
            "dataset",
            "role",
            "model",
            "ADE_mean",
            "FDE_mean",
            "ADE_track_macro",
            "FDE_track_macro",
            "ADE_source_macro",
            "FDE_source_macro"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nResults saved to:"
)

print(
    OUT
)