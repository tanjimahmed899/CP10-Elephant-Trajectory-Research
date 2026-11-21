from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\HP\Downloads\unified_trajectory_10hz"
)

PREVIOUS = Path(
    r"C:\Users\HP\Downloads\baseline_constant_velocity"
)

OUT = Path(
    r"C:\Users\HP\Downloads\baseline_kalman"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

FPS = 10
DT = 1.0 / FPS


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
# GRID
#
# IMPORTANT:
# These parameters are selected using TRAINREAL ONLY.
# ============================================================

Q_GRID = [
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1
]

R_GRID = [
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2
]


# ============================================================
# STATE-SPACE MODEL
#
# state = [x, y, vx, vy]
# ============================================================

F = np.array(
    [
        [1, 0, DT, 0],
        [0, 1, 0, DT],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ],
    dtype=np.float64
)


H = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ],
    dtype=np.float64
)


I4 = np.eye(
    4,
    dtype=np.float64
)


# ============================================================
# PROCESS NOISE
# ============================================================

def make_Q(q):

    dt = DT

    return q * np.array(
        [
            [
                dt**4 / 4,
                0,
                dt**3 / 2,
                0
            ],

            [
                0,
                dt**4 / 4,
                0,
                dt**3 / 2
            ],

            [
                dt**3 / 2,
                0,
                dt**2,
                0
            ],

            [
                0,
                dt**3 / 2,
                0,
                dt**2
            ]
        ],
        dtype=np.float64
    )


# ============================================================
# INITIAL VELOCITY
# ============================================================

def initial_velocity(
    history,
    n_frames=5
):

    n = min(
        n_frames,
        len(history)
    )

    points = history[
        :n
    ]

    times = (

        np.arange(
            n,
            dtype=np.float64
        )

        *
        DT
    )


    tc = (
        times
        -
        times.mean()
    )


    denom = np.sum(
        tc ** 2
    )


    if denom <= 1e-12:

        return np.zeros(
            2,
            dtype=np.float64
        )


    pc = (

        points
        -
        points.mean(
            axis=0,
            keepdims=True
        )

    )


    velocity = (

        tc[:, None]
        *
        pc

    ).sum(
        axis=0
    ) / denom


    return velocity


# ============================================================
# KALMAN FORECAST
# ============================================================

def kalman_forecast(
    history,
    future_length,
    q,
    r
):

    Q = make_Q(
        q
    )


    R = (

        np.eye(
            2,
            dtype=np.float64
        )

        *
        r
    )


    v0 = initial_velocity(
        history
    )


    x = np.array(
        [
            history[0, 0],
            history[0, 1],
            v0[0],
            v0[1]
        ],
        dtype=np.float64
    )


    # deliberately diffuse initialization
    P = np.eye(
        4,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # FILTER OBSERVATION HISTORY
    # --------------------------------------------------------

    for z in history:


        # predict
        x = F @ x

        P = (
            F
            @
            P
            @
            F.T
            +
            Q
        )


        # innovation
        innovation = (
            z
            -
            H @ x
        )


        S = (
            H
            @
            P
            @
            H.T
            +
            R
        )


        K = (
            P
            @
            H.T
            @
            np.linalg.inv(
                S
            )
        )


        x = (
            x
            +
            K @ innovation
        )


        # Joseph covariance update
        KH = K @ H


        P = (

            (I4 - KH)
            @
            P
            @
            (I4 - KH).T

            +

            K
            @
            R
            @
            K.T

        )


    # --------------------------------------------------------
    # FORECAST WITHOUT FUTURE MEASUREMENTS
    # --------------------------------------------------------

    predictions = []


    for _ in range(
        future_length
    ):


        x = F @ x

        P = (
            F
            @
            P
            @
            F.T
            +
            Q
        )


        predictions.append(
            x[:2].copy()
        )


    return np.asarray(
        predictions,
        dtype=np.float64
    )


# ============================================================
# METRICS
# ============================================================

def calculate_errors(
    prediction,
    truth
):

    errors = np.linalg.norm(

        prediction
        -
        truth,

        axis=1

    )


    return (
        float(
            errors.mean()
        ),

        float(
            errors[-1]
        ),

        errors
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(
    npz_path,
    metadata_path
):


    arr = np.load(
        npz_path
    )


    history = arr[
        "history_abs"
    ].astype(
        np.float64
    )


    future = arr[
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
            "History/metadata length mismatch"
        )


    return (
        history,
        future,
        metadata
    )


# ============================================================
# TUNING ON BIRDSAI TRAINREAL ONLY
# ============================================================

tuning_rows = []

best_params = {}


for task, configs in (
    DATASETS.items()
):


    train_config = [

        item

        for item in configs

        if item[0]
        ==
        "BIRDSAI_train"

    ][0]


    (
        dataset_name,
        role,
        npz_path,
        metadata_path

    ) = train_config


    history, future, metadata = (
        load_dataset(
            npz_path,
            metadata_path
        )
    )


    print(
        "\n======================================"
    )

    print(
        "TUNING:",
        task
    )

    print(
        "TrainReal samples:",
        len(history)
    )

    print(
        "======================================"
    )


    candidate_records = []


    for q in Q_GRID:

        for r in R_GRID:


            sample_ade = []

            sample_fde = []


            for i in range(
                len(history)
            ):


                pred = kalman_forecast(

                    history[i],

                    len(
                        future[i]
                    ),

                    q,
                    r

                )


                ade, fde, _ = (
                    calculate_errors(

                        pred,
                        future[i]

                    )
                )


                sample_ade.append(
                    ade
                )

                sample_fde.append(
                    fde
                )


            temp = metadata[
                [
                    "track_key",
                    "source_group"
                ]
            ].copy()


            temp[
                "ADE"
            ] = sample_ade


            temp[
                "FDE"
            ] = sample_fde


            # ------------------------------------
            # Equal weight per trajectory/track
            # ------------------------------------

            per_track = (

                temp.groupby(
                    "track_key",
                    as_index=False
                )

                .agg(
                    ADE=(
                        "ADE",
                        "mean"
                    ),

                    FDE=(
                        "FDE",
                        "mean"
                    )
                )

            )


            track_macro_ade = float(
                per_track[
                    "ADE"
                ].mean()
            )


            track_macro_fde = float(
                per_track[
                    "FDE"
                ].mean()
            )


            record = {

                "task":
                    task,

                "q":
                    q,

                "r":
                    r,

                "track_macro_ADE":
                    track_macro_ade,

                "track_macro_FDE":
                    track_macro_fde,

                "window_ADE":
                    float(
                        np.mean(
                            sample_ade
                        )
                    ),

                "window_FDE":
                    float(
                        np.mean(
                            sample_fde
                        )
                    )
            }


            tuning_rows.append(
                record
            )


            candidate_records.append(
                record
            )


    # --------------------------------------------
    # Select using track-macro ADE;
    # FDE only breaks ties
    # --------------------------------------------

    best = sorted(

        candidate_records,

        key=lambda x: (

            x[
                "track_macro_ADE"
            ],

            x[
                "track_macro_FDE"
            ]

        )

    )[0]


    best_params[
        task
    ] = (

        best["q"],
        best["r"]

    )


    print(
        "Best q:",
        best["q"]
    )

    print(
        "Best r:",
        best["r"]
    )

    print(
        "Train track-macro ADE:",
        best[
            "track_macro_ADE"
        ]
    )


pd.DataFrame(
    tuning_rows
).to_csv(

    OUT /
    "kalman_tuning.csv",

    index=False
)


# ============================================================
# FINAL EVALUATION
# ============================================================

sample_rows = []

curve_rows = []


for task, configs in (
    DATASETS.items()
):


    q, r = best_params[
        task
    ]


    print(
        "\nEvaluating:",
        task,
        "q=",
        q,
        "r=",
        r
    )


    for (
        dataset_name,
        role,
        npz_path,
        metadata_path

    ) in configs:


        history, future, metadata = (
            load_dataset(

                npz_path,

                metadata_path

            )
        )


        predictions = []


        for i in range(
            len(history)
        ):


            pred = kalman_forecast(

                history[i],

                len(
                    future[i]
                ),

                q,
                r

            )


            ade, fde, errors = (
                calculate_errors(

                    pred,

                    future[i]

                )
            )


            predictions.append(
                pred.astype(
                    np.float32
                )
            )


            meta = metadata.iloc[
                i
            ]


            source = str(
                meta.get(
                    "source_group",
                    ""
                )
            )


            track = str(
                meta.get(
                    "track_key",
                    ""
                )
            )


            sample_rows.append({

                "task":
                    task,

                "dataset":
                    dataset_name,

                "role":
                    role,

                "model":
                    "KalmanCV",

                "sample_index":
                    i,

                "source_group":
                    source,

                "track_key":
                    track,

                "ADE":
                    ade,

                "FDE":
                    fde,

                "q":
                    q,

                "r":
                    r
            })


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
                        "KalmanCV",

                    "sample_index":
                        i,

                    "source_group":
                        source,

                    "track_key":
                        track,

                    "forecast_time_sec":
                        step
                        /
                        FPS,

                    "error":
                        float(
                            error
                        )
                })


        np.savez_compressed(

            OUT /
            f"{task}_{dataset_name}_kalman_predictions.npz",

            truth=
                future.astype(
                    np.float32
                ),

            prediction=
                np.stack(
                    predictions
                )
        )


# ============================================================
# SAMPLE TABLE
# ============================================================

samples = pd.DataFrame(
    sample_rows
)


samples.to_csv(

    OUT /
    "kalman_sample_metrics.csv",

    index=False
)


# ============================================================
# OVERALL
# ============================================================

overall = (

    samples.groupby(

        [
            "task",
            "dataset",
            "role",
            "model"
        ],

        as_index=False

    )

    .agg(

        samples=(
            "ADE",
            "size"
        ),

        tracks=(
            "track_key",
            "nunique"
        ),

        sources=(
            "source_group",
            "nunique"
        ),

        ADE_mean=(
            "ADE",
            "mean"
        ),

        ADE_median=(
            "ADE",
            "median"
        ),

        ADE_std=(
            "ADE",
            "std"
        ),

        FDE_mean=(
            "FDE",
            "mean"
        ),

        FDE_median=(
            "FDE",
            "median"
        ),

        FDE_std=(
            "FDE",
            "std"
        )
    )

)


# ============================================================
# TRACK MACRO
# ============================================================

track_table = (

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


track_macro = (

    track_table.groupby(

        [
            "task",
            "dataset",
            "role",
            "model"
        ],

        as_index=False

    )

    .agg(

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


# ============================================================
# SOURCE MACRO
# ============================================================

source_table = (

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


source_table.to_csv(

    OUT /
    "kalman_per_source_metrics.csv",

    index=False
)


source_macro = (

    source_table.groupby(

        [
            "task",
            "dataset",
            "role",
            "model"
        ],

        as_index=False

    )

    .agg(

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


# ============================================================
# KALMAN FINAL TABLE
# ============================================================

kalman_summary = (

    overall

    .merge(

        track_macro,

        on=[
            "task",
            "dataset",
            "role",
            "model"
        ]

    )

    .merge(

        source_macro,

        on=[
            "task",
            "dataset",
            "role",
            "model"
        ]

    )

)


kalman_summary.to_csv(

    OUT /
    "kalman_metrics.csv",

    index=False
)


# ============================================================
# FORECAST CURVE
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
    "kalman_forecast_error_curve.csv",

    index=False
)


# ============================================================
# MERGE WITH PREVIOUS BASELINES
# ============================================================

previous_file = (

    PREVIOUS /
    "baseline_comparison.csv"

)


if previous_file.exists():


    previous = pd.read_csv(
        previous_file
    )


    # Harmonize columns

    keep = [

        "task",
        "dataset",
        "role",
        "model",
        "samples",
        "ADE_mean",
        "ADE_median",
        "ADE_std",
        "FDE_mean",
        "FDE_median",
        "FDE_std",
        "ADE_track_macro",
        "FDE_track_macro",
        "ADE_source_macro",
        "FDE_source_macro"

    ]


    old = previous[
        keep
    ].copy()


    new = kalman_summary[
        keep
    ].copy()


    combined = pd.concat(

        [
            old,
            new
        ],

        ignore_index=True

    )


else:

    combined = kalman_summary.copy()


combined.to_csv(

    OUT /
    "combined_baseline_comparison.csv",

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
        "KALMAN BASELINE\n"
    )

    f.write(
        "=" * 78
        +
        "\n\n"
    )


    f.write(
        "Kalman state: [x, y, vx, vy]\n"
    )

    f.write(
        "Model: constant-velocity state-space model\n"
    )

    f.write(
        "Sampling: 10 Hz\n"
    )

    f.write(
        "Observation: 2 seconds\n"
    )

    f.write(
        "Q/R selection: BIRDSAI TrainReal only\n"
    )

    f.write(
        "Selection objective: track-macro ADE\n"
    )

    f.write(
        "BIRDSAI TestReal and WildBox were NOT "
        "used for tuning.\n\n"
    )


    for task, (
        q,
        r
    ) in best_params.items():


        f.write(

            f"{task} selected "
            f"q={q}, r={r}\n"

        )


    f.write(
        "\nFINAL RESULTS\n"
    )

    f.write(
        "-" * 78
        +
        "\n\n"
    )


    for _, row in (
        combined.sort_values(

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
            f"{row['model']}\n"

        )


        f.write(

            f"ADE={row['ADE_mean']:.6f}, "
            f"FDE={row['FDE_mean']:.6f}\n"

        )


        f.write(

            f"Track-macro ADE="
            f"{row['ADE_track_macro']:.6f}, "

            f"FDE="
            f"{row['FDE_track_macro']:.6f}\n"

        )


        f.write(

            f"Source-macro ADE="
            f"{row['ADE_source_macro']:.6f}, "

            f"FDE="
            f"{row['FDE_source_macro']:.6f}\n\n"

        )


print(
    "\n=========================================="
)

print(
    "KALMAN BASELINE COMPLETE"
)

print(
    "=========================================="
)

print(
    "\nBest parameters:"
)

for task, params in (
    best_params.items()
):

    print(
        task,
        "q=",
        params[0],
        "r=",
        params[1]
    )


print(
    "\nCombined comparison:"
)

print(

    combined[
        [
            "task",
            "dataset",
            "model",
            "ADE_mean",
            "FDE_mean",
            "ADE_track_macro",
            "FDE_track_macro",
            "ADE_source_macro",
            "FDE_source_macro"
        ]
    ]

    .sort_values(
        [
            "task",
            "dataset",
            "model"
        ]
    )

    .to_string(
        index=False
    )

)


print(
    "\nSaved to:"
)

print(
    OUT
)