from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA = Path(
    r"C:\Users\HP\Downloads\final_neural_trajectory_dataset"
)

BASELINE = Path(
    r"C:\Users\HP\Downloads\baseline_constant_velocity"
)

KALMAN = Path(
    r"C:\Users\HP\Downloads\baseline_kalman"
)

RNN = Path(
    r"C:\Users\HP\Downloads\lstm_gru_results"
)

TRANSFORMER = Path(
    r"C:\Users\HP\Downloads\transformer_results"
)

OUT = Path(
    r"C:\Users\HP\Downloads\motion_regime_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


TASKS = [
    "2s_to_2s",
    "2s_to_5s"
]

SEEDS = [
    42,
    52,
    62
]

FPS = 10


# ============================================================
# HISTORY FEATURES
# ============================================================

def history_features(history):

    increments = np.diff(
        history,
        axis=0
    )

    distances = np.linalg.norm(
        increments,
        axis=1
    )

    path_length = float(
        distances.sum()
    )

    net_vector = (
        history[-1]
        -
        history[0]
    )

    net_displacement = float(
        np.linalg.norm(
            net_vector
        )
    )

    straightness = (

        net_displacement
        /
        path_length

        if path_length > 1e-12

        else 0.0
    )

    speeds = (
        distances
        *
        FPS
    )

    mean_speed = float(
        speeds.mean()
    )

    speed_std = float(
        speeds.std()
    )


    # --------------------------------------------------------
    # First-half and second-half directions
    # --------------------------------------------------------

    midpoint = (
        len(history)
        //
        2
    )

    v1 = (

        history[
            midpoint - 1
        ]

        -
        history[0]

    )

    v2 = (

        history[-1]

        -
        history[
            midpoint
        ]

    )


    n1 = np.linalg.norm(
        v1
    )

    n2 = np.linalg.norm(
        v2
    )


    if (
        n1 > 1e-10
        and
        n2 > 1e-10
    ):

        cosine = np.clip(

            np.dot(
                v1,
                v2
            )
            /
            (
                n1
                *
                n2
            ),

            -1.0,
            1.0
        )

        heading_change = float(
            np.arccos(
                cosine
            )
        )

    else:

        heading_change = 0.0


    # --------------------------------------------------------
    # acceleration proxy
    # --------------------------------------------------------

    velocity = (
        increments
        *
        FPS
    )


    if len(
        velocity
    ) > 1:

        acceleration = (

            np.diff(
                velocity,
                axis=0
            )

            *
            FPS
        )

        acceleration_magnitude = (
            np.linalg.norm(
                acceleration,
                axis=1
            )
        )

        mean_acceleration = float(
            acceleration_magnitude.mean()
        )

    else:

        mean_acceleration = 0.0


    return {

        "path_length":
            path_length,

        "net_displacement":
            net_displacement,

        "straightness":
            straightness,

        "mean_speed":
            mean_speed,

        "speed_std":
            speed_std,

        "heading_change_rad":
            heading_change,

        "mean_acceleration":
            mean_acceleration
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_set(
    task,
    dataset
):

    if dataset == "BIRDSAI_train":

        npz = DATA / (
            f"{task}_birdsai_trainval_canonical.npz"
        )

        meta = DATA / (
            f"{task}_birdsai_trainval_metadata.csv"
        )

    elif dataset == "BIRDSAI_test":

        npz = DATA / (
            f"{task}_birdsai_test_canonical.npz"
        )

        meta = DATA / (
            f"{task}_birdsai_test_metadata.csv"
        )

    elif dataset == "WildBox_external":

        npz = DATA / (
            f"{task}_wildbox_external_canonical.npz"
        )

        meta = DATA / (
            f"{task}_wildbox_external_metadata.csv"
        )

    else:

        raise ValueError(
            dataset
        )


    arrays = np.load(
        npz
    )


    metadata = pd.read_csv(
        meta,
        dtype={
            "source_group": str,
            "track_key": str
        }
    )


    history = arrays[
        "history_abs"
    ]


    return (
        arrays,
        metadata,
        history
    )


# ============================================================
# TRAIN-ONLY THRESHOLDS
# ============================================================

threshold_rows = []

feature_tables = {}


for task in TASKS:


    arrays, meta, histories = (
        load_set(
            task,
            "BIRDSAI_train"
        )
    )


    train_indices = arrays[
        "train_indices"
    ].astype(int)


    rows = []


    for idx in train_indices:

        feats = history_features(
            histories[idx]
        )

        feats[
            "sample_index"
        ] = idx

        rows.append(
            feats
        )


    train_features = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # All thresholds from neural TRAIN only
    # --------------------------------------------------------

    slow_threshold = float(

        train_features[
            "path_length"
        ].quantile(
            0.25
        )
    )


    straight_threshold = float(

        train_features[
            "straightness"
        ].quantile(
            0.50
        )
    )


    heading_threshold = float(

        train_features[
            "heading_change_rad"
        ].quantile(
            0.50
        )
    )


    threshold_rows.append({

        "task":
            task,

        "slow_path_threshold_q25":
            slow_threshold,

        "straightness_threshold_median":
            straight_threshold,

        "heading_change_threshold_median":
            heading_threshold
    })


threshold_df = pd.DataFrame(
    threshold_rows
)


threshold_df.to_csv(

    OUT /
    "regime_thresholds.csv",

    index=False
)


# ============================================================
# ASSIGN REGIME
# ============================================================

def assign_regime(
    features,
    thresholds
):


    if (

        features[
            "path_length"
        ]

        <=

        thresholds[
            "slow_path_threshold_q25"
        ]

    ):

        return "slow_stationary"


    if (

        features[
            "straightness"
        ]

        >=

        thresholds[
            "straightness_threshold_median"
        ]

        and

        features[
            "heading_change_rad"
        ]

        <=

        thresholds[
            "heading_change_threshold_median"
        ]

    ):

        return "persistent_linear"


    return "irregular_turning"


# ============================================================
# BUILD REGIME TABLE
# ============================================================

all_feature_rows = []


for task in TASKS:


    threshold = (

        threshold_df[
            threshold_df[
                "task"
            ]
            ==
            task
        ]

        .iloc[0]
    )


    for dataset in [

        "BIRDSAI_train",
        "BIRDSAI_test",
        "WildBox_external"

    ]:


        arrays, metadata, histories = (
            load_set(
                task,
                dataset
            )
        )


        if dataset == "BIRDSAI_train":

            indices = arrays[
                "train_indices"
            ].astype(int)

        else:

            indices = np.arange(
                len(histories)
            )


        for idx in indices:


            features = history_features(
                histories[idx]
            )


            regime = assign_regime(

                features,

                threshold
            )


            meta = metadata.iloc[
                idx
            ]


            all_feature_rows.append({

                "task":
                    task,

                "dataset":
                    dataset,

                "sample_index":
                    idx,

                "source_group":
                    str(
                        meta[
                            "source_group"
                        ]
                    ),

                "track_key":
                    str(
                        meta[
                            "track_key"
                        ]
                    ),

                "regime":
                    regime,

                **features
            })


feature_df = pd.DataFrame(
    all_feature_rows
)


feature_df.to_csv(

    OUT /
    "motion_regime_samples.csv",

    index=False
)


# ============================================================
# REGIME DISTRIBUTION
# ============================================================

distribution = (

    feature_df.groupby(

        [
            "task",
            "dataset",
            "regime"
        ],

        as_index=False

    )

    .agg(

        windows=(
            "sample_index",
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

        median_path_length=(
            "path_length",
            "median"
        ),

        median_straightness=(
            "straightness",
            "median"
        ),

        median_heading_change=(
            "heading_change_rad",
            "median"
        )
    )
)


distribution.to_csv(

    OUT /
    "regime_distribution.csv",

    index=False
)


# ============================================================
# LOAD MODEL PREDICTIONS
# ============================================================

def trajectory_errors(
    pred,
    truth
):

    error = np.linalg.norm(

        pred
        -
        truth,

        axis=2
    )


    return (
        error.mean(
            axis=1
        ),

        error[
            :,
            -1
        ]
    )


error_rows = []


# ------------------------------------------------------------
# Classical predictions
# ------------------------------------------------------------

for task in TASKS:


    for dataset in [

        "BIRDSAI_test",
        "WildBox_external"

    ]:


        pred_file = (

            BASELINE /
            f"{task}_{dataset}_predictions.npz"

        )


        data = np.load(
            pred_file
        )


        truth = data[
            "truth"
        ]


        for model, key in [

            (
                "Stationary",
                "stationary"
            ),

            (
                "ConstantVelocity",
                "constant_velocity"
            )

        ]:


            ADE, FDE = trajectory_errors(

                data[
                    key
                ],

                truth

            )


            for i in range(
                len(ADE)
            ):


                error_rows.append({

                    "task":
                        task,

                    "dataset":
                        dataset,

                    "sample_index":
                        i,

                    "model":
                        model,

                    "seed":
                        -1,

                    "ADE":
                        ADE[i],

                    "FDE":
                        FDE[i]
                })


        # Kalman

        kalman_file = (

            KALMAN /
            f"{task}_{dataset}_kalman_predictions.npz"

        )


        k = np.load(
            kalman_file
        )


        ADE, FDE = trajectory_errors(

            k[
                "prediction"
            ],

            k[
                "truth"
            ]

        )


        for i in range(
            len(ADE)
        ):


            error_rows.append({

                "task":
                    task,

                "dataset":
                    dataset,

                "sample_index":
                    i,

                "model":
                    "KalmanCV",

                "seed":
                    -1,

                "ADE":
                    ADE[i],

                "FDE":
                    FDE[i]
            })


# ------------------------------------------------------------
# LSTM / GRU predictions
# ------------------------------------------------------------

for task in TASKS:


    for dataset in [

        "BIRDSAI_test",
        "WildBox_external"

    ]:


        for model in [
            "LSTM",
            "GRU"
        ]:


            for seed in SEEDS:


                pred_file = (

                    RNN /
                    (
                        f"{task}_{dataset}_"
                        f"{model}_seed{seed}_predictions.npz"
                    )

                )


                data = np.load(
                    pred_file
                )


                ADE, FDE = trajectory_errors(

                    data[
                        "prediction_abs"
                    ],

                    data[
                        "truth_abs"
                    ]

                )


                for i in range(
                    len(ADE)
                ):


                    error_rows.append({

                        "task":
                            task,

                        "dataset":
                            dataset,

                        "sample_index":
                            i,

                        "model":
                            model,

                        "seed":
                            seed,

                        "ADE":
                            ADE[i],

                        "FDE":
                            FDE[i]
                    })


# ------------------------------------------------------------
# Transformer predictions
# ------------------------------------------------------------

for task in TASKS:


    for dataset in [

        "BIRDSAI_test",
        "WildBox_external"

    ]:


        for seed in SEEDS:


            pred_file = (

                TRANSFORMER /
                (
                    f"{task}_{dataset}_"
                    f"Transformer_seed{seed}_predictions.npz"
                )

            )


            data = np.load(
                pred_file
            )


            ADE, FDE = trajectory_errors(

                data[
                    "prediction_abs"
                ],

                data[
                    "truth_abs"
                ]

            )


            for i in range(
                len(ADE)
            ):


                error_rows.append({

                    "task":
                        task,

                    "dataset":
                        dataset,

                    "sample_index":
                        i,

                    "model":
                        "Transformer",

                    "seed":
                        seed,

                    "ADE":
                        ADE[i],

                    "FDE":
                        FDE[i]
                })


errors_df = pd.DataFrame(
    error_rows
)


# ============================================================
# JOIN REGIMES
# ============================================================

test_features = feature_df[

    feature_df[
        "dataset"
    ].isin(
        [
            "BIRDSAI_test",
            "WildBox_external"
        ]
    )

][
    [
        "task",
        "dataset",
        "sample_index",
        "source_group",
        "track_key",
        "regime"
    ]
]


merged = errors_df.merge(

    test_features,

    on=[
        "task",
        "dataset",
        "sample_index"
    ],

    how="left"
)


if merged[
    "regime"
].isna().any():

    raise RuntimeError(
        "Some predictions could not be matched "
        "to motion regimes."
    )


merged.to_csv(

    OUT /
    "model_errors_with_regimes.csv",

    index=False
)


# ============================================================
# AVERAGE MULTI-SEED MODELS FIRST
# ============================================================

seed_average = (

    merged.groupby(

        [
            "task",
            "dataset",
            "sample_index",
            "source_group",
            "track_key",
            "regime",
            "model"
        ],

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


# ============================================================
# REGIME RESULTS
# ============================================================

regime_results = (

    seed_average.groupby(

        [
            "task",
            "dataset",
            "regime",
            "model"
        ],

        as_index=False

    )

    .agg(

        windows=(
            "sample_index",
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
        ),

        ADE_median=(
            "ADE",
            "median"
        ),

        FDE_median=(
            "FDE",
            "median"
        )
    )
)


regime_results.to_csv(

    OUT /
    "regime_model_comparison.csv",

    index=False
)


# ============================================================
# BEST MODEL PER REGIME
# ============================================================

best_rows = []


for (
    task,
    dataset,
    regime
), group in regime_results.groupby(

    [
        "task",
        "dataset",
        "regime"
    ]

):


    best = group.sort_values(
        [
            "ADE_mean",
            "FDE_mean"
        ]
    ).iloc[0]


    best_rows.append({

        "task":
            task,

        "dataset":
            dataset,

        "regime":
            regime,

        "best_model":
            best[
                "model"
            ],

        "best_ADE":
            best[
                "ADE_mean"
            ],

        "best_FDE":
            best[
                "FDE_mean"
            ]
    })


best_df = pd.DataFrame(
    best_rows
)


best_df.to_csv(

    OUT /
    "best_model_by_regime.csv",

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
        "MOTION REGIME AUDIT\n"
    )

    f.write(
        "=" * 76
        +
        "\n\n"
    )


    f.write(
        "Regime thresholds were derived from "
        "BIRDSAI neural-training samples ONLY.\n"
    )

    f.write(
        "No TestReal or WildBox outcome was used "
        "to define thresholds.\n\n"
    )


    f.write(
        "REGIME THRESHOLDS\n"
    )

    f.write(
        "-" * 76
        +
        "\n"
    )


    f.write(
        threshold_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nBEST MODEL BY REGIME\n"
    )

    f.write(
        "-" * 76
        +
        "\n"
    )


    f.write(
        best_df.to_string(
            index=False
        )
    )

    f.write(
        "\n"
    )


print(
    "\n======================================"
)

print(
    "MOTION REGIME AUDIT COMPLETE"
)

print(
    "======================================"
)

print(
    "\nThresholds:"
)

print(
    threshold_df.to_string(
        index=False
    )
)

print(
    "\nRegime distribution:"
)

print(
    distribution.to_string(
        index=False
    )
)

print(
    "\nBest model by regime:"
)

print(
    best_df.to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    OUT
)