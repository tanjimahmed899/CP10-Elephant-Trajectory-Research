from pathlib import Path
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\HP\Downloads\final_neural_trajectory_dataset"
)

FRESH_ROOT = Path(
    r"C:\Users\HP\Downloads\wildbox_fresh_confirmatory_results"
)

MODEL_ROOT = Path(
    r"C:\Users\HP\Downloads\transformer_results"
)

GEOFENCE_INSTANCES = Path(
    r"C:\Users\HP\Downloads\step18_geofence"
    r"\step18b_evaluation"
    r"\step18b_all_geofence_instances.csv"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step19_clustered_bootstrap"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FROZEN SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

SEEDS = [
    42,
    52,
    62
]

TASKS = {
    "2s_to_2s": {
        "future_frames": 20
    },

    "2s_to_5s": {
        "future_frames": 50
    },
}

DATASETS = [
    "BIRDSAI_TestReal",
    "WildBox_Fresh",
]

BOOTSTRAP_REPS = 2000

BOOTSTRAP_SEED = 20261017

CI_LEVEL = 0.95

EPS = 1e-12


# ============================================================
# BOOTSTRAP RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=32
    ):

        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            max_len,
            dtype=torch.float32
        ).unsqueeze(
            1
        )

        div_term = torch.exp(

            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )

            *
            (
                -math.log(10000.0)
                /
                d_model
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )


    def forward(
        self,
        x
    ):

        return (
            x
            +
            self.pe[
                :,
                :x.size(1)
            ]
        )


# ============================================================
# FROZEN TRANSFORMER
# ============================================================

class TrajectoryTransformer(nn.Module):

    def __init__(
        self,
        future_length,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.10
    ):

        super().__init__()

        self.future_length = future_length

        self.input_projection = nn.Linear(
            2,
            d_model
        )

        self.position = PositionalEncoding(
            d_model,
            max_len=32
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.head = nn.Sequential(

            nn.Linear(
                d_model,
                d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model,
                future_length * 2
            )
        )


    def forward(
        self,
        history
    ):

        x = self.input_projection(
            history
        )

        x = self.position(
            x
        )

        x = self.encoder(
            x
        )

        x = x.mean(
            dim=1
        )

        x = self.norm(
            x
        )

        return (
            self.head(
                x
            )
            .view(
                -1,
                self.future_length,
                2
            )
        )


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

def load_transformer(
    task,
    seed,
    future_length
):

    path = (
        MODEL_ROOT
        /
        f"{task}_Transformer_seed{seed}.pt"
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    model = TrajectoryTransformer(

        future_length=future_length,

        d_model=int(
            checkpoint.get(
                "d_model",
                64
            )
        ),

        nhead=int(
            checkpoint.get(
                "nhead",
                4
            )
        ),

        num_layers=int(
            checkpoint.get(
                "num_layers",
                2
            )
        ),

        dim_feedforward=int(
            checkpoint.get(
                "dim_feedforward",
                128
            )
        ),

        dropout=float(
            checkpoint.get(
                "dropout",
                0.10
            )
        )

    ).to(
        DEVICE
    )

    state_dict = {

        k: v

        for k, v
        in state_dict.items()

        if k != "position.pe"
    }

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    missing = [

        key

        for key
        in result.missing_keys

        if key != "position.pe"
    ]

    if (
        missing
        or
        result.unexpected_keys
    ):

        raise RuntimeError(

            "Checkpoint mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: "
            f"{result.unexpected_keys}"
        )

    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    histories,
    batch_size=256
):

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(histories),
            batch_size
        ):

            x = torch.tensor(

                histories[
                    start:
                    start + batch_size
                ],

                dtype=torch.float32,

                device=DEVICE
            )

            y = model(
                x
            )

            predictions.append(
                y.cpu().numpy()
            )

    return np.concatenate(
        predictions,
        axis=0
    )


# ============================================================
# INVERSE CANONICAL TRANSFORM
# ============================================================

def inverse_transform_batch(
    canonical,
    origin,
    angle,
    scale
):

    output = []

    for i in range(
        len(canonical)
    ):

        a = float(
            angle[i]
        )

        c = np.cos(a)
        s = np.sin(a)

        R = np.asarray(
            [
                [c, s],
                [-s, c]
            ],
            dtype=np.float64
        )

        local = (

            canonical[i]

            *
            float(
                scale[i]
            )

        ) @ R

        output.append(
            local
            +
            origin[i]
        )

    return np.asarray(
        output,
        dtype=np.float64
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    dataset,
    task
):

    if dataset == "BIRDSAI_TestReal":

        npz_path = (
            DATA_ROOT
            /
            f"{task}_birdsai_test_canonical.npz"
        )

        metadata_path = (
            DATA_ROOT
            /
            f"{task}_birdsai_test_metadata.csv"
        )

    elif dataset == "WildBox_Fresh":

        npz_path = (
            FRESH_ROOT
            /
            f"{task}_fresh_dataset.npz"
        )

        metadata_path = (
            FRESH_ROOT
            /
            f"{task}_fresh_metadata.csv"
        )

    else:

        raise ValueError(
            dataset
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
        npz_path,
        allow_pickle=False
    )

    metadata = pd.read_csv(

        metadata_path,

        dtype={
            "source_group": str,
            "track_key": str
        }
    )

    return (
        arrays,
        metadata
    )


# ============================================================
# CREATE FROZEN ENSEMBLE FORECAST ERRORS
# ============================================================

def build_forecast_error_table(
    dataset,
    task
):

    arrays, metadata = load_dataset(
        dataset,
        task
    )

    history = arrays[
        "history_canonical"
    ].astype(
        np.float32
    )

    truth = arrays[
        "future_abs"
    ].astype(
        np.float64
    )

    origin = arrays[
        "origin"
    ]

    angle = arrays[
        "heading_angle"
    ]

    scale = arrays[
        "scale"
    ]

    future_length = truth.shape[
        1
    ]

    seed_predictions = []

    for seed in SEEDS:

        model = load_transformer(

            task,
            seed,
            future_length
        )

        pred_can = predict(
            model,
            history
        )

        pred_abs = inverse_transform_batch(

            pred_can,
            origin,
            angle,
            scale
        )

        seed_predictions.append(
            pred_abs
        )

    ensemble = np.mean(
        np.stack(
            seed_predictions,
            axis=0
        ),
        axis=0
    )

    point_error = np.linalg.norm(

        ensemble
        -
        truth,

        axis=2
    )

    ADE = point_error.mean(
        axis=1
    )

    FDE = point_error[
        :,
        -1
    ]

    table = metadata[
        [
            "source_group",
            "track_key"
        ]
    ].copy()

    table[
        "dataset"
    ] = dataset

    table[
        "task"
    ] = task

    table[
        "ADE"
    ] = ADE

    table[
        "FDE"
    ] = FDE

    return table


# ============================================================
# FORECAST POINT METRICS
# ============================================================

def forecast_metrics(
    table
):

    window_ADE = float(
        table[
            "ADE"
        ].mean()
    )

    window_FDE = float(
        table[
            "FDE"
        ].mean()
    )

    track_table = (

        table.groupby(
            [
                "source_group",
                "track_key"
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

    track_ADE = float(
        track_table[
            "ADE"
        ].mean()
    )

    track_FDE = float(
        track_table[
            "FDE"
        ].mean()
    )

    source_table = (

        table.groupby(
            "source_group",
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

    source_ADE = float(
        source_table[
            "ADE"
        ].mean()
    )

    source_FDE = float(
        source_table[
            "FDE"
        ].mean()
    )

    return {

        "ADE":
            window_ADE,

        "FDE":
            window_FDE,

        "track_macro_ADE":
            track_ADE,

        "track_macro_FDE":
            track_FDE,

        "source_macro_ADE":
            source_ADE,

        "source_macro_FDE":
            source_FDE,
    }


# ============================================================
# STRATIFIED TRACK BOOTSTRAP
# ============================================================

def sample_track_blocks(
    table,
    rng
):

    pieces = []

    # Resample tracks WITHIN each source.
    # This preserves source representation while keeping
    # all overlapping windows from sampled tracks together.

    for source, source_table in table.groupby(
        "source_group"
    ):

        tracks = (
            source_table[
                "track_key"
            ]
            .drop_duplicates()
            .to_numpy()
        )

        sampled_tracks = rng.choice(

            tracks,

            size=len(tracks),

            replace=True
        )

        for bootstrap_track_id, track in enumerate(
            sampled_tracks
        ):

            block = source_table[
                source_table[
                    "track_key"
                ]
                ==
                track
            ].copy()

            # Give repeated sampled clusters a unique
            # bootstrap track identity.
            block[
                "_bootstrap_track"
            ] = (

                str(source)
                +
                "|"
                +
                str(
                    bootstrap_track_id
                )
            )

            pieces.append(
                block
            )

    sampled = pd.concat(
        pieces,
        ignore_index=True
    )

    return sampled


def forecast_metrics_bootstrap(
    table
):

    # Must use bootstrap cluster IDs for track macro,
    # because same original track can be sampled twice.

    window_ADE = float(
        table[
            "ADE"
        ].mean()
    )

    window_FDE = float(
        table[
            "FDE"
        ].mean()
    )

    track_table = (

        table.groupby(
            [
                "source_group",
                "_bootstrap_track"
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

    source_table = (

        table.groupby(
            "source_group",
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

    return {

        "ADE":
            window_ADE,

        "FDE":
            window_FDE,

        "track_macro_ADE":
            float(
                track_table[
                    "ADE"
                ].mean()
            ),

        "track_macro_FDE":
            float(
                track_table[
                    "FDE"
                ].mean()
            ),

        "source_macro_ADE":
            float(
                source_table[
                    "ADE"
                ].mean()
            ),

        "source_macro_FDE":
            float(
                source_table[
                    "FDE"
                ].mean()
            ),
    }


# ============================================================
# GEOFENCE METRICS
# ============================================================

def auroc_rank(
    y_true,
    scores
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    positive = (
        y_true == 1
    )

    negative = (
        y_true == 0
    )

    n_positive = int(
        positive.sum()
    )

    n_negative = int(
        negative.sum()
    )

    if (
        n_positive == 0
        or
        n_negative == 0
    ):

        return np.nan

    ranks = (

        pd.Series(
            scores
        )
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    positive_rank_sum = float(
        ranks[
            positive
        ].sum()
    )

    return float(

        (
            positive_rank_sum

            -
            (
                n_positive
                *
                (
                    n_positive
                    +
                    1
                )
                /
                2.0
            )
        )

        /
        (
            n_positive
            *
            n_negative
        )
    )


def average_precision(
    y_true,
    scores
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    positives = int(
        y_true.sum()
    )

    if positives == 0:

        return np.nan

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    sorted_y = y_true[
        order
    ]

    cumulative_tp = np.cumsum(
        sorted_y
    )

    ranks = np.arange(
        1,
        len(sorted_y) + 1
    )

    precision = (
        cumulative_tp
        /
        ranks
    )

    return float(

        precision[
            sorted_y == 1
        ].sum()

        /
        positives
    )


def geofence_metrics(
    table
):

    true = table[
        "true_crossing"
    ].to_numpy(
        dtype=int
    )

    pred = table[
        "predicted_crossing"
    ].to_numpy(
        dtype=int
    )

    TP = int(
        np.sum(
            (true == 1)
            &
            (pred == 1)
        )
    )

    FP = int(
        np.sum(
            (true == 0)
            &
            (pred == 1)
        )
    )

    FN = int(
        np.sum(
            (true == 1)
            &
            (pred == 0)
        )
    )

    TN = int(
        np.sum(
            (true == 0)
            &
            (pred == 0)
        )
    )

    positive = TP + FN
    negative = TN + FP

    precision = (

        TP / (TP + FP)

        if (TP + FP) > 0

        else np.nan
    )

    recall = (

        TP / positive

        if positive > 0

        else np.nan
    )

    if (
        np.isfinite(
            precision
        )
        and
        np.isfinite(
            recall
        )
        and
        (
            precision
            +
            recall
        ) > 0
    ):

        F1 = (

            2
            *
            precision
            *
            recall

            /
            (
                precision
                +
                recall
            )
        )

    else:

        F1 = np.nan

    false_warning_rate = (

        FP / negative

        if negative > 0

        else np.nan
    )

    miss_rate = (

        FN / positive

        if positive > 0

        else np.nan
    )

    TP_table = table[
        (table["true_crossing"] == 1)
        &
        (table["predicted_crossing"] == 1)
    ]

    if len(
        TP_table
    ) > 0:

        TTE_MAE = float(
            TP_table[
                "tte_abs_error_sec"
            ].mean()
        )

        TTE_bias = float(
            TP_table[
                "tte_error_sec"
            ].mean()
        )

        mean_lead = float(
            TP_table[
                "true_tte_sec"
            ].mean()
        )

        median_lead = float(
            TP_table[
                "true_tte_sec"
            ].median()
        )

    else:

        TTE_MAE = np.nan
        TTE_bias = np.nan
        mean_lead = np.nan
        median_lead = np.nan

    return {

        "precision":
            precision,

        "recall":
            recall,

        "F1":
            F1,

        "false_warning_rate":
            false_warning_rate,

        "miss_rate":
            miss_rate,

        "AUROC":
            auroc_rank(
                true,
                table[
                    "risk_score"
                ]
            ),

        "AUPRC":
            average_precision(
                true,
                table[
                    "risk_score"
                ]
            ),

        "TTE_MAE_sec":
            TTE_MAE,

        "TTE_bias_sec":
            TTE_bias,

        "mean_warning_lead_sec":
            mean_lead,

        "median_warning_lead_sec":
            median_lead,
    }


# ============================================================
# GENERIC CI
# ============================================================

def percentile_ci(
    values,
    ci_level=0.95
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:

        return (
            np.nan,
            np.nan
        )

    alpha = (
        1.0
        -
        ci_level
    )

    low = np.quantile(
        values,
        alpha / 2.0
    )

    high = np.quantile(
        values,
        1.0
        -
        alpha / 2.0
    )

    return (
        float(low),
        float(high)
    )


# ============================================================
# LOAD GEOFENCE INSTANCES
# ============================================================

if not GEOFENCE_INSTANCES.exists():

    raise FileNotFoundError(
        GEOFENCE_INSTANCES
    )


geofence_all = pd.read_csv(

    GEOFENCE_INSTANCES,

    dtype={
        "dataset": str,
        "task": str,
        "source_group": str,
        "track_key": str,
    }
)


# ============================================================
# FORECASTING BOOTSTRAP
# ============================================================

forecast_ci_rows = []

forecast_bootstrap_store = {}


for dataset in DATASETS:

    for task in TASKS:

        print(
            "\nForecast bootstrap:",
            dataset,
            task
        )

        table = build_forecast_error_table(
            dataset,
            task
        )

        point = forecast_metrics(
            table
        )

        distributions = {
            metric: []
            for metric
            in point.keys()
        }

        for rep in range(
            BOOTSTRAP_REPS
        ):

            sampled = sample_track_blocks(
                table,
                rng
            )

            metrics = forecast_metrics_bootstrap(
                sampled
            )

            for metric, value in metrics.items():

                distributions[
                    metric
                ].append(
                    value
                )

        forecast_bootstrap_store[
            (dataset, task)
        ] = distributions

        for metric, point_value in point.items():

            low, high = percentile_ci(
                distributions[
                    metric
                ],
                CI_LEVEL
            )

            forecast_ci_rows.append({

                "dataset":
                    dataset,

                "task":
                    task,

                "metric":
                    metric,

                "estimate":
                    point_value,

                "CI95_low":
                    low,

                "CI95_high":
                    high,

                "bootstrap_reps":
                    BOOTSTRAP_REPS,

                "cluster_unit":
                    "track",

                "stratified_by":
                    "source_group",
            })


forecast_ci_df = pd.DataFrame(
    forecast_ci_rows
)


# ============================================================
# GEOFENCE BOOTSTRAP
# ============================================================

geofence_ci_rows = []

geofence_bootstrap_store = {}


for dataset in DATASETS:

    for task in TASKS:

        print(
            "\nGeofence bootstrap:",
            dataset,
            task
        )

        table = geofence_all[
            (
                geofence_all[
                    "dataset"
                ]
                ==
                dataset
            )
            &
            (
                geofence_all[
                    "task"
                ]
                ==
                task
            )
        ].copy()

        if len(
            table
        ) == 0:

            raise RuntimeError(
                f"No geofence rows for "
                f"{dataset} {task}"
            )

        point = geofence_metrics(
            table
        )

        distributions = {
            metric: []
            for metric
            in point.keys()
        }

        for rep in range(
            BOOTSTRAP_REPS
        ):

            sampled = sample_track_blocks(
                table,
                rng
            )

            metrics = geofence_metrics(
                sampled
            )

            for metric, value in metrics.items():

                distributions[
                    metric
                ].append(
                    value
                )

        geofence_bootstrap_store[
            (dataset, task)
        ] = distributions

        for metric, point_value in point.items():

            low, high = percentile_ci(
                distributions[
                    metric
                ],
                CI_LEVEL
            )

            geofence_ci_rows.append({

                "dataset":
                    dataset,

                "task":
                    task,

                "metric":
                    metric,

                "estimate":
                    point_value,

                "CI95_low":
                    low,

                "CI95_high":
                    high,

                "bootstrap_reps":
                    BOOTSTRAP_REPS,

                "cluster_unit":
                    "track",

                "stratified_by":
                    "source_group",
            })


geofence_ci_df = pd.DataFrame(
    geofence_ci_rows
)


# ============================================================
# LEAVE-ONE-SOURCE-OUT SENSITIVITY
# ============================================================

loso_rows = []


# Forecasting
for dataset in DATASETS:

    for task in TASKS:

        table = build_forecast_error_table(
            dataset,
            task
        )

        sources = sorted(
            table[
                "source_group"
            ].unique()
        )

        for source in sources:

            subset = table[
                table[
                    "source_group"
                ]
                !=
                source
            ].copy()

            metrics = forecast_metrics(
                subset
            )

            for metric, value in metrics.items():

                loso_rows.append({

                    "analysis":
                        "forecast",

                    "dataset":
                        dataset,

                    "task":
                        task,

                    "left_out_source":
                        source,

                    "metric":
                        metric,

                    "value":
                        value,
                })


# Geofence
for dataset in DATASETS:

    for task in TASKS:

        table = geofence_all[
            (
                geofence_all[
                    "dataset"
                ]
                ==
                dataset
            )
            &
            (
                geofence_all[
                    "task"
                ]
                ==
                task
            )
        ].copy()

        sources = sorted(
            table[
                "source_group"
            ].unique()
        )

        for source in sources:

            subset = table[
                table[
                    "source_group"
                ]
                !=
                source
            ].copy()

            metrics = geofence_metrics(
                subset
            )

            for metric, value in metrics.items():

                loso_rows.append({

                    "analysis":
                        "geofence",

                    "dataset":
                        dataset,

                    "task":
                        task,

                    "left_out_source":
                        source,

                    "metric":
                        metric,

                    "value":
                        value,
                })


loso_df = pd.DataFrame(
    loso_rows
)


# ============================================================
# LOSO RANGE SUMMARY
# ============================================================

loso_summary_rows = []


for (
    analysis,
    dataset,
    task,
    metric
), group in loso_df.groupby(

    [
        "analysis",
        "dataset",
        "task",
        "metric"
    ]
):

    values = group[
        "value"
    ].to_numpy(
        dtype=np.float64
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    loso_summary_rows.append({

        "analysis":
            analysis,

        "dataset":
            dataset,

        "task":
            task,

        "metric":
            metric,

        "sources":
            len(group),

        "LOSO_min":
            (
                float(
                    np.min(
                        values
                    )
                )
                if len(values)
                else np.nan
            ),

        "LOSO_max":
            (
                float(
                    np.max(
                        values
                    )
                )
                if len(values)
                else np.nan
            ),

        "LOSO_mean":
            (
                float(
                    np.mean(
                        values
                    )
                )
                if len(values)
                else np.nan
            ),
    })


loso_summary_df = pd.DataFrame(
    loso_summary_rows
)


# ============================================================
# HORIZON CONTRASTS
#
# 5s - 2s
#
# These are descriptive independent clustered-bootstrap
# contrasts, NOT paired hypothesis tests.
# ============================================================

horizon_rows = []


forecast_contrast_metrics = [
    "ADE",
    "FDE",
    "track_macro_ADE",
    "track_macro_FDE",
]


for dataset in DATASETS:

    for metric in forecast_contrast_metrics:

        dist_2 = np.asarray(

            forecast_bootstrap_store[
                (
                    dataset,
                    "2s_to_2s"
                )
            ][
                metric
            ],

            dtype=np.float64
        )

        dist_5 = np.asarray(

            forecast_bootstrap_store[
                (
                    dataset,
                    "2s_to_5s"
                )
            ][
                metric
            ],

            dtype=np.float64
        )

        difference = (
            dist_5
            -
            dist_2
        )

        point_2 = float(

            forecast_ci_df[
                (
                    forecast_ci_df[
                        "dataset"
                    ]
                    ==
                    dataset
                )
                &
                (
                    forecast_ci_df[
                        "task"
                    ]
                    ==
                    "2s_to_2s"
                )
                &
                (
                    forecast_ci_df[
                        "metric"
                    ]
                    ==
                    metric
                )
            ][
                "estimate"
            ].iloc[0]
        )

        point_5 = float(

            forecast_ci_df[
                (
                    forecast_ci_df[
                        "dataset"
                    ]
                    ==
                    dataset
                )
                &
                (
                    forecast_ci_df[
                        "task"
                    ]
                    ==
                    "2s_to_5s"
                )
                &
                (
                    forecast_ci_df[
                        "metric"
                    ]
                    ==
                    metric
                )
            ][
                "estimate"
            ].iloc[0]
        )

        low, high = percentile_ci(
            difference
        )

        horizon_rows.append({

            "analysis":
                "forecast",

            "dataset":
                dataset,

            "metric":
                metric,

            "contrast":
                "5s_minus_2s",

            "estimate_difference":
                (
                    point_5
                    -
                    point_2
                ),

            "CI95_low":
                low,

            "CI95_high":
                high,

            "interpretation":
                (
                    "descriptive independent "
                    "cluster-bootstrap contrast"
                ),
        })


geofence_contrast_metrics = [
    "precision",
    "recall",
    "F1",
    "false_warning_rate",
    "AUROC",
    "AUPRC",
    "TTE_MAE_sec",
]


for dataset in DATASETS:

    for metric in geofence_contrast_metrics:

        dist_2 = np.asarray(

            geofence_bootstrap_store[
                (
                    dataset,
                    "2s_to_2s"
                )
            ][
                metric
            ],

            dtype=np.float64
        )

        dist_5 = np.asarray(

            geofence_bootstrap_store[
                (
                    dataset,
                    "2s_to_5s"
                )
            ][
                metric
            ],

            dtype=np.float64
        )

        mask = (
            np.isfinite(
                dist_2
            )
            &
            np.isfinite(
                dist_5
            )
        )

        difference = (
            dist_5[
                mask
            ]
            -
            dist_2[
                mask
            ]
        )

        point_2 = float(

            geofence_ci_df[
                (
                    geofence_ci_df[
                        "dataset"
                    ]
                    ==
                    dataset
                )
                &
                (
                    geofence_ci_df[
                        "task"
                    ]
                    ==
                    "2s_to_2s"
                )
                &
                (
                    geofence_ci_df[
                        "metric"
                    ]
                    ==
                    metric
                )
            ][
                "estimate"
            ].iloc[0]
        )

        point_5 = float(

            geofence_ci_df[
                (
                    geofence_ci_df[
                        "dataset"
                    ]
                    ==
                    dataset
                )
                &
                (
                    geofence_ci_df[
                        "task"
                    ]
                    ==
                    "2s_to_5s"
                )
                &
                (
                    geofence_ci_df[
                        "metric"
                    ]
                    ==
                    metric
                )
            ][
                "estimate"
            ].iloc[0]
        )

        low, high = percentile_ci(
            difference
        )

        horizon_rows.append({

            "analysis":
                "geofence",

            "dataset":
                dataset,

            "metric":
                metric,

            "contrast":
                "5s_minus_2s",

            "estimate_difference":
                (
                    point_5
                    -
                    point_2
                ),

            "CI95_low":
                low,

            "CI95_high":
                high,

            "interpretation":
                (
                    "descriptive independent "
                    "cluster-bootstrap contrast"
                ),
        })


horizon_df = pd.DataFrame(
    horizon_rows
)


# ============================================================
# TESTREAL vs FRESH WILDBOX CONTRAST
#
# WildBox - TestReal
# ============================================================

domain_rows = []


for task in TASKS:

    for metric in forecast_contrast_metrics:

        test_dist = np.asarray(

            forecast_bootstrap_store[
                (
                    "BIRDSAI_TestReal",
                    task
                )
            ][
                metric
            ],

            dtype=np.float64
        )

        wild_dist = np.asarray(

            forecast_bootstrap_store[
                (
                    "WildBox_Fresh",
                    task
                )
            ][
                metric
            ],

            dtype=np.float64
        )

        difference = (
            wild_dist
            -
            test_dist
        )

        test_point = float(

            forecast_ci_df[
                (
                    forecast_ci_df[
                        "dataset"
                    ]
                    ==
                    "BIRDSAI_TestReal"
                )
                &
                (
                    forecast_ci_df[
                        "task"
                    ]
                    ==
                    task
                )
                &
                (
                    forecast_ci_df[
                        "metric"
                    ]
                    ==
                    metric
                )
            ][
                "estimate"
            ].iloc[0]
        )

        wild_point = float(

            forecast_ci_df[
                (
                    forecast_ci_df[
                        "dataset"
                    ]
                    ==
                    "WildBox_Fresh"
                )
                &
                (
                    forecast_ci_df[
                        "task"
                    ]
                    ==
                    task
                )
                &
                (
                    forecast_ci_df[
                        "metric"
                    ]
                    ==
                    metric
                )
            ][
                "estimate"
            ].iloc[0]
        )

        low, high = percentile_ci(
            difference
        )

        domain_rows.append({

            "analysis":
                "forecast",

            "task":
                task,

            "metric":
                metric,

            "contrast":
                "WildBox_minus_TestReal",

            "estimate_difference":
                (
                    wild_point
                    -
                    test_point
                ),

            "CI95_low":
                low,

            "CI95_high":
                high,

            "interpretation":
                (
                    "descriptive independent "
                    "cluster-bootstrap domain contrast"
                ),
        })


for task in TASKS:

    for metric in geofence_contrast_metrics:

        test_dist = np.asarray(

            geofence_bootstrap_store[
                (
                    "BIRDSAI_TestReal",
                    task
                )
            ][
                metric
            ],

            dtype=np.float64
        )

        wild_dist = np.asarray(

            geofence_bootstrap_store[
                (
                    "WildBox_Fresh",
                    task
                )
            ][
                metric
            ],

            dtype=np.float64
        )

        mask = (
            np.isfinite(
                test_dist
            )
            &
            np.isfinite(
                wild_dist
            )
        )

        difference = (

            wild_dist[
                mask
            ]
            -
            test_dist[
                mask
            ]

        )

        test_point = float(

            geofence_ci_df[
                (
                    geofence_ci_df[
                        "dataset"
                    ]
                    ==
                    "BIRDSAI_TestReal"
                )
                &
                (
                    geofence_ci_df[
                        "task"
                    ]
                    ==
                    task
                )
                &
                (
                    geofence_ci_df[
                        "metric"
                    ]
                    ==
                    metric
                )
            ][
                "estimate"
            ].iloc[0]
        )

        wild_point = float(

            geofence_ci_df[
                (
                    geofence_ci_df[
                        "dataset"
                    ]
                    ==
                    "WildBox_Fresh"
                )
                &
                (
                    geofence_ci_df[
                        "task"
                    ]
                    ==
                    task
                )
                &
                (
                    geofence_ci_df[
                        "metric"
                    ]
                    ==
                    metric
                )
            ][
                "estimate"
            ].iloc[0]
        )

        low, high = percentile_ci(
            difference
        )

        domain_rows.append({

            "analysis":
                "geofence",

            "task":
                task,

            "metric":
                metric,

            "contrast":
                "WildBox_minus_TestReal",

            "estimate_difference":
                (
                    wild_point
                    -
                    test_point
                ),

            "CI95_low":
                low,

            "CI95_high":
                high,

            "interpretation":
                (
                    "descriptive independent "
                    "cluster-bootstrap domain contrast"
                ),
        })


domain_df = pd.DataFrame(
    domain_rows
)


# ============================================================
# SAVE CSV FILES
# ============================================================

forecast_ci_df.to_csv(

    OUT /
    "step19_forecasting_clustered_CI.csv",

    index=False
)


geofence_ci_df.to_csv(

    OUT /
    "step19_geofence_clustered_CI.csv",

    index=False
)


loso_df.to_csv(

    OUT /
    "step19_leave_one_source_out.csv",

    index=False
)


loso_summary_df.to_csv(

    OUT /
    "step19_source_sensitivity_summary.csv",

    index=False
)


horizon_df.to_csv(

    OUT /
    "step19_horizon_contrasts.csv",

    index=False
)


domain_df.to_csv(

    OUT /
    "step19_domain_contrasts.csv",

    index=False
)


# ============================================================
# PUBLICATION-FRIENDLY TABLES
# ============================================================

forecast_publication = forecast_ci_df.copy()

forecast_publication[
    "estimate_95CI"
] = (

    forecast_publication[
        "estimate"
    ].map(
        lambda x:
        f"{x:.4f}"
    )

    +
    " ["

    +
    forecast_publication[
        "CI95_low"
    ].map(
        lambda x:
        f"{x:.4f}"
    )

    +
    ", "

    +
    forecast_publication[
        "CI95_high"
    ].map(
        lambda x:
        f"{x:.4f}"
    )

    +
    "]"
)


geofence_publication = geofence_ci_df.copy()


def format_geofence_metric(
    metric,
    value
):

    if metric in [
        "precision",
        "recall",
        "F1",
        "false_warning_rate",
        "miss_rate",
        "AUROC",
        "AUPRC",
    ]:

        return f"{value:.3f}"

    return f"{value:.3f}"


geofence_publication[
    "estimate_95CI"
] = [

    (
        format_geofence_metric(
            row["metric"],
            row["estimate"]
        )

        +
        " ["

        +
        format_geofence_metric(
            row["metric"],
            row["CI95_low"]
        )

        +
        ", "

        +
        format_geofence_metric(
            row["metric"],
            row["CI95_high"]
        )

        +
        "]"
    )

    for _, row
    in geofence_publication.iterrows()
]


forecast_publication[
    [
        "dataset",
        "task",
        "metric",
        "estimate_95CI"
    ]
].to_csv(

    OUT /
    "TABLE_forecasting_95CI.csv",

    index=False
)


geofence_publication[
    [
        "dataset",
        "task",
        "metric",
        "estimate_95CI"
    ]
].to_csv(

    OUT /
    "TABLE_geofence_95CI.csv",

    index=False
)


# ============================================================
# SUMMARY
# ============================================================

with open(

    OUT /
    "SUMMARY.txt",

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "STEP 19 — CLUSTERED BOOTSTRAP FINAL STATISTICAL ANALYSIS\n"
    )

    f.write(
        "=" * 92
        +
        "\n\n"
    )


    f.write(
        "METHOD\n"
    )

    f.write(
        f"- Bootstrap replicates: {BOOTSTRAP_REPS}\n"
    )

    f.write(
        f"- Random seed: {BOOTSTRAP_SEED}\n"
    )

    f.write(
        "- Primary bootstrap cluster: trajectory track.\n"
    )

    f.write(
        "- Resampling is stratified within source_group.\n"
    )

    f.write(
        "- All overlapping windows belonging to a sampled "
        "track remain together as one resampled cluster.\n"
    )

    f.write(
        "- 95% percentile bootstrap confidence intervals.\n"
    )

    f.write(
        "- Source-level uncertainty is NOT presented as a "
        "bootstrap CI because only 3 fresh WildBox sources "
        "and 5-6 TestReal sources are available.\n"
    )

    f.write(
        "- Source robustness is instead summarized using "
        "leave-one-source-out sensitivity.\n"
    )

    f.write(
        "- No model retraining, model selection, warning "
        "threshold tuning, or protocol modification occurred.\n\n"
    )


    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "- BIRDSAI TestReal remains a frozen in-domain "
        "benchmark previously inspected during development.\n"
    )

    f.write(
        "- Fresh WildBox remains the preregistered unseen "
        "source-video external holdout.\n"
    )

    f.write(
        "- Forecasting errors are normalized image-plane "
        "coordinate errors, not metres.\n"
    )

    f.write(
        "- Virtual-geofence events are experimental image-plane "
        "boundary crossings, not actual human-elephant conflict "
        "events.\n"
    )

    f.write(
        "- Horizon and domain contrasts are descriptive "
        "independent clustered-bootstrap contrasts, not paired "
        "hypothesis tests.\n\n"
    )


    f.write(
        "FORECASTING 95% CI\n"
    )

    f.write(
        "-" * 92
        +
        "\n"
    )

    f.write(
        forecast_publication[
            [
                "dataset",
                "task",
                "metric",
                "estimate_95CI"
            ]
        ].to_string(
            index=False
        )
    )


    f.write(
        "\n\nGEOFENCE 95% CI\n"
    )

    f.write(
        "-" * 92
        +
        "\n"
    )

    f.write(
        geofence_publication[
            [
                "dataset",
                "task",
                "metric",
                "estimate_95CI"
            ]
        ].to_string(
            index=False
        )
    )


    f.write(
        "\n\nSOURCE SENSITIVITY\n"
    )

    f.write(
        "-" * 92
        +
        "\n"
    )

    f.write(
        loso_summary_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nHORIZON CONTRASTS\n"
    )

    f.write(
        "-" * 92
        +
        "\n"
    )

    f.write(
        horizon_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nDOMAIN CONTRASTS\n"
    )

    f.write(
        "-" * 92
        +
        "\n"
    )

    f.write(
        domain_df.to_string(
            index=False
        )
    )


# ============================================================
# CONSOLE
# ============================================================

print(
    "\n"
    +
    "=" * 92
)

print(
    "STEP 19 COMPLETE"
)

print(
    "=" * 92
)


print(
    "\nForecasting confidence intervals:"
)

print(

    forecast_publication[
        [
            "dataset",
            "task",
            "metric",
            "estimate_95CI"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nGeofence confidence intervals:"
)

print(

    geofence_publication[
        [
            "dataset",
            "task",
            "metric",
            "estimate_95CI"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    OUT
)