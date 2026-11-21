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

MODEL_ROOT = Path(
    r"C:\Users\HP\Downloads\transformer_results"
)

OUT = Path(
    r"C:\Users\HP\Downloads\transformer_ensemble_validation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

TASKS = [
    "2s_to_2s",
    "2s_to_5s"
]

SEEDS = [
    42,
    52,
    62
]

FPS = 10.0

EPS = 1e-12


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
                -math.log(
                    10000.0
                )

                /
                d_model
            )

        )


        pe[:, 0::2] = torch.sin(

            position
            *
            div_term

        )


        pe[:, 1::2] = torch.cos(

            position
            *
            div_term

        )


        self.register_buffer(

            "pe",

            pe.unsqueeze(
                0
            )

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
# ORIGINAL FROZEN TRANSFORMER
#
# Must match Step 11 checkpoint architecture.
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


        self.future_length = (
            future_length
        )


        self.input_projection = nn.Linear(

            2,

            d_model

        )


        self.position = PositionalEncoding(

            d_model,

            max_len=32

        )


        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead,

                dim_feedforward=dim_feedforward,

                dropout=dropout,

                activation="gelu",

                batch_first=True,

                norm_first=True

            )
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


        pooled = x.mean(
            dim=1
        )


        pooled = self.norm(
            pooled
        )


        future = (

            self.head(
                pooled
            )

            .view(
                -1,
                self.future_length,
                2
            )

        )


        return future


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_transformer(
    task,
    seed,
    future_length
):


    checkpoint_file = (

        MODEL_ROOT
        /
        f"{task}_Transformer_seed{seed}.pt"

    )


    if not checkpoint_file.exists():

        raise FileNotFoundError(

            f"\nCheckpoint not found:\n"
            f"{checkpoint_file}\n\n"
            f"Expected Step 11 checkpoint name:\n"
            f"{task}_Transformer_seed{seed}.pt"

        )


    checkpoint = torch.load(

        checkpoint_file,

        map_location=DEVICE

    )


    if isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Architecture metadata from the original checkpoint
    # --------------------------------------------------------

    d_model = int(
        checkpoint.get(
            "d_model",
            64
        )
    )


    nhead = int(
        checkpoint.get(
            "nhead",
            4
        )
    )


    num_layers = int(
        checkpoint.get(
            "num_layers",
            2
        )
    )


    dim_feedforward = int(
        checkpoint.get(
            "dim_feedforward",
            128
        )
    )


    dropout = float(
        checkpoint.get(
            "dropout",
            0.10
        )
    )


    model = TrajectoryTransformer(

        future_length=
            future_length,

        d_model=
            d_model,

        nhead=
            nhead,

        num_layers=
            num_layers,

        dim_feedforward=
            dim_feedforward,

        dropout=
            dropout

    ).to(
        DEVICE
    )


    # --------------------------------------------------------
    # Positional encoding is deterministic.
    #
    # Drop the saved PE buffer so checkpoint max_len
    # differences cannot cause a harmless shape mismatch.
    # --------------------------------------------------------

    state_dict = {
        key: value
        for key, value
        in state_dict.items()
        if key != "position.pe"
    }


    load_result = model.load_state_dict(

        state_dict,

        strict=False

    )


    allowed_missing = {
        "position.pe"
    }


    unexpected_missing = [

        key
        for key
        in load_result.missing_keys
        if key not in allowed_missing

    ]


    if (
        len(
            unexpected_missing
        ) > 0

        or

        len(
            load_result.unexpected_keys
        ) > 0
    ):

        raise RuntimeError(

            "\nCheckpoint architecture mismatch.\n"
            f"Missing keys: {unexpected_missing}\n"
            f"Unexpected keys: "
            f"{load_result.unexpected_keys}\n"

        )


    model.eval()


    return (
        model,
        checkpoint_file,
        checkpoint
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_model(
    model,
    history,
    batch_size=256
):


    predictions = []


    with torch.no_grad():


        for start in range(

            0,
            len(history),
            batch_size

        ):


            x = torch.tensor(

                history[
                    start:
                    start + batch_size
                ],

                dtype=torch.float32,

                device=DEVICE

            )


            prediction = model(
                x
            )


            predictions.append(

                prediction
                .cpu()
                .numpy()

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


    outputs = []


    for i in range(
        len(canonical)
    ):


        a = float(
            angle[i]
        )


        c = np.cos(a)
        s = np.sin(a)


        R = np.array(

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


        absolute = (

            local
            +
            origin[i]

        )


        outputs.append(
            absolute
        )


    return np.asarray(

        outputs,

        dtype=np.float64

    )


# ============================================================
# ADE / FDE METRICS
# ============================================================

def calculate_metrics(
    prediction,
    truth,
    metadata
):


    error = np.linalg.norm(

        prediction
        -
        truth,

        axis=2

    )


    ADE = error.mean(
        axis=1
    )


    FDE = error[
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
        "ADE"
    ] = ADE


    table[
        "FDE"
    ] = FDE


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

        "ADE_mean":
            float(
                ADE.mean()
            ),

        "FDE_mean":
            float(
                FDE.mean()
            ),

        "ADE_track_macro":
            float(
                track_table[
                    "ADE"
                ].mean()
            ),

        "FDE_track_macro":
            float(
                track_table[
                    "FDE"
                ].mean()
            ),

        "ADE_source_macro":
            float(
                source_table[
                    "ADE"
                ].mean()
            ),

        "FDE_source_macro":
            float(
                source_table[
                    "FDE"
                ].mean()
            ),

        "ADE_per_sample":
            ADE,

        "FDE_per_sample":
            FDE,

        "point_error":
            error

    }


# ============================================================
# CORRELATION HELPERS
# ============================================================

def safe_pearson(
    x,
    y
):


    x = np.asarray(
        x,
        dtype=np.float64
    ).reshape(
        -1
    )


    y = np.asarray(
        y,
        dtype=np.float64
    ).reshape(
        -1
    )


    mask = (

        np.isfinite(
            x
        )

        &

        np.isfinite(
            y
        )

    )


    x = x[
        mask
    ]

    y = y[
        mask
    ]


    if (
        len(x) < 3

        or

        np.std(x) < EPS

        or

        np.std(y) < EPS
    ):

        return np.nan


    return float(

        np.corrcoef(
            x,
            y
        )[
            0,
            1
        ]

    )


def safe_spearman(
    x,
    y
):


    x = np.asarray(
        x,
        dtype=np.float64
    ).reshape(
        -1
    )


    y = np.asarray(
        y,
        dtype=np.float64
    ).reshape(
        -1
    )


    mask = (

        np.isfinite(
            x
        )

        &

        np.isfinite(
            y
        )

    )


    x = x[
        mask
    ]

    y = y[
        mask
    ]


    if len(
        x
    ) < 3:

        return np.nan


    rank_x = (

        pd.Series(
            x
        )

        .rank(
            method="average"
        )

        .to_numpy(
            dtype=np.float64
        )

    )


    rank_y = (

        pd.Series(
            y
        )

        .rank(
            method="average"
        )

        .to_numpy(
            dtype=np.float64
        )

    )


    return safe_pearson(

        rank_x,

        rank_y

    )


# ============================================================
# MAIN
# ============================================================

all_seed_rows = []
ensemble_rows = []
uncertainty_rows = []
uncertainty_time_rows = []
uncertainty_bin_rows = []
selective_rows = []


for task in TASKS:


    print(
        "\n============================================"
    )

    print(
        "TASK:",
        task
    )

    print(
        "============================================"
    )


    # ========================================================
    # LOAD VALIDATION DATA ONLY
    # ========================================================

    npz_file = (

        DATA_ROOT
        /
        f"{task}_birdsai_trainval_canonical.npz"

    )


    metadata_file = (

        DATA_ROOT
        /
        f"{task}_birdsai_trainval_metadata.csv"

    )


    arrays = np.load(
        npz_file
    )


    metadata = pd.read_csv(

        metadata_file,

        dtype={
            "source_group": str,
            "track_key": str
        }

    )


    validation_indices = arrays[
        "validation_indices"
    ].astype(
        int
    )


    history = arrays[
        "history_canonical"
    ][
        validation_indices
    ].astype(
        np.float32
    )


    truth_abs = arrays[
        "future_abs"
    ][
        validation_indices
    ].astype(
        np.float64
    )


    origin = arrays[
        "origin"
    ][
        validation_indices
    ]


    angle = arrays[
        "heading_angle"
    ][
        validation_indices
    ]


    scale = arrays[
        "scale"
    ][
        validation_indices
    ]


    val_metadata = (

        metadata.iloc[
            validation_indices
        ]

        .reset_index(
            drop=True
        )

    )


    future_length = (
        truth_abs.shape[
            1
        ]
    )


    # ========================================================
    # LOAD FROZEN SEEDS
    # ========================================================

    canonical_seed_predictions = []
    absolute_seed_predictions = []


    for seed in SEEDS:


        model, checkpoint_file, checkpoint = (
            load_transformer(

                task,
                seed,
                future_length

            )
        )


        prediction_canonical = (
            predict_model(

                model,

                history

            )
        )


        prediction_abs = (
            inverse_transform_batch(

                prediction_canonical,

                origin,

                angle,

                scale

            )
        )


        canonical_seed_predictions.append(
            prediction_canonical
        )


        absolute_seed_predictions.append(
            prediction_abs
        )


        metrics = calculate_metrics(

            prediction_abs,

            truth_abs,

            val_metadata

        )


        all_seed_rows.append({

            "task":
                task,

            "seed":
                seed,

            "checkpoint":
                checkpoint_file.name,

            "best_epoch":
                checkpoint.get(
                    "best_epoch",
                    np.nan
                ),

            "validation_samples":
                len(
                    validation_indices
                ),

            "validation_tracks":
                val_metadata[
                    "track_key"
                ].nunique(),

            "validation_sources":
                val_metadata[
                    "source_group"
                ].nunique(),

            "ADE_mean":
                metrics[
                    "ADE_mean"
                ],

            "FDE_mean":
                metrics[
                    "FDE_mean"
                ],

            "ADE_track_macro":
                metrics[
                    "ADE_track_macro"
                ],

            "FDE_track_macro":
                metrics[
                    "FDE_track_macro"
                ],

            "ADE_source_macro":
                metrics[
                    "ADE_source_macro"
                ],

            "FDE_source_macro":
                metrics[
                    "FDE_source_macro"
                ]

        })


        print(

            f"Seed {seed}: "
            f"ADE={metrics['ADE_mean']:.6f}, "
            f"FDE={metrics['FDE_mean']:.6f}, "
            f"track-ADE="
            f"{metrics['ADE_track_macro']:.6f}"

        )


    # ========================================================
    # STACK SEED PREDICTIONS
    # ========================================================

    canonical_seed_predictions = np.stack(

        canonical_seed_predictions,

        axis=0

    )


    absolute_seed_predictions = np.stack(

        absolute_seed_predictions,

        axis=0

    )


    # Shape:
    # [seed, sample, future_step, xy]


    # ========================================================
    # ENSEMBLE MEAN
    # ========================================================

    ensemble_abs = (

        absolute_seed_predictions.mean(
            axis=0
        )

    )


    ensemble_metrics = calculate_metrics(

        ensemble_abs,

        truth_abs,

        val_metadata

    )


    # ========================================================
    # SEED-DISAGREEMENT UNCERTAINTY
    # ========================================================

    deviations = (

        absolute_seed_predictions

        -
        ensemble_abs[
            None,
            ...,
        ]

    )


    squared_radial_deviation = np.sum(

        deviations ** 2,

        axis=-1

    )


    # RMS radial disagreement among the 3 seeds
    # [sample, time]

    disagreement = np.sqrt(

        squared_radial_deviation.mean(
            axis=0
        )

    )


    actual_error = (
        ensemble_metrics[
            "point_error"
        ]
    )


    sample_uncertainty = disagreement.mean(
        axis=1
    )


    sample_ADE = actual_error.mean(
        axis=1
    )


    endpoint_uncertainty = disagreement[
        :,
        -1
    ]


    endpoint_error = actual_error[
        :,
        -1
    ]


    # ========================================================
    # SINGLE-SEED MEAN VS ENSEMBLE
    # ========================================================

    task_seed_table = pd.DataFrame(
        [
            row
            for row in all_seed_rows
            if row[
                "task"
            ] == task
        ]
    )


    single_ADE_mean = float(

        task_seed_table[
            "ADE_mean"
        ].mean()

    )


    single_FDE_mean = float(

        task_seed_table[
            "FDE_mean"
        ].mean()

    )


    single_track_ADE_mean = float(

        task_seed_table[
            "ADE_track_macro"
        ].mean()

    )


    single_track_FDE_mean = float(

        task_seed_table[
            "FDE_track_macro"
        ].mean()

    )


    def improvement_percent(
        baseline,
        ensemble
    ):

        return (

            100.0

            *
            (
                baseline
                -
                ensemble
            )

            /
            baseline

        )


    ensemble_rows.append({

        "task":
            task,

        "ensemble":
            "Transformer-3Seed-Mean",

        "seeds":
            len(
                SEEDS
            ),

        "validation_samples":
            len(
                validation_indices
            ),

        "validation_tracks":
            val_metadata[
                "track_key"
            ].nunique(),

        "validation_sources":
            val_metadata[
                "source_group"
            ].nunique(),

        "single_seed_ADE_mean":
            single_ADE_mean,

        "ensemble_ADE":
            ensemble_metrics[
                "ADE_mean"
            ],

        "ADE_improvement_pct":
            improvement_percent(

                single_ADE_mean,

                ensemble_metrics[
                    "ADE_mean"
                ]

            ),

        "single_seed_FDE_mean":
            single_FDE_mean,

        "ensemble_FDE":
            ensemble_metrics[
                "FDE_mean"
            ],

        "FDE_improvement_pct":
            improvement_percent(

                single_FDE_mean,

                ensemble_metrics[
                    "FDE_mean"
                ]

            ),

        "single_seed_track_ADE_mean":
            single_track_ADE_mean,

        "ensemble_track_ADE":
            ensemble_metrics[
                "ADE_track_macro"
            ],

        "track_ADE_improvement_pct":
            improvement_percent(

                single_track_ADE_mean,

                ensemble_metrics[
                    "ADE_track_macro"
                ]

            ),

        "single_seed_track_FDE_mean":
            single_track_FDE_mean,

        "ensemble_track_FDE":
            ensemble_metrics[
                "FDE_track_macro"
            ],

        "track_FDE_improvement_pct":
            improvement_percent(

                single_track_FDE_mean,

                ensemble_metrics[
                    "FDE_track_macro"
                ]

            ),

        "ensemble_source_ADE":
            ensemble_metrics[
                "ADE_source_macro"
            ],

        "ensemble_source_FDE":
            ensemble_metrics[
                "FDE_source_macro"
            ]

    })


    # ========================================================
    # OVERALL UNCERTAINTY CORRELATION
    # ========================================================

    point_pearson = safe_pearson(

        disagreement.reshape(
            -1
        ),

        actual_error.reshape(
            -1
        )

    )


    point_spearman = safe_spearman(

        disagreement.reshape(
            -1
        ),

        actual_error.reshape(
            -1
        )

    )


    sample_pearson = safe_pearson(

        sample_uncertainty,

        sample_ADE

    )


    sample_spearman = safe_spearman(

        sample_uncertainty,

        sample_ADE

    )


    endpoint_pearson = safe_pearson(

        endpoint_uncertainty,

        endpoint_error

    )


    endpoint_spearman = safe_spearman(

        endpoint_uncertainty,

        endpoint_error

    )


    # top / bottom uncertainty comparison

    order = np.argsort(
        sample_uncertainty
    )


    n = len(
        order
    )


    q = max(
        1,
        int(
            round(
                0.20
                *
                n
            )
        )
    )


    bottom_idx = order[
        :q
    ]


    top_idx = order[
        -q:
    ]


    bottom20_error = float(

        sample_ADE[
            bottom_idx
        ].mean()

    )


    top20_error = float(

        sample_ADE[
            top_idx
        ].mean()

    )


    uncertainty_rows.append({

        "task":
            task,

        "samples":
            len(
                sample_ADE
            ),

        "future_steps":
            future_length,

        "mean_disagreement":
            float(
                disagreement.mean()
            ),

        "median_disagreement":
            float(
                np.median(
                    disagreement
                )
            ),

        "pointwise_Pearson":
            point_pearson,

        "pointwise_Spearman":
            point_spearman,

        "sample_ADE_Pearson":
            sample_pearson,

        "sample_ADE_Spearman":
            sample_spearman,

        "endpoint_FDE_Pearson":
            endpoint_pearson,

        "endpoint_FDE_Spearman":
            endpoint_spearman,

        "bottom20_uncertainty_ADE":
            bottom20_error,

        "top20_uncertainty_ADE":
            top20_error,

        "top_bottom_error_ratio":
            (

                top20_error
                /
                bottom20_error

                if bottom20_error
                > EPS

                else np.nan

            )

    })


    # ========================================================
    # UNCERTAINTY BY FORECAST TIME
    # ========================================================

    for step in range(
        future_length
    ):


        step_uncertainty = disagreement[
            :,
            step
        ]


        step_error = actual_error[
            :,
            step
        ]


        uncertainty_time_rows.append({

            "task":
                task,

            "forecast_step":
                step + 1,

            "forecast_time_sec":
                (
                    step + 1
                )
                /
                FPS,

            "mean_disagreement":
                float(
                    step_uncertainty.mean()
                ),

            "median_disagreement":
                float(
                    np.median(
                        step_uncertainty
                    )
                ),

            "mean_error":
                float(
                    step_error.mean()
                ),

            "median_error":
                float(
                    np.median(
                        step_error
                    )
                ),

            "Pearson":
                safe_pearson(

                    step_uncertainty,

                    step_error

                ),

            "Spearman":
                safe_spearman(

                    step_uncertainty,

                    step_error

                )

        })


    # ========================================================
    # UNCERTAINTY QUINTILES
    # ========================================================

    uncertainty_frame = pd.DataFrame({

        "uncertainty":
            sample_uncertainty,

        "ADE":
            sample_ADE

    })


    # rank first avoids qcut problems caused by tied values

    uncertainty_frame[
        "rank"
    ] = (

        uncertainty_frame[
            "uncertainty"
        ]

        .rank(
            method="first"
        )

    )


    uncertainty_frame[
        "uncertainty_bin"
    ] = pd.qcut(

        uncertainty_frame[
            "rank"
        ],

        q=5,

        labels=[
            "Q1_lowest",
            "Q2",
            "Q3",
            "Q4",
            "Q5_highest"
        ]

    )


    for bin_name, group in (
        uncertainty_frame.groupby(

            "uncertainty_bin",

            observed=False

        )
    ):


        uncertainty_bin_rows.append({

            "task":
                task,

            "uncertainty_bin":
                str(
                    bin_name
                ),

            "samples":
                len(
                    group
                ),

            "mean_uncertainty":
                float(
                    group[
                        "uncertainty"
                    ].mean()
                ),

            "mean_ADE":
                float(
                    group[
                        "ADE"
                    ].mean()
                ),

            "median_ADE":
                float(
                    group[
                        "ADE"
                    ].median()
                )

        })


    # ========================================================
    # SELECTIVE RISK
    #
    # If uncertainty is useful, retaining only low-uncertainty
    # samples should reduce average error.
    # ========================================================

    ascending = np.argsort(
        sample_uncertainty
    )


    for retain_fraction in [

        1.00,
        0.90,
        0.80,
        0.70,
        0.50

    ]:


        keep = max(

            1,

            int(

                round(

                    retain_fraction

                    *
                    len(
                        ascending
                    )

                )

            )

        )


        selected = ascending[
            :keep
        ]


        selective_rows.append({

            "task":
                task,

            "retained_fraction":
                retain_fraction,

            "retained_samples":
                keep,

            "mean_ADE":
                float(

                    sample_ADE[
                        selected
                    ].mean()

                ),

            "mean_uncertainty":
                float(

                    sample_uncertainty[
                        selected
                    ].mean()

                )

        })


    print(
        "\nEnsemble:"
    )

    print(

        f"ADE="
        f"{ensemble_metrics['ADE_mean']:.6f} | "

        f"FDE="
        f"{ensemble_metrics['FDE_mean']:.6f} | "

        f"track-ADE="
        f"{ensemble_metrics['ADE_track_macro']:.6f}"

    )


    print(
        "\nUncertainty diagnostic:"
    )

    print(

        f"Point Spearman = "
        f"{point_spearman:.4f}"

    )

    print(

        f"Sample ADE Spearman = "
        f"{sample_spearman:.4f}"

    )

    print(

        f"Endpoint FDE Spearman = "
        f"{endpoint_spearman:.4f}"

    )


# ============================================================
# SAVE TABLES
# ============================================================

seed_df = pd.DataFrame(
    all_seed_rows
)


ensemble_df = pd.DataFrame(
    ensemble_rows
)


uncertainty_df = pd.DataFrame(
    uncertainty_rows
)


uncertainty_time_df = pd.DataFrame(
    uncertainty_time_rows
)


uncertainty_bins_df = pd.DataFrame(
    uncertainty_bin_rows
)


selective_df = pd.DataFrame(
    selective_rows
)


seed_df.to_csv(

    OUT /
    "transformer_ensemble_seed_metrics.csv",

    index=False

)


ensemble_df.to_csv(

    OUT /
    "transformer_ensemble_validation_summary.csv",

    index=False

)


uncertainty_df.to_csv(

    OUT /
    "transformer_ensemble_uncertainty_summary.csv",

    index=False

)


uncertainty_time_df.to_csv(

    OUT /
    "transformer_ensemble_uncertainty_by_time.csv",

    index=False

)


uncertainty_bins_df.to_csv(

    OUT /
    "transformer_ensemble_uncertainty_bins.csv",

    index=False

)


selective_df.to_csv(

    OUT /
    "transformer_ensemble_selective_risk.csv",

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
        "FROZEN TRANSFORMER 3-SEED ENSEMBLE — "
        "VALIDATION ONLY\n"
    )

    f.write(
        "=" * 82
        +
        "\n\n"
    )


    f.write(
        "PROTOCOL\n"
    )

    f.write(
        "- No retraining performed.\n"
    )

    f.write(
        "- Frozen Compact Temporal Transformer checkpoints "
        "only.\n"
    )

    f.write(
        "- Seeds: 42, 52, 62.\n"
    )

    f.write(
        "- Evaluation: fixed BIRDSAI TrainReal "
        "track-disjoint validation only.\n"
    )

    f.write(
        "- BIRDSAI TestReal NOT loaded.\n"
    )

    f.write(
        "- WildBox NOT loaded.\n"
    )

    f.write(
        "- Ensemble prediction = arithmetic mean of "
        "three frozen seed predictions.\n"
    )

    f.write(
        "- Uncertainty diagnostic = RMS spatial "
        "disagreement among the three seed predictions.\n"
    )

    f.write(
        "- Seed disagreement is diagnostic only; "
        "it is NOT claimed as calibrated probabilistic "
        "uncertainty.\n\n"
    )


    f.write(
        "ENSEMBLE VALIDATION SUMMARY\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        ensemble_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nUNCERTAINTY SUMMARY\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        uncertainty_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nSEED-LEVEL METRICS\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        seed_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nSELECTIVE RISK\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        selective_df.to_string(
            index=False
        )
    )


print(
    "\n============================================"
)

print(
    "STEP 16 COMPLETE"
)

print(
    "============================================"
)


print(
    "\nEnsemble validation summary:"
)

print(
    ensemble_df.to_string(
        index=False
    )
)


print(
    "\nUncertainty summary:"
)

print(
    uncertainty_df.to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    OUT
)