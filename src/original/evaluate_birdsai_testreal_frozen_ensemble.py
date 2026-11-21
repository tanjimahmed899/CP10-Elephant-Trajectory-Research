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
    r"C:\Users\HP\Downloads\birdsai_testreal_frozen_ensemble"
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
        "samples": 1053,
        "tracks": 89,
        "sources": 6,
        "future_frames": 20,
    },

    "2s_to_5s": {
        "samples": 826,
        "tracks": 63,
        "sources": 5,
        "future_frames": 50,
    },

}

FPS = 10.0

EPS = 1e-12


# ============================================================
# IMPORTANT PROTOCOL
# ============================================================

PROTOCOL_LABEL = (
    "Frozen in-domain BIRDSAI TestReal benchmark. "
    "TestReal was previously inspected during exploratory "
    "development and is therefore NOT described as an "
    "untouched confirmatory test set."
)


# ============================================================
# FIND CORRECT TESTREAL DATA WITHOUT GUESSING FILENAMES
# ============================================================

REQUIRED_NPZ_KEYS = {
    "history_canonical",
    "future_canonical",
    "history_abs",
    "future_abs",
    "origin",
    "heading_angle",
    "scale",
}


def locate_test_npz(
    task,
    expected_samples,
    expected_future
):

    candidates = []


    for path in DATA_ROOT.glob("*.npz"):

        lower_name = path.name.lower()


        # Never accidentally use development/external files
        if any(
            token in lower_name
            for token in [
                "trainval",
                "wildbox",
                "external",
                "validation",
            ]
        ):
            continue


        try:

            arrays = np.load(
                path,
                allow_pickle=False
            )

        except Exception:

            continue


        keys = set(
            arrays.files
        )


        if not REQUIRED_NPZ_KEYS.issubset(
            keys
        ):

            continue


        history = arrays[
            "history_canonical"
        ]

        future = arrays[
            "future_canonical"
        ]


        if history.shape != (
            expected_samples,
            20,
            2
        ):

            continue


        if future.shape != (
            expected_samples,
            expected_future,
            2
        ):

            continue


        candidates.append(
            path
        )


    if len(candidates) != 1:

        print(
            "\nNPZ candidates found:"
        )

        for candidate in candidates:

            print(
                " -",
                candidate
            )


        raise RuntimeError(

            f"\nExpected exactly ONE BIRDSAI TestReal NPZ "
            f"for {task}, but found {len(candidates)}.\n"
            f"No file was guessed or selected automatically."

        )


    return candidates[0]


def locate_test_metadata(
    task,
    expected_samples,
    expected_tracks,
    expected_sources
):

    candidates = []


    for path in DATA_ROOT.glob("*.csv"):

        lower_name = path.name.lower()


        if any(
            token in lower_name
            for token in [
                "trainval",
                "wildbox",
                "external",
                "validation",
            ]
        ):
            continue


        try:

            table = pd.read_csv(
                path,
                dtype={
                    "source_group": str,
                    "track_key": str,
                }
            )

        except Exception:

            continue


        required = {
            "source_group",
            "track_key",
        }


        if not required.issubset(
            table.columns
        ):

            continue


        if len(
            table
        ) != expected_samples:

            continue


        sources = table[
            "source_group"
        ].nunique()


        tracks = (
            table[
                [
                    "source_group",
                    "track_key"
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )


        if sources != expected_sources:

            continue


        if tracks != expected_tracks:

            continue


        candidates.append(
            path
        )


    if len(candidates) != 1:

        print(
            "\nMetadata candidates found:"
        )

        for candidate in candidates:

            print(
                " -",
                candidate
            )


        raise RuntimeError(

            f"\nExpected exactly ONE BIRDSAI TestReal metadata "
            f"CSV for {task}, but found {len(candidates)}.\n"
            f"No metadata file was guessed."

        )


    return candidates[0]


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


        pe[
            :,
            0::2
        ] = torch.sin(

            position
            *
            div_term

        )


        pe[
            :,
            1::2
        ] = torch.cos(

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
# EXACT FROZEN TRANSFORMER ARCHITECTURE
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
# LOAD FROZEN CHECKPOINT
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

            f"\nFrozen checkpoint missing:\n"
            f"{checkpoint_file}"

        )


    checkpoint = torch.load(

        checkpoint_file,

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

        future_length=
            future_length,

        d_model=
            int(
                checkpoint.get(
                    "d_model",
                    64
                )
            ),

        nhead=
            int(
                checkpoint.get(
                    "nhead",
                    4
                )
            ),

        num_layers=
            int(
                checkpoint.get(
                    "num_layers",
                    2
                )
            ),

        dim_feedforward=
            int(
                checkpoint.get(
                    "dim_feedforward",
                    128
                )
            ),

        dropout=
            float(
                checkpoint.get(
                    "dropout",
                    0.10
                )
            ),

    ).to(
        DEVICE
    )


    # Positional encoding is deterministic,
    # so do not require saved PE buffer to match shape.

    state_dict = {

        key: value

        for key, value
        in state_dict.items()

        if key != "position.pe"

    }


    result = model.load_state_dict(

        state_dict,

        strict=False

    )


    missing = [

        key

        for key in result.missing_keys

        if key != "position.pe"

    ]


    if (
        missing
        or
        result.unexpected_keys
    ):

        raise RuntimeError(

            "\nCheckpoint architecture mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {result.unexpected_keys}"

        )


    model.eval()


    return (
        model,
        checkpoint_file,
        checkpoint
    )


# ============================================================
# PREDICT
# ============================================================

def predict_model(
    model,
    history,
    batch_size=256
):


    outputs = []


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


            outputs.append(

                prediction
                .cpu()
                .numpy()

            )


    return np.concatenate(

        outputs,

        axis=0

    )


# ============================================================
# EXACT INVERSE CANONICAL TRANSFORM
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
                [-s, c],
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


        output.append(
            absolute
        )


    return np.asarray(

        output,

        dtype=np.float64

    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    truth,
    metadata
):


    point_error = np.linalg.norm(

        prediction
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


    frame = metadata[
        [
            "source_group",
            "track_key"
        ]
    ].copy()


    frame[
        "ADE"
    ] = ADE


    frame[
        "FDE"
    ] = FDE


    track_table = (

        frame.groupby(

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

        frame.groupby(

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
            float(
                ADE.mean()
            ),

        "FDE":
            float(
                FDE.mean()
            ),

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

        "point_error":
            point_error,

        "sample_ADE":
            ADE,

        "sample_FDE":
            FDE,

    }


# ============================================================
# CLASSICAL BASELINES
# ============================================================

def stationary_prediction(
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


def constant_velocity_prediction(
    history,
    future_length
):


    T = len(
        history
    )


    t = np.arange(
        T,
        dtype=np.float64
    )


    tc = (
        t
        -
        t.mean()
    )


    denom = np.sum(
        tc ** 2
    )


    centered = (

        history

        -
        history.mean(
            axis=0,
            keepdims=True
        )

    )


    velocity = (

        (
            centered

            *
            tc[
                :,
                None
            ]

        ).sum(
            axis=0
        )

        /
        denom

    )


    future_steps = np.arange(

        1,
        future_length + 1,

        dtype=np.float64

    )[
        :,
        None
    ]


    return (

        history[
            -1
        ]

        +
        future_steps
        *
        velocity

    )


# ============================================================
# CORRELATION
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
        np.std(x) <= EPS

        or
        np.std(y) <= EPS
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


    x_rank = (

        pd.Series(
            np.asarray(
                x
            ).reshape(
                -1
            )
        )

        .rank(
            method="average"
        )

        .to_numpy()

    )


    y_rank = (

        pd.Series(
            np.asarray(
                y
            ).reshape(
                -1
            )
        )

        .rank(
            method="average"
        )

        .to_numpy()

    )


    return safe_pearson(

        x_rank,

        y_rank

    )


# ============================================================
# RUN
# ============================================================

seed_rows = []

ensemble_rows = []

baseline_rows = []

source_rows = []

uncertainty_rows = []

data_audit_rows = []


for task, config in TASKS.items():


    print(
        "\n"
        +
        "=" * 82
    )

    print(
        "FROZEN BIRDSAI TESTREAL BENCHMARK:",
        task
    )

    print(
        "=" * 82
    )


    # ========================================================
    # FIND DATA SAFELY
    # ========================================================

    npz_file = locate_test_npz(

        task,

        config[
            "samples"
        ],

        config[
            "future_frames"
        ]

    )


    metadata_file = locate_test_metadata(

        task,

        config[
            "samples"
        ],

        config[
            "tracks"
        ],

        config[
            "sources"
        ]

    )


    print(
        "\nNPZ:"
    )

    print(
        npz_file
    )


    print(
        "\nMetadata:"
    )

    print(
        metadata_file
    )


    arrays = np.load(

        npz_file,

        allow_pickle=False

    )


    metadata = pd.read_csv(

        metadata_file,

        dtype={
            "source_group": str,
            "track_key": str,
        }

    )


    history_can = arrays[
        "history_canonical"
    ].astype(
        np.float32
    )


    history_abs = arrays[
        "history_abs"
    ].astype(
        np.float64
    )


    truth_abs = arrays[
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


    # ========================================================
    # FINAL SAFETY AUDIT
    # ========================================================

    n_samples = len(
        metadata
    )


    n_tracks = (

        metadata[
            [
                "source_group",
                "track_key"
            ]
        ]

        .drop_duplicates()

        .shape[0]

    )


    n_sources = metadata[
        "source_group"
    ].nunique()


    assert (
        n_samples
        ==
        config[
            "samples"
        ]
    )


    assert (
        n_tracks
        ==
        config[
            "tracks"
        ]
    )


    assert (
        n_sources
        ==
        config[
            "sources"
        ]
    )


    assert history_can.shape == (
        config[
            "samples"
        ],
        20,
        2
    )


    assert truth_abs.shape == (
        config[
            "samples"
        ],
        config[
            "future_frames"
        ],
        2
    )


    data_audit_rows.append({

        "task":
            task,

        "npz_file":
            npz_file.name,

        "metadata_file":
            metadata_file.name,

        "samples":
            n_samples,

        "tracks":
            n_tracks,

        "sources":
            n_sources,

        "status":
            "PASS",

    })


    # ========================================================
    # FROZEN SEED PREDICTIONS
    # ========================================================

    seed_predictions_abs = []


    for seed in SEEDS:


        (
            model,
            checkpoint_file,
            checkpoint

        ) = load_transformer(

            task,

            seed,

            config[
                "future_frames"
            ]

        )


        prediction_can = predict_model(

            model,

            history_can

        )


        prediction_abs = inverse_transform_batch(

            prediction_can,

            origin,

            angle,

            scale

        )


        seed_predictions_abs.append(
            prediction_abs
        )


        metrics = calculate_metrics(

            prediction_abs,

            truth_abs,

            metadata

        )


        seed_rows.append({

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

            "ADE":
                metrics[
                    "ADE"
                ],

            "FDE":
                metrics[
                    "FDE"
                ],

            "track_macro_ADE":
                metrics[
                    "track_macro_ADE"
                ],

            "track_macro_FDE":
                metrics[
                    "track_macro_FDE"
                ],

            "source_macro_ADE":
                metrics[
                    "source_macro_ADE"
                ],

            "source_macro_FDE":
                metrics[
                    "source_macro_FDE"
                ],

        })


    seed_predictions_abs = np.stack(

        seed_predictions_abs,

        axis=0

    )


    # ========================================================
    # FINAL FROZEN 3-SEED ENSEMBLE
    # ========================================================

    ensemble_prediction = (

        seed_predictions_abs.mean(
            axis=0
        )

    )


    ensemble_metrics = calculate_metrics(

        ensemble_prediction,

        truth_abs,

        metadata

    )


    ensemble_rows.append({

        "task":
            task,

        "model":
            "Frozen_Transformer_3Seed_Ensemble",

        "samples":
            n_samples,

        "tracks":
            n_tracks,

        "sources":
            n_sources,

        "ADE":
            ensemble_metrics[
                "ADE"
            ],

        "FDE":
            ensemble_metrics[
                "FDE"
            ],

        "track_macro_ADE":
            ensemble_metrics[
                "track_macro_ADE"
            ],

        "track_macro_FDE":
            ensemble_metrics[
                "track_macro_FDE"
            ],

        "source_macro_ADE":
            ensemble_metrics[
                "source_macro_ADE"
            ],

        "source_macro_FDE":
            ensemble_metrics[
                "source_macro_FDE"
            ],

    })


    # ========================================================
    # CLASSICAL COMPARISON — SAME DATA
    # ========================================================

    stationary = np.stack([

        stationary_prediction(

            h,

            config[
                "future_frames"
            ]

        )

        for h in history_abs

    ])


    constant_velocity = np.stack([

        constant_velocity_prediction(

            h,

            config[
                "future_frames"
            ]

        )

        for h in history_abs

    ])


    for model_name, prediction in [

        (
            "Stationary",
            stationary
        ),

        (
            "ConstantVelocity",
            constant_velocity
        ),

    ]:


        metrics = calculate_metrics(

            prediction,

            truth_abs,

            metadata

        )


        baseline_rows.append({

            "task":
                task,

            "model":
                model_name,

            "ADE":
                metrics[
                    "ADE"
                ],

            "FDE":
                metrics[
                    "FDE"
                ],

            "track_macro_ADE":
                metrics[
                    "track_macro_ADE"
                ],

            "track_macro_FDE":
                metrics[
                    "track_macro_FDE"
                ],

            "source_macro_ADE":
                metrics[
                    "source_macro_ADE"
                ],

            "source_macro_FDE":
                metrics[
                    "source_macro_FDE"
                ],

        })


    # ========================================================
    # PER-SOURCE ENSEMBLE RESULT
    # ========================================================

    source_values = metadata[
        "source_group"
    ].to_numpy()


    for source in sorted(
        metadata[
            "source_group"
        ].unique()
    ):


        mask = (
            source_values
            ==
            source
        )


        source_metadata = (

            metadata.loc[
                mask
            ]

            .reset_index(
                drop=True
            )

        )


        metrics = calculate_metrics(

            ensemble_prediction[
                mask
            ],

            truth_abs[
                mask
            ],

            source_metadata

        )


        source_rows.append({

            "task":
                task,

            "source_group":
                source,

            "samples":
                int(
                    mask.sum()
                ),

            "tracklets":
                source_metadata[
                    [
                        "source_group",
                        "track_key"
                    ]
                ]
                .drop_duplicates()
                .shape[0],

            "ADE":
                metrics[
                    "ADE"
                ],

            "FDE":
                metrics[
                    "FDE"
                ],

            "track_macro_ADE":
                metrics[
                    "track_macro_ADE"
                ],

            "track_macro_FDE":
                metrics[
                    "track_macro_FDE"
                ],

        })


    # ========================================================
    # ENSEMBLE DISAGREEMENT DIAGNOSTIC
    # ========================================================

    deviations = (

        seed_predictions_abs

        -
        ensemble_prediction[
            None,
            ...
        ]

    )


    disagreement = np.sqrt(

        np.mean(

            np.sum(
                deviations ** 2,
                axis=-1
            ),

            axis=0

        )

    )


    actual_error = ensemble_metrics[
        "point_error"
    ]


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


    endpoint_FDE = actual_error[
        :,
        -1
    ]


    uncertainty_rows.append({

        "task":
            task,

        "samples":
            n_samples,

        "mean_disagreement":
            float(
                disagreement.mean()
            ),

        "pointwise_Pearson":
            safe_pearson(

                disagreement.reshape(
                    -1
                ),

                actual_error.reshape(
                    -1
                )

            ),

        "pointwise_Spearman":
            safe_spearman(

                disagreement.reshape(
                    -1
                ),

                actual_error.reshape(
                    -1
                )

            ),

        "sample_ADE_Pearson":
            safe_pearson(

                sample_uncertainty,

                sample_ADE

            ),

        "sample_ADE_Spearman":
            safe_spearman(

                sample_uncertainty,

                sample_ADE

            ),

        "endpoint_FDE_Pearson":
            safe_pearson(

                endpoint_uncertainty,

                endpoint_FDE

            ),

        "endpoint_FDE_Spearman":
            safe_spearman(

                endpoint_uncertainty,

                endpoint_FDE

            ),

    })


# ============================================================
# SAVE
# ============================================================

seed_df = pd.DataFrame(
    seed_rows
)


ensemble_df = pd.DataFrame(
    ensemble_rows
)


baseline_df = pd.DataFrame(
    baseline_rows
)


source_df = pd.DataFrame(
    source_rows
)


uncertainty_df = pd.DataFrame(
    uncertainty_rows
)


audit_df = pd.DataFrame(
    data_audit_rows
)


seed_df.to_csv(

    OUT /
    "testreal_seed_metrics.csv",

    index=False

)


ensemble_df.to_csv(

    OUT /
    "testreal_ensemble_metrics.csv",

    index=False

)


baseline_df.to_csv(

    OUT /
    "testreal_classical_baselines.csv",

    index=False

)


source_df.to_csv(

    OUT /
    "testreal_per_source_metrics.csv",

    index=False

)


uncertainty_df.to_csv(

    OUT /
    "testreal_uncertainty_diagnostic.csv",

    index=False

)


audit_df.to_csv(

    OUT /
    "testreal_data_audit.csv",

    index=False

)


# ============================================================
# COMBINED MODEL TABLE
# ============================================================

ensemble_for_table = (

    ensemble_df[
        [
            "task",
            "model",
            "ADE",
            "FDE",
            "track_macro_ADE",
            "track_macro_FDE",
            "source_macro_ADE",
            "source_macro_FDE",
        ]
    ]
)


combined = pd.concat(

    [
        ensemble_for_table,
        baseline_df
    ],

    ignore_index=True

)


combined.to_csv(

    OUT /
    "testreal_model_comparison.csv",

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
        "BIRDSAI TESTREAL — FROZEN IN-DOMAIN BENCHMARK\n"
    )

    f.write(
        "=" * 82
        +
        "\n\n"
    )


    f.write(
        "INTERPRETATION STATUS\n"
    )

    f.write(
        "- This is a frozen-model in-domain benchmark.\n"
    )

    f.write(
        "- BIRDSAI TestReal was previously inspected during "
        "exploratory development.\n"
    )

    f.write(
        "- Therefore it is NOT described as an untouched or "
        "preregistered confirmatory test set.\n"
    )

    f.write(
        "- No retraining, architecture selection, or "
        "hyperparameter adjustment was performed in Step 17C.\n"
    )

    f.write(
        "- The genuinely preregistered confirmatory external "
        "evaluation remains the fresh WildBox holdout.\n\n"
    )


    f.write(
        "FROZEN MODEL\n"
    )

    f.write(
        "- Compact Temporal Transformer.\n"
    )

    f.write(
        "- Seeds: 42, 52, 62.\n"
    )

    f.write(
        "- Ensemble prediction: arithmetic mean of the "
        "three frozen seed predictions.\n"
    )

    f.write(
        "- Reported ADE/FDE are camera-corrected normalized "
        "image-plane coordinate errors, NOT metres.\n"
    )

    f.write(
        "- Seed disagreement is diagnostic only; no calibrated "
        "uncertainty claim is made.\n\n"
    )


    f.write(
        "DATA AUDIT\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        audit_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nENSEMBLE METRICS\n"
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
        "\n\nMODEL COMPARISON\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        combined.to_string(
            index=False
        )
    )


    f.write(
        "\n\nSEED METRICS\n"
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
        "\n\nPER-SOURCE ENSEMBLE METRICS\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        source_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nENSEMBLE DISAGREEMENT DIAGNOSTIC\n"
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


print(
    "\n"
    +
    "=" * 82
)

print(
    "STEP 17C COMPLETE"
)

print(
    "=" * 82
)


print(
    "\nProtocol:"
)

print(
    PROTOCOL_LABEL
)


print(
    "\nData audit:"
)

print(
    audit_df.to_string(
        index=False
    )
)


print(
    "\nFrozen ensemble:"
)

print(
    ensemble_df.to_string(
        index=False
    )
)


print(
    "\nComparison:"
)

print(
    combined.to_string(
        index=False
    )
)


print(
    "\nUncertainty diagnostic:"
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