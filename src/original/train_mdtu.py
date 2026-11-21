from pathlib import Path
import copy
import math
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\HP\Downloads\final_neural_trajectory_dataset"
)

RNN_DIR = Path(
    r"C:\Users\HP\Downloads\lstm_gru_results"
)

TRANSFORMER_DIR = Path(
    r"C:\Users\HP\Downloads\transformer_results"
)

V2_DIR = Path(
    r"C:\Users\HP\Downloads\rmpgu_v2_fixed_validation"
)

OUT = Path(
    r"C:\Users\HP\Downloads\mdtu_validation"
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

BATCH_SIZE = 64

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.10

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 300
PATIENCE = 30
MIN_DELTA = 1e-5
GRAD_CLIP = 5.0


# ============================================================
# FIXED LOSS WEIGHTS
#
# No TestReal/WildBox tuning.
# ============================================================

LAMBDA_FDE = 0.50
LAMBDA_INCREMENT = 0.25
LAMBDA_ACCELERATION = 0.10
LAMBDA_NLL = 0.10


# ============================================================
# UNCERTAINTY
# ============================================================

CHI2_50 = 1.38629436112
CHI2_90 = 4.60517018599
CHI2_95 = 5.99146454711


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True
    )


# ============================================================
# FIRST + SECOND MOTION DIFFERENCES
# ============================================================

def build_derivative_input(history):

    """
    history:
        [N,T,2] canonical trajectory

    Returns:
        [N,T,6]

        [x,y,dx,dy,ddx,ddy]

    dx/ddx are discrete trajectory differences,
    not claimed as metric velocity/acceleration.
    """

    first = np.zeros_like(
        history,
        dtype=np.float32
    )

    first[:, 1:, :] = (

        history[:, 1:, :]
        -
        history[:, :-1, :]

    )


    second = np.zeros_like(
        history,
        dtype=np.float32
    )

    second[:, 1:, :] = (

        first[:, 1:, :]
        -
        first[:, :-1, :]

    )


    features = np.concatenate(

        [
            history.astype(
                np.float32
            ),
            first,
            second
        ],

        axis=2

    )


    return features.astype(
        np.float32
    )


# ============================================================
# FUTURE INCREMENT TARGET
# ============================================================

def future_to_increments(
    history,
    future
):

    """
    history [N,obs,2]
    future  [N,future,2]

    First future increment is measured from
    last observed point.
    """

    last_observation = history[
        :,
        -1:,
        :
    ]


    previous = np.concatenate(

        [
            last_observation,
            future[:, :-1, :]
        ],

        axis=1

    )


    increments = (

        future
        -
        previous

    )


    return increments.astype(
        np.float32
    )


# ============================================================
# RECONSTRUCT POSITIONS FROM PREDICTED INCREMENTS
# ============================================================

def increments_to_positions_torch(
    history,
    increments
):

    last = history[
        :,
        -1:,
        :
    ]


    cumulative = torch.cumsum(

        increments,

        dim=1

    )


    return (
        last
        +
        cumulative
    )


# ============================================================
# DATASET
# ============================================================

class TrajectoryDataset(Dataset):

    def __init__(
        self,
        history_features,
        history_xy,
        future_positions,
        future_increments,
        weights
    ):

        self.history_features = torch.tensor(
            history_features,
            dtype=torch.float32
        )

        self.history_xy = torch.tensor(
            history_xy,
            dtype=torch.float32
        )

        self.future_positions = torch.tensor(
            future_positions,
            dtype=torch.float32
        )

        self.future_increments = torch.tensor(
            future_increments,
            dtype=torch.float32
        )

        self.weights = torch.tensor(
            weights,
            dtype=torch.float32
        )


    def __len__(self):

        return len(
            self.history_features
        )


    def __getitem__(
        self,
        index
    ):

        return (
            self.history_features[index],
            self.history_xy[index],
            self.future_positions[index],
            self.future_increments[index],
            self.weights[index]
        )


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=64
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
# MDT-U MODEL
# ============================================================

class MDTU(nn.Module):

    """
    Motion-Derivative Transformer with Uncertainty.

    Input:
      x,y,dx,dy,ddx,ddy

    Decoder:
      predicts future increments.

    Position trajectory:
      cumulative sum from last observation.

    Uncertainty:
      diagonal Gaussian sigma on reconstructed
      future positions.
    """

    def __init__(
        self,
        future_length
    ):

        super().__init__()


        self.future_length = (
            future_length
        )


        # ----------------------------------------------------
        # 6-D motion input -> Transformer dimension
        # ----------------------------------------------------

        self.input_projection = nn.Linear(
            6,
            D_MODEL
        )


        self.position = PositionalEncoding(

            D_MODEL,

            max_len=32

        )


        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=D_MODEL,

                nhead=NHEAD,

                dim_feedforward=DIM_FEEDFORWARD,

                dropout=DROPOUT,

                activation="gelu",

                batch_first=True,

                norm_first=True

            )
        )


        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=NUM_LAYERS

        )


        self.norm = nn.LayerNorm(
            D_MODEL
        )


        # ----------------------------------------------------
        # Future displacement increment head
        # ----------------------------------------------------

        self.increment_head = nn.Sequential(

            nn.Linear(
                D_MODEL,
                D_MODEL
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                D_MODEL,
                future_length * 2
            )

        )


        # ----------------------------------------------------
        # Position uncertainty
        # ----------------------------------------------------

        self.uncertainty_head = nn.Sequential(

            nn.Linear(
                D_MODEL,
                D_MODEL
            ),

            nn.GELU(),

            nn.Linear(
                D_MODEL,
                future_length * 2
            )

        )


    def forward(
        self,
        history_features,
        history_xy
    ):


        x = self.input_projection(
            history_features
        )


        x = self.position(
            x
        )


        x = self.encoder(
            x
        )


        # Same simple pooling philosophy as
        # compact Transformer baseline.

        latent = x.mean(
            dim=1
        )


        latent = self.norm(
            latent
        )


        # ================================================
        # FUTURE INCREMENTS
        # ================================================

        increments = (

            self.increment_head(
                latent
            )

            .view(
                -1,
                self.future_length,
                2
            )

        )


        positions = (
            increments_to_positions_torch(

                history_xy,

                increments

            )
        )


        # ================================================
        # UNCERTAINTY ON FUTURE POSITIONS
        # ================================================

        log_sigma = (

            self.uncertainty_head(
                latent
            )

            .view(
                -1,
                self.future_length,
                2
            )

        )


        log_sigma = torch.clamp(

            log_sigma,

            min=-4.5,

            max=1.5

        )


        sigma = torch.exp(
            log_sigma
        )


        return (

            positions,
            increments,
            log_sigma,
            sigma

        )


# ============================================================
# WEIGHTED LOSS
# ============================================================

def weighted_loss(
    prediction_position,
    prediction_increment,
    log_sigma,
    truth_position,
    truth_increment,
    weights
):


    # --------------------------------------------------------
    # Main trajectory position loss
    # --------------------------------------------------------

    position_loss = (

        nn.functional.smooth_l1_loss(

            prediction_position,

            truth_position,

            reduction="none"

        )

        .mean(
            dim=(1, 2)
        )

    )


    # --------------------------------------------------------
    # Endpoint loss
    # --------------------------------------------------------

    fde_loss = (

        nn.functional.smooth_l1_loss(

            prediction_position[
                :,
                -1,
                :
            ],

            truth_position[
                :,
                -1,
                :
            ],

            reduction="none"

        )

        .mean(
            dim=1
        )

    )


    # --------------------------------------------------------
    # Increment consistency
    # --------------------------------------------------------

    increment_loss = (

        nn.functional.smooth_l1_loss(

            prediction_increment,

            truth_increment,

            reduction="none"

        )

        .mean(
            dim=(1, 2)
        )

    )


    # --------------------------------------------------------
    # Acceleration/change-of-increment consistency
    # --------------------------------------------------------

    if prediction_increment.shape[1] > 1:


        pred_acc = (

            prediction_increment[
                :,
                1:,
                :
            ]

            -
            prediction_increment[
                :,
                :-1,
                :
            ]

        )


        truth_acc = (

            truth_increment[
                :,
                1:,
                :
            ]

            -
            truth_increment[
                :,
                :-1,
                :
            ]

        )


        acceleration_loss = (

            nn.functional.smooth_l1_loss(

                pred_acc,

                truth_acc,

                reduction="none"

            )

            .mean(
                dim=(1, 2)
            )

        )


    else:


        acceleration_loss = torch.zeros_like(
            position_loss
        )


    # --------------------------------------------------------
    # Gaussian NLL on position trajectory
    # --------------------------------------------------------

    sigma = torch.exp(
        log_sigma
    )


    normalized_error = (

        truth_position
        -
        prediction_position

    ) / sigma


    nll = (

        0.5
        *
        normalized_error ** 2

        +
        log_sigma

    ).mean(
        dim=(1, 2)
    )


    # --------------------------------------------------------
    # Combined sample loss
    # --------------------------------------------------------

    sample_loss = (

        position_loss

        +

        LAMBDA_FDE
        *
        fde_loss

        +

        LAMBDA_INCREMENT
        *
        increment_loss

        +

        LAMBDA_ACCELERATION
        *
        acceleration_loss

        +

        LAMBDA_NLL
        *
        nll

    )


    return (

        (
            sample_loss
            *
            weights
        ).sum()

        /
        weights.sum()

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
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    truth,
    metadata
):


    errors = np.linalg.norm(

        prediction
        -
        truth,

        axis=2

    )


    ADE = errors.mean(
        axis=1
    )


    FDE = errors[
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

        "point_errors":
            errors

    }


# ============================================================
# UNCERTAINTY METRICS
# ============================================================

def uncertainty_metrics(
    prediction,
    truth,
    sigma
):


    standardized = (

        truth
        -
        prediction

    ) / sigma


    mahalanobis_sq = np.sum(

        standardized ** 2,

        axis=2

    )


    return {

        "coverage50":
            float(
                np.mean(
                    mahalanobis_sq
                    <=
                    CHI2_50
                )
            ),

        "coverage90":
            float(
                np.mean(
                    mahalanobis_sq
                    <=
                    CHI2_90
                )
            ),

        "coverage95":
            float(
                np.mean(
                    mahalanobis_sq
                    <=
                    CHI2_95
                )
            ),

        "mean_sigma":
            float(
                sigma.mean()
            )

    }


# ============================================================
# PREDICT
# ============================================================

def predict_model(
    model,
    history_features,
    history_xy,
    batch_size=256
):


    model.eval()


    positions = []
    increments = []
    sigmas = []


    with torch.no_grad():


        for start in range(

            0,
            len(history_features),
            batch_size

        ):


            x_feature = torch.tensor(

                history_features[
                    start:
                    start + batch_size
                ],

                dtype=torch.float32,

                device=DEVICE

            )


            x_history = torch.tensor(

                history_xy[
                    start:
                    start + batch_size
                ],

                dtype=torch.float32,

                device=DEVICE

            )


            (
                position,
                increment,
                _,
                sigma

            ) = model(

                x_feature,

                x_history

            )


            positions.append(

                position
                .cpu()
                .numpy()

            )


            increments.append(

                increment
                .cpu()
                .numpy()

            )


            sigmas.append(

                sigma
                .cpu()
                .numpy()

            )


    return (

        np.concatenate(
            positions,
            axis=0
        ),

        np.concatenate(
            increments,
            axis=0
        ),

        np.concatenate(
            sigmas,
            axis=0
        )

    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation(
    model,
    X_features,
    X_history,
    Y_can,
    Y_abs,
    metadata,
    origin,
    angle,
    scale
):


    (
        pred_can,
        pred_increment,
        sigma

    ) = predict_model(

        model,

        X_features,

        X_history

    )


    pred_abs = inverse_transform_batch(

        pred_can,

        origin,

        angle,

        scale

    )


    metrics = calculate_metrics(

        pred_abs,

        Y_abs,

        metadata

    )


    uncertainty = uncertainty_metrics(

        pred_can,

        Y_can,

        sigma

    )


    return (

        metrics,
        uncertainty,
        pred_can,
        pred_increment,
        sigma,
        pred_abs

    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_one(
    task,
    seed
):


    set_seed(
        seed
    )


    print(
        "\n============================================"
    )

    print(
        task,
        "| MDT-U | seed",
        seed
    )

    print(
        "============================================"
    )


    npz_file = (

        ROOT /
        f"{task}_birdsai_trainval_canonical.npz"

    )


    metadata_file = (

        ROOT /
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


    train_idx = arrays[
        "train_indices"
    ].astype(
        int
    )


    val_idx = arrays[
        "validation_indices"
    ].astype(
        int
    )


    train_weights = arrays[
        "train_sample_weights"
    ].astype(
        np.float32
    )


    history_can = arrays[
        "history_canonical"
    ].astype(
        np.float32
    )


    future_can = arrays[
        "future_canonical"
    ].astype(
        np.float32
    )


    future_abs = arrays[
        "future_abs"
    ].astype(
        np.float32
    )


    origins = arrays[
        "origin"
    ]


    angles = arrays[
        "heading_angle"
    ]


    scales = arrays[
        "scale"
    ]


    # ========================================================
    # MOTION-DERIVATIVE INPUT
    # ========================================================

    derivative_features = (
        build_derivative_input(

            history_can

        )
    )


    future_increments = (
        future_to_increments(

            history_can,

            future_can

        )
    )


    # ========================================================
    # TRAIN
    # ========================================================

    X_train_features = (
        derivative_features[
            train_idx
        ]
    )


    X_train_history = (
        history_can[
            train_idx
        ]
    )


    Y_train_positions = (
        future_can[
            train_idx
        ]
    )


    Y_train_increments = (
        future_increments[
            train_idx
        ]
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    X_val_features = (
        derivative_features[
            val_idx
        ]
    )


    X_val_history = (
        history_can[
            val_idx
        ]
    )


    Y_val_can = (
        future_can[
            val_idx
        ]
    )


    Y_val_abs = (
        future_abs[
            val_idx
        ]
    )


    val_origin = origins[
        val_idx
    ]


    val_angle = angles[
        val_idx
    ]


    val_scale = scales[
        val_idx
    ]


    val_meta = (

        metadata.iloc[
            val_idx
        ]

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # DATASET
    # ========================================================

    dataset = TrajectoryDataset(

        history_features=
            X_train_features,

        history_xy=
            X_train_history,

        future_positions=
            Y_train_positions,

        future_increments=
            Y_train_increments,

        weights=
            train_weights

    )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        generator=generator,

        num_workers=0

    )


    future_length = (
        Y_train_positions.shape[
            1
        ]
    )


    model = MDTU(

        future_length

    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    scheduler = (

        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=10,

            min_lr=1e-5

        )

    )


    best_metric = float(
        "inf"
    )


    best_epoch = 0
    best_state = None

    no_improvement = 0

    history_rows = []


    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):


        model.train()

        losses = []


        for (
            X_feature,
            X_history,
            Y_position,
            Y_increment,
            W

        ) in loader:


            X_feature = X_feature.to(
                DEVICE
            )

            X_history = X_history.to(
                DEVICE
            )

            Y_position = Y_position.to(
                DEVICE
            )

            Y_increment = Y_increment.to(
                DEVICE
            )

            W = W.to(
                DEVICE
            )


            optimizer.zero_grad()


            (
                pred_position,
                pred_increment,
                log_sigma,
                _

            ) = model(

                X_feature,

                X_history

            )


            loss = weighted_loss(

                prediction_position=
                    pred_position,

                prediction_increment=
                    pred_increment,

                log_sigma=
                    log_sigma,

                truth_position=
                    Y_position,

                truth_increment=
                    Y_increment,

                weights=
                    W

            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                GRAD_CLIP

            )


            optimizer.step()


            losses.append(
                float(
                    loss.item()
                )
            )


        # ====================================================
        # VALIDATION
        # ====================================================

        (
            metrics,
            uncertainty,
            _,
            _,
            _,
            _

        ) = evaluate_validation(

            model,

            X_val_features,

            X_val_history,

            Y_val_can,

            Y_val_abs,

            val_meta,

            val_origin,

            val_angle,

            val_scale

        )


        validation_track_ADE = (
            metrics[
                "ADE_track_macro"
            ]
        )


        scheduler.step(
            validation_track_ADE
        )


        history_rows.append({

            "task":
                task,

            "model":
                "MDT-U",

            "seed":
                seed,

            "epoch":
                epoch,

            "train_loss":
                float(
                    np.mean(
                        losses
                    )
                ),

            "validation_ADE":
                metrics[
                    "ADE_mean"
                ],

            "validation_FDE":
                metrics[
                    "FDE_mean"
                ],

            "validation_track_macro_ADE":
                metrics[
                    "ADE_track_macro"
                ],

            "validation_track_macro_FDE":
                metrics[
                    "FDE_track_macro"
                ],

            "validation_source_macro_ADE":
                metrics[
                    "ADE_source_macro"
                ],

            "coverage90":
                uncertainty[
                    "coverage90"
                ],

            "coverage95":
                uncertainty[
                    "coverage95"
                ],

            "learning_rate":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]

        })


        # ====================================================
        # CHECKPOINT SELECTION
        # ====================================================

        if (

            validation_track_ADE

            <
            best_metric
            -
            MIN_DELTA

        ):


            best_metric = (
                validation_track_ADE
            )


            best_epoch = epoch


            best_state = copy.deepcopy(
                model.state_dict()
            )


            no_improvement = 0


        else:

            no_improvement += 1


        if (

            epoch == 1

            or

            epoch % 10 == 0

        ):


            print(

                f"Epoch {epoch:03d} | "
                f"loss={np.mean(losses):.6f} | "
                f"val track-ADE="
                f"{validation_track_ADE:.6f} | "
                f"best={best_metric:.6f} | "
                f"cov95="
                f"{uncertainty['coverage95']:.3f}"

            )


        if (
            no_improvement
            >=
            PATIENCE
        ):


            print(

                "Early stopping at epoch",
                epoch

            )

            break


    # ========================================================
    # RESTORE BEST
    # ========================================================

    model.load_state_dict(
        best_state
    )


    (
        metrics,
        uncertainty,
        pred_can,
        pred_increment,
        sigma,
        pred_abs

    ) = evaluate_validation(

        model,

        X_val_features,

        X_val_history,

        Y_val_can,

        Y_val_abs,

        val_meta,

        val_origin,

        val_angle,

        val_scale

    )


    # ========================================================
    # ERROR CURVE
    # ========================================================

    point_errors = metrics[
        "point_errors"
    ]


    error_curve_rows = []


    for step in range(
        point_errors.shape[
            1
        ]
    ):


        step_error = point_errors[
            :,
            step
        ]


        error_curve_rows.append({

            "task":
                task,

            "seed":
                seed,

            "forecast_step":
                step + 1,

            "forecast_time_sec":
                (
                    step + 1
                )
                /
                FPS,

            "mean_error":
                float(
                    step_error.mean()
                ),

            "median_error":
                float(
                    np.median(
                        step_error
                    )
                )

        })


    error_curve = pd.DataFrame(
        error_curve_rows
    )


    # ========================================================
    # UNCERTAINTY BY FORECAST TIME
    # ========================================================

    standardized = (

        Y_val_can
        -
        pred_can

    ) / sigma


    mahalanobis_sq = np.sum(

        standardized ** 2,

        axis=2

    )


    uncertainty_time_rows = []


    for step in range(
        mahalanobis_sq.shape[
            1
        ]
    ):


        d2 = mahalanobis_sq[
            :,
            step
        ]


        uncertainty_time_rows.append({

            "task":
                task,

            "seed":
                seed,

            "forecast_step":
                step + 1,

            "forecast_time_sec":
                (
                    step + 1
                )
                /
                FPS,

            "coverage50":
                float(
                    np.mean(
                        d2
                        <=
                        CHI2_50
                    )
                ),

            "coverage90":
                float(
                    np.mean(
                        d2
                        <=
                        CHI2_90
                    )
                ),

            "coverage95":
                float(
                    np.mean(
                        d2
                        <=
                        CHI2_95
                    )
                ),

            "mean_sigma":
                float(
                    sigma[
                        :,
                        step,
                        :
                    ].mean()
                )

        })


    uncertainty_time = pd.DataFrame(
        uncertainty_time_rows
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(

        {
            "state_dict":
                best_state,

            "task":
                task,

            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "architecture":
                "MDT-U",

            "d_model":
                D_MODEL,

            "nhead":
                NHEAD,

            "num_layers":
                NUM_LAYERS,

            "dim_feedforward":
                DIM_FEEDFORWARD,

            "dropout":
                DROPOUT

        },

        OUT /
        f"{task}_MDTU_seed{seed}.pt"

    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "task":
            task,

        "model":
            "MDT-U",

        "seed":
            seed,

        "best_epoch":
            best_epoch,

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
            ],

        "coverage50":
            uncertainty[
                "coverage50"
            ],

        "coverage90":
            uncertainty[
                "coverage90"
            ],

        "coverage95":
            uncertainty[
                "coverage95"
            ],

        "mean_sigma":
            uncertainty[
                "mean_sigma"
            ]

    }


    return (

        pd.DataFrame(
            history_rows
        ),

        result,

        error_curve,

        uncertainty_time

    )


# ============================================================
# RUN ALL
# ============================================================

all_history = []
all_results = []
all_error_curves = []
all_uncertainty_time = []


for task in TASKS:


    for seed in SEEDS:


        (
            history,
            result,
            error_curve,
            uncertainty_time

        ) = train_one(

            task,

            seed

        )


        all_history.append(
            history
        )


        all_results.append(
            result
        )


        all_error_curves.append(
            error_curve
        )


        all_uncertainty_time.append(
            uncertainty_time
        )


# ============================================================
# TRAINING HISTORY
# ============================================================

training_df = pd.concat(

    all_history,

    ignore_index=True

)


training_df.to_csv(

    OUT /
    "mdtu_training_history.csv",

    index=False

)


# ============================================================
# SEED RESULTS
# ============================================================

results = pd.DataFrame(
    all_results
)


results.to_csv(

    OUT /
    "mdtu_validation_seed_results.csv",

    index=False

)


# ============================================================
# MULTI-SEED MDT-U SUMMARY
# ============================================================

metric_columns = [

    "ADE_mean",
    "FDE_mean",
    "ADE_track_macro",
    "FDE_track_macro",
    "ADE_source_macro",
    "FDE_source_macro",
    "coverage50",
    "coverage90",
    "coverage95",
    "mean_sigma"

]


summary_rows = []


for (
    task,
    model
), group in results.groupby(

    [
        "task",
        "model"
    ]

):


    row = {

        "task":
            task,

        "model":
            model,

        "seeds":
            len(
                group
            ),

        "mean_best_epoch":
            group[
                "best_epoch"
            ].mean()

    }


    for metric in metric_columns:


        row[
            f"{metric}_mean"
        ] = group[
            metric
        ].mean()


        row[
            f"{metric}_std"
        ] = group[
            metric
        ].std(
            ddof=1
        )


    summary_rows.append(
        row
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(

    OUT /
    "mdtu_validation_summary.csv",

    index=False

)


# ============================================================
# ERROR CURVE
# ============================================================

error_curve_df = pd.concat(

    all_error_curves,

    ignore_index=True

)


error_curve_df.to_csv(

    OUT /
    "mdtu_validation_error_curve_seed.csv",

    index=False

)


error_curve_mean = (

    error_curve_df.groupby(

        [
            "task",
            "forecast_step",
            "forecast_time_sec"
        ],

        as_index=False

    )

    .agg(

        mean_error=(
            "mean_error",
            "mean"
        ),

        mean_error_std=(
            "mean_error",
            "std"
        ),

        median_error=(
            "median_error",
            "mean"
        ),

        median_error_std=(
            "median_error",
            "std"
        )

    )

)


error_curve_mean.to_csv(

    OUT /
    "mdtu_validation_error_curve.csv",

    index=False

)


# ============================================================
# UNCERTAINTY BY TIME
# ============================================================

uncertainty_time_df = pd.concat(

    all_uncertainty_time,

    ignore_index=True

)


uncertainty_time_df.to_csv(

    OUT /
    "mdtu_uncertainty_by_time_seed.csv",

    index=False

)


uncertainty_time_mean = (

    uncertainty_time_df.groupby(

        [
            "task",
            "forecast_step",
            "forecast_time_sec"
        ],

        as_index=False

    )

    .agg(

        coverage50_mean=(
            "coverage50",
            "mean"
        ),

        coverage90_mean=(
            "coverage90",
            "mean"
        ),

        coverage95_mean=(
            "coverage95",
            "mean"
        ),

        mean_sigma_mean=(
            "mean_sigma",
            "mean"
        )

    )

)


uncertainty_time_mean.to_csv(

    OUT /
    "mdtu_uncertainty_by_time.csv",

    index=False

)


# ============================================================
# VALIDATION BASELINE COMPARISON
# ============================================================

comparison_seed_rows = []


def add_baseline_history(
    file_path
):


    if not file_path.exists():

        return


    table = pd.read_csv(
        file_path
    )


    for (
        task,
        model,
        seed
    ), group in table.groupby(

        [
            "task",
            "model",
            "seed"
        ]

    ):


        best = group.sort_values(

            "validation_track_macro_ADE"

        ).iloc[0]


        comparison_seed_rows.append({

            "task":
                task,

            "model":
                model,

            "seed":
                seed,

            "ADE_mean":
                best[
                    "validation_ADE"
                ],

            "FDE_mean":
                best[
                    "validation_FDE"
                ],

            "ADE_track_macro":
                best[
                    "validation_track_macro_ADE"
                ],

            "FDE_track_macro":
                best[
                    "validation_track_macro_FDE"
                ]

        })


# ============================================================
# LSTM + GRU
# ============================================================

add_baseline_history(

    RNN_DIR /
    "training_history.csv"

)


# ============================================================
# TRANSFORMER
# ============================================================

add_baseline_history(

    TRANSFORMER_DIR /
    "transformer_training_history.csv"

)


# ============================================================
# FIXED RMPG-U v2
# ============================================================

v2_file = (

    V2_DIR /
    "rmpgu_v2_fixed_validation_seed_results.csv"

)


if v2_file.exists():


    v2 = pd.read_csv(
        v2_file
    )


    for _, row in v2.iterrows():


        comparison_seed_rows.append({

            "task":
                row[
                    "task"
                ],

            "model":
                "RMPG-U-v2",

            "seed":
                row[
                    "seed"
                ],

            "ADE_mean":
                row[
                    "ADE_mean"
                ],

            "FDE_mean":
                row[
                    "FDE_mean"
                ],

            "ADE_track_macro":
                row[
                    "ADE_track_macro"
                ],

            "FDE_track_macro":
                row[
                    "FDE_track_macro"
                ]

        })


# ============================================================
# MDT-U
# ============================================================

for _, row in results.iterrows():


    comparison_seed_rows.append({

        "task":
            row[
                "task"
            ],

        "model":
            "MDT-U",

        "seed":
            row[
                "seed"
            ],

        "ADE_mean":
            row[
                "ADE_mean"
            ],

        "FDE_mean":
            row[
                "FDE_mean"
            ],

        "ADE_track_macro":
            row[
                "ADE_track_macro"
            ],

        "FDE_track_macro":
            row[
                "FDE_track_macro"
            ]

    })


comparison_seed = pd.DataFrame(
    comparison_seed_rows
)


comparison_rows = []


for (
    task,
    model
), group in comparison_seed.groupby(

    [
        "task",
        "model"
    ]

):


    comparison_rows.append({

        "task":
            task,

        "model":
            model,

        "seeds":
            len(
                group
            ),

        "ADE_mean":
            group[
                "ADE_mean"
            ].mean(),

        "ADE_std":
            group[
                "ADE_mean"
            ].std(
                ddof=1
            ),

        "FDE_mean":
            group[
                "FDE_mean"
            ].mean(),

        "FDE_std":
            group[
                "FDE_mean"
            ].std(
                ddof=1
            ),

        "track_macro_ADE_mean":
            group[
                "ADE_track_macro"
            ].mean(),

        "track_macro_ADE_std":
            group[
                "ADE_track_macro"
            ].std(
                ddof=1
            ),

        "track_macro_FDE_mean":
            group[
                "FDE_track_macro"
            ].mean(),

        "track_macro_FDE_std":
            group[
                "FDE_track_macro"
            ].std(
                ddof=1
            )

    })


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(

    OUT /
    "mdtu_model_comparison.csv",

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
        "MDT-U — TRAINREAL VALIDATION EXPERIMENT\n"
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
        "- Training: BIRDSAI TrainReal neural-training tracks only.\n"
    )

    f.write(
        "- Validation: fixed track-disjoint TrainReal validation only.\n"
    )

    f.write(
        "- BIRDSAI TestReal NOT loaded.\n"
    )

    f.write(
        "- WildBox NOT loaded.\n"
    )

    f.write(
        "- Same three random seeds: 42, 52, 62.\n"
    )

    f.write(
        "- Checkpoint selection: validation track-macro ADE.\n\n"
    )


    f.write(
        "ARCHITECTURE\n"
    )

    f.write(
        "- Compact temporal Transformer encoder.\n"
    )

    f.write(
        "- Input per observation step: "
        "[x,y,dx,dy,ddx,ddy].\n"
    )

    f.write(
        "- Future displacement-increment decoder.\n"
    )

    f.write(
        "- Future positions reconstructed by cumulative sum.\n"
    )

    f.write(
        "- Endpoint-aware trajectory loss.\n"
    )

    f.write(
        "- Increment and acceleration-consistency losses.\n"
    )

    f.write(
        "- Heteroscedastic Gaussian uncertainty head.\n\n"
    )


    f.write(
        "MODEL COMPARISON\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        comparison.to_string(
            index=False
        )
    )


    f.write(
        "\n\nMDT-U SUMMARY\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        summary.to_string(
            index=False
        )
    )


    f.write(
        "\n\nADDITIONAL FILES\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        "mdtu_validation_error_curve.csv\n"
    )

    f.write(
        "mdtu_uncertainty_by_time.csv\n"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print(
    "\n============================================"
)

print(
    "MDT-U VALIDATION COMPLETE"
)

print(
    "============================================"
)


print(
    "\nModel comparison:"
)

print(
    comparison.to_string(
        index=False
    )
)


print(
    "\nMDT-U summary:"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    OUT
)