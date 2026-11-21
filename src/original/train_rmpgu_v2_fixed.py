from pathlib import Path
import copy
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

V1_DIR = Path(
    r"C:\Users\HP\Downloads\proposed_ampmu_validation"
)

OLD_V2_DIR = Path(
    r"C:\Users\HP\Downloads\rmpgu_v2_validation"
)

OUT = Path(
    r"C:\Users\HP\Downloads\rmpgu_v2_fixed_validation"
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
DT = 1.0 / FPS

BATCH_SIZE = 64

HIDDEN_SIZE = 64
FEATURE_HIDDEN = 16
FUSED_SIZE = 64

NUM_LAYERS = 2
DROPOUT = 0.10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 300
PATIENCE = 30
MIN_DELTA = 1e-5

GRAD_CLIP = 5.0

CV_RECENT_FRAMES = 10

# ============================================================
# SAME LOSS WEIGHTS AS PREVIOUS V2
# ============================================================

LAMBDA_NLL = 0.25
LAMBDA_GATE = 0.25
LAMBDA_RESIDUAL = 0.01

EPS = 1e-8


# ============================================================
# 2-D GAUSSIAN CONFIDENCE ELLIPSE THRESHOLDS
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
# MOTION FEATURES
# ============================================================

def motion_features(history):

    """
    History-only motion descriptors.

    Input:
        history [T,2]
        original camera-corrected normalized xy

    Output:
        7-D feature vector
    """

    increments = np.diff(
        history,
        axis=0
    )

    step_distance = np.linalg.norm(
        increments,
        axis=1
    )

    path_length = float(
        step_distance.sum()
    )

    net_displacement = float(

        np.linalg.norm(

            history[-1]
            -
            history[0]

        )

    )

    straightness = (

        net_displacement
        /
        max(
            path_length,
            EPS
        )

    )


    speeds = (
        step_distance
        *
        FPS
    )

    mean_speed = float(
        speeds.mean()
    )

    speed_std = float(
        speeds.std()
    )

    speed_cv = (

        speed_std
        /
        max(
            mean_speed,
            EPS
        )

    )


    # --------------------------------------------------------
    # Recent speed
    # --------------------------------------------------------

    recent_k = min(
        10,
        len(history)
    )

    recent = history[
        -recent_k:
    ]

    recent_step_distance = np.linalg.norm(

        np.diff(
            recent,
            axis=0
        ),

        axis=1

    )

    recent_speed = float(

        recent_step_distance.mean()
        *
        FPS

    )


    # --------------------------------------------------------
    # Heading change
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
        n1 > EPS
        and
        n2 > EPS
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
    # Acceleration variability
    # --------------------------------------------------------

    velocity = (
        increments
        *
        FPS
    )

    if len(
        velocity
    ) >= 2:

        acceleration = (

            np.diff(
                velocity,
                axis=0
            )

            *
            FPS

        )

        mean_acceleration = float(

            np.linalg.norm(
                acceleration,
                axis=1
            ).mean()

        )

    else:

        mean_acceleration = 0.0


    normalized_acceleration = (

        mean_acceleration
        /
        max(
            mean_speed,
            EPS
        )

    )


    return np.asarray(

        [
            path_length,
            net_displacement,
            straightness,
            recent_speed,
            speed_cv,
            heading_change,
            normalized_acceleration
        ],

        dtype=np.float32

    )


def build_motion_features(
    histories
):

    return np.stack([

        motion_features(
            h
        )

        for h in histories

    ]).astype(
        np.float32
    )


# ============================================================
# FEATURE NORMALIZATION
#
# FIT ON TRAINING SAMPLES ONLY
# ============================================================

def fit_feature_normalizer(
    features
):

    mean = features.mean(
        axis=0
    )

    std = features.std(
        axis=0
    )

    std = np.maximum(
        std,
        1e-6
    )

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        )
    )


def normalize_features(
    features,
    mean,
    std
):

    normalized = (

        features
        -
        mean

    ) / std


    normalized = np.clip(
        normalized,
        -5.0,
        5.0
    )


    return normalized.astype(
        np.float32
    )


# ============================================================
# CONSTANT VELOCITY PRIOR
# ============================================================

def constant_velocity_prior_torch(
    history,
    future_length
):

    """
    history:
        [B,T,2] canonical coordinates
    """

    k = min(
        CV_RECENT_FRAMES,
        history.shape[1]
    )


    recent = history[
        :,
        -k:,
        :
    ]


    t = torch.arange(

        k,

        dtype=history.dtype,

        device=history.device

    ) * DT


    tc = (
        t
        -
        t.mean()
    )


    denom = torch.sum(
        tc ** 2
    )


    centered = (

        recent

        -
        recent.mean(
            dim=1,
            keepdim=True
        )

    )


    velocity = (

        centered

        *
        tc.view(
            1,
            k,
            1
        )

    ).sum(
        dim=1
    ) / denom


    future_time = (

        torch.arange(

            1,
            future_length + 1,

            dtype=history.dtype,

            device=history.device

        )

        *
        DT

    )


    prediction = (

        history[
            :,
            -1:,
            :
        ]

        +

        velocity.unsqueeze(1)

        *
        future_time.view(
            1,
            future_length,
            1
        )

    )


    return prediction


def constant_velocity_prior_numpy(
    history,
    future_length
):

    x = torch.tensor(

        history[None],

        dtype=torch.float32

    )


    with torch.no_grad():

        pred = (
            constant_velocity_prior_torch(

                x,

                future_length

            )
        )


    return (

        pred[0]
        .numpy()

    )


# ============================================================
# GATE TARGET DIFFERENCE
# ============================================================

def calculate_gate_differences(
    histories_can,
    futures_can
):

    """
    Positive value:
        CV error < Stationary error
        => CV preferred

    Negative value:
        Stationary preferred
    """

    differences = []


    for i in range(
        len(histories_can)
    ):


        history = histories_can[
            i
        ]

        truth = futures_can[
            i
        ]


        stationary = np.repeat(

            history[
                -1:
            ],

            len(
                truth
            ),

            axis=0

        )


        cv = constant_velocity_prior_numpy(

            history,

            len(
                truth
            )

        )


        stationary_error = np.linalg.norm(

            stationary
            -
            truth,

            axis=1

        )


        cv_error = np.linalg.norm(

            cv
            -
            truth,

            axis=1

        )


        differences.append(

            stationary_error
            -
            cv_error

        )


    return np.stack(
        differences
    ).astype(
        np.float32
    )


# ============================================================
# TRAIN-ONLY GATE TARGET
# ============================================================

def build_gate_targets(
    histories_can,
    futures_can
):

    differences = (
        calculate_gate_differences(

            histories_can,

            futures_can

        )
    )


    absolute_difference = np.abs(
        differences
    ).reshape(
        -1
    )


    temperature = float(

        np.median(
            absolute_difference
        )

    )


    temperature = max(
        temperature,
        1e-3
    )


    return (

        gate_targets_with_temperature(

            histories_can,

            futures_can,

            temperature

        ),

        temperature

    )


def gate_targets_with_temperature(
    histories_can,
    futures_can,
    temperature
):

    differences = (
        calculate_gate_differences(

            histories_can,

            futures_can

        )
    )


    logits = (

        differences
        /
        float(
            temperature
        )

    )


    logits = np.clip(
        logits,
        -10.0,
        10.0
    )


    target = (

        1.0

        /

        (
            1.0
            +
            np.exp(
                -logits
            )
        )

    )


    return target.astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

class TrajectoryDataset(Dataset):

    def __init__(
        self,
        history,
        future,
        features,
        weights,
        gate_targets
    ):

        self.history = torch.tensor(
            history,
            dtype=torch.float32
        )

        self.future = torch.tensor(
            future,
            dtype=torch.float32
        )

        self.features = torch.tensor(
            features,
            dtype=torch.float32
        )

        self.weights = torch.tensor(
            weights,
            dtype=torch.float32
        )

        self.gate_targets = torch.tensor(
            gate_targets,
            dtype=torch.float32
        )


    def __len__(self):

        return len(
            self.history
        )


    def __getitem__(
        self,
        index
    ):

        return (
            self.history[index],
            self.future[index],
            self.features[index],
            self.weights[index],
            self.gate_targets[index]
        )


# ============================================================
# RMPG-U v2 MODEL
# ============================================================

class RMPGU(nn.Module):

    """
    Residual Motion-Prior Gated Forecaster
    with Uncertainty.

    Physical prior:
      Stationary <-> Constant Velocity

    Final:
      prior + learned nonlinear residual
    """

    def __init__(
        self,
        future_length,
        feature_dim=7
    ):

        super().__init__()


        self.future_length = (
            future_length
        )


        # ----------------------------------------------------
        # GRU trajectory encoder
        # ----------------------------------------------------

        self.encoder = nn.GRU(

            input_size=2,

            hidden_size=HIDDEN_SIZE,

            num_layers=NUM_LAYERS,

            batch_first=True,

            dropout=DROPOUT

        )


        # ----------------------------------------------------
        # Explicit motion feature encoder
        # ----------------------------------------------------

        self.feature_encoder = nn.Sequential(

            nn.Linear(
                feature_dim,
                FEATURE_HIDDEN
            ),

            nn.GELU(),

            nn.Linear(
                FEATURE_HIDDEN,
                FEATURE_HIDDEN
            ),

            nn.GELU()

        )


        # ----------------------------------------------------
        # Shared fused representation
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(

                HIDDEN_SIZE
                +
                FEATURE_HIDDEN,

                FUSED_SIZE

            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            )

        )


        # ----------------------------------------------------
        # Gate:
        #
        # alpha = 0 -> stationary
        # alpha = 1 -> constant velocity
        # ----------------------------------------------------

        self.gate_head = nn.Linear(

            FUSED_SIZE,

            future_length

        )


        # ----------------------------------------------------
        # Neural residual
        # ----------------------------------------------------

        self.residual_head = nn.Sequential(

            nn.Linear(
                FUSED_SIZE,
                FUSED_SIZE
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                FUSED_SIZE,
                future_length * 2
            )

        )


        # ----------------------------------------------------
        # Diagonal Gaussian uncertainty
        # ----------------------------------------------------

        self.uncertainty_head = nn.Sequential(

            nn.Linear(
                FUSED_SIZE,
                FUSED_SIZE
            ),

            nn.GELU(),

            nn.Linear(
                FUSED_SIZE,
                future_length * 2
            )

        )


    def forward(
        self,
        history,
        motion_feature
    ):


        # ====================================================
        # ENCODING
        # ====================================================

        _, hidden = self.encoder(
            history
        )


        temporal = hidden[
            -1
        ]


        feature_embedding = (
            self.feature_encoder(
                motion_feature
            )
        )


        fused = self.fusion(

            torch.cat(

                [
                    temporal,
                    feature_embedding
                ],

                dim=1

            )

        )


        # ====================================================
        # STATIONARY PRIOR
        # ====================================================

        stationary = (

            history[
                :,
                -1:,
                :
            ]

            .repeat(
                1,
                self.future_length,
                1
            )

        )


        # ====================================================
        # CONSTANT VELOCITY PRIOR
        # ====================================================

        cv = constant_velocity_prior_torch(

            history,

            self.future_length

        )


        # ====================================================
        # GATE
        # ====================================================

        gate_logits = self.gate_head(
            fused
        )


        alpha = torch.sigmoid(
            gate_logits
        )


        prior = (

            (
                1.0
                -
                alpha.unsqueeze(
                    -1
                )
            )

            *
            stationary

            +

            alpha.unsqueeze(
                -1
            )

            *
            cv

        )


        # ====================================================
        # RESIDUAL
        # ====================================================

        residual = (

            self.residual_head(
                fused
            )

            .view(
                -1,
                self.future_length,
                2
            )

        )


        mean = (

            prior
            +
            residual

        )


        # ====================================================
        # UNCERTAINTY
        # ====================================================

        log_sigma = (

            self.uncertainty_head(
                fused
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


        # ====================================================
        # IMPORTANT FIX:
        #
        # gate_logits returned from THIS SAME forward pass.
        # No second encoder/fusion/dropout pass.
        # ====================================================

        return (

            mean,
            log_sigma,
            sigma,
            alpha,
            residual,
            prior,
            gate_logits

        )


# ============================================================
# LOSS
# ============================================================

def weighted_loss(
    mean,
    log_sigma,
    residual,
    gate_logits,
    target,
    gate_target,
    weights
):


    # --------------------------------------------------------
    # Smooth-L1 trajectory loss
    # --------------------------------------------------------

    point_loss = (

        nn.functional.smooth_l1_loss(

            mean,

            target,

            reduction="none"

        )

        .mean(
            dim=(1, 2)
        )

    )


    # --------------------------------------------------------
    # Heteroscedastic Gaussian NLL
    # --------------------------------------------------------

    sigma = torch.exp(
        log_sigma
    )


    normalized_error = (

        target
        -
        mean

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
    # Gate supervision
    #
    # SAME gate_logits used in forward prediction.
    # --------------------------------------------------------

    gate_loss = (

        nn.functional.binary_cross_entropy_with_logits(

            gate_logits,

            gate_target,

            reduction="none"

        )

        .mean(
            dim=1
        )

    )


    # --------------------------------------------------------
    # Residual regularization
    # --------------------------------------------------------

    residual_penalty = (

        residual ** 2

    ).mean(
        dim=(1, 2)
    )


    sample_loss = (

        point_loss

        +

        LAMBDA_NLL
        *
        nll

        +

        LAMBDA_GATE
        *
        gate_loss

        +

        LAMBDA_RESIDUAL
        *
        residual_penalty

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


    output = []


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


        output.append(
            absolute
        )


    return np.asarray(

        output,

        dtype=np.float64

    )


# ============================================================
# TRAJECTORY METRICS
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


    # --------------------------------------------------------
    # Track macro
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Source macro
    # --------------------------------------------------------

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
            )

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
# PREDICTION
# ============================================================

def predict_model(
    model,
    history,
    features,
    batch_size=256
):


    model.eval()


    means = []
    sigmas = []
    alphas = []
    residuals = []
    priors = []


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


            f = torch.tensor(

                features[
                    start:
                    start + batch_size
                ],

                dtype=torch.float32,

                device=DEVICE

            )


            (
                mean,
                _,
                sigma,
                alpha,
                residual,
                prior,
                _

            ) = model(
                x,
                f
            )


            means.append(
                mean.cpu().numpy()
            )


            sigmas.append(
                sigma.cpu().numpy()
            )


            alphas.append(
                alpha.cpu().numpy()
            )


            residuals.append(
                residual.cpu().numpy()
            )


            priors.append(
                prior.cpu().numpy()
            )


    return (

        np.concatenate(
            means,
            axis=0
        ),

        np.concatenate(
            sigmas,
            axis=0
        ),

        np.concatenate(
            alphas,
            axis=0
        ),

        np.concatenate(
            residuals,
            axis=0
        ),

        np.concatenate(
            priors,
            axis=0
        )

    )


# ============================================================
# MOTION REGIME FUNCTIONS
# ============================================================

def regime_features(
    history
):


    increments = np.diff(
        history,
        axis=0
    )


    distances = np.linalg.norm(
        increments,
        axis=1
    )


    path = float(
        distances.sum()
    )


    net = float(

        np.linalg.norm(

            history[-1]
            -
            history[0]

        )

    )


    straightness = (

        net
        /
        max(
            path,
            EPS
        )

    )


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
        n1 > EPS
        and
        n2 > EPS
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


    return (
        path,
        straightness,
        heading_change
    )


def derive_regime_thresholds(
    histories
):


    values = np.asarray([

        regime_features(
            h
        )

        for h in histories

    ])


    return {

        "slow":
            float(
                np.quantile(
                    values[
                        :,
                        0
                    ],
                    0.25
                )
            ),

        "straight":
            float(
                np.quantile(
                    values[
                        :,
                        1
                    ],
                    0.50
                )
            ),

        "heading":
            float(
                np.quantile(
                    values[
                        :,
                        2
                    ],
                    0.50
                )
            )

    }


def assign_regime(
    history,
    thresholds
):


    (
        path,
        straightness,
        heading

    ) = regime_features(
        history
    )


    if path <= thresholds[
        "slow"
    ]:

        return "slow_stationary"


    if (

        straightness
        >=
        thresholds[
            "straight"
        ]

        and

        heading
        <=
        thresholds[
            "heading"
        ]

    ):

        return "persistent_linear"


    return "irregular_turning"


# ============================================================
# VALIDATION EVALUATION
# ============================================================

def evaluate_validation(
    model,
    X_val,
    F_val,
    Y_val_can,
    Y_val_abs,
    metadata,
    origin,
    angle,
    scale
):


    (
        pred_can,
        sigma,
        alpha,
        residual,
        prior

    ) = predict_model(

        model,

        X_val,

        F_val

    )


    pred_abs = inverse_transform_batch(

        pred_can,

        origin,

        angle,

        scale

    )


    metrics = calculate_metrics(

        pred_abs,

        Y_val_abs,

        metadata

    )


    uncertainty = uncertainty_metrics(

        pred_can,

        Y_val_can,

        sigma

    )


    return (

        metrics,
        uncertainty,
        pred_can,
        sigma,
        alpha,
        residual,
        prior

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
        "| RMPG-U v2 FIXED | seed",
        seed
    )

    print(
        "============================================"
    )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

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
    ]


    future_can = arrays[
        "future_canonical"
    ]


    history_abs = arrays[
        "history_abs"
    ]


    future_abs = arrays[
        "future_abs"
    ]


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
    # EXPLICIT FEATURES
    # ========================================================

    all_features = build_motion_features(
        history_abs
    )


    feature_mean, feature_std = (
        fit_feature_normalizer(

            all_features[
                train_idx
            ]

        )
    )


    normalized_features = (
        normalize_features(

            all_features,

            feature_mean,

            feature_std

        )
    )


    # ========================================================
    # TRAIN
    # ========================================================

    X_train = history_can[
        train_idx
    ]


    Y_train = future_can[
        train_idx
    ]


    F_train = normalized_features[
        train_idx
    ]


    train_abs_history = history_abs[
        train_idx
    ]


    # ========================================================
    # VALIDATION
    # ========================================================

    X_val = history_can[
        val_idx
    ]


    Y_val_can = future_can[
        val_idx
    ]


    Y_val_abs = future_abs[
        val_idx
    ]


    F_val = normalized_features[
        val_idx
    ]


    val_abs_history = history_abs[
        val_idx
    ]


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
    # TRAIN-ONLY GATE SUPERVISION
    # ========================================================

    gate_targets, gate_temperature = (
        build_gate_targets(

            X_train,

            Y_train

        )
    )


    # ========================================================
    # TRAIN-ONLY MOTION REGIME THRESHOLDS
    # ========================================================

    regime_thresholds = (
        derive_regime_thresholds(

            train_abs_history

        )
    )


    val_regimes = [

        assign_regime(

            history,

            regime_thresholds

        )

        for history in
        val_abs_history

    ]


    # ========================================================
    # DATA LOADER
    # ========================================================

    dataset = TrajectoryDataset(

        X_train,

        Y_train,

        F_train,

        train_weights,

        gate_targets

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
        Y_train.shape[
            1
        ]
    )


    model = RMPGU(

        future_length=
            future_length,

        feature_dim=
            F_train.shape[
                1
            ]

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

    training_rows = []


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):


        model.train()

        losses = []


        for (
            X,
            Y,
            F,
            W,
            G

        ) in loader:


            X = X.to(
                DEVICE
            )

            Y = Y.to(
                DEVICE
            )

            F = F.to(
                DEVICE
            )

            W = W.to(
                DEVICE
            )

            G = G.to(
                DEVICE
            )


            optimizer.zero_grad()


            # =================================================
            # SINGLE FORWARD PASS ONLY
            # =================================================

            (
                mean,
                log_sigma,
                _,
                _,
                residual,
                _,
                gate_logits

            ) = model(
                X,
                F
            )


            loss = weighted_loss(

                mean=
                    mean,

                log_sigma=
                    log_sigma,

                residual=
                    residual,

                gate_logits=
                    gate_logits,

                target=
                    Y,

                gate_target=
                    G,

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
            _,
            _

        ) = evaluate_validation(

            model,

            X_val,

            F_val,

            Y_val_can,

            Y_val_abs,

            val_meta,

            val_origin,

            val_angle,

            val_scale

        )


        val_track_ADE = (
            metrics[
                "ADE_track_macro"
            ]
        )


        scheduler.step(
            val_track_ADE
        )


        training_rows.append({

            "task":
                task,

            "model":
                "RMPG-U-v2-fixed",

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
        # BEST CHECKPOINT
        # ====================================================

        if (

            val_track_ADE

            <
            best_metric
            -
            MIN_DELTA

        ):


            best_metric = (
                val_track_ADE
            )


            best_epoch = (
                epoch
            )


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
                f"loss={np.mean(losses):.5f} | "
                f"val track-ADE={val_track_ADE:.6f} | "
                f"best={best_metric:.6f} | "
                f"cov95={uncertainty['coverage95']:.3f}"

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
        sigma,
        alpha,
        residual,
        prior

    ) = evaluate_validation(

        model,

        X_val,

        F_val,

        Y_val_can,

        Y_val_abs,

        val_meta,

        val_origin,

        val_angle,

        val_scale

    )


    # ========================================================
    # VALIDATION GATE DIAGNOSTIC
    #
    # Uses TRAIN-DERIVED gate temperature.
    # Validation future is used only after model selection
    # for diagnostic scoring.
    # ========================================================

    val_gate_target = (
        gate_targets_with_temperature(

            X_val,

            Y_val_can,

            gate_temperature

        )
    )


    hard_prediction = (
        alpha
        >=
        0.5
    )


    hard_target = (
        val_gate_target
        >=
        0.5
    )


    gate_accuracy = float(

        np.mean(

            hard_prediction
            ==
            hard_target

        )

    )


    gate_brier = float(

        np.mean(

            (
                alpha
                -
                val_gate_target
            )
            **
            2

        )

    )


    # ========================================================
    # REGIME-AVERAGED GATE
    # ========================================================

    alpha_sample = alpha.mean(
        axis=1
    )


    residual_sample = np.linalg.norm(

        residual,

        axis=2

    ).mean(
        axis=1
    )


    gate_table = pd.DataFrame({

        "task":
            task,

        "seed":
            seed,

        "regime":
            val_regimes,

        "mean_cv_weight":
            alpha_sample,

        "mean_stationary_weight":
            1.0
            -
            alpha_sample,

        "residual_magnitude":
            residual_sample

    })


    gate_by_regime = (

        gate_table.groupby(

            [
                "task",
                "seed",
                "regime"
            ],

            as_index=False

        )

        .agg(

            samples=(
                "regime",
                "size"
            ),

            stationary_weight=(
                "mean_stationary_weight",
                "mean"
            ),

            cv_weight=(
                "mean_cv_weight",
                "mean"
            ),

            residual_magnitude=(
                "residual_magnitude",
                "mean"
            )

        )

    )


    # ========================================================
    # NEW:
    # GATE BY FORECAST TIME
    # ========================================================

    time_rows = []


    val_regimes_array = np.asarray(
        val_regimes
    )


    residual_per_step = np.linalg.norm(

        residual,

        axis=2

    )


    for regime in sorted(
        set(
            val_regimes
        )
    ):


        mask = (
            val_regimes_array
            ==
            regime
        )


        for step in range(
            future_length
        ):


            cv_weight = float(

                alpha[
                    mask,
                    step
                ].mean()

            )


            stationary_weight = (
                1.0
                -
                cv_weight
            )


            residual_magnitude = float(

                residual_per_step[
                    mask,
                    step
                ].mean()

            )


            time_rows.append({

                "task":
                    task,

                "seed":
                    seed,

                "regime":
                    regime,

                "forecast_step":
                    step + 1,

                "forecast_time_sec":
                    (
                        step + 1
                    )
                    /
                    FPS,

                "stationary_weight":
                    stationary_weight,

                "cv_weight":
                    cv_weight,

                "residual_magnitude":
                    residual_magnitude

            })


    gate_by_time = pd.DataFrame(
        time_rows
    )


    # ========================================================
    # SAVE CHECKPOINT
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

            "feature_mean":
                feature_mean,

            "feature_std":
                feature_std,

            "gate_temperature":
                gate_temperature,

            "regime_thresholds":
                regime_thresholds,

            "architecture":
                "RMPG-U-v2-fixed",

            "implementation":
                "single-forward gate supervision"

        },

        OUT /
        f"{task}_RMPGU_v2_fixed_seed{seed}.pt"

    )


    # ========================================================
    # RESULT ROW
    # ========================================================

    result = {

        "task":
            task,

        "model":
            "RMPG-U-v2-fixed",

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
            ],

        "gate_accuracy":
            gate_accuracy,

        "gate_brier":
            gate_brier,

        "gate_temperature":
            gate_temperature,

        "mean_cv_gate":
            float(
                alpha.mean()
            ),

        "mean_residual_magnitude":
            float(
                residual_sample.mean()
            )

    }


    return (

        pd.DataFrame(
            training_rows
        ),

        result,

        gate_by_regime,

        gate_by_time

    )


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

all_training = []
all_results = []
all_gate_regime = []
all_gate_time = []


for task in TASKS:


    for seed in SEEDS:


        (
            training_history,
            result,
            gate_regime,
            gate_time

        ) = train_one(

            task,
            seed

        )


        all_training.append(
            training_history
        )


        all_results.append(
            result
        )


        all_gate_regime.append(
            gate_regime
        )


        all_gate_time.append(
            gate_time
        )


# ============================================================
# TRAINING HISTORY
# ============================================================

training_df = pd.concat(

    all_training,

    ignore_index=True

)


training_df.to_csv(

    OUT /
    "rmpgu_v2_fixed_training_history.csv",

    index=False

)


# ============================================================
# SEED-LEVEL RESULTS
# ============================================================

results = pd.DataFrame(
    all_results
)


results.to_csv(

    OUT /
    "rmpgu_v2_fixed_validation_seed_results.csv",

    index=False

)


# ============================================================
# MULTI-SEED SUMMARY
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
    "mean_sigma",
    "gate_accuracy",
    "gate_brier",
    "mean_cv_gate",
    "mean_residual_magnitude"

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
    "rmpgu_v2_fixed_validation_summary.csv",

    index=False

)


# ============================================================
# GATE BY REGIME
# ============================================================

gate_regime_df = pd.concat(

    all_gate_regime,

    ignore_index=True

)


gate_regime_df.to_csv(

    OUT /
    "rmpgu_v2_fixed_gate_by_regime.csv",

    index=False

)


# ============================================================
# GATE BY FORECAST TIME
# ============================================================

gate_time_df = pd.concat(

    all_gate_time,

    ignore_index=True

)


gate_time_df.to_csv(

    OUT /
    "rmpgu_v2_fixed_gate_by_time.csv",

    index=False

)


# ============================================================
# BASELINE VALIDATION COMPARISON
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
# LSTM / GRU
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
# AMPM-U v1
# ============================================================

v1_file = (

    V1_DIR /
    "ampmu_validation_seed_results.csv"

)


if v1_file.exists():


    v1 = pd.read_csv(
        v1_file
    )


    for _, row in v1.iterrows():


        comparison_seed_rows.append({

            "task":
                row[
                    "task"
                ],

            "model":
                "AMPM-U-v1",

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
# OLD RMPG-U v2
# ============================================================

old_v2_file = (

    OLD_V2_DIR /
    "rmpgu_v2_validation_seed_results.csv"

)


if old_v2_file.exists():


    old_v2 = pd.read_csv(
        old_v2_file
    )


    for _, row in old_v2.iterrows():


        comparison_seed_rows.append({

            "task":
                row[
                    "task"
                ],

            "model":
                "RMPG-U-v2-old",

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
# FIXED V2
# ============================================================

for _, row in results.iterrows():


    comparison_seed_rows.append({

        "task":
            row[
                "task"
            ],

        "model":
            "RMPG-U-v2-fixed",

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


# ============================================================
# SUMMARY COMPARISON
# ============================================================

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
    "rmpgu_v2_fixed_model_comparison.csv",

    index=False

)


# ============================================================
# MEAN GATE-BY-TIME ACROSS SEEDS
# ============================================================

gate_time_mean = (

    gate_time_df.groupby(

        [
            "task",
            "regime",
            "forecast_step",
            "forecast_time_sec"
        ],

        as_index=False

    )

    .agg(

        stationary_weight_mean=(
            "stationary_weight",
            "mean"
        ),

        stationary_weight_std=(
            "stationary_weight",
            "std"
        ),

        cv_weight_mean=(
            "cv_weight",
            "mean"
        ),

        cv_weight_std=(
            "cv_weight",
            "std"
        ),

        residual_magnitude_mean=(
            "residual_magnitude",
            "mean"
        ),

        residual_magnitude_std=(
            "residual_magnitude",
            "std"
        )

    )

)


gate_time_mean.to_csv(

    OUT /
    "rmpgu_v2_fixed_gate_by_time_mean.csv",

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
        "RMPG-U v2 FIXED — TRAINREAL VALIDATION EXPERIMENT\n"
    )

    f.write(
        "=" * 82
        +
        "\n\n"
    )


    f.write(
        "IMPLEMENTATION FIX\n"
    )

    f.write(
        "- Prediction and auxiliary gate supervision now use "
        "gate_logits from the SAME forward pass.\n"
    )

    f.write(
        "- The previous second encoder/fusion/dropout pass "
        "has been removed.\n"
    )

    f.write(
        "- Architecture and hyperparameters are otherwise unchanged.\n\n"
    )


    f.write(
        "PROTOCOL\n"
    )

    f.write(
        "- Training: BIRDSAI TrainReal neural-training tracks only.\n"
    )

    f.write(
        "- Validation: fixed track-disjoint TrainReal validation.\n"
    )

    f.write(
        "- BIRDSAI TestReal NOT loaded.\n"
    )

    f.write(
        "- WildBox NOT loaded.\n"
    )

    f.write(
        "- Feature normalization: training samples only.\n"
    )

    f.write(
        "- Gate temperature: training targets only.\n"
    )

    f.write(
        "- Validation future is used only after checkpoint selection "
        "for gate diagnostic scoring.\n\n"
    )


    f.write(
        "ARCHITECTURE\n"
    )

    f.write(
        "- GRU temporal encoder\n"
    )

    f.write(
        "- explicit history-only motion features\n"
    )

    f.write(
        "- stationary/CV physical-prior gate\n"
    )

    f.write(
        "- nonlinear residual trajectory branch\n"
    )

    f.write(
        "- heteroscedastic Gaussian uncertainty head\n\n"
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
        "\n\nFIXED RMPG-U v2 SUMMARY\n"
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
        "\n\nGATE BY MOTION REGIME\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        gate_regime_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nGATE-BY-TIME FILES\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        "Detailed seed-level temporal gate behavior:\n"
    )

    f.write(
        "rmpgu_v2_fixed_gate_by_time.csv\n\n"
    )

    f.write(
        "Mean ± std temporal gate behavior across seeds:\n"
    )

    f.write(
        "rmpgu_v2_fixed_gate_by_time_mean.csv\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print(
    "\n============================================"
)

print(
    "RMPG-U v2 FIXED VALIDATION COMPLETE"
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
    "\nFixed v2 summary:"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nGate by regime:"
)

print(
    gate_regime_df.to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    OUT
)