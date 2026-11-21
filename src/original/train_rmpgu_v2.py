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

OUT = Path(
    r"C:\Users\HP\Downloads\rmpgu_v2_validation"
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

# ------------------------------------------------------------
# Fixed loss weights.
# These will NOT be tuned using TestReal/WildBox.
# ------------------------------------------------------------

LAMBDA_NLL = 0.25
LAMBDA_GATE = 0.25
LAMBDA_RESIDUAL = 0.01

EPS = 1e-8


# Gaussian 2D ellipse thresholds

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
    history:
        original camera-corrected normalized xy
        shape [T, 2]

    Returns 7 history-only features.
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


    recent_steps = np.linalg.norm(

        np.diff(
            recent,
            axis=0
        ),

        axis=1

    )


    recent_speed = float(

        recent_steps.mean()
        *
        FPS

    )


    # --------------------------------------------------------
    # Heading change:
    # compare first half vs second half motion
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
# STANDARDIZATION
#
# FIT ON NEURAL TRAIN INDICES ONLY
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


    # protect model from extreme feature values

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
        canonical coordinates [B,T,2]
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

        pred = constant_velocity_prior_torch(
            x,
            future_length
        )


    return (
        pred[0]
        .numpy()
    )


# ============================================================
# TRAIN-ONLY SOFT GATE TARGET
# ============================================================

def build_gate_targets(
    histories_can,
    futures_can
):

    """
    Target represents how much CV should be preferred
    over Stationary at each future step.

    0 = stationary better
    1 = CV better

    Soft labels are derived ONLY from training targets.
    """

    differences = []


    for i in range(
        len(histories_can)
    ):


        h = histories_can[i]

        y = futures_can[i]


        stationary = np.repeat(

            h[
                -1:
            ],

            len(y),

            axis=0

        )


        cv = constant_velocity_prior_numpy(

            h,

            len(y)

        )


        stat_error = np.linalg.norm(

            stationary
            -
            y,

            axis=1

        )


        cv_error = np.linalg.norm(

            cv
            -
            y,

            axis=1

        )


        # Positive => CV is better

        differences.append(

            stat_error
            -
            cv_error

        )


    differences = np.stack(
        differences
    )


    # --------------------------------------------------------
    # Temperature derived from TRAIN ONLY.
    # --------------------------------------------------------

    nonzero = np.abs(
        differences
    ).reshape(
        -1
    )


    temperature = float(

        np.median(
            nonzero
        )

    )


    temperature = max(
        temperature,
        1e-3
    )


    logits = (

        differences
        /
        temperature

    )


    logits = np.clip(
        logits,
        -10.0,
        10.0
    )


    soft_target = (

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


    return (

        soft_target.astype(
            np.float32
        ),

        temperature

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
        idx
    ):

        return (
            self.history[idx],
            self.future[idx],
            self.features[idx],
            self.weights[idx],
            self.gate_targets[idx]
        )


# ============================================================
# MODEL
# ============================================================

class RMPGU(nn.Module):

    """
    Residual Motion-Prior Gated Forecaster
    with Uncertainty.

    Base prior:
      Stationary <-> Constant Velocity

    Final:
      prior + nonlinear residual
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
        # Temporal encoder
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

        self.feature_encoder = (
            nn.Sequential(

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
        )


        # ----------------------------------------------------
        # Fusion
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
        # Binary physical prior gate
        #
        # alpha=0 -> stationary
        # alpha=1 -> CV
        # ----------------------------------------------------

        self.gate_head = nn.Linear(

            FUSED_SIZE,

            future_length

        )


        # ----------------------------------------------------
        # Nonlinear residual
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
        # Uncertainty
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


        # ================================================
        # PHYSICAL PRIORS
        # ================================================

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


        cv = constant_velocity_prior_torch(

            history,

            self.future_length

        )


        # ================================================
        # PRIOR GATE
        # ================================================

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


        # ================================================
        # NONLINEAR RESIDUAL
        # ================================================

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


        # ================================================
        # UNCERTAINTY
        # ================================================

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


        return (
            mean,
            log_sigma,
            sigma,
            alpha,
            residual,
            prior
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
    # Point prediction loss
    # --------------------------------------------------------

    point = (

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


    normalized = (

        target
        -
        mean

    ) / sigma


    nll = (

        0.5
        *
        normalized ** 2

        +
        log_sigma

    ).mean(
        dim=(1, 2)
    )


    # --------------------------------------------------------
    # Gate supervision
    #
    # Future labels used here belong ONLY to training data.
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
    # Keep residual correction modest unless needed.
    # --------------------------------------------------------

    residual_penalty = (

        residual ** 2

    ).mean(
        dim=(1, 2)
    )


    total = (

        point

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
            total
            *
            weights
        ).sum()

        /
        weights.sum()

    )


# ============================================================
# INVERSE TRANSFORM
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
# TRAJECTORY METRICS
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


    temp = metadata[
        [
            "source_group",
            "track_key"
        ]
    ].copy()


    temp[
        "ADE"
    ] = ADE


    temp[
        "FDE"
    ] = FDE


    track_table = (

        temp.groupby(

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

        temp.groupby(

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
# UNCERTAINTY
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


    d2 = np.sum(

        standardized ** 2,

        axis=2

    )


    return {

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
                sigma.mean()
            )
    }


# ============================================================
# PREDICT
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
        )
    )


# ============================================================
# MOTION REGIME
#
# Only for interpreting validation gate behavior.
# Thresholds derived from neural training histories.
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

            -1,
            1
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
                    values[:, 0],
                    0.25
                )
            ),

        "straight":
            float(
                np.quantile(
                    values[:, 1],
                    0.50
                )
            ),

        "heading":
            float(
                np.quantile(
                    values[:, 2],
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
        residual

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
        residual
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
        "| RMPG-U v2 | seed",
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
    ].astype(int)


    val_idx = arrays[
        "validation_indices"
    ].astype(int)


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
    # MOTION FEATURES
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


    all_features_normalized = (
        normalize_features(

            all_features,

            feature_mean,

            feature_std

        )
    )


    # ========================================================
    # SPLITS
    # ========================================================

    X_train = history_can[
        train_idx
    ]


    Y_train = future_can[
        train_idx
    ]


    F_train = all_features_normalized[
        train_idx
    ]


    X_val = history_can[
        val_idx
    ]


    Y_val_can = future_can[
        val_idx
    ]


    Y_val_abs = future_abs[
        val_idx
    ]


    F_val = all_features_normalized[
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
    # GATE TARGETS — TRAIN TARGETS ONLY
    # ========================================================

    gate_targets, gate_temperature = (
        build_gate_targets(

            X_train,

            Y_train

        )
    )


    # ========================================================
    # MOTION REGIMES — TRAIN THRESHOLDS ONLY
    # ========================================================

    train_abs_histories = history_abs[
        train_idx
    ]


    val_abs_histories = history_abs[
        val_idx
    ]


    regime_thresholds = (
        derive_regime_thresholds(

            train_abs_histories

        )
    )


    val_regimes = [

        assign_regime(

            h,

            regime_thresholds

        )

        for h
        in val_abs_histories

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
        Y_train.shape[1]
    )


    model = RMPGU(

        future_length=
            future_length,

        feature_dim=
            F_train.shape[1]

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
    # TRAIN
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


            (
                mean,
                log_sigma,
                _,
                _,
                residual,
                _

            ) = model(
                X,
                F
            )


            # Gate logits needed separately

            _, hidden = model.encoder(
                X
            )

            temporal = hidden[
                -1
            ]


            feature_emb = (
                model.feature_encoder(
                    F
                )
            )


            fused = model.fusion(

                torch.cat(

                    [
                        temporal,
                        feature_emb
                    ],

                    dim=1
                )

            )


            gate_logits = (
                model.gate_head(
                    fused
                )
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
                "RMPG-U-v2",

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
        # BEST MODEL
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
                f"val track-ADE="
                f"{val_track_ADE:.6f} | "
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
        sigma,
        alpha,
        residual

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
    # VALIDATION ORACLE GATE
    #
    # FOR DIAGNOSTIC ONLY.
    # Not used for training/model selection.
    # ========================================================

    val_gate_target, _ = (
        build_gate_targets(

            X_val,

            Y_val_can

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
    # GATE BY REGIME
    # ========================================================

    alpha_sample = alpha.mean(
        axis=1
    )


    residual_magnitude = np.linalg.norm(

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
            residual_magnitude

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
    # CHECKPOINT
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
                "RMPG-U-v2"

        },

        OUT /
        f"{task}_RMPGU_v2_seed{seed}.pt"

    )


    result = {

        "task":
            task,

        "model":
            "RMPG-U-v2",

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
                residual_magnitude.mean()
            )

    }


    return (

        pd.DataFrame(
            training_rows
        ),

        result,

        gate_by_regime

    )


# ============================================================
# RUN
# ============================================================

all_training = []
all_results = []
all_gates = []


for task in TASKS:


    for seed in SEEDS:


        (
            history,
            result,
            gate_table

        ) = train_one(

            task,
            seed

        )


        all_training.append(
            history
        )

        all_results.append(
            result
        )

        all_gates.append(
            gate_table
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

training_df = pd.concat(

    all_training,

    ignore_index=True

)


training_df.to_csv(

    OUT /
    "rmpgu_v2_training_history.csv",

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
    "rmpgu_v2_validation_seed_results.csv",

    index=False
)


# ============================================================
# V2 SUMMARY
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
    "rmpgu_v2_validation_summary.csv",

    index=False
)


# ============================================================
# GATE BY REGIME
# ============================================================

gate_summary = pd.concat(

    all_gates,

    ignore_index=True

)


gate_summary.to_csv(

    OUT /
    "rmpgu_v2_gate_by_regime.csv",

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


add_baseline_history(

    RNN_DIR /
    "training_history.csv"

)


add_baseline_history(

    TRANSFORMER_DIR /
    "transformer_training_history.csv"

)


# ------------------------------------------------------------
# Add v1 AMPM-U if available
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Add v2
# ------------------------------------------------------------

for _, row in results.iterrows():


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
    "rmpgu_v2_model_comparison.csv",

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
        "RMPG-U v2 — TRAINREAL VALIDATION EXPERIMENT\n"
    )

    f.write(
        "=" * 80
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
        "- Validation: fixed track-disjoint TrainReal validation.\n"
    )

    f.write(
        "- TestReal NOT loaded.\n"
    )

    f.write(
        "- WildBox NOT loaded.\n"
    )

    f.write(
        "- Feature normalization fit on training samples only.\n"
    )

    f.write(
        "- Gate auxiliary targets derived from training future "
        "trajectories only.\n\n"
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
        "- Stationary/CV physical-prior gate\n"
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
        "-" * 80
        +
        "\n"
    )


    f.write(
        comparison.to_string(
            index=False
        )
    )


    f.write(
        "\n\nRMPG-U v2 SUMMARY\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )


    f.write(
        summary.to_string(
            index=False
        )
    )


    f.write(
        "\n\nGATE BY VALIDATION MOTION REGIME\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )


    f.write(
        gate_summary.to_string(
            index=False
        )
    )


    f.write(
        "\n"
    )


print(
    "\n============================================"
)

print(
    "RMPG-U v2 VALIDATION COMPLETE"
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
    "\nV2 summary:"
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
    gate_summary.to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    OUT
)