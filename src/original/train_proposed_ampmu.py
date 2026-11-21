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

OUT = Path(
    r"C:\Users\HP\Downloads\proposed_ampmu_validation"
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
NUM_LAYERS = 2
DROPOUT = 0.10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 300
PATIENCE = 30
MIN_DELTA = 1e-5
GRAD_CLIP = 5.0

CV_RECENT_FRAMES = 10

LAMBDA_POINT = 1.0

# Gaussian confidence ellipse chi-square thresholds
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
# DATASET
# ============================================================

class TrajectoryDataset(Dataset):

    def __init__(
        self,
        history,
        future,
        weights
    ):

        self.history = torch.tensor(
            history,
            dtype=torch.float32
        )

        self.future = torch.tensor(
            future,
            dtype=torch.float32
        )

        self.weights = torch.tensor(
            weights,
            dtype=torch.float32
        )


    def __len__(self):

        return len(
            self.history
        )


    def __getitem__(self, idx):

        return (
            self.history[idx],
            self.future[idx],
            self.weights[idx]
        )


# ============================================================
# CONSTANT VELOCITY PRIOR
# ============================================================

def constant_velocity_prior(
    history,
    future_length
):

    """
    history: [B, T, 2]

    Estimate recent velocity with least squares,
    then extrapolate.

    Operates in canonical coordinate space.
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


    times = torch.arange(

        k,

        dtype=history.dtype,

        device=history.device

    ) * DT


    times = (
        times
        -
        times.mean()
    )


    denom = torch.sum(
        times ** 2
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
        times.view(
            1,
            k,
            1
        )

    ).sum(
        dim=1
    ) / denom


    future_times = (

        torch.arange(

            1,
            future_length + 1,

            dtype=history.dtype,

            device=history.device

        )

        *
        DT

    )


    prior = (

        history[
            :,
            -1:,
            :
        ]

        +

        velocity.unsqueeze(1)

        *
        future_times.view(
            1,
            future_length,
            1
        )

    )


    return prior


# ============================================================
# PROPOSED MODEL
# ============================================================

class AMPMU(nn.Module):

    """
    Adaptive Motion-Prior Mixture with Uncertainty

    Experts:
      1. Stationary prior
      2. Constant-velocity prior
      3. Learned nonlinear trajectory expert

    The gate predicts per-future-step mixture weights.

    An uncertainty head predicts diagonal Gaussian sigma
    for x and y at every future step.
    """

    def __init__(
        self,
        future_length
    ):

        super().__init__()


        self.future_length = (
            future_length
        )


        self.encoder = nn.GRU(

            input_size=2,

            hidden_size=HIDDEN_SIZE,

            num_layers=NUM_LAYERS,

            batch_first=True,

            dropout=DROPOUT
        )


        # --------------------------------------------
        # Nonlinear trajectory expert
        # --------------------------------------------

        self.neural_head = nn.Sequential(

            nn.Linear(
                HIDDEN_SIZE,
                HIDDEN_SIZE
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_SIZE,
                future_length * 2
            )
        )


        # --------------------------------------------
        # Per-step expert gate:
        # stationary / CV / nonlinear
        # --------------------------------------------

        self.gate_head = nn.Sequential(

            nn.Linear(
                HIDDEN_SIZE,
                HIDDEN_SIZE
            ),

            nn.GELU(),

            nn.Linear(
                HIDDEN_SIZE,
                future_length * 3
            )
        )


        # --------------------------------------------
        # Diagonal Gaussian uncertainty
        # --------------------------------------------

        self.uncertainty_head = nn.Sequential(

            nn.Linear(
                HIDDEN_SIZE,
                HIDDEN_SIZE
            ),

            nn.GELU(),

            nn.Linear(
                HIDDEN_SIZE,
                future_length * 2
            )
        )


    def forward(self, history):


        _, hidden = self.encoder(
            history
        )


        latent = hidden[
            -1
        ]


        # ============================================
        # EXPERT 1 — STATIONARY
        # ============================================

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


        # ============================================
        # EXPERT 2 — CONSTANT VELOCITY
        # ============================================

        cv = constant_velocity_prior(

            history,

            self.future_length

        )


        # ============================================
        # EXPERT 3 — NONLINEAR NEURAL
        # ============================================

        neural = (

            self.neural_head(
                latent
            )

            .view(
                -1,
                self.future_length,
                2
            )

        )


        # ============================================
        # ADAPTIVE MIXTURE
        # ============================================

        gate_logits = (

            self.gate_head(
                latent
            )

            .view(
                -1,
                self.future_length,
                3
            )

        )


        gate = torch.softmax(

            gate_logits,

            dim=-1

        )


        experts = torch.stack(

            [
                stationary,
                cv,
                neural
            ],

            dim=2

        )
        # [B, future, expert, xy]


        mean = (

            gate.unsqueeze(
                -1
            )

            *
            experts

        ).sum(
            dim=2
        )


        # ============================================
        # UNCERTAINTY
        # ============================================

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


        # numerical stability

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
            gate
        )


# ============================================================
# LOSS
# ============================================================

def weighted_probabilistic_loss(
    mean,
    log_sigma,
    target,
    weights
):


    sigma = torch.exp(
        log_sigma
    )


    normalized_error = (

        target
        -
        mean

    ) / sigma


    # diagonal Gaussian negative log likelihood

    nll = (

        0.5
        *
        normalized_error ** 2

        +
        log_sigma

    ).mean(
        dim=(1, 2)
    )


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


    sample_loss = (

        nll

        +
        LAMBDA_POINT
        *
        point_loss

    )


    weighted = (

        sample_loss

        *
        weights

    )


    return (

        weighted.sum()

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
            )
    }


# ============================================================
# UNCERTAINTY CALIBRATION
# ============================================================

def uncertainty_metrics(
    prediction_canonical,
    truth_canonical,
    sigma
):


    standardized = (

        truth_canonical
        -
        prediction_canonical

    ) / sigma


    mahalanobis_sq = np.sum(

        standardized ** 2,

        axis=2

    )


    coverage50 = float(

        np.mean(
            mahalanobis_sq
            <=
            CHI2_50
        )

    )


    coverage90 = float(

        np.mean(
            mahalanobis_sq
            <=
            CHI2_90
        )

    )


    coverage95 = float(

        np.mean(
            mahalanobis_sq
            <=
            CHI2_95
        )

    )


    mean_sigma = float(
        np.mean(
            sigma
        )
    )


    return {

        "coverage50":
            coverage50,

        "coverage90":
            coverage90,

        "coverage95":
            coverage95,

        "mean_sigma":
            mean_sigma
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_model(
    model,
    history,
    batch_size=256
):


    model.eval()


    means = []
    sigmas = []
    gates = []


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


            (
                mean,
                _,
                sigma,
                gate

            ) = model(
                x
            )


            means.append(

                mean.cpu()
                .numpy()

            )


            sigmas.append(

                sigma.cpu()
                .numpy()

            )


            gates.append(

                gate.cpu()
                .numpy()

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
            gates,
            axis=0
        )
    )


# ============================================================
# MOTION REGIME FEATURES
# ============================================================

def history_features(
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


    path_length = float(
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
        path_length

        if path_length > 1e-12

        else 0.0
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


    return (
        path_length,
        straightness,
        heading_change
    )


def derive_regime_thresholds(
    histories
):


    features = [

        history_features(
            h
        )

        for h in histories

    ]


    features = np.asarray(
        features,
        dtype=np.float64
    )


    return {

        "slow":
            float(
                np.quantile(
                    features[:, 0],
                    0.25
                )
            ),

        "straight":
            float(
                np.quantile(
                    features[:, 1],
                    0.50
                )
            ),

        "heading":
            float(
                np.quantile(
                    features[:, 2],
                    0.50
                )
            )
    }


def assign_regime(
    history,
    threshold
):


    (
        path,
        straightness,
        heading

    ) = history_features(
        history
    )


    if path <= threshold[
        "slow"
    ]:

        return "slow_stationary"


    if (

        straightness
        >=
        threshold[
            "straight"
        ]

        and

        heading
        <=
        threshold[
            "heading"
        ]

    ):

        return "persistent_linear"


    return "irregular_turning"


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation(
    model,
    X_val,
    Y_val_can,
    Y_val_abs,
    val_meta,
    origin,
    angle,
    scale
):


    (
        pred_can,
        sigma,
        gate

    ) = predict_model(

        model,
        X_val

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

        val_meta

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
        gate
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
        "| AMPM-U | seed",
        seed
    )

    print(
        "============================================"
    )


    npz_file = (

        ROOT /
        f"{task}_birdsai_trainval_canonical.npz"

    )


    meta_file = (

        ROOT /
        f"{task}_birdsai_trainval_metadata.csv"

    )


    arrays = np.load(
        npz_file
    )


    metadata = pd.read_csv(

        meta_file,

        dtype={
            "source_group": str,
            "track_key": str
        }
    )


    train_indices = arrays[
        "train_indices"
    ].astype(int)


    val_indices = arrays[
        "validation_indices"
    ].astype(int)


    weights = arrays[
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


    origin = arrays[
        "origin"
    ]


    angle = arrays[
        "heading_angle"
    ]


    scale = arrays[
        "scale"
    ]


    # ============================================
    # TRAIN
    # ============================================

    X_train = history_can[
        train_indices
    ]


    Y_train = future_can[
        train_indices
    ]


    train_abs_history = history_abs[
        train_indices
    ]


    # ============================================
    # VALIDATION
    # ============================================

    X_val = history_can[
        val_indices
    ]


    Y_val_can = future_can[
        val_indices
    ]


    Y_val_abs = future_abs[
        val_indices
    ]


    val_abs_history = history_abs[
        val_indices
    ]


    val_origin = origin[
        val_indices
    ]


    val_angle = angle[
        val_indices
    ]


    val_scale = scale[
        val_indices
    ]


    val_meta = (

        metadata.iloc[
            val_indices
        ]

        .reset_index(
            drop=True
        )

    )


    # ============================================
    # TRAIN-ONLY MOTION THRESHOLDS
    # ============================================

    regime_threshold = (
        derive_regime_thresholds(

            train_abs_history

        )
    )


    validation_regimes = [

        assign_regime(

            h,

            regime_threshold

        )

        for h
        in val_abs_history

    ]


    # ============================================
    # DATA LOADER
    # ============================================

    dataset = TrajectoryDataset(

        X_train,

        Y_train,

        weights

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


    model = AMPMU(

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


    # ============================================
    # TRAINING LOOP
    # ============================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):


        model.train()

        losses = []


        for (
            X,
            Y,
            W

        ) in loader:


            X = X.to(
                DEVICE
            )


            Y = Y.to(
                DEVICE
            )


            W = W.to(
                DEVICE
            )


            optimizer.zero_grad()


            (
                mean,
                log_sigma,
                _,
                _

            ) = model(
                X
            )


            loss = (
                weighted_probabilistic_loss(

                    mean,

                    log_sigma,

                    Y,

                    W

                )
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


        # ========================================
        # VALIDATION
        # ========================================

        (
            metrics,
            uncertainty,
            _,
            _,
            _

        ) = evaluate_validation(

            model,

            X_val,

            Y_val_can,

            Y_val_abs,

            val_meta,

            val_origin,

            val_angle,

            val_scale

        )


        val_track_ADE = metrics[
            "ADE_track_macro"
        ]


        scheduler.step(
            val_track_ADE
        )


        history_rows.append({

            "task":
                task,

            "model":
                "AMPM-U",

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

            "mean_sigma":
                uncertainty[
                    "mean_sigma"
                ],

            "learning_rate":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
        })


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
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_state
    )


    (
        metrics,
        uncertainty,
        pred_can,
        sigma,
        gate

    ) = evaluate_validation(

        model,

        X_val,

        Y_val_can,

        Y_val_abs,

        val_meta,

        val_origin,

        val_angle,

        val_scale

    )


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

            "best_validation_track_macro_ADE":
                best_metric,

            "hidden_size":
                HIDDEN_SIZE,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT
        },

        OUT /
        f"{task}_AMPMU_seed{seed}.pt"

    )


    # ========================================================
    # GATE INTERPRETATION ON VALIDATION ONLY
    # ========================================================

    gate_sample = gate.mean(
        axis=1
    )
    # Average future-step gate for each sample


    gate_table = pd.DataFrame({

        "task":
            task,

        "seed":
            seed,

        "regime":
            validation_regimes,

        "stationary_weight":
            gate_sample[
                :,
                0
            ],

        "cv_weight":
            gate_sample[
                :,
                1
            ],

        "neural_weight":
            gate_sample[
                :,
                2
            ]

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
                "stationary_weight",
                "mean"
            ),

            cv_weight=(
                "cv_weight",
                "mean"
            ),

            neural_weight=(
                "neural_weight",
                "mean"
            )
        )

    )


    result = {

        "task":
            task,

        "model":
            "AMPM-U",

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "validation_samples":
            len(
                val_indices
            ),

        "validation_tracks":
            val_meta[
                "track_key"
            ].nunique(),

        "validation_sources":
            val_meta[
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

        "slow_threshold":
            regime_threshold[
                "slow"
            ],

        "straight_threshold":
            regime_threshold[
                "straight"
            ],

        "heading_threshold":
            regime_threshold[
                "heading"
            ]
    }


    return (

        model,

        pd.DataFrame(
            history_rows
        ),

        result,

        gate_by_regime
    )


# ============================================================
# RUN PROPOSED MODEL
# ============================================================

all_history = []
all_results = []
all_gates = []


for task in TASKS:


    for seed in SEEDS:


        (
            model,
            history,
            result,
            gate_summary

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


        all_gates.append(
            gate_summary
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

pd.concat(

    all_history,

    ignore_index=True

).to_csv(

    OUT /
    "ampmu_training_history.csv",

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
    "ampmu_validation_seed_results.csv",

    index=False
)


# ============================================================
# MULTI-SEED SUMMARY
# ============================================================

metric_names = [

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


    for metric in metric_names:


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
    "ampmu_validation_summary.csv",

    index=False
)


# ============================================================
# GATE BY VALIDATION MOTION REGIME
# ============================================================

gate_summary = pd.concat(

    all_gates,

    ignore_index=True

)


gate_summary.to_csv(

    OUT /
    "validation_gate_by_regime.csv",

    index=False
)


# ============================================================
# PREVIOUS LEARNED BASELINE VALIDATION RESULTS
# ============================================================

baseline_rows = []


def collect_history(
    filename
):


    if not filename.exists():

        return


    table = pd.read_csv(
        filename
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


        # Same selection criterion used during training:
        # lowest validation track-macro ADE

        best = group.sort_values(

            "validation_track_macro_ADE"

        ).iloc[0]


        baseline_rows.append({

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


collect_history(

    RNN_DIR /
    "training_history.csv"

)


collect_history(

    TRANSFORMER_DIR /
    "transformer_training_history.csv"

)


baseline_validation = pd.DataFrame(
    baseline_rows
)


# ============================================================
# ADD PROPOSED MODEL
# ============================================================

proposed_for_comparison = results[
    [
        "task",
        "model",
        "seed",
        "ADE_mean",
        "FDE_mean",
        "ADE_track_macro",
        "FDE_track_macro"
    ]
].copy()


comparison_seed = pd.concat(

    [
        baseline_validation,
        proposed_for_comparison
    ],

    ignore_index=True
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
    "validation_model_comparison.csv",

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
        "AMPM-U PROPOSED MODEL — "
        "TRAINREAL VALIDATION ONLY\n"
    )

    f.write(
        "=" * 80
        +
        "\n\n"
    )


    f.write(
        "IMPORTANT PROTOCOL:\n"
    )

    f.write(
        "- Proposed model trained only on BIRDSAI "
        "TrainReal neural-training tracks.\n"
    )

    f.write(
        "- Model selection uses only track-disjoint "
        "TrainReal validation.\n"
    )

    f.write(
        "- BIRDSAI TestReal is NOT loaded by this script.\n"
    )

    f.write(
        "- WildBox is NOT loaded by this script.\n\n"
    )


    f.write(
        "Architecture:\n"
    )

    f.write(
        "- GRU temporal encoder\n"
    )

    f.write(
        "- stationary motion prior\n"
    )

    f.write(
        "- constant-velocity motion prior\n"
    )

    f.write(
        "- nonlinear neural trajectory expert\n"
    )

    f.write(
        "- per-future-step adaptive softmax gate\n"
    )

    f.write(
        "- per-step x/y Gaussian uncertainty\n\n"
    )


    f.write(
        "VALIDATION MODEL COMPARISON\n"
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
        "\n\nAMPM-U UNCERTAINTY SUMMARY\n"
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
        "\n\nVALIDATION GATE BY MOTION REGIME\n"
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


print(
    "\n============================================"
)

print(
    "AMPM-U VALIDATION EXPERIMENT COMPLETE"
)

print(
    "============================================"
)

print(
    "\nValidation comparison:"
)

print(
    comparison.to_string(
        index=False
    )
)

print(
    "\nAMPM-U summary:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nGate behavior:"
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