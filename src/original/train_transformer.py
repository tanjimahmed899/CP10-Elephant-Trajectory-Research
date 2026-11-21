from pathlib import Path
import copy
import random
import math

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

OUT = Path(
    r"C:\Users\HP\Downloads\transformer_results"
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


    def __getitem__(self, index):

        return (
            self.history[index],
            self.future[index],
            self.weights[index]
        )


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=100
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
            pe.unsqueeze(0)
        )


    def forward(self, x):

        return (

            x

            +

            self.pe[
                :,
                :x.size(1)
            ]

        )


# ============================================================
# TRANSFORMER
# ============================================================

class TrajectoryTransformer(nn.Module):

    def __init__(
        self,
        future_length
    ):

        super().__init__()


        self.input_projection = nn.Linear(
            2,
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


        self.head = nn.Sequential(

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


        self.future_length = (
            future_length
        )


    def forward(self, x):


        x = self.input_projection(
            x
        )


        x = self.position(
            x
        )


        x = self.encoder(
            x
        )


        # Mean pooling over the observed trajectory
        x = x.mean(
            dim=1
        )


        x = self.norm(
            x
        )


        output = self.head(
            x
        )


        return output.view(

            -1,

            self.future_length,

            2
        )


# ============================================================
# LOSS
# ============================================================

def weighted_smooth_l1(
    prediction,
    target,
    weights
):


    element_loss = (
        nn.functional.smooth_l1_loss(

            prediction,

            target,

            reduction="none"

        )
    )


    sample_loss = element_loss.mean(
        dim=(1, 2)
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
# PREDICT
# ============================================================

def predict_model(
    model,
    history,
    batch_size=256
):


    model.eval()

    predictions = []


    with torch.no_grad():


        for start in range(

            0,
            len(history),
            batch_size

        ):


            batch = torch.tensor(

                history[
                    start:
                    start + batch_size
                ],

                dtype=torch.float32,

                device=DEVICE
            )


            prediction = model(
                batch
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


    temp = metadata[
        [
            "source_group",
            "track_key"
        ]
    ].copy()


    temp["ADE"] = ADE
    temp["FDE"] = FDE


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
            ),

        "ADE_array":
            ADE,

        "FDE_array":
            FDE
    }


# ============================================================
# VALIDATION
# ============================================================

def validation_metric(
    model,
    history,
    truth_abs,
    metadata,
    origin,
    angle,
    scale
):


    canonical_prediction = (
        predict_model(
            model,
            history
        )
    )


    prediction_abs = (
        inverse_transform_batch(

            canonical_prediction,

            origin,

            angle,

            scale
        )
    )


    metrics = calculate_metrics(

        prediction_abs,

        truth_abs,

        metadata
    )


    return (

        metrics[
            "ADE_track_macro"
        ],

        metrics
    )


# ============================================================
# TRAIN ONE SEED
# ============================================================

def train_one(
    task,
    seed
):


    set_seed(
        seed
    )


    print(
        "\n=========================================="
    )

    print(
        task,
        "| Transformer | seed",
        seed
    )

    print(
        "=========================================="
    )


    npz_path = (

        ROOT /
        f"{task}_birdsai_trainval_canonical.npz"

    )


    metadata_path = (

        ROOT /
        f"{task}_birdsai_trainval_metadata.csv"

    )


    arrays = np.load(
        npz_path
    )


    metadata = pd.read_csv(

        metadata_path,

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


    histories = arrays[
        "history_canonical"
    ]


    futures = arrays[
        "future_canonical"
    ]


    truth_abs = arrays[
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


    X_train = histories[
        train_indices
    ]


    Y_train = futures[
        train_indices
    ]


    X_val = histories[
        val_indices
    ]


    val_truth_abs = truth_abs[
        val_indices
    ]


    val_origin = origins[
        val_indices
    ]


    val_angle = angles[
        val_indices
    ]


    val_scale = scales[
        val_indices
    ]


    val_metadata = (

        metadata.iloc[
            val_indices
        ]

        .reset_index(
            drop=True
        )
    )


    train_dataset = (
        TrajectoryDataset(

            X_train,

            Y_train,

            weights
        )
    )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        generator=generator,

        num_workers=0
    )


    future_length = (
        Y_train.shape[1]
    )


    model = TrajectoryTransformer(

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


            prediction = model(
                X
            )


            loss = weighted_smooth_l1(

                prediction,

                Y,

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


        train_loss = float(
            np.mean(
                losses
            )
        )


        val_value, val_metrics = (
            validation_metric(

                model,

                X_val,

                val_truth_abs,

                val_metadata,

                val_origin,

                val_angle,

                val_scale
            )
        )


        scheduler.step(
            val_value
        )


        history_rows.append({

            "task":
                task,

            "model":
                "Transformer",

            "seed":
                seed,

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_ADE":
                val_metrics[
                    "ADE_mean"
                ],

            "validation_FDE":
                val_metrics[
                    "FDE_mean"
                ],

            "validation_track_macro_ADE":
                val_metrics[
                    "ADE_track_macro"
                ],

            "validation_track_macro_FDE":
                val_metrics[
                    "FDE_track_macro"
                ],

            "learning_rate":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
        })


        if (

            val_value

            <
            best_metric
            -
            MIN_DELTA

        ):


            best_metric = (
                val_value
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
                f"loss={train_loss:.6f} | "
                f"val track-ADE={val_value:.6f} | "
                f"best={best_metric:.6f}"

            )


        if no_improvement >= PATIENCE:


            print(

                "Early stopping at epoch",
                epoch

            )

            break


    model.load_state_dict(
        best_state
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
        f"{task}_Transformer_seed{seed}.pt"
    )


    return (

        model,

        pd.DataFrame(
            history_rows
        ),

        best_epoch
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    model,
    task,
    dataset_name,
    role,
    npz_path,
    metadata_path,
    seed,
    best_epoch
):


    arrays = np.load(
        npz_path
    )


    metadata = pd.read_csv(

        metadata_path,

        dtype={
            "source_group": str,
            "track_key": str
        }
    )


    history = arrays[
        "history_canonical"
    ]


    truth_abs = arrays[
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


    pred_can = predict_model(

        model,

        history
    )


    pred_abs = inverse_transform_batch(

        pred_can,

        origin,

        angle,

        scale
    )


    metrics = calculate_metrics(

        pred_abs,

        truth_abs,

        metadata
    )


    np.savez_compressed(

        OUT /
        (
            f"{task}_{dataset_name}_"
            f"Transformer_seed{seed}_predictions.npz"
        ),

        truth_abs=
            truth_abs.astype(
                np.float32
            ),

        prediction_abs=
            pred_abs.astype(
                np.float32
            ),

        prediction_canonical=
            pred_can.astype(
                np.float32
            )
    )


    sample_table = metadata[
        [
            "source_group",
            "track_key"
        ]
    ].copy()


    sample_table[
        "ADE"
    ] = metrics[
        "ADE_array"
    ]


    sample_table[
        "FDE"
    ] = metrics[
        "FDE_array"
    ]


    sample_table[
        "task"
    ] = task


    sample_table[
        "dataset"
    ] = dataset_name


    sample_table[
        "role"
    ] = role


    sample_table[
        "model"
    ] = "Transformer"


    sample_table[
        "seed"
    ] = seed


    result = {

        "task":
            task,

        "dataset":
            dataset_name,

        "role":
            role,

        "model":
            "Transformer",

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "samples":
            len(
                metadata
            ),

        "tracks":
            metadata[
                "track_key"
            ].nunique(),

        "sources":
            metadata[
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
    }


    return (
        result,
        sample_table
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

all_results = []

all_samples = []

all_history = []


for task in TASKS:


    for seed in SEEDS:


        (
            model,
            training_history,
            best_epoch

        ) = train_one(

            task,
            seed

        )


        all_history.append(
            training_history
        )


        # BIRDSAI TestReal

        result, samples = evaluate(

            model=model,

            task=task,

            dataset_name=
                "BIRDSAI_test",

            role=
                "in_domain_test",

            npz_path=
                ROOT /
                f"{task}_birdsai_test_canonical.npz",

            metadata_path=
                ROOT /
                f"{task}_birdsai_test_metadata.csv",

            seed=seed,

            best_epoch=
                best_epoch
        )


        all_results.append(
            result
        )


        all_samples.append(
            samples
        )


        # WildBox external

        result, samples = evaluate(

            model=model,

            task=task,

            dataset_name=
                "WildBox_external",

            role=
                "external_test",

            npz_path=
                ROOT /
                f"{task}_wildbox_external_canonical.npz",

            metadata_path=
                ROOT /
                f"{task}_wildbox_external_metadata.csv",

            seed=seed,

            best_epoch=
                best_epoch
        )


        all_results.append(
            result
        )


        all_samples.append(
            samples
        )


# ============================================================
# SAVE SEED RESULTS
# ============================================================

result_df = pd.DataFrame(
    all_results
)


result_df.to_csv(

    OUT /
    "transformer_seed_results.csv",

    index=False
)


# ============================================================
# TRAINING HISTORY
# ============================================================

pd.concat(

    all_history,

    ignore_index=True

).to_csv(

    OUT /
    "transformer_training_history.csv",

    index=False
)


# ============================================================
# SAMPLE RESULTS
# ============================================================

sample_df = pd.concat(

    all_samples,

    ignore_index=True
)


sample_df.to_csv(

    OUT /
    "transformer_sample_results.csv",

    index=False
)


# ============================================================
# MULTI-SEED SUMMARY
# ============================================================

metrics = [

    "ADE_mean",
    "FDE_mean",
    "ADE_track_macro",
    "FDE_track_macro",
    "ADE_source_macro",
    "FDE_source_macro"
]


summary_rows = []


for (
    task,
    dataset,
    role,
    model
), group in result_df.groupby(

    [
        "task",
        "dataset",
        "role",
        "model"
    ]

):


    row = {

        "task":
            task,

        "dataset":
            dataset,

        "role":
            role,

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


    for metric in metrics:


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
    "transformer_summary.csv",

    index=False
)


# ============================================================
# PER SOURCE
# ============================================================

per_source = (

    sample_df.groupby(

        [
            "task",
            "dataset",
            "role",
            "model",
            "seed",
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


per_source.to_csv(

    OUT /
    "transformer_per_source.csv",

    index=False
)


# ============================================================
# SUMMARY TXT
# ============================================================

with open(

    OUT /
    "SUMMARY.txt",

    "w",
    encoding="utf-8"

) as f:


    f.write(
        "TEMPORAL TRANSFORMER ELEPHANT TRAJECTORY FORECASTING\n"
    )

    f.write(
        "=" * 78
        +
        "\n\n"
    )


    f.write(
        "Training: BIRDSAI TrainReal only\n"
    )

    f.write(
        "Validation: fixed track-disjoint TrainReal validation\n"
    )

    f.write(
        "Model selection: validation track-macro ADE\n"
    )

    f.write(
        "TestReal and WildBox never used for model selection.\n"
    )

    f.write(
        f"Seeds: {SEEDS}\n"
    )

    f.write(
        "Architecture: 2-layer compact Transformer encoder, "
        "d_model=64, 4 heads, FFN=128.\n"
    )

    f.write(
        "Metrics are calculated after inverse canonical transform.\n\n"
    )


    for _, row in (

        summary.sort_values(

            [
                "task",
                "dataset"
            ]

        ).iterrows()

    ):


        f.write(

            f"{row['task']} | "
            f"{row['dataset']}\n"

        )


        f.write(

            f"ADE = "
            f"{row['ADE_mean_mean']:.6f} ± "
            f"{row['ADE_mean_std']:.6f}\n"

        )


        f.write(

            f"FDE = "
            f"{row['FDE_mean_mean']:.6f} ± "
            f"{row['FDE_mean_std']:.6f}\n"

        )


        f.write(

            f"Track-macro ADE = "
            f"{row['ADE_track_macro_mean']:.6f} ± "
            f"{row['ADE_track_macro_std']:.6f}\n"

        )


        f.write(

            f"Source-macro ADE = "
            f"{row['ADE_source_macro_mean']:.6f} ± "
            f"{row['ADE_source_macro_std']:.6f}\n"

        )


        f.write(

            f"Mean best epoch = "
            f"{row['mean_best_epoch']:.1f}\n\n"

        )


print(
    "\n============================================"
)

print(
    "TRANSFORMER EXPERIMENT COMPLETE"
)

print(
    "============================================"
)

print(
    "\nResults:"
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