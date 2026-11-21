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

OUT = Path(
    r"C:\Users\HP\Downloads\lstm_gru_results"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

SEEDS = [42, 52, 62]

MODELS = [
    "LSTM",
    "GRU"
]

TASKS = [
    "2s_to_2s",
    "2s_to_5s"
]

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
        weights=None
    ):

        self.history = torch.tensor(
            history,
            dtype=torch.float32
        )

        self.future = torch.tensor(
            future,
            dtype=torch.float32
        )

        if weights is None:

            weights = np.ones(
                len(history),
                dtype=np.float32
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
# MODEL
# ============================================================

class RecurrentForecaster(nn.Module):

    def __init__(
        self,
        model_type,
        future_length
    ):

        super().__init__()

        self.model_type = model_type

        self.future_length = future_length


        if model_type == "LSTM":

            self.encoder = nn.LSTM(

                input_size=2,

                hidden_size=HIDDEN_SIZE,

                num_layers=NUM_LAYERS,

                batch_first=True,

                dropout=DROPOUT
            )


        elif model_type == "GRU":

            self.encoder = nn.GRU(

                input_size=2,

                hidden_size=HIDDEN_SIZE,

                num_layers=NUM_LAYERS,

                batch_first=True,

                dropout=DROPOUT
            )


        else:

            raise ValueError(
                model_type
            )


        self.head = nn.Sequential(

            nn.Linear(
                HIDDEN_SIZE,
                HIDDEN_SIZE
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE,
                future_length * 2
            )
        )


    def forward(self, x):


        if self.model_type == "LSTM":

            _, (hidden, _) = self.encoder(
                x
            )

        else:

            _, hidden = self.encoder(
                x
            )


        latent = hidden[
            -1
        ]


        output = self.head(
            latent
        )


        return output.view(

            -1,

            self.future_length,

            2
        )


# ============================================================
# WEIGHTED LOSS
# ============================================================

def weighted_smooth_l1(
    prediction,
    target,
    weights
):

    # loss per coordinate
    element_loss = nn.functional.smooth_l1_loss(

        prediction,

        target,

        reduction="none"

    )


    # one loss per sample
    sample_loss = element_loss.mean(
        dim=(1, 2)
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


        rotated = (

            canonical[i]
            *
            float(
                scale[i]
            )

        )


        # canonical = local @ R.T / scale
        # therefore local = canonical*scale @ R

        local = (
            rotated
            @
            R
        )


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
# BASIC TRAJECTORY ERRORS
# ============================================================

def sample_ade_fde(
    prediction,
    truth
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


    return ADE, FDE


# ============================================================
# MACRO METRICS
# ============================================================

def calculate_metrics(
    prediction,
    truth,
    metadata
):

    ADE, FDE = sample_ade_fde(

        prediction,

        truth

    )


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


    # --------------------------------------------
    # Window-level
    # --------------------------------------------

    window_ADE = float(
        ADE.mean()
    )

    window_FDE = float(
        FDE.mean()
    )


    # --------------------------------------------
    # Track macro
    # --------------------------------------------

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


    # --------------------------------------------
    # Source macro
    # --------------------------------------------

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

        "ADE_mean":
            window_ADE,

        "FDE_mean":
            window_FDE,

        "ADE_track_macro":
            track_ADE,

        "FDE_track_macro":
            track_FDE,

        "ADE_source_macro":
            source_ADE,

        "FDE_source_macro":
            source_FDE,

        "ADE_array":
            ADE,

        "FDE_array":
            FDE
    }


# ============================================================
# PREDICT CANONICAL
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


            x = torch.tensor(

                history[
                    start:
                    start
                    +
                    batch_size
                ],

                dtype=torch.float32,

                device=DEVICE

            )


            pred = model(
                x
            )


            predictions.append(

                pred.cpu()
                .numpy()

            )


    return np.concatenate(

        predictions,

        axis=0

    )


# ============================================================
# VALIDATION METRIC
# ============================================================

def validation_track_macro_ADE(
    model,
    val_history,
    val_future_abs,
    val_meta,
    val_origin,
    val_angle,
    val_scale
):


    prediction_canonical = (
        predict_model(

            model,

            val_history

        )
    )


    prediction_abs = (
        inverse_transform_batch(

            prediction_canonical,

            val_origin,

            val_angle,

            val_scale

        )
    )


    metrics = calculate_metrics(

        prediction_abs,

        val_future_abs,

        val_meta

    )


    return (
        metrics[
            "ADE_track_macro"
        ],

        metrics
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_one(
    task,
    model_type,
    seed
):

    set_seed(
        seed
    )


    print(
        "\n========================================"
    )

    print(
        task,
        model_type,
        "seed",
        seed
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # LOAD TRAIN / VALIDATION
    # --------------------------------------------------------

    trainval_file = (

        ROOT /
        f"{task}_birdsai_trainval_canonical.npz"

    )


    trainval_meta_file = (

        ROOT /
        f"{task}_birdsai_trainval_metadata.csv"

    )


    arrays = np.load(
        trainval_file
    )


    metadata = pd.read_csv(

        trainval_meta_file,

        dtype={
            "source_group": str,
            "track_key": str
        }
    )


    train_indices = arrays[
        "train_indices"
    ].astype(
        int
    )


    val_indices = arrays[
        "validation_indices"
    ].astype(
        int
    )


    train_weights = arrays[
        "train_sample_weights"
    ].astype(
        np.float32
    )


    history_canonical = arrays[
        "history_canonical"
    ]


    future_canonical = arrays[
        "future_canonical"
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


    X_train = history_canonical[
        train_indices
    ]


    Y_train = future_canonical[
        train_indices
    ]


    X_val = history_canonical[
        val_indices
    ]


    Y_val_abs = future_abs[
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


    # --------------------------------------------------------
    # TRACK BALANCED TRAIN DATA
    # --------------------------------------------------------

    train_dataset = TrajectoryDataset(

        X_train,

        Y_train,

        train_weights

    )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        generator=generator,

        num_workers=0

    )


    future_length = (
        Y_train.shape[1]
    )


    model = RecurrentForecaster(

        model_type,

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
        torch.optim.lr_scheduler.ReduceLROnPlateau(

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

    patience_counter = 0

    history_rows = []


    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):


        model.train()

        epoch_losses = []


        for (
            X_batch,
            Y_batch,
            W_batch

        ) in train_loader:


            X_batch = X_batch.to(
                DEVICE
            )


            Y_batch = Y_batch.to(
                DEVICE
            )


            W_batch = W_batch.to(
                DEVICE
            )


            optimizer.zero_grad()


            prediction = model(
                X_batch
            )


            loss = weighted_smooth_l1(

                prediction,

                Y_batch,

                W_batch

            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                GRAD_CLIP

            )


            optimizer.step()


            epoch_losses.append(
                float(
                    loss.item()
                )
            )


        train_loss = float(
            np.mean(
                epoch_losses
            )
        )


        # ----------------------------------------
        # TRACK-MACRO VALIDATION ADE
        # ----------------------------------------

        val_metric, val_metrics = (
            validation_track_macro_ADE(

                model,

                X_val,

                Y_val_abs,

                val_meta,

                val_origin,

                val_angle,

                val_scale

            )
        )


        scheduler.step(
            val_metric
        )


        learning_rate = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )


        history_rows.append({

            "task":
                task,

            "model":
                model_type,

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
                learning_rate
        })


        if (

            val_metric

            <
            best_metric
            -
            MIN_DELTA

        ):


            best_metric = (
                val_metric
            )


            best_epoch = epoch


            best_state = copy.deepcopy(
                model.state_dict()
            )


            patience_counter = 0


        else:


            patience_counter += 1


        if (

            epoch == 1

            or

            epoch % 10 == 0

        ):


            print(

                f"Epoch {epoch:03d} | "
                f"loss={train_loss:.6f} | "
                f"val track-ADE="
                f"{val_metric:.6f} | "
                f"best="
                f"{best_metric:.6f}"

            )


        if (

            patience_counter
            >=
            PATIENCE

        ):


            print(

                "Early stopping at epoch",
                epoch

            )

            break


    # --------------------------------------------------------
    # RESTORE BEST
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )


    model_file = (

        OUT /
        f"{task}_{model_type}_seed{seed}.pt"

    )


    torch.save(

        {
            "state_dict":
                best_state,

            "task":
                task,

            "model_type":
                model_type,

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

        model_file

    )


    history_df = pd.DataFrame(
        history_rows
    )


    return (
        model,
        history_df,
        best_epoch,
        best_metric
    )


# ============================================================
# EVALUATE FIXED MODEL
# ============================================================

def evaluate_dataset(
    model,
    task,
    dataset_name,
    role,
    npz_path,
    meta_path,
    model_type,
    seed,
    best_epoch
):


    arrays = np.load(
        npz_path
    )


    metadata = pd.read_csv(

        meta_path,

        dtype={
            "source_group": str,
            "track_key": str
        }
    )


    X = arrays[
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

        X

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


    # --------------------------------------------------------
    # PER-SAMPLE RESULTS
    # --------------------------------------------------------

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
    ] = model_type


    sample_table[
        "seed"
    ] = seed


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    np.savez_compressed(

        OUT /
        (
            f"{task}_{dataset_name}_"
            f"{model_type}_seed{seed}_predictions.npz"
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


    summary = {

        "task":
            task,

        "dataset":
            dataset_name,

        "role":
            role,

        "model":
            model_type,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "samples":
            len(
                X
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
        summary,
        sample_table
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

all_training_history = []

all_results = []

all_sample_results = []


for task in TASKS:


    for model_type in MODELS:


        for seed in SEEDS:


            (
                model,
                training_history,
                best_epoch,
                best_val

            ) = train_one(

                task,

                model_type,

                seed

            )


            all_training_history.append(
                training_history
            )


            # ====================================
            # BIRDSAI TESTREAL
            # ====================================

            result, sample_table = (
                evaluate_dataset(

                    model=model,

                    task=task,

                    dataset_name=
                        "BIRDSAI_test",

                    role=
                        "in_domain_test",

                    npz_path=
                        ROOT /
                        f"{task}_birdsai_test_canonical.npz",

                    meta_path=
                        ROOT /
                        f"{task}_birdsai_test_metadata.csv",

                    model_type=
                        model_type,

                    seed=
                        seed,

                    best_epoch=
                        best_epoch

                )
            )


            all_results.append(
                result
            )


            all_sample_results.append(
                sample_table
            )


            # ====================================
            # WILDBOX EXTERNAL
            # ====================================

            result, sample_table = (
                evaluate_dataset(

                    model=model,

                    task=task,

                    dataset_name=
                        "WildBox_external",

                    role=
                        "external_test",

                    npz_path=
                        ROOT /
                        f"{task}_wildbox_external_canonical.npz",

                    meta_path=
                        ROOT /
                        f"{task}_wildbox_external_metadata.csv",

                    model_type=
                        model_type,

                    seed=
                        seed,

                    best_epoch=
                        best_epoch

                )
            )


            all_results.append(
                result
            )


            all_sample_results.append(
                sample_table
            )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

training_history_df = pd.concat(

    all_training_history,

    ignore_index=True

)


training_history_df.to_csv(

    OUT /
    "training_history.csv",

    index=False
)


# ============================================================
# SAVE SEED-LEVEL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_df.to_csv(

    OUT /
    "seed_level_results.csv",

    index=False
)


# ============================================================
# SAVE SAMPLE RESULTS
# ============================================================

sample_results_df = pd.concat(

    all_sample_results,

    ignore_index=True

)


sample_results_df.to_csv(

    OUT /
    "sample_results.csv",

    index=False
)


# ============================================================
# MEAN ± STD ACROSS SEEDS
# ============================================================

metric_columns = [

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
), group in results_df.groupby(

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
            len(group),

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


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(

    OUT /
    "lstm_gru_summary.csv",

    index=False
)


# ============================================================
# PER-SOURCE RESULTS
# ============================================================

per_source = (

    sample_results_df.groupby(

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
    "lstm_gru_per_source.csv",

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
        "LSTM + GRU ELEPHANT TRAJECTORY FORECASTING\n"
    )

    f.write(
        "=" * 78
        +
        "\n\n"
    )


    f.write(
        "Training data: BIRDSAI TrainReal only\n"
    )

    f.write(
        "Validation: track-disjoint BIRDSAI "
        "TrainReal validation\n"
    )

    f.write(
        "Model selection metric: validation "
        "track-macro ADE\n"
    )

    f.write(
        "Evaluation: untouched BIRDSAI TestReal "
        "+ untouched WildBox external RGB\n"
    )

    f.write(
        "Seeds: "
        +
        str(
            SEEDS
        )
        +
        "\n"
    )

    f.write(
        "Final ADE/FDE computed after inverse "
        "canonical transform.\n\n"
    )


    for _, row in (
        summary_df.sort_values(

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

            f"ADE = "
            f"{row['ADE_mean_mean']:.6f} "
            f"± "
            f"{row['ADE_mean_std']:.6f}\n"

        )


        f.write(

            f"FDE = "
            f"{row['FDE_mean_mean']:.6f} "
            f"± "
            f"{row['FDE_mean_std']:.6f}\n"

        )


        f.write(

            f"Track-macro ADE = "
            f"{row['ADE_track_macro_mean']:.6f} "
            f"± "
            f"{row['ADE_track_macro_std']:.6f}\n"

        )


        f.write(

            f"Source-macro ADE = "
            f"{row['ADE_source_macro_mean']:.6f} "
            f"± "
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
    "LSTM + GRU EXPERIMENT COMPLETE"
)

print(
    "============================================"
)

print(
    "\nFinal mean ± std results:"
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    OUT
)