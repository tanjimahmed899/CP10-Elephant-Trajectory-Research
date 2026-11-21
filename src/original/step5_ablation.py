from pathlib import Path
import ast
import copy
import math
import random
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# STEP 5: CONTROLLED REPRESENTATION ABLATION
#
# Baseline:
#   Canonical + scale-normalized (existing frozen Transformer)
#
# New ablations:
#   1) NO_SCALE
#      Translation + heading alignment retained,
#      but division by path-length scale removed.
#
#   2) NO_CANONICALIZATION
#      Ego-motion-corrected absolute normalized x,y used directly.
#
# Same architecture, seeds, split, optimizer, LR, scheduler,
# early stopping, sample weights, and validation criterion.
#
# Existing Main Result files are READ ONLY.
# All NEW outputs go to Downloads.
# ============================================================

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

NEURAL_DIR = (
    ROOT /
    "final_neural_trajectory_dataset"
)

FRESH_DIR = (
    ROOT /
    "wildbox_fresh_confirmatory_results"
)

TRANSFORMER_DIR = (
    ROOT /
    "transformer_results"
)

ORIGINAL_TRAIN = (
    ROOT /
    "train_transformer.py"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step5_ablation"
)

CHECKPOINT_DIR = (
    OUT /
    "checkpoints"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ORIGINAL TRAINING CONSTANTS
# ============================================================

DEVICE = torch.device("cpu")

TASKS = {
    "2s_to_2s": {
        "future_length": 20,
        "expected_trainval": 531,
        "expected_train": 434,
        "expected_val": 97,
        "expected_birdsai_test": 1053,
        "expected_fresh": 865,
    },

    "2s_to_5s": {
        "future_length": 50,
        "expected_trainval": 382,
        "expected_train": 294,
        "expected_val": 88,
        "expected_birdsai_test": 826,
        "expected_fresh": 637,
    }
}

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
DROPOUT = 0.1

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001

MAX_EPOCHS = 300
PATIENCE = 30
MIN_DELTA = 1e-5
GRAD_CLIP = 5.0

VARIANTS = [
    "BASELINE_CANONICAL_SCALE",
    "NO_SCALE",
    "NO_CANONICALIZATION",
]


# ============================================================
# EXTRACT ORIGINAL MODEL / LOSS DEFINITIONS
# WITHOUT EXECUTING ORIGINAL TOP-LEVEL CODE
# ============================================================

def load_original_definitions():

    if not ORIGINAL_TRAIN.exists():

        raise FileNotFoundError(
            ORIGINAL_TRAIN
        )


    source = ORIGINAL_TRAIN.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(
            ORIGINAL_TRAIN
        )
    )


    required_names = {

        "set_seed",

        "TrajectoryDataset",

        "PositionalEncoding",

        "TrajectoryTransformer",

        "weighted_smooth_l1"
    }


    selected = []


    for node in tree.body:

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.ClassDef
            )
        ):

            if node.name in required_names:

                selected.append(
                    node
                )


    found = {
        node.name
        for node in selected
    }


    missing = (

        required_names

        -

        found
    )


    if missing:

        raise RuntimeError(

            "Missing original definitions: "

            +

            ", ".join(
                sorted(
                    missing
                )
            )
        )


    selected.sort(
        key=lambda n: n.lineno
    )


    module_ast = ast.Module(

        body=selected,

        type_ignores=[]
    )


    ast.fix_missing_locations(
        module_ast
    )


    ns = {

        "__name__":
            "original_transformer_training_definitions",

        "np":
            np,

        "pd":
            pd,

        "torch":
            torch,

        "nn":
            nn,

        "Dataset":
            Dataset,

        "DataLoader":
            DataLoader,

        "math":
            math,

        "random":
            random,

        "Path":
            Path,

        # Original architecture globals
        "D_MODEL":
            D_MODEL,

        "NHEAD":
            NHEAD,

        "NUM_LAYERS":
            NUM_LAYERS,

        "DIM_FEEDFORWARD":
            DIM_FEEDFORWARD,

        "DROPOUT":
            DROPOUT,
    }


    exec(

        compile(

            module_ast,

            filename=str(
                ORIGINAL_TRAIN
            ),

            mode="exec"
        ),

        ns,

        ns
    )


    return types.SimpleNamespace(

        set_seed=
            ns["set_seed"],

        TrajectoryDataset=
            ns["TrajectoryDataset"],

        PositionalEncoding=
            ns["PositionalEncoding"],

        TrajectoryTransformer=
            ns["TrajectoryTransformer"],

        weighted_smooth_l1=
            ns["weighted_smooth_l1"],
    )


ORIG = load_original_definitions()


# ============================================================
# REPRESENTATION CONSTRUCTION
# ============================================================

def build_representation(
    arrays,
    variant
):

    if variant == "BASELINE_CANONICAL_SCALE":

        X = np.asarray(
            arrays[
                "history_canonical"
            ],
            dtype=np.float32
        )

        Y = np.asarray(
            arrays[
                "future_canonical"
            ],
            dtype=np.float32
        )

        return X, Y


    if variant == "NO_SCALE":

        history_canonical = np.asarray(

            arrays[
                "history_canonical"
            ],

            dtype=np.float32
        )


        future_canonical = np.asarray(

            arrays[
                "future_canonical"
            ],

            dtype=np.float32
        )


        scale = np.asarray(

            arrays["scale"],

            dtype=np.float32
        )


        # Existing canonical representation:
        # rotated_relative_xy / scale
        #
        # Therefore multiplying by scale exactly removes
        # scale normalization while retaining translation
        # and heading alignment.

        X = (

            history_canonical

            *

            scale[
                :,
                None,
                None
            ]
        )


        Y = (

            future_canonical

            *

            scale[
                :,
                None,
                None
            ]
        )


        return (

            X.astype(
                np.float32
            ),

            Y.astype(
                np.float32
            )
        )


    if variant == "NO_CANONICALIZATION":

        X = np.asarray(

            arrays[
                "history_abs"
            ],

            dtype=np.float32
        )


        Y = np.asarray(

            arrays[
                "future_abs"
            ],

            dtype=np.float32
        )


        return X, Y


    raise ValueError(
        variant
    )


# ============================================================
# REPRESENTATION -> ABSOLUTE FRAME
# ============================================================

def inverse_representation(
    prediction,
    arrays,
    variant,
    indices=None
):

    prediction = np.asarray(
        prediction,
        dtype=np.float64
    )


    if indices is None:

        origin = np.asarray(
            arrays["origin"],
            dtype=np.float64
        )

        angle = np.asarray(
            arrays["heading_angle"],
            dtype=np.float64
        )

        scale = np.asarray(
            arrays["scale"],
            dtype=np.float64
        )

    else:

        origin = np.asarray(
            arrays["origin"],
            dtype=np.float64
        )[
            indices
        ]

        angle = np.asarray(
            arrays["heading_angle"],
            dtype=np.float64
        )[
            indices
        ]

        scale = np.asarray(
            arrays["scale"],
            dtype=np.float64
        )[
            indices
        ]


    if variant == "NO_CANONICALIZATION":

        return prediction


    if variant == "BASELINE_CANONICAL_SCALE":

        local = (

            prediction

            *

            scale[
                :,
                None,
                None
            ]
        )


    elif variant == "NO_SCALE":

        local = prediction


    else:

        raise ValueError(
            variant
        )


    output = np.empty_like(
        local,
        dtype=np.float64
    )


    for i in range(
        len(local)
    ):

        a = float(
            angle[i]
        )

        c = np.cos(a)
        s = np.sin(a)


        # Same row-vector inverse rotation used in the
        # verified Fresh WildBox evaluation.

        R = np.asarray(

            [
                [c, s],
                [-s, c]
            ],

            dtype=np.float64
        )


        output[i] = (

            local[i]
            @
            R

            +

            origin[i]
        )


    return output


# ============================================================
# PREDICTION
# ============================================================

def predict_model(
    model,
    history,
    batch_size=256
):

    model.eval()

    outputs = []


    with torch.no_grad():

        for start in range(

            0,

            len(history),

            batch_size
        ):

            X = torch.tensor(

                history[
                    start:
                    start
                    +
                    batch_size
                ],

                dtype=torch.float32,

                device=DEVICE
            )


            prediction = model(
                X
            )


            outputs.append(

                prediction
                .detach()
                .cpu()
                .numpy()
            )


    return np.concatenate(

        outputs,

        axis=0
    )


# ============================================================
# METRICS
# ============================================================

def sample_errors(
    prediction_abs,
    truth_abs
):

    errors = np.linalg.norm(

        prediction_abs

        -

        truth_abs,

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


def calculate_metrics(
    prediction_abs,
    truth_abs,
    metadata
):

    ADE, FDE = sample_errors(

        prediction_abs,

        truth_abs
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


    track = (

        temp
        .groupby(
            "track_key",
            as_index=False
        )
        .agg(
            ADE=("ADE", "mean"),
            FDE=("FDE", "mean")
        )
    )


    source = (

        temp
        .groupby(
            "source_group",
            as_index=False
        )
        .agg(
            ADE=("ADE", "mean"),
            FDE=("FDE", "mean")
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
                track[
                    "ADE"
                ].mean()
            ),

        "track_macro_FDE":
            float(
                track[
                    "FDE"
                ].mean()
            ),

        "source_macro_ADE":
            float(
                source[
                    "ADE"
                ].mean()
            ),

        "source_macro_FDE":
            float(
                source[
                    "FDE"
                ].mean()
            ),

        "sample_ADE":
            ADE,

        "sample_FDE":
            FDE,
    }


# ============================================================
# VALIDATION METRIC
# Same selection criterion as original:
# validation track-macro ADE
# ============================================================

def validation_metric(
    model,
    X_val,
    truth_abs,
    metadata,
    arrays,
    variant,
    validation_indices
):

    prediction_rep = predict_model(

        model,

        X_val
    )


    prediction_abs = inverse_representation(

        prediction_rep,

        arrays,

        variant,

        indices=
            validation_indices
    )


    metrics = calculate_metrics(

        prediction_abs,

        truth_abs,

        metadata
    )


    return (

        metrics[
            "track_macro_ADE"
        ],

        metrics
    )


# ============================================================
# TRAIN ONE ABLATION MODEL
# ============================================================

def train_one(
    task,
    variant,
    seed
):

    if variant == "BASELINE_CANONICAL_SCALE":

        raise ValueError(

            "Baseline is frozen and "
            "must not be retrained."
        )


    ORIG.set_seed(
        seed
    )


    print(
        "\n"
        +
        "=" * 78
    )


    print(
        task,
        "|",
        variant,
        "| seed",
        seed
    )


    print(
        "=" * 78
    )


    npz_path = (

        NEURAL_DIR

        /

        f"{task}_birdsai_trainval_canonical.npz"
    )


    metadata_path = (

        NEURAL_DIR

        /

        f"{task}_birdsai_trainval_metadata.csv"
    )


    arrays = np.load(
        npz_path
    )


    metadata = pd.read_csv(

        metadata_path,

        dtype={
            "source_group":
                str,

            "track_key":
                str
        }
    )


    train_indices = np.asarray(

        arrays[
            "train_indices"
        ],

        dtype=int
    )


    validation_indices = np.asarray(

        arrays[
            "validation_indices"
        ],

        dtype=int
    )


    weights = np.asarray(

        arrays[
            "train_sample_weights"
        ],

        dtype=np.float32
    )


    X_all, Y_all = build_representation(

        arrays,

        variant
    )


    if len(X_all) != TASKS[
        task
    ][
        "expected_trainval"
    ]:

        raise RuntimeError(
            "Unexpected trainval size."
        )


    if len(
        train_indices
    ) != TASKS[
        task
    ][
        "expected_train"
    ]:

        raise RuntimeError(
            "Unexpected train split."
        )


    if len(
        validation_indices
    ) != TASKS[
        task
    ][
        "expected_val"
    ]:

        raise RuntimeError(
            "Unexpected validation split."
        )


    X_train = X_all[
        train_indices
    ]


    Y_train = Y_all[
        train_indices
    ]


    X_val = X_all[
        validation_indices
    ]


    val_truth_abs = np.asarray(

        arrays[
            "future_abs"
        ],

        dtype=np.float64
    )[
        validation_indices
    ]


    val_metadata = (

        metadata
        .iloc[
            validation_indices
        ]
        .reset_index(
            drop=True
        )
    )


    train_dataset = ORIG.TrajectoryDataset(

        X_train,

        Y_train,

        weights
    )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    loader = DataLoader(

        train_dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=
            True,

        generator=
            generator,

        num_workers=
            0
    )


    future_length = (
        Y_train.shape[
            1
        ]
    )


    model = ORIG.TrajectoryTransformer(

        future_length
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=
            LEARNING_RATE,

        weight_decay=
            WEIGHT_DECAY
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


        for X, Y, W in loader:

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


            loss = ORIG.weighted_smooth_l1(

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


        (
            val_value,
            val_metrics
        ) = validation_metric(

            model,

            X_val,

            val_truth_abs,

            val_metadata,

            arrays,

            variant,

            validation_indices
        )


        scheduler.step(
            val_value
        )


        history_rows.append({

            "task":
                task,

            "variant":
                variant,

            "seed":
                seed,

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_ADE":
                val_metrics[
                    "ADE"
                ],

            "validation_FDE":
                val_metrics[
                    "FDE"
                ],

            "validation_track_macro_ADE":
                val_metrics[
                    "track_macro_ADE"
                ],

            "validation_track_macro_FDE":
                val_metrics[
                    "track_macro_FDE"
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

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
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


    if best_state is None:

        raise RuntimeError(
            "No best state captured."
        )


    model.load_state_dict(
        best_state
    )


    checkpoint_path = (

        CHECKPOINT_DIR

        /

        (
            f"{task}_"
            f"{variant}_"
            f"seed{seed}.pt"
        )
    )


    torch.save(

        {

            "state_dict":
                best_state,

            "task":
                task,

            "variant":
                variant,

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
                DROPOUT,

            "training_protocol":
                "same_as_original_transformer"
        },

        checkpoint_path
    )


    return (

        model,

        pd.DataFrame(
            history_rows
        ),

        best_epoch,

        best_metric
    )


# ============================================================
# LOAD FROZEN BASELINE
# ============================================================

def load_baseline_model(
    task,
    seed
):

    checkpoint_path = (

        TRANSFORMER_DIR

        /

        f"{task}_Transformer_seed{seed}.pt"
    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=
            DEVICE
    )


    model = ORIG.TrajectoryTransformer(

        TASKS[
            task
        ][
            "future_length"
        ]
    ).to(
        DEVICE
    )


    model.load_state_dict(

        checkpoint[
            "state_dict"
        ],

        strict=True
    )


    model.eval()


    return (

        model,

        int(
            checkpoint[
                "best_epoch"
            ]
        ),

        float(
            checkpoint[
                "best_validation_track_macro_ADE"
            ]
        )
    )


# ============================================================
# LOAD NEW ABLATION CHECKPOINT
# ============================================================

def load_ablation_model(
    task,
    variant,
    seed
):

    checkpoint_path = (

        CHECKPOINT_DIR

        /

        (
            f"{task}_"
            f"{variant}_"
            f"seed{seed}.pt"
        )
    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=
            DEVICE
    )


    model = ORIG.TrajectoryTransformer(

        TASKS[
            task
        ][
            "future_length"
        ]
    ).to(
        DEVICE
    )


    model.load_state_dict(

        checkpoint[
            "state_dict"
        ],

        strict=True
    )


    model.eval()


    return (

        model,

        int(
            checkpoint[
                "best_epoch"
            ]
        ),

        float(
            checkpoint[
                "best_validation_track_macro_ADE"
            ]
        )
    )


# ============================================================
# EVALUATE ONE MODEL ON ONE DATASET
# ============================================================

def evaluate_model(
    model,
    arrays,
    metadata,
    variant
):

    X, _ = build_representation(

        arrays,

        variant
    )


    prediction_rep = predict_model(

        model,

        X
    )


    prediction_abs = inverse_representation(

        prediction_rep,

        arrays,

        variant
    )


    truth_abs = np.asarray(

        arrays[
            "future_abs"
        ],

        dtype=np.float64
    )


    return calculate_metrics(

        prediction_abs,

        truth_abs,

        metadata
    ), prediction_abs


# ============================================================
# DATASET LOADING
# ============================================================

def load_eval_dataset(
    task,
    dataset_name
):

    if dataset_name == "BIRDSAI_TEST":

        npz_path = (

            NEURAL_DIR

            /

            f"{task}_birdsai_test_canonical.npz"
        )


        metadata_path = (

            NEURAL_DIR

            /

            f"{task}_birdsai_test_metadata.csv"
        )


        expected = TASKS[
            task
        ][
            "expected_birdsai_test"
        ]


    elif dataset_name == "FRESH_WILDBOX":

        npz_path = (

            FRESH_DIR

            /

            f"{task}_fresh_dataset.npz"
        )


        metadata_path = (

            FRESH_DIR

            /

            f"{task}_fresh_metadata.csv"
        )


        expected = TASKS[
            task
        ][
            "expected_fresh"
        ]


    else:

        raise ValueError(
            dataset_name
        )


    arrays = np.load(
        npz_path
    )


    metadata = pd.read_csv(

        metadata_path,

        dtype={
            "source_group":
                str,

            "track_key":
                str
        }
    )


    if len(
        metadata
    ) != expected:

        raise RuntimeError(

            f"{task} {dataset_name}: "
            f"{len(metadata)} != {expected}"
        )


    return arrays, metadata


# ============================================================
# FRESH BASELINE SELF-CHECK
# ============================================================

def fresh_baseline_self_check(
    task,
    seed_metrics,
    ensemble_metrics
):

    saved = pd.read_csv(

        FRESH_DIR

        /

        "fresh_holdout_model_metrics.csv"
    )


    saved = saved[
        saved[
            "task"
        ]
        ==
        task
    ]


    rows = []


    for seed in SEEDS:

        actual = seed_metrics[
            seed
        ]


        expected_row = saved[

            saved[
                "model"
            ]

            ==

            f"Transformer_seed{seed}"
        ]


        if len(
            expected_row
        ) != 1:

            raise RuntimeError(
                "Missing saved baseline metric."
            )


        expected_row = expected_row.iloc[
            0
        ]


        ade_diff = abs(

            actual[
                "ADE"
            ]

            -

            float(
                expected_row[
                    "ADE"
                ]
            )
        )


        fde_diff = abs(

            actual[
                "FDE"
            ]

            -

            float(
                expected_row[
                    "FDE"
                ]
            )
        )


        rows.append({

            "task":
                task,

            "model":
                f"Transformer_seed{seed}",

            "ADE_diff":
                ade_diff,

            "FDE_diff":
                fde_diff
        })


        if (
            ade_diff > 5e-6

            or

            fde_diff > 5e-6
        ):

            raise RuntimeError(

                "Frozen baseline Fresh WildBox "
                "self-check failed."
            )


    ensemble_saved = saved[

        saved[
            "model"
        ]

        ==

        "Transformer_3Seed_Ensemble"
    ]


    if len(
        ensemble_saved
    ) != 1:

        raise RuntimeError(
            "Missing saved ensemble metric."
        )


    ensemble_saved = ensemble_saved.iloc[
        0
    ]


    ade_diff = abs(

        ensemble_metrics[
            "ADE"
        ]

        -

        float(
            ensemble_saved[
                "ADE"
            ]
        )
    )


    fde_diff = abs(

        ensemble_metrics[
            "FDE"
        ]

        -

        float(
            ensemble_saved[
                "FDE"
            ]
        )
    )


    rows.append({

        "task":
            task,

        "model":
            "Transformer_3Seed_Ensemble",

        "ADE_diff":
            ade_diff,

        "FDE_diff":
            fde_diff
    })


    if (
        ade_diff > 5e-6

        or

        fde_diff > 5e-6
    ):

        raise RuntimeError(

            "Frozen baseline ensemble Fresh "
            "WildBox self-check failed."
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# IMPROVEMENT / DEGRADATION
# ============================================================

def pct_change(
    baseline,
    variant
):

    if abs(
        baseline
    ) <= 1e-12:

        return np.nan


    return (

        100.0

        *

        (
            variant
            -
            baseline
        )

        /

        baseline
    )


# ============================================================
# MAIN
# ============================================================

def main():

    training_history = []

    training_summary = []

    seed_metric_rows = []

    sample_metric_rows = []

    self_check_rows = []


    # ========================================================
    # TRAIN NEW VARIANTS
    # ========================================================

    for task in TASKS:

        for variant in [

            "NO_SCALE",

            "NO_CANONICALIZATION"
        ]:

            for seed in SEEDS:

                (
                    model,
                    history,
                    best_epoch,
                    best_metric
                ) = train_one(

                    task,

                    variant,

                    seed
                )


                training_history.append(
                    history
                )


                training_summary.append({

                    "task":
                        task,

                    "variant":
                        variant,

                    "seed":
                        seed,

                    "best_epoch":
                        best_epoch,

                    "best_validation_track_macro_ADE":
                        best_metric
                })


    pd.concat(

        training_history,

        ignore_index=True
    ).to_csv(

        OUT /
        "training_history.csv",

        index=False
    )


    pd.DataFrame(

        training_summary

    ).to_csv(

        OUT /
        "training_summary.csv",

        index=False
    )


    # ========================================================
    # EVALUATE ALL VARIANTS
    # ========================================================

    for task in TASKS:

        for dataset_name in [

            "BIRDSAI_TEST",

            "FRESH_WILDBOX"
        ]:

            arrays, metadata = load_eval_dataset(

                task,

                dataset_name
            )


            for variant in VARIANTS:

                predictions_by_seed = []

                metrics_by_seed = {}


                for seed in SEEDS:

                    if (
                        variant

                        ==

                        "BASELINE_CANONICAL_SCALE"
                    ):

                        (
                            model,
                            best_epoch,
                            best_val
                        ) = load_baseline_model(

                            task,

                            seed
                        )


                    else:

                        (
                            model,
                            best_epoch,
                            best_val
                        ) = load_ablation_model(

                            task,

                            variant,

                            seed
                        )


                    (
                        metrics,
                        prediction_abs
                    ) = evaluate_model(

                        model,

                        arrays,

                        metadata,

                        variant
                    )


                    predictions_by_seed.append(
                        prediction_abs
                    )


                    metrics_by_seed[
                        seed
                    ] = metrics


                    seed_metric_rows.append({

                        "task":
                            task,

                        "dataset":
                            dataset_name,

                        "variant":
                            variant,

                        "model":
                            "Transformer",

                        "seed":
                            seed,

                        "best_epoch":
                            best_epoch,

                        "best_validation_track_macro_ADE":
                            best_val,

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
                            ]
                    })


                    for i in range(
                        len(metadata)
                    ):

                        sample_metric_rows.append({

                            "task":
                                task,

                            "dataset":
                                dataset_name,

                            "variant":
                                variant,

                            "aggregation":
                                f"seed{seed}",

                            "sample_id":
                                int(
                                    metadata.iloc[
                                        i
                                    ][
                                        "sample_id"
                                    ]
                                ),

                            "source_group":
                                metadata.iloc[
                                    i
                                ][
                                    "source_group"
                                ],

                            "track_key":
                                metadata.iloc[
                                    i
                                ][
                                    "track_key"
                                ],

                            "ADE":
                                float(
                                    metrics[
                                        "sample_ADE"
                                    ][
                                        i
                                    ]
                                ),

                            "FDE":
                                float(
                                    metrics[
                                        "sample_FDE"
                                    ][
                                        i
                                    ]
                                )
                        })


                # --------------------------------------------
                # 3-SEED ENSEMBLE
                # --------------------------------------------

                ensemble_prediction = np.mean(

                    np.stack(

                        predictions_by_seed,

                        axis=0
                    ),

                    axis=0
                )


                ensemble_metrics = calculate_metrics(

                    ensemble_prediction,

                    np.asarray(
                        arrays[
                            "future_abs"
                        ],
                        dtype=np.float64
                    ),

                    metadata
                )


                seed_metric_rows.append({

                    "task":
                        task,

                    "dataset":
                        dataset_name,

                    "variant":
                        variant,

                    "model":
                        "Transformer_3Seed_Ensemble",

                    "seed":
                        "ensemble",

                    "best_epoch":
                        np.nan,

                    "best_validation_track_macro_ADE":
                        np.nan,

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
                        ]
                })


                for i in range(
                    len(metadata)
                ):

                    sample_metric_rows.append({

                        "task":
                            task,

                        "dataset":
                            dataset_name,

                        "variant":
                            variant,

                        "aggregation":
                            "ensemble",

                        "sample_id":
                            int(
                                metadata.iloc[
                                    i
                                ][
                                    "sample_id"
                                ]
                            ),

                        "source_group":
                            metadata.iloc[
                                i
                            ][
                                "source_group"
                            ],

                        "track_key":
                            metadata.iloc[
                                i
                            ][
                                "track_key"
                            ],

                        "ADE":
                            float(
                                ensemble_metrics[
                                    "sample_ADE"
                                ][
                                    i
                                ]
                            ),

                        "FDE":
                            float(
                                ensemble_metrics[
                                    "sample_FDE"
                                ][
                                    i
                                ]
                            )
                    })


                # --------------------------------------------
                # Verify existing frozen baseline on Fresh WB
                # --------------------------------------------

                if (

                    dataset_name
                    ==
                    "FRESH_WILDBOX"

                    and

                    variant
                    ==
                    "BASELINE_CANONICAL_SCALE"
                ):

                    check = fresh_baseline_self_check(

                        task,

                        metrics_by_seed,

                        ensemble_metrics
                    )


                    self_check_rows.append(
                        check
                    )


    seed_metrics_df = pd.DataFrame(
        seed_metric_rows
    )


    seed_metrics_df.to_csv(

        OUT /
        "variant_seed_and_ensemble_metrics.csv",

        index=False
    )


    pd.DataFrame(

        sample_metric_rows

    ).to_csv(

        OUT /
        "variant_sample_metrics.csv",

        index=False
    )


    checks = pd.concat(

        self_check_rows,

        ignore_index=True
    )


    checks.to_csv(

        OUT /
        "baseline_fresh_self_check.csv",

        index=False
    )


    # ========================================================
    # ABLATION SUMMARY
    # Compare ENSEMBLE variants
    # ========================================================

    ensemble_rows = seed_metrics_df[

        seed_metrics_df[
            "seed"
        ].astype(
            str
        )
        ==
        "ensemble"

    ].copy()


    summary_rows = []


    for (
        task,
        dataset_name
    ), group in ensemble_rows.groupby(

        [
            "task",
            "dataset"
        ],

        sort=False
    ):

        baseline = group[

            group[
                "variant"
            ]

            ==

            "BASELINE_CANONICAL_SCALE"

        ]


        if len(
            baseline
        ) != 1:

            raise RuntimeError(
                "Missing baseline ensemble row."
            )


        baseline = baseline.iloc[
            0
        ]


        for variant in VARIANTS:

            row = group[

                group[
                    "variant"
                ]

                ==

                variant
            ]


            if len(
                row
            ) != 1:

                raise RuntimeError(

                    f"Missing row: "
                    f"{task} "
                    f"{dataset_name} "
                    f"{variant}"
                )


            row = row.iloc[
                0
            ]


            summary_rows.append({

                "task":
                    task,

                "dataset":
                    dataset_name,

                "variant":
                    variant,

                "ADE":
                    row[
                        "ADE"
                    ],

                "FDE":
                    row[
                        "FDE"
                    ],

                "track_macro_ADE":
                    row[
                        "track_macro_ADE"
                    ],

                "track_macro_FDE":
                    row[
                        "track_macro_FDE"
                    ],

                "source_macro_ADE":
                    row[
                        "source_macro_ADE"
                    ],

                "source_macro_FDE":
                    row[
                        "source_macro_FDE"
                    ],

                # Positive degradation means worse
                # than canonical+scale baseline.

                "ADE_degradation_vs_baseline_pct":
                    pct_change(

                        baseline[
                            "ADE"
                        ],

                        row[
                            "ADE"
                        ]
                    ),

                "FDE_degradation_vs_baseline_pct":
                    pct_change(

                        baseline[
                            "FDE"
                        ],

                        row[
                            "FDE"
                        ]
                    ),

                "track_macro_ADE_degradation_vs_baseline_pct":
                    pct_change(

                        baseline[
                            "track_macro_ADE"
                        ],

                        row[
                            "track_macro_ADE"
                        ]
                    )
            })


    ablation_summary = pd.DataFrame(
        summary_rows
    )


    ablation_summary.to_csv(

        OUT /
        "FINAL_ABLATION_SUMMARY.csv",

        index=False
    )


    # ========================================================
    # ENSEMBLE BENEFIT
    # Compare ensemble against arithmetic mean of seed metrics
    # ========================================================

    benefit_rows = []


    for (
        task,
        dataset_name,
        variant
    ), group in seed_metrics_df.groupby(

        [
            "task",
            "dataset",
            "variant"
        ],

        sort=False
    ):

        singles = group[

            group[
                "seed"
            ].astype(
                str
            )
            !=
            "ensemble"

        ]


        ensemble = group[

            group[
                "seed"
            ].astype(
                str
            )
            ==
            "ensemble"

        ]


        if (
            len(
                singles
            ) != 3

            or

            len(
                ensemble
            ) != 1
        ):

            raise RuntimeError(
                "Unexpected seed/ensemble count."
            )


        ensemble = ensemble.iloc[
            0
        ]


        mean_single_ADE = float(

            singles[
                "ADE"
            ].mean()
        )


        mean_single_FDE = float(

            singles[
                "FDE"
            ].mean()
        )


        ensemble_ADE = float(

            ensemble[
                "ADE"
            ]
        )


        ensemble_FDE = float(

            ensemble[
                "FDE"
            ]
        )


        benefit_rows.append({

            "task":
                task,

            "dataset":
                dataset_name,

            "variant":
                variant,

            "mean_single_seed_ADE":
                mean_single_ADE,

            "ensemble_ADE":
                ensemble_ADE,

            "ADE_ensemble_improvement_pct":
                (
                    100.0
                    *
                    (
                        mean_single_ADE
                        -
                        ensemble_ADE
                    )
                    /
                    mean_single_ADE
                ),

            "mean_single_seed_FDE":
                mean_single_FDE,

            "ensemble_FDE":
                ensemble_FDE,

            "FDE_ensemble_improvement_pct":
                (
                    100.0
                    *
                    (
                        mean_single_FDE
                        -
                        ensemble_FDE
                    )
                    /
                    mean_single_FDE
                )
        })


    benefit_df = pd.DataFrame(
        benefit_rows
    )


    benefit_df.to_csv(

        OUT /
        "ensemble_benefit.csv",

        index=False
    )


    # ========================================================
    # HUMAN-READABLE SUMMARY
    # ========================================================

    with open(

        OUT /
        "SUMMARY.txt",

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "STEP 5: CONTROLLED REPRESENTATION ABLATION\n"
        )

        f.write(
            "=" * 78
            +
            "\n\n"
        )


        f.write(
            "Protocol:\n"
        )

        f.write(
            "- Baseline uses the existing frozen "
            "canonical + scale-normalized Transformer.\n"
        )

        f.write(
            "- NO_SCALE removes only path-length "
            "scale normalization while retaining "
            "translation and heading alignment.\n"
        )

        f.write(
            "- NO_CANONICALIZATION uses corrected "
            "absolute normalized x,y directly.\n"
        )

        f.write(
            "- Same track-disjoint train/validation split.\n"
        )

        f.write(
            "- Same seeds: 42, 52, 62.\n"
        )

        f.write(
            "- Same architecture and training protocol.\n"
        )

        f.write(
            "- Model selection uses validation "
            "track-macro ADE only.\n"
        )

        f.write(
            "- Fresh WildBox is used only after "
            "training/model selection as frozen "
            "external evaluation.\n\n"
        )


        f.write(
            "BASELINE FRESH SELF-CHECK\n"
        )

        f.write(
            "-------------------------\n"
        )

        f.write(
            checks.to_string(
                index=False
            )
        )


        f.write(
            "\n\nFINAL ENSEMBLE ABLATION SUMMARY\n"
        )

        f.write(
            "-------------------------------\n"
        )

        f.write(
            ablation_summary.to_string(
                index=False
            )
        )


        f.write(
            "\n\nENSEMBLE BENEFIT\n"
        )

        f.write(
            "----------------\n"
        )

        f.write(
            benefit_df.to_string(
                index=False
            )
        )


        f.write(
            "\n\nInterpretation:\n"
        )

        f.write(
            "Positive degradation percentages mean "
            "the ablated representation performs "
            "worse than the canonical+scale baseline.\n"
        )


    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print(
        "\nSTEP 5 COMPLETE"
    )


    print(
        "Outputs:",
        OUT
    )


    print(
        "\nBASELINE FRESH SELF-CHECK:\n"
    )


    print(
        checks.to_string(
            index=False
        )
    )


    print(
        "\nFINAL ENSEMBLE ABLATION SUMMARY:\n"
    )


    print(

        ablation_summary[
            [
                "task",
                "dataset",
                "variant",
                "ADE",
                "FDE",
                "track_macro_ADE",
                "ADE_degradation_vs_baseline_pct",
                "FDE_degradation_vs_baseline_pct",
                "track_macro_ADE_degradation_vs_baseline_pct"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nENSEMBLE BENEFIT:\n"
    )


    print(
        benefit_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()
