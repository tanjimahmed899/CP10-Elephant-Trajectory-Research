from pathlib import Path
import ast
import copy
import math
import random
import types
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# STEP 6C: TRACK-LEVEL CROSS-FITTED CONFORMAL CALIBRATION
#
# Purpose
# -------
# Build leakage-resistant calibration residuals from ALL TrainReal
# tracks using outer track-level cross-fitting.
#
# Outer held-out tracks are NEVER used for training or early stopping.
# Each outer training complement gets its own INNER track-disjoint
# validation split for early stopping.
#
# Original training protocol retained:
# - Transformer architecture
# - seeds 42/52/62
# - AdamW
# - same LR / WD / scheduler
# - same early stopping
# - track-balanced sample weights
#
# Exact original weight rule:
#   each training track contributes equal total weight
#   => each window weight = 1 / windows_in_that_training_track
#
# Primary conformal unit:
#   TRACK
#
# Primary score for each track:
#   maximum canonical Euclidean forecast error across
#   ALL windows and ALL future steps in that track.
#
# This avoids treating overlapping windows as independent
# calibration observations.
#
# IMPORTANT
# ---------
# TestReal and Fresh WildBox have already been inspected in Step 6A/6B.
# Therefore Step 6C evaluations on them are retrospective/diagnostic,
# NOT a new untouched confirmatory uncertainty test.
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
    r"C:\Users\HP\Downloads\step6c_crossfit_conformal"
)

CHECKPOINT_DIR = (
    OUT /
    "crossfit_checkpoints"
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
# FIXED ORIGINAL TRAINING PROTOCOL
# ============================================================

DEVICE = torch.device("cpu")

TASKS = {

    "2s_to_2s": {
        "future_length": 20,
        "expected_trainval_windows": 531,
        "expected_trainval_tracks": 62,
        "expected_test_windows": 1053,
        "expected_test_tracks": 89,
        "expected_fresh_windows": 865,
        "expected_fresh_tracks": 78
    },

    "2s_to_5s": {
        "future_length": 50,
        "expected_trainval_windows": 382,
        "expected_trainval_tracks": 34,
        "expected_test_windows": 826,
        "expected_test_tracks": 63,
        "expected_fresh_windows": 637,
        "expected_fresh_tracks": 71
    }
}

SEEDS = [
    42,
    52,
    62
]

N_OUTER_FOLDS = 3

INNER_VAL_FRACTION = 0.20

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

ALPHAS = [
    0.10,
    0.05
]

N_BOOT = 10000

RNG_SEED = 20261026

SELF_CHECK_TOL = 5e-6


# ============================================================
# EXTRACT ORIGINAL TRAINING DEFINITIONS WITHOUT RUNNING
# ORIGINAL TOP-LEVEL CODE
# ============================================================

def load_original_definitions():

    source = ORIGINAL_TRAIN.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(
            ORIGINAL_TRAIN
        )
    )

    required = {

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

            if node.name in required:

                selected.append(
                    node
                )

    found = {
        node.name
        for node in selected
    }

    missing = (
        required
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
            "original_transformer_crossfit",

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
            ns[
                "set_seed"
            ],

        TrajectoryDataset=
            ns[
                "TrajectoryDataset"
            ],

        TrajectoryTransformer=
            ns[
                "TrajectoryTransformer"
            ],

        weighted_smooth_l1=
            ns[
                "weighted_smooth_l1"
            ]
    )


ORIG = load_original_definitions()


# ============================================================
# BASIC HELPERS
# ============================================================

def stable_int(
    text
):

    digest = hashlib.sha256(
        str(
            text
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[
            :16
        ],
        16
    )


def make_track_weights(
    metadata,
    indices
):

    subset = (
        metadata
        .iloc[
            indices
        ]
        .copy()
    )

    keys = (
        subset[
            "track_key"
        ]
        .astype(
            str
        )
    )

    counts = keys.value_counts()

    weights = np.asarray(

        [
            1.0
            /
            float(
                counts[
                    str(
                        metadata.iloc[
                            idx
                        ][
                            "track_key"
                        ]
                    )
                ]
            )

            for idx in indices
        ],

        dtype=np.float32
    )

    # Exact rule check: every included track must sum to 1.
    check = pd.DataFrame({

        "track_key":
            metadata.iloc[
                indices
            ][
                "track_key"
            ].astype(
                str
            ).to_numpy(),

        "weight":
            weights
    })

    sums = (
        check
        .groupby(
            "track_key"
        )[
            "weight"
        ]
        .sum()
        .to_numpy()
    )

    if not np.allclose(
        sums,
        1.0,
        atol=1e-6
    ):

        raise RuntimeError(
            "Track-weight equality check failed."
        )

    return weights


def inverse_canonical(
    canonical_prediction,
    origin,
    angle,
    scale
):

    prediction = np.asarray(
        canonical_prediction,
        dtype=np.float64
    )

    origin = np.asarray(
        origin,
        dtype=np.float64
    )

    angle = np.asarray(
        angle,
        dtype=np.float64
    )

    scale = np.asarray(
        scale,
        dtype=np.float64
    )

    local = (

        prediction

        *

        scale[
            :,
            None,
            None
        ]
    )

    output = np.empty_like(
        local,
        dtype=np.float64
    )

    for i in range(
        len(
            local
        )
    ):

        a = float(
            angle[
                i
            ]
        )

        c = np.cos(
            a
        )

        s = np.sin(
            a
        )

        R = np.asarray(

            [
                [c, s],
                [-s, c]
            ],

            dtype=np.float64
        )

        output[
            i
        ] = (

            local[
                i
            ]

            @

            R

            +

            origin[
                i
            ]
        )

    return output


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

    return (
        errors.mean(
            axis=1
        ),

        errors[
            :,
            -1
        ],

        errors
    )


def track_macro_ADE(
    prediction_abs,
    truth_abs,
    metadata
):

    ADE, _, _ = sample_errors(

        prediction_abs,

        truth_abs
    )

    temp = pd.DataFrame({

        "track_key":
            metadata[
                "track_key"
            ].astype(
                str
            ).to_numpy(),

        "ADE":
            ADE
    })

    return float(

        temp
        .groupby(
            "track_key"
        )[
            "ADE"
        ]
        .mean()
        .mean()
    )


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

            len(
                history
            ),

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

            pred = model(
                X
            )

            outputs.append(

                pred
                .detach()
                .cpu()
                .numpy()
            )

    return np.concatenate(
        outputs,
        axis=0
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_trainval(
    task
):

    arrays = np.load(

        NEURAL_DIR

        /

        f"{task}_birdsai_trainval_canonical.npz"
    )

    metadata = pd.read_csv(

        NEURAL_DIR

        /

        f"{task}_birdsai_trainval_metadata.csv",

        dtype={
            "source_group":
                str,

            "track_key":
                str
        }
    )

    if len(
        metadata
    ) != TASKS[
        task
    ][
        "expected_trainval_windows"
    ]:

        raise RuntimeError(
            f"{task}: unexpected TrainReal window count."
        )

    if metadata[
        "track_key"
    ].nunique() != TASKS[
        task
    ][
        "expected_trainval_tracks"
    ]:

        raise RuntimeError(
            f"{task}: unexpected TrainReal track count."
        )

    return arrays, metadata


def load_testreal(
    task
):

    arrays = np.load(

        NEURAL_DIR

        /

        f"{task}_birdsai_test_canonical.npz"
    )

    metadata = pd.read_csv(

        NEURAL_DIR

        /

        f"{task}_birdsai_test_metadata.csv",

        dtype={
            "source_group":
                str,

            "track_key":
                str
        }
    )

    if len(
        metadata
    ) != TASKS[
        task
    ][
        "expected_test_windows"
    ]:

        raise RuntimeError(
            f"{task}: unexpected TestReal windows."
        )

    if metadata[
        "track_key"
    ].nunique() != TASKS[
        task
    ][
        "expected_test_tracks"
    ]:

        raise RuntimeError(
            f"{task}: unexpected TestReal tracks."
        )

    return arrays, metadata


def load_fresh(
    task
):

    arrays = np.load(

        FRESH_DIR

        /

        f"{task}_fresh_dataset.npz"
    )

    metadata = pd.read_csv(

        FRESH_DIR

        /

        f"{task}_fresh_metadata.csv",

        dtype={
            "source_group":
                str,

            "track_key":
                str
        }
    )

    if len(
        metadata
    ) != TASKS[
        task
    ][
        "expected_fresh_windows"
    ]:

        raise RuntimeError(
            f"{task}: unexpected Fresh windows."
        )

    if metadata[
        "track_key"
    ].nunique() != TASKS[
        task
    ][
        "expected_fresh_tracks"
    ]:

        raise RuntimeError(
            f"{task}: unexpected Fresh tracks."
        )

    return arrays, metadata


# ============================================================
# DETERMINISTIC SOURCE-AWARE OUTER FOLDS
# ============================================================

def make_track_table(
    metadata
):

    table = (

        metadata
        .groupby(
            "track_key",
            as_index=False
        )
        .agg(
            source_group=(
                "source_group",
                "first"
            ),

            windows=(
                "track_key",
                "size"
            )
        )
    )

    # Safety: one track must belong to one source.
    source_counts = (

        metadata
        .groupby(
            "track_key"
        )[
            "source_group"
        ]
        .nunique()
    )

    if int(
        source_counts.max()
    ) != 1:

        raise RuntimeError(
            "A track appears in multiple source groups."
        )

    return table


def assign_outer_folds(
    metadata,
    task
):

    track_table = make_track_table(
        metadata
    )

    track_table[
        "outer_fold"
    ] = -1

    global_fold_windows = np.zeros(
        N_OUTER_FOLDS,
        dtype=int
    )

    global_fold_tracks = np.zeros(
        N_OUTER_FOLDS,
        dtype=int
    )

    task_seed = (
        RNG_SEED
        +
        stable_int(
            task
        )
        %
        100000
    )

    rng = np.random.default_rng(
        task_seed
    )

    # Process larger sources first.
    source_order = (

        track_table
        .groupby(
            "source_group"
        )[
            "windows"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .index
        .tolist()
    )

    for source_group in source_order:

        source_idx = track_table.index[

            track_table[
                "source_group"
            ]

            ==

            source_group

        ].to_numpy()

        # Deterministic randomized tie-breaking.
        tie = rng.random(
            len(
                source_idx
            )
        )

        source_items = pd.DataFrame({

            "idx":
                source_idx,

            "windows":
                track_table.loc[
                    source_idx,
                    "windows"
                ].to_numpy(),

            "tie":
                tie
        })

        source_items = source_items.sort_values(

            [
                "windows",
                "tie"
            ],

            ascending=[
                False,
                True
            ]
        )

        source_fold_windows = np.zeros(
            N_OUTER_FOLDS,
            dtype=int
        )

        source_fold_tracks = np.zeros(
            N_OUTER_FOLDS,
            dtype=int
        )

        for _, item in source_items.iterrows():

            candidates = list(
                range(
                    N_OUTER_FOLDS
                )
            )

            candidates.sort(

                key=lambda f: (

                    source_fold_windows[
                        f
                    ],

                    source_fold_tracks[
                        f
                    ],

                    global_fold_windows[
                        f
                    ],

                    global_fold_tracks[
                        f
                    ],

                    f
                )
            )

            fold = candidates[
                0
            ]

            idx = int(
                item[
                    "idx"
                ]
            )

            windows = int(
                item[
                    "windows"
                ]
            )

            track_table.loc[
                idx,
                "outer_fold"
            ] = fold

            source_fold_windows[
                fold
            ] += windows

            source_fold_tracks[
                fold
            ] += 1

            global_fold_windows[
                fold
            ] += windows

            global_fold_tracks[
                fold
            ] += 1

    if (
        track_table[
            "outer_fold"
        ]
        <
        0
    ).any():

        raise RuntimeError(
            "Outer fold assignment incomplete."
        )

    if set(
        track_table[
            "outer_fold"
        ].astype(
            int
        )
    ) != set(
        range(
            N_OUTER_FOLDS
        )
    ):

        raise RuntimeError(
            "Not all outer folds populated."
        )

    return track_table


# ============================================================
# INNER TRACK-DISJOINT VALIDATION SPLIT
# ============================================================

def make_inner_split(
    outer_train_track_table,
    task,
    outer_fold
):

    df = outer_train_track_table.copy()

    inner_val_tracks = []

    seed = (
        RNG_SEED
        +
        1000
        *
        (
            outer_fold
            +
            1
        )
        +
        stable_int(
            task
        )
        %
        100000
    )

    rng = np.random.default_rng(
        seed
    )

    for source_group, group in df.groupby(
        "source_group"
    ):

        tracks = group[
            "track_key"
        ].astype(
            str
        ).tolist()

        if len(
            tracks
        ) <= 1:

            continue

        tracks = sorted(
            tracks
        )

        perm = rng.permutation(
            len(
                tracks
            )
        )

        tracks = [
            tracks[
                int(
                    i
                )
            ]
            for i in perm
        ]

        n_val = int(
            round(
                INNER_VAL_FRACTION
                *
                len(
                    tracks
                )
            )
        )

        n_val = max(
            1,
            n_val
        )

        n_val = min(
            n_val,
            len(
                tracks
            )
            -
            1
        )

        inner_val_tracks.extend(
            tracks[
                :n_val
            ]
        )

    # If source-wise logic produced too few validation tracks,
    # add deterministic tracks from the remaining pool.
    minimum_val = max(
        2,
        int(
            round(
                INNER_VAL_FRACTION
                *
                len(
                    df
                )
            )
        )
    )

    inner_val_tracks = list(
        dict.fromkeys(
            inner_val_tracks
        )
    )

    if len(
        inner_val_tracks
    ) < minimum_val:

        remaining = sorted(

            set(
                df[
                    "track_key"
                ].astype(
                    str
                )
            )

            -

            set(
                inner_val_tracks
            )
        )

        perm = rng.permutation(
            len(
                remaining
            )
        )

        for pos in perm:

            candidate = remaining[
                int(
                    pos
                )
            ]

            # Never remove the last track of a source from training.
            source = str(

                df.loc[
                    df[
                        "track_key"
                    ].astype(
                        str
                    )
                    ==
                    candidate,
                    "source_group"
                ].iloc[
                    0
                ]
            )

            source_tracks = set(

                df.loc[
                    df[
                        "source_group"
                    ].astype(
                        str
                    )
                    ==
                    source,
                    "track_key"
                ].astype(
                    str
                )
            )

            already_val_source = (
                source_tracks
                &
                set(
                    inner_val_tracks
                )
            )

            if (
                len(
                    source_tracks
                )
                -
                len(
                    already_val_source
                )
                <=
                1
            ):

                continue

            inner_val_tracks.append(
                candidate
            )

            if len(
                inner_val_tracks
            ) >= minimum_val:

                break

    inner_val_tracks = set(
        inner_val_tracks
    )

    all_tracks = set(

        df[
            "track_key"
        ].astype(
            str
        )
    )

    inner_train_tracks = (
        all_tracks
        -
        inner_val_tracks
    )

    if not inner_train_tracks:

        raise RuntimeError(
            "Empty inner training track set."
        )

    if not inner_val_tracks:

        raise RuntimeError(
            "Empty inner validation track set."
        )

    if (
        inner_train_tracks
        &
        inner_val_tracks
    ):

        raise RuntimeError(
            "Inner split track overlap."
        )

    return (
        inner_train_tracks,
        inner_val_tracks
    )


# ============================================================
# TRAIN ONE CROSS-FIT MODEL
# ============================================================

def train_crossfit_model(
    task,
    outer_fold,
    seed,
    arrays,
    metadata,
    inner_train_indices,
    inner_val_indices
):

    ORIG.set_seed(
        seed
    )

    X_all = np.asarray(

        arrays[
            "history_canonical"
        ],

        dtype=np.float32
    )

    Y_all = np.asarray(

        arrays[
            "future_canonical"
        ],

        dtype=np.float32
    )

    truth_abs = np.asarray(

        arrays[
            "future_abs"
        ],

        dtype=np.float64
    )

    origin = np.asarray(

        arrays[
            "origin"
        ],

        dtype=np.float64
    )

    angle = np.asarray(

        arrays[
            "heading_angle"
        ],

        dtype=np.float64
    )

    scale = np.asarray(

        arrays[
            "scale"
        ],

        dtype=np.float64
    )

    weights = make_track_weights(

        metadata,

        inner_train_indices
    )

    train_dataset = ORIG.TrajectoryDataset(

        X_all[
            inner_train_indices
        ],

        Y_all[
            inner_train_indices
        ],

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

    model = ORIG.TrajectoryTransformer(

        TASKS[
            task
        ][
            "future_length"
        ]
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

    X_val = X_all[
        inner_val_indices
    ]

    val_truth_abs = truth_abs[
        inner_val_indices
    ]

    val_meta = (

        metadata
        .iloc[
            inner_val_indices
        ]
        .reset_index(
            drop=True
        )
    )

    val_origin = origin[
        inner_val_indices
    ]

    val_angle = angle[
        inner_val_indices
    ]

    val_scale = scale[
        inner_val_indices
    ]


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

        pred_val_can = predict_model(

            model,

            X_val
        )

        pred_val_abs = inverse_canonical(

            pred_val_can,

            val_origin,

            val_angle,

            val_scale
        )

        val_track_ade = track_macro_ADE(

            pred_val_abs,

            val_truth_abs,

            val_meta
        )

        scheduler.step(
            val_track_ade
        )

        history_rows.append({

            "task":
                task,

            "outer_fold":
                outer_fold,

            "seed":
                seed,

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "inner_validation_track_macro_ADE":
                val_track_ade,

            "learning_rate":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
        })

        if (

            val_track_ade

            <

            best_metric
            -
            MIN_DELTA
        ):

            best_metric = (
                val_track_ade
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

                f"{task} | outer {outer_fold} | "
                f"seed {seed} | epoch {epoch:03d} | "
                f"loss={train_loss:.6f} | "
                f"inner-val track-ADE={val_track_ade:.6f} | "
                f"best={best_metric:.6f}"
            )

        if no_improvement >= PATIENCE:

            print(

                f"{task} | outer {outer_fold} | "
                f"seed {seed} | early stopping "
                f"at epoch {epoch}"
            )

            break

    if best_state is None:

        raise RuntimeError(
            "No best cross-fit state captured."
        )

    model.load_state_dict(
        best_state
    )

    checkpoint_path = (

        CHECKPOINT_DIR

        /

        (
            f"{task}_outerfold{outer_fold}_"
            f"seed{seed}.pt"
        )
    )

    torch.save(

        {

            "state_dict":
                best_state,

            "task":
                task,

            "outer_fold":
                outer_fold,

            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "best_inner_validation_track_macro_ADE":
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

            "protocol":
                "outer-track-crossfit_with_inner-track-validation"
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
# CROSS-FIT ONE TASK
# ============================================================

def crossfit_task(
    task
):

    arrays, metadata = load_trainval(
        task
    )

    outer_table = assign_outer_folds(

        metadata,

        task
    )

    outer_table[
        "task"
    ] = task

    X_all = np.asarray(

        arrays[
            "history_canonical"
        ],

        dtype=np.float32
    )

    Y_true_can = np.asarray(

        arrays[
            "future_canonical"
        ],

        dtype=np.float64
    )

    Y_true_abs = np.asarray(

        arrays[
            "future_abs"
        ],

        dtype=np.float64
    )

    origin = np.asarray(

        arrays[
            "origin"
        ],

        dtype=np.float64
    )

    angle = np.asarray(

        arrays[
            "heading_angle"
        ],

        dtype=np.float64
    )

    scale = np.asarray(

        arrays[
            "scale"
        ],

        dtype=np.float64
    )

    n = len(
        metadata
    )

    oof_prediction_can = np.full(

        (
            n,
            TASKS[
                task
            ][
                "future_length"
            ],
            2
        ),

        np.nan,

        dtype=np.float64
    )

    oof_prediction_abs = np.full_like(

        oof_prediction_can,

        np.nan
    )

    oof_fold = np.full(
        n,
        -1,
        dtype=int
    )

    training_history = []

    training_summary = []

    inner_assignment_rows = []


    track_to_fold = dict(

        zip(

            outer_table[
                "track_key"
            ].astype(
                str
            ),

            outer_table[
                "outer_fold"
            ].astype(
                int
            )
        )
    )

    sample_outer_fold = np.asarray(

        [
            track_to_fold[
                str(
                    t
                )
            ]

            for t in metadata[
                "track_key"
            ]
        ],

        dtype=int
    )


    for outer_fold in range(
        N_OUTER_FOLDS
    ):

        outer_test_tracks = set(

            outer_table.loc[

                outer_table[
                    "outer_fold"
                ]

                ==

                outer_fold,

                "track_key"

            ].astype(
                str
            )
        )

        outer_train_table = outer_table.loc[

            outer_table[
                "outer_fold"
            ]

            !=

            outer_fold

        ].copy()

        (
            inner_train_tracks,
            inner_val_tracks
        ) = make_inner_split(

            outer_train_table,

            task,

            outer_fold
        )

        # Hard leakage checks.
        if (
            outer_test_tracks
            &
            inner_train_tracks
        ):

            raise RuntimeError(
                "Outer test leaks into inner training."
            )

        if (
            outer_test_tracks
            &
            inner_val_tracks
        ):

            raise RuntimeError(
                "Outer test leaks into inner validation."
            )

        inner_train_mask = metadata[
            "track_key"
        ].astype(
            str
        ).isin(
            inner_train_tracks
        )

        inner_val_mask = metadata[
            "track_key"
        ].astype(
            str
        ).isin(
            inner_val_tracks
        )

        outer_test_mask = metadata[
            "track_key"
        ].astype(
            str
        ).isin(
            outer_test_tracks
        )

        inner_train_indices = np.where(
            inner_train_mask
        )[0]

        inner_val_indices = np.where(
            inner_val_mask
        )[0]

        outer_test_indices = np.where(
            outer_test_mask
        )[0]

        if len(
            outer_test_indices
        ) == 0:

            raise RuntimeError(
                "Empty outer held-out fold."
            )

        print(
            "\n"
            +
            "=" * 78
        )

        print(
            f"{task} | OUTER FOLD {outer_fold}"
        )

        print(
            "inner train:",
            len(
                inner_train_indices
            ),
            "windows /",
            len(
                inner_train_tracks
            ),
            "tracks"
        )

        print(
            "inner val:",
            len(
                inner_val_indices
            ),
            "windows /",
            len(
                inner_val_tracks
            ),
            "tracks"
        )

        print(
            "outer held-out:",
            len(
                outer_test_indices
            ),
            "windows /",
            len(
                outer_test_tracks
            ),
            "tracks"
        )

        print(
            "=" * 78
        )


        # Save inner split assignments for audit.
        for track in sorted(
            inner_train_tracks
        ):

            row = outer_train_table.loc[

                outer_train_table[
                    "track_key"
                ].astype(
                    str
                )
                ==
                track

            ].iloc[
                0
            ]

            inner_assignment_rows.append({

                "task":
                    task,

                "outer_fold":
                    outer_fold,

                "track_key":
                    track,

                "source_group":
                    row[
                        "source_group"
                    ],

                "windows":
                    int(
                        row[
                            "windows"
                        ]
                    ),

                "inner_role":
                    "train"
            })

        for track in sorted(
            inner_val_tracks
        ):

            row = outer_train_table.loc[

                outer_train_table[
                    "track_key"
                ].astype(
                    str
                )
                ==
                track

            ].iloc[
                0
            ]

            inner_assignment_rows.append({

                "task":
                    task,

                "outer_fold":
                    outer_fold,

                "track_key":
                    track,

                "source_group":
                    row[
                        "source_group"
                    ],

                "windows":
                    int(
                        row[
                            "windows"
                        ]
                    ),

                "inner_role":
                    "validation"
            })


        seed_outer_predictions_can = []

        seed_outer_predictions_abs = []


        for seed in SEEDS:

            (
                model,
                history,
                best_epoch,
                best_metric
            ) = train_crossfit_model(

                task,

                outer_fold,

                seed,

                arrays,

                metadata,

                inner_train_indices,

                inner_val_indices
            )

            training_history.append(
                history
            )

            training_summary.append({

                "task":
                    task,

                "outer_fold":
                    outer_fold,

                "seed":
                    seed,

                "inner_train_windows":
                    len(
                        inner_train_indices
                    ),

                "inner_train_tracks":
                    len(
                        inner_train_tracks
                    ),

                "inner_validation_windows":
                    len(
                        inner_val_indices
                    ),

                "inner_validation_tracks":
                    len(
                        inner_val_tracks
                    ),

                "outer_heldout_windows":
                    len(
                        outer_test_indices
                    ),

                "outer_heldout_tracks":
                    len(
                        outer_test_tracks
                    ),

                "best_epoch":
                    best_epoch,

                "best_inner_validation_track_macro_ADE":
                    best_metric
            })

            pred_can = predict_model(

                model,

                X_all[
                    outer_test_indices
                ]
            )

            pred_abs = inverse_canonical(

                pred_can,

                origin[
                    outer_test_indices
                ],

                angle[
                    outer_test_indices
                ],

                scale[
                    outer_test_indices
                ]
            )

            seed_outer_predictions_can.append(
                pred_can
            )

            seed_outer_predictions_abs.append(
                pred_abs
            )


        ensemble_can = np.mean(

            np.stack(

                seed_outer_predictions_can,

                axis=0
            ),

            axis=0
        )

        ensemble_abs = np.mean(

            np.stack(

                seed_outer_predictions_abs,

                axis=0
            ),

            axis=0
        )

        oof_prediction_can[
            outer_test_indices
        ] = ensemble_can

        oof_prediction_abs[
            outer_test_indices
        ] = ensemble_abs

        oof_fold[
            outer_test_indices
        ] = outer_fold


    # ========================================================
    # COMPLETE OOF CHECKS
    # ========================================================

    if np.isnan(
        oof_prediction_can
    ).any():

        raise RuntimeError(
            "Some TrainReal windows lack OOF prediction."
        )

    if np.isnan(
        oof_prediction_abs
    ).any():

        raise RuntimeError(
            "Some TrainReal windows lack OOF absolute prediction."
        )

    if np.any(
        oof_fold
        <
        0
    ):

        raise RuntimeError(
            "Some TrainReal windows lack outer fold."
        )

    # Every track must appear in exactly one outer fold.
    check = pd.DataFrame({

        "track_key":
            metadata[
                "track_key"
            ].astype(
                str
            ),

        "outer_fold":
            oof_fold
    })

    per_track_folds = (

        check
        .groupby(
            "track_key"
        )[
            "outer_fold"
        ]
        .nunique()
    )

    if int(
        per_track_folds.max()
    ) != 1:

        raise RuntimeError(
            "A track appears in multiple outer folds."
        )


    # Exact canonical -> absolute distance consistency.
    err_can = np.linalg.norm(

        oof_prediction_can

        -

        Y_true_can,

        axis=2
    )

    err_abs = np.linalg.norm(

        oof_prediction_abs

        -

        Y_true_abs,

        axis=2
    )

    reconstructed_abs = (

        err_can

        *

        scale[
            :,
            None
        ]
    )

    consistency_diff = float(

        np.max(

            np.abs(

                reconstructed_abs

                -

                err_abs
            )
        )
    )

    if consistency_diff > 1e-5:

        raise RuntimeError(

            "OOF canonical/absolute error "
            f"consistency failed: {consistency_diff}"
        )


    # ========================================================
    # OOF SAMPLE SCORES
    # ========================================================

    sample_scores = err_can.max(
        axis=1
    )

    sample_ADE = err_abs.mean(
        axis=1
    )

    sample_FDE = err_abs[
        :,
        -1
    ]

    oof_sample_df = metadata[
        [
            "sample_id",
            "source_group",
            "track_key"
        ]
    ].copy()

    oof_sample_df[
        "task"
    ] = task

    oof_sample_df[
        "outer_fold"
    ] = oof_fold

    oof_sample_df[
        "scale"
    ] = scale

    oof_sample_df[
        "max_canonical_error"
    ] = sample_scores

    oof_sample_df[
        "ADE_absolute"
    ] = sample_ADE

    oof_sample_df[
        "FDE_absolute"
    ] = sample_FDE


    # ========================================================
    # PRIMARY TRACK-LEVEL SCORES
    #
    # One score per independent-ish calibration unit:
    # max over windows AND future steps for each track.
    # ========================================================

    track_score_df = (

        oof_sample_df
        .groupby(
            "track_key",
            as_index=False
        )
        .agg(

            source_group=(
                "source_group",
                "first"
            ),

            outer_fold=(
                "outer_fold",
                "first"
            ),

            windows=(
                "sample_id",
                "size"
            ),

            track_max_canonical_error=(
                "max_canonical_error",
                "max"
            ),

            track_mean_ADE_absolute=(
                "ADE_absolute",
                "mean"
            ),

            track_mean_FDE_absolute=(
                "FDE_absolute",
                "mean"
            )
        )
    )

    track_score_df[
        "task"
    ] = task


    # ========================================================
    # STEPWISE TRACK-LEVEL SCORES
    # For each forecast step and track:
    # max error over windows in that track.
    # ========================================================

    stepwise_rows = []

    for step in range(
        err_can.shape[
            1
        ]
    ):

        temp = pd.DataFrame({

            "track_key":
                metadata[
                    "track_key"
                ].astype(
                    str
                ).to_numpy(),

            "error":
                err_can[
                    :,
                    step
                ]
        })

        by_track = (

            temp
            .groupby(
                "track_key",
                as_index=False
            )
            .agg(
                track_step_max_error=(
                    "error",
                    "max"
                )
            )
        )

        by_track[
            "task"
        ] = task

        by_track[
            "forecast_step"
        ] = step + 1

        by_track[
            "forecast_time_sec"
        ] = (
            step + 1
        ) / 10.0

        stepwise_rows.append(
            by_track
        )

    stepwise_track_scores = pd.concat(

        stepwise_rows,

        ignore_index=True
    )


    return {

        "outer_fold_table":
            outer_table,

        "inner_assignment":
            pd.DataFrame(
                inner_assignment_rows
            ),

        "training_history":
            pd.concat(
                training_history,
                ignore_index=True
            ),

        "training_summary":
            pd.DataFrame(
                training_summary
            ),

        "oof_sample_scores":
            oof_sample_df,

        "track_scores":
            track_score_df,

        "stepwise_track_scores":
            stepwise_track_scores,

        "consistency_diff":
            consistency_diff,

        "metadata":
            metadata,

        "arrays":
            arrays
    }


# ============================================================
# CONFORMAL QUANTILE ON TRACK SCORES
# ============================================================

def conformal_quantile(
    scores,
    alpha
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    scores = scores[
        np.isfinite(
            scores
        )
    ]

    n = len(
        scores
    )

    if n == 0:

        raise RuntimeError(
            "No finite conformal scores."
        )

    k = int(

        np.ceil(

            (
                n
                +
                1
            )

            *

            (
                1.0
                -
                alpha
            )
        )
    )

    k = min(
        max(
            k,
            1
        ),
        n
    )

    ordered = np.sort(
        scores
    )

    return (

        float(
            ordered[
                k
                -
                1
            ]
        ),

        k,

        n
    )


def build_track_calibration(
    task,
    track_scores,
    stepwise_track_scores
):

    rows = []

    step_rows = []

    objects = {}

    primary_scores = track_scores[
        "track_max_canonical_error"
    ].to_numpy(
        dtype=float
    )

    for alpha in ALPHAS:

        target = (
            1.0
            -
            alpha
        )

        q_track, k, n = conformal_quantile(

            primary_scores,

            alpha
        )

        cal_track_coverage = float(

            (
                primary_scores

                <=

                q_track
            ).mean()
        )

        q_steps = []

        for step in sorted(

            stepwise_track_scores[
                "forecast_step"
            ].unique()
        ):

            step_scores = stepwise_track_scores.loc[

                stepwise_track_scores[
                    "forecast_step"
                ]

                ==

                step,

                "track_step_max_error"

            ].to_numpy(
                dtype=float
            )

            q_step, k_step, n_step = conformal_quantile(

                step_scores,

                alpha
            )

            q_steps.append(
                q_step
            )

            step_rows.append({

                "task":
                    task,

                "alpha":
                    alpha,

                "target_coverage":
                    target,

                "forecast_step":
                    int(
                        step
                    ),

                "forecast_time_sec":
                    float(
                        step
                    )
                    /
                    10.0,

                "normalized_track_step_radius":
                    q_step,

                "order_statistic_k":
                    k_step,

                "calibration_tracks":
                    n_step,

                "calibration_track_step_coverage":
                    float(

                        (
                            step_scores

                            <=

                            q_step
                        ).mean()
                    )
            })

        q_steps = np.asarray(
            q_steps,
            dtype=np.float64
        )

        objects[
            alpha
        ] = {

            "q_track":
                q_track,

            "q_steps":
                q_steps
        }

        rows.append({

            "task":
                task,

            "alpha":
                alpha,

            "target_coverage":
                target,

            "calibration_tracks":
                n,

            "normalized_track_simultaneous_radius":
                q_track,

            "order_statistic_k":
                k,

            "calibration_track_simultaneous_coverage":
                cal_track_coverage,

            "mean_normalized_track_step_radius":
                float(
                    q_steps.mean()
                ),

            "endpoint_normalized_track_step_radius":
                float(
                    q_steps[
                        -1
                    ]
                )
        })

    return (

        pd.DataFrame(
            rows
        ),

        pd.DataFrame(
            step_rows
        ),

        objects
    )


# ============================================================
# LOAD EXISTING FINAL FROZEN ENSEMBLE
# ============================================================

def load_final_model(
    task,
    seed
):

    checkpoint = torch.load(

        TRANSFORMER_DIR

        /

        f"{task}_Transformer_seed{seed}.pt",

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

    return model


def final_ensemble_predict(
    task,
    arrays
):

    X = np.asarray(

        arrays[
            "history_canonical"
        ],

        dtype=np.float32
    )

    origin = np.asarray(

        arrays[
            "origin"
        ],

        dtype=np.float64
    )

    angle = np.asarray(

        arrays[
            "heading_angle"
        ],

        dtype=np.float64
    )

    scale = np.asarray(

        arrays[
            "scale"
        ],

        dtype=np.float64
    )

    seed_can = []

    seed_abs = []

    for seed in SEEDS:

        model = load_final_model(

            task,

            seed
        )

        pred_can = predict_model(

            model,

            X
        )

        pred_abs = inverse_canonical(

            pred_can,

            origin,

            angle,

            scale
        )

        seed_can.append(
            np.asarray(
                pred_can,
                dtype=np.float64
            )
        )

        seed_abs.append(
            pred_abs
        )

    stacked_can = np.stack(

        seed_can,

        axis=0
    )

    stacked_abs = np.stack(

        seed_abs,

        axis=0
    )

    mean_can = np.mean(

        stacked_can,

        axis=0
    )

    mean_abs = np.mean(

        stacked_abs,

        axis=0
    )

    disagreement_abs = np.mean(

        np.linalg.norm(

            stacked_abs

            -

            mean_abs[
                None,
                ...
            ],

            axis=3
        ),

        axis=0
    )

    return (

        stacked_can,

        stacked_abs,

        mean_can,

        mean_abs,

        disagreement_abs
    )


# ============================================================
# FRESH MODEL SELF-CHECK
# ============================================================

def fresh_self_check(
    task,
    stacked_abs,
    mean_abs,
    truth_abs
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

    for seed_index, seed in enumerate(
        SEEDS
    ):

        err = np.linalg.norm(

            stacked_abs[
                seed_index
            ]

            -

            truth_abs,

            axis=2
        )

        ade = float(
            err.mean(
                axis=1
            ).mean()
        )

        fde = float(
            err[
                :,
                -1
            ].mean()
        )

        expected = saved[

            saved[
                "model"
            ]

            ==

            f"Transformer_seed{seed}"
        ]

        if len(
            expected
        ) != 1:

            raise RuntimeError(
                "Missing Fresh seed metric."
            )

        expected = expected.iloc[
            0
        ]

        ade_diff = abs(

            ade

            -

            float(
                expected[
                    "ADE"
                ]
            )
        )

        fde_diff = abs(

            fde

            -

            float(
                expected[
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

            ade_diff
            >
            SELF_CHECK_TOL

            or

            fde_diff
            >
            SELF_CHECK_TOL
        ):

            raise RuntimeError(
                "Fresh seed self-check failed."
            )

    err = np.linalg.norm(

        mean_abs

        -

        truth_abs,

        axis=2
    )

    ade = float(
        err.mean(
            axis=1
        ).mean()
    )

    fde = float(
        err[
            :,
            -1
        ].mean()
    )

    expected = saved[

        saved[
            "model"
        ]

        ==

        "Transformer_3Seed_Ensemble"
    ]

    if len(
        expected
    ) != 1:

        raise RuntimeError(
            "Missing Fresh ensemble metric."
        )

    expected = expected.iloc[
        0
    ]

    ade_diff = abs(

        ade

        -

        float(
            expected[
                "ADE"
            ]
        )
    )

    fde_diff = abs(

        fde

        -

        float(
            expected[
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

        ade_diff
        >
        SELF_CHECK_TOL

        or

        fde_diff
        >
        SELF_CHECK_TOL
    ):

        raise RuntimeError(
            "Fresh ensemble self-check failed."
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TRACK-CLUSTER BOOTSTRAP EVALUATION
# ============================================================

def bootstrap_track_metrics(
    metadata,
    window_covered,
    rng
):

    temp = pd.DataFrame({

        "track_key":
            metadata[
                "track_key"
            ].astype(
                str
            ).to_numpy(),

        "covered":
            np.asarray(
                window_covered,
                dtype=float
            )
    })

    by_track = (

        temp
        .groupby(
            "track_key",
            as_index=False
        )
        .agg(

            covered_windows=(
                "covered",
                "sum"
            ),

            windows=(
                "covered",
                "size"
            ),

            track_macro_window_coverage=(
                "covered",
                "mean"
            ),

            track_fully_covered=(
                "covered",
                "min"
            )
        )
    )

    n_tracks = len(
        by_track
    )

    covered_windows = by_track[
        "covered_windows"
    ].to_numpy(
        dtype=float
    )

    windows = by_track[
        "windows"
    ].to_numpy(
        dtype=float
    )

    macro = by_track[
        "track_macro_window_coverage"
    ].to_numpy(
        dtype=float
    )

    full = by_track[
        "track_fully_covered"
    ].to_numpy(
        dtype=float
    )

    boot_window = np.empty(
        N_BOOT,
        dtype=float
    )

    boot_macro = np.empty(
        N_BOOT,
        dtype=float
    )

    boot_full = np.empty(
        N_BOOT,
        dtype=float
    )

    for b in range(
        N_BOOT
    ):

        idx = rng.integers(

            0,

            n_tracks,

            size=n_tracks
        )

        boot_window[
            b
        ] = (

            covered_windows[
                idx
            ].sum()

            /

            windows[
                idx
            ].sum()
        )

        boot_macro[
            b
        ] = macro[
            idx
        ].mean()

        boot_full[
            b
        ] = full[
            idx
        ].mean()

    return {

        "window_coverage":
            float(
                np.asarray(
                    window_covered,
                    dtype=float
                ).mean()
            ),

        "window_CI_low":
            float(
                np.quantile(
                    boot_window,
                    0.025
                )
            ),

        "window_CI_high":
            float(
                np.quantile(
                    boot_window,
                    0.975
                )
            ),

        "track_macro_window_coverage":
            float(
                macro.mean()
            ),

        "track_macro_CI_low":
            float(
                np.quantile(
                    boot_macro,
                    0.025
                )
            ),

        "track_macro_CI_high":
            float(
                np.quantile(
                    boot_macro,
                    0.975
                )
            ),

        "fully_covered_track_fraction":
            float(
                full.mean()
            ),

        "fully_covered_track_CI_low":
            float(
                np.quantile(
                    boot_full,
                    0.025
                )
            ),

        "fully_covered_track_CI_high":
            float(
                np.quantile(
                    boot_full,
                    0.975
                )
            )
    }


# ============================================================
# EVALUATE TRACK-CALIBRATED RADII
# ============================================================

def evaluate_dataset(
    task,
    dataset_name,
    dataset_role,
    arrays,
    metadata,
    mean_can,
    mean_abs,
    disagreement_abs,
    calibration_objects,
    rng
):

    truth_can = np.asarray(

        arrays[
            "future_canonical"
        ],

        dtype=np.float64
    )

    truth_abs = np.asarray(

        arrays[
            "future_abs"
        ],

        dtype=np.float64
    )

    scale = np.asarray(

        arrays[
            "scale"
        ],

        dtype=np.float64
    )

    err_can = np.linalg.norm(

        mean_can

        -

        truth_can,

        axis=2
    )

    err_abs = np.linalg.norm(

        mean_abs

        -

        truth_abs,

        axis=2
    )

    reconstructed_abs = (

        err_can

        *

        scale[
            :,
            None
        ]
    )

    consistency_diff = float(

        np.max(

            np.abs(

                reconstructed_abs

                -

                err_abs
            )
        )
    )

    if consistency_diff > 1e-5:

        raise RuntimeError(

            f"{task} {dataset_name}: "
            f"distance consistency failed {consistency_diff}"
        )

    sample_max_can = err_can.max(
        axis=1
    )

    summary_rows = []

    time_rows = []

    source_rows = []

    failure_rows = []


    for alpha in ALPHAS:

        target = (
            1.0
            -
            alpha
        )

        q_track = calibration_objects[
            alpha
        ][
            "q_track"
        ]

        q_steps = calibration_objects[
            alpha
        ][
            "q_steps"
        ]

        absolute_radius = (

            q_track

            *

            scale
        )

        window_covered = (

            sample_max_can

            <=

            q_track
        )

        boot = bootstrap_track_metrics(

            metadata,

            window_covered,

            rng
        )

        step_radius_abs = (

            scale[
                :,
                None
            ]

            *

            q_steps[
                None,
                :
            ]
        )

        step_covered = (

            err_abs

            <=

            step_radius_abs
        )

        summary_rows.append({

            "task":
                task,

            "dataset":
                dataset_name,

            "dataset_role_for_uncertainty":
                dataset_role,

            "alpha":
                alpha,

            "target_coverage":
                target,

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

            "normalized_track_simultaneous_radius":
                q_track,

            "mean_absolute_radius":
                float(
                    absolute_radius.mean()
                ),

            "median_absolute_radius":
                float(
                    np.median(
                        absolute_radius
                    )
                ),

            "p95_absolute_radius":
                float(
                    np.quantile(
                        absolute_radius,
                        0.95
                    )
                ),

            "window_simultaneous_coverage":
                boot[
                    "window_coverage"
                ],

            "window_coverage_gap_pp":
                100.0
                *
                (
                    boot[
                        "window_coverage"
                    ]
                    -
                    target
                ),

            "window_CI_low":
                boot[
                    "window_CI_low"
                ],

            "window_CI_high":
                boot[
                    "window_CI_high"
                ],

            "track_macro_window_coverage":
                boot[
                    "track_macro_window_coverage"
                ],

            "track_macro_CI_low":
                boot[
                    "track_macro_CI_low"
                ],

            "track_macro_CI_high":
                boot[
                    "track_macro_CI_high"
                ],

            "fully_covered_track_fraction":
                boot[
                    "fully_covered_track_fraction"
                ],

            "fully_covered_track_CI_low":
                boot[
                    "fully_covered_track_CI_low"
                ],

            "fully_covered_track_CI_high":
                boot[
                    "fully_covered_track_CI_high"
                ],

            "stepwise_window_coverage":
                float(
                    step_covered.mean()
                ),

            "mean_ADE":
                float(
                    err_abs.mean(
                        axis=1
                    ).mean()
                ),

            "mean_FDE":
                float(
                    err_abs[
                        :,
                        -1
                    ].mean()
                ),

            "mean_ensemble_disagreement":
                float(
                    disagreement_abs.mean()
                ),

            "max_abs_vs_scaled_canonical_error_diff":
                consistency_diff
        })


        # ----------------------------------------------------
        # TIME-WISE
        # ----------------------------------------------------

        for step in range(
            err_abs.shape[
                1
            ]
        ):

            this_covered = (

                err_can[
                    :,
                    step
                ]

                <=

                q_steps[
                    step
                ]
            )

            time_rows.append({

                "task":
                    task,

                "dataset":
                    dataset_name,

                "dataset_role_for_uncertainty":
                    dataset_role,

                "alpha":
                    alpha,

                "target_coverage":
                    target,

                "forecast_step":
                    step + 1,

                "forecast_time_sec":
                    (
                        step + 1
                    )
                    /
                    10.0,

                "normalized_track_step_radius":
                    q_steps[
                        step
                    ],

                "mean_absolute_step_radius":
                    float(

                        (
                            q_steps[
                                step
                            ]

                            *

                            scale
                        ).mean()
                    ),

                "window_coverage":
                    float(
                        this_covered.mean()
                    ),

                "coverage_gap_pp":
                    100.0
                    *
                    (
                        this_covered.mean()
                        -
                        target
                    ),

                "mean_error_absolute":
                    float(
                        err_abs[
                            :,
                            step
                        ].mean()
                    ),

                "mean_disagreement_absolute":
                    float(
                        disagreement_abs[
                            :,
                            step
                        ].mean()
                    )
            })


        # ----------------------------------------------------
        # SOURCE-WISE
        # ----------------------------------------------------

        temp = metadata[
            [
                "source_group",
                "track_key"
            ]
        ].copy()

        temp[
            "covered"
        ] = window_covered.astype(
            int
        )

        temp[
            "absolute_radius"
        ] = absolute_radius

        temp[
            "max_error_absolute"
        ] = err_abs.max(
            axis=1
        )

        for source_group, group in temp.groupby(
            "source_group"
        ):

            by_track = (

                group
                .groupby(
                    "track_key"
                )[
                    "covered"
                ]
                .agg(
                    [
                        "mean",
                        "min"
                    ]
                )
            )

            source_rows.append({

                "task":
                    task,

                "dataset":
                    dataset_name,

                "dataset_role_for_uncertainty":
                    dataset_role,

                "alpha":
                    alpha,

                "target_coverage":
                    target,

                "source_group":
                    source_group,

                "samples":
                    len(
                        group
                    ),

                "tracks":
                    group[
                        "track_key"
                    ].nunique(),

                "window_coverage":
                    float(
                        group[
                            "covered"
                        ].mean()
                    ),

                "track_macro_window_coverage":
                    float(
                        by_track[
                            "mean"
                        ].mean()
                    ),

                "fully_covered_track_fraction":
                    float(
                        by_track[
                            "min"
                        ].mean()
                    ),

                "mean_absolute_radius":
                    float(
                        group[
                            "absolute_radius"
                        ].mean()
                    ),

                "mean_max_error_absolute":
                    float(
                        group[
                            "max_error_absolute"
                        ].mean()
                    )
            })


        # ----------------------------------------------------
        # FAILURE CASES
        # ----------------------------------------------------

        failures = np.where(
            ~window_covered
        )[0]

        if len(
            failures
        ) > 0:

            excess = (

                sample_max_can[
                    failures
                ]

                -

                q_track
            )

            order = failures[
                np.argsort(
                    -excess
                )
            ]

            top = order[
                :min(
                    30,
                    len(
                        order
                    )
                )
            ]

            for idx in top:

                failure_rows.append({

                    "task":
                        task,

                    "dataset":
                        dataset_name,

                    "dataset_role_for_uncertainty":
                        dataset_role,

                    "alpha":
                        alpha,

                    "target_coverage":
                        target,

                    "sample_id":
                        metadata.iloc[
                            idx
                        ].get(
                            "sample_id",
                            idx
                        ),

                    "source_group":
                        metadata.iloc[
                            idx
                        ][
                            "source_group"
                        ],

                    "track_key":
                        metadata.iloc[
                            idx
                        ][
                            "track_key"
                        ],

                    "scale":
                        float(
                            scale[
                                idx
                            ]
                        ),

                    "max_canonical_error":
                        float(
                            sample_max_can[
                                idx
                            ]
                        ),

                    "normalized_radius":
                        q_track,

                    "max_absolute_error":
                        float(
                            err_abs[
                                idx
                            ].max()
                        ),

                    "absolute_radius":
                        float(
                            absolute_radius[
                                idx
                            ]
                        ),

                    "ADE":
                        float(
                            err_abs[
                                idx
                            ].mean()
                        ),

                    "FDE":
                        float(
                            err_abs[
                                idx,
                                -1
                            ]
                        ),

                    "mean_disagreement":
                        float(
                            disagreement_abs[
                                idx
                            ].mean()
                        )
                })

    return (

        pd.DataFrame(
            summary_rows
        ),

        pd.DataFrame(
            time_rows
        ),

        pd.DataFrame(
            source_rows
        ),

        pd.DataFrame(
            failure_rows
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rng = np.random.default_rng(
        RNG_SEED
    )

    outer_tables = []

    inner_tables = []

    training_histories = []

    training_summaries = []

    oof_samples = []

    track_scores_all = []

    stepwise_track_scores_all = []

    calibration_all = []

    calibration_stepwise_all = []

    evaluation_all = []

    timewise_all = []

    sourcewise_all = []

    failures_all = []

    self_checks_all = []

    protocol_audit_rows = []


    calibration_objects_by_task = {}


    # ========================================================
    # CROSS-FIT CALIBRATION
    # ========================================================

    for task in TASKS:

        print(
            "\n"
            +
            "#" * 78
        )

        print(
            "CROSS-FITTING:",
            task
        )

        print(
            "#" * 78
        )

        result = crossfit_task(
            task
        )

        outer_tables.append(
            result[
                "outer_fold_table"
            ]
        )

        inner_tables.append(
            result[
                "inner_assignment"
            ]
        )

        training_histories.append(
            result[
                "training_history"
            ]
        )

        training_summaries.append(
            result[
                "training_summary"
            ]
        )

        oof_samples.append(
            result[
                "oof_sample_scores"
            ]
        )

        track_scores_all.append(
            result[
                "track_scores"
            ]
        )

        stepwise_track_scores_all.append(
            result[
                "stepwise_track_scores"
            ]
        )

        (
            calibration_df,
            calibration_step_df,
            calibration_objects
        ) = build_track_calibration(

            task,

            result[
                "track_scores"
            ],

            result[
                "stepwise_track_scores"
            ]
        )

        calibration_all.append(
            calibration_df
        )

        calibration_stepwise_all.append(
            calibration_step_df
        )

        calibration_objects_by_task[
            task
        ] = calibration_objects


        protocol_audit_rows.append({

            "task":
                task,

            "trainval_windows":
                len(
                    result[
                        "metadata"
                    ]
                ),

            "trainval_tracks":
                result[
                    "metadata"
                ][
                    "track_key"
                ].nunique(),

            "outer_folds":
                N_OUTER_FOLDS,

            "seeds_per_outer_fold":
                len(
                    SEEDS
                ),

            "total_crossfit_models":
                N_OUTER_FOLDS
                *
                len(
                    SEEDS
                ),

            "all_windows_have_oof_prediction":
                True,

            "all_tracks_single_outer_fold":
                True,

            "max_abs_vs_scaled_canonical_error_diff":
                result[
                    "consistency_diff"
                ],

            "primary_calibration_unit":
                "track",

            "primary_score":
                (
                    "max canonical Euclidean error "
                    "across all windows and future steps "
                    "within track"
                )
        })


    outer_df = pd.concat(
        outer_tables,
        ignore_index=True
    )

    inner_df = pd.concat(
        inner_tables,
        ignore_index=True
    )

    training_history_df = pd.concat(
        training_histories,
        ignore_index=True
    )

    training_summary_df = pd.concat(
        training_summaries,
        ignore_index=True
    )

    oof_sample_df = pd.concat(
        oof_samples,
        ignore_index=True
    )

    track_score_df = pd.concat(
        track_scores_all,
        ignore_index=True
    )

    stepwise_track_score_df = pd.concat(
        stepwise_track_scores_all,
        ignore_index=True
    )

    calibration_df = pd.concat(
        calibration_all,
        ignore_index=True
    )

    calibration_step_df = pd.concat(
        calibration_stepwise_all,
        ignore_index=True
    )

    protocol_audit_df = pd.DataFrame(
        protocol_audit_rows
    )


    # ========================================================
    # RETROSPECTIVE EVALUATION ON EXISTING DATASETS
    # ========================================================

    for task in TASKS:

        cal_objects = calibration_objects_by_task[
            task
        ]

        # ----------------------------------------------------
        # BIRDSAI TESTREAL
        # Already inspected in 6A/6B => retrospective
        # ----------------------------------------------------

        test_arrays, test_meta = load_testreal(
            task
        )

        (
            _,
            _,
            test_mean_can,
            test_mean_abs,
            test_disagreement
        ) = final_ensemble_predict(

            task,

            test_arrays
        )

        (
            eval_df,
            time_df,
            source_df,
            failure_df
        ) = evaluate_dataset(

            task,

            "BIRDSAI_TESTREAL",

            "RETROSPECTIVE_AFTER_STEP6A_6B",

            test_arrays,

            test_meta,

            test_mean_can,

            test_mean_abs,

            test_disagreement,

            cal_objects,

            rng
        )

        evaluation_all.append(
            eval_df
        )

        timewise_all.append(
            time_df
        )

        sourcewise_all.append(
            source_df
        )

        if len(
            failure_df
        ) > 0:

            failures_all.append(
                failure_df
            )


        # ----------------------------------------------------
        # FRESH WILDBOX
        # Already used to judge 6A/6B => diagnostic only
        # ----------------------------------------------------

        fresh_arrays, fresh_meta = load_fresh(
            task
        )

        (
            _,
            fresh_stacked_abs,
            fresh_mean_can,
            fresh_mean_abs,
            fresh_disagreement
        ) = final_ensemble_predict(

            task,

            fresh_arrays
        )

        check = fresh_self_check(

            task,

            fresh_stacked_abs,

            fresh_mean_abs,

            np.asarray(
                fresh_arrays[
                    "future_abs"
                ],
                dtype=np.float64
            )
        )

        self_checks_all.append(
            check
        )

        (
            eval_df,
            time_df,
            source_df,
            failure_df
        ) = evaluate_dataset(

            task,

            "FRESH_WILDBOX",

            "DIAGNOSTIC_ONLY_ALREADY_INSPECTED",

            fresh_arrays,

            fresh_meta,

            fresh_mean_can,

            fresh_mean_abs,

            fresh_disagreement,

            cal_objects,

            rng
        )

        evaluation_all.append(
            eval_df
        )

        timewise_all.append(
            time_df
        )

        sourcewise_all.append(
            source_df
        )

        if len(
            failure_df
        ) > 0:

            failures_all.append(
                failure_df
            )


    evaluation_df = pd.concat(
        evaluation_all,
        ignore_index=True
    )

    timewise_df = pd.concat(
        timewise_all,
        ignore_index=True
    )

    sourcewise_df = pd.concat(
        sourcewise_all,
        ignore_index=True
    )

    self_check_df = pd.concat(
        self_checks_all,
        ignore_index=True
    )

    if failures_all:

        failures_df = pd.concat(
            failures_all,
            ignore_index=True
        )

    else:

        failures_df = pd.DataFrame()


    # ========================================================
    # SAVE EVERYTHING
    # ========================================================

    outer_df.to_csv(

        OUT /
        "outer_track_fold_assignment.csv",

        index=False
    )

    inner_df.to_csv(

        OUT /
        "inner_track_split_assignment.csv",

        index=False
    )

    training_history_df.to_csv(

        OUT /
        "crossfit_training_history.csv",

        index=False
    )

    training_summary_df.to_csv(

        OUT /
        "crossfit_training_summary.csv",

        index=False
    )

    oof_sample_df.to_csv(

        OUT /
        "crossfit_oof_sample_scores.csv",

        index=False
    )

    track_score_df.to_csv(

        OUT /
        "crossfit_track_calibration_scores.csv",

        index=False
    )

    stepwise_track_score_df.to_csv(

        OUT /
        "crossfit_stepwise_track_scores.csv",

        index=False
    )

    calibration_df.to_csv(

        OUT /
        "track_conformal_calibration_summary.csv",

        index=False
    )

    calibration_step_df.to_csv(

        OUT /
        "track_conformal_stepwise_radii.csv",

        index=False
    )

    protocol_audit_df.to_csv(

        OUT /
        "crossfit_protocol_audit.csv",

        index=False
    )

    evaluation_df.to_csv(

        OUT /
        "track_conformal_retrospective_evaluation.csv",

        index=False
    )

    timewise_df.to_csv(

        OUT /
        "track_conformal_timewise_evaluation.csv",

        index=False
    )

    sourcewise_df.to_csv(

        OUT /
        "track_conformal_sourcewise_evaluation.csv",

        index=False
    )

    failures_df.to_csv(

        OUT /
        "track_conformal_failure_cases.csv",

        index=False
    )

    self_check_df.to_csv(

        OUT /
        "fresh_transformer_self_check.csv",

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
            "STEP 6C: TRACK-LEVEL CROSS-FITTED CONFORMAL CALIBRATION\n"
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
            "- All TrainReal tracks receive out-of-fold predictions.\n"
        )

        f.write(
            "- Three outer folds are track-disjoint.\n"
        )

        f.write(
            "- Outer held-out tracks are never used for training or early stopping.\n"
        )

        f.write(
            "- Each outer training complement has a separate inner "
            "track-disjoint validation split for early stopping.\n"
        )

        f.write(
            "- Three Transformer seeds (42/52/62) are trained per outer fold.\n"
        )

        f.write(
            "- Training weights reproduce the original exact rule: "
            "each training track contributes equal total weight.\n"
        )

        f.write(
            "- Primary conformal unit is TRACK, not sliding window.\n"
        )

        f.write(
            "- Primary track score is the maximum canonical Euclidean "
            "forecast error over all windows and future steps in that track.\n"
        )

        f.write(
            "- This handles within-track overlapping-window dependence more "
            "conservatively than window-level conformal calibration.\n"
        )

        f.write(
            "- Existing TestReal and Fresh WildBox were already inspected "
            "during Step 6A/6B; Step 6C results on them are retrospective/"
            "diagnostic and are NOT a new untouched confirmatory uncertainty test.\n"
        )

        f.write(
            "- A new untouched uncertainty holdout is still required before "
            "making a final confirmatory coverage claim.\n\n"
        )


        f.write(
            "PROTOCOL AUDIT\n"
        )

        f.write(
            "--------------\n"
        )

        f.write(
            protocol_audit_df.to_string(
                index=False
            )
        )


        f.write(
            "\n\nTRACK CONFORMAL CALIBRATION\n"
        )

        f.write(
            "---------------------------\n"
        )

        f.write(
            calibration_df.to_string(
                index=False
            )
        )


        f.write(
            "\n\nRETROSPECTIVE / DIAGNOSTIC EVALUATION\n"
        )

        f.write(
            "-------------------------------------\n"
        )

        f.write(
            evaluation_df.to_string(
                index=False
            )
        )


        f.write(
            "\n\nFRESH TRANSFORMER SELF-CHECK\n"
        )

        f.write(
            "----------------------------\n"
        )

        f.write(
            self_check_df.to_string(
                index=False
            )
        )


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(
        "\nSTEP 6C COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


    print(
        "\nCROSS-FIT PROTOCOL AUDIT:\n"
    )

    print(
        protocol_audit_df.to_string(
            index=False
        )
    )


    print(
        "\nTRACK-LEVEL CONFORMAL CALIBRATION:\n"
    )

    print(
        calibration_df[
            [
                "task",
                "alpha",
                "target_coverage",
                "calibration_tracks",
                "normalized_track_simultaneous_radius",
                "order_statistic_k",
                "calibration_track_simultaneous_coverage",
                "endpoint_normalized_track_step_radius"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nRETROSPECTIVE / DIAGNOSTIC COVERAGE:\n"
    )

    print(
        evaluation_df[
            [
                "task",
                "dataset",
                "dataset_role_for_uncertainty",
                "alpha",
                "target_coverage",
                "samples",
                "tracks",
                "normalized_track_simultaneous_radius",
                "mean_absolute_radius",
                "window_simultaneous_coverage",
                "window_coverage_gap_pp",
                "track_macro_window_coverage",
                "fully_covered_track_fraction",
                "fully_covered_track_CI_low",
                "fully_covered_track_CI_high",
                "stepwise_window_coverage",
                "mean_ADE",
                "mean_FDE"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nFRESH TRANSFORMER SELF-CHECK:\n"
    )

    print(
        self_check_df.to_string(
            index=False
        )
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "Step 6C creates stronger track-level calibration, "
        "but a NEW untouched uncertainty holdout is still "
        "required for a final confirmatory coverage claim."
    )


if __name__ == "__main__":

    main()
