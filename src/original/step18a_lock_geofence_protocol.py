from pathlib import Path
import copy
import hashlib
import json
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
    r"C:\Users\HP\Downloads\step18_geofence"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORTANT
#
# STEP 18A USES ONLY:
# BIRDSAI TrainReal fixed track-disjoint validation
#
# It does NOT load:
# - BIRDSAI TestReal
# - WildBox old external set
# - Fresh WildBox confirmatory holdout
# ============================================================


DEVICE = torch.device("cpu")

SEEDS = [
    42,
    52,
    62
]

TASKS = [
    "2s_to_2s",
    "2s_to_5s"
]

FPS = 10.0

EPS = 1e-10


# ============================================================
# LOCKED VIRTUAL-GEOFENCE PROTOCOL
# ============================================================

GEOFENCE_THRESHOLDS = [
    0.25,
    0.50,
    0.75
]

AXES = {
    "x": 0,
    "y": 1,
}


# ============================================================
# WARNING DECISION
#
# risk_score >= 0
#
# means:
# predicted mean trajectory reaches or crosses
# the virtual boundary.
#
# No F1/AUROC-based threshold tuning.
# ============================================================

WARNING_THRESHOLD = 0.0


PROTOCOL_CONFIG = {

    "version":
        "STEP18_VIRTUAL_GEOFENCE_V1",

    "coordinate_system":
        "camera-corrected normalized image-plane coordinates",

    "observation_seconds":
        2.0,

    "forecast_horizons_seconds":
        [
            2.0,
            5.0
        ],

    "sampling_hz":
        FPS,

    "axes":
        [
            "x",
            "y"
        ],

    "boundary_thresholds":
        GEOFENCE_THRESHOLDS,

    "direction_rule":
        (
            "For each boundary, the current side of the last "
            "observed position is treated as the safe side; "
            "the opposite side is the virtual risk zone."
        ),

    "eligibility_rule":
        (
            "Exactly one directional instance per axis-threshold "
            "pair, except exact boundary ties."
        ),

    "true_crossing_rule":
        (
            "At least one ground-truth future point reaches or "
            "crosses the virtual risk side."
        ),

    "prediction_score":
        (
            "Maximum directional signed boundary penetration of "
            "the frozen 3-seed ensemble-mean forecast."
        ),

    "warning_threshold":
        WARNING_THRESHOLD,

    "warning_rule":
        "risk_score >= 0",

    "tte_rule":
        (
            "First future timestep at which trajectory reaches "
            "or crosses the boundary."
        ),

    "model":
        "Frozen Compact Temporal Transformer",

    "seeds":
        SEEDS,

    "ensemble":
        "arithmetic mean of frozen seed predictions",

    "threshold_selection":
        (
            "No performance-based warning-threshold tuning. "
            "Zero is fixed by boundary-crossing semantics."
        ),

    "development_data":
        (
            "BIRDSAI TrainReal track-disjoint validation only "
            "for Step18A diagnostic evaluation."
        ),

    "testreal_used":
        False,

    "wildbox_used":
        False,

    "calibrated_probability_claim":
        False,

}


# ============================================================
# SAVE PROTOCOL IMMEDIATELY
# BEFORE ANY PERFORMANCE IS COMPUTED
# ============================================================

protocol_json = json.dumps(

    PROTOCOL_CONFIG,

    indent=2,

    sort_keys=True

)


protocol_hash = hashlib.sha256(

    protocol_json.encode(
        "utf-8"
    )

).hexdigest()


with open(

    OUT /
    "STEP18_PROTOCOL_LOCKED.json",

    "w",

    encoding="utf-8"

) as f:

    f.write(
        protocol_json
    )


with open(

    OUT /
    "STEP18_PROTOCOL_HASH.txt",

    "w",

    encoding="utf-8"

) as f:

    f.write(
        protocol_hash
        +
        "\n"
    )


print(
    "\n============================================"
)

print(
    "STEP 18 PROTOCOL LOCKED"
)

print(
    "============================================"
)

print(
    "Protocol SHA256:"
)

print(
    protocol_hash
)


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
# FROZEN TRANSFORMER
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
# CHECKPOINT
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

            f"Missing checkpoint:\n"
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


    # deterministic positional encoding
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

            "Checkpoint architecture mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {result.unexpected_keys}"

        )


    model.eval()


    return model


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    histories,
    batch_size=256
):


    predictions = []


    with torch.no_grad():


        for start in range(

            0,
            len(histories),
            batch_size

        ):


            x = torch.tensor(

                histories[
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


        R = np.asarray(

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


        outputs.append(

            local
            +
            origin[i]

        )


    return np.asarray(

        outputs,

        dtype=np.float64

    )


# ============================================================
# FIRST CROSSING TIME
# ============================================================

def first_crossing_time(
    signed_margin
):


    indices = np.where(

        signed_margin
        >=
        0.0

    )[0]


    if len(
        indices
    ) == 0:

        return np.nan


    first_index = int(
        indices[0]
    )


    return (
        first_index + 1
    ) / FPS


# ============================================================
# CREATE GEOFENCE INSTANCES
# ============================================================

def build_geofence_instances(
    task,
    histories_abs,
    truth_abs,
    ensemble_prediction,
    seed_predictions,
    metadata
):


    rows = []


    for sample_index in range(
        len(
            histories_abs
        )
    ):


        current = histories_abs[
            sample_index,
            -1
        ]


        true_future = truth_abs[
            sample_index
        ]


        pred_future = ensemble_prediction[
            sample_index
        ]


        source_group = str(

            metadata.iloc[
                sample_index
            ][
                "source_group"
            ]

        )


        track_key = str(

            metadata.iloc[
                sample_index
            ][
                "track_key"
            ]

        )


        for axis_name, axis_index in AXES.items():


            current_coordinate = float(

                current[
                    axis_index
                ]

            )


            for boundary in GEOFENCE_THRESHOLDS:


                # ============================================
                # Exact tie is ambiguous.
                # Normally extremely rare.
                # ============================================

                if abs(

                    current_coordinate
                    -
                    boundary

                ) <= EPS:

                    continue


                # ============================================
                # DEFINE RISK DIRECTION
                #
                # Current < boundary:
                # risk is HIGH direction (>= boundary)
                #
                # Current > boundary:
                # risk is LOW direction (<= boundary)
                # ============================================

                if current_coordinate < boundary:

                    direction = "high"

                    sign = 1.0

                else:

                    direction = "low"

                    sign = -1.0


                boundary_id = (

                    f"{axis_name}_"
                    f"{boundary:.2f}_"
                    f"{direction}"

                )


                # ============================================
                # TRUE CROSSING
                # ============================================

                true_signed = (

                    sign

                    *
                    (
                        true_future[
                            :,
                            axis_index
                        ]

                        -
                        boundary
                    )

                )


                true_score = float(

                    np.max(
                        true_signed
                    )

                )


                true_crossing = (

                    true_score
                    >=
                    WARNING_THRESHOLD

                )


                true_tte = (

                    first_crossing_time(
                        true_signed
                    )

                    if true_crossing

                    else np.nan

                )


                # ============================================
                # ENSEMBLE PREDICTED CROSSING
                # ============================================

                predicted_signed = (

                    sign

                    *
                    (
                        pred_future[
                            :,
                            axis_index
                        ]

                        -
                        boundary
                    )

                )


                risk_score = float(

                    np.max(
                        predicted_signed
                    )

                )


                predicted_crossing = (

                    risk_score
                    >=
                    WARNING_THRESHOLD

                )


                predicted_tte = (

                    first_crossing_time(
                        predicted_signed
                    )

                    if predicted_crossing

                    else np.nan

                )


                # ============================================
                # SEED-LEVEL CROSSING / DISAGREEMENT
                # ============================================

                seed_risk_scores = []

                seed_crossings = []


                for seed_index in range(
                    seed_predictions.shape[
                        0
                    ]
                ):


                    seed_future = seed_predictions[
                        seed_index,
                        sample_index
                    ]


                    seed_signed = (

                        sign

                        *
                        (
                            seed_future[
                                :,
                                axis_index
                            ]

                            -
                            boundary
                        )

                    )


                    seed_score = float(

                        np.max(
                            seed_signed
                        )

                    )


                    seed_risk_scores.append(
                        seed_score
                    )


                    seed_crossings.append(

                        seed_score
                        >=
                        WARNING_THRESHOLD

                    )


                seed_risk_scores = np.asarray(

                    seed_risk_scores,

                    dtype=np.float64

                )


                vote_fraction = float(

                    np.mean(
                        seed_crossings
                    )

                )


                score_disagreement = float(

                    np.std(

                        seed_risk_scores,

                        ddof=0

                    )

                )


                # ============================================
                # CONFUSION TYPE
                # ============================================

                if (
                    true_crossing
                    and
                    predicted_crossing
                ):

                    outcome = "TP"

                elif (
                    not true_crossing
                    and
                    predicted_crossing
                ):

                    outcome = "FP"

                elif (
                    true_crossing
                    and
                    not predicted_crossing
                ):

                    outcome = "FN"

                else:

                    outcome = "TN"


                # ============================================
                # TTE ERROR
                # ============================================

                if (
                    true_crossing
                    and
                    predicted_crossing
                ):

                    tte_error = (

                        predicted_tte
                        -
                        true_tte

                    )


                    tte_abs_error = abs(
                        tte_error
                    )

                else:

                    tte_error = np.nan
                    tte_abs_error = np.nan


                rows.append({

                    "task":
                        task,

                    "sample_index":
                        sample_index,

                    "source_group":
                        source_group,

                    "track_key":
                        track_key,

                    "axis":
                        axis_name,

                    "boundary":
                        boundary,

                    "direction":
                        direction,

                    "boundary_id":
                        boundary_id,

                    "current_coordinate":
                        current_coordinate,

                    "distance_to_boundary":
                        abs(
                            current_coordinate
                            -
                            boundary
                        ),

                    "true_crossing":
                        int(
                            true_crossing
                        ),

                    "predicted_crossing":
                        int(
                            predicted_crossing
                        ),

                    "risk_score":
                        risk_score,

                    "true_score":
                        true_score,

                    "seed_vote_fraction":
                        vote_fraction,

                    "seed_score_disagreement":
                        score_disagreement,

                    "true_tte_sec":
                        true_tte,

                    "predicted_tte_sec":
                        predicted_tte,

                    "tte_error_sec":
                        tte_error,

                    "tte_abs_error_sec":
                        tte_abs_error,

                    "outcome":
                        outcome,

                })


    return pd.DataFrame(
        rows
    )


# ============================================================
# RANKING METRICS
# ============================================================

def auroc_rank(
    y_true,
    scores
):


    y_true = np.asarray(
        y_true,
        dtype=int
    )


    scores = np.asarray(
        scores,
        dtype=np.float64
    )


    positive = (
        y_true == 1
    )


    negative = (
        y_true == 0
    )


    n_positive = int(
        positive.sum()
    )


    n_negative = int(
        negative.sum()
    )


    if (
        n_positive == 0
        or
        n_negative == 0
    ):

        return np.nan


    ranks = (

        pd.Series(
            scores
        )

        .rank(
            method="average"
        )

        .to_numpy(
            dtype=np.float64
        )

    )


    positive_rank_sum = float(

        ranks[
            positive
        ].sum()

    )


    auc = (

        positive_rank_sum

        -
        (
            n_positive
            *
            (
                n_positive
                +
                1
            )
            /
            2.0
        )

    ) / (

        n_positive
        *
        n_negative

    )


    return float(
        auc
    )


def average_precision(
    y_true,
    scores
):


    y_true = np.asarray(
        y_true,
        dtype=int
    )


    scores = np.asarray(
        scores,
        dtype=np.float64
    )


    positives = int(
        y_true.sum()
    )


    if positives == 0:

        return np.nan


    order = np.argsort(
        -scores,
        kind="mergesort"
    )


    sorted_y = y_true[
        order
    ]


    cumulative_tp = np.cumsum(
        sorted_y
    )


    ranks = np.arange(

        1,
        len(
            sorted_y
        ) + 1

    )


    precision = (

        cumulative_tp
        /
        ranks

    )


    ap = (

        precision[
            sorted_y == 1
        ].sum()

        /
        positives

    )


    return float(
        ap
    )


# ============================================================
# BINARY METRICS
# ============================================================

def summarize_binary(
    table
):


    TP = int(
        (
            table[
                "outcome"
            ]
            ==
            "TP"
        ).sum()
    )


    FP = int(
        (
            table[
                "outcome"
            ]
            ==
            "FP"
        ).sum()
    )


    FN = int(
        (
            table[
                "outcome"
            ]
            ==
            "FN"
        ).sum()
    )


    TN = int(
        (
            table[
                "outcome"
            ]
            ==
            "TN"
        ).sum()
    )


    positive = (
        TP
        +
        FN
    )


    negative = (
        TN
        +
        FP
    )


    precision = (

        TP
        /
        (
            TP
            +
            FP
        )

        if (
            TP
            +
            FP
        ) > 0

        else np.nan

    )


    recall = (

        TP
        /
        positive

        if positive > 0

        else np.nan

    )


    specificity = (

        TN
        /
        negative

        if negative > 0

        else np.nan

    )


    false_warning_rate = (

        FP
        /
        negative

        if negative > 0

        else np.nan

    )


    miss_rate = (

        FN
        /
        positive

        if positive > 0

        else np.nan

    )


    if (
        np.isfinite(
            precision
        )
        and
        np.isfinite(
            recall
        )
        and
        (
            precision
            +
            recall
        ) > 0
    ):

        f1 = (

            2
            *
            precision
            *
            recall

            /
            (
                precision
                +
                recall
            )

        )

    else:

        f1 = np.nan


    positive_rate = (

        positive
        /
        len(
            table
        )

        if len(
            table
        ) > 0

        else np.nan

    )


    tte_table = table[

        (
            table[
                "true_crossing"
            ]
            ==
            1
        )

        &

        (
            table[
                "predicted_crossing"
            ]
            ==
            1
        )

    ]


    tte_mae = (

        float(

            tte_table[
                "tte_abs_error_sec"
            ].mean()

        )

        if len(
            tte_table
        ) > 0

        else np.nan

    )


    tte_bias = (

        float(

            tte_table[
                "tte_error_sec"
            ].mean()

        )

        if len(
            tte_table
        ) > 0

        else np.nan

    )


    # Warning is emitted at forecast time t=0.
    # Therefore actual warning lead time for a TP
    # is the true time until crossing.

    mean_warning_lead = (

        float(

            tte_table[
                "true_tte_sec"
            ].mean()

        )

        if len(
            tte_table
        ) > 0

        else np.nan

    )


    median_warning_lead = (

        float(

            tte_table[
                "true_tte_sec"
            ].median()

        )

        if len(
            tte_table
        ) > 0

        else np.nan

    )


    return {

        "instances":
            len(
                table
            ),

        "positive":
            positive,

        "negative":
            negative,

        "positive_rate":
            positive_rate,

        "TP":
            TP,

        "FP":
            FP,

        "FN":
            FN,

        "TN":
            TN,

        "precision":
            precision,

        "recall":
            recall,

        "F1":
            f1,

        "specificity":
            specificity,

        "false_warning_rate":
            false_warning_rate,

        "miss_rate":
            miss_rate,

        "AUROC":
            auroc_rank(

                table[
                    "true_crossing"
                ],

                table[
                    "risk_score"
                ]

            ),

        "AUPRC":
            average_precision(

                table[
                    "true_crossing"
                ],

                table[
                    "risk_score"
                ]

            ),

        "TTE_MAE_sec":
            tte_mae,

        "TTE_bias_sec":
            tte_bias,

        "mean_warning_lead_sec":
            mean_warning_lead,

        "median_warning_lead_sec":
            median_warning_lead,

    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

all_instances = []

overall_rows = []

boundary_rows = []

source_rows = []

audit_rows = []


for task in TASKS:


    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "STEP18A:",
        task
    )

    print(
        "=" * 80
    )


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


    if not npz_file.exists():

        raise FileNotFoundError(
            npz_file
        )


    if not metadata_file.exists():

        raise FileNotFoundError(
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


    validation_indices = arrays[
        "validation_indices"
    ].astype(
        int
    )


    # ========================================================
    # VALIDATION ONLY
    # ========================================================

    history_can = arrays[
        "history_canonical"
    ][
        validation_indices
    ].astype(
        np.float32
    )


    history_abs = arrays[
        "history_abs"
    ][
        validation_indices
    ].astype(
        np.float64
    )


    future_abs = arrays[
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


    future_length = future_abs.shape[
        1
    ]


    # ========================================================
    # FROZEN ENSEMBLE
    # ========================================================

    seed_predictions = []


    for seed in SEEDS:


        model = load_transformer(

            task,

            seed,

            future_length

        )


        prediction_can = predict(

            model,

            history_can

        )


        prediction_abs = inverse_transform_batch(

            prediction_can,

            origin,

            angle,

            scale

        )


        seed_predictions.append(
            prediction_abs
        )


    seed_predictions = np.stack(

        seed_predictions,

        axis=0

    )


    ensemble_prediction = (

        seed_predictions.mean(
            axis=0
        )

    )


    # ========================================================
    # CREATE GEOFENCE INSTANCES
    # ========================================================

    instances = build_geofence_instances(

        task=
            task,

        histories_abs=
            history_abs,

        truth_abs=
            future_abs,

        ensemble_prediction=
            ensemble_prediction,

        seed_predictions=
            seed_predictions,

        metadata=
            val_metadata,

    )


    all_instances.append(
        instances
    )


    expected_instances = (

        len(
            val_metadata
        )

        *
        len(
            AXES
        )

        *
        len(
            GEOFENCE_THRESHOLDS
        )

    )


    audit_rows.append({

        "task":
            task,

        "validation_samples":
            len(
                val_metadata
            ),

        "expected_instances":
            expected_instances,

        "actual_instances":
            len(
                instances
            ),

        "exact_boundary_ties_excluded":
            (
                expected_instances
                -
                len(
                    instances
                )
            ),

        "sources":
            val_metadata[
                "source_group"
            ].nunique(),

        "tracks":
            (
                val_metadata[
                    [
                        "source_group",
                        "track_key"
                    ]
                ]

                .drop_duplicates()

                .shape[
                    0
                ]
            ),

    })


    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    overall = summarize_binary(
        instances
    )


    overall[
        "task"
    ] = task


    overall[
        "warning_threshold"
    ] = WARNING_THRESHOLD


    overall_rows.append(
        overall
    )


    # ========================================================
    # BY BOUNDARY
    # ========================================================

    for boundary_id, group in instances.groupby(

        "boundary_id"

    ):


        summary = summarize_binary(
            group
        )


        summary[
            "task"
        ] = task


        summary[
            "boundary_id"
        ] = boundary_id


        boundary_rows.append(
            summary
        )


    # ========================================================
    # BY SOURCE
    # ========================================================

    for source_group, group in instances.groupby(

        "source_group"

    ):


        summary = summarize_binary(
            group
        )


        summary[
            "task"
        ] = task


        summary[
            "source_group"
        ] = source_group


        source_rows.append(
            summary
        )


# ============================================================
# COMBINE
# ============================================================

instances_df = pd.concat(

    all_instances,

    ignore_index=True

)


overall_df = pd.DataFrame(
    overall_rows
)


boundary_df = pd.DataFrame(
    boundary_rows
)


source_df = pd.DataFrame(
    source_rows
)


audit_df = pd.DataFrame(
    audit_rows
)


# ============================================================
# SAVE
# ============================================================

instances_df.to_csv(

    OUT /
    "step18a_validation_geofence_instances.csv",

    index=False

)


overall_df.to_csv(

    OUT /
    "step18a_validation_summary.csv",

    index=False

)


boundary_df.to_csv(

    OUT /
    "step18a_validation_by_boundary.csv",

    index=False

)


source_df.to_csv(

    OUT /
    "step18a_validation_by_source.csv",

    index=False

)


audit_df.to_csv(

    OUT /
    "step18a_validation_audit.csv",

    index=False

)


# ============================================================
# HUMAN-READABLE LOCKED PROTOCOL
# ============================================================

with open(

    OUT /
    "STEP18_PROTOCOL_LOCKED.txt",

    "w",

    encoding="utf-8"

) as f:


    f.write(
        "STEP 18 — VIRTUAL-GEOFENCE EARLY-WARNING PROTOCOL\n"
    )

    f.write(
        "=" * 82
        +
        "\n\n"
    )


    f.write(
        "STATUS\n"
    )

    f.write(
        "- Protocol frozen before TestReal / fresh WildBox "
        "geofence evaluation.\n"
    )

    f.write(
        "- No downstream classifier is trained.\n"
    )

    f.write(
        "- No warning threshold is optimized using F1, "
        "AUROC, AUPRC or test performance.\n"
    )

    f.write(
        "- Warning threshold = 0 signed normalized-coordinate "
        "units, corresponding to predicted boundary crossing.\n"
    )

    f.write(
        "- Step18A uses only fixed TrainReal track-disjoint "
        "validation for diagnostic evaluation.\n"
    )

    f.write(
        "- BIRDSAI TestReal is not loaded in Step18A.\n"
    )

    f.write(
        "- WildBox is not loaded in Step18A.\n\n"
    )


    f.write(
        "MODEL\n"
    )

    f.write(
        "- Frozen Compact Temporal Transformer.\n"
    )

    f.write(
        "- Frozen seeds: 42, 52, 62.\n"
    )

    f.write(
        "- Forecast = arithmetic mean of the three seed "
        "trajectories.\n\n"
    )


    f.write(
        "VIRTUAL GEOFENCES\n"
    )

    f.write(
        "- Axes: x and y.\n"
    )

    f.write(
        "- Fixed normalized boundaries: "
        "0.25, 0.50, 0.75.\n"
    )

    f.write(
        "- For each axis-boundary pair, the last observed "
        "position determines the current safe side.\n"
    )

    f.write(
        "- The opposite side is treated as the virtual "
        "risk zone.\n"
    )

    f.write(
        "- Therefore each trajectory normally produces "
        "six eligible directional boundary instances.\n"
    )

    f.write(
        "- These are experimental virtual boundaries; they "
        "are NOT labeled crop/village conflict zones.\n\n"
    )


    f.write(
        "EVENT DEFINITION\n"
    )

    f.write(
        "- True crossing: ground-truth future reaches/crosses "
        "the boundary into the risk side.\n"
    )

    f.write(
        "- Predicted warning: ensemble-mean future "
        "reaches/crosses the same boundary.\n"
    )

    f.write(
        "- Continuous risk score: maximum directional signed "
        "boundary penetration.\n"
    )

    f.write(
        "- Decision threshold: risk_score >= 0.\n"
    )

    f.write(
        "- TTE: first future crossing timestep.\n"
    )

    f.write(
        "- Warning lead time for a true positive is actual "
        "TTE because the warning is generated at forecast "
        "time zero.\n\n"
    )


    f.write(
        "METRICS\n"
    )

    f.write(
        "- Precision, recall, F1.\n"
    )

    f.write(
        "- False-warning rate and miss rate.\n"
    )

    f.write(
        "- AUROC and AUPRC using continuous risk score.\n"
    )

    f.write(
        "- TTE MAE and TTE bias among correctly warned "
        "crossings.\n"
    )

    f.write(
        "- Warning lead time.\n"
    )

    f.write(
        "- Boundary-level and source-level metrics.\n"
    )

    f.write(
        "- Seed vote/disagreement retained as diagnostic "
        "information only.\n\n"
    )


    f.write(
        "LIMITATION\n"
    )

    f.write(
        "- Geofence positives are rare and source-concentrated.\n"
    )

    f.write(
        "- This experiment is a secondary downstream "
        "early-warning benchmark, not evidence of actual "
        "human-elephant conflict prediction.\n"
    )

    f.write(
        "- Coordinates are normalized image-plane units, "
        "not metres.\n"
    )

    f.write(
        "- Seed vote fraction is not interpreted as a "
        "calibrated probability.\n\n"
    )


    f.write(
        "PROTOCOL SHA256\n"
    )

    f.write(
        protocol_hash
        +
        "\n\n"
    )


    f.write(
        "VALIDATION AUDIT\n"
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
        "\n\nVALIDATION SUMMARY\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        overall_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nBOUNDARY SUMMARY\n"
    )

    f.write(
        "-" * 82
        +
        "\n"
    )

    f.write(
        boundary_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nSOURCE SUMMARY\n"
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


# ============================================================
# CONSOLE
# ============================================================

print(
    "\n============================================"
)

print(
    "STEP 18A COMPLETE"
)

print(
    "============================================"
)


print(
    "\nValidation audit:"
)

print(
    audit_df.to_string(
        index=False
    )
)


print(
    "\nValidation geofence results:"
)

print(
    overall_df.to_string(
        index=False
    )
)


print(
    "\nProtocol hash:"
)

print(
    protocol_hash
)


print(
    "\nSaved to:"
)

print(
    OUT
)