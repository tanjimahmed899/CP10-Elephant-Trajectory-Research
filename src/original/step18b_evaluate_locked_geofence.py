from pathlib import Path
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

FRESH_ROOT = Path(
    r"C:\Users\HP\Downloads\wildbox_fresh_confirmatory_results"
)

MODEL_ROOT = Path(
    r"C:\Users\HP\Downloads\transformer_results"
)

PROTOCOL_FILE = Path(
    r"C:\Users\HP\Downloads\step18_geofence"
    r"\STEP18_PROTOCOL_LOCKED.json"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step18_geofence"
    r"\step18b_evaluation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXACT LOCKED HASH FROM STEP 18A
# ============================================================

EXPECTED_PROTOCOL_HASH = (
    "23c47c8204c09dbc96bbb06bd2f9038929e9405fb5a64be95fce491f3155fa65"
)


# ============================================================
# LOAD + VERIFY LOCKED PROTOCOL
# ============================================================

if not PROTOCOL_FILE.exists():

    raise FileNotFoundError(
        f"Locked protocol missing:\n{PROTOCOL_FILE}"
    )


raw_protocol = PROTOCOL_FILE.read_text(
    encoding="utf-8"
)


computed_hash = hashlib.sha256(
    raw_protocol.encode("utf-8")
).hexdigest()


if computed_hash != EXPECTED_PROTOCOL_HASH:

    raise RuntimeError(
        "\nLOCKED PROTOCOL HASH MISMATCH.\n"
        f"Expected: {EXPECTED_PROTOCOL_HASH}\n"
        f"Found:    {computed_hash}\n"
        "STOPPING. Do not evaluate with a modified protocol."
    )


protocol = json.loads(
    raw_protocol
)


# ============================================================
# VERIFY CRITICAL SETTINGS
# ============================================================

assert protocol["version"] == "STEP18_VIRTUAL_GEOFENCE_V1"

assert protocol["warning_threshold"] == 0.0

assert protocol["boundary_thresholds"] == [
    0.25,
    0.5,
    0.75
]

assert protocol["axes"] == [
    "x",
    "y"
]

assert protocol["seeds"] == [
    42,
    52,
    62
]


print("\nLOCKED PROTOCOL VERIFIED")
print("SHA256:", computed_hash)


# ============================================================
# SETTINGS FROM LOCKED PROTOCOL
# ============================================================

DEVICE = torch.device("cpu")

SEEDS = protocol["seeds"]

GEOFENCE_THRESHOLDS = (
    protocol[
        "boundary_thresholds"
    ]
)

WARNING_THRESHOLD = float(
    protocol[
        "warning_threshold"
    ]
)

AXES = {
    "x": 0,
    "y": 1,
}

FPS = float(
    protocol[
        "sampling_hz"
    ]
)

EPS = 1e-10


TASKS = {

    "2s_to_2s": {
        "future_frames": 20,
    },

    "2s_to_5s": {
        "future_frames": 50,
    },

}


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
        ).unsqueeze(1)


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
            position * div_term
        )


        pe[:, 1::2] = torch.cos(
            position * div_term
        )


        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
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


        self.future_length = future_length


        self.input_projection = nn.Linear(
            2,
            d_model
        )


        self.position = PositionalEncoding(
            d_model,
            max_len=32
        )


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=dim_feedforward,

            dropout=dropout,

            activation="gelu",

            batch_first=True,

            norm_first=True

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
            self.head(x)
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
            checkpoint_file
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

        for key
        in result.missing_keys

        if key != "position.pe"

    ]


    if (
        missing
        or
        result.unexpected_keys
    ):

        raise RuntimeError(

            "Checkpoint mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: "
            f"{result.unexpected_keys}"

        )


    model.eval()

    return model


# ============================================================
# PREDICTION
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


            y = model(
                x
            )


            predictions.append(
                y.cpu().numpy()
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
# FIRST CROSSING TIME
# ============================================================

def first_crossing_time(
    signed_margin
):

    indices = np.where(
        signed_margin >= 0.0
    )[0]


    if len(indices) == 0:

        return np.nan


    return (
        int(indices[0]) + 1
    ) / FPS


# ============================================================
# BUILD LOCKED GEOFENCE INSTANCES
# ============================================================

def build_geofence_instances(
    dataset_name,
    task,
    histories_abs,
    truth_abs,
    ensemble_prediction,
    seed_predictions,
    metadata
):

    rows = []


    for sample_index in range(
        len(histories_abs)
    ):

        current = histories_abs[
            sample_index,
            -1
        ]


        true_future = truth_abs[
            sample_index
        ]


        predicted_future = ensemble_prediction[
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

                if abs(
                    current_coordinate
                    -
                    boundary
                ) <= EPS:

                    continue


                # --------------------------------------------
                # LOCKED SAFE/RISK SIDE RULE
                # --------------------------------------------

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


                # --------------------------------------------
                # TRUE CROSSING
                # --------------------------------------------

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


                # --------------------------------------------
                # ENSEMBLE WARNING
                # --------------------------------------------

                predicted_signed = (

                    sign

                    *
                    (
                        predicted_future[
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


                # --------------------------------------------
                # FROZEN SEED DIAGNOSTICS
                # --------------------------------------------

                seed_scores = []
                seed_crossings = []


                for seed_index in range(
                    seed_predictions.shape[0]
                ):

                    seed_future = (
                        seed_predictions[
                            seed_index,
                            sample_index
                        ]
                    )


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


                    seed_scores.append(
                        seed_score
                    )


                    seed_crossings.append(

                        seed_score
                        >=
                        WARNING_THRESHOLD

                    )


                seed_scores = np.asarray(
                    seed_scores,
                    dtype=np.float64
                )


                vote_fraction = float(
                    np.mean(
                        seed_crossings
                    )
                )


                score_disagreement = float(
                    np.std(
                        seed_scores,
                        ddof=0
                    )
                )


                # --------------------------------------------
                # CONFUSION TYPE
                # --------------------------------------------

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


                # --------------------------------------------
                # TTE
                # --------------------------------------------

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

                    "dataset":
                        dataset_name,

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
# AUROC
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


# ============================================================
# AUPRC / AVERAGE PRECISION
#
# Exact same implementation as Step18A.
# ============================================================

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
        len(sorted_y) + 1
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
# BINARY SUMMARY
# ============================================================

def summarize_binary(
    table
):

    TP = int(
        (
            table["outcome"]
            ==
            "TP"
        ).sum()
    )


    FP = int(
        (
            table["outcome"]
            ==
            "FP"
        ).sum()
    )


    FN = int(
        (
            table["outcome"]
            ==
            "FN"
        ).sum()
    )


    TN = int(
        (
            table["outcome"]
            ==
            "TN"
        ).sum()
    )


    positive = TP + FN
    negative = TN + FP


    precision = (

        TP / (TP + FP)

        if (TP + FP) > 0

        else np.nan

    )


    recall = (

        TP / positive

        if positive > 0

        else np.nan

    )


    specificity = (

        TN / negative

        if negative > 0

        else np.nan

    )


    false_warning_rate = (

        FP / negative

        if negative > 0

        else np.nan

    )


    miss_rate = (

        FN / positive

        if positive > 0

        else np.nan

    )


    if (
        np.isfinite(precision)
        and
        np.isfinite(recall)
        and
        (
            precision
            +
            recall
        ) > 0
    ):

        F1 = (

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

        F1 = np.nan


    positive_rate = (

        positive
        /
        len(table)

        if len(table) > 0

        else np.nan

    )


    correctly_warned = table[

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


    if len(
        correctly_warned
    ) > 0:

        TTE_MAE = float(
            correctly_warned[
                "tte_abs_error_sec"
            ].mean()
        )


        TTE_bias = float(
            correctly_warned[
                "tte_error_sec"
            ].mean()
        )


        mean_lead = float(
            correctly_warned[
                "true_tte_sec"
            ].mean()
        )


        median_lead = float(
            correctly_warned[
                "true_tte_sec"
            ].median()
        )

    else:

        TTE_MAE = np.nan
        TTE_bias = np.nan
        mean_lead = np.nan
        median_lead = np.nan


    return {

        "instances":
            len(table),

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
            F1,

        "specificity":
            specificity,

        "false_warning_rate":
            false_warning_rate,

        "miss_rate":
            miss_rate,

        "AUROC":
            auroc_rank(
                table["true_crossing"],
                table["risk_score"]
            ),

        "AUPRC":
            average_precision(
                table["true_crossing"],
                table["risk_score"]
            ),

        "TTE_MAE_sec":
            TTE_MAE,

        "TTE_bias_sec":
            TTE_bias,

        "mean_warning_lead_sec":
            mean_lead,

        "median_warning_lead_sec":
            median_lead,

    }


# ============================================================
# LOAD BIRDSAI TESTREAL
# ============================================================

def load_testreal(
    task
):

    npz_file = (
        DATA_ROOT
        /
        f"{task}_birdsai_test_canonical.npz"
    )


    metadata_file = (
        DATA_ROOT
        /
        f"{task}_birdsai_test_metadata.csv"
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


    return (
        arrays,
        metadata
    )


# ============================================================
# LOAD FRESH WILDBOX
# ============================================================

def load_fresh_wildbox(
    task
):

    npz_file = (
        FRESH_ROOT
        /
        f"{task}_fresh_dataset.npz"
    )


    metadata_file = (
        FRESH_ROOT
        /
        f"{task}_fresh_metadata.csv"
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


    return (
        arrays,
        metadata
    )


# ============================================================
# EVALUATE ONE DATASET/TASK
# ============================================================

def evaluate_dataset(
    dataset_name,
    dataset_status,
    task,
    arrays,
    metadata
):

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


    n_samples = len(
        metadata
    )


    if len(
        history_can
    ) != n_samples:

        raise RuntimeError(
            "Metadata / trajectory sample mismatch."
        )


    future_length = truth_abs.shape[
        1
    ]


    if future_length != TASKS[
        task
    ][
        "future_frames"
    ]:

        raise RuntimeError(
            "Unexpected forecast length."
        )


    # ========================================================
    # FROZEN SEEDS
    # ========================================================

    seed_predictions = []


    for seed in SEEDS:

        model = load_transformer(

            task,

            seed,

            future_length

        )


        pred_can = predict(

            model,

            history_can

        )


        pred_abs = inverse_transform_batch(

            pred_can,

            origin,

            angle,

            scale

        )


        seed_predictions.append(
            pred_abs
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
    # LOCKED GEOFENCE
    # ========================================================

    instances = build_geofence_instances(

        dataset_name=
            dataset_name,

        task=
            task,

        histories_abs=
            history_abs,

        truth_abs=
            truth_abs,

        ensemble_prediction=
            ensemble_prediction,

        seed_predictions=
            seed_predictions,

        metadata=
            metadata,

    )


    expected_instances = (

        n_samples

        *
        len(AXES)

        *
        len(
            GEOFENCE_THRESHOLDS
        )

    )


    audit = {

        "dataset":
            dataset_name,

        "dataset_status":
            dataset_status,

        "task":
            task,

        "samples":
            n_samples,

        "tracks":
            (
                metadata[
                    [
                        "source_group",
                        "track_key"
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        "sources":
            metadata[
                "source_group"
            ].nunique(),

        "expected_instances":
            expected_instances,

        "actual_instances":
            len(instances),

        "boundary_ties_excluded":
            (
                expected_instances
                -
                len(instances)
            ),

        "protocol_hash":
            computed_hash,

    }


    overall = summarize_binary(
        instances
    )


    overall[
        "dataset"
    ] = dataset_name


    overall[
        "dataset_status"
    ] = dataset_status


    overall[
        "task"
    ] = task


    overall[
        "warning_threshold"
    ] = WARNING_THRESHOLD


    return (
        instances,
        audit,
        overall
    )


# ============================================================
# MAIN
# ============================================================

all_instances = []

audit_rows = []

overall_rows = []

boundary_rows = []

source_rows = []

vote_rows = []


DATASETS = [

    (
        "BIRDSAI_TestReal",
        (
            "frozen in-domain benchmark; "
            "previously inspected during development"
        ),
        load_testreal
    ),

    (
        "WildBox_Fresh",
        (
            "preregistered unseen source-video "
            "external holdout"
        ),
        load_fresh_wildbox
    ),

]


for (
    dataset_name,
    dataset_status,
    loader

) in DATASETS:


    for task in TASKS:


        print(
            "\n"
            +
            "=" * 82
        )

        print(
            dataset_name,
            "|",
            task
        )

        print(
            "=" * 82
        )


        arrays, metadata = loader(
            task
        )


        (
            instances,
            audit,
            overall

        ) = evaluate_dataset(

            dataset_name=
                dataset_name,

            dataset_status=
                dataset_status,

            task=
                task,

            arrays=
                arrays,

            metadata=
                metadata,

        )


        all_instances.append(
            instances
        )


        audit_rows.append(
            audit
        )


        overall_rows.append(
            overall
        )


        # ====================================================
        # BOUNDARY LEVEL
        # ====================================================

        for boundary_id, group in instances.groupby(
            "boundary_id"
        ):

            summary = summarize_binary(
                group
            )


            summary[
                "dataset"
            ] = dataset_name


            summary[
                "task"
            ] = task


            summary[
                "boundary_id"
            ] = boundary_id


            boundary_rows.append(
                summary
            )


        # ====================================================
        # SOURCE LEVEL
        # ====================================================

        for source_group, group in instances.groupby(
            "source_group"
        ):

            summary = summarize_binary(
                group
            )


            summary[
                "dataset"
            ] = dataset_name


            summary[
                "task"
            ] = task


            summary[
                "source_group"
            ] = source_group


            source_rows.append(
                summary
            )


        # ====================================================
        # SEED-VOTE DESCRIPTIVE DIAGNOSTIC
        #
        # No new decision rule.
        # ====================================================

        for vote, group in instances.groupby(
            "seed_vote_fraction"
        ):

            vote_rows.append({

                "dataset":
                    dataset_name,

                "task":
                    task,

                "seed_vote_fraction":
                    vote,

                "instances":
                    len(group),

                "positive":
                    int(
                        group[
                            "true_crossing"
                        ].sum()
                    ),

                "positive_rate":
                    float(
                        group[
                            "true_crossing"
                        ].mean()
                    ),

                "mean_risk_score":
                    float(
                        group[
                            "risk_score"
                        ].mean()
                    ),

                "mean_score_disagreement":
                    float(
                        group[
                            "seed_score_disagreement"
                        ].mean()
                    ),

            })


# ============================================================
# COMBINE
# ============================================================

instances_df = pd.concat(

    all_instances,

    ignore_index=True

)


audit_df = pd.DataFrame(
    audit_rows
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


vote_df = pd.DataFrame(
    vote_rows
)


# ============================================================
# SAVE
# ============================================================

instances_df.to_csv(

    OUT /
    "step18b_all_geofence_instances.csv",

    index=False

)


audit_df.to_csv(

    OUT /
    "step18b_data_audit.csv",

    index=False

)


overall_df.to_csv(

    OUT /
    "step18b_overall_metrics.csv",

    index=False

)


boundary_df.to_csv(

    OUT /
    "step18b_by_boundary.csv",

    index=False

)


source_df.to_csv(

    OUT /
    "step18b_by_source.csv",

    index=False

)


vote_df.to_csv(

    OUT /
    "step18b_seed_vote_diagnostic.csv",

    index=False

)


# ============================================================
# SUMMARY
# ============================================================

with open(

    OUT /
    "SUMMARY.txt",

    "w",

    encoding="utf-8"

) as f:


    f.write(
        "STEP 18B — LOCKED VIRTUAL-GEOFENCE EVALUATION\n"
    )

    f.write(
        "=" * 88
        +
        "\n\n"
    )


    f.write(
        "PROTOCOL STATUS\n"
    )

    f.write(
        f"- Protocol SHA256: {computed_hash}\n"
    )

    f.write(
        "- Exact Step18A protocol verified before evaluation.\n"
    )

    f.write(
        "- Warning threshold remained fixed at 0.\n"
    )

    f.write(
        "- No downstream classifier trained.\n"
    )

    f.write(
        "- No threshold optimization performed.\n"
    )

    f.write(
        "- Frozen Transformer seeds 42, 52, 62 only.\n\n"
    )


    f.write(
        "DATASET INTERPRETATION\n"
    )

    f.write(
        "- BIRDSAI TestReal: frozen in-domain benchmark; "
        "previously inspected during development and therefore "
        "not described as untouched/preregistered.\n"
    )

    f.write(
        "- Fresh WildBox: preregistered unseen source-video "
        "external holdout.\n\n"
    )


    f.write(
        "EVENT INTERPRETATION\n"
    )

    f.write(
        "- Events are crossings of experimental virtual "
        "image-plane boundaries.\n"
    )

    f.write(
        "- They are NOT labels of actual human-elephant "
        "conflict, crop intrusion or village entry.\n"
    )

    f.write(
        "- Coordinates are normalized image-plane units, "
        "not metres.\n\n"
    )


    f.write(
        "DATA AUDIT\n"
    )

    f.write(
        "-" * 88
        +
        "\n"
    )

    f.write(
        audit_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nOVERALL METRICS\n"
    )

    f.write(
        "-" * 88
        +
        "\n"
    )

    f.write(
        overall_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nBOUNDARY METRICS\n"
    )

    f.write(
        "-" * 88
        +
        "\n"
    )

    f.write(
        boundary_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nSOURCE METRICS\n"
    )

    f.write(
        "-" * 88
        +
        "\n"
    )

    f.write(
        source_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nSEED-VOTE DIAGNOSTIC\n"
    )

    f.write(
        "-" * 88
        +
        "\n"
    )

    f.write(
        vote_df.to_string(
            index=False
        )
    )


# ============================================================
# CONSOLE
# ============================================================

print(
    "\n"
    +
    "=" * 82
)

print(
    "STEP 18B COMPLETE"
)

print(
    "=" * 82
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
    "\nOverall locked geofence results:"
)

print(
    overall_df.to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)

print(
    OUT
)