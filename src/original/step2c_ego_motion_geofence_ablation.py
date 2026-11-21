from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# SOURCE PATHS
# Original project files are READ ONLY.
# New outputs go only to Downloads.
# ============================================================

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

TRAJ_DIR = ROOT / "trajectory_dataset"

CORR_DIR = ROOT / "corrected_trajectory_dataset"

OUT = Path(
    r"C:\Users\HP\Downloads\step2c_ego_motion_geofence_ablation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROTOCOL
# ============================================================

RAW_FPS = 30
TARGET_FPS = 10
DOWNSAMPLE = 3

OBS_RAW = 60

GEOFENCE_THRESHOLDS = [
    0.25,
    0.50,
    0.75
]

AXES = {
    "x": 0,
    "y": 1
}

WARNING_THRESHOLD = 0.0

EPS = 1e-10


TASKS = {

    "2s_to_2s": {
        "future_raw": 60,
        "future_10hz": 20,
        "expected_samples": 1053
    },

    "2s_to_5s": {
        "future_raw": 150,
        "future_10hz": 50,
        "expected_samples": 826
    }
}


# ============================================================
# CONSTANT VELOCITY
# ============================================================

def estimate_velocity(history, fps=10):

    history = np.asarray(
        history,
        dtype=np.float64
    )

    n = len(history)

    t = (
        np.arange(n, dtype=np.float64)
        / float(fps)
    )

    t_centered = (
        t - t.mean()
    )

    denominator = np.sum(
        t_centered ** 2
    )

    if denominator <= 1e-12:
        return np.zeros(
            2,
            dtype=np.float64
        )

    position_centered = (
        history
        -
        history.mean(
            axis=0,
            keepdims=True
        )
    )

    velocity = (

        (
            t_centered[:, None]
            *
            position_centered
        ).sum(axis=0)

        /

        denominator
    )

    return velocity


def predict_constant_velocity(
    history,
    future_length,
    fps=10
):

    velocity = estimate_velocity(
        history,
        fps
    )

    future_t = (

        np.arange(
            1,
            future_length + 1,
            dtype=np.float64
        )

        / float(fps)
    )

    prediction = (

        history[-1][None, :]

        +

        future_t[:, None]
        *
        velocity[None, :]
    )

    return prediction


# ============================================================
# STATIONARY
# ============================================================

def predict_stationary(
    history,
    future_length
):

    return np.repeat(
        history[-1:, :],
        future_length,
        axis=0
    )


def predict_model(
    model,
    history,
    future_length
):

    if model == "Stationary":

        return predict_stationary(
            history,
            future_length
        )

    if model == "ConstantVelocity":

        return predict_constant_velocity(
            history,
            future_length,
            TARGET_FPS
        )

    raise ValueError(
        model
    )


# ============================================================
# TRACK LOOKUP
# ============================================================

def build_track_lookup(cleaned):

    lookup = {}

    grouped = cleaned.groupby(
        [
            "split",
            "sequence",
            "segment_id"
        ],
        sort=False
    )

    for key, group in grouped:

        group = (
            group
            .sort_values("frame")
            .copy()
        )

        group["frame"] = (
            group["frame"]
            .astype(int)
        )

        lookup[
            tuple(
                str(x)
                for x in key
            )
        ] = (
            group
            .set_index("frame")
        )

    return lookup


# ============================================================
# RAW WINDOW
# ============================================================

def extract_raw_window(
    lookup,
    row,
    future_raw
):

    key = (
        str(row["split"]),
        str(row["sequence"]),
        str(row["segment_id"])
    )

    if key not in lookup:

        raise KeyError(
            f"Raw track not found: {key}"
        )

    track = lookup[key]

    start = int(
        row["obs_start_frame"]
    )

    future_end = int(
        row["future_end_frame"]
    )

    required_frames = list(
        range(
            start,
            future_end + 1
        )
    )

    missing = [
        f for f in required_frames
        if f not in track.index
    ]

    if missing:

        raise RuntimeError(
            f"Missing raw frame "
            f"{missing[0]} for {key}"
        )

    xy = (

        track.loc[
            required_frames,
            ["cx_n", "cy_n"]
        ]

        .to_numpy(
            dtype=np.float64
        )
    )

    expected = (
        OBS_RAW
        +
        future_raw
    )

    if len(xy) != expected:

        raise RuntimeError(
            f"Unexpected raw window length "
            f"{len(xy)} != {expected}"
        )

    history = (
        xy[:OBS_RAW]
        [::DOWNSAMPLE]
    )

    future = (
        xy[OBS_RAW:]
        [::DOWNSAMPLE]
    )

    return history, future


# ============================================================
# TTE
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
    ) / TARGET_FPS


# ============================================================
# AUROC
# ============================================================

def auroc_rank(
    labels,
    scores
):

    labels = np.asarray(
        labels,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    valid = (
        np.isfinite(scores)
    )

    labels = labels[valid]
    scores = scores[valid]

    n_pos = int(
        labels.sum()
    )

    n_neg = (
        len(labels)
        -
        n_pos
    )

    if (
        n_pos == 0
        or
        n_neg == 0
    ):
        return np.nan

    ranks = (
        pd.Series(scores)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    rank_sum_pos = (
        ranks[
            labels == 1
        ]
        .sum()
    )

    auc = (

        rank_sum_pos
        -
        n_pos
        *
        (n_pos + 1)
        / 2.0

    ) / (
        n_pos * n_neg
    )

    return float(auc)


# ============================================================
# AUPRC / AVERAGE PRECISION
# ============================================================

def average_precision(
    labels,
    scores
):

    labels = np.asarray(
        labels,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    valid = (
        np.isfinite(scores)
    )

    labels = labels[valid]
    scores = scores[valid]

    positives = int(
        labels.sum()
    )

    if positives == 0:
        return np.nan

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    y = labels[order]

    cumulative_tp = np.cumsum(
        y
    )

    precision = (
        cumulative_tp
        /
        np.arange(
            1,
            len(y) + 1
        )
    )

    ap = (
        precision[y == 1]
        .mean()
    )

    return float(ap)


# ============================================================
# WARNING INSTANCE
#
# IMPORTANT:
# Corrected trajectory defines:
# - current safe/risk side
# - true crossing event
#
# Both RAW and CORRECTED predictors are scored against
# exactly the same corrected reference event.
# ============================================================

def make_instance(
    task,
    model,
    representation,
    sample_id,
    sequence,
    track_key,
    corrected_current,
    raw_current,
    corrected_future,
    raw_future,
    predicted_future,
    axis_name,
    axis_index,
    boundary
):

    current_coordinate = float(
        corrected_current[
            axis_index
        ]
    )

    if abs(
        current_coordinate
        -
        boundary
    ) <= EPS:

        return None


    # ========================================================
    # LOCKED DIRECTION FROM CORRECTED REFERENCE FRAME
    # ========================================================

    if current_coordinate < boundary:

        direction = "high"
        sign = 1.0

    else:

        direction = "low"
        sign = -1.0


    # ========================================================
    # TRUE EVENT:
    # ALWAYS FROM CORRECTED FUTURE
    # ========================================================

    true_signed = (

        sign

        *

        (
            corrected_future[
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


    # ========================================================
    # PREDICTED WARNING
    # ========================================================

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


    # ========================================================
    # DIAGNOSTIC:
    # WHAT LABEL WOULD THE RAW IMAGE-PLANE TRAJECTORY CREATE?
    # ========================================================

    raw_current_coordinate = float(
        raw_current[
            axis_index
        ]
    )

    if raw_current_coordinate < boundary:

        raw_native_sign = 1.0
        raw_native_direction = "high"

    else:

        raw_native_sign = -1.0
        raw_native_direction = "low"


    raw_native_signed = (

        raw_native_sign

        *

        (
            raw_future[
                :,
                axis_index
            ]

            -

            boundary
        )
    )

    raw_native_true_crossing = (

        np.max(
            raw_native_signed
        )

        >=
        WARNING_THRESHOLD
    )


    current_side_disagreement = int(

        (
            corrected_current[
                axis_index
            ]
            <
            boundary
        )

        !=

        (
            raw_current[
                axis_index
            ]
            <
            boundary
        )
    )


    raw_truth_disagreement = int(

        bool(
            raw_native_true_crossing
        )

        !=

        bool(
            true_crossing
        )
    )


    # ========================================================
    # OUTCOME
    # ========================================================

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


    return {

        "task":
            task,

        "model":
            model,

        "representation":
            representation,

        "sample_id":
            sample_id,

        "sequence":
            sequence,

        "track_key":
            track_key,

        "axis":
            axis_name,

        "boundary":
            boundary,

        "direction":
            direction,

        "corrected_current":
            current_coordinate,

        "raw_current":
            raw_current_coordinate,

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

        "raw_native_direction":
            raw_native_direction,

        "raw_native_true_crossing":
            int(
                raw_native_true_crossing
            ),

        "raw_truth_label_disagreement":
            raw_truth_disagreement,

        "current_side_disagreement":
            current_side_disagreement
    }


# ============================================================
# SUMMARY METRICS
# ============================================================

def summarize(
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

    positive = (
        TP + FN
    )

    negative = (
        TN + FP
    )


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
        precision + recall > 0
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


    correctly_warned = table[

        (
            table["true_crossing"]
            ==
            1
        )

        &

        (
            table["predicted_crossing"]
            ==
            1
        )
    ]


    if len(correctly_warned) > 0:

        tte_mae = float(

            correctly_warned[
                "tte_abs_error_sec"
            ]
            .mean()
        )

        mean_lead = float(

            correctly_warned[
                "true_tte_sec"
            ]
            .mean()
        )

        median_lead = float(

            correctly_warned[
                "true_tte_sec"
            ]
            .median()
        )

    else:

        tte_mae = np.nan
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
            (
                positive
                /
                len(table)
            ),

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

        "mean_warning_lead_sec":
            mean_lead,

        "median_warning_lead_sec":
            median_lead,

        "raw_truth_label_disagreement_rate":
            float(

                table[
                    "raw_truth_label_disagreement"
                ]
                .mean()
            ),

        "current_side_disagreement_rate":
            float(

                table[
                    "current_side_disagreement"
                ]
                .mean()
            )
    }


# ============================================================
# IMPROVEMENT
# ============================================================

def improvement_pct(
    raw,
    corrected
):

    if (
        not np.isfinite(raw)
        or
        abs(raw) < 1e-12
    ):

        return np.nan

    return (

        100.0

        *
        (
            raw
            -
            corrected
        )

        /
        raw
    )


# ============================================================
# MAIN
# ============================================================

def main():


    # ========================================================
    # LOAD RAW TRACKS
    # ========================================================

    cleaned_file = (

        TRAJ_DIR /
        "cleaned_track_points.csv"
    )

    if not cleaned_file.exists():

        raise FileNotFoundError(
            cleaned_file
        )


    print(
        "Loading raw tracks..."
    )

    cleaned = pd.read_csv(

        cleaned_file,

        dtype={
            "split": str,
            "sequence": str,
            "segment_id": str
        }
    )


    lookup = build_track_lookup(
        cleaned
    )


    print(
        "Track lookup:",
        len(lookup)
    )


    instance_rows = []


    # ========================================================
    # TASK LOOP
    # ========================================================

    for task, config in TASKS.items():


        print(
            "\n"
            +
            "=" * 70
        )

        print(
            "TASK:",
            task
        )

        print(
            "=" * 70
        )


        metadata_file = (

            CORR_DIR /
            f"corrected_windows_{task}.csv"
        )

        npz_file = (

            CORR_DIR /
            f"corrected_{task}.npz"
        )


        metadata = pd.read_csv(

            metadata_file,

            dtype={
                "split": str,
                "sequence": str,
                "segment_id": str
            }
        )


        arrays = np.load(
            npz_file
        )


        corrected_history_all = (
            arrays["history"]
        )

        corrected_future_all = (
            arrays["future"]
        )


        core = metadata[

            (
                metadata["accepted"]
                ==
                1
            )

            &

            (
                metadata[
                    "quality_tier"
                ]
                .astype(str)
                .str.lower()
                ==
                "core"
            )

            &

            (
                metadata["split"]
                ==
                "test_real"
            )

        ].copy()


        core = (

            core
            .sort_values(
                "sample_id"
            )
            .reset_index(
                drop=True
            )
        )


        print(
            "Core TestReal samples:",
            len(core)
        )


        if (
            len(core)
            !=
            config[
                "expected_samples"
            ]
        ):

            print(
                "WARNING expected:",
                config[
                    "expected_samples"
                ],
                "found:",
                len(core)
            )


        # ====================================================
        # SAMPLE LOOP
        # ====================================================

        for _, row in core.iterrows():


            sample_id = int(
                row["sample_id"]
            )


            # -----------------------------------------------
            # RAW 10-Hz
            # -----------------------------------------------

            (
                raw_history,
                raw_future
            ) = extract_raw_window(

                lookup,

                row,

                config[
                    "future_raw"
                ]
            )


            # -----------------------------------------------
            # CORRECTED 10-Hz
            # -----------------------------------------------

            corrected_history = (

                np.asarray(

                    corrected_history_all[
                        sample_id,
                        :,
                        :2
                    ],

                    dtype=np.float64
                )

                [::DOWNSAMPLE]
            )


            corrected_future = (

                np.asarray(

                    corrected_future_all[
                        sample_id,
                        :,
                        :2
                    ],

                    dtype=np.float64
                )

                [::DOWNSAMPLE]
            )


            if corrected_history.shape != (
                20,
                2
            ):

                raise RuntimeError(
                    corrected_history.shape
                )


            if corrected_future.shape != (

                config[
                    "future_10hz"
                ],

                2
            ):

                raise RuntimeError(
                    corrected_future.shape
                )


            # -----------------------------------------------
            # PREDICT BOTH REPRESENTATIONS
            # -----------------------------------------------

            for model in [
                "Stationary",
                "ConstantVelocity"
            ]:


                raw_prediction = predict_model(

                    model,

                    raw_history,

                    len(raw_future)
                )


                corrected_prediction = predict_model(

                    model,

                    corrected_history,

                    len(corrected_future)
                )


                # -------------------------------------------
                # SAME REFERENCE EVENT FOR BOTH
                # -------------------------------------------

                for (
                    axis_name,
                    axis_index
                ) in AXES.items():


                    for boundary in (
                        GEOFENCE_THRESHOLDS
                    ):


                        for (
                            representation,
                            prediction
                        ) in [

                            (
                                "RAW_UNCORRECTED",
                                raw_prediction
                            ),

                            (
                                "EGO_CORRECTED",
                                corrected_prediction
                            )
                        ]:


                            result = make_instance(

                                task=
                                    task,

                                model=
                                    model,

                                representation=
                                    representation,

                                sample_id=
                                    sample_id,

                                sequence=
                                    str(
                                        row[
                                            "sequence"
                                        ]
                                    ),

                                track_key=
                                    str(
                                        row[
                                            "segment_id"
                                        ]
                                    ),

                                corrected_current=
                                    corrected_history[-1],

                                raw_current=
                                    raw_history[-1],

                                corrected_future=
                                    corrected_future,

                                raw_future=
                                    raw_future,

                                predicted_future=
                                    prediction,

                                axis_name=
                                    axis_name,

                                axis_index=
                                    axis_index,

                                boundary=
                                    boundary
                            )


                            if result is not None:

                                instance_rows.append(
                                    result
                                )


    # ========================================================
    # SAVE INSTANCE TABLE
    # ========================================================

    instances = pd.DataFrame(
        instance_rows
    )


    instances.to_csv(

        OUT /
        "raw_vs_corrected_geofence_instances.csv",

        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []


    grouped = instances.groupby(

        [
            "task",
            "model",
            "representation"
        ],

        sort=False
    )


    for (
        task,
        model,
        representation
    ), group in grouped:


        row = summarize(
            group
        )


        row[
            "task"
        ] = task

        row[
            "model"
        ] = model

        row[
            "representation"
        ] = representation

        row[
            "samples"
        ] = group[
            "sample_id"
        ].nunique()

        row[
            "tracks"
        ] = group[
            "track_key"
        ].nunique()

        row[
            "sequences"
        ] = group[
            "sequence"
        ].nunique()


        summary_rows.append(
            row
        )


    summary = pd.DataFrame(
        summary_rows
    )


    summary.to_csv(

        OUT /
        "raw_vs_corrected_geofence_summary.csv",

        index=False
    )


    # ========================================================
    # RAW LABEL DISTORTION AUDIT
    #
    # This is model-independent.
    # Take one copy only.
    # ========================================================

    diagnostic = (

        instances[

            instances[
                "representation"
            ]
            ==
            "EGO_CORRECTED"

        ]

        .drop_duplicates(

            subset=[

                "task",
                "sample_id",
                "axis",
                "boundary"

            ]
        )
    )


    diagnostic_rows = []


    for task, group in diagnostic.groupby(
        "task"
    ):


        diagnostic_rows.append({

            "task":
                task,

            "instances":
                len(group),

            "reference_positive":
                int(
                    group[
                        "true_crossing"
                    ]
                    .sum()
                ),

            "raw_native_positive":
                int(
                    group[
                        "raw_native_true_crossing"
                    ]
                    .sum()
                ),

            "raw_vs_corrected_truth_disagreement_count":
                int(
                    group[
                        "raw_truth_label_disagreement"
                    ]
                    .sum()
                ),

            "raw_vs_corrected_truth_disagreement_pct":
                100.0
                *
                float(
                    group[
                        "raw_truth_label_disagreement"
                    ]
                    .mean()
                ),

            "current_safe_side_disagreement_count":
                int(
                    group[
                        "current_side_disagreement"
                    ]
                    .sum()
                ),

            "current_safe_side_disagreement_pct":
                100.0
                *
                float(
                    group[
                        "current_side_disagreement"
                    ]
                    .mean()
                )
        })


    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )


    diagnostic_df.to_csv(

        OUT /
        "raw_imageplane_label_distortion_audit.csv",

        index=False
    )


    # ========================================================
    # CORRECTION IMPACT COMPARISON
    # ========================================================

    comparison_rows = []


    for (
        task,
        model
    ), group in summary.groupby(
        [
            "task",
            "model"
        ]
    ):


        raw = group[
            group[
                "representation"
            ]
            ==
            "RAW_UNCORRECTED"
        ].iloc[0]


        corrected = group[
            group[
                "representation"
            ]
            ==
            "EGO_CORRECTED"
        ].iloc[0]


        comparison_rows.append({

            "task":
                task,

            "model":
                model,

            "raw_precision":
                raw["precision"],

            "corrected_precision":
                corrected["precision"],

            "precision_change_pp":
                100.0
                *
                (
                    corrected["precision"]
                    -
                    raw["precision"]
                ),

            "raw_recall":
                raw["recall"],

            "corrected_recall":
                corrected["recall"],

            "recall_change_pp":
                100.0
                *
                (
                    corrected["recall"]
                    -
                    raw["recall"]
                ),

            "raw_F1":
                raw["F1"],

            "corrected_F1":
                corrected["F1"],

            "F1_change_pp":
                100.0
                *
                (
                    corrected["F1"]
                    -
                    raw["F1"]
                ),

            "raw_false_warning_rate":
                raw[
                    "false_warning_rate"
                ],

            "corrected_false_warning_rate":
                corrected[
                    "false_warning_rate"
                ],

            "false_warning_rate_reduction_pct":
                improvement_pct(

                    raw[
                        "false_warning_rate"
                    ],

                    corrected[
                        "false_warning_rate"
                    ]
                ),

            "raw_AUROC":
                raw["AUROC"],

            "corrected_AUROC":
                corrected["AUROC"],

            "raw_AUPRC":
                raw["AUPRC"],

            "corrected_AUPRC":
                corrected["AUPRC"],

            "raw_TTE_MAE_sec":
                raw["TTE_MAE_sec"],

            "corrected_TTE_MAE_sec":
                corrected[
                    "TTE_MAE_sec"
                ],

            "TTE_MAE_reduction_pct":
                improvement_pct(

                    raw[
                        "TTE_MAE_sec"
                    ],

                    corrected[
                        "TTE_MAE_sec"
                    ]
                )
        })


    comparison = pd.DataFrame(
        comparison_rows
    )


    comparison.to_csv(

        OUT /
        "ego_motion_warning_impact_summary.csv",

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
            "STEP 2C: EGO-MOTION CORRECTION "
            "IMPACT ON VIRTUAL-GEOFENCE WARNING\n"
        )

        f.write(
            "=" * 80
            +
            "\n\n"
        )


        f.write(
            "REFERENCE EVENT DEFINITION\n"
        )

        f.write(
            "--------------------------\n"
        )

        f.write(
            "True boundary-crossing events and the "
            "safe/risk-side direction are defined ONLY "
            "from the ego-motion-corrected TestReal "
            "trajectory. Therefore RAW and CORRECTED "
            "warnings are scored against exactly the "
            "same reference events.\n\n"
        )


        f.write(
            "This avoids changing the target event "
            "between representations.\n\n"
        )


        f.write(
            "WARNING PERFORMANCE\n"
        )

        f.write(
            "-------------------\n"
        )

        f.write(
            comparison.to_string(
                index=False
            )
        )


        f.write(
            "\n\nRAW IMAGE-PLANE LABEL DISTORTION\n"
        )

        f.write(
            "--------------------------------\n"
        )

        f.write(
            diagnostic_df.to_string(
                index=False
            )
        )


        f.write(
            "\n\nIMPORTANT:\n"
        )

        f.write(
            "This is a representation-level "
            "geofence ablation using non-learning "
            "Stationary and Constant-Velocity predictors. "
            "It isolates the effect of camera-motion "
            "correction without retraining a neural model.\n"
        )

        f.write(
            "It does NOT claim independent metric "
            "world-coordinate ground truth.\n"
        )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\nDONE"
    )

    print(
        "Outputs:",
        OUT
    )


    print(
        "\nWARNING IMPACT SUMMARY:\n"
    )

    print(

        comparison
        .to_string(
            index=False
        )
    )


    print(
        "\nRAW LABEL DISTORTION AUDIT:\n"
    )

    print(

        diagnostic_df
        .to_string(
            index=False
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()