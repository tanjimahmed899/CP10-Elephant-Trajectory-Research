from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\HP\Downloads\corrected_trajectory_dataset"
)

OUT = Path(
    r"C:\Users\HP\Downloads\multigeofence_audit"
)

OUT.mkdir(exist_ok=True)


FILES = {

    "2s_to_2s": {

        "meta":
            ROOT /
            "corrected_windows_2s_to_2s.csv",

        "npz":
            ROOT /
            "corrected_2s_to_2s.npz",

        "horizon_sec":
            2
    },

    "2s_to_5s": {

        "meta":
            ROOT /
            "corrected_windows_2s_to_5s.csv",

        "npz":
            ROOT /
            "corrected_2s_to_5s.npz",

        "horizon_sec":
            5
    }
}


FPS = 30


# Fixed boundaries.
# These are chosen BEFORE looking at TestReal.

THRESHOLDS = [
    0.25,
    0.50,
    0.75
]


# ============================================================
# ZONE DEFINITION
# ============================================================

def inside_zone(
    x,
    y,
    axis,
    direction,
    threshold
):

    if axis == "x":

        if direction == "low":

            return x <= threshold

        else:

            return x >= threshold


    else:

        if direction == "low":

            return y <= threshold

        else:

            return y >= threshold


# ============================================================
# DISTANCE TO BOUNDARY
# ============================================================

def boundary_distance(
    x,
    y,
    axis,
    threshold
):

    if axis == "x":

        return abs(
            x - threshold
        )

    else:

        return abs(
            y - threshold
        )


# ============================================================
# CREATE GEOFENCE BANK
# ============================================================

GEOFENCES = []


for axis in [
    "x",
    "y"
]:

    for threshold in (
        THRESHOLDS
    ):

        for direction in [
            "low",
            "high"
        ]:

            GEOFENCES.append({

                "axis":
                    axis,

                "threshold":
                    threshold,

                "direction":
                    direction,

                "geofence_id":

                    f"{axis}_"
                    f"{threshold:.2f}_"
                    f"{direction}"
            })


# ============================================================
# AUDIT
# ============================================================

def process(
    forecast,
    config
):

    meta = pd.read_csv(

        config["meta"],

        dtype={
            "sequence": str,
            "segment_id": str
        }
    )


    arrays = np.load(
        config["npz"]
    )


    history = arrays[
        "history"
    ]


    future = arrays[
        "future"
    ]


    # =============================================
    # TRAIN + CORE ONLY FOR BENCHMARK DESIGN
    # =============================================

    meta = meta[

        (
            meta["split"]
            ==
            "train_real"
        )

        &

        (
            meta["accepted"]
            ==
            1
        )

        &

        (
            meta["quality_tier"]
            ==
            "core"
        )

        &

        (
            meta["sample_id"]
            >=
            0
        )

    ].copy()


    instances = []


    for _, row in (
        meta.iterrows()
    ):


        sid = int(
            row["sample_id"]
        )


        if sid >= len(history):

            continue


        hist = history[
            sid
        ]


        fut = future[
            sid
        ]


        current_x = float(
            hist[-1, 0]
        )

        current_y = float(
            hist[-1, 1]
        )


        if not (

            np.isfinite(
                current_x
            )

            and

            np.isfinite(
                current_y
            )

            and

            np.isfinite(
                fut[:, :2]
            ).all()

        ):

            continue


        for zone in GEOFENCES:


            axis = zone[
                "axis"
            ]


            threshold = zone[
                "threshold"
            ]


            direction = zone[
                "direction"
            ]


            # Elephant must currently
            # be OUTSIDE protected zone.

            if inside_zone(

                current_x,
                current_y,

                axis,
                direction,
                threshold

            ):

                continue


            labels = []


            for x, y in (
                fut[:, :2]
            ):


                labels.append(

                    inside_zone(

                        float(x),
                        float(y),

                        axis,
                        direction,
                        threshold

                    )

                )


            labels = np.array(
                labels,
                dtype=bool
            )


            positive = int(
                labels.any()
            )


            if positive:

                first = int(
                    np.argmax(
                        labels
                    )
                )

                tte = (

                    first + 1

                ) / FPS


            else:

                tte = np.nan


            instances.append({

                "forecast":
                    forecast,

                "horizon_sec":
                    config[
                        "horizon_sec"
                    ],

                "sequence":
                    row[
                        "sequence"
                    ],

                "object_id":
                    row[
                        "object_id"
                    ],

                "segment_id":
                    row[
                        "segment_id"
                    ],

                "sample_id":
                    sid,

                "obs_start_frame":
                    row[
                        "obs_start_frame"
                    ],

                "obs_end_frame":
                    row[
                        "obs_end_frame"
                    ],

                "geofence_id":
                    zone[
                        "geofence_id"
                    ],

                "axis":
                    axis,

                "threshold":
                    threshold,

                "direction":
                    direction,

                "current_x":
                    current_x,

                "current_y":
                    current_y,

                "distance_to_boundary":
                    boundary_distance(

                        current_x,
                        current_y,

                        axis,
                        threshold

                    ),

                "entry_label":
                    positive,

                "tte_sec":
                    tte
            })


    return pd.DataFrame(
        instances
    )


# ============================================================
# RUN
# ============================================================

results = []


for forecast, config in (
    FILES.items()
):

    print(
        "Processing:",
        forecast
    )


    result = process(

        forecast,
        config

    )


    results.append(
        result
    )


instances = pd.concat(

    results,

    ignore_index=True

)


instances.to_csv(

    OUT /
    "multigeofence_train_core_instances.csv",

    index=False
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall_rows = []


for forecast, group in (
    instances.groupby(
        "forecast"
    )
):


    positive = int(

        group[
            "entry_label"
        ].sum()

    )


    total = len(
        group
    )


    positive_histories = (

        group.loc[

            group[
                "entry_label"
            ]
            ==
            1,

            [
                "sequence",
                "segment_id",
                "sample_id"
            ]

        ]

        .drop_duplicates()

    )


    positive_sequences = (

        group.loc[

            group[
                "entry_label"
            ]
            ==
            1,

            "sequence"

        ]

        .nunique()

    )


    tte = (

        group.loc[

            group[
                "entry_label"
            ]
            ==
            1,

            "tte_sec"

        ]

        .dropna()

    )


    overall_rows.append({

        "forecast":
            forecast,

        "total_instances":
            total,

        "positive":
            positive,

        "negative":
            total - positive,

        "positive_pct":

            100
            *
            positive
            /
            total,

        "unique_positive_histories":
            len(
                positive_histories
            ),

        "positive_source_sequences":
            positive_sequences,

        "median_tte_sec":

            (
                tte.median()
                if len(tte)
                else np.nan
            )
    })


overall = pd.DataFrame(
    overall_rows
)


overall.to_csv(

    OUT /
    "overall_balance.csv",

    index=False
)


# ============================================================
# PER-GEOFENCE SUMMARY
# ============================================================

zone_rows = []


for (
    forecast,
    geofence
), group in instances.groupby(

    [
        "forecast",
        "geofence_id"
    ]

):


    positive = int(

        group[
            "entry_label"
        ].sum()

    )


    total = len(
        group
    )


    positive_sequences = (

        group.loc[

            group[
                "entry_label"
            ]
            ==
            1,

            "sequence"

        ].nunique()

    )


    zone_rows.append({

        "forecast":
            forecast,

        "geofence_id":
            geofence,

        "eligible":
            total,

        "positive":
            positive,

        "negative":
            total - positive,

        "positive_pct":

            (
                100
                *
                positive
                /
                total

                if total
                else 0
            ),

        "positive_sequences":
            positive_sequences
    })


zone_summary = pd.DataFrame(
    zone_rows
)


zone_summary.to_csv(

    OUT /
    "per_geofence_balance.csv",

    index=False
)


# ============================================================
# PER-SEQUENCE POSITIVES
# ============================================================

sequence_rows = []


for (
    forecast,
    sequence
), group in instances.groupby(

    [
        "forecast",
        "sequence"
    ]

):


    positive = int(

        group[
            "entry_label"
        ].sum()

    )


    sequence_rows.append({

        "forecast":
            forecast,

        "sequence":
            sequence,

        "instances":
            len(group),

        "positive":
            positive,

        "positive_pct":

            (
                100
                *
                positive
                /
                len(group)

                if len(group)
                else 0
            )
    })


sequence_summary = pd.DataFrame(
    sequence_rows
)


sequence_summary.to_csv(

    OUT /
    "per_sequence_positive_balance.csv",

    index=False
)


# ============================================================
# SUMMARY TEXT
# ============================================================

with open(

    OUT /
    "SUMMARY.txt",

    "w"

) as f:


    f.write(
        "MULTI-GEOFENCE AUDIT\n"
    )

    f.write(
        "=" * 65
        +
        "\n\n"
    )


    f.write(
        "DATA: TrainReal CORE only\n"
    )


    f.write(
        "Thresholds: "
        "0.25, 0.50, 0.75\n"
    )


    f.write(
        "Axes: x and y\n"
    )


    f.write(
        "Directions: low and high\n"
    )


    f.write(
        "Total fixed geofences: 12\n\n"
    )


    for _, row in (
        overall.iterrows()
    ):


        f.write(

            f"{row['forecast']}\n"

        )


        f.write(

            f"Instances: "
            f"{int(row['total_instances'])}\n"

        )


        f.write(

            f"Positive: "
            f"{int(row['positive'])}\n"

        )


        f.write(

            f"Negative: "
            f"{int(row['negative'])}\n"

        )


        f.write(

            f"Positive rate: "
            f"{row['positive_pct']:.2f}%\n"

        )


        f.write(

            f"Unique positive histories: "
            f"{int(row['unique_positive_histories'])}\n"

        )


        f.write(

            f"Positive source sequences: "
            f"{int(row['positive_source_sequences'])}\n"

        )


        f.write(

            f"Median TTE: "
            f"{row['median_tte_sec']:.3f}s\n\n"

        )


print(
    "\nMULTI-GEOFENCE AUDIT COMPLETE"
)

print(
    overall
)

print(
    "\nResults:"
)

print(
    OUT
)