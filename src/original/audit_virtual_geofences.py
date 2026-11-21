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
    r"C:\Users\HP\Downloads\virtual_zone_audit"
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

        "horizon":
            2
    },

    "2s_to_5s": {

        "meta":
            ROOT /
            "corrected_windows_2s_to_5s.csv",

        "npz":
            ROOT /
            "corrected_2s_to_5s.npz",

        "horizon":
            5
    }
}


FPS = 30

ZONE_WIDTHS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]

SIDES = [
    "left",
    "right",
    "top",
    "bottom"
]


# ============================================================
# ZONE FUNCTIONS
# ============================================================

def inside_zone(
    x,
    y,
    side,
    width
):

    if side == "left":

        return x <= width

    if side == "right":

        return x >= (
            1 - width
        )

    if side == "top":

        return y <= width

    if side == "bottom":

        return y >= (
            1 - width
        )


# ============================================================
# AUDIT ONE FORECAST HORIZON
# ============================================================

def audit_forecast(
    forecast_name,
    config
):

    meta = pd.read_csv(
        config["meta"],
        dtype={
            "sequence": str,
            "segment_id": str
        }
    )


    npz = np.load(
        config["npz"]
    )


    history = npz[
        "history"
    ]

    future = npz[
        "future"
    ]


    # ==========================================
    # TRAIN + CORE ONLY
    # ==========================================

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


    rows = []


    for _, row in (
        meta.iterrows()
    ):


        sample_id = int(
            row["sample_id"]
        )


        if (
            sample_id
            >=
            len(history)
        ):

            continue


        hist = history[
            sample_id
        ]

        fut = future[
            sample_id
        ]


        # Current location =
        # final observation frame

        current_x = float(
            hist[-1, 0]
        )

        current_y = float(
            hist[-1, 1]
        )


        future_xy = fut[
            :, :2
        ]


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
                future_xy
            ).all()
        ):

            continue


        # ==================================
        # TRY EACH CANDIDATE ZONE
        # ==================================

        for width in (
            ZONE_WIDTHS
        ):


            for side in (
                SIDES
            ):


                # Prospective forecasting:
                # elephant must currently
                # be outside zone.

                if inside_zone(

                    current_x,

                    current_y,

                    side,

                    width

                ):

                    continue


                future_inside = []


                for x, y in (
                    future_xy
                ):


                    future_inside.append(

                        inside_zone(

                            float(x),

                            float(y),

                            side,

                            width

                        )

                    )


                future_inside = (
                    np.array(
                        future_inside,
                        dtype=bool
                    )
                )


                positive = int(

                    future_inside.any()

                )


                if positive:

                    first_entry = int(

                        np.argmax(
                            future_inside
                        )

                    )


                    # Future frame 0
                    # = 1/30 sec ahead

                    tte = (

                        first_entry + 1

                    ) / FPS


                else:

                    tte = np.nan


                rows.append({

                    "forecast":
                        forecast_name,

                    "zone_width":
                        width,

                    "zone_side":
                        side,

                    "entry_label":
                        positive,

                    "tte_sec":
                        tte
                })


    return pd.DataFrame(
        rows
    )


# ============================================================
# RUN BOTH HORIZONS
# ============================================================

results = []


for name, config in (
    FILES.items()
):

    print(
        "Processing:",
        name
    )


    result = audit_forecast(

        name,

        config

    )


    results.append(
        result
    )


all_results = pd.concat(

    results,

    ignore_index=True

)


# ============================================================
# COMBINED BALANCE
# ============================================================

combined_rows = []


for (
    forecast,
    width
), group in all_results.groupby(

    [
        "forecast",
        "zone_width"
    ]

):


    total = len(
        group
    )


    positives = int(

        group[
            "entry_label"
        ].sum()

    )


    negatives = (

        total
        -
        positives

    )


    positive_rate = (

        100
        *
        positives
        /
        total

        if total
        else 0
    )


    positive_tte = (

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


    combined_rows.append({

        "forecast":
            forecast,

        "zone_width":
            width,

        "eligible":
            total,

        "positive":
            positives,

        "negative":
            negatives,

        "positive_pct":
            positive_rate,

        "median_tte":
            (
                positive_tte.median()
                if len(
                    positive_tte
                )
                else np.nan
            )
    })


combined = pd.DataFrame(
    combined_rows
)


combined.to_csv(

    OUT /
    "candidate_zone_balance_train_core.csv",

    index=False
)


# ============================================================
# BALANCE BY SIDE
# ============================================================

side_rows = []


for (
    forecast,
    width,
    side
), group in all_results.groupby(

    [
        "forecast",
        "zone_width",
        "zone_side"
    ]

):


    total = len(
        group
    )


    positive = int(

        group[
            "entry_label"
        ].sum()

    )


    side_rows.append({

        "forecast":
            forecast,

        "zone_width":
            width,

        "side":
            side,

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
            )
    })


side_df = pd.DataFrame(
    side_rows
)


side_df.to_csv(

    OUT /
    "candidate_zone_balance_by_side_train_core.csv",

    index=False
)


# ============================================================
# TRAIN-ONLY SCREENING RECOMMENDATION
# ============================================================

recommendations = []


for forecast, group in (
    combined.groupby(
        "forecast"
    )
):


    group = (
        group.copy()
    )


    # Screening target:
    # approximately 25% positive.
    #
    # Final decision will NOT
    # depend on this score alone.

    group[
        "balance_score"
    ] = (

        group[
            "positive_pct"
        ]
        -
        25

    ).abs()


    # Avoid tiny event sets

    group.loc[

        (
            group["positive"]
            <
            30
        )

        |

        (
            group["negative"]
            <
            30
        ),

        "balance_score"

    ] += 1000


    best = (

        group

        .sort_values(
            [
                "balance_score",
                "zone_width"
            ]
        )

        .iloc[0]

    )


    recommendations.append(
        best
    )


recommendation_df = pd.DataFrame(
    recommendations
)


recommendation_df.to_csv(

    OUT /
    "train_only_zone_recommendation.csv",

    index=False
)


# ============================================================
# TEXT SUMMARY
# ============================================================

with open(

    OUT /
    "SUMMARY.txt",

    "w"

) as f:


    f.write(
        "VIRTUAL GEOFENCE AUDIT\n"
    )

    f.write(
        "=" * 60
        +
        "\n\n"
    )


    f.write(
        "DATA USED: "
        "TrainReal CORE samples only\n\n"
    )


    for forecast in (
        combined[
            "forecast"
        ].unique()
    ):


        f.write(
            forecast
            +
            "\n"
        )


        subset = combined[

            combined[
                "forecast"
            ]
            ==
            forecast

        ].sort_values(
            "zone_width"
        )


        for _, row in (
            subset.iterrows()
        ):


            f.write(

                f"width={row['zone_width']:.2f} | "

                f"eligible="
                f"{int(row['eligible'])} | "

                f"positive="
                f"{int(row['positive'])} | "

                f"negative="
                f"{int(row['negative'])} | "

                f"positive%="
                f"{row['positive_pct']:.2f} | "

                f"median TTE="
                f"{row['median_tte']:.3f}s\n"

            )


        rec = recommendation_df[

            recommendation_df[
                "forecast"
            ]
            ==
            forecast

        ].iloc[0]


        f.write(

            "\nScreening recommendation: "

            f"width="
            f"{rec['zone_width']:.2f}, "

            f"positive%="
            f"{rec['positive_pct']:.2f}\n\n"

        )


    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "TestReal was NOT used "
        "for zone selection.\n"
    )

    f.write(
        "This is only benchmark-design "
        "screening, not final labels.\n"
    )


print(
    "\nVIRTUAL ZONE AUDIT COMPLETE"
)

print(
    "\nResults:"
)

print(
    OUT
)