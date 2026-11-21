from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

ROOT = Path(
    r"C:\Users\HP\Downloads\unified_trajectory_10hz"
)

OUT = Path(
    r"C:\Users\HP\Downloads\canonical_motion_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASETS
# ============================================================

SETS = {

    "2s_to_2s": [

        (
            "BIRDSAI_train",
            ROOT / "birdsai_train_real_core_2s_to_2s_10hz.npz",
            ROOT / "birdsai_train_real_core_2s_to_2s_10hz_metadata.csv"
        ),

        (
            "BIRDSAI_test",
            ROOT / "birdsai_test_real_core_2s_to_2s_10hz.npz",
            ROOT / "birdsai_test_real_core_2s_to_2s_10hz_metadata.csv"
        ),

        (
            "WildBox_external",
            ROOT / "wildbox_external_2s_to_2s_10hz.npz",
            ROOT / "wildbox_external_2s_to_2s_10hz_metadata.csv"
        )
    ],

    "2s_to_5s": [

        (
            "BIRDSAI_train",
            ROOT / "birdsai_train_real_core_2s_to_5s_10hz.npz",
            ROOT / "birdsai_train_real_core_2s_to_5s_10hz_metadata.csv"
        ),

        (
            "BIRDSAI_test",
            ROOT / "birdsai_test_real_core_2s_to_5s_10hz.npz",
            ROOT / "birdsai_test_real_core_2s_to_5s_10hz_metadata.csv"
        ),

        (
            "WildBox_external",
            ROOT / "wildbox_external_2s_to_5s_10hz.npz",
            ROOT / "wildbox_external_2s_to_5s_10hz_metadata.csv"
        )
    ]
}


FPS = 10

RECENT_HEADING_FRAMES = 10


# ============================================================
# HEADING ESTIMATION
# ============================================================

def estimate_heading(history):

    recent = history[
        -RECENT_HEADING_FRAMES:
    ]

    t = np.arange(
        len(recent),
        dtype=np.float64
    )

    vx = np.polyfit(
        t,
        recent[:, 0],
        1
    )[0]

    vy = np.polyfit(
        t,
        recent[:, 1],
        1
    )[0]

    speed = np.hypot(
        vx,
        vy
    )

    if speed < 1e-12:

        return 0.0, speed

    angle = np.arctan2(
        vy,
        vx
    )

    return angle, speed


# ============================================================
# ROTATION
# ============================================================

def rotate_to_heading(
    points,
    angle
):

    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )

    R = np.array(
        [
            [c, s],
            [-s, c]
        ],
        dtype=np.float64
    )

    return points @ R.T


# ============================================================
# SAMPLE AUDIT
# ============================================================

rows = []


for task, datasets in SETS.items():

    print(
        "\nTASK:",
        task
    )

    for dataset_name, npz_path, meta_path in datasets:

        print(
            "  ",
            dataset_name
        )

        arrays = np.load(
            npz_path
        )

        metadata = pd.read_csv(
            meta_path
        )

        history = arrays[
            "history_abs"
        ]

        future = arrays[
            "future_abs"
        ]

        if len(history) != len(metadata):

            raise RuntimeError(
                f"Length mismatch: "
                f"{dataset_name}"
            )


        for i in range(
            len(history)
        ):

            h = history[i].astype(
                np.float64
            )

            f = future[i].astype(
                np.float64
            )


            # ====================================
            # HISTORY MOTION
            # ====================================

            increments = np.diff(
                h,
                axis=0
            )

            step_dist = np.linalg.norm(
                increments,
                axis=1
            )

            path_length = float(
                step_dist.sum()
            )

            net_displacement = float(
                np.linalg.norm(
                    h[-1] - h[0]
                )
            )

            straightness = (

                net_displacement
                /
                path_length

                if path_length > 1e-12

                else 0.0
            )


            median_step = float(
                np.median(
                    step_dist
                )
            )


            # ====================================
            # LOCAL COORDINATES
            # ====================================

            origin = h[-1]

            h_local = (
                h - origin
            )

            f_local = (
                f - origin
            )


            raw_future_max = float(
                np.max(
                    np.abs(
                        f_local
                    )
                )
            )


            # ====================================
            # HEADING
            # ====================================

            angle, recent_speed = (
                estimate_heading(
                    h
                )
            )


            h_rot = rotate_to_heading(
                h_local,
                angle
            )

            f_rot = rotate_to_heading(
                f_local,
                angle
            )


            # ====================================
            # CANONICAL SCALE PREVIEW
            #
            # Do NOT lock this scale yet.
            # This audit only tests feasibility.
            # ====================================

            if path_length > 1e-8:

                h_can = (
                    h_rot
                    /
                    path_length
                )

                f_can = (
                    f_rot
                    /
                    path_length
                )

                canonical_future_max = float(
                    np.max(
                        np.abs(
                            f_can
                        )
                    )
                )

                canonical_final_distance = float(
                    np.linalg.norm(
                        f_can[-1]
                    )
                )

            else:

                canonical_future_max = np.nan
                canonical_final_distance = np.nan


            meta = metadata.iloc[i]


            rows.append({

                "task":
                    task,

                "dataset":
                    dataset_name,

                "sample_index":
                    i,

                "source_group":
                    str(
                        meta.get(
                            "source_group",
                            ""
                        )
                    ),

                "track_key":
                    str(
                        meta.get(
                            "track_key",
                            ""
                        )
                    ),

                "history_path_length":
                    path_length,

                "history_net_displacement":
                    net_displacement,

                "straightness":
                    straightness,

                "median_step":
                    median_step,

                "recent_heading_speed_per_frame":
                    recent_speed,

                "recent_heading_speed_per_sec":
                    recent_speed
                    *
                    FPS,

                "raw_relative_future_max":
                    raw_future_max,

                "canonical_future_max":
                    canonical_future_max,

                "canonical_final_distance":
                    canonical_final_distance
            })


audit = pd.DataFrame(
    rows
)


audit.to_csv(

    OUT /
    "canonical_motion_samples.csv",

    index=False
)


# ============================================================
# SUMMARY QUANTILES
# ============================================================

summary_rows = []


for (
    task,
    dataset
), g in audit.groupby(

    [
        "task",
        "dataset"
    ]

):

    summary_rows.append({

        "task":
            task,

        "dataset":
            dataset,

        "samples":
            len(g),

        "path_q01":
            g[
                "history_path_length"
            ].quantile(
                0.01
            ),

        "path_q05":
            g[
                "history_path_length"
            ].quantile(
                0.05
            ),

        "path_q25":
            g[
                "history_path_length"
            ].quantile(
                0.25
            ),

        "path_median":
            g[
                "history_path_length"
            ].median(),

        "path_q95":
            g[
                "history_path_length"
            ].quantile(
                0.95
            ),

        "straightness_median":
            g[
                "straightness"
            ].median(),

        "straightness_q10":
            g[
                "straightness"
            ].quantile(
                0.10
            ),

        "heading_speed_median":
            g[
                "recent_heading_speed_per_sec"
            ].median(),

        "raw_future_max_median":
            g[
                "raw_relative_future_max"
            ].median(),

        "raw_future_max_q95":
            g[
                "raw_relative_future_max"
            ].quantile(
                0.95
            ),

        "canonical_future_median":
            g[
                "canonical_future_max"
            ].median(),

        "canonical_future_q95":
            g[
                "canonical_future_max"
            ].quantile(
                0.95
            ),

        "canonical_future_max":
            g[
                "canonical_future_max"
            ].max()
    })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(

    OUT /
    "canonical_motion_summary.csv",

    index=False
)


# ============================================================
# PER SOURCE
# ============================================================

source_rows = []


for (
    task,
    dataset,
    source
), g in audit.groupby(

    [
        "task",
        "dataset",
        "source_group"
    ]

):

    source_rows.append({

        "task":
            task,

        "dataset":
            dataset,

        "source_group":
            source,

        "samples":
            len(g),

        "tracks":
            g[
                "track_key"
            ].nunique(),

        "path_median":
            g[
                "history_path_length"
            ].median(),

        "straightness_median":
            g[
                "straightness"
            ].median(),

        "canonical_future_q95":
            g[
                "canonical_future_max"
            ].quantile(
                0.95
            )
    })


pd.DataFrame(
    source_rows
).to_csv(

    OUT /
    "canonical_per_source.csv",

    index=False
)


# ============================================================
# OUTLIERS
# ============================================================

outliers = (

    audit.sort_values(

        "canonical_future_max",

        ascending=False

    )

    .head(
        50
    )
)


outliers.to_csv(

    OUT /
    "top_canonical_outliers.csv",

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
        "CANONICAL MOTION AUDIT\n"
    )

    f.write(
        "=" * 72
        +
        "\n\n"
    )


    f.write(
        "No samples were removed.\n"
    )

    f.write(
        "This stage only evaluates whether "
        "translation + heading rotation + "
        "history-path normalization is suitable.\n\n"
    )


    for _, row in (
        summary.iterrows()
    ):

        f.write(

            f"{row['task']} | "
            f"{row['dataset']}\n"

        )

        f.write(

            f"samples="
            f"{int(row['samples'])}\n"

        )

        f.write(

            f"path median="
            f"{row['path_median']:.6f}, "

            f"q05="
            f"{row['path_q05']:.6f}\n"

        )

        f.write(

            f"straightness median="
            f"{row['straightness_median']:.4f}\n"

        )

        f.write(

            f"canonical future median="
            f"{row['canonical_future_median']:.4f}, "

            f"q95="
            f"{row['canonical_future_q95']:.4f}, "

            f"max="
            f"{row['canonical_future_max']:.4f}\n\n"

        )


print(
    "\n===================================="
)

print(
    "CANONICAL MOTION AUDIT COMPLETE"
)

print(
    "===================================="
)

print(
    "\nSummary:"
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