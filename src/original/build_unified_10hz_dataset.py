from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\HP\Downloads"
)

BIRD_DIR = (
    ROOT /
    "corrected_trajectory_dataset"
)

WILD_DIR = (
    ROOT /
    "wildbox_trajectory_dataset"
)

OUT = (
    ROOT /
    "unified_trajectory_10hz"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_FPS = 10

TASKS = {

    "2s_to_2s": {
        "future_frames_10hz": 20
    },

    "2s_to_5s": {
        "future_frames_10hz": 50
    }
}


# ============================================================
# RELATIVE REPRESENTATION
# ============================================================

def make_relative(
    history,
    future
):

    # Last observed point = local origin
    origin = history[
        :,
        -1:,
        :
    ]


    history_rel = (
        history
        -
        origin
    )


    future_rel = (
        future
        -
        origin
    )


    return (
        history_rel,
        future_rel
    )


# ============================================================
# SAMPLE-LEVEL QUALITY AUDIT
# ============================================================

def audit_samples(
    history,
    future,
    metadata,
    dataset,
    modality,
    split_name
):

    rows = []


    for i in range(
        len(history)
    ):


        h = history[i]

        f = future[i]


        all_points = np.concatenate(
            [
                h,
                f
            ],
            axis=0
        )


        finite = bool(
            np.isfinite(
                all_points
            ).all()
        )


        outside = (

            (all_points[:, 0] < 0)
            |
            (all_points[:, 0] > 1)
            |
            (all_points[:, 1] < 0)
            |
            (all_points[:, 1] > 1)

        )


        extreme = (

            (np.abs(
                all_points[:, 0]
            ) > 2)

            |

            (np.abs(
                all_points[:, 1]
            ) > 2)

        )


        history_disp = float(

            np.linalg.norm(

                h[-1]
                -
                h[0]

            )

        )


        future_disp = float(

            np.linalg.norm(

                f[-1]
                -
                h[-1]

            )

        )


        # displacement per 0.1 sec -> normalized units/sec

        if len(h) > 1:

            speeds = (

                np.linalg.norm(

                    np.diff(
                        h,
                        axis=0
                    ),

                    axis=1

                )

                *
                TARGET_FPS

            )


            median_speed = float(
                np.median(
                    speeds
                )
            )


            max_speed = float(
                np.max(
                    speeds
                )
            )

        else:

            median_speed = np.nan
            max_speed = np.nan


        meta_row = (
            metadata.iloc[i]
        )


        rows.append({

            "dataset":
                dataset,

            "modality":
                modality,

            "split":
                split_name,

            "source_group":
                meta_row[
                    "source_group"
                ],

            "track_key":
                meta_row[
                    "track_key"
                ],

            "finite":
                int(finite),

            "outside_01_pct":
                100
                *
                outside.mean(),

            "extreme_pct":
                100
                *
                extreme.mean(),

            "history_displacement":
                history_disp,

            "future_displacement":
                future_disp,

            "median_history_speed":
                median_speed,

            "max_history_speed":
                max_speed
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# LOAD BIRDSAI
# ============================================================

def load_birdsai(
    task
):

    meta_file = (
        BIRD_DIR
        /
        f"corrected_windows_{task}.csv"
    )


    npz_file = (
        BIRD_DIR
        /
        f"corrected_{task}.npz"
    )


    meta = pd.read_csv(
        meta_file,
        dtype={
            "sequence": str,
            "segment_id": str
        }
    )


    arrays = np.load(
        npz_file
    )


    histories = arrays[
        "history"
    ]


    futures = arrays[
        "future"
    ]


    # MAIN PAPER DATA = CORE ONLY

    core = meta[

        (
            meta[
                "accepted"
            ]
            ==
            1
        )

        &

        (
            meta[
                "quality_tier"
            ]
            ==
            "core"
        )

        &

        (
            meta[
                "sample_id"
            ]
            >=
            0
        )

    ].copy()


    outputs = {}


    for split in [
        "train_real",
        "test_real"
    ]:


        temp = core[

            core[
                "split"
            ]
            ==
            split

        ].copy()


        ids = (
            temp[
                "sample_id"
            ]
            .astype(int)
            .to_numpy()
        )


        h = histories[
            ids,
            :,
            :2
        ]


        f = futures[
            ids,
            :,
            :2
        ]


        # ----------------------------------------
        # BIRDSAI 30 Hz -> 10 Hz
        # ----------------------------------------

        h = h[
            :,
            ::3,
            :
        ]


        f = f[
            :,
            ::3,
            :
        ]


        temp = (
            temp
            .reset_index(
                drop=True
            )
        )


        temp[
            "dataset"
        ] = "BIRDSAI"


        temp[
            "modality"
        ] = "thermal"


        temp[
            "source_group"
        ] = temp[
            "sequence"
        ]


        temp[
            "track_key"
        ] = temp[
            "segment_id"
        ]


        outputs[
            split
        ] = (
            temp,
            h.astype(
                np.float32
            ),
            f.astype(
                np.float32
            )
        )


    return outputs


# ============================================================
# LOAD WILDBOX
# ============================================================

def load_wildbox(
    task
):


    meta_file = (

        WILD_DIR
        /
        f"wildbox_windows_{task}.csv"

    )


    npz_file = (

        WILD_DIR
        /
        f"wildbox_corrected_{task}.npz"

    )


    meta = pd.read_csv(
        meta_file,
        dtype={
            "video": str,
            "segment": str,
            "track_id": str
        }
    )


    arrays = np.load(
        npz_file
    )


    # use common x,y only

    h = arrays[
        "history"
    ][
        :,
        :,
        :2
    ]


    f = arrays[
        "future"
    ][
        :,
        :,
        :2
    ]


    meta = (
        meta
        .sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )


    meta[
        "dataset"
    ] = "WildBox"


    meta[
        "modality"
    ] = "RGB"


    meta[
        "source_group"
    ] = meta[
        "video"
    ]


    meta[
        "track_key"
    ] = (

        meta[
            "video"
        ].astype(str)

        +
        "__"

        +
        meta[
            "segment"
        ].astype(str)

        +
        "__track"

        +
        meta[
            "track_id"
        ].astype(str)

    )


    return (
        meta,
        h.astype(
            np.float32
        ),
        f.astype(
            np.float32
        )
    )


# ============================================================
# PROCESS TASKS
# ============================================================

all_summary = []

all_audits = []


for task, config in (
    TASKS.items()
):


    print(
        "\nProcessing:",
        task
    )


    # ========================================================
    # BIRDSAI
    # ========================================================

    birds = load_birdsai(
        task
    )


    for split, (
        meta,
        history,
        future
    ) in birds.items():


        expected_future = (
            config[
                "future_frames_10hz"
            ]
        )


        if history.shape[1] != 20:

            raise RuntimeError(
                f"{task} {split}: "
                f"history shape wrong "
                f"{history.shape}"
            )


        if future.shape[1] != expected_future:

            raise RuntimeError(
                f"{task} {split}: "
                f"future shape wrong "
                f"{future.shape}"
            )


        (
            history_rel,
            future_rel
        ) = make_relative(
            history,
            future
        )


        prefix = (

            f"birdsai_"
            f"{split}_core_"
            f"{task}_10hz"

        )


        np.savez_compressed(

            OUT /
            f"{prefix}.npz",

            history_abs=
                history,

            future_abs=
                future,

            history_rel=
                history_rel,

            future_rel=
                future_rel
        )


        meta.to_csv(

            OUT /
            f"{prefix}_metadata.csv",

            index=False
        )


        audit = audit_samples(

            history,
            future,
            meta,

            dataset=
                "BIRDSAI",

            modality=
                "thermal",

            split_name=
                split
        )


        audit[
            "task"
        ] = task


        all_audits.append(
            audit
        )


        all_summary.append({

            "task":
                task,

            "dataset":
                "BIRDSAI",

            "modality":
                "thermal",

            "role":
                (
                    "training"
                    if split
                    ==
                    "train_real"
                    else
                    "in_domain_test"
                ),

            "samples":
                len(
                    history
                ),

            "source_groups":
                meta[
                    "source_group"
                ].nunique(),

            "tracks":
                meta[
                    "track_key"
                ].nunique(),

            "history_shape":
                str(
                    history.shape
                ),

            "future_shape":
                str(
                    future.shape
                )
        })


    # ========================================================
    # WILDBOX
    # ========================================================

    (
        wild_meta,
        wild_history,
        wild_future
    ) = load_wildbox(
        task
    )


    if wild_history.shape[1] != 20:

        raise RuntimeError(
            f"{task} WildBox "
            f"history shape wrong "
            f"{wild_history.shape}"
        )


    if (
        wild_future.shape[1]
        !=
        config[
            "future_frames_10hz"
        ]
    ):

        raise RuntimeError(
            f"{task} WildBox "
            f"future shape wrong "
            f"{wild_future.shape}"
        )


    (
        wild_history_rel,
        wild_future_rel
    ) = make_relative(

        wild_history,

        wild_future

    )


    prefix = (

        f"wildbox_external_"
        f"{task}_10hz"

    )


    np.savez_compressed(

        OUT /
        f"{prefix}.npz",

        history_abs=
            wild_history,

        future_abs=
            wild_future,

        history_rel=
            wild_history_rel,

        future_rel=
            wild_future_rel

    )


    wild_meta.to_csv(

        OUT /
        f"{prefix}_metadata.csv",

        index=False
    )


    audit = audit_samples(

        wild_history,
        wild_future,
        wild_meta,

        dataset=
            "WildBox",

        modality=
            "RGB",

        split_name=
            "external_test"

    )


    audit[
        "task"
    ] = task


    all_audits.append(
        audit
    )


    all_summary.append({

        "task":
            task,

        "dataset":
            "WildBox",

        "modality":
            "RGB",

        "role":
            "external_test",

        "samples":
            len(
                wild_history
            ),

        "source_groups":
            wild_meta[
                "source_group"
            ].nunique(),

        "tracks":
            wild_meta[
                "track_key"
            ].nunique(),

        "history_shape":
            str(
                wild_history.shape
            ),

        "future_shape":
            str(
                wild_future.shape
            )
    })


# ============================================================
# SAVE AUDIT
# ============================================================

summary_df = pd.DataFrame(
    all_summary
)


summary_df.to_csv(

    OUT /
    "dataset_summary.csv",

    index=False
)


audit_df = pd.concat(

    all_audits,

    ignore_index=True

)


audit_df.to_csv(

    OUT /
    "trajectory_quality_audit.csv",

    index=False
)


# ============================================================
# AGGREGATE QUALITY
# ============================================================

quality_rows = []


for (
    task,
    dataset,
    split
), group in audit_df.groupby(

    [
        "task",
        "dataset",
        "split"
    ]

):


    quality_rows.append({

        "task":
            task,

        "dataset":
            dataset,

        "split":
            split,

        "samples":
            len(
                group
            ),

        "finite_pct":
            100
            *
            group[
                "finite"
            ].mean(),

        "mean_outside_01_pct":
            group[
                "outside_01_pct"
            ].mean(),

        "max_extreme_pct":
            group[
                "extreme_pct"
            ].max(),

        "median_history_displacement":
            group[
                "history_displacement"
            ].median(),

        "median_future_displacement":
            group[
                "future_displacement"
            ].median(),

        "median_history_speed":
            group[
                "median_history_speed"
            ].median(),

        "p95_max_history_speed":
            group[
                "max_history_speed"
            ].quantile(
                0.95
            )
    })


quality_df = pd.DataFrame(
    quality_rows
)


quality_df.to_csv(

    OUT /
    "trajectory_quality_summary.csv",

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
        "UNIFIED ELEPHANT TRAJECTORY DATASET — 10 HZ\n"
    )

    f.write(
        "=" * 72
        +
        "\n\n"
    )


    f.write(
        "Common representation:\n"
    )

    f.write(
        "- Sampling frequency: 10 Hz\n"
    )

    f.write(
        "- Observation: 2 s / 20 frames\n"
    )

    f.write(
        "- Forecast: 2 s / 20 frames "
        "or 5 s / 50 frames\n"
    )

    f.write(
        "- Features used for common model: "
        "camera-corrected normalized x,y\n"
    )

    f.write(
        "- Both absolute and relative-to-last-observation "
        "coordinates are saved.\n\n"
    )


    f.write(
        "Experimental roles:\n"
    )

    f.write(
        "- BIRDSAI TrainReal CORE = training\n"
    )

    f.write(
        "- BIRDSAI TestReal CORE = untouched in-domain test\n"
    )

    f.write(
        "- WildBox RGB = untouched external cross-modal test\n\n"
    )


    for _, row in (
        summary_df.iterrows()
    ):


        f.write(

            f"{row['task']} | "
            f"{row['dataset']} | "
            f"{row['role']}\n"

        )


        f.write(

            f"Samples: "
            f"{int(row['samples'])}\n"

        )


        f.write(

            f"Source groups: "
            f"{int(row['source_groups'])}\n"

        )


        f.write(

            f"Tracks: "
            f"{int(row['tracks'])}\n"

        )


        f.write(

            f"History: "
            f"{row['history_shape']}\n"

        )


        f.write(

            f"Future: "
            f"{row['future_shape']}\n\n"

        )


print(
    "\n========================================"
)

print(
    "UNIFIED 10 HZ DATASET BUILD COMPLETE"
)

print(
    "========================================"
)

print(
    "\nDataset summary:"
)

print(
    summary_df
)

print(
    "\nQuality summary:"
)

print(
    quality_df
)

print(
    "\nSaved to:"
)

print(
    OUT
)