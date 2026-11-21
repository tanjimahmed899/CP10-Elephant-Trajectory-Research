from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\HP\Downloads\unified_trajectory_10hz"
)

OUT = Path(
    r"C:\Users\HP\Downloads\final_neural_trajectory_dataset"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

VAL_TRACK_FRACTION = 0.20

RECENT_HEADING_FRAMES = 10

SCALE_FLOOR_QUANTILE = 0.05


TASKS = {

    "2s_to_2s": {

        "train_npz":
            ROOT /
            "birdsai_train_real_core_2s_to_2s_10hz.npz",

        "train_meta":
            ROOT /
            "birdsai_train_real_core_2s_to_2s_10hz_metadata.csv",

        "test_npz":
            ROOT /
            "birdsai_test_real_core_2s_to_2s_10hz.npz",

        "test_meta":
            ROOT /
            "birdsai_test_real_core_2s_to_2s_10hz_metadata.csv",

        "wild_npz":
            ROOT /
            "wildbox_external_2s_to_2s_10hz.npz",

        "wild_meta":
            ROOT /
            "wildbox_external_2s_to_2s_10hz_metadata.csv"
    },


    "2s_to_5s": {

        "train_npz":
            ROOT /
            "birdsai_train_real_core_2s_to_5s_10hz.npz",

        "train_meta":
            ROOT /
            "birdsai_train_real_core_2s_to_5s_10hz_metadata.csv",

        "test_npz":
            ROOT /
            "birdsai_test_real_core_2s_to_5s_10hz.npz",

        "test_meta":
            ROOT /
            "birdsai_test_real_core_2s_to_5s_10hz_metadata.csv",

        "wild_npz":
            ROOT /
            "wildbox_external_2s_to_5s_10hz.npz",

        "wild_meta":
            ROOT /
            "wildbox_external_2s_to_5s_10hz_metadata.csv"
    }
}


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def history_path_length(
    history
):

    return float(

        np.linalg.norm(

            np.diff(
                history,
                axis=0
            ),

            axis=1

        ).sum()

    )


def estimate_heading(
    history
):

    """
    Linear regression over the most recent
    observation frames.

    Returns angle in radians.
    """

    recent = history[
        -RECENT_HEADING_FRAMES:
    ]


    t = np.arange(
        len(recent),
        dtype=np.float64
    )


    tc = (
        t
        -
        t.mean()
    )


    denom = np.sum(
        tc ** 2
    )


    if denom <= 1e-12:

        return 0.0


    centered = (

        recent

        -
        recent.mean(
            axis=0,
            keepdims=True
        )

    )


    velocity = (

        tc[:, None]
        *
        centered

    ).sum(
        axis=0
    ) / denom


    magnitude = np.linalg.norm(
        velocity
    )


    if magnitude <= 1e-10:

        return 0.0


    return float(

        np.arctan2(
            velocity[1],
            velocity[0]
        )

    )


def rotation_to_heading(
    angle
):

    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )


    # Rotates heading angle to +X

    return np.array(
        [
            [c, s],
            [-s, c]
        ],
        dtype=np.float64
    )


# ============================================================
# CANONICAL TRANSFORM
# ============================================================

def canonicalize_sample(
    history,
    future,
    scale_floor
):

    origin = history[
        -1
    ].copy()


    angle = estimate_heading(
        history
    )


    R = rotation_to_heading(
        angle
    )


    path_length = history_path_length(
        history
    )


    scale = max(
        path_length,
        scale_floor
    )


    h_local = (
        history
        -
        origin
    )


    f_local = (
        future
        -
        origin
    )


    h_rot = (
        h_local
        @
        R.T
    )


    f_rot = (
        f_local
        @
        R.T
    )


    h_can = (
        h_rot
        /
        scale
    )


    f_can = (
        f_rot
        /
        scale
    )


    return {

        "history_canonical":
            h_can.astype(
                np.float32
            ),

        "future_canonical":
            f_can.astype(
                np.float32
            ),

        "origin":
            origin.astype(
                np.float32
            ),

        "angle":
            np.float32(
                angle
            ),

        "scale":
            np.float32(
                scale
            ),

        "raw_path_length":
            np.float32(
                path_length
            )
    }


# ============================================================
# LOAD DATASET
# ============================================================

def load_data(
    npz_path,
    meta_path
):

    arrays = np.load(
        npz_path
    )


    history = arrays[
        "history_abs"
    ].astype(
        np.float64
    )


    future = arrays[
        "future_abs"
    ].astype(
        np.float64
    )


    metadata = pd.read_csv(
        meta_path,
        dtype={
            "source_group": str,
            "track_key": str
        }
    )


    if len(history) != len(
        metadata
    ):

        raise RuntimeError(
            "Metadata/data length mismatch"
        )


    return (
        history,
        future,
        metadata
    )


# ============================================================
# TRANSFORM ENTIRE DATASET
# ============================================================

def transform_dataset(
    history,
    future,
    scale_floor
):

    hc = []
    fc = []

    origins = []
    angles = []
    scales = []
    raw_paths = []


    for i in range(
        len(history)
    ):


        result = canonicalize_sample(

            history[i],

            future[i],

            scale_floor

        )


        hc.append(
            result[
                "history_canonical"
            ]
        )


        fc.append(
            result[
                "future_canonical"
            ]
        )


        origins.append(
            result[
                "origin"
            ]
        )


        angles.append(
            result[
                "angle"
            ]
        )


        scales.append(
            result[
                "scale"
            ]
        )


        raw_paths.append(
            result[
                "raw_path_length"
            ]
        )


    return {

        "history_canonical":
            np.stack(
                hc
            ),

        "future_canonical":
            np.stack(
                fc
            ),

        "history_abs":
            history.astype(
                np.float32
            ),

        "future_abs":
            future.astype(
                np.float32
            ),

        "origin":
            np.stack(
                origins
            ),

        "heading_angle":
            np.asarray(
                angles,
                dtype=np.float32
            ),

        "scale":
            np.asarray(
                scales,
                dtype=np.float32
            ),

        "raw_path_length":
            np.asarray(
                raw_paths,
                dtype=np.float32
            )
    }


# ============================================================
# TRACK-DISJOINT VALIDATION SPLIT
# ============================================================

def create_track_split(
    metadata,
    seed
):

    rng = np.random.default_rng(
        seed
    )


    train_indices = []
    val_indices = []


    split_records = []


    # Split tracks separately within each source.
    # This preserves every possible TrainReal source
    # in training while preventing track overlap.


    for source, source_df in metadata.groupby(
        "source_group"
    ):


        tracks = np.asarray(
            sorted(
                source_df[
                    "track_key"
                ].astype(
                    str
                ).unique()
            )
        )


        shuffled = tracks.copy()

        rng.shuffle(
            shuffled
        )


        if len(
            shuffled
        ) >= 2:


            n_val = max(

                1,

                int(
                    round(
                        len(shuffled)
                        *
                        VAL_TRACK_FRACTION
                    )
                )

            )


            n_val = min(

                n_val,

                len(shuffled)
                -
                1

            )


        else:

            # Cannot place sole track in validation
            # without removing this source from training.

            n_val = 0


        val_tracks = set(
            shuffled[
                :n_val
            ]
        )


        train_tracks = set(
            shuffled[
                n_val:
            ]
        )


        source_train = source_df[
            source_df[
                "track_key"
            ].astype(str).isin(
                train_tracks
            )
        ].index.tolist()


        source_val = source_df[
            source_df[
                "track_key"
            ].astype(str).isin(
                val_tracks
            )
        ].index.tolist()


        train_indices.extend(
            source_train
        )


        val_indices.extend(
            source_val
        )


        split_records.append({

            "source_group":
                source,

            "total_tracks":
                len(
                    tracks
                ),

            "train_tracks":
                len(
                    train_tracks
                ),

            "val_tracks":
                len(
                    val_tracks
                ),

            "train_windows":
                len(
                    source_train
                ),

            "val_windows":
                len(
                    source_val
                )
        })


    train_indices = np.asarray(
        sorted(
            train_indices
        ),
        dtype=np.int64
    )


    val_indices = np.asarray(
        sorted(
            val_indices
        ),
        dtype=np.int64
    )


    # --------------------------------------------
    # Leakage check
    # --------------------------------------------

    train_tracks = set(

        metadata.iloc[
            train_indices
        ][
            "track_key"
        ].astype(str)

    )


    val_tracks = set(

        metadata.iloc[
            val_indices
        ][
            "track_key"
        ].astype(str)

    )


    overlap = (
        train_tracks
        &
        val_tracks
    )


    if overlap:

        raise RuntimeError(

            "TRACK LEAKAGE DETECTED: "
            +
            str(
                overlap
            )

        )


    return (
        train_indices,
        val_indices,
        pd.DataFrame(
            split_records
        )
    )


# ============================================================
# TRAIN SAMPLE WEIGHTS
# ============================================================

def make_track_weights(
    metadata,
    train_indices
):

    train_meta = (

        metadata.iloc[
            train_indices
        ]
        .copy()
    )


    counts = (

        train_meta[
            "track_key"
        ]

        .astype(str)

        .value_counts()

    )


    weights = []


    for idx in train_indices:


        track = str(

            metadata.iloc[
                idx
            ][
                "track_key"
            ]

        )


        # Each track contributes equal total weight.

        w = (
            1.0
            /
            counts[
                track
            ]
        )


        weights.append(
            w
        )


    weights = np.asarray(

        weights,

        dtype=np.float32

    )


    # normalize mean to 1

    weights = (

        weights

        /
        weights.mean()

    )


    return weights


# ============================================================
# MAIN
# ============================================================

summary_rows = []

split_rows = []


for task_index, (
    task,
    paths
) in enumerate(
    TASKS.items()
):


    print(
        "\n========================================"
    )

    print(
        task
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # LOAD BIRDSAI TRAIN
    # --------------------------------------------------------

    (
        train_history,
        train_future,
        train_meta

    ) = load_data(

        paths[
            "train_npz"
        ],

        paths[
            "train_meta"
        ]

    )


    # --------------------------------------------------------
    # SCALE FLOOR FROM TRAINREAL ONLY
    # --------------------------------------------------------

    train_path_lengths = np.asarray([

        history_path_length(
            h
        )

        for h
        in train_history

    ])


    scale_floor = float(

        np.quantile(

            train_path_lengths,

            SCALE_FLOOR_QUANTILE

        )

    )


    print(
        "Training-derived scale floor:",
        scale_floor
    )


    # --------------------------------------------------------
    # CANONICALIZE TRAIN
    # --------------------------------------------------------

    train_data = transform_dataset(

        train_history,

        train_future,

        scale_floor

    )


    # --------------------------------------------------------
    # TRACK-DISJOINT TRAIN / VAL
    # --------------------------------------------------------

    (
        train_idx,
        val_idx,
        split_table

    ) = create_track_split(

        train_meta,

        SEED
        +
        task_index

    )


    train_weights = make_track_weights(

        train_meta,

        train_idx

    )


    # Save split labels in metadata

    train_meta = (
        train_meta.copy()
    )


    train_meta[
        "neural_split"
    ] = "unused"


    train_meta.loc[
        train_idx,
        "neural_split"
    ] = "train"


    train_meta.loc[
        val_idx,
        "neural_split"
    ] = "validation"


    train_meta[
        "scale_floor"
    ] = scale_floor


    # --------------------------------------------------------
    # SAVE TRAINREAL
    # --------------------------------------------------------

    np.savez_compressed(

        OUT /
        f"{task}_birdsai_trainval_canonical.npz",

        **train_data,

        train_indices=
            train_idx,

        validation_indices=
            val_idx,

        train_sample_weights=
            train_weights
    )


    train_meta.to_csv(

        OUT /
        f"{task}_birdsai_trainval_metadata.csv",

        index=False
    )


    split_table[
        "task"
    ] = task


    split_rows.append(
        split_table
    )


    # --------------------------------------------------------
    # TESTREAL
    # --------------------------------------------------------

    (
        test_history,
        test_future,
        test_meta

    ) = load_data(

        paths[
            "test_npz"
        ],

        paths[
            "test_meta"
        ]

    )


    test_data = transform_dataset(

        test_history,

        test_future,

        scale_floor

    )


    np.savez_compressed(

        OUT /
        f"{task}_birdsai_test_canonical.npz",

        **test_data
    )


    test_meta.to_csv(

        OUT /
        f"{task}_birdsai_test_metadata.csv",

        index=False
    )


    # --------------------------------------------------------
    # WILDBOX EXTERNAL
    # --------------------------------------------------------

    (
        wild_history,
        wild_future,
        wild_meta

    ) = load_data(

        paths[
            "wild_npz"
        ],

        paths[
            "wild_meta"
        ]

    )


    wild_data = transform_dataset(

        wild_history,

        wild_future,

        scale_floor

    )


    np.savez_compressed(

        OUT /
        f"{task}_wildbox_external_canonical.npz",

        **wild_data
    )


    wild_meta.to_csv(

        OUT /
        f"{task}_wildbox_external_metadata.csv",

        index=False
    )


    # --------------------------------------------------------
    # FINAL LEAKAGE CHECK
    # --------------------------------------------------------

    tr_tracks = set(

        train_meta.iloc[
            train_idx
        ][
            "track_key"
        ].astype(str)

    )


    va_tracks = set(

        train_meta.iloc[
            val_idx
        ][
            "track_key"
        ].astype(str)

    )


    overlap = len(
        tr_tracks
        &
        va_tracks
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows.extend([

        {

            "task":
                task,

            "dataset":
                "BIRDSAI",

            "split":
                "neural_train",

            "samples":
                len(
                    train_idx
                ),

            "tracks":
                train_meta.iloc[
                    train_idx
                ][
                    "track_key"
                ].nunique(),

            "sources":
                train_meta.iloc[
                    train_idx
                ][
                    "source_group"
                ].nunique(),

            "scale_floor":
                scale_floor,

            "track_overlap_with_validation":
                overlap
        },

        {

            "task":
                task,

            "dataset":
                "BIRDSAI",

            "split":
                "neural_validation",

            "samples":
                len(
                    val_idx
                ),

            "tracks":
                train_meta.iloc[
                    val_idx
                ][
                    "track_key"
                ].nunique(),

            "sources":
                train_meta.iloc[
                    val_idx
                ][
                    "source_group"
                ].nunique(),

            "scale_floor":
                scale_floor,

            "track_overlap_with_validation":
                overlap
        },

        {

            "task":
                task,

            "dataset":
                "BIRDSAI",

            "split":
                "untouched_test",

            "samples":
                len(
                    test_history
                ),

            "tracks":
                test_meta[
                    "track_key"
                ].nunique(),

            "sources":
                test_meta[
                    "source_group"
                ].nunique(),

            "scale_floor":
                scale_floor,

            "track_overlap_with_validation":
                np.nan
        },

        {

            "task":
                task,

            "dataset":
                "WildBox",

            "split":
                "external_test",

            "samples":
                len(
                    wild_history
                ),

            "tracks":
                wild_meta[
                    "track_key"
                ].nunique(),

            "sources":
                wild_meta[
                    "source_group"
                ].nunique(),

            "scale_floor":
                scale_floor,

            "track_overlap_with_validation":
                np.nan
        }
    ])


# ============================================================
# SAVE SUMMARIES
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(

    OUT /
    "neural_dataset_summary.csv",

    index=False
)


split_df = pd.concat(

    split_rows,

    ignore_index=True

)


split_df.to_csv(

    OUT /
    "train_validation_split_summary.csv",

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
        "FINAL NEURAL TRAJECTORY DATASET\n"
    )

    f.write(
        "=" * 76
        +
        "\n\n"
    )


    f.write(
        "Input representation:\n"
    )

    f.write(
        "1. Camera-motion corrected normalized x,y\n"
    )

    f.write(
        "2. 10 Hz sampling\n"
    )

    f.write(
        "3. Last observed point translated to origin\n"
    )

    f.write(
        "4. Recent heading aligned to +X\n"
    )

    f.write(
        "5. Divided by observed history path length\n"
    )

    f.write(
        "6. Scale floor = TrainReal 5th percentile "
        "of history path length\n\n"
    )


    f.write(
        "Important:\n"
    )

    f.write(
        "- Scale floor derived from BIRDSAI TrainReal ONLY.\n"
    )

    f.write(
        "- Validation split is TRACK-DISJOINT.\n"
    )

    f.write(
        "- BIRDSAI TestReal remains untouched.\n"
    )

    f.write(
        "- WildBox remains untouched external test.\n"
    )

    f.write(
        "- Final ADE/FDE must be evaluated after inverse "
        "transform in original corrected normalized x,y.\n"
    )

    f.write(
        "- Training sample weights equalize total "
        "contribution of each training track.\n\n"
    )


    for task in TASKS.keys():


        temp = summary_df[
            summary_df[
                "task"
            ]
            ==
            task
        ]


        floor = temp[
            "scale_floor"
        ].iloc[0]


        f.write(
            f"{task}\n"
        )


        f.write(
            f"Training-derived scale floor: "
            f"{floor:.8f}\n"
        )


        for _, row in (
            temp.iterrows()
        ):


            f.write(

                f"  {row['dataset']} | "
                f"{row['split']} | "
                f"samples={int(row['samples'])}, "
                f"tracks={int(row['tracks'])}, "
                f"sources={int(row['sources'])}\n"

            )


        f.write(
            "\n"
        )


print(
    "\n============================================"
)

print(
    "FINAL NEURAL DATASET BUILD COMPLETE"
)

print(
    "============================================"
)

print(
    "\nDataset summary:"
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\nPer-source split:"
)

print(
    split_df.to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    OUT
)