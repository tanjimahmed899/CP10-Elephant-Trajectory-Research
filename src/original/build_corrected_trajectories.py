from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import math


# ==========================================================
# PATHS
# ==========================================================

MOTION_FILE = Path(
    r"C:\Users\HP\Downloads\ego_motion_full\all_ego_motion.csv"
)

TRACK_FILE = Path(
    r"C:\Users\HP\Downloads\trajectory_dataset\cleaned_track_points.csv"
)

WINDOW_2_FILE = Path(
    r"C:\Users\HP\Downloads\trajectory_dataset\windows_2s_to_2s.csv"
)

WINDOW_5_FILE = Path(
    r"C:\Users\HP\Downloads\trajectory_dataset\windows_2s_to_5s.csv"
)

TRAIN_IMAGES = Path(
    r"C:\Users\HP\Downloads\conservation_drones_train_real\TrainReal\images"
)

TEST_IMAGES = Path(
    r"C:\Users\HP\Downloads\conservation_drones_test_real\TestReal\images"
)

OUT = Path(
    r"C:\Users\HP\Downloads\corrected_trajectory_dataset"
)

OUT.mkdir(exist_ok=True)


# ==========================================================
# SETTINGS
# ==========================================================

FPS = 30

OBS_FRAMES = 60

MAX_REPAIR_RUN = 5

# Core = cleaner samples
CORE_MAX_REPAIRED_PCT = 10

# Extended = still usable
EXTENDED_MAX_REPAIRED_PCT = 20


# ==========================================================
# LOAD DATA
# ==========================================================

motion = pd.read_csv(
    MOTION_FILE,
    dtype={"sequence": str}
)

tracks = pd.read_csv(
    TRACK_FILE,
    dtype={
        "sequence": str,
        "segment_id": str
    }
)

windows2 = pd.read_csv(
    WINDOW_2_FILE,
    dtype={
        "sequence": str,
        "segment_id": str
    }
)

windows5 = pd.read_csv(
    WINDOW_5_FILE,
    dtype={
        "sequence": str,
        "segment_id": str
    }
)


# motion file uses train/test
motion["split"] = motion["split"].replace({
    "train": "train_real",
    "test": "test_real"
})


# ==========================================================
# IMAGE SIZE
# ==========================================================

size_cache = {}


def get_image_size(split, sequence):

    key = (split, sequence)

    if key in size_cache:
        return size_cache[key]

    if split == "train_real":

        folder = (
            TRAIN_IMAGES
            /
            sequence
        )

    else:

        folder = (
            TEST_IMAGES
            /
            sequence
        )

    for file in sorted(folder.iterdir()):

        if file.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:

            img = cv2.imread(str(file))

            if img is not None:

                h, w = img.shape[:2]

                size_cache[key] = (
                    w,
                    h
                )

                return w, h

    raise RuntimeError(
        f"Image size unavailable: "
        f"{split} {sequence}"
    )


# ==========================================================
# REPAIR SHORT MOTION FAILURES
# ==========================================================

def repair_motion_group(group):

    group = (
        group
        .sort_values("pair")
        .copy()
    )

    first_pair = int(
        group["pair"].min()
    )

    last_pair = int(
        group["pair"].max()
    )

    group = (
        group
        .set_index("pair")
        .reindex(
            range(
                first_pair,
                last_pair + 1
            )
        )
    )

    group.index.name = "pair"


    params = [

        "dx_pixels",

        "dy_pixels",

        "rotation_deg",

        "scale"
    ]


    reliable = (

        group["reliable"]
        .fillna(0)
        .astype(int)
        ==
        1
    )


    group["motion_status"] = (
        "invalid"
    )


    group.loc[
        reliable,
        "motion_status"
    ] = "original"


    # --------------------------------------------
    # Find runs where transform is unreliable
    # --------------------------------------------

    bad = ~reliable

    indices = list(
        group.index
    )

    i = 0


    while i < len(indices):

        pair = indices[i]

        if not bad.loc[pair]:

            i += 1

            continue


        start_i = i


        while (

            i < len(indices)

            and

            bad.loc[
                indices[i]
            ]

        ):

            i += 1


        end_i = i - 1

        run_pairs = indices[
            start_i:end_i + 1
        ]

        run_length = len(
            run_pairs
        )


        if run_length > MAX_REPAIR_RUN:

            continue


        # Need reliable transform
        # on both sides.

        left_i = (
            start_i - 1
        )

        right_i = i


        if (

            left_i < 0

            or

            right_i >= len(indices)

        ):

            continue


        left_pair = indices[
            left_i
        ]

        right_pair = indices[
            right_i
        ]


        if (

            not reliable.loc[
                left_pair
            ]

            or

            not reliable.loc[
                right_pair
            ]

        ):

            continue


        # ------------------------------------------
        # Linear interpolation
        # ------------------------------------------

        for pair_number in run_pairs:

            alpha = (

                pair_number
                -
                left_pair

            ) / (

                right_pair
                -
                left_pair

            )


            for p in params:

                left_value = group.loc[
                    left_pair,
                    p
                ]

                right_value = group.loc[
                    right_pair,
                    p
                ]


                group.loc[
                    pair_number,
                    p
                ] = (

                    left_value

                    +

                    alpha
                    *
                    (
                        right_value
                        -
                        left_value
                    )

                )


            group.loc[
                pair_number,
                "motion_status"
            ] = "repaired"


    group = (
        group
        .reset_index()
    )


    group["usable_motion"] = (

        group["motion_status"]
        .isin([
            "original",
            "repaired"
        ])

    ).astype(int)


    return group


# ==========================================================
# BUILD REPAIRED MOTION DATABASE
# ==========================================================

repaired_groups = []

motion_lookup = {}


for (
    split,
    sequence
), group in motion.groupby(
    [
        "split",
        "sequence"
    ]
):

    repaired = (
        repair_motion_group(
            group
        )
    )

    repaired["split"] = split

    repaired["sequence"] = sequence


    repaired_groups.append(
        repaired
    )


    motion_lookup[
        (
            split,
            sequence
        )
    ] = (
        repaired
        .set_index("pair")
    )


repaired_motion = pd.concat(
    repaired_groups,
    ignore_index=True
)


repaired_motion.to_csv(

    OUT
    /
    "repaired_ego_motion.csv",

    index=False
)


# ==========================================================
# TRACK LOOKUP
# ==========================================================

track_lookup = {}


for (
    split,
    sequence,
    segment_id
), group in tracks.groupby(

    [
        "split",
        "sequence",
        "segment_id"
    ]

):

    group = (
        group
        .sort_values("frame")
        .set_index("frame")
    )


    track_lookup[
        (
            split,
            sequence,
            segment_id
        )
    ] = group


# ==========================================================
# AFFINE MATRIX
# ==========================================================

def motion_matrix(row):

    theta = math.radians(
        float(
            row["rotation_deg"]
        )
    )

    scale = float(
        row["scale"]
    )

    dx = float(
        row["dx_pixels"]
    )

    dy = float(
        row["dy_pixels"]
    )


    c = math.cos(
        theta
    )

    s = math.sin(
        theta
    )


    M = np.array([

        [
            scale * c,
            -scale * s,
            dx
        ],

        [
            scale * s,
            scale * c,
            dy
        ],

        [
            0.0,
            0.0,
            1.0
        ]

    ], dtype=np.float64)


    return M


# ==========================================================
# CORRECT ONE WINDOW
# ==========================================================

def correct_window(row, horizon_frames):

    split = row["split"]

    sequence = str(
        row["sequence"]
    )

    segment_id = str(
        row["segment_id"]
    )


    start_frame = int(
        row["obs_start_frame"]
    )

    obs_end = int(
        row["obs_end_frame"]
    )

    future_end = int(
        row["future_end_frame"]
    )


    motion_key = (
        split,
        sequence
    )

    track_key = (
        split,
        sequence,
        segment_id
    )


    if motion_key not in motion_lookup:

        return None


    if track_key not in track_lookup:

        return None


    mg = motion_lookup[
        motion_key
    ]

    tg = track_lookup[
        track_key
    ]


    required_pairs = list(
        range(
            start_frame,
            future_end
        )
    )


    # Must have every transform

    if not all(
        p in mg.index
        for p in required_pairs
    ):

        return None


    motion_slice = mg.loc[
        required_pairs
    ]


    if (
        motion_slice[
            "usable_motion"
        ]
        .min()
        !=
        1
    ):

        return None


    repaired_pct = (

        100
        *
        (
            motion_slice[
                "motion_status"
            ]
            ==
            "repaired"
        ).mean()

    )


    if (
        repaired_pct
        <=
        CORE_MAX_REPAIRED_PCT
    ):

        tier = "core"


    elif (
        repaired_pct
        <=
        EXTENDED_MAX_REPAIRED_PCT
    ):

        tier = "extended"


    else:

        tier = "reject_quality"


    if tier == "reject_quality":

        return {
            "accepted": False,
            "tier": tier,
            "repaired_pct": repaired_pct
        }


    required_frames = list(
        range(
            start_frame,
            future_end + 1
        )
    )


    if not all(
        f in tg.index
        for f in required_frames
    ):

        return None


    width, height = get_image_size(
        split,
        sequence
    )


    corrected = []


    # T maps current frame coordinates
    # back to observation-start frame.

    T = np.eye(
        3,
        dtype=np.float64
    )


    for frame in required_frames:

        point_row = tg.loc[
            frame
        ]


        point = np.array([

            float(
                point_row["cx"]
            ),

            float(
                point_row["cy"]
            ),

            1.0

        ])


        corrected_point = (
            T
            @
            point
        )


        corrected.append([

            corrected_point[0]
            /
            width,

            corrected_point[1]
            /
            height,

            float(
                point_row["w_n"]
            ),

            float(
                point_row["h_n"]
            ),

            float(
                point_row["occlusion"]
            ),

            float(
                point_row["noise"]
            ),

            float(
                point_row["interpolated"]
            )
        ])


        # transform from next frame
        # into this frame

        if frame < future_end:

            motion_row = mg.loc[
                frame
            ]

            M = motion_matrix(
                motion_row
            )

            T = (
                T
                @
                M
            )


    corrected = np.array(
        corrected,
        dtype=np.float32
    )


    # History includes 60 frames.

    history = corrected[
        :OBS_FRAMES
    ]


    # Future starts immediately
    # after observation window.

    future = corrected[
        OBS_FRAMES:
        OBS_FRAMES
        +
        horizon_frames
    ]


    if (

        len(history)
        != OBS_FRAMES

        or

        len(future)
        != horizon_frames

    ):

        return None


    return {

        "accepted":
            True,

        "tier":
            tier,

        "repaired_pct":
            repaired_pct,

        "history":
            history,

        "future":
            future
    }


# ==========================================================
# PROCESS WINDOWS
# ==========================================================

def process_windows(
    window_df,
    horizon_sec
):

    horizon_frames = (
        horizon_sec
        *
        FPS
    )


    histories = []

    futures = []

    metadata = []


    for i, row in (
        window_df.iterrows()
    ):


        result = correct_window(
            row,
            horizon_frames
        )


        if (
            result is None
        ):

            accepted = False

            tier = "invalid_motion"

            repaired_pct = np.nan


        else:

            accepted = bool(
                result["accepted"]
            )

            tier = result[
                "tier"
            ]

            repaired_pct = result[
                "repaired_pct"
            ]


        meta = {

            "source_index":
                i,

            "split":
                row["split"],

            "sequence":
                row["sequence"],

            "object_id":
                row["object_id"],

            "segment_id":
                row["segment_id"],

            "obs_start_frame":
                row[
                    "obs_start_frame"
                ],

            "obs_end_frame":
                row[
                    "obs_end_frame"
                ],

            "future_end_frame":
                row[
                    "future_end_frame"
                ],

            "horizon_sec":
                horizon_sec,

            "accepted":
                int(accepted),

            "quality_tier":
                tier,

            "repaired_motion_pct":
                repaired_pct
        }


        if accepted:

            sample_id = len(
                histories
            )

            meta[
                "sample_id"
            ] = sample_id


            histories.append(
                result["history"]
            )

            futures.append(
                result["future"]
            )


        else:

            meta[
                "sample_id"
            ] = -1


        metadata.append(
            meta
        )


    meta_df = pd.DataFrame(
        metadata
    )


    if len(histories) > 0:

        history_array = np.stack(
            histories
        )

        future_array = np.stack(
            futures
        )


    else:

        history_array = np.empty(
            (
                0,
                OBS_FRAMES,
                7
            ),
            dtype=np.float32
        )

        future_array = np.empty(
            (
                0,
                horizon_frames,
                7
            ),
            dtype=np.float32
        )


    return (
        meta_df,
        history_array,
        future_array
    )


# ==========================================================
# 2 SECOND FORECAST
# ==========================================================

meta2, hist2, future2 = (

    process_windows(
        windows2,
        2
    )

)


meta2.to_csv(

    OUT
    /
    "corrected_windows_2s_to_2s.csv",

    index=False
)


np.savez_compressed(

    OUT
    /
    "corrected_2s_to_2s.npz",

    history=hist2,

    future=future2
)


# ==========================================================
# 5 SECOND FORECAST
# ==========================================================

meta5, hist5, future5 = (

    process_windows(
        windows5,
        5
    )

)


meta5.to_csv(

    OUT
    /
    "corrected_windows_2s_to_5s.csv",

    index=False
)


np.savez_compressed(

    OUT
    /
    "corrected_2s_to_5s.npz",

    history=hist5,

    future=future5
)


# ==========================================================
# SUMMARY
# ==========================================================

summary_rows = []


for label, df in [

    (
        "2s_to_2s",
        meta2
    ),

    (
        "2s_to_5s",
        meta5
    )

]:

    for split in [

        "train_real",
        "test_real"

    ]:

        temp = df[
            df["split"]
            ==
            split
        ]


        accepted = temp[
            temp["accepted"]
            ==
            1
        ]


        core = accepted[
            accepted[
                "quality_tier"
            ]
            ==
            "core"
        ]


        extended = accepted[
            accepted[
                "quality_tier"
            ]
            ==
            "extended"
        ]


        summary_rows.append({

            "forecast":
                label,

            "split":
                split,

            "total":
                len(temp),

            "accepted":
                len(accepted),

            "accepted_pct":
                (
                    100
                    *
                    len(accepted)
                    /
                    len(temp)
                )
                if len(temp)
                else 0,

            "core":
                len(core),

            "extended":
                len(extended),

            "rejected":
                (
                    len(temp)
                    -
                    len(accepted)
                )
        })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(

    OUT
    /
    "correction_summary.csv",

    index=False
)


with open(

    OUT
    /
    "SUMMARY.txt",

    "w"

) as f:


    f.write(
        "CAMERA-CORRECTED TRAJECTORY SUMMARY\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )


    f.write(
        "Correction strategy:\n"
    )

    f.write(
        "KLT+RANSAC reliable transforms retained.\n"
    )

    f.write(
        "Unreliable runs <=5 frames "
        "interpolated between reliable transforms.\n"
    )

    f.write(
        "Longer/unresolved transform gaps rejected.\n"
    )

    f.write(
        "Each forecast window uses its own "
        "observation-start frame as the local anchor.\n\n"
    )


    for _, row in (
        summary.iterrows()
    ):

        f.write(

            f"{row['forecast']} "
            f"{row['split']}\n"
        )

        f.write(

            f"Total: "
            f"{int(row['total'])}\n"
        )

        f.write(

            f"Accepted: "
            f"{int(row['accepted'])} "
            f"({row['accepted_pct']:.2f}%)\n"
        )

        f.write(

            f"Core: "
            f"{int(row['core'])}\n"
        )

        f.write(

            f"Extended: "
            f"{int(row['extended'])}\n"
        )

        f.write(

            f"Rejected: "
            f"{int(row['rejected'])}\n\n"
        )


print(
    "\nCAMERA CORRECTION COMPLETE"
)

print(summary)

print(
    "\nOutput folder:"
)

print(OUT)