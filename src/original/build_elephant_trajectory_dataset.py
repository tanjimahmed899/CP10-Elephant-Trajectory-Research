from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================

TRAIN_ROOT = Path(
    r"C:\Users\HP\Downloads\conservation_drones_train_real\TrainReal"
)

TEST_ROOT = Path(
    r"C:\Users\HP\Downloads\conservation_drones_test_real\TestReal"
)

OUT = Path(
    r"C:\Users\HP\Downloads\trajectory_dataset"
)

OUT.mkdir(exist_ok=True)

# =========================
# SETTINGS
# =========================

FPS = 30

OBS_SEC = 2
OBS_FRAMES = OBS_SEC * FPS       # 60 frames

HORIZONS = {
    2: 60,
    5: 150
}

# Missing <=3 frames -> interpolate
# Missing >3 frames -> split track
MAX_SMALL_GAP = 3

# Generate one sample every 1 second
WINDOW_STRIDE = 30

COLS = [
    "frame",
    "object_id",
    "x",
    "y",
    "w",
    "h",
    "class",
    "species",
    "occlusion",
    "noise"
]


# =========================
# IMAGE SIZE
# =========================

def get_image_size(images_root, sequence):

    folder = images_root / sequence

    if not folder.exists():
        return None, None

    for file in folder.iterdir():

        if file.suffix.lower() in [
            ".jpg", ".jpeg", ".png"
        ]:

            img = cv2.imread(str(file))

            if img is not None:

                h, w = img.shape[:2]

                return w, h

    return None, None


# =========================
# TRACK SPLITTING
# =========================

def split_track(track):

    track = (
        track
        .sort_values("frame")
        .drop_duplicates("frame")
        .reset_index(drop=True)
    )

    segments = []

    start = 0

    frames = track["frame"].values

    for i in range(1, len(frames)):

        missing = (
            frames[i]
            - frames[i - 1]
            - 1
        )

        if missing > MAX_SMALL_GAP:

            segments.append(
                track.iloc[start:i].copy()
            )

            start = i

    segments.append(
        track.iloc[start:].copy()
    )

    return segments


# =========================
# SMALL GAP INTERPOLATION
# =========================

def interpolate_segment(segment):

    segment = segment.copy()

    original_frames = set(
        segment["frame"].astype(int)
    )

    first_frame = int(
        segment["frame"].min()
    )

    last_frame = int(
        segment["frame"].max()
    )

    full_frames = range(
        first_frame,
        last_frame + 1
    )

    segment = (
        segment
        .set_index("frame")
        .reindex(full_frames)
    )

    # geometry
    for col in ["x", "y", "w", "h"]:

        segment[col] = (
            segment[col]
            .interpolate()
            .ffill()
            .bfill()
        )

    # identity
    for col in [
        "object_id",
        "class",
        "species"
    ]:

        segment[col] = (
            segment[col]
            .ffill()
            .bfill()
        )

    # quality flags
    for col in [
        "occlusion",
        "noise"
    ]:

        segment[col] = (
            segment[col]
            .ffill()
            .bfill()
            .fillna(0)
        )

    segment = segment.reset_index()

    segment = segment.rename(
        columns={"index": "frame"}
    )

    segment["interpolated"] = (
        ~segment["frame"].isin(
            original_frames
        )
    ).astype(int)

    return segment


# =========================
# FEATURE CREATION
# =========================

def add_motion_features(
    segment,
    img_w,
    img_h
):

    segment = segment.copy()

    # Bounding box center
    segment["cx"] = (
        segment["x"]
        + segment["w"] / 2
    )

    segment["cy"] = (
        segment["y"]
        + segment["h"] / 2
    )

    # Normalized coordinates
    segment["cx_n"] = (
        segment["cx"] / img_w
    )

    segment["cy_n"] = (
        segment["cy"] / img_h
    )

    segment["w_n"] = (
        segment["w"] / img_w
    )

    segment["h_n"] = (
        segment["h"] / img_h
    )

    # Velocity
    segment["vx_n"] = (
        segment["cx_n"]
        .diff()
        .fillna(0)
    )

    segment["vy_n"] = (
        segment["cy_n"]
        .diff()
        .fillna(0)
    )

    # Speed
    segment["speed_n"] = np.sqrt(

        segment["vx_n"] ** 2
        +
        segment["vy_n"] ** 2
    )

    # Heading angle
    segment["heading_rad"] = np.arctan2(

        segment["vy_n"],
        segment["vx_n"]
    )

    # Acceleration
    segment["accel_n"] = (
        segment["speed_n"]
        .diff()
        .fillna(0)
    )

    return segment


# =========================
# WINDOW GENERATION
# =========================

def make_windows(
    segment,
    split,
    sequence,
    object_id,
    segment_id,
    horizon_sec
):

    rows = []

    future_frames = HORIZONS[
        horizon_sec
    ]

    minimum_length = (
        OBS_FRAMES
        +
        future_frames
    )

    if len(segment) < minimum_length:
        return rows

    max_start = (
        len(segment)
        -
        minimum_length
    )

    for start in range(
        0,
        max_start + 1,
        WINDOW_STRIDE
    ):

        obs_end = (
            start
            +
            OBS_FRAMES
            - 1
        )

        future_end = (
            obs_end
            +
            future_frames
        )

        current = segment.iloc[
            obs_end
        ]

        future = segment.iloc[
            future_end
        ]

        history = segment.iloc[
            start:obs_end + 1
        ]

        rows.append({

            "split":
                split,

            "sequence":
                sequence,

            "object_id":
                object_id,

            "segment_id":
                segment_id,

            "obs_start_frame":
                int(
                    segment.iloc[start][
                        "frame"
                    ]
                ),

            "obs_end_frame":
                int(
                    current["frame"]
                ),

            "future_end_frame":
                int(
                    future["frame"]
                ),

            "horizon_sec":
                horizon_sec,

            "current_x":
                current["cx_n"],

            "current_y":
                current["cy_n"],

            "future_x":
                future["cx_n"],

            "future_y":
                future["cy_n"],

            "current_speed":
                current["speed_n"],

            "current_heading":
                current["heading_rad"],

            "occlusion_pct":
                (
                    history["occlusion"]
                    == 1
                ).mean() * 100,

            "noise_pct":
                (
                    history["noise"]
                    == 1
                ).mean() * 100,

            "interpolated_pct":
                (
                    history[
                        "interpolated"
                    ]
                    == 1
                ).mean() * 100
        })

    return rows


# =========================
# PROCESS SPLIT
# =========================

def process_dataset(
    root,
    split_name
):

    annotation_dir = (
        root / "annotations"
    )

    image_dir = (
        root / "images"
    )

    cleaned_points = []

    segment_summary = []

    windows_2s = []

    windows_5s = []

    sequence_summary = []


    for csv_file in sorted(
        annotation_dir.glob("*.csv")
    ):

        sequence = csv_file.stem

        df = pd.read_csv(
            csv_file,
            header=None,
            names=COLS
        )

        # elephant only
        df = df[
            df["species"] == 1
        ].copy()

        if len(df) == 0:
            continue

        img_w, img_h = (
            get_image_size(
                image_dir,
                sequence
            )
        )

        if img_w is None:

            print(
                "Image size missing:",
                sequence
            )

            continue

        sequence_segments = 0

        sequence_w2 = 0

        sequence_w5 = 0


        for object_id, track in (
            df.groupby("object_id")
        ):

            parts = split_track(
                track
            )

            for number, part in enumerate(
                parts,
                start=1
            ):

                if len(part) == 0:
                    continue

                segment = (
                    interpolate_segment(
                        part
                    )
                )

                segment = (
                    add_motion_features(
                        segment,
                        img_w,
                        img_h
                    )
                )

                segment_id = (
                    f"{split_name}_"
                    f"{sequence}_"
                    f"obj{object_id}_"
                    f"seg{number}"
                )

                segment["split"] = (
                    split_name
                )

                segment["sequence"] = (
                    sequence
                )

                segment["segment_id"] = (
                    segment_id
                )

                cleaned_points.append(
                    segment
                )

                # segment stats
                segment_summary.append({

                    "split":
                        split_name,

                    "sequence":
                        sequence,

                    "object_id":
                        object_id,

                    "segment_id":
                        segment_id,

                    "frames":
                        len(segment),

                    "duration_sec":
                        len(segment) / FPS,

                    "occlusion_pct":
                        (
                            segment[
                                "occlusion"
                            ] == 1
                        ).mean() * 100,

                    "noise_pct":
                        (
                            segment[
                                "noise"
                            ] == 1
                        ).mean() * 100,

                    "interpolated_pct":
                        (
                            segment[
                                "interpolated"
                            ] == 1
                        ).mean() * 100
                })

                sequence_segments += 1

                w2 = make_windows(
                    segment,
                    split_name,
                    sequence,
                    object_id,
                    segment_id,
                    2
                )

                w5 = make_windows(
                    segment,
                    split_name,
                    sequence,
                    object_id,
                    segment_id,
                    5
                )

                windows_2s.extend(w2)
                windows_5s.extend(w5)

                sequence_w2 += len(w2)
                sequence_w5 += len(w5)


        sequence_summary.append({

            "split":
                split_name,

            "sequence":
                sequence,

            "elephant_boxes":
                len(df),

            "segments":
                sequence_segments,

            "windows_2s_to_2s":
                sequence_w2,

            "windows_2s_to_5s":
                sequence_w5
        })


    return (

        cleaned_points,

        segment_summary,

        windows_2s,

        windows_5s,

        sequence_summary
    )


# =========================
# RUN
# =========================

train = process_dataset(
    TRAIN_ROOT,
    "train_real"
)

test = process_dataset(
    TEST_ROOT,
    "test_real"
)


# =========================
# COMBINE
# =========================

cleaned = pd.concat(

    [
        pd.concat(
            train[0],
            ignore_index=True
        ),

        pd.concat(
            test[0],
            ignore_index=True
        )
    ],

    ignore_index=True
)


segments = pd.DataFrame(
    train[1] + test[1]
)

windows2 = pd.DataFrame(
    train[2] + test[2]
)

windows5 = pd.DataFrame(
    train[3] + test[3]
)

sequence_stats = pd.DataFrame(
    train[4] + test[4]
)


# =========================
# SAVE
# =========================

cleaned.to_csv(

    OUT /
    "cleaned_track_points.csv",

    index=False
)


segments.to_csv(

    OUT /
    "track_segments.csv",

    index=False
)


windows2.to_csv(

    OUT /
    "windows_2s_to_2s.csv",

    index=False
)


windows5.to_csv(

    OUT /
    "windows_2s_to_5s.csv",

    index=False
)


sequence_stats.to_csv(

    OUT /
    "per_sequence_statistics.csv",

    index=False
)


# =========================
# SUMMARY
# =========================

with open(
    OUT / "SUMMARY.txt",
    "w"
) as f:

    f.write(
        "TRAJECTORY DATASET SUMMARY\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    for split in [
        "train_real",
        "test_real"
    ]:

        seg = segments[
            segments["split"] == split
        ]

        seq = sequence_stats[
            sequence_stats["split"]
            == split
        ]

        w2 = windows2[
            windows2["split"] == split
        ]

        w5 = windows5[
            windows5["split"] == split
        ]

        f.write(
            split.upper()
            + "\n"
        )

        f.write(
            f"Sequences: {len(seq)}\n"
        )

        f.write(
            f"Continuous segments: "
            f"{len(seg)}\n"
        )

        if len(seg) > 0:

            f.write(
                "Segment duration sec "
                "(min / median / max): "
            )

            f.write(
                f"{seg['duration_sec'].min():.2f} / "
                f"{seg['duration_sec'].median():.2f} / "
                f"{seg['duration_sec'].max():.2f}\n"
            )

        f.write(
            f"2s history -> 2s future windows: "
            f"{len(w2)}\n"
        )

        f.write(
            f"2s history -> 5s future windows: "
            f"{len(w5)}\n\n"
        )


print("\nDONE")

print(
    "\nOutput folder:"
)

print(OUT)