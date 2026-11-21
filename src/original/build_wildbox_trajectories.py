from pathlib import Path
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\HP\Downloads"
)

ZIP_FILES = [

    ROOT /
    "DJI_20240119152552_0006_V.zip",

    ROOT /
    "DJI_20240626143832_0004_V.zip",

    ROOT /
    "DJI_20260227084907_0004_V.zip"
]


OUT = ROOT / "wildbox_trajectory_dataset"

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

FPS = 10

OBS_SEC = 2
OBS_FRAMES = 20

HORIZONS = {

    2: 20,

    5: 50
}

# one new sample every second
STRIDE = 10


# ============================================================
# CAMERA FUNCTIONS
# ============================================================

def world_from_camera(
    center_camera,
    extrinsic
):

    E = np.asarray(
        extrinsic,
        dtype=np.float64
    )


    if E.shape == (4, 4):

        E = E[:3, :]


    if E.shape != (3, 4):

        raise ValueError(
            f"Unexpected extrinsic shape: "
            f"{E.shape}"
        )


    R = E[:, :3]

    t = E[:, 3]


    X_cam = np.asarray(
        center_camera,
        dtype=np.float64
    )


    # X_cam = R X_world + t

    X_world = np.linalg.solve(

        R,

        X_cam - t

    )


    return X_world


def camera_from_world(
    point_world,
    extrinsic
):

    E = np.asarray(
        extrinsic,
        dtype=np.float64
    )


    if E.shape == (4, 4):

        E = E[:3, :]


    R = E[:, :3]

    t = E[:, 3]


    X_world = np.asarray(
        point_world,
        dtype=np.float64
    )


    return (
        R @ X_world
        +
        t
    )


def project_point(
    point_camera,
    intrinsic,
    width,
    height
):

    K = np.asarray(
        intrinsic,
        dtype=np.float64
    )


    X = np.asarray(
        point_camera,
        dtype=np.float64
    )


    if X[2] <= 1e-8:

        return None


    p = (
        K
        @
        X
    )


    u = (
        p[0]
        /
        p[2]
    )

    v = (
        p[1]
        /
        p[2]
    )


    return (

        u / width,

        v / height,

        X[2]
    )


# ============================================================
# CONTINUOUS RUN SPLITTING
# ============================================================

def split_continuous(
    frames
):

    runs = []

    current = []


    for frame in frames:

        frame = int(
            frame
        )


        if not current:

            current = [
                frame
            ]

            continue


        if (
            frame
            ==
            current[-1] + 1
        ):

            current.append(
                frame
            )

        else:

            runs.append(
                current
            )

            current = [
                frame
            ]


    if current:

        runs.append(
            current
        )


    return runs


# ============================================================
# OUTPUT CONTAINERS
# ============================================================

all_track_rows = []

window_metadata = {

    2: [],

    5: []
}


window_history = {

    2: [],

    5: []
}


window_future = {

    2: [],

    5: []
}


# ============================================================
# PROCESS ZIPS
# ============================================================

for zip_path in ZIP_FILES:


    if not zip_path.exists():

        print(
            "MISSING:",
            zip_path
        )

        continue


    video = zip_path.stem


    print(
        "\nProcessing:",
        video
    )


    with ZipFile(
        zip_path,
        "r"
    ) as z:


        names = z.namelist()


        tracking_files = sorted([

            name

            for name
            in names

            if name.endswith(
                "/tracking_summary.json"
            )

        ])


        for tracking_path in (
            tracking_files
        ):


            segment = Path(
                tracking_path
            ).parent.name


            camera_path = (

                str(
                    Path(
                        tracking_path
                    ).parent
                )
                .replace(
                    "\\",
                    "/"
                )

                +
                "/cameras.json"
            )


            if camera_path not in names:

                print(
                    "Camera metadata missing:",
                    video,
                    segment
                )

                continue


            tracking = json.loads(

                z.read(
                    tracking_path
                ).decode(
                    "utf-8"
                )

            )


            cameras_json = json.loads(

                z.read(
                    camera_path
                ).decode(
                    "utf-8"
                )

            )


            camera_lookup = {

                int(
                    camera[
                        "frame_index"
                    ]
                ):
                camera

                for camera
                in cameras_json[
                    "cameras"
                ]

            }


            tracks = tracking.get(
                "tracks",
                {}
            )


            print(
                segment,
                "tracks:",
                len(tracks)
            )


            # ================================================
            # EACH TRACK
            # ================================================

            for track_id, track in (
                tracks.items()
            ):


                frames = [

                    int(x)

                    for x
                    in track.get(
                        "frames",
                        []
                    )

                ]


                centers = track.get(
                    "centers",
                    []
                )


                if (
                    len(frames)
                    !=
                    len(centers)
                ):

                    print(
                        "Length mismatch:",
                        video,
                        segment,
                        track_id
                    )

                    continue


                center_lookup = {

                    frame:
                    np.asarray(
                        center,
                        dtype=np.float64
                    )

                    for frame, center
                    in zip(
                        frames,
                        centers
                    )

                }


                # ============================================
                # RAW + WORLD TRAJECTORY AUDIT
                # ============================================

                valid_frames = []


                for frame in frames:


                    if frame not in (
                        camera_lookup
                    ):

                        continue


                    center_cam = (
                        center_lookup[
                            frame
                        ]
                    )


                    camera = (
                        camera_lookup[
                            frame
                        ]
                    )


                    try:

                        center_world = (
                            world_from_camera(

                                center_cam,

                                camera[
                                    "extrinsic"
                                ]

                            )
                        )

                    except Exception:

                        continue


                    if not (

                        np.isfinite(
                            center_cam
                        ).all()

                        and

                        np.isfinite(
                            center_world
                        ).all()

                    ):

                        continue


                    valid_frames.append(
                        frame
                    )


                    all_track_rows.append({

                        "video":
                            video,

                        "segment":
                            segment,

                        "track_id":
                            str(
                                track_id
                            ),

                        "frame_index":
                            frame,

                        "cam_x":
                            center_cam[0],

                        "cam_y":
                            center_cam[1],

                        "cam_z":
                            center_cam[2],

                        "world_x":
                            center_world[0],

                        "world_y":
                            center_world[1],

                        "world_z":
                            center_world[2]
                    })


                # ============================================
                # CONTINUOUS TRACK RUNS
                # ============================================

                runs = split_continuous(
                    valid_frames
                )


                for run_id, run in enumerate(
                    runs
                ):


                    # create lookup index

                    for horizon_sec, horizon_frames in (
                        HORIZONS.items()
                    ):


                        required = (

                            OBS_FRAMES
                            +
                            horizon_frames

                        )


                        if len(run) < required:

                            continue


                        max_start = (

                            len(run)
                            -
                            required

                        )


                        for start_pos in range(

                            0,

                            max_start + 1,

                            STRIDE

                        ):


                            window_frames = run[

                                start_pos:
                                start_pos
                                +
                                required

                            ]


                            anchor_frame = (
                                window_frames[0]
                            )


                            anchor_camera = (
                                camera_lookup[
                                    anchor_frame
                                ]
                            )


                            anchor_extrinsic = (
                                anchor_camera[
                                    "extrinsic"
                                ]
                            )


                            anchor_K = (
                                anchor_camera[
                                    "intrinsic"
                                ]
                            )


                            width = int(
                                anchor_camera[
                                    "image_width"
                                ]
                            )


                            height = int(
                                anchor_camera[
                                    "image_height"
                                ]
                            )


                            corrected = []

                            valid_window = True


                            for frame in (
                                window_frames
                            ):


                                center_cam = (
                                    center_lookup[
                                        frame
                                    ]
                                )


                                camera = (
                                    camera_lookup[
                                        frame
                                    ]
                                )


                                try:

                                    X_world = (
                                        world_from_camera(

                                            center_cam,

                                            camera[
                                                "extrinsic"
                                            ]

                                        )
                                    )


                                    X_anchor = (
                                        camera_from_world(

                                            X_world,

                                            anchor_extrinsic

                                        )
                                    )


                                    projected = (
                                        project_point(

                                            X_anchor,

                                            anchor_K,

                                            width,

                                            height

                                        )
                                    )


                                except Exception:

                                    projected = None


                                if projected is None:

                                    valid_window = False

                                    break


                                x_n, y_n, z_a = (
                                    projected
                                )


                                if not np.isfinite(
                                    [
                                        x_n,
                                        y_n,
                                        z_a
                                    ]
                                ).all():

                                    valid_window = False

                                    break


                                corrected.append(

                                    [
                                        x_n,
                                        y_n,
                                        z_a
                                    ]

                                )


                            if not valid_window:

                                continue


                            corrected = np.asarray(

                                corrected,

                                dtype=np.float32

                            )


                            history = corrected[
                                :OBS_FRAMES
                            ]


                            future = corrected[
                                OBS_FRAMES:
                            ]


                            sample_id = len(

                                window_history[
                                    horizon_sec
                                ]

                            )


                            window_history[
                                horizon_sec
                            ].append(
                                history
                            )


                            window_future[
                                horizon_sec
                            ].append(
                                future
                            )


                            window_metadata[
                                horizon_sec
                            ].append({

                                "sample_id":
                                    sample_id,

                                "video":
                                    video,

                                "segment":
                                    segment,

                                "track_id":
                                    str(
                                        track_id
                                    ),

                                "run_id":
                                    run_id,

                                "fps":
                                    FPS,

                                "obs_sec":
                                    OBS_SEC,

                                "horizon_sec":
                                    horizon_sec,

                                "obs_start_track_frame":
                                    window_frames[
                                        0
                                    ],

                                "obs_end_track_frame":
                                    window_frames[
                                        OBS_FRAMES - 1
                                    ],

                                "future_start_track_frame":
                                    window_frames[
                                        OBS_FRAMES
                                    ],

                                "future_end_track_frame":
                                    window_frames[
                                        -1
                                    ],

                                "anchor_frame_index":
                                    anchor_frame,

                                "history_frames":
                                    OBS_FRAMES,

                                "future_frames":
                                    horizon_frames
                            })


# ============================================================
# SAVE TRACK POINTS
# ============================================================

track_df = pd.DataFrame(
    all_track_rows
)


track_df.to_csv(

    OUT /
    "wildbox_world_track_points.csv",

    index=False
)


# ============================================================
# SAVE WINDOWS
# ============================================================

summary_rows = []


for horizon_sec in [
    2,
    5
]:


    meta = pd.DataFrame(

        window_metadata[
            horizon_sec
        ]

    )


    meta.to_csv(

        OUT /
        f"wildbox_windows_2s_to_{horizon_sec}s.csv",

        index=False
    )


    if len(
        window_history[
            horizon_sec
        ]
    ) > 0:


        histories = np.stack(

            window_history[
                horizon_sec
            ]

        )


        futures = np.stack(

            window_future[
                horizon_sec
            ]

        )


    else:


        histories = np.empty(

            (
                0,
                OBS_FRAMES,
                3
            ),

            dtype=np.float32

        )


        futures = np.empty(

            (
                0,
                HORIZONS[
                    horizon_sec
                ],
                3
            ),

            dtype=np.float32

        )


    np.savez_compressed(

        OUT /
        f"wildbox_corrected_2s_to_{horizon_sec}s.npz",

        history=histories,

        future=futures
    )


    # ==========================================
    # PER-VIDEO COUNTS
    # ==========================================

    if len(meta):

        for video, group in (
            meta.groupby(
                "video"
            )
        ):


            summary_rows.append({

                "task":
                    f"2s_to_{horizon_sec}s",

                "video":
                    video,

                "windows":
                    len(
                        group
                    ),

                "segments":
                    group[
                        "segment"
                    ].nunique(),

                "tracks":
                    (
                        group[
                            [
                                "segment",
                                "track_id"
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    )
            })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(

    OUT /
    "wildbox_window_summary.csv",

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
        "WILDBOX CAMERA-CORRECTED TRAJECTORY DATASET\n"
    )

    f.write(
        "=" * 72
        +
        "\n\n"
    )


    f.write(
        "Sampling frequency: 10 Hz\n"
    )

    f.write(
        "Observation history: 2 s = 20 frames\n"
    )

    f.write(
        "Forecast horizons: 2 s and 5 s\n"
    )

    f.write(
        "Window stride: 1 s = 10 frames\n\n"
    )


    f.write(
        "Camera correction:\n"
    )

    f.write(
        "1. Track centroid camera -> world "
        "using inverse per-frame extrinsic.\n"
    )

    f.write(
        "2. World point -> observation-start "
        "anchor camera.\n"
    )

    f.write(
        "3. Anchor-camera 3D point projected "
        "with anchor intrinsic.\n"
    )

    f.write(
        "4. Result normalized by anchor image "
        "width and height.\n\n"
    )


    for task in [

        "2s_to_2s",

        "2s_to_5s"

    ]:


        subset = summary_df[

            summary_df[
                "task"
            ]
            ==
            task

        ]


        f.write(
            task
            +
            "\n"
        )


        f.write(

            f"Total windows: "
            f"{int(subset['windows'].sum())}\n"

        )


        for _, row in (
            subset.iterrows()
        ):


            f.write(

                f"  {row['video']}: "
                f"{int(row['windows'])} windows\n"

            )


        f.write(
            "\n"
        )


print(
    "\n=========================================="
)

print(
    "WILDBOX TRAJECTORY BUILD COMPLETE"
)

print(
    "=========================================="
)

print(
    "\nOutput:"
)

print(
    OUT
)

print(
    "\nWindow summary:"
)

print(
    summary_df
)