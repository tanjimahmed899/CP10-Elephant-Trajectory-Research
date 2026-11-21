from pathlib import Path
import json
import math
import zipfile
import os
import hashlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn



# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("CP10_JOURNAL_ROOT", "/mnt/e/AIUB/Capstone Project/Journal Based Results"))
LOCKED_FILE = Path(__file__).with_name("CANDIDATES_RETROSPECTIVE.csv")
AUDIT_FILE = Path(__file__).with_name("SEMANTIC_AUDIT_VERIFIED.json")
MODEL_ROOT = ROOT / "Main Result" / "transformer_results"
DOWNLOAD_ROOT = ROOT / "Main Result"
OUT = Path(os.environ.get("CP10_STAGE5_OUT", str(Path.home() / "CP10_Stage5_Frozen_Inference_Results")))
OUT.mkdir(parents=True, exist_ok=True)
if not AUDIT_FILE.exists():
    raise FileNotFoundError(AUDIT_FILE)
AUDIT = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
SEMANTIC_TRACKS = {
    (x["video"], x["segment"], str(x["track_id"]))
    for x in AUDIT["tracks"] if x.get("semantic_elephant") is True
}
AUDIT_SEGMENTS = {(x["video"], x["segment"]): x for x in AUDIT["segments"]}

REPO_ID = "wildbox-anon-2026/wildbox-review"
REPO_TYPE = "dataset"


# ============================================================
# FROZEN EXPERIMENT SETTINGS
# ============================================================

SEEDS = [
    42,
    52,
    62
]

TASKS = {
    "2s_to_2s": {
        "obs_frames": 20,
        "future_frames": 20,
        "scale_floor": 0.06820264416877736,
    },

    "2s_to_5s": {
        "obs_frames": 20,
        "future_frames": 50,
        "scale_floor": 0.06732638773715147,
    },
}

FPS = 10.0

STRIDE = 10

HEADING_FRAMES = 10

DEVICE = torch.device(
    "cpu"
)

EPS = 1e-10


# ============================================================
# LOAD LOCKED HOLDOUT
# ============================================================

if not LOCKED_FILE.exists():

    raise FileNotFoundError(
        f"Locked holdout file not found:\n{LOCKED_FILE}"
    )


locked = pd.read_csv(
    LOCKED_FILE
)


required_columns = {
    "holdout_id",
    "repo_path",
    "basename",
    "elephant_group",
}


missing_columns = (
    required_columns
    -
    set(
        locked.columns
    )
)


if missing_columns:

    raise RuntimeError(
        f"Locked CSV missing columns: {missing_columns}"
    )


if len(
    locked
) != 3:

    raise RuntimeError(
        "Retrospective adaptation/evaluation requires exactly three specified local videos."
    )


print(
    "\nRETROSPECTIVE SAME-GROUP CANDIDATES (NOT INDEPENDENT HOLDOUT)"
)

print(
    locked[
        [
            "holdout_id",
            "elephant_group",
            "repo_path"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# DOWNLOAD ONLY LOCKED ZIP FILES
# ============================================================

download_rows = []
for _, row in locked.iterrows():
    video = Path(row["basename"]).stem
    possible = [
        ROOT / "Main Result" / row["basename"],
        ROOT / "ZIP 4.0" / row["basename"],
        ROOT / "Q1_Research_Completion_Sprint" / "new_data" / row["elephant_group"] / row["basename"],
    ]
    local_path = next((p for p in possible if p.is_file()), None)
    if local_path is None:
        raise FileNotFoundError(f"Source not found for {video}: {possible}")
    download_rows.append({
        "holdout_id": row["holdout_id"],
        "repo_path": row["repo_path"],
        "basename": row["basename"],
        "elephant_group": row["elephant_group"],
        "analysis_role": row["analysis_role"],
        "local_zip": str(local_path),
        "size_mb": local_path.stat().st_size/(1024**2),
    })
    print("Local candidate:", local_path)
downloads = pd.DataFrame(download_rows)
downloads.to_csv(OUT / "candidate_archive_manifest.csv", index=False)

def load_json_member(
    archive,
    member_name
):

    with archive.open(
        member_name
    ) as f:

        return json.load(
            f
        )


def find_json_members(
    archive
):

    members = archive.namelist()


    tracking_members = sorted(
        [
            m
            for m in members
            if m.endswith(
                "/tracking_summary.json"
            )
        ]
    )


    camera_members = sorted(
        [
            m
            for m in members
            if m.endswith(
                "/cameras.json"
            )
        ]
    )


    return (
        tracking_members,
        camera_members
    )


def member_parent(
    member
):

    return str(
        Path(
            member
        ).parent
    ).replace(
        "\\",
        "/"
    )


# ============================================================
# CAMERA GEOMETRY
# ============================================================

def split_extrinsic(
    extrinsic
):

    E = np.asarray(
        extrinsic,
        dtype=np.float64
    )


    if E.shape != (
        3,
        4
    ):

        raise ValueError(
            f"Unexpected extrinsic shape: {E.shape}"
        )


    R = E[
        :,
        :3
    ]


    t = E[
        :,
        3
    ]


    return (
        R,
        t
    )


def camera_to_world(
    point_camera,
    extrinsic
):

    R, t = split_extrinsic(
        extrinsic
    )


    point_camera = np.asarray(
        point_camera,
        dtype=np.float64
    )


    # Dataset convention:
    #
    # X_cam = R X_world + t
    #
    # therefore
    #
    # X_world = R.T (X_cam - t)

    return (
        R.T
        @
        (
            point_camera
            -
            t
        )
    )


def world_to_camera(
    point_world,
    extrinsic
):

    R, t = split_extrinsic(
        extrinsic
    )


    return (

        R
        @
        point_world

        +
        t

    )


def project_point(
    point_camera,
    intrinsic,
    width,
    height
):

    point_camera = np.asarray(
        point_camera,
        dtype=np.float64
    )


    K = np.asarray(
        intrinsic,
        dtype=np.float64
    )


    if K.shape != (
        3,
        3
    ):

        raise ValueError(
            f"Unexpected intrinsic shape: {K.shape}"
        )


    if (
        not np.all(
            np.isfinite(
                point_camera
            )
        )
    ):

        return None


    if point_camera[
        2
    ] <= EPS:

        return None


    projected = (
        K
        @
        point_camera
    )


    if abs(
        projected[
            2
        ]
    ) <= EPS:

        return None


    u = (
        projected[
            0
        ]
        /
        projected[
            2
        ]
    )


    v = (
        projected[
            1
        ]
        /
        projected[
            2
        ]
    )


    x_norm = (
        u
        /
        float(
            width
        )
    )


    y_norm = (
        v
        /
        float(
            height
        )
    )


    return np.asarray(
        [
            x_norm,
            y_norm
        ],
        dtype=np.float64
    )


# ============================================================
# CAMERA LOOKUP
# ============================================================

def build_camera_lookup(
    camera_json
):

    cameras = camera_json[
        "cameras"
    ]


    lookup = {}


    for camera in cameras:

        frame_index = int(
            camera[
                "frame_index"
            ]
        )


        lookup[
            frame_index
        ] = camera


    return lookup


# ============================================================
# SPLIT TRACK INTO CONTIGUOUS RUNS
# ============================================================

def contiguous_runs(
    frames
):

    frames = np.asarray(
        frames,
        dtype=int
    )


    if len(
        frames
    ) == 0:

        return []


    order = np.argsort(
        frames
    )


    frames = frames[
        order
    ]


    breaks = np.where(

        np.diff(
            frames
        )
        !=
        1

    )[0]


    starts = np.concatenate(
        [
            [0],
            breaks + 1
        ]
    )


    ends = np.concatenate(
        [
            breaks + 1,
            [len(frames)]
        ]
    )


    runs = []


    for run_id, (
        start,
        end
    ) in enumerate(
        zip(
            starts,
            ends
        )
    ):

        runs.append(
            {
                "run_id":
                    run_id,

                "indices":
                    order[
                        start:end
                    ],

                "frames":
                    frames[
                        start:end
                    ],
            }
        )


    return runs


# ============================================================
# BUILD ONE WINDOW IN OBSERVATION-START ANCHOR FRAME
# ============================================================

def build_anchor_window(
    track,
    selected_indices,
    camera_lookup
):

    frames = np.asarray(
        track[
            "frames"
        ],
        dtype=int
    )


    centers = np.asarray(
        track[
            "centers"
        ],
        dtype=np.float64
    )


    selected_frames = frames[
        selected_indices
    ]


    selected_centers = centers[
        selected_indices
    ]


    anchor_frame = int(
        selected_frames[
            0
        ]
    )


    if anchor_frame not in camera_lookup:

        return None


    anchor_camera = camera_lookup[
        anchor_frame
    ]


    anchor_extrinsic = anchor_camera[
        "extrinsic"
    ]


    anchor_intrinsic = anchor_camera[
        "intrinsic"
    ]


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


    trajectory = []


    for frame_index, center in zip(
        selected_frames,
        selected_centers
    ):


        frame_index = int(
            frame_index
        )


        if frame_index not in camera_lookup:

            return None


        current_camera = camera_lookup[
            frame_index
        ]


        # ----------------------------------------------------
        # Current camera coordinates -> dataset world frame
        # ----------------------------------------------------

        world_point = camera_to_world(

            center,

            current_camera[
                "extrinsic"
            ]

        )


        # ----------------------------------------------------
        # Dataset world frame -> fixed observation-start camera
        # ----------------------------------------------------

        anchor_camera_point = world_to_camera(

            world_point,

            anchor_extrinsic

        )


        projected = project_point(

            anchor_camera_point,

            anchor_intrinsic,

            width,

            height

        )


        if projected is None:

            return None


        trajectory.append(
            projected
        )


    trajectory = np.asarray(
        trajectory,
        dtype=np.float64
    )


    if not np.all(
        np.isfinite(
            trajectory
        )
    ):

        return None


    return (
        trajectory,
        anchor_frame
    )


# ============================================================
# CANONICALIZATION
#
# MUST MATCH FINAL TRAINING REPRESENTATION
# ============================================================

def heading_angle_from_history(
    history
):

    recent = history[
        -min(
            HEADING_FRAMES,
            len(
                history
            )
        ):
    ]


    t = np.arange(
        len(
            recent
        ),
        dtype=np.float64
    )


    tc = (
        t
        -
        t.mean()
    )


    denominator = np.sum(
        tc ** 2
    )


    if denominator <= EPS:

        return 0.0


    x_centered = (
        recent[
            :,
            0
        ]
        -
        recent[
            :,
            0
        ].mean()
    )


    y_centered = (
        recent[
            :,
            1
        ]
        -
        recent[
            :,
            1
        ].mean()
    )


    vx = (
        np.sum(
            tc
            *
            x_centered
        )
        /
        denominator
    )


    vy = (
        np.sum(
            tc
            *
            y_centered
        )
        /
        denominator
    )


    if np.hypot(
        vx,
        vy
    ) <= EPS:

        return 0.0


    return float(
        np.arctan2(
            vy,
            vx
        )
    )


def canonicalize_window(
    history_abs,
    future_abs,
    scale_floor
):

    origin = history_abs[
        -1
    ].copy()


    angle = heading_angle_from_history(
        history_abs
    )


    history_steps = np.linalg.norm(

        np.diff(
            history_abs,
            axis=0
        ),

        axis=1

    )


    path_length = float(
        history_steps.sum()
    )


    scale = max(
        path_length,
        scale_floor
    )


    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )


    # inverse rotation used during reconstruction:
    #
    # canonical @ R -> absolute-local

    R = np.asarray(

        [
            [c, s],
            [-s, c]
        ],

        dtype=np.float64

    )


    history_can = (

        (
            history_abs
            -
            origin
        )

        @
        R.T

        /
        scale

    )


    future_can = (

        (
            future_abs
            -
            origin
        )

        @
        R.T

        /
        scale

    )


    return {
        "history_canonical":
            history_can.astype(
                np.float32
            ),

        "future_canonical":
            future_can.astype(
                np.float32
            ),

        "origin":
            origin.astype(
                np.float64
            ),

        "heading_angle":
            float(
                angle
            ),

        "scale":
            float(
                scale
            ),

        "history_path":
            path_length,
    }


# ============================================================
# BUILD FRESH HOLDOUt WINDOWS
# ============================================================


def strict_matched_elephant_frames(archive, member_dir, tracks, camera_json, video, segment):
    """One-to-one KITTI vs 512x288 tracker box pairing per frame.
    No GT future position enters model input; labels only govern sample eligibility.
    """
    audit = AUDIT_SEGMENTS.get((video, segment))
    if audit is None:
        raise RuntimeError(f"No semantic audit segment for {video}/{segment}")
    sw = float(audit["assumed_tracker_input_width"])
    sh = float(audit["assumed_tracker_input_height"])
    byframe = {}
    for tid, tr in tracks.items():
        frames = tr.get("frames", [])
        boxes = tr.get("bbox_2d", [])
        if len(frames) != len(boxes):
            raise RuntimeError(f"bbox/frame count mismatch: {video}/{segment}/{tid}")
        for frame, bbox in zip(frames, boxes):
            if bbox and len(bbox) == 4:
                byframe.setdefault(int(frame), []).append((str(tid), bbox))
    eligible = {str(tid): set() for tid in tracks}
    names = set(archive.namelist())
    for cam in camera_json["cameras"]:
        frame = int(cam["frame_index"])
        frame_image = cam.get("image_name", f"frame_{frame:06d}.jpg")
        lbl = member_dir + "/kitti_labels/" + Path(frame_image).with_suffix(".txt").name
        if lbl not in names:
            continue
        gts=[]
        for line in archive.read(lbl).decode('utf-8', 'replace').splitlines():
            bits=line.split()
            if len(bits) < 8: continue
            try: box=tuple(float(q) for q in bits[4:8])
            except ValueError: continue
            if box[2] > box[0] and box[3] > box[1]:
                gts.append((bits[0].strip().lower(),box))
        width = float(cam.get("image_width", 1920))
        height = float(cam.get("image_height", 1080))
        matched=[]
        for p,(tid,raw) in enumerate(byframe.get(frame, [])):
            pred=(float(raw[0])*width/sw,float(raw[1])*height/sh,
                  float(raw[2])*width/sw,float(raw[3])*height/sh)
            for g,(cl,gt) in enumerate(gts):
                ix=max(0.,min(pred[2],gt[2])-max(pred[0],gt[0]))
                iy=max(0.,min(pred[3],gt[3])-max(pred[1],gt[1]))
                a=max(0.,pred[2]-pred[0])*max(0.,pred[3]-pred[1])
                b=max(0.,gt[2]-gt[0])*max(0.,gt[3]-gt[1])
                u=a+b-ix*iy
                overlap=ix*iy/u if u>0 else 0.
                if overlap >= 0.5:matched.append((overlap, p, g))
        used_p=set();used_g=set()
        for val,p,g in sorted(matched, reverse=True):
            if p in used_p or g in used_g:continue
            used_p.add(p);used_g.add(g)
            if gts[g][0] == 'elephant':
                eligible[byframe[frame][p][0]].add(frame)
    return eligible

def build_task_windows(
    task,
    config
):

    obs_frames = config[
        "obs_frames"
    ]


    future_frames = config[
        "future_frames"
    ]


    total_frames = (
        obs_frames
        +
        future_frames
    )


    scale_floor = config[
        "scale_floor"
    ]


    history_list = []
    future_list = []

    history_abs_list = []
    future_abs_list = []

    origins = []
    angles = []
    scales = []

    metadata_rows = []

    audit_rows = []


    sample_id = 0


    for _, video_row in downloads.iterrows():

        video_name = Path(
            video_row[
                "basename"
            ]
        ).stem


        zip_path = Path(
            video_row[
                "local_zip"
            ]
        )


        print(
            f"\nBuilding {task} from {video_name}"
        )


        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as archive:


            tracking_members, camera_members = (
                find_json_members(
                    archive
                )
            )


            camera_by_parent = {

                member_parent(
                    m
                ):
                    m

                for m in camera_members
            }


            if len(
                tracking_members
            ) == 0:

                audit_rows.append({
                    "task":
                        task,

                    "video":
                        video_name,

                    "segment":
                        "NONE",

                    "tracks":
                        0,

                    "valid_windows":
                        0,

                    "status":
                        "NO_TRACKING_JSON",
                })

                continue


            for tracking_member in tracking_members:

                parent = member_parent(
                    tracking_member
                )


                segment = Path(
                    parent
                ).name


                camera_member = camera_by_parent.get(
                    parent
                )


                if camera_member is None:

                    audit_rows.append({
                        "task":
                            task,

                        "video":
                            video_name,

                        "segment":
                            segment,

                        "tracks":
                            0,

                        "valid_windows":
                            0,

                        "status":
                            "NO_CAMERAS_JSON",
                    })

                    continue


                tracking_json = load_json_member(

                    archive,

                    tracking_member

                )


                camera_json = load_json_member(

                    archive,

                    camera_member

                )


                camera_lookup = build_camera_lookup(
                    camera_json
                )


                tracks = tracking_json.get(
                    "tracks",
                    {}
                )


                gt_verified_frames = strict_matched_elephant_frames(
                    archive, parent, tracks, camera_json, video_name, segment
                )

                segment_windows = 0


                for track_id, track in tracks.items():
                    if (video_name, segment, str(track_id)) not in SEMANTIC_TRACKS:
                        continue


                    frames = np.asarray(
                        track.get(
                            "frames",
                            []
                        ),
                        dtype=int
                    )


                    centers = np.asarray(
                        track.get(
                            "centers",
                            []
                        ),
                        dtype=np.float64
                    )


                    if (
                        len(
                            frames
                        )
                        !=
                        len(
                            centers
                        )
                    ):

                        continue


                    if len(
                        frames
                    ) < total_frames:

                        continue


                    runs = contiguous_runs(
                        frames
                    )


                    for run in runs:

                        run_indices = run[
                            "indices"
                        ]


                        if len(
                            run_indices
                        ) < total_frames:

                            continue


                        for start in range(

                            0,

                            len(
                                run_indices
                            )
                            -
                            total_frames
                            +
                            1,

                            STRIDE

                        ):


                            selected = run_indices[
                                start:
                                start
                                +
                                total_frames
                            ]


                            if not all(int(frames[int(i)]) in gt_verified_frames.get(str(track_id),set()) for i in selected):
                                continue

                            result = build_anchor_window(

                                track,

                                selected,

                                camera_lookup

                            )


                            if result is None:

                                continue


                            full_abs, anchor_frame = result


                            history_abs = full_abs[
                                :obs_frames
                            ]


                            future_abs = full_abs[
                                obs_frames:
                            ]


                            canonical = canonicalize_window(

                                history_abs,

                                future_abs,

                                scale_floor

                            )


                            history_list.append(

                                canonical[
                                    "history_canonical"
                                ]

                            )


                            future_list.append(

                                canonical[
                                    "future_canonical"
                                ]

                            )


                            history_abs_list.append(

                                history_abs.astype(
                                    np.float64
                                )

                            )


                            future_abs_list.append(

                                future_abs.astype(
                                    np.float64
                                )

                            )


                            origins.append(

                                canonical[
                                    "origin"
                                ]

                            )


                            angles.append(

                                canonical[
                                    "heading_angle"
                                ]

                            )


                            scales.append(

                                canonical[
                                    "scale"
                                ]

                            )


                            metadata_rows.append({

                                "sample_id":
                                    sample_id,

                                "task":
                                    task,

                                "holdout_id":
                                    video_row[
                                        "holdout_id"
                                    ],

                                "elephant_group":
                                    video_row[
                                        "elephant_group"
                                    ],

                                "source_group":
                                    video_name,

                                "analysis_role":
                                    video_row["analysis_role"],

                                "video":
                                    video_name,

                                "segment":
                                    segment,

                                "track_id":
                                    str(
                                        track_id
                                    ),

                                "track_key":
                                    (
                                        f"{video_name}"
                                        f"|{segment}"
                                        f"|{track_id}"
                                    ),

                                "run_id":
                                    run[
                                        "run_id"
                                    ],

                                "anchor_frame_index":
                                    anchor_frame,

                                "history_frames":
                                    obs_frames,

                                "future_frames":
                                    future_frames,

                                "history_path":
                                    canonical[
                                        "history_path"
                                    ],

                                "scale":
                                    canonical[
                                        "scale"
                                    ],

                            })


                            sample_id += 1

                            segment_windows += 1


                audit_rows.append({

                    "task":
                        task,

                    "video":
                        video_name,

                    "segment":
                        segment,

                    "tracks":
                        len(
                            tracks
                        ),

                    "camera_frames":
                        len(
                            camera_lookup
                        ),

                    "valid_windows":
                        segment_windows,

                    "status":
                        "OK",

                })


    metadata = pd.DataFrame(
        metadata_rows
    )


    audit = pd.DataFrame(
        audit_rows
    )


    if len(
        history_list
    ) == 0:

        return None, metadata, audit


    dataset = {

        "history_canonical":
            np.stack(
                history_list
            ),

        "future_canonical":
            np.stack(
                future_list
            ),

        "history_abs":
            np.stack(
                history_abs_list
            ),

        "future_abs":
            np.stack(
                future_abs_list
            ),

        "origin":
            np.stack(
                origins
            ),

        "heading_angle":
            np.asarray(
                angles,
                dtype=np.float64
            ),

        "scale":
            np.asarray(
                scales,
                dtype=np.float64
            ),

    }


    return (
        dataset,
        metadata,
        audit
    )


# ============================================================
# TRANSFORMER MODEL
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

        ).unsqueeze(
            1
        )


        div_term = torch.exp(

            torch.arange(

                0,
                d_model,
                2,

                dtype=torch.float32

            )

            *
            (
                -math.log(
                    10000.0
                )
                /
                d_model
            )

        )


        pe[
            :,
            0::2
        ] = torch.sin(

            position
            *
            div_term

        )


        pe[
            :,
            1::2
        ] = torch.cos(

            position
            *
            div_term

        )


        self.register_buffer(

            "pe",

            pe.unsqueeze(
                0
            )

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
                :x.size(
                    1
                )
            ]

        )


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


        self.future_length = (
            future_length
        )


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

            self.head(
                x
            )

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
            f"Missing frozen checkpoint:\n"
            f"{checkpoint_file}"
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


    # Positional encoding is deterministic.
    state_dict = {

        k: v

        for k, v
        in state_dict.items()

        if k != "position.pe"

    }


    result = model.load_state_dict(

        state_dict,

        strict=False

    )


    unexpected_missing = [

        k
        for k in result.missing_keys
        if k != "position.pe"

    ]


    if (
        unexpected_missing
        or
        result.unexpected_keys
    ):

        raise RuntimeError(
            "Checkpoint architecture mismatch.\n"
            f"Missing: {unexpected_missing}\n"
            f"Unexpected: {result.unexpected_keys}"
        )


    model.eval()


    return model


# ============================================================
# MODEL PREDICTION
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
            len(
                histories
            ),
            batch_size

        ):


            x = torch.tensor(

                histories[
                    start:
                    start
                    +
                    batch_size
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
        len(
            canonical
        )
    ):

        a = float(
            angle[
                i
            ]
        )


        c = np.cos(
            a
        )

        s = np.sin(
            a
        )


        R = np.asarray(

            [
                [c, s],
                [-s, c]
            ],

            dtype=np.float64

        )


        local = (

            canonical[
                i
            ]

            *
            float(
                scale[
                    i
                ]
            )

        ) @ R


        output.append(

            local
            +
            origin[
                i
            ]

        )


    return np.asarray(
        output,
        dtype=np.float64
    )


# ============================================================
# FIXED CLASSICAL BASELINES
# ============================================================

def stationary_prediction(
    history,
    future_length
):

    return np.repeat(

        history[
            -1:
        ],

        future_length,

        axis=0

    )


def constant_velocity_prediction(
    history,
    future_length
):

    T = len(
        history
    )


    t = np.arange(
        T,
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


    centered = (

        history

        -
        history.mean(
            axis=0,
            keepdims=True
        )

    )


    velocity_per_step = (

        (
            centered
            *
            tc[
                :,
                None
            ]
        ).sum(
            axis=0
        )

        /
        denom

    )


    future_steps = np.arange(

        1,
        future_length + 1,

        dtype=np.float64

    )[
        :,
        None
    ]


    return (

        history[
            -1
        ]

        +
        future_steps
        *
        velocity_per_step

    )


# ============================================================
# METRICS
# ============================================================

def evaluate_metrics(
    prediction,
    truth,
    metadata
):

    error = np.linalg.norm(

        prediction
        -
        truth,

        axis=2

    )


    ADE = error.mean(
        axis=1
    )


    FDE = error[
        :,
        -1
    ]


    temp = metadata[
        [
            "source_group",
            "track_key"
        ]
    ].copy()


    temp[
        "ADE"
    ] = ADE


    temp[
        "FDE"
    ] = FDE


    track_metrics = (

        temp.groupby(

            [
                "source_group",
                "track_key"
            ],

            as_index=False

        )

        .agg(

            ADE=(
                "ADE",
                "mean"
            ),

            FDE=(
                "FDE",
                "mean"
            )

        )

    )


    source_metrics = (

        temp.groupby(

            "source_group",

            as_index=False

        )

        .agg(

            ADE=(
                "ADE",
                "mean"
            ),

            FDE=(
                "FDE",
                "mean"
            )

        )

    )


    return {

        "ADE":
            float(
                ADE.mean()
            ),

        "FDE":
            float(
                FDE.mean()
            ),

        "track_ADE":
            float(
                track_metrics[
                    "ADE"
                ].mean()
            ),

        "track_FDE":
            float(
                track_metrics[
                    "FDE"
                ].mean()
            ),

        "source_ADE":
            float(
                source_metrics[
                    "ADE"
                ].mean()
            ),

        "source_FDE":
            float(
                source_metrics[
                    "FDE"
                ].mean()
            ),

        "point_error":
            error,

        "sample_ADE":
            ADE,

        "sample_FDE":
            FDE,

    }


# ============================================================
# SAFE CORRELATION
# ============================================================

def safe_corr(
    x,
    y
):

    x = np.asarray(
        x
    ).reshape(
        -1
    )


    y = np.asarray(
        y
    ).reshape(
        -1
    )


    mask = (

        np.isfinite(
            x
        )

        &
        np.isfinite(
            y
        )

    )


    x = x[
        mask
    ]

    y = y[
        mask
    ]


    if (
        len(
            x
        ) < 3
        or
        np.std(
            x
        ) <= EPS
        or
        np.std(
            y
        ) <= EPS
    ):

        return np.nan


    return float(

        np.corrcoef(
            x,
            y
        )[
            0,
            1
        ]

    )


def safe_spearman(
    x,
    y
):

    x = np.asarray(
        x
    ).reshape(
        -1
    )


    y = np.asarray(
        y
    ).reshape(
        -1
    )


    rank_x = pd.Series(
        x
    ).rank(
        method="average"
    )


    rank_y = pd.Series(
        y
    ).rank(
        method="average"
    )


    return safe_corr(

        rank_x,

        rank_y

    )


# ============================================================
# RUN DATASET BUILD + ONE-SHOT EVALUATION
# ============================================================

dataset_summary_rows = []
audit_tables = []

metric_rows = []
per_source_rows = []

uncertainty_rows = []

all_metadata_to_save = []


for task, config in TASKS.items():

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "BUILDING RETROSPECTIVE CANDIDATE:",
        task
    )

    print(
        "=" * 80
    )


    dataset, metadata, audit = (
        build_task_windows(
            task,
            config
        )
    )


    audit_tables.append(
        audit
    )


    if (
        dataset is None
        or
        len(
            metadata
        ) == 0
    ):

        print(
            "No valid windows for:",
            task
        )

        continue


    metadata.to_csv(

        OUT /
        f"{task}_fresh_metadata.csv",

        index=False

    )


    np.savez_compressed(

        OUT /
        f"{task}_fresh_dataset.npz",

        **dataset

    )


    all_metadata_to_save.append(
        metadata
    )


    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    dataset_summary_rows.append({

        "task":
            task,

        "samples":
            len(
                metadata
            ),

        "sources":
            metadata[
                "source_group"
            ].nunique(),

        "segments":
            metadata[
                [
                    "source_group",
                    "segment"
                ]
            ].drop_duplicates().shape[
                0
            ],

        "tracklets":
            metadata[
                "track_key"
            ].nunique(),

        "median_history_path":
            float(
                metadata[
                    "history_path"
                ].median()
            ),

        "scale_floor":
            config[
                "scale_floor"
            ],

        "floor_applied_pct":
            float(
                100.0
                *
                np.mean(

                    metadata[
                        "history_path"
                    ].to_numpy()

                    <
                    config[
                        "scale_floor"
                    ]

                )
            ),

    })


    histories_can = dataset[
        "history_canonical"
    ]


    truth_abs = dataset[
        "future_abs"
    ]


    future_length = truth_abs.shape[
        1
    ]


    # ========================================================
    # FROZEN TRANSFORMER SEEDS
    # ========================================================

    seed_predictions_abs = []


    for seed in SEEDS:

        model = load_transformer(

            task,

            seed,

            future_length

        )


        pred_can = predict(

            model,

            histories_can

        )


        pred_abs = inverse_transform_batch(

            pred_can,

            dataset[
                "origin"
            ],

            dataset[
                "heading_angle"
            ],

            dataset[
                "scale"
            ]

        )


        seed_predictions_abs.append(
            pred_abs
        )


        metrics = evaluate_metrics(

            pred_abs,

            truth_abs,

            metadata

        )


        metric_rows.append({

            "task":
                task,

            "model":
                f"Transformer_seed{seed}",

            "ADE":
                metrics[
                    "ADE"
                ],

            "FDE":
                metrics[
                    "FDE"
                ],

            "track_macro_ADE":
                metrics[
                    "track_ADE"
                ],

            "track_macro_FDE":
                metrics[
                    "track_FDE"
                ],

            "source_macro_ADE":
                metrics[
                    "source_ADE"
                ],

            "source_macro_FDE":
                metrics[
                    "source_FDE"
                ],

        })


    seed_predictions_abs = np.stack(

        seed_predictions_abs,

        axis=0

    )


    # ========================================================
    # FROZEN ENSEMBLE
    # ========================================================

    ensemble_prediction = (

        seed_predictions_abs.mean(
            axis=0
        )

    )


    np.savez_compressed(
        OUT / f"{task}_FROZEN_PREDICTIONS_FOR_CALIBRATION.npz",
        history_abs=dataset["history_abs"],
        future_abs=dataset["future_abs"],
        ensemble_prediction_abs=ensemble_prediction,
        three_seed_predictions_abs=seed_predictions_abs,
        history_canonical=dataset["history_canonical"],
        scale=dataset["scale"],
        origin=dataset["origin"],
        heading_angle=dataset["heading_angle"],
    )

    ensemble_metrics = evaluate_metrics(

        ensemble_prediction,

        truth_abs,

        metadata

    )


    metric_rows.append({

        "task":
            task,

        "model":
            "Transformer_3Seed_Ensemble",

        "ADE":
            ensemble_metrics[
                "ADE"
            ],

        "FDE":
            ensemble_metrics[
                "FDE"
            ],

        "track_macro_ADE":
            ensemble_metrics[
                "track_ADE"
            ],

        "track_macro_FDE":
            ensemble_metrics[
                "track_FDE"
            ],

        "source_macro_ADE":
            ensemble_metrics[
                "source_ADE"
            ],

        "source_macro_FDE":
            ensemble_metrics[
                "source_FDE"
            ],

    })


    # ========================================================
    # STATIONARY + CONSTANT VELOCITY BASELINES
    # ========================================================

    stationary_predictions = []

    cv_predictions = []


    for history in dataset[
        "history_abs"
    ]:

        stationary_predictions.append(

            stationary_prediction(

                history,

                future_length

            )

        )


        cv_predictions.append(

            constant_velocity_prediction(

                history,

                future_length

            )

        )


    stationary_predictions = np.stack(
        stationary_predictions
    )


    cv_predictions = np.stack(
        cv_predictions
    )


    for model_name, prediction in [

        (
            "Stationary",
            stationary_predictions
        ),

        (
            "ConstantVelocity",
            cv_predictions
        ),

    ]:


        metrics = evaluate_metrics(

            prediction,

            truth_abs,

            metadata

        )


        metric_rows.append({

            "task":
                task,

            "model":
                model_name,

            "ADE":
                metrics[
                    "ADE"
                ],

            "FDE":
                metrics[
                    "FDE"
                ],

            "track_macro_ADE":
                metrics[
                    "track_ADE"
                ],

            "track_macro_FDE":
                metrics[
                    "track_FDE"
                ],

            "source_macro_ADE":
                metrics[
                    "source_ADE"
                ],

            "source_macro_FDE":
                metrics[
                    "source_FDE"
                ],

        })


    # ========================================================
    # PER-SOURCE ENSEMBLE METRICS
    # ========================================================

    for source in sorted(

        metadata[
            "source_group"
        ].unique()

    ):


        mask = (

            metadata[
                "source_group"
            ].to_numpy()

            ==
            source

        )


        source_meta = metadata.loc[
            mask
        ].reset_index(
            drop=True
        )


        source_metrics = evaluate_metrics(

            ensemble_prediction[
                mask
            ],

            truth_abs[
                mask
            ],

            source_meta

        )


        per_source_rows.append({

            "task":
                task,

            "source_group":
                source,

            "samples":
                int(
                    mask.sum()
                ),

            "tracklets":
                source_meta[
                    "track_key"
                ].nunique(),

            "ADE":
                source_metrics[
                    "ADE"
                ],

            "FDE":
                source_metrics[
                    "FDE"
                ],

            "track_macro_ADE":
                source_metrics[
                    "track_ADE"
                ],

            "track_macro_FDE":
                source_metrics[
                    "track_FDE"
                ],

        })


    # ========================================================
    # ENSEMBLE DISAGREEMENT — CONFIRMATORY DIAGNOSTIC
    # ========================================================

    deviations = (

        seed_predictions_abs

        -
        ensemble_prediction[
            None,
            ...
        ]

    )


    disagreement = np.sqrt(

        np.mean(

            np.sum(
                deviations ** 2,
                axis=-1
            ),

            axis=0

        )

    )


    actual_error = ensemble_metrics[
        "point_error"
    ]


    sample_uncertainty = disagreement.mean(
        axis=1
    )


    sample_error = actual_error.mean(
        axis=1
    )


    endpoint_uncertainty = disagreement[
        :,
        -1
    ]


    endpoint_error = actual_error[
        :,
        -1
    ]


    uncertainty_rows.append({

        "task":
            task,

        "samples":
            len(
                metadata
            ),

        "mean_disagreement":
            float(
                disagreement.mean()
            ),

        "pointwise_Pearson":
            safe_corr(

                disagreement.reshape(
                    -1
                ),

                actual_error.reshape(
                    -1
                )

            ),

        "pointwise_Spearman":
            safe_spearman(

                disagreement.reshape(
                    -1
                ),

                actual_error.reshape(
                    -1
                )

            ),

        "sample_ADE_Pearson":
            safe_corr(

                sample_uncertainty,

                sample_error

            ),

        "sample_ADE_Spearman":
            safe_spearman(

                sample_uncertainty,

                sample_error

            ),

        "endpoint_FDE_Pearson":
            safe_corr(

                endpoint_uncertainty,

                endpoint_error

            ),

        "endpoint_FDE_Spearman":
            safe_spearman(

                endpoint_uncertainty,

                endpoint_error

            ),

    })


# ============================================================
# SAVE AUDIT
# ============================================================

if audit_tables:

    audit_df = pd.concat(

        audit_tables,

        ignore_index=True

    )

else:

    audit_df = pd.DataFrame()


audit_df.to_csv(

    OUT /
    "fresh_holdout_build_audit.csv",

    index=False
)


# ============================================================
# SAVE DATASET SUMMARY
# ============================================================

dataset_summary = pd.DataFrame(
    dataset_summary_rows
)


dataset_summary.to_csv(

    OUT /
    "fresh_holdout_dataset_summary.csv",

    index=False
)


# ============================================================
# SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame(
    metric_rows
)


metrics_df.to_csv(

    OUT /
    "fresh_holdout_model_metrics.csv",

    index=False
)


# ============================================================
# SAVE PER-SOURCE
# ============================================================

per_source_df = pd.DataFrame(
    per_source_rows
)


per_source_df.to_csv(

    OUT /
    "fresh_holdout_per_source.csv",

    index=False
)


# ============================================================
# SAVE UNCERTAINTY
# ============================================================

uncertainty_df = pd.DataFrame(
    uncertainty_rows
)


uncertainty_df.to_csv(

    OUT /
    "fresh_holdout_uncertainty.csv",

    index=False
)


# ============================================================
# SUMMARY TEXT
# ============================================================

with open(OUT/"STAGE5_RETROSPECTIVE_PROTOCOL.txt", "w", encoding="utf-8") as f:
    f.write("Retrospective same-recording-group candidate assessment, not a locked independent holdout.\n")
    f.write("Model weights frozen before inference; overlapping windows are not independent. Candidate 0006=adaptation/development; the other 0004 videos=retrospective evaluation.\n")
    f.write("Only semantically verified track/frame positions meeting GT IoU>=0.50 are eligible.\n")
    f.write("GT annotations are used for evaluation and eligibility, never as detector predictions.\n")
    f.write("Prediction arrays in .npz support subsequent calibration analyses; none are calibrated here.\n")
    f.write("No formal distribution-shift coverage guarantee is claimed.\n")

# ============================================================
# CONSOLE OUTPUT
# ============================================================

print(
    "\n"
    +
    "=" * 82
)

print(
    "STEP 17B COMPLETE"
)

print(
    "=" * 82
)


print(
    "\nDataset summary:"
)

print(
    dataset_summary.to_string(
        index=False
    )
)


print(
    "\nModel metrics:"
)

print(
    metrics_df.to_string(
        index=False
    )
)


print(
    "\nRetrospective same-group ensemble per source:"
)

print(
    per_source_df.to_string(
        index=False
    )
)


print(
    "\nUncertainty diagnostic:"
)

print(
    uncertainty_df.to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)

print(
    OUT
)