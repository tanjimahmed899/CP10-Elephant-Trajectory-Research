from pathlib import Path
import hashlib
import importlib
import importlib.util
import json
import math

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

PRELOCK = (
    BASE
    / "k3_operational_warning"
    / "step7_external_prelock"
    / "STEP6D_K3C10_BAMBI_STEP7_PRELOCK.json"
)

EXPECTED_PRELOCK_SHA256 = (
    "8cc687c3f91ecbfb2a3c0f491427d91ccfdca959659a52e0836a98889c890bd1"
)

WORLD_RAW = (
    BASE
    / "step6dk_valid_world_trajectory_rows.csv"
)

RESAMPLED = (
    BASE
    / "k2a_10hz"
    / "step6dk2a_resampled_10hz_world_trajectory.csv"
)

K2A_DIR = (
    BASE
    / "k2a_10hz"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_reconstruction_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

FOVY_DEG = 50.0
IMAGE_W = 1024.0
IMAGE_H = 1024.0
TIME_TOL_SEC = 2e-4


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def require(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)


def discover_one(filename, root=DOWNLOADS, prefer_tokens=()):
    matches = sorted(
        p
        for p in Path(root).rglob(filename)
        if p.is_file()
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {root}"
        )

    if prefer_tokens:
        preferred = [
            p
            for p in matches
            if all(
                token.lower()
                in
                str(p).lower()
                for token in prefer_tokens
            )
        ]

        if len(preferred) == 1:
            return preferred[0]

    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(
        f"Ambiguous {filename}:\n"
        +
        "\n".join(
            str(p)
            for p in matches
        )
    )


def discover_enriched_pose():
    matches = sorted(
        p
        for p in BASE.rglob("6_matched_poses.json")
        if p.is_file()
        and
        "correction"
        in
        str(p.parent).lower()
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one enriched correction/6_matched_poses.json:\n"
            +
            "\n".join(
                str(p)
                for p in matches
            )
        )

    path = matches[0]

    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    images = raw.get(
        "images",
        []
    )

    if not images:
        raise RuntimeError(
            "Enriched pose JSON has no images."
        )

    if not all(
        isinstance(
            images[i],
            dict
        )
        and
        "location"
        in
        images[i]
        for i in [
            0,
            min(
                1000,
                len(images) - 1
            ),
            len(images) - 1,
        ]
    ):
        raise RuntimeError(
            "Enriched pose JSON does not consistently contain location."
        )

    return (
        path,
        raw
    )


def load_bambi_module():
    spec = importlib.util.find_spec(
        "bambi.georeference_mot17_tracks"
    )

    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Installed bambi.georeference_mot17_tracks not found."
        )

    module = importlib.import_module(
        "bambi.georeference_mot17_tracks"
    )

    return (
        module,
        Path(
            spec.origin
        )
    )


def normalize_correction_dict(
    raw,
    frame_idx
):
    rotation = raw.get(
        "rotation",
        {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
    )

    translation = raw.get(
        "translation",
        {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
    )

    additions = (
        raw.get(
            "fine_corrections"
        )
        or
        raw.get(
            "additional_corrections"
        )
        or
        raw.get(
            "additional corrections"
        )
        or
        []
    )

    chosen_rot = dict(
        rotation
    )

    chosen_trans = dict(
        translation
    )

    for item in additions:
        start = (
            item.get(
                "start frame"
            )
            if
            "start frame"
            in
            item
            else
            item.get(
                "start_frame"
            )
        )

        end = (
            item.get(
                "end frame"
            )
            if
            "end frame"
            in
            item
            else
            item.get(
                "end_frame"
            )
        )

        if (
            start is not None
            and
            end is not None
            and
            int(
                start
            )
            <
            int(
                frame_idx
            )
            <
            int(
                end
            )
        ):
            chosen_rot = dict(
                item.get(
                    "rotation",
                    chosen_rot
                )
            )

            chosen_trans = dict(
                item.get(
                    "translation",
                    chosen_trans
                )
            )

            break

    return (
        chosen_rot,
        chosen_trans
    )


def make_camera_matrices(
    module,
    poses,
    correction,
    frame_idx
):
    rot, trans = normalize_correction_dict(
        correction,
        frame_idx
    )

    Vector3 = module.Vector3

    cor_rotation_eulers = Vector3(
        [
            float(
                rot.get(
                    "x",
                    0.0
                )
            ),
            float(
                rot.get(
                    "y",
                    0.0
                )
            ),
            float(
                rot.get(
                    "z",
                    0.0
                )
            ),
        ],
        dtype="f4",
    )

    cor_translation = Vector3(
        [
            float(
                trans.get(
                    "x",
                    0.0
                )
            ),
            float(
                trans.get(
                    "y",
                    0.0
                )
            ),
            float(
                trans.get(
                    "z",
                    0.0
                )
            ),
        ],
        dtype="f4",
    )

    idx = int(
        frame_idx
    )

    poses_temp = dict(
        poses
    )

    images = list(
        poses[
            "images"
        ]
    )

    item = dict(
        images[
            idx
        ]
    )

    fovy = item.get(
        "fovy"
    )

    if isinstance(
        fovy,
        (
            list,
            tuple,
        )
    ) and len(
        fovy
    ):
        pass

    elif isinstance(
        fovy,
        (
            int,
            float,
        )
    ):
        item[
            "fovy"
        ] = [
            float(
                fovy
            )
        ]

    else:
        item[
            "fovy"
        ] = [
            FOVY_DEG
        ]

    images[
        idx
    ] = item

    poses_temp[
        "images"
    ] = images

    camera = module.get_camera_for_frame(
        poses_temp,
        idx,
        cor_rotation_eulers,
        cor_translation,
        overrule_fov=FOVY_DEG,
    )

    view = np.asarray(
        camera.get_view(
            dtype="f8"
        ),
        dtype=float,
    ).reshape(
        4,
        4
    )

    proj = np.asarray(
        camera.get_proj(
            dtype="f8"
        ),
        dtype=float,
    ).reshape(
        4,
        4
    )

    return (
        view,
        proj
    )


def project_local_xyz_to_normalized(
    xyz_local,
    view,
    proj
):
    pts = np.asarray(
        xyz_local,
        dtype=float
    )

    if (
        pts.ndim
        !=
        2
        or
        pts.shape[
            1
        ]
        !=
        3
    ):
        raise ValueError(
            f"Expected Nx3 local world points, got {pts.shape}"
        )

    ones = np.ones(
        (
            len(
                pts
            ),
            1,
        ),
        dtype=float,
    )

    p4 = np.concatenate(
        [
            pts,
            ones,
        ],
        axis=1,
    )

    # Exact K3C9-locked convention:
    # clip_col = P.T @ V.T @ p_col
    # Batch row equivalent:
    # clip_row = p_row @ V @ P
    clip = (
        p4
        @
        view
        @
        proj
    )

    w = clip[
        :,
        3
    ]

    finite_w = (
        np.isfinite(
            w
        )
        &
        (
            np.abs(
                w
            )
            >
            1e-12
        )
    )

    ndc = np.full(
        (
            len(
                pts
            ),
            3,
        ),
        np.nan,
        dtype=float,
    )

    ndc[
        finite_w
    ] = (
        clip[
            finite_w,
            :3
        ]
        /
        w[
            finite_w,
            None
        ]
    )

    u = (
        (
            ndc[
                :,
                0
            ]
            +
            1.0
        )
        *
        0.5
        *
        IMAGE_W
    )

    v = (
        (
            1.0
            -
            ndc[
                :,
                1
            ]
        )
        *
        0.5
        *
        IMAGE_H
    )

    xy_norm = np.stack(
        [
            u
            /
            IMAGE_W,
            v
            /
            IMAGE_H,
        ],
        axis=1,
    )

    return (
        xy_norm,
        ndc,
        w
    )


def fit_frame_time_mapping(
    world_raw
):
    df = world_raw[
        [
            "frame_id",
            "time_sec",
        ]
    ].drop_duplicates().copy()

    x = pd.to_numeric(
        df[
            "frame_id"
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    y = pd.to_numeric(
        df[
            "time_sec"
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    ok = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )

    x = x[
        ok
    ]

    y = y[
        ok
    ]

    if len(
        x
    ) < 2:
        raise RuntimeError(
            "Insufficient frame/time pairs for mapping."
        )

    A = np.column_stack(
        [
            x,
            np.ones_like(
                x
            ),
        ]
    )

    slope, intercept = np.linalg.lstsq(
        A,
        y,
        rcond=None
    )[0]

    pred = (
        slope
        *
        x
        +
        intercept
    )

    residual = (
        y
        -
        pred
    )

    fps = (
        1.0
        /
        slope
    )

    return {
        "seconds_per_frame":
            float(
                slope
            ),

        "intercept_sec":
            float(
                intercept
            ),

        "fps":
            float(
                fps
            ),

        "fit_max_abs_residual_sec":
            float(
                np.max(
                    np.abs(
                        residual
                    )
                )
            ),

        "fit_p95_abs_residual_sec":
            float(
                np.quantile(
                    np.abs(
                        residual
                    ),
                    0.95
                )
            ),
    }


def anchor_frame_for_time(
    t_sec,
    mapping,
    pose_count
):
    slope = mapping[
        "seconds_per_frame"
    ]

    intercept = mapping[
        "intercept_sec"
    ]

    frame_float = (
        (
            float(
                t_sec
            )
            -
            intercept
        )
        /
        slope
    )

    frame_idx = int(
        np.rint(
            frame_float
        )
    )

    if (
        frame_idx
        <
        0
        or
        frame_idx
        >=
        pose_count
    ):
        raise IndexError(
            f"Anchor frame {frame_idx} outside pose range 0..{pose_count-1}"
        )

    pose_time = (
        slope
        *
        frame_idx
        +
        intercept
    )

    offset = (
        pose_time
        -
        float(
            t_sec
        )
    )

    return (
        frame_idx,
        frame_float,
        pose_time,
        offset,
    )


def task_metadata_path(
    task
):
    matches = sorted(
        p
        for p in K2A_DIR.glob(
            "*.csv"
        )
        if (
            task.lower()
            in
            p.name.lower()
            and
            "metadata"
            in
            p.name.lower()
        )
    )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Expected exactly one K2A metadata CSV for {task}:\n"
            +
            "\n".join(
                str(p)
                for p in matches
            )
        )

    return matches[
        0
    ]


def extract_window_xyz(
    resampled_group,
    start_time,
    expected_points,
):
    g = resampled_group.sort_values(
        "time_sec"
    ).reset_index(
        drop=True
    )

    times = g[
        "time_sec"
    ].to_numpy(
        dtype=float
    )

    start_idx = int(
        np.argmin(
            np.abs(
                times
                -
                float(
                    start_time
                )
            )
        )
    )

    start_err = float(
        abs(
            times[
                start_idx
            ]
            -
            float(
                start_time
            )
        )
    )

    if start_err > TIME_TOL_SEC:
        raise RuntimeError(
            f"No resampled point sufficiently close to window start {start_time}; "
            f"nearest error={start_err} sec"
        )

    end_idx = (
        start_idx
        +
        expected_points
    )

    if end_idx > len(
        g
    ):
        raise RuntimeError(
            "Window extends beyond resampled segment."
        )

    sub = g.iloc[
        start_idx:end_idx
    ].copy()

    xyz = sub[
        [
            "utm_x",
            "utm_y",
            "utm_z",
        ]
    ].to_numpy(
        dtype=float
    )

    sub_times = sub[
        "time_sec"
    ].to_numpy(
        dtype=float
    )

    if len(
        sub_times
    ) > 1:
        dt = np.diff(
            sub_times
        )

        max_dt_error = float(
            np.max(
                np.abs(
                    dt
                    -
                    0.1
                )
            )
        )
    else:
        max_dt_error = 0.0

    return (
        xyz,
        sub_times,
        start_err,
        max_dt_error,
    )


def build_task(
    task,
    horizon_steps,
    metadata,
    resampled,
    dem_origin,
    poses,
    correction,
    module,
    frame_time_mapping,
    camera_cache,
):
    history_steps = 20

    expected_points = (
        history_steps
        +
        horizon_steps
    )

    required_meta = {
        "sample_index",
        "track_id",
        "global_segment_id",
        "gap_segment_id",
        "window_start_time_sec",
    }

    missing = (
        required_meta
        -
        set(
            metadata.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{task} metadata missing columns: {sorted(missing)}"
        )

    metadata = metadata.sort_values(
        "sample_index"
    ).reset_index(
        drop=True
    )

    sample_indices = pd.to_numeric(
        metadata[
            "sample_index"
        ],
        errors="raise"
    ).astype(
        int
    ).to_numpy()

    expected_indices = np.arange(
        len(
            metadata
        ),
        dtype=int,
    )

    if not np.array_equal(
        sample_indices,
        expected_indices
    ):
        raise RuntimeError(
            f"{task} sample_index is not exactly 0..N-1."
        )

    history_norm = np.empty(
        (
            len(
                metadata
            ),
            history_steps,
            2,
        ),
        dtype=np.float64,
    )

    future_norm = np.empty(
        (
            len(
                metadata
            ),
            horizon_steps,
            2,
        ),
        dtype=np.float64,
    )

    history_world = np.empty(
        (
            len(
                metadata
            ),
            history_steps,
            3,
        ),
        dtype=np.float64,
    )

    future_world = np.empty(
        (
            len(
                metadata
            ),
            horizon_steps,
            3,
        ),
        dtype=np.float64,
    )

    audit_rows = []

    grouped = {}

    group_cols = [
        "track_id",
        "global_segment_id",
        "gap_segment_id",
    ]

    for key, group in resampled.groupby(
        group_cols,
        sort=False,
    ):
        norm_key = tuple(
            int(
                x
            )
            for x in key
        )

        grouped[
            norm_key
        ] = group.copy()

    pose_count = len(
        poses[
            "images"
        ]
    )

    for i, row in metadata.iterrows():
        key = (
            int(
                row[
                    "track_id"
                ]
            ),
            int(
                row[
                    "global_segment_id"
                ]
            ),
            int(
                row[
                    "gap_segment_id"
                ]
            ),
        )

        if key not in grouped:
            raise RuntimeError(
                f"{task} sample {i}: no resampled segment for key {key}"
            )

        xyz_abs, times, start_err, max_dt_error = (
            extract_window_xyz(
                grouped[
                    key
                ],
                float(
                    row[
                        "window_start_time_sec"
                    ]
                ),
                expected_points,
            )
        )

        anchor_frame, frame_float, pose_time, anchor_offset = (
            anchor_frame_for_time(
                float(
                    row[
                        "window_start_time_sec"
                    ]
                ),
                frame_time_mapping,
                pose_count,
            )
        )

        if anchor_frame not in camera_cache:
            camera_cache[
                anchor_frame
            ] = make_camera_matrices(
                module,
                poses,
                correction,
                anchor_frame,
            )

        view, proj = camera_cache[
            anchor_frame
        ]

        xyz_local = (
            xyz_abs
            -
            dem_origin[
                None,
                :
            ]
        )

        xy_norm, ndc, clip_w = (
            project_local_xyz_to_normalized(
                xyz_local,
                view,
                proj,
            )
        )

        if not np.isfinite(
            xy_norm
        ).all():
            raise RuntimeError(
                f"{task} sample {i}: nonfinite normalized projection."
            )

        history_world[
            i
        ] = xyz_abs[
            :history_steps
        ]

        future_world[
            i
        ] = xyz_abs[
            history_steps:
        ]

        history_norm[
            i
        ] = xy_norm[
            :history_steps
        ]

        future_norm[
            i
        ] = xy_norm[
            history_steps:
        ]

        hist = history_norm[
            i
        ]

        fut = future_norm[
            i
        ]

        all_xy = np.concatenate(
            [
                hist,
                fut,
            ],
            axis=0,
        )

        in_frame = (
            (
                all_xy[
                    :,
                    0
                ]
                >=
                0.0
            )
            &
            (
                all_xy[
                    :,
                    0
                ]
                <=
                1.0
            )
            &
            (
                all_xy[
                    :,
                    1
                ]
                >=
                0.0
            )
            &
            (
                all_xy[
                    :,
                    1
                ]
                <=
                1.0
            )
        )

        audit_rows.append({
            "sample_index":
                int(
                    row[
                        "sample_index"
                    ]
                ),

            "track_id":
                int(
                    row[
                        "track_id"
                    ]
                ),

            "global_segment_id":
                int(
                    row[
                        "global_segment_id"
                    ]
                ),

            "gap_segment_id":
                int(
                    row[
                        "gap_segment_id"
                    ]
                ),

            "window_start_time_sec":
                float(
                    row[
                        "window_start_time_sec"
                    ]
                ),

            "anchor_frame":
                int(
                    anchor_frame
                ),

            "anchor_frame_float":
                float(
                    frame_float
                ),

            "anchor_pose_time_sec":
                float(
                    pose_time
                ),

            "anchor_time_offset_sec":
                float(
                    anchor_offset
                ),

            "anchor_time_offset_ms":
                float(
                    1000.0
                    *
                    anchor_offset
                ),

            "resampled_start_match_error_sec":
                float(
                    start_err
                ),

            "max_10hz_dt_error_sec":
                float(
                    max_dt_error
                ),

            "history_x_min":
                float(
                    np.min(
                        hist[
                            :,
                            0
                        ]
                    )
                ),

            "history_x_max":
                float(
                    np.max(
                        hist[
                            :,
                            0
                        ]
                    )
                ),

            "history_y_min":
                float(
                    np.min(
                        hist[
                            :,
                            1
                        ]
                    )
                ),

            "history_y_max":
                float(
                    np.max(
                        hist[
                            :,
                            1
                        ]
                    )
                ),

            "future_x_min":
                float(
                    np.min(
                        fut[
                            :,
                            0
                        ]
                    )
                ),

            "future_x_max":
                float(
                    np.max(
                        fut[
                            :,
                            0
                        ]
                    )
                ),

            "future_y_min":
                float(
                    np.min(
                        fut[
                            :,
                            1
                        ]
                    )
                ),

            "future_y_max":
                float(
                    np.max(
                        fut[
                            :,
                            1
                        ]
                    )
                ),

            "fraction_points_inside_image":
                float(
                    np.mean(
                        in_frame
                    )
                ),

            "clip_w_min":
                float(
                    np.min(
                        clip_w
                    )
                ),

            "clip_w_max":
                float(
                    np.max(
                        clip_w
                    )
                ),

            "ndc_z_min":
                float(
                    np.min(
                        ndc[
                            :,
                            2
                        ]
                    )
                ),

            "ndc_z_max":
                float(
                    np.max(
                        ndc[
                            :,
                            2
                        ]
                    )
                ),
        })

    audit = pd.DataFrame(
        audit_rows
    )

    npz_path = (
        OUT
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_windows.npz"
    )

    np.savez_compressed(
        npz_path,
        history_anchor_norm=history_norm,
        future_anchor_norm=future_norm,
        history_world_xyz=history_world,
        future_world_xyz=future_world,
    )

    audit_path = (
        OUT
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_metadata.csv"
    )

    merged_meta = metadata.merge(
        audit,
        on=[
            "sample_index",
            "track_id",
            "global_segment_id",
            "gap_segment_id",
            "window_start_time_sec",
        ],
        how="left",
        validate="one_to_one",
    )

    merged_meta.to_csv(
        audit_path,
        index=False
    )

    abs_offsets = np.abs(
        audit[
            "anchor_time_offset_sec"
        ].to_numpy(
            dtype=float
        )
    )

    frame_period = frame_time_mapping[
        "seconds_per_frame"
    ]

    half_frame = (
        0.5
        *
        frame_period
    )

    timing_pass = bool(
        np.max(
            abs_offsets
        )
        <=
        (
            half_frame
            +
            1e-9
        )
    )

    finite_pass = bool(
        np.isfinite(
            history_norm
        ).all()
        and
        np.isfinite(
            future_norm
        ).all()
    )

    summary = {
        "task":
            task,

        "samples":
            int(
                len(
                    metadata
                )
            ),

        "history_steps":
            int(
                history_steps
            ),

        "future_steps":
            int(
                horizon_steps
            ),

        "unique_tracks":
            int(
                metadata[
                    "track_id"
                ].nunique()
            ),

        "unique_anchor_frames":
            int(
                audit[
                    "anchor_frame"
                ].nunique()
            ),

        "median_abs_anchor_time_offset_ms":
            float(
                1000.0
                *
                np.median(
                    abs_offsets
                )
            ),

        "p95_abs_anchor_time_offset_ms":
            float(
                1000.0
                *
                np.quantile(
                    abs_offsets,
                    0.95
                )
            ),

        "max_abs_anchor_time_offset_ms":
            float(
                1000.0
                *
                np.max(
                    abs_offsets
                )
            ),

        "half_original_frame_period_ms":
            float(
                1000.0
                *
                half_frame
            ),

        "timing_alignment_within_nearest_frame_bound":
            timing_pass,

        "all_projected_values_finite":
            finite_pass,

        "fraction_all_points_inside_image":
            float(
                np.mean(
                    audit[
                        "fraction_points_inside_image"
                    ].to_numpy(
                        dtype=float
                    )
                )
            ),

        "history_norm_min":
            float(
                np.min(
                    history_norm
                )
            ),

        "history_norm_max":
            float(
                np.max(
                    history_norm
                )
            ),

        "future_norm_min":
            float(
                np.min(
                    future_norm
                )
            ),

        "future_norm_max":
            float(
                np.max(
                    future_norm
                )
            ),

        "npz":
            str(
                npz_path
            ),

        "npz_sha256":
            sha256_file(
                npz_path
            ),

        "metadata_csv":
            str(
                audit_path
            ),

        "metadata_sha256":
            sha256_file(
                audit_path
            ),
    }

    return (
        summary,
        timing_pass
        and
        finite_pass,
    )


def main():
    print(
        "\nSTEP 6D-K3C11 — BAMBI OBSERVATION-START ANCHOR NORMALIZED RECONSTRUCTION"
    )

    print(
        "Geometry-only reconstruction after K3C10 prelock."
    )

    print(
        "NO new model inference, NO q recalibration, NO Step7 geofence metrics, and NO alarm evaluation.\n"
    )

    require(
        PRELOCK
    )

    actual_prelock_hash = sha256_file(
        PRELOCK
    )

    if (
        actual_prelock_hash
        !=
        EXPECTED_PRELOCK_SHA256
    ):
        raise RuntimeError(
            "K3C10 prelock hash mismatch. "
            f"Expected {EXPECTED_PRELOCK_SHA256}, got {actual_prelock_hash}"
        )

    prelock = json.loads(
        PRELOCK.read_text(
            encoding="utf-8"
        )
    )

    if (
        prelock.get(
            "status"
        )
        !=
        "LOCKED_BEFORE_ANY_BAMBI_STEP7_OPERATIONAL_METRICS"
    ):
        raise RuntimeError(
            "K3C10 prelock status is not the expected locked state."
        )

    require(
        WORLD_RAW
    )

    require(
        RESAMPLED
    )

    world_raw = pd.read_csv(
        WORLD_RAW
    )

    resampled = pd.read_csv(
        RESAMPLED
    )

    required_resampled = {
        "track_id",
        "global_segment_id",
        "gap_segment_id",
        "time_sec",
        "utm_x",
        "utm_y",
        "utm_z",
    }

    missing = (
        required_resampled
        -
        set(
            resampled.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Resampled trajectory missing columns: {sorted(missing)}"
        )

    frame_time_mapping = fit_frame_time_mapping(
        world_raw
    )

    pose_path, poses = discover_enriched_pose()

    correction_path = discover_one(
        "6_correction.json",
        prefer_tokens=(
            "bambi_downloads",
        )
    )

    correction = json.loads(
        correction_path.read_text(
            encoding="utf-8"
        )
    )

    dem_json_path = discover_one(
        "6_matched_dem.json"
    )

    dem_meta = json.loads(
        dem_json_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        "origin"
        not in
        dem_meta
    ):
        raise RuntimeError(
            "DEM metadata has no origin."
        )

    dem_origin = np.asarray(
        dem_meta[
            "origin"
        ],
        dtype=float,
    )

    if dem_origin.size < 3:
        raise RuntimeError(
            "DEM origin must contain x,y,z."
        )

    dem_origin = dem_origin[
        :3
    ]

    module, module_path = load_bambi_module()

    print(
        "PRELOCK SHA256:"
    )
    print(
        actual_prelock_hash
    )

    print(
        "\nFRAME↔TIME MAPPING:"
    )
    print(
        json.dumps(
            frame_time_mapping,
            indent=2
        )
    )

    print(
        "\nENRICHED POSE SOURCE:"
    )
    print(
        pose_path
    )

    print(
        "\nDEM ORIGIN:"
    )
    print(
        dem_origin.tolist()
    )

    camera_cache = {}

    task_specs = [
        (
            "2s_to_2s",
            20,
        ),
        (
            "2s_to_5s",
            50,
        ),
    ]

    summaries = []

    all_pass = True

    for task, horizon_steps in task_specs:
        metadata_path = task_metadata_path(
            task
        )

        metadata = pd.read_csv(
            metadata_path
        )

        summary, passed = build_task(
            task,
            horizon_steps,
            metadata,
            resampled,
            dem_origin,
            poses,
            correction,
            module,
            frame_time_mapping,
            camera_cache,
        )

        summaries.append(
            summary
        )

        all_pass = (
            all_pass
            and
            passed
        )

        print(
            f"\n{task} SUMMARY:"
        )

        print(
            json.dumps(
                summary,
                indent=2
            )
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        OUT
        / "TABLE_step6dk3c11_anchor_normalized_reconstruction_summary.csv",
        index=False
    )

    if all_pass:
        status = (
            "ANCHOR_NORMALIZED_RECONSTRUCTION_COMPLETE_"
            "READY_FOR_DERIVED_DATASET_LOCK_BEFORE_SECONDARY_FROZEN_INFERENCE"
        )
    else:
        status = (
            "ANCHOR_NORMALIZED_RECONSTRUCTION_QC_FAILED_"
            "DO_NOT_RUN_SECONDARY_INFERENCE_OR_STEP7"
        )

    final = {
        "step":
            "STEP_6D_K3C11_BAMBI_OBSERVATION_START_ANCHOR_NORMALIZED_RECONSTRUCTION",

        "status":
            status,

        "prelock":
            str(
                PRELOCK
            ),

        "prelock_sha256_verified":
            actual_prelock_hash,

        "frame_time_mapping":
            frame_time_mapping,

        "anchor_pose_rule":
            (
                "Nearest original BAMBI pose frame to each 10 Hz window_start_time_sec. "
                "Timing error is audited against the half-original-frame bound."
            ),

        "world_to_pixel_convention":
            "col_PT_VT",

        "pixel_mapping":
            "top-left",

        "normalization":
            "x_norm=u/1024, y_norm=v/1024",

        "fovy_degrees":
            FOVY_DEG,

        "fovy_status":
            (
                "Assumed/default BAMBI reference-pipeline value; "
                "not independently measured camera calibration."
            ),

        "dem_origin":
            dem_origin.tolist(),

        "pose_source":
            str(
                pose_path
            ),

        "correction_source":
            str(
                correction_path
            ),

        "dem_metadata":
            str(
                dem_json_path
            ),

        "official_bambi_source":
            str(
                module_path
            ),

        "tasks":
            summaries,

        "analysis_role":
            (
                "Secondary frozen operational-transfer preparation. "
                "Not a new untouched confirmatory uncertainty holdout."
            ),

        "nonclaims": [
            "No BAMBI Step7 metric or alarm was generated.",
            "No model was retrained or tuned.",
            "No conformal q value was recalibrated.",
            "No virtual-geofence threshold or persistence rule was changed.",
            "The 50 degree FOVY remains an assumption/default.",
            "Virtual geofences remain normalized synthetic boundaries, not field boundaries.",
        ],
    }

    final_path = (
        OUT
        / "STEP6D_K3C11_SUMMARY.json"
    )

    final_path.write_text(
        json.dumps(
            final,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nSTATUS:"
    )
    print(
        status
    )

    print(
        "\nSTEP 6D-K3C11 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
