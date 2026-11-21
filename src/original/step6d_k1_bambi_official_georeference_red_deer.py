from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


# ============================================================
# STEP 6D-K1
# BAMBI FLIGHT 6 — OFFICIAL DEM GEOREFERENCING OF RED-DEER GT
#
# PURPOSE
# -------
# 1) Build a clean Red-deer-only MOT file from 6_gt.txt.
# 2) Convert matched poses to DEM-local location/rotation using BAMBI's
#    official add_relative_dem_position_to_poses.py utility.
# 3) Use BAMBI's official georeference_mot17_tracks implementation to
#    ray-project MOT bounding boxes onto the DEM.
# 4) Build per-detection UTM/world-coordinate trajectory centres.
# 5) Run geometry / continuity diagnostics ONLY.
#
# NO 10 Hz RESAMPLING.
# NO FORECASTING.
# NO CONFORMAL COVERAGE.
# NO STEP-7 WARNING EVALUATION.
#
# READS:
#   C:\Users\HP\Downloads\Dataset-main\Dataset-main
#   C:\Users\HP\Downloads\step6d_uncertainty_holdout\bambi_flight6\DEM
#
# WRITES ONLY:
#   C:\Users\HP\Downloads\step6d_uncertainty_holdout\bambi_flight6\step6dk_georef
# ============================================================


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

DATASET_REPO = (
    DOWNLOADS
    / "Dataset-main"
    / "Dataset-main"
)

BAMBI_DOWNLOADS = (
    DATASET_REPO
    / "bambi_downloads"
)

HOLDOUT = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
)

DEM_DIR = (
    HOLDOUT
    / "DEM"
)

OUT = (
    HOLDOUT
    / "step6dk_georef"
)

INPUT_BASE = (
    OUT
    / "mot_input"
)

INPUT_TEST = (
    INPUT_BASE
    / "test"
)

CORRECTION_DIR = (
    OUT
    / "correction"
)

GEOREF_BASE = (
    OUT
    / "georeferenced"
)

GEOREF_TEST = (
    GEOREF_BASE
    / "test"
)

GT_ORIGINAL = (
    BAMBI_DOWNLOADS
    / "6_gt.txt"
)

POSE_ORIGINAL = (
    BAMBI_DOWNLOADS
    / "6_matched_poses.json"
)

CORRECTION_ORIGINAL = (
    BAMBI_DOWNLOADS
    / "6_correction.json"
)

DEM_GLTF_ORIGINAL = (
    DEM_DIR
    / "6_matched_dem.glb"
)

DEM_META_ORIGINAL = (
    DEM_DIR
    / "6_matched_dem.json"
)

POSE_UTILITY = (
    DATASET_REPO
    / "add_relative_dem_position_to_poses.py"
)

RED_DEER_MOT = (
    INPUT_TEST
    / "6_gt.txt"
)

POSE_ENRICHED = (
    CORRECTION_DIR
    / "6_matched_poses.json"
)

WORK_DEM_GLTF = (
    CORRECTION_DIR
    / "6_dem.glb"
)

WORK_DEM_META = (
    CORRECTION_DIR
    / "6_dem_mesh_r2.json"
)

WORK_CORRECTION = (
    CORRECTION_DIR
    / "6_correction.json"
)

EMPTY_ADDITIONAL = (
    CORRECTION_DIR
    / "additional_corrections_empty.json"
)

GEOREF_OUTPUT = (
    GEOREF_TEST
    / "6_gt.txt"
)

FPS_LOCKED = 29.411834004179347

SPECIES_MATCH = "red deer"


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def prepare_dirs():
    for p in [
        OUT,
        INPUT_TEST,
        CORRECTION_DIR,
        GEOREF_TEST,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True
        )


def dependency_check():
    errors = []

    try:
        import pyproj  # noqa
    except Exception as e:
        errors.append(
            f"pyproj: {e}"
        )

    try:
        import trimesh  # noqa
    except Exception as e:
        errors.append(
            f"trimesh: {e}"
        )

    try:
        import scipy  # noqa
    except Exception as e:
        errors.append(
            f"scipy: {e}"
        )

    try:
        import bambi  # noqa
    except Exception as e:
        errors.append(
            f"bambi-detection: {e}"
        )

    try:
        import alfspy  # noqa
    except Exception as e:
        errors.append(
            f"alfspy: {e}"
        )

    if errors:
        print(
            "\nDEPENDENCY CHECK FAILED"
        )

        for e in errors:
            print(
                " -",
                e
            )

        print(
            "\nInstall the OFFICIAL BAMBI packages, then rerun:"
        )

        print(
            'python -m pip install "git+https://github.com/bambi-eco/bambi_detection.git@v1.0.0"'
        )

        print(
            'python -m pip install "AlfsPy[torch,embree] @ git+https://github.com/bambi-eco/alfs_py.git@v3.0.0"'
        )

        raise SystemExit(
            2
        )


def inspect_gt_and_write_red_deer():
    total = 0
    red_rows = []
    human_rows = 0
    other_rows = 0
    red_tracks = set()
    frames = set()

    with open(
        GT_ORIGINAL,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            parts = [
                p.strip()
                for p in line.split(",")
            ]

            if len(
                parts
            ) < 9:
                continue

            total += 1

            species = (
                parts[9]
                if len(
                    parts
                ) > 9
                else ""
            )

            species_low = species.lower()

            if SPECIES_MATCH in species_low:
                try:
                    frame_id = int(
                        float(
                            parts[0]
                        )
                    )

                    track_id = int(
                        float(
                            parts[1]
                        )
                    )
                except Exception:
                    continue

                # Canonical 9-column MOT only.
                # Official BAMBI parser needs:
                # frame, track, left, top, width, height, conf, class, visibility
                canonical = ",".join(
                    parts[:9]
                )

                red_rows.append(
                    canonical
                )

                red_tracks.add(
                    track_id
                )

                frames.add(
                    frame_id
                )

            elif "human" in species_low:
                human_rows += 1

            else:
                other_rows += 1

    if not red_rows:
        raise RuntimeError(
            "No Red-deer rows were found in 6_gt.txt."
        )

    with open(
        RED_DEER_MOT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        for row in red_rows:
            f.write(
                row
                +
                "\n"
            )

    return {
        "original_nonempty_rows":
            total,

        "red_deer_rows":
            len(
                red_rows
            ),

        "red_deer_tracks":
            len(
                red_tracks
            ),

        "red_deer_unique_frames":
            len(
                frames
            ),

        "human_rows":
            human_rows,

        "other_rows":
            other_rows,

        "red_deer_frame_min":
            min(
                frames
            ),

        "red_deer_frame_max":
            max(
                frames
            ),
    }


def create_dem_local_pose_copy():
    # Official BAMBI utility creates:
    #   location = [easting-origin_x, northing-origin_y, alt-origin_z]
    #   rotation = [pitch, roll, yaw]
    #
    # It writes the enriched pose file into CORRECTION_DIR.
    cmd = [
        sys.executable,
        str(
            POSE_UTILITY
        ),
        "--poses",
        str(
            POSE_ORIGINAL
        ),
        "--dem",
        str(
            DEM_META_ORIGINAL
        ),
        "--output",
        str(
            CORRECTION_DIR
        ),
    ]

    print(
        "\nRUNNING OFFICIAL BAMBI POSE->DEM-LOCAL UTILITY:"
    )

    print(
        " ".join(
            f'"{x}"'
            if " " in x
            else x
            for x in cmd
        )
    )

    subprocess.run(
        cmd,
        check=True
    )

    require_file(
        POSE_ENRICHED
    )

    with open(
        POSE_ENRICHED,
        "r",
        encoding="utf-8"
    ) as f:
        poses = json.load(
            f
        )

    images = poses.get(
        "images",
        []
    )

    if not images:
        raise RuntimeError(
            "Enriched pose JSON contains no images."
        )

    required = [
        "location",
        "rotation",
        "lat",
        "lng",
        "alt",
    ]

    missing_examples = []

    for i, img in enumerate(
        images[:50]
    ):
        missing = [
            key
            for key in required
            if key not in img
        ]

        if missing:
            missing_examples.append(
                {
                    "index":
                        i,

                    "missing":
                        missing,
                }
            )

    if missing_examples:
        raise RuntimeError(
            "Enriched pose structure is incomplete: "
            +
            str(
                missing_examples[:5]
            )
        )

    return {
        "pose_records":
            len(
                images
            ),

        "first_location":
            images[0].get(
                "location"
            ),

        "first_rotation":
            images[0].get(
                "rotation"
            ),

        "has_fovy":
            bool(
                images[0].get(
                    "fovy"
                )
            ),
    }


def prepare_official_georef_correction_folder():
    # The official georeference_mot17_tracks implementation expects:
    #   6_matched_poses.json
    #   6_correction.json
    #   6_dem.glb
    #   6_dem_mesh_r2.json
    #
    # Keep originals untouched; copy/rename only inside Downloads output folder.
    shutil.copy2(
        CORRECTION_ORIGINAL,
        WORK_CORRECTION
    )

    shutil.copy2(
        DEM_GLTF_ORIGINAL,
        WORK_DEM_GLTF
    )

    shutil.copy2(
        DEM_META_ORIGINAL,
        WORK_DEM_META
    )

    with open(
        EMPTY_ADDITIONAL,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "corrections":
                    {}
            },
            f,
            indent=2
        )

    return {
        "work_pose":
            str(
                POSE_ENRICHED
            ),

        "work_correction":
            str(
                WORK_CORRECTION
            ),

        "work_dem_glb":
            str(
                WORK_DEM_GLTF
            ),

        "work_dem_meta":
            str(
                WORK_DEM_META
            ),

        "additional_corrections":
            str(
                EMPTY_ADDITIONAL
            ),
    }


def run_official_georeferencing():
    # Force a headless CPU-friendly official backend choice.
    os.environ.setdefault(
        "ALFS_ENGINE",
        "torch"
    )

    os.environ.setdefault(
        "ALFS_RAYCASTER",
        "embree"
    )

    os.environ.setdefault(
        "ALFS_DEVICE",
        "cpu"
    )

    from alfspy.core.rendering import Resolution
    from bambi.georeference_mot17_tracks import run

    print(
        "\nRUNNING OFFICIAL BAMBI MOT->DEM GEOREFERENCING"
    )

    print(
        "Primary settings:"
    )

    print(
        "  input_resolution = 1024x1024"
    )

    print(
        "  add_offsets = True"
    )

    print(
        "  apply_smoothing = True (official default)"
    )

    print(
        "  smoothing = Savitzky-Golay window 11, polyorder 2"
    )

    print(
        "  subfolder = test"
    )

    run(
        base_dir=str(
            INPUT_BASE
        ),
        correction_folder=str(
            CORRECTION_DIR
        ),
        target_base=str(
            GEOREF_BASE
        ),
        additional_corrections_path=str(
            EMPTY_ADDITIONAL
        ),
        input_resolution=Resolution(
            1024,
            1024
        ),
        skip_existing=False,
        transform_to_target_crs=False,
        add_offsets=True,
        apply_smoothing=True,
        window_length=11,
        polyorder=2,
        include_visibility=True,
        subfolders=[
            "test"
        ],
    )

    require_file(
        GEOREF_OUTPUT
    )


def parse_georef_output():
    rows = []

    with open(
        GEOREF_OUTPUT,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        for line in f:
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            parts = s.split()

            if len(
                parts
            ) < 10:
                continue

            try:
                row = {
                    "frame_id":
                        int(
                            float(
                                parts[0]
                            )
                        ),

                    "track_id":
                        int(
                            float(
                                parts[1]
                            )
                        ),

                    "utm_x_min":
                        float(
                            parts[2]
                        ),

                    "utm_y_min":
                        float(
                            parts[3]
                        ),

                    "utm_z_min":
                        float(
                            parts[4]
                        ),

                    "utm_x_max":
                        float(
                            parts[5]
                        ),

                    "utm_y_max":
                        float(
                            parts[6]
                        ),

                    "utm_z_max":
                        float(
                            parts[7]
                        ),

                    "confidence":
                        float(
                            parts[8]
                        ),

                    "class_id":
                        int(
                            float(
                                parts[9]
                            )
                        ),

                    "visibility":
                        (
                            float(
                                parts[10]
                            )
                            if len(
                                parts
                            )
                            > 10
                            else np.nan
                        ),
                }

            except Exception:
                continue

            rows.append(
                row
            )

    df = pd.DataFrame(
        rows
    )

    if len(
        df
    ) == 0:
        raise RuntimeError(
            "Official georeferencing produced no parseable rows."
        )

    coords = [
        "utm_x_min",
        "utm_y_min",
        "utm_z_min",
        "utm_x_max",
        "utm_y_max",
        "utm_z_max",
    ]

    valid = np.ones(
        len(
            df
        ),
        dtype=bool
    )

    for c in coords:
        v = pd.to_numeric(
            df[
                c
            ],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        valid &= np.isfinite(
            v
        )

        valid &= (
            v
            !=
            -1
        )

    df[
        "georef_valid"
    ] = valid

    # Centre of the official georeferenced 3D AABB.
    df[
        "utm_x_center"
    ] = (
        df[
            "utm_x_min"
        ]
        +
        df[
            "utm_x_max"
        ]
    ) / 2.0

    df[
        "utm_y_center"
    ] = (
        df[
            "utm_y_min"
        ]
        +
        df[
            "utm_y_max"
        ]
    ) / 2.0

    df[
        "utm_z_center"
    ] = (
        df[
            "utm_z_min"
        ]
        +
        df[
            "utm_z_max"
        ]
    ) / 2.0

    df[
        "time_sec"
    ] = (
        df[
            "frame_id"
        ]
        /
        FPS_LOCKED
    )

    return df


def trajectory_diagnostics(df):
    valid = df[
        df[
            "georef_valid"
        ]
    ].copy()

    valid = valid.sort_values(
        [
            "track_id",
            "frame_id"
        ]
    )

    step_rows = []

    track_rows = []

    for track_id, g in valid.groupby(
        "track_id"
    ):
        g = g.sort_values(
            "frame_id"
        )

        frames = g[
            "frame_id"
        ].to_numpy(
            dtype=int
        )

        x = g[
            "utm_x_center"
        ].to_numpy(
            dtype=float
        )

        y = g[
            "utm_y_center"
        ].to_numpy(
            dtype=float
        )

        z = g[
            "utm_z_center"
        ].to_numpy(
            dtype=float
        )

        if len(
            g
        ) >= 2:
            dframe = np.diff(
                frames
            )

            dt = (
                dframe
                /
                FPS_LOCKED
            )

            dx = np.diff(
                x
            )

            dy = np.diff(
                y
            )

            dz = np.diff(
                z
            )

            dist_xy = np.sqrt(
                dx * dx
                +
                dy * dy
            )

            dist_3d = np.sqrt(
                dx * dx
                +
                dy * dy
                +
                dz * dz
            )

            speed_xy = np.divide(
                dist_xy,
                dt,
                out=np.full_like(
                    dist_xy,
                    np.nan,
                    dtype=float
                ),
                where=dt > 0
            )

            for i in range(
                len(
                    dframe
                )
            ):
                step_rows.append({
                    "track_id":
                        int(
                            track_id
                        ),

                    "frame_from":
                        int(
                            frames[i]
                        ),

                    "frame_to":
                        int(
                            frames[i + 1]
                        ),

                    "frame_gap":
                        int(
                            dframe[i]
                        ),

                    "dt_sec":
                        float(
                            dt[i]
                        ),

                    "step_xy_m":
                        float(
                            dist_xy[i]
                        ),

                    "step_3d_m":
                        float(
                            dist_3d[i]
                        ),

                    "speed_xy_mps":
                        float(
                            speed_xy[i]
                        ),
                })

        track_rows.append({
            "track_id":
                int(
                    track_id
                ),

            "rows":
                int(
                    len(
                        g
                    )
                ),

            "frame_min":
                int(
                    frames.min()
                ),

            "frame_max":
                int(
                    frames.max()
                ),

            "temporal_span_sec":
                float(
                    (
                        frames.max()
                        -
                        frames.min()
                    )
                    /
                    FPS_LOCKED
                ),

            "x_span_m":
                float(
                    x.max()
                    -
                    x.min()
                ),

            "y_span_m":
                float(
                    y.max()
                    -
                    y.min()
                ),

            "z_span_m":
                float(
                    z.max()
                    -
                    z.min()
                ),

            "path_extent_xy_m":
                float(
                    np.sqrt(
                        (
                            x.max()
                            -
                            x.min()
                        )
                        ** 2
                        +
                        (
                            y.max()
                            -
                            y.min()
                        )
                        ** 2
                    )
                ),
        })

    step_df = pd.DataFrame(
        step_rows
    )

    track_df = pd.DataFrame(
        track_rows
    )

    if len(
        step_df
    ):
        speed = pd.to_numeric(
            step_df[
                "speed_xy_mps"
            ],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        finite_speed = speed[
            np.isfinite(
                speed
            )
        ]

        if len(
            finite_speed
        ):
            med = float(
                np.median(
                    finite_speed
                )
            )

            mad = float(
                np.median(
                    np.abs(
                        finite_speed
                        -
                        med
                    )
                )
            )

            robust_limit = (
                med
                +
                10.0
                *
                1.4826
                *
                mad
                if mad
                >
                0
                else float(
                    np.quantile(
                        finite_speed,
                        0.995
                    )
                )
            )

            step_df[
                "robust_jump_flag"
            ] = (
                step_df[
                    "speed_xy_mps"
                ]
                >
                robust_limit
            )

        else:
            robust_limit = np.nan

            step_df[
                "robust_jump_flag"
            ] = False

    else:
        robust_limit = np.nan

    return (
        valid,
        track_df,
        step_df,
        robust_limit
    )


def summarise(
    gt_audit,
    pose_audit,
    df,
    valid,
    track_df,
    step_df,
    robust_limit
):
    summary = {
        "step":
            "STEP_6D_K1_BAMBI_OFFICIAL_GEOREFERENCING",

        "flight_id":
            6,

        "species":
            "Cervus elaphus (Red deer)",

        "fps":
            FPS_LOCKED,

        "gt_audit":
            gt_audit,

        "pose_audit":
            pose_audit,

        "georeferencing":
            {
                "output_rows":
                    int(
                        len(
                            df
                        )
                    ),

                "valid_rows":
                    int(
                        df[
                            "georef_valid"
                        ].sum()
                    ),

                "invalid_rows":
                    int(
                        (
                            ~df[
                                "georef_valid"
                            ]
                        ).sum()
                    ),

                "success_rate":
                    float(
                        df[
                            "georef_valid"
                        ].mean()
                    ),

                "valid_tracks":
                    int(
                        valid[
                            "track_id"
                        ].nunique()
                    ),

                "valid_frames":
                    int(
                        valid[
                            "frame_id"
                        ].nunique()
                    ),
            },

        "coordinate_extent_valid":
            (
                {
                    "utm_x_min":
                        float(
                            valid[
                                "utm_x_center"
                            ].min()
                        ),

                    "utm_x_max":
                        float(
                            valid[
                                "utm_x_center"
                            ].max()
                        ),

                    "utm_y_min":
                        float(
                            valid[
                                "utm_y_center"
                            ].min()
                        ),

                    "utm_y_max":
                        float(
                            valid[
                                "utm_y_center"
                            ].max()
                        ),

                    "utm_z_min":
                        float(
                            valid[
                                "utm_z_center"
                            ].min()
                        ),

                    "utm_z_max":
                        float(
                            valid[
                                "utm_z_center"
                            ].max()
                        ),
                }
                if len(
                    valid
                )
                else {}
            ),

        "track_diagnostics":
            {
                "tracks":
                    int(
                        len(
                            track_df
                        )
                    ),

                "median_rows_per_track":
                    (
                        float(
                            track_df[
                                "rows"
                            ].median()
                        )
                        if len(
                            track_df
                        )
                        else np.nan
                    ),

                "median_temporal_span_sec":
                    (
                        float(
                            track_df[
                                "temporal_span_sec"
                            ].median()
                        )
                        if len(
                            track_df
                        )
                        else np.nan
                    ),
            },

        "step_diagnostics":
            {
                "steps":
                    int(
                        len(
                            step_df
                        )
                    ),

                "median_speed_xy_mps":
                    (
                        float(
                            step_df[
                                "speed_xy_mps"
                            ].median()
                        )
                        if len(
                            step_df
                        )
                        else np.nan
                    ),

                "p95_speed_xy_mps":
                    (
                        float(
                            step_df[
                                "speed_xy_mps"
                            ].quantile(
                                0.95
                            )
                        )
                        if len(
                            step_df
                        )
                        else np.nan
                    ),

                "p99_speed_xy_mps":
                    (
                        float(
                            step_df[
                                "speed_xy_mps"
                            ].quantile(
                                0.99
                            )
                        )
                        if len(
                            step_df
                        )
                        else np.nan
                    ),

                "max_speed_xy_mps":
                    (
                        float(
                            step_df[
                                "speed_xy_mps"
                            ].max()
                        )
                        if len(
                            step_df
                        )
                        else np.nan
                    ),

                "robust_jump_flag_threshold_mps":
                    robust_limit,

                "robust_jump_flags":
                    (
                        int(
                            step_df[
                                "robust_jump_flag"
                            ].sum()
                        )
                        if (
                            len(
                                step_df
                            )
                            and
                            "robust_jump_flag"
                            in step_df.columns
                        )
                        else 0
                    ),
            },

        "status":
            (
                "K1_GEOMETRY_BUILT_PENDING_QUALITY_REVIEW"
                if len(
                    valid
                )
                    >
                    0
                else
                "K1_FAILED_NO_VALID_GEOREFERENCED_ROWS"
            ),

        "next_if_passed":
            (
                "Step 6D-K2: quality filtering + 10 Hz resampling and locked "
                "2s->2s / 2s->5s window construction."
            ),
    }

    return summary


def main():
    print(
        "\nSTEP 6D-K1 — BAMBI FLIGHT 6 OFFICIAL GEOREFERENCING"
    )

    print(
        "\nThis step builds world/UTM ground trajectories only."
    )

    print(
        "It does NOT run forecasting, conformal coverage, or Step-7 alarms."
    )

    prepare_dirs()

    for p in [
        GT_ORIGINAL,
        POSE_ORIGINAL,
        CORRECTION_ORIGINAL,
        DEM_GLTF_ORIGINAL,
        DEM_META_ORIGINAL,
        POSE_UTILITY,
    ]:
        require_file(
            p
        )

    dependency_check()

    print(
        "\nINPUT HASHES:"
    )

    input_hash_rows = []

    for label, path in [
        (
            "gt",
            GT_ORIGINAL
        ),
        (
            "matched_poses",
            POSE_ORIGINAL
        ),
        (
            "correction",
            CORRECTION_ORIGINAL
        ),
        (
            "dem_glb",
            DEM_GLTF_ORIGINAL
        ),
        (
            "dem_metadata",
            DEM_META_ORIGINAL
        ),
    ]:
        h = sha256_file(
            path
        )

        input_hash_rows.append({
            "label":
                label,

            "path":
                str(
                    path
                ),

            "sha256":
                h,
        })

        print(
            label,
            h
        )

    pd.DataFrame(
        input_hash_rows
    ).to_csv(
        OUT
        /
        "step6dk_input_hashes.csv",
        index=False
    )

    print(
        "\nFILTERING RED-DEER GT:"
    )

    gt_audit = inspect_gt_and_write_red_deer()

    print(
        json.dumps(
            gt_audit,
            indent=2
        )
    )

    print(
        "\nCREATING DEM-LOCAL POSE COPY:"
    )

    pose_audit = create_dem_local_pose_copy()

    print(
        json.dumps(
            pose_audit,
            indent=2
        )
    )

    prep_audit = prepare_official_georef_correction_folder()

    with open(
        OUT
        /
        "step6dk_working_paths.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            prep_audit,
            f,
            indent=2
        )

    run_official_georeferencing()

    df = parse_georef_output()

    df.to_csv(
        OUT
        /
        "step6dk_all_georeferenced_red_deer_rows.csv",
        index=False
    )

    (
        valid,
        track_df,
        step_df,
        robust_limit
    ) = trajectory_diagnostics(
        df
    )

    valid.to_csv(
        OUT
        /
        "step6dk_valid_world_trajectory_rows.csv",
        index=False
    )

    track_df.to_csv(
        OUT
        /
        "step6dk_track_geometry_summary.csv",
        index=False
    )

    step_df.to_csv(
        OUT
        /
        "step6dk_step_motion_diagnostics.csv",
        index=False
    )

    summary = summarise(
        gt_audit,
        pose_audit,
        df,
        valid,
        track_df,
        step_df,
        robust_limit
    )

    with open(
        OUT
        /
        "STEP6D_K1_SUMMARY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    with open(
        OUT
        /
        "STEP6D_K1_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K1 — BAMBI FLIGHT 6 OFFICIAL GEOREFERENCING\n"
        )

        f.write(
            "="
            *
            78
            +
            "\n\n"
        )

        f.write(
            "GT AUDIT\n"
        )

        f.write(
            "--------\n"
        )

        f.write(
            json.dumps(
                gt_audit,
                indent=2
            )
        )

        f.write(
            "\n\nPOSE AUDIT\n"
        )

        f.write(
            "----------\n"
        )

        f.write(
            json.dumps(
                pose_audit,
                indent=2
            )
        )

        f.write(
            "\n\nGEOREFERENCING SUMMARY\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            json.dumps(
                summary[
                    "georeferencing"
                ],
                indent=2
            )
        )

        f.write(
            "\n\nCOORDINATE EXTENT\n"
        )

        f.write(
            "-----------------\n"
        )

        f.write(
            json.dumps(
                summary[
                    "coordinate_extent_valid"
                ],
                indent=2
            )
        )

        f.write(
            "\n\nTRACK DIAGNOSTICS\n"
        )

        f.write(
            "-----------------\n"
        )

        f.write(
            json.dumps(
                summary[
                    "track_diagnostics"
                ],
                indent=2
            )
        )

        f.write(
            "\n\nSTEP-MOTION DIAGNOSTICS\n"
        )

        f.write(
            "-----------------------\n"
        )

        f.write(
            json.dumps(
                summary[
                    "step_diagnostics"
                ],
                indent=2
            )
        )

        f.write(
            "\n\nSTATUS\n"
        )

        f.write(
            "------\n"
        )

        f.write(
            summary[
                "status"
            ]
            +
            "\n"
        )

        f.write(
            "\nIMPORTANT: robust jump flags are diagnostic only. "
            "No rows/tracks are removed in K1.\n"
        )

    print(
        "\nGEOREFERENCING SUMMARY:"
    )

    print(
        json.dumps(
            summary[
                "georeferencing"
            ],
            indent=2
        )
    )

    print(
        "\nCOORDINATE EXTENT:"
    )

    print(
        json.dumps(
            summary[
                "coordinate_extent_valid"
            ],
            indent=2
        )
    )

    print(
        "\nTRACK DIAGNOSTICS:"
    )

    print(
        json.dumps(
            summary[
                "track_diagnostics"
            ],
            indent=2
        )
    )

    print(
        "\nSTEP-MOTION DIAGNOSTICS:"
    )

    print(
        json.dumps(
            summary[
                "step_diagnostics"
            ],
            indent=2
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        summary[
            "status"
        ]
    )

    print(
        "\nSTEP 6D-K1 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
