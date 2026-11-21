from pathlib import Path
import csv
import hashlib
import json
import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================
# STEP 6D-I
# BAMBI FLIGHT 6: PRE-RESULT LOCK + STRUCTURAL / GEO AUDIT
#
# IMPORTANT:
# - NO forecasting model is loaded.
# - NO forecast error or coverage is computed.
# - Selection/performance is not changed based on this audit.
# - The five downloaded Flight-6 files are SHA-256 locked first.
# ============================================================

BASE = Path(
    r"C:\Users\HP\Downloads\Dataset-main\Dataset-main\bambi_downloads"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step6d_uncertainty_holdout\bambi_flight6"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

FLIGHT = "6"

FILES = {
    "gt":
        BASE / f"{FLIGHT}_gt.txt",

    "poses":
        BASE / f"{FLIGHT}_matched_poses.json",

    "metadata":
        BASE / f"{FLIGHT}_metadata.json",

    "mapping":
        BASE / f"{FLIGHT}_track_mapping.json",

    "correction":
        BASE / f"{FLIGHT}_correction.json",
}


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def recursive_items(obj, prefix=""):
    rows = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = (
                f"{prefix}.{k}"
                if prefix
                else str(k)
            )

            if isinstance(v, (dict, list)):
                rows.extend(
                    recursive_items(
                        v,
                        key
                    )
                )
            else:
                rows.append(
                    (
                        key,
                        v
                    )
                )

    elif isinstance(obj, list):
        # Do not explode huge pose arrays completely.
        rows.append(
            (
                prefix + ".__list_length__",
                len(obj)
            )
        )

        for i, v in enumerate(
            obj[:5]
        ):
            key = f"{prefix}[{i}]"

            if isinstance(v, (dict, list)):
                rows.extend(
                    recursive_items(
                        v,
                        key
                    )
                )
            else:
                rows.append(
                    (
                        key,
                        v
                    )
                )

    return rows


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def find_values_by_key_terms(
    obj,
    terms
):
    hits = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                p = (
                    f"{path}.{k}"
                    if path
                    else str(k)
                )

                kl = str(k).lower()

                if any(
                    term in kl
                    for term in terms
                ):
                    if not isinstance(
                        v,
                        (
                            dict,
                            list
                        )
                    ):
                        hits.append(
                            (
                                p,
                                v
                            )
                        )

                walk(
                    v,
                    p
                )

        elif isinstance(x, list):
            for i, v in enumerate(
                x[:20]
            ):
                walk(
                    v,
                    f"{path}[{i}]"
                )

    walk(
        obj
    )

    return hits


def detect_delimiter(path):
    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        first = ""

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            first = line
            break

    if not first:
        raise RuntimeError(
            "GT file is empty."
        )

    if "," in first:
        return ","

    return None


def read_mot(path):
    """
    BAMBI extended MOT parser.

    Expected observed row shape:
    frame,track,x,y,w,h,confidence,class_id,visibility,
    species_name,extra_1,extra_2,extra_3

    Species is textual, so the whole row must NOT be cast to float.
    """
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
        newline=""
    ) as f:

        reader = csv.reader(
            f
        )

        for parts in reader:
            if not parts:
                continue

            parts = [
                str(x).strip()
                for x in parts
            ]

            if len(parts) < 10:
                continue

            try:
                frame_id = int(
                    round(
                        float(
                            parts[0]
                        )
                    )
                )

                track_id = int(
                    round(
                        float(
                            parts[1]
                        )
                    )
                )

                x = float(
                    parts[2]
                )

                y = float(
                    parts[3]
                )

                width = float(
                    parts[4]
                )

                height = float(
                    parts[5]
                )

                confidence_or_valid = float(
                    parts[6]
                )

                class_id = int(
                    round(
                        float(
                            parts[7]
                        )
                    )
                )

                visibility = float(
                    parts[8]
                )

            except Exception:
                continue

            species_name = parts[9]

            extras = []

            for value in parts[10:]:
                try:
                    extras.append(
                        float(
                            value
                        )
                    )
                except Exception:
                    extras.append(
                        np.nan
                    )

            row = {
                "frame_id":
                    frame_id,

                "track_id":
                    track_id,

                "x":
                    x,

                "y":
                    y,

                "width":
                    width,

                "height":
                    height,

                "confidence_or_valid":
                    confidence_or_valid,

                "class_id":
                    class_id,

                "visibility":
                    visibility,

                "species_name":
                    species_name,
            }

            for i, value in enumerate(
                extras,
                start=1
            ):
                row[
                    f"extra_{i}"
                ] = value

            rows.append(
                row
            )

    if not rows:
        raise RuntimeError(
            "No BAMBI extended-MOT rows parsed."
        )

    df = pd.DataFrame(
        rows
    )

    return (
        df,
        ",",
        len(
            df.columns
        )
    )


def track_temporal_stats(
    frames,
    fps
):
    """
    Temporal structural audit using actual source frame IDs.

    BAMBI annotations may not exist on every source frame, so strict +1
    adjacency is not used as the primary viability criterion.
    """
    f = sorted(
        set(
            int(x)
            for x in frames
        )
    )

    if not f:
        return {
            "first_frame": None,
            "last_frame": None,
            "span_frames": 0,
            "unique_annotation_frames": 0,
            "duration_sec": None,
            "median_positive_gap_frames": None,
            "max_gap_frames": None,
        }

    gaps = np.diff(
        np.asarray(
            f,
            dtype=int
        )
    )

    positive = gaps[
        gaps > 0
    ]

    duration_sec = None

    if fps is not None and fps > 0:
        duration_sec = (
            (
                f[-1]
                -
                f[0]
            )
            /
            float(
                fps
            )
        )

    return {
        "first_frame":
            int(
                f[0]
            ),

        "last_frame":
            int(
                f[-1]
            ),

        "span_frames":
            int(
                f[-1]
                -
                f[0]
                +
                1
            ),

        "unique_annotation_frames":
            int(
                len(
                    f
                )
            ),

        "duration_sec":
            (
                float(
                    duration_sec
                )
                if duration_sec is not None
                else None
            ),

        "median_positive_gap_frames":
            (
                float(
                    np.median(
                        positive
                    )
                )
                if len(
                    positive
                )
                else None
            ),

        "max_gap_frames":
            (
                int(
                    np.max(
                        positive
                    )
                )
                if len(
                    positive
                )
                else None
            ),
    }


def extract_pose_frames(obj):
    frames = []

    def frame_from_string(s):
        m = re.search(
            r"(\d{4,10})",
            str(s)
        )

        if m:
            return int(
                m.group(1)
            )

        return None

    if isinstance(obj, dict):
        candidates = []

        for key in [
            "images",
            "poses",
            "frames",
            "data"
        ]:
            if isinstance(
                obj.get(key),
                list
            ):
                candidates = obj[
                    key
                ]
                break

        if candidates:
            for item in candidates:
                if not isinstance(
                    item,
                    dict
                ):
                    continue

                found = None

                for key in [
                    "frame",
                    "frame_id",
                    "frame_idx",
                    "frame_index",
                    "index"
                ]:
                    if key in item:
                        try:
                            found = int(
                                item[
                                    key
                                ]
                            )
                            break
                        except Exception:
                            pass

                if found is None:
                    for key in [
                        "imagefile",
                        "image",
                        "filename",
                        "file"
                    ]:
                        if key in item:
                            found = frame_from_string(
                                item[
                                    key
                                ]
                            )

                            if found is not None:
                                break

                if found is not None:
                    frames.append(
                        found
                    )

    return sorted(
        set(
            frames
        )
    )


def pick_numeric_value(
    hits
):
    for path, value in hits:
        try:
            v = float(
                value
            )

            if np.isfinite(
                v
            ):
                return (
                    path,
                    v
                )
        except Exception:
            pass

    return (
        None,
        None
    )


def summarize_track_mapping(
    mapping
):
    if isinstance(
        mapping,
        dict
    ):
        return {
            "top_type":
                "dict",

            "top_keys":
                ";".join(
                    list(
                        map(
                            str,
                            mapping.keys()
                        )
                    )[:50]
                ),

            "top_length":
                len(
                    mapping
                )
        }

    if isinstance(
        mapping,
        list
    ):
        return {
            "top_type":
                "list",

            "top_keys":
                "",

            "top_length":
                len(
                    mapping
                )
        }

    return {
        "top_type":
            type(
                mapping
            ).__name__,

        "top_keys":
            "",

        "top_length":
            None
    }


def main():
    print(
        "\nSTEP 6D-I: BAMBI FLIGHT 6 PRE-RESULT LOCK + STRUCTURAL AUDIT"
    )

    # --------------------------------------------------------
    # Required file check
    # --------------------------------------------------------

    missing = [
        str(p)
        for p in FILES.values()
        if not p.exists()
    ]

    if missing:
        print(
            "\nFILES FOUND IN BAMBI DOWNLOAD DIRECTORY:"
        )

        if BASE.exists():
            for p in sorted(
                BASE.iterdir()
            ):
                print(
                    p.name
                )

        raise FileNotFoundError(
            "Missing required BAMBI files:\n"
            +
            "\n".join(
                missing
            )
        )

    # --------------------------------------------------------
    # LOCK FIRST -- BEFORE STRUCTURAL OUTCOME IS INSPECTED
    # --------------------------------------------------------

    file_lock_rows = []

    for role, path in FILES.items():
        file_lock_rows.append({
            "role":
                role,

            "path":
                str(
                    path
                ),

            "size_bytes":
                int(
                    path.stat().st_size
                ),

            "sha256":
                sha256_file(
                    path
                )
        })

    lock_df = pd.DataFrame(
        file_lock_rows
    )

    lock_df.to_csv(
        OUT
        /
        "bambi_flight6_file_lock.csv",
        index=False
    )

    prelock = {
        "lock_stage":
            "STEP_6D_BAMBI_FLIGHT6_PRE_RESULT_LOCK",

        "dataset":
            "BAMBI",

        "flight_id":
            6,

        "selection_note":
            (
                "Flight 6 was selected and downloaded before any model "
                "prediction, forecast error, or uncertainty coverage result "
                "was inspected."
            ),

        "model_predictions_or_forecast_errors_seen_before_lock":
            False,

        "files":
            file_lock_rows,

        "scope_note":
            (
                "BAMBI is an external UAV wildlife dataset. Species and "
                "trajectory suitability are audited after this file lock. "
                "If the species is non-elephant, any successful evaluation "
                "is cross-species robustness evidence, not elephant-specific "
                "external validation."
            )
    }

    prelock_path = (
        OUT
        /
        "STEP6D_BAMBI_FLIGHT6_PRELOCK.json"
    )

    if prelock_path.exists():
        existing_prelock = load_json(
            prelock_path
        )

        existing_hashes = {
            x[
                "role"
            ]:
            x[
                "sha256"
            ]
            for x in existing_prelock.get(
                "files",
                []
            )
        }

        current_hashes = {
            x[
                "role"
            ]:
            x[
                "sha256"
            ]
            for x in file_lock_rows
        }

        if existing_hashes != current_hashes:
            raise RuntimeError(
                "Existing pre-lock hashes do not match current BAMBI files."
            )

        print(
            "\nEXISTING PRE-RESULT FILE LOCK VERIFIED:"
        )

    else:
        with open(
            prelock_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                prelock,
                f,
                indent=2
            )

        print(
            "\nPRE-RESULT FILE LOCK CREATED:"
        )

    print(
        prelock_path
    )

    # --------------------------------------------------------
    # Parse files AFTER lock
    # --------------------------------------------------------

    metadata = load_json(
        FILES[
            "metadata"
        ]
    )

    poses = load_json(
        FILES[
            "poses"
        ]
    )

    mapping = load_json(
        FILES[
            "mapping"
        ]
    )

    correction = load_json(
        FILES[
            "correction"
        ]
    )

    mot, delimiter, ncols = read_mot(
        FILES[
            "gt"
        ]
    )

    # --------------------------------------------------------
    # Metadata audit
    # --------------------------------------------------------

    metadata_flat = pd.DataFrame(
        recursive_items(
            metadata
        ),
        columns=[
            "key_path",
            "value"
        ]
    )

    metadata_flat.to_csv(
        OUT
        /
        "flight6_metadata_flat_preview.csv",
        index=False
    )

    fps_hits = find_values_by_key_terms(
        metadata,
        [
            "fps",
            "frame_rate",
            "framerate"
        ]
    )

    fps_path, fps = pick_numeric_value(
        fps_hits
    )

    species_hits = find_values_by_key_terms(
        metadata,
        [
            "species",
            "animal",
            "class",
            "taxon"
        ]
    )

    altitude_hits = find_values_by_key_terms(
        metadata,
        [
            "altitude",
            "height",
            "agl"
        ]
    )

    calib_hits = find_values_by_key_terms(
        metadata,
        [
            "intrinsic",
            "calib",
            "focal",
            "fov",
            "camera_matrix"
        ]
    )

    dem_hits = find_values_by_key_terms(
        metadata,
        [
            "dem",
            "terrain",
            "elevation"
        ]
    )

    # --------------------------------------------------------
    # MOT track audit
    # --------------------------------------------------------

    track_rows = []

    for track_id, g in mot.groupby(
        "track_id"
    ):
        frame_ids = g[
            "frame_id"
        ].astype(
            int
        ).tolist()

        temporal = track_temporal_stats(
            frame_ids,
            fps
        )

        classes = sorted(
            set(
                int(x)
                for x in g[
                    "class_id"
                ].dropna()
            )
        )

        species = sorted(
            set(
                str(x)
                for x in g[
                    "species_name"
                ].dropna()
            )
        )

        track_rows.append({
            "track_id":
                int(
                    track_id
                ),

            "rows":
                len(
                    g
                ),

            **temporal,

            "class_ids":
                ";".join(
                    map(
                        str,
                        classes
                    )
                ),

            "species_names":
                ";".join(
                    species
                )
        })

    tracks = pd.DataFrame(
        track_rows
    ).sort_values(
        [
            "duration_sec",
            "rows"
        ],
        ascending=[
            False,
            False
        ],
        na_position="last"
    )

    # --------------------------------------------------------
    # Pose coverage audit
    # --------------------------------------------------------

    pose_frames = extract_pose_frames(
        poses
    )

    gt_frames = sorted(
        set(
            mot[
                "frame_id"
            ].astype(
                int
            )
        )
    )

    if pose_frames:
        pose_set = set(
            pose_frames
        )

        gt_pose_overlap = sum(
            1
            for x in gt_frames
            if x in pose_set
        )

        pose_coverage = (
            gt_pose_overlap
            /
            len(
                gt_frames
            )
            if gt_frames
            else np.nan
        )
    else:
        gt_pose_overlap = 0
        pose_coverage = np.nan

    # --------------------------------------------------------
    # Horizon viability
    # --------------------------------------------------------

    if fps is not None and fps > 0:
        frames_2_to_2 = int(
            math.ceil(
                fps
                *
                4.0
            )
        )

        frames_2_to_5 = int(
            math.ceil(
                fps
                *
                7.0
            )
        )

        tracks[
            "supports_2s_to_2s_by_duration"
        ] = (
            tracks[
                "duration_sec"
            ]
            >=
            4.0
        )

        tracks[
            "supports_2s_to_5s_by_duration"
        ] = (
            tracks[
                "duration_sec"
            ]
            >=
            7.0
        )

        # Require enough temporal observations to construct at least the
        # canonical 10 Hz history+future samples after interpolation/resampling.
        tracks[
            "enough_annotations_for_2s_to_2s_10hz"
        ] = (
            tracks[
                "unique_annotation_frames"
            ]
            >=
            4
        )

        tracks[
            "enough_annotations_for_2s_to_5s_10hz"
        ] = (
            tracks[
                "unique_annotation_frames"
            ]
            >=
            7
        )

        tracks[
            "supports_2s_to_2s_original_rate"
        ] = (
            tracks[
                "supports_2s_to_2s_by_duration"
            ]
            &
            tracks[
                "enough_annotations_for_2s_to_2s_10hz"
            ]
        )

        tracks[
            "supports_2s_to_5s_original_rate"
        ] = (
            tracks[
                "supports_2s_to_5s_by_duration"
            ]
            &
            tracks[
                "enough_annotations_for_2s_to_5s_10hz"
            ]
        )

        n_22 = int(
            tracks[
                "supports_2s_to_2s_original_rate"
            ].sum()
        )

        n_25 = int(
            tracks[
                "supports_2s_to_5s_original_rate"
            ].sum()
        )

    else:
        frames_2_to_2 = None
        frames_2_to_5 = None

        tracks[
            "supports_2s_to_2s_by_duration"
        ] = False

        tracks[
            "supports_2s_to_5s_by_duration"
        ] = False

        tracks[
            "enough_annotations_for_2s_to_2s_10hz"
        ] = False

        tracks[
            "enough_annotations_for_2s_to_5s_10hz"
        ] = False

        tracks[
            "supports_2s_to_2s_original_rate"
        ] = False

        tracks[
            "supports_2s_to_5s_original_rate"
        ] = False

        n_22 = None
        n_25 = None

    tracks.to_csv(
        OUT
        /
        "flight6_track_structure.csv",
        index=False
    )

    mot.to_csv(
        OUT
        /
        "flight6_gt_parsed.csv",
        index=False
    )

    # --------------------------------------------------------
    # Mapping/correction structure
    # --------------------------------------------------------

    mapping_summary = summarize_track_mapping(
        mapping
    )

    correction_flat = pd.DataFrame(
        recursive_items(
            correction
        ),
        columns=[
            "key_path",
            "value"
        ]
    )

    correction_flat.to_csv(
        OUT
        /
        "flight6_correction_flat.csv",
        index=False
    )

    pose_flat_preview = pd.DataFrame(
        recursive_items(
            poses
        ),
        columns=[
            "key_path",
            "value"
        ]
    )

    pose_flat_preview.to_csv(
        OUT
        /
        "flight6_pose_structure_preview.csv",
        index=False
    )

    # --------------------------------------------------------
    # Georeferencing readiness
    # --------------------------------------------------------

    has_pose_data = (
        len(
            pose_frames
        )
        >
        0
        or
        (
            isinstance(
                poses,
                dict
            )
            and
            len(
                poses
            )
            >
            0
        )
    )

    has_calib_hint = (
        len(
            calib_hits
        )
        >
        0
    )

    has_dem_hint = (
        len(
            dem_hits
        )
        >
        0
    )

    georef_status = (
        "POSES_PRESENT_NEED_GROUND_PROJECTION_RESOURCE_AUDIT"
    )

    if (
        has_pose_data
        and
        has_calib_hint
        and
        has_dem_hint
    ):
        georef_status = (
            "METADATA_APPEARS_TO_INCLUDE_POSE_CALIBRATION_AND_TERRAIN_HINTS"
        )

    elif not has_pose_data:
        georef_status = (
            "NOT_READY_POSE_DATA_NOT_DETECTED"
        )

    species_counts_df = (
        mot
        .groupby(
            [
                "species_name",
                "class_id"
            ],
            as_index=False
        )
        .agg(
            detections=(
                "track_id",
                "size"
            ),
            tracks=(
                "track_id",
                "nunique"
            ),
            frames=(
                "frame_id",
                "nunique"
            )
        )
        .sort_values(
            [
                "tracks",
                "detections"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    species_counts_df.to_csv(
        OUT
        /
        "flight6_species_counts_from_gt.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save compact audit
    # --------------------------------------------------------

    audit = {
        "dataset":
            "BAMBI",

        "flight":
            6,

        "mot_rows":
            int(
                len(
                    mot
                )
            ),

        "mot_columns":
            list(
                mot.columns
            ),

        "mot_numeric_column_count":
            int(
                ncols
            ),

        "mot_delimiter":
            (
                "comma"
                if delimiter == ","
                else "whitespace"
            ),

        "unique_tracks":
            int(
                mot[
                    "track_id"
                ].nunique()
            ),

        "unique_gt_frames":
            int(
                len(
                    gt_frames
                )
            ),

        "gt_min_frame":
            int(
                min(
                    gt_frames
                )
            ),

        "gt_max_frame":
            int(
                max(
                    gt_frames
                )
            ),

        "fps_value":
            fps,

        "fps_metadata_path":
            fps_path,

        "frames_required_2s_to_2s_original_rate":
            frames_2_to_2,

        "frames_required_2s_to_5s_original_rate":
            frames_2_to_5,

        "tracks_supporting_2s_to_2s":
            n_22,

        "tracks_supporting_2s_to_5s":
            n_25,

        "pose_frames_detected":
            int(
                len(
                    pose_frames
                )
            ),

        "gt_frames_with_pose_match":
            int(
                gt_pose_overlap
            ),

        "gt_pose_frame_coverage":
            (
                float(
                    pose_coverage
                )
                if np.isfinite(
                    pose_coverage
                )
                else None
            ),

        "species_metadata_hits":
            [
                {
                    "path":
                        p,

                    "value":
                        str(
                            v
                        )
                }
                for p, v in species_hits[
                    :50
                ]
            ],

        "altitude_metadata_hits":
            [
                {
                    "path":
                        p,

                    "value":
                        str(
                            v
                        )
                }
                for p, v in altitude_hits[
                    :50
                ]
            ],

        "calibration_metadata_hits":
            [
                {
                    "path":
                        p,

                    "value":
                        str(
                            v
                        )
                }
                for p, v in calib_hits[
                    :50
                ]
            ],

        "terrain_dem_metadata_hits":
            [
                {
                    "path":
                        p,

                    "value":
                        str(
                            v
                        )
                }
                for p, v in dem_hits[
                    :50
                ]
            ],

        "track_mapping_structure":
            mapping_summary,

        "georeferencing_readiness":
            georef_status,

        "forecast_model_loaded":
            False,

        "forecast_error_computed":
            False,

        "uncertainty_coverage_computed":
            False,
    }

    with open(
        OUT
        /
        "STEP6D_I_AUDIT.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    with open(
        OUT
        /
        "STEP6D_I_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 6D-I: BAMBI FLIGHT 6 PRE-RESULT LOCK + STRUCTURAL AUDIT\n"
        )

        f.write(
            "=" * 82
            +
            "\n\n"
        )

        f.write(
            "PRE-RESULT LOCK\n"
        )

        f.write(
            "---------------\n"
        )

        f.write(
            lock_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nMOT STRUCTURE\n"
        )

        f.write(
            "-------------\n"
        )

        f.write(
            f"Rows: {len(mot)}\n"
        )

        f.write(
            f"Columns: {list(mot.columns)}\n"
        )

        f.write(
            f"Tracks: {mot['track_id'].nunique()}\n"
        )

        f.write(
            f"GT frames: {len(gt_frames)} "
            f"({min(gt_frames)} to {max(gt_frames)})\n"
        )

        f.write(
            f"FPS detected: {fps} via {fps_path}\n"
        )

        f.write(
            f"2->2 required contiguous original frames: "
            f"{frames_2_to_2}\n"
        )

        f.write(
            f"2->5 required contiguous original frames: "
            f"{frames_2_to_5}\n"
        )

        f.write(
            f"Tracks structurally supporting 2->2: {n_22}\n"
        )

        f.write(
            f"Tracks structurally supporting 2->5: {n_25}\n"
        )

        f.write(
            "\nTOP TRACKS\n"
        )

        f.write(
            "----------\n"
        )

        f.write(
            tracks.head(
                30
            ).to_string(
                index=False
            )
        )

        f.write(
            "\n\nPOSE AUDIT\n"
        )

        f.write(
            "----------\n"
        )

        f.write(
            f"Pose frames detected: {len(pose_frames)}\n"
        )

        f.write(
            f"GT frame / pose overlap: {gt_pose_overlap} "
            f"of {len(gt_frames)}\n"
        )

        f.write(
            f"Pose coverage: {pose_coverage}\n"
        )

        f.write(
            "\nSPECIES METADATA HITS\n"
        )

        f.write(
            "---------------------\n"
        )

        if species_hits:
            for p, v in species_hits[
                :50
            ]:
                f.write(
                    f"{p}: {v}\n"
                )
        else:
            f.write(
                "NONE DETECTED\n"
            )

        f.write(
            "\nSPECIES COUNTS FROM GT\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            species_counts_df.to_string(
                index=False
            )
        )

        f.write(
            "\n"
        )

        f.write(
            "\nGEOREFERENCE RESOURCE HINTS\n"
        )

        f.write(
            "---------------------------\n"
        )

        f.write(
            f"Status: {georef_status}\n"
        )

        f.write(
            f"Calibration-like metadata hits: {len(calib_hits)}\n"
        )

        f.write(
            f"DEM/terrain-like metadata hits: {len(dem_hits)}\n"
        )

        f.write(
            f"Altitude-like metadata hits: {len(altitude_hits)}\n"
        )

        f.write(
            "\nNo forecasting model, forecast error, or uncertainty "
            "coverage was computed in this step.\n"
        )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print(
        "\nMOT / TRACK STRUCTURE:"
    )

    print(
        "mot_rows =",
        len(
            mot
        )
    )

    print(
        "mot_columns =",
        list(
            mot.columns
        )
    )

    print(
        "unique_tracks =",
        mot[
            "track_id"
        ].nunique()
    )

    print(
        "unique_gt_frames =",
        len(
            gt_frames
        )
    )

    print(
        "fps =",
        fps,
        "from",
        fps_path
    )

    print(
        "tracks_supporting_2s_to_2s =",
        n_22
    )

    print(
        "tracks_supporting_2s_to_5s =",
        n_25
    )

    print(
        "\nTOP TRACKS:"
    )

    print(
        tracks.head(
            20
        ).to_string(
            index=False
        )
    )

    print(
        "\nPOSE COVERAGE:"
    )

    print(
        "pose_frames_detected =",
        len(
            pose_frames
        )
    )

    print(
        "gt_frames_with_pose_match =",
        gt_pose_overlap
    )

    print(
        "gt_pose_frame_coverage =",
        pose_coverage
    )

    print(
        "\nSPECIES METADATA HITS:"
    )

    if species_hits:
        for p, v in species_hits[
            :30
        ]:
            print(
                f"{p}: {v}"
            )
    else:
        print(
            "NONE"
        )

    print(
        "\nSPECIES COUNTS FROM GT:"
    )

    print(
        species_counts_df.to_string(
            index=False
        )
    )

    print(
        "\nGEOREFERENCE RESOURCE AUDIT:"
    )

    print(
        "status =",
        georef_status
    )

    print(
        "calibration_like_hits =",
        len(
            calib_hits
        )
    )

    print(
        "dem_or_terrain_like_hits =",
        len(
            dem_hits
        )
    )

    print(
        "altitude_like_hits =",
        len(
            altitude_hits
        )
    )

    print(
        "\nSTEP 6D-I COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
