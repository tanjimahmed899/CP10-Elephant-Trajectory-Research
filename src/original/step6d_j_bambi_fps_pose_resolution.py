from pathlib import Path
import csv
import hashlib
import json
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# STEP 6D-J
# BAMBI FLIGHT 6: FPS + POSE STRUCTURE RESOLUTION
#
# NO model prediction / forecast error / coverage is computed.
# Existing Step 6D pre-lock MUST already exist and is verified.
# ============================================================

REPO = Path(
    r"C:\Users\HP\Downloads\Dataset-main\Dataset-main"
)

BASE = REPO / "bambi_downloads"

OUT = Path(
    r"C:\Users\HP\Downloads\step6d_uncertainty_holdout\bambi_flight6"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

LOCK = OUT / "STEP6D_BAMBI_FLIGHT6_PRELOCK.json"

GT = BASE / "6_gt.txt"
POSES = BASE / "6_matched_poses.json"
META = BASE / "6_metadata.json"

FLIGHT_META_DIR = REPO / "flight_metadata"


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_prelock():
    if not LOCK.exists():
        raise FileNotFoundError(
            f"Pre-lock missing: {LOCK}"
        )

    with open(
        LOCK,
        "r",
        encoding="utf-8"
    ) as f:
        lock = json.load(f)

    expected = {
        x["role"]: x["sha256"]
        for x in lock.get(
            "files",
            []
        )
    }

    current_map = {
        "gt": GT,
        "poses": POSES,
        "metadata": META,
        "mapping": BASE / "6_track_mapping.json",
        "correction": BASE / "6_correction.json",
    }

    current = {
        role: sha256_file(path)
        for role, path in current_map.items()
    }

    if expected != current:
        raise RuntimeError(
            "Pre-lock hash verification FAILED."
        )

    return lock


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def parse_gt():
    rows = []

    with open(
        GT,
        "r",
        encoding="utf-8",
        errors="ignore",
        newline=""
    ) as f:
        reader = csv.reader(f)

        for p in reader:
            if len(p) < 10:
                continue

            try:
                rows.append({
                    "frame_id": int(round(float(p[0]))),
                    "track_id": int(round(float(p[1]))),
                    "x": float(p[2]),
                    "y": float(p[3]),
                    "width": float(p[4]),
                    "height": float(p[5]),
                    "confidence_or_valid": float(p[6]),
                    "class_id": int(round(float(p[7]))),
                    "visibility": float(p[8]),
                    "species_name": str(p[9]).strip(),
                })
            except Exception:
                continue

    if not rows:
        raise RuntimeError(
            "Could not parse BAMBI GT."
        )

    return pd.DataFrame(rows)


def flatten_scalars(obj, prefix=""):
    rows = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)

            if isinstance(v, (dict, list)):
                rows.extend(
                    flatten_scalars(
                        v,
                        p
                    )
                )
            else:
                rows.append(
                    (p, v)
                )

    elif isinstance(obj, list):
        for i, v in enumerate(obj[:100]):
            p = f"{prefix}[{i}]"

            if isinstance(v, (dict, list)):
                rows.extend(
                    flatten_scalars(
                        v,
                        p
                    )
                )
            else:
                rows.append(
                    (p, v)
                )

    return rows


def numeric(v):
    try:
        x = float(v)

        if np.isfinite(x):
            return x
    except Exception:
        pass

    return None


def parse_frame_from_text(value):
    if value is None:
        return None

    s = str(value)

    # Prefer a long numeric token near the end of a filename.
    nums = re.findall(
        r"(?<!\d)(\d{1,10})(?!\d)",
        s
    )

    if not nums:
        return None

    try:
        return int(
            nums[-1]
        )
    except Exception:
        return None


def parse_timestamp_value(v):
    if v is None:
        return None

    # ISO-like time
    if isinstance(v, str):
        s = v.strip()

        try:
            ts = pd.to_datetime(
                s,
                utc=True
            )

            if not pd.isna(ts):
                return float(
                    ts.timestamp()
                )
        except Exception:
            pass

        x = numeric(s)

    else:
        x = numeric(v)

    if x is None:
        return None

    # Heuristic unit normalization.
    ax = abs(x)

    if ax > 1e17:
        return x / 1e9

    if ax > 1e14:
        return x / 1e6

    if ax > 1e11:
        return x / 1e3

    return x


def collect_pose_records(obj):
    records = []

    frame_keys = [
        "frame",
        "frame_id",
        "frame_idx",
        "frame_index",
        "index",
        "id",
    ]

    image_keys = [
        "imagefile",
        "image",
        "filename",
        "file",
        "path",
    ]

    time_keys = [
        "timestamp",
        "time",
        "datetime",
        "utc",
        "utc_time",
        "timestamp_ms",
        "time_ms",
    ]

    def walk(x, path="", parent_key=None):
        if isinstance(x, dict):
            keys_lower = {
                str(k).lower(): k
                for k in x.keys()
            }

            pose_signal = any(
                k in keys_lower
                for k in [
                    "location",
                    "rotation",
                    "lat",
                    "latitude",
                    "lng",
                    "lon",
                    "longitude",
                    "fovy",
                    "alt",
                    "altitude",
                ]
            )

            frame = None
            frame_source = None

            for k in frame_keys:
                if k in keys_lower:
                    val = x[
                        keys_lower[k]
                    ]

                    try:
                        frame = int(
                            round(
                                float(
                                    val
                                )
                            )
                        )
                        frame_source = k
                        break
                    except Exception:
                        pass

            if frame is None:
                for k in image_keys:
                    if k in keys_lower:
                        frame = parse_frame_from_text(
                            x[
                                keys_lower[k]
                            ]
                        )

                        if frame is not None:
                            frame_source = k
                            break

            if frame is None and parent_key is not None:
                try:
                    frame = int(
                        str(
                            parent_key
                        )
                    )
                    frame_source = "parent_key"
                except Exception:
                    pass

            timestamp = None
            timestamp_source = None

            for k in time_keys:
                if k in keys_lower:
                    timestamp = parse_timestamp_value(
                        x[
                            keys_lower[k]
                        ]
                    )

                    if timestamp is not None:
                        timestamp_source = k
                        break

            if pose_signal:
                rec = {
                    "json_path": path,
                    "frame_id": frame,
                    "frame_source": frame_source,
                    "timestamp_sec": timestamp,
                    "timestamp_source": timestamp_source,
                    "has_location": "location" in keys_lower,
                    "has_rotation": "rotation" in keys_lower,
                    "has_fovy": "fovy" in keys_lower,
                    "has_lat": (
                        "lat" in keys_lower
                        or "latitude" in keys_lower
                    ),
                    "has_lng": (
                        "lng" in keys_lower
                        or "lon" in keys_lower
                        or "longitude" in keys_lower
                    ),
                    "has_altitude": (
                        "alt" in keys_lower
                        or "altitude" in keys_lower
                    ),
                    "keys": ";".join(
                        map(
                            str,
                            x.keys()
                        )
                    ),
                }

                # Keep selected values as compact text.
                for wanted in [
                    "imagefile",
                    "location",
                    "rotation",
                    "lat",
                    "lng",
                    "lon",
                    "alt",
                    "altitude",
                    "fovy",
                ]:
                    if wanted in keys_lower:
                        rec[wanted] = str(
                            x[
                                keys_lower[wanted]
                            ]
                        )

                records.append(rec)

            for k, v in x.items():
                walk(
                    v,
                    f"{path}.{k}" if path else str(k),
                    k
                )

        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(
                    v,
                    f"{path}[{i}]",
                    i
                )

    walk(obj)

    df = pd.DataFrame(records)

    if len(df):
        # Remove exact duplicate structures created by nested traversal overlap.
        keep_cols = [
            c
            for c in [
                "json_path",
                "frame_id",
                "timestamp_sec",
                "keys",
            ]
            if c in df.columns
        ]

        df = df.drop_duplicates(
            subset=keep_cols
        )

    return df


def candidate_fps_from_flat(flat_rows, source_name):
    candidates = []

    for key, value in flat_rows:
        kl = key.lower()

        if any(
            token in kl
            for token in [
                "fps",
                "frame_rate",
                "framerate",
                "frames_per_second",
            ]
        ):
            x = numeric(value)

            if x is not None and 1 <= x <= 240:
                candidates.append({
                    "source": source_name,
                    "key": key,
                    "value": x,
                    "method": "explicit_fps_key",
                })

    return candidates


def frame_duration_fps(flat_rows, source_name):
    frame_values = []
    duration_values = []

    for key, value in flat_rows:
        kl = key.lower()
        x = numeric(value)

        if x is None:
            continue

        if any(
            token in kl
            for token in [
                "frame_count",
                "frames_count",
                "num_frames",
                "number_of_frames",
                "total_frames",
            ]
        ):
            if x > 0:
                frame_values.append(
                    (key, x)
                )

        if any(
            token in kl
            for token in [
                "duration_sec",
                "duration_seconds",
                "duration",
            ]
        ):
            if x > 0:
                duration_values.append(
                    (key, x)
                )

    rows = []

    for fk, frames in frame_values:
        for dk, duration in duration_values:
            fps = frames / duration

            if 1 <= fps <= 240:
                rows.append({
                    "source": source_name,
                    "key": f"{fk}/{dk}",
                    "value": fps,
                    "method": "frames_div_duration",
                })

    return rows


def infer_fps_from_pose_times(pose_df):
    if len(pose_df) == 0:
        return []

    q = pose_df.dropna(
        subset=[
            "frame_id",
            "timestamp_sec"
        ]
    ).copy()

    if len(q) < 3:
        return []

    q = q.sort_values(
        "frame_id"
    )

    frame = q[
        "frame_id"
    ].to_numpy(
        dtype=float
    )

    t = q[
        "timestamp_sec"
    ].to_numpy(
        dtype=float
    )

    df = np.diff(
        frame
    )

    dt = np.diff(
        t
    )

    good = (
        (df > 0)
        &
        (dt > 0)
        &
        np.isfinite(
            df
        )
        &
        np.isfinite(
            dt
        )
    )

    if not np.any(good):
        return []

    ratios = df[
        good
    ] / dt[
        good
    ]

    ratios = ratios[
        (ratios >= 1)
        &
        (ratios <= 240)
    ]

    if len(ratios) == 0:
        return []

    return [
        {
            "source": "6_matched_poses.json",
            "key": "frame_delta/time_delta",
            "value": float(
                np.median(
                    ratios
                )
            ),
            "method": "pose_frame_timestamp_ratio",
        }
    ]


def scan_flight_metadata():
    rows = []
    fps_candidates = []

    if not FLIGHT_META_DIR.exists():
        return (
            pd.DataFrame(),
            []
        )

    for p in FLIGHT_META_DIR.rglob("*"):
        if not p.is_file():
            continue

        suffix = p.suffix.lower()

        try:
            if suffix == ".json":
                obj = load_json(p)

                # Include file if filename suggests flight 6 or a scalar exact 6
                # appears under a flight/id key.
                flat = flatten_scalars(
                    obj
                )

                relevant = (
                    re.search(
                        r"(^|[^0-9])6([^0-9]|$)",
                        p.stem
                    )
                    is not None
                )

                for key, value in flat:
                    kl = key.lower()

                    if (
                        (
                            "flight" in kl
                            or kl.endswith(".id")
                            or kl == "id"
                        )
                        and
                        str(value).strip() == "6"
                    ):
                        relevant = True

                if relevant:
                    rows.append({
                        "path": str(p),
                        "type": "json",
                        "scalar_preview": "; ".join(
                            f"{k}={v}"
                            for k, v in flat[:80]
                        ),
                    })

                    fps_candidates.extend(
                        candidate_fps_from_flat(
                            flat,
                            str(p)
                        )
                    )

                    fps_candidates.extend(
                        frame_duration_fps(
                            flat,
                            str(p)
                        )
                    )

            elif suffix == ".csv":
                df = pd.read_csv(p)

                id_cols = [
                    c
                    for c in df.columns
                    if str(c).lower() in [
                        "flight",
                        "flight_id",
                        "id",
                        "flightid",
                    ]
                ]

                if not id_cols:
                    continue

                mask = pd.Series(
                    False,
                    index=df.index
                )

                for c in id_cols:
                    mask = (
                        mask
                        |
                        (
                            df[c]
                            .astype(str)
                            .str.strip()
                            ==
                            "6"
                        )
                    )

                subset = df[
                    mask
                ]

                if len(subset):
                    rows.append({
                        "path": str(p),
                        "type": "csv",
                        "scalar_preview": subset.head(
                            5
                        ).to_json(
                            orient="records"
                        ),
                    })

                    for c in df.columns:
                        cl = str(c).lower()

                        if any(
                            term in cl
                            for term in [
                                "fps",
                                "frame_rate",
                                "framerate",
                                "frames_per_second",
                            ]
                        ):
                            for v in subset[c]:
                                x = numeric(v)

                                if x is not None and 1 <= x <= 240:
                                    fps_candidates.append({
                                        "source": str(p),
                                        "key": str(c),
                                        "value": x,
                                        "method": "flight_metadata_csv",
                                    })

        except Exception:
            continue

    return (
        pd.DataFrame(rows),
        fps_candidates
    )


def choose_fps(candidates):
    if not candidates:
        return (
            None,
            None
        )

    df = pd.DataFrame(
        candidates
    )

    # Prefer explicit keys, then flight metadata CSV, then pose timestamp inference.
    priority = {
        "explicit_fps_key": 0,
        "flight_metadata_csv": 1,
        "frames_div_duration": 2,
        "pose_frame_timestamp_ratio": 3,
    }

    df["priority"] = df[
        "method"
    ].map(
        priority
    ).fillna(
        9
    )

    df = df.sort_values(
        [
            "priority",
            "source",
            "key"
        ]
    )

    best = df.iloc[0]

    return (
        float(
            best[
                "value"
            ]
        ),
        df
    )


def main():
    print(
        "\nSTEP 6D-J: BAMBI FLIGHT 6 FPS + POSE RESOLUTION"
    )

    verify_prelock()

    print(
        "\nPRE-RESULT LOCK HASHES VERIFIED."
    )

    gt = parse_gt()

    meta = load_json(
        META
    )

    poses = load_json(
        POSES
    )

    meta_flat = flatten_scalars(
        meta
    )

    poses_flat = flatten_scalars(
        poses
    )

    pose_df = collect_pose_records(
        poses
    )

    pose_df.to_csv(
        OUT
        /
        "flight6_pose_records_resolved.csv",
        index=False
    )

    metadata_relevant = []

    for k, v in meta_flat:
        kl = k.lower()

        if any(
            term in kl
            for term in [
                "fps",
                "frame",
                "duration",
                "video",
                "species",
                "camera",
                "thermal",
                "resolution",
                "width",
                "height",
            ]
        ):
            metadata_relevant.append({
                "key": k,
                "value": str(v)
            })

    pd.DataFrame(
        metadata_relevant
    ).to_csv(
        OUT
        /
        "flight6_metadata_relevant_keys.csv",
        index=False
    )

    flight_meta_df, flight_meta_fps = scan_flight_metadata()

    flight_meta_df.to_csv(
        OUT
        /
        "flight6_repo_flight_metadata_hits.csv",
        index=False
    )

    fps_candidates = []

    fps_candidates.extend(
        candidate_fps_from_flat(
            meta_flat,
            "6_metadata.json"
        )
    )

    fps_candidates.extend(
        frame_duration_fps(
            meta_flat,
            "6_metadata.json"
        )
    )

    fps_candidates.extend(
        candidate_fps_from_flat(
            poses_flat,
            "6_matched_poses.json"
        )
    )

    fps_candidates.extend(
        frame_duration_fps(
            poses_flat,
            "6_matched_poses.json"
        )
    )

    fps_candidates.extend(
        infer_fps_from_pose_times(
            pose_df
        )
    )

    fps_candidates.extend(
        flight_meta_fps
    )

    fps, fps_df = choose_fps(
        fps_candidates
    )

    if fps_df is None:
        fps_df = pd.DataFrame(
            columns=[
                "source",
                "key",
                "value",
                "method",
                "priority",
            ]
        )

    fps_df.to_csv(
        OUT
        /
        "flight6_fps_candidates.csv",
        index=False
    )

    # Pose-to-GT frame overlap.
    pose_frame_values = set()

    if len(pose_df) and "frame_id" in pose_df.columns:
        pose_frame_values = set(
            int(x)
            for x in pose_df[
                "frame_id"
            ].dropna()
        )

    gt_frames = set(
        int(x)
        for x in gt[
            "frame_id"
        ]
    )

    overlap = (
        gt_frames
        &
        pose_frame_values
    )

    pose_coverage = (
        len(
            overlap
        )
        /
        len(
            gt_frames
        )
        if gt_frames
        else np.nan
    )

    # Track structure with resolved FPS.
    track_rows = []

    for track_id, g in gt.groupby(
        "track_id"
    ):
        frames = sorted(
            set(
                int(x)
                for x in g[
                    "frame_id"
                ]
            )
        )

        first = frames[
            0
        ]

        last = frames[
            -1
        ]

        span = (
            last
            -
            first
            +
            1
        )

        duration = (
            (
                last
                -
                first
            )
            /
            fps
            if fps is not None
            else None
        )

        gaps = np.diff(
            np.asarray(
                frames,
                dtype=int
            )
        )

        species = sorted(
            set(
                str(x)
                for x in g[
                    "species_name"
                ]
            )
        )

        track_rows.append({
            "track_id": int(track_id),
            "species_names": ";".join(species),
            "rows": len(g),
            "first_frame": first,
            "last_frame": last,
            "span_frames": span,
            "unique_annotation_frames": len(frames),
            "duration_sec": duration,
            "median_gap_frames": (
                float(
                    np.median(
                        gaps
                    )
                )
                if len(gaps)
                else None
            ),
            "max_gap_frames": (
                int(
                    np.max(
                        gaps
                    )
                )
                if len(gaps)
                else None
            ),
            "supports_2s_to_2s_by_duration": (
                bool(
                    duration >= 4.0
                )
                if duration is not None
                else False
            ),
            "supports_2s_to_5s_by_duration": (
                bool(
                    duration >= 7.0
                )
                if duration is not None
                else False
            ),
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

    tracks.to_csv(
        OUT
        /
        "flight6_track_structure_resolved.csv",
        index=False
    )

    # Red-deer-only eligibility counts.
    red = tracks[
        tracks[
            "species_names"
        ].str.contains(
            "Red deer",
            case=False,
            na=False
        )
    ]

    n_red = int(
        len(
            red
        )
    )

    n22 = int(
        red[
            "supports_2s_to_2s_by_duration"
        ].sum()
    )

    n25 = int(
        red[
            "supports_2s_to_5s_by_duration"
        ].sum()
    )

    # Pose structure summary.
    pose_summary = {
        "pose_records_detected":
            int(
                len(
                    pose_df
                )
            ),

        "pose_records_with_frame_id":
            int(
                pose_df[
                    "frame_id"
                ].notna().sum()
            )
            if len(
                pose_df
            )
            else 0,

        "pose_records_with_location":
            int(
                pose_df[
                    "has_location"
                ].sum()
            )
            if len(
                pose_df
            )
            else 0,

        "pose_records_with_rotation":
            int(
                pose_df[
                    "has_rotation"
                ].sum()
            )
            if len(
                pose_df
            )
            else 0,

        "pose_records_with_fovy":
            int(
                pose_df[
                    "has_fovy"
                ].sum()
            )
            if len(
                pose_df
            )
            else 0,

        "pose_records_with_lat":
            int(
                pose_df[
                    "has_lat"
                ].sum()
            )
            if len(
                pose_df
            )
            else 0,

        "pose_records_with_lng":
            int(
                pose_df[
                    "has_lng"
                ].sum()
            )
            if len(
                pose_df
            )
            else 0,

        "gt_unique_frames":
            int(
                len(
                    gt_frames
                )
            ),

        "gt_pose_frame_overlap":
            int(
                len(
                    overlap
                )
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
    }

    with open(
        OUT
        /
        "STEP6D_J_AUDIT.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "fps_selected":
                    fps,

                "red_deer_tracks":
                    n_red,

                "red_deer_tracks_supporting_2s_to_2s":
                    n22,

                "red_deer_tracks_supporting_2s_to_5s":
                    n25,

                "pose_summary":
                    pose_summary,

                "forecast_model_loaded":
                    False,

                "forecast_error_computed":
                    False,

                "uncertainty_coverage_computed":
                    False,
            },
            f,
            indent=2
        )

    with open(
        OUT
        /
        "STEP6D_J_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-J: BAMBI FLIGHT 6 FPS + POSE RESOLUTION\n"
        )
        f.write(
            "=" * 78 + "\n\n"
        )

        f.write(
            f"Selected FPS: {fps}\n\n"
        )

        f.write(
            "FPS CANDIDATES\n"
        )
        f.write(
            "--------------\n"
        )
        f.write(
            fps_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPOSE SUMMARY\n"
        )
        f.write(
            "------------\n"
        )
        f.write(
            json.dumps(
                pose_summary,
                indent=2
            )
        )

        f.write(
            "\n\nRED-DEER TRACK VIABILITY\n"
        )
        f.write(
            "------------------------\n"
        )
        f.write(
            f"Red-deer tracks: {n_red}\n"
        )
        f.write(
            f"Support 2->2 by temporal span: {n22}\n"
        )
        f.write(
            f"Support 2->5 by temporal span: {n25}\n\n"
        )
        f.write(
            red.head(
                40
            ).to_string(
                index=False
            )
        )

        f.write(
            "\n\nNo model prediction/error/coverage was computed.\n"
        )

    print(
        "\nFPS RESOLUTION:"
    )

    print(
        "selected_fps =",
        fps
    )

    if len(
        fps_df
    ):
        print(
            fps_df.to_string(
                index=False
            )
        )
    else:
        print(
            "NO FPS CANDIDATE FOUND"
        )

    print(
        "\nPOSE STRUCTURE RESOLUTION:"
    )

    for k, v in pose_summary.items():
        print(
            f"{k} = {v}"
        )

    print(
        "\nRED-DEER TRACK VIABILITY:"
    )

    print(
        "red_deer_tracks =",
        n_red
    )

    print(
        "supports_2s_to_2s =",
        n22
    )

    print(
        "supports_2s_to_5s =",
        n25
    )

    print(
        "\nTOP RED-DEER TRACKS:"
    )

    if len(
        red
    ):
        print(
            red.head(
                25
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nSTEP 6D-J COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
