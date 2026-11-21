
from pathlib import Path
import json
import hashlib
import importlib.util
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
CAPSTONE = Path(r"E:\AIUB\Capstone Project")
MAIN1 = CAPSTONE / "Journal Based Results" / "Main Result"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R13_ROOT = REVISION / "R13_end_to_end_pipeline"

R13A_V2_DIR = R13_ROOT / "R13A_V2_project_aware_asset_audit"
R13A_V2_MANIFEST = R13A_V2_DIR / "R13A_V2_MANIFEST.json"
R13A_V2_CKPTS = R13A_V2_DIR / "R13A_V2_DETECTOR_CHECKPOINTS.csv"

FINAL_META_2 = (
    MAIN1
    / "final_neural_trajectory_dataset"
    / "2s_to_2s_birdsai_test_metadata.csv"
)

CLEANED_TRACKS = (
    MAIN1
    / "trajectory_dataset"
    / "cleaned_track_points.csv"
)

TESTREAL_IMAGES = (
    MAIN1
    / "conservation_drones_test_real"
    / "TestReal"
    / "images"
)

OUT = R13_ROOT / "R13B_detector_data_tracker_compatibility_audit"
OUT.mkdir(parents=True, exist_ok=True)

# Frozen detector results from the established full RGB experiment.
TARGET_DETECTOR = {
    "precision": 0.918,
    "recall": 0.884,
    "map50": 0.940,
    "map50_95": 0.698,
}

TOL = {
    "precision": 0.04,
    "recall": 0.04,
    "map50": 0.04,
    "map50_95": 0.04,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def norm_col(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_metric_column(columns, metric):
    cmap = {c: norm_col(c) for c in columns}

    patterns = {
        "precision": [
            "metricsprecisionb",
            "precision",
        ],
        "recall": [
            "metricsrecallb",
            "recall",
        ],
        "map50": [
            "metricsmap50b",
            "map50",
        ],
        "map50_95": [
            "metricsmap5095b",
            "map5095",
            "map50-95",
        ],
    }

    wanted = patterns[metric]

    # Prefer exact normalized endings.
    for c, nc in cmap.items():
        if metric == "map50":
            # Avoid picking mAP50-95.
            if "map5095" in nc:
                continue
        if any(p.replace("-", "") == nc or nc.endswith(p.replace("-", "")) for p in wanted):
            return c

    # Fallback contains.
    for c, nc in cmap.items():
        if metric == "map50":
            if "map5095" in nc:
                continue
            if "map50" in nc:
                return c
        elif metric == "map50_95" and "map5095" in nc:
            return c
        elif metric == "precision" and "precision" in nc:
            return c
        elif metric == "recall" and "recall" in nc:
            return c

    return None


def find_run_dir_from_checkpoint(checkpoint):
    p = Path(checkpoint)

    # Common layout: run_dir/weights/best.pt
    if p.parent.name.lower() == "weights":
        return p.parent.parent

    return p.parent


def detector_candidate_audit():
    require(R13A_V2_CKPTS)

    ckpts = pd.read_csv(R13A_V2_CKPTS)
    rows = []

    for _, rec in ckpts.iterrows():
        p = Path(str(rec["path"]))

        if not p.exists():
            continue

        if p.name.lower() != "best.pt":
            continue

        run_dir = find_run_dir_from_checkpoint(p)

        results_candidates = [
            run_dir / "results.csv",
            run_dir / "results.csv ",
        ]

        results_path = None

        for rp in results_candidates:
            if rp.exists():
                results_path = rp
                break

        args_path = run_dir / "args.yaml"

        row = {
            "checkpoint": str(p),
            "checkpoint_sha256": sha256_file(p),
            "run_dir": str(run_dir),
            "results_csv": str(results_path) if results_path else "",
            "args_yaml": str(args_path) if args_path.exists() else "",
            "has_results_csv": bool(results_path),
            "has_args_yaml": args_path.exists(),
            "project_relevant_from_v2": bool(rec.get("project_relevant", False)),
            "precision": np.nan,
            "recall": np.nan,
            "map50": np.nan,
            "map50_95": np.nan,
            "best_epoch_index": np.nan,
            "metric_distance": np.nan,
            "matches_full_detector_target": False,
        }

        if results_path:
            try:
                df = pd.read_csv(results_path)
            except Exception:
                df = None

            if df is not None and len(df):
                cols = {
                    m: find_metric_column(df.columns, m)
                    for m in ["precision", "recall", "map50", "map50_95"]
                }

                if all(cols.values()):
                    # Use row with best mAP50-95.
                    vals = pd.to_numeric(
                        df[cols["map50_95"]],
                        errors="coerce",
                    )

                    if vals.notna().any():
                        idx = vals.idxmax()
                        best = df.loc[idx]

                        metrics = {}

                        for m, c in cols.items():
                            metrics[m] = float(
                                pd.to_numeric(
                                    pd.Series([best[c]]),
                                    errors="coerce",
                                ).iloc[0]
                            )

                        row.update(metrics)
                        row["best_epoch_index"] = int(idx)

                        dist = 0.0
                        ok = True

                        for m, target in TARGET_DETECTOR.items():
                            v = metrics[m]
                            d = abs(v - target)
                            dist += d
                            if d > TOL[m]:
                                ok = False

                        row["metric_distance"] = float(dist)
                        row["matches_full_detector_target"] = bool(ok)

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            [
                "matches_full_detector_target",
                "metric_distance",
                "checkpoint",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).reset_index(drop=True)

    return out


def canon_sequence(s):
    return str(s).strip()


def birdsai_sequence_compatibility():
    require(FINAL_META_2)
    require(TESTREAL_IMAGES)

    meta = pd.read_csv(FINAL_META_2, low_memory=False)

    final_sequences = sorted(
        {
            canon_sequence(x)
            for x in meta["sequence"].dropna().tolist()
        }
    )

    frame_dirs = sorted(
        [
            p.name
            for p in TESTREAL_IMAGES.iterdir()
            if p.is_dir()
        ]
    )

    final_set = set(final_sequences)
    frame_set = set(frame_dirs)

    rows = []

    for seq in sorted(final_set | frame_set):
        p = TESTREAL_IMAGES / seq
        image_count = 0

        if p.exists() and p.is_dir():
            image_count = sum(
                1
                for x in p.iterdir()
                if x.is_file()
                and x.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            )

        rows.append(
            {
                "sequence": seq,
                "in_final_2s_metadata": seq in final_set,
                "frame_directory_exists": seq in frame_set,
                "image_count": image_count,
            }
        )

    return pd.DataFrame(rows), final_set, frame_set


def gt_schema_audit():
    require(CLEANED_TRACKS)

    df = pd.read_csv(
        CLEANED_TRACKS,
        nrows=5000,
        low_memory=False,
    )

    columns = list(df.columns)
    norm = {c: norm_col(c) for c in columns}

    def find_any(tokens):
        found = []
        for c, nc in norm.items():
            if any(t in nc for t in tokens):
                found.append(c)
        return found

    frame_cols = find_any(["frame", "frameid", "framenumber"])
    object_cols = find_any(["objectid", "trackid", "identity", "id"])
    center_x_cols = find_any(["cx", "centerx", "xcenter"])
    center_y_cols = find_any(["cy", "centery", "ycenter"])
    width_cols = find_any(["bboxw", "boxw", "width"])
    height_cols = find_any(["bboxh", "boxh", "height"])
    x1_cols = find_any(["x1", "xmin", "left"])
    y1_cols = find_any(["y1", "ymin", "top"])
    x2_cols = find_any(["x2", "xmax", "right"])
    y2_cols = find_any(["y2", "ymax", "bottom"])

    center_box_possible = (
        len(center_x_cols) > 0
        and len(center_y_cols) > 0
        and len(width_cols) > 0
        and len(height_cols) > 0
    )

    corner_box_possible = (
        len(x1_cols) > 0
        and len(y1_cols) > 0
        and len(x2_cols) > 0
        and len(y2_cols) > 0
    )

    track_gt_possible = (
        len(frame_cols) > 0
        and len(object_cols) > 0
        and (center_box_possible or corner_box_possible)
    )

    summary = pd.DataFrame(
        [
            {"field_group": "frame", "columns": "|".join(frame_cols), "available": bool(frame_cols)},
            {"field_group": "object/track id", "columns": "|".join(object_cols), "available": bool(object_cols)},
            {"field_group": "center x", "columns": "|".join(center_x_cols), "available": bool(center_x_cols)},
            {"field_group": "center y", "columns": "|".join(center_y_cols), "available": bool(center_y_cols)},
            {"field_group": "bbox width", "columns": "|".join(width_cols), "available": bool(width_cols)},
            {"field_group": "bbox height", "columns": "|".join(height_cols), "available": bool(height_cols)},
            {"field_group": "x1", "columns": "|".join(x1_cols), "available": bool(x1_cols)},
            {"field_group": "y1", "columns": "|".join(y1_cols), "available": bool(y1_cols)},
            {"field_group": "x2", "columns": "|".join(x2_cols), "available": bool(x2_cols)},
            {"field_group": "y2", "columns": "|".join(y2_cols), "available": bool(y2_cols)},
            {"field_group": "track-GT compatible schema", "columns": "", "available": track_gt_possible},
        ]
    )

    return summary, track_gt_possible, columns


def ultralytics_tracker_audit():
    rows = []

    spec = importlib.util.find_spec("ultralytics")

    if spec is None:
        return pd.DataFrame(
            [
                {
                    "ultralytics_installed": False,
                    "package_root": "",
                    "bytetrack_yaml": "",
                    "bytetrack_exists": False,
                    "botsort_yaml": "",
                    "botsort_exists": False,
                }
            ]
        )

    import ultralytics

    package_root = Path(ultralytics.__file__).resolve().parent

    candidates = [
        package_root / "cfg" / "trackers" / "bytetrack.yaml",
        package_root / "cfg" / "trackers" / "botsort.yaml",
    ]

    byte = candidates[0]
    bot = candidates[1]

    rows.append(
        {
            "ultralytics_installed": True,
            "ultralytics_version": getattr(ultralytics, "__version__", ""),
            "package_root": str(package_root),
            "bytetrack_yaml": str(byte),
            "bytetrack_exists": byte.exists(),
            "botsort_yaml": str(bot),
            "botsort_exists": bot.exists(),
        }
    )

    return pd.DataFrame(rows)


def main():
    print("\nR13-B — DETECTOR / DATA / TRACKER COMPATIBILITY AUDIT")
    print(
        "AUDIT ONLY. NO detector inference and NO tracking inference. "
        "This step locks the detector candidate by historical metrics, verifies sequence-level "
        "compatibility between frozen BIRDSAI/TestReal metadata and conservation_drones frame folders, "
        "checks whether cleaned trajectory data can serve as tracking GT, and verifies whether "
        "Ultralytics provides ByteTrack/BoT-SORT for a NEW revision experiment. "
        "No original file is modified.\n"
    )

    require(R13A_V2_MANIFEST)

    v2 = json.loads(
        R13A_V2_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    allowed = {
        "R13A_V2_END_TO_END_ASSETS_INCOMPLETE_MISSING_GENUINE_TRACKER_IMPLEMENTATION",
        "R13A_V2_CORE_END_TO_END_ASSETS_FOUND_READY_FOR_R13B_PROTOCOL_AND_GT_COMPATIBILITY_AUDIT",
    }

    if v2.get("status") not in allowed:
        raise RuntimeError(
            f"Unexpected R13-A-V2 status: {v2.get('status')}"
        )

    detector = detector_candidate_audit()
    seq_df, final_set, frame_set = birdsai_sequence_compatibility()
    gt_df, gt_possible, gt_columns = gt_schema_audit()
    tracker_df = ultralytics_tracker_audit()

    detector_matches = (
        detector[
            detector["matches_full_detector_target"] == True
        ]
        if len(detector)
        else pd.DataFrame()
    )

    unique_detector_lock = (
        len(detector_matches) == 1
    )

    detector_lock = (
        str(detector_matches.iloc[0]["checkpoint"])
        if unique_detector_lock
        else ""
    )

    sequence_match = final_set.issubset(frame_set)

    tracker_available = bool(
        len(tracker_df)
        and tracker_df.iloc[0].get("ultralytics_installed", False)
        and (
            tracker_df.iloc[0].get("bytetrack_exists", False)
            or tracker_df.iloc[0].get("botsort_exists", False)
        )
    )

    summary = pd.DataFrame(
        [
            {
                "check": "Unique historical full-dataset detector lock",
                "pass": unique_detector_lock,
                "detail": detector_lock if detector_lock else f"{len(detector_matches)} matching candidates",
            },
            {
                "check": "All final TestReal sequences have frame directories",
                "pass": sequence_match,
                "detail": f"{len(final_set & frame_set)}/{len(final_set)} final sequences present",
            },
            {
                "check": "Tracking GT-compatible schema in cleaned_track_points",
                "pass": gt_possible,
                "detail": "|".join(gt_columns),
            },
            {
                "check": "Ultralytics tracker available for NEW revision experiment",
                "pass": tracker_available,
                "detail": (
                    tracker_df.iloc[0].to_json()
                    if len(tracker_df)
                    else "not installed"
                ),
            },
        ]
    )

    if unique_detector_lock and sequence_match and gt_possible and tracker_available:
        status = (
            "R13B_COMPATIBILITY_PASS_DETECTOR_DATA_GT_AND_NEW_TRACKER_AVAILABLE_READY_FOR_R13C_FROZEN_END_TO_END_PROTOCOL_LOCK"
        )
    else:
        failed = summary.loc[
            ~summary["pass"].astype(bool),
            "check",
        ].tolist()

        status = (
            "R13B_COMPATIBILITY_INCOMPLETE_REVIEW_"
            + "_".join(
                re.sub(r"[^A-Za-z0-9]+", "_", x).upper().strip("_")
                for x in failed
            )
        )

    detector_path = OUT / "R13B_DETECTOR_CANDIDATE_METRIC_AUDIT.csv"
    seq_path = OUT / "R13B_TESTREAL_SEQUENCE_FRAME_COMPATIBILITY.csv"
    gt_path = OUT / "R13B_TRACKING_GT_SCHEMA_AUDIT.csv"
    tracker_path = OUT / "R13B_ULTRALYTICS_TRACKER_AVAILABILITY.csv"
    summary_path = OUT / "R13B_COMPATIBILITY_SUMMARY.csv"

    detector.to_csv(detector_path, index=False)
    seq_df.to_csv(seq_path, index=False)
    gt_df.to_csv(gt_path, index=False)
    tracker_df.to_csv(tracker_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "step": "R13B_DETECTOR_DATA_TRACKER_COMPATIBILITY_AUDIT",
        "created_utc": utc_now(),
        "status": status,
        "training": False,
        "detector_inference": False,
        "tracker_inference": False,
        "original_files_modified": False,
        "detector_lock": detector_lock,
        "detector_lock_unique": unique_detector_lock,
        "sequence_compatibility_pass": sequence_match,
        "tracking_gt_schema_pass": gt_possible,
        "ultralytics_tracker_available": tracker_available,
        "new_tracker_experiment_only": True,
        "historical_pipeline_end_to_end_claim_allowed": False,
        "outputs": {
            "detector_candidates": str(detector_path),
            "sequence_compatibility": str(seq_path),
            "gt_schema": str(gt_path),
            "tracker_availability": str(tracker_path),
            "summary": str(summary_path),
        },
    }

    (OUT / "R13B_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("COMPATIBILITY SUMMARY:")
    print(summary.to_string(index=False))

    print("\nDETECTOR CANDIDATE METRIC AUDIT:")
    if len(detector):
        cols = [
            "checkpoint",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "metric_distance",
            "matches_full_detector_target",
        ]
        print(detector[cols].to_string(index=False))
    else:
        print("No best.pt candidates.")

    print("\nTESTREAL SEQUENCE / FRAME COMPATIBILITY:")
    print(seq_df.to_string(index=False))

    print("\nTRACKING GT SCHEMA AUDIT:")
    print(gt_df.to_string(index=False))

    print("\nULTRALYTICS TRACKER AVAILABILITY:")
    print(tracker_df.to_string(index=False))

    print("\nSTATUS:")
    print(status)

    print("\nR13-B COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
