
from pathlib import Path
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
JOURNAL_ROOT = Path(r"E:\AIUB\Capstone Project\Journal Based Results")
MAIN1 = JOURNAL_ROOT / "Main Result"
MAIN2 = JOURNAL_ROOT / "Main Result 2.0"

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R12F = REVISION_ROOT / "R12_track_accounting" / "R12F_final_track_accounting_lock" / "R12F_MANIFEST.json"

R13_ROOT = REVISION_ROOT / "R13_end_to_end_pipeline"
OUT = R13_ROOT / "R13A_end_to_end_asset_provenance_audit"
OUT.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    JOURNAL_ROOT,
    DOWNLOADS,
]

SKIP_DIR_TOKENS = [
    "site-packages",
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "anaconda",
    "miniconda",
    "paper_strengthening_revision",
    "cache",
]

MAX_TEXT_BYTES = 20_000_000

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MODEL_EXTS = {".pt", ".pth", ".onnx", ".engine"}
CONFIG_NAMES = {"data.yaml", "dataset.yaml", "tracker.yaml", "bytetrack.yaml", "botsort.yaml"}

DETECTOR_PATTERNS = [
    "yolo",
    "ultralytics",
    "detect",
    "best.pt",
    "last.pt",
]

TRACKER_PATTERNS = [
    "bytetrack",
    "byte_track",
    "botsort",
    "bo-sort",
    "deep sort",
    "deepsort",
    "multi-object",
    "multi object",
    "tracker",
    "track_id",
    "object_id",
    "idf1",
    "hota",
    "id_switch",
    "idswitch",
    "fragmentation",
]

TRAJECTORY_PATTERNS = [
    "trajectory",
    "track_key",
    "segment_id",
    "cleaned_track_points",
    "corrected_windows",
]

RESULT_PATTERNS = [
    "results.csv",
    "metrics",
    "map50",
    "precision",
    "recall",
]


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


def should_skip_dir(path):
    low = str(path).lower()
    return any(token.lower() in low for token in SKIP_DIR_TOKENS)


def iter_files():
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for current_root, dirs, files in os.walk(root):
            current = Path(current_root)

            dirs[:] = [
                d for d in dirs
                if not should_skip_dir(current / d)
            ]

            if should_skip_dir(current):
                continue

            for name in files:
                p = current / name

                key = str(p).lower()
                if key in seen:
                    continue
                seen.add(key)

                yield p


def safe_text(path):
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return ""
    except Exception:
        return ""

    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def classify_file(path):
    name = path.name.lower()
    suffix = path.suffix.lower()
    low_path = str(path).lower()

    categories = []
    evidence = []

    if suffix in VIDEO_EXTS:
        categories.append("RAW_OR_PROCESSED_VIDEO")
        evidence.append("video_extension")

    if suffix in MODEL_EXTS:
        categories.append("MODEL_CHECKPOINT_OR_EXPORT")
        evidence.append("model_extension")

        if name in {"best.pt", "last.pt"}:
            categories.append("YOLO_LIKELY_CHECKPOINT")
            evidence.append("best_or_last_pt")

    if name in CONFIG_NAMES:
        categories.append("CONFIG")
        evidence.append("known_config_filename")

    if suffix in {".csv", ".txt", ".json", ".yaml", ".yml", ".py", ".md"}:
        text = safe_text(path)
        low = text.lower()

        detector_hits = [p for p in DETECTOR_PATTERNS if p in low or p in low_path]
        tracker_hits = [p for p in TRACKER_PATTERNS if p in low or p in low_path]
        trajectory_hits = [p for p in TRAJECTORY_PATTERNS if p in low or p in low_path]
        result_hits = [p for p in RESULT_PATTERNS if p in low or p in low_path]

        if detector_hits:
            categories.append("DETECTOR_RELATED")
            evidence.append("detector:" + "|".join(detector_hits))

        if tracker_hits:
            categories.append("TRACKER_RELATED")
            evidence.append("tracker:" + "|".join(tracker_hits))

        if trajectory_hits:
            categories.append("TRAJECTORY_RELATED")
            evidence.append("trajectory:" + "|".join(trajectory_hits))

        if result_hits:
            categories.append("EVALUATION_OR_TRAINING_RESULT")
            evidence.append("result:" + "|".join(result_hits))

        if suffix == ".py":
            if "yolo" in low or "ultralytics" in low:
                categories.append("DETECTOR_SOURCE")
            if any(x in low for x in ["bytetrack", "botsort", "deepsort", "tracker"]):
                categories.append("TRACKER_SOURCE")

    if name in {"data.yaml", "dataset.yaml"}:
        categories.append("DETECTION_DATASET_CONFIG")

    if "runs" in low_path and "detect" in low_path:
        categories.append("YOLO_RUN_ARTIFACT")

    if "weights" in low_path and suffix == ".pt":
        categories.append("WEIGHT_FILE")

    if not categories:
        return None

    try:
        size = path.stat().st_size
    except Exception:
        size = None

    return {
        "path": str(path),
        "name": path.name,
        "suffix": suffix,
        "size_bytes": size,
        "categories": "|".join(sorted(set(categories))),
        "evidence": "|".join(evidence),
    }


def main():
    print("\nR13-A — END-TO-END ASSET / PROVENANCE AUDIT")
    print(
        "AUDIT ONLY. NO training and NO inference. "
        "This step verifies whether the current project actually contains the detector, "
        "raw video, tracker implementation/results, and trajectory assets required to build "
        "a real detector -> tracker -> forecast -> warning evaluation. No original file is modified.\n"
    )

    require(R12F)

    r12f = json.loads(
        R12F.read_text(
            encoding="utf-8"
        )
    )

    expected_r12 = (
        "R12_TRACK_ACCOUNTING_FULLY_RESOLVED_AND_LOCKED_COMPLETE_READY_FOR_R13_END_TO_END_PIPELINE"
    )

    if r12f.get("status") != expected_r12:
        raise RuntimeError(
            f"R12-F is not locked complete: {r12f.get('status')}"
        )

    rows = []

    for p in iter_files():
        rec = classify_file(p)
        if rec is not None:
            rows.append(rec)

    inventory = pd.DataFrame(rows)

    if len(inventory):
        inventory = inventory.sort_values(
            ["categories", "path"]
        ).reset_index(drop=True)

    def subset_contains(token):
        if not len(inventory):
            return pd.DataFrame()
        return inventory[
            inventory["categories"]
            .astype(str)
            .str.contains(token, regex=False)
        ].copy()

    models = subset_contains("YOLO_LIKELY_CHECKPOINT")
    videos = subset_contains("RAW_OR_PROCESSED_VIDEO")
    detector_sources = subset_contains("DETECTOR_SOURCE")
    tracker_sources = subset_contains("TRACKER_SOURCE")
    tracker_related = subset_contains("TRACKER_RELATED")
    trajectory_related = subset_contains("TRAJECTORY_RELATED")
    configs = subset_contains("DETECTION_DATASET_CONFIG")
    run_artifacts = subset_contains("YOLO_RUN_ARTIFACT")

    summary = pd.DataFrame(
        [
            {"asset_class": "YOLO likely checkpoints", "count": len(models)},
            {"asset_class": "Video files", "count": len(videos)},
            {"asset_class": "Detector source files", "count": len(detector_sources)},
            {"asset_class": "Tracker source files", "count": len(tracker_sources)},
            {"asset_class": "Tracker-related files", "count": len(tracker_related)},
            {"asset_class": "Trajectory-related files", "count": len(trajectory_related)},
            {"asset_class": "Detection dataset configs", "count": len(configs)},
            {"asset_class": "YOLO run artifacts", "count": len(run_artifacts)},
        ]
    )

    checkpoint_found = len(models) > 0
    video_found = len(videos) > 0
    tracker_impl_found = len(tracker_sources) > 0
    trajectory_found = len(trajectory_related) > 0

    if checkpoint_found and video_found and tracker_impl_found and trajectory_found:
        status = (
            "R13A_END_TO_END_ASSET_AUDIT_PASS_REQUIRED_COMPONENTS_FOUND_READY_FOR_R13B_PIPELINE_LOCK"
        )
    else:
        missing = []

        if not checkpoint_found:
            missing.append("detector_checkpoint")

        if not video_found:
            missing.append("raw_or_processed_video")

        if not tracker_impl_found:
            missing.append("tracker_implementation")

        if not trajectory_found:
            missing.append("trajectory_assets")

        status = (
            "R13A_END_TO_END_ASSET_AUDIT_INCOMPLETE_MISSING_"
            + "_".join(missing).upper()
        )

    inventory_path = OUT / "R13A_ASSET_INVENTORY.csv"
    summary_path = OUT / "R13A_ASSET_SUMMARY.csv"
    models_path = OUT / "R13A_DETECTOR_CHECKPOINT_CANDIDATES.csv"
    videos_path = OUT / "R13A_VIDEO_CANDIDATES.csv"
    trackers_path = OUT / "R13A_TRACKER_SOURCE_CANDIDATES.csv"
    trajectories_path = OUT / "R13A_TRAJECTORY_ASSETS.csv"

    inventory.to_csv(inventory_path, index=False)
    summary.to_csv(summary_path, index=False)
    models.to_csv(models_path, index=False)
    videos.to_csv(videos_path, index=False)
    tracker_sources.to_csv(trackers_path, index=False)
    trajectory_related.to_csv(trajectories_path, index=False)

    manifest = {
        "step": "R13A_END_TO_END_ASSET_PROVENANCE_AUDIT",
        "created_utc": utc_now(),
        "status": status,
        "training": False,
        "inference": False,
        "original_files_modified": False,
        "checkpoint_found": checkpoint_found,
        "video_found": video_found,
        "tracker_implementation_found": tracker_impl_found,
        "trajectory_assets_found": trajectory_found,
        "outputs": {
            "inventory": str(inventory_path),
            "summary": str(summary_path),
            "detector_checkpoints": str(models_path),
            "videos": str(videos_path),
            "tracker_sources": str(trackers_path),
            "trajectory_assets": str(trajectories_path),
        },
    }

    (OUT / "R13A_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("ASSET SUMMARY:")
    print(summary.to_string(index=False))

    print("\nTOP DETECTOR CHECKPOINT CANDIDATES:")
    if len(models):
        print(
            models[["path", "size_bytes", "categories"]]
            .head(30)
            .to_string(index=False)
        )
    else:
        print("None found.")

    print("\nTOP VIDEO CANDIDATES:")
    if len(videos):
        print(
            videos[["path", "size_bytes"]]
            .head(30)
            .to_string(index=False)
        )
    else:
        print("None found.")

    print("\nTOP TRACKER SOURCE CANDIDATES:")
    if len(tracker_sources):
        print(
            tracker_sources[["path", "categories", "evidence"]]
            .head(40)
            .to_string(index=False)
        )
    else:
        print("None found.")

    print("\nSTATUS:")
    print(status)

    print("\nR13-A COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
