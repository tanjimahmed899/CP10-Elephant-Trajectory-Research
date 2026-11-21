
from pathlib import Path
import json
import os
import hashlib
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
CAPSTONE = Path(r"E:\AIUB\Capstone Project")
JOURNAL = CAPSTONE / "Journal Based Results"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R13_ROOT = REVISION / "R13_end_to_end_pipeline"
R13A = R13_ROOT / "R13A_end_to_end_asset_provenance_audit" / "R13A_MANIFEST.json"

OUT = R13_ROOT / "R13A_V2_project_aware_asset_audit"
OUT.mkdir(parents=True, exist_ok=True)

KNOWN_CHECKPOINT_CANDIDATES = [
    DOWNLOADS
    / "elephant_200_yolov8_140_40_20"
    / "runs"
    / "detect"
    / "Elephant_200"
    / "baseline_50"
    / "weights"
    / "best.pt",
]

SEARCH_ROOTS = [
    CAPSTONE,
    DOWNLOADS,
]

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_EXTS = {".txt", ".json", ".csv", ".xml"}

SKIP_DIRS = {
    "site-packages",
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "anaconda",
    "miniconda",
    "paper_strengthening_revision",
}

TRACKER_IMPLEMENTATION_TOKENS = [
    "model.track(",
    ".track(",
    "bytetrack.yaml",
    "botsort.yaml",
    "byte_tracker",
    "bytetrack",
    "botsort",
    "supervision.bytetrack",
    "deep_sort",
    "deepsort",
]

TRACK_METRIC_TOKENS = [
    "idf1",
    "hota",
    "id switch",
    "id_switch",
    "idswitch",
    "fragmentation",
    "mota",
    "motmetrics",
    "trackeval",
]

PROJECT_HINTS = [
    "elephant",
    "birdsai",
    "wildbox",
    "bambi",
    "trajectory",
    "uav",
    "drone",
]

MAX_TEXT_BYTES = 10_000_000


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


def skipped(path):
    low_parts = {p.lower() for p in Path(path).parts}
    return any(x.lower() in low_parts for x in SKIP_DIRS)


def iter_files(root):
    if not root.exists():
        return

    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)

        dirs[:] = [
            d for d in dirs
            if not skipped(current / d)
        ]

        if skipped(current):
            continue

        for name in files:
            yield current / name


def project_relevance(path):
    low = str(path).lower()
    return any(h in low for h in PROJECT_HINTS)


def read_small_text(path):
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def scan_checkpoints():
    rows = []
    seen = set()

    for p in KNOWN_CHECKPOINT_CANDIDATES:
        if p.exists():
            rows.append(
                {
                    "path": str(p),
                    "exists": True,
                    "source": "known_previous_run_path",
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                    "project_relevant": True,
                }
            )
            seen.add(str(p).lower())

    roots = [
        DOWNLOADS,
        JOURNAL,
        CAPSTONE,
    ]

    for root in roots:
        if not root.exists():
            continue

        for current_root, dirs, files in os.walk(root):
            current = Path(current_root)

            dirs[:] = [
                d for d in dirs
                if not skipped(current / d)
            ]

            low_current = str(current).lower()

            likely_tree = (
                "runs" in low_current
                or "weights" in low_current
                or "yolo" in low_current
                or "elephant" in low_current
                or root == CAPSTONE
            )

            if not likely_tree:
                continue

            for name in files:
                if name.lower() not in {"best.pt", "last.pt"}:
                    continue

                p = current / name
                key = str(p).lower()

                if key in seen:
                    continue

                seen.add(key)

                rows.append(
                    {
                        "path": str(p),
                        "exists": True,
                        "source": "targeted_filesystem_search",
                        "size_bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                        "project_relevant": project_relevance(p),
                    }
                )

    return pd.DataFrame(rows)


def scan_project_media_and_sequences():
    media_rows = []
    sequence_rows = []

    candidate_dirs = {}

    for root in [CAPSTONE, JOURNAL]:
        if not root.exists():
            continue

        for current_root, dirs, files in os.walk(root):
            current = Path(current_root)

            dirs[:] = [
                d for d in dirs
                if not skipped(current / d)
            ]

            if skipped(current):
                continue

            relevant = project_relevance(current)

            image_count = 0
            label_count = 0

            for name in files:
                p = current / name
                suffix = p.suffix.lower()

                if suffix in VIDEO_EXTS and relevant:
                    media_rows.append(
                        {
                            "path": str(p),
                            "suffix": suffix,
                            "size_bytes": p.stat().st_size,
                            "project_relevant": True,
                        }
                    )

                if suffix in IMAGE_EXTS:
                    image_count += 1

                if suffix in LABEL_EXTS:
                    label_count += 1

            if relevant and image_count >= 20:
                candidate_dirs[str(current)] = {
                    "directory": str(current),
                    "image_count": image_count,
                    "label_like_file_count": label_count,
                    "birdsai_hint": "birdsai" in str(current).lower(),
                    "elephant_hint": "elephant" in str(current).lower(),
                    "wildbox_hint": "wildbox" in str(current).lower(),
                    "bambi_hint": "bambi" in str(current).lower(),
                }

    if candidate_dirs:
        sequence_rows = list(candidate_dirs.values())

    return pd.DataFrame(media_rows), pd.DataFrame(sequence_rows)


def scan_tracker_sources():
    rows = []

    roots = [
        JOURNAL,
        CAPSTONE,
    ]

    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for p in iter_files(root):
            if p.suffix.lower() != ".py":
                continue

            key = str(p).lower()
            if key in seen:
                continue
            seen.add(key)

            text = read_small_text(p)
            low = text.lower()

            impl_hits = [
                t for t in TRACKER_IMPLEMENTATION_TOKENS
                if t.lower() in low
            ]

            metric_hits = [
                t for t in TRACK_METRIC_TOKENS
                if t.lower() in low
            ]

            if not impl_hits and not metric_hits:
                continue

            rows.append(
                {
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "implementation_hits": "|".join(impl_hits),
                    "metric_hits": "|".join(metric_hits),
                    "actual_tracker_implementation_signal": bool(impl_hits),
                    "tracking_metric_signal": bool(metric_hits),
                }
            )

    return pd.DataFrame(rows)


def scan_birdsai_annotation_assets():
    rows = []

    if not CAPSTONE.exists():
        return pd.DataFrame()

    for current_root, dirs, files in os.walk(CAPSTONE):
        current = Path(current_root)

        dirs[:] = [
            d for d in dirs
            if not skipped(current / d)
        ]

        low = str(current).lower()

        if "birdsai" not in low:
            continue

        for name in files:
            p = current / name
            suffix = p.suffix.lower()

            if suffix not in {".csv", ".txt", ".json", ".xml"}:
                continue

            low_name = name.lower()

            likely = any(
                t in low_name
                for t in [
                    "track",
                    "annot",
                    "label",
                    "gt",
                    "ground",
                    "metadata",
                    "trajectory",
                ]
            )

            if not likely:
                continue

            rows.append(
                {
                    "path": str(p),
                    "suffix": suffix,
                    "size_bytes": p.stat().st_size,
                }
            )

    return pd.DataFrame(rows)


def main():
    print("\nR13-A-V2 — PROJECT-AWARE END-TO-END ASSET AUDIT")
    print(
        "AUDIT ONLY. NO training and NO inference. "
        "This revision removes false positives from revision scripts and unrelated personal videos, "
        "checks a known prior YOLO checkpoint path, searches the full Capstone tree, detects image-sequence "
        "datasets such as BIRDSAI even when no MP4 exists, and requires genuine tracker implementation code. "
        "No original file is modified.\n"
    )

    require(R13A)

    old = json.loads(R13A.read_text(encoding="utf-8"))

    checkpoints = scan_checkpoints()
    videos, sequences = scan_project_media_and_sequences()
    trackers = scan_tracker_sources()
    annotations = scan_birdsai_annotation_assets()

    checkpoint_ok = len(checkpoints) > 0
    project_video_or_sequence_ok = len(videos) > 0 or len(sequences) > 0

    tracker_impl_ok = (
        len(trackers) > 0
        and bool(trackers["actual_tracker_implementation_signal"].any())
    )

    birdsai_sequence_ok = (
        len(sequences) > 0
        and "birdsai_hint" in sequences.columns
        and bool(sequences["birdsai_hint"].any())
    )

    birdsai_annotation_ok = len(annotations) > 0

    summary = pd.DataFrame(
        [
            {"component": "Detector checkpoint", "found": checkpoint_ok, "count": len(checkpoints)},
            {"component": "Project-relevant videos", "found": len(videos) > 0, "count": len(videos)},
            {"component": "Project image-sequence directories", "found": len(sequences) > 0, "count": len(sequences)},
            {"component": "BIRDSAI image-sequence directory", "found": birdsai_sequence_ok, "count": int(sequences["birdsai_hint"].sum()) if len(sequences) and "birdsai_hint" in sequences else 0},
            {"component": "Genuine tracker implementation source", "found": tracker_impl_ok, "count": int(trackers["actual_tracker_implementation_signal"].sum()) if len(trackers) else 0},
            {"component": "BIRDSAI annotation/GT candidates", "found": birdsai_annotation_ok, "count": len(annotations)},
        ]
    )

    if checkpoint_ok and project_video_or_sequence_ok and tracker_impl_ok:
        status = (
            "R13A_V2_CORE_END_TO_END_ASSETS_FOUND_READY_FOR_R13B_PROTOCOL_AND_GT_COMPATIBILITY_AUDIT"
        )
    else:
        missing = []

        if not checkpoint_ok:
            missing.append("DETECTOR_CHECKPOINT")

        if not project_video_or_sequence_ok:
            missing.append("PROJECT_MEDIA_OR_IMAGE_SEQUENCE")

        if not tracker_impl_ok:
            missing.append("GENUINE_TRACKER_IMPLEMENTATION")

        status = (
            "R13A_V2_END_TO_END_ASSETS_INCOMPLETE_MISSING_"
            + "_".join(missing)
        )

    checkpoints_path = OUT / "R13A_V2_DETECTOR_CHECKPOINTS.csv"
    videos_path = OUT / "R13A_V2_PROJECT_VIDEOS.csv"
    sequences_path = OUT / "R13A_V2_PROJECT_IMAGE_SEQUENCE_DIRECTORIES.csv"
    trackers_path = OUT / "R13A_V2_GENUINE_TRACKER_SOURCE_AUDIT.csv"
    annotations_path = OUT / "R13A_V2_BIRDSAI_ANNOTATION_GT_CANDIDATES.csv"
    summary_path = OUT / "R13A_V2_SUMMARY.csv"

    checkpoints.to_csv(checkpoints_path, index=False)
    videos.to_csv(videos_path, index=False)
    sequences.to_csv(sequences_path, index=False)
    trackers.to_csv(trackers_path, index=False)
    annotations.to_csv(annotations_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "step": "R13A_V2_PROJECT_AWARE_END_TO_END_ASSET_AUDIT",
        "created_utc": utc_now(),
        "status": status,
        "training": False,
        "inference": False,
        "original_files_modified": False,
        "r13a_v1_status": old.get("status"),
        "detector_checkpoint_found": checkpoint_ok,
        "project_media_or_sequence_found": project_video_or_sequence_ok,
        "genuine_tracker_implementation_found": tracker_impl_ok,
        "birdsai_sequence_found": birdsai_sequence_ok,
        "birdsai_annotation_gt_candidate_found": birdsai_annotation_ok,
        "outputs": {
            "summary": str(summary_path),
            "checkpoints": str(checkpoints_path),
            "project_videos": str(videos_path),
            "image_sequences": str(sequences_path),
            "tracker_sources": str(trackers_path),
            "birdsai_gt_candidates": str(annotations_path),
        },
    }

    (OUT / "R13A_V2_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("PROJECT-AWARE ASSET SUMMARY:")
    print(summary.to_string(index=False))

    print("\nDETECTOR CHECKPOINTS:")
    if len(checkpoints):
        print(checkpoints.to_string(index=False))
    else:
        print("None found.")

    print("\nPROJECT IMAGE-SEQUENCE DIRECTORIES:")
    if len(sequences):
        print(sequences.head(60).to_string(index=False))
    else:
        print("None found.")

    print("\nGENUINE TRACKER SOURCE AUDIT:")
    if len(trackers):
        print(trackers.head(80).to_string(index=False))
    else:
        print("No genuine tracker implementation source found.")

    print("\nBIRDSAI ANNOTATION / GT CANDIDATES:")
    if len(annotations):
        print(annotations.head(80).to_string(index=False))
    else:
        print("None found.")

    print("\nSTATUS:")
    print(status)

    print("\nR13-A-V2 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
