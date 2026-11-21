from pathlib import Path
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict

import pandas as pd


PROJECT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

ZIP_PATH = PROJECT / "Elephant.zip"

AUDIT_CSV = (
    PROJECT
    /
    "dataset_audit_results"
    /
    "nestler_video_audit.csv"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step6d_uncertainty_holdout"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATION_EXTENSIONS = {
    ".csv",
    ".json",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".mat",
    ".npy",
    ".npz",
    ".geojson",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

ANNOTATION_KEYWORDS = {
    "label",
    "labels",
    "annot",
    "annotation",
    "annotations",
    "track",
    "tracks",
    "trajectory",
    "trajectories",
    "bbox",
    "box",
    "boxes",
    "groundtruth",
    "ground_truth",
    "gt",
    "object",
    "objects",
    "mask",
    "masks",
    "position",
    "positions",
    "coordinate",
    "coordinates",
}


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def keyword_hits(name):
    low = name.lower()
    return sorted(
        kw
        for kw in ANNOTATION_KEYWORDS
        if kw in low
    )


def main():
    print("\nSTEP 6D-B: NESTLER / ELEPHANT.ZIP HOLDOUT ELIGIBILITY AUDIT")

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Missing ZIP: {ZIP_PATH}"
        )

    zip_sha = sha256_file(
        ZIP_PATH
    )

    rows = []
    extension_counts = Counter()
    total_uncompressed = 0

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zf:

        members = zf.infolist()

        for info in members:
            if info.is_dir():
                continue

            p = Path(
                info.filename
            )

            ext = p.suffix.lower()

            extension_counts[
                ext if ext else "<no_ext>"
            ] += 1

            total_uncompressed += int(
                info.file_size
            )

            hits = keyword_hits(
                info.filename
            )

            category = "other"

            if ext in VIDEO_EXTENSIONS:
                category = "video"

            elif ext in IMAGE_EXTENSIONS:
                category = "image"

            elif ext in ANNOTATION_EXTENSIONS:
                category = "possible_annotation"

            rows.append({
                "archive_path":
                    info.filename,

                "extension":
                    ext,

                "category":
                    category,

                "size_bytes":
                    int(
                        info.file_size
                    ),

                "size_mb":
                    round(
                        info.file_size
                        /
                        (1024 * 1024),
                        6
                    ),

                "annotation_keyword_hits":
                    ";".join(
                        hits
                    ),

                "has_annotation_keyword":
                    bool(
                        hits
                    ),
            })

    archive_df = pd.DataFrame(
        rows
    )

    archive_df.to_csv(
        OUT
        /
        "nestler_elephant_zip_inventory.csv",
        index=False
    )

    likely_annotation = archive_df[
        (
            archive_df[
                "category"
            ]
            ==
            "possible_annotation"
        )
        |
        (
            archive_df[
                "has_annotation_keyword"
            ]
            ==
            True
        )
    ].copy()

    likely_annotation.to_csv(
        OUT
        /
        "nestler_possible_annotation_files.csv",
        index=False
    )

    videos = archive_df[
        archive_df[
            "category"
        ]
        ==
        "video"
    ].copy()

    videos.to_csv(
        OUT
        /
        "nestler_video_files_in_zip.csv",
        index=False
    )

    images = archive_df[
        archive_df[
            "category"
        ]
        ==
        "image"
    ].copy()

    images.to_csv(
        OUT
        /
        "nestler_image_files_in_zip.csv",
        index=False
    )

    # Existing previous video audit: metadata only, no outcome use.
    if AUDIT_CSV.exists():
        nestler_audit = pd.read_csv(
            AUDIT_CSV
        )
    else:
        nestler_audit = pd.DataFrame()

    nestler_audit.to_csv(
        OUT
        /
        "nestler_existing_video_audit_copy.csv",
        index=False
    )

    # Decide HOLDOUT ELIGIBILITY conservatively.
    has_annotation_candidate = (
        len(
            likely_annotation
        )
        >
        0
    )

    has_video = (
        len(
            videos
        )
        >
        0
    )

    # Raw video alone is not enough for final trajectory uncertainty
    # confirmation because model-generated tracks cannot serve as
    # independent trajectory ground truth.
    if (
        has_video
        and
        not has_annotation_candidate
    ):
        eligibility = (
            "NOT_READY_FOR_CONFIRMATORY_TRAJECTORY_UNCERTAINTY"
        )

        reason = (
            "Archive contains video/image data but no obvious "
            "independent trajectory/annotation asset. "
            "Model-generated tracks must not be used as ground truth."
        )

    elif has_annotation_candidate:
        eligibility = (
            "POTENTIALLY_ELIGIBLE_REQUIRES_ANNOTATION_SEMANTIC_AUDIT"
        )

        reason = (
            "One or more possible annotation files exist. "
            "Their semantics must be checked before pre-locking "
            "a trajectory holdout."
        )

    else:
        eligibility = (
            "NOT_READY_NO_VIDEO_OR_TRAJECTORY_ASSET_FOUND"
        )

        reason = (
            "No suitable raw video/trajectory asset identified."
        )

    summary = {
        "step":
            "6D-B",

        "zip_path":
            str(
                ZIP_PATH
            ),

        "zip_sha256":
            zip_sha,

        "zip_size_bytes":
            int(
                ZIP_PATH.stat().st_size
            ),

        "archive_file_count":
            int(
                len(
                    archive_df
                )
            ),

        "video_files":
            int(
                len(
                    videos
                )
            ),

        "image_files":
            int(
                len(
                    images
                )
            ),

        "possible_annotation_files":
            int(
                len(
                    likely_annotation
                )
            ),

        "existing_nestler_video_audit_present":
            bool(
                AUDIT_CSV.exists()
            ),

        "holdout_eligibility":
            eligibility,

        "reason":
            reason,

        "model_predictions_or_errors_computed":
            False,

        "holdout_locked":
            False,
    }

    with open(
        OUT
        /
        "STEP6D_NESTLER_ELIGIBILITY.json",
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
        "STEP6D_B_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 6D-B: NESTLER / ELEPHANT.ZIP ELIGIBILITY AUDIT\n"
        )

        f.write(
            "=" * 76
            +
            "\n\n"
        )

        f.write(
            f"ZIP: {ZIP_PATH}\n"
        )

        f.write(
            f"SHA-256: {zip_sha}\n"
        )

        f.write(
            f"Archive files: {len(archive_df)}\n"
        )

        f.write(
            f"Video files: {len(videos)}\n"
        )

        f.write(
            f"Image files: {len(images)}\n"
        )

        f.write(
            "Possible annotation/trajectory files: "
            f"{len(likely_annotation)}\n\n"
        )

        f.write(
            f"Eligibility: {eligibility}\n"
        )

        f.write(
            f"Reason: {reason}\n\n"
        )

        f.write(
            "No model predictions or forecast errors were computed.\n"
        )

        f.write(
            "No holdout was locked at this stage.\n\n"
        )

        f.write(
            "ARCHIVE EXTENSION COUNTS\n"
        )

        f.write(
            "------------------------\n"
        )

        for ext, count in sorted(
            extension_counts.items()
        ):
            f.write(
                f"{ext}: {count}\n"
            )

        f.write(
            "\nPOSSIBLE ANNOTATION FILES\n"
        )

        f.write(
            "-------------------------\n"
        )

        if len(
            likely_annotation
        ):

            f.write(
                likely_annotation[
                    [
                        "archive_path",
                        "extension",
                        "size_mb",
                        "annotation_keyword_hits",
                    ]
                ].to_string(
                    index=False
                )
            )

        else:
            f.write(
                "NONE\n"
            )

        f.write(
            "\n\nEXISTING NESTLER VIDEO AUDIT\n"
        )

        f.write(
            "----------------------------\n"
        )

        if len(
            nestler_audit
        ):
            f.write(
                nestler_audit.to_string(
                    index=False
                )
            )
        else:
            f.write(
                "Not found.\n"
            )

    print("\nSTEP 6D-B AUDIT COMPLETE")
    print(
        "Outputs:",
        OUT
    )

    print("\nZIP SHA-256:")
    print(
        zip_sha
    )

    print("\nARCHIVE COUNTS:")
    print(
        "files =",
        len(
            archive_df
        )
    )
    print(
        "videos =",
        len(
            videos
        )
    )
    print(
        "images =",
        len(
            images
        )
    )
    print(
        "possible_annotations =",
        len(
            likely_annotation
        )
    )

    print("\nHOLDOUT ELIGIBILITY:")
    print(
        eligibility
    )

    print("\nREASON:")
    print(
        reason
    )

    print("\nPOSSIBLE ANNOTATION FILES:")
    if len(
        likely_annotation
    ):
        print(
            likely_annotation[
                [
                    "archive_path",
                    "extension",
                    "size_mb",
                    "annotation_keyword_hits",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print("NONE")

    print("\nEXISTING NESTLER VIDEO AUDIT:")
    if len(
        nestler_audit
    ):
        print(
            nestler_audit.to_string(
                index=False
            )
        )
    else:
        print("NOT FOUND")


if __name__ == "__main__":
    main()
