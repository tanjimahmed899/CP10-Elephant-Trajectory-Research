from pathlib import Path
import hashlib
import json
import re
import pandas as pd


ROOT_MAIN = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

ROOT_DL = Path(
    r"C:\Users\HP\Downloads"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step6d_uncertainty_holdout"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

# Known-used locations / terms from Steps 1-6C.
USED_PATH_TERMS = [
    "wildbox_fresh_holdout_zips",
    "wildbox_fresh_confirmatory_results",
    "wildbox_trajectory_dataset",
    "step3_fresh_wildbox_full_comparison",
    "step6_conformal_uncertainty",
    "step6b_scale_adaptive_conformal",
    "step6c_crossfit_conformal",
]

# Candidate metadata pair names.
TRACKING_NAMES = {
    "tracking_summary.json",
}

CAMERA_NAMES = {
    "cameras.json",
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


def normalized_segment_key(path):
    # Normalize likely WildBox source/video/segment identity.
    parts = [p for p in path.parts]

    # Try to retain DJI video name + segment.
    video = None
    seg = None

    for p in parts:
        if p.startswith("DJI_"):
            video = p

        if re.fullmatch(
            r"seg\d+",
            p,
            flags=re.IGNORECASE
        ):
            seg = p.lower()

    if video and seg:
        return f"{video}::{seg}"

    # Handle flattened audit names such as:
    # DJI_...__seg1__tracking_summary.json
    name = path.name

    m = re.search(
        r"(DJI_[^_]+(?:_[^_]+)*)__(seg\d+)__",
        name,
        flags=re.IGNORECASE
    )

    if m:
        return f"{m.group(1)}::{m.group(2).lower()}"

    return str(path.parent).lower()


def is_known_used(path):
    s = str(path).lower()
    return any(
        term.lower() in s
        for term in USED_PATH_TERMS
    )


def find_all_metadata_files():
    rows = []

    for root in [ROOT_MAIN, ROOT_DL]:
        if not root.exists():
            continue

        for p in root.rglob("*.json"):
            if OUT in p.parents:
                continue

            low_name = p.name.lower()

            kind = None

            if low_name in TRACKING_NAMES or "tracking_summary" in low_name:
                kind = "tracking_summary"

            elif low_name in CAMERA_NAMES or "cameras" in low_name:
                kind = "cameras"

            if kind is None:
                continue

            rows.append({
                "path": str(p),
                "kind": kind,
                "segment_key": normalized_segment_key(p),
                "known_used_by_path": is_known_used(p),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })

    return pd.DataFrame(rows)


def summarize_json_structure(path):
    row = {
        "json_path": str(path),
        "top_type": "",
        "top_keys": "",
        "list_length": None,
        "contains_bbox_term": False,
        "contains_track_term": False,
        "contains_position_term": False,
        "contains_frame_term": False,
        "parse_error": "",
    }

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            obj = json.load(f)

        row["top_type"] = type(obj).__name__

        if isinstance(obj, dict):
            row["top_keys"] = ";".join(
                sorted(
                    map(
                        str,
                        obj.keys()
                    )
                )[:100]
            )

        elif isinstance(obj, list):
            row["list_length"] = len(obj)

        text = json.dumps(
            obj
        ).lower()

        row["contains_bbox_term"] = (
            "bbox" in text
            or "box" in text
        )

        row["contains_track_term"] = (
            "track" in text
            or "object_id" in text
            or "track_id" in text
        )

        row["contains_position_term"] = (
            "position" in text
            or "translation" in text
            or "center" in text
            or "centroid" in text
        )

        row["contains_frame_term"] = (
            "frame" in text
            or "timestamp" in text
        )

    except Exception as e:
        row["parse_error"] = (
            f"{type(e).__name__}: {e}"
        )

    return row


def main():
    meta = find_all_metadata_files()

    if len(meta) == 0:
        print("No WildBox-style JSON metadata found.")
        return

    meta.to_csv(
        OUT / "wildbox_all_metadata_files.csv",
        index=False
    )

    # Pair tracking + cameras by normalized segment key.
    grouped_rows = []

    for key, g in meta.groupby(
        "segment_key"
    ):
        tracking = g[
            g["kind"] == "tracking_summary"
        ]

        cameras = g[
            g["kind"] == "cameras"
        ]

        grouped_rows.append({
            "segment_key": key,
            "tracking_files": len(tracking),
            "camera_files": len(cameras),
            "has_tracking": len(tracking) > 0,
            "has_cameras": len(cameras) > 0,
            "any_known_used_path": bool(
                g["known_used_by_path"].any()
            ),
            "tracking_path": (
                tracking.iloc[0]["path"]
                if len(tracking)
                else ""
            ),
            "camera_path": (
                cameras.iloc[0]["path"]
                if len(cameras)
                else ""
            ),
            "tracking_sha256": (
                tracking.iloc[0]["sha256"]
                if len(tracking)
                else ""
            ),
            "camera_sha256": (
                cameras.iloc[0]["sha256"]
                if len(cameras)
                else ""
            ),
        })

    segments = pd.DataFrame(
        grouped_rows
    )

    segments["paired_metadata"] = (
        segments["has_tracking"]
        &
        segments["has_cameras"]
    )

    # Conservative eligibility:
    # - must have BOTH tracking summary and camera metadata
    # - must NOT live in a known-used path
    candidates = segments[
        (
            segments["paired_metadata"] == True
        )
        &
        (
            segments["any_known_used_path"] == False
        )
    ].copy()

    # Parse only structure/field names; no forecast/error computation.
    structure_rows = []

    for _, row in candidates.iterrows():
        tpath = Path(
            row["tracking_path"]
        )

        structure = summarize_json_structure(
            tpath
        )

        structure["segment_key"] = (
            row["segment_key"]
        )

        structure_rows.append(
            structure
        )

    structure_df = pd.DataFrame(
        structure_rows
    )

    if len(structure_df):
        candidates = candidates.merge(
            structure_df,
            on="segment_key",
            how="left"
        )

    segments.to_csv(
        OUT / "wildbox_segment_inventory.csv",
        index=False
    )

    candidates.to_csv(
        OUT / "wildbox_unused_annotated_candidates.csv",
        index=False
    )

    # Create a PRE-RESULT lock only if exactly one unused paired candidate exists.
    lock = None

    if len(candidates) == 1:
        row = candidates.iloc[0]

        lock = {
            "lock_stage": "STEP_6D_UNUSED_WILDBOX_PRELOCK",
            "selection_rule": (
                "the only WildBox segment found with both tracking_summary "
                "and cameras metadata outside all known-used WildBox/result paths"
            ),
            "model_predictions_or_forecast_errors_seen_before_lock": False,
            "segment_key": row["segment_key"],
            "tracking_path": row["tracking_path"],
            "camera_path": row["camera_path"],
            "tracking_sha256": row["tracking_sha256"],
            "camera_sha256": row["camera_sha256"],
        }

        with open(
            OUT / "STEP6D_UNUSED_WILDBOX_PRELOCK.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                lock,
                f,
                indent=2
            )

    with open(
        OUT / "STEP6D_C_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-C: UNUSED ANNOTATED WILDBOX AUDIT\n"
        )
        f.write(
            "=" * 72 + "\n\n"
        )

        f.write(
            "No model predictions or forecast errors were computed.\n"
        )
        f.write(
            "Selection is based only on metadata-pair presence and path history.\n\n"
        )

        f.write(
            f"Total distinct metadata segments: {len(segments)}\n"
        )
        f.write(
            f"Paired tracking+camera segments: "
            f"{int(segments['paired_metadata'].sum())}\n"
        )
        f.write(
            f"Unused paired candidates: {len(candidates)}\n\n"
        )

        if len(candidates):
            cols = [
                "segment_key",
                "tracking_path",
                "camera_path",
                "top_type",
                "top_keys",
                "contains_bbox_term",
                "contains_track_term",
                "contains_position_term",
                "contains_frame_term",
            ]

            cols = [
                c for c in cols
                if c in candidates.columns
            ]

            f.write(
                candidates[
                    cols
                ].to_string(
                    index=False
                )
            )
            f.write("\n\n")
        else:
            f.write(
                "No unused paired annotated WildBox candidate found.\n\n"
            )

        if lock is not None:
            f.write(
                "PRE-RESULT LOCK CREATED:\n"
            )
            f.write(
                json.dumps(
                    lock,
                    indent=2
                )
            )
            f.write("\n")
        else:
            f.write(
                "No automatic pre-lock created.\n"
            )

    print("\nSTEP 6D-C UNUSED WILDBOX AUDIT COMPLETE")
    print("Outputs:", OUT)

    print("\nCOUNTS:")
    print(
        "distinct_segments =",
        len(
            segments
        )
    )
    print(
        "paired_tracking_camera =",
        int(
            segments[
                "paired_metadata"
            ].sum()
        )
    )
    print(
        "unused_paired_candidates =",
        len(
            candidates
        )
    )

    print("\nUNUSED ANNOTATED CANDIDATES:")
    if len(candidates):
        show_cols = [
            "segment_key",
            "tracking_path",
            "camera_path",
            "top_type",
            "top_keys",
            "contains_bbox_term",
            "contains_track_term",
            "contains_position_term",
            "contains_frame_term",
        ]

        show_cols = [
            c
            for c in show_cols
            if c in candidates.columns
        ]

        print(
            candidates[
                show_cols
            ].to_string(
                index=False
            )
        )
    else:
        print("NONE")

    if lock is not None:
        print("\nPRE-RESULT LOCK CREATED:")
        print(
            OUT / "STEP6D_UNUSED_WILDBOX_PRELOCK.json"
        )
    else:
        print("\nNO AUTOMATIC PRE-LOCK CREATED.")


if __name__ == "__main__":
    main()
