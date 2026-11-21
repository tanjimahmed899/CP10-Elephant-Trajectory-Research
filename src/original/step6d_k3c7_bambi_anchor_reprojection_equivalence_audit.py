from pathlib import Path
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import sys

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "bambi_anchor_reprojection_equivalence_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

KEYWORDS = [
    "project",
    "projection",
    "intrinsic",
    "extrinsic",
    "fovy",
    "fov",
    "camera",
    "world",
    "pixel",
    "ray",
    "georef",
    "georeference",
    "pose",
    "rotation",
    "quaternion",
    "resolution",
]


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
    if not path.exists():
        raise FileNotFoundError(path)


def discover_one(filename):
    matches = sorted(
        p
        for p in DOWNLOADS.rglob(filename)
        if p.is_file()
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {DOWNLOADS}"
        )

    # Prefer the BAMBI download folder used in K3C6.
    preferred = [
        p
        for p in matches
        if "bambi_downloads" in str(p).lower()
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(
        f"Multiple candidates for {filename}:\n"
        + "\n".join(str(p) for p in matches)
    )


def json_summary(path):
    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    out = {
        "file":
            str(path),

        "sha256":
            sha256_file(path),

        "type":
            type(raw).__name__,
    }

    if isinstance(raw, dict):
        out["top_level_keys"] = list(
            raw.keys()
        )

        for k, v in raw.items():
            if isinstance(
                v,
                (
                    str,
                    int,
                    float,
                    bool,
                )
            ) or v is None:
                out.setdefault(
                    "top_level_scalars",
                    {}
                )[k] = v

        # Expand likely pose/camera lists.
        for candidate in [
            "images",
            "poses",
            "frames",
            "camera",
            "origin",
        ]:
            if candidate not in raw:
                continue

            v = raw[candidate]

            if isinstance(v, list):
                out[f"{candidate}_length"] = len(v)

                if v:
                    out[
                        f"{candidate}_first_item_type"
                    ] = type(
                        v[0]
                    ).__name__

                    if isinstance(
                        v[0],
                        dict
                    ):
                        out[
                            f"{candidate}_first_item_keys"
                        ] = list(
                            v[0].keys()
                        )

                        out[
                            f"{candidate}_first_item"
                        ] = v[0]

            elif isinstance(v, dict):
                out[
                    f"{candidate}_keys"
                ] = list(
                    v.keys()
                )

                scalar_part = {
                    kk: vv
                    for kk, vv in v.items()
                    if isinstance(
                        vv,
                        (
                            str,
                            int,
                            float,
                            bool,
                        )
                    ) or vv is None
                }

                out[
                    f"{candidate}_scalar_values"
                ] = scalar_part

    elif isinstance(raw, list):
        out["length"] = len(raw)

        if raw:
            out["first_item_type"] = type(
                raw[0]
            ).__name__

            if isinstance(raw[0], dict):
                out["first_item_keys"] = list(
                    raw[0].keys()
                )
                out["first_item"] = raw[0]

    return out


def collect_json_field_inventory(path):
    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    rows = []

    def visit(obj, prefix, depth=0):
        if depth > 5:
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                key = (
                    f"{prefix}.{k}"
                    if prefix
                    else str(k)
                )

                rows.append({
                    "path":
                        key,

                    "type":
                        type(v).__name__,

                    "preview":
                        repr(v)[:300],
                })

                visit(
                    v,
                    key,
                    depth + 1
                )

        elif isinstance(obj, list):
            if not obj:
                return

            # Inspect only first few representative items.
            for idx, v in enumerate(
                obj[:3]
            ):
                key = (
                    f"{prefix}[{idx}]"
                )

                rows.append({
                    "path":
                        key,

                    "type":
                        type(v).__name__,

                    "preview":
                        repr(v)[:300],
                })

                visit(
                    v,
                    key,
                    depth + 1
                )

    visit(
        raw,
        ""
    )

    return pd.DataFrame(
        rows
    )


def python_source_hits(root):
    rows = []

    if not root.exists():
        return pd.DataFrame(rows)

    for path in root.rglob("*.py"):
        try:
            if path.stat().st_size > 3_000_000:
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue

        low = text.lower()

        if not any(
            k in low
            for k in KEYWORDS
        ):
            continue

        lines = text.splitlines()

        for i, line in enumerate(
            lines,
            start=1
        ):
            ll = line.lower()

            hits = [
                k
                for k in KEYWORDS
                if k in ll
            ]

            if not hits:
                continue

            strong = any(
                token in ll
                for token in [
                    "def ",
                    "fovy",
                    "intrinsic",
                    "extrinsic",
                    "project",
                    "georef",
                    "ray",
                    "camera",
                    "rotation",
                    "quaternion",
                ]
            )

            if not strong:
                continue

            lo = max(
                1,
                i - 4
            )

            hi = min(
                len(lines),
                i + 4
            )

            context = "\n".join(
                f"{j:05d}: {lines[j-1]}"
                for j in range(
                    lo,
                    hi + 1
                )
            )

            rows.append({
                "file":
                    str(path),

                "line":
                    int(i),

                "hits":
                    ";".join(
                        sorted(
                            set(
                                hits
                            )
                        )
                    ),

                "text":
                    line.strip(),

                "context":
                    context,
            })

    return pd.DataFrame(
        rows
    )


def installed_bambi_source():
    rows = []

    modules = [
        "bambi.georeference_mot17_tracks",
        "bambi",
    ]

    for name in modules:
        try:
            spec = importlib.util.find_spec(
                name
            )
        except Exception as e:
            rows.append({
                "module":
                    name,

                "status":
                    f"find_spec_error:{type(e).__name__}:{e}",

                "origin":
                    None,
            })

            continue

        if spec is None:
            rows.append({
                "module":
                    name,

                "status":
                    "not_found",

                "origin":
                    None,
            })

            continue

        rows.append({
            "module":
                name,

            "status":
                "found",

            "origin":
                str(
                    spec.origin
                )
                if spec.origin
                else None,
        })

    return pd.DataFrame(
        rows
    )


def csv_inventory(root):
    rows = []

    if not root.exists():
        return pd.DataFrame(rows)

    for p in root.rglob("*.csv"):
        try:
            df = pd.read_csv(
                p,
                nrows=50
            )
        except Exception:
            continue

        cols = list(
            df.columns
        )

        interesting = [
            c
            for c in cols
            if any(
                token in c.lower()
                for token in [
                    "frame",
                    "track",
                    "x",
                    "y",
                    "z",
                    "lat",
                    "lon",
                    "utm",
                    "world",
                    "pose",
                    "time",
                ]
            )
        ]

        if not interesting:
            continue

        rows.append({
            "file":
                str(p),

            "columns":
                ";".join(
                    cols
                ),

            "interesting_columns":
                ";".join(
                    interesting
                ),

            "sample_rows":
                int(
                    len(
                        df
                    )
                ),
        })

    return pd.DataFrame(
        rows
    )


def frame_overlap_audit(
    georef_csv_inventory,
    poses_path
):
    raw = json.loads(
        poses_path.read_text(
            encoding="utf-8"
        )
    )

    pose_frames = set()

    def maybe_add_frame(d):
        if not isinstance(
            d,
            dict
        ):
            return

        for key in [
            "frame_id",
            "frame",
            "frame_index",
            "image_id",
            "id",
        ]:
            if key in d:
                try:
                    pose_frames.add(
                        int(
                            d[key]
                        )
                    )
                except Exception:
                    pass

    if isinstance(raw, dict):
        for k in [
            "images",
            "poses",
            "frames",
        ]:
            v = raw.get(
                k
            )

            if isinstance(v, list):
                for item in v:
                    maybe_add_frame(
                        item
                    )

    elif isinstance(raw, list):
        for item in raw:
            maybe_add_frame(
                item
            )

    rows = []

    for _, rec in georef_csv_inventory.iterrows():
        p = Path(
            rec[
                "file"
            ]
        )

        try:
            df = pd.read_csv(
                p
            )
        except Exception:
            continue

        frame_col = None

        for c in df.columns:
            if c.lower() in {
                "frame",
                "frame_id",
                "frame_index",
            }:
                frame_col = c
                break

        if frame_col is None:
            continue

        frames = pd.to_numeric(
            df[
                frame_col
            ],
            errors="coerce"
        ).dropna()

        if len(
            frames
        ) == 0:
            continue

        frame_set = set(
            frames.astype(
                int
            ).tolist()
        )

        overlap = (
            frame_set
            &
            pose_frames
        )

        rows.append({
            "file":
                str(p),

            "frame_column":
                frame_col,

            "unique_csv_frames":
                int(
                    len(
                        frame_set
                    )
                ),

            "unique_pose_frames":
                int(
                    len(
                        pose_frames
                    )
                ),

            "overlap_frames":
                int(
                    len(
                        overlap
                    )
                ),

            "csv_overlap_fraction":
                (
                    float(
                        len(
                            overlap
                        )
                        /
                        len(
                            frame_set
                        )
                    )
                    if frame_set
                    else None
                ),
        })

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nSTEP 6D-K3C7 — BAMBI ANCHOR-CAMERA REPROJECTION EQUIVALENCE AUDIT"
    )

    print(
        "Read-only audit. NO BAMBI coordinate conversion or Step7 alarms are generated.\n"
    )

    gt = discover_one(
        "6_gt.txt"
    )

    metadata = discover_one(
        "6_metadata.json"
    )

    poses = discover_one(
        "6_matched_poses.json"
    )

    correction = discover_one(
        "6_correction.json"
    )

    print(
        "RESOLVED BAMBI FILES:"
    )

    for label, p in [
        ("GT", gt),
        ("Metadata", metadata),
        ("Matched poses", poses),
        ("Correction", correction),
    ]:
        print(
            f"{label}: {p}"
        )

    meta_summary = json_summary(
        metadata
    )

    poses_summary = json_summary(
        poses
    )

    corr_summary = json_summary(
        correction
    )

    with open(
        OUT
        / "STEP6D_K3C7_BAMBI_JSON_SUMMARIES.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "metadata":
                    meta_summary,

                "poses":
                    poses_summary,

                "correction":
                    corr_summary,
            },
            f,
            indent=2
        )

    pose_fields = collect_json_field_inventory(
        poses
    )

    correction_fields = collect_json_field_inventory(
        correction
    )

    pose_fields.to_csv(
        OUT
        / "TABLE_step6dk3c7_pose_field_inventory.csv",
        index=False
    )

    correction_fields.to_csv(
        OUT
        / "TABLE_step6dk3c7_correction_field_inventory.csv",
        index=False
    )

    repo_roots = []

    # Search local downloaded BAMBI repo(s).
    for p in DOWNLOADS.rglob(
        "georeference_mot17_tracks.py"
    ):
        repo_roots.append(
            p.parent.parent
        )

    for p in DOWNLOADS.rglob(
        "Dataset-main"
    ):
        if p.is_dir():
            repo_roots.append(
                p
            )

    repo_roots = list(
        dict.fromkeys(
            repo_roots
        )
    )

    source_frames = []

    for root in repo_roots:
        source_frames.append(
            python_source_hits(
                root
            )
        )

    source_hits = (
        pd.concat(
            source_frames,
            ignore_index=True
        )
        if source_frames
        else pd.DataFrame()
    )

    source_hits.to_csv(
        OUT
        / "TABLE_step6dk3c7_bambi_projection_source_hits.csv",
        index=False
    )

    module_inventory = installed_bambi_source()

    module_inventory.to_csv(
        OUT
        / "TABLE_step6dk3c7_installed_bambi_module_inventory.csv",
        index=False
    )

    georef_csvs = csv_inventory(
        BASE
    )

    georef_csvs.to_csv(
        OUT
        / "TABLE_step6dk3c7_georef_csv_inventory.csv",
        index=False
    )

    overlap = frame_overlap_audit(
        georef_csvs,
        poses
    )

    overlap.to_csv(
        OUT
        / "TABLE_step6dk3c7_worldpoint_pose_frame_overlap.csv",
        index=False
    )

    print(
        "\nMATCHED-POSE JSON SUMMARY:"
    )

    print(
        json.dumps(
            poses_summary,
            indent=2
        )[:16000]
    )

    print(
        "\nPOSE/CAMERA FIELD INVENTORY:"
    )

    print(
        pose_fields.head(
            150
        ).to_string(
            index=False
        )
        if len(
            pose_fields
        )
        else "NONE"
    )

    print(
        "\nCORRECTION FIELD INVENTORY:"
    )

    print(
        correction_fields.head(
            100
        ).to_string(
            index=False
        )
        if len(
            correction_fields
        )
        else "NONE"
    )

    print(
        "\nBAMBI PROJECTION/GEOREFERENCE SOURCE HITS:"
    )

    print(
        source_hits.head(
            160
        ).to_string(
            index=False
        )
        if len(
            source_hits
        )
        else "NONE"
    )

    print(
        "\nINSTALLED BAMBI MODULE INVENTORY:"
    )

    print(
        module_inventory.to_string(
            index=False
        )
    )

    print(
        "\nGEOREFERENCED CSV INVENTORY:"
    )

    print(
        georef_csvs.head(
            80
        ).to_string(
            index=False
        )
        if len(
            georef_csvs
        )
        else "NONE"
    )

    print(
        "\nWORLD-POINT / POSE FRAME OVERLAP:"
    )

    print(
        overlap.to_string(
            index=False
        )
        if len(
            overlap
        )
        else "NONE"
    )

    # ------------------------------------------------------------
    # Conservative evidence-based decision.
    # ------------------------------------------------------------
    combined_fields = " ".join(
        pose_fields[
            "path"
        ].astype(
            str
        ).tolist()
    ).lower()

    combined_sources = (
        " ".join(
            source_hits[
                "context"
            ].astype(
                str
            ).tolist()
        ).lower()
        if len(
            source_hits
        )
        else ""
    )

    has_pose_orientation = any(
        token in combined_fields
        for token in [
            "rotation",
            "quaternion",
            "orientation",
            "yaw",
            "pitch",
            "roll",
            "transform",
            "extrinsic",
        ]
    )

    has_camera_model = any(
        token in (
            combined_fields
            +
            " "
            +
            combined_sources
        )
        for token in [
            "intrinsic",
            "fovy",
            "fov",
            "projection",
            "camera",
        ]
    )

    has_projection_code = any(
        token in combined_sources
        for token in [
            "project",
            "projection",
            "intrinsic",
            "fovy",
            "ray",
        ]
    )

    full_frame_overlap = False

    if len(
        overlap
    ):
        vals = pd.to_numeric(
            overlap[
                "csv_overlap_fraction"
            ],
            errors="coerce"
        )

        full_frame_overlap = bool(
            (
                vals.dropna()
                >=
                0.999
            ).any()
        )

    if (
        has_pose_orientation
        and
        has_camera_model
        and
        has_projection_code
        and
        full_frame_overlap
    ):
        status = (
            "BAMBI_ANCHOR_REPROJECTION_INPUTS_APPEAR_AVAILABLE_"
            "READY_FOR_NUMERIC_EQUIVALENCE_CONSTRUCTION_TEST"
        )

    else:
        status = (
            "BAMBI_ANCHOR_REPROJECTION_NOT_YET_PROVEN_"
            "DO_NOT_RUN_STEP7"
        )

    summary = {
        "step":
            "STEP_6D_K3C7_BAMBI_ANCHOR_CAMERA_REPROJECTION_EQUIVALENCE_AUDIT",

        "has_pose_orientation_evidence":
            bool(
                has_pose_orientation
            ),

        "has_camera_model_evidence":
            bool(
                has_camera_model
            ),

        "has_projection_code_evidence":
            bool(
                has_projection_code
            ),

        "has_full_worldpoint_pose_frame_overlap_candidate":
            bool(
                full_frame_overlap
            ),

        "status":
            status,

        "important_limit": (
            "Flight-6 FOVY=50 degrees remains an assumed/default BAMBI "
            "reference-pipeline value, not independently measured camera calibration."
        ),

        "decision_rule": (
            "Only proceed to BAMBI Step18/Step7 reconstruction if the same "
            "observation-start anchor-camera reprojection can be numerically "
            "constructed from the existing BAMBI world points and camera pose/model."
        ),
    }

    (
        OUT
        / "STEP6D_K3C7_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
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
        "\nSTEP 6D-K3C7 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
