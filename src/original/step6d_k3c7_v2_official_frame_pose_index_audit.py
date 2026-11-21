from pathlib import Path
import ast
import hashlib
import importlib.util
import inspect
import json
import re

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

WORLD_CSV = (
    BASE
    / "step6dk_valid_world_trajectory_rows.csv"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "bambi_anchor_reprojection_equivalence_audit_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


def require(path):
    if not path.exists():
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


def official_module_source():
    spec = importlib.util.find_spec(
        "bambi.georeference_mot17_tracks"
    )

    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Installed bambi.georeference_mot17_tracks source not found."
        )

    path = Path(
        spec.origin
    )

    require(
        path
    )

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return path, text


def source_contexts(path, text):
    lines = text.splitlines()

    patterns = [
        "images[",
        "images [",
        "enumerate(",
        "frame_id",
        "frame_idx",
        "frame_index",
        "frame_no",
        "frame_number",
        "mot",
        "pose",
        "poses",
        "track",
    ]

    rows = []

    for i, line in enumerate(
        lines,
        start=1
    ):
        low = line.lower()

        hits = [
            p
            for p in patterns
            if p.lower() in low
        ]

        if not hits:
            continue

        # Prefer likely frame-to-pose indexing contexts.
        neighborhood = "\n".join(
            lines[
                max(
                    0,
                    i - 5
                ):
                min(
                    len(lines),
                    i + 5
                )
            ]
        ).lower()

        strong = any(
            token in neighborhood
            for token in [
                "images[",
                "enumerate(",
                "frame_id",
                "frame_idx",
                "frame_index",
                "frame_no",
                "frame_number",
            ]
        )

        if not strong:
            continue

        lo = max(
            1,
            i - 5
        )

        hi = min(
            len(lines),
            i + 5
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
                    hits
                ),

            "text":
                line.strip(),

            "context":
                context,
        })

    return pd.DataFrame(
        rows
    )


def function_inventory(path, text):
    tree = ast.parse(
        text
    )

    rows = []

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            continue

        src = ast.get_source_segment(
            text,
            node
        ) or ""

        low = src.lower()

        score = 0
        hits = []

        for token, weight in [
            ("frame", 2),
            ("pose", 3),
            ("images", 3),
            ("mot", 2),
            ("track", 1),
            ("georef", 2),
            ("projection", 1),
            ("ray", 1),
        ]:
            if token in low:
                score += weight
                hits.append(
                    token
                )

        if score <= 0:
            continue

        rows.append({
            "file":
                str(path),

            "function":
                node.name,

            "line_start":
                int(
                    node.lineno
                ),

            "line_end":
                int(
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno
                    )
                ),

            "score":
                int(
                    score
                ),

            "hits":
                ";".join(
                    hits
                ),

            "source":
                src,
        })

    df = pd.DataFrame(
        rows
    )

    if len(
        df
    ):
        df = df.sort_values(
            [
                "score",
                "line_start",
            ],
            ascending=[
                False,
                True,
            ]
        )

    return df


def parse_pose_json(path):
    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    images = raw.get(
        "images",
        []
    )

    if not isinstance(
        images,
        list
    ) or not images:
        raise RuntimeError(
            "Pose JSON has no non-empty images list."
        )

    explicit_frame_keys = [
        "frame_id",
        "frame",
        "frame_index",
        "image_id",
        "id",
    ]

    explicit = []

    for idx, item in enumerate(
        images
    ):
        found = None

        if isinstance(
            item,
            dict
        ):
            for key in explicit_frame_keys:
                if key in item:
                    try:
                        found = int(
                            item[
                                key
                            ]
                        )
                    except Exception:
                        found = None
                    break

        explicit.append(
            found
        )

    return raw, images, explicit


def candidate_frame_mapping_audit(
    world_df,
    images,
    explicit_frames
):
    world_frames = set(
        pd.to_numeric(
            world_df[
                "frame_id"
            ],
            errors="coerce"
        )
        .dropna()
        .astype(
            int
        )
        .tolist()
    )

    n = len(
        images
    )

    candidate_sets = {
        "list_index_0_based":
            set(
                range(
                    0,
                    n
                )
            ),

        "list_index_1_based":
            set(
                range(
                    1,
                    n + 1
                )
            ),
    }

    explicit_valid = {
        x
        for x in explicit_frames
        if x is not None
    }

    if explicit_valid:
        candidate_sets[
            "explicit_frame_fields"
        ] = explicit_valid

    rows = []

    for name, pose_frames in (
        candidate_sets.items()
    ):
        overlap = (
            world_frames
            &
            pose_frames
        )

        rows.append({
            "candidate_mapping":
                name,

            "world_unique_frames":
                int(
                    len(
                        world_frames
                    )
                ),

            "pose_unique_frames":
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

            "world_overlap_fraction":
                (
                    float(
                        len(
                            overlap
                        )
                        /
                        len(
                            world_frames
                        )
                    )
                    if world_frames
                    else None
                ),

            "world_frame_min":
                (
                    int(
                        min(
                            world_frames
                        )
                    )
                    if world_frames
                    else None
                ),

            "world_frame_max":
                (
                    int(
                        max(
                            world_frames
                        )
                    )
                    if world_frames
                    else None
                ),

            "pose_frame_min":
                (
                    int(
                        min(
                            pose_frames
                        )
                    )
                    if pose_frames
                    else None
                ),

            "pose_frame_max":
                (
                    int(
                        max(
                            pose_frames
                        )
                    )
                    if pose_frames
                    else None
                ),
        })

    return pd.DataFrame(
        rows
    )


def detect_index_convention_from_source(text):
    """
    Conservative lexical inference. We only declare a convention if source
    contains a clear pose-list indexing expression tied to a frame variable.
    """
    compact = re.sub(
        r"\s+",
        " ",
        text
    )

    evidence = []

    patterns = [
        (
            "direct_frame_index",
            r'images\s*\[\s*(?:frame|frame_id|frame_idx|frame_index|frame_no|frame_number)\s*\]'
        ),
        (
            "frame_minus_one",
            r'images\s*\[\s*(?:frame|frame_id|frame_idx|frame_index|frame_no|frame_number)\s*-\s*1\s*\]'
        ),
        (
            "frame_plus_one",
            r'images\s*\[\s*(?:frame|frame_id|frame_idx|frame_index|frame_no|frame_number)\s*\+\s*1\s*\]'
        ),
    ]

    for label, pattern in patterns:
        for m in re.finditer(
            pattern,
            compact,
            flags=re.IGNORECASE
        ):
            evidence.append({
                "label":
                    label,

                "matched_expression":
                    m.group(
                        0
                    ),
            })

    return evidence


def main():
    print(
        "\nSTEP 6D-K3C7-V2 — OFFICIAL BAMBI FRAME↔POSE INDEX CONVENTION AUDIT"
    )

    print(
        "Read-only patch of K3C7. NO BAMBI conversion, predictions, geofence instances, or alarms are generated.\n"
    )

    pose_path = discover_one(
        "6_matched_poses.json"
    )

    require(
        WORLD_CSV
    )

    module_path, module_text = (
        official_module_source()
    )

    print(
        "OFFICIAL BAMBI GEOREFERENCE SOURCE:"
    )

    print(
        module_path
    )

    print(
        "\nPOSE JSON:"
    )

    print(
        pose_path
    )

    print(
        "\nWORLD TRAJECTORY:"
    )

    print(
        WORLD_CSV
    )

    raw_pose, images, explicit_frames = (
        parse_pose_json(
            pose_path
        )
    )

    world_df = pd.read_csv(
        WORLD_CSV
    )

    if "frame_id" not in world_df.columns:
        raise RuntimeError(
            "World trajectory CSV has no frame_id column."
        )

    contexts = source_contexts(
        module_path,
        module_text
    )

    functions = function_inventory(
        module_path,
        module_text
    )

    mapping = candidate_frame_mapping_audit(
        world_df,
        images,
        explicit_frames
    )

    lexical_evidence = (
        detect_index_convention_from_source(
            module_text
        )
    )

    contexts.to_csv(
        OUT
        / "TABLE_step6dk3c7v2_official_source_frame_contexts.csv",
        index=False
    )

    functions.to_csv(
        OUT
        / "TABLE_step6dk3c7v2_official_source_functions.csv",
        index=False
    )

    mapping.to_csv(
        OUT
        / "TABLE_step6dk3c7v2_candidate_frame_pose_overlap.csv",
        index=False
    )

    pd.DataFrame(
        lexical_evidence
    ).to_csv(
        OUT
        / "TABLE_step6dk3c7v2_indexing_expression_evidence.csv",
        index=False
    )

    with open(
        OUT
        / "STEP6D_K3C7V2_OFFICIAL_SOURCE_BUNDLE.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            f"SOURCE: {module_path}\n"
            f"SHA256: {sha256_file(module_path)}\n\n"
        )

        for _, row in functions.head(
            30
        ).iterrows():
            f.write(
                "=" * 100
                +
                "\n"
            )

            f.write(
                f"FUNCTION: {row['function']}\n"
                f"LINES: {row['line_start']}-{row['line_end']}\n"
                f"SCORE: {row['score']}\n"
                f"HITS: {row['hits']}\n"
            )

            f.write(
                "=" * 100
                +
                "\n"
            )

            f.write(
                row[
                    "source"
                ]
            )

            f.write(
                "\n\n"
            )

    print(
        "\nPOSE LIST SUMMARY:"
    )

    print(
        "images:",
        len(
            images
        )
    )

    print(
        "explicit frame fields:",
        sum(
            x is not None
            for x in explicit_frames
        )
    )

    print(
        "\nOFFICIAL SOURCE FRAME/POSE CONTEXTS:"
    )

    print(
        contexts.head(
            120
        ).to_string(
            index=False
        )
        if len(
            contexts
        )
        else "NONE"
    )

    print(
        "\nEXACT INDEXING EXPRESSION EVIDENCE:"
    )

    if lexical_evidence:
        print(
            pd.DataFrame(
                lexical_evidence
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nCANDIDATE FRAME↔POSE OVERLAP:"
    )

    print(
        mapping.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Conservative decision.
    # ------------------------------------------------------------
    labels = {
        x[
            "label"
        ]
        for x in lexical_evidence
    }

    proven_mapping = None

    if (
        "direct_frame_index"
        in
        labels
        and
        "frame_minus_one"
        not in
        labels
    ):
        proven_mapping = (
            "list_index_0_based"
        )

    elif (
        "frame_minus_one"
        in
        labels
        and
        "direct_frame_index"
        not in
        labels
    ):
        proven_mapping = (
            "list_index_1_based"
        )

    if proven_mapping is not None:
        row = mapping[
            mapping[
                "candidate_mapping"
            ]
            ==
            proven_mapping
        ].iloc[
            0
        ]

        overlap_fraction = float(
            row[
                "world_overlap_fraction"
            ]
        )

        if overlap_fraction >= 0.999999:
            status = (
                "OFFICIAL_BAMBI_FRAME_POSE_MAPPING_PROVEN_"
                "FULL_WORLD_FRAME_OVERLAP_READY_FOR_NUMERIC_ANCHOR_REPROJECTION_TEST"
            )
        else:
            status = (
                "OFFICIAL_FRAME_POSE_MAPPING_PROVEN_BUT_"
                "WORLD_FRAME_OVERLAP_INCOMPLETE_DO_NOT_RUN_STEP7"
            )

    else:
        # If both 0- and 1-based candidates give full overlap but source
        # convention is not explicit, we do NOT guess.
        status = (
            "POSE_LIST_INDEX_RANGE_OVERLAPS_WORLD_FRAMES_BUT_"
            "OFFICIAL_ZERO_VS_ONE_BASED_CONVENTION_NOT_YET_PROVEN"
        )

    summary = {
        "step":
            "STEP_6D_K3C7_V2_OFFICIAL_BAMBI_FRAME_POSE_INDEX_CONVENTION_AUDIT",

        "official_georeference_source":
            str(
                module_path
            ),

        "official_georeference_source_sha256":
            sha256_file(
                module_path
            ),

        "pose_json":
            str(
                pose_path
            ),

        "pose_images":
            int(
                len(
                    images
                )
            ),

        "explicit_pose_frame_fields":
            int(
                sum(
                    x is not None
                    for x in explicit_frames
                )
            ),

        "proven_mapping":
            proven_mapping,

        "status":
            status,

        "note":
            (
                "K3C7 reported zero pose frames because the JSON image records "
                "do not carry explicit frame IDs. K3C7-V2 resolves the official "
                "list-index convention from the installed BAMBI georeference source."
            ),
    }

    (
        OUT
        / "STEP6D_K3C7V2_SUMMARY.json"
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
        "\nSTEP 6D-K3C7-V2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
