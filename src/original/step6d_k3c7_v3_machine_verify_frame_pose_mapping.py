from pathlib import Path
import ast
import hashlib
import importlib.util
import json
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
    / "bambi_anchor_reprojection_equivalence_audit_v3"
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


def discover_pose_json():
    matches = sorted(
        p for p in DOWNLOADS.rglob("6_matched_poses.json")
        if p.is_file()
    )

    preferred = [
        p for p in matches
        if "bambi_downloads" in str(p).lower()
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(
        "Could not resolve one 6_matched_poses.json:\n"
        + "\n".join(str(p) for p in matches)
    )


def get_official_source():
    spec = importlib.util.find_spec(
        "bambi.georeference_mot17_tracks"
    )

    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Installed bambi.georeference_mot17_tracks not found."
        )

    path = Path(spec.origin)
    require(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return path, text


def source_segment(text, node):
    return ast.get_source_segment(
        text,
        node
    ) or ""


def extract_ast_evidence(text):
    tree = ast.parse(text)

    rows = []

    # ------------------------------------------------------------
    # Evidence A:
    # matched_poses['images'][frame_idx]
    # ------------------------------------------------------------
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue

        seg = source_segment(text, node)
        compact = "".join(seg.split())

        if (
            "['images'][" in compact
            or '["images"][' in compact
        ):
            slice_node = node.slice

            if isinstance(slice_node, ast.Name):
                idx_name = slice_node.id
            else:
                idx_name = source_segment(
                    text,
                    slice_node
                ).strip()

            rows.append({
                "evidence_type":
                    "pose_list_subscript",

                "expression":
                    seg.strip(),

                "index_expression":
                    idx_name,

                "line":
                    int(
                        getattr(
                            node,
                            "lineno",
                            -1
                        )
                    ),

                "proves_direct_zero_based":
                    bool(
                        idx_name
                        ==
                        "frame_idx"
                    ),
            })

    # ------------------------------------------------------------
    # Evidence B:
    # camera = get_camera_for_frame(poses, frame_idx, ...)
    # ------------------------------------------------------------
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id

        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != "get_camera_for_frame":
            continue

        args = [
            source_segment(
                text,
                a
            ).strip()
            for a in node.args
        ]

        rows.append({
            "evidence_type":
                "get_camera_for_frame_call",

            "expression":
                source_segment(
                    text,
                    node
                ).strip(),

            "index_expression":
                (
                    args[1]
                    if len(args) > 1
                    else None
                ),

            "line":
                int(
                    getattr(
                        node,
                        "lineno",
                        -1
                    )
                ),

            "proves_direct_zero_based":
                bool(
                    len(args) > 1
                    and
                    args[1]
                    ==
                    "frame_idx"
                ),
        })

    # ------------------------------------------------------------
    # Evidence C:
    # frame_idx = int(parts[0])
    # ------------------------------------------------------------
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        targets = [
            t.id
            for t in node.targets
            if isinstance(
                t,
                ast.Name
            )
        ]

        if "frame_idx" not in targets:
            continue

        value_seg = source_segment(
            text,
            node.value
        ).strip()

        if "parts[0]" not in "".join(
            value_seg.split()
        ):
            continue

        rows.append({
            "evidence_type":
                "mot_frame_parse",

            "expression":
                source_segment(
                    text,
                    node
                ).strip(),

            "index_expression":
                "frame_idx",

            "line":
                int(
                    getattr(
                        node,
                        "lineno",
                        -1
                    )
                ),

            "proves_direct_zero_based":
                True,
        })

    return pd.DataFrame(rows)


def overlap_audit(world_csv, pose_json):
    world = pd.read_csv(world_csv)

    if "frame_id" not in world.columns:
        raise RuntimeError(
            "World trajectory CSV has no frame_id."
        )

    world_frames = set(
        pd.to_numeric(
            world["frame_id"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    raw = json.loads(
        pose_json.read_text(
            encoding="utf-8"
        )
    )

    images = raw.get("images", [])

    if not isinstance(images, list) or not images:
        raise RuntimeError(
            "Pose JSON images list missing/empty."
        )

    pose_frames_0 = set(
        range(
            len(images)
        )
    )

    overlap = (
        world_frames
        &
        pose_frames_0
    )

    return {
        "pose_images":
            int(len(images)),

        "world_unique_frames":
            int(len(world_frames)),

        "world_frame_min":
            int(min(world_frames)),

        "world_frame_max":
            int(max(world_frames)),

        "overlap_frames":
            int(len(overlap)),

        "world_overlap_fraction":
            float(
                len(overlap)
                /
                len(world_frames)
            ),

        "missing_world_frames":
            sorted(
                world_frames
                -
                pose_frames_0
            ),
    }


def main():
    print(
        "\nSTEP 6D-K3C7-V3 — MACHINE-VERIFIED OFFICIAL BAMBI FRAME↔POSE INDEX PROOF"
    )

    print(
        "Read-only audit. NO BAMBI conversion, prediction, geofence instance, or alarm is generated.\n"
    )

    require(WORLD_CSV)

    pose_json = discover_pose_json()
    source_path, source_text = get_official_source()

    evidence = extract_ast_evidence(
        source_text
    )

    evidence.to_csv(
        OUT
        / "TABLE_step6dk3c7v3_ast_indexing_evidence.csv",
        index=False
    )

    overlap = overlap_audit(
        WORLD_CSV,
        pose_json
    )

    proof_types = set(
        evidence.loc[
            evidence[
                "proves_direct_zero_based"
            ]
            ==
            True,
            "evidence_type"
        ].tolist()
    )

    needed = {
        "pose_list_subscript",
        "get_camera_for_frame_call",
        "mot_frame_parse",
    }

    direct_mapping_proven = (
        needed
        <=
        proof_types
    )

    full_overlap = (
        overlap[
            "world_overlap_fraction"
        ]
        >=
        0.999999999
    )

    print(
        "OFFICIAL SOURCE:"
    )
    print(source_path)

    print(
        "\nAST INDEXING EVIDENCE:"
    )
    print(
        evidence.to_string(
            index=False
        )
        if len(evidence)
        else "NONE"
    )

    print(
        "\n0-BASED FRAME↔POSE OVERLAP:"
    )
    print(
        json.dumps(
            overlap,
            indent=2
        )
    )

    if (
        direct_mapping_proven
        and
        full_overlap
    ):
        status = (
            "OFFICIAL_BAMBI_ZERO_BASED_FRAME_POSE_MAPPING_PROVEN_"
            "FULL_WORLD_FRAME_OVERLAP_READY_FOR_NUMERIC_ANCHOR_REPROJECTION_TEST"
        )

    elif direct_mapping_proven:
        status = (
            "OFFICIAL_BAMBI_ZERO_BASED_MAPPING_PROVEN_"
            "BUT_WORLD_FRAME_OVERLAP_INCOMPLETE_DO_NOT_RUN_STEP7"
        )

    else:
        status = (
            "OFFICIAL_BAMBI_ZERO_BASED_MAPPING_NOT_MACHINE_PROVEN_"
            "DO_NOT_RUN_STEP7"
        )

    summary = {
        "step":
            "STEP_6D_K3C7_V3_MACHINE_VERIFIED_OFFICIAL_BAMBI_FRAME_POSE_INDEX_PROOF",

        "official_source":
            str(source_path),

        "official_source_sha256":
            sha256_file(
                source_path
            ),

        "pose_json":
            str(pose_json),

        "world_csv":
            str(WORLD_CSV),

        "required_ast_proof_types":
            sorted(
                needed
            ),

        "observed_proof_types":
            sorted(
                proof_types
            ),

        "direct_zero_based_mapping_proven":
            bool(
                direct_mapping_proven
            ),

        "full_world_frame_overlap":
            bool(
                full_overlap
            ),

        "overlap":
            overlap,

        "status":
            status,
    }

    (
        OUT
        / "STEP6D_K3C7V3_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    with open(
        OUT
        / "STEP6D_K3C7V3_SOURCE_PROOF.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            f"OFFICIAL SOURCE: {source_path}\n"
            f"SHA256: {sha256_file(source_path)}\n\n"
        )

        for _, row in evidence.iterrows():
            f.write(
                f"{row['evidence_type']} | "
                f"line={row['line']} | "
                f"proves_direct_zero_based="
                f"{row['proves_direct_zero_based']}\n"
                f"{row['expression']}\n\n"
            )

    print(
        "\nSTATUS:"
    )
    print(status)

    print(
        "\nSTEP 6D-K3C7-V3 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
