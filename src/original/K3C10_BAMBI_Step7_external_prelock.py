from pathlib import Path
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone

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
    / "step7_external_prelock"
)

OUT.mkdir(parents=True, exist_ok=True)

ORIGINAL_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

FOVY_DEG = 50.0
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def add_file(records, label, path, required=True):
    path = Path(path)

    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required lock input missing: {label}: {path}"
            )

        records.append({
            "label": label,
            "path": str(path),
            "exists": False,
            "sha256": None,
            "size_bytes": None,
        })
        return

    records.append({
        "label": label,
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    })


def discover_one(root, filename, prefer=()):
    matches = sorted(
        p for p in Path(root).rglob(filename)
        if p.is_file()
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {root}"
        )

    if prefer:
        preferred = [
            p for p in matches
            if all(
                token.lower() in str(p).lower()
                for token in prefer
            )
        ]
        if len(preferred) == 1:
            return preferred[0]

    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(
        f"Ambiguous {filename} under {root}:\n"
        + "\n".join(str(p) for p in matches)
    )


def discover_by_name_fragment(root, fragments, suffix=None, require_unique=True):
    root = Path(root)

    matches = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        name = p.name.lower()

        if suffix is not None and p.suffix.lower() != suffix.lower():
            continue

        if all(
            fragment.lower() in name
            for fragment in fragments
        ):
            matches.append(p)

    matches = sorted(matches)

    if not matches:
        raise FileNotFoundError(
            f"No file under {root} matched fragments {fragments}"
        )

    if require_unique and len(matches) != 1:
        raise RuntimeError(
            f"Ambiguous fragment match {fragments}:\n"
            + "\n".join(str(p) for p in matches)
        )

    return matches[0] if require_unique else matches


def official_bambi_source():
    spec = importlib.util.find_spec(
        "bambi.georeference_mot17_tracks"
    )

    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Installed bambi.georeference_mot17_tracks not found."
        )

    return Path(spec.origin)


def enriched_pose_source():
    matches = sorted(
        p
        for p in BASE.rglob("6_matched_poses.json")
        if p.is_file()
        and "correction" in str(p.parent).lower()
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one enriched correction/6_matched_poses.json:\n"
            + "\n".join(str(p) for p in matches)
        )

    return matches[0]


def first_inference_npz(task):
    folder = (
        BASE
        / "k3_frozen_ensemble"
        / "first_inference"
    )

    matches = sorted(
        p
        for p in folder.glob("*.npz")
        if task.lower() in p.name.lower()
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one first-inference NPZ for {task}:\n"
            + "\n".join(str(p) for p in matches)
        )

    return matches[0]


def main():
    print(
        "\nSTEP 6D-K3C10 — BAMBI EXTERNAL STEP7 PRELOCK"
    )
    print(
        "Creates a provenance/rule lock BEFORE any BAMBI Step7 operational metrics."
    )
    print(
        "NO BAMBI warning metrics, alarms, q recalibration, or threshold tuning are generated.\n"
    )

    records = []

    # ------------------------------------------------------------
    # Core frozen development rule / uncertainty sources
    # ------------------------------------------------------------
    step7_manifest = (
        DOWNLOADS
        / "step7_response_aware_warning"
        / "STEP7_FINAL_DEVELOPMENT_MANIFEST.json"
    )
    add_file(
        records,
        "frozen_step7h_development_manifest",
        step7_manifest,
    )

    step6c_script = (
        DOWNLOADS
        / "step6c_crossfit_conformal.py"
    )
    add_file(
        records,
        "step6c_original_crossfit_conformal_source",
        step6c_script,
    )

    # Exact Step6C source/formula audit already completed.
    k3b0v3_dir = (
        BASE
        / "k3_fixed_conformal"
        / "source_audit_v3"
    )

    k3b0v3_files = sorted(
        p
        for p in k3b0v3_dir.iterdir()
        if p.is_file()
    )

    if not k3b0v3_files:
        raise FileNotFoundError(
            f"No K3B0-V3 exact conformal audit files found in {k3b0v3_dir}"
        )

    for p in k3b0v3_files:
        add_file(
            records,
            f"k3b0v3_exact_conformal_audit::{p.name}",
            p,
        )

    # ------------------------------------------------------------
    # Frozen BAMBI representation/model/inference
    # ------------------------------------------------------------
    k2b3_lock = (
        BASE
        / "k2b3_final_canonical"
        / "STEP6D_K2B3_CANONICAL_DATASET_LOCK.json"
    )
    add_file(
        records,
        "bambi_k2b3_canonical_dataset_lock",
        k2b3_lock,
    )

    model_lock_dir = (
        BASE
        / "k3_frozen_ensemble"
        / "model_lock"
    )

    model_lock_files = sorted(
        p
        for p in model_lock_dir.iterdir()
        if p.is_file()
    )

    if not model_lock_files:
        raise FileNotFoundError(
            f"No K3A1 model-lock files found in {model_lock_dir}"
        )

    for p in model_lock_files:
        add_file(
            records,
            f"k3a1_model_lock::{p.name}",
            p,
        )

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        add_file(
            records,
            f"first_locked_inference::{task}",
            first_inference_npz(task),
        )

    # Final external conformal result, animal-track unit.
    k3b1v2_dir = (
        BASE
        / "k3_fixed_conformal"
        / "k3b1_v2_fixed_no_recalibration_animal_track_unit"
    )

    k3b1v2_primary = discover_by_name_fragment(
        k3b1v2_dir,
        [
            "primary",
            "coverage",
        ],
        suffix=".csv",
    )
    add_file(
        records,
        "k3b1v2_final_external_conformal_primary_coverage",
        k3b1v2_primary,
    )

    # ------------------------------------------------------------
    # Exact geometry / coordinate-equivalence provenance
    # ------------------------------------------------------------
    add_file(
        records,
        "bambi_official_georeference_source",
        official_bambi_source(),
    )

    add_file(
        records,
        "bambi_enriched_pose_source",
        enriched_pose_source(),
    )

    correction_json = discover_one(
        DOWNLOADS,
        "6_correction.json",
        prefer=("bambi_downloads",),
    )
    add_file(
        records,
        "bambi_flight6_correction_json",
        correction_json,
    )

    dem_tif = discover_one(
        DOWNLOADS,
        "6_matched_dem.tif",
    )
    dem_glb = discover_one(
        DOWNLOADS,
        "6_matched_dem.glb",
    )
    dem_json = discover_one(
        DOWNLOADS,
        "6_matched_dem.json",
    )

    add_file(
        records,
        "bambi_dem_tif",
        dem_tif,
    )
    add_file(
        records,
        "bambi_dem_glb",
        dem_glb,
    )
    add_file(
        records,
        "bambi_dem_metadata",
        dem_json,
    )

    k3c9_summary = (
        BASE
        / "k3_operational_warning"
        / "camera_matrix_pixel_roundtrip_v5"
        / "STEP6D_K3C9_SUMMARY.json"
    )

    if not k3c9_summary.exists():
        # Handle V5 summary name if version was embedded.
        candidates = sorted(
            (
                BASE
                / "k3_operational_warning"
                / "camera_matrix_pixel_roundtrip_v5"
            ).glob("*SUMMARY*.json")
        )

        if len(candidates) != 1:
            raise FileNotFoundError(
                "Could not uniquely resolve K3C9 V5 summary JSON."
            )

        k3c9_summary = candidates[0]

    add_file(
        records,
        "k3c9_exact_pixel_roundtrip_summary",
        k3c9_summary,
    )

    # Step18/Step7 exact source audits.
    operational_root = (
        BASE
        / "k3_operational_warning"
    )

    audit_dirs = [
        "exact_rule_audit",
        "base_instance_rule_audit",
        "step18_geometry_audit",
        "coordinate_compatibility_audit",
        "coordinate_mapping_provenance",
        "exact_normalized_mapping_confirmation_v2",
        "bambi_anchor_reprojection_equivalence_audit_v3",
    ]

    for dirname in audit_dirs:
        folder = operational_root / dirname

        if not folder.exists():
            raise FileNotFoundError(
                f"Required provenance audit folder missing: {folder}"
            )

        files = sorted(
            p
            for p in folder.iterdir()
            if p.is_file()
        )

        if not files:
            raise FileNotFoundError(
                f"Required provenance audit folder empty: {folder}"
            )

        for p in files:
            add_file(
                records,
                f"operational_provenance::{dirname}::{p.name}",
                p,
            )

    # ------------------------------------------------------------
    # Lock the exact frozen formulas/rules explicitly.
    # ------------------------------------------------------------
    fixed_rule = {
        "coordinate_system":
            "camera-corrected normalized image-plane coordinates",

        "image_width":
            IMAGE_WIDTH,

        "image_height":
            IMAGE_HEIGHT,

        "bambi_fovy_degrees":
            FOVY_DEG,

        "bambi_fovy_status":
            (
                "Assumed/default value from BAMBI reference pipeline; "
                "not independently measured camera calibration."
            ),

        "frame_pose_mapping":
            "official direct zero-based frame_idx -> matched_poses['images'][frame_idx]",

        "world_to_pixel_matrix_convention":
            "col_PT_VT",

        "world_to_pixel_expression":
            "clip = projection_matrix.T @ view_matrix.T @ [x,y,z,1]^T",

        "pixel_mapping":
            "top-left image coordinates",

        "pixel_from_ndc":
            {
                "u":
                    "(ndc_x + 1) * 0.5 * width",

                "v":
                    "(1 - ndc_y) * 0.5 * height",
            },

        "normalization":
            {
                "x_norm":
                    "u / width",

                "y_norm":
                    "v / height",
            },

        "anchor_camera":
            "observation-start camera for each forecast window",

        "primary_conformal_level":
            0.90,

        "persistence_windows":
            2,

        "persistence_semantics":
            (
                "2 consecutive available forecast windows for the same virtual geofence; "
                "not 2 seconds"
            ),

        "step18_virtual_boundaries":
            [
                0.25,
                0.50,
                0.75,
            ],

        "step18_axes":
            [
                "x",
                "y",
            ],

        "step18_direction_rule":
            (
                "Current side is safe; opposite side is risk. "
                "If current < boundary, sign=+1/high side risk; "
                "otherwise sign=-1/low side risk."
            ),

        "step18_prediction_score":
            "max directional signed boundary penetration over forecast",

        "step18_true_crossing":
            "any GT future reaches/crosses the risk side",

        "step18_tte":
            "first future timestep reaching/crossing the risk side",

        "step7_red":
            (
                "Nominal predicted crossing at/before assumed response threshold; "
                "or already at/inside boundary if TTE unavailable."
            ),

        "step7_yellow":
            (
                "Not RED and (nominal predicted crossing after threshold OR "
                "90% conformal intersection with same virtual geofence for "
                "2 consecutive available forecast windows)."
            ),

        "step7_green":
            "Neither RED nor YELLOW.",

        "evaluation_unit":
            "individual virtual-geofence scenario; not OR-union across six boundaries",

        "no_bambi_recalibration":
            True,

        "no_bambi_q_tuning":
            True,

        "no_bambi_model_tuning":
            True,

        "no_bambi_threshold_tuning":
            True,

        "step7h_development_rule_frozen":
            True,
    }

    manifest = {
        "step":
            "STEP_6D_K3C10_BAMBI_EXTERNAL_STEP7_PRELOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "LOCKED_BEFORE_ANY_BAMBI_STEP7_OPERATIONAL_METRICS",

        "dataset":
            "BAMBI Flight 6",

        "role":
            (
                "Cross-species/cross-domain external operational transfer stress test. "
                "Not elephant-specific validation and not field validation."
            ),

        "fixed_rule":
            fixed_rule,

        "locked_files":
            records,

        "critical_nonclaims": [
            "BAMBI Flight 6 is red-deer/human dominated, not an elephant dataset.",
            "The 50 degree FOVY is an assumption/default from the BAMBI reference pipeline.",
            "Virtual geofences are normalized synthetic boundaries, not measured village/farm boundaries.",
            "The 2s/5s forecast horizon is not a measured human/team response time.",
            "No BAMBI conformal recalibration is permitted.",
            "Step7H remains a development-frozen rule; BAMBI must not be used to tune it.",
            "A failed BAMBI operational transfer result must remain visible and must not be repaired by tuning.",
        ],
    }

    manifest_path = (
        OUT
        / "STEP6D_K3C10_BAMBI_STEP7_PRELOCK.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    manifest_hash = sha256_file(
        manifest_path
    )

    receipt = {
        "manifest":
            str(manifest_path),

        "manifest_sha256":
            manifest_hash,

        "locked_file_count":
            len(records),

        "status":
            "PRELOCK_COMPLETE_NO_OPERATIONAL_METRICS_GENERATED",
    }

    (
        OUT
        / "STEP6D_K3C10_PRELOCK_RECEIPT.json"
    ).write_text(
        json.dumps(
            receipt,
            indent=2
        ),
        encoding="utf-8"
    )

    print("LOCKED FILE COUNT:", len(records))
    print("MANIFEST:", manifest_path)
    print("MANIFEST SHA256:", manifest_hash)

    print("\nFROZEN RULE:")
    print(
        json.dumps(
            fixed_rule,
            indent=2
        )
    )

    print("\nSTATUS:")
    print(
        "PRELOCK_COMPLETE_NO_OPERATIONAL_METRICS_GENERATED"
    )

    print("\nSTEP 6D-K3C10 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
