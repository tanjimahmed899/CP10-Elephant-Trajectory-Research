
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R18_ROOT = REVISION / "R18_meter_space_wildbox"

R18A2_DIR = R18_ROOT / "R18A2_wildbox_only_metric_scale_audit"
R18A2_MANIFEST = R18A2_DIR / "R18A2_MANIFEST.json"
R18A2_SUMMARY = R18A2_DIR / "R18A2_SCALE_READINESS_SUMMARY.csv"
R18A2_SOURCE = R18A2_DIR / "R18A2_PER_SOURCE_SCALE_READINESS.csv"

OUT = R18_ROOT / "R18B_meter_space_limitation_lock"
OUT.mkdir(parents=True, exist_ok=True)


EXPECTED_STATUS = (
    "R18A2_WILDBOX_CAMERA_GEOMETRY_PRESENT_BUT_NO_VERIFIED_METRIC_SCALE_ANCHOR_"
    "NO_DEFENSIBLE_METER_CONVERSION_READY_FOR_R18B_LIMITATION_LOCK"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    print("\nR18-B — WILDBOX METER-SPACE LIMITATION LOCK")
    print(
        "LIMITATION LOCK ONLY. NO coordinate conversion, NO forecasting rerun, and NO "
        "assumed altitude/FOV/scale insertion. This step freezes the scientifically "
        "defensible interpretation of the strict R18-A2 audit.\n"
    )

    for p in [R18A2_MANIFEST, R18A2_SUMMARY, R18A2_SOURCE]:
        require(p)

    manifest = json.loads(
        R18A2_MANIFEST.read_text(encoding="utf-8")
    )

    if manifest.get("status") != EXPECTED_STATUS:
        raise RuntimeError(
            f"Unexpected R18-A2 status: {manifest.get('status')}"
        )

    summary = pd.read_csv(R18A2_SUMMARY)
    source = pd.read_csv(R18A2_SOURCE)

    if len(summary) != 1:
        raise RuntimeError("Expected exactly one R18-A2 summary row.")

    s = summary.iloc[0]

    sources_expected = int(s["sources_expected"])
    sources_found = int(s["sources_found"])
    scale_anchors = int(s["sources_with_metric_scale_anchor"])
    intrinsics = int(s["sources_with_intrinsics"])
    extrinsics = int(s["sources_with_extrinsics"])
    meter_ready = int(s["sources_direct_meter_ready"])

    if not (
        sources_expected == 3
        and sources_found == 3
        and scale_anchors == 0
        and intrinsics == 3
        and extrinsics == 3
        and meter_ready == 0
    ):
        raise RuntimeError(
            "R18-A2 source counts do not match the expected locked audit state."
        )

    claim_rows = [
        {
            "claim_id": "R18_C1",
            "claim": "WildBox trajectories/results can be reported directly in meters.",
            "status": "REJECTED",
            "reason": (
                "No verified source-specific metric-scale anchor was found in any of the "
                "three WildBox sources."
            ),
        },
        {
            "claim_id": "R18_C2",
            "claim": "WildBox camera intrinsics/extrinsics alone establish metric distance.",
            "status": "REJECTED",
            "reason": (
                "Camera geometry exists, but the reconstruction scale is not independently "
                "verified in physical units."
            ),
        },
        {
            "claim_id": "R18_C3",
            "claim": "Normalized/image-plane WildBox ADE/FDE may be converted to meters using an assumed altitude.",
            "status": "REJECTED",
            "reason": (
                "No measured or source-verified altitude/metric anchor was available; adding "
                "an assumed scale would create unsupported physical-error claims."
            ),
        },
        {
            "claim_id": "R18_C4",
            "claim": "WildBox supports external trajectory-forecasting evaluation in normalized/image space.",
            "status": "SUPPORTED",
            "reason": (
                "The existing frozen WildBox forecasting results remain valid in their original "
                "reported coordinate representation; R18 does not invalidate those results."
            ),
        },
        {
            "claim_id": "R18_C5",
            "claim": "Meter-space evaluation is a current limitation that should be disclosed in the paper.",
            "status": "SUPPORTED",
            "reason": (
                "All three sources contain camera geometry but zero verified metric-scale anchors."
            ),
        },
        {
            "claim_id": "R18_C6",
            "claim": "The available WildBox camera matrices may be described as world-ground-truth metric poses.",
            "status": "REJECTED",
            "reason": (
                "Available intrinsics/extrinsics do not prove metric world coordinates or a "
                "known physical scale."
            ),
        },
    ]

    claims = pd.DataFrame(claim_rows)

    manuscript_rows = [
        {
            "section": "Methods / Evaluation",
            "locked_text": (
                "WildBox evaluation is reported in the original normalized/image-coordinate "
                "representation. Although per-frame camera intrinsics and extrinsics were "
                "available, no source-specific metric-scale anchor such as measured altitude, "
                "known distance, ground-sample distance, or calibrated meters-per-pixel factor "
                "was verified for the three evaluated WildBox sources."
            ),
        },
        {
            "section": "Limitations",
            "locked_text": (
                "Accordingly, WildBox ADE/FDE values were not converted to meters. The camera "
                "geometry is sufficient for geometric consistency analysis but, without an "
                "independently verified physical scale, does not justify metric-distance claims."
            ),
        },
        {
            "section": "Claims",
            "locked_text": (
                "No claim is made that WildBox forecast errors, geofence distances, or camera "
                "translations correspond directly to meters."
            ),
        },
    ]

    manuscript = pd.DataFrame(manuscript_rows)

    claim_path = OUT / "R18B_CLAIM_MATRIX_LOCKED.csv"
    manuscript_path = OUT / "R18B_MANUSCRIPT_WORDING_LOCK.csv"

    claims.to_csv(claim_path, index=False)
    manuscript.to_csv(manuscript_path, index=False)

    status = (
        "R18_WILDBOX_METER_SPACE_NOT_DEFENSIBLE_WITH_AVAILABLE_METADATA_"
        "LIMITATION_LOCKED_NORMALIZED_IMAGE_SPACE_RESULTS_RETAINED_READY_FOR_R19_CLAIM_MATRIX_AND_PAPER_REWRITE"
    )

    lock = {
        "step": "R18B_WILDBOX_METER_SPACE_LIMITATION_LOCK",
        "created_utc": utc_now(),
        "status": status,
        "r18_complete": True,
        "coordinate_conversion_performed": False,
        "forecast_rerun_performed": False,
        "sources_expected": sources_expected,
        "sources_found": sources_found,
        "sources_with_verified_metric_scale_anchor": scale_anchors,
        "sources_with_intrinsics": intrinsics,
        "sources_with_extrinsics": extrinsics,
        "sources_direct_meter_ready": meter_ready,
        "meter_space_claim_allowed": False,
        "normalized_image_space_results_retained": True,
        "locked_interpretation": (
            "WildBox camera geometry is available, but no independently verified metric-scale "
            "anchor exists for the three frozen sources. Therefore, no defensible meter-space "
            "conversion is performed. Existing normalized/image-space forecasting results remain "
            "valid and must be reported in their original coordinate representation."
        ),
        "prohibited_shortcuts": [
            "Do not insert an assumed drone altitude.",
            "Do not infer meters from extrinsic translation magnitude alone.",
            "Do not infer meters from focal length/intrinsics alone.",
            "Do not call reconstructed camera coordinates world-ground-truth meters.",
            "Do not convert normalized ADE/FDE to meters without external scale calibration.",
        ],
        "future_resolution": (
            "A future metric-space extension would require source-specific measured altitude, "
            "known ground distance/GCPs, calibrated GSD/meters-per-pixel, or another independently "
            "validated physical-scale reference."
        ),
        "input_hashes": {
            "R18A2_manifest": sha256_file(R18A2_MANIFEST),
            "R18A2_summary": sha256_file(R18A2_SUMMARY),
            "R18A2_per_source": sha256_file(R18A2_SOURCE),
        },
        "output_hashes": {
            "claim_matrix": sha256_file(claim_path),
            "manuscript_wording": sha256_file(manuscript_path),
        },
        "next_roadmap_step": "R19_CLAIM_MATRIX_AND_PAPER_REWRITE",
    }

    lock_path = OUT / "R18B_METER_SPACE_LIMITATION_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2),
        encoding="utf-8",
    )

    print("AUDIT STATE:")
    print(
        f"  WildBox sources found = {sources_found}/{sources_expected}\n"
        f"  sources with intrinsics = {intrinsics}\n"
        f"  sources with extrinsics = {extrinsics}\n"
        f"  sources with verified metric-scale anchor = {scale_anchors}\n"
        f"  sources directly meter-ready = {meter_ready}"
    )

    print("\nCLAIM MATRIX:")
    print(claims.to_string(index=False))

    print("\nLOCKED PAPER INTERPRETATION:")
    print(
        "WildBox results remain valid in normalized/image space, but no meter-space "
        "conversion or meter-valued error claim is allowed with the available metadata."
    )

    print("\nSTATUS:")
    print(status)

    print("\nR18-B COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
