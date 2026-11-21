
from pathlib import Path
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION = DOWNLOADS / "paper_strengthening_revision"
R13_ROOT = REVISION / "R13_end_to_end_pipeline"

R13B_V2_DIR = R13_ROOT / "R13B_V2_bbox_gt_and_detector_provenance_fix"
R13B_V2_MANIFEST = R13B_V2_DIR / "R13B_V2_MANIFEST.json"

FINAL_META_2 = (
    MAIN1
    / "final_neural_trajectory_dataset"
    / "2s_to_2s_birdsai_test_metadata.csv"
)

CLEANED = (
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

OUT = R13_ROOT / "R13C_frozen_end_to_end_protocol_lock"
OUT.mkdir(parents=True, exist_ok=True)


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


def canon_id(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    try:
        x = float(s)
        if np.isfinite(x) and x.is_integer():
            return str(int(x))
    except Exception:
        pass
    return s


def parse_last_integer(stem):
    nums = re.findall(r"(\d+)", stem)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except Exception:
        return None


def verify_upstream():
    require(R13B_V2_MANIFEST)

    m = json.loads(
        R13B_V2_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R13B_V2_COMPATIBILITY_PASS_DETECTOR_FRAMES_TRACKING_GT_AND_NEW_TRACKER_AVAILABLE_READY_FOR_R13C_FROZEN_END_TO_END_PROTOCOL_LOCK"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"R13-B-V2 is not in PASS state: {m.get('status')}"
        )

    return m


def discover_tracker_yamls():
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent

    byte = pkg / "cfg" / "trackers" / "bytetrack.yaml"
    bot = pkg / "cfg" / "trackers" / "botsort.yaml"

    require(byte)
    require(bot)

    return {
        "ultralytics_version": getattr(ultralytics, "__version__", ""),
        "package_root": str(pkg),
        "bytetrack": byte,
        "botsort": bot,
    }


def lock_sequences_and_frame_mapping():
    require(FINAL_META_2)
    require(CLEANED)
    require(TESTREAL_IMAGES)

    meta = pd.read_csv(
        FINAL_META_2,
        low_memory=False,
    )

    sequences = sorted(
        meta["sequence"]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    tracks = pd.read_csv(
        CLEANED,
        low_memory=False,
    )

    split = (
        tracks["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    test_mask = split.isin(
        ["test_real", "testreal", "test-real"]
    )

    if not test_mask.any():
        test_mask = split.str.contains("test", regex=False)

    t = tracks.loc[test_mask].copy()
    t["sequence"] = t["sequence"].astype(str).str.strip()
    t = t[t["sequence"].isin(sequences)].copy()

    rows = []

    for seq in sequences:
        seq_dir = TESTREAL_IMAGES / seq
        require(seq_dir)

        imgs = sorted(
            [
                p for p in seq_dir.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            ]
        )

        parsed = {
            parse_last_integer(p.stem)
            for p in imgs
        }
        parsed.discard(None)

        gt = t[t["sequence"] == seq].copy()
        gt_frames = {
            int(x)
            for x in pd.to_numeric(
                gt["frame"],
                errors="coerce",
            ).dropna().tolist()
        }

        missing_gt_frames = sorted(
            gt_frames - parsed
        )

        extra_image_frames = sorted(
            parsed - gt_frames
        )

        rows.append(
            {
                "sequence": seq,
                "image_dir": str(seq_dir),
                "image_count": len(imgs),
                "parsed_frame_count": len(parsed),
                "gt_row_count": len(gt),
                "gt_unique_frame_count": len(gt_frames),
                "gt_frames_missing_image": len(missing_gt_frames),
                "image_frames_without_gt": len(extra_image_frames),
                "frame_mapping_complete_for_gt": len(missing_gt_frames) == 0,
                "first_missing_gt_frames": "|".join(
                    map(str, missing_gt_frames[:20])
                ),
            }
        )

    df = pd.DataFrame(rows)

    return sequences, df


def gt_policy_audit(sequences):
    tracks = pd.read_csv(
        CLEANED,
        low_memory=False,
    )

    split = (
        tracks["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    test_mask = split.isin(
        ["test_real", "testreal", "test-real"]
    )

    if not test_mask.any():
        test_mask = split.str.contains("test", regex=False)

    t = tracks.loc[test_mask].copy()
    t["sequence"] = t["sequence"].astype(str).str.strip()
    t = t[t["sequence"].isin(sequences)].copy()

    rows = []

    total = len(t)

    rows.append(
        {
            "field": "rows_total_six_sequences",
            "value": total,
        }
    )

    if "interpolated" in t.columns:
        vc = (
            t["interpolated"]
            .astype(str)
            .value_counts(dropna=False)
        )
        for k, v in vc.items():
            rows.append(
                {
                    "field": f"interpolated={k}",
                    "value": int(v),
                }
            )

    if "noise" in t.columns:
        vc = (
            t["noise"]
            .astype(str)
            .value_counts(dropna=False)
        )
        for k, v in vc.items():
            rows.append(
                {
                    "field": f"noise={k}",
                    "value": int(v),
                }
            )

    if "occlusion" in t.columns:
        vc = (
            t["occlusion"]
            .astype(str)
            .value_counts(dropna=False)
        )
        for k, v in vc.head(20).items():
            rows.append(
                {
                    "field": f"occlusion={k}",
                    "value": int(v),
                }
            )

    # Conservative primary tracking GT:
    # observed annotations only; interpolated boxes are excluded from metric GT.
    if "interpolated" in t.columns:
        interp_num = pd.to_numeric(
            t["interpolated"],
            errors="coerce",
        )

        observed_mask = (
            interp_num.fillna(0) == 0
        )

        primary_gt = t.loc[observed_mask].copy()
    else:
        primary_gt = t.copy()

    primary_tracks = (
        primary_gt["sequence"].astype(str)
        + "|"
        + primary_gt["object_id"].map(canon_id)
    ).nunique()

    rows.extend(
        [
            {
                "field": "primary_tracking_gt_rows_observed_only",
                "value": int(len(primary_gt)),
            },
            {
                "field": "primary_tracking_gt_unique_original_tracks",
                "value": int(primary_tracks),
            },
        ]
    )

    return pd.DataFrame(rows)


def main():
    print("\nR13-C — FROZEN END-TO-END PROTOCOL LOCK")
    print(
        "PROTOCOL LOCK ONLY. NO detector inference and NO tracking inference. "
        "This step freezes the new revision experiment before any TestReal end-to-end results are seen. "
        "It locks the detector checkpoint, six TestReal sequences, frame-to-GT mapping, primary and sensitivity "
        "trackers, detector inference settings, tracker reset policy, GT policy, metrics, and claim boundaries. "
        "No original file is modified.\n"
    )

    upstream = verify_upstream()

    detector_ckpt = Path(
        upstream["detector_checkpoint_lock"]
    )

    require(detector_ckpt)

    detector_hash = sha256_file(detector_ckpt)

    tracker_info = discover_tracker_yamls()

    sequences, frame_map = lock_sequences_and_frame_mapping()
    gt_audit = gt_policy_audit(sequences)

    if not bool(
        frame_map["frame_mapping_complete_for_gt"].all()
    ):
        raise RuntimeError(
            "At least one frozen TestReal GT frame cannot be mapped to an image file. "
            "Do not run end-to-end inference until frame mapping is resolved."
        )

    # Snapshot tracker configs into revision output so the exact defaults are frozen.
    byte_src = Path(tracker_info["bytetrack"])
    bot_src = Path(tracker_info["botsort"])

    byte_copy = OUT / "bytetrack_FROZEN.yaml"
    bot_copy = OUT / "botsort_FROZEN.yaml"

    shutil.copy2(byte_src, byte_copy)
    shutil.copy2(bot_src, bot_copy)

    byte_hash = sha256_file(byte_copy)
    bot_hash = sha256_file(bot_copy)

    # Fixed before seeing TestReal tracking results.
    detector_settings = {
        "imgsz": 640,
        "conf": 0.10,
        "iou": 0.70,
        "max_det": 300,
        "agnostic_nms": False,
        "augment": False,
        "classes": None,
        "reason_for_conf_0_10": (
            "Preserve detections down to ByteTrack's low-confidence association range; "
            "not tuned on TestReal."
        ),
    }

    protocol = {
        "protocol_name": "R13C_FROZEN_NEW_REVISION_END_TO_END_PROTOCOL",
        "frozen_utc": utc_now(),

        "claim_boundary": {
            "historical_forecasting_pipeline_end_to_end": False,
            "new_revision_experiment_end_to_end_candidate": True,
            "field_effectiveness_claim": False,
            "detector_checkpoint_metric_identity_claim": False,
            "detector_checkpoint_provenance_claim": True,
        },

        "detector": {
            "checkpoint": str(detector_ckpt),
            "checkpoint_sha256": detector_hash,
            "provenance_note": (
                "Full elephant-detector checkpoint provenance lock. "
                "Its run metrics are close to, but not asserted identical with, "
                "the authoritative held-out detector metrics."
            ),
            "settings": detector_settings,
        },

        "data": {
            "dataset_role": "BIRDSAI TestReal / conservation_drones TestReal frame sequences",
            "sequences": sequences,
            "sequence_count": len(sequences),
            "frame_root": str(TESTREAL_IMAGES),
            "split_use": "test-only; no parameter tuning",
        },

        "tracking": {
            "primary_tracker": "ByteTrack",
            "primary_tracker_yaml": str(byte_copy),
            "primary_tracker_yaml_sha256": byte_hash,
            "sensitivity_tracker": "BoT-SORT",
            "sensitivity_tracker_yaml": str(bot_copy),
            "sensitivity_tracker_yaml_sha256": bot_hash,
            "ultralytics_version": tracker_info["ultralytics_version"],
            "tracker_state_reset": "reset at the start of every sequence",
            "parameter_tuning_on_testreal": False,
            "comparison_rule": (
                "Use frozen default tracker YAMLs. Do not alter tracker thresholds after viewing results."
            ),
        },

        "tracking_ground_truth": {
            "source": str(CLEANED),
            "box_format": "XYWH where cx == x+w/2 and cy == y+h/2",
            "identity": "object_id within sequence",
            "primary_metric_gt": "observed rows only; exclude interpolated==1 rows",
            "occlusion_policy": "retain; do not tune by occlusion state",
            "noise_policy": (
                "Do not introduce a new noise-based exclusion unless source semantics are separately locked."
            ),
        },

        "metrics": {
            "detection": [
                "precision",
                "recall",
                "AP50",
            ],
            "tracking_primary": [
                "HOTA",
                "IDF1",
                "ID-switches",
                "fragmentation",
            ],
            "tracking_supporting": [
                "MOTA",
                "mostly-tracked / mostly-lost if evaluator supports",
            ],
            "downstream": [
                "forecast ADE/FDE from tracker-derived trajectories",
                "geofence warning precision/recall/F1",
                "false-warning rate",
                "TTE_MAE",
            ],
        },

        "evaluation_design": {
            "primary_tracker": "ByteTrack",
            "secondary_tracker": "BoT-SORT sensitivity only",
            "testreal_tuning": False,
            "sequence_level_outputs": True,
            "aggregate_outputs": True,
            "detector_tracker_outputs_saved_before_forecasting": True,
            "error_propagation_comparison_required": (
                "GT trajectory forecast versus tracker-derived trajectory forecast"
            ),
        },
    }

    protocol_path = OUT / "R13C_FROZEN_PROTOCOL.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    frame_path = OUT / "R13C_SEQUENCE_FRAME_MAPPING_LOCK.csv"
    gt_path = OUT / "R13C_TRACKING_GT_POLICY_AUDIT.csv"

    frame_map.to_csv(frame_path, index=False)
    gt_audit.to_csv(gt_path, index=False)

    claim_matrix = pd.DataFrame(
        [
            {
                "claim": "Historical forecasting results are end-to-end detector-to-warning results.",
                "status": "REJECTED",
            },
            {
                "claim": "The new R13 experiment may evaluate detector -> tracker -> forecasting -> warning propagation.",
                "status": "ALLOWED AFTER EXECUTION",
            },
            {
                "claim": "ByteTrack is the frozen primary tracker.",
                "status": "LOCKED",
            },
            {
                "claim": "BoT-SORT is a frozen sensitivity tracker.",
                "status": "LOCKED",
            },
            {
                "claim": "Tracker parameters may be tuned after viewing TestReal results.",
                "status": "REJECTED",
            },
            {
                "claim": "Interpolated trajectory points are used as primary tracking GT.",
                "status": "REJECTED",
            },
            {
                "claim": "Tracking metrics must include HOTA, IDF1, ID-switches, and fragmentation.",
                "status": "LOCKED",
            },
        ]
    )

    claim_path = OUT / "R13C_CLAIM_AND_PROTOCOL_MATRIX.csv"
    claim_matrix.to_csv(claim_path, index=False)

    status = (
        "R13C_FROZEN_END_TO_END_PROTOCOL_LOCKED_NO_TEST_TUNING_READY_FOR_R13D_DETECTOR_TRACKER_EXECUTION"
    )

    manifest = {
        "step": "R13C_FROZEN_END_TO_END_PROTOCOL_LOCK",
        "created_utc": utc_now(),
        "status": status,
        "training": False,
        "detector_inference": False,
        "tracker_inference": False,
        "original_files_modified": False,
        "protocol_sha256": sha256_file(protocol_path),
        "detector_checkpoint": str(detector_ckpt),
        "detector_checkpoint_sha256": detector_hash,
        "sequence_count": len(sequences),
        "frame_mapping_pass": True,
        "primary_tracker": "ByteTrack",
        "sensitivity_tracker": "BoT-SORT",
        "testreal_parameter_tuning_allowed": False,
        "outputs": {
            "protocol": str(protocol_path),
            "sequence_frame_mapping": str(frame_path),
            "gt_policy_audit": str(gt_path),
            "claim_matrix": str(claim_path),
            "bytetrack_frozen_yaml": str(byte_copy),
            "botsort_frozen_yaml": str(bot_copy),
        },
    }

    manifest_path = OUT / "R13C_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("SEQUENCE / FRAME MAPPING LOCK:")
    print(frame_map.to_string(index=False))

    print("\nTRACKING GT POLICY AUDIT:")
    print(gt_audit.to_string(index=False))

    print("\nFROZEN DETECTOR:")
    print(detector_ckpt)
    print("SHA256:", detector_hash)

    print("\nFROZEN TRACKERS:")
    print("Primary ByteTrack:", byte_copy)
    print("SHA256:", byte_hash)
    print("Sensitivity BoT-SORT:", bot_copy)
    print("SHA256:", bot_hash)

    print("\nFROZEN DETECTOR SETTINGS:")
    print(json.dumps(detector_settings, indent=2))

    print("\nSTATUS:")
    print(status)

    print("\nR13-C COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
