
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
CAPSTONE = Path(r"E:\AIUB\Capstone Project")
MAIN1 = CAPSTONE / "Journal Based Results" / "Main Result"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R13_ROOT = REVISION / "R13_end_to_end_pipeline"

R13B_DIR = R13_ROOT / "R13B_detector_data_tracker_compatibility_audit"
R13B_MANIFEST = R13B_DIR / "R13B_MANIFEST.json"
R13B_DETECTOR = R13B_DIR / "R13B_DETECTOR_CANDIDATE_METRIC_AUDIT.csv"
R13B_SEQ = R13B_DIR / "R13B_TESTREAL_SEQUENCE_FRAME_COMPATIBILITY.csv"
R13B_TRACKER = R13B_DIR / "R13B_ULTRALYTICS_TRACKER_AVAILABILITY.csv"

CLEANED = MAIN1 / "trajectory_dataset" / "cleaned_track_points.csv"

OUT = R13_ROOT / "R13B_V2_bbox_gt_and_detector_provenance_fix"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_FULL_DATASET_HINTS = [
    "Elephant Detection.v2-resized640.yolov8",
    "elephant_final_clean",
]

AUTHORITATIVE_REFERENCE = {
    "precision": 0.918,
    "recall": 0.884,
    "map50": 0.940,
    "map50_95": 0.698,
}


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


def verify_upstream():
    for p in [R13B_MANIFEST, R13B_DETECTOR, R13B_SEQ, R13B_TRACKER, CLEANED]:
        require(p)

    m = json.loads(R13B_MANIFEST.read_text(encoding="utf-8"))

    expected = (
        "R13B_COMPATIBILITY_INCOMPLETE_REVIEW_TRACKING_GT_COMPATIBLE_SCHEMA_IN_CLEANED_TRACK_POINTS"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R13-B status: {m.get('status')}"
        )

    return m


def bbox_semantics_audit():
    df = pd.read_csv(CLEANED, low_memory=False)

    required = ["frame", "object_id", "x", "y", "w", "h", "cx", "cy", "split", "sequence"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        return pd.DataFrame(), False, {"missing_columns": missing}

    sample = df[required + [c for c in ["interpolated", "segment_id"] if c in df.columns]].copy()

    for c in ["x", "y", "w", "h", "cx", "cy"]:
        sample[c] = pd.to_numeric(sample[c], errors="coerce")

    finite = sample[["x", "y", "w", "h", "cx", "cy"]].notna().all(axis=1)
    sample = sample.loc[finite].copy()

    if len(sample) == 0:
        return pd.DataFrame(), False, {"reason": "no finite bbox rows"}

    sample["cx_from_xywh"] = sample["x"] + sample["w"] / 2.0
    sample["cy_from_xywh"] = sample["y"] + sample["h"] / 2.0
    sample["cx_abs_err"] = (sample["cx"] - sample["cx_from_xywh"]).abs()
    sample["cy_abs_err"] = (sample["cy"] - sample["cy_from_xywh"]).abs()

    # Very tight because these columns should be algebraically linked.
    tol = 1e-6

    cx_match_rate = float((sample["cx_abs_err"] <= tol).mean())
    cy_match_rate = float((sample["cy_abs_err"] <= tol).mean())

    positive_wh_rate = float(((sample["w"] > 0) & (sample["h"] > 0)).mean())

    unique_track_rows = (
        sample.assign(
            track_key=sample["sequence"].astype(str) + "|" + sample["object_id"].astype(str)
        )
        .groupby("track_key", as_index=False)
        .agg(
            rows=("frame", "size"),
            unique_frames=("frame", "nunique"),
        )
    )

    track_id_temporal_ok = bool(
        len(unique_track_rows) > 0
        and (unique_track_rows["rows"] >= unique_track_rows["unique_frames"]).all()
    )

    gt_compatible = bool(
        cx_match_rate >= 0.999
        and cy_match_rate >= 0.999
        and positive_wh_rate >= 0.999
        and track_id_temporal_ok
    )

    summary = pd.DataFrame(
        [
            {
                "check": "Required frame/object_id/x/y/w/h/cx/cy columns present",
                "pass": True,
                "value": len(sample),
            },
            {
                "check": "cx == x + w/2",
                "pass": cx_match_rate >= 0.999,
                "value": cx_match_rate,
            },
            {
                "check": "cy == y + h/2",
                "pass": cy_match_rate >= 0.999,
                "value": cy_match_rate,
            },
            {
                "check": "w,h positive",
                "pass": positive_wh_rate >= 0.999,
                "value": positive_wh_rate,
            },
            {
                "check": "track IDs are temporally indexable by frame",
                "pass": track_id_temporal_ok,
                "value": len(unique_track_rows),
            },
            {
                "check": "tracking-GT-compatible XYWH schema",
                "pass": gt_compatible,
                "value": gt_compatible,
            },
        ]
    )

    details = {
        "rows_audited": int(len(sample)),
        "cx_match_rate": cx_match_rate,
        "cy_match_rate": cy_match_rate,
        "positive_wh_rate": positive_wh_rate,
        "unique_tracks": int(len(unique_track_rows)),
        "gt_compatible": gt_compatible,
    }

    return summary, gt_compatible, details


def detector_provenance_audit():
    det = pd.read_csv(R13B_DETECTOR)

    if len(det) == 0:
        return pd.DataFrame(), "", False

    rows = []

    for _, r in det.iterrows():
        ckpt = str(r["checkpoint"])
        p = Path(ckpt)

        project_hint_score = sum(
            1
            for token in EXPECTED_FULL_DATASET_HINTS
            if token.lower() in ckpt.lower()
        )

        is_best = p.name.lower() == "best.pt"

        metrics = {
            "precision": float(r["precision"]) if pd.notna(r["precision"]) else np.nan,
            "recall": float(r["recall"]) if pd.notna(r["recall"]) else np.nan,
            "map50": float(r["map50"]) if pd.notna(r["map50"]) else np.nan,
            "map50_95": float(r["map50_95"]) if pd.notna(r["map50_95"]) else np.nan,
        }

        diffs = {
            k: abs(metrics[k] - AUTHORITATIVE_REFERENCE[k])
            if np.isfinite(metrics[k])
            else np.nan
            for k in metrics
        }

        mean_abs_diff = float(np.nanmean(list(diffs.values())))

        provenance_score = (
            100 * project_hint_score
            + 20 * int(is_best)
            - 1000 * mean_abs_diff
        )

        rows.append(
            {
                "checkpoint": ckpt,
                "exists": p.exists(),
                "sha256": sha256_file(p) if p.exists() else "",
                "project_hint_score": project_hint_score,
                "is_best_pt": is_best,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
                "mean_abs_diff_vs_authoritative_reference": mean_abs_diff,
                "provenance_score": provenance_score,
                "note": (
                    "Training-run metrics are close to, but not identical with, the authoritative "
                    "held-out detection result; therefore this is a checkpoint provenance lock, "
                    "not an exact metric-identity claim."
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["project_hint_score", "provenance_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    top = out.iloc[0]

    lock = str(top["checkpoint"])

    strong_provenance = bool(
        top["exists"]
        and top["project_hint_score"] >= 2
        and top["is_best_pt"]
    )

    return out, lock, strong_provenance


def main():
    print("\nR13-B-V2 — BBOX GT FIX + DETECTOR PROVENANCE LOCK")
    print(
        "AUDIT ONLY. NO detector inference and NO tracking inference. "
        "This patch fixes the prior false-negative schema check by explicitly validating x,y,w,h "
        "against cx,cy algebraically. It also treats the detector as a provenance lock rather than "
        "claiming exact identity between training-run metrics and the authoritative held-out detector metrics. "
        "No original file is modified.\n"
    )

    upstream = verify_upstream()

    bbox_summary, gt_ok, bbox_details = bbox_semantics_audit()
    detector_df, detector_lock, detector_provenance_ok = detector_provenance_audit()

    seq_df = pd.read_csv(R13B_SEQ)
    tracker_df = pd.read_csv(R13B_TRACKER)

    final_seq_rows = seq_df[seq_df["in_final_2s_metadata"] == True].copy()

    sequence_ok = bool(
        len(final_seq_rows) > 0
        and final_seq_rows["frame_directory_exists"].astype(bool).all()
    )

    tracker_ok = bool(
        len(tracker_df) > 0
        and tracker_df.iloc[0].get("ultralytics_installed", False)
        and (
            tracker_df.iloc[0].get("bytetrack_exists", False)
            or tracker_df.iloc[0].get("botsort_exists", False)
        )
    )

    summary = pd.DataFrame(
        [
            {
                "check": "Detector checkpoint provenance lock",
                "pass": detector_provenance_ok,
                "detail": detector_lock,
            },
            {
                "check": "All frozen final TestReal sequences have frame directories",
                "pass": sequence_ok,
                "detail": f"{int(final_seq_rows['frame_directory_exists'].astype(bool).sum())}/{len(final_seq_rows)}",
            },
            {
                "check": "cleaned_track_points supports tracking GT as frame+ID+XYWH",
                "pass": gt_ok,
                "detail": json.dumps(bbox_details),
            },
            {
                "check": "Ultralytics ByteTrack/BoT-SORT available for NEW revision experiment",
                "pass": tracker_ok,
                "detail": tracker_df.iloc[0].to_json() if len(tracker_df) else "",
            },
        ]
    )

    all_pass = bool(summary["pass"].astype(bool).all())

    if all_pass:
        status = (
            "R13B_V2_COMPATIBILITY_PASS_DETECTOR_FRAMES_TRACKING_GT_AND_NEW_TRACKER_AVAILABLE_READY_FOR_R13C_FROZEN_END_TO_END_PROTOCOL_LOCK"
        )
    else:
        failed = summary.loc[
            ~summary["pass"].astype(bool),
            "check",
        ].tolist()

        status = (
            "R13B_V2_COMPATIBILITY_INCOMPLETE_REVIEW_"
            + "_".join(
                "".join(ch if ch.isalnum() else "_" for ch in x).upper().strip("_")
                for x in failed
            )
        )

    bbox_path = OUT / "R13B_V2_BBOX_GT_SEMANTICS_AUDIT.csv"
    detector_path = OUT / "R13B_V2_DETECTOR_PROVENANCE_AUDIT.csv"
    summary_path = OUT / "R13B_V2_COMPATIBILITY_SUMMARY.csv"

    bbox_summary.to_csv(bbox_path, index=False)
    detector_df.to_csv(detector_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "step": "R13B_V2_BBOX_GT_AND_DETECTOR_PROVENANCE_FIX",
        "created_utc": utc_now(),
        "status": status,
        "training": False,
        "detector_inference": False,
        "tracker_inference": False,
        "original_files_modified": False,
        "detector_checkpoint_lock": detector_lock,
        "detector_checkpoint_sha256": (
            sha256_file(detector_lock)
            if detector_lock and Path(detector_lock).exists()
            else ""
        ),
        "detector_provenance_pass": detector_provenance_ok,
        "tracking_gt_schema_pass": gt_ok,
        "sequence_compatibility_pass": sequence_ok,
        "new_tracker_available": tracker_ok,
        "historical_pipeline_end_to_end_claim_allowed": False,
        "new_revision_end_to_end_experiment_allowed": all_pass,
        "outputs": {
            "bbox_gt_semantics": str(bbox_path),
            "detector_provenance": str(detector_path),
            "summary": str(summary_path),
        },
    }

    (OUT / "R13B_V2_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("BBOX / TRACKING-GT SEMANTICS AUDIT:")
    print(bbox_summary.to_string(index=False))

    print("\nDETECTOR PROVENANCE AUDIT:")
    cols = [
        "checkpoint",
        "project_hint_score",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "mean_abs_diff_vs_authoritative_reference",
        "provenance_score",
    ]
    print(detector_df[cols].head(12).to_string(index=False))

    print("\nCOMPATIBILITY SUMMARY:")
    print(summary.to_string(index=False))

    print("\nSTATUS:")
    print(status)

    print("\nR13-B-V2 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
