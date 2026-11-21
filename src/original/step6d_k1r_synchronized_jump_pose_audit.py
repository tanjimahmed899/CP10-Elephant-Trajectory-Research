from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BAMBI_ROOT = DOWNLOADS / "step6d_uncertainty_holdout" / "bambi_flight6"
GEO_ROOT = BAMBI_ROOT / "step6dk_georef"
QC_ROOT = GEO_ROOT / "quality_review"
OUT = QC_ROOT / "synchronized_jump_audit"
OUT.mkdir(parents=True, exist_ok=True)

TRAJ = GEO_ROOT / "step6dk_valid_world_trajectory_rows.csv"
STEPS = QC_ROOT / "step6dk1q_step_motion_flags.csv"
POSES = GEO_ROOT / "correction" / "6_matched_poses.json"
GT = DOWNLOADS / "Dataset-main" / "Dataset-main" / "bambi_downloads" / "6_gt.txt"

FPS = 29.411834004179347

def wrap_deg(x):
    x = np.asarray(x, dtype=float)
    return (x + 180.0) % 360.0 - 180.0

def unit_resultant(dx, dy):
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    mag = np.sqrt(dx*dx + dy*dy)
    good = np.isfinite(mag) & (mag > 1e-12)
    if not np.any(good):
        return np.nan
    ux = dx[good] / mag[good]
    uy = dy[good] / mag[good]
    return float(np.sqrt(np.mean(ux)**2 + np.mean(uy)**2))

def load_red_deer_gt():
    rows = []
    with open(GT, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            p = [x.strip() for x in s.split(",")]
            if len(p) < 10:
                continue
            species = p[9].lower()
            if "red deer" not in species:
                continue
            try:
                frame = int(float(p[0]))
                tid = int(float(p[1]))
                x = float(p[2]); y = float(p[3])
                w = float(p[4]); h = float(p[5])
            except Exception:
                continue
            rows.append({
                "frame_id": frame,
                "track_id": tid,
                "bbox_cx_px": x + w/2.0,
                "bbox_cy_px": y + h/2.0,
                "bbox_w_px": w,
                "bbox_h_px": h,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No Red-deer GT rows parsed.")
    return df

def infer_pose_frame_id(img, fallback_index):
    candidate_keys = [
        "frame_id", "frame_idx", "frame", "image_id", "id",
        "frameIndex", "frame_number", "frameNumber"
    ]
    for k in candidate_keys:
        if k in img:
            v = img[k]
            try:
                if isinstance(v, (list, tuple)) and len(v):
                    v = v[0]
                return int(float(v))
            except Exception:
                pass

    text_keys = [
        "file_name", "filename", "name", "path", "image", "image_path"
    ]
    for k in text_keys:
        if k in img and img[k] is not None:
            txt = str(img[k])
            nums = re.findall(r"(\d+)", Path(txt).stem)
            if nums:
                try:
                    return int(nums[-1])
                except Exception:
                    pass

    # Last resort: BAMBI matched-pose arrays are normally frame ordered.
    return int(fallback_index)

def load_pose_table():
    with open(POSES, "r", encoding="utf-8") as f:
        obj = json.load(f)

    images = obj.get("images", [])
    if not images:
        raise RuntimeError("No images in enriched pose JSON.")

    rows = []
    for i, img in enumerate(images):
        fid = infer_pose_frame_id(img, i)

        loc = img.get("location")
        rot = img.get("rotation")

        if not (isinstance(loc, (list, tuple)) and len(loc) >= 3):
            continue
        if not (isinstance(rot, (list, tuple)) and len(rot) >= 3):
            continue

        rows.append({
            "frame_id": int(fid),
            "cam_x": float(loc[0]),
            "cam_y": float(loc[1]),
            "cam_z": float(loc[2]),
            "rot_0_deg": float(rot[0]),
            "rot_1_deg": float(rot[1]),
            "rot_2_deg": float(rot[2]),
        })

    df = pd.DataFrame(rows).drop_duplicates("frame_id", keep="first")
    if df.empty:
        raise RuntimeError("No usable pose rows.")
    return df

def pose_delta(pose, f0, f1):
    a = pose[pose["frame_id"] == f0]
    b = pose[pose["frame_id"] == f1]

    if len(a) != 1 or len(b) != 1:
        return {
            "pose_available": False,
            "cam_translation_m": np.nan,
            "rot_delta_norm_deg": np.nan,
            "rot0_delta_deg": np.nan,
            "rot1_delta_deg": np.nan,
            "rot2_delta_deg": np.nan,
        }

    a = a.iloc[0]
    b = b.iloc[0]

    trans = np.array([
        b["cam_x"] - a["cam_x"],
        b["cam_y"] - a["cam_y"],
        b["cam_z"] - a["cam_z"],
    ], dtype=float)

    rdelta = wrap_deg(np.array([
        b["rot_0_deg"] - a["rot_0_deg"],
        b["rot_1_deg"] - a["rot_1_deg"],
        b["rot_2_deg"] - a["rot_2_deg"],
    ], dtype=float))

    return {
        "pose_available": True,
        "cam_translation_m": float(np.linalg.norm(trans)),
        "rot_delta_norm_deg": float(np.linalg.norm(rdelta)),
        "rot0_delta_deg": float(rdelta[0]),
        "rot1_delta_deg": float(rdelta[1]),
        "rot2_delta_deg": float(rdelta[2]),
    }

def raw_image_transition(gt, f0, f1):
    a = gt[gt["frame_id"] == f0][
        ["track_id", "bbox_cx_px", "bbox_cy_px"]
    ].copy()
    b = gt[gt["frame_id"] == f1][
        ["track_id", "bbox_cx_px", "bbox_cy_px"]
    ].copy()

    m = a.merge(b, on="track_id", suffixes=("_0", "_1"))
    if m.empty:
        return {
            "shared_raw_tracks": 0,
            "median_raw_shift_px": np.nan,
            "p95_raw_shift_px": np.nan,
            "raw_direction_coherence": np.nan,
        }

    dx = m["bbox_cx_px_1"].to_numpy(float) - m["bbox_cx_px_0"].to_numpy(float)
    dy = m["bbox_cy_px_1"].to_numpy(float) - m["bbox_cy_px_0"].to_numpy(float)
    mag = np.sqrt(dx*dx + dy*dy)

    return {
        "shared_raw_tracks": int(len(m)),
        "median_raw_shift_px": float(np.median(mag)),
        "p95_raw_shift_px": float(np.percentile(mag, 95)),
        "raw_direction_coherence": unit_resultant(dx, dy),
    }

def world_transition(traj, f0, f1):
    a = traj[traj["frame_id"] == f0][
        ["track_id", "utm_x_center", "utm_y_center"]
    ].copy()
    b = traj[traj["frame_id"] == f1][
        ["track_id", "utm_x_center", "utm_y_center"]
    ].copy()

    m = a.merge(b, on="track_id", suffixes=("_0", "_1"))
    if m.empty:
        return {
            "shared_world_tracks": 0,
            "median_world_shift_m": np.nan,
            "p95_world_shift_m": np.nan,
            "world_direction_coherence": np.nan,
            "median_world_speed_mps": np.nan,
        }

    dx = m["utm_x_center_1"].to_numpy(float) - m["utm_x_center_0"].to_numpy(float)
    dy = m["utm_y_center_1"].to_numpy(float) - m["utm_y_center_0"].to_numpy(float)
    mag = np.sqrt(dx*dx + dy*dy)
    dt = (f1 - f0) / FPS

    return {
        "shared_world_tracks": int(len(m)),
        "median_world_shift_m": float(np.median(mag)),
        "p95_world_shift_m": float(np.percentile(mag, 95)),
        "world_direction_coherence": unit_resultant(dx, dy),
        "median_world_speed_mps": float(np.median(mag / dt)) if dt > 0 else np.nan,
    }

def local_pose_baseline(pose, f0, f1, radius=5):
    # Adjacent-pose deltas in a local neighborhood, excluding the target transition.
    rows = []
    lo = max(int(f0 - radius), 0)
    hi = int(f1 + radius)

    frame_set = set(pose["frame_id"].astype(int).tolist())

    for a in range(lo, hi):
        b = a + 1
        if a == f0 and b == f1:
            continue
        if a not in frame_set or b not in frame_set:
            continue
        d = pose_delta(pose, a, b)
        if d["pose_available"]:
            rows.append(d)

    if not rows:
        return {
            "local_pose_pairs": 0,
            "local_median_cam_translation_m": np.nan,
            "local_p95_cam_translation_m": np.nan,
            "local_median_rot_delta_norm_deg": np.nan,
            "local_p95_rot_delta_norm_deg": np.nan,
        }

    q = pd.DataFrame(rows)
    return {
        "local_pose_pairs": int(len(q)),
        "local_median_cam_translation_m": float(q["cam_translation_m"].median()),
        "local_p95_cam_translation_m": float(q["cam_translation_m"].quantile(0.95)),
        "local_median_rot_delta_norm_deg": float(q["rot_delta_norm_deg"].median()),
        "local_p95_rot_delta_norm_deg": float(q["rot_delta_norm_deg"].quantile(0.95)),
    }

def main():
    print("\nSTEP 6D-K1R — SYNCHRONIZED JUMP / CAMERA-POSE DISCONTINUITY AUDIT")
    print("No filtering. No resampling. No forecasting.\n")

    for p in [TRAJ, STEPS, POSES, GT]:
        if not p.exists():
            raise FileNotFoundError(p)

    traj = pd.read_csv(TRAJ)
    steps = pd.read_csv(STEPS)
    gt = load_red_deer_gt()
    pose = load_pose_table()

    if "robust_jump_flag_recomputed" not in steps.columns:
        raise RuntimeError("Missing robust_jump_flag_recomputed in K1Q steps.")

    jumps = steps[steps["robust_jump_flag_recomputed"].astype(bool)].copy()

    # Transition-level robust-jump concentration.
    jump_events = (
        jumps.groupby(["frame_from", "frame_to"], as_index=False)
        .agg(
            robust_jump_tracks=("track_id", "nunique"),
            robust_jump_steps=("track_id", "size"),
            median_flagged_speed_mps=("speed_xy_mps", "median"),
            max_flagged_speed_mps=("speed_xy_mps", "max"),
        )
        .sort_values(
            ["robust_jump_tracks", "max_flagged_speed_mps"],
            ascending=False
        )
    )

    # Evaluate every unique robust-jump transition in camera, raw-image, and world space.
    event_rows = []
    for _, e in jump_events.iterrows():
        f0 = int(e["frame_from"])
        f1 = int(e["frame_to"])

        row = {
            "frame_from": f0,
            "frame_to": f1,
            "frame_gap": int(f1 - f0),
            "dt_sec": float((f1 - f0) / FPS),
            "robust_jump_tracks": int(e["robust_jump_tracks"]),
            "robust_jump_steps": int(e["robust_jump_steps"]),
            "median_flagged_speed_mps": float(e["median_flagged_speed_mps"]),
            "max_flagged_speed_mps": float(e["max_flagged_speed_mps"]),
        }

        row.update(pose_delta(pose, f0, f1))
        row.update(local_pose_baseline(pose, f0, f1))
        row.update(raw_image_transition(gt, f0, f1))
        row.update(world_transition(traj, f0, f1))

        # Diagnostic ratios; descriptive only.
        row["cam_translation_vs_local_median"] = (
            row["cam_translation_m"] / row["local_median_cam_translation_m"]
            if row["pose_available"]
            and np.isfinite(row["local_median_cam_translation_m"])
            and row["local_median_cam_translation_m"] > 0
            else np.nan
        )

        row["rotation_vs_local_median"] = (
            row["rot_delta_norm_deg"] / row["local_median_rot_delta_norm_deg"]
            if row["pose_available"]
            and np.isfinite(row["local_median_rot_delta_norm_deg"])
            and row["local_median_rot_delta_norm_deg"] > 0
            else np.nan
        )

        event_rows.append(row)

    events = pd.DataFrame(event_rows)

    # Major synchronized events: descriptive flag only.
    # We do not delete anything here.
    if len(events):
        events["synchronized_multi_track_event"] = (
            events["robust_jump_tracks"] >= 3
        )
    else:
        events["synchronized_multi_track_event"] = []

    # Comprehensive consecutive-frame transition table wherever >=2 tracks are shared.
    # This supplies context so the suspect events can be compared against ordinary transitions.
    frames = sorted(set(traj["frame_id"].astype(int).tolist()))
    transition_rows = []

    frame_set = set(frames)
    for f0 in frames:
        f1 = f0 + 1
        if f1 not in frame_set:
            continue

        w = world_transition(traj, f0, f1)
        if w["shared_world_tracks"] < 2:
            continue

        r = raw_image_transition(gt, f0, f1)
        p = pose_delta(pose, f0, f1)

        rg = jumps[
            (jumps["frame_from"].astype(int) == f0) &
            (jumps["frame_to"].astype(int) == f1)
        ]

        transition_rows.append({
            "frame_from": int(f0),
            "frame_to": int(f1),
            "shared_world_tracks": w["shared_world_tracks"],
            "median_world_shift_m": w["median_world_shift_m"],
            "p95_world_shift_m": w["p95_world_shift_m"],
            "world_direction_coherence": w["world_direction_coherence"],
            "median_world_speed_mps": w["median_world_speed_mps"],
            "shared_raw_tracks": r["shared_raw_tracks"],
            "median_raw_shift_px": r["median_raw_shift_px"],
            "p95_raw_shift_px": r["p95_raw_shift_px"],
            "raw_direction_coherence": r["raw_direction_coherence"],
            "cam_translation_m": p["cam_translation_m"],
            "rot_delta_norm_deg": p["rot_delta_norm_deg"],
            "robust_jump_tracks": int(rg["track_id"].nunique()),
        })

    all_consecutive = pd.DataFrame(transition_rows)

    if len(all_consecutive):
        # Context ranks only; no filtering threshold.
        all_consecutive["world_shift_percentile"] = (
            all_consecutive["median_world_shift_m"].rank(pct=True)
        )
        all_consecutive["pose_translation_percentile"] = (
            all_consecutive["cam_translation_m"].rank(pct=True)
        )
        all_consecutive["pose_rotation_percentile"] = (
            all_consecutive["rot_delta_norm_deg"].rank(pct=True)
        )

        suspect_lookup = set(
            zip(
                events.loc[
                    events["synchronized_multi_track_event"],
                    "frame_from"
                ].astype(int),
                events.loc[
                    events["synchronized_multi_track_event"],
                    "frame_to"
                ].astype(int),
            )
        )

        all_consecutive["is_synchronized_jump_event"] = [
            (int(a), int(b)) in suspect_lookup
            for a, b in zip(
                all_consecutive["frame_from"],
                all_consecutive["frame_to"]
            )
        ]

    # Track/frame evidence around major events.
    detail_parts = []
    if len(events):
        major = events[events["synchronized_multi_track_event"]]
        for _, e in major.iterrows():
            f0 = int(e["frame_from"])
            f1 = int(e["frame_to"])

            tg = traj[
                traj["frame_id"].between(f0 - 3, f1 + 3)
            ].copy()
            tg["audit_event_from"] = f0
            tg["audit_event_to"] = f1
            detail_parts.append(tg)

    detail = (
        pd.concat(detail_parts, ignore_index=True)
        if detail_parts else pd.DataFrame()
    )

    # Candidate segmentation breakpoints are only PROPOSED from synchronized events.
    # They are not applied in this step.
    proposed_breaks = []
    if len(events):
        for _, e in events[events["synchronized_multi_track_event"]].iterrows():
            proposed_breaks.append({
                "break_before_frame": int(e["frame_to"]),
                "event_frame_from": int(e["frame_from"]),
                "event_frame_to": int(e["frame_to"]),
                "robust_jump_tracks": int(e["robust_jump_tracks"]),
                "reason": "synchronized_multi_track_world_jump; candidate global pose/projection discontinuity",
                "applied": False,
            })

    breaks = pd.DataFrame(proposed_breaks)

    # Save
    jump_events.to_csv(
        OUT / "step6dk1r_robust_jump_transition_counts.csv",
        index=False
    )
    events.to_csv(
        OUT / "TABLE_step6dk1r_synchronized_event_audit.csv",
        index=False
    )
    all_consecutive.to_csv(
        OUT / "step6dk1r_all_consecutive_transition_context.csv",
        index=False
    )
    detail.to_csv(
        OUT / "step6dk1r_major_event_neighborhood_world_rows.csv",
        index=False
    )
    breaks.to_csv(
        OUT / "step6dk1r_proposed_segmentation_breaks_NOT_APPLIED.csv",
        index=False
    )

    # Human-readable evidence summary.
    verdict = "NO_SYNCHRONIZED_MULTI_TRACK_EVENTS"
    if len(events) and events["synchronized_multi_track_event"].any():
        verdict = "SYNCHRONIZED_GLOBAL_DISCONTINUITY_EVIDENCE_FOUND"

    summary = {
        "step": "STEP_6D_K1R_SYNCHRONIZED_JUMP_CAMERA_POSE_AUDIT",
        "robust_jump_steps": int(len(jumps)),
        "unique_robust_jump_transitions": int(len(jump_events)),
        "synchronized_events_ge_3_tracks": int(
            events["synchronized_multi_track_event"].sum()
        ) if len(events) else 0,
        "verdict": verdict,
        "proposed_breaks_applied": False,
        "important_notes": [
            "No animal-speed cutoff is used to delete detections.",
            "Synchronized jumps across multiple animals at the same frame transition are evaluated as possible camera-pose/projection artefacts.",
            "Raw-image box motion, camera-pose deltas, and projected-world motion are reported together.",
            "Any segmentation breakpoint must be explicitly locked in a later step before 10 Hz interpolation.",
            "The Flight-6 working pose uses the BAMBI reference 50-degree FOV assumption."
        ],
    }

    with open(OUT / "STEP6D_K1R_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "STEP6D_K1R_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("STEP 6D-K1R — SYNCHRONIZED JUMP / CAMERA-POSE DISCONTINUITY AUDIT\n")
        f.write("=" * 82 + "\n\n")
        f.write("ROBUST-JUMP TRANSITION COUNTS\n")
        f.write("-----------------------------\n")
        f.write(jump_events.to_string(index=False))
        f.write("\n\nSYNCHRONIZED EVENT AUDIT\n")
        f.write("------------------------\n")
        f.write(events.to_string(index=False))
        f.write("\n\nPROPOSED SEGMENTATION BREAKS — NOT APPLIED\n")
        f.write("------------------------------------------\n")
        f.write(breaks.to_string(index=False) if len(breaks) else "NONE")
        f.write("\n\nVERDICT\n-------\n")
        f.write(verdict + "\n")

    print("ROBUST-JUMP TRANSITION COUNTS:")
    print(jump_events.to_string(index=False))

    print("\nSYNCHRONIZED EVENT AUDIT:")
    print(events.to_string(index=False))

    print("\nPROPOSED SEGMENTATION BREAKS — NOT APPLIED:")
    print(breaks.to_string(index=False) if len(breaks) else "NONE")

    print("\nVERDICT:")
    print(verdict)

    print("\nSTEP 6D-K1R COMPLETE")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
