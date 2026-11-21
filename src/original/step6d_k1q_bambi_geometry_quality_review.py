from pathlib import Path
import json
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
ROOT = DOWNLOADS / "step6d_uncertainty_holdout" / "bambi_flight6" / "step6dk_georef"

TRAJ = ROOT / "step6dk_valid_world_trajectory_rows.csv"
STEPS = ROOT / "step6dk_step_motion_diagnostics.csv"
TRACKS = ROOT / "step6dk_track_geometry_summary.csv"

OUT = ROOT / "quality_review"
OUT.mkdir(parents=True, exist_ok=True)

FPS = 29.411834004179347

def safe_quantile(s, q):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.quantile(q)) if len(x) else np.nan

def longest_contiguous_run(frames):
    frames = np.sort(np.unique(np.asarray(frames, dtype=int)))
    if len(frames) == 0:
        return 0
    best = cur = 1
    for a, b in zip(frames[:-1], frames[1:]):
        if b - a == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return int(best)

def main():
    print("\nSTEP 6D-K1Q — BAMBI GEOREFERENCING QUALITY REVIEW")
    print("No filtering, no resampling, no forecasting.\n")

    for p in [TRAJ, STEPS, TRACKS]:
        if not p.exists():
            raise FileNotFoundError(p)

    traj = pd.read_csv(TRAJ)
    steps = pd.read_csv(STEPS)
    tracks = pd.read_csv(TRACKS)

    # -----------------------------
    # Basic integrity
    # -----------------------------
    integrity = {
        "trajectory_rows": int(len(traj)),
        "tracks": int(traj["track_id"].nunique()),
        "frames": int(traj["frame_id"].nunique()),
        "duplicate_track_frame_rows": int(
            traj.duplicated(["track_id", "frame_id"]).sum()
        ),
        "nonfinite_xy_rows": int(
            (~np.isfinite(pd.to_numeric(traj["utm_x_center"], errors="coerce")) |
             ~np.isfinite(pd.to_numeric(traj["utm_y_center"], errors="coerce"))).sum()
        ),
    }

    # -----------------------------
    # Frame-gap / continuity audit
    # -----------------------------
    continuity_rows = []
    for tid, g in traj.groupby("track_id"):
        g = g.sort_values("frame_id")
        frames = g["frame_id"].astype(int).to_numpy()
        diffs = np.diff(frames) if len(frames) > 1 else np.array([], dtype=int)

        continuity_rows.append({
            "track_id": int(tid),
            "rows": int(len(g)),
            "frame_min": int(frames.min()),
            "frame_max": int(frames.max()),
            "temporal_span_sec": float((frames.max()-frames.min())/FPS),
            "median_frame_gap": float(np.median(diffs)) if len(diffs) else np.nan,
            "p95_frame_gap": float(np.percentile(diffs, 95)) if len(diffs) else np.nan,
            "max_frame_gap": int(np.max(diffs)) if len(diffs) else np.nan,
            "fraction_gap_eq_1": float(np.mean(diffs == 1)) if len(diffs) else np.nan,
            "longest_strict_contiguous_run_frames": longest_contiguous_run(frames),
        })

    continuity = pd.DataFrame(continuity_rows)

    # -----------------------------
    # Robust speed distribution
    # -----------------------------
    speed = pd.to_numeric(steps["speed_xy_mps"], errors="coerce")
    finite_speed = speed[np.isfinite(speed)]

    med = float(np.median(finite_speed))
    mad = float(np.median(np.abs(finite_speed - med)))
    robust_sigma = 1.4826 * mad
    robust_jump_threshold = med + 10.0 * robust_sigma if robust_sigma > 0 else safe_quantile(finite_speed, 0.995)

    # Add independent quantile flags for diagnostic context only.
    p99 = safe_quantile(finite_speed, 0.99)
    p995 = safe_quantile(finite_speed, 0.995)

    steps2 = steps.copy()
    steps2["robust_jump_flag_recomputed"] = (
        pd.to_numeric(steps2["speed_xy_mps"], errors="coerce") > robust_jump_threshold
    )
    steps2["p99_speed_flag"] = (
        pd.to_numeric(steps2["speed_xy_mps"], errors="coerce") > p99
    )
    steps2["p995_speed_flag"] = (
        pd.to_numeric(steps2["speed_xy_mps"], errors="coerce") > p995
    )

    # -----------------------------
    # Acceleration-like diagnostic
    # Only between consecutive step records in same track.
    # Not interpreted biologically; used to find isolated projection spikes.
    # -----------------------------
    accel_rows = []
    for tid, g in steps2.groupby("track_id"):
        g = g.sort_values("frame_from")
        v = pd.to_numeric(g["speed_xy_mps"], errors="coerce").to_numpy(float)
        t = pd.to_numeric(g["dt_sec"], errors="coerce").to_numpy(float)

        if len(g) < 2:
            continue

        for i in range(1, len(g)):
            dt_mid = 0.5 * (t[i-1] + t[i])
            a = abs(v[i] - v[i-1]) / dt_mid if dt_mid > 0 and np.isfinite(v[i-1]) and np.isfinite(v[i]) else np.nan
            accel_rows.append({
                "track_id": int(tid),
                "prev_frame_from": int(g.iloc[i-1]["frame_from"]),
                "frame_from": int(g.iloc[i]["frame_from"]),
                "abs_delta_speed_mps": float(abs(v[i] - v[i-1])) if np.isfinite(v[i-1]) and np.isfinite(v[i]) else np.nan,
                "approx_abs_accel_mps2": float(a) if np.isfinite(a) else np.nan,
            })

    accel = pd.DataFrame(accel_rows)

    if len(accel):
        a99 = safe_quantile(accel["approx_abs_accel_mps2"], 0.99)
        a995 = safe_quantile(accel["approx_abs_accel_mps2"], 0.995)
        accel["p99_accel_flag"] = pd.to_numeric(
            accel["approx_abs_accel_mps2"], errors="coerce"
        ) > a99
        accel["p995_accel_flag"] = pd.to_numeric(
            accel["approx_abs_accel_mps2"], errors="coerce"
        ) > a995
    else:
        a99 = a995 = np.nan

    # -----------------------------
    # Per-track suspiciousness
    # -----------------------------
    track_qc_rows = []
    for tid, g in steps2.groupby("track_id"):
        c = continuity[continuity["track_id"] == tid].iloc[0]
        speeds = pd.to_numeric(g["speed_xy_mps"], errors="coerce")

        track_qc_rows.append({
            "track_id": int(tid),
            "steps": int(len(g)),
            "rows": int(c["rows"]),
            "temporal_span_sec": float(c["temporal_span_sec"]),
            "median_frame_gap": float(c["median_frame_gap"]),
            "max_frame_gap": float(c["max_frame_gap"]),
            "longest_strict_contiguous_run_frames": int(c["longest_strict_contiguous_run_frames"]),
            "median_speed_xy_mps": float(speeds.median()),
            "p95_speed_xy_mps": float(speeds.quantile(0.95)),
            "max_speed_xy_mps": float(speeds.max()),
            "robust_jump_steps": int(g["robust_jump_flag_recomputed"].sum()),
            "p99_speed_steps": int(g["p99_speed_flag"].sum()),
            "p995_speed_steps": int(g["p995_speed_flag"].sum()),
            "robust_jump_fraction": float(g["robust_jump_flag_recomputed"].mean()),
        })

    track_qc = pd.DataFrame(track_qc_rows)

    # Rank tracks for manual review; do not exclude anything.
    track_qc["review_priority_score"] = (
        track_qc["robust_jump_steps"] * 1000
        + track_qc["p995_speed_steps"] * 100
        + track_qc["p99_speed_steps"] * 10
        + track_qc["max_speed_xy_mps"].fillna(0)
    )
    track_qc = track_qc.sort_values(
        ["review_priority_score", "max_speed_xy_mps"],
        ascending=False
    )

    # -----------------------------
    # Neighborhood export around robust jumps
    # -----------------------------
    jump_rows = steps2[steps2["robust_jump_flag_recomputed"]].copy()
    neighborhoods = []

    for _, r in jump_rows.iterrows():
        tid = int(r["track_id"])
        f0 = int(r["frame_from"])
        f1 = int(r["frame_to"])

        tg = traj[traj["track_id"] == tid].sort_values("frame_id")
        local = tg[
            (tg["frame_id"] >= f0 - 3) &
            (tg["frame_id"] <= f1 + 3)
        ].copy()

        local["jump_frame_from"] = f0
        local["jump_frame_to"] = f1
        local["jump_speed_xy_mps"] = float(r["speed_xy_mps"])
        neighborhoods.append(local)

    jump_neighborhoods = (
        pd.concat(neighborhoods, ignore_index=True)
        if neighborhoods else pd.DataFrame()
    )

    # -----------------------------
    # Summaries
    # -----------------------------
    motion_summary = {
        "steps": int(len(steps2)),
        "median_speed_xy_mps": med,
        "p95_speed_xy_mps": safe_quantile(finite_speed, 0.95),
        "p99_speed_xy_mps": p99,
        "p995_speed_xy_mps": p995,
        "max_speed_xy_mps": float(finite_speed.max()),
        "mad_speed_xy_mps": mad,
        "robust_sigma_speed_xy_mps": robust_sigma,
        "robust_jump_threshold_mps": robust_jump_threshold,
        "robust_jump_steps": int(steps2["robust_jump_flag_recomputed"].sum()),
        "tracks_with_robust_jumps": int(
            steps2.loc[
                steps2["robust_jump_flag_recomputed"], "track_id"
            ].nunique()
        ),
        "p99_accel_mps2": a99,
        "p995_accel_mps2": a995,
    }

    continuity_summary = {
        "tracks": int(len(continuity)),
        "median_rows_per_track": float(continuity["rows"].median()),
        "median_frame_gap_across_tracks": float(continuity["median_frame_gap"].median()),
        "median_max_frame_gap": float(continuity["max_frame_gap"].median()),
        "max_frame_gap_any_track": float(continuity["max_frame_gap"].max()),
        "median_longest_strict_contiguous_run_frames": float(
            continuity["longest_strict_contiguous_run_frames"].median()
        ),
        "tracks_with_longest_strict_run_ge_20": int(
            (continuity["longest_strict_contiguous_run_frames"] >= 20).sum()
        ),
        "tracks_with_longest_strict_run_ge_40": int(
            (continuity["longest_strict_contiguous_run_frames"] >= 40).sum()
        ),
        "tracks_with_longest_strict_run_ge_70": int(
            (continuity["longest_strict_contiguous_run_frames"] >= 70).sum()
        ),
    }

    # Data-quality verdict: no automatic exclusion decision.
    if integrity["duplicate_track_frame_rows"] > 0 or integrity["nonfinite_xy_rows"] > 0:
        verdict = "STRUCTURAL_ISSUE_REQUIRES_FIX"
    elif motion_summary["robust_jump_steps"] > 0:
        verdict = "GEOMETRY_VALID_BUT_OUTLIER_REVIEW_REQUIRED"
    else:
        verdict = "GEOMETRY_QC_PASSED_NO_ROBUST_JUMPS"

    # Write outputs
    continuity.to_csv(OUT / "step6dk1q_track_continuity_audit.csv", index=False)
    steps2.to_csv(OUT / "step6dk1q_step_motion_flags.csv", index=False)
    accel.to_csv(OUT / "step6dk1q_acceleration_diagnostic.csv", index=False)
    track_qc.to_csv(OUT / "TABLE_step6dk1q_track_review_priority.csv", index=False)
    jump_rows.to_csv(OUT / "TABLE_step6dk1q_robust_jump_steps.csv", index=False)
    jump_neighborhoods.to_csv(OUT / "step6dk1q_jump_neighborhood_trajectory_rows.csv", index=False)

    summary = {
        "step": "STEP_6D_K1Q_BAMBI_GEOREFERENCING_QUALITY_REVIEW",
        "integrity": integrity,
        "motion_summary": motion_summary,
        "continuity_summary": continuity_summary,
        "verdict": verdict,
        "important_interpretation": [
            "No biological speed threshold was used to delete data.",
            "Robust jump flags are distributional diagnostics only.",
            "No point or track is removed in K1Q.",
            "K2 filtering/resampling rules must be decided only after reviewing whether flagged jumps are isolated projection artefacts, temporal gaps, or sustained track motion.",
            "The 50-degree FOV remains a reference-pipeline assumption and not a Flight-6 independently calibrated intrinsic."
        ],
    }

    with open(OUT / "STEP6D_K1Q_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "STEP6D_K1Q_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("STEP 6D-K1Q — BAMBI GEOREFERENCING QUALITY REVIEW\n")
        f.write("=" * 76 + "\n\n")
        f.write("INTEGRITY\n---------\n")
        f.write(json.dumps(integrity, indent=2))
        f.write("\n\nMOTION SUMMARY\n--------------\n")
        f.write(json.dumps(motion_summary, indent=2))
        f.write("\n\nCONTINUITY SUMMARY\n------------------\n")
        f.write(json.dumps(continuity_summary, indent=2))
        f.write("\n\nTOP 15 TRACKS FOR REVIEW\n------------------------\n")
        f.write(track_qc.head(15).to_string(index=False))
        f.write("\n\nROBUST JUMP STEPS\n-----------------\n")
        f.write(jump_rows.to_string(index=False))
        f.write("\n\nVERDICT\n-------\n")
        f.write(verdict + "\n")

    print("INTEGRITY:")
    print(json.dumps(integrity, indent=2))

    print("\nMOTION SUMMARY:")
    print(json.dumps(motion_summary, indent=2))

    print("\nCONTINUITY SUMMARY:")
    print(json.dumps(continuity_summary, indent=2))

    print("\nTOP TRACKS FOR REVIEW:")
    print(track_qc.head(15).to_string(index=False))

    print("\nROBUST JUMP STEPS:")
    if len(jump_rows):
        print(jump_rows.to_string(index=False))
    else:
        print("NONE")

    print("\nVERDICT:")
    print(verdict)

    print("\nSTEP 6D-K1Q COMPLETE")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
