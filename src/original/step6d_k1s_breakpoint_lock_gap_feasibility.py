from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
ROOT = DOWNLOADS / "step6d_uncertainty_holdout" / "bambi_flight6" / "step6dk_georef"
TRAJ = ROOT / "step6dk_valid_world_trajectory_rows.csv"
AUDIT_DIR = ROOT / "quality_review" / "synchronized_jump_audit"
EVENTS = AUDIT_DIR / "TABLE_step6dk1r_synchronized_event_audit.csv"

OUT = ROOT / "preprocessing_lock"
OUT.mkdir(parents=True, exist_ok=True)

FPS = 29.411834004179347
TARGET_HZ = 10.0

# Locked global discontinuities from K1R.
# A breakpoint "before frame 1035" means no interpolation or trajectory block
# is allowed to bridge from <1035 to >=1035.
LOCKED_BREAK_BEFORE_FRAMES = [1035, 3093]

# Geometry-only candidate interpolation-gap thresholds.
# No forecasting predictions/errors/coverage are used here.
CANDIDATE_MAX_GAP_SEC = [0.25, 0.50, 0.75, 1.00]

HISTORY_SEC = 2.0
HORIZONS_SEC = [2.0, 5.0]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def global_segment_id(frame_id):
    f = int(frame_id)
    # Segment 0: frame < 1035
    # Segment 1: 1035 <= frame < 3093
    # Segment 2: frame >= 3093
    return int(sum(f >= b for b in LOCKED_BREAK_BEFORE_FRAMES))


def make_blocks(g, max_gap_sec):
    """
    Split one track/global-segment into blocks whenever observed timestamp gap
    exceeds max_gap_sec. No interpolation is performed.
    """
    g = g.sort_values("frame_id")
    frames = g["frame_id"].astype(int).to_numpy()

    if len(frames) == 0:
        return []

    times = frames / FPS
    block_ids = np.zeros(len(g), dtype=int)

    if len(frames) > 1:
        dt = np.diff(times)
        cuts = dt > (max_gap_sec + 1e-12)
        block_ids[1:] = np.cumsum(cuts)

    out = []
    for bid in np.unique(block_ids):
        idx = np.where(block_ids == bid)[0]
        sub = g.iloc[idx]
        f0 = int(sub["frame_id"].min())
        f1 = int(sub["frame_id"].max())
        t0 = f0 / FPS
        t1 = f1 / FPS
        duration = float(t1 - t0)
        out.append({
            "block_id_local": int(bid),
            "rows": int(len(sub)),
            "frame_min": f0,
            "frame_max": f1,
            "time_min_sec": t0,
            "time_max_sec": t1,
            "duration_sec": duration,
        })
    return out


def main():
    print("\nSTEP 6D-K1S — BREAKPOINT LOCK + INTERPOLATION-GAP FEASIBILITY AUDIT")
    print("No interpolation. No forecasting. No conformal evaluation.\n")

    if not TRAJ.exists():
        raise FileNotFoundError(TRAJ)
    if not EVENTS.exists():
        raise FileNotFoundError(EVENTS)

    traj = pd.read_csv(TRAJ)
    events = pd.read_csv(EVENTS)

    # ------------------------------------------------------------
    # Verify locked breakpoints are supported by synchronized events.
    # ------------------------------------------------------------
    required_events = {
        (1034, 1035),
        (3092, 3093),
    }

    observed_sync = set()
    for _, r in events.iterrows():
        try:
            sync = bool(r["synchronized_multi_track_event"])
        except Exception:
            sync = str(r["synchronized_multi_track_event"]).strip().lower() == "true"

        if sync:
            observed_sync.add((int(r["frame_from"]), int(r["frame_to"])))

    missing_support = sorted(required_events - observed_sync)
    if missing_support:
        raise RuntimeError(
            f"Locked breakpoint support missing from K1R: {missing_support}"
        )

    # ------------------------------------------------------------
    # Add immutable global segment label.
    # ------------------------------------------------------------
    traj = traj.copy()
    traj["global_segment_id"] = traj["frame_id"].map(global_segment_id)

    # Verify no duplicated track/frame.
    dup = int(traj.duplicated(["track_id", "frame_id"]).sum())
    if dup:
        raise RuntimeError(f"Duplicate track/frame rows found: {dup}")

    # ------------------------------------------------------------
    # Observed gap distribution AFTER global breakpoint segmentation.
    # ------------------------------------------------------------
    gap_rows = []

    for (tid, seg), g in traj.groupby(["track_id", "global_segment_id"]):
        g = g.sort_values("frame_id")
        frames = g["frame_id"].astype(int).to_numpy()

        if len(frames) < 2:
            continue

        dframe = np.diff(frames)
        dt = dframe / FPS

        for a, b, df, ds in zip(frames[:-1], frames[1:], dframe, dt):
            gap_rows.append({
                "track_id": int(tid),
                "global_segment_id": int(seg),
                "frame_from": int(a),
                "frame_to": int(b),
                "frame_gap": int(df),
                "gap_sec": float(ds),
            })

    gaps = pd.DataFrame(gap_rows)

    gap_summary = {
        "observed_gaps": int(len(gaps)),
        "median_gap_frames": float(gaps["frame_gap"].median()),
        "p75_gap_frames": float(gaps["frame_gap"].quantile(0.75)),
        "p90_gap_frames": float(gaps["frame_gap"].quantile(0.90)),
        "p95_gap_frames": float(gaps["frame_gap"].quantile(0.95)),
        "p99_gap_frames": float(gaps["frame_gap"].quantile(0.99)),
        "max_gap_frames": int(gaps["frame_gap"].max()),
        "median_gap_sec": float(gaps["gap_sec"].median()),
        "p90_gap_sec": float(gaps["gap_sec"].quantile(0.90)),
        "p95_gap_sec": float(gaps["gap_sec"].quantile(0.95)),
        "p99_gap_sec": float(gaps["gap_sec"].quantile(0.99)),
        "max_gap_sec": float(gaps["gap_sec"].max()),
    }

    # ------------------------------------------------------------
    # Candidate max-gap feasibility.
    # ------------------------------------------------------------
    block_rows = []
    summary_rows = []

    base_groups = list(
        traj.groupby(["track_id", "global_segment_id"], sort=True)
    )

    for max_gap_sec in CANDIDATE_MAX_GAP_SEC:
        candidate_blocks = []

        for (tid, seg), g in base_groups:
            blocks = make_blocks(g, max_gap_sec)

            for b in blocks:
                row = {
                    "max_gap_sec": float(max_gap_sec),
                    "track_id": int(tid),
                    "global_segment_id": int(seg),
                    **b,
                }

                # Potential exact 10 Hz grid points if later interpolated.
                # This is only a count estimate; K1S does NOT interpolate.
                row["potential_10hz_points"] = (
                    int(np.floor(b["duration_sec"] * TARGET_HZ + 1e-9)) + 1
                )

                for horizon in HORIZONS_SEC:
                    need = HISTORY_SEC + horizon
                    key = f"eligible_{int(HISTORY_SEC)}to{int(horizon)}"
                    row[key] = bool(
                        b["duration_sec"] + 1e-12 >= need
                    )

                    # Potential number of sliding windows at 10 Hz, stride 1 grid point.
                    # Observation 2 s = 20 points; future 2 s = 20, future 5 s = 50.
                    required_points = int(round((HISTORY_SEC + horizon) * TARGET_HZ))
                    row[f"potential_windows_{int(HISTORY_SEC)}to{int(horizon)}"] = max(
                        0,
                        row["potential_10hz_points"] - required_points + 1
                    )

                candidate_blocks.append(row)
                block_rows.append(row)

        bdf = pd.DataFrame(candidate_blocks)

        if len(bdf) == 0:
            continue

        s = {
            "max_gap_sec": float(max_gap_sec),
            "max_gap_frames_equivalent": float(max_gap_sec * FPS),
            "blocks": int(len(bdf)),
            "unique_tracks_with_any_block": int(bdf["track_id"].nunique()),
            "median_block_duration_sec": float(bdf["duration_sec"].median()),
            "p90_block_duration_sec": float(bdf["duration_sec"].quantile(0.90)),
            "max_block_duration_sec": float(bdf["duration_sec"].max()),
        }

        for horizon in HORIZONS_SEC:
            h = int(horizon)
            elig_col = f"eligible_{int(HISTORY_SEC)}to{h}"
            win_col = f"potential_windows_{int(HISTORY_SEC)}to{h}"

            elig = bdf[bdf[elig_col]]
            s[f"eligible_blocks_2to{h}"] = int(len(elig))
            s[f"eligible_tracks_2to{h}"] = int(elig["track_id"].nunique())
            s[f"potential_windows_2to{h}"] = int(bdf[win_col].sum())

        # Fraction of observed adjacent intervals that would be bridged.
        s["fraction_observed_gaps_within_threshold"] = float(
            (gaps["gap_sec"] <= max_gap_sec + 1e-12).mean()
        )

        summary_rows.append(s)

    block_df = pd.DataFrame(block_rows)
    feasibility = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------
    # Breakpoint lock manifest.
    # ------------------------------------------------------------
    manifest = {
        "step": "STEP_6D_K1S_PREPROCESSING_GEOMETRY_LOCK",
        "status": "GLOBAL_BREAKPOINTS_LOCKED_GAP_THRESHOLD_NOT_YET_SELECTED",
        "flight_id": 6,
        "species": "Cervus elaphus (Red deer)",
        "fps": FPS,
        "target_hz_future_step": TARGET_HZ,
        "locked_global_break_before_frames": LOCKED_BREAK_BEFORE_FRAMES,
        "locked_global_break_evidence": [
            {
                "event_frame_from": 1034,
                "event_frame_to": 1035,
                "reason": (
                    "K1R synchronized multi-track projected-world jump with highly "
                    "coherent raw-image and world motion."
                ),
            },
            {
                "event_frame_from": 3092,
                "event_frame_to": 3093,
                "reason": (
                    "K1R synchronized multi-track projected-world jump with highly "
                    "coherent raw-image and world motion."
                ),
            },
        ],
        "not_locked_as_global_break": [
            {
                "event": "1033->1035",
                "reason": "same 1035 discontinuity manifested through a 2-frame observation gap",
            },
            {
                "event": "989->990",
                "reason": "only one robust-jump track; insufficient evidence for a global breakpoint",
            },
        ],
        "candidate_max_interpolation_gap_sec": CANDIDATE_MAX_GAP_SEC,
        "gap_threshold_selection_status": "PENDING_GEOMETRY_ONLY_FEASIBILITY_REVIEW",
        "no_model_predictions_or_errors_used": True,
        "source_trajectory_sha256": sha256_file(TRAJ),
        "source_k1r_event_audit_sha256": sha256_file(EVENTS),
        "important_rules": [
            "No interpolation may bridge a locked global breakpoint.",
            "No trajectory block may cross from one global_segment_id to another.",
            "No animal-speed threshold is used to delete points.",
            "The maximum interpolation gap will be selected using geometry/continuity only, before BAMBI forecasting or conformal evaluation.",
            "K1S performs no interpolation and creates no forecasting windows.",
        ],
    }

    with open(
        OUT / "STEP6D_K1S_PREPROCESSING_LOCK.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    traj.to_csv(
        OUT / "step6dk1s_world_trajectory_with_locked_global_segments.csv",
        index=False
    )
    gaps.to_csv(
        OUT / "step6dk1s_observed_gap_distribution.csv",
        index=False
    )
    block_df.to_csv(
        OUT / "step6dk1s_candidate_gap_blocks.csv",
        index=False
    )
    feasibility.to_csv(
        OUT / "TABLE_step6dk1s_gap_feasibility.csv",
        index=False
    )

    with open(
        OUT / "STEP6D_K1S_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("STEP 6D-K1S — BREAKPOINT LOCK + INTERPOLATION-GAP FEASIBILITY AUDIT\n")
        f.write("=" * 86 + "\n\n")
        f.write("LOCKED GLOBAL BREAKPOINTS\n")
        f.write("-------------------------\n")
        f.write("before frame 1035\nbefore frame 3093\n\n")
        f.write("OBSERVED GAP SUMMARY AFTER GLOBAL SEGMENTATION\n")
        f.write("----------------------------------------------\n")
        f.write(json.dumps(gap_summary, indent=2))
        f.write("\n\nMAX-GAP FEASIBILITY\n")
        f.write("-------------------\n")
        f.write(feasibility.to_string(index=False))
        f.write("\n\nSTATUS\n")
        f.write("------\n")
        f.write(manifest["status"] + "\n")
        f.write(
            "\nNo interpolation, forecasting, conformal evaluation, or warning "
            "evaluation was run in K1S.\n"
        )

    print("LOCKED GLOBAL BREAKPOINTS:")
    print(LOCKED_BREAK_BEFORE_FRAMES)

    print("\nOBSERVED GAP SUMMARY AFTER GLOBAL SEGMENTATION:")
    print(json.dumps(gap_summary, indent=2))

    print("\nMAX-GAP FEASIBILITY:")
    print(feasibility.to_string(index=False))

    print("\nSTATUS:")
    print(manifest["status"])

    print("\nSTEP 6D-K1S COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
