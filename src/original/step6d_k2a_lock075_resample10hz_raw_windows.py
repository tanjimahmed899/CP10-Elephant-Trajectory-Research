from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")

ROOT = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

PRE = ROOT / "preprocessing_lock"

TRAJ_IN = (
    PRE
    / "step6dk1s_world_trajectory_with_locked_global_segments.csv"
)

K1S_LOCK = (
    PRE
    / "STEP6D_K1S_PREPROCESSING_LOCK.json"
)

OUT = ROOT / "k2a_10hz"

FINAL_LOCK = (
    PRE
    / "STEP6D_K2A_FINAL_PREPROCESSING_LOCK.json"
)

FPS = 29.411834004179347
TARGET_HZ = 10.0
DT = 1.0 / TARGET_HZ

LOCKED_BREAK_BEFORE_FRAMES = [1035, 3093]
LOCKED_MAX_INTERPOLATION_GAP_SEC = 0.75

HISTORY_STEPS = 20
FUTURE_STEPS = {
    "2s_to_2s": 20,
    "2s_to_5s": 50,
}

SOURCE_GROUP = "BAMBI_Flight6"
SPECIES = "Cervus elaphus (Red deer)"


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


def split_by_gap(g, max_gap_sec):
    g = g.sort_values("frame_id").copy()

    frames = g["frame_id"].astype(int).to_numpy()
    times = frames / FPS

    if len(g) == 0:
        return []

    local_seg = np.zeros(len(g), dtype=int)

    if len(g) > 1:
        dt = np.diff(times)
        cuts = dt > (max_gap_sec + 1e-12)
        local_seg[1:] = np.cumsum(cuts)

    g["gap_segment_id"] = local_seg

    return [
        q.copy()
        for _, q in g.groupby("gap_segment_id", sort=True)
    ]


def build_10hz_grid(t0, t1):
    # Global 0.1-s clock, no extrapolation.
    start_tick = int(np.ceil((t0 - 1e-12) * TARGET_HZ))
    end_tick = int(np.floor((t1 + 1e-12) * TARGET_HZ))

    if end_tick < start_tick:
        return np.array([], dtype=float)

    ticks = np.arange(
        start_tick,
        end_tick + 1,
        dtype=int
    )

    return ticks.astype(float) / TARGET_HZ


def bracket_gap_for_times(obs_t, grid_t):
    """
    For each grid time, report the observed timestamp gap of the interpolation
    bracket containing it. This verifies no grid point was produced across an
    observed gap > locked threshold.
    """
    obs_t = np.asarray(obs_t, dtype=float)
    grid_t = np.asarray(grid_t, dtype=float)

    out = np.full(len(grid_t), np.nan, dtype=float)

    if len(obs_t) < 2 or len(grid_t) == 0:
        return out

    idx_right = np.searchsorted(
        obs_t,
        grid_t,
        side="left"
    )

    for i, (t, r) in enumerate(zip(grid_t, idx_right)):
        # Exact observed timestamp.
        if r < len(obs_t) and abs(obs_t[r] - t) <= 1e-9:
            out[i] = 0.0
            continue

        if r == 0 or r >= len(obs_t):
            continue

        l = r - 1
        out[i] = obs_t[r] - obs_t[l]

    return out


def interpolate_block(g, track_id, global_seg, gap_seg):
    g = g.sort_values("frame_id").copy()

    frames = g["frame_id"].astype(int).to_numpy()
    t = frames / FPS

    if len(g) < 2:
        return pd.DataFrame()

    grid_t = build_10hz_grid(
        float(t.min()),
        float(t.max())
    )

    if len(grid_t) == 0:
        return pd.DataFrame()

    x = pd.to_numeric(
        g["utm_x_center"],
        errors="coerce"
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        g["utm_y_center"],
        errors="coerce"
    ).to_numpy(dtype=float)

    z = pd.to_numeric(
        g["utm_z_center"],
        errors="coerce"
    ).to_numpy(dtype=float)

    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(z))
    ):
        raise RuntimeError(
            f"Nonfinite coordinate in track {track_id}"
        )

    gx = np.interp(grid_t, t, x)
    gy = np.interp(grid_t, t, y)
    gz = np.interp(grid_t, t, z)

    bracket_gap = bracket_gap_for_times(
        t,
        grid_t
    )

    if np.nanmax(bracket_gap) > LOCKED_MAX_INTERPOLATION_GAP_SEC + 1e-9:
        raise RuntimeError(
            "Interpolation crossed the locked max gap."
        )

    block_key = (
        f"BAMBI6_T{int(track_id):03d}"
        f"_G{int(global_seg)}"
        f"_B{int(gap_seg):03d}"
    )

    out = pd.DataFrame({
        "source_group": SOURCE_GROUP,
        "species": SPECIES,
        "track_id": int(track_id),
        "global_segment_id": int(global_seg),
        "gap_segment_id": int(gap_seg),
        "block_key": block_key,
        "time_sec": grid_t,
        "utm_x": gx,
        "utm_y": gy,
        "utm_z": gz,
        "interp_bracket_gap_sec": bracket_gap,
    })

    return out


def make_windows(resampled, task, future_steps):
    xs = []
    ys = []
    meta_rows = []

    total_steps = HISTORY_STEPS + future_steps

    sample_index = 0

    for block_key, g in resampled.groupby(
        "block_key",
        sort=True
    ):
        g = g.sort_values("time_sec").reset_index(drop=True)

        arr = g[["utm_x", "utm_y"]].to_numpy(dtype=np.float64)
        t = g["time_sec"].to_numpy(dtype=float)

        if len(g) < total_steps:
            continue

        # Strictly verify 10 Hz.
        if len(t) > 1:
            dt = np.diff(t)
            if not np.allclose(
                dt,
                DT,
                atol=1e-9,
                rtol=0
            ):
                raise RuntimeError(
                    f"Non-10Hz grid in block {block_key}"
                )

        for start in range(
            0,
            len(g) - total_steps + 1
        ):
            h0 = start
            h1 = start + HISTORY_STEPS
            f1 = h1 + future_steps

            x = arr[h0:h1]
            y = arr[h1:f1]

            xs.append(x)
            ys.append(y)

            first = g.iloc[h0]
            hist_last = g.iloc[h1 - 1]
            fut_first = g.iloc[h1]
            fut_last = g.iloc[f1 - 1]

            meta_rows.append({
                "sample_index": int(sample_index),
                "dataset": "BAMBI_Flight6",
                "task": task,
                "source_group": SOURCE_GROUP,
                "species": SPECIES,
                "track_id": int(first["track_id"]),
                "track_key": str(block_key),
                "global_segment_id": int(first["global_segment_id"]),
                "gap_segment_id": int(first["gap_segment_id"]),
                "window_start_time_sec": float(first["time_sec"]),
                "history_end_time_sec": float(hist_last["time_sec"]),
                "future_start_time_sec": float(fut_first["time_sec"]),
                "future_end_time_sec": float(fut_last["time_sec"]),
                "max_interp_bracket_gap_sec_in_window": float(
                    g.iloc[h0:f1]["interp_bracket_gap_sec"].max()
                ),
            })

            sample_index += 1

    if xs:
        X = np.stack(xs).astype(np.float32)
        Y = np.stack(ys).astype(np.float32)
    else:
        X = np.empty(
            (0, HISTORY_STEPS, 2),
            dtype=np.float32
        )
        Y = np.empty(
            (0, future_steps, 2),
            dtype=np.float32
        )

    meta = pd.DataFrame(meta_rows)

    return X, Y, meta


def motion_qc(resampled):
    rows = []

    for block_key, g in resampled.groupby("block_key"):
        g = g.sort_values("time_sec")

        if len(g) < 2:
            continue

        x = g["utm_x"].to_numpy(float)
        y = g["utm_y"].to_numpy(float)
        t = g["time_sec"].to_numpy(float)

        dt = np.diff(t)
        d = np.sqrt(
            np.diff(x) ** 2
            +
            np.diff(y) ** 2
        )

        speed = d / dt

        for i in range(len(speed)):
            rows.append({
                "block_key": block_key,
                "track_id": int(g.iloc[0]["track_id"]),
                "global_segment_id": int(g.iloc[0]["global_segment_id"]),
                "gap_segment_id": int(g.iloc[0]["gap_segment_id"]),
                "time_from_sec": float(t[i]),
                "time_to_sec": float(t[i + 1]),
                "speed_xy_mps": float(speed[i]),
            })

    q = pd.DataFrame(rows)

    if len(q) == 0:
        return q, {}

    s = q["speed_xy_mps"]

    summary = {
        "steps": int(len(q)),
        "median_speed_xy_mps": float(s.median()),
        "p95_speed_xy_mps": float(s.quantile(0.95)),
        "p99_speed_xy_mps": float(s.quantile(0.99)),
        "max_speed_xy_mps": float(s.max()),
    }

    return q, summary


def main():
    print(
        "\nSTEP 6D-K2A — LOCK 0.75 s GAP + 10 Hz RESAMPLING + RAW WINDOWS"
    )
    print(
        "No canonicalization. No forecasting. No conformal evaluation.\n"
    )

    require(TRAJ_IN)
    require(K1S_LOCK)

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        K1S_LOCK,
        "r",
        encoding="utf-8"
    ) as f:
        k1s = json.load(f)

    if k1s.get(
        "locked_global_break_before_frames"
    ) != LOCKED_BREAK_BEFORE_FRAMES:
        raise RuntimeError(
            "K1S breakpoint lock mismatch."
        )

    traj = pd.read_csv(
        TRAJ_IN
    )

    required_cols = [
        "frame_id",
        "track_id",
        "global_segment_id",
        "utm_x_center",
        "utm_y_center",
        "utm_z_center",
    ]

    missing = [
        c
        for c in required_cols
        if c not in traj.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    # ------------------------------------------------------------
    # Final preprocessing lock
    # ------------------------------------------------------------
    lock = {
        "step":
            "STEP_6D_K2A_FINAL_PREPROCESSING_LOCK",

        "status":
            "LOCKED_BEFORE_BAMBI_FORECASTING",

        "flight_id":
            6,

        "species":
            SPECIES,

        "fps":
            FPS,

        "target_hz":
            TARGET_HZ,

        "locked_global_break_before_frames":
            LOCKED_BREAK_BEFORE_FRAMES,

        "locked_max_interpolation_gap_sec":
            LOCKED_MAX_INTERPOLATION_GAP_SEC,

        "selection_rationale":
            (
                "0.75 s and 1.00 s produced identical geometry-only feasibility, "
                "while 0.75 s is stricter; 0.50 s substantially reduced usable "
                "especially 2s-to-5s blocks/windows. 0.75 s covers 99.94% of "
                "post-breakpoint observed adjacent gaps in K1S."
            ),

        "history_steps":
            HISTORY_STEPS,

        "future_steps":
            FUTURE_STEPS,

        "interpolation":
            "linear in UTM x/y/z on an absolute 10 Hz clock; no extrapolation",

        "window_stride":
            "1 resampled point = 0.1 s",

        "no_model_predictions_or_errors_used_for_selection":
            True,

        "source_trajectory_sha256":
            sha256_file(
                TRAJ_IN
            ),

        "important_rules": [
            "No interpolation may bridge a locked global breakpoint.",
            "No interpolation may bridge an observed gap > 0.75 s.",
            "No animal-speed cutoff is used for deletion.",
            "No BAMBI forecasting/conformal results were inspected before this preprocessing lock.",
            "The BAMBI working georeferencing uses the reference 50-degree FOV assumption.",
        ],
    }

    if FINAL_LOCK.exists():
        with open(
            FINAL_LOCK,
            "r",
            encoding="utf-8"
        ) as f:
            old = json.load(f)

        if old != lock:
            raise RuntimeError(
                "Existing K2A lock differs; refusing overwrite."
            )

        lock_status = "EXISTING_LOCK_VERIFIED"

    else:
        with open(
            FINAL_LOCK,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                lock,
                f,
                indent=2
            )

        lock_status = "NEW_FINAL_PREPROCESSING_LOCK_CREATED"

    print(
        "PREPROCESSING LOCK STATUS:",
        lock_status
    )

    # ------------------------------------------------------------
    # Split by locked 0.75-s observed gap and resample each block.
    # ------------------------------------------------------------
    resampled_parts = []
    block_audit = []

    for (
        track_id,
        global_seg
    ), g in traj.groupby(
        [
            "track_id",
            "global_segment_id"
        ],
        sort=True
    ):
        blocks = split_by_gap(
            g,
            LOCKED_MAX_INTERPOLATION_GAP_SEC
        )

        for local_id, block in enumerate(blocks):
            block = block.copy()
            block["gap_segment_id"] = int(local_id)

            r = interpolate_block(
                block,
                track_id,
                global_seg,
                local_id
            )

            observed_frames = block[
                "frame_id"
            ].astype(int)

            observed_times = (
                observed_frames.to_numpy()
                /
                FPS
            )

            block_audit.append({
                "track_id":
                    int(track_id),

                "global_segment_id":
                    int(global_seg),

                "gap_segment_id":
                    int(local_id),

                "observed_rows":
                    int(len(block)),

                "observed_frame_min":
                    int(
                        observed_frames.min()
                    ),

                "observed_frame_max":
                    int(
                        observed_frames.max()
                    ),

                "observed_time_min_sec":
                    float(
                        observed_times.min()
                    ),

                "observed_time_max_sec":
                    float(
                        observed_times.max()
                    ),

                "observed_duration_sec":
                    float(
                        observed_times.max()
                        -
                        observed_times.min()
                    ),

                "resampled_points_10hz":
                    int(len(r)),

                "max_interp_bracket_gap_sec":
                    (
                        float(
                            r[
                                "interp_bracket_gap_sec"
                            ].max()
                        )
                        if len(r)
                        else np.nan
                    ),
            })

            if len(r):
                resampled_parts.append(r)

    resampled = (
        pd.concat(
            resampled_parts,
            ignore_index=True
        )
        if resampled_parts
        else pd.DataFrame()
    )

    block_audit_df = pd.DataFrame(
        block_audit
    )

    if len(resampled) == 0:
        raise RuntimeError(
            "No 10 Hz resampled points produced."
        )

    # Verify exact 10 Hz within blocks.
    grid_failures = 0

    for block_key, g in resampled.groupby(
        "block_key"
    ):
        t = np.sort(
            g["time_sec"].to_numpy(float)
        )

        if len(t) > 1:
            if not np.allclose(
                np.diff(t),
                DT,
                atol=1e-9,
                rtol=0
            ):
                grid_failures += 1

    if grid_failures:
        raise RuntimeError(
            f"{grid_failures} blocks failed exact 10 Hz QC."
        )

    if (
        resampled[
            "interp_bracket_gap_sec"
        ].max()
        >
        LOCKED_MAX_INTERPOLATION_GAP_SEC
        +
        1e-9
    ):
        raise RuntimeError(
            "Resampled data crosses locked max gap."
        )

    resampled.to_csv(
        OUT
        / "step6dk2a_resampled_10hz_world_trajectory.csv",
        index=False
    )

    block_audit_df.to_csv(
        OUT
        / "step6dk2a_block_audit.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Raw absolute-coordinate windows
    # ------------------------------------------------------------
    task_summaries = {}

    for task, future_steps in FUTURE_STEPS.items():
        X, Y, meta = make_windows(
            resampled,
            task,
            future_steps
        )

        np.savez_compressed(
            OUT
            / f"{task}_bambi_flight6_world_raw.npz",
            X_abs_xy=X,
            Y_abs_xy=Y,
        )

        meta.to_csv(
            OUT
            / f"{task}_bambi_flight6_metadata.csv",
            index=False
        )

        task_summaries[task] = {
            "windows":
                int(len(X)),

            "unique_tracks":
                int(
                    meta[
                        "track_id"
                    ].nunique()
                )
                if len(meta)
                else 0,

            "unique_blocks":
                int(
                    meta[
                        "track_key"
                    ].nunique()
                )
                if len(meta)
                else 0,

            "X_shape":
                list(
                    X.shape
                ),

            "Y_shape":
                list(
                    Y.shape
                ),

            "max_interp_bracket_gap_sec":
                (
                    float(
                        meta[
                            "max_interp_bracket_gap_sec_in_window"
                        ].max()
                    )
                    if len(meta)
                    else np.nan
                ),
        }

    # ------------------------------------------------------------
    # Resampled motion QC
    # ------------------------------------------------------------
    motion_df, motion_summary = motion_qc(
        resampled
    )

    motion_df.to_csv(
        OUT
        / "step6dk2a_resampled_motion_qc.csv",
        index=False
    )

    overall = {
        "resampled_points_10hz":
            int(
                len(
                    resampled
                )
            ),

        "resampled_tracks":
            int(
                resampled[
                    "track_id"
                ].nunique()
            ),

        "resampled_blocks":
            int(
                resampled[
                    "block_key"
                ].nunique()
            ),

        "max_interp_bracket_gap_sec":
            float(
                resampled[
                    "interp_bracket_gap_sec"
                ].max()
            ),

        "exact_10hz_grid_failures":
            int(
                grid_failures
            ),
    }

    summary = {
        "step":
            "STEP_6D_K2A_10HZ_RESAMPLING_RAW_WINDOW_CONSTRUCTION",

        "lock_status":
            lock_status,

        "preprocessing_lock":
            lock,

        "overall":
            overall,

        "tasks":
            task_summaries,

        "resampled_motion_qc":
            motion_summary,

        "status":
            "K2A_RESAMPLING_AND_RAW_WINDOWS_COMPLETE_PENDING_QC_REVIEW",
    }

    with open(
        OUT
        / "STEP6D_K2A_SUMMARY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    with open(
        OUT
        / "STEP6D_K2A_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K2A — 10 Hz RESAMPLING + RAW WINDOW CONSTRUCTION\n"
        )

        f.write(
            "="
            *
            82
            +
            "\n\n"
        )

        f.write(
            "FINAL PREPROCESSING LOCK\n"
        )

        f.write(
            "------------------------\n"
        )

        f.write(
            json.dumps(
                {
                    "locked_global_break_before_frames":
                        LOCKED_BREAK_BEFORE_FRAMES,

                    "locked_max_interpolation_gap_sec":
                        LOCKED_MAX_INTERPOLATION_GAP_SEC,

                    "target_hz":
                        TARGET_HZ,
                },
                indent=2
            )
        )

        f.write(
            "\n\nRESAMPLING SUMMARY\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            json.dumps(
                overall,
                indent=2
            )
        )

        f.write(
            "\n\nRAW WINDOW SUMMARY\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            json.dumps(
                task_summaries,
                indent=2
            )
        )

        f.write(
            "\n\nRESAMPLED MOTION QC\n"
        )

        f.write(
            "-------------------\n"
        )

        f.write(
            json.dumps(
                motion_summary,
                indent=2
            )
        )

        f.write(
            "\n\nSTATUS\n"
        )

        f.write(
            "------\n"
        )

        f.write(
            summary[
                "status"
            ]
            +
            "\n"
        )

        f.write(
            "\nNo canonicalization, forecasting, conformal coverage, "
            "or Step-7 warning evaluation was run.\n"
        )

    print(
        "\nRESAMPLING SUMMARY:"
    )

    print(
        json.dumps(
            overall,
            indent=2
        )
    )

    print(
        "\nRAW WINDOW SUMMARY:"
    )

    print(
        json.dumps(
            task_summaries,
            indent=2
        )
    )

    print(
        "\nRESAMPLED MOTION QC:"
    )

    print(
        json.dumps(
            motion_summary,
            indent=2
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        summary[
            "status"
        ]
    )

    print(
        "\nSTEP 6D-K2A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
