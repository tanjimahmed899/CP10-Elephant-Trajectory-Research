
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

STEP2B_SCRIPT = MAIN2 / "step2b_raw_vs_corrected_forecasting.py"
CORR_DIR = MAIN1 / "corrected_trajectory_dataset"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9F_V3B_exact_mapping_recovery"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "expected_core": 1053,
        "expected_npz_n": 1704,
        "history_raw": 60,
        "future_raw": 60,
        "future_10hz": 20,
    },
    "2s_to_5s": {
        "expected_core": 826,
        "expected_npz_n": 1319,
        "history_raw": 60,
        "future_raw": 150,
        "future_10hz": 50,
    },
}

FRAME_HINTS = [
    "start_frame",
    "history_start_frame",
    "history_end_frame",
    "future_start_frame",
    "future_end_frame",
    "anchor_frame",
    "window_start_frame",
    "window_end_frame",
    "current_frame",
]

TRACK_HINTS = [
    "track_key",
    "track_id",
    "segment_id",
    "sequence",
    "split",
    "accepted",
    "quality_tier",
    "sample_id",
]


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


def extract_source_function_context():
    require(STEP2B_SCRIPT)

    text = STEP2B_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    # Exact region containing extract_raw_window and surrounding helper logic.
    start = 360
    end = 570

    rows = []

    for i in range(
        start,
        min(end, len(lines)) + 1,
    ):
        rows.append(
            {
                "line_number": i,
                "line": lines[i - 1],
            }
        )

    return pd.DataFrame(rows)


def analyze_task(task, cfg):
    meta_path = CORR_DIR / f"corrected_windows_{task}.csv"
    npz_path = CORR_DIR / f"corrected_{task}.npz"

    require(meta_path)
    require(npz_path)

    metadata = pd.read_csv(
        meta_path,
        dtype={
            "split": str,
            "sequence": str,
            "segment_id": str,
        },
    )

    arrays = np.load(npz_path)

    if "history" not in arrays.files or "future" not in arrays.files:
        raise RuntimeError(
            f"{task}: NPZ missing history/future arrays."
        )

    history = np.asarray(arrays["history"])
    future = np.asarray(arrays["future"])

    core = metadata[
        (pd.to_numeric(metadata["accepted"], errors="coerce") == 1)
        &
        (
            metadata["quality_tier"]
            .astype(str)
            .str.lower()
            ==
            "core"
        )
        &
        (
            metadata["split"].astype(str)
            ==
            "test_real"
        )
    ].copy()

    core = (
        core
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    if len(core) != cfg["expected_core"]:
        raise RuntimeError(
            f"{task}: filtered core count {len(core)} "
            f"!= expected {cfg['expected_core']}"
        )

    sample_ids = pd.to_numeric(
        core["sample_id"],
        errors="raise",
    ).astype(int)

    if sample_ids.duplicated().any():
        dup = sample_ids[
            sample_ids.duplicated()
        ].unique().tolist()

        raise RuntimeError(
            f"{task}: duplicate filtered sample IDs: {dup[:20]}"
        )

    if (sample_ids < 0).any():
        raise RuntimeError(
            f"{task}: filtered core contains negative sample_id."
        )

    if (sample_ids >= len(history)).any():
        bad = sample_ids[
            sample_ids >= len(history)
        ].tolist()

        raise RuntimeError(
            f"{task}: sample IDs out of NPZ range: {bad[:20]}"
        )

    if len(history) != cfg["expected_npz_n"]:
        raise RuntimeError(
            f"{task}: history NPZ length {len(history)} "
            f"!= expected {cfg['expected_npz_n']}"
        )

    if len(future) != cfg["expected_npz_n"]:
        raise RuntimeError(
            f"{task}: future NPZ length {len(future)} "
            f"!= expected {cfg['expected_npz_n']}"
        )

    if history.shape[1:] != (cfg["history_raw"], 7):
        raise RuntimeError(
            f"{task}: history shape {history.shape} unexpected."
        )

    if future.shape[1:] != (cfg["future_raw"], 7):
        raise RuntimeError(
            f"{task}: future shape {future.shape} unexpected."
        )

    corrected_history = history[
        sample_ids.to_numpy(),
        :,
        :2,
    ][
        :,
        ::3,
        :,
    ]

    corrected_future = future[
        sample_ids.to_numpy(),
        :,
        :2,
    ][
        :,
        ::3,
        :,
    ]

    expected_history_shape = (
        cfg["expected_core"],
        20,
        2,
    )

    expected_future_shape = (
        cfg["expected_core"],
        cfg["future_10hz"],
        2,
    )

    if corrected_history.shape != expected_history_shape:
        raise RuntimeError(
            f"{task}: filtered corrected history shape "
            f"{corrected_history.shape} != {expected_history_shape}"
        )

    if corrected_future.shape != expected_future_shape:
        raise RuntimeError(
            f"{task}: filtered corrected future shape "
            f"{corrected_future.shape} != {expected_future_shape}"
        )

    frame_cols = [
        c
        for c in FRAME_HINTS
        if c in metadata.columns
    ]

    key_cols = [
        c
        for c in TRACK_HINTS
        if c in metadata.columns
    ]

    # Include any other frame-like columns to avoid missing exact names.
    for c in metadata.columns:
        lc = str(c).lower()

        if (
            "frame" in lc
            and c not in frame_cols
        ):
            frame_cols.append(c)

    # Core per-sequence / track accounting.
    track_col = (
        "track_key"
        if "track_key" in core.columns
        else
        "segment_id"
        if "segment_id" in core.columns
        else
        None
    )

    seq_rows = []

    for seq, g in core.groupby(
        "sequence",
        sort=True,
        dropna=False,
    ):
        row = {
            "task": task,
            "sequence": str(seq),
            "samples": int(len(g)),
        }

        if track_col is not None:
            row["tracks"] = int(
                g[track_col]
                .astype(str)
                .nunique()
            )
        else:
            row["tracks"] = None

        row["sample_id_min"] = int(
            pd.to_numeric(
                g["sample_id"],
                errors="raise",
            ).min()
        )

        row["sample_id_max"] = int(
            pd.to_numeric(
                g["sample_id"],
                errors="raise",
            ).max()
        )

        seq_rows.append(row)

    # Compact lock table: only identity + frame bounds.
    lock_cols = []

    for c in [
        "sample_id",
        "sequence",
        "segment_id",
        "track_key",
        "track_id",
        *frame_cols,
    ]:
        if c in core.columns and c not in lock_cols:
            lock_cols.append(c)

    lock = core[lock_cols].copy()

    summary = {
        "task": task,
        "metadata_rows": int(len(metadata)),
        "filtered_core_testreal_rows": int(len(core)),
        "expected_core_testreal_rows": int(cfg["expected_core"]),
        "npz_history_shape": list(history.shape),
        "npz_future_shape": list(future.shape),
        "filtered_corrected_history_shape":
            list(corrected_history.shape),
        "filtered_corrected_future_shape":
            list(corrected_future.shape),
        "sample_id_min": int(sample_ids.min()),
        "sample_id_max": int(sample_ids.max()),
        "sample_id_unique": int(sample_ids.nunique()),
        "sample_id_all_valid_npz_indices": True,
        "feature_slice": ":2",
        "downsample": "::3",
        "frame_columns": frame_cols,
        "key_columns": key_cols,
        "metadata_path": str(meta_path),
        "metadata_sha256": sha256_file(meta_path),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
    }

    return (
        summary,
        pd.DataFrame(seq_rows),
        lock,
    )


def main():
    print(
        "\nR9-F-V3B — EXACT STEP2B FILTER / SAMPLE-ID / FRAME-BOUND MAPPING RECOVERY"
    )

    print(
        "DIAGNOSTIC + LOCK ONLY. NO isotropic window build, NO model training, "
        "NO frozen modification. This step reproduces the exact Step2b filter, "
        "proves sample_id->NPZ indexing, verifies :2 feature slicing and ::3 "
        "downsampling, and locks the exact frame-bound columns needed for R9F-V3C.\n"
    )

    source_context = extract_source_function_context()

    summaries = []
    seq_tables = []
    lock_tables = []

    for task, cfg in TASKS.items():
        summary, seq_table, lock = analyze_task(
            task,
            cfg,
        )

        summaries.append(summary)
        seq_tables.append(seq_table)

        lock = lock.copy()
        lock.insert(0, "task", task)
        lock_tables.append(lock)

    summary_df = pd.DataFrame(
        summaries
    )

    seq_df = pd.concat(
        seq_tables,
        ignore_index=True,
    )

    lock_df = pd.concat(
        lock_tables,
        ignore_index=True,
    )

    source_path = (
        OUT
        / "R9F_V3B_EXTRACT_RAW_WINDOW_SOURCE_CONTEXT.csv"
    )

    summary_path = (
        OUT
        / "R9F_V3B_MAPPING_SUMMARY.csv"
    )

    seq_path = (
        OUT
        / "R9F_V3B_FILTERED_CORE_SEQUENCE_ACCOUNTING.csv"
    )

    lock_path = (
        OUT
        / "R9F_V3B_EXACT_CORE_SAMPLE_FRAME_LOCK.csv"
    )

    source_context.to_csv(
        source_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    seq_df.to_csv(
        seq_path,
        index=False,
    )

    lock_df.to_csv(
        lock_path,
        index=False,
    )

    status = (
        "R9F_V3B_EXACT_STEP2B_MAPPING_RECOVERED_AND_LOCKED_READY_FOR_R9F_V3C_ISOTROPIC_WINDOW_BUILD"
    )

    manifest = {
        "step":
            "R9F_V3B_EXACT_STEP2B_MAPPING_RECOVERY",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "isotropic_windows_built":
            False,

        "frozen_results_modified":
            False,

        "step2b_script":
            str(STEP2B_SCRIPT),

        "step2b_script_sha256":
            sha256_file(STEP2B_SCRIPT),

        "mapping_summary_csv":
            str(summary_path),

        "mapping_summary_sha256":
            sha256_file(summary_path),

        "exact_core_sample_frame_lock_csv":
            str(lock_path),

        "exact_core_sample_frame_lock_sha256":
            sha256_file(lock_path),

        "filter":
            {
                "accepted": 1,
                "quality_tier": "core",
                "split": "test_real",
            },

        "corrected_coordinate_slice":
            ":2",

        "downsample":
            "::3",
    }

    (
        OUT
        / "R9F_V3B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "EXACT MAPPING SUMMARY:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nFILTERED CORE SEQUENCE ACCOUNTING:"
    )

    print(
        seq_df.to_string(
            index=False
        )
    )

    print(
        "\nEXACT FRAME/IDENTITY LOCK COLUMNS:"
    )

    print(
        json.dumps(
            {
                task:
                    [
                        c
                        for c in lock_df[
                            lock_df["task"] == task
                        ].columns
                        if c != "task"
                    ]
                for task in TASKS
            },
            indent=2,
        )
    )

    print(
        "\nEXTRACT_RAW_WINDOW SOURCE CONTEXT:"
    )

    print(
        source_context.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )

    print(status)

    print(
        "\nR9-F-V3B COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
