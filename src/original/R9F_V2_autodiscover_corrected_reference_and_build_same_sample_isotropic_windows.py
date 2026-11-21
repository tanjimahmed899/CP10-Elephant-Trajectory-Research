
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

R9D_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9D_isotropic_point_dataset_build"
)

R9E_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9E_V2_exact_window_protocol_match"
)

R9D_MANIFEST = R9D_DIR / "R9D_MANIFEST.json"
ISO_POINTS = R9D_DIR / "R9D_isotropic_cleaned_track_points.csv"

R9E_MANIFEST = R9E_DIR / "R9E_V2_MANIFEST.json"
REFERENCE_LOCK = R9E_DIR / "R9E_V2_FROZEN_REFERENCE_SAMPLE_KEY_LOCK.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9F_V2_same_sample_isotropic_windows"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "expected_samples": 1053,
        "history_raw": 60,
        "future_raw": 60,
        "history_10hz": 20,
        "future_10hz": 20,
    },
    "2s_to_5s": {
        "expected_samples": 826,
        "history_raw": 60,
        "future_raw": 150,
        "history_10hz": 20,
        "future_10hz": 50,
    },
}

DOWNSAMPLE = 3

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
    DOWNLOADS,
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


def verify_inputs():
    for p in [
        R9D_MANIFEST,
        ISO_POINTS,
        R9E_MANIFEST,
        REFERENCE_LOCK,
    ]:
        require(p)

    d = json.loads(R9D_MANIFEST.read_text(encoding="utf-8"))
    e = json.loads(R9E_MANIFEST.read_text(encoding="utf-8"))

    if d.get("status") != (
        "R9D_NONDESTRUCTIVE_ISOTROPIC_POINT_DATASET_BUILT_READY_FOR_R9E_WINDOW_PROTOCOL_MATCH"
    ):
        raise RuntimeError(f"Unexpected R9D status: {d.get('status')}")

    if e.get("status") != (
        "R9E_V2_FROZEN_SAMPLE_UNIVERSE_AND_WINDOW_PROTOCOL_MATCH_LOCKED_READY_FOR_R9F_ISOTROPIC_WINDOW_BUILD"
    ):
        raise RuntimeError(f"Unexpected R9E-V2 status: {e.get('status')}")

    return d, e


def identify_iso_columns(iso):
    required = {"sequence", "cx_iso", "cy_iso"}
    missing = required - set(iso.columns)

    if missing:
        raise RuntimeError(
            f"Isotropic point table missing columns: {sorted(missing)}"
        )

    track_col = next(
        (
            c for c in [
                "track_key",
                "track_id",
                "obj_id",
                "object_id",
            ]
            if c in iso.columns
        ),
        None,
    )

    frame_col = next(
        (
            c for c in [
                "frame",
                "frame_id",
                "frame_idx",
                "frame_index",
            ]
            if c in iso.columns
        ),
        None,
    )

    if track_col is None:
        raise RuntimeError("Could not identify isotropic track column.")

    if frame_col is None:
        raise RuntimeError("Could not identify isotropic frame column.")

    return track_col, frame_col


def corrected_candidate_pairs(task):
    meta_name = f"corrected_windows_{task}.csv"
    npz_name = f"corrected_{task}.npz"

    metas = []
    npzs = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        try:
            metas.extend(root.rglob(meta_name))
        except Exception:
            pass

        try:
            npzs.extend(root.rglob(npz_name))
        except Exception:
            pass

    pairs = []

    for meta in metas:
        for npz in npzs:
            # Strong preference: same parent directory.
            same_parent = meta.parent == npz.parent

            try:
                m = pd.read_csv(meta)
            except Exception:
                continue

            try:
                arr = np.load(npz)
                if "history" not in arr.files or "future" not in arr.files:
                    continue

                h = np.asarray(arr["history"])
                f = np.asarray(arr["future"])
            except Exception:
                continue

            expected = TASKS[task]

            sample_ok = (
                len(m) == expected["expected_samples"]
                and h.shape[0] == expected["expected_samples"]
                and f.shape[0] == expected["expected_samples"]
            )

            shape_ok = (
                h.shape[1:] == (expected["history_10hz"], 2)
                and f.shape[1:] == (expected["future_10hz"], 2)
            )

            pairs.append(
                {
                    "task": task,
                    "metadata_path": str(meta),
                    "npz_path": str(npz),
                    "same_parent": same_parent,
                    "metadata_rows": int(len(m)),
                    "history_shape": str(tuple(h.shape)),
                    "future_shape": str(tuple(f.shape)),
                    "sample_count_match": bool(sample_ok),
                    "array_shape_match": bool(shape_ok),
                    "metadata_sha256": sha256_file(meta),
                    "npz_sha256": sha256_file(npz),
                }
            )

    return pd.DataFrame(pairs)


def select_corrected_pair(task):
    audit = corrected_candidate_pairs(task)

    if len(audit) == 0:
        raise RuntimeError(
            f"No corrected metadata/NPZ candidates found for {task}."
        )

    valid = audit[
        audit["sample_count_match"]
        & audit["array_shape_match"]
    ].copy()

    if len(valid) == 0:
        raise RuntimeError(
            f"No valid corrected metadata/NPZ pair for {task}.\n"
            + audit.to_string(index=False)
        )

    valid = valid.sort_values(
        [
            "same_parent",
            "metadata_path",
            "npz_path",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort",
    )

    best = valid.iloc[0]

    # If multiple valid same-parent pairs exist with different hashes, stop.
    same_quality = valid[
        valid["same_parent"] == best["same_parent"]
    ]

    unique_hash_pairs = same_quality[
        ["metadata_sha256", "npz_sha256"]
    ].drop_duplicates()

    if len(unique_hash_pairs) > 1:
        raise RuntimeError(
            f"Ambiguous multiple valid corrected pairs for {task}; "
            "manual provenance lock required.\n"
            + same_quality.to_string(index=False)
        )

    return (
        Path(best["metadata_path"]),
        Path(best["npz_path"]),
        audit,
    )


def identify_metadata_columns(meta):
    cols = set(meta.columns)

    sample_col = next(
        (
            c for c in [
                "sample_id",
                "sample_index",
                "window_id",
            ]
            if c in cols
        ),
        None,
    )

    sequence_col = next(
        (
            c for c in [
                "sequence",
                "source_group",
                "sequence_id",
                "source",
            ]
            if c in cols
        ),
        None,
    )

    track_col = next(
        (
            c for c in [
                "track_key",
                "track_id",
                "track",
            ]
            if c in cols
        ),
        None,
    )

    if sample_col is None:
        raise RuntimeError(
            f"Corrected metadata missing sample ID. Columns={sorted(cols)}"
        )

    if sequence_col is None:
        raise RuntimeError(
            f"Corrected metadata missing sequence/source. Columns={sorted(cols)}"
        )

    if track_col is None:
        raise RuntimeError(
            f"Corrected metadata missing track key. Columns={sorted(cols)}"
        )

    return sample_col, sequence_col, track_col


def identify_frame_bounds(meta, history_raw, future_raw):
    cols = set(meta.columns)

    start_col = next(
        (
            c for c in [
                "window_start_frame",
                "history_start_frame",
                "start_frame",
                "window_start",
            ]
            if c in cols
        ),
        None,
    )

    history_end_col = next(
        (
            c for c in [
                "history_end_frame",
                "current_frame",
                "anchor_frame",
                "window_history_end_frame",
            ]
            if c in cols
        ),
        None,
    )

    future_end_col = next(
        (
            c for c in [
                "future_end_frame",
                "window_end_frame",
                "end_frame",
            ]
            if c in cols
        ),
        None,
    )

    if future_end_col is None:
        raise RuntimeError(
            "Corrected metadata has no recognizable future-end frame column. "
            f"Columns={sorted(cols)}"
        )

    if start_col is None and history_end_col is None:
        raise RuntimeError(
            "Corrected metadata has neither window-start nor history-end frame. "
            f"Columns={sorted(cols)}"
        )

    rows = []

    for idx, row in meta.iterrows():
        future_end = int(row[future_end_col])

        if start_col is not None:
            start = int(row[start_col])
        else:
            history_end = int(row[history_end_col])
            start = history_end - history_raw + 1

        expected_future_end = (
            start + history_raw + future_raw - 1
        )

        if future_end != expected_future_end:
            alt_start = (
                future_end - (history_raw + future_raw) + 1
            )

            if (
                history_end_col is not None
                and int(row[history_end_col])
                == alt_start + history_raw - 1
            ):
                start = alt_start
            else:
                raise RuntimeError(
                    f"Frame-bound mismatch at metadata row {idx}: "
                    f"start={start}, future_end={future_end}, "
                    f"expected={expected_future_end}"
                )

        rows.append(
            {
                "_metadata_row_index": int(idx),
                "_window_start_frame_locked": int(start),
                "_future_end_frame_locked": int(future_end),
            }
        )

    bounds = pd.DataFrame(rows).set_index(
        "_metadata_row_index"
    )

    out = meta.copy()

    out["_window_start_frame_locked"] = bounds[
        "_window_start_frame_locked"
    ]

    out["_future_end_frame_locked"] = bounds[
        "_future_end_frame_locked"
    ]

    return (
        out,
        {
            "start_col": start_col,
            "history_end_col": history_end_col,
            "future_end_col": future_end_col,
        },
    )


def normalize_sequence(s):
    s = str(s)

    for prefix in [
        "test_real_",
        "train_real_",
    ]:
        if s.startswith(prefix):
            s = s[len(prefix):]

    return s


def build_track_lookup(iso, iso_track_col, frame_col):
    lookup = {}

    for (sequence, track), g in iso.groupby(
        ["sequence", iso_track_col],
        sort=False,
        dropna=False,
    ):
        gg = g.sort_values(
            frame_col,
            kind="mergesort",
        ).copy()

        frames = pd.to_numeric(
            gg[frame_col],
            errors="raise",
        ).astype(int)

        if frames.duplicated().any():
            dup = frames[
                frames.duplicated()
            ].unique().tolist()

            raise RuntimeError(
                f"Duplicate frames for sequence={sequence}, "
                f"track={track}: {dup[:10]}"
            )

        lookup[
            (str(sequence), str(track))
        ] = gg.set_index(frames)

    return lookup


def resolve_track_key(
    sequence,
    reference_track,
    meta_track,
    lookup,
):
    seq = normalize_sequence(sequence)

    candidates = []

    for t in [reference_track, meta_track]:
        if pd.notna(t):
            candidates.append(str(t))

    for t in candidates:
        key = (seq, t)
        if key in lookup:
            return key

    same_seq = [
        key for key in lookup
        if key[0] == seq
    ]

    for t in candidates:
        matches = [
            key for key in same_seq
            if (
                key[1] == t
                or key[1].endswith(t)
                or t.endswith(key[1])
            )
        ]

        if len(matches) == 1:
            return matches[0]

    raise RuntimeError(
        f"Could not resolve isotropic track for sequence={seq}, "
        f"reference_track={reference_track}, meta_track={meta_track}"
    )


def extract_window(
    track_df,
    start_frame,
    history_raw,
    future_raw,
):
    end_frame = (
        start_frame + history_raw + future_raw - 1
    )

    wanted = np.arange(
        start_frame,
        end_frame + 1,
        dtype=int,
    )

    missing = [
        int(f) for f in wanted
        if f not in track_df.index
    ]

    if missing:
        raise RuntimeError(
            f"Missing isotropic raw frames: {missing[:20]}"
        )

    block = track_df.loc[wanted]

    coords = block[
        ["cx_iso", "cy_iso"]
    ].to_numpy(dtype=np.float64)

    expected_len = history_raw + future_raw

    if coords.shape != (expected_len, 2):
        raise RuntimeError(
            f"Window shape {coords.shape} != {(expected_len,2)}"
        )

    history = coords[:history_raw][::DOWNSAMPLE]
    future = coords[history_raw:][::DOWNSAMPLE]

    return history, future


def verify_reference_vs_metadata(
    task,
    ref,
    meta,
    sample_col,
    sequence_col,
    track_col,
):
    ref_task = ref[
        ref["task"].astype(str) == task
    ].copy()

    cfg = TASKS[task]

    if len(ref_task) != cfg["expected_samples"]:
        raise RuntimeError(
            f"{task}: reference count mismatch."
        )

    ref_task["_sample_id_int"] = pd.to_numeric(
        ref_task["sample_id"],
        errors="raise",
    ).astype(int)

    meta["_sample_id_int"] = pd.to_numeric(
        meta[sample_col],
        errors="raise",
    ).astype(int)

    if meta["_sample_id_int"].duplicated().any():
        raise RuntimeError(
            f"{task}: metadata duplicate sample IDs."
        )

    ref_ids = set(
        ref_task["_sample_id_int"].tolist()
    )

    meta_ids = set(
        meta["_sample_id_int"].tolist()
    )

    if ref_ids != meta_ids:
        raise RuntimeError(
            f"{task}: sample ID set mismatch. "
            f"missing={sorted(ref_ids-meta_ids)[:20]}, "
            f"extra={sorted(meta_ids-ref_ids)[:20]}"
        )

    merged = ref_task.merge(
        meta,
        on="_sample_id_int",
        how="left",
        suffixes=("_ref", "_meta"),
        validate="one_to_one",
    )

    seq_meta_col = (
        f"{sequence_col}_meta"
        if f"{sequence_col}_meta" in merged.columns
        else sequence_col
    )

    track_meta_col = (
        f"{track_col}_meta"
        if f"{track_col}_meta" in merged.columns
        else track_col
    )

    merged["_sequence_ref_normalized"] = (
        merged["sequence_normalized"]
        .map(normalize_sequence)
    )

    merged["_sequence_meta_normalized"] = (
        merged[seq_meta_col]
        .map(normalize_sequence)
    )

    if not (
        merged["_sequence_ref_normalized"]
        == merged["_sequence_meta_normalized"]
    ).all():
        bad = merged[
            merged["_sequence_ref_normalized"]
            != merged["_sequence_meta_normalized"]
        ]

        raise RuntimeError(
            f"{task}: sequence mismatch:\n"
            + bad[
                [
                    "_sample_id_int",
                    "_sequence_ref_normalized",
                    "_sequence_meta_normalized",
                ]
            ].head(20).to_string(index=False)
        )

    return merged, seq_meta_col, track_meta_col


def build_task(
    task,
    ref,
    lookup,
):
    cfg = TASKS[task]

    meta_path, npz_path, discovery = (
        select_corrected_pair(task)
    )

    meta = pd.read_csv(meta_path)

    sample_col, sequence_col, track_col = (
        identify_metadata_columns(meta)
    )

    meta, frame_cols = identify_frame_bounds(
        meta,
        cfg["history_raw"],
        cfg["future_raw"],
    )

    merged, seq_meta_col, track_meta_col = (
        verify_reference_vs_metadata(
            task,
            ref,
            meta,
            sample_col,
            sequence_col,
            track_col,
        )
    )

    merged = merged.sort_values(
        "_sample_id_int",
        kind="mergesort",
    ).reset_index(drop=True)

    histories = []
    futures = []
    rows = []

    for _, row in merged.iterrows():
        sequence = row["_sequence_meta_normalized"]

        reference_track = (
            row["track_key_ref"]
            if "track_key_ref" in row.index
            else row["track_key"]
        )

        meta_track = row[track_meta_col]

        key = resolve_track_key(
            sequence,
            reference_track,
            meta_track,
            lookup,
        )

        history, future = extract_window(
            lookup[key],
            int(row["_window_start_frame_locked"]),
            cfg["history_raw"],
            cfg["future_raw"],
        )

        if history.shape != (
            cfg["history_10hz"],
            2,
        ):
            raise RuntimeError(
                f"{task}: history shape {history.shape}"
            )

        if future.shape != (
            cfg["future_10hz"],
            2,
        ):
            raise RuntimeError(
                f"{task}: future shape {future.shape}"
            )

        histories.append(history)
        futures.append(future)

        rows.append(
            {
                "task": task,
                "sample_id": int(row["_sample_id_int"]),
                "sequence": str(sequence),
                "reference_track_key": str(reference_track),
                "metadata_track_key": str(meta_track),
                "resolved_isotropic_track": str(key[1]),
                "window_start_frame":
                    int(row["_window_start_frame_locked"]),
                "future_end_frame":
                    int(row["_future_end_frame_locked"]),
            }
        )

    history_arr = np.stack(histories)
    future_arr = np.stack(futures)

    frozen = np.load(npz_path)

    frozen_history = np.asarray(frozen["history"])
    frozen_future = np.asarray(frozen["future"])

    if history_arr.shape != frozen_history.shape:
        raise RuntimeError(
            f"{task}: isotropic history shape {history_arr.shape} "
            f"!= frozen corrected {frozen_history.shape}"
        )

    if future_arr.shape != frozen_future.shape:
        raise RuntimeError(
            f"{task}: isotropic future shape {future_arr.shape} "
            f"!= frozen corrected {frozen_future.shape}"
        )

    return (
        history_arr,
        future_arr,
        pd.DataFrame(rows),
        discovery,
        {
            "metadata_path": str(meta_path),
            "metadata_sha256": sha256_file(meta_path),
            "npz_path": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "metadata_columns": {
                "sample_col": sample_col,
                "sequence_col": sequence_col,
                "track_col": track_col,
                **frame_cols,
            },
        },
    )


def main():
    print(
        "\nR9-F-V2 — AUTODISCOVERED EXACT SAME-SAMPLE ISOTROPIC WINDOW BUILD"
    )

    print(
        "NO model training and NO frozen overwrite. "
        "This revision first autodiscovers and hash-locks the original corrected "
        "metadata/NPZ pair for each task, verifies frozen counts/shapes, then builds "
        "parallel isotropic windows on the exact same sample IDs.\n"
    )

    verify_inputs()

    iso = pd.read_csv(ISO_POINTS)
    ref = pd.read_csv(REFERENCE_LOCK)

    iso_track_col, frame_col = identify_iso_columns(iso)

    lookup = build_track_lookup(
        iso,
        iso_track_col,
        frame_col,
    )

    summary_rows = []
    discovery_tables = []
    provenance = {}

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        (
            history,
            future,
            meta_out,
            discovery,
            prov,
        ) = build_task(
            task,
            ref,
            lookup,
        )

        discovery = discovery.copy()
        discovery["selected_pair"] = (
            (
                discovery["metadata_path"]
                == prov["metadata_path"]
            )
            &
            (
                discovery["npz_path"]
                == prov["npz_path"]
            )
        )

        discovery_tables.append(discovery)

        npz_out = OUT / f"R9F_V2_isotropic_{task}.npz"
        meta_out_path = OUT / f"R9F_V2_isotropic_windows_{task}.csv"

        np.savez_compressed(
            npz_out,
            history=history,
            future=future,
        )

        meta_out.to_csv(
            meta_out_path,
            index=False,
        )

        summary_rows.append(
            {
                "task": task,
                "samples": int(history.shape[0]),
                "history_shape": str(tuple(history.shape)),
                "future_shape": str(tuple(future.shape)),
                "expected_samples": TASKS[task]["expected_samples"],
                "sample_count_match":
                    int(history.shape[0])
                    == TASKS[task]["expected_samples"],
                "history_all_finite":
                    bool(np.isfinite(history).all()),
                "future_all_finite":
                    bool(np.isfinite(future).all()),
                "output_npz": str(npz_out),
                "output_npz_sha256": sha256_file(npz_out),
                "output_metadata": str(meta_out_path),
                "output_metadata_sha256":
                    sha256_file(meta_out_path),
            }
        )

        provenance[task] = prov

    summary = pd.DataFrame(summary_rows)

    discovery_all = pd.concat(
        discovery_tables,
        ignore_index=True,
    )

    if not summary["sample_count_match"].all():
        raise RuntimeError("Sample count mismatch.")

    if not summary["history_all_finite"].all():
        raise RuntimeError("Non-finite history.")

    if not summary["future_all_finite"].all():
        raise RuntimeError("Non-finite future.")

    discovery_path = OUT / "R9F_V2_CORRECTED_REFERENCE_AUTODISCOVERY.csv"
    summary_path = OUT / "R9F_V2_WINDOW_BUILD_SUMMARY.csv"

    discovery_all.to_csv(
        discovery_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    status = (
        "R9F_V2_EXACT_SAME_SAMPLE_ISOTROPIC_WINDOWS_BUILT_READY_FOR_R9G_BASELINE_COMPARISON"
    )

    manifest = {
        "step":
            "R9F_V2_AUTODISCOVERED_EXACT_SAME_SAMPLE_ISOTROPIC_WINDOW_BUILD",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "frozen_results_modified":
            False,

        "corrected_reference_provenance":
            provenance,

        "autodiscovery_csv":
            str(discovery_path),

        "autodiscovery_sha256":
            sha256_file(discovery_path),

        "summary_csv":
            str(summary_path),

        "summary_sha256":
            sha256_file(summary_path),

        "isotropic_source":
            str(ISO_POINTS),

        "isotropic_source_sha256":
            sha256_file(ISO_POINTS),

        "sample_lock":
            str(REFERENCE_LOCK),

        "sample_lock_sha256":
            sha256_file(REFERENCE_LOCK),
    }

    (
        OUT / "R9F_V2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("CORRECTED REFERENCE AUTODISCOVERY:")
    print(
        discovery_all.to_string(
            index=False
        )
    )

    print("\nISOTROPIC WINDOW BUILD SUMMARY:")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nLOCKED CORRECTED REFERENCE PROVENANCE:")
    print(
        json.dumps(
            provenance,
            indent=2,
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR9-F-V2 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
