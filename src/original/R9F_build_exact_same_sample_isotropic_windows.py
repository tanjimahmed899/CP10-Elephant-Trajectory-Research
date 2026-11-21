
from pathlib import Path
import json
import hashlib
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
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

STEP2B_SCRIPT = MAIN2 / "step2b_raw_vs_corrected_forecasting.py"
CORR_DIR = MAIN2 / "corrected_trajectory_dataset"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9F_same_sample_isotropic_windows"
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
        STEP2B_SCRIPT,
    ]:
        require(p)

    d = json.loads(
        R9D_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    e = json.loads(
        R9E_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        d.get("status")
        !=
        "R9D_NONDESTRUCTIVE_ISOTROPIC_POINT_DATASET_BUILT_READY_FOR_R9E_WINDOW_PROTOCOL_MATCH"
    ):
        raise RuntimeError(
            f"Unexpected R9D status: {d.get('status')}"
        )

    if (
        e.get("status")
        !=
        "R9E_V2_FROZEN_SAMPLE_UNIVERSE_AND_WINDOW_PROTOCOL_MATCH_LOCKED_READY_FOR_R9F_ISOTROPIC_WINDOW_BUILD"
    ):
        raise RuntimeError(
            f"Unexpected R9E-V2 status: {e.get('status')}"
        )

    return d, e


def verify_protocol_constants():
    text = STEP2B_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    checks = []

    def add(name, passed, detail):
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "detail": str(detail),
            }
        )

    # Soft code-evidence checks, hard locked protocol checks below.
    add(
        "script_mentions_future_raw_60",
        "60" in text,
        "2s->2s future_raw expected 60",
    )

    add(
        "script_mentions_future_raw_150",
        "150" in text,
        "2s->5s future_raw expected 150",
    )

    add(
        "script_mentions_future_10hz_20",
        "20" in text,
        "2s->2s future_10hz expected 20",
    )

    add(
        "script_mentions_future_10hz_50",
        "50" in text,
        "2s->5s future_10hz expected 50",
    )

    add(
        "locked_downsample_is_3",
        DOWNSAMPLE == 3,
        "30 Hz raw -> 10 Hz by ::3",
    )

    audit = pd.DataFrame(checks)

    if not audit["passed"].all():
        raise RuntimeError(
            "Protocol-constant audit failed:\n"
            + audit.to_string(index=False)
        )

    return audit


def identify_iso_columns(iso):
    required = {
        "sequence",
        "cx_iso",
        "cy_iso",
    }

    missing = required - set(
        iso.columns
    )

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
        raise RuntimeError(
            "Could not identify isotropic point track column."
        )

    if frame_col is None:
        raise RuntimeError(
            "Could not identify isotropic point frame column."
        )

    return track_col, frame_col


def corrected_paths(task):
    meta = CORR_DIR / f"corrected_windows_{task}.csv"
    npz = CORR_DIR / f"corrected_{task}.npz"

    require(meta)
    require(npz)

    return meta, npz


def identify_metadata_columns(meta):
    cols = set(
        meta.columns
    )

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
    cols = set(
        meta.columns
    )

    # Preferred exact start/end fields.
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

    # We need either start+future_end OR history_end+future_end.
    if future_end_col is None:
        raise RuntimeError(
            "Corrected metadata has no recognizable future-end frame column. "
            f"Columns={sorted(cols)}"
        )

    if start_col is None and history_end_col is None:
        raise RuntimeError(
            "Corrected metadata has neither recognizable window-start nor history-end frame. "
            f"Columns={sorted(cols)}"
        )

    rows = []

    for idx, row in meta.iterrows():
        future_end = int(
            row[
                future_end_col
            ]
        )

        if start_col is not None:
            start = int(
                row[
                    start_col
                ]
            )
        else:
            history_end = int(
                row[
                    history_end_col
                ]
            )

            start = (
                history_end
                -
                history_raw
                +
                1
            )

        expected_future_end = (
            start
            +
            history_raw
            +
            future_raw
            -
            1
        )

        if future_end != expected_future_end:
            # If metadata uses history-end as inclusive anchor with a different
            # start convention, try reconstructing from future_end instead.
            alt_start = (
                future_end
                -
                (
                    history_raw
                    +
                    future_raw
                )
                +
                1
            )

            if (
                history_end_col is not None
                and
                int(
                    row[
                        history_end_col
                    ]
                )
                ==
                alt_start
                +
                history_raw
                -
                1
            ):
                start = alt_start
            else:
                raise RuntimeError(
                    f"Frame-bound mismatch at metadata row {idx}: "
                    f"start={start}, future_end={future_end}, "
                    f"expected_future_end={expected_future_end}"
                )

        rows.append(
            {
                "_metadata_row_index":
                    int(idx),

                "_window_start_frame_locked":
                    int(start),

                "_future_end_frame_locked":
                    int(future_end),
            }
        )

    bounds = pd.DataFrame(
        rows
    ).set_index(
        "_metadata_row_index"
    )

    out = meta.copy()

    out[
        "_window_start_frame_locked"
    ] = bounds[
        "_window_start_frame_locked"
    ]

    out[
        "_future_end_frame_locked"
    ] = bounds[
        "_future_end_frame_locked"
    ]

    return (
        out,
        {
            "start_col":
                start_col,

            "history_end_col":
                history_end_col,

            "future_end_col":
                future_end_col,
        },
    )


def normalize_sequence(s):
    s = str(s)

    for prefix in [
        "test_real_",
        "train_real_",
    ]:
        if s.startswith(
            prefix
        ):
            s = s[
                len(prefix):
            ]

    return s


def build_track_lookup(
    iso,
    iso_track_col,
    frame_col,
):
    lookup = {}

    for (
        sequence,
        track,
    ), g in iso.groupby(
        [
            "sequence",
            iso_track_col,
        ],
        sort=False,
        dropna=False,
    ):
        gg = g.sort_values(
            frame_col,
            kind="mergesort",
        ).copy()

        frames = pd.to_numeric(
            gg[
                frame_col
            ],
            errors="raise",
        ).astype(int)

        if frames.duplicated().any():
            dup = frames[
                frames.duplicated()
            ].unique().tolist()

            raise RuntimeError(
                f"Duplicate frames for sequence={sequence}, track={track}: {dup[:10]}"
            )

        lookup[
            (
                str(sequence),
                str(track),
            )
        ] = gg.set_index(
            frames
        )

    return lookup


def resolve_track_key(
    sequence,
    reference_track,
    meta_track,
    lookup,
):
    seq = normalize_sequence(
        sequence
    )

    candidates = []

    for t in [
        reference_track,
        meta_track,
    ]:
        if pd.notna(
            t
        ):
            candidates.append(
                str(
                    t
                )
            )

    # Exact lookup first.
    for t in candidates:
        key = (
            seq,
            t,
        )

        if key in lookup:
            return key

    # If point table uses track_key with a test_real_ prefix or equivalent,
    # try suffix matching inside the same sequence.
    same_seq_keys = [
        key
        for key in lookup
        if key[
            0
        ]
        ==
        seq
    ]

    for t in candidates:
        matches = [
            key
            for key in same_seq_keys
            if (
                key[
                    1
                ]
                ==
                t
                or
                key[
                    1
                ].endswith(
                    t
                )
                or
                t.endswith(
                    key[
                        1
                    ]
                )
            )
        ]

        if len(
            matches
        ) == 1:
            return matches[
                0
            ]

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
        start_frame
        +
        history_raw
        +
        future_raw
        -
        1
    )

    wanted_frames = np.arange(
        start_frame,
        end_frame + 1,
        dtype=int,
    )

    missing = [
        int(f)
        for f in wanted_frames
        if f not in track_df.index
    ]

    if missing:
        raise RuntimeError(
            f"Missing raw frames in isotropic source: {missing[:20]}"
        )

    block = track_df.loc[
        wanted_frames
    ]

    coords = block[
        [
            "cx_iso",
            "cy_iso",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    expected_len = (
        history_raw
        +
        future_raw
    )

    if coords.shape != (
        expected_len,
        2,
    ):
        raise RuntimeError(
            f"Raw isotropic window shape {coords.shape} != {(expected_len,2)}"
        )

    history_raw_arr = coords[
        :history_raw
    ]

    future_raw_arr = coords[
        history_raw:
    ]

    history = history_raw_arr[
        ::DOWNSAMPLE
    ]

    future = future_raw_arr[
        ::DOWNSAMPLE
    ]

    return (
        history,
        future,
    )


def verify_reference_vs_metadata(
    task,
    ref,
    meta,
    sample_col,
    sequence_col,
    track_col,
):
    ref_task = ref[
        ref[
            "task"
        ].astype(str)
        ==
        task
    ].copy()

    if len(
        ref_task
    ) != TASKS[
        task
    ][
        "expected_samples"
    ]:
        raise RuntimeError(
            f"{task}: reference-lock sample count mismatch."
        )

    ref_task[
        "_sample_id_int"
    ] = pd.to_numeric(
        ref_task[
            "sample_id"
        ],
        errors="raise",
    ).astype(int)

    meta[
        "_sample_id_int"
    ] = pd.to_numeric(
        meta[
            sample_col
        ],
        errors="raise",
    ).astype(int)

    if meta[
        "_sample_id_int"
    ].duplicated().any():
        raise RuntimeError(
            f"{task}: corrected metadata contains duplicate sample IDs."
        )

    ref_ids = set(
        ref_task[
            "_sample_id_int"
        ].tolist()
    )

    meta_ids = set(
        meta[
            "_sample_id_int"
        ].tolist()
    )

    if ref_ids != meta_ids:
        raise RuntimeError(
            f"{task}: sample-ID set mismatch. "
            f"missing_in_meta={sorted(ref_ids-meta_ids)[:20]}, "
            f"extra_in_meta={sorted(meta_ids-ref_ids)[:20]}"
        )

    merged = ref_task.merge(
        meta,
        on="_sample_id_int",
        how="left",
        suffixes=(
            "_ref",
            "_meta",
        ),
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

    merged[
        "_sequence_ref_normalized"
    ] = merged[
        "sequence_normalized"
    ].map(
        normalize_sequence
    )

    merged[
        "_sequence_meta_normalized"
    ] = merged[
        seq_meta_col
    ].map(
        normalize_sequence
    )

    if not (
        merged[
            "_sequence_ref_normalized"
        ]
        ==
        merged[
            "_sequence_meta_normalized"
        ]
    ).all():
        bad = merged[
            merged[
                "_sequence_ref_normalized"
            ]
            !=
            merged[
                "_sequence_meta_normalized"
            ]
        ]

        raise RuntimeError(
            f"{task}: sequence mismatch between reference lock and corrected metadata:\n"
            +
            bad[
                [
                    "_sample_id_int",
                    "_sequence_ref_normalized",
                    "_sequence_meta_normalized",
                ]
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    return (
        merged,
        seq_meta_col,
        track_meta_col,
    )


def build_task_windows(
    task,
    ref,
    iso,
    lookup,
    iso_track_col,
):
    cfg = TASKS[
        task
    ]

    meta_path, corr_npz_path = corrected_paths(
        task
    )

    meta = pd.read_csv(
        meta_path
    )

    (
        sample_col,
        sequence_col,
        track_col,
    ) = identify_metadata_columns(
        meta
    )

    meta, frame_cols = identify_frame_bounds(
        meta,
        cfg[
            "history_raw"
        ],
        cfg[
            "future_raw"
        ],
    )

    (
        merged,
        seq_meta_col,
        track_meta_col,
    ) = verify_reference_vs_metadata(
        task,
        ref,
        meta,
        sample_col,
        sequence_col,
        track_col,
    )

    # Order exactly by frozen sample ID because corrected arrays use sample_id indexing.
    merged = merged.sort_values(
        "_sample_id_int",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    histories = []
    futures = []
    meta_rows = []

    for _, row in merged.iterrows():
        sequence = row[
            "_sequence_meta_normalized"
        ]

        reference_track = (
            row[
                "track_key_ref"
            ]
            if "track_key_ref" in row.index
            else row[
                "track_key"
            ]
        )

        meta_track = row[
            track_meta_col
        ]

        lookup_key = resolve_track_key(
            sequence,
            reference_track,
            meta_track,
            lookup,
        )

        track_df = lookup[
            lookup_key
        ]

        start_frame = int(
            row[
                "_window_start_frame_locked"
            ]
        )

        history, future = extract_window(
            track_df,
            start_frame,
            cfg[
                "history_raw"
            ],
            cfg[
                "future_raw"
            ],
        )

        if history.shape != (
            cfg[
                "history_10hz"
            ],
            2,
        ):
            raise RuntimeError(
                f"{task} sample {row['_sample_id_int']}: "
                f"history shape {history.shape}"
            )

        if future.shape != (
            cfg[
                "future_10hz"
            ],
            2,
        ):
            raise RuntimeError(
                f"{task} sample {row['_sample_id_int']}: "
                f"future shape {future.shape}"
            )

        histories.append(
            history
        )

        futures.append(
            future
        )

        meta_rows.append(
            {
                "task":
                    task,

                "sample_id":
                    int(
                        row[
                            "_sample_id_int"
                        ]
                    ),

                "sequence":
                    str(
                        sequence
                    ),

                "reference_track_key":
                    str(
                        reference_track
                    ),

                "metadata_track_key":
                    str(
                        meta_track
                    ),

                "resolved_isotropic_track":
                    str(
                        lookup_key[
                            1
                        ]
                    ),

                "window_start_frame":
                    int(
                        start_frame
                    ),

                "future_end_frame":
                    int(
                        row[
                            "_future_end_frame_locked"
                        ]
                    ),

                "history_raw_frames":
                    int(
                        cfg[
                            "history_raw"
                        ]
                    ),

                "future_raw_frames":
                    int(
                        cfg[
                            "future_raw"
                        ]
                    ),

                "history_10hz_frames":
                    int(
                        cfg[
                            "history_10hz"
                        ]
                    ),

                "future_10hz_frames":
                    int(
                        cfg[
                            "future_10hz"
                        ]
                    ),
            }
        )

    history_arr = np.stack(
        histories,
        axis=0,
    )

    future_arr = np.stack(
        futures,
        axis=0,
    )

    # Compare only shapes/counts to frozen corrected arrays.
    corr = np.load(
        corr_npz_path
    )

    corr_history = np.asarray(
        corr[
            "history"
        ]
    )

    corr_future = np.asarray(
        corr[
            "future"
        ]
    )

    if history_arr.shape != corr_history.shape:
        raise RuntimeError(
            f"{task}: isotropic history shape {history_arr.shape} "
            f"!= frozen corrected history shape {corr_history.shape}"
        )

    if future_arr.shape != corr_future.shape:
        raise RuntimeError(
            f"{task}: isotropic future shape {future_arr.shape} "
            f"!= frozen corrected future shape {corr_future.shape}"
        )

    return (
        history_arr,
        future_arr,
        pd.DataFrame(
            meta_rows
        ),
        {
            "corrected_metadata_path":
                str(
                    meta_path
                ),

            "corrected_metadata_sha256":
                sha256_file(
                    meta_path
                ),

            "corrected_npz_path":
                str(
                    corr_npz_path
                ),

            "corrected_npz_sha256":
                sha256_file(
                    corr_npz_path
                ),

            "metadata_columns":
                {
                    "sample":
                        sample_col,

                    "sequence":
                        sequence_col,

                    "track":
                        track_col,

                    **frame_cols,
                },
        },
    )


def main():
    print(
        "\nR9-F — EXACT SAME-SAMPLE ISOTROPIC WINDOW BUILD"
    )

    print(
        "NO model training and NO frozen-output overwrite. "
        "This step uses the locked frozen sample IDs plus the original corrected-window "
        "frame bounds to extract parallel isotropic 10 Hz windows with exactly the same "
        "sample counts and array shapes as Step2b.\n"
    )

    verify_inputs()

    protocol_audit = verify_protocol_constants()

    iso = pd.read_csv(
        ISO_POINTS
    )

    ref = pd.read_csv(
        REFERENCE_LOCK
    )

    iso_track_col, frame_col = identify_iso_columns(
        iso
    )

    lookup = build_track_lookup(
        iso,
        iso_track_col,
        frame_col,
    )

    task_summaries = []
    provenance = {}

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        (
            history,
            future,
            metadata,
            task_prov,
        ) = build_task_windows(
            task,
            ref,
            iso,
            lookup,
            iso_track_col,
        )

        npz_path = (
            OUT
            / f"R9F_isotropic_{task}.npz"
        )

        meta_path = (
            OUT
            / f"R9F_isotropic_windows_{task}.csv"
        )

        np.savez_compressed(
            npz_path,
            history=history,
            future=future,
        )

        metadata.to_csv(
            meta_path,
            index=False,
        )

        task_summaries.append(
            {
                "task":
                    task,

                "samples":
                    int(
                        history.shape[
                            0
                        ]
                    ),

                "history_shape":
                    str(
                        tuple(
                            history.shape
                        )
                    ),

                "future_shape":
                    str(
                        tuple(
                            future.shape
                        )
                    ),

                "expected_samples":
                    TASKS[
                        task
                    ][
                        "expected_samples"
                    ],

                "sample_count_match":
                    bool(
                        history.shape[
                            0
                        ]
                        ==
                        TASKS[
                            task
                        ][
                            "expected_samples"
                        ]
                    ),

                "history_all_finite":
                    bool(
                        np.isfinite(
                            history
                        ).all()
                    ),

                "future_all_finite":
                    bool(
                        np.isfinite(
                            future
                        ).all()
                    ),

                "history_min":
                    float(
                        history.min()
                    ),

                "history_max":
                    float(
                        history.max()
                    ),

                "future_min":
                    float(
                        future.min()
                    ),

                "future_max":
                    float(
                        future.max()
                    ),

                "output_npz":
                    str(
                        npz_path
                    ),

                "output_npz_sha256":
                    sha256_file(
                        npz_path
                    ),

                "output_metadata":
                    str(
                        meta_path
                    ),

                "output_metadata_sha256":
                    sha256_file(
                        meta_path
                    ),
            }
        )

        provenance[
            task
        ] = task_prov

    summary = pd.DataFrame(
        task_summaries
    )

    if not summary[
        "sample_count_match"
    ].all():
        raise RuntimeError(
            "R9F sample-count mismatch."
        )

    if not summary[
        "history_all_finite"
    ].all():
        raise RuntimeError(
            "R9F history contains non-finite values."
        )

    if not summary[
        "future_all_finite"
    ].all():
        raise RuntimeError(
            "R9F future contains non-finite values."
        )

    summary_path = (
        OUT
        / "R9F_WINDOW_BUILD_SUMMARY.csv"
    )

    protocol_path = (
        OUT
        / "R9F_PROTOCOL_CONSTANT_AUDIT.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    protocol_audit.to_csv(
        protocol_path,
        index=False,
    )

    status = (
        "R9F_EXACT_SAME_SAMPLE_ISOTROPIC_WINDOWS_BUILT_READY_FOR_R9G_BASELINE_COMPARISON"
    )

    manifest = {
        "step":
            "R9F_EXACT_SAME_SAMPLE_ISOTROPIC_WINDOW_BUILD",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "frozen_results_modified":
            False,

        "reference_sample_lock":
            str(
                REFERENCE_LOCK
            ),

        "reference_sample_lock_sha256":
            sha256_file(
                REFERENCE_LOCK
            ),

        "isotropic_point_source":
            str(
                ISO_POINTS
            ),

        "isotropic_point_source_sha256":
            sha256_file(
                ISO_POINTS
            ),

        "isotropic_track_column":
            iso_track_col,

        "isotropic_frame_column":
            frame_col,

        "downsample":
            DOWNSAMPLE,

        "task_protocols":
            TASKS,

        "corrected_reference_provenance":
            provenance,

        "summary_csv":
            str(
                summary_path
            ),

        "summary_sha256":
            sha256_file(
                summary_path
            ),
    }

    (
        OUT
        / "R9F_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "PROTOCOL CONSTANT AUDIT:"
    )

    print(
        protocol_audit.to_string(
            index=False
        )
    )

    print(
        "\nISOTROPIC WINDOW BUILD SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nCORRECTED REFERENCE PROVENANCE:"
    )

    print(
        json.dumps(
            provenance,
            indent=2,
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-F COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
