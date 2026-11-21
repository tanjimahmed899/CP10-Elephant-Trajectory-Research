
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"

R12A_DIR = (
    REVISION_ROOT
    / "R12_track_accounting"
    / "R12A_track_segment_provenance_audit"
)
R12A_MANIFEST = R12A_DIR / "R12A_MANIFEST.json"

FINAL_NEURAL = MAIN1 / "final_neural_trajectory_dataset"

OUT = (
    REVISION_ROOT
    / "R12_track_accounting"
    / "R12B_exact_original_track_to_forecast_segment_mapping"
)
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "metadata":
            FINAL_NEURAL
            / "2s_to_2s_birdsai_test_metadata.csv",

        "expected_samples":
            1053,

        "expected_forecast_units":
            89,
    },

    "2s_to_5s": {
        "metadata":
            FINAL_NEURAL
            / "2s_to_5s_birdsai_test_metadata.csv",

        "expected_samples":
            826,

        "expected_forecast_units":
            63,
    },
}

EXPECTED_DATASET_LEVEL_TESTREAL_TRACKS = 45


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def require(path):
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(p)


def sha256_file(
    path,
    chunk=1024 * 1024,
):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            b = f.read(
                chunk
            )

            if not b:
                break

            h.update(
                b
            )

    return h.hexdigest()


def verify_r12a():
    require(
        R12A_MANIFEST
    )

    m = json.loads(
        R12A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    allowed = {
        "R12A_45_DATASET_TRACK_SOURCE_AND_89_63_FORECAST_UNITS_IDENTIFIED_READY_FOR_R12B_EXACT_MAPPING",
        "R12A_89_63_FORECAST_UNITS_CONFIRMED_BUT_45_TRACK_SOURCE_NOT_YET_UNIQUELY_IDENTIFIED_REVIEW_CANDIDATES",
    }

    if m.get(
        "status"
    ) not in allowed:
        raise RuntimeError(
            f"Unexpected R12-A status: {m.get('status')}"
        )

    return m


def read_metadata(
    task,
    cfg,
):
    p = cfg[
        "metadata"
    ]

    require(
        p
    )

    df = pd.read_csv(
        p,
        dtype={
            "sample_id":
                str,

            "sequence":
                str,

            "object_id":
                str,

            "segment_id":
                str,

            "track_key":
                str,

            "source_group":
                str,
        },
    )

    required = {
        "sample_id",
        "sequence",
        "object_id",
        "segment_id",
        "track_key",
        "obs_start_frame",
        "obs_end_frame",
        "future_end_frame",
    }

    missing = (
        required
        -
        set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{task}: missing columns {sorted(missing)}"
        )

    if len(
        df
    ) != cfg[
        "expected_samples"
    ]:
        raise RuntimeError(
            f"{task}: rows={len(df)}, expected={cfg['expected_samples']}"
        )

    if df[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{task}: duplicate sample_id."
        )

    for col in [
        "sequence",
        "object_id",
        "segment_id",
        "track_key",
    ]:
        if df[
            col
        ].isna().any():
            raise RuntimeError(
                f"{task}: missing values in {col}."
            )

    df[
        "original_track_key"
    ] = (
        df[
            "sequence"
        ].astype(
            str
        )
        +
        "|"
        +
        df[
            "object_id"
        ].astype(
            str
        )
    )

    df[
        "canonical_segment_key"
    ] = (
        df[
            "sequence"
        ].astype(
            str
        )
        +
        "|"
        +
        df[
            "object_id"
        ].astype(
            str
        )
        +
        "|"
        +
        df[
            "segment_id"
        ].astype(
            str
        )
    )

    return (
        df,
        p,
    )


def audit_track_key_formula(
    df,
    task,
):
    rows = []

    # Determine whether track_key is exactly one-to-one with
    # (sequence, object_id, segment_id), regardless of string formatting.
    seg_to_track = (
        df[
            [
                "canonical_segment_key",
                "track_key",
            ]
        ]
        .drop_duplicates()
    )

    track_to_seg = (
        df[
            [
                "track_key",
                "canonical_segment_key",
            ]
        ]
        .drop_duplicates()
    )

    seg_counts = (
        seg_to_track.groupby(
            "canonical_segment_key"
        )[
            "track_key"
        ]
        .nunique()
    )

    track_counts = (
        track_to_seg.groupby(
            "track_key"
        )[
            "canonical_segment_key"
        ]
        .nunique()
    )

    rows.append(
        {
            "task":
                task,

            "unique_canonical_segments":
                int(
                    df[
                        "canonical_segment_key"
                    ].nunique()
                ),

            "unique_track_keys":
                int(
                    df[
                        "track_key"
                    ].nunique()
                ),

            "every_segment_maps_to_one_track_key":
                bool(
                    (
                        seg_counts
                        ==
                        1
                    ).all()
                ),

            "every_track_key_maps_to_one_segment":
                bool(
                    (
                        track_counts
                        ==
                        1
                    ).all()
                ),

            "bijection":
                bool(
                    (
                        seg_counts
                        ==
                        1
                    ).all()
                    and
                    (
                        track_counts
                        ==
                        1
                    ).all()
                    and
                    df[
                        "canonical_segment_key"
                    ].nunique()
                    ==
                    df[
                        "track_key"
                    ].nunique()
                ),
        }
    )

    return pd.DataFrame(
        rows
    )


def build_mapping(
    df,
    task,
):
    rows = []

    grouped = df.groupby(
        [
            "sequence",
            "object_id",
            "original_track_key",
        ],
        dropna=False,
        sort=True,
    )

    for (
        sequence,
        object_id,
        original_track_key,
    ), g in grouped:
        segment_keys = sorted(
            g[
                "canonical_segment_key"
            ]
            .astype(
                str
            )
            .unique()
            .tolist()
        )

        track_keys = sorted(
            g[
                "track_key"
            ]
            .astype(
                str
            )
            .unique()
            .tolist()
        )

        segment_ids = sorted(
            g[
                "segment_id"
            ]
            .astype(
                str
            )
            .unique()
            .tolist()
        )

        rows.append(
            {
                "task":
                    task,

                "sequence":
                    sequence,

                "object_id":
                    object_id,

                "original_track_key":
                    original_track_key,

                "forecast_segments":
                    int(
                        len(
                            segment_keys
                        )
                    ),

                "segment_ids":
                    "|".join(
                        segment_ids
                    ),

                "track_keys":
                    "|".join(
                        track_keys
                    ),

                "samples":
                    int(
                        len(
                            g
                        )
                    ),

                "min_obs_start_frame":
                    int(
                        pd.to_numeric(
                            g[
                                "obs_start_frame"
                            ],
                            errors="raise",
                        ).min()
                    ),

                "max_future_end_frame":
                    int(
                        pd.to_numeric(
                            g[
                                "future_end_frame"
                            ],
                            errors="raise",
                        ).max()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_segment_table(
    df,
    task,
):
    work = df.copy()

    for col in [
        "obs_start_frame",
        "obs_end_frame",
        "future_end_frame",
    ]:
        work[
            col
        ] = pd.to_numeric(
            work[
                col
            ],
            errors="raise",
        )

    seg = (
        work.groupby(
            [
                "sequence",
                "object_id",
                "original_track_key",
                "segment_id",
                "canonical_segment_key",
                "track_key",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            samples=(
                "sample_id",
                "size",
            ),

            min_obs_start_frame=(
                "obs_start_frame",
                "min",
            ),

            max_obs_end_frame=(
                "obs_end_frame",
                "max",
            ),

            max_future_end_frame=(
                "future_end_frame",
                "max",
            ),
        )
    )

    seg[
        "task"
    ] = task

    return seg


def per_source_original_vs_segments(
    df,
    task,
):
    rows = []

    for sequence, g in df.groupby(
        "sequence",
        dropna=False,
        sort=True,
    ):
        rows.append(
            {
                "task":
                    task,

                "sequence":
                    sequence,

                "original_tracks":
                    int(
                        g[
                            "original_track_key"
                        ].nunique()
                    ),

                "forecast_segments":
                    int(
                        g[
                            "track_key"
                        ].nunique()
                    ),

                "samples":
                    int(
                        len(
                            g
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def segment_multiplicity_summary(
    mapping_df,
):
    rows = []

    for task, g in mapping_df.groupby(
        "task",
        sort=True,
    ):
        counts = pd.to_numeric(
            g[
                "forecast_segments"
            ],
            errors="raise",
        )

        rows.append(
            {
                "task":
                    task,

                "original_tracks":
                    int(
                        len(
                            g
                        )
                    ),

                "forecast_segments":
                    int(
                        counts.sum()
                    ),

                "tracks_with_1_segment":
                    int(
                        (
                            counts
                            ==
                            1
                        ).sum()
                    ),

                "tracks_with_2_segments":
                    int(
                        (
                            counts
                            ==
                            2
                        ).sum()
                    ),

                "tracks_with_3plus_segments":
                    int(
                        (
                            counts
                            >=
                            3
                        ).sum()
                    ),

                "max_segments_from_one_original_track":
                    int(
                        counts.max()
                    ),

                "mean_segments_per_original_track":
                    float(
                        counts.mean()
                    ),

                "median_segments_per_original_track":
                    float(
                        counts.median()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def cross_task_track_overlap(
    mapping_all,
):
    a = mapping_all[
        mapping_all[
            "task"
        ]
        ==
        "2s_to_2s"
    ]

    b = mapping_all[
        mapping_all[
            "task"
        ]
        ==
        "2s_to_5s"
    ]

    a_tracks = set(
        a[
            "original_track_key"
        ].astype(
            str
        )
    )

    b_tracks = set(
        b[
            "original_track_key"
        ].astype(
            str
        )
    )

    return pd.DataFrame(
        [
            {
                "2s_original_tracks":
                    int(
                        len(
                            a_tracks
                        )
                    ),

                "5s_original_tracks":
                    int(
                        len(
                            b_tracks
                        )
                    ),

                "common_original_tracks":
                    int(
                        len(
                            a_tracks
                            &
                            b_tracks
                        )
                    ),

                "2s_only_original_tracks":
                    int(
                        len(
                            a_tracks
                            -
                            b_tracks
                        )
                    ),

                "5s_only_original_tracks":
                    int(
                        len(
                            b_tracks
                            -
                            a_tracks
                        )
                    ),

                "all_5s_original_tracks_subset_of_2s":
                    bool(
                        b_tracks.issubset(
                            a_tracks
                        )
                    ),
            }
        ]
    )


def main():
    print(
        "\nR12-B — EXACT ORIGINAL-TRACK → FORECAST-SEGMENT MAPPING"
    )

    print(
        "NO training and NO inference. "
        "This step defines the original dataset-level track identity directly from "
        "the frozen TestReal forecasting metadata as (sequence, object_id), while the "
        "forecastable segment identity is (sequence, object_id, segment_id). It then "
        "tests whether track_key is a one-to-one alias of that segment identity, counts "
        "the exact original-track population, and quantifies how original tracks split "
        "into 2s/5s forecastable segments.\n"
    )

    verify_r12a()

    mapping_frames = []
    segment_frames = []
    per_source_frames = []
    formula_frames = []
    metadata_lock_rows = []

    meta_frames = {}

    for task, cfg in TASKS.items():
        df, p = read_metadata(
            task,
            cfg,
        )

        meta_frames[
            task
        ] = df

        formula = audit_track_key_formula(
            df,
            task,
        )

        formula_frames.append(
            formula
        )

        mapping = build_mapping(
            df,
            task,
        )

        mapping_frames.append(
            mapping
        )

        segments = build_segment_table(
            df,
            task,
        )

        segment_frames.append(
            segments
        )

        per_source = per_source_original_vs_segments(
            df,
            task,
        )

        per_source_frames.append(
            per_source
        )

        metadata_lock_rows.append(
            {
                "task":
                    task,

                "metadata_path":
                    str(
                        p
                    ),

                "metadata_sha256":
                    sha256_file(
                        p
                    ),

                "samples":
                    int(
                        len(
                            df
                        )
                    ),

                "unique_sequence_object_tracks":
                    int(
                        df[
                            "original_track_key"
                        ].nunique()
                    ),

                "unique_sequence_object_segment_units":
                    int(
                        df[
                            "canonical_segment_key"
                        ].nunique()
                    ),

                "unique_track_keys":
                    int(
                        df[
                            "track_key"
                        ].nunique()
                    ),
            }
        )

    mapping_all = pd.concat(
        mapping_frames,
        ignore_index=True
    )

    segments_all = pd.concat(
        segment_frames,
        ignore_index=True
    )

    per_source_all = pd.concat(
        per_source_frames,
        ignore_index=True
    )

    formula_all = pd.concat(
        formula_frames,
        ignore_index=True
    )

    metadata_lock_df = pd.DataFrame(
        metadata_lock_rows
    )

    multiplicity_df = segment_multiplicity_summary(
        mapping_all
    )

    overlap_df = cross_task_track_overlap(
        mapping_all
    )

    # Strong resolution checks.
    row_2s = metadata_lock_df[
        metadata_lock_df[
            "task"
        ]
        ==
        "2s_to_2s"
    ].iloc[
        0
    ]

    row_5s = metadata_lock_df[
        metadata_lock_df[
            "task"
        ]
        ==
        "2s_to_5s"
    ].iloc[
        0
    ]

    original_2s = int(
        row_2s[
            "unique_sequence_object_tracks"
        ]
    )

    segments_2s = int(
        row_2s[
            "unique_sequence_object_segment_units"
        ]
    )

    segments_5s = int(
        row_5s[
            "unique_sequence_object_segment_units"
        ]
    )

    track_key_bijection = bool(
        formula_all[
            "bijection"
        ].all()
    )

    count_45_resolved = bool(
        original_2s
        ==
        EXPECTED_DATASET_LEVEL_TESTREAL_TRACKS
    )

    count_89_resolved = bool(
        segments_2s
        ==
        TASKS[
            "2s_to_2s"
        ][
            "expected_forecast_units"
        ]
    )

    count_63_resolved = bool(
        segments_5s
        ==
        TASKS[
            "2s_to_5s"
        ][
            "expected_forecast_units"
        ]
    )

    five_is_subset = bool(
        overlap_df[
            "all_5s_original_tracks_subset_of_2s"
        ].iloc[
            0
        ]
    )

    # Save outputs.
    metadata_lock_path = (
        OUT
        /
        "R12B_METADATA_IDENTITY_LOCK.csv"
    )

    formula_path = (
        OUT
        /
        "R12B_TRACK_KEY_SEGMENT_BIJECTION.csv"
    )

    mapping_path = (
        OUT
        /
        "R12B_ORIGINAL_TRACK_TO_FORECAST_SEGMENT_MAPPING.csv"
    )

    segments_path = (
        OUT
        /
        "R12B_FORECAST_SEGMENT_TABLE.csv"
    )

    per_source_path = (
        OUT
        /
        "R12B_PER_SOURCE_ORIGINAL_TRACK_VS_SEGMENT_COUNTS.csv"
    )

    multiplicity_path = (
        OUT
        /
        "R12B_SEGMENT_MULTIPLICITY_SUMMARY.csv"
    )

    overlap_path = (
        OUT
        /
        "R12B_2S_VS_5S_ORIGINAL_TRACK_OVERLAP.csv"
    )

    metadata_lock_df.to_csv(
        metadata_lock_path,
        index=False
    )

    formula_all.to_csv(
        formula_path,
        index=False
    )

    mapping_all.to_csv(
        mapping_path,
        index=False
    )

    segments_all.to_csv(
        segments_path,
        index=False
    )

    per_source_all.to_csv(
        per_source_path,
        index=False
    )

    multiplicity_df.to_csv(
        multiplicity_path,
        index=False
    )

    overlap_df.to_csv(
        overlap_path,
        index=False
    )

    if (
        count_45_resolved
        and
        count_89_resolved
        and
        count_63_resolved
        and
        track_key_bijection
        and
        five_is_subset
    ):
        status = (
            "R12B_45_ORIGINAL_TESTREAL_TRACKS_TO_89_2S_AND_63_5S_FORECAST_SEGMENTS_EXACTLY_RESOLVED_READY_FOR_R12C_BUILD_RULE_PROVENANCE"
        )
    else:
        status = (
            "R12B_TRACK_ACCOUNTING_NOT_FULLY_RESOLVED_REVIEW_MAPPING_BEFORE_R12C"
        )

    manuscript = f"""R12 track-accounting resolution

Terminology
-----------
Original dataset-level track:
    unique (sequence, object_id)

Forecastable segment / tracklet:
    unique (sequence, object_id, segment_id)

Frozen track_key:
    one-to-one alias of the forecastable segment identity, provided the bijection audit passes.

Resolved counts
---------------
2 s forecasting metadata:
    original TestReal tracks = {original_2s}
    forecastable segments/tracklets = {segments_2s}
    windows = {TASKS['2s_to_2s']['expected_samples']}

5 s forecasting metadata:
    original tracks represented = {int(row_5s['unique_sequence_object_tracks'])}
    forecastable segments/tracklets = {segments_5s}
    windows = {TASKS['2s_to_5s']['expected_samples']}

Interpretation
--------------
The previously reported 45 and 89 values are not contradictory counts of the same object. The 45 value is the number of original TestReal object trajectories identified by (sequence, object_id). During forecasting-data construction, continuity/usable-window segmentation creates multiple segment_id units from some original tracks, yielding 89 forecastable 2 s segments. The 5 s task applies a longer future-window requirement, retaining only 63 forecastable segments; all 5 s original tracks are checked as a subset of those represented in the 2 s task.

Manuscript wording
------------------
Use:
    "45 original TestReal tracks produced 89 forecastable 2-s segments/tracklets and 63 forecastable 5-s segments/tracklets."

Do not use:
    "TestReal has 89 tracks."

The exact reason each original track splits into one or more segment_id units should be documented from the dataset-build continuity/gap logic in R12-C before R12 is fully closed.
"""

    manuscript_path = (
        OUT
        /
        "R12B_MANUSCRIPT_READY_TRACK_ACCOUNTING_RESOLUTION.txt"
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8"
    )

    manifest = {
        "step":
            "R12B_EXACT_ORIGINAL_TRACK_TO_FORECAST_SEGMENT_MAPPING",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "original_track_identity":
            "(sequence, object_id)",

        "forecast_segment_identity":
            "(sequence, object_id, segment_id)",

        "track_key_is_segment_alias_bijection":
            track_key_bijection,

        "resolved_2s_original_tracks":
            original_2s,

        "resolved_2s_forecast_segments":
            segments_2s,

        "resolved_5s_original_tracks":
            int(
                row_5s[
                    "unique_sequence_object_tracks"
                ]
            ),

        "resolved_5s_forecast_segments":
            segments_5s,

        "count_45_resolved":
            count_45_resolved,

        "count_89_resolved":
            count_89_resolved,

        "count_63_resolved":
            count_63_resolved,

        "all_5s_original_tracks_subset_of_2s":
            five_is_subset,

        "provisional_manuscript_wording":
            (
                "45 original TestReal tracks produced 89 forecastable 2-s "
                "segments/tracklets and 63 forecastable 5-s segments/tracklets."
            ),

        "remaining_requirement":
            (
                "R12-C must identify and lock the dataset-build continuity/gap rule "
                "that creates segment_id from original object tracks."
            ),

        "outputs": {
            "metadata_identity_lock":
                str(
                    metadata_lock_path
                ),

            "track_key_segment_bijection":
                str(
                    formula_path
                ),

            "original_track_to_segment_mapping":
                str(
                    mapping_path
                ),

            "forecast_segment_table":
                str(
                    segments_path
                ),

            "per_source_counts":
                str(
                    per_source_path
                ),

            "segment_multiplicity":
                str(
                    multiplicity_path
                ),

            "2s_vs_5s_original_track_overlap":
                str(
                    overlap_path
                ),

            "manuscript_ready_resolution":
                str(
                    manuscript_path
                ),
        },
    }

    (
        OUT
        /
        "R12B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "METADATA IDENTITY LOCK:"
    )

    print(
        metadata_lock_df.to_string(
            index=False
        )
    )

    print(
        "\nTRACK_KEY ↔ SEGMENT IDENTITY BIJECTION:"
    )

    print(
        formula_all.to_string(
            index=False
        )
    )

    print(
        "\nSEGMENT MULTIPLICITY SUMMARY:"
    )

    print(
        multiplicity_df.to_string(
            index=False
        )
    )

    print(
        "\nPER-SOURCE ORIGINAL TRACK vs FORECAST SEGMENT COUNTS:"
    )

    print(
        per_source_all.to_string(
            index=False
        )
    )

    print(
        "\n2s vs 5s ORIGINAL-TRACK OVERLAP:"
    )

    print(
        overlap_df.to_string(
            index=False
        )
    )

    print(
        "\nRESOLUTION:"
    )

    print(
        f"  2s: {original_2s} original (sequence, object_id) tracks -> {segments_2s} forecastable segments."
    )

    print(
        f"  5s: {int(row_5s['unique_sequence_object_tracks'])} represented original tracks -> {segments_5s} forecastable segments."
    )

    print(
        f"  track_key is one-to-one with (sequence, object_id, segment_id): {track_key_bijection}"
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-B COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
