
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"

R11E_MANIFEST = (
    REVISION_ROOT
    / "R11_noise_floor_analysis"
    / "R11E_final_noise_floor_conclusion"
    / "R11E_MANIFEST.json"
)

FINAL_NEURAL = MAIN1 / "final_neural_trajectory_dataset"
CORRECTED_ROOT = MAIN1 / "corrected_trajectory_dataset"

OUT = (
    REVISION_ROOT
    / "R12_track_accounting"
    / "R12A_track_segment_provenance_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "metadata": FINAL_NEURAL / "2s_to_2s_birdsai_test_metadata.csv",
        "expected_samples": 1053,
        "expected_forecast_units": 89,
        "expected_sources": 6,
    },
    "2s_to_5s": {
        "metadata": FINAL_NEURAL / "2s_to_5s_birdsai_test_metadata.csv",
        "expected_samples": 826,
        "expected_forecast_units": 63,
        "expected_sources": 5,
    },
}

KNOWN_DATASET_LEVEL_TESTREAL_TRACKS = 45

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
    REVISION_ROOT,
]

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".txt",
    ".md",
    ".csv",
}

MAX_TEXT_BYTES = 5_000_000


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


def verify_r11():
    require(R11E_MANIFEST)

    m = json.loads(
        R11E_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R11E_EMPIRICAL_NOISE_FLOOR_CONCLUSION_LOCKED_R11_COMPLETE_READY_FOR_R12_TRACK_ACCOUNTING"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R11-E status: {m.get('status')}"
        )

    return m


def normalize_series(s):
    return (
        s.astype(str)
        .str.strip()
        .replace(
            {
                "nan": "",
                "None": "",
                "<NA>": "",
            }
        )
    )


def candidate_original_track_columns(df):
    cols = []

    for c in df.columns:
        lc = str(c).lower()

        if (
            "track" in lc
            or lc in {
                "id",
                "object_id",
                "animal_id",
                "elephant_id",
            }
        ):
            cols.append(c)

    return cols


def source_column(df):
    preferred = [
        "sequence",
        "source_group",
        "source_id",
        "video",
    ]

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in preferred:
        if name in lower:
            return lower[name]

    raise RuntimeError(
        f"No source/sequence column found. Columns={list(df.columns)}"
    )


def forecast_unit_column(df):
    preferred = [
        "track_key",
        "segment_id",
        "segment",
        "tracklet_id",
    ]

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in preferred:
        if name in lower:
            return lower[name]

    raise RuntimeError(
        f"No forecast-unit column found. Columns={list(df.columns)}"
    )


def audit_task_metadata():
    rows = []
    per_source_rows = []
    unit_rows = []
    meta_frames = {}

    for task, cfg in TASKS.items():
        p = cfg["metadata"]
        require(p)

        df = pd.read_csv(
            p,
            dtype=str,
        )

        meta_frames[task] = df.copy()

        if len(df) != cfg["expected_samples"]:
            raise RuntimeError(
                f"{task}: metadata rows={len(df)}, expected={cfg['expected_samples']}"
            )

        src_col = source_column(df)
        unit_col = forecast_unit_column(df)

        src = normalize_series(
            df[src_col]
        )

        unit = normalize_series(
            df[unit_col]
        )

        n_sources = int(
            src.nunique()
        )

        n_units = int(
            unit.nunique()
        )

        rows.append(
            {
                "task":
                    task,

                "metadata_path":
                    str(p),

                "metadata_sha256":
                    sha256_file(p),

                "rows":
                    int(
                        len(df)
                    ),

                "source_column":
                    src_col,

                "forecast_unit_column":
                    unit_col,

                "sources":
                    n_sources,

                "forecast_units":
                    n_units,

                "expected_sources":
                    cfg["expected_sources"],

                "expected_forecast_units":
                    cfg["expected_forecast_units"],

                "source_count_match":
                    bool(
                        n_sources
                        ==
                        cfg["expected_sources"]
                    ),

                "forecast_unit_count_match":
                    bool(
                        n_units
                        ==
                        cfg["expected_forecast_units"]
                    ),

                "all_columns":
                    "|".join(
                        str(c)
                        for c in df.columns
                    ),
            }
        )

        work = pd.DataFrame(
            {
                "source":
                    src,

                "forecast_unit":
                    unit,
            }
        )

        for source, g in work.groupby(
            "source",
            dropna=False,
        ):
            per_source_rows.append(
                {
                    "task":
                        task,

                    "source":
                        source,

                    "samples":
                        int(
                            len(g)
                        ),

                    "forecast_units":
                        int(
                            g[
                                "forecast_unit"
                            ].nunique()
                        ),
                }
            )

        counts = (
            work.groupby(
                [
                    "source",
                    "forecast_unit",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="samples"
            )
        )

        counts[
            "task"
        ] = task

        unit_rows.append(
            counts
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(per_source_rows),
        pd.concat(
            unit_rows,
            ignore_index=True
        ),
        meta_frames,
    )


def inventory_corrected_csvs():
    rows = []

    if not CORRECTED_ROOT.exists():
        return pd.DataFrame(rows)

    for p in CORRECTED_ROOT.glob(
        "**/*.csv"
    ):
        try:
            if p.stat().st_size > 20_000_000:
                continue

            df = pd.read_csv(
                p,
                nrows=5000,
            )

        except Exception:
            continue

        rows.append(
            {
                "path":
                    str(p),

                "name":
                    p.name,

                "sha256":
                    sha256_file(p),

                "size_bytes":
                    int(
                        p.stat().st_size
                    ),

                "columns":
                    "|".join(
                        str(c)
                        for c in df.columns
                    ),

                "candidate_track_columns":
                    "|".join(
                        str(c)
                        for c in candidate_original_track_columns(
                            df
                        )
                    ),

                "has_sequence":
                    bool(
                        any(
                            str(c).lower()
                            in {
                                "sequence",
                                "source_group",
                                "video",
                            }
                            for c in df.columns
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def score_track_source_candidate(row):
    score = 0
    reasons = []

    path = str(
        row[
            "path"
        ]
    ).lower()

    cols = str(
        row[
            "columns"
        ]
    ).lower()

    track_cols = str(
        row[
            "candidate_track_columns"
        ]
    ).lower()

    if "test" in path:
        score += 3
        reasons.append(
            "test artifact"
        )

    if "track" in track_cols:
        score += 3
        reasons.append(
            "track column"
        )

    if "sequence" in cols or "source_group" in cols:
        score += 2
        reasons.append(
            "source identity"
        )

    if "frame" in cols:
        score += 1
        reasons.append(
            "frame-level data"
        )

    if (
        "corrected" in cols
        or
        "cx_n" in cols
        or
        "cy_n" in cols
    ):
        score += 2
        reasons.append(
            "trajectory coordinates"
        )

    return score, "|".join(
        reasons
    )


def rank_corrected_candidates(inv):
    if len(inv) == 0:
        return inv

    rows = []

    for _, row in inv.iterrows():
        score, reasons = score_track_source_candidate(
            row
        )

        rec = row.to_dict()
        rec[
            "track_source_relevance_score"
        ] = score
        rec[
            "relevance_reasons"
        ] = reasons

        rows.append(
            rec
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "track_source_relevance_score",
            "path",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def search_track_segment_logic():
    needles = [
        "track_key",
        "segment_id",
        "segment",
        "tracklet",
        "gap",
        "continu",
        "split",
        "max_gap",
        "frame_gap",
        "45",
        "89",
    ]

    rows = []
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.glob(
            "**/*"
        ):
            if not p.is_file():
                continue

            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue

            try:
                if p.stat().st_size > MAX_TEXT_BYTES:
                    continue
            except Exception:
                continue

            low_path = str(
                p
            ).lower()

            if any(
                token in low_path
                for token in [
                    "site-packages",
                    "__pycache__",
                    ".git",
                ]
            ):
                continue

            key = str(
                p.resolve()
            ).lower()

            if key in seen:
                continue

            seen.add(
                key
            )

            # Prefer files likely related to trajectory build/accounting.
            name_low = p.name.lower()

            if not any(
                token in name_low
                for token in [
                    "track",
                    "trajectory",
                    "neural",
                    "dataset",
                    "audit",
                    "build",
                    "clean",
                    "correct",
                    "protocol",
                    "summary",
                ]
            ):
                continue

            try:
                lines = p.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines()
            except Exception:
                continue

            for i, line in enumerate(
                lines,
                start=1
            ):
                ll = line.lower()

                matched = [
                    needle
                    for needle in needles
                    if needle in ll
                ]

                if not matched:
                    continue

                rows.append(
                    {
                        "source_file":
                            str(p),

                        "line_number":
                            i,

                        "matched_terms":
                            "|".join(
                                matched
                            ),

                        "line":
                            line,
                    }
                )

    return pd.DataFrame(
        rows
    )


def inspect_top_candidate_files(
    ranked_inv,
):
    rows = []

    if len(
        ranked_inv
    ) == 0:
        return pd.DataFrame(
            rows
        )

    top = ranked_inv.head(
        20
    )

    for _, row in top.iterrows():
        p = Path(
            str(
                row[
                    "path"
                ]
            )
        )

        try:
            df = pd.read_csv(
                p,
            )
        except Exception:
            continue

        if len(df) == 0:
            continue

        src_col = None

        for c in df.columns:
            if str(c).lower() in {
                "sequence",
                "source_group",
                "video",
            }:
                src_col = c
                break

        track_cols = candidate_original_track_columns(
            df
        )

        if (
            src_col is None
            or
            not track_cols
        ):
            continue

        for track_col in track_cols:
            work = pd.DataFrame(
                {
                    "source":
                        normalize_series(
                            df[
                                src_col
                            ]
                        ),

                    "track":
                        normalize_series(
                            df[
                                track_col
                            ]
                        ),
                }
            )

            work = work[
                (
                    work[
                        "source"
                    ]
                    !=
                    ""
                )
                &
                (
                    work[
                        "track"
                    ]
                    !=
                    ""
                )
            ]

            if len(work) == 0:
                continue

            unique_pair_count = int(
                work.drop_duplicates(
                    [
                        "source",
                        "track",
                    ]
                ).shape[
                    0
                ]
            )

            rows.append(
                {
                    "path":
                        str(p),

                    "source_column":
                        src_col,

                    "track_column":
                        track_col,

                    "rows":
                        int(
                            len(df)
                        ),

                    "sources":
                        int(
                            work[
                                "source"
                            ].nunique()
                        ),

                    "source_track_pairs":
                        unique_pair_count,

                    "equals_known_dataset_level_45":
                        bool(
                            unique_pair_count
                            ==
                            KNOWN_DATASET_LEVEL_TESTREAL_TRACKS
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def cross_task_overlap(meta_frames):
    a = meta_frames[
        "2s_to_2s"
    ]

    b = meta_frames[
        "2s_to_5s"
    ]

    a_src_col = source_column(
        a
    )

    b_src_col = source_column(
        b
    )

    a_unit_col = forecast_unit_column(
        a
    )

    b_unit_col = forecast_unit_column(
        b
    )

    a_units = set(
        zip(
            normalize_series(
                a[
                    a_src_col
                ]
            ),
            normalize_series(
                a[
                    a_unit_col
                ]
            ),
        )
    )

    b_units = set(
        zip(
            normalize_series(
                b[
                    b_src_col
                ]
            ),
            normalize_series(
                b[
                    b_unit_col
                ]
            ),
        )
    )

    return pd.DataFrame(
        [
            {
                "2s_to_2s_units":
                    len(
                        a_units
                    ),

                "2s_to_5s_units":
                    len(
                        b_units
                    ),

                "common_units":
                    len(
                        a_units
                        &
                        b_units
                    ),

                "2s_only_units":
                    len(
                        a_units
                        -
                        b_units
                    ),

                "5s_only_units":
                    len(
                        b_units
                        -
                        a_units
                    ),

                "all_5s_units_subset_of_2s_units":
                    bool(
                        b_units.issubset(
                            a_units
                        )
                    ),
            }
        ]
    )


def main():
    print(
        "\nR12-A — DATASET TRACK vs FORECAST SEGMENT PROVENANCE AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "This step does not assume that the reported 45 TestReal dataset-level tracks "
        "and the 89/63 forecasting units are the same type of object. It audits the exact "
        "forecast metadata, inventories corrected trajectory sources, searches the build "
        "code for track/segment continuity logic, and attempts to identify an authoritative "
        "source containing exactly 45 unique (source, original-track) pairs. The goal is to "
        "replace the ambiguous phrase '89 tracks' with provenance-correct terminology.\n"
    )

    verify_r11()

    (
        task_df,
        per_source_df,
        unit_df,
        meta_frames,
    ) = audit_task_metadata()

    corrected_inv = inventory_corrected_csvs()
    ranked_inv = rank_corrected_candidates(
        corrected_inv
    )

    candidate_count_df = inspect_top_candidate_files(
        ranked_inv
    )

    logic_df = search_track_segment_logic()

    overlap_df = cross_task_overlap(
        meta_frames
    )

    task_path = (
        OUT
        /
        "R12A_FORECAST_METADATA_ACCOUNTING.csv"
    )

    per_source_path = (
        OUT
        /
        "R12A_FORECAST_UNITS_PER_SOURCE.csv"
    )

    unit_path = (
        OUT
        /
        "R12A_FORECAST_UNIT_SAMPLE_COUNTS.csv"
    )

    corrected_inv_path = (
        OUT
        /
        "R12A_CORRECTED_TRAJECTORY_CSV_INVENTORY.csv"
    )

    ranked_path = (
        OUT
        /
        "R12A_RANKED_ORIGINAL_TRACK_SOURCE_CANDIDATES.csv"
    )

    candidate_count_path = (
        OUT
        /
        "R12A_CANDIDATE_SOURCE_TRACK_PAIR_COUNTS.csv"
    )

    logic_path = (
        OUT
        /
        "R12A_TRACK_SEGMENT_BUILD_LOGIC_EVIDENCE.csv"
    )

    overlap_path = (
        OUT
        /
        "R12A_2S_VS_5S_FORECAST_UNIT_OVERLAP.csv"
    )

    task_df.to_csv(
        task_path,
        index=False
    )

    per_source_df.to_csv(
        per_source_path,
        index=False
    )

    unit_df.to_csv(
        unit_path,
        index=False
    )

    corrected_inv.to_csv(
        corrected_inv_path,
        index=False
    )

    ranked_inv.to_csv(
        ranked_path,
        index=False
    )

    candidate_count_df.to_csv(
        candidate_count_path,
        index=False
    )

    logic_df.to_csv(
        logic_path,
        index=False
    )

    overlap_df.to_csv(
        overlap_path,
        index=False
    )

    exact_45 = (
        candidate_count_df[
            candidate_count_df[
                "equals_known_dataset_level_45"
            ]
            ==
            True
        ].copy()
        if len(
            candidate_count_df
        )
        else
        pd.DataFrame()
    )

    forecast_counts_ok = bool(
        task_df[
            "forecast_unit_count_match"
        ].all()
        and
        task_df[
            "source_count_match"
        ].all()
    )

    if (
        forecast_counts_ok
        and
        len(
            exact_45
        )
        >
        0
    ):
        status = (
            "R12A_45_DATASET_TRACK_SOURCE_AND_89_63_FORECAST_UNITS_IDENTIFIED_READY_FOR_R12B_EXACT_MAPPING"
        )
    elif forecast_counts_ok:
        status = (
            "R12A_89_63_FORECAST_UNITS_CONFIRMED_BUT_45_TRACK_SOURCE_NOT_YET_UNIQUELY_IDENTIFIED_REVIEW_CANDIDATES"
        )
    else:
        status = (
            "R12A_FORECAST_ACCOUNTING_MISMATCH_REVIEW_BEFORE_R12B"
        )

    manifest = {
        "step":
            "R12A_DATASET_TRACK_VS_FORECAST_SEGMENT_PROVENANCE_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "known_dataset_level_testreal_tracks":
            KNOWN_DATASET_LEVEL_TESTREAL_TRACKS,

        "forecast_counts_confirmed":
            forecast_counts_ok,

        "authoritative_45_pair_candidate_count":
            int(
                len(
                    exact_45
                )
            ),

        "scientific_guardrail":
            (
                "Do not call the 89/63 forecast units 'dataset tracks' until an exact mapping "
                "to original source-track identities proves equivalence. Use 'forecastable "
                "segments/tracklets' provisionally."
            ),

        "outputs": {
            "forecast_metadata_accounting":
                str(
                    task_path
                ),

            "forecast_units_per_source":
                str(
                    per_source_path
                ),

            "forecast_unit_sample_counts":
                str(
                    unit_path
                ),

            "corrected_csv_inventory":
                str(
                    corrected_inv_path
                ),

            "ranked_original_track_candidates":
                str(
                    ranked_path
                ),

            "candidate_source_track_pair_counts":
                str(
                    candidate_count_path
                ),

            "build_logic_evidence":
                str(
                    logic_path
                ),

            "2s_vs_5s_overlap":
                str(
                    overlap_path
                ),
        },
    }

    (
        OUT
        /
        "R12A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "FORECAST METADATA ACCOUNTING:"
    )

    print(
        task_df.to_string(
            index=False
        )
    )

    print(
        "\nFORECAST UNITS PER SOURCE:"
    )

    print(
        per_source_df.to_string(
            index=False
        )
    )

    print(
        "\n2s vs 5s FORECAST-UNIT OVERLAP:"
    )

    print(
        overlap_df.to_string(
            index=False
        )
    )

    print(
        "\nTOP ORIGINAL-TRACK SOURCE CANDIDATES:"
    )

    if len(
        ranked_inv
    ):
        print(
            ranked_inv[
                [
                    "track_source_relevance_score",
                    "path",
                    "candidate_track_columns",
                    "relevance_reasons",
                ]
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )
    else:
        print(
            "No corrected trajectory CSV candidates found."
        )

    print(
        "\nCANDIDATE SOURCE-TRACK PAIR COUNTS:"
    )

    if len(
        candidate_count_df
    ):
        print(
            candidate_count_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No candidate source-track pair counts could be computed."
        )

    print(
        "\nEXACT 45-PAIR CANDIDATES:"
    )

    if len(
        exact_45
    ):
        print(
            exact_45.to_string(
                index=False
            )
        )
    else:
        print(
            "No exact 45-pair source identified in this audit."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
