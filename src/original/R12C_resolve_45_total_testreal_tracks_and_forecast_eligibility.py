
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R12_ROOT = REVISION_ROOT / "R12_track_accounting"

R12B_DIR = (
    R12_ROOT
    / "R12B_exact_original_track_to_forecast_segment_mapping"
)
R12B_MANIFEST = R12B_DIR / "R12B_MANIFEST.json"

FINAL_NEURAL = MAIN1 / "final_neural_trajectory_dataset"

OUT = (
    R12_ROOT
    / "R12C_resolve_45_total_tracks_and_forecast_eligibility"
)
OUT.mkdir(parents=True, exist_ok=True)

TASK_META = {
    "2s_to_2s":
        FINAL_NEURAL
        / "2s_to_2s_birdsai_test_metadata.csv",

    "2s_to_5s":
        FINAL_NEURAL
        / "2s_to_5s_birdsai_test_metadata.csv",
}

EXPECTED_TOTAL_TESTREAL_TRACKS = 45
EXPECTED_2S_REPRESENTED_TRACKS = 37
EXPECTED_5S_REPRESENTED_TRACKS = 31

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
]

# Avoid very large result dumps / irrelevant folders where possible.
SKIP_TOKENS = [
    "site-packages",
    "__pycache__",
    ".git",
    "transformer_results",
    "runs",
]

MAX_CSV_BYTES = 100_000_000


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


def verify_r12b():
    require(R12B_MANIFEST)

    m = json.loads(
        R12B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    allowed = {
        "R12B_TRACK_ACCOUNTING_NOT_FULLY_RESOLVED_REVIEW_MAPPING_BEFORE_R12C",
        "R12B_45_ORIGINAL_TESTREAL_TRACKS_TO_89_2S_AND_63_5S_FORECAST_SEGMENTS_EXACTLY_RESOLVED_READY_FOR_R12C_BUILD_RULE_PROVENANCE",
    }

    if m.get("status") not in allowed:
        raise RuntimeError(
            f"Unexpected R12-B status: {m.get('status')}"
        )

    return m


def normalize_name(name):
    return str(name).strip().lower()


def candidate_sequence_col(columns):
    preferred = [
        "sequence",
        "source_group",
        "video",
        "source_id",
    ]

    lower = {
        normalize_name(c): c
        for c in columns
    }

    for key in preferred:
        if key in lower:
            return lower[key]

    return None


def candidate_object_col(columns):
    preferred = [
        "object_id",
        "track_id",
        "animal_id",
        "elephant_id",
    ]

    lower = {
        normalize_name(c): c
        for c in columns
    }

    for key in preferred:
        if key in lower:
            return lower[key]

    return None


def candidate_split_col(columns):
    preferred = [
        "split",
        "dataset_split",
        "subset",
        "partition",
        "role",
    ]

    lower = {
        normalize_name(c): c
        for c in columns
    }

    for key in preferred:
        if key in lower:
            return lower[key]

    return None


def is_testreal_series(series):
    s = series.astype(str).str.lower()

    return (
        s.str.contains("test", regex=False)
        |
        s.str.contains("test_real", regex=False)
        |
        s.str.contains("testreal", regex=False)
    )


def load_forecast_track_sets():
    sets = {}
    rows = []

    for task, path in TASK_META.items():
        require(path)

        df = pd.read_csv(
            path,
            dtype={
                "sequence": str,
                "object_id": str,
                "segment_id": str,
                "track_key": str,
                "sample_id": str,
            },
        )

        required = {
            "sequence",
            "object_id",
            "segment_id",
            "track_key",
        }

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{task}: missing metadata columns {sorted(missing)}"
            )

        original_key = (
            df["sequence"].astype(str)
            +
            "|"
            +
            df["object_id"].astype(str)
        )

        track_set = set(
            original_key.unique().tolist()
        )

        sets[task] = track_set

        rows.append(
            {
                "task": task,
                "metadata_path": str(path),
                "metadata_sha256": sha256_file(path),
                "rows": int(len(df)),
                "represented_original_tracks": int(len(track_set)),
                "forecast_segments": int(df["track_key"].nunique()),
            }
        )

    return sets, pd.DataFrame(rows)


def scan_candidate_csvs():
    rows = []

    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.glob("**/*.csv"):
            low = str(p).lower()

            if any(
                token.lower() in low
                for token in SKIP_TOKENS
            ):
                continue

            try:
                if p.stat().st_size > MAX_CSV_BYTES:
                    continue
            except Exception:
                continue

            key = str(p.resolve()).lower()

            if key in seen:
                continue

            seen.add(key)

            try:
                head = pd.read_csv(
                    p,
                    nrows=100,
                    low_memory=False,
                )
            except Exception:
                continue

            seq_col = candidate_sequence_col(
                head.columns
            )

            obj_col = candidate_object_col(
                head.columns
            )

            if seq_col is None or obj_col is None:
                continue

            split_col = candidate_split_col(
                head.columns
            )

            # Load full CSV only after schema passes.
            try:
                df = pd.read_csv(
                    p,
                    low_memory=False,
                    dtype={
                        seq_col: str,
                        obj_col: str,
                    },
                )
            except Exception:
                continue

            if len(df) == 0:
                continue

            work = df.copy()

            test_filter_method = "none"

            if split_col is not None:
                mask = is_testreal_series(
                    work[split_col]
                )

                if mask.any():
                    work = work.loc[mask].copy()
                    test_filter_method = f"{split_col}:contains_test"

            # Some files may already be TestReal-only.
            seq = (
                work[seq_col]
                .astype(str)
                .str.strip()
            )

            obj = (
                work[obj_col]
                .astype(str)
                .str.strip()
            )

            valid = (
                seq.ne("")
                &
                obj.ne("")
                &
                seq.ne("nan")
                &
                obj.ne("nan")
            )

            work = work.loc[valid].copy()

            if len(work) == 0:
                continue

            key_series = (
                work[seq_col].astype(str)
                +
                "|"
                +
                work[obj_col].astype(str)
            )

            unique_tracks = int(
                key_series.nunique()
            )

            unique_sources = int(
                work[seq_col].astype(str).nunique()
            )

            rows.append(
                {
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "rows_total": int(len(df)),
                    "rows_after_test_filter": int(len(work)),
                    "sequence_column": str(seq_col),
                    "object_column": str(obj_col),
                    "split_column": (
                        str(split_col)
                        if split_col is not None
                        else ""
                    ),
                    "test_filter_method": test_filter_method,
                    "unique_sources": unique_sources,
                    "unique_source_object_tracks": unique_tracks,
                    "equals_45": bool(
                        unique_tracks
                        ==
                        EXPECTED_TOTAL_TESTREAL_TRACKS
                    ),
                    "columns": "|".join(
                        str(c)
                        for c in df.columns
                    ),
                }
            )

    return pd.DataFrame(rows)


def rank_candidates(df):
    if len(df) == 0:
        return df

    rows = []

    for _, row in df.iterrows():
        score = 0
        reasons = []

        path = str(row["path"]).lower()
        cols = str(row["columns"]).lower()

        if bool(row["equals_45"]):
            score += 100
            reasons.append("exact_45")

        if "test" in path:
            score += 5
            reasons.append("test_in_path")

        if "birdsai" in path:
            score += 3
            reasons.append("birdsai_in_path")

        if "cleaned_track_points" in path:
            score += 10
            reasons.append("cleaned_track_points")

        if "trajectory" in path:
            score += 3
            reasons.append("trajectory_artifact")

        if "corrected" in path:
            score += 2
            reasons.append("corrected_artifact")

        if "frame" in cols:
            score += 2
            reasons.append("frame_level")

        if "split" in cols:
            score += 2
            reasons.append("explicit_split")

        rec = row.to_dict()
        rec["relevance_score"] = score
        rec["relevance_reasons"] = "|".join(reasons)

        rows.append(rec)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "relevance_score",
                "path",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def extract_track_set_from_candidate(row):
    p = Path(
        str(row["path"])
    )

    seq_col = str(
        row["sequence_column"]
    )

    obj_col = str(
        row["object_column"]
    )

    split_col = str(
        row["split_column"]
    ).strip()

    df = pd.read_csv(
        p,
        low_memory=False,
        dtype={
            seq_col: str,
            obj_col: str,
        },
    )

    if split_col:
        mask = is_testreal_series(
            df[split_col]
        )

        if mask.any():
            df = df.loc[mask].copy()

    seq = df[seq_col].astype(str).str.strip()
    obj = df[obj_col].astype(str).str.strip()

    valid = (
        seq.ne("")
        &
        obj.ne("")
        &
        seq.ne("nan")
        &
        obj.ne("nan")
    )

    df = df.loc[valid].copy()

    df["original_track_key"] = (
        df[seq_col].astype(str)
        +
        "|"
        +
        df[obj_col].astype(str)
    )

    track_table = (
        df[
            [
                seq_col,
                obj_col,
                "original_track_key",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                seq_col: "sequence",
                obj_col: "object_id",
            }
        )
        .sort_values(
            [
                "sequence",
                "object_id",
            ]
        )
        .reset_index(drop=True)
    )

    return df, track_table


def summarize_track_eligibility(
    total_tracks,
    track_sets,
):
    rows = []

    set_2s = track_sets[
        "2s_to_2s"
    ]

    set_5s = track_sets[
        "2s_to_5s"
    ]

    for _, row in total_tracks.iterrows():
        key = str(
            row["original_track_key"]
        )

        in_2s = key in set_2s
        in_5s = key in set_5s

        if in_5s and not in_2s:
            eligibility = "INVALID_5S_WITHOUT_2S"

        elif in_5s:
            eligibility = "ELIGIBLE_2S_AND_5S"

        elif in_2s:
            eligibility = "ELIGIBLE_2S_ONLY"

        else:
            eligibility = "NO_VALID_2S_FORECAST_WINDOWS"

        rows.append(
            {
                "sequence": row["sequence"],
                "object_id": row["object_id"],
                "original_track_key": key,
                "represented_in_2s": in_2s,
                "represented_in_5s": in_5s,
                "eligibility_class": eligibility,
            }
        )

    return pd.DataFrame(rows)


def add_track_length_evidence(
    eligibility_df,
    source_df,
    candidate_row,
):
    seq_col = str(
        candidate_row["sequence_column"]
    )

    obj_col = str(
        candidate_row["object_column"]
    )

    frame_candidates = [
        c
        for c in source_df.columns
        if "frame" in str(c).lower()
    ]

    frame_col = None

    preferred = [
        "frame_index",
        "frame",
        "frame_id",
        "frame_number",
    ]

    lower = {
        str(c).lower(): c
        for c in frame_candidates
    }

    for p in preferred:
        if p in lower:
            frame_col = lower[p]
            break

    rows = []

    for _, tr in eligibility_df.iterrows():
        seq = str(
            tr["sequence"]
        )

        obj = str(
            tr["object_id"]
        )

        g = source_df[
            (
                source_df[seq_col].astype(str)
                ==
                seq
            )
            &
            (
                source_df[obj_col].astype(str)
                ==
                obj
            )
        ].copy()

        rec = tr.to_dict()
        rec["source_rows"] = int(
            len(g)
        )

        if frame_col is not None and len(g):
            vals = pd.to_numeric(
                g[frame_col],
                errors="coerce"
            ).dropna()

            if len(vals):
                rec["frame_column"] = str(frame_col)
                rec["min_frame"] = float(
                    vals.min()
                )
                rec["max_frame"] = float(
                    vals.max()
                )
                rec["unique_frames"] = int(
                    vals.nunique()
                )
                rec["frame_span"] = float(
                    vals.max()
                    -
                    vals.min()
                    +
                    1
                )

                diffs = (
                    np.diff(
                        np.sort(
                            vals.unique()
                        )
                    )
                    if vals.nunique() > 1
                    else np.array([])
                )

                if len(diffs):
                    rec["max_frame_gap"] = float(
                        diffs.max()
                    )
                    rec["median_frame_gap"] = float(
                        np.median(
                            diffs
                        )
                    )
                else:
                    rec["max_frame_gap"] = None
                    rec["median_frame_gap"] = None

            else:
                rec["frame_column"] = str(frame_col)
                rec["min_frame"] = None
                rec["max_frame"] = None
                rec["unique_frames"] = None
                rec["frame_span"] = None
                rec["max_frame_gap"] = None
                rec["median_frame_gap"] = None

        else:
            rec["frame_column"] = (
                str(frame_col)
                if frame_col is not None
                else ""
            )
            rec["min_frame"] = None
            rec["max_frame"] = None
            rec["unique_frames"] = None
            rec["frame_span"] = None
            rec["max_frame_gap"] = None
            rec["median_frame_gap"] = None

        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    print(
        "\nR12-C — RESOLVE 45 TOTAL TESTREAL TRACKS AND FORECAST ELIGIBILITY"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R12-B showed that forecasting metadata contains 37 original tracks at 2s and "
        "31 at 5s, not 45. This step searches the upstream BIRDSAI TestReal trajectory "
        "artifacts for an authoritative 45-track population, then maps every total track "
        "to 2s/5s forecasting eligibility. The scientific goal is to distinguish total "
        "dataset tracks from forecast-eligible tracks and from forecastable segments.\n"
    )

    verify_r12b()

    track_sets, forecast_lock_df = load_forecast_track_sets()

    candidates = scan_candidate_csvs()
    ranked = rank_candidates(
        candidates
    )

    exact45 = (
        ranked[
            ranked["equals_45"] == True
        ].copy()
        if len(ranked)
        else pd.DataFrame()
    )

    if len(exact45) == 0:
        status = (
            "R12C_NO_AUTHORITATIVE_45_TRACK_SOURCE_FOUND_NEEDS_TARGETED_SOURCE_RECOVERY"
        )

        # Save what we have before exiting normally.
        forecast_lock_path = (
            OUT
            / "R12C_FORECAST_REPRESENTATION_LOCK.csv"
        )

        candidate_path = (
            OUT
            / "R12C_UPSTREAM_TRACK_SOURCE_CANDIDATES.csv"
        )

        exact_path = (
            OUT
            / "R12C_EXACT_45_TRACK_SOURCE_CANDIDATES.csv"
        )

        forecast_lock_df.to_csv(
            forecast_lock_path,
            index=False
        )

        ranked.to_csv(
            candidate_path,
            index=False
        )

        exact45.to_csv(
            exact_path,
            index=False
        )

        manifest = {
            "step":
                "R12C_RESOLVE_45_TOTAL_TESTREAL_TRACKS_AND_FORECAST_ELIGIBILITY",

            "created_utc":
                utc_now(),

            "status":
                status,

            "training":
                False,

            "inference":
                False,

            "forecast_2s_original_tracks":
                len(
                    track_sets[
                        "2s_to_2s"
                    ]
                ),

            "forecast_5s_original_tracks":
                len(
                    track_sets[
                        "2s_to_5s"
                    ]
                ),

            "exact_45_candidate_count":
                0,

            "outputs": {
                "forecast_lock":
                    str(
                        forecast_lock_path
                    ),

                "candidate_sources":
                    str(
                        candidate_path
                    ),

                "exact_45_sources":
                    str(
                        exact_path
                    ),
            },
        }

        (
            OUT
            / "R12C_MANIFEST.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2
            ),
            encoding="utf-8"
        )

        print(
            "FORECAST REPRESENTATION LOCK:"
        )

        print(
            forecast_lock_df.to_string(
                index=False
            )
        )

        print(
            "\nTOP UPSTREAM TRACK SOURCE CANDIDATES:"
        )

        if len(ranked):
            print(
                ranked.head(
                    30
                ).to_string(
                    index=False
                )
            )
        else:
            print(
                "No candidate upstream CSVs found."
            )

        print(
            "\nEXACT 45-TRACK SOURCE CANDIDATES:"
        )

        print(
            "None found."
        )

        print(
            "\nSTATUS:"
        )

        print(
            status
        )

        print(
            "\nR12-C COMPLETE"
        )

        print(
            "Outputs:",
            OUT
        )

        return

    # Prefer highest-ranked exact-45 source.
    chosen = exact45.iloc[0]

    source_df, total_track_table = extract_track_set_from_candidate(
        chosen
    )

    total_track_count = int(
        len(
            total_track_table
        )
    )

    if (
        total_track_count
        !=
        EXPECTED_TOTAL_TESTREAL_TRACKS
    ):
        raise RuntimeError(
            f"Chosen authoritative source produced {total_track_count} tracks, expected 45."
        )

    eligibility_df = summarize_track_eligibility(
        total_track_table,
        track_sets,
    )

    detailed_df = add_track_length_evidence(
        eligibility_df,
        source_df,
        chosen,
    )

    class_summary = (
        eligibility_df.groupby(
            "eligibility_class",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "tracks"
            }
        )
    )

    represented_2s = int(
        eligibility_df[
            "represented_in_2s"
        ].sum()
    )

    represented_5s = int(
        eligibility_df[
            "represented_in_5s"
        ].sum()
    )

    no_2s = int(
        (
            eligibility_df[
                "eligibility_class"
            ]
            ==
            "NO_VALID_2S_FORECAST_WINDOWS"
        ).sum()
    )

    only_2s = int(
        (
            eligibility_df[
                "eligibility_class"
            ]
            ==
            "ELIGIBLE_2S_ONLY"
        ).sum()
    )

    both = int(
        (
            eligibility_df[
                "eligibility_class"
            ]
            ==
            "ELIGIBLE_2S_AND_5S"
        ).sum()
    )

    if (
        represented_2s
        ==
        EXPECTED_2S_REPRESENTED_TRACKS
        and
        represented_5s
        ==
        EXPECTED_5S_REPRESENTED_TRACKS
        and
        no_2s
        ==
        (
            EXPECTED_TOTAL_TESTREAL_TRACKS
            -
            EXPECTED_2S_REPRESENTED_TRACKS
        )
        and
        only_2s
        ==
        (
            EXPECTED_2S_REPRESENTED_TRACKS
            -
            EXPECTED_5S_REPRESENTED_TRACKS
        )
        and
        both
        ==
        EXPECTED_5S_REPRESENTED_TRACKS
    ):
        status = (
            "R12C_45_TOTAL_TESTREAL_TRACKS_RESOLVED_37_2S_ELIGIBLE_31_5S_ELIGIBLE_READY_FOR_R12D_SEGMENT_BUILD_RULE_LOCK"
        )
    else:
        status = (
            "R12C_45_SOURCE_FOUND_BUT_ELIGIBILITY_COUNTS_DO_NOT_MATCH_EXPECTATION_REVIEW"
        )

    # Save outputs.
    forecast_lock_path = (
        OUT
        / "R12C_FORECAST_REPRESENTATION_LOCK.csv"
    )

    candidate_path = (
        OUT
        / "R12C_UPSTREAM_TRACK_SOURCE_CANDIDATES.csv"
    )

    exact_path = (
        OUT
        / "R12C_EXACT_45_TRACK_SOURCE_CANDIDATES.csv"
    )

    chosen_path = (
        OUT
        / "R12C_CHOSEN_AUTHORITATIVE_45_TRACK_SOURCE.csv"
    )

    total_track_path = (
        OUT
        / "R12C_TOTAL_TESTREAL_45_TRACKS.csv"
    )

    eligibility_path = (
        OUT
        / "R12C_TRACK_FORECAST_ELIGIBILITY.csv"
    )

    detailed_path = (
        OUT
        / "R12C_TRACK_FORECAST_ELIGIBILITY_WITH_LENGTH_EVIDENCE.csv"
    )

    class_summary_path = (
        OUT
        / "R12C_ELIGIBILITY_CLASS_SUMMARY.csv"
    )

    forecast_lock_df.to_csv(
        forecast_lock_path,
        index=False
    )

    ranked.to_csv(
        candidate_path,
        index=False
    )

    exact45.to_csv(
        exact_path,
        index=False
    )

    pd.DataFrame(
        [chosen.to_dict()]
    ).to_csv(
        chosen_path,
        index=False
    )

    total_track_table.to_csv(
        total_track_path,
        index=False
    )

    eligibility_df.to_csv(
        eligibility_path,
        index=False
    )

    detailed_df.to_csv(
        detailed_path,
        index=False
    )

    class_summary.to_csv(
        class_summary_path,
        index=False
    )

    manuscript = f"""R12 track-accounting population resolution

Authoritative TestReal population
---------------------------------
The upstream BIRDSAI TestReal trajectory source contains {total_track_count} unique original tracks, where original track identity is defined as (sequence, object_id).

Forecast eligibility
--------------------
Of these {total_track_count} total original tracks:

- {both} tracks contribute valid forecasting windows to both the 2 s and 5 s tasks.
- {only_2s} additional tracks contribute valid 2 s windows but not valid 5 s windows.
- {no_2s} tracks do not contribute a valid 2 s forecasting window.

Therefore:

- total TestReal original tracks = {total_track_count}
- 2 s represented original tracks = {represented_2s}
- 5 s represented original tracks = {represented_5s}

Combined with R12-B:

- {represented_2s} 2-s represented original tracks generate 89 forecastable segments/tracklets.
- {represented_5s} 5-s represented original tracks generate 63 forecastable segments/tracklets.

Reviewer-safe wording
---------------------
Use:

"The BIRDSAI TestReal split contains 45 original tracks. After forecasting-window eligibility filtering, 37 original tracks contribute to the 2-s task and 31 contribute to the 5-s task. Continuity-based segmentation of these eligible trajectories yields 89 forecastable 2-s segments and 63 forecastable 5-s segments."

Do not write:

"TestReal contains 89 tracks."

Remaining task
--------------
R12-D must still lock the exact continuity/gap/build rule that creates segment_id from an original (sequence, object_id) trajectory.
"""

    manuscript_path = (
        OUT
        / "R12C_MANUSCRIPT_READY_45_TO_37_TO_31_TRACK_RESOLUTION.txt"
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8"
    )

    manifest = {
        "step":
            "R12C_RESOLVE_45_TOTAL_TESTREAL_TRACKS_AND_FORECAST_ELIGIBILITY",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "authoritative_45_source":
            str(
                chosen["path"]
            ),

        "authoritative_45_source_sha256":
            str(
                chosen["sha256"]
            ),

        "total_testreal_original_tracks":
            total_track_count,

        "represented_original_tracks_2s":
            represented_2s,

        "represented_original_tracks_5s":
            represented_5s,

        "tracks_both_2s_and_5s":
            both,

        "tracks_2s_only":
            only_2s,

        "tracks_no_valid_2s_window":
            no_2s,

        "remaining_requirement":
            (
                "R12-D must lock the exact continuity/gap/segment construction rule."
            ),

        "outputs": {
            "forecast_lock":
                str(
                    forecast_lock_path
                ),

            "candidate_sources":
                str(
                    candidate_path
                ),

            "exact_45_sources":
                str(
                    exact_path
                ),

            "chosen_45_source":
                str(
                    chosen_path
                ),

            "total_45_tracks":
                str(
                    total_track_path
                ),

            "track_eligibility":
                str(
                    eligibility_path
                ),

            "track_eligibility_with_length":
                str(
                    detailed_path
                ),

            "eligibility_summary":
                str(
                    class_summary_path
                ),

            "manuscript_ready_resolution":
                str(
                    manuscript_path
                ),
        },
    }

    (
        OUT
        / "R12C_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "FORECAST REPRESENTATION LOCK:"
    )

    print(
        forecast_lock_df.to_string(
            index=False
        )
    )

    print(
        "\nCHOSEN AUTHORITATIVE 45-TRACK SOURCE:"
    )

    print(
        pd.DataFrame(
            [chosen.to_dict()]
        )[
            [
                "path",
                "sha256",
                "rows_total",
                "rows_after_test_filter",
                "sequence_column",
                "object_column",
                "split_column",
                "unique_sources",
                "unique_source_object_tracks",
                "relevance_score",
                "relevance_reasons",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nELIGIBILITY CLASS SUMMARY:"
    )

    print(
        class_summary.to_string(
            index=False
        )
    )

    print(
        "\nTRACK ACCOUNTING RESOLUTION:"
    )

    print(
        f"  Total TestReal original tracks: {total_track_count}"
    )

    print(
        f"  Represented in 2s forecasting: {represented_2s}"
    )

    print(
        f"  Represented in 5s forecasting: {represented_5s}"
    )

    print(
        f"  2s and 5s eligible: {both}"
    )

    print(
        f"  2s-only eligible: {only_2s}"
    )

    print(
        f"  No valid 2s forecast window: {no_2s}"
    )

    print(
        "\nTRACKS NOT REPRESENTED IN 2s:"
    )

    print(
        detailed_df[
            detailed_df[
                "represented_in_2s"
            ]
            ==
            False
        ].to_string(
            index=False
        )
    )

    print(
        "\nTRACKS REPRESENTED IN 2s BUT NOT 5s:"
    )

    print(
        detailed_df[
            (
                detailed_df[
                    "represented_in_2s"
                ]
                ==
                True
            )
            &
            (
                detailed_df[
                    "represented_in_5s"
                ]
                ==
                False
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-C COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
