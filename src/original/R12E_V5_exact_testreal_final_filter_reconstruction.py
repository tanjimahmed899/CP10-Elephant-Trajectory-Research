
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R12_ROOT = REVISION_ROOT / "R12_track_accounting"

R12E_V4_DIR = (
    R12_ROOT
    / "R12E_v4_post_segmentation_filter_audit"
)
R12E_V4_MANIFEST = R12E_V4_DIR / "R12E_V4_MANIFEST.json"

CORR_2 = (
    MAIN1
    / "corrected_trajectory_dataset"
    / "corrected_windows_2s_to_2s.csv"
)

CORR_5 = (
    MAIN1
    / "corrected_trajectory_dataset"
    / "corrected_windows_2s_to_5s.csv"
)

FINAL_2 = (
    MAIN1
    / "final_neural_trajectory_dataset"
    / "2s_to_2s_birdsai_test_metadata.csv"
)

FINAL_5 = (
    MAIN1
    / "final_neural_trajectory_dataset"
    / "2s_to_5s_birdsai_test_metadata.csv"
)

FINAL_BUILDER = MAIN1 / "build_final_neural_dataset.py"
CORRECTOR = MAIN1 / "build_corrected_trajectories.py"

OUT = (
    R12_ROOT
    / "R12E_v5_exact_testreal_filter_rule_reconstruction"
)
OUT.mkdir(parents=True, exist_ok=True)


EXPECTED_FINAL = {
    "2s_to_2s": {
        "rows": 1053,
        "tracks": 37,
        "segments": 89,
    },
    "2s_to_5s": {
        "rows": 826,
        "tracks": 31,
        "segments": 63,
    },
}


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


def verify_v4():
    require(R12E_V4_MANIFEST)

    m = json.loads(
        R12E_V4_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R12E_V4_POST_SEGMENTATION_FILTER_STAGE_AND_REMOVED_WINDOWS_IDENTIFIED_READY_FOR_R12E_V5_EXACT_FILTER_RULE_RECONSTRUCTION"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R12-E-V4 status: {m.get('status')}"
        )

    return m


def canon_numeric_id(value):
    if pd.isna(value):
        return ""

    s = str(value).strip()

    if not s:
        return ""

    try:
        x = float(s)
        if np.isfinite(x) and x.is_integer():
            return str(int(x))
    except Exception:
        pass

    return s


def normalize(df):
    out = df.copy()

    if "sequence" in out.columns:
        out["sequence_canon"] = (
            out["sequence"]
            .astype(str)
            .str.strip()
        )

    if "object_id" in out.columns:
        out["object_id_canon"] = (
            out["object_id"]
            .map(canon_numeric_id)
        )

    if "segment_id" in out.columns:
        # segment_id can be numeric or a provenance string.
        out["segment_id_canon"] = (
            out["segment_id"]
            .astype(str)
            .str.strip()
        )

    for c in [
        "obs_start_frame",
        "obs_end_frame",
        "future_end_frame",
    ]:
        if c in out.columns:
            out[c + "_canon"] = (
                pd.to_numeric(
                    out[c],
                    errors="coerce",
                )
                .map(
                    lambda x:
                        ""
                        if pd.isna(x)
                        else str(int(x))
                )
            )

    if {
        "sequence_canon",
        "object_id_canon",
    }.issubset(out.columns):
        out["original_track_key_canon"] = (
            out["sequence_canon"]
            + "|"
            + out["object_id_canon"]
        )

    # If downstream track_key exists, keep as independent segment identity.
    if "track_key" in out.columns:
        out["track_key_canon"] = (
            out["track_key"]
            .astype(str)
            .str.strip()
        )

    return out


def is_testreal(df):
    if "split" not in df.columns:
        raise RuntimeError(
            "Corrected window file has no split column."
        )

    s = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    exact = s.isin(
        [
            "test_real",
            "testreal",
            "test-real",
        ]
    )

    if exact.any():
        return exact

    return s.str.contains(
        "test",
        regex=False,
    )


def composite_key(df):
    required = [
        "sequence_canon",
        "object_id_canon",
        "segment_id_canon",
        "obs_start_frame_canon",
        "future_end_frame_canon",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Cannot build composite identity; missing {missing}"
        )

    key = df[required[0]].astype(str)

    for c in required[1:]:
        key = key + "|" + df[c].astype(str)

    return key


def stage_stats(df):
    rec = {
        "rows": int(len(df)),
    }

    if "original_track_key_canon" in df.columns:
        rec["tracks"] = int(
            df["original_track_key_canon"].nunique()
        )

    if "track_key_canon" in df.columns:
        rec["segments_track_key"] = int(
            df["track_key_canon"].nunique()
        )

    if {
        "sequence_canon",
        "object_id_canon",
        "segment_id_canon",
    }.issubset(df.columns):
        seg = (
            df["sequence_canon"].astype(str)
            + "|"
            + df["object_id_canon"].astype(str)
            + "|"
            + df["segment_id_canon"].astype(str)
        )
        rec["segments_composite"] = int(
            seg.nunique()
        )

    return rec


def source_context(path, terms):
    require(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    rows = []

    for i, line in enumerate(
        lines,
        start=1,
    ):
        ll = line.lower()

        matched = [
            t
            for t in terms
            if t.lower() in ll
        ]

        if not matched:
            continue

        start = max(
            1,
            i - 6,
        )

        end = min(
            len(lines),
            i + 6,
        )

        rows.append(
            {
                "source_file":
                    str(path),

                "source_sha256":
                    sha256_file(path),

                "anchor_line":
                    i,

                "matched_terms":
                    "|".join(matched),

                "anchor_text":
                    line,

                "context_start":
                    start,

                "context_end":
                    end,

                "context":
                    "\n".join(
                        f"{j}: {lines[j-1]}"
                        for j in range(
                            start,
                            end + 1,
                        )
                    ),
            }
        )

    return pd.DataFrame(rows)


def candidate_filters(test_corr):
    """
    Evaluate candidate downstream filters without assuming which one is correct.
    """
    candidates = []

    def add(name, mask):
        subset = test_corr.loc[
            mask
        ].copy()

        candidates.append(
            (
                name,
                subset,
            )
        )

    all_mask = pd.Series(
        True,
        index=test_corr.index,
    )

    add(
        "testreal_all",
        all_mask,
    )

    if "accepted" in test_corr.columns:
        accepted = (
            pd.to_numeric(
                test_corr["accepted"],
                errors="coerce",
            )
            ==
            1
        )

        add(
            "testreal_accepted_1",
            accepted,
        )

    else:
        accepted = all_mask

    if "quality_tier" in test_corr.columns:
        core = (
            test_corr["quality_tier"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            "core"
        )

        add(
            "testreal_quality_core",
            core,
        )

    else:
        core = all_mask

    add(
        "testreal_accepted_1_and_core",
        accepted
        &
        core,
    )

    # Optional threshold candidates for repaired_motion_pct,
    # tested only diagnostically.
    if "repaired_motion_pct" in test_corr.columns:
        rpm = pd.to_numeric(
            test_corr["repaired_motion_pct"],
            errors="coerce",
        )

        for threshold in [
            5.0,
            10.0,
            20.0,
        ]:
            add(
                f"testreal_accepted_1_and_repaired_le_{threshold:g}",
                accepted
                &
                rpm.notna()
                &
                (
                    rpm
                    <=
                    threshold
                ),
            )

    return candidates


def compare_to_final(
    task,
    candidate_name,
    candidate,
    final_df,
):
    c = candidate.copy()
    f = final_df.copy()

    c["_physical_key"] = composite_key(
        c
    )

    f["_physical_key"] = composite_key(
        f
    )

    cset = set(
        c["_physical_key"].astype(str)
    )

    fset = set(
        f["_physical_key"].astype(str)
    )

    only_candidate = cset - fset
    only_final = fset - cset

    stats = stage_stats(
        c
    )

    row = {
        "task":
            task,

        "candidate_filter":
            candidate_name,

        "candidate_rows":
            int(
                len(
                    c
                )
            ),

        "final_rows":
            int(
                len(
                    f
                )
            ),

        "candidate_unique_physical_windows":
            int(
                len(
                    cset
                )
            ),

        "final_unique_physical_windows":
            int(
                len(
                    fset
                )
            ),

        "candidate_tracks":
            stats.get(
                "tracks",
                np.nan,
            ),

        "candidate_segments":
            stats.get(
                "segments_composite",
                np.nan,
            ),

        "only_candidate":
            int(
                len(
                    only_candidate
                )
            ),

        "only_final":
            int(
                len(
                    only_final
                )
            ),

        "exact_window_set_match":
            bool(
                cset
                ==
                fset
            ),
    }

    extras = c[
        c["_physical_key"].isin(
            only_candidate
        )
    ].copy()

    missing = f[
        f["_physical_key"].isin(
            only_final
        )
    ].copy()

    return row, extras, missing


def final_rule_evidence_score(evidence):
    if len(evidence) == 0:
        return {
            "has_split": False,
            "has_accepted": False,
            "has_quality_tier": False,
            "has_core": False,
        }

    text = "\n".join(
        evidence["context"]
        .astype(str)
        .tolist()
    ).lower()

    return {
        "has_split":
            "split" in text,

        "has_accepted":
            "accepted" in text,

        "has_quality_tier":
            "quality_tier" in text,

        "has_core":
            "core" in text,
    }


def main():
    print(
        "\nR12-E-V5 — EXACT TESTREAL FINAL-FILTER RULE RECONSTRUCTION"
    )

    print(
        "NO training and NO inference. "
        "This step fixes the V4 comparison issue by never using re-indexed sample_id "
        "as cross-stage identity. It restricts corrected windows to TestReal, compares "
        "candidate final filters to frozen final metadata using the physical window key "
        "(sequence, object_id, segment_id, obs_start_frame, future_end_frame), and "
        "extracts the exact filter evidence from build_final_neural_dataset.py. "
        "No original file is modified.\n"
    )

    verify_v4()

    for p in [
        CORR_2,
        CORR_5,
        FINAL_2,
        FINAL_5,
        FINAL_BUILDER,
        CORRECTOR,
    ]:
        require(p)

    corr2 = normalize(
        pd.read_csv(
            CORR_2,
            low_memory=False,
        )
    )

    corr5 = normalize(
        pd.read_csv(
            CORR_5,
            low_memory=False,
        )
    )

    final2 = normalize(
        pd.read_csv(
            FINAL_2,
            low_memory=False,
        )
    )

    final5 = normalize(
        pd.read_csv(
            FINAL_5,
            low_memory=False,
        )
    )

    test2 = corr2.loc[
        is_testreal(
            corr2
        )
    ].copy()

    test5 = corr5.loc[
        is_testreal(
            corr5
        )
    ].copy()

    baseline_rows = []

    for task, test_corr, final_df in [
        (
            "2s_to_2s",
            test2,
            final2,
        ),
        (
            "2s_to_5s",
            test5,
            final5,
        ),
    ]:
        a = stage_stats(
            test_corr
        )

        b = stage_stats(
            final_df
        )

        baseline_rows.extend(
            [
                {
                    "task":
                        task,

                    "stage":
                        "corrected_testreal_before_final_filter",

                    **a,
                },
                {
                    "task":
                        task,

                    "stage":
                        "frozen_final_metadata",

                    **b,
                },
            ]
        )

    baseline_df = pd.DataFrame(
        baseline_rows
    )

    candidate_rows = []
    extras_frames = []
    missing_frames = []

    exact_candidates = []

    for task, test_corr, final_df in [
        (
            "2s_to_2s",
            test2,
            final2,
        ),
        (
            "2s_to_5s",
            test5,
            final5,
        ),
    ]:
        for name, subset in candidate_filters(
            test_corr
        ):
            row, extras, missing = compare_to_final(
                task,
                name,
                subset,
                final_df,
            )

            candidate_rows.append(
                row
            )

            if row[
                "exact_window_set_match"
            ]:
                exact_candidates.append(
                    (
                        task,
                        name,
                    )
                )

            if len(extras):
                extras["_task"] = task
                extras["_candidate_filter"] = name
                extras_frames.append(
                    extras
                )

            if len(missing):
                missing["_task"] = task
                missing["_candidate_filter"] = name
                missing_frames.append(
                    missing
                )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    extras_df = (
        pd.concat(
            extras_frames,
            ignore_index=True,
            sort=False,
        )
        if extras_frames
        else
        pd.DataFrame()
    )

    missing_df = (
        pd.concat(
            missing_frames,
            ignore_index=True,
            sort=False,
        )
        if missing_frames
        else
        pd.DataFrame()
    )

    final_evidence = source_context(
        FINAL_BUILDER,
        [
            "split",
            "accepted",
            "quality_tier",
            "core",
            "test_real",
            "testreal",
            "filter",
            "continue",
        ],
    )

    corrector_evidence = source_context(
        CORRECTOR,
        [
            "accepted",
            "quality_tier",
            "core",
            "extended",
            "reject_quality",
            "invalid_motion",
            "repaired_motion_pct",
            "20",
            "10",
        ],
    )

    source_score = final_rule_evidence_score(
        final_evidence
    )

    # We require the same candidate to match both tasks exactly.
    exact_by_name = {}

    for task, name in exact_candidates:
        exact_by_name.setdefault(
            name,
            set(),
        ).add(
            task
        )

    both_tasks = [
        name
        for name, tasks in exact_by_name.items()
        if tasks
        ==
        {
            "2s_to_2s",
            "2s_to_5s",
        }
    ]

    exact_rule_name = (
        both_tasks[0]
        if len(
            both_tasks
        )
        ==
        1
        else ""
    )

    # Track the TestReal windows removed by the exact rule.
    exact_removed_frames = []

    if exact_rule_name:
        for task, test_corr, final_df in [
            (
                "2s_to_2s",
                test2,
                final2,
            ),
            (
                "2s_to_5s",
                test5,
                final5,
            ),
        ]:
            subset_dict = {
                name: subset
                for name, subset in candidate_filters(
                    test_corr
                )
            }

            chosen = subset_dict[
                exact_rule_name
            ].copy()

            all_test = test_corr.copy()

            all_test[
                "_physical_key"
            ] = composite_key(
                all_test
            )

            chosen[
                "_physical_key"
            ] = composite_key(
                chosen
            )

            removed_keys = (
                set(
                    all_test[
                        "_physical_key"
                    ].astype(str)
                )
                -
                set(
                    chosen[
                        "_physical_key"
                    ].astype(str)
                )
            )

            removed = all_test[
                all_test[
                    "_physical_key"
                ].isin(
                    removed_keys
                )
            ].copy()

            removed["_task"] = task
            removed["_exact_filter_rule"] = exact_rule_name

            exact_removed_frames.append(
                removed
            )

    exact_removed_df = (
        pd.concat(
            exact_removed_frames,
            ignore_index=True,
            sort=False,
        )
        if exact_removed_frames
        else
        pd.DataFrame()
    )

    # Summarize exact removed TestReal windows by reason.
    reason_rows = []

    if len(exact_removed_df):
        for task, g in exact_removed_df.groupby(
            "_task"
        ):
            for col in [
                "accepted",
                "quality_tier",
                "repaired_motion_pct",
            ]:
                if col not in g.columns:
                    continue

                if col == "repaired_motion_pct":
                    # Numeric descriptive summary rather than giant value count.
                    vals = pd.to_numeric(
                        g[col],
                        errors="coerce",
                    )

                    reason_rows.append(
                        {
                            "task":
                                task,

                            "column":
                                col,

                            "value":
                                "numeric_summary",

                            "count":
                                int(
                                    vals.notna().sum()
                                ),

                            "min":
                                float(
                                    vals.min()
                                )
                                if vals.notna().any()
                                else np.nan,

                            "median":
                                float(
                                    vals.median()
                                )
                                if vals.notna().any()
                                else np.nan,

                            "max":
                                float(
                                    vals.max()
                                )
                                if vals.notna().any()
                                else np.nan,
                        }
                    )

                else:
                    vc = (
                        g[col]
                        .astype(str)
                        .value_counts(
                            dropna=False
                        )
                    )

                    for value, count in vc.items():
                        reason_rows.append(
                            {
                                "task":
                                    task,

                                "column":
                                    col,

                                "value":
                                    value,

                                "count":
                                    int(
                                        count
                                    ),

                                "min":
                                    np.nan,

                                "median":
                                    np.nan,

                                "max":
                                    np.nan,
                            }
                        )

    reason_df = pd.DataFrame(
        reason_rows
    )

    # Validate final counts.
    final_count_ok = True

    for task, final_df in [
        (
            "2s_to_2s",
            final2,
        ),
        (
            "2s_to_5s",
            final5,
        ),
    ]:
        stats = stage_stats(
            final_df
        )

        exp = EXPECTED_FINAL[
            task
        ]

        segment_count = (
            stats.get(
                "segments_track_key",
                stats.get(
                    "segments_composite",
                    None,
                ),
            )
        )

        if not (
            len(
                final_df
            )
            ==
            exp[
                "rows"
            ]
            and
            stats[
                "tracks"
            ]
            ==
            exp[
                "tracks"
            ]
            and
            segment_count
            ==
            exp[
                "segments"
            ]
        ):
            final_count_ok = False

    source_rule_support = bool(
        source_score[
            "has_accepted"
        ]
        and
        source_score[
            "has_quality_tier"
        ]
        and
        source_score[
            "has_core"
        ]
    )

    if (
        exact_rule_name
        and
        final_count_ok
        and
        source_rule_support
    ):
        status = (
            "R12E_V5_TESTREAL_FINAL_FILTER_EXACTLY_RECONSTRUCTED_AND_SOURCE_LOCKED_READY_FOR_R12F_FINAL_TRACK_ACCOUNTING_LOCK"
        )

    elif exact_rule_name and final_count_ok:
        status = (
            "R12E_V5_EXACT_DATA_FILTER_MATCH_FOUND_BUT_SOURCE_EVIDENCE_NEEDS_REVIEW"
        )

    else:
        status = (
            "R12E_V5_FINAL_FILTER_NOT_YET_EXACTLY_RESOLVED_REVIEW_CANDIDATES"
        )

    # Save outputs.
    baseline_path = (
        OUT
        /
        "R12E_V5_TESTREAL_STAGE_COUNTS.csv"
    )

    candidate_path = (
        OUT
        /
        "R12E_V5_FILTER_CANDIDATE_EXACT_MATCH_TABLE.csv"
    )

    exact_removed_path = (
        OUT
        /
        "R12E_V5_EXACT_FILTER_REMOVED_TESTREAL_WINDOWS.csv"
    )

    reason_path = (
        OUT
        /
        "R12E_V5_EXACT_FILTER_REMOVED_REASON_SUMMARY.csv"
    )

    final_evidence_path = (
        OUT
        /
        "R12E_V5_FINAL_BUILDER_FILTER_SOURCE_EVIDENCE.csv"
    )

    corrector_evidence_path = (
        OUT
        /
        "R12E_V5_CORRECTOR_QUALITY_RULE_SOURCE_EVIDENCE.csv"
    )

    extras_path = (
        OUT
        /
        "R12E_V5_CANDIDATE_EXTRA_WINDOWS.csv"
    )

    missing_path = (
        OUT
        /
        "R12E_V5_CANDIDATE_MISSING_WINDOWS.csv"
    )

    baseline_df.to_csv(
        baseline_path,
        index=False,
    )

    candidate_df.to_csv(
        candidate_path,
        index=False,
    )

    exact_removed_df.to_csv(
        exact_removed_path,
        index=False,
    )

    reason_df.to_csv(
        reason_path,
        index=False,
    )

    final_evidence.to_csv(
        final_evidence_path,
        index=False,
    )

    corrector_evidence.to_csv(
        corrector_evidence_path,
        index=False,
    )

    extras_df.to_csv(
        extras_path,
        index=False,
    )

    missing_df.to_csv(
        missing_path,
        index=False,
    )

    manifest = {
        "step":
            "R12E_V5_EXACT_TESTREAL_FINAL_FILTER_RECONSTRUCTION",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "cross_stage_identity":
            (
                "sequence + object_id + segment_id + "
                "obs_start_frame + future_end_frame"
            ),

        "sample_id_used_for_cross_stage_matching":
            False,

        "exact_rule_name":
            exact_rule_name,

        "exact_rule_matches_both_tasks":
            bool(
                exact_rule_name
            ),

        "final_count_lock_pass":
            final_count_ok,

        "source_rule_support":
            source_rule_support,

        "outputs": {
            "testreal_stage_counts":
                str(
                    baseline_path
                ),

            "filter_candidate_table":
                str(
                    candidate_path
                ),

            "exact_removed_testreal_windows":
                str(
                    exact_removed_path
                ),

            "removed_reason_summary":
                str(
                    reason_path
                ),

            "final_builder_source_evidence":
                str(
                    final_evidence_path
                ),

            "corrector_source_evidence":
                str(
                    corrector_evidence_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_V5_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "TESTREAL STAGE COUNTS:"
    )

    print(
        baseline_df.to_string(
            index=False
        )
    )

    print(
        "\nFILTER CANDIDATE EXACT-MATCH TABLE:"
    )

    print(
        candidate_df.to_string(
            index=False
        )
    )

    print(
        "\nEXACT FILTER RULE:"
    )

    print(
        exact_rule_name
        if exact_rule_name
        else
        "No unique candidate matched both tasks exactly."
    )

    print(
        "\nEXACT FILTER REMOVED TESTREAL REASON SUMMARY:"
    )

    if len(
        reason_df
    ):
        print(
            reason_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No exact removed-window summary available."
        )

    print(
        "\nFINAL BUILDER SOURCE RULE SUPPORT:"
    )

    print(
        json.dumps(
            source_score,
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
        "\nR12-E-V5 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
