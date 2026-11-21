
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

R12E_V3_DIR = (
    R12_ROOT
    / "R12E_v3_robust_segment_rule_reconstruction"
)
R12E_V3_MANIFEST = R12E_V3_DIR / "R12E_V3_MANIFEST.json"

RAW_WINDOWS_2 = (
    MAIN1
    / "trajectory_dataset"
    / "windows_2s_to_2s.csv"
)

RAW_WINDOWS_5 = (
    MAIN1
    / "trajectory_dataset"
    / "windows_2s_to_5s.csv"
)

CORR_WINDOWS_2 = (
    MAIN1
    / "corrected_trajectory_dataset"
    / "corrected_windows_2s_to_2s.csv"
)

CORR_WINDOWS_5 = (
    MAIN1
    / "corrected_trajectory_dataset"
    / "corrected_windows_2s_to_5s.csv"
)

FINAL_META_2 = (
    MAIN1
    / "final_neural_trajectory_dataset"
    / "2s_to_2s_birdsai_test_metadata.csv"
)

FINAL_META_5 = (
    MAIN1
    / "final_neural_trajectory_dataset"
    / "2s_to_5s_birdsai_test_metadata.csv"
)

CORRECTOR = MAIN1 / "build_corrected_trajectories.py"
FINAL_BUILDER = MAIN1 / "build_final_neural_dataset.py"

OUT = (
    R12_ROOT
    / "R12E_v4_post_segmentation_filter_audit"
)
OUT.mkdir(parents=True, exist_ok=True)


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


def verify_r12e_v3():
    require(R12E_V3_MANIFEST)

    m = json.loads(
        R12E_V3_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R12E_V3_RULE_EXTRACTED_BUT_RECONSTRUCTION_NOT_EXACT_REVIEW_BEFORE_R12F"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R12-E-V3 status: {m.get('status')}"
        )

    return m


def canon_obj(v):
    if pd.isna(v):
        return ""

    s = str(v).strip()

    try:
        x = float(s)
        if np.isfinite(x) and x.is_integer():
            return str(int(x))
    except Exception:
        pass

    return s


def load_csv(path):
    require(path)
    return pd.read_csv(path, low_memory=False)


def normalize_ids(df):
    out = df.copy()

    if "sequence" in out.columns:
        out["sequence"] = (
            out["sequence"]
            .astype(str)
            .str.strip()
        )

    if "object_id" in out.columns:
        out["object_id_canon"] = out["object_id"].map(canon_obj)

    if "segment_id" in out.columns:
        out["segment_id_canon"] = out["segment_id"].map(canon_obj)

    if {
        "sequence",
        "object_id_canon",
    }.issubset(out.columns):
        out["original_track_key_canon"] = (
            out["sequence"]
            + "|"
            + out["object_id_canon"]
        )

    if {
        "sequence",
        "object_id_canon",
        "segment_id_canon",
    }.issubset(out.columns):
        out["segment_key_canon"] = (
            out["sequence"]
            + "|"
            + out["object_id_canon"]
            + "|"
            + out["segment_id_canon"]
        )

    return out


def candidate_window_identity(df):
    preferred = [
        ["sample_id"],
        ["sequence", "object_id_canon", "segment_id_canon", "obs_start_frame", "future_end_frame"],
        ["sequence", "object_id_canon", "segment_id_canon", "obs_start_frame"],
        ["sequence", "object_id_canon", "segment_id_canon", "obs_end_frame", "future_end_frame"],
    ]

    for cols in preferred:
        if all(
            c in df.columns
            for c in cols
        ):
            return cols

    raise RuntimeError(
        f"Could not determine window identity from columns: {list(df.columns)}"
    )


def identity_series(df, cols):
    if len(cols) == 1:
        return df[cols[0]].astype(str)

    pieces = []

    for c in cols:
        if "frame" in c:
            vals = pd.to_numeric(
                df[c],
                errors="coerce",
            )

            vals = vals.map(
                lambda x:
                    str(int(x))
                    if pd.notna(x)
                    and float(x).is_integer()
                    else str(x)
            )

            pieces.append(vals)

        else:
            pieces.append(
                df[c].astype(str)
            )

    out = pieces[0]

    for s in pieces[1:]:
        out = out + "|" + s

    return out


def summarize_stage(task, raw_df, corr_df, final_df):
    rows = []

    for stage, df in [
        ("raw_windows", raw_df),
        ("corrected_windows", corr_df),
        ("final_metadata", final_df),
    ]:
        rec = {
            "task": task,
            "stage": stage,
            "rows": int(len(df)),
        }

        if "sequence" in df.columns:
            rec["sources"] = int(
                df["sequence"].astype(str).nunique()
            )

        if "original_track_key_canon" in df.columns:
            rec["original_tracks"] = int(
                df["original_track_key_canon"].nunique()
            )

        if "segment_key_canon" in df.columns:
            rec["segments"] = int(
                df["segment_key_canon"].nunique()
            )

        elif "track_key" in df.columns:
            rec["segments"] = int(
                df["track_key"].astype(str).nunique()
            )

        rows.append(rec)

    return pd.DataFrame(rows)


def value_count_inventory(df, task, stage):
    rows = []

    interesting = [
        "accepted",
        "quality_tier",
        "repaired_motion_pct",
        "usable_motion",
        "reason",
        "status",
        "reject_reason",
        "rejection_reason",
        "quality_reason",
        "motion_quality",
    ]

    for c in interesting:
        if c not in df.columns:
            continue

        vc = (
            df[c]
            .astype(str)
            .value_counts(
                dropna=False
            )
        )

        for value, count in vc.items():
            rows.append(
                {
                    "task": task,
                    "stage": stage,
                    "column": c,
                    "value": value,
                    "count": int(count),
                }
            )

    return rows


def compare_stages(task, left_name, left, right_name, right):
    left_cols = candidate_window_identity(left)
    right_cols = candidate_window_identity(right)

    # Prefer the richer common identity if schemas allow.
    common_candidates = [
        ["sample_id"],
        ["sequence", "object_id_canon", "segment_id_canon", "obs_start_frame", "future_end_frame"],
        ["sequence", "object_id_canon", "segment_id_canon", "obs_start_frame"],
    ]

    chosen = None

    for cols in common_candidates:
        if all(c in left.columns for c in cols) and all(c in right.columns for c in cols):
            chosen = cols
            break

    if chosen is None:
        raise RuntimeError(
            f"{task}: no common identity for {left_name} vs {right_name}. "
            f"left={list(left.columns)}, right={list(right.columns)}"
        )

    l = left.copy()
    r = right.copy()

    l["_window_key"] = identity_series(
        l,
        chosen
    )

    r["_window_key"] = identity_series(
        r,
        chosen
    )

    left_set = set(
        l["_window_key"].astype(str)
    )

    right_set = set(
        r["_window_key"].astype(str)
    )

    only_left = left_set - right_set
    only_right = right_set - left_set

    summary = pd.DataFrame(
        [
            {
                "task": task,
                "left_stage": left_name,
                "right_stage": right_name,
                "identity_columns": "|".join(chosen),
                "left_rows": int(len(l)),
                "right_rows": int(len(r)),
                "left_unique_windows": int(len(left_set)),
                "right_unique_windows": int(len(right_set)),
                "only_left": int(len(only_left)),
                "only_right": int(len(only_right)),
                "right_subset_of_left": bool(
                    right_set.issubset(left_set)
                ),
            }
        ]
    )

    removed = l[
        l["_window_key"].isin(
            only_left
        )
    ].copy()

    added = r[
        r["_window_key"].isin(
            only_right
        )
    ].copy()

    return summary, removed, added


def extract_source_evidence(path, terms):
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
            term
            for term in terms
            if term.lower() in ll
        ]

        if not matched:
            continue

        start = max(
            1,
            i - 5,
        )

        end = min(
            len(lines),
            i + 5,
        )

        context = "\n".join(
            f"{j}: {lines[j-1]}"
            for j in range(
                start,
                end + 1,
            )
        )

        rows.append(
            {
                "source_file": str(path),
                "source_sha256": sha256_file(path),
                "anchor_line": i,
                "matched_terms": "|".join(matched),
                "anchor_text": line,
                "context_start": start,
                "context_end": end,
                "context": context,
            }
        )

    return pd.DataFrame(rows)


def removed_reason_summary(removed_df, task, transition):
    rows = []

    if len(removed_df) == 0:
        return pd.DataFrame(rows)

    candidate_cols = [
        "accepted",
        "quality_tier",
        "repaired_motion_pct",
        "usable_motion",
        "reason",
        "status",
        "reject_reason",
        "rejection_reason",
        "quality_reason",
        "motion_quality",
    ]

    present = [
        c
        for c in candidate_cols
        if c in removed_df.columns
    ]

    if not present:
        return pd.DataFrame(
            [
                {
                    "task": task,
                    "transition": transition,
                    "column": "<none>",
                    "value": "<no explicit reason column in removed-stage schema>",
                    "count": int(len(removed_df)),
                }
            ]
        )

    for c in present:
        vc = (
            removed_df[c]
            .astype(str)
            .value_counts(
                dropna=False
            )
        )

        for value, count in vc.items():
            rows.append(
                {
                    "task": task,
                    "transition": transition,
                    "column": c,
                    "value": value,
                    "count": int(count),
                }
            )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR12-E-V4 — POST-SEGMENTATION CORRECTION / QUALITY FILTER AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R12-E-V3 proved the continuity segmentation rule but reconstructed 17 extra 2s "
        "windows and 1 extra 5s window. This step traces those exact windows through "
        "raw-window, corrected-window, and final-neural metadata stages and audits the "
        "source code for accepted/quality/repaired-motion filtering. No original file "
        "is modified.\n"
    )

    verify_r12e_v3()

    for p in [
        RAW_WINDOWS_2,
        RAW_WINDOWS_5,
        CORR_WINDOWS_2,
        CORR_WINDOWS_5,
        FINAL_META_2,
        FINAL_META_5,
        CORRECTOR,
        FINAL_BUILDER,
    ]:
        require(p)

    raw2 = normalize_ids(
        load_csv(
            RAW_WINDOWS_2
        )
    )

    raw5 = normalize_ids(
        load_csv(
            RAW_WINDOWS_5
        )
    )

    corr2 = normalize_ids(
        load_csv(
            CORR_WINDOWS_2
        )
    )

    corr5 = normalize_ids(
        load_csv(
            CORR_WINDOWS_5
        )
    )

    final2 = normalize_ids(
        load_csv(
            FINAL_META_2
        )
    )

    final5 = normalize_ids(
        load_csv(
            FINAL_META_5
        )
    )

    stage_summary = pd.concat(
        [
            summarize_stage(
                "2s_to_2s",
                raw2,
                corr2,
                final2,
            ),
            summarize_stage(
                "2s_to_5s",
                raw5,
                corr5,
                final5,
            ),
        ],
        ignore_index=True,
    )

    value_rows = []

    for task, stage, df in [
        ("2s_to_2s", "raw_windows", raw2),
        ("2s_to_2s", "corrected_windows", corr2),
        ("2s_to_2s", "final_metadata", final2),
        ("2s_to_5s", "raw_windows", raw5),
        ("2s_to_5s", "corrected_windows", corr5),
        ("2s_to_5s", "final_metadata", final5),
    ]:
        value_rows.extend(
            value_count_inventory(
                df,
                task,
                stage,
            )
        )

    value_df = pd.DataFrame(
        value_rows
    )

    comparisons = []
    removed_outputs = []
    added_outputs = []
    reason_outputs = []

    for task, raw, corr, final in [
        (
            "2s_to_2s",
            raw2,
            corr2,
            final2,
        ),
        (
            "2s_to_5s",
            raw5,
            corr5,
            final5,
        ),
    ]:
        s1, rem1, add1 = compare_stages(
            task,
            "raw_windows",
            raw,
            "corrected_windows",
            corr,
        )

        s2, rem2, add2 = compare_stages(
            task,
            "corrected_windows",
            corr,
            "final_metadata",
            final,
        )

        comparisons.extend(
            [
                s1,
                s2,
            ]
        )

        if len(rem1):
            rem1[
                "_task"
            ] = task

            rem1[
                "_transition"
            ] = (
                "raw_windows -> corrected_windows"
            )

            removed_outputs.append(
                rem1
            )

        if len(rem2):
            rem2[
                "_task"
            ] = task

            rem2[
                "_transition"
            ] = (
                "corrected_windows -> final_metadata"
            )

            removed_outputs.append(
                rem2
            )

        if len(add1):
            add1[
                "_task"
            ] = task

            add1[
                "_transition"
            ] = (
                "raw_windows -> corrected_windows"
            )

            added_outputs.append(
                add1
            )

        if len(add2):
            add2[
                "_task"
            ] = task

            add2[
                "_transition"
            ] = (
                "corrected_windows -> final_metadata"
            )

            added_outputs.append(
                add2
            )

        reason_outputs.append(
            removed_reason_summary(
                rem1,
                task,
                "raw_windows -> corrected_windows",
            )
        )

        reason_outputs.append(
            removed_reason_summary(
                rem2,
                task,
                "corrected_windows -> final_metadata",
            )
        )

    compare_df = pd.concat(
        comparisons,
        ignore_index=True,
    )

    removed_df = (
        pd.concat(
            removed_outputs,
            ignore_index=True,
            sort=False,
        )
        if removed_outputs
        else pd.DataFrame()
    )

    added_df = (
        pd.concat(
            added_outputs,
            ignore_index=True,
            sort=False,
        )
        if added_outputs
        else pd.DataFrame()
    )

    reason_df = (
        pd.concat(
            [
                x
                for x in reason_outputs
                if len(x)
            ],
            ignore_index=True,
        )
        if any(
            len(x)
            for x in reason_outputs
        )
        else pd.DataFrame()
    )

    corrector_evidence = extract_source_evidence(
        CORRECTOR,
        [
            "accepted",
            "quality_tier",
            "repaired_motion_pct",
            "usable_motion",
            "return None",
            "reject",
            "continue",
            "history",
            "future",
        ],
    )

    final_builder_evidence = extract_source_evidence(
        FINAL_BUILDER,
        [
            "accepted",
            "quality_tier",
            "repaired_motion_pct",
            "track_key",
            "10hz",
            "10 hz",
            "history",
            "future",
            "continue",
            "filter",
        ],
    )

    source_evidence = pd.concat(
        [
            corrector_evidence,
            final_builder_evidence,
        ],
        ignore_index=True,
        sort=False,
    )

    stage_path = (
        OUT
        /
        "R12E_V4_STAGE_COUNTS.csv"
    )

    values_path = (
        OUT
        /
        "R12E_V4_ACCEPTANCE_QUALITY_VALUE_COUNTS.csv"
    )

    compare_path = (
        OUT
        /
        "R12E_V4_STAGE_WINDOW_SET_COMPARISON.csv"
    )

    removed_path = (
        OUT
        /
        "R12E_V4_REMOVED_WINDOWS.csv"
    )

    added_path = (
        OUT
        /
        "R12E_V4_ADDED_WINDOWS_UNEXPECTED.csv"
    )

    reasons_path = (
        OUT
        /
        "R12E_V4_REMOVED_WINDOW_REASON_SUMMARY.csv"
    )

    evidence_path = (
        OUT
        /
        "R12E_V4_FILTER_SOURCE_CODE_EVIDENCE.csv"
    )

    stage_summary.to_csv(
        stage_path,
        index=False,
    )

    value_df.to_csv(
        values_path,
        index=False,
    )

    compare_df.to_csv(
        compare_path,
        index=False,
    )

    removed_df.to_csv(
        removed_path,
        index=False,
    )

    added_df.to_csv(
        added_path,
        index=False,
    )

    reason_df.to_csv(
        reasons_path,
        index=False,
    )

    source_evidence.to_csv(
        evidence_path,
        index=False,
    )

    # Determine where the 17/1 reduction occurs.
    def transition_removed(
        task,
        left,
        right,
    ):
        row = compare_df[
            (
                compare_df[
                    "task"
                ]
                ==
                task
            )
            &
            (
                compare_df[
                    "left_stage"
                ]
                ==
                left
            )
            &
            (
                compare_df[
                    "right_stage"
                ]
                ==
                right
            )
        ]

        if len(row) != 1:
            return None

        return int(
            row.iloc[
                0
            ][
                "only_left"
            ]
        )

    raw_to_corr_2 = transition_removed(
        "2s_to_2s",
        "raw_windows",
        "corrected_windows",
    )

    corr_to_final_2 = transition_removed(
        "2s_to_2s",
        "corrected_windows",
        "final_metadata",
    )

    raw_to_corr_5 = transition_removed(
        "2s_to_5s",
        "raw_windows",
        "corrected_windows",
    )

    corr_to_final_5 = transition_removed(
        "2s_to_5s",
        "corrected_windows",
        "final_metadata",
    )

    # We don't force where the filtering happens; we report it.
    status = (
        "R12E_V4_POST_SEGMENTATION_FILTER_STAGE_AND_REMOVED_WINDOWS_IDENTIFIED_READY_FOR_R12E_V5_EXACT_FILTER_RULE_RECONSTRUCTION"
    )

    manifest = {
        "step":
            "R12E_V4_POST_SEGMENTATION_CORRECTION_QUALITY_FILTER_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "raw_to_corrected_removed_2s":
            raw_to_corr_2,

        "corrected_to_final_removed_2s":
            corr_to_final_2,

        "raw_to_corrected_removed_5s":
            raw_to_corr_5,

        "corrected_to_final_removed_5s":
            corr_to_final_5,

        "outputs": {
            "stage_counts":
                str(
                    stage_path
                ),

            "acceptance_quality_value_counts":
                str(
                    values_path
                ),

            "stage_window_set_comparison":
                str(
                    compare_path
                ),

            "removed_windows":
                str(
                    removed_path
                ),

            "added_windows":
                str(
                    added_path
                ),

            "removed_reason_summary":
                str(
                    reasons_path
                ),

            "source_code_evidence":
                str(
                    evidence_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_V4_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "STAGE COUNTS:"
    )

    print(
        stage_summary.to_string(
            index=False
        )
    )

    print(
        "\nSTAGE WINDOW-SET COMPARISON:"
    )

    print(
        compare_df.to_string(
            index=False
        )
    )

    print(
        "\nACCEPTANCE / QUALITY VALUE COUNTS:"
    )

    if len(value_df):
        print(
            value_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No explicit acceptance/quality columns found."
        )

    print(
        "\nREMOVED WINDOW REASON SUMMARY:"
    )

    if len(reason_df):
        print(
            reason_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No explicit removed-window reason values found."
        )

    print(
        "\nREMOVED WINDOWS:"
    )

    if len(removed_df):
        preferred_cols = [
            c
            for c in [
                "_task",
                "_transition",
                "sample_id",
                "sequence",
                "object_id",
                "object_id_canon",
                "segment_id",
                "segment_id_canon",
                "obs_start_frame",
                "obs_end_frame",
                "future_end_frame",
                "accepted",
                "quality_tier",
                "repaired_motion_pct",
                "reason",
                "status",
                "reject_reason",
            ]
            if c in removed_df.columns
        ]

        print(
            removed_df[
                preferred_cols
            ].to_string(
                index=False
            )
        )

    else:
        print(
            "No removed windows identified."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-E-V4 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
