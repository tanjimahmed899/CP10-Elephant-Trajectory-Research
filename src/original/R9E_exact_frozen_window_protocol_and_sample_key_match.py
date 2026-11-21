
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

R9D_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9D_isotropic_point_dataset_build"
)

R9D_MANIFEST = R9D_DIR / "R9D_MANIFEST.json"
ISO_POINTS = R9D_DIR / "R9D_isotropic_cleaned_track_points.csv"

STEP2B_DIR = MAIN2 / "step2b_raw_vs_corrected_baseline"
STEP2B_SAMPLE_METRICS = (
    STEP2B_DIR
    / "raw_vs_corrected_sample_metrics.csv"
)
STEP2B_SCRIPT = (
    MAIN2
    / "step2b_raw_vs_corrected_forecasting.py"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9E_exact_window_protocol_match"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_COUNTS = {
    "2s_to_2s": 1053,
    "2s_to_5s": 826,
}

EXPECTED_TRACKS = {
    "2s_to_2s": 89,
    "2s_to_5s": 63,
}

EXPECTED_SEQUENCES = {
    "2s_to_2s": 6,
    "2s_to_5s": 5,
}

TESTREAL_SEQUENCES = {
    "0000000055_0000000000",
    "0000000058_0000000000",
    "0000000349_0000000000",
    "0000000352_0000000000",
    "0000000353_0000000000",
    "0000000357_0000000000",
}


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


def verify_r9d():
    for p in [
        R9D_MANIFEST,
        ISO_POINTS,
        STEP2B_SAMPLE_METRICS,
        STEP2B_SCRIPT,
    ]:
        require(p)

    manifest = json.loads(
        R9D_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("status")
        !=
        "R9D_NONDESTRUCTIVE_ISOTROPIC_POINT_DATASET_BUILT_READY_FOR_R9E_WINDOW_PROTOCOL_MATCH"
    ):
        raise RuntimeError(
            f"Unexpected R9D status: {manifest.get('status')}"
        )

    return manifest


def infer_columns(ref):
    columns = set(
        ref.columns
    )

    task_col = (
        "task"
        if "task" in columns
        else None
    )

    if task_col is None:
        raise RuntimeError(
            "Reference sample metrics has no task column."
        )

    sequence_candidates = [
        "source_group",
        "sequence",
        "sequence_id",
        "source",
    ]

    sequence_col = next(
        (
            c
            for c in sequence_candidates
            if c in columns
        ),
        None,
    )

    if sequence_col is None:
        raise RuntimeError(
            "Could not identify sequence/source column in reference sample metrics."
        )

    track_candidates = [
        "track_key",
        "track_id",
        "track",
    ]

    track_col = next(
        (
            c
            for c in track_candidates
            if c in columns
        ),
        None,
    )

    if track_col is None:
        raise RuntimeError(
            "Could not identify track column in reference sample metrics."
        )

    sample_candidates = [
        "sample_index",
        "window_index",
        "sample_id",
    ]

    sample_col = next(
        (
            c
            for c in sample_candidates
            if c in columns
        ),
        None,
    )

    return (
        task_col,
        sequence_col,
        track_col,
        sample_col,
    )


def normalize_sequence_value(value):
    s = str(
        value
    )

    for prefix in [
        "test_real_",
        "train_real_",
    ]:
        if s.startswith(
            prefix
        ):
            s = s[
                len(
                    prefix
                ):
            ]

    return s


def build_reference_lock(ref):
    (
        task_col,
        seq_col,
        track_col,
        sample_col,
    ) = infer_columns(
        ref
    )

    work = ref.copy()

    work[
        "_sequence_normalized"
    ] = work[
        seq_col
    ].map(
        normalize_sequence_value
    )

    summary_rows = []

    locked_tables = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        sub = work[
            work[
                task_col
            ].astype(str)
            ==
            task
        ].copy()

        if len(
            sub
        ) != EXPECTED_COUNTS[
            task
        ]:
            raise RuntimeError(
                f"{task}: expected {EXPECTED_COUNTS[task]} reference samples, "
                f"found {len(sub)}"
            )

        track_count = int(
            sub[
                track_col
            ].astype(str)
            .nunique()
        )

        seq_count = int(
            sub[
                "_sequence_normalized"
            ]
            .astype(str)
            .nunique()
        )

        if (
            track_count
            != EXPECTED_TRACKS[
                task
            ]
        ):
            raise RuntimeError(
                f"{task}: expected {EXPECTED_TRACKS[task]} tracks, found {track_count}"
            )

        if (
            seq_count
            != EXPECTED_SEQUENCES[
                task
            ]
        ):
            raise RuntimeError(
                f"{task}: expected {EXPECTED_SEQUENCES[task]} sequences, found {seq_count}"
            )

        unexpected = sorted(
            set(
                sub[
                    "_sequence_normalized"
                ].astype(str)
            )
            -
            TESTREAL_SEQUENCES
        )

        if unexpected:
            raise RuntimeError(
                f"{task}: unexpected non-TestReal sequences: {unexpected}"
            )

        key_cols = [
            task_col,
            seq_col,
            track_col,
        ]

        if sample_col is not None:
            key_cols.append(
                sample_col
            )
        else:
            sub[
                "_within_task_row_index"
            ] = np.arange(
                len(
                    sub
                ),
                dtype=int,
            )

            key_cols.append(
                "_within_task_row_index"
            )

        lock = sub[
            key_cols
        ].copy()

        lock[
            "sequence_normalized"
        ] = sub[
            "_sequence_normalized"
        ].to_numpy()

        lock[
            "reference_row_index"
        ] = sub.index.to_numpy(
            dtype=int
        )

        locked_tables.append(
            lock
        )

        summary_rows.append(
            {
                "task":
                    task,

                "samples":
                    int(
                        len(
                            sub
                        )
                    ),

                "tracks":
                    track_count,

                "sequences":
                    seq_count,

                "sample_id_column":
                    sample_col
                    if sample_col is not None
                    else "_within_task_row_index",
            }
        )

    return (
        pd.concat(
            locked_tables,
            ignore_index=True,
        ),
        pd.DataFrame(
            summary_rows
        ),
        {
            "task_col":
                task_col,

            "sequence_col":
                seq_col,

            "track_col":
                track_col,

            "sample_col":
                sample_col,
        },
    )


def audit_iso_source_coverage(iso, ref_lock):
    if "sequence" not in iso.columns:
        raise RuntimeError(
            "R9D isotropic points lacks sequence column."
        )

    iso_sequences = set(
        iso[
            "sequence"
        ].astype(str)
        .unique()
    )

    ref_sequences = set(
        ref_lock[
            "sequence_normalized"
        ].astype(str)
        .unique()
    )

    missing_sequences = sorted(
        ref_sequences
        -
        iso_sequences
    )

    extra_relevant = sorted(
        (
            iso_sequences
            &
            TESTREAL_SEQUENCES
        )
        -
        ref_sequences
    )

    rows = [
        {
            "check":
                "all_reference_sequences_present_in_isotropic_points",

            "passed":
                len(
                    missing_sequences
                )
                ==
                0,

            "detail":
                json.dumps(
                    {
                        "missing_sequences":
                            missing_sequences,
                    }
                ),
        },

        {
            "check":
                "reference_sequences_are_testreal_only",

            "passed":
                ref_sequences
                <=
                TESTREAL_SEQUENCES,

            "detail":
                json.dumps(
                    {
                        "reference_sequences":
                            sorted(
                                ref_sequences
                            ),
                    }
                ),
        },

        {
            "check":
                "isotropic_point_source_contains_expected_testreal_sequences",

            "passed":
                TESTREAL_SEQUENCES
                <=
                iso_sequences,

            "detail":
                json.dumps(
                    {
                        "missing_from_isotropic":
                            sorted(
                                TESTREAL_SEQUENCES
                                -
                                iso_sequences
                            ),

                        "extra_relevant_not_used_in_reference":
                            extra_relevant,
                    }
                ),
        },
    ]

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "passed"
    ].all():
        raise RuntimeError(
            "R9E isotropic source coverage audit failed:\n"
            +
            audit.to_string(
                index=False
            )
        )

    return audit


def inspect_step2b_script():
    text = STEP2B_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    keywords = [
        "history",
        "future",
        "2s_to_2s",
        "2s_to_5s",
        "20",
        "50",
        "10",
        "sample_index",
        "track_key",
        "source_group",
        "cx_n",
        "cy_n",
        "corrected",
    ]

    rows = []

    lines = text.splitlines()

    for i, line in enumerate(
        lines,
        start=1,
    ):
        low = line.lower()

        if any(
            k.lower()
            in low
            for k in keywords
        ):
            a = max(
                1,
                i - 2,
            )

            b = min(
                len(
                    lines
                ),
                i + 2,
            )

            context = "\n".join(
                f"{j}: {lines[j-1]}"
                for j in range(
                    a,
                    b + 1,
                )
            )

            rows.append(
                {
                    "line_number":
                        i,

                    "line":
                        line,

                    "context":
                        context,
                }
            )

    return pd.DataFrame(
        rows
    )


def discover_window_candidates():
    rows = []

    for root in [
        MAIN1,
        MAIN2,
    ]:
        if not root.exists():
            continue

        for p in root.rglob(
            "*.csv"
        ):
            try:
                header = pd.read_csv(
                    p,
                    nrows=3,
                )
            except Exception:
                continue

            cols = set(
                map(
                    str,
                    header.columns,
                )
            )

            score = 0

            for c in [
                "task",
                "sample_index",
                "track_key",
                "source_group",
                "history",
                "future",
                "history_path",
                "future_path",
                "history_x",
                "history_y",
                "future_x",
                "future_y",
            ]:
                if c in cols:
                    score += 1

            if score < 2:
                continue

            try:
                nrows = sum(
                    1
                    for _ in open(
                        p,
                        "rb",
                    )
                ) - 1
            except Exception:
                nrows = None

            rows.append(
                {
                    "path":
                        str(
                            p
                        ),

                    "filename":
                        p.name,

                    "root":
                        root.name,

                    "approx_rows":
                        nrows,

                    "score":
                        score,

                    "columns":
                        " | ".join(
                            sorted(
                                cols
                            )
                        ),

                    "sha256":
                        sha256_file(
                            p
                        ),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(
        rows
    )

    out[
        "exact_expected_row_count_candidate"
    ] = out[
        "approx_rows"
    ].isin(
        [
            1053,
            826,
            1879,
        ]
    )

    return out.sort_values(
        [
            "exact_expected_row_count_candidate",
            "score",
            "filename",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="mergesort",
    )


def main():
    print(
        "\nR9-E — EXACT FROZEN WINDOW-PROTOCOL / SAMPLE-KEY MATCH"
    )

    print(
        "NO model training. NO isotropic model inference. "
        "This step locks the exact frozen TestReal sample universe used by Step2b "
        "and verifies that the R9D isotropic source contains every required source sequence. "
        "Any later isotropic comparison must use these exact sample keys and counts.\n"
    )

    verify_r9d()

    ref = pd.read_csv(
        STEP2B_SAMPLE_METRICS
    )

    iso = pd.read_csv(
        ISO_POINTS
    )

    (
        ref_lock,
        ref_summary,
        ref_columns,
    ) = build_reference_lock(
        ref
    )

    coverage = audit_iso_source_coverage(
        iso,
        ref_lock,
    )

    script_context = inspect_step2b_script()

    candidates = discover_window_candidates()

    ref_lock_path = (
        OUT
        / "R9E_FROZEN_REFERENCE_SAMPLE_KEY_LOCK.csv"
    )

    summary_path = (
        OUT
        / "R9E_REFERENCE_SAMPLE_ACCOUNTING.csv"
    )

    coverage_path = (
        OUT
        / "R9E_ISOTROPIC_SOURCE_COVERAGE_AUDIT.csv"
    )

    script_path = (
        OUT
        / "R9E_STEP2B_WINDOW_PROTOCOL_CODE_CONTEXT.csv"
    )

    candidate_path = (
        OUT
        / "R9E_EXISTING_WINDOW_SOURCE_CANDIDATES.csv"
    )

    ref_lock.to_csv(
        ref_lock_path,
        index=False,
    )

    ref_summary.to_csv(
        summary_path,
        index=False,
    )

    coverage.to_csv(
        coverage_path,
        index=False,
    )

    script_context.to_csv(
        script_path,
        index=False,
    )

    candidates.to_csv(
        candidate_path,
        index=False,
    )

    exact_counts_pass = bool(
        (
            ref_summary.set_index(
                "task"
            )[
                "samples"
            ].to_dict()
            ==
            EXPECTED_COUNTS
        )
    )

    status = (
        "R9E_FROZEN_SAMPLE_UNIVERSE_AND_WINDOW_PROTOCOL_MATCH_LOCKED_READY_FOR_R9F_ISOTROPIC_WINDOW_BUILD"
    )

    manifest = {
        "step":
            "R9E_EXACT_FROZEN_WINDOW_PROTOCOL_SAMPLE_MATCH",

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

        "reference_step2b_sample_metrics":
            str(
                STEP2B_SAMPLE_METRICS
            ),

        "reference_step2b_sample_metrics_sha256":
            sha256_file(
                STEP2B_SAMPLE_METRICS
            ),

        "reference_columns":
            ref_columns,

        "expected_counts":
            EXPECTED_COUNTS,

        "exact_counts_pass":
            exact_counts_pass,

        "reference_sample_lock_csv":
            str(
                ref_lock_path
            ),

        "reference_sample_lock_sha256":
            sha256_file(
                ref_lock_path
            ),

        "isotropic_source":
            str(
                ISO_POINTS
            ),

        "isotropic_source_sha256":
            sha256_file(
                ISO_POINTS
            ),

        "isotropic_source_coverage_all_pass":
            bool(
                coverage[
                    "passed"
                ].all()
            ),

        "existing_window_candidate_count":
            int(
                len(
                    candidates
                )
            ),
    }

    (
        OUT
        / "R9E_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "FROZEN REFERENCE SAMPLE ACCOUNTING:"
    )

    print(
        ref_summary.to_string(
            index=False
        )
    )

    print(
        "\nISOTROPIC SOURCE COVERAGE AUDIT:"
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    print(
        "\nREFERENCE KEY COLUMNS:"
    )

    print(
        json.dumps(
            ref_columns,
            indent=2,
        )
    )

    print(
        "\nTOP EXISTING WINDOW SOURCE CANDIDATES:"
    )

    if len(
        candidates
    ):
        print(
            candidates[
                [
                    "filename",
                    "root",
                    "approx_rows",
                    "score",
                    "exact_expected_row_count_candidate",
                    "columns",
                    "path",
                ]
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )
    else:
        print(
            "No existing window-table candidates found."
        )

    print(
        "\nSTEP2B WINDOW-PROTOCOL CODE CONTEXT:"
    )

    print(
        script_context.head(
            80
        ).to_string(
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
        "\nR9-E COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
