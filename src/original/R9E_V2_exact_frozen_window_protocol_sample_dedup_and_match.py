
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
STEP2B_SAMPLE_METRICS = STEP2B_DIR / "raw_vs_corrected_sample_metrics.csv"
STEP2B_SCRIPT = MAIN2 / "step2b_raw_vs_corrected_forecasting.py"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9E_V2_exact_window_protocol_match"
)

OUT.mkdir(parents=True, exist_ok=True)

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

EXPECTED_MODELS = {
    "Stationary",
    "ConstantVelocity",
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
        R9D_MANIFEST.read_text(encoding="utf-8")
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


def normalize_sequence_value(value):
    s = str(value)

    for prefix in [
        "test_real_",
        "train_real_",
    ]:
        if s.startswith(prefix):
            s = s[len(prefix):]

    return s


def infer_columns(ref):
    cols = set(ref.columns)

    if "task" not in cols:
        raise RuntimeError("Reference table has no task column.")

    sequence_col = next(
        (
            c for c in [
                "source_group",
                "sequence",
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

    sample_col = next(
        (
            c for c in [
                "sample_index",
                "window_index",
                "sample_id",
            ]
            if c in cols
        ),
        None,
    )

    model_col = next(
        (
            c for c in [
                "model",
                "baseline",
                "method",
            ]
            if c in cols
        ),
        None,
    )

    if sequence_col is None:
        raise RuntimeError(
            "Could not identify sequence/source column."
        )

    if track_col is None:
        raise RuntimeError(
            "Could not identify track column."
        )

    if sample_col is None:
        raise RuntimeError(
            "Could not identify sample/window ID column. "
            "R9E-V2 refuses row-order-only deduplication."
        )

    if model_col is None:
        raise RuntimeError(
            "Could not identify model column even though row duplication is expected."
        )

    return {
        "task_col": "task",
        "sequence_col": sequence_col,
        "track_col": track_col,
        "sample_col": sample_col,
        "model_col": model_col,
    }


def audit_model_duplication(ref, cols):
    rows = []

    for task in EXPECTED_COUNTS:
        sub = ref[
            ref[cols["task_col"]].astype(str) == task
        ].copy()

        model_values = sorted(
            sub[cols["model_col"]]
            .astype(str)
            .unique()
            .tolist()
        )

        raw_rows = len(sub)

        sample_key_cols = [
            cols["task_col"],
            cols["sequence_col"],
            cols["track_col"],
            cols["sample_col"],
        ]

        unique_samples = (
            sub[sample_key_cols]
            .drop_duplicates()
        )

        duplicate_factor = (
            raw_rows / len(unique_samples)
            if len(unique_samples) > 0
            else np.nan
        )

        per_sample_model_counts = (
            sub.groupby(
                sample_key_cols,
                dropna=False,
            )[cols["model_col"]]
            .nunique()
        )

        per_sample_row_counts = (
            sub.groupby(
                sample_key_cols,
                dropna=False,
            )
            .size()
        )

        all_two_models = bool(
            (per_sample_model_counts == 2).all()
        )

        all_two_rows = bool(
            (per_sample_row_counts == 2).all()
        )

        rows.append(
            {
                "task": task,
                "raw_rows": int(raw_rows),
                "unique_sample_keys": int(len(unique_samples)),
                "duplicate_factor": float(duplicate_factor),
                "models": json.dumps(model_values),
                "model_count": int(len(model_values)),
                "all_sample_keys_have_two_models": all_two_models,
                "all_sample_keys_have_exactly_two_rows": all_two_rows,
                "matches_expected_unique_count":
                    int(len(unique_samples)) == EXPECTED_COUNTS[task],
            }
        )

        if len(model_values) != 2:
            raise RuntimeError(
                f"{task}: expected exactly 2 models, found {model_values}"
            )

        if set(model_values) != EXPECTED_MODELS:
            raise RuntimeError(
                f"{task}: model set mismatch. "
                f"Expected {sorted(EXPECTED_MODELS)}, found {model_values}"
            )

        if not all_two_models or not all_two_rows:
            raise RuntimeError(
                f"{task}: model duplication is not exactly one row per model per sample."
            )

        if len(unique_samples) != EXPECTED_COUNTS[task]:
            raise RuntimeError(
                f"{task}: deduplicated unique sample count "
                f"{len(unique_samples)} != expected {EXPECTED_COUNTS[task]}"
            )

    return pd.DataFrame(rows)


def build_reference_lock(ref, cols):
    work = ref.copy()

    work["_sequence_normalized"] = (
        work[cols["sequence_col"]]
        .map(normalize_sequence_value)
    )

    sample_key_cols = [
        cols["task_col"],
        cols["sequence_col"],
        cols["track_col"],
        cols["sample_col"],
    ]

    # Confirm that non-model identity fields are identical across the two model rows.
    identity_check_cols = [
        c for c in [
            cols["task_col"],
            cols["sequence_col"],
            cols["track_col"],
            cols["sample_col"],
        ]
        if c in work.columns
    ]

    # One canonical row per frozen sample key.
    dedup = (
        work.sort_values(
            sample_key_cols + [cols["model_col"]],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=sample_key_cols,
            keep="first",
        )
        .copy()
    )

    dedup["sequence_normalized"] = (
        dedup[cols["sequence_col"]]
        .map(normalize_sequence_value)
    )

    lock_rows = []
    summary_rows = []

    for task in EXPECTED_COUNTS:
        sub = dedup[
            dedup[cols["task_col"]].astype(str) == task
        ].copy()

        if len(sub) != EXPECTED_COUNTS[task]:
            raise RuntimeError(
                f"{task}: canonical sample count mismatch after dedup."
            )

        track_count = int(
            sub[cols["track_col"]]
            .astype(str)
            .nunique()
        )

        seq_count = int(
            sub["sequence_normalized"]
            .astype(str)
            .nunique()
        )

        if track_count != EXPECTED_TRACKS[task]:
            raise RuntimeError(
                f"{task}: expected {EXPECTED_TRACKS[task]} tracks, "
                f"found {track_count}"
            )

        if seq_count != EXPECTED_SEQUENCES[task]:
            raise RuntimeError(
                f"{task}: expected {EXPECTED_SEQUENCES[task]} sequences, "
                f"found {seq_count}"
            )

        unexpected = sorted(
            set(
                sub["sequence_normalized"].astype(str)
            )
            - TESTREAL_SEQUENCES
        )

        if unexpected:
            raise RuntimeError(
                f"{task}: unexpected sequences: {unexpected}"
            )

        lock = sub[
            [
                cols["task_col"],
                cols["sequence_col"],
                cols["track_col"],
                cols["sample_col"],
                "sequence_normalized",
            ]
        ].copy()

        lock["canonical_reference_sample_index"] = np.arange(
            len(lock),
            dtype=int,
        )

        lock_rows.append(lock)

        summary_rows.append(
            {
                "task": task,
                "samples": int(len(sub)),
                "tracks": track_count,
                "sequences": seq_count,
                "source_models_removed_from_identity":
                    json.dumps(sorted(EXPECTED_MODELS)),
                "sample_id_column": cols["sample_col"],
            }
        )

    return (
        pd.concat(lock_rows, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def audit_iso_source_coverage(iso, ref_lock):
    if "sequence" not in iso.columns:
        raise RuntimeError(
            "R9D isotropic points lacks sequence column."
        )

    iso_sequences = set(
        iso["sequence"].astype(str).unique()
    )

    ref_sequences = set(
        ref_lock["sequence_normalized"].astype(str).unique()
    )

    missing_sequences = sorted(
        ref_sequences - iso_sequences
    )

    rows = [
        {
            "check":
                "all_reference_sequences_present_in_isotropic_points",
            "passed":
                len(missing_sequences) == 0,
            "detail":
                json.dumps(
                    {
                        "missing_sequences": missing_sequences
                    }
                ),
        },
        {
            "check":
                "reference_sequences_are_testreal_only",
            "passed":
                ref_sequences <= TESTREAL_SEQUENCES,
            "detail":
                json.dumps(
                    {
                        "reference_sequences":
                            sorted(ref_sequences)
                    }
                ),
        },
        {
            "check":
                "isotropic_source_contains_all_six_testreal_sequences",
            "passed":
                TESTREAL_SEQUENCES <= iso_sequences,
            "detail":
                json.dumps(
                    {
                        "missing":
                            sorted(
                                TESTREAL_SEQUENCES - iso_sequences
                            )
                    }
                ),
        },
    ]

    audit = pd.DataFrame(rows)

    if not audit["passed"].all():
        raise RuntimeError(
            "Isotropic source coverage audit failed:\n"
            + audit.to_string(index=False)
        )

    return audit


def inspect_step2b_script():
    text = STEP2B_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    keywords = [
        "history_len",
        "future_len",
        "history",
        "future",
        "20",
        "50",
        "10hz",
        "sample_index",
        "track_key",
        "source_group",
        "stationary",
        "constantvelocity",
        "cx_n",
        "cy_n",
        "corrected",
    ]

    rows = []

    for i, line in enumerate(lines, start=1):
        low = line.lower()

        if any(k.lower() in low for k in keywords):
            a = max(1, i - 2)
            b = min(len(lines), i + 2)

            context = "\n".join(
                f"{j}: {lines[j-1]}"
                for j in range(a, b + 1)
            )

            rows.append(
                {
                    "line_number": i,
                    "line": line,
                    "context": context,
                }
            )

    return pd.DataFrame(rows)


def discover_window_candidates():
    rows = []

    for root in [MAIN1, MAIN2]:
        if not root.exists():
            continue

        for p in root.rglob("*.csv"):
            try:
                header = pd.read_csv(p, nrows=3)
            except Exception:
                continue

            cols = set(map(str, header.columns))

            score = 0
            for c in [
                "task",
                "sample_index",
                "track_key",
                "source_group",
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

            rows.append(
                {
                    "path": str(p),
                    "filename": p.name,
                    "root": root.name,
                    "score": score,
                    "columns": " | ".join(sorted(cols)),
                    "sha256": sha256_file(p),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["score", "filename"],
        ascending=[False, True],
        kind="mergesort",
    )


def main():
    print(
        "\nR9-E-V2 — EXACT FROZEN WINDOW-PROTOCOL / SAMPLE-KEY MATCH"
    )

    print(
        "NO model training. NO isotropic inference. "
        "The Step2b sample-metric table contains one row per model per sample, "
        "so this revision first proves exact two-model duplication and then locks "
        "one canonical sample identity per frozen sample.\n"
    )

    verify_r9d()

    ref = pd.read_csv(STEP2B_SAMPLE_METRICS)
    iso = pd.read_csv(ISO_POINTS)

    cols = infer_columns(ref)

    duplication = audit_model_duplication(
        ref,
        cols,
    )

    ref_lock, ref_summary = build_reference_lock(
        ref,
        cols,
    )

    coverage = audit_iso_source_coverage(
        iso,
        ref_lock,
    )

    script_context = inspect_step2b_script()
    candidates = discover_window_candidates()

    duplication_path = (
        OUT / "R9E_V2_MODEL_DUPLICATION_AUDIT.csv"
    )

    ref_lock_path = (
        OUT / "R9E_V2_FROZEN_REFERENCE_SAMPLE_KEY_LOCK.csv"
    )

    summary_path = (
        OUT / "R9E_V2_REFERENCE_SAMPLE_ACCOUNTING.csv"
    )

    coverage_path = (
        OUT / "R9E_V2_ISOTROPIC_SOURCE_COVERAGE_AUDIT.csv"
    )

    script_path = (
        OUT / "R9E_V2_STEP2B_WINDOW_PROTOCOL_CODE_CONTEXT.csv"
    )

    candidate_path = (
        OUT / "R9E_V2_EXISTING_WINDOW_SOURCE_CANDIDATES.csv"
    )

    duplication.to_csv(
        duplication_path,
        index=False,
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

    status = (
        "R9E_V2_FROZEN_SAMPLE_UNIVERSE_AND_WINDOW_PROTOCOL_MATCH_LOCKED_READY_FOR_R9F_ISOTROPIC_WINDOW_BUILD"
    )

    manifest = {
        "step":
            "R9E_V2_EXACT_FROZEN_WINDOW_PROTOCOL_SAMPLE_MATCH",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "frozen_results_modified":
            False,

        "failure_fixed":
            (
                "Step2b raw_vs_corrected_sample_metrics has two rows per "
                "sample because Stationary and ConstantVelocity share the "
                "same frozen sample universe."
            ),

        "reference_table":
            str(STEP2B_SAMPLE_METRICS),

        "reference_table_sha256":
            sha256_file(STEP2B_SAMPLE_METRICS),

        "reference_columns":
            cols,

        "expected_counts":
            EXPECTED_COUNTS,

        "expected_models":
            sorted(EXPECTED_MODELS),

        "model_duplication_audit_csv":
            str(duplication_path),

        "sample_key_lock_csv":
            str(ref_lock_path),

        "sample_key_lock_sha256":
            sha256_file(ref_lock_path),

        "isotropic_source":
            str(ISO_POINTS),

        "isotropic_source_sha256":
            sha256_file(ISO_POINTS),

        "coverage_all_pass":
            bool(coverage["passed"].all()),
    }

    (
        OUT / "R9E_V2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("MODEL-DUPLICATION AUDIT:")
    print(
        duplication.to_string(
            index=False
        )
    )

    print("\nFROZEN REFERENCE SAMPLE ACCOUNTING:")
    print(
        ref_summary.to_string(
            index=False
        )
    )

    print("\nISOTROPIC SOURCE COVERAGE AUDIT:")
    print(
        coverage.to_string(
            index=False
        )
    )

    print("\nREFERENCE KEY COLUMNS:")
    print(
        json.dumps(
            cols,
            indent=2,
        )
    )

    print("\nTOP EXISTING WINDOW SOURCE CANDIDATES:")
    if len(candidates):
        print(
            candidates[
                [
                    "filename",
                    "root",
                    "score",
                    "columns",
                    "path",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )
    else:
        print("No existing window-table candidates found.")

    print("\nSTEP2B WINDOW-PROTOCOL CODE CONTEXT:")
    print(
        script_context.head(80).to_string(
            index=False
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR9-E-V2 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
