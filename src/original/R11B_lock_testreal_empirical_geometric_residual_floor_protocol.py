
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"

R11A_DIR = (
    REVISION_ROOT
    / "R11_noise_floor_analysis"
    / "R11A_empirical_noise_floor_source_audit"
)
R11A_MANIFEST = R11A_DIR / "R11A_MANIFEST.json"

R8B_CSV = (
    REVISION_ROOT
    / "R8_ego_motion_ablation"
    / "R8B_exact_result_extraction"
    / "R8B_stationary_landmark_residual_results.csv"
)

MAIN2_SPLIT_CSV = (
    MAIN2
    / "step2_ego_motion_validation"
    / "heldout_background_residual_split_summary.csv"
)

MAIN2_SEQUENCE_CSV = (
    MAIN2
    / "step2_ego_motion_validation"
    / "heldout_background_residual_summary.csv"
)

OUT = (
    REVISION_ROOT
    / "R11_noise_floor_analysis"
    / "R11B_empirical_geometric_residual_floor_protocol_lock"
)
OUT.mkdir(parents=True, exist_ok=True)


PRIMARY_REQUIRED_METRICS = [
    "median_raw_background_motion_px",
    "median_corrected_residual_px",
    "p95_corrected_residual_px",
]


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


def verify_r11a():
    require(R11A_MANIFEST)

    m = json.loads(
        R11A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R11A_EMPIRICAL_NOISE_SOURCES_FOUND_READY_FOR_R11B_NOISE_FLOOR_PROTOCOL_LOCK"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R11-A status: {m.get('status')}"
        )

    return m


def read_csv_locked(path):
    require(path)

    return (
        pd.read_csv(path),
        sha256_file(path),
    )


def find_split_label_column(df):
    preferred = [
        "split",
        "dataset_split",
        "role",
        "dataset",
        "subset",
        "partition",
        "source_split",
    ]

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    # Fall back to any object/string column that contains train/test labels.
    for col in df.columns:
        s = df[col].astype(str).str.lower()

        if (
            s.str.contains("test", regex=False).any()
            and
            s.str.contains("train", regex=False).any()
        ):
            return col

    return None


def identify_testreal_row(df, source_name):
    split_col = find_split_label_column(df)

    if split_col is None:
        raise RuntimeError(
            f"{source_name}: could not identify a split-label column. "
            f"Columns={list(df.columns)}"
        )

    labels = (
        df[split_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Strongest match first.
    exact_tokens = {
        "test_real",
        "testreal",
        "test-real",
        "heldout_test_real",
        "heldout_testreal",
    }

    exact_mask = labels.isin(
        exact_tokens
    )

    if int(exact_mask.sum()) == 1:
        return (
            df.loc[exact_mask].iloc[0],
            split_col,
            "exact_testreal_label",
        )

    test_mask = labels.str.contains(
        "test",
        regex=False
    )

    train_mask = labels.str.contains(
        "train",
        regex=False
    )

    candidate_mask = (
        test_mask
        &
        ~train_mask
    )

    if int(candidate_mask.sum()) == 1:
        return (
            df.loc[candidate_mask].iloc[0],
            split_col,
            "unique_test_label",
        )

    raise RuntimeError(
        f"{source_name}: TestReal row not uniquely identified. "
        f"{split_col} values={df[split_col].astype(str).tolist()}"
    )


def validate_metrics(row, source_name):
    missing = [
        col
        for col in PRIMARY_REQUIRED_METRICS
        if col not in row.index
    ]

    if missing:
        raise RuntimeError(
            f"{source_name}: missing required metrics {missing}"
        )

    values = {}

    for col in PRIMARY_REQUIRED_METRICS:
        value = float(row[col])

        if not np.isfinite(value):
            raise RuntimeError(
                f"{source_name}: non-finite {col}={value}"
            )

        if value < 0:
            raise RuntimeError(
                f"{source_name}: negative {col}={value}"
            )

        values[col] = value

    if (
        values["p95_corrected_residual_px"]
        <
        values["median_corrected_residual_px"]
    ):
        raise RuntimeError(
            f"{source_name}: corrected P95 is below corrected median."
        )

    return values


def compare_rows(row_a, row_b):
    rows = []

    for metric in PRIMARY_REQUIRED_METRICS:
        a = float(row_a[metric])
        b = float(row_b[metric])
        diff = abs(a - b)

        rows.append(
            {
                "metric": metric,
                "R8B_value": a,
                "Main2_split_summary_value": b,
                "abs_difference": diff,
                "match_within_1e_12": bool(
                    diff <= 1e-12
                ),
            }
        )

    return pd.DataFrame(rows)


def sequence_level_testreal_context(df):
    """
    Keep sequence-level residual rows for descriptive sensitivity only.
    Do not use their pooled median as the primary floor.
    """
    out = df.copy()

    split_col = find_split_label_column(
        out
    )

    if split_col is not None:
        labels = (
            out[split_col]
            .astype(str)
            .str.lower()
        )

        mask = labels.str.contains(
            "test",
            regex=False
        )

        if mask.any():
            out = out.loc[mask].copy()

    return out


def main():
    print(
        "\nR11-B — EMPIRICAL GEOMETRIC RESIDUAL FLOOR PROTOCOL LOCK"
    )

    print(
        "PROTOCOL LOCK ONLY. NO training, NO inference, NO threshold tuning. "
        "The primary reference is NOT the median across TrainReal and TestReal summary rows. "
        "Instead, this step isolates the held-out BIRDSAI TestReal stationary-landmark row "
        "and locks two empirical post-correction reference scales: "
        "(1) F50 = median corrected stationary-landmark residual in pixels, and "
        "(2) F95 = 95th-percentile corrected stationary-landmark residual in pixels. "
        "These are described as an empirical geometric residual floor/envelope after ego-motion "
        "correction, not as a universal irreducible annotation-noise bound.\n"
    )

    verify_r11a()

    r8b, r8b_hash = read_csv_locked(
        R8B_CSV
    )

    main2_split, main2_split_hash = read_csv_locked(
        MAIN2_SPLIT_CSV
    )

    main2_seq, main2_seq_hash = read_csv_locked(
        MAIN2_SEQUENCE_CSV
    )

    (
        r8b_test,
        r8b_split_col,
        r8b_method,
    ) = identify_testreal_row(
        r8b,
        "R8B stationary-landmark residual result"
    )

    (
        main2_test,
        main2_split_col,
        main2_method,
    ) = identify_testreal_row(
        main2_split,
        "Main Result 2.0 heldout split summary"
    )

    r8b_metrics = validate_metrics(
        r8b_test,
        "R8B TestReal row"
    )

    main2_metrics = validate_metrics(
        main2_test,
        "Main2 TestReal row"
    )

    crosscheck_df = compare_rows(
        r8b_test,
        main2_test
    )

    exact_crosscheck = bool(
        crosscheck_df[
            "match_within_1e_12"
        ].all()
    )

    if not exact_crosscheck:
        raise RuntimeError(
            "R8B and Main Result 2.0 TestReal residual metrics do not match exactly."
        )

    raw_median = float(
        r8b_metrics[
            "median_raw_background_motion_px"
        ]
    )

    f50 = float(
        r8b_metrics[
            "median_corrected_residual_px"
        ]
    )

    f95 = float(
        r8b_metrics[
            "p95_corrected_residual_px"
        ]
    )

    reduction_pct = (
        100.0
        *
        (
            raw_median
            -
            f50
        )
        /
        raw_median
        if raw_median > 0
        else np.nan
    )

    protocol_df = pd.DataFrame(
        [
            {
                "reference_name":
                    "F50",

                "value_px":
                    f50,

                "definition":
                    (
                        "Median corrected stationary-landmark residual on held-out "
                        "BIRDSAI TestReal after ego-motion correction."
                    ),

                "role":
                    (
                        "Primary central empirical geometric residual reference."
                    ),

                "allowed_use":
                    (
                        "Compare pixel-space forecast error magnitude against the typical "
                        "post-correction residual scale on the same held-out BIRDSAI domain."
                    ),

                "not_allowed":
                    (
                        "Do not call this a universal irreducible annotation-noise floor; "
                        "do not directly transfer it to WildBox/BAMBI without separate validation."
                    ),
            },
            {
                "reference_name":
                    "F95",

                "value_px":
                    f95,

                "definition":
                    (
                        "95th-percentile corrected stationary-landmark residual on held-out "
                        "BIRDSAI TestReal after ego-motion correction."
                    ),

                "role":
                    (
                        "Conservative empirical geometric residual envelope."
                    ),

                "allowed_use":
                    (
                        "Quantify what fraction of pixel-space forecast errors lie above a "
                        "high residual envelope on held-out BIRDSAI TestReal."
                    ),

                "not_allowed":
                    (
                        "Do not use as a tuned forecasting threshold; do not infer field "
                        "measurement uncertainty outside the validated BIRDSAI correction setting."
                    ),
            },
        ]
    )

    protocol_path = (
        OUT
        /
        "R11B_EMPIRICAL_GEOMETRIC_RESIDUAL_PROTOCOL.csv"
    )

    protocol_df.to_csv(
        protocol_path,
        index=False
    )

    row_lock_df = pd.DataFrame(
        [
            {
                "source":
                    "R8B_exact_result_extraction",

                "path":
                    str(
                        R8B_CSV
                    ),

                "sha256":
                    r8b_hash,

                "split_column":
                    r8b_split_col,

                "selection_method":
                    r8b_method,

                "split_value":
                    str(
                        r8b_test[
                            r8b_split_col
                        ]
                    ),

                "median_raw_background_motion_px":
                    raw_median,

                "median_corrected_residual_px":
                    f50,

                "p95_corrected_residual_px":
                    f95,

                "median_residual_reduction_pct_recomputed":
                    reduction_pct,
            },
            {
                "source":
                    "Main_Result_2.0_heldout_split_summary",

                "path":
                    str(
                        MAIN2_SPLIT_CSV
                    ),

                "sha256":
                    main2_split_hash,

                "split_column":
                    main2_split_col,

                "selection_method":
                    main2_method,

                "split_value":
                    str(
                        main2_test[
                            main2_split_col
                        ]
                    ),

                "median_raw_background_motion_px":
                    float(
                        main2_test[
                            "median_raw_background_motion_px"
                        ]
                    ),

                "median_corrected_residual_px":
                    float(
                        main2_test[
                            "median_corrected_residual_px"
                        ]
                    ),

                "p95_corrected_residual_px":
                    float(
                        main2_test[
                            "p95_corrected_residual_px"
                        ]
                    ),

                "median_residual_reduction_pct_recomputed":
                    reduction_pct,
            },
        ]
    )

    row_lock_path = (
        OUT
        /
        "R11B_TESTREAL_RESIDUAL_ROW_LOCK.csv"
    )

    row_lock_df.to_csv(
        row_lock_path,
        index=False
    )

    crosscheck_path = (
        OUT
        /
        "R11B_SOURCE_CROSSCHECK.csv"
    )

    crosscheck_df.to_csv(
        crosscheck_path,
        index=False
    )

    test_seq = sequence_level_testreal_context(
        main2_seq
    )

    seq_path = (
        OUT
        /
        "R11B_TESTREAL_SEQUENCE_RESIDUAL_CONTEXT.csv"
    )

    test_seq.to_csv(
        seq_path,
        index=False
    )

    manuscript_text = f"""R11 empirical geometric residual reference

Protocol definition
-------------------
The noise-floor analysis uses held-out BIRDSAI TestReal stationary-landmark residual motion after ego-motion correction as an empirical geometric reference. It does not use forecast error to define the reference.

Primary central reference:
F50 = {f50:.6f} px
Definition: median corrected stationary-landmark residual on held-out BIRDSAI TestReal.

Conservative residual envelope:
F95 = {f95:.6f} px
Definition: 95th-percentile corrected stationary-landmark residual on held-out BIRDSAI TestReal.

For context, the held-out TestReal median raw background motion was {raw_median:.6f} px and the corrected median residual was reduced by {reduction_pct:.3f}% relative to that raw median.

Interpretation guardrail
------------------------
F50 and F95 quantify residual geometric motion remaining after the ego-motion correction procedure in the held-out BIRDSAI camera setting. They are not claimed to be universal annotation noise, an irreducible forecasting error bound, or a transferable uncertainty threshold for Fresh WildBox or BAMBI. Their purpose is to place BIRDSAI TestReal pixel-space forecasting errors on an empirically measured residual-motion scale.
"""

    manuscript_path = (
        OUT
        /
        "R11B_MANUSCRIPT_READY_PROTOCOL.txt"
    )

    manuscript_path.write_text(
        manuscript_text,
        encoding="utf-8"
    )

    status = (
        "R11B_TESTREAL_EMPIRICAL_GEOMETRIC_RESIDUAL_F50_F95_LOCKED_READY_FOR_R11C_FORECAST_ERROR_ABOVE_FLOOR_ANALYSIS"
    )

    manifest = {
        "step":
            "R11B_EMPIRICAL_GEOMETRIC_RESIDUAL_FLOOR_PROTOCOL_LOCK",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "threshold_tuning":
            False,

        "primary_domain":
            "BIRDSAI TestReal",

        "primary_evidence_type":
            "stationary-landmark residual after ego-motion correction",

        "F50_px":
            f50,

        "F95_px":
            f95,

        "raw_median_background_motion_px":
            raw_median,

        "median_reduction_pct":
            reduction_pct,

        "source_crosscheck_exact_within_1e_12":
            exact_crosscheck,

        "terminology":
            "empirical geometric residual reference/floor and residual envelope",

        "claim_guardrails": [
            "Do not call F50 a universal irreducible annotation-noise floor.",
            "Do not transfer BIRDSAI F50/F95 directly to Fresh WildBox or BAMBI.",
            "Do not tune F50/F95 using forecast errors.",
            "Use only pixel-space forecast errors for direct comparison with these pixel-space residual references.",
        ],

        "outputs": {
            "protocol":
                str(
                    protocol_path
                ),

            "testreal_row_lock":
                str(
                    row_lock_path
                ),

            "source_crosscheck":
                str(
                    crosscheck_path
                ),

            "sequence_context":
                str(
                    seq_path
                ),

            "manuscript_ready_protocol":
                str(
                    manuscript_path
                ),
        },
    }

    (
        OUT
        /
        "R11B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "TESTREAL RESIDUAL ROW LOCK:"
    )

    print(
        row_lock_df.to_string(
            index=False
        )
    )

    print(
        "\nSOURCE CROSSCHECK:"
    )

    print(
        crosscheck_df.to_string(
            index=False
        )
    )

    print(
        "\nLOCKED EMPIRICAL RESIDUAL REFERENCES:"
    )

    print(
        protocol_df[
            [
                "reference_name",
                "value_px",
                "role",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nINTERPRETATION:"
    )

    print(
        f"  F50 = {f50:.6f} px: typical post-correction residual geometric motion."
    )

    print(
        f"  F95 = {f95:.6f} px: conservative post-correction residual envelope."
    )

    print(
        "  These are BIRDSAI TestReal geometric residual references, not universal irreducible noise bounds."
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR11-B COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
