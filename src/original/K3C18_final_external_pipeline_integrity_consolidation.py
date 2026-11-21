from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

FILES = {
    "k3c10_prelock":
        BASE
        / "k3_operational_warning"
        / "step7_external_prelock"
        / "STEP6D_K3C10_BAMBI_STEP7_PRELOCK.json",

    "k3c13_canonical_lock":
        BASE
        / "k3_operational_warning"
        / "anchor_normalized_canonical_lock_v1"
        / "STEP6D_K3C13_CANONICAL_DATASET_LOCK.json",

    "k3c15_prediction_lock":
        BASE
        / "k3_operational_warning"
        / "secondary_frozen_inference_v2"
        / "STEP6D_K3C15_SECONDARY_FROZEN_INFERENCE_LOCK.json",

    "k3c16_summary":
        BASE
        / "k3_operational_warning"
        / "secondary_fixed_conformal_audit_v3"
        / "STEP6D_K3C16_SUMMARY.json",

    "k3c16_metrics":
        BASE
        / "k3_operational_warning"
        / "secondary_fixed_conformal_audit_v3"
        / "TABLE_step6dk3c16_fixed_conformal_secondary_summary.csv",

    "k3c17_manifest":
        BASE
        / "k3_operational_warning"
        / "frozen_step7_external_evaluation_v1"
        / "STEP6D_K3C17_FROZEN_STEP7_EXTERNAL_EVALUATION_MANIFEST.json",

    "k3c17_metrics":
        BASE
        / "k3_operational_warning"
        / "frozen_step7_external_evaluation_v1"
        / "TABLE_step6dk3c17_frozen_step7_metrics.csv",

    "k3c17_boundary_macro":
        BASE
        / "k3_operational_warning"
        / "frozen_step7_external_evaluation_v1"
        / "TABLE_step6dk3c17_boundary_macro_metrics.csv",
}

EXPECTED_HASHES = {
    "k3c10_prelock":
        "8cc687c3f91ecbfb2a3c0f491427d91ccfdca959659a52e0836a98889c890bd1",

    "k3c13_canonical_lock":
        "71d83c774233538342f4b410a789750943e61a8513fe983249c8f8f5a2aee605",

    "k3c15_prediction_lock":
        "06a863cf9d86d1cf9faa3416d7fbfe41f0eadcb3d9ffba25845e7cf801a65c33",

    "k3c16_summary":
        "21685938c4995476cd6a45acca525c800a30ff12913690b059e92f16506e3104",

    "k3c17_manifest":
        "5f89576a3d27114b7a8d599c6a2c3a7f88574777a0d2dae810420b7a862d22ab",
}

OUT = (
    BASE
    / "k3_operational_warning"
    / "final_external_pipeline_consolidation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def require(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)


def verify_core_hashes():
    rows = []

    for key, expected in EXPECTED_HASHES.items():
        path = FILES[key]

        require(path)

        actual = sha256_file(
            path
        )

        ok = (
            actual
            ==
            expected
        )

        rows.append({
            "artifact":
                key,

            "path":
                str(
                    path
                ),

            "expected_sha256":
                expected,

            "actual_sha256":
                actual,

            "match":
                ok,
        })

        if not ok:
            raise RuntimeError(
                f"Hash mismatch for {key}\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}\n"
                f"Path: {path}"
            )

    return pd.DataFrame(
        rows
    )


def extract_conformal_summary():
    require(
        FILES[
            "k3c16_metrics"
        ]
    )

    df = pd.read_csv(
        FILES[
            "k3c16_metrics"
        ]
    )

    needed = [
        "task",
        "alpha",
        "target_coverage",
        "q_fixed_step6c",
        "window_coverage",
        "track_macro_window_coverage",
        "fully_covered_track_fraction",
        "recalibrated_on_bambi",
    ]

    missing = [
        c
        for c in needed
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"K3C16 metrics missing columns: {missing}"
        )

    return df[
        needed
    ].copy()


def extract_step7_summary():
    require(
        FILES[
            "k3c17_metrics"
        ]
    )

    df = pd.read_csv(
        FILES[
            "k3c17_metrics"
        ]
    )

    needed = [
        "task",
        "response_threshold_sec",
        "instances",
        "true_crossing_instances",
        "urgent_instances",
        "combined_alert_instances",
        "red_instances",
        "yellow_instances",
        "green_instances",
        "combined_alert_sensitivity",
        "combined_false_alert_rate",
        "red_urgent_sensitivity",
        "red_nonurgent_false_positive_rate",
        "red_precision_for_urgent",
        "combined_alert_precision_for_crossing",
    ]

    missing = [
        c
        for c in needed
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"K3C17 metrics missing columns: {missing}"
        )

    return df[
        needed
    ].copy()


def summarize_task_ranges(step7):
    rows = []

    for task, group in step7.groupby(
        "task",
        sort=False,
    ):
        rows.append({
            "task":
                task,

            "response_threshold_min_sec":
                float(
                    group[
                        "response_threshold_sec"
                    ].min()
                ),

            "response_threshold_max_sec":
                float(
                    group[
                        "response_threshold_sec"
                    ].max()
                ),

            "combined_alert_sensitivity_min":
                float(
                    group[
                        "combined_alert_sensitivity"
                    ].min()
                ),

            "combined_alert_sensitivity_max":
                float(
                    group[
                        "combined_alert_sensitivity"
                    ].max()
                ),

            "combined_false_alert_rate_min":
                float(
                    group[
                        "combined_false_alert_rate"
                    ].min()
                ),

            "combined_false_alert_rate_max":
                float(
                    group[
                        "combined_false_alert_rate"
                    ].max()
                ),

            "red_urgent_sensitivity_min":
                float(
                    group[
                        "red_urgent_sensitivity"
                    ].min()
                ),

            "red_urgent_sensitivity_max":
                float(
                    group[
                        "red_urgent_sensitivity"
                    ].max()
                ),

            "red_nonurgent_false_positive_rate_min":
                float(
                    group[
                        "red_nonurgent_false_positive_rate"
                    ].min()
                ),

            "red_nonurgent_false_positive_rate_max":
                float(
                    group[
                        "red_nonurgent_false_positive_rate"
                    ].max()
                ),

            "red_precision_for_urgent_min":
                float(
                    group[
                        "red_precision_for_urgent"
                    ].min()
                ),

            "red_precision_for_urgent_max":
                float(
                    group[
                        "red_precision_for_urgent"
                    ].max()
                ),

            "combined_alert_precision_for_crossing":
                float(
                    group[
                        "combined_alert_precision_for_crossing"
                    ].iloc[
                        0
                    ]
                ),
        })

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nSTEP 6D-K3C18 — FINAL EXTERNAL PIPELINE INTEGRITY + CONSOLIDATION"
    )

    print(
        "Read-only finalization. NO model inference, NO q recalibration, "
        "NO threshold/persistence tuning, and NO Step7 rerun.\n"
    )

    integrity = verify_core_hashes()

    conformal = extract_conformal_summary()

    step7 = extract_step7_summary()

    task_ranges = summarize_task_ranges(
        step7
    )

    integrity_path = (
        OUT
        / "TABLE_step6dk3c18_core_hash_integrity.csv"
    )

    conformal_path = (
        OUT
        / "TABLE_step6dk3c18_conformal_summary.csv"
    )

    step7_path = (
        OUT
        / "TABLE_step6dk3c18_step7_summary.csv"
    )

    ranges_path = (
        OUT
        / "TABLE_step6dk3c18_step7_task_ranges.csv"
    )

    integrity.to_csv(
        integrity_path,
        index=False,
    )

    conformal.to_csv(
        conformal_path,
        index=False,
    )

    step7.to_csv(
        step7_path,
        index=False,
    )

    task_ranges.to_csv(
        ranges_path,
        index=False,
    )

    # Core interpretation, frozen from existing outputs only.
    q90 = conformal[
        np.isclose(
            pd.to_numeric(
                conformal[
                    "alpha"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            ),
            0.10,
        )
    ].copy()

    q90_cov = {
        str(
            row[
                "task"
            ]
        ):
            float(
                row[
                    "window_coverage"
                ]
            )
        for _, row in q90.iterrows()
    }

    final = {
        "step":
            "STEP_6D_K3C18_FINAL_EXTERNAL_PIPELINE_INTEGRITY_CONSOLIDATION",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "BAMBI_SECONDARY_EXTERNAL_OPERATIONAL_PIPELINE_FINALIZED_AND_FROZEN",

        "analysis_role":
            (
                "BAMBI Flight 6 cross-species/cross-domain secondary external "
                "operational-transfer stress test."
            ),

        "core_hashes_verified":
            True,

        "q90_window_coverage":
            q90_cov,

        "primary_findings": {
            "2s_to_2s":
                (
                    "Forecast transfer is substantially stronger than 5s. "
                    "Fixed q90 coverage is 100%, indicating a highly conservative "
                    "uncertainty radius on this secondary representation. The combined "
                    "warning layer is extremely sensitive but produces many false alerts; "
                    "RED alerts remain much more selective and precise."
                ),

            "2s_to_5s":
                (
                    "Long-horizon forecast and uncertainty transfer degrade strongly. "
                    "Fixed q90 coverage is severely below nominal, so the 5s uncertainty-assisted "
                    "warning layer must not be described as 90%-calibrated. Combined alerts "
                    "retain high event sensitivity but RED urgent-event sensitivity is modest."
                ),
        },

        "critical_nonclaims": [
            "BAMBI Flight 6 is red-deer/human dominated and is not elephant-specific validation.",
            "The analysis is a secondary operational-transfer stress test, not a pristine confirmatory uncertainty holdout.",
            "The Flight-6 50 degree FOVY is an assumed/default reference-pipeline value, not independently measured calibration.",
            "Virtual geofences are normalized synthetic boundaries, not real village/farm boundaries.",
            "Response thresholds are scenario assumptions, not measured field-team response times.",
            "No BAMBI q recalibration, model tuning, threshold tuning, persistence tuning, or response-threshold tuning was performed.",
            "The 5s q90 uncertainty layer is not calibrated on this secondary representation.",
            "This is not field validation.",
        ],

        "files": {
            "integrity_csv":
                str(
                    integrity_path
                ),

            "conformal_summary_csv":
                str(
                    conformal_path
                ),

            "step7_summary_csv":
                str(
                    step7_path
                ),

            "step7_task_ranges_csv":
                str(
                    ranges_path
                ),
        },
    }

    final_path = (
        OUT
        / "STEP6D_K3C18_FINAL_EXTERNAL_PIPELINE_SUMMARY.json"
    )

    final_path.write_text(
        json.dumps(
            final,
            indent=2,
        ),
        encoding="utf-8",
    )

    receipt = {
        "summary":
            str(
                final_path
            ),

        "summary_sha256":
            sha256_file(
                final_path
            ),

        "status":
            final[
                "status"
            ],
    }

    receipt_path = (
        OUT
        / "STEP6D_K3C18_FINAL_RECEIPT.json"
    )

    receipt_path.write_text(
        json.dumps(
            receipt,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "CORE HASH INTEGRITY:"
    )
    print(
        integrity.to_string(
            index=False
        )
    )

    print(
        "\nFINAL CONFORMAL SUMMARY:"
    )
    print(
        conformal.to_string(
            index=False
        )
    )

    print(
        "\nFINAL STEP7 TASK RANGES:"
    )
    print(
        task_ranges.to_string(
            index=False
        )
    )

    print(
        "\nFINAL SUMMARY:"
    )
    print(
        final_path
    )

    print(
        "FINAL SUMMARY SHA256:"
    )
    print(
        receipt[
            "summary_sha256"
        ]
    )

    print(
        "\nSTATUS:"
    )
    print(
        final[
            "status"
        ]
    )

    print(
        "\nSTEP 6D-K3C18 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
