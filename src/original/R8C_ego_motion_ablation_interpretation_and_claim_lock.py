
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

R8B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R8_ego_motion_ablation"
    / "R8B_exact_result_extraction"
)

R8B_MANIFEST = R8B_DIR / "R8B_MANIFEST.json"
RESIDUAL = R8B_DIR / "R8B_stationary_landmark_residual_results.csv"
FORECAST = R8B_DIR / "R8B_raw_vs_corrected_forecasting_results.csv"
WARNING = R8B_DIR / "R8B_raw_vs_corrected_warning_results.csv"
LABEL = R8B_DIR / "R8B_raw_imageplane_label_distortion.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R8_ego_motion_ablation"
    / "R8C_interpretation_claim_lock"
)

OUT.mkdir(parents=True, exist_ok=True)


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


def verify():
    for p in [
        R8B_MANIFEST,
        RESIDUAL,
        FORECAST,
        WARNING,
        LABEL,
    ]:
        require(p)

    m = json.loads(
        R8B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        m.get("status")
        !=
        "R8B_EXACT_EXISTING_EGO_MOTION_RESULTS_EXTRACTED_READY_FOR_R8C_CLAIM_LOCK"
    ):
        raise RuntimeError(
            f"Unexpected R8B status: {m.get('status')}"
        )

    return m


def build_compact_summary():
    residual = pd.read_csv(
        RESIDUAL
    )

    forecast = pd.read_csv(
        FORECAST
    )

    warning = pd.read_csv(
        WARNING
    )

    label = pd.read_csv(
        LABEL
    )

    rows = []

    for _, r in residual.iterrows():
        rows.append(
            {
                "evidence_type":
                    "stationary_landmark_residual",

                "condition":
                    str(
                        r["split"]
                    ),

                "metric":
                    "median_background_motion_px",

                "raw_value":
                    float(
                        r[
                            "median_raw_background_motion_px"
                        ]
                    ),

                "corrected_value":
                    float(
                        r[
                            "median_corrected_residual_px"
                        ]
                    ),

                "change":
                    float(
                        r[
                            "median_residual_reduction_pct"
                        ]
                    ),

                "change_unit":
                    "percent_reduction",

                "interpretation":
                    (
                        "correction_reduces_background_residual"
                        if
                        r[
                            "median_corrected_residual_px"
                        ]
                        <
                        r[
                            "median_raw_background_motion_px"
                        ]
                        else
                        "no_reduction"
                    ),
            }
        )

    for _, r in forecast.iterrows():
        for metric in [
            "ADE",
            "FDE",
        ]:
            raw_col = f"raw_mean_{metric}"
            corr_col = f"corrected_mean_{metric}"
            imp_col = f"{metric}_improvement_pct"

            rows.append(
                {
                    "evidence_type":
                        "forecast_raw_vs_corrected",

                    "condition":
                        f"{r['task']} | {r['model']}",

                    "metric":
                        metric,

                    "raw_value":
                        float(
                            r[
                                raw_col
                            ]
                        ),

                    "corrected_value":
                        float(
                            r[
                                corr_col
                            ]
                        ),

                    "change":
                        float(
                            r[
                                imp_col
                            ]
                        ),

                    "change_unit":
                        "percent_improvement",

                    "interpretation":
                        (
                            "corrected_better"
                            if
                            r[
                                corr_col
                            ]
                            <
                            r[
                                raw_col
                            ]
                            else
                            "not_improved"
                        ),
                }
            )

    cv = warning[
        warning["model"]
        ==
        "ConstantVelocity"
    ].copy()

    for _, r in cv.iterrows():
        rows.append(
            {
                "evidence_type":
                    "warning_raw_vs_corrected",

                "condition":
                    f"{r['task']} | ConstantVelocity",

                "metric":
                    "F1",

                "raw_value":
                    float(
                        r[
                            "raw_F1"
                        ]
                    ),

                "corrected_value":
                    float(
                        r[
                            "corrected_F1"
                        ]
                    ),

                "change":
                    float(
                        r[
                            "F1_change_pp"
                        ]
                    ),

                "change_unit":
                    "percentage_points",

                "interpretation":
                    "corrected_better",
            }
        )

        rows.append(
            {
                "evidence_type":
                    "warning_raw_vs_corrected",

                "condition":
                    f"{r['task']} | ConstantVelocity",

                "metric":
                    "false_warning_rate",

                "raw_value":
                    float(
                        r[
                            "raw_false_warning_rate"
                        ]
                    ),

                "corrected_value":
                    float(
                        r[
                            "corrected_false_warning_rate"
                        ]
                    ),

                "change":
                    float(
                        r[
                            "false_warning_rate_reduction_pct"
                        ]
                    ),

                "change_unit":
                    "percent_reduction",

                "interpretation":
                    "corrected_better",
            }
        )

        rows.append(
            {
                "evidence_type":
                    "warning_raw_vs_corrected",

                "condition":
                    f"{r['task']} | ConstantVelocity",

                "metric":
                    "TTE_MAE_sec",

                "raw_value":
                    float(
                        r[
                            "raw_TTE_MAE_sec"
                        ]
                    ),

                "corrected_value":
                    float(
                        r[
                            "corrected_TTE_MAE_sec"
                        ]
                    ),

                "change":
                    float(
                        r[
                            "TTE_MAE_reduction_pct"
                        ]
                    ),

                "change_unit":
                    "percent_reduction",

                "interpretation":
                    "corrected_better",
            }
        )

    stationary = warning[
        warning["model"]
        ==
        "Stationary"
    ].copy()

    for _, r in stationary.iterrows():
        rows.append(
            {
                "evidence_type":
                    "warning_stationary_degenerate_case",

                "condition":
                    f"{r['task']} | Stationary",

                "metric":
                    "recall",

                "raw_value":
                    float(
                        r[
                            "raw_recall"
                        ]
                    ),

                "corrected_value":
                    float(
                        r[
                            "corrected_recall"
                        ]
                    ),

                "change":
                    float(
                        r[
                            "recall_change_pp"
                        ]
                    ),

                "change_unit":
                    "percentage_points",

                "interpretation":
                    (
                        "zero_false_warning_but_zero_recall_not_operational_improvement"
                    ),
            }
        )

    for _, r in label.iterrows():
        rows.append(
            {
                "evidence_type":
                    "raw_imageplane_label_distortion",

                "condition":
                    str(
                        r[
                            "task"
                        ]
                    ),

                "metric":
                    "truth_label_disagreement_pct",

                "raw_value":
                    np.nan,

                "corrected_value":
                    np.nan,

                "change":
                    float(
                        r[
                            "raw_vs_corrected_truth_disagreement_pct"
                        ]
                    ),

                "change_unit":
                    "percent",

                "interpretation":
                    "raw_image_plane_changes_crossing_labels",
            }
        )

    return pd.DataFrame(
        rows
    )


def build_claim_lock():
    return pd.DataFrame(
        [
            {
                "claim_id":
                    "R8C_CLAIM_01",

                "status":
                    "SUPPORTED",

                "claim":
                    (
                        "The ego-motion correction is geometrically meaningful "
                        "because stationary/background residual motion decreases "
                        "after correction in both held-out TestReal and TrainReal."
                    ),

                "basis":
                    (
                        "Median residual: TestReal 0.7055 px -> 0.2033 px; "
                        "TrainReal 1.2008 px -> 0.7540 px."
                    ),
            },

            {
                "claim_id":
                    "R8C_CLAIM_02",

                "status":
                    "SUPPORTED_WITH_LIMITATION",

                "claim":
                    (
                        "Residual suppression is not uniform across sequences/splits."
                    ),

                "basis":
                    (
                        "Median residual reduction is 46.85% on TestReal but "
                        "only 7.48% on TrainReal; difficult sequences remain."
                    ),
            },

            {
                "claim_id":
                    "R8C_CLAIM_03",

                "status":
                    "SUPPORTED",

                "claim":
                    (
                        "Using camera-corrected trajectories substantially improves "
                        "forecasting error relative to raw image-plane trajectories "
                        "for Stationary and Constant-Velocity baselines at both 2s "
                        "and 5s horizons."
                    ),

                "basis":
                    (
                        "Mean ADE reductions span 50.62% to 74.10%; "
                        "mean FDE reductions span 40.73% to 74.66%."
                    ),
            },

            {
                "claim_id":
                    "R8C_CLAIM_04",

                "status":
                    "SUPPORTED",

                "claim":
                    (
                        "For the Constant-Velocity warning experiment, ego-motion "
                        "correction improves warning discrimination and reduces false "
                        "warnings at both horizons."
                    ),

                "basis":
                    (
                        "2s: F1 0.2368->0.5046 and false-warning rate "
                        "0.1065->0.0112; 5s: F1 0.2990->0.5645 and "
                        "false-warning rate 0.1562->0.0267."
                    ),
            },

            {
                "claim_id":
                    "R8C_CLAIM_05",

                "status":
                    "SUPPORTED",

                "claim":
                    (
                        "For Constant Velocity, corrected trajectories reduce "
                        "TTE error at both horizons."
                    ),

                "basis":
                    (
                        "TTE MAE reduction: 38.92% at 2s and 33.58% at 5s."
                    ),
            },

            {
                "claim_id":
                    "R8C_CLAIM_06",

                "status":
                    "SUPPORTED",

                "claim":
                    (
                        "Raw image-plane coordinates can distort the binary crossing "
                        "target relative to the corrected reference representation."
                    ),

                "basis":
                    (
                        "Raw-vs-corrected truth-label disagreement is 8.58% at 2s "
                        "and 12.45% at 5s."
                    ),
            },

            {
                "claim_id":
                    "R8C_CLAIM_07",

                "status":
                    "NOT_SUPPORTED",

                "claim":
                    (
                        "The corrected Stationary warning rule is operationally better "
                        "because its false-warning rate becomes zero."
                    ),

                "basis":
                    (
                        "Corrected Stationary recall also becomes zero at both horizons; "
                        "precision/F1/TTE are undefined because it produces no positive warnings."
                    ),
            },

            {
                "claim_id":
                    "R8C_CLAIM_08",

                "status":
                    "NOT_SUPPORTED",

                "claim":
                    (
                        "Ego-motion correction perfectly removes camera motion."
                    ),

                "basis":
                    (
                        "Corrected stationary residuals remain non-zero and "
                        "suppression varies substantially across splits/sequences."
                    ),
            },

            {
                "claim_id":
                    "R8C_CLAIM_09",

                "status":
                    "METHOD_NOTE",

                "claim":
                    (
                        "The manuscript should call the corrected trajectory a "
                        "camera-corrected estimated reference-frame trajectory, "
                        "not world-coordinate ground truth."
                    ),

                "basis":
                    (
                        "Correction reduces apparent camera motion but does not establish "
                        "metric world-space ground truth."
                    ),
            },

            {
                "claim_id":
                    "R8C_CLAIM_10",

                "status":
                    "METHOD_NOTE",

                "claim":
                    (
                        "Raw-vs-corrected results are an ego-motion ablation and should "
                        "not replace the frozen primary forecasting benchmark."
                    ),

                "basis":
                    (
                        "The ablation isolates representation/camera-motion effects; "
                        "model-comparison claims remain governed by the frozen fair benchmark."
                    ),
            },
        ]
    )


def manuscript_map():
    return pd.DataFrame(
        [
            {
                "paper_section":
                    "Methods — Ego-motion correction",

                "content":
                    (
                        "Define the camera-corrected estimated reference-frame trajectory "
                        "and explain stationary-landmark residual validation."
                    ),
            },

            {
                "paper_section":
                    "Results — Geometric validation",

                "content":
                    (
                        "Report TestReal and TrainReal raw vs corrected background residuals, "
                        "including the weaker TrainReal reduction."
                    ),
            },

            {
                "paper_section":
                    "Results — Forecasting ablation",

                "content":
                    (
                        "Report raw-vs-corrected Stationary/CV ADE and FDE at 2s and 5s."
                    ),
            },

            {
                "paper_section":
                    "Results — Warning ablation",

                "content":
                    (
                        "Use Constant-Velocity warning results for meaningful F1/FWR/TTE "
                        "comparison; explicitly note the degenerate zero-recall Stationary case."
                    ),
            },

            {
                "paper_section":
                    "Discussion — Representation limitation",

                "content":
                    (
                        "State that raw image-plane labels disagree with corrected crossing "
                        "labels by 8.58% (2s) and 12.45% (5s), showing camera-motion-induced "
                        "decision distortion."
                    ),
            },
        ]
    )


def main():
    print(
        "\nR8-C — EGO-MOTION ABLATION INTERPRETATION + CLAIM LOCK"
    )

    print(
        "NO new experiment. This step converts the existing R8 evidence "
        "into manuscript-safe claims and explicitly blocks overclaims.\n"
    )

    verify()

    summary = build_compact_summary()
    claims = build_claim_lock()
    mapping = manuscript_map()

    summary_path = (
        OUT
        / "R8C_EGO_MOTION_EVIDENCE_SUMMARY.csv"
    )

    claim_path = (
        OUT
        / "R8C_CLAIM_LOCK.csv"
    )

    mapping_path = (
        OUT
        / "R8C_MANUSCRIPT_SECTION_MAP.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    claims.to_csv(
        claim_path,
        index=False,
    )

    mapping.to_csv(
        mapping_path,
        index=False,
    )

    status = (
        "R8C_EGO_MOTION_ABLATION_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R9_CANONICALIZATION_ABLATION"
    )

    manifest = {
        "step":
            "R8C_EGO_MOTION_ABLATION_INTERPRETATION_CLAIM_LOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_experiment":
            False,

        "existing_evidence_sufficient":
            True,

        "additional_ego_motion_experiment_required":
            False,

        "source_r8b_manifest":
            str(
                R8B_MANIFEST
            ),

        "source_r8b_manifest_sha256":
            sha256_file(
                R8B_MANIFEST
            ),

        "evidence_summary_csv":
            str(
                summary_path
            ),

        "evidence_summary_sha256":
            sha256_file(
                summary_path
            ),

        "claim_lock_csv":
            str(
                claim_path
            ),

        "claim_lock_sha256":
            sha256_file(
                claim_path
            ),

        "manuscript_map_csv":
            str(
                mapping_path
            ),
    }

    (
        OUT
        / "R8C_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "EGO-MOTION EVIDENCE SUMMARY:"
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nCLAIM LOCK:"
    )
    print(
        claims.to_string(
            index=False
        )
    )

    print(
        "\nMANUSCRIPT SECTION MAP:"
    )
    print(
        mapping.to_string(
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
        "\nR8-C COMPLETE"
    )
    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
