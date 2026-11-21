
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np


BASE = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0"
)

STEP2 = BASE / "step2_ego_motion_validation"
STEP2B = BASE / "step2b_raw_vs_corrected_baseline"
STEP2C = BASE / "step2c_ego_motion_geofence_ablation"

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "paper_strengthening_revision"
    / "R8_ego_motion_ablation"
    / "R8B_exact_result_extraction"
)

OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "residual_split":
        STEP2 / "heldout_background_residual_split_summary.csv",

    "residual_sequence":
        STEP2 / "heldout_background_residual_summary.csv",

    "forecast_summary":
        STEP2B / "raw_vs_corrected_summary.csv",

    "forecast_sequence":
        STEP2B / "raw_vs_corrected_per_sequence.csv",

    "warning_impact":
        STEP2C / "ego_motion_warning_impact_summary.csv",

    "warning_summary":
        STEP2C / "raw_vs_corrected_geofence_summary.csv",

    "label_distortion":
        STEP2C / "raw_imageplane_label_distortion_audit.csv",
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


def read_all():
    out = {}

    for key, path in FILES.items():
        require(path)
        out[key] = pd.read_csv(path)

    return out


def compact_residual(df):
    keep = [
        c for c in [
            "split",
            "rows",
            "sequences",
            "median_raw_background_motion_px",
            "median_corrected_residual_px",
            "median_residual_reduction_pct",
            "p95_corrected_residual_px",
        ]
        if c in df.columns
    ]

    return df[keep].copy()


def compact_forecast(df):
    identity = [
        c for c in [
            "dataset",
            "split",
            "task",
            "model",
            "samples",
            "tracks",
            "sequences",
        ]
        if c in df.columns
    ]

    metrics = [
        c for c in [
            "raw_mean_ADE",
            "corrected_mean_ADE",
            "ADE_improvement_pct",
            "raw_mean_FDE",
            "corrected_mean_FDE",
            "FDE_improvement_pct",
            "raw_track_macro_ADE",
            "corrected_track_macro_ADE",
            "track_macro_ADE_improvement_pct",
            "raw_track_macro_FDE",
            "corrected_track_macro_FDE",
            "track_macro_FDE_improvement_pct",
            "raw_source_macro_ADE",
            "corrected_source_macro_ADE",
            "source_macro_ADE_improvement_pct",
            "raw_source_macro_FDE",
            "corrected_source_macro_FDE",
            "source_macro_FDE_improvement_pct",
        ]
        if c in df.columns
    ]

    return df[identity + metrics].copy()


def compact_warning(df):
    identity = [
        c for c in [
            "dataset",
            "split",
            "task",
            "model",
        ]
        if c in df.columns
    ]

    metrics = [
        c for c in [
            "raw_precision",
            "corrected_precision",
            "precision_change_pp",
            "raw_recall",
            "corrected_recall",
            "recall_change_pp",
            "raw_F1",
            "corrected_F1",
            "F1_change_pp",
            "raw_false_warning_rate",
            "corrected_false_warning_rate",
            "false_warning_rate_reduction_pct",
            "raw_AUROC",
            "corrected_AUROC",
            "raw_AUPRC",
            "corrected_AUPRC",
            "raw_TTE_MAE_sec",
            "corrected_TTE_MAE_sec",
            "TTE_MAE_reduction_pct",
        ]
        if c in df.columns
    ]

    return df[identity + metrics].copy()


def derive_claim_checks(residual, forecast, warning, label_distortion):
    rows = []

    # Residual consistency.
    for _, r in residual.iterrows():
        raw = r.get("median_raw_background_motion_px", np.nan)
        corr = r.get("median_corrected_residual_px", np.nan)

        rows.append(
            {
                "claim_area": "stationary_landmark_residual",
                "condition": str(
                    r.get(
                        "split",
                        r.get(
                            "dataset",
                            "unspecified",
                        ),
                    )
                ),
                "supported": bool(
                    pd.notna(raw)
                    and pd.notna(corr)
                    and corr < raw
                ),
                "evidence": (
                    f"median raw={raw}, corrected={corr}, "
                    f"reduction_pct={r.get('median_residual_reduction_pct', np.nan)}"
                ),
            }
        )

    # Forecast improvement.
    for _, r in forecast.iterrows():
        task = str(
            r.get(
                "task",
                "unspecified",
            )
        )

        raw_ade = r.get(
            "raw_mean_ADE",
            np.nan,
        )

        corr_ade = r.get(
            "corrected_mean_ADE",
            np.nan,
        )

        raw_fde = r.get(
            "raw_mean_FDE",
            np.nan,
        )

        corr_fde = r.get(
            "corrected_mean_FDE",
            np.nan,
        )

        supported = bool(
            pd.notna(raw_ade)
            and pd.notna(corr_ade)
            and pd.notna(raw_fde)
            and pd.notna(corr_fde)
            and corr_ade < raw_ade
            and corr_fde < raw_fde
        )

        rows.append(
            {
                "claim_area": "forecast_raw_vs_corrected",
                "condition": task,
                "supported": supported,
                "evidence": (
                    f"ADE {raw_ade}->{corr_ade}; "
                    f"FDE {raw_fde}->{corr_fde}; "
                    f"ADE_improvement_pct={r.get('ADE_improvement_pct', np.nan)}; "
                    f"FDE_improvement_pct={r.get('FDE_improvement_pct', np.nan)}"
                ),
            }
        )

    # Warning improvement.
    for _, r in warning.iterrows():
        task = str(
            r.get(
                "task",
                r.get(
                    "dataset",
                    "unspecified",
                ),
            )
        )

        raw_fwr = r.get(
            "raw_false_warning_rate",
            np.nan,
        )

        corr_fwr = r.get(
            "corrected_false_warning_rate",
            np.nan,
        )

        raw_f1 = r.get(
            "raw_F1",
            np.nan,
        )

        corr_f1 = r.get(
            "corrected_F1",
            np.nan,
        )

        rows.append(
            {
                "claim_area": "warning_raw_vs_corrected",
                "condition": task,
                "supported": bool(
                    pd.notna(raw_fwr)
                    and pd.notna(corr_fwr)
                    and corr_fwr <= raw_fwr
                ),
                "evidence": (
                    f"F1 {raw_f1}->{corr_f1}; "
                    f"false_warning_rate {raw_fwr}->{corr_fwr}; "
                    f"reduction_pct={r.get('false_warning_rate_reduction_pct', np.nan)}; "
                    f"TTE_MAE {r.get('raw_TTE_MAE_sec', np.nan)}"
                    f"->{r.get('corrected_TTE_MAE_sec', np.nan)}"
                ),
            }
        )

    # Raw-label distortion warning.
    for _, r in label_distortion.iterrows():
        disagreement_pct = r.get(
            "raw_vs_corrected_truth_disagreement_pct",
            np.nan,
        )

        rows.append(
            {
                "claim_area": "raw_imageplane_label_distortion",
                "condition": str(
                    r.get(
                        "task",
                        r.get(
                            "dataset",
                            "unspecified",
                        ),
                    )
                ),
                "supported": bool(
                    pd.notna(disagreement_pct)
                ),
                "evidence": (
                    f"raw_vs_corrected_truth_disagreement_pct={disagreement_pct}"
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR8-B — EXACT EGO-MOTION RESULT EXTRACTION"
    )

    print(
        "NO new experiment. This step extracts the existing stationary-landmark, "
        "forecasting, warning and raw-label-distortion evidence exactly as stored.\n"
    )

    data = read_all()

    residual = compact_residual(
        data["residual_split"]
    )

    forecast = compact_forecast(
        data["forecast_summary"]
    )

    warning = compact_warning(
        data["warning_impact"]
    )

    label = data[
        "label_distortion"
    ].copy()

    claim_checks = derive_claim_checks(
        residual,
        forecast,
        warning,
        label,
    )

    residual_path = (
        OUT
        / "R8B_stationary_landmark_residual_results.csv"
    )

    forecast_path = (
        OUT
        / "R8B_raw_vs_corrected_forecasting_results.csv"
    )

    warning_path = (
        OUT
        / "R8B_raw_vs_corrected_warning_results.csv"
    )

    label_path = (
        OUT
        / "R8B_raw_imageplane_label_distortion.csv"
    )

    claims_path = (
        OUT
        / "R8B_evidence_sufficiency_checks.csv"
    )

    residual.to_csv(
        residual_path,
        index=False,
    )

    forecast.to_csv(
        forecast_path,
        index=False,
    )

    warning.to_csv(
        warning_path,
        index=False,
    )

    label.to_csv(
        label_path,
        index=False,
    )

    claim_checks.to_csv(
        claims_path,
        index=False,
    )

    status = (
        "R8B_EXACT_EXISTING_EGO_MOTION_RESULTS_EXTRACTED_READY_FOR_R8C_CLAIM_LOCK"
    )

    manifest = {
        "step":
            "R8B_EXACT_EGO_MOTION_RESULT_EXTRACTION",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_experiment":
            False,

        "source_files":
            {
                key: {
                    "path":
                        str(path),

                    "sha256":
                        sha256_file(path),
                }
                for key, path in FILES.items()
            },

        "residual_result_csv":
            str(residual_path),

        "forecast_result_csv":
            str(forecast_path),

        "warning_result_csv":
            str(warning_path),

        "label_distortion_csv":
            str(label_path),

        "evidence_checks_csv":
            str(claims_path),
    }

    (
        OUT
        / "R8B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "STATIONARY-LANDMARK RESIDUAL RESULTS:"
    )
    print(
        residual.to_string(
            index=False
        )
    )

    print(
        "\nRAW vs CORRECTED FORECASTING RESULTS:"
    )
    print(
        forecast.to_string(
            index=False
        )
    )

    print(
        "\nRAW vs CORRECTED WARNING RESULTS:"
    )
    print(
        warning.to_string(
            index=False
        )
    )

    print(
        "\nRAW IMAGE-PLANE LABEL DISTORTION:"
    )
    print(
        label.to_string(
            index=False
        )
    )

    print(
        "\nEVIDENCE SUFFICIENCY CHECKS:"
    )
    print(
        claim_checks.to_string(
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
        "\nR8-B COMPLETE"
    )
    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
