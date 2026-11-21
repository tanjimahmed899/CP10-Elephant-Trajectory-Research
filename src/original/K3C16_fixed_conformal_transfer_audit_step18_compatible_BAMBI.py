from pathlib import Path
import json
import math
import re
import hashlib

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

PRED_DIR = (
    BASE
    / "k3_operational_warning"
    / "secondary_frozen_inference_v2"
)

PRED_MANIFEST = (
    PRED_DIR
    / "STEP6D_K3C15_SECONDARY_FROZEN_INFERENCE_LOCK.json"
)

EXPECTED_PRED_MANIFEST_SHA256 = (
    "06a863cf9d86d1cf9faa3416d7fbfe41f0eadcb3d9ffba25845e7cf801a65c33"
)

CANON_DIR = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_canonical_lock_v1"
)

K3B1_V2_DIR = (
    BASE
    / "k3_fixed_conformal"
    / "k3b1_v2_fixed_no_recalibration_animal_track_unit"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "secondary_fixed_conformal_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]


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


def normalize_task_text(value):
    text = str(value).lower().replace(" ", "")

    if any(
        token in text
        for token in [
            "2s_to_2s",
            "2s-to-2s",
            "2to2",
            "2_to_2",
            "2→2",
        ]
    ):
        return "2s_to_2s"

    if any(
        token in text
        for token in [
            "2s_to_5s",
            "2s-to-5s",
            "2to5",
            "2_to_5",
            "2→5",
        ]
    ):
        return "2s_to_5s"

    return None


def alpha_from_value(value):
    try:
        x = float(value)
    except Exception:
        return None

    candidates = {
        0.10:
            0.10,

        0.1:
            0.10,

        0.90:
            0.10,

        0.9:
            0.10,

        90.0:
            0.10,

        0.05:
            0.05,

        0.95:
            0.05,

        95.0:
            0.05,
    }

    for key, alpha in candidates.items():
        if abs(
            x
            -
            key
        ) < 1e-8:
            return alpha

    return None


def candidate_q_columns(columns):
    result = []

    for col in columns:
        low = str(col).lower()

        if (
            low == "q"
            or
            low.startswith("q_")
            or
            "q_track"
            in low
            or
            "normalized_radius"
            in low
            or
            "conformal_q"
            in low
        ):
            if not any(
                bad
                in
                low
                for bad in [
                    "endpoint",
                    "step",
                    "absolute",
                    "radius_m",
                    "radius_abs",
                ]
            ):
                result.append(
                    col
                )

    return result


def infer_task_from_row(row):
    for col, value in row.items():
        task = normalize_task_text(
            value
        )

        if task is not None:
            return task

    return None


def infer_alpha_from_row(row):
    # Prefer explicitly named alpha/coverage/level columns.
    for col, value in row.items():
        low = str(col).lower()

        if any(
            token in low
            for token in [
                "alpha",
                "coverage",
                "level",
                "target",
            ]
        ):
            alpha = alpha_from_value(
                value
            )

            if alpha is not None:
                return alpha

    # Then inspect all scalar values.
    for value in row.values:
        alpha = alpha_from_value(
            value
        )

        if alpha is not None:
            return alpha

    return None


def scan_csv_for_q(path):
    rows = []

    try:
        df = pd.read_csv(
            path
        )

    except Exception:
        return rows

    q_cols = candidate_q_columns(
        df.columns
    )

    if not q_cols:
        return rows

    for idx, row in df.iterrows():
        task = infer_task_from_row(
            row
        )

        alpha = infer_alpha_from_row(
            row
        )

        if task is None or alpha is None:
            continue

        for q_col in q_cols:
            try:
                q = float(
                    row[
                        q_col
                    ]
                )

            except Exception:
                continue

            if (
                not np.isfinite(
                    q
                )
                or
                q
                <=
                0
            ):
                continue

            rows.append({
                "source_file":
                    str(
                        path
                    ),

                "source_kind":
                    "csv",

                "row_index":
                    int(
                        idx
                    ),

                "task":
                    task,

                "alpha":
                    float(
                        alpha
                    ),

                "q":
                    float(
                        q
                    ),

                "q_column":
                    str(
                        q_col
                    ),
            })

    return rows


def recursively_scan_json(
    obj,
    source_file,
    context="root",
    inherited_task=None,
    inherited_alpha=None,
):
    rows = []

    if isinstance(
        obj,
        dict
    ):
        task = inherited_task

        alpha = inherited_alpha

        for key, value in obj.items():
            low = str(
                key
            ).lower()

            if task is None:
                maybe_task = normalize_task_text(
                    value
                )

                if maybe_task is not None:
                    task = maybe_task

                maybe_key_task = normalize_task_text(
                    key
                )

                if maybe_key_task is not None:
                    task = maybe_key_task

            if alpha is None and any(
                token in low
                for token in [
                    "alpha",
                    "coverage",
                    "level",
                    "target",
                ]
            ):
                maybe_alpha = alpha_from_value(
                    value
                )

                if maybe_alpha is not None:
                    alpha = maybe_alpha

        for key, value in obj.items():
            low = str(
                key
            ).lower()

            if (
                task is not None
                and
                alpha is not None
                and
                isinstance(
                    value,
                    (
                        int,
                        float,
                    )
                )
                and
                (
                    low == "q"
                    or
                    low.startswith(
                        "q_"
                    )
                    or
                    "q_track"
                    in
                    low
                    or
                    "conformal_q"
                    in
                    low
                )
                and
                not any(
                    bad
                    in
                    low
                    for bad in [
                        "endpoint",
                        "step",
                        "absolute",
                    ]
                )
            ):
                q = float(
                    value
                )

                if np.isfinite(
                    q
                ) and q > 0:
                    rows.append({
                        "source_file":
                            str(
                                source_file
                            ),

                        "source_kind":
                            "json",

                        "row_index":
                            context,

                        "task":
                            task,

                        "alpha":
                            float(
                                alpha
                            ),

                        "q":
                            q,

                        "q_column":
                            str(
                                key
                            ),
                    })

        for key, value in obj.items():
            rows.extend(
                recursively_scan_json(
                    value,
                    source_file,
                    context=f"{context}.{key}",
                    inherited_task=task,
                    inherited_alpha=alpha,
                )
            )

    elif isinstance(
        obj,
        list
    ):
        for idx, value in enumerate(
            obj
        ):
            rows.extend(
                recursively_scan_json(
                    value,
                    source_file,
                    context=f"{context}[{idx}]",
                    inherited_task=inherited_task,
                    inherited_alpha=inherited_alpha,
                )
            )

    return rows


def scan_json_for_q(path):
    try:
        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return []

    return recursively_scan_json(
        obj,
        path,
    )


def recover_exact_q_values():
    require(
        K3B1_V2_DIR
    )

    rows = []

    for p in sorted(
        K3B1_V2_DIR.rglob(
            "*"
        )
    ):
        if not p.is_file():
            continue

        if p.suffix.lower() == ".csv":
            rows.extend(
                scan_csv_for_q(
                    p
                )
            )

        elif p.suffix.lower() == ".json":
            rows.extend(
                scan_json_for_q(
                    p
                )
            )

    candidates = pd.DataFrame(
        rows
    )

    if len(
        candidates
    ) == 0:
        raise RuntimeError(
            "Could not recover any fixed Step6C q values from K3B1-V2 outputs."
        )

    candidates.to_csv(
        OUT
        / "TABLE_step6dk3c16_q_value_candidates.csv",
        index=False,
    )

    selected = []

    for task in TASKS:
        for alpha in [
            0.10,
            0.05,
        ]:
            subset = candidates[
                (
                    candidates[
                        "task"
                    ]
                    ==
                    task
                )
                &
                (
                    np.isclose(
                        candidates[
                            "alpha"
                        ].to_numpy(
                            dtype=float
                        ),
                        alpha,
                    )
                )
            ].copy()

            if len(
                subset
            ) == 0:
                raise RuntimeError(
                    f"No fixed q candidate found for {task}, alpha={alpha}"
                )

            q_values = np.unique(
                np.round(
                    subset[
                        "q"
                    ].to_numpy(
                        dtype=float
                    ),
                    12,
                )
            )

            if len(
                q_values
            ) != 1:
                raise RuntimeError(
                    f"Ambiguous q values for {task}, alpha={alpha}: {q_values.tolist()}"
                )

            q = float(
                subset.iloc[
                    0
                ][
                    "q"
                ]
            )

            selected.append({
                "task":
                    task,

                "alpha":
                    alpha,

                "target_coverage":
                    1.0
                    -
                    alpha,

                "q":
                    q,

                "source_files":
                    " | ".join(
                        sorted(
                            set(
                                subset[
                                    "source_file"
                                ].astype(
                                    str
                                )
                            )
                        )
                    ),
            })

    selected_df = pd.DataFrame(
        selected
    )

    selected_df.to_csv(
        OUT
        / "TABLE_step6dk3c16_fixed_q_values.csv",
        index=False,
    )

    return (
        selected_df,
        candidates,
    )


def load_task_data(task):
    pred_npz = (
        PRED_DIR
        /
        f"{task}_BAMBI_Flight6_secondary_frozen_ensemble_predictions_LOCKED.npz"
    )

    metadata_csv = (
        PRED_DIR
        /
        f"{task}_BAMBI_Flight6_secondary_frozen_inference_metadata_LOCKED.csv"
    )

    canonical_npz = (
        CANON_DIR
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_canonical_LOCKED.npz"
    )

    for p in [
        pred_npz,
        metadata_csv,
        canonical_npz,
    ]:
        require(
            p
        )

    pred = np.load(
        pred_npz,
        allow_pickle=False,
    )

    canon = np.load(
        canonical_npz,
        allow_pickle=False,
    )

    metadata = pd.read_csv(
        metadata_csv
    )

    prediction_canonical = np.asarray(
        pred[
            "prediction_canonical"
        ],
        dtype=np.float64,
    )

    prediction_anchor = np.asarray(
        pred[
            "prediction_anchor_norm"
        ],
        dtype=np.float64,
    )

    gt_anchor = np.asarray(
        pred[
            "ground_truth_anchor_norm"
        ],
        dtype=np.float64,
    )

    future_canonical = np.asarray(
        canon[
            "future_canonical"
        ],
        dtype=np.float64,
    )

    scale = np.asarray(
        canon[
            "scale"
        ],
        dtype=np.float64,
    )

    if prediction_canonical.shape != future_canonical.shape:
        raise RuntimeError(
            f"{task}: prediction/GT canonical shape mismatch "
            f"{prediction_canonical.shape} vs {future_canonical.shape}"
        )

    if prediction_anchor.shape != gt_anchor.shape:
        raise RuntimeError(
            f"{task}: prediction/GT anchor shape mismatch "
            f"{prediction_anchor.shape} vs {gt_anchor.shape}"
        )

    if len(
        metadata
    ) != prediction_canonical.shape[
        0
    ]:
        raise RuntimeError(
            f"{task}: metadata/sample count mismatch."
        )

    if scale.shape != (
        prediction_canonical.shape[
            0
        ],
    ):
        raise RuntimeError(
            f"{task}: scale shape mismatch {scale.shape}"
        )

    return {
        "pred_npz":
            pred_npz,

        "metadata_csv":
            metadata_csv,

        "canonical_npz":
            canonical_npz,

        "prediction_canonical":
            prediction_canonical,

        "prediction_anchor":
            prediction_anchor,

        "gt_anchor":
            gt_anchor,

        "future_canonical":
            future_canonical,

        "scale":
            scale,

        "metadata":
            metadata,
    }


def coverage_metrics(
    covered,
    metadata,
):
    covered = np.asarray(
        covered,
        dtype=bool,
    )

    window = float(
        covered.mean()
    )

    if "track_id" not in metadata.columns:
        return {
            "window_coverage":
                window,

            "track_macro_coverage":
                None,

            "fully_covered_track_fraction":
                None,
        }

    temp = pd.DataFrame({
        "track_id":
            metadata[
                "track_id"
            ].to_numpy(),

        "covered":
            covered.astype(
                float
            ),
    })

    per_track = (
        temp
        .groupby(
            "track_id",
            as_index=False,
        )
        .agg(
            mean_covered=(
                "covered",
                "mean"
            ),

            fully_covered=(
                "covered",
                "min"
            ),
        )
    )

    return {
        "window_coverage":
            window,

        "track_macro_coverage":
            float(
                per_track[
                    "mean_covered"
                ].mean()
            ),

        "fully_covered_track_fraction":
            float(
                per_track[
                    "fully_covered"
                ].mean()
            ),
    }


def evaluate_task(
    task,
    q_rows,
):
    data = load_task_data(
        task
    )

    err_can = np.linalg.norm(
        data[
            "prediction_canonical"
        ]
        -
        data[
            "future_canonical"
        ],
        axis=2,
    )

    err_anchor = np.linalg.norm(
        data[
            "prediction_anchor"
        ]
        -
        data[
            "gt_anchor"
        ],
        axis=2,
    )

    reconstructed_anchor_err = (
        err_can
        *
        data[
            "scale"
        ][
            :,
            None
        ]
    )

    consistency = np.abs(
        reconstructed_anchor_err
        -
        err_anchor
    )

    max_consistency_diff = float(
        consistency.max()
    )

    if max_consistency_diff > 5e-6:
        raise RuntimeError(
            f"{task}: canonical-to-anchor error consistency failed; "
            f"max diff={max_consistency_diff}"
        )

    sample_max_can = err_can.max(
        axis=1
    )

    endpoint_can = err_can[
        :,
        -1
    ]

    rows = []

    instance_rows = []

    for _, qrow in q_rows.iterrows():
        alpha = float(
            qrow[
                "alpha"
            ]
        )

        target = float(
            qrow[
                "target_coverage"
            ]
        )

        q = float(
            qrow[
                "q"
            ]
        )

        window_covered = (
            sample_max_can
            <=
            q
        )

        endpoint_covered = (
            endpoint_can
            <=
            q
        )

        window_metrics = coverage_metrics(
            window_covered,
            data[
                "metadata"
            ],
        )

        endpoint_metrics = coverage_metrics(
            endpoint_covered,
            data[
                "metadata"
            ],
        )

        abs_radius_norm = (
            q
            *
            data[
                "scale"
            ]
        )

        rows.append({
            "task":
                task,

            "alpha":
                alpha,

            "target_coverage":
                target,

            "q_fixed_step6c":
                q,

            "samples":
                int(
                    len(
                        sample_max_can
                    )
                ),

            "tracks":
                int(
                    data[
                        "metadata"
                    ][
                        "track_id"
                    ].nunique()
                ),

            "window_coverage":
                window_metrics[
                    "window_coverage"
                ],

            "track_macro_window_coverage":
                window_metrics[
                    "track_macro_coverage"
                ],

            "fully_covered_track_fraction":
                window_metrics[
                    "fully_covered_track_fraction"
                ],

            "endpoint_coverage_same_track_q":
                endpoint_metrics[
                    "window_coverage"
                ],

            "endpoint_track_macro_same_track_q":
                endpoint_metrics[
                    "track_macro_coverage"
                ],

            "mean_radius_anchor_norm":
                float(
                    abs_radius_norm.mean()
                ),

            "median_radius_anchor_norm":
                float(
                    np.median(
                        abs_radius_norm
                    )
                ),

            "p95_radius_anchor_norm":
                float(
                    np.quantile(
                        abs_radius_norm,
                        0.95,
                    )
                ),

            "mean_max_canonical_error":
                float(
                    sample_max_can.mean()
                ),

            "median_max_canonical_error":
                float(
                    np.median(
                        sample_max_can
                    )
                ),

            "p95_max_canonical_error":
                float(
                    np.quantile(
                        sample_max_can,
                        0.95,
                    )
                ),

            "max_canonical_anchor_error_consistency_diff":
                max_consistency_diff,

            "recalibrated_on_bambi":
                False,
        })

        for i in range(
            len(
                sample_max_can
            )
        ):
            instance_rows.append({
                "task":
                    task,

                "sample_index":
                    int(
                        i
                    ),

                "track_id":
                    int(
                        data[
                            "metadata"
                        ].iloc[
                            i
                        ][
                            "track_id"
                        ]
                    ),

                "window_start_time_sec":
                    float(
                        data[
                            "metadata"
                        ].iloc[
                            i
                        ][
                            "window_start_time_sec"
                        ]
                    ),

                "alpha":
                    alpha,

                "target_coverage":
                    target,

                "q_fixed_step6c":
                    q,

                "scale_anchor_norm":
                    float(
                        data[
                            "scale"
                        ][
                            i
                        ]
                    ),

                "absolute_radius_anchor_norm":
                    float(
                        abs_radius_norm[
                            i
                        ]
                    ),

                "max_canonical_error":
                    float(
                        sample_max_can[
                            i
                        ]
                    ),

                "endpoint_canonical_error":
                    float(
                        endpoint_can[
                            i
                        ]
                    ),

                "window_covered":
                    bool(
                        window_covered[
                            i
                        ]
                    ),

                "endpoint_covered_same_track_q":
                    bool(
                        endpoint_covered[
                            i
                        ]
                    ),
            })

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            instance_rows
        ),
    )


def main():
    print(
        "\nSTEP 6D-K3C16 — FIXED STEP6C CONFORMAL TRANSFER AUDIT "
        "ON STEP18-COMPATIBLE BAMBI REPRESENTATION"
    )

    print(
        "NO BAMBI recalibration, NO q tuning, NO threshold tuning, "
        "NO geofence metric, and NO Step7 alarm is performed.\n"
    )

    require(
        PRED_MANIFEST
    )

    actual_manifest_hash = sha256_file(
        PRED_MANIFEST
    )

    if actual_manifest_hash != EXPECTED_PRED_MANIFEST_SHA256:
        raise RuntimeError(
            "K3C15 prediction manifest hash mismatch. "
            f"Expected {EXPECTED_PRED_MANIFEST_SHA256}, got {actual_manifest_hash}"
        )

    q_df, q_candidates = recover_exact_q_values()

    print(
        "VERIFIED K3C15 MANIFEST SHA256:"
    )
    print(
        actual_manifest_hash
    )

    print(
        "\nRECOVERED FIXED STEP6C q VALUES:"
    )
    print(
        q_df.to_string(
            index=False
        )
    )

    all_summaries = []

    all_instances = []

    for task in TASKS:
        q_rows = q_df[
            q_df[
                "task"
            ]
            ==
            task
        ].copy()

        summary, instances = evaluate_task(
            task,
            q_rows,
        )

        all_summaries.append(
            summary
        )

        all_instances.append(
            instances
        )

        print(
            f"\n{task} FIXED CONFORMAL SUMMARY:"
        )

        print(
            summary.to_string(
                index=False
            )
        )

    summary_df = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    instance_df = pd.concat(
        all_instances,
        ignore_index=True,
    )

    summary_path = (
        OUT
        / "TABLE_step6dk3c16_fixed_conformal_secondary_summary.csv"
    )

    instance_path = (
        OUT
        / "step6dk3c16_fixed_conformal_secondary_instances.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    instance_df.to_csv(
        instance_path,
        index=False,
    )

    result = {
        "step":
            "STEP_6D_K3C16_FIXED_STEP6C_CONFORMAL_TRANSFER_AUDIT_STEP18_COMPATIBLE_BAMBI",

        "status":
            "FIXED_CONFORMAL_TRANSFER_AUDIT_COMPLETE_NO_RECALIBRATION_READY_FOR_FROZEN_STEP7_EVALUATION",

        "analysis_role":
            (
                "Secondary frozen operational-transfer uncertainty audit. "
                "Not a new pristine confirmatory holdout."
            ),

        "k3c15_prediction_manifest":
            str(
                PRED_MANIFEST
            ),

        "k3c15_prediction_manifest_sha256":
            actual_manifest_hash,

        "fixed_q_source_directory":
            str(
                K3B1_V2_DIR
            ),

        "fixed_q_values":
            q_df.to_dict(
                orient="records"
            ),

        "summary_csv":
            str(
                summary_path
            ),

        "summary_csv_sha256":
            sha256_file(
                summary_path
            ),

        "instance_csv":
            str(
                instance_path
            ),

        "instance_csv_sha256":
            sha256_file(
                instance_path
            ),

        "no_bambi_recalibration":
            True,

        "no_q_tuning":
            True,

        "no_step7_metric_generated":
            True,

        "nonclaims": [
            "This secondary representation has already been methodologically inspected during development.",
            "Coverage here must not be described as a new untouched confirmatory uncertainty holdout.",
            "No failed coverage result may be repaired by recalibrating q on BAMBI.",
            "No Step7 warning/alarm result is produced in K3C16.",
        ],
    }

    result_path = (
        OUT
        / "STEP6D_K3C16_SUMMARY.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSTATUS:"
    )

    print(
        "FIXED_CONFORMAL_TRANSFER_AUDIT_COMPLETE_NO_RECALIBRATION_READY_FOR_FROZEN_STEP7_EVALUATION"
    )

    print(
        "\nSTEP 6D-K3C16 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
