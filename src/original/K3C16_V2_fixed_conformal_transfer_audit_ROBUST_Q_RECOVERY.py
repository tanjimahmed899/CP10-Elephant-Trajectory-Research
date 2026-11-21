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
    / "secondary_fixed_conformal_audit_v2"
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
    text = str(value).lower()
    compact = re.sub(r"[^a-z0-9→]+", "", text)

    patterns_22 = [
        "2sto2s",
        "2to2",
        "2→2",
        "2s2s",
        "h20",
        "future20",
        "pred20",
    ]

    patterns_25 = [
        "2sto5s",
        "2to5",
        "2→5",
        "2s5s",
        "h50",
        "future50",
        "pred50",
    ]

    if any(p in compact or p in text for p in patterns_22):
        return "2s_to_2s"

    if any(p in compact or p in text for p in patterns_25):
        return "2s_to_5s"

    return None


def alpha_from_text(value):
    text = str(value).lower().strip()

    # Explicit alpha wording.
    m = re.search(r"alpha[^0-9]*([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            x = float(m.group(1))
            if abs(x - 0.10) < 1e-8:
                return 0.10
            if abs(x - 0.05) < 1e-8:
                return 0.05
        except Exception:
            pass

    # q90 / q95 / 90% / 95% / coverage level.
    if re.search(r"q[_\-\s]*90\b", text) or "90%" in text:
        return 0.10

    if re.search(r"q[_\-\s]*95\b", text) or "95%" in text:
        return 0.05

    # Pure numeric forms.
    try:
        x = float(text)
    except Exception:
        return None

    if any(abs(x - v) < 1e-8 for v in [0.10, 0.1, 0.90, 0.9, 90.0]):
        return 0.10

    if any(abs(x - v) < 1e-8 for v in [0.05, 0.95, 95.0]):
        return 0.05

    return None


def is_primary_q_label(label):
    low = str(label).lower()

    # Must look q-like.
    q_like = (
        low == "q"
        or
        "q90" in low
        or
        "q95" in low
        or
        "q_90" in low
        or
        "q_95" in low
        or
        "q-track" in low
        or
        "q_track" in low
        or
        "track_q" in low
        or
        "conformal_q" in low
        or
        "normalized_radius" in low
    )

    if not q_like:
        return False

    # Exclude non-primary quantities.
    forbidden = [
        "endpoint",
        "stepwise",
        "step_",
        "per_step",
        "absolute_radius",
        "radius_abs",
        "_abs",
        "metre",
        "meter",
        "_m",
    ]

    return not any(token in low for token in forbidden)


def infer_task_from_context(*values):
    for value in values:
        task = normalize_task_text(value)
        if task is not None:
            return task
    return None


def infer_alpha_from_context(*values):
    for value in values:
        alpha = alpha_from_text(value)
        if alpha is not None:
            return alpha
    return None


def scan_csv_for_q(path):
    rows = []

    try:
        df = pd.read_csv(path)
    except Exception:
        return rows

    file_task = normalize_task_text(path.name)
    file_alpha = alpha_from_text(path.name)

    # CASE A: q90/q95 stored as wide columns.
    for col in df.columns:
        if not is_primary_q_label(col):
            continue

        col_alpha = alpha_from_text(col)
        col_task = normalize_task_text(col)

        for idx, row in df.iterrows():
            task = infer_task_from_context(
                col_task,
                file_task,
                *row.astype(str).tolist(),
            )

            alpha = infer_alpha_from_context(
                col_alpha,
                file_alpha,
                *[
                    row[c]
                    for c in df.columns
                    if any(
                        token in str(c).lower()
                        for token in ["alpha", "coverage", "level", "target"]
                    )
                ],
            )

            if task is None or alpha is None:
                continue

            try:
                q = float(row[col])
            except Exception:
                continue

            if not np.isfinite(q) or q <= 0:
                continue

            rows.append({
                "source_file": str(path),
                "source_kind": "csv",
                "row_index": int(idx),
                "task": task,
                "alpha": float(alpha),
                "q": float(q),
                "q_label": str(col),
                "evidence": "wide-column/row-context",
            })

    # CASE B: long form, where a generic q/q_track column is paired with alpha.
    generic_q_cols = [
        c
        for c in df.columns
        if is_primary_q_label(c)
        and alpha_from_text(c) is None
    ]

    for idx, row in df.iterrows():
        row_values = [row[c] for c in df.columns]

        task = infer_task_from_context(
            file_task,
            *row_values,
        )

        alpha = infer_alpha_from_context(
            file_alpha,
            *[
                row[c]
                for c in df.columns
                if any(
                    token in str(c).lower()
                    for token in ["alpha", "coverage", "level", "target"]
                )
            ],
            *row_values,
        )

        if task is None or alpha is None:
            continue

        for col in generic_q_cols:
            try:
                q = float(row[col])
            except Exception:
                continue

            if not np.isfinite(q) or q <= 0:
                continue

            rows.append({
                "source_file": str(path),
                "source_kind": "csv",
                "row_index": int(idx),
                "task": task,
                "alpha": float(alpha),
                "q": float(q),
                "q_label": str(col),
                "evidence": "long-form-row-context",
            })

    return rows


def recursively_scan_json(obj, source_file, context="root", inherited_task=None, inherited_alpha=None):
    rows = []

    if isinstance(obj, dict):
        task = inherited_task or normalize_task_text(context)
        alpha = inherited_alpha or alpha_from_text(context)

        # Learn task/alpha from this dictionary.
        for key, value in obj.items():
            if task is None:
                task = infer_task_from_context(key, value)

            if alpha is None:
                alpha = infer_alpha_from_context(key, value)

        for key, value in obj.items():
            key_task = normalize_task_text(key) or task
            key_alpha = alpha_from_text(key) or alpha

            if (
                key_task is not None
                and
                key_alpha is not None
                and
                isinstance(value, (int, float))
                and
                is_primary_q_label(key)
            ):
                q = float(value)

                if np.isfinite(q) and q > 0:
                    rows.append({
                        "source_file": str(source_file),
                        "source_kind": "json",
                        "row_index": context,
                        "task": key_task,
                        "alpha": float(key_alpha),
                        "q": q,
                        "q_label": str(key),
                        "evidence": "json-key/context",
                    })

        for key, value in obj.items():
            rows.extend(
                recursively_scan_json(
                    value,
                    source_file,
                    context=f"{context}.{key}",
                    inherited_task=normalize_task_text(key) or task,
                    inherited_alpha=alpha_from_text(key) or alpha,
                )
            )

    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
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
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    return recursively_scan_json(obj, path)


def scan_text_for_q(path):
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    rows = []
    lines = raw.splitlines()
    file_task = normalize_task_text(path.name)
    file_alpha = alpha_from_text(path.name)

    # q labels with exact numeric values.
    q_pattern = re.compile(
        r"(?P<label>(?:track[_\-\s]*)?q[_\-\s]*(?:90|95)|q[_\-\s]*track(?:[_\-\s]*(?:90|95))?|conformal[_\-\s]*q(?:[_\-\s]*(?:90|95))?)"
        r"[^0-9+\-]{0,40}"
        r"(?P<value>[+\-]?(?:\d+\.\d+|\d+)(?:[eE][+\-]?\d+)?)",
        flags=re.IGNORECASE,
    )

    for i, line in enumerate(lines):
        # Include nearby lines for task context.
        lo = max(0, i - 4)
        hi = min(len(lines), i + 5)
        context = "\n".join(lines[lo:hi])

        for match in q_pattern.finditer(line):
            label = match.group("label")

            if not is_primary_q_label(label):
                continue

            task = infer_task_from_context(
                line,
                context,
                file_task,
            )

            alpha = infer_alpha_from_context(
                label,
                line,
                context,
                file_alpha,
            )

            if task is None or alpha is None:
                continue

            try:
                q = float(match.group("value"))
            except Exception:
                continue

            if not np.isfinite(q) or q <= 0:
                continue

            rows.append({
                "source_file": str(path),
                "source_kind": "text",
                "row_index": int(i + 1),
                "task": task,
                "alpha": float(alpha),
                "q": float(q),
                "q_label": label,
                "evidence": line.strip()[:500],
            })

    return rows


def recover_exact_q_values():
    search_dirs = [
        K3B1_V2_DIR,
        BASE / "k3_fixed_conformal" / "source_audit_v3",
    ]

    rows = []

    for folder in search_dirs:
        require(folder)

        for p in sorted(folder.rglob("*")):
            if not p.is_file():
                continue

            suffix = p.suffix.lower()

            if suffix == ".csv":
                rows.extend(scan_csv_for_q(p))

            elif suffix == ".json":
                rows.extend(scan_json_for_q(p))

            elif suffix in {".txt", ".py", ".md", ".log"}:
                rows.extend(scan_text_for_q(p))

    candidates = pd.DataFrame(rows)

    if len(candidates) == 0:
        raise RuntimeError(
            "Could not recover any fixed primary Step6C q values from "
            "K3B1-V2 or source_audit_v3."
        )

    candidates = candidates.drop_duplicates().reset_index(drop=True)

    candidates.to_csv(
        OUT / "TABLE_step6dk3c16_q_value_candidates_v2.csv",
        index=False,
    )

    selected = []

    for task in TASKS:
        for alpha in [0.10, 0.05]:
            subset = candidates[
                (candidates["task"] == task)
                &
                np.isclose(
                    candidates["alpha"].to_numpy(dtype=float),
                    alpha,
                )
            ].copy()

            if len(subset) == 0:
                raise RuntimeError(
                    f"No fixed primary q candidate found for {task}, alpha={alpha}. "
                    "See TABLE_step6dk3c16_q_value_candidates_v2.csv."
                )

            # Cluster numerically identical values despite formatting precision.
            subset["q_rounded"] = np.round(
                subset["q"].to_numpy(dtype=float),
                9,
            )

            unique_values = sorted(
                subset["q_rounded"].unique().tolist()
            )

            if len(unique_values) != 1:
                details = subset[
                    [
                        "q",
                        "q_label",
                        "source_file",
                        "evidence",
                    ]
                ].to_dict(orient="records")

                raise RuntimeError(
                    f"Ambiguous fixed primary q values for {task}, alpha={alpha}: "
                    f"{unique_values}. Evidence={details}"
                )

            # Preserve the highest-precision candidate, using longest decimal
            # representation as a harmless provenance tie-breaker.
            subset["precision_score"] = subset["q"].astype(str).str.len()
            best = subset.sort_values(
                ["precision_score"],
                ascending=False,
            ).iloc[0]

            selected.append({
                "task": task,
                "alpha": float(alpha),
                "target_coverage": float(1.0 - alpha),
                "q": float(best["q"]),
                "source_file": str(best["source_file"]),
                "q_label": str(best["q_label"]),
                "evidence": str(best["evidence"]),
                "candidate_count": int(len(subset)),
            })

    selected_df = pd.DataFrame(selected)

    selected_df.to_csv(
        OUT / "TABLE_step6dk3c16_fixed_q_values_v2.csv",
        index=False,
    )

    return selected_df, candidates

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
        "\nSTEP 6D-K3C16-V2 — FIXED STEP6C CONFORMAL TRANSFER AUDIT "
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
            "STEP_6D_K3C16_V2_FIXED_STEP6C_CONFORMAL_TRANSFER_AUDIT_STEP18_COMPATIBLE_BAMBI",

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
