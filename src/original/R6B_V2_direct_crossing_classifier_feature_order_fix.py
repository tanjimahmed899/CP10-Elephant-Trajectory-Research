from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

R6A_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6A_validation_only_preflight"
)

R6A_MANIFEST = R6A_DIR / "R6A_MANIFEST.json"

VALIDATION = (
    MAIN_RESULT
    / "step18_geofence"
    / "step18a_validation_geofence_instances.csv"
)

EVALUATION = (
    MAIN_RESULT
    / "step18_geofence"
    / "step18b_evaluation"
    / "step18b_all_geofence_instances.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6B_feature_model_threshold_protocol_lock"
)

OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Locked direct-classifier design.
# ---------------------------------------------------------------------
LABEL = "true_crossing"

RAW_FEATURES = [
    "distance_to_boundary",
    "predicted_crossing",
    "predicted_tte_sec",
]

DERIVED_FEATURES = [
    "distance_to_boundary",
    "predicted_crossing_bool",
    "predicted_tte_norm",
    "predicted_tte_available",
]

EXPLICITLY_EXCLUDED = [
    "true_tte_sec",
    "tte_error_sec",
    "tte_abs_error_sec",
    "current_coordinate",
    "boundary",
    "boundary_id",
    "source_group",
    "track_key",
    "sample_index",
]

LEAKAGE_PATTERNS = [
    "true_",
    "actual_",
    "ground_truth",
    "groundtruth",
    "gt_",
    "tte_error",
    "tte_abs_error",
    "future_",
]

TASK_HORIZON_SEC = {
    "2s_to_2s": 2.0,
    "2s_to_5s": 5.0,
}

MODEL_PROTOCOL = {
    "model_family":
        "LogisticRegression",

    "separate_model_per_task":
        True,

    "penalty":
        "l2",

    "C":
        1.0,

    "class_weight":
        "balanced",

    "solver":
        "liblinear",

    "max_iter":
        5000,

    "feature_scaling":
        "StandardScaler fitted inside each development fold; binary features retained numerically",

    "hyperparameter_tuning":
        False,
}

THRESHOLD_PROTOCOL = {
    "development_only":
        True,

    "oof_scheme":
        "Leave-One-Source-Out using source_group on step18a validation/development instances",

    "selection_metric":
        "balanced_accuracy",

    "primary_rule":
        (
            "Choose the probability threshold that maximizes balanced accuracy "
            "on pooled out-of-fold validation predictions within each task."
        ),

    "candidate_thresholds":
        "unique OOF probabilities plus 0.5, clipped to [0,1]",

    "tie_break_1":
        "lower false-positive rate",

    "tie_break_2":
        "higher sensitivity",

    "tie_break_3":
        "threshold closest to 0.5",

    "evaluation_threshold_reselection":
        False,

    "evaluation_recalibration":
        False,
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


def verify_r6a():
    require(R6A_MANIFEST)

    obj = json.loads(
        R6A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        obj.get("status")
        !=
        "R6A_PREFLIGHT_COMPLETE_LEAKAGE_SAFE_VALIDATION_CANDIDATE_FOUND_READY_FOR_R6B_FEATURE_LOCK"
    ):
        raise RuntimeError(
            f"Unexpected R6A status: {obj.get('status')}"
        )

    return obj


def normalize_bool_series(s, name):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")

        bad = vals.dropna()[~vals.dropna().isin([0, 1])]

        if len(bad) > 0:
            raise RuntimeError(
                f"{name} contains numeric values outside 0/1: "
                f"{sorted(bad.unique().tolist())[:20]}"
            )

        return vals.astype("Int64")

    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
    }

    out = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    bad = s[out.isna() & s.notna()].astype(str).unique().tolist()

    if bad:
        raise RuntimeError(
            f"{name} contains unrecognized boolean labels: {bad[:20]}"
        )

    return out.astype("Int64")


def validate_schema(df, name):
    required = {
        "task",
        "source_group",
        "track_key",
        LABEL,
        *RAW_FEATURES,
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{name} missing required columns: {sorted(missing)}"
        )

    bad_tasks = sorted(
        set(df["task"].astype(str))
        -
        set(TASK_HORIZON_SEC)
    )

    if bad_tasks:
        raise RuntimeError(
            f"{name} has unexpected tasks: {bad_tasks}"
        )

    # Label must be binary.
    label_bool = normalize_bool_series(
        df[LABEL],
        f"{name}.{LABEL}",
    )

    if label_bool.isna().any():
        raise RuntimeError(
            f"{name}.{LABEL} contains missing values."
        )

    # predicted_crossing must be parseable binary.
    pred_bool = normalize_bool_series(
        df["predicted_crossing"],
        f"{name}.predicted_crossing",
    )

    if pred_bool.isna().any():
        raise RuntimeError(
            f"{name}.predicted_crossing contains missing/unparseable values."
        )

    for col in [
        "distance_to_boundary",
        "predicted_tte_sec",
    ]:
        converted = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        # distance must be present. predicted_tte may legitimately be missing.
        if col == "distance_to_boundary" and converted.isna().any():
            raise RuntimeError(
                f"{name}.{col} has missing/non-numeric values."
            )

    return label_bool, pred_bool


def leakage_audit(df):
    rows = []

    for col in df.columns:
        low = str(col).lower()

        direct_label = (
            col == LABEL
        )

        explicit_exclusion = (
            col in EXPLICITLY_EXCLUDED
        )

        pattern_leakage = any(
            token in low
            for token in LEAKAGE_PATTERNS
        )

        locked_raw_feature = (
            col in RAW_FEATURES
        )

        allowed = locked_raw_feature

        reason = None

        if direct_label:
            reason = "target_label"
        elif explicit_exclusion:
            reason = "explicitly_excluded"
        elif pattern_leakage:
            reason = "ground_truth_or_future_dependent_pattern"
        elif locked_raw_feature:
            reason = "locked_safe_raw_feature"
        else:
            reason = "not_used_by_minimal_classifier"

        rows.append(
            {
                "column": col,
                "locked_feature": bool(allowed),
                "target_label": bool(direct_label),
                "explicitly_excluded": bool(explicit_exclusion),
                "leakage_pattern_hit": bool(pattern_leakage),
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


def transform_features(df):
    out = pd.DataFrame(
        index=df.index
    )

    out["distance_to_boundary"] = pd.to_numeric(
        df["distance_to_boundary"],
        errors="raise",
    ).astype(float)

    out["predicted_crossing_bool"] = normalize_bool_series(
        df["predicted_crossing"],
        "predicted_crossing",
    ).astype(int)

    raw_tte = pd.to_numeric(
        df["predicted_tte_sec"],
        errors="coerce",
    )

    horizon = (
        df["task"]
        .astype(str)
        .map(TASK_HORIZON_SEC)
        .astype(float)
    )

    if horizon.isna().any():
        raise RuntimeError(
            "Cannot map one or more task values to a forecast horizon."
        )

    out["predicted_tte_available"] = (
        raw_tte.notna()
    ).astype(int)

    # Deterministic, task-aware transformation:
    # - finite predicted TTE is normalized by horizon and clipped;
    # - missing TTE (typically no predicted crossing) receives 1.25,
    #   and availability indicator preserves the distinction.
    norm = raw_tte / horizon

    norm = norm.clip(
        lower=0.0,
        upper=1.25,
    )

    norm = norm.fillna(
        1.25
    )

    out["predicted_tte_norm"] = norm.astype(float)

    # Enforce the exact frozen feature order declared in DERIVED_FEATURES.
    return out[DERIVED_FEATURES].copy()


def audit_distribution(df, name):
    label = normalize_bool_series(
        df[LABEL],
        f"{name}.{LABEL}",
    ).astype(int)

    rows = []

    for (task, source), idx in df.groupby(
        ["task", "source_group"],
        sort=True,
    ).groups.items():
        y = label.loc[idx]

        rows.append(
            {
                "table": name,
                "task": str(task),
                "source_group": str(source),
                "rows": int(len(idx)),
                "positives": int(y.sum()),
                "negatives": int((1 - y).sum()),
                "prevalence": float(y.mean()),
                "has_both_classes": bool(y.nunique() == 2),
            }
        )

    return pd.DataFrame(rows)


def loso_trainability(validation):
    label = normalize_bool_series(
        validation[LABEL],
        f"validation.{LABEL}",
    ).astype(int)

    rows = []

    for task, group in validation.groupby(
        "task",
        sort=True,
    ):
        sources = sorted(
            group["source_group"]
            .astype(str)
            .unique()
            .tolist()
        )

        for heldout in sources:
            train = group[
                group["source_group"].astype(str)
                !=
                heldout
            ]

            test = group[
                group["source_group"].astype(str)
                ==
                heldout
            ]

            y_train = label.loc[
                train.index
            ]

            y_test = label.loc[
                test.index
            ]

            rows.append(
                {
                    "task": str(task),
                    "heldout_source": heldout,
                    "train_rows": int(len(train)),
                    "train_positives": int(y_train.sum()),
                    "train_negatives": int((1 - y_train).sum()),
                    "train_has_both_classes": bool(y_train.nunique() == 2),
                    "heldout_rows": int(len(test)),
                    "heldout_positives": int(y_test.sum()),
                    "heldout_negatives": int((1 - y_test).sum()),
                    "heldout_has_both_classes": bool(y_test.nunique() == 2),
                }
            )

    out = pd.DataFrame(rows)

    if not out["train_has_both_classes"].all():
        raise RuntimeError(
            "At least one leave-one-source-out development training fold lacks both classes."
        )

    return out


def source_overlap(validation, evaluation):
    val = set(
        validation["source_group"]
        .astype(str)
        .unique()
    )

    eva = set(
        evaluation["source_group"]
        .astype(str)
        .unique()
    )

    overlap = sorted(
        val & eva
    )

    return {
        "validation_sources": sorted(val),
        "evaluation_sources": sorted(eva),
        "overlap_sources": overlap,
        "overlap_count": len(overlap),
    }


def evaluation_dataset_audit(df):
    if "dataset" not in df.columns:
        return pd.DataFrame(
            [
                {
                    "dataset": "COLUMN_NOT_PRESENT",
                    "rows": len(df),
                    "sources": df["source_group"].astype(str).nunique(),
                    "tracks": df["track_key"].astype(str).nunique(),
                }
            ]
        )

    return (
        df.groupby(
            ["dataset", "task"],
            as_index=False,
        )
        .agg(
            rows=(LABEL, "size"),
            positives=(LABEL, lambda x: int(normalize_bool_series(x, LABEL).sum())),
            sources=("source_group", "nunique"),
            tracks=("track_key", "nunique"),
        )
    )


def main():
    print(
        "\nR6-B-V2 — DIRECT CROSSING CLASSIFIER FEATURE / MODEL / THRESHOLD PROTOCOL LOCK"
    )

    print(
        "NO classifier training. NO threshold selection. "
        "This step locks a minimal leakage-safe feature set, a fixed logistic-regression "
        "benchmark, and a validation-only leave-one-source-out threshold-selection rule.\n"
    )

    verify_r6a()

    require(VALIDATION)
    require(EVALUATION)

    val = pd.read_csv(
        VALIDATION
    )

    eva = pd.read_csv(
        EVALUATION
    )

    validate_schema(
        val,
        "validation",
    )

    validate_schema(
        eva,
        "evaluation",
    )

    leakage = leakage_audit(
        val
    )

    # Hard guard: no locked feature may be ground-truth dependent.
    bad_locked = leakage[
        leakage["locked_feature"]
        &
        leakage["leakage_pattern_hit"]
    ]

    if len(bad_locked) > 0:
        raise RuntimeError(
            "A locked feature matched a leakage pattern:\n"
            +
            bad_locked.to_string(
                index=False
            )
        )

    val_x = transform_features(
        val
    )

    eva_x = transform_features(
        eva
    )

    if list(val_x.columns) != DERIVED_FEATURES:
        raise RuntimeError(
            f"Validation derived feature order mismatch: {list(val_x.columns)}"
        )

    if list(eva_x.columns) != DERIVED_FEATURES:
        raise RuntimeError(
            f"Evaluation derived feature order mismatch: {list(eva_x.columns)}"
        )

    val_dist = audit_distribution(
        val,
        "validation",
    )

    eva_dist = audit_distribution(
        eva,
        "evaluation",
    )

    loso = loso_trainability(
        val
    )

    overlap = source_overlap(
        val,
        eva,
    )

    if overlap["overlap_count"] != 0:
        raise RuntimeError(
            f"Validation/evaluation source overlap detected: {overlap['overlap_sources']}"
        )

    eval_dataset = evaluation_dataset_audit(
        eva
    )

    feature_stats = pd.DataFrame(
        [
            {
                "table": "validation",
                "rows": len(val_x),
                **{
                    f"{c}_missing": int(val_x[c].isna().sum())
                    for c in DERIVED_FEATURES
                },
            },
            {
                "table": "evaluation",
                "rows": len(eva_x),
                **{
                    f"{c}_missing": int(eva_x[c].isna().sum())
                    for c in DERIVED_FEATURES
                },
            },
        ]
    )

    if feature_stats.filter(
        like="_missing"
    ).to_numpy().sum() != 0:
        raise RuntimeError(
            "Derived feature matrix still contains missing values."
        )

    protocol = {
        "label":
            LABEL,

        "raw_features":
            RAW_FEATURES,

        "derived_features_in_order":
            DERIVED_FEATURES,

        "feature_transformations": {
            "distance_to_boundary":
                "numeric raw value; standardized inside model pipeline",

            "predicted_crossing_bool":
                "binary parse of predicted_crossing",

            "predicted_tte_available":
                "1 when predicted_tte_sec is finite, else 0",

            "predicted_tte_norm":
                (
                    "predicted_tte_sec / task forecast horizon, clipped to [0,1.25]; "
                    "missing set deterministically to 1.25"
                ),
        },

        "excluded_ground_truth_dependent_features": [
            "true_tte_sec",
            "tte_error_sec",
            "tte_abs_error_sec",
        ],

        "excluded_identity_geometry_features": [
            "boundary",
            "boundary_id",
            "current_coordinate",
            "source_group",
            "track_key",
            "sample_index",
        ],

        "model_protocol":
            MODEL_PROTOCOL,

        "threshold_protocol":
            THRESHOLD_PROTOCOL,

        "validation_source":
            str(VALIDATION),

        "validation_source_sha256":
            sha256_file(VALIDATION),

        "evaluation_source":
            str(EVALUATION),

        "evaluation_source_sha256":
            sha256_file(EVALUATION),

        "source_overlap_count":
            overlap["overlap_count"],
    }

    protocol_path = OUT / "R6B_LOCKED_DIRECT_CLASSIFIER_PROTOCOL.json"
    leakage_path = OUT / "R6B_feature_leakage_audit.csv"
    val_dist_path = OUT / "R6B_validation_class_distribution.csv"
    eva_dist_path = OUT / "R6B_evaluation_class_distribution.csv"
    loso_path = OUT / "R6B_validation_LOSO_trainability.csv"
    eval_dataset_path = OUT / "R6B_evaluation_dataset_audit.csv"
    feature_stats_path = OUT / "R6B_derived_feature_completeness.csv"
    overlap_path = OUT / "R6B_source_overlap.json"

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    leakage.to_csv(
        leakage_path,
        index=False,
    )

    val_dist.to_csv(
        val_dist_path,
        index=False,
    )

    eva_dist.to_csv(
        eva_dist_path,
        index=False,
    )

    loso.to_csv(
        loso_path,
        index=False,
    )

    eval_dataset.to_csv(
        eval_dataset_path,
        index=False,
    )

    feature_stats.to_csv(
        feature_stats_path,
        index=False,
    )

    overlap_path.write_text(
        json.dumps(
            overlap,
            indent=2,
        ),
        encoding="utf-8",
    )

    status = (
        "R6B_FEATURE_MODEL_THRESHOLD_PROTOCOL_LOCKED_READY_FOR_R6C_VALIDATION_ONLY_FIT"
    )

    manifest = {
        "step":
            "R6B_DIRECT_CROSSING_CLASSIFIER_PROTOCOL_LOCK",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "status":
            status,

        "classifier_trained":
            False,

        "threshold_selected":
            False,

        "evaluation_used_for_selection":
            False,

        "validation_source":
            str(VALIDATION),

        "validation_source_sha256":
            sha256_file(VALIDATION),

        "evaluation_source":
            str(EVALUATION),

        "evaluation_source_sha256":
            sha256_file(EVALUATION),

        "locked_protocol":
            str(protocol_path),

        "locked_protocol_sha256":
            sha256_file(protocol_path),

        "source_overlap_count":
            overlap["overlap_count"],
    }

    manifest_path = OUT / "R6B_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("LOCKED SAFE FEATURES:")
    print(
        json.dumps(
            {
                "raw_features": RAW_FEATURES,
                "derived_features": DERIVED_FEATURES,
                "label": LABEL,
            },
            indent=2,
        )
    )

    print("\nLEAKAGE EXCLUSIONS:")
    print(
        leakage[
            leakage["reason"].isin(
                [
                    "target_label",
                    "explicitly_excluded",
                    "ground_truth_or_future_dependent_pattern",
                ]
            )
        ].to_string(
            index=False
        )
    )

    print("\nVALIDATION LOSO TRAINABILITY:")
    print(
        loso.to_string(
            index=False
        )
    )

    print("\nVALIDATION / EVALUATION SOURCE OVERLAP:")
    print(
        json.dumps(
            overlap,
            indent=2,
        )
    )

    print("\nEVALUATION DATASET AUDIT:")
    print(
        eval_dataset.to_string(
            index=False
        )
    )

    print("\nLOCKED MODEL PROTOCOL:")
    print(
        json.dumps(
            MODEL_PROTOCOL,
            indent=2,
        )
    )

    print("\nLOCKED THRESHOLD PROTOCOL:")
    print(
        json.dumps(
            THRESHOLD_PROTOCOL,
            indent=2,
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR6-B COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
