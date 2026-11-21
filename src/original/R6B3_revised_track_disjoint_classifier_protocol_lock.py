from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

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

R6B2V3_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6B2V3_track_group_cv_no_sklearn"
)

R6B2V3_MANIFEST = R6B2V3_DIR / "R6B2V3_MANIFEST.json"
SELECTED_SCHEME = R6B2V3_DIR / "R6B2V3_SELECTED_TRACK_GROUP_CV_SCHEME.csv"
FOLD_DETAILS = R6B2V3_DIR / "R6B2V3_track_group_cv_fold_details.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6B3_revised_protocol_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

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

GROUND_TRUTH_EXCLUSIONS = [
    "true_tte_sec",
    "tte_error_sec",
    "tte_abs_error_sec",
]

IDENTITY_EXCLUSIONS = [
    "source_group",
    "track_key",
    "sample_index",
    "boundary",
    "boundary_id",
    "current_coordinate",
]

TASK_HORIZON_SEC = {
    "2s_to_2s": 2.0,
    "2s_to_5s": 5.0,
}

MODEL_PROTOCOL = {
    "model_family":
        "binary logistic regression implemented with numpy only",

    "separate_model_per_task":
        True,

    "objective":
        "weighted binary cross-entropy + L2 penalty",

    "class_weight":
        "balanced within each training fold",

    "l2_lambda":
        1.0,

    "optimizer":
        "deterministic Newton/IRLS with numerical damping",

    "max_iter":
        200,

    "convergence_tolerance":
        1e-9,

    "feature_scaling":
        (
            "Fit mean/std on training fold only for all four numeric features; "
            "apply the same fold scaler to held-out fold. "
            "For final frozen evaluation, fit scaler on full validation/development set."
        ),

    "hyperparameter_tuning":
        False,

    "external_ml_dependency":
        False,
}

THRESHOLD_PROTOCOL = {
    "development_only":
        True,

    "oof_grouping_unit":
        "track_key",

    "oof_scheme":
        "exact deterministic 5-fold track-disjoint partition locked by R6B2-V3",

    "selection_metric":
        "balanced_accuracy",

    "primary_rule":
        (
            "Pool all out-of-fold validation probabilities within each task and choose "
            "the threshold that maximizes balanced accuracy."
        ),

    "candidate_thresholds":
        (
            "sorted unique OOF probabilities plus 0.5 and endpoints 0/1; "
            "selection uses development OOF predictions only"
        ),

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


def verify_b2v3():
    for p in [
        R6B2V3_MANIFEST,
        SELECTED_SCHEME,
        FOLD_DETAILS,
        VALIDATION,
        EVALUATION,
    ]:
        require(p)

    manifest = json.loads(
        R6B2V3_MANIFEST.read_text(encoding="utf-8")
    )

    if (
        manifest.get("status")
        !=
        "R6B2V3_TRACK_DISJOINT_GROUPED_CV_FEASIBILITY_PROVEN_WITHOUT_SKLEARN_READY_FOR_PROTOCOL_REVISION"
    ):
        raise RuntimeError(
            f"Unexpected R6B2-V3 status: {manifest.get('status')}"
        )

    return manifest


def parse_bool_series(s, name):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        if vals.dropna().isin([0, 1]).all():
            return vals.astype(int)

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

    if out.isna().any():
        bad = s[out.isna()].astype(str).unique().tolist()
        raise RuntimeError(
            f"{name}: unparseable binary values {bad[:20]}"
        )

    return out.astype(int)


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
            f"{name} contains unexpected tasks: {bad_tasks}"
        )

    y = parse_bool_series(
        df[LABEL],
        f"{name}.{LABEL}",
    )

    if y.nunique() != 2:
        raise RuntimeError(
            f"{name}: target does not contain both classes overall."
        )


def source_overlap():
    val = pd.read_csv(VALIDATION)
    eva = pd.read_csv(EVALUATION)

    val_sources = set(
        val["source_group"].astype(str).unique()
    )

    eva_sources = set(
        eva["source_group"].astype(str).unique()
    )

    overlap = sorted(
        val_sources & eva_sources
    )

    return {
        "validation_source_count": len(val_sources),
        "evaluation_source_count": len(eva_sources),
        "overlap_count": len(overlap),
        "overlap_sources": overlap,
    }


def verify_selected_scheme():
    selected = pd.read_csv(SELECTED_SCHEME)
    folds = pd.read_csv(FOLD_DETAILS)

    if len(selected) != 2:
        raise RuntimeError(
            f"Expected two selected scheme rows, found {len(selected)}"
        )

    expected_tasks = {
        "2s_to_2s",
        "2s_to_5s",
    }

    if set(selected["task"].astype(str)) != expected_tasks:
        raise RuntimeError(
            "Selected scheme tasks mismatch."
        )

    if not (selected["selected_n_splits"] == 5).all():
        raise RuntimeError(
            "R6B3 requires exact 5-fold scheme for both tasks."
        )

    locked_rows = []

    for task in sorted(expected_tasks):
        sub = folds[
            (folds["task"].astype(str) == task)
            &
            (folds["n_splits"] == 5)
        ].copy()

        if len(sub) != 5:
            raise RuntimeError(
                f"{task}: expected 5 fold-detail rows, found {len(sub)}"
            )

        if not sub["train_has_both_classes"].all():
            raise RuntimeError(
                f"{task}: at least one training fold lacks both classes."
            )

        if not (sub["track_overlap_count"] == 0).all():
            raise RuntimeError(
                f"{task}: track overlap detected."
            )

        for _, row in sub.sort_values("fold").iterrows():
            locked_rows.append(
                {
                    "task": task,
                    "fold": int(row["fold"]),
                    "val_track_keys_json": row["val_track_keys_json"],
                    "train_rows": int(row["train_rows"]),
                    "val_rows": int(row["val_rows"]),
                    "train_positives": int(row["train_positives"]),
                    "train_negatives": int(row["train_negatives"]),
                    "val_positives": int(row["val_positives"]),
                    "val_negatives": int(row["val_negatives"]),
                    "track_overlap_count": int(row["track_overlap_count"]),
                    "train_has_both_classes": bool(row["train_has_both_classes"]),
                }
            )

    return selected, pd.DataFrame(locked_rows)


def leakage_lock_table(validation):
    rows = []

    for col in validation.columns:
        if col == LABEL:
            role = "target_label"
            allowed = False
        elif col in GROUND_TRUTH_EXCLUSIONS:
            role = "ground_truth_dependent_excluded"
            allowed = False
        elif col in IDENTITY_EXCLUSIONS:
            role = "identity_or_geometry_excluded"
            allowed = False
        elif col in RAW_FEATURES:
            role = "locked_safe_raw_feature"
            allowed = True
        else:
            role = "unused"
            allowed = False

        rows.append(
            {
                "column": col,
                "role": role,
                "locked_as_feature": allowed,
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR6-B3 — REVISED DIRECT-CROSSING CLASSIFIER PROTOCOL LOCK"
    )

    print(
        "NO classifier training. NO threshold selection. "
        "Source-level CV was proven infeasible, so this protocol locks the exact "
        "5-fold track-disjoint development partition established by R6B2-V3.\n"
    )

    verify_b2v3()

    validation = pd.read_csv(VALIDATION)
    evaluation = pd.read_csv(EVALUATION)

    validate_schema(validation, "validation")
    validate_schema(evaluation, "evaluation")

    overlap = source_overlap()

    if overlap["overlap_count"] != 0:
        raise RuntimeError(
            f"Validation/evaluation source overlap detected: {overlap['overlap_sources']}"
        )

    selected, locked_folds = verify_selected_scheme()

    leakage = leakage_lock_table(
        validation
    )

    fold_path = OUT / "R6B3_LOCKED_5FOLD_TRACK_PARTITIONS.csv"
    leakage_path = OUT / "R6B3_FEATURE_LEAKAGE_LOCK.csv"
    protocol_path = OUT / "R6B3_LOCKED_DIRECT_CLASSIFIER_PROTOCOL.json"

    locked_folds.to_csv(
        fold_path,
        index=False,
    )

    leakage.to_csv(
        leakage_path,
        index=False,
    )

    protocol = {
        "label":
            LABEL,

        "raw_features":
            RAW_FEATURES,

        "derived_features_in_order":
            DERIVED_FEATURES,

        "task_horizon_seconds":
            TASK_HORIZON_SEC,

        "feature_transformations": {
            "distance_to_boundary":
                "numeric raw value",

            "predicted_crossing_bool":
                "binary parse of predicted_crossing",

            "predicted_tte_norm":
                (
                    "predicted_tte_sec / task horizon; clip to [0,1.25]; "
                    "missing -> 1.25"
                ),

            "predicted_tte_available":
                "1 if predicted_tte_sec is finite, else 0",
        },

        "ground_truth_features_forbidden":
            GROUND_TRUTH_EXCLUSIONS,

        "identity_or_geometry_features_forbidden":
            IDENTITY_EXCLUSIONS,

        "model_protocol":
            MODEL_PROTOCOL,

        "threshold_protocol":
            THRESHOLD_PROTOCOL,

        "locked_cv_partition_csv":
            str(fold_path),

        "validation_source":
            str(VALIDATION),

        "validation_source_sha256":
            sha256_file(VALIDATION),

        "evaluation_source":
            str(EVALUATION),

        "evaluation_source_sha256":
            sha256_file(EVALUATION),

        "validation_evaluation_source_overlap":
            overlap,
    }

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    status = (
        "R6B3_REVISED_TRACK_DISJOINT_FEATURE_MODEL_THRESHOLD_PROTOCOL_LOCKED_READY_FOR_R6C_OOF_FIT"
    )

    manifest = {
        "step":
            "R6B3_REVISED_DIRECT_CROSSING_CLASSIFIER_PROTOCOL_LOCK",

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

        "external_sklearn_dependency":
            False,

        "cv_grouping_unit":
            "track_key",

        "cv_folds_per_task":
            5,

        "locked_protocol":
            str(protocol_path),

        "locked_protocol_sha256":
            sha256_file(protocol_path),

        "locked_fold_csv":
            str(fold_path),

        "locked_fold_csv_sha256":
            sha256_file(fold_path),

        "validation_source_sha256":
            sha256_file(VALIDATION),

        "evaluation_source_sha256":
            sha256_file(EVALUATION),

        "source_overlap_count":
            overlap["overlap_count"],
    }

    manifest_path = OUT / "R6B3_MANIFEST.json"

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
                "label": LABEL,
                "raw_features": RAW_FEATURES,
                "derived_features": DERIVED_FEATURES,
            },
            indent=2,
        )
    )

    print("\nLEAKAGE EXCLUSIONS:")
    print(
        leakage[
            leakage["role"]
            .str.contains(
                "excluded|target",
                regex=True,
            )
        ].to_string(
            index=False
        )
    )

    print("\nLOCKED 5-FOLD TRACK-DISJOINT PARTITIONS:")
    print(
        locked_folds.to_string(
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

    print("\nR6-B3 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
