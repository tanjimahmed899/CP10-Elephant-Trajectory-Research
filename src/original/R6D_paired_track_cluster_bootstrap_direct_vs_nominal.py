from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

R6C_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6C_oof_threshold_lock_and_frozen_evaluation"
)

R6C_MANIFEST = R6C_DIR / "R6C_MANIFEST.json"
PREDICTIONS = R6C_DIR / "R6C_FROZEN_EVALUATION_PREDICTIONS.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6D_paired_track_cluster_bootstrap"
)

OUT.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20261029


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


def verify_r6c():
    for p in [R6C_MANIFEST, PREDICTIONS]:
        require(p)

    manifest = json.loads(
        R6C_MANIFEST.read_text(encoding="utf-8")
    )

    if (
        manifest.get("status")
        !=
        "R6C_VALIDATION_ONLY_THRESHOLD_LOCK_AND_FROZEN_DIRECT_CLASSIFIER_EVALUATION_COMPLETE_READY_FOR_R6D_INTERPRETATION"
    ):
        raise RuntimeError(
            f"Unexpected R6C status: {manifest.get('status')}"
        )

    expected = manifest.get("evaluation_metrics_sha256")

    # Predictions hash was not separately stored in R6C manifest, so verify
    # only existence here and preserve its hash in the R6D manifest.
    return manifest


def confusion_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (
            pd.notna(precision)
            and pd.notna(sensitivity)
            and (precision + sensitivity) > 0
        )
        else np.nan
    )

    balanced_accuracy = (
        0.5 * (sensitivity + specificity)
        if pd.notna(sensitivity) and pd.notna(specificity)
        else np.nan
    )

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true) > 0
        else np.nan
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "precision": precision,
        "F1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def metric_deltas(group):
    y = group["true_crossing"].to_numpy(dtype=int)

    direct = group[
        "direct_classifier_prediction"
    ].to_numpy(dtype=int)

    baseline = group[
        "nominal_predicted_crossing_baseline"
    ].to_numpy(dtype=int)

    d = confusion_metrics(y, direct)
    b = confusion_metrics(y, baseline)

    return {
        "balanced_accuracy_delta":
            d["balanced_accuracy"] - b["balanced_accuracy"],

        "sensitivity_delta":
            d["sensitivity"] - b["sensitivity"],

        "false_positive_rate_delta":
            d["false_positive_rate"] - b["false_positive_rate"],

        "precision_delta":
            d["precision"] - b["precision"]
            if pd.notna(d["precision"]) and pd.notna(b["precision"])
            else np.nan,

        "F1_delta":
            d["F1"] - b["F1"]
            if pd.notna(d["F1"]) and pd.notna(b["F1"])
            else np.nan,

        "accuracy_delta":
            d["accuracy"] - b["accuracy"],
    }


def bootstrap_one_group(group, rng):
    tracks = sorted(
        group["track_key"].astype(str).unique().tolist()
    )

    n_tracks = len(tracks)

    if n_tracks < 2:
        raise RuntimeError(
            f"Need at least 2 tracks, got {n_tracks}"
        )

    by_track = {
        track: group[
            group["track_key"].astype(str) == track
        ].copy()
        for track in tracks
    }

    point = metric_deltas(group)

    metrics = list(point.keys())

    boots = {
        metric: np.empty(
            BOOTSTRAP_REPS,
            dtype=float,
        )
        for metric in metrics
    }

    for rep in range(BOOTSTRAP_REPS):
        sampled = rng.choice(
            tracks,
            size=n_tracks,
            replace=True,
        )

        pieces = [
            by_track[t]
            for t in sampled
        ]

        boot_df = pd.concat(
            pieces,
            ignore_index=True,
        )

        vals = metric_deltas(
            boot_df
        )

        for metric in metrics:
            boots[metric][rep] = vals[metric]

    rows = []

    for metric in metrics:
        arr = boots[metric]

        finite = arr[
            np.isfinite(arr)
        ]

        if len(finite) == 0:
            ci_low = np.nan
            ci_high = np.nan
            prob_positive = np.nan
            prob_negative = np.nan
        else:
            ci_low, ci_high = np.quantile(
                finite,
                [0.025, 0.975],
            )

            prob_positive = float(
                np.mean(finite > 0)
            )

            prob_negative = float(
                np.mean(finite < 0)
            )

        if metric in {
            "balanced_accuracy_delta",
            "sensitivity_delta",
            "precision_delta",
            "F1_delta",
            "accuracy_delta",
        }:
            positive_direction_favors_direct = True
        elif metric == "false_positive_rate_delta":
            positive_direction_favors_direct = False
        else:
            positive_direction_favors_direct = None

        if pd.notna(ci_low) and pd.notna(ci_high):
            excludes_zero = bool(
                ci_low > 0
                or ci_high < 0
            )
        else:
            excludes_zero = False

        rows.append(
            {
                "metric": metric,
                "point_delta_direct_minus_baseline":
                    float(point[metric]),

                "ci95_low":
                    float(ci_low)
                    if pd.notna(ci_low)
                    else np.nan,

                "ci95_high":
                    float(ci_high)
                    if pd.notna(ci_high)
                    else np.nan,

                "ci95_excludes_zero":
                    excludes_zero,

                "bootstrap_prob_delta_positive":
                    prob_positive,

                "bootstrap_prob_delta_negative":
                    prob_negative,

                "positive_direction_favors_direct":
                    positive_direction_favors_direct,

                "tracks":
                    n_tracks,

                "bootstrap_reps":
                    BOOTSTRAP_REPS,
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR6-D — PAIRED TRACK-CLUSTER BOOTSTRAP: DIRECT CLASSIFIER vs NOMINAL CROSSING"
    )

    print(
        "Thresholds remain frozen from validation OOF. "
        "No retraining and no threshold reselection. "
        "This step quantifies uncertainty in the paired evaluation differences "
        "by resampling whole tracks with replacement.\n"
    )

    verify_r6c()

    df = pd.read_csv(
        PREDICTIONS
    )

    required = {
        "dataset",
        "task",
        "track_key",
        "true_crossing",
        "direct_classifier_prediction",
        "nominal_predicted_crossing_baseline",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"R6C prediction table missing columns: {sorted(missing)}"
        )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    all_rows = []

    for (dataset, task), group in df.groupby(
        ["dataset", "task"],
        sort=True,
    ):
        result = bootstrap_one_group(
            group,
            rng,
        )

        result.insert(
            0,
            "task",
            task,
        )

        result.insert(
            0,
            "dataset",
            dataset,
        )

        all_rows.append(
            result
        )

    results = pd.concat(
        all_rows,
        ignore_index=True,
    )

    # Compact reviewer-facing subset.
    primary_metrics = [
        "balanced_accuracy_delta",
        "sensitivity_delta",
        "false_positive_rate_delta",
        "precision_delta",
        "F1_delta",
    ]

    primary = results[
        results["metric"].isin(
            primary_metrics
        )
    ].copy()

    results_path = OUT / "R6D_TRACK_CLUSTER_BOOTSTRAP_RESULTS.csv"
    primary_path = OUT / "R6D_REVIEWER_FACING_PRIMARY_COMPARISON.csv"

    results.to_csv(
        results_path,
        index=False,
    )

    primary.to_csv(
        primary_path,
        index=False,
    )

    status = (
        "R6D_PAIRED_TRACK_CLUSTER_BOOTSTRAP_COMPLETE_READY_FOR_R6E_CLAIM_LOCK"
    )

    manifest = {
        "step":
            "R6D_PAIRED_TRACK_CLUSTER_BOOTSTRAP_DIRECT_VS_NOMINAL",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "status":
            status,

        "retraining":
            False,

        "threshold_reselection":
            False,

        "bootstrap_unit":
            "track_key",

        "bootstrap_reps":
            BOOTSTRAP_REPS,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "source_predictions":
            str(PREDICTIONS),

        "source_predictions_sha256":
            sha256_file(PREDICTIONS),

        "results_csv":
            str(results_path),

        "results_sha256":
            sha256_file(results_path),

        "primary_csv":
            str(primary_path),

        "primary_sha256":
            sha256_file(primary_path),
    }

    manifest_path = OUT / "R6D_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("PAIRED TRACK-CLUSTER BOOTSTRAP — PRIMARY METRICS:")
    print(
        primary[
            [
                "dataset",
                "task",
                "metric",
                "tracks",
                "point_delta_direct_minus_baseline",
                "ci95_low",
                "ci95_high",
                "ci95_excludes_zero",
                "bootstrap_prob_delta_positive",
                "bootstrap_prob_delta_negative",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR6-D COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
