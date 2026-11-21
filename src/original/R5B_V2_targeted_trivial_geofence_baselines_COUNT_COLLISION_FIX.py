from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

STEP7 = MAIN_RESULT_2 / "step7_response_aware_warning"

INSTANCES = STEP7 / "step7h_final_response_aware_instances.csv"
FINAL_METRICS = STEP7 / "TABLE_step7h_final_response_aware_metrics.csv"
BOUNDARY_METRICS = STEP7 / "TABLE_step7h_boundary_macro_metrics.csv"

R5A_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R5_geofence_baselines"
    / "R5A_existing_geofence_baseline_audit"
    / "R5A_MANIFEST.json"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R5_geofence_baselines"
    / "R5B_trivial_always_red_always_green"
)

OUT.mkdir(parents=True, exist_ok=True)


PRIMARY_OUTPUT_COLUMNS = {
    "actual_crossings",
    "urgent_actual_crossings",
    "green_rate",
    "yellow_rate",
    "red_rate",
    "alert_sensitivity_any_crossing",
    "red_sensitivity_urgent_crossing",
    "false_alert_rate_non_crossing",
    "false_red_rate_non_urgent",
    "alert_precision_any_crossing",
    "red_precision_urgent_crossing",
}

LABEL_OR_PREDICTION_COLUMNS = {
    "true_crossing",
    "predicted_crossing",
    "true_tte_sec",
    "predicted_tte_sec",
    "tte_error_sec",
    "tte_abs_error_sec",
    "true_crossing_bool",
    "predicted_crossing_bool",
    "alarm_state",
    "alarm_state_final",
}

BASELINES = [
    "ALWAYS_RED",
    "ALWAYS_GREEN",
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


def safe_rate(num, den):
    if den <= 0:
        return np.nan
    return float(num) / float(den)


def verify_r5a():
    require(R5A_MANIFEST)

    obj = json.loads(
        R5A_MANIFEST.read_text(encoding="utf-8")
    )

    if (
        obj.get("status")
        !=
        "R5A_AUDIT_COMPLETE_MISSING_TRIVIAL_BASELINES_REQUIRE_TARGETED_ADDITION"
    ):
        raise RuntimeError(
            f"Unexpected R5A status: {obj.get('status')}"
        )

    missing = set(
        obj.get("missing_trivial_baselines", [])
    )

    expected = {
        "always_alert_or_always_red",
        "never_alert_or_always_green",
    }

    if not expected.issubset(missing):
        raise RuntimeError(
            f"R5A does not require both targeted trivial baselines. "
            f"Missing list={sorted(missing)}"
        )

    return obj


def infer_group_keys(instances, metrics):
    """
    Recover the exact Step7H primary grouping by using columns shared by
    instances and final metrics, excluding labels/predictions and metric outputs.

    We accept a candidate key set only if:
      - grouping the instance table produces exactly the same number of groups
        as FINAL_METRICS rows;
      - every final-metric group matches exactly one instance group.
    """
    shared = [
        c
        for c in metrics.columns
        if c in instances.columns
        and c not in PRIMARY_OUTPUT_COLUMNS
        and c not in LABEL_OR_PREDICTION_COLUMNS
    ]

    # Preserve likely protocol/group columns first.
    preferred_tokens = [
        "dataset",
        "task",
        "response",
        "threshold",
        "coverage",
        "persistence",
        "horizon",
        "split",
        "role",
    ]

    likely = [
        c
        for c in shared
        if any(tok in c.lower() for tok in preferred_tokens)
    ]

    # Try likely keys first. Then progressively add shared keys if necessary.
    candidate_sets = []

    if likely:
        candidate_sets.append(likely)

    candidate_sets.append(shared)

    # Also try reduced variants dropping obvious sample/entity identifiers.
    reduced = [
        c
        for c in shared
        if not any(
            tok in c.lower()
            for tok in [
                "sample",
                "track",
                "source",
                "boundary",
                "geofence",
                "frame",
                "window_id",
            ]
        )
    ]

    if reduced:
        candidate_sets.append(reduced)

    seen = set()

    for keys in candidate_sets:
        keys = tuple(dict.fromkeys(keys))

        if not keys or keys in seen:
            continue

        seen.add(keys)

        grouped = (
            instances.groupby(
                list(keys),
                dropna=False,
            )
            .size()
            .reset_index(name="_group_instance_count")
        )

        metric_groups = metrics[
            list(keys)
        ].drop_duplicates()

        if len(grouped) != len(metrics):
            continue

        if len(metric_groups) != len(metrics):
            continue

        merged = metrics.merge(
            grouped,
            on=list(keys),
            how="left",
            validate="one_to_one",
        )

        if merged["_group_instance_count"].notna().all():
            return list(keys), grouped

    raise RuntimeError(
        "Could not prove exact Step7H primary grouping from shared instance/metric columns. "
        f"Shared columns={shared}"
    )


def build_trivial_metrics(metrics_with_n, group_keys):
    rows = []

    for _, row in metrics_with_n.iterrows():
        n = int(row["instances"])
        actual = int(row["actual_crossings"])
        urgent = int(row["urgent_actual_crossings"])

        noncrossing = n - actual
        nonurgent = n - urgent

        base_common = {
            key: row[key]
            for key in group_keys
        }

        base_common.update(
            {
                "instances": n,
                "actual_crossings": actual,
                "urgent_actual_crossings": urgent,
                "actual_crossing_prevalence":
                    safe_rate(actual, n),
                "urgent_crossing_prevalence":
                    safe_rate(urgent, n),
            }
        )

        # ----------------------------------------------------------
        # ALWAYS RED = always positive for alert and urgent RED.
        # ----------------------------------------------------------
        always_red = dict(base_common)

        always_red.update(
            {
                "baseline": "ALWAYS_RED",
                "green_rate": 0.0,
                "yellow_rate": 0.0,
                "red_rate": 1.0,
                "alert_sensitivity_any_crossing":
                    1.0 if actual > 0 else np.nan,
                "red_sensitivity_urgent_crossing":
                    1.0 if urgent > 0 else np.nan,
                "false_alert_rate_non_crossing":
                    1.0 if noncrossing > 0 else np.nan,
                "false_red_rate_non_urgent":
                    1.0 if nonurgent > 0 else np.nan,
                "alert_precision_any_crossing":
                    safe_rate(actual, n),
                "red_precision_urgent_crossing":
                    safe_rate(urgent, n),
                "alert_precision_defined": True,
                "red_precision_defined": True,
            }
        )

        rows.append(always_red)

        # ----------------------------------------------------------
        # ALWAYS GREEN = always negative / never alert.
        # Precision is mathematically undefined because there are
        # zero positive predictions; keep NaN rather than invent 0.
        # ----------------------------------------------------------
        always_green = dict(base_common)

        always_green.update(
            {
                "baseline": "ALWAYS_GREEN",
                "green_rate": 1.0,
                "yellow_rate": 0.0,
                "red_rate": 0.0,
                "alert_sensitivity_any_crossing":
                    0.0 if actual > 0 else np.nan,
                "red_sensitivity_urgent_crossing":
                    0.0 if urgent > 0 else np.nan,
                "false_alert_rate_non_crossing":
                    0.0 if noncrossing > 0 else np.nan,
                "false_red_rate_non_urgent":
                    0.0 if nonurgent > 0 else np.nan,
                "alert_precision_any_crossing":
                    np.nan,
                "red_precision_urgent_crossing":
                    np.nan,
                "alert_precision_defined": False,
                "red_precision_defined": False,
            }
        )

        rows.append(always_green)

    return pd.DataFrame(rows)


def build_current_comparison(metrics, group_keys):
    out = metrics.copy()
    out["baseline"] = "STEP7H_PRIMARY"

    # Keep the same core columns as trivial baselines.
    keep = (
        group_keys
        + [
            "baseline",
            "actual_crossings",
            "urgent_actual_crossings",
            "green_rate",
            "yellow_rate",
            "red_rate",
            "alert_sensitivity_any_crossing",
            "red_sensitivity_urgent_crossing",
            "false_alert_rate_non_crossing",
            "false_red_rate_non_urgent",
            "alert_precision_any_crossing",
            "red_precision_urgent_crossing",
        ]
    )

    return out[
        [
            c
            for c in keep
            if c in out.columns
        ]
    ].copy()


def derive_review_table(comparison, group_keys):
    """
    Reviewer-facing compact table: preserve every frozen operating point,
    but add simple dominance indicators against trivial baselines.
    """
    rows = []

    for key_values, group in comparison.groupby(
        group_keys,
        dropna=False,
        sort=True,
    ):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)

        key_dict = dict(
            zip(group_keys, key_values)
        )

        current = group[
            group["baseline"] == "STEP7H_PRIMARY"
        ]

        red = group[
            group["baseline"] == "ALWAYS_RED"
        ]

        green = group[
            group["baseline"] == "ALWAYS_GREEN"
        ]

        if len(current) != 1 or len(red) != 1 or len(green) != 1:
            raise RuntimeError(
                f"Expected one row per method for group {key_dict}"
            )

        c = current.iloc[0]
        r = red.iloc[0]
        g = green.iloc[0]

        rows.append(
            {
                **key_dict,
                "step7h_alert_sensitivity":
                    float(c["alert_sensitivity_any_crossing"]),
                "step7h_false_alert_rate":
                    float(c["false_alert_rate_non_crossing"]),
                "step7h_alert_precision":
                    float(c["alert_precision_any_crossing"]),
                "step7h_red_sensitivity_urgent":
                    float(c["red_sensitivity_urgent_crossing"]),
                "step7h_false_red_rate_non_urgent":
                    float(c["false_red_rate_non_urgent"]),
                "step7h_red_precision_urgent":
                    float(c["red_precision_urgent_crossing"]),
                "always_red_alert_sensitivity":
                    float(r["alert_sensitivity_any_crossing"])
                    if pd.notna(r["alert_sensitivity_any_crossing"])
                    else np.nan,
                "always_red_false_alert_rate":
                    float(r["false_alert_rate_non_crossing"])
                    if pd.notna(r["false_alert_rate_non_crossing"])
                    else np.nan,
                "always_red_alert_precision":
                    float(r["alert_precision_any_crossing"])
                    if pd.notna(r["alert_precision_any_crossing"])
                    else np.nan,
                "always_green_alert_sensitivity":
                    float(g["alert_sensitivity_any_crossing"])
                    if pd.notna(g["alert_sensitivity_any_crossing"])
                    else np.nan,
                "always_green_false_alert_rate":
                    float(g["false_alert_rate_non_crossing"])
                    if pd.notna(g["false_alert_rate_non_crossing"])
                    else np.nan,
                "step7h_nontrivial_tradeoff":
                    bool(
                        pd.notna(c["alert_sensitivity_any_crossing"])
                        and
                        pd.notna(c["false_alert_rate_non_crossing"])
                        and
                        float(c["alert_sensitivity_any_crossing"]) > 0.0
                        and
                        float(c["false_alert_rate_non_crossing"]) < 1.0
                    ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR5-B-V2 — TARGETED TRIVIAL GEOFENCE BASELINES: ALWAYS RED + ALWAYS GREEN"
    )

    print(
        "NO existing Step7H result is modified. "
        "The two missing reviewer-facing trivial baselines are evaluated "
        "on the exact frozen Step7H operating-point groups.\n"
    )

    verify_r5a()

    for p in [
        INSTANCES,
        FINAL_METRICS,
    ]:
        require(p)

    instances = pd.read_csv(INSTANCES)
    metrics = pd.read_csv(FINAL_METRICS)

    required_metrics = PRIMARY_OUTPUT_COLUMNS

    missing = required_metrics - set(metrics.columns)

    if missing:
        raise RuntimeError(
            f"Step7H final metrics missing required columns: {sorted(missing)}"
        )

    group_keys, grouped_counts = infer_group_keys(
        instances,
        metrics,
    )

    metrics_with_n = metrics.merge(
        grouped_counts,
        on=group_keys,
        how="left",
        validate="one_to_one",
    )

    if "_group_instance_count" not in metrics_with_n.columns:
        raise RuntimeError(
            "Exact Step7H group instance count column was not attached."
        )

    if metrics_with_n["_group_instance_count"].isna().any():
        raise RuntimeError(
            "Failed to attach exact Step7H instance counts to final metric rows."
        )

    # Preserve any pre-existing Step7H 'instances' column for audit,
    # but use the count proven directly from the frozen instance table.
    if "instances" in metrics_with_n.columns:
        metrics_with_n["instances_reported_step7h"] = metrics_with_n["instances"]

    metrics_with_n["instances"] = (
        metrics_with_n["_group_instance_count"]
        .astype(int)
    )

    trivial = build_trivial_metrics(
        metrics_with_n,
        group_keys,
    )

    current = build_current_comparison(
        metrics,
        group_keys,
    )

    # Add instance/prevalence fields to current rows for consistent comparison.
    current = current.merge(
        metrics_with_n[
            group_keys
            + [
                "instances",
                "actual_crossings",
                "urgent_actual_crossings",
            ]
        ],
        on=group_keys
        + [
            "actual_crossings",
            "urgent_actual_crossings",
        ],
        how="left",
        validate="one_to_one",
        suffixes=("", "_countproof"),
    )

    # If current already carried an 'instances' field, prefer the direct
    # count proven from frozen Step7H instances.
    if "instances_countproof" in current.columns:
        current["instances"] = current["instances_countproof"]
        current = current.drop(columns=["instances_countproof"])

    current["actual_crossing_prevalence"] = (
        current["actual_crossings"]
        /
        current["instances"]
    )

    current["urgent_crossing_prevalence"] = (
        current["urgent_actual_crossings"]
        /
        current["instances"]
    )

    current["alert_precision_defined"] = (
        current["alert_precision_any_crossing"]
        .notna()
    )

    current["red_precision_defined"] = (
        current["red_precision_urgent_crossing"]
        .notna()
    )

    comparison = pd.concat(
        [
            current,
            trivial,
        ],
        ignore_index=True,
        sort=False,
    )

    review = derive_review_table(
        comparison,
        group_keys,
    )

    trivial_path = OUT / "R5B_trivial_baseline_metrics.csv"
    comparison_path = OUT / "R5B_step7h_vs_trivial_baselines.csv"
    review_path = OUT / "R5B_reviewer_facing_trivial_baseline_comparison.csv"
    group_path = OUT / "R5B_grouping_audit.csv"

    trivial.to_csv(
        trivial_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    review.to_csv(
        review_path,
        index=False,
    )

    pd.DataFrame(
        {
            "group_key": group_keys,
        }
    ).to_csv(
        group_path,
        index=False,
    )

    status = (
        "R5B_TRIVIAL_ALWAYS_RED_ALWAYS_GREEN_BASELINES_COMPLETE_READY_FOR_R5C_INTERPRETATION"
    )

    manifest = {
        "step":
            "R5B_TARGETED_TRIVIAL_GEOFENCE_BASELINES",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "status":
            status,

        "existing_step7h_modified":
            False,

        "new_prediction_model":
            False,

        "new_threshold_selection":
            False,

        "baselines_added":
            BASELINES,

        "source_instances":
            str(INSTANCES),

        "source_instances_sha256":
            sha256_file(INSTANCES),

        "source_final_metrics":
            str(FINAL_METRICS),

        "source_final_metrics_sha256":
            sha256_file(FINAL_METRICS),

        "group_keys":
            group_keys,

        "trivial_metrics_csv":
            str(trivial_path),

        "comparison_csv":
            str(comparison_path),

        "review_table_csv":
            str(review_path),
    }

    manifest_path = OUT / "R5B_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("PROVEN STEP7H GROUP KEYS:")
    print(group_keys)

    print("\nTRIVIAL BASELINE METRICS:")
    print(
        trivial.to_string(
            index=False
        )
    )

    print("\nSTEP7H vs TRIVIAL BASELINES — COMPACT REVIEW:")
    print(
        review.to_string(
            index=False
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR5-B COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
