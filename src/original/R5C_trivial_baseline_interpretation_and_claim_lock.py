from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

R5B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R5_geofence_baselines"
    / "R5B_trivial_always_red_always_green"
)

R5B_MANIFEST = R5B_DIR / "R5B_MANIFEST.json"
COMPARISON_CSV = R5B_DIR / "R5B_step7h_vs_trivial_baselines.csv"
REVIEW_CSV = R5B_DIR / "R5B_reviewer_facing_trivial_baseline_comparison.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R5_geofence_baselines"
    / "R5C_interpretation_claim_lock"
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
        R5B_MANIFEST,
        COMPARISON_CSV,
        REVIEW_CSV,
    ]:
        require(p)

    manifest = json.loads(
        R5B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("status")
        !=
        "R5B_TRIVIAL_ALWAYS_RED_ALWAYS_GREEN_BASELINES_COMPLETE_READY_FOR_R5C_INTERPRETATION"
    ):
        raise RuntimeError(
            f"Unexpected R5B status: {manifest.get('status')}"
        )

    return manifest


def summarize(review):
    out = review.copy()

    out["alert_sensitivity_retained_vs_always_red_pct"] = (
        100.0
        *
        out["step7h_alert_sensitivity"]
    )

    out["false_alert_reduction_vs_always_red_pct"] = (
        100.0
        *
        (
            1.0
            -
            out["step7h_false_alert_rate"]
        )
    )

    out["alert_precision_lift_vs_always_red_x"] = (
        out["step7h_alert_precision"]
        /
        out["always_red_alert_precision"]
    )

    # ALWAYS-RED urgent precision equals urgent event prevalence.
    # Recover that prevalence from the full comparison table later.
    return out


def add_urgent_prevalence(summary, comparison):
    red = comparison[
        comparison["baseline"] == "ALWAYS_RED"
    ].copy()

    keys = [
        "dataset",
        "task",
        "response_time_sec",
    ]

    needed = red[
        keys
        + [
            "urgent_crossing_prevalence",
            "red_precision_urgent_crossing",
        ]
    ].copy()

    needed = needed.rename(
        columns={
            "urgent_crossing_prevalence":
                "always_red_urgent_prevalence",
            "red_precision_urgent_crossing":
                "always_red_red_precision_urgent",
        }
    )

    out = summary.merge(
        needed,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    out["red_precision_lift_vs_always_red_x"] = (
        out["step7h_red_precision_urgent"]
        /
        out["always_red_red_precision_urgent"]
    )

    return out


def aggregate_by_dataset_task(summary):
    grouped = (
        summary.groupby(
            ["dataset", "task"],
            as_index=False,
        )
        .agg(
            response_points=("response_time_sec", "nunique"),
            alert_sensitivity_min=("step7h_alert_sensitivity", "min"),
            alert_sensitivity_max=("step7h_alert_sensitivity", "max"),
            false_alert_rate_min=("step7h_false_alert_rate", "min"),
            false_alert_rate_max=("step7h_false_alert_rate", "max"),
            alert_precision_min=("step7h_alert_precision", "min"),
            alert_precision_max=("step7h_alert_precision", "max"),
            red_sensitivity_urgent_min=("step7h_red_sensitivity_urgent", "min"),
            red_sensitivity_urgent_max=("step7h_red_sensitivity_urgent", "max"),
            false_red_rate_non_urgent_min=("step7h_false_red_rate_non_urgent", "min"),
            false_red_rate_non_urgent_max=("step7h_false_red_rate_non_urgent", "max"),
            red_precision_urgent_min=("step7h_red_precision_urgent", "min"),
            red_precision_urgent_max=("step7h_red_precision_urgent", "max"),
            false_alert_reduction_vs_always_red_min_pct=(
                "false_alert_reduction_vs_always_red_pct",
                "min",
            ),
            false_alert_reduction_vs_always_red_max_pct=(
                "false_alert_reduction_vs_always_red_pct",
                "max",
            ),
            alert_precision_lift_vs_always_red_min_x=(
                "alert_precision_lift_vs_always_red_x",
                "min",
            ),
            alert_precision_lift_vs_always_red_max_x=(
                "alert_precision_lift_vs_always_red_x",
                "max",
            ),
            red_precision_lift_vs_always_red_min_x=(
                "red_precision_lift_vs_always_red_x",
                "min",
            ),
            red_precision_lift_vs_always_red_max_x=(
                "red_precision_lift_vs_always_red_x",
                "max",
            ),
            all_nontrivial_tradeoff=("step7h_nontrivial_tradeoff", "all"),
        )
    )

    return grouped


def claim_lock():
    rows = [
        {
            "claim_id": "R5C_CLAIM_01",
            "status": "SUPPORTED",
            "claim": (
                "The frozen Step7H warning rule is non-trivial: it does not reduce "
                "to always alerting or never alerting."
            ),
            "basis": (
                "Across all frozen operating points, Step7H sensitivity is >0 and "
                "false-alert rate is <1, while ALWAYS-RED has sensitivity=1/FAR=1 "
                "and ALWAYS-GREEN has sensitivity=0/FAR=0."
            ),
        },
        {
            "claim_id": "R5C_CLAIM_02",
            "status": "SUPPORTED",
            "claim": (
                "Relative to ALWAYS-RED, Step7H sacrifices some crossing sensitivity "
                "but removes a large fraction of false alerts."
            ),
            "basis": (
                "Step7H alert sensitivity is roughly 0.77–0.87 while false-alert "
                "rates are roughly 0.31–0.37 instead of 1.0."
            ),
        },
        {
            "claim_id": "R5C_CLAIM_03",
            "status": "SUPPORTED",
            "claim": (
                "Step7H alert precision is materially higher than the prevalence-level "
                "precision of an ALWAYS-RED rule."
            ),
            "basis": (
                "ALWAYS-RED precision equals crossing prevalence, whereas Step7H "
                "selectively alerts and achieves higher precision at every frozen group."
            ),
        },
        {
            "claim_id": "R5C_CLAIM_04",
            "status": "SUPPORTED",
            "claim": (
                "The RED channel is selective for urgent crossings: false-RED rates "
                "among non-urgent instances remain very low while urgent RED precision "
                "is substantially above the ALWAYS-RED urgent-prevalence baseline."
            ),
            "basis": (
                "Step7H false-RED rates are below 1% across the reported groups, while "
                "urgent RED precision is roughly 0.44–0.89."
            ),
        },
        {
            "claim_id": "R5C_CLAIM_05",
            "status": "SUPPORTED_WITH_LIMITATION",
            "claim": (
                "Step7H provides a useful sensitivity–false-alert trade-off in the "
                "virtual-geofence simulation."
            ),
            "basis": (
                "It lies between ALWAYS-RED and ALWAYS-GREEN on the expected trade-off "
                "curve, but this remains a simulation/decision-layer result."
            ),
        },
        {
            "claim_id": "R5C_CLAIM_06",
            "status": "NOT_SUPPORTED",
            "claim": (
                "Step7H achieves near-perfect crossing detection without substantial "
                "false alerts."
            ),
            "basis": (
                "Alert sensitivity is below 0.9 and false-alert rates remain around "
                "0.31–0.37 at the frozen operating points."
            ),
        },
        {
            "claim_id": "R5C_CLAIM_07",
            "status": "NOT_SUPPORTED",
            "claim": (
                "The trivial-baseline comparison establishes real-world warning effectiveness."
            ),
            "basis": (
                "The experiment uses a normalized virtual geofence and simulated response "
                "thresholds rather than field deployment."
            ),
        },
        {
            "claim_id": "R5C_CLAIM_08",
            "status": "METHOD_NOTE",
            "claim": (
                "Step7H remains the primary operational warning protocol; Step18 should "
                "be reported as secondary/ablation evidence."
            ),
            "basis": (
                "This hierarchy was already frozen in the revision protocol and R5A audit."
            ),
        },
    ]

    return pd.DataFrame(rows)


def main():
    print(
        "\nR5-C — TRIVIAL-BASELINE INTERPRETATION + CLAIM LOCK"
    )

    print(
        "NO new warning experiment is run. "
        "This step summarizes the frozen Step7H vs ALWAYS-RED / ALWAYS-GREEN "
        "comparison and locks manuscript-safe claims.\n"
    )

    verify()

    comparison = pd.read_csv(COMPARISON_CSV)
    review = pd.read_csv(REVIEW_CSV)

    summary = summarize(review)
    summary = add_urgent_prevalence(
        summary,
        comparison,
    )

    aggregate = aggregate_by_dataset_task(
        summary
    )

    claims = claim_lock()

    detailed_path = OUT / "R5C_trivial_baseline_detailed_interpretation.csv"
    aggregate_path = OUT / "R5C_dataset_task_summary.csv"
    claim_path = OUT / "R5C_CLAIM_LOCK.csv"

    summary.to_csv(
        detailed_path,
        index=False,
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    claims.to_csv(
        claim_path,
        index=False,
    )

    status = (
        "R5C_TRIVIAL_BASELINE_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R6_DIRECT_CROSSING_CLASSIFIER"
    )

    manifest = {
        "step":
            "R5C_TRIVIAL_BASELINE_INTERPRETATION_CLAIM_LOCK",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "status":
            status,

        "new_warning_experiment":
            False,

        "source_r5b_comparison":
            str(COMPARISON_CSV),

        "source_r5b_comparison_sha256":
            sha256_file(COMPARISON_CSV),

        "source_r5b_review":
            str(REVIEW_CSV),

        "source_r5b_review_sha256":
            sha256_file(REVIEW_CSV),

        "claim_lock_csv":
            str(claim_path),

        "claim_lock_sha256":
            sha256_file(claim_path),

        "dataset_task_summary_csv":
            str(aggregate_path),
    }

    manifest_path = OUT / "R5C_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("DATASET/TASK SUMMARY:")
    print(
        aggregate.to_string(
            index=False
        )
    )

    print("\nCLAIM LOCK:")
    print(
        claims.to_string(
            index=False
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR5-C COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
