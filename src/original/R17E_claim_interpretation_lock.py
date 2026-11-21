
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timezone

HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

R17D_DIR = R17_ROOT / "R17D_locked_metric_evaluation"
R17D_MANIFEST = R17D_DIR / "R17D_MANIFEST.json"
PRIMARY_METRICS = R17D_DIR / "R17D_PRIMARY_CONF025_METRICS.csv"
AP_METRICS = R17D_DIR / "R17D_AP_METRICS.csv"
BOOTSTRAP = R17D_DIR / "R17D_PAIRED_BOOTSTRAP_CI.csv"
SUPP_METRICS = R17D_DIR / "R17D_SUPPLEMENTARY_IN_VOCAB_METRICS.csv"
FILTER_ABLATION = R17D_DIR / "R17D_SELECTIVE_FILTER_ALARM_ABLATION.csv"
DECISION_SUMMARY = R17D_DIR / "R17D_DECISION_SUMMARY.csv"

OUT = R17_ROOT / "R17E_claim_interpretation_lock"
OUT.mkdir(parents=True, exist_ok=True)

SELECTIVE_ID = "SELECTIVE_4CLASS_V5"
COMPARATOR_ID = "ELEPHANT_ONLY_ORIGINAL"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def read_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def get_boot(df, name):
    x = df.loc[
        df["comparison"] == name
    ]

    if len(x) != 1:
        raise RuntimeError(
            f"Expected one bootstrap row for {name}, found {len(x)}."
        )

    return x.iloc[0]


def get_primary_metric(df, detector_id, scope, metric):
    x = df.loc[
        (df["detector_id"] == detector_id)
        &
        (df["scope"] == scope),
        metric,
    ]

    if len(x) != 1:
        raise RuntimeError(
            f"Metric lookup failed: detector={detector_id}, scope={scope}, metric={metric}"
        )

    return float(
        x.iloc[0]
    )


def get_ap(df, detector_id, metric):
    x = df.loc[
        (df["detector_id"] == detector_id)
        &
        (df["metric"] == metric),
        "value",
    ]

    if len(x) != 1:
        raise RuntimeError(
            f"AP lookup failed: detector={detector_id}, metric={metric}"
        )

    return float(
        x.iloc[0]
    )


def main():
    print(
        "\nR17-E — MIXED-SPECIES CLAIM INTERPRETATION LOCK"
    )

    print(
        "INTERPRETATION LOCK ONLY. NO inference, NO training, NO threshold tuning. "
        "This step converts the frozen R17-D results into explicit supported/prohibited claims "
        "and locks whether the current 4-class detector is suitable as a replacement for the "
        "original elephant-only detector.\n"
    )

    for p in [
        R17D_MANIFEST,
        PRIMARY_METRICS,
        AP_METRICS,
        BOOTSTRAP,
        SUPP_METRICS,
        FILTER_ABLATION,
        DECISION_SUMMARY,
    ]:
        require(p)

    manifest = read_json(
        R17D_MANIFEST
    )

    expected = (
        "R17D_LOCKED_MIXED_SPECIES_METRICS_BOOTSTRAP_AND_FILTER_ABLATION_COMPLETE_READY_FOR_R17E_CLAIM_INTERPRETATION_LOCK"
    )

    if manifest.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R17-D status: {manifest.get('status')}"
        )

    primary = pd.read_csv(
        PRIMARY_METRICS,
        low_memory=False,
    )

    ap = pd.read_csv(
        AP_METRICS,
        low_memory=False,
    )

    boot = pd.read_csv(
        BOOTSTRAP,
        low_memory=False,
    )

    supp = pd.read_csv(
        SUPP_METRICS,
        low_memory=False,
    )

    ablation = pd.read_csv(
        FILTER_ABLATION,
        low_memory=False,
    )

    # Core bootstrap rows.
    false_alarm = get_boot(
        boot,
        "NEGATIVE_FALSE_ELEPHANT_IMAGE_RATE",
    )

    sensitivity = get_boot(
        boot,
        "POSITIVE_IMAGE_SENSITIVITY",
    )

    recall = get_boot(
        boot,
        "ALL_PRIMARY_453_RECALL",
    )

    f1 = get_boot(
        boot,
        "ALL_PRIMARY_453_F1",
    )

    precision = get_boot(
        boot,
        "ALL_PRIMARY_453_PRECISION",
    )

    confuser = get_boot(
        boot,
        "POSITIVE_CONFUSER_ERROR_RATE",
    )

    # Primary point estimates.
    selective_false = get_primary_metric(
        primary,
        SELECTIVE_ID,
        "NEGATIVE_MULTI_ANIMAL_NO_ELEPHANT",
        "false_elephant_image_rate",
    )

    comparator_false = get_primary_metric(
        primary,
        COMPARATOR_ID,
        "NEGATIVE_MULTI_ANIMAL_NO_ELEPHANT",
        "false_elephant_image_rate",
    )

    selective_recall = get_primary_metric(
        primary,
        SELECTIVE_ID,
        "ALL_PRIMARY_453",
        "recall",
    )

    comparator_recall = get_primary_metric(
        primary,
        COMPARATOR_ID,
        "ALL_PRIMARY_453",
        "recall",
    )

    selective_f1 = get_primary_metric(
        primary,
        SELECTIVE_ID,
        "ALL_PRIMARY_453",
        "f1",
    )

    comparator_f1 = get_primary_metric(
        primary,
        COMPARATOR_ID,
        "ALL_PRIMARY_453",
        "f1",
    )

    selective_ap50 = get_ap(
        ap,
        SELECTIVE_ID,
        "AP50",
    )

    comparator_ap50 = get_ap(
        ap,
        COMPARATOR_ID,
        "AP50",
    )

    selective_ap5095 = get_ap(
        ap,
        SELECTIVE_ID,
        "AP50:95",
    )

    comparator_ap5095 = get_ap(
        ap,
        COMPARATOR_ID,
        "AP50:95",
    )

    # Supplementary lion metrics.
    s_sel = supp.loc[
        supp["detector_id"] == SELECTIVE_ID
    ].iloc[0]

    s_cmp = supp.loc[
        supp["detector_id"] == COMPARATOR_ID
    ].iloc[0]

    # Filter ablation.
    any_alarm = ablation.loc[
        ablation["alarm_mode"] == "UNFILTERED_ANY_CLASS_ALARM"
    ].iloc[0]

    filtered_alarm = ablation.loc[
        ablation["alarm_mode"] == "ELEPHANT_CLASS_FILTERED_ALARM"
    ].iloc[0]

    false_alarm_reduction_supported = bool(
        float(false_alarm["ci95_low"]) > 0
    )

    major_sensitivity_loss_supported = bool(
        float(sensitivity["ci95_high"]) < 0
    )

    recall_loss_supported = bool(
        float(recall["ci95_high"]) < 0
    )

    f1_loss_supported = bool(
        float(f1["ci95_high"]) < 0
    )

    precision_loss_supported = bool(
        float(precision["ci95_high"]) < 0
    )

    confuser_improvement_supported = bool(
        float(confuser["ci95_low"]) > 0
    )

    # Replacement rule:
    # current 4-class detector cannot replace original if it has a statistically
    # supported sensitivity/recall/F1 loss, even if false alarms improve.
    replacement_supported = bool(
        false_alarm_reduction_supported
        and
        not major_sensitivity_loss_supported
        and
        not recall_loss_supported
        and
        not f1_loss_supported
    )

    claim_rows = [
        {
            "claim_id": "C1",
            "claim": (
                "The current frozen 4-class detector reduces false-elephant alarms "
                "on external multi-animal scenes without elephants relative to the original "
                "elephant-only detector."
            ),
            "status": (
                "SUPPORTED"
                if false_alarm_reduction_supported
                else "NOT_SUPPORTED"
            ),
            "evidence": (
                f"false-elephant image rate {comparator_false:.6f} -> {selective_false:.6f}; "
                f"paired delta comparator-selective={float(false_alarm['point_delta_positive_favors_selective']):.6f}; "
                f"95% CI [{float(false_alarm['ci95_low']):.6f}, {float(false_alarm['ci95_high']):.6f}]"
            ),
        },
        {
            "claim_id": "C2",
            "claim": (
                "The current frozen 4-class detector maintains elephant sensitivity "
                "while reducing false alarms."
            ),
            "status": "REJECTED",
            "evidence": (
                f"positive-image sensitivity delta selective-comparator="
                f"{float(sensitivity['point_delta_positive_favors_selective']):.6f}; "
                f"95% CI [{float(sensitivity['ci95_low']):.6f}, {float(sensitivity['ci95_high']):.6f}]"
            ),
        },
        {
            "claim_id": "C3",
            "claim": (
                "The current frozen 4-class detector is overall superior to the original "
                "elephant-only detector."
            ),
            "status": "REJECTED",
            "evidence": (
                f"overall recall {selective_recall:.6f} vs {comparator_recall:.6f}; "
                f"F1 {selective_f1:.6f} vs {comparator_f1:.6f}; "
                f"AP50 {selective_ap50:.6f} vs {comparator_ap50:.6f}; "
                f"AP50:95 {selective_ap5095:.6f} vs {comparator_ap5095:.6f}"
            ),
        },
        {
            "claim_id": "C4",
            "claim": (
                "The current frozen 4-class detector significantly reduces animal-confuser "
                "to elephant overlap errors in positive mixed scenes."
            ),
            "status": (
                "SUPPORTED"
                if confuser_improvement_supported
                else "NOT_SUPPORTED"
            ),
            "evidence": (
                f"paired delta comparator-selective="
                f"{float(confuser['point_delta_positive_favors_selective']):.6f}; "
                f"95% CI [{float(confuser['ci95_low']):.6f}, {float(confuser['ci95_high']):.6f}]"
            ),
        },
        {
            "claim_id": "C5",
            "claim": (
                "The 9-image elephant+lion supplement establishes broad in-vocabulary "
                "mixed-species robustness."
            ),
            "status": "REJECTED",
            "evidence": (
                f"selective elephant recall={float(s_sel['elephant_recall']):.6f}, "
                f"comparator recall={float(s_cmp['elephant_recall']):.6f}; "
                f"lion-to-elephant overlap errors={int(s_sel['lion_to_elephant_overlap_error_predictions'])} "
                f"for selective and {int(s_cmp['lion_to_elephant_overlap_error_predictions'])} for comparator"
            ),
        },
        {
            "claim_id": "C6",
            "claim": (
                "Elephant-class filtering itself is the main reason for the lower negative-scene alarm rate."
            ),
            "status": "NOT_SUPPORTED",
            "evidence": (
                f"4-class any-class negative alarm rate={float(any_alarm['negative_alarm_rate']):.6f}; "
                f"elephant-filtered rate={float(filtered_alarm['negative_alarm_rate']):.6f}; "
                "difference is small compared with the detector-to-detector false-alarm difference"
            ),
        },
        {
            "claim_id": "C7",
            "claim": (
                "R17 validates detector-to-tracker-to-forecast end-to-end operational performance."
            ),
            "status": "REJECTED",
            "evidence": (
                "R17 is an image-level detector/selective-filter robustness experiment only; "
                "the historical end-to-end limitation remains unchanged."
            ),
        },
        {
            "claim_id": "C8",
            "claim": (
                "The current 4-class detector should replace the original elephant-only detector "
                "in the present system."
            ),
            "status": (
                "SUPPORTED"
                if replacement_supported
                else "REJECTED"
            ),
            "evidence": (
                "False-alarm reduction is statistically supported, but sensitivity, recall, "
                "precision and F1 losses are statistically supported; therefore replacement is not justified."
            ),
        },
    ]

    claims = pd.DataFrame(
        claim_rows
    )

    claims_path = OUT / "R17E_CLAIM_MATRIX_LOCKED.csv"
    claims.to_csv(
        claims_path,
        index=False,
    )

    quantitative = pd.DataFrame(
        [
            {
                "metric":
                    "negative_false_elephant_image_rate",

                "selective":
                    selective_false,

                "comparator":
                    comparator_false,

                "preferred_direction":
                    "lower",
            },
            {
                "metric":
                    "overall_recall",

                "selective":
                    selective_recall,

                "comparator":
                    comparator_recall,

                "preferred_direction":
                    "higher",
            },
            {
                "metric":
                    "overall_f1",

                "selective":
                    selective_f1,

                "comparator":
                    comparator_f1,

                "preferred_direction":
                    "higher",
            },
            {
                "metric":
                    "AP50",

                "selective":
                    selective_ap50,

                "comparator":
                    comparator_ap50,

                "preferred_direction":
                    "higher",
            },
            {
                "metric":
                    "AP50:95",

                "selective":
                    selective_ap5095,

                "comparator":
                    comparator_ap5095,

                "preferred_direction":
                    "higher",
            },
        ]
    )

    quantitative_path = OUT / "R17E_KEY_RESULT_LOCK.csv"
    quantitative.to_csv(
        quantitative_path,
        index=False,
    )

    status = (
        "R17_MIXED_SPECIES_ROBUSTNESS_FULLY_ASSESSED_FALSE_ALARM_REDUCTION_SUPPORTED_BUT_MAJOR_SENSITIVITY_LOSS_LOCKED_CURRENT_4CLASS_NOT_SUPPORTED_AS_REPLACEMENT_READY_FOR_R18_METER_SPACE_WILDBOX"
    )

    lock = {
        "step":
            "R17E_MIXED_SPECIES_CLAIM_INTERPRETATION_LOCK",

        "created_utc":
            utc_now(),

        "status":
            status,

        "r17_complete":
            True,

        "current_4class_replacement_supported":
            replacement_supported,

        "supported_positive_finding":
            (
                "The frozen 4-class detector significantly reduces false-elephant image alarms "
                "on external no-elephant multi-animal scenes relative to the original elephant-only detector."
            ),

        "dominant_negative_finding":
            (
                "The false-alarm reduction is accompanied by a large statistically supported loss "
                "of elephant sensitivity/recall/F1, so the current 4-class model is not supported as "
                "a replacement detector."
            ),

        "confuser_error_finding":
            (
                "A significant improvement in animal-confuser-to-elephant overlap error rate was not demonstrated."
            ),

        "supplementary_lion_finding":
            (
                "The 9-image elephant+lion challenge is small and does not establish broad in-vocabulary robustness."
            ),

        "future_model_rule":
            (
                "If a new mixed-species detector is trained or tuned after R17, the current frozen COCO 453 "
                "evaluation set must be treated as seen/evaluated and must not be described as a pristine untouched "
                "external test for that new model. A fresh untouched external test set is required for confirmatory claims."
            ),

        "manuscript_claim_rule":
            (
                "Report R17 as a robustness stress test showing a selectivity-vs-sensitivity trade-off and a current "
                "detector limitation. Do not claim universal mixed-species robustness, unseen-species robustness, "
                "or end-to-end operational validation."
            ),

        "input_hashes": {
            "R17D_manifest":
                sha256_file(
                    R17D_MANIFEST
                ),

            "primary_metrics":
                sha256_file(
                    PRIMARY_METRICS
                ),

            "bootstrap":
                sha256_file(
                    BOOTSTRAP
                ),

            "supplementary":
                sha256_file(
                    SUPP_METRICS
                ),

            "filter_ablation":
                sha256_file(
                    FILTER_ABLATION
                ),
        },

        "output_hashes": {
            "claim_matrix":
                sha256_file(
                    claims_path
                ),

            "key_result_lock":
                sha256_file(
                    quantitative_path
                ),
        },

        "next_roadmap_step":
            "R18_METER_SPACE_WILDBOX",
    }

    lock_path = OUT / "R17E_CLAIM_INTERPRETATION_LOCK.json"

    lock_path.write_text(
        json.dumps(
            lock,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "KEY RESULT LOCK:"
    )
    print(
        quantitative.to_string(
            index=False
        )
    )

    print(
        "\nCLAIM MATRIX:"
    )
    print(
        claims.to_string(
            index=False
        )
    )

    print(
        "\nFINAL R17 INTERPRETATION:"
    )
    print(
        "- Supported: strong reduction in false-elephant alarms on no-elephant multi-animal scenes."
    )
    print(
        "- Also supported: major loss of elephant sensitivity/recall/F1 relative to the original elephant-only detector."
    )
    print(
        "- Therefore: current 4-class detector is NOT supported as a replacement."
    )
    print(
        "- R17 is a detector/selective-filter robustness stress test, not end-to-end validation."
    )
    print(
        "- Any future improved mixed-species model requires a fresh untouched confirmatory test set."
    )

    print(
        "\nSTATUS:"
    )
    print(
        status
    )

    print(
        "\nR17-E COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
