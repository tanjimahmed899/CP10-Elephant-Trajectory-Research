
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

R5C_MANIFEST = (
    DOWNLOADS / "paper_strengthening_revision" / "R5_geofence_baselines"
    / "R5C_interpretation_claim_lock" / "R5C_MANIFEST.json"
)

R6E_MANIFEST = (
    DOWNLOADS / "paper_strengthening_revision" / "R6_direct_crossing_classifier"
    / "R6E_interpretation_claim_lock" / "R6E_MANIFEST.json"
)

STEP7H_METRICS = (
    MAIN_RESULT_2 / "step7_response_aware_warning"
    / "TABLE_step7h_final_response_aware_metrics.csv"
)

STEP7H_INSTANCES = (
    MAIN_RESULT_2 / "step7_response_aware_warning"
    / "step7h_final_response_aware_instances.csv"
)

STEP18_ROOT = MAIN_RESULT / "step18_geofence"

OUT = (
    DOWNLOADS / "paper_strengthening_revision"
    / "R7_warning_hierarchy"
    / "R7A_operational_hierarchy_lock"
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


def verify_prior():
    for p in [
        R5C_MANIFEST,
        R6E_MANIFEST,
        STEP7H_METRICS,
        STEP7H_INSTANCES,
    ]:
        require(p)

    r5 = json.loads(
        R5C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    r6 = json.loads(
        R6E_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        r5.get("status")
        !=
        "R5C_TRIVIAL_BASELINE_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R6_DIRECT_CROSSING_CLASSIFIER"
    ):
        raise RuntimeError(
            f"Unexpected R5C status: {r5.get('status')}"
        )

    if (
        r6.get("status")
        !=
        "R6E_DIRECT_CROSSING_CLASSIFIER_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R7_WARNING_HIERARCHY"
    ):
        raise RuntimeError(
            f"Unexpected R6E status: {r6.get('status')}"
        )


def step18_inventory():
    rows = []

    if not STEP18_ROOT.exists():
        return pd.DataFrame()

    for p in STEP18_ROOT.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in {
            ".csv",
            ".json",
            ".txt",
            ".py",
        }:
            continue

        row = {
            "path": str(p),
            "suffix": p.suffix.lower(),
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }

        if p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(p)
                row["rows"] = int(len(df))
                row["columns"] = " | ".join(
                    map(str, df.columns)
                )
            except Exception as e:
                row["rows"] = None
                row["columns"] = None
                row["error"] = (
                    f"{type(e).__name__}: {e}"
                )
        else:
            row["rows"] = None
            row["columns"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def hierarchy_table():
    return pd.DataFrame(
        [
            {
                "component":
                    "Step7H response-aware RED/YELLOW/GREEN rule",

                "paper_role":
                    "PRIMARY_OPERATIONAL_WARNING_PROTOCOL",

                "decision_output":
                    "GREEN / YELLOW / RED",

                "uses_nominal_forecast":
                    True,

                "uses_uncertainty":
                    True,

                "uses_persistence":
                    True,

                "uses_response_threshold":
                    True,

                "main_interpretation":
                    (
                        "Primary operational decision logic. "
                        "RED = selective/actionable urgent warning; "
                        "YELLOW = watch/uncertainty flag; "
                        "GREEN = no current warning."
                    ),

                "allowed_claim":
                    (
                        "Use as the main warning-system result and discuss "
                        "sensitivity, false-alert rate, RED selectivity, "
                        "uncertainty contribution, persistence and "
                        "response-aware behavior."
                    ),

                "forbidden_claim":
                    (
                        "Do not call field effectiveness, real village/farm "
                        "deployment, or real response-time validation."
                    ),
            },

            {
                "component":
                    "Step18 binary geofence crossing benchmark",

                "paper_role":
                    "SECONDARY_BINARY_CROSSING_BENCHMARK",

                "decision_output":
                    "cross / no-cross",

                "uses_nominal_forecast":
                    True,

                "uses_uncertainty":
                    False,

                "uses_persistence":
                    False,

                "uses_response_threshold":
                    False,

                "main_interpretation":
                    (
                        "Simplified binary crossing benchmark retained "
                        "for ablation/diagnostic comparison."
                    ),

                "allowed_claim":
                    (
                        "Use as secondary evidence for crossing prediction "
                        "and as a basis for the direct-classifier benchmark."
                    ),

                "forbidden_claim":
                    (
                        "Do not present Step18 as the primary operational "
                        "warning protocol."
                    ),
            },

            {
                "component":
                    "R6 direct logistic crossing classifier",

                "paper_role":
                    "SECONDARY_DECISION_LAYER_CLASSIFIER_ABLATION",

                "decision_output":
                    "cross / no-cross probability + locked threshold",

                "uses_nominal_forecast":
                    True,

                "uses_uncertainty":
                    False,

                "uses_persistence":
                    False,

                "uses_response_threshold":
                    False,

                "main_interpretation":
                    (
                        "Reviewer-facing direct classifier benchmark. "
                        "Provides a higher-sensitivity operating point "
                        "with higher false-positive rate and lower precision."
                    ),

                "allowed_claim":
                    (
                        "Report statistically supported balanced-accuracy/"
                        "sensitivity gains together with the false-positive/"
                        "precision cost."
                    ),

                "forbidden_claim":
                    (
                        "Do not call universally superior and do not "
                        "replace Step7H with this classifier."
                    ),
            },
        ]
    )


def claim_lock():
    return pd.DataFrame(
        [
            {
                "claim_id": "R7A_CLAIM_01",
                "status": "METHOD_LOCK",
                "claim": (
                    "Step7H is the primary operational warning "
                    "protocol for the paper."
                ),
                "basis": (
                    "It uniquely implements the frozen response-aware "
                    "RED/YELLOW/GREEN logic with nominal crossing, "
                    "q90 uncertainty and persistence."
                ),
            },
            {
                "claim_id": "R7A_CLAIM_02",
                "status": "METHOD_LOCK",
                "claim": (
                    "Step18 is secondary binary-crossing "
                    "benchmark/ablation evidence."
                ),
                "basis": (
                    "It lacks the full response-aware uncertainty/"
                    "persistence hierarchy used by Step7H."
                ),
            },
            {
                "claim_id": "R7A_CLAIM_03",
                "status": "METHOD_LOCK",
                "claim": (
                    "R6 direct logistic crossing classifier is a "
                    "secondary decision-layer benchmark."
                ),
                "basis": (
                    "It was added to answer the reviewer-style direct "
                    "classifier comparison and has a different "
                    "sensitivity/false-positive operating point."
                ),
            },
            {
                "claim_id": "R7A_CLAIM_04",
                "status": "SUPPORTED",
                "claim": (
                    "RED should be described as the selective/actionable "
                    "channel; YELLOW as a watch/uncertainty flag."
                ),
                "basis": (
                    "Frozen Step7H definitions and R5 trivial-baseline "
                    "interpretation."
                ),
            },
            {
                "claim_id": "R7A_CLAIM_05",
                "status": "NOT_SUPPORTED",
                "claim": (
                    "Step18 and Step7H are interchangeable operational "
                    "protocols."
                ),
                "basis": (
                    "They answer different decision questions and use "
                    "different logic."
                ),
            },
            {
                "claim_id": "R7A_CLAIM_06",
                "status": "NOT_SUPPORTED",
                "claim": (
                    "The direct classifier should replace Step7H because "
                    "it improves balanced accuracy."
                ),
                "basis": (
                    "R6E shows a clear cost in false-positive rate and "
                    "precision, while Step7H remains the frozen primary "
                    "operational protocol."
                ),
            },
            {
                "claim_id": "R7A_CLAIM_07",
                "status": "NOT_SUPPORTED",
                "claim": (
                    "Any of these warning experiments establish real-world "
                    "field effectiveness."
                ),
                "basis": (
                    "The geofence is normalized/synthetic and the response "
                    "thresholds are simulated."
                ),
            },
        ]
    )


def manuscript_map():
    return pd.DataFrame(
        [
            {
                "paper_section":
                    "Methods — Warning protocol",

                "primary_content":
                    (
                        "Describe Step7H first: normalized virtual geofence, "
                        "RED/YELLOW/GREEN states, q90 uncertainty, "
                        "persistence N=2, response thresholds."
                    ),

                "secondary_content":
                    (
                        "Mention Step18 and R6 classifier only as secondary "
                        "binary-crossing benchmarks."
                    ),
            },
            {
                "paper_section":
                    "Results — Operational warning",

                "primary_content":
                    (
                        "Lead with Step7H sensitivity, false-alert rate, "
                        "RED urgent precision/selectivity, uncertainty "
                        "contribution and persistence behavior."
                    ),

                "secondary_content":
                    (
                        "Then report Step18/R6 binary classifier as "
                        "ablation/benchmark comparison."
                    ),
            },
            {
                "paper_section":
                    "Discussion",

                "primary_content":
                    (
                        "Emphasize operating-point trade-offs, "
                        "synthetic-geofence limitation and lack of "
                        "field-effectiveness validation."
                    ),

                "secondary_content":
                    (
                        "State that classifier gains in sensitivity come "
                        "with higher FPR/lower precision."
                    ),
            },
        ]
    )


def main():
    print(
        "\nR7-A — WARNING HIERARCHY FORMAL LOCK"
    )

    print(
        "NO new experiment. This step removes hierarchy ambiguity by "
        "formally locking Step7H as primary and Step18/R6 as secondary evidence.\n"
    )

    verify_prior()

    inv = step18_inventory()
    hierarchy = hierarchy_table()
    claims = claim_lock()
    mapping = manuscript_map()

    inv_path = (
        OUT
        / "R7A_step18_artifact_inventory.csv"
    )

    hierarchy_path = (
        OUT
        / "R7A_WARNING_HIERARCHY_LOCK.csv"
    )

    claims_path = (
        OUT
        / "R7A_CLAIM_LOCK.csv"
    )

    mapping_path = (
        OUT
        / "R7A_MANUSCRIPT_SECTION_MAP.csv"
    )

    inv.to_csv(
        inv_path,
        index=False,
    )

    hierarchy.to_csv(
        hierarchy_path,
        index=False,
    )

    claims.to_csv(
        claims_path,
        index=False,
    )

    mapping.to_csv(
        mapping_path,
        index=False,
    )

    status = (
        "R7A_WARNING_HIERARCHY_FORMALLY_LOCKED_READY_FOR_R8_EGO_MOTION_ABLATION"
    )

    manifest = {
        "step":
            "R7A_WARNING_HIERARCHY_FORMAL_LOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_experiment":
            False,

        "step7h_metrics":
            str(
                STEP7H_METRICS
            ),

        "step7h_metrics_sha256":
            sha256_file(
                STEP7H_METRICS
            ),

        "step7h_instances":
            str(
                STEP7H_INSTANCES
            ),

        "step7h_instances_sha256":
            sha256_file(
                STEP7H_INSTANCES
            ),

        "hierarchy_csv":
            str(
                hierarchy_path
            ),

        "hierarchy_sha256":
            sha256_file(
                hierarchy_path
            ),

        "claim_lock_csv":
            str(
                claims_path
            ),

        "claim_lock_sha256":
            sha256_file(
                claims_path
            ),

        "manuscript_map_csv":
            str(
                mapping_path
            ),
    }

    (
        OUT
        / "R7A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "WARNING HIERARCHY LOCK:"
    )
    print(
        hierarchy.to_string(
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
        "\nR7-A COMPLETE"
    )
    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
