
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
REVISION = DOWNLOADS / "paper_strengthening_revision"

R17E = (
    REVISION
    / "R17_mixed_species_robustness"
    / "R17E_claim_interpretation_lock"
    / "R17E_CLAIM_INTERPRETATION_LOCK.json"
)

R18B = (
    REVISION
    / "R18_meter_space_wildbox"
    / "R18B_meter_space_limitation_lock"
    / "R18B_METER_SPACE_LIMITATION_LOCK.json"
)

OUT = REVISION / "R19_final_claim_matrix_and_paper_rewrite_lock"
OUT.mkdir(parents=True, exist_ok=True)

R17_EXPECTED = (
    "R17_MIXED_SPECIES_ROBUSTNESS_FULLY_ASSESSED_FALSE_ALARM_REDUCTION_SUPPORTED_"
    "BUT_MAJOR_SENSITIVITY_LOSS_LOCKED_CURRENT_4CLASS_NOT_SUPPORTED_AS_REPLACEMENT_"
    "READY_FOR_R18_METER_SPACE_WILDBOX"
)

R18_EXPECTED = (
    "R18_WILDBOX_METER_SPACE_NOT_DEFENSIBLE_WITH_AVAILABLE_METADATA_"
    "LIMITATION_LOCKED_NORMALIZED_IMAGE_SPACE_RESULTS_RETAINED_"
    "READY_FOR_R19_CLAIM_MATRIX_AND_PAPER_REWRITE"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract_status_inventory():
    rows = []

    for p in REVISION.rglob("*.json"):
        if OUT in p.parents:
            continue

        try:
            data = read_json(p)
        except Exception:
            continue

        # R19 SAFE STATUS PARSER — dict/list compatible
        if isinstance(data, dict):
            status = str(data.get("status", "") or data.get("final_status", "") or "")
        elif isinstance(data, list):
            status_values = []
            for item in data:
                if isinstance(item, dict):
                    value = item.get("status", "") or item.get("final_status", "")
                    if value:
                        status_values.append(str(value))
            status = " | ".join(status_values)
        else:
            status = ""
        if not status:
            continue

        rows.append(
            {
                "path": str(p),
                "file_name": p.name,
                "step": str(data.get("step", "")),
                "status": str(status),
                "sha256": sha256_file(p),
            }
        )

    return pd.DataFrame(rows)


def roadmap_rows():
    return [
        {
            "step": "R1",
            "topic": "Frozen protocol / artifact lock",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Use frozen protocol/checkpoint/data hashes as provenance; do not imply post-test tuning."
            ),
        },
        {
            "step": "R2",
            "topic": "Fair TestReal + Fresh WildBox comparison",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Use matched evaluation across the same windows/samples; do not compare mismatched legacy WildBox counts."
            ),
        },
        {
            "step": "R3",
            "topic": "Paired clustered bootstrap CI",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Model superiority claims require paired clustered confidence intervals, not point estimates alone."
            ),
        },
        {
            "step": "R4",
            "topic": "Fixed conformal coverage",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Report frozen q90/q95 coverage honestly; do not recalibrate external test sets after inspection."
            ),
        },
        {
            "step": "R5",
            "topic": "Geofence trivial baselines",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Warning performance must be interpreted relative to always-RED/always-GREEN trivial baselines."
            ),
        },
        {
            "step": "R6",
            "topic": "Direct crossing classifier",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Treat direct crossing classifier as a higher-sensitivity secondary operating point, not the primary operational protocol."
            ),
        },
        {
            "step": "R7",
            "topic": "Warning hierarchy",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Primary operational hierarchy is RED actionable/selective warning, YELLOW watch/uncertainty, GREEN no alert."
            ),
        },
        {
            "step": "R8",
            "topic": "Ego-motion ablation",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Camera-motion correction improves geometric consistency and some downstream warning behavior; do not call corrected trajectories world ground truth."
            ),
        },
        {
            "step": "R9",
            "topic": "Canonicalization / isotropic sensitivity",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Original anisotropic representation remains the historical primary. Isotropic scaling is a matched sensitivity analysis and does not universally improve forecasting."
            ),
        },
        {
            "step": "R10",
            "topic": "Transformer ensemble",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Three-seed Transformer arithmetic-mean ensemble provides small but consistent gains versus seed comparators; do not claim overall model superiority."
            ),
        },
        {
            "step": "R11",
            "topic": "Noise-floor context",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Forecast error is much larger than residual camera-motion geometry. Ensemble gain is modest; F50/F95 are contextual residual scales, not significance thresholds."
            ),
        },
        {
            "step": "R12",
            "topic": "Track accounting",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Use original dataset-level track versus forecastable segment/tracklet terminology exactly; do not call 89/63 dataset tracks."
            ),
        },
        {
            "step": "R13",
            "topic": "Detector-to-tracker feasibility",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Historical trajectory forecasting remains GT-trajectory based, not end-to-end. New stress test shows detector domain-transfer/coverage is the dominant bottleneck."
            ),
        },
        {
            "step": "R14",
            "topic": "Tracking metrics",
            "final_state": "SATISFIED_BY_R13E",
            "paper_effect": (
                "Report HOTA, IDF1, ID switches and fragmentation as feasibility-limit evidence, not as strong tracker performance."
            ),
        },
        {
            "step": "R15",
            "topic": "Representative tracker-derived forecast propagation",
            "final_state": "NOT_ESTIMABLE_REPRESENTATIVELY",
            "paper_effect": (
                "Do not report the two available 2 s tracker-derived windows as representative forecasting performance; 5 s has zero tracker-derived windows."
            ),
        },
        {
            "step": "R16",
            "topic": "Latency",
            "final_state": "COMPLETE",
            "paper_effect": (
                "CPU compute fits inside 2 s/5 s horizons but does not establish operational response sufficiency; measured CPU pipeline also did not meet 10 Hz."
            ),
        },
        {
            "step": "R17",
            "topic": "Mixed-species robustness",
            "final_state": "COMPLETE",
            "paper_effect": (
                "Current four-class detector significantly reduces false-elephant alarms on no-elephant mixed-animal scenes but has major sensitivity/recall/F1 loss; it is not supported as a replacement."
            ),
        },
        {
            "step": "R18",
            "topic": "WildBox meter-space feasibility",
            "final_state": "COMPLETE_LIMITATION_LOCK",
            "paper_effect": (
                "WildBox results remain in normalized/image space. No meter-valued ADE/FDE or distance claim is allowed because no verified metric-scale anchor exists."
            ),
        },
    ]


def claim_rows():
    return [
        {
            "claim_id": "P01",
            "topic": "Overall system",
            "claim_status": "ALLOWED_WITH_QUALIFICATION",
            "recommended_claim": (
                "The study evaluates a UAV-oriented elephant trajectory-forecasting and uncertainty-aware virtual-geofence warning framework using frozen internal and external evaluations."
            ),
            "prohibited_claim": (
                "The full detector-to-warning system has been validated end-to-end in field conditions."
            ),
        },
        {
            "claim_id": "P02",
            "topic": "Detection foundation",
            "claim_status": "ALLOWED",
            "recommended_claim": (
                "The elephant detector provides the historical detection foundation, while later end-to-end stress testing shows substantial domain-transfer limitations on BIRDSAI TestReal."
            ),
            "prohibited_claim": (
                "The detector is universally robust across UAV datasets and domains."
            ),
        },
        {
            "claim_id": "P03",
            "topic": "Historical trajectory forecasting",
            "claim_status": "ALLOWED_WITH_QUALIFICATION",
            "recommended_claim": (
                "Historical forecasting experiments use annotated/ground-truth trajectory observations rather than fully detector-generated tracks."
            ),
            "prohibited_claim": (
                "All reported forecasting results are produced end-to-end from detector outputs."
            ),
        },
        {
            "claim_id": "P04",
            "topic": "Ego motion",
            "claim_status": "ALLOWED",
            "recommended_claim": (
                "Camera-motion correction improves reference-frame geometric consistency and supports a raw-versus-corrected forecasting/warning ablation."
            ),
            "prohibited_claim": (
                "Camera-corrected trajectories are world-coordinate ground truth."
            ),
        },
        {
            "claim_id": "P05",
            "topic": "Canonicalization",
            "claim_status": "ALLOWED_WITH_QUALIFICATION",
            "recommended_claim": (
                "Isotropic scaling removes aspect-ratio distortion in Euclidean geometry but does not universally improve forecasting performance."
            ),
            "prohibited_claim": (
                "Isotropic normalization is uniformly superior for all models and horizons."
            ),
        },
        {
            "claim_id": "P06",
            "topic": "Transformer",
            "claim_status": "ALLOWED_WITH_QUALIFICATION",
            "recommended_claim": (
                "The frozen three-seed Transformer ensemble provides small, statistically consistent benefits over individual-seed comparators on supported datasets/horizons."
            ),
            "prohibited_claim": (
                "Transformer is the best model overall or is novel by itself."
            ),
        },
        {
            "claim_id": "P07",
            "topic": "Uncertainty",
            "claim_status": "ALLOWED_WITH_QUALIFICATION",
            "recommended_claim": (
                "Frozen conformal thresholds provide uncertainty sets with dataset- and horizon-dependent empirical coverage."
            ),
            "prohibited_claim": (
                "Ensemble spread is a calibrated probability or BAMBI 5 s q90 is well calibrated."
            ),
        },
        {
            "claim_id": "P08",
            "topic": "Virtual geofence",
            "claim_status": "ALLOWED_WITH_QUALIFICATION",
            "recommended_claim": (
                "The virtual-geofence experiments quantify warning behavior under a normalized synthetic boundary protocol."
            ),
            "prohibited_claim": (
                "The geofence represents a surveyed real village/farm boundary or proves field effectiveness."
            ),
        },
        {
            "claim_id": "P09",
            "topic": "Response horizon",
            "claim_status": "PROHIBITED",
            "recommended_claim": (
                "Prediction horizons of 2 s and 5 s are forecasting horizons; measured compute latency is only one component of total operational response latency."
            ),
            "prohibited_claim": (
                "A 2 s or 5 s prediction horizon proves a human or drone can respond within that time."
            ),
        },
        {
            "claim_id": "P10",
            "topic": "Tracking",
            "claim_status": "ALLOWED_AS_LIMITATION",
            "recommended_claim": (
                "Frozen end-to-end stress testing exposes severe detector/tracker coverage limitations, with detector domain transfer as the dominant observed bottleneck."
            ),
            "prohibited_claim": (
                "Tracking performance is strong enough to support representative downstream forecasting."
            ),
        },
        {
            "claim_id": "P11",
            "topic": "Track accounting",
            "claim_status": "ALLOWED",
            "recommended_claim": (
                "Report 45 TestReal original dataset-level tracks, 37 represented original tracks and 89 forecastable segments for 2 s, and 31 represented original tracks and 63 forecastable segments for 5 s."
            ),
            "prohibited_claim": (
                "Call 89 or 63 the number of original dataset tracks."
            ),
        },
        {
            "claim_id": "P12",
            "topic": "BAMBI",
            "claim_status": "ALLOWED_AS_STRESS_TEST",
            "recommended_claim": (
                "BAMBI Flight 6 is a red-deer/human cross-species, cross-domain transfer stress test."
            ),
            "prohibited_claim": (
                "BAMBI Flight 6 is elephant-specific validation or a pristine untouched confirmatory holdout."
            ),
        },
        {
            "claim_id": "P13",
            "topic": "Mixed species",
            "claim_status": "ALLOWED_AS_TRADEOFF_FINDING",
            "recommended_claim": (
                "The current four-class detector reduces false-elephant alarms in external no-elephant mixed-animal scenes but incurs a large loss in elephant sensitivity/recall/F1."
            ),
            "prohibited_claim": (
                "The current four-class detector is a robust replacement for the elephant-only detector."
            ),
        },
        {
            "claim_id": "P14",
            "topic": "WildBox physical scale",
            "claim_status": "PROHIBITED_METER_CLAIM",
            "recommended_claim": (
                "WildBox evaluation remains in normalized/image coordinates because no verified source-specific metric-scale anchor was available."
            ),
            "prohibited_claim": (
                "Report WildBox ADE/FDE or camera translations directly in meters."
            ),
        },
        {
            "claim_id": "P15",
            "topic": "External validation",
            "claim_status": "ALLOWED_WITH_QUALIFICATION",
            "recommended_claim": (
                "Fresh WildBox, BIRDSAI TestReal and BAMBI provide complementary external or transfer evaluations with different limitations."
            ),
            "prohibited_claim": (
                "Treat all external datasets as equivalent pristine confirmatory tests."
            ),
        },
    ]


def rewrite_rows():
    return [
        {
            "paper_section": "Title",
            "action": "CHECK",
            "rewrite_rule": (
                "Avoid wording that implies field-deployed end-to-end operational validation."
            ),
        },
        {
            "paper_section": "Abstract",
            "action": "REWRITE",
            "rewrite_rule": (
                "State the strongest forecasting/warning results, but explicitly separate historical GT-trajectory forecasting from detector/tracker feasibility stress testing. Mention external transfer and major limitations."
            ),
        },
        {
            "paper_section": "Introduction / Contributions",
            "action": "REWRITE",
            "rewrite_rule": (
                "List contribution as an uncertainty-aware forecasting + virtual-geofence evaluation framework, ego-motion ablation, external transfer evaluation, and rigorous failure-mode audits. Do not list Transformer architecture itself as novelty."
            ),
        },
        {
            "paper_section": "Dataset",
            "action": "REWRITE",
            "rewrite_rule": (
                "Use exact original-track versus forecastable-segment terminology and exact TestReal accounting."
            ),
        },
        {
            "paper_section": "Detection / Tracking",
            "action": "REWRITE",
            "rewrite_rule": (
                "Separate historical detector baseline from the new frozen detector/tracker feasibility stress test. Report HOTA/IDF1/IDSW/fragmentation and detector-coverage bottleneck."
            ),
        },
        {
            "paper_section": "Ego-motion",
            "action": "REWRITE",
            "rewrite_rule": (
                "Use 'camera-corrected estimated reference-frame trajectory'; do not call it world GT."
            ),
        },
        {
            "paper_section": "Forecasting Models",
            "action": "REWRITE",
            "rewrite_rule": (
                "Do not imply Transformer universally wins. Report GRU/LSTM/Transformer/ensemble by dataset and horizon with paired-CI-supported conclusions."
            ),
        },
        {
            "paper_section": "Uncertainty",
            "action": "REWRITE",
            "rewrite_rule": (
                "Describe conformal calibration unit/score exactly and report undercoverage honestly, especially BAMBI 5 s q90."
            ),
        },
        {
            "paper_section": "Warning Evaluation",
            "action": "REWRITE",
            "rewrite_rule": (
                "Use Step7H as primary operational protocol; Step18 binary crossing as secondary/ablation. Define RED/YELLOW/GREEN exactly and include trivial baselines."
            ),
        },
        {
            "paper_section": "Latency",
            "action": "REWRITE",
            "rewrite_rule": (
                "Report compute latency only. State non-compute latency components are unmeasured and CPU throughput did not meet 10 Hz."
            ),
        },
        {
            "paper_section": "Mixed-species robustness",
            "action": "ADD",
            "rewrite_rule": (
                "Add R17 as a robustness/failure-mode experiment. Report false-alarm reduction together with the major sensitivity loss and reject replacement claim."
            ),
        },
        {
            "paper_section": "WildBox physical scale",
            "action": "ADD_LIMITATION",
            "rewrite_rule": (
                "State that no defensible meter conversion was performed because no verified metric-scale anchor was found despite available camera geometry."
            ),
        },
        {
            "paper_section": "Discussion",
            "action": "REWRITE",
            "rewrite_rule": (
                "Frame detector domain transfer, tracker forecastability, response latency, mixed-species selectivity trade-off, and missing metric scale as explicit limitations."
            ),
        },
        {
            "paper_section": "Conclusion",
            "action": "REWRITE",
            "rewrite_rule": (
                "Conclude with evidence-supported forecasting/warning findings and external stress-test insights; avoid field-effectiveness or universal robustness claims."
            ),
        },
    ]


def main():
    print("\nR19 — FINAL CLAIM MATRIX + PAPER REWRITE LOCK")
    print(
        "CONSOLIDATION ONLY. NO new experiment, NO model rerun, and NO post-test tuning. "
        "This step freezes the manuscript-safe claims and rewrite rules after R1-R18.\n"
    )

    require(R17E)
    require(R18B)

    r17 = read_json(R17E)
    r18 = read_json(R18B)

    if r17.get("status") != R17_EXPECTED:
        raise RuntimeError(
            f"Unexpected R17 status: {r17.get('status')}"
        )

    if r18.get("status") != R18_EXPECTED:
        raise RuntimeError(
            f"Unexpected R18 status: {r18.get('status')}"
        )

    inventory = extract_status_inventory()
    roadmap = pd.DataFrame(roadmap_rows())
    claims = pd.DataFrame(claim_rows())
    rewrites = pd.DataFrame(rewrite_rows())

    inventory_path = OUT / "R19_REVISION_STATUS_INVENTORY.csv"
    roadmap_path = OUT / "R19_ROADMAP_FINAL_STATUS.csv"
    claims_path = OUT / "R19_FINAL_CLAIM_MATRIX.csv"
    rewrites_path = OUT / "R19_PAPER_REWRITE_RULES.csv"

    inventory.to_csv(inventory_path, index=False)
    roadmap.to_csv(roadmap_path, index=False)
    claims.to_csv(claims_path, index=False)
    rewrites.to_csv(rewrites_path, index=False)

    md_lines = [
        "# R19 Final Manuscript Claim Lock",
        "",
        "## Core manuscript position",
        "",
        (
            "The paper may present a UAV-oriented elephant trajectory-forecasting, uncertainty, "
            "and virtual-geofence warning framework with extensive frozen internal/external "
            "evaluation and failure-mode auditing. It must not present the historical forecasting "
            "results as fully detector-to-warning end-to-end validation."
        ),
        "",
        "## Mandatory limitations",
        "",
        "- Detector domain-transfer/coverage is the dominant observed end-to-end bottleneck on BIRDSAI TestReal.",
        "- Representative tracker-derived downstream forecasting is not estimable (2 windows at 2 s; 0 at 5 s).",
        "- CPU compute latency alone does not establish operational response sufficiency and did not meet 10 Hz.",
        "- Current four-class mixed-species detector reduces false alarms but has a major elephant-sensitivity loss.",
        "- WildBox has camera geometry but no verified metric-scale anchor; no meter-valued ADE/FDE claim is allowed.",
        "- BAMBI Flight 6 is a cross-species/cross-domain stress test, not elephant-specific confirmatory validation.",
        "",
        "## Final writing rule",
        "",
        (
            "Any numerical or superiority statement must remain tied to the exact dataset, horizon, "
            "coordinate representation, frozen protocol, and confidence interval that supports it."
        ),
        "",
    ]

    md_path = OUT / "R19_FINAL_MANUSCRIPT_POSITION.md"
    md_path.write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )

    status = (
        "R19_FINAL_CLAIM_MATRIX_AND_PAPER_REWRITE_RULES_LOCKED_"
        "R1_TO_R18_STRENGTHENING_COMPLETE_READY_FOR_MANUSCRIPT_REVISION"
    )

    final_lock = {
        "step": "R19_FINAL_CLAIM_MATRIX_AND_PAPER_REWRITE_LOCK",
        "created_utc": utc_now(),
        "status": status,
        "new_experiment_performed": False,
        "model_rerun_performed": False,
        "post_test_tuning": False,
        "r17_status": r17.get("status"),
        "r18_status": r18.get("status"),
        "final_manuscript_position": (
            "Evidence-supported trajectory forecasting, uncertainty and virtual-geofence warning "
            "results may be reported with explicit separation from detector/tracker feasibility "
            "stress tests and with all locked limitations retained."
        ),
        "critical_prohibitions": [
            "Do not claim the historical forecasting pipeline is fully end-to-end.",
            "Do not claim 2 s or 5 s forecasting horizon equals operational response time.",
            "Do not claim Transformer is universally best.",
            "Do not call 89/63 original dataset tracks.",
            "Do not call BAMBI elephant-specific validation.",
            "Do not claim current 4-class detector is a suitable replacement.",
            "Do not convert WildBox errors to meters without a verified metric-scale anchor.",
            "Do not claim field effectiveness.",
        ],
        "outputs": {
            "revision_status_inventory": str(inventory_path),
            "roadmap_final_status": str(roadmap_path),
            "final_claim_matrix": str(claims_path),
            "paper_rewrite_rules": str(rewrites_path),
            "final_manuscript_position": str(md_path),
        },
        "output_hashes": {
            "roadmap_final_status": sha256_file(roadmap_path),
            "final_claim_matrix": sha256_file(claims_path),
            "paper_rewrite_rules": sha256_file(rewrites_path),
            "final_manuscript_position": sha256_file(md_path),
        },
        "next_action": (
            "Apply the locked R19 rules to the actual manuscript section-by-section without changing "
            "the frozen experimental results."
        ),
    }

    lock_path = OUT / "R19_FINAL_LOCK.json"
    lock_path.write_text(
        json.dumps(final_lock, indent=2),
        encoding="utf-8",
    )

    print("ROADMAP FINAL STATUS:")
    print(roadmap.to_string(index=False))

    print("\nFINAL CLAIM MATRIX:")
    print(claims.to_string(index=False))

    print("\nPAPER REWRITE RULES:")
    print(rewrites.to_string(index=False))

    print("\nSTATUS:")
    print(status)

    print("\nR19 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
