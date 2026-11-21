
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R13_ROOT = REVISION / "R13_end_to_end_pipeline"

R13D = R13_ROOT / "R13D_frozen_detector_tracker_execution" / "R13D_MANIFEST.json"
R13E = R13_ROOT / "R13E_tracking_metrics" / "R13E_MANIFEST.json"
R13FA = R13_ROOT / "R13F_A_detector_tracker_bottleneck_audit" / "R13F_A_MANIFEST.json"
R13FB = R13_ROOT / "R13F_B_tracker_forecastability_audit" / "R13F_B_MANIFEST.json"

OUT = R13_ROOT / "R13F_C_end_to_end_feasibility_and_claim_lock"
OUT.mkdir(parents=True, exist_ok=True)


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


def load_and_verify():
    for p in [R13D, R13E, R13FA, R13FB]:
        require(p)

    d = json.loads(R13D.read_text(encoding="utf-8"))
    e = json.loads(R13E.read_text(encoding="utf-8"))
    a = json.loads(R13FA.read_text(encoding="utf-8"))
    b = json.loads(R13FB.read_text(encoding="utf-8"))

    expected = {
        "R13D": "R13D_FROZEN_DETECTOR_BYTETRACK_BOTSORT_EXECUTION_COMPLETE_RAW_OUTPUTS_LOCKED_READY_FOR_R13E_TRACKING_METRICS",
        "R13E": "R13E_TRACKING_METRICS_COMPLETE_HOTA_IDF1_IDSW_FRAGMENTATION_LOCKED_READY_FOR_R13F_ERROR_PROPAGATION",
        "R13FA": "R13F_A_DETECTOR_TRACKER_BOTTLENECK_DECOMPOSITION_COMPLETE_REVIEW_BEFORE_FORECAST_PROPAGATION",
        "R13FB": "R13F_B_TRACKER_DERIVED_FORECASTABILITY_AUDIT_COMPLETE_REVIEW_COVERAGE_BEFORE_FORECAST_INFERENCE",
    }

    actual = {
        "R13D": d.get("status"),
        "R13E": e.get("status"),
        "R13FA": a.get("status"),
        "R13FB": b.get("status"),
    }

    for k in expected:
        if actual[k] != expected[k]:
            raise RuntimeError(
                f"{k} unexpected status: {actual[k]}"
            )

    return d, e, a, b


def main():
    print("\nR13-F-C — END-TO-END FEASIBILITY / CLAIM LOCK")
    print(
        "FINAL CONSOLIDATION ONLY. NO inference, NO retraining, and NO tuning. "
        "This step locks the scientifically supported conclusion from the frozen detector/tracker execution, "
        "tracking metrics, detector-vs-tracker bottleneck decomposition, and tracker-derived forecastability audit. "
        "It explicitly prevents representative forecast/warning claims when temporal coverage is inadequate.\n"
    )

    d, e, a, b = load_and_verify()

    agg_tracking = pd.DataFrame(
        e["aggregate_results"]
    )

    byte_tracking = agg_tracking.loc[
        agg_tracking["tracker"] == "ByteTrack"
    ].iloc[0]

    bot_tracking = agg_tracking.loc[
        agg_tracking["tracker"] == "BoT-SORT"
    ].iloc[0]

    forecastability = pd.DataFrame(
        b["aggregate_results"]
    )

    byte_f = forecastability.loc[
        forecastability["tracker"] == "ByteTrack"
    ].iloc[0]

    bot_f = forecastability.loc[
        forecastability["tracker"] == "BoT-SORT"
    ].iloc[0]

    raw_detector_recall = float(
        a["raw_detector_recall_iou050"]
    )

    raw_detector_upper = float(
        a["raw_detector_count_upper_bound_recall"]
    )

    byte_id_recall = float(
        a["bytetrack_id_recall_iou050"]
    )

    bot_id_recall = float(
        a["botsort_id_recall_iou050"]
    )

    max_2s_windows = int(
        max(
            int(byte_f["windows_2s_to_2s"]),
            int(bot_f["windows_2s_to_2s"]),
        )
    )

    max_5s_windows = int(
        max(
            int(byte_f["windows_2s_to_5s"]),
            int(bot_f["windows_2s_to_5s"]),
        )
    )

    max_2s_coverage = float(
        max(
            float(byte_f["window_coverage_vs_gt_final_2s"]),
            float(bot_f["window_coverage_vs_gt_final_2s"]),
        )
    )

    max_5s_coverage = float(
        max(
            float(byte_f["window_coverage_vs_gt_final_5s"]),
            float(bot_f["window_coverage_vs_gt_final_5s"]),
        )
    )

    representative_forecast_allowed = bool(
        max_2s_coverage >= 0.10
        and max_5s_coverage >= 0.10
    )

    if representative_forecast_allowed:
        raise RuntimeError(
            "Unexpectedly sufficient tracker-derived forecast coverage. "
            "This lock script was designed for the sparse-coverage result; review before proceeding."
        )

    claims = pd.DataFrame(
        [
            {
                "claim":
                    "The historical GT-trajectory forecasting pipeline was an end-to-end detector-to-warning evaluation.",
                "status":
                    "REJECTED",
                "basis":
                    "Historical forecasting used GT-derived trajectories rather than detector/tracker-derived trajectories.",
            },
            {
                "claim":
                    "A new frozen detector+tracker revision experiment was executed on all six TestReal sequences.",
                "status":
                    "SUPPORTED",
                "basis":
                    "R13-D processed 9,085 frames with the locked detector, ByteTrack, and BoT-SORT.",
            },
            {
                "claim":
                    "Tracking quality is strong enough to support representative downstream forecasting.",
                "status":
                    "REJECTED",
                "basis":
                    f"ByteTrack HOTA={float(byte_tracking['HOTA']):.6f}, IDF1={float(byte_tracking['IDF1']):.6f}; "
                    f"BoT-SORT HOTA={float(bot_tracking['HOTA']):.6f}, IDF1={float(bot_tracking['IDF1']):.6f}.",
            },
            {
                "claim":
                    "The main end-to-end bottleneck is detector coverage on BIRDSAI TestReal.",
                "status":
                    "SUPPORTED",
                "basis":
                    f"Raw detector recall@IoU0.50={raw_detector_recall:.6f}; "
                    f"count-only upper-bound recall={raw_detector_upper:.6f}; "
                    f"ByteTrack ID-assigned recall={byte_id_recall:.6f}; "
                    f"BoT-SORT ID-assigned recall={bot_id_recall:.6f}.",
            },
            {
                "claim":
                    "Representative 2-s downstream forecast evaluation is supported by tracker-derived windows.",
                "status":
                    "REJECTED",
                "basis":
                    f"At most {max_2s_windows} 2→2 windows, only {100*max_2s_coverage:.3f}% of the historical 1,053-window reference.",
            },
            {
                "claim":
                    "Representative 5-s downstream forecast evaluation is supported by tracker-derived windows.",
                "status":
                    "REJECTED",
                "basis":
                    f"At most {max_5s_windows} 2→5 windows, {100*max_5s_coverage:.3f}% of the historical 826-window reference.",
            },
            {
                "claim":
                    "Running ADE/FDE or warning metrics on the two available 2-s tracker windows would establish end-to-end performance.",
                "status":
                    "REJECTED",
                "basis":
                    "Two windows are exploratory only and are not a representative evaluation population.",
            },
            {
                "claim":
                    "The new R13 experiment validly demonstrates detector/tracker error propagation feasibility limits.",
                "status":
                    "SUPPORTED",
                "basis":
                    "Frozen detector/tracker outputs, tracking metrics, bottleneck decomposition, and forecastability were evaluated without TestReal tuning.",
            },
        ]
    )

    roadmap = pd.DataFrame(
        [
            {
                "item":
                    "R13 end-to-end detector/tracker execution",
                "status":
                    "COMPLETE",
                "note":
                    "Frozen detector + ByteTrack + BoT-SORT executed on six TestReal sequences.",
            },
            {
                "item":
                    "R14 tracking metrics requirement",
                "status":
                    "SATISFIED BY R13-E",
                "note":
                    "HOTA, IDF1, ID-switches, fragmentation, MOTA and per-sequence results already computed.",
            },
            {
                "item":
                    "R15 representative forecast error propagation",
                "status":
                    "NOT ESTIMABLE REPRESENTATIVELY",
                "note":
                    "Only 2 tracker-derived 2-s windows and 0 tracker-derived 5-s windows.",
            },
            {
                "item":
                    "Next strengthening target",
                "status":
                    "READY",
                "note":
                    "Proceed to latency analysis / mixed-species robustness while keeping detector-domain limitation explicit.",
            },
        ]
    )

    manuscript = f"""R13 end-to-end feasibility conclusion — locked wording

A frozen revision experiment was conducted to assess whether the historical trajectory-forecasting pipeline could be extended to a detector-to-tracker-to-forecast workflow on BIRDSAI TestReal without test-set tuning. The locked elephant detector was applied to all six TestReal sequences, followed by ByteTrack as the primary tracker and BoT-SORT as a sensitivity tracker.

The detector/tracker stage showed severe domain-transfer limitations. The raw detector achieved a recall of {raw_detector_recall:.4f} at IoU >= 0.50 against the observed TestReal annotations, while the total number of detector boxes implied a count-only recall ceiling of {raw_detector_upper:.4f}. After identity assignment, observation recall was {byte_id_recall:.4f} for ByteTrack and {bot_id_recall:.4f} for BoT-SORT. Tracking performance was correspondingly low: ByteTrack achieved HOTA={float(byte_tracking['HOTA']):.4f} and IDF1={float(byte_tracking['IDF1']):.4f}, while BoT-SORT achieved HOTA={float(bot_tracking['HOTA']):.4f} and IDF1={float(bot_tracking['IDF1']):.4f}.

This lack of temporal coverage prevented a representative downstream forecasting evaluation. Each tracker produced only {max_2s_windows} valid 2-s forecasting windows, corresponding to approximately {100*max_2s_coverage:.2f}% of the 1,053-window historical 2-s evaluation set, and neither tracker produced any valid 5-s forecasting windows. Therefore, detector/tracker-derived ADE/FDE and warning metrics are not reported as representative end-to-end performance.

Reviewer-safe interpretation:
The historical forecasting results remain trajectory-level evaluations based on curated GT-derived trajectories. The new frozen end-to-end stress test does not validate operational detector-to-warning performance; instead, it identifies detector domain transfer as the dominant bottleneck and shows that the present detector/tracker combination does not provide sufficient temporal coverage for representative downstream forecasting on BIRDSAI TestReal.

Forbidden overclaims:
- Do not call the historical results end-to-end.
- Do not report two 2-s windows as representative forecast performance.
- Do not infer 5-s end-to-end forecasting performance because zero valid 5-s tracker-derived windows exist.
- Do not attribute the failure solely to tracker association; detector coverage is the primary observed bottleneck.
"""

    claims_path = OUT / "R13F_C_CLAIM_MATRIX.csv"
    roadmap_path = OUT / "R13F_C_ROADMAP_LOCK.csv"
    manuscript_path = OUT / "R13F_C_MANUSCRIPT_READY_CONCLUSION.txt"

    claims.to_csv(
        claims_path,
        index=False,
    )

    roadmap.to_csv(
        roadmap_path,
        index=False,
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8",
    )

    evidence = pd.DataFrame(
        [
            {
                "artifact": "R13D_MANIFEST",
                "path": str(R13D),
                "sha256": sha256_file(R13D),
            },
            {
                "artifact": "R13E_MANIFEST",
                "path": str(R13E),
                "sha256": sha256_file(R13E),
            },
            {
                "artifact": "R13F_A_MANIFEST",
                "path": str(R13FA),
                "sha256": sha256_file(R13FA),
            },
            {
                "artifact": "R13F_B_MANIFEST",
                "path": str(R13FB),
                "sha256": sha256_file(R13FB),
            },
        ]
    )

    evidence_path = OUT / "R13F_C_EVIDENCE_HASH_LOCK.csv"
    evidence.to_csv(
        evidence_path,
        index=False,
    )

    status = (
        "R13_END_TO_END_FEASIBILITY_FULLY_ASSESSED_DETECTOR_BOTTLENECK_LOCKED_REPRESENTATIVE_FORECAST_PROPAGATION_NOT_SUPPORTED_READY_FOR_NEXT_STRENGTHENING_STEP"
    )

    final_manifest = {
        "step":
            "R13F_C_END_TO_END_FEASIBILITY_AND_CLAIM_LOCK",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "new_inference":
            False,

        "tuning":
            False,

        "historical_pipeline_end_to_end_claim":
            False,

        "new_r13_detector_tracker_execution_complete":
            True,

        "representative_forecast_propagation_supported":
            False,

        "representative_warning_propagation_supported":
            False,

        "dominant_observed_bottleneck":
            "detector domain-transfer / coverage on BIRDSAI TestReal",

        "raw_detector_recall_iou050":
            raw_detector_recall,

        "raw_detector_count_upper_bound_recall":
            raw_detector_upper,

        "max_tracker_2s_windows":
            max_2s_windows,

        "max_tracker_5s_windows":
            max_5s_windows,

        "max_tracker_2s_coverage_vs_historical":
            max_2s_coverage,

        "max_tracker_5s_coverage_vs_historical":
            max_5s_coverage,

        "r14_tracking_metrics_requirement":
            "satisfied by R13-E",

        "r15_representative_error_propagation":
            "not estimable due inadequate tracker-derived temporal coverage",

        "outputs": {
            "claim_matrix":
                str(claims_path),

            "roadmap_lock":
                str(roadmap_path),

            "manuscript_ready_conclusion":
                str(manuscript_path),

            "evidence_hash_lock":
                str(evidence_path),
        },
    }

    manifest_path = OUT / "R13F_C_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            final_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("CLAIM MATRIX:")
    print(
        claims.to_string(
            index=False
        )
    )

    print("\nROADMAP LOCK:")
    print(
        roadmap.to_string(
            index=False
        )
    )

    print("\nSTATUS:")
    print(
        status
    )

    print("\nR13-F-C COMPLETE")
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
