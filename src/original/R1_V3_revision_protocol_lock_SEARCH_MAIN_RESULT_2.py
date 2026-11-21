from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

SEARCH_ROOTS = [
    DOWNLOADS,
    MAIN_RESULT,
    MAIN_RESULT_2,
]

OUT = DOWNLOADS / "paper_strengthening_revision" / "R1_protocol_lock_v3"
OUT.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {
    "K3C10_step7_prelock": {
        "patterns": [
            "STEP6D_K3C10_BAMBI_STEP7_PRELOCK.json",
            "*K3C10*STEP7*PRELOCK*.json",
            "*K3C10*PRELOCK*.json",
        ],
        "expected_sha256":
            "8cc687c3f91ecbfb2a3c0f491427d91ccfdca959659a52e0836a98889c890bd1",
    },

    "K3C13_canonical_lock": {
        "patterns": [
            "STEP6D_K3C13_CANONICAL_DATASET_LOCK.json",
            "*K3C13*CANONICAL*LOCK*.json",
        ],
        "expected_sha256":
            "71d83c774233538342f4b410a789750943e61a8513fe983249c8f8f5a2aee605",
    },

    "K3C15_prediction_lock": {
        "patterns": [
            "STEP6D_K3C15_SECONDARY_FROZEN_INFERENCE_LOCK.json",
            "*K3C15*FROZEN*INFERENCE*LOCK*.json",
        ],
        "expected_sha256":
            "06a863cf9d86d1cf9faa3416d7fbfe41f0eadcb3d9ffba25845e7cf801a65c33",
    },

    "K3C16_conformal_summary": {
        "patterns": [
            "STEP6D_K3C16_SUMMARY.json",
            "*K3C16*SUMMARY*.json",
        ],
        "expected_sha256":
            "21685938c4995476cd6a45acca525c800a30ff12913690b059e92f16506e3104",
    },

    "K3C17_step7_manifest": {
        "patterns": [
            "STEP6D_K3C17_FROZEN_STEP7_EXTERNAL_EVALUATION_MANIFEST.json",
            "*K3C17*STEP7*MANIFEST*.json",
            "*K3C17*EXTERNAL*EVALUATION*MANIFEST*.json",
        ],
        "expected_sha256":
            "5f89576a3d27114b7a8d599c6a2c3a7f88574777a0d2dae810420b7a862d22ab",
    },

    "K3C18_final_summary": {
        "patterns": [
            "STEP6D_K3C18_FINAL_EXTERNAL_PIPELINE_SUMMARY.json",
            "*K3C18*FINAL*PIPELINE*SUMMARY*.json",
            "*K3C18*FINAL*SUMMARY*.json",
        ],
        "expected_sha256":
            "b2c7ff95095c19dbe564b239c20b8631820b962321fc84cc6d28c8277fb41050",
    },
}


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def discover_by_hash(name, config):
    expected = config["expected_sha256"]
    candidates = []
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for pattern in config["patterns"]:
            for p in root.rglob(pattern):
                if not p.is_file():
                    continue

                rp = str(p.resolve()).lower()

                if rp in seen:
                    continue

                seen.add(rp)
                candidates.append(p)

    checked = []
    matches = []

    for p in candidates:
        try:
            actual = sha256_file(p)
        except Exception as e:
            checked.append({
                "path": str(p),
                "sha256": None,
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        checked.append({
            "path": str(p),
            "sha256": actual,
            "error": None,
        })

        if actual == expected:
            matches.append(p)

    if matches:
        # Prefer Main Result 2.0, then Main Result, then Downloads.
        def rank(p):
            s = str(p).lower()

            if str(MAIN_RESULT_2).lower() in s:
                root_rank = 0
            elif str(MAIN_RESULT).lower() in s:
                root_rank = 1
            elif str(DOWNLOADS).lower() in s:
                root_rank = 2
            else:
                root_rank = 9

            return (
                root_rank,
                len(str(p)),
                str(p).lower(),
            )

        matches = sorted(matches, key=rank)

        return matches[0], checked, matches

    raise RuntimeError(
        f"\nCould not prove frozen artifact: {name}\n"
        f"Expected SHA256: {expected}\n"
        f"Search roots:\n"
        + "\n".join(f"  - {root}" for root in SEARCH_ROOTS)
        + f"\nPatterns searched: {config['patterns']}\n"
        + f"Candidates found: {len(candidates)}\n"
        + "Checked candidates:\n"
        + "\n".join(
            f"  - {row['path']} | {row['sha256']} | {row['error']}"
            for row in checked
        )
    )


def verify_frozen():
    rows = []

    for name, config in ARTIFACTS.items():
        selected, checked, matches = discover_by_hash(
            name,
            config,
        )

        actual = sha256_file(selected)

        rows.append({
            "artifact":
                name,

            "selected_path":
                str(selected),

            "expected_sha256":
                config["expected_sha256"],

            "actual_sha256":
                actual,

            "match":
                actual
                ==
                config["expected_sha256"],

            "candidate_count_checked":
                len(checked),

            "hash_matching_copies":
                len(matches),
        })

    return pd.DataFrame(rows)


PROTOCOL = {
    "protocol_name":
        "CP1.0 Paper Strengthening Revision Protocol",

    "protocol_version":
        "R1-V3",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "purpose":
        (
            "Freeze the paper-strengthening revision plan before any new revision "
            "results are inspected, while preserving K3C10-K3C18 as read-only."
        ),

    "search_roots_used_for_frozen_provenance": [
        str(DOWNLOADS),
        str(MAIN_RESULT),
        str(MAIN_RESULT_2),
    ],

    "existing_snapshot_rule": {
        "K3C10_to_K3C18_read_only":
            True,

        "new_revision_root":
            r"C:\Users\HP\Downloads\paper_strengthening_revision",

        "no_overwrite":
            True,

        "no_post_test_tuning":
            True,
    },

    "primary_warning_protocol": {
        "primary":
            "Step7H response-aware GREEN/YELLOW/RED",

        "secondary":
            "Step18 binary geofence-crossing benchmark",

        "RED":
            "actionable/selective warning",

        "YELLOW":
            "watch/uncertainty flag",

        "GREEN":
            "no warning",
    },

    "execution_order": [
        "R1 Protocol lock",
        "R2 Fair TestReal + Fresh WildBox all-baseline benchmark",
        "R3 Paired track/source-clustered bootstrap CI",
        "R4 In-domain/fresh fixed-conformal coverage",
        "R5 Geofence trivial baselines",
        "R6 Direct crossing classifier",
        "R7 Warning-protocol hierarchy cleanup",
        "R8 Ego-motion ablation",
        "R9 Canonicalization/scale/isotropic-normalization ablation",
        "R10 Ensemble ablation",
        "R11 Noise-floor proxy",
        "R12 Track/tracklet accounting",
        "R13 End-to-end detector->tracker->forecast->warning",
        "R14 Tracking metrics",
        "R15 Error propagation",
        "R16 Latency",
        "R17 Mixed-species elephant filtering",
        "R18 Meter-space WildBox",
        "R19 Final claim matrix + paper rewrite",
    ],

    "R2_models": [
        "Stationary",
        "Constant Velocity",
        "Kalman constant-velocity",
        "GRU",
        "LSTM",
        "Frozen Transformer seed42",
        "Frozen Transformer seed52",
        "Frozen Transformer seed62",
        "Frozen 3-seed Transformer ensemble",
    ],

    "R3_clustered_bootstrap": {
        "replicates":
            5000,

        "seed":
            20261029,

        "primary":
            "paired track-clustered bootstrap 95% percentile CI",

        "secondary":
            "paired source-clustered bootstrap sensitivity analysis",

        "ranking_rule":
            (
                "Do not claim superiority when the paired 95% CI for the "
                "metric difference includes zero."
            ),
    },

    "R4_fixed_q": {
        "2s_to_2s": {
            "q90":
                2.158767888280567,

            "q95":
                2.560772554862189,
        },

        "2s_to_5s": {
            "q90":
                1.975129668255982,

            "q95":
                2.950976253259077,
        },
    },

    "R5_R6_geofence_models": [
        "Distance-only current-position score",
        "Stationary future",
        "Constant Velocity extrapolation",
        "Direct crossing classifier from 20-frame history",
        "Frozen Transformer ensemble",
    ],

    "R8_R10_ablations": [
        "raw vs camera-corrected trajectory",
        "no canonicalization vs translation-only vs full canonicalization",
        "no scale vs locked path-length scale",
        "anisotropic x/W,y/H vs isotropic x/S,y/S with S=max(width,height)",
        "seed42 vs seed52 vs seed62 vs 3-seed ensemble",
    ],

    "R13_R16_end_to_end": {
        "pipeline":
            (
                "video -> frozen elephant detector -> fixed tracker -> elephant tracks -> "
                "camera correction -> forecast -> warning"
            ),

        "metrics": [
            "detection precision/recall/AP",
            "HOTA",
            "IDF1",
            "ID switches",
            "fragmentation",
            "forecast ADE/FDE degradation",
            "warning degradation",
            "latency mean/median/P95",
        ],
    },

    "R17_mixed_species_elephant_filtering": {
        "systems": {
            "A":
                "Current elephant-only detector baseline",

            "B":
                "Explicit elephant vs other-animal/human detector + elephant filter",

            "C":
                (
                    "Class-agnostic tracking first, then track-level class voting, "
                    "then keep elephant tracks only"
                ),
        },

        "preferred_primary_candidate":
            "C",

        "negative_stress_test":
            (
                "BAMBI deer+human frames may be used as elephant-negative distractor "
                "data; expected elephant count is zero."
            ),

        "metrics": [
            "elephant precision/recall/AP",
            "elephant-vs-other confusion",
            "false-elephant detections per 1000 non-elephant frames",
            "HOTA/IDF1/ID switches for elephant tracks",
            "forecast ADE/FDE",
            "warning sensitivity",
            "non-elephant false alarms",
            "distractor-induced warning changes",
        ],

        "domain_rule":
            "Report RGB and thermal detector evaluations separately.",
    },

    "global_nonclaims_until_supported": [
        "No Transformer superiority claim without clustered paired CI.",
        "No end-to-end system validation claim before detector/tracker evaluation.",
        "No field-effectiveness claim.",
        "No real-geofence claim for normalized synthetic boundaries.",
        "No elephant-specific BAMBI validation claim.",
        "No 5s BAMBI q90 calibration claim.",
    ],
}


def main():
    print(
        "\nR1-V3 — PAPER-STRENGTHENING REVISION PROTOCOL LOCK"
    )

    print(
        "NO new experiment is run. Frozen artifacts are searched in Downloads, "
        "Main Result, and Main Result 2.0 and accepted ONLY by exact SHA256.\n"
    )

    print(
        "SEARCH ROOTS:"
    )

    for root in SEARCH_ROOTS:
        print(
            "-",
            root,
            "| exists =",
            root.exists(),
        )

    integrity = verify_frozen()

    integrity_path = (
        OUT
        / "R1_V3_existing_frozen_snapshot_integrity.csv"
    )

    integrity.to_csv(
        integrity_path,
        index=False,
    )

    protocol_path = (
        OUT
        / "R1_V3_REVISION_PROTOCOL_LOCK.json"
    )

    protocol_path.write_text(
        json.dumps(
            PROTOCOL,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    protocol_hash = sha256_file(
        protocol_path
    )

    roadmap_path = (
        OUT
        / "R1_V3_revision_execution_order.csv"
    )

    pd.DataFrame(
        [
            {
                "order":
                    i,

                "revision_step":
                    step,
            }
            for i, step in enumerate(
                PROTOCOL[
                    "execution_order"
                ],
                start=1,
            )
        ]
    ).to_csv(
        roadmap_path,
        index=False,
    )

    receipt = {
        "status":
            "R1_REVISION_PROTOCOL_LOCKED_READY_FOR_R2",

        "protocol_path":
            str(
                protocol_path
            ),

        "protocol_sha256":
            protocol_hash,

        "integrity_csv":
            str(
                integrity_path
            ),

        "roadmap_csv":
            str(
                roadmap_path
            ),

        "existing_snapshot_verified_by_sha256":
            True,

        "new_results_generated":
            False,
    }

    receipt_path = (
        OUT
        / "R1_V3_REVISION_PROTOCOL_RECEIPT.json"
    )

    receipt_path.write_text(
        json.dumps(
            receipt,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nDISCOVERED + VERIFIED FROZEN ARTIFACTS:"
    )

    print(
        integrity[
            [
                "artifact",
                "selected_path",
                "match",
                "hash_matching_copies",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nPROTOCOL:"
    )
    print(
        protocol_path
    )

    print(
        "PROTOCOL SHA256:"
    )
    print(
        protocol_hash
    )

    print(
        "\nSTATUS:"
    )
    print(
        "R1_REVISION_PROTOCOL_LOCKED_READY_FOR_R2"
    )

    print(
        "\nR1-V3 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
