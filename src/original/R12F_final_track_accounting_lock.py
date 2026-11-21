
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

ROOT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R12_track_accounting"
)

V11_DIR = (
    ROOT
    / "R12E_v11_full_exact_track_accounting_reconstruction"
)

V11_MANIFEST = (
    V11_DIR
    / "R12E_V11_MANIFEST.json"
)

V11_CHAIN = (
    V11_DIR
    / "R12E_V11_FULL_TRACK_ACCOUNTING_CHAIN.csv"
)

V11_MATCH = (
    V11_DIR
    / "R12E_V11_FILTERED_VS_FROZEN_PHYSICAL_WINDOW_MATCH.csv"
)

V11_FILTER = (
    V11_DIR
    / "R12E_V11_EXACT_SOURCE_FILTER_EXPRESSION.csv"
)

OUT = (
    ROOT
    / "R12F_final_track_accounting_lock"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def require(path):
    p = Path(
        path
    )

    if not p.exists():
        raise FileNotFoundError(
            p
        )


def sha256_file(
    path,
    chunk=1024 * 1024,
):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            b = f.read(
                chunk
            )

            if not b:
                break

            h.update(
                b
            )

    return h.hexdigest()


def main():
    print(
        "\nR12-F — FINAL TRACK ACCOUNTING LOCK"
    )

    print(
        "FINAL CONSOLIDATION ONLY. NO training and NO inference. "
        "This step verifies the exact R12-E-V11 reconstruction, freezes terminology, "
        "writes the reviewer-safe manuscript wording, and officially closes R12. "
        "No original experiment file is modified.\n"
    )

    for p in [
        V11_MANIFEST,
        V11_CHAIN,
        V11_MATCH,
        V11_FILTER,
    ]:
        require(
            p
        )

    manifest = json.loads(
        V11_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_status = (
        "R12E_V11_FULL_EXACT_RECONSTRUCTION_PASS_45_ORIGINAL_37_31_TRACKS_89_63_SEGMENTS_1053_826_WINDOWS_ZERO_PHYSICAL_MISMATCH_READY_FOR_R12F_FINAL_LOCK"
    )

    if (
        manifest.get(
            "status"
        )
        !=
        expected_status
    ):
        raise RuntimeError(
            "R12-E-V11 is not in the required PASS state."
        )

    chain = pd.read_csv(
        V11_CHAIN
    )

    match = pd.read_csv(
        V11_MATCH
    )

    source_filter = pd.read_csv(
        V11_FILTER
    )

    # Hard lock the decisive values.
    checks = {
        "total_original_tracks":
            int(
                manifest[
                    "total_original_tracks"
                ]
            )
            ==
            45,

        "2s_windows":
            int(
                manifest[
                    "final_2s"
                ][
                    "windows"
                ]
            )
            ==
            1053,

        "2s_original_tracks":
            int(
                manifest[
                    "final_2s"
                ][
                    "original_tracks"
                ]
            )
            ==
            37,

        "2s_segments":
            int(
                manifest[
                    "final_2s"
                ][
                    "segments"
                ]
            )
            ==
            89,

        "2s_zero_mismatch":
            int(
                manifest[
                    "final_2s"
                ][
                    "physical_window_mismatch"
                ]
            )
            ==
            0,

        "5s_windows":
            int(
                manifest[
                    "final_5s"
                ][
                    "windows"
                ]
            )
            ==
            826,

        "5s_original_tracks":
            int(
                manifest[
                    "final_5s"
                ][
                    "original_tracks"
                ]
            )
            ==
            31,

        "5s_segments":
            int(
                manifest[
                    "final_5s"
                ][
                    "segments"
                ]
            )
            ==
            63,

        "5s_zero_mismatch":
            int(
                manifest[
                    "final_5s"
                ][
                    "physical_window_mismatch"
                ]
            )
            ==
            0,

        "exact_filter_locked":
            (
                manifest[
                    "exact_filter"
                ]
                ==
                "accepted == 1 AND quality_tier == 'core' AND sample_id >= 0"
            ),
    }

    failed = [
        key
        for key, value in checks.items()
        if not bool(
            value
        )
    ]

    if failed:
        raise RuntimeError(
            f"R12-F verification failed: {failed}"
        )

    if not bool(
        match[
            "exact_physical_window_set_match"
        ].all()
    ):
        raise RuntimeError(
            "Physical window identity mismatch detected."
        )

    if not (
        (
            match[
                "only_filtered"
            ]
            ==
            0
        ).all()
        and
        (
            match[
                "only_final"
            ]
            ==
            0
        ).all()
    ):
        raise RuntimeError(
            "Non-zero physical window mismatch detected."
        )

    terminology = pd.DataFrame(
        [
            {
                "quantity":
                    "45",

                "locked_term":
                    "original dataset-level tracks",

                "definition":
                    "unique (sequence, object_id) in BIRDSAI TestReal",

                "allowed_claim":
                    "The TestReal split contains 45 original tracks.",

                "forbidden_claim":
                    "Do not call 89 or 63 original tracks.",
            },
            {
                "quantity":
                    "37",

                "locked_term":
                    "original tracks represented in final 2-s evaluation set",

                "definition":
                    "unique (sequence, object_id) after exact source-implemented filtering",

                "allowed_claim":
                    "37 original tracks are represented in the final 2-s set.",

                "forbidden_claim":
                    "Do not call 37 total TestReal tracks.",
            },
            {
                "quantity":
                    "31",

                "locked_term":
                    "original tracks represented in final 5-s evaluation set",

                "definition":
                    "unique (sequence, object_id) after exact source-implemented filtering",

                "allowed_claim":
                    "31 original tracks are represented in the final 5-s set.",

                "forbidden_claim":
                    "Do not call 31 total TestReal tracks.",
            },
            {
                "quantity":
                    "89",

                "locked_term":
                    "forecastable segments / tracklets for 2-s task",

                "definition":
                    "unique (sequence, object_id, segment_id)",

                "allowed_claim":
                    "The final 2-s set contains 89 forecastable segments.",

                "forbidden_claim":
                    "Never report 89 as dataset-level tracks.",
            },
            {
                "quantity":
                    "63",

                "locked_term":
                    "forecastable segments / tracklets for 5-s task",

                "definition":
                    "unique (sequence, object_id, segment_id)",

                "allowed_claim":
                    "The final 5-s set contains 63 forecastable segments.",

                "forbidden_claim":
                    "Never report 63 as dataset-level tracks.",
            },
            {
                "quantity":
                    "1053",

                "locked_term":
                    "final 2-s evaluation windows",

                "definition":
                    "physical windows after source filter",

                "allowed_claim":
                    "The final 2-s evaluation set contains 1,053 windows.",

                "forbidden_claim":
                    "",
            },
            {
                "quantity":
                    "826",

                "locked_term":
                    "final 5-s evaluation windows",

                "definition":
                    "physical windows after source filter",

                "allowed_claim":
                    "The final 5-s evaluation set contains 826 windows.",

                "forbidden_claim":
                    "",
            },
        ]
    )

    claim_matrix = pd.DataFrame(
        [
            {
                "claim":
                    "BIRDSAI TestReal contains 45 original dataset-level tracks.",

                "status":
                    "SUPPORTED",

                "basis":
                    "R12C-V2 + R12E-V11 exact reconstruction",
            },
            {
                "claim":
                    "Original tracks are segmented when more than 3 consecutive frames are missing.",

                "status":
                    "SUPPORTED",

                "basis":
                    "build_elephant_trajectory_dataset.py source lock",
            },
            {
                "claim":
                    "Smaller gaps are retained within a segment and interpolated.",

                "status":
                    "SUPPORTED",

                "basis":
                    "historical trajectory-builder interpolation rule",
            },
            {
                "claim":
                    "The unified 10-Hz core rule is accepted=1 AND quality_tier='core' AND sample_id>=0.",

                "status":
                    "SUPPORTED",

                "basis":
                    "build_unified_10hz_dataset.py exact source expression",
            },
            {
                "claim":
                    "Final 2-s set: 1,053 windows, 37 original tracks, 89 forecastable segments.",

                "status":
                    "SUPPORTED",

                "basis":
                    "zero-mismatch physical-window reconstruction",
            },
            {
                "claim":
                    "Final 5-s set: 826 windows, 31 original tracks, 63 forecastable segments.",

                "status":
                    "SUPPORTED",

                "basis":
                    "zero-mismatch physical-window reconstruction",
            },
            {
                "claim":
                    "89 and 63 are original dataset-level track counts.",

                "status":
                    "REJECTED",

                "basis":
                    "They are forecastable segment/tracklet counts.",
            },
            {
                "claim":
                    "45 vs 89 reflects contradictory counting.",

                "status":
                    "REJECTED",

                "basis":
                    "45 counts original tracks; 89 counts final 2-s segments.",
            },
        ]
    )

    manuscript_text = """R12 — Final track-accounting wording

The BIRDSAI TestReal split contains 45 original dataset-level tracks, where an original track is defined as a unique (sequence, object_id) pair. Original trajectories were segmented when more than three consecutive frames were missing; smaller gaps were retained within the same segment and interpolated. Candidate windows were subsequently restricted by the source-implemented unified 10-Hz core-quality rule: accepted = 1, quality_tier = 'core', and sample_id >= 0.

After this filtering, the final 2-s evaluation set contains 1,053 windows from 37 original tracks and 89 forecastable segments/tracklets. The final 5-s evaluation set contains 826 windows from 31 original tracks and 63 forecastable segments/tracklets. Reconstruction of the exact physical window identity, defined by sequence, object_id, segment_id, observation-start frame, and future-end frame, matched the frozen final metadata with zero mismatches for both horizons.

Interpretation:
- 45 is the total number of original dataset-level TestReal tracks.
- 37 and 31 are the original tracks represented in the final 2-s and 5-s evaluation sets.
- 89 and 63 are forecastable segments/tracklets, not original track counts.
- 1,053 and 826 are the final evaluation-window counts.

Reviewer-safe short version:
"The BIRDSAI TestReal split contains 45 original tracks. After source-defined segmentation and core-quality filtering, 37 original tracks yielded 89 forecastable 2-s segments and 1,053 evaluation windows, while 31 original tracks yielded 63 forecastable 5-s segments and 826 evaluation windows."

R12 conclusion:
The previously apparent 45-versus-89 discrepancy is fully resolved as a difference in counting units rather than a data inconsistency.
"""

    terminology_path = (
        OUT
        / "R12F_TERMINOLOGY_LOCK.csv"
    )

    claim_matrix_path = (
        OUT
        / "R12F_CLAIM_MATRIX.csv"
    )

    manuscript_path = (
        OUT
        / "R12F_MANUSCRIPT_READY_WORDING.txt"
    )

    evidence_path = (
        OUT
        / "R12F_EVIDENCE_HASH_LOCK.csv"
    )

    terminology.to_csv(
        terminology_path,
        index=False,
    )

    claim_matrix.to_csv(
        claim_matrix_path,
        index=False,
    )

    manuscript_path.write_text(
        manuscript_text,
        encoding="utf-8",
    )

    evidence = pd.DataFrame(
        [
            {
                "artifact":
                    "R12E_V11_MANIFEST",

                "path":
                    str(
                        V11_MANIFEST
                    ),

                "sha256":
                    sha256_file(
                        V11_MANIFEST
                    ),
            },
            {
                "artifact":
                    "R12E_V11_FULL_TRACK_ACCOUNTING_CHAIN",

                "path":
                    str(
                        V11_CHAIN
                    ),

                "sha256":
                    sha256_file(
                        V11_CHAIN
                    ),
            },
            {
                "artifact":
                    "R12E_V11_PHYSICAL_WINDOW_MATCH",

                "path":
                    str(
                        V11_MATCH
                    ),

                "sha256":
                    sha256_file(
                        V11_MATCH
                    ),
            },
            {
                "artifact":
                    "R12E_V11_SOURCE_FILTER",

                "path":
                    str(
                        V11_FILTER
                    ),

                "sha256":
                    sha256_file(
                        V11_FILTER
                    ),
            },
        ]
    )

    evidence.to_csv(
        evidence_path,
        index=False,
    )

    status = (
        "R12_TRACK_ACCOUNTING_FULLY_RESOLVED_AND_LOCKED_COMPLETE_READY_FOR_R13_END_TO_END_PIPELINE"
    )

    final_manifest = {
        "step":
            "R12F_FINAL_TRACK_ACCOUNTING_LOCK",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "original_files_modified":
            False,

        "locked_values": {
            "original_testreal_tracks":
                45,

            "final_2s_original_tracks":
                37,

            "final_2s_segments":
                89,

            "final_2s_windows":
                1053,

            "final_5s_original_tracks":
                31,

            "final_5s_segments":
                63,

            "final_5s_windows":
                826,

            "physical_window_mismatch_2s":
                0,

            "physical_window_mismatch_5s":
                0,
        },

        "locked_filter":
            "accepted == 1 AND quality_tier == 'core' AND sample_id >= 0",

        "locked_terminology": {
            "45":
                "original dataset-level tracks",

            "37_31":
                "original tracks represented in final evaluation sets",

            "89_63":
                "forecastable segments / tracklets",

            "1053_826":
                "final evaluation windows",
        },

        "outputs": {
            "terminology_lock":
                str(
                    terminology_path
                ),

            "claim_matrix":
                str(
                    claim_matrix_path
                ),

            "manuscript_ready_wording":
                str(
                    manuscript_path
                ),

            "evidence_hash_lock":
                str(
                    evidence_path
                ),
        },
    }

    manifest_path = (
        OUT
        / "R12F_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            final_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "LOCKED TERMINOLOGY:"
    )

    print(
        terminology.to_string(
            index=False
        )
    )

    print(
        "\nCLAIM MATRIX:"
    )

    print(
        claim_matrix.to_string(
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
        "\nR12-F COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
