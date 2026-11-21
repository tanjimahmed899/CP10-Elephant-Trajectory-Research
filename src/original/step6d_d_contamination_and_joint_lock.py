from pathlib import Path
import hashlib
import json
import re
import pandas as pd

VIDEO_ID = "DJI_20240119152552_0006_V"

ROOT_MAIN = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

ROOT_DL = Path(
    r"C:\Users\HP\Downloads"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step6d_uncertainty_holdout"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

SEGMENTS = ["seg1", "seg2", "seg3"]

AUDIT_DIR = (
    ROOT_MAIN
    /
    "wildbox_zip_audit"
)

# Any occurrence here would mean this source was already used in model
# development/evaluation and should NOT be treated as untouched.
CONTAMINATION_DIR_TERMS = [
    "final_neural_trajectory_dataset",
    "unified_trajectory_10hz",
    "wildbox_trajectory_dataset",
    "wildbox_fresh_confirmatory_results",
    "transformer_results",
    "lstm_gru_results",
    "baseline_constant_velocity",
    "baseline_kalman",
    "transformer_ensemble_validation",
    "step2",
    "step3_fresh_wildbox_full_comparison",
    "step4",
    "step5",
    "step6_conformal_uncertainty",
    "step6b_scale_adaptive_conformal",
    "step6c_crossfit_conformal",
]

TEXT_EXTENSIONS = {
    ".csv", ".json", ".txt", ".md", ".py", ".yaml", ".yml"
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


def path_is_contaminating(path):
    s = str(path).lower()
    return any(
        term.lower() in s
        for term in CONTAMINATION_DIR_TERMS
    )


def scan_occurrences():
    rows = []

    for root in [ROOT_MAIN, ROOT_DL]:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            # Ignore Step6D outputs themselves.
            if OUT in p.parents:
                continue

            path_hit = (
                VIDEO_ID.lower()
                in
                str(p).lower()
            )

            content_hit = False
            hit_count = 0

            if p.suffix.lower() in TEXT_EXTENSIONS:
                try:
                    # Avoid reading very large text files into memory.
                    if p.stat().st_size <= 50 * 1024 * 1024:
                        text = p.read_text(
                            encoding="utf-8",
                            errors="ignore"
                        )

                        hit_count = text.lower().count(
                            VIDEO_ID.lower()
                        )

                        content_hit = (
                            hit_count > 0
                        )
                except Exception:
                    pass

            if path_hit or content_hit:
                rows.append({
                    "path": str(p),
                    "path_hit": path_hit,
                    "content_hit": content_hit,
                    "content_hit_count": hit_count,
                    "contaminating_used_path": path_is_contaminating(p),
                })

    return pd.DataFrame(rows)


def summarize_segment(seg):
    tracking = (
        AUDIT_DIR
        /
        f"{VIDEO_ID}__{seg}__tracking_summary.json"
    )

    cameras = (
        AUDIT_DIR
        /
        f"{VIDEO_ID}__{seg}__cameras.json"
    )

    if not tracking.exists():
        raise FileNotFoundError(
            f"Missing tracking summary: {tracking}"
        )

    if not cameras.exists():
        raise FileNotFoundError(
            f"Missing camera metadata: {cameras}"
        )

    with open(
        tracking,
        "r",
        encoding="utf-8"
    ) as f:
        obj = json.load(f)

    tracks = obj.get(
        "tracks",
        []
    )

    # Only structural/non-performance metadata.
    total_tracks = obj.get(
        "total_tracks",
        len(tracks) if isinstance(tracks, list) else None
    )

    total_frames = obj.get(
        "total_frames",
        None
    )

    total_detections = obj.get(
        "total_detections",
        None
    )

    avg_tracklet_length = obj.get(
        "avg_tracklet_length",
        None
    )

    max_tracklet_length = obj.get(
        "max_tracklet_length",
        None
    )

    # Collect field names only from first few tracks/detections.
    track_keys = set()
    detection_keys = set()

    if isinstance(tracks, list):
        for t in tracks[:10]:
            if isinstance(t, dict):
                track_keys.update(
                    map(
                        str,
                        t.keys()
                    )
                )

                for possible in [
                    "detections",
                    "frames",
                    "observations",
                    "trajectory",
                    "points"
                ]:
                    value = t.get(
                        possible
                    )

                    if isinstance(value, list):
                        for d in value[:5]:
                            if isinstance(d, dict):
                                detection_keys.update(
                                    map(
                                        str,
                                        d.keys()
                                    )
                                )

    return {
        "segment": seg,
        "tracking_path": str(tracking),
        "camera_path": str(cameras),
        "tracking_sha256": sha256_file(tracking),
        "camera_sha256": sha256_file(cameras),
        "total_tracks": total_tracks,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "avg_tracklet_length": avg_tracklet_length,
        "max_tracklet_length": max_tracklet_length,
        "track_field_names": ";".join(
            sorted(
                track_keys
            )
        ),
        "nested_observation_field_names": ";".join(
            sorted(
                detection_keys
            )
        ),
    }


def main():
    print(
        "\nSTEP 6D-D: WILDBOX CONTAMINATION AUDIT + JOINT PRE-LOCK"
    )

    occurrences = scan_occurrences()

    if len(occurrences) == 0:
        occurrences = pd.DataFrame(
            columns=[
                "path",
                "path_hit",
                "content_hit",
                "content_hit_count",
                "contaminating_used_path",
            ]
        )

    occurrences.to_csv(
        OUT
        /
        "step6d_video_id_occurrence_audit.csv",
        index=False
    )

    contaminating = occurrences[
        occurrences[
            "contaminating_used_path"
        ]
        ==
        True
    ].copy()

    segment_rows = [
        summarize_segment(
            seg
        )
        for seg in SEGMENTS
    ]

    segment_df = pd.DataFrame(
        segment_rows
    )

    segment_df.to_csv(
        OUT
        /
        "step6d_joint_holdout_segment_metadata.csv",
        index=False
    )

    # Predeclared selection rule:
    # use ALL three candidate segments together.
    # No performance result is used to choose between them.
    lock_created = (
        len(
            contaminating
        )
        ==
        0
    )

    lock = {
        "lock_stage":
            "STEP_6D_JOINT_WILDBOX_PRE_RESULT_LOCK",

        "selection_rule":
            (
                "Use ALL three structurally eligible unused segments "
                "seg1, seg2, seg3 from the single candidate video "
                f"{VIDEO_ID}; no segment-level performance selection."
            ),

        "video_id":
            VIDEO_ID,

        "segments":
            SEGMENTS,

        "model_predictions_or_forecast_errors_seen_before_lock":
            False,

        "contaminating_occurrences_in_known_used_paths":
            int(
                len(
                    contaminating
                )
            ),

        "segment_files":
            segment_rows,

        "holdout_scope_note":
            (
                "This is one video/source with three segments. "
                "It can provide a new untouched source-level uncertainty test, "
                "but it does not establish broad multi-source external generalization."
            )
    }

    if lock_created:
        with open(
            OUT
            /
            "STEP6D_JOINT_WILDBOX_PRELOCK.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                lock,
                f,
                indent=2
            )

    with open(
        OUT
        /
        "STEP6D_D_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 6D-D: CONTAMINATION AUDIT + JOINT PRE-LOCK\n"
        )

        f.write(
            "=" * 76
            +
            "\n\n"
        )

        f.write(
            f"Candidate video: {VIDEO_ID}\n"
        )

        f.write(
            "Predeclared rule: use ALL seg1/seg2/seg3 together; "
            "do not choose based on performance.\n\n"
        )

        f.write(
            f"All exact ID occurrences found: {len(occurrences)}\n"
        )

        f.write(
            "Occurrences inside known-used training/evaluation paths: "
            f"{len(contaminating)}\n\n"
        )

        f.write(
            "SEGMENT STRUCTURAL METADATA\n"
        )

        f.write(
            "---------------------------\n"
        )

        f.write(
            segment_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nCONTAMINATION OCCURRENCES\n"
        )

        f.write(
            "-------------------------\n"
        )

        if len(
            contaminating
        ):
            f.write(
                contaminating.to_string(
                    index=False
                )
            )
        else:
            f.write(
                "NONE in known-used paths.\n"
            )

        f.write(
            "\n\n"
        )

        if lock_created:
            f.write(
                "JOINT PRE-RESULT LOCK CREATED.\n"
            )

            f.write(
                "No model prediction/error was used for selection.\n"
            )
        else:
            f.write(
                "NO LOCK CREATED because contamination was detected.\n"
            )

    print("\nEXACT VIDEO-ID OCCURRENCES:")
    print(
        len(
            occurrences
        )
    )

    print(
        "\nCONTAMINATING OCCURRENCES IN KNOWN-USED PATHS:"
    )
    print(
        len(
            contaminating
        )
    )

    if len(
        contaminating
    ):
        print(
            contaminating[
                [
                    "path",
                    "path_hit",
                    "content_hit",
                    "content_hit_count",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print("NONE")

    print("\nCANDIDATE SEGMENT STRUCTURE:")
    print(
        segment_df[
            [
                "segment",
                "total_tracks",
                "total_frames",
                "total_detections",
                "avg_tracklet_length",
                "max_tracklet_length",
                "track_field_names",
                "nested_observation_field_names",
            ]
        ].to_string(
            index=False
        )
    )

    if lock_created:
        print(
            "\nJOINT PRE-RESULT HOLDOUT LOCK CREATED:"
        )

        print(
            OUT
            /
            "STEP6D_JOINT_WILDBOX_PRELOCK.json"
        )

        print(
            "\nLOCK RULE: ALL seg1 + seg2 + seg3 are included."
        )
    else:
        print(
            "\nNO HOLDOUT LOCK CREATED."
        )


if __name__ == "__main__":
    main()
