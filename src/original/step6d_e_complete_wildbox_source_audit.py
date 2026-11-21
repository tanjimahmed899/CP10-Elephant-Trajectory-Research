from pathlib import Path
import hashlib
import json
import re
import pandas as pd

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

# These are places that represent actual model development/evaluation usage.
# Presence of a source ID here disqualifies it as an untouched Step 6D holdout.
CONTAMINATION_TERMS = [
    "final_neural_trajectory_dataset",
    "unified_trajectory_10hz",
    "wildbox_trajectory_dataset",
    "wildbox_fresh_confirmatory_results",
    "wildbox_fresh_holdout_zips",
    "baseline_constant_velocity",
    "baseline_kalman",
    "lstm_gru_results",
    "transformer_results",
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

VIDEO_ID_PATTERN = re.compile(
    r"(DJI_\d{14}_\d{4}_V)",
    flags=re.IGNORECASE
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


def is_contaminating_path(path):
    s = str(path).lower()

    return any(
        term.lower() in s
        for term in CONTAMINATION_TERMS
    )


def extract_video_id(text):
    m = VIDEO_ID_PATTERN.search(
        str(text)
    )

    if not m:
        return None

    return m.group(1)


def collect_metadata_segments():
    rows = []

    for root in [ROOT_MAIN, ROOT_DL]:
        if not root.exists():
            continue

        for p in root.rglob("*.json"):
            if OUT in p.parents:
                continue

            low = p.name.lower()

            if (
                "tracking_summary" not in low
                and "cameras" not in low
            ):
                continue

            video_id = extract_video_id(
                str(p)
            )

            if video_id is None:
                continue

            seg_match = re.search(
                r"(seg\d+)",
                str(p),
                flags=re.IGNORECASE
            )

            seg = (
                seg_match.group(1).lower()
                if seg_match
                else ""
            )

            kind = (
                "tracking_summary"
                if "tracking_summary" in low
                else "cameras"
            )

            rows.append({
                "video_id": video_id,
                "segment": seg,
                "kind": kind,
                "path": str(p),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            })

    return pd.DataFrame(
        rows
    )


def scan_id_occurrences(video_id):
    rows = []

    for root in [ROOT_MAIN, ROOT_DL]:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            if OUT in p.parents:
                continue

            path_hit = (
                video_id.lower()
                in
                str(p).lower()
            )

            content_hit = False
            content_hit_count = 0

            if p.suffix.lower() in TEXT_EXTENSIONS:
                try:
                    if p.stat().st_size <= 50 * 1024 * 1024:
                        text = p.read_text(
                            encoding="utf-8",
                            errors="ignore"
                        )

                        content_hit_count = text.lower().count(
                            video_id.lower()
                        )

                        content_hit = (
                            content_hit_count > 0
                        )
                except Exception:
                    pass

            if path_hit or content_hit:
                rows.append({
                    "video_id": video_id,
                    "path": str(p),
                    "path_hit": path_hit,
                    "content_hit": content_hit,
                    "content_hit_count": content_hit_count,
                    "contaminating_used_path": is_contaminating_path(p),
                })

    return rows


def structural_summary(tracking_path):
    row = {
        "total_tracks": None,
        "total_frames": None,
        "total_detections": None,
        "avg_tracklet_length": None,
        "max_tracklet_length": None,
        "parse_error": "",
    }

    try:
        with open(
            tracking_path,
            "r",
            encoding="utf-8"
        ) as f:
            obj = json.load(f)

        if isinstance(
            obj,
            dict
        ):
            row["total_tracks"] = obj.get(
                "total_tracks",
                None
            )

            row["total_frames"] = obj.get(
                "total_frames",
                None
            )

            row["total_detections"] = obj.get(
                "total_detections",
                None
            )

            row["avg_tracklet_length"] = obj.get(
                "avg_tracklet_length",
                None
            )

            row["max_tracklet_length"] = obj.get(
                "max_tracklet_length",
                None
            )

    except Exception as e:
        row["parse_error"] = (
            f"{type(e).__name__}: {e}"
        )

    return row


def main():
    meta = collect_metadata_segments()

    if len(
        meta
    ) == 0:
        print("No WildBox-style metadata sources found.")
        return

    meta.to_csv(
        OUT
        /
        "step6d_all_wildbox_metadata_segments.csv",
        index=False
    )

    all_ids = sorted(
        meta[
            "video_id"
        ].dropna().unique()
    )

    occurrence_rows = []
    source_rows = []
    segment_rows = []

    for video_id in all_ids:
        occurrences = scan_id_occurrences(
            video_id
        )

        occurrence_rows.extend(
            occurrences
        )

        occ_df = pd.DataFrame(
            occurrences
        )

        if len(
            occ_df
        ):
            contaminated = bool(
                occ_df[
                    "contaminating_used_path"
                ].any()
            )

            contaminating_files = int(
                occ_df[
                    "contaminating_used_path"
                ].sum()
            )
        else:
            contaminated = False
            contaminating_files = 0

        source_meta = meta[
            meta[
                "video_id"
            ]
            ==
            video_id
        ].copy()

        segments = sorted(
            set(
                source_meta[
                    "segment"
                ]
            )
        )

        paired_count = 0

        for seg in segments:
            g = source_meta[
                source_meta[
                    "segment"
                ]
                ==
                seg
            ]

            tracking = g[
                g[
                    "kind"
                ]
                ==
                "tracking_summary"
            ]

            cameras = g[
                g[
                    "kind"
                ]
                ==
                "cameras"
            ]

            paired = (
                len(
                    tracking
                )
                >
                0
                and
                len(
                    cameras
                )
                >
                0
            )

            if paired:
                paired_count += 1

            tracking_path = (
                Path(
                    tracking.iloc[0][
                        "path"
                    ]
                )
                if len(
                    tracking
                )
                else None
            )

            structure = (
                structural_summary(
                    tracking_path
                )
                if tracking_path
                else {
                    "total_tracks": None,
                    "total_frames": None,
                    "total_detections": None,
                    "avg_tracklet_length": None,
                    "max_tracklet_length": None,
                    "parse_error": "no tracking summary",
                }
            )

            segment_rows.append({
                "video_id": video_id,
                "segment": seg,
                "paired_tracking_camera": paired,
                "tracking_path": (
                    str(
                        tracking_path
                    )
                    if tracking_path
                    else ""
                ),
                **structure,
            })

        source_rows.append({
            "video_id": video_id,
            "segments_found": len(
                segments
            ),
            "paired_segments": paired_count,
            "contaminated_by_prior_use": contaminated,
            "contaminating_file_count": contaminating_files,
            "eligible_as_step6d_untouched_source": (
                paired_count > 0
                and not contaminated
            ),
        })

    source_df = pd.DataFrame(
        source_rows
    )

    segment_df = pd.DataFrame(
        segment_rows
    )

    occurrence_df = pd.DataFrame(
        occurrence_rows
    )

    source_df.to_csv(
        OUT
        /
        "step6d_wildbox_source_contamination_summary.csv",
        index=False
    )

    segment_df.to_csv(
        OUT
        /
        "step6d_wildbox_segment_structure_summary.csv",
        index=False
    )

    occurrence_df.to_csv(
        OUT
        /
        "step6d_wildbox_all_id_occurrences.csv",
        index=False
    )

    eligible = source_df[
        source_df[
            "eligible_as_step6d_untouched_source"
        ]
        ==
        True
    ].copy()

    eligible.to_csv(
        OUT
        /
        "step6d_wildbox_remaining_untouched_sources.csv",
        index=False
    )

    with open(
        OUT
        /
        "STEP6D_E_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 6D-E: COMPLETE WILDBOX SOURCE CONTAMINATION AUDIT\n"
        )

        f.write(
            "=" * 78
            +
            "\n\n"
        )

        f.write(
            "This audit uses only source IDs, file paths, "
            "and structural metadata. No prediction/error "
            "results are computed.\n\n"
        )

        f.write(
            f"Distinct WildBox video IDs found: {len(source_df)}\n"
        )

        f.write(
            "Eligible untouched sources remaining: "
            f"{len(eligible)}\n\n"
        )

        f.write(
            "SOURCE SUMMARY\n"
        )

        f.write(
            "--------------\n"
        )

        f.write(
            source_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nELIGIBLE UNTOUCHED SOURCES\n"
        )

        f.write(
            "--------------------------\n"
        )

        if len(
            eligible
        ):
            f.write(
                eligible.to_string(
                    index=False
                )
            )
        else:
            f.write(
                "NONE\n"
            )

        f.write(
            "\n\nSEGMENT STRUCTURE\n"
        )

        f.write(
            "-----------------\n"
        )

        f.write(
            segment_df.to_string(
                index=False
            )
        )

    print(
        "\nSTEP 6D-E COMPLETE WILDBOX SOURCE AUDIT"
    )

    print(
        "Outputs:",
        OUT
    )

    print(
        "\nSOURCE CONTAMINATION SUMMARY:"
    )

    print(
        source_df.to_string(
            index=False
        )
    )

    print(
        "\nELIGIBLE UNTOUCHED WILDBOX SOURCES:"
    )

    if len(
        eligible
    ):
        print(
            eligible.to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nSEGMENT STRUCTURE SUMMARY:"
    )

    print(
        segment_df[
            [
                "video_id",
                "segment",
                "paired_tracking_camera",
                "total_tracks",
                "total_frames",
                "total_detections",
                "avg_tracklet_length",
                "max_tracklet_length",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
