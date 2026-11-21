
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R13_ROOT = REVISION / "R13_end_to_end_pipeline"

R13D_DIR = R13_ROOT / "R13D_frozen_detector_tracker_execution"
R13D_MANIFEST = R13D_DIR / "R13D_MANIFEST.json"
BYTE_BOXES = R13D_DIR / "R13D_BYTETRACK_ALL_BOXES.csv"
BOT_BOXES = R13D_DIR / "R13D_BOTSORT_ALL_BOXES.csv"

R13FA_DIR = R13_ROOT / "R13F_A_detector_tracker_bottleneck_audit"
R13FA_MANIFEST = R13FA_DIR / "R13F_A_MANIFEST.json"

OUT = R13_ROOT / "R13F_B_tracker_forecastability_audit"
OUT.mkdir(parents=True, exist_ok=True)

# Frozen diagnostic construction rule for this NEW revision experiment.
SOURCE_FPS = 30
TARGET_FPS = 10
RESAMPLE_STEP = 3

MAX_SMALL_GAP = 3

HISTORY_10HZ = 20
FUTURE_2S_10HZ = 20
FUTURE_5S_10HZ = 50
WINDOW_STRIDE_10HZ = 10

REFERENCE_GT_FINAL = {
    "2s": {
        "windows": 1053,
        "segments": 89,
        "original_tracks": 37,
    },
    "5s": {
        "windows": 826,
        "segments": 63,
        "original_tracks": 31,
    },
}


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


def verify_upstream():
    for p in [
        R13D_MANIFEST,
        R13FA_MANIFEST,
        BYTE_BOXES,
        BOT_BOXES,
    ]:
        require(p)

    d = json.loads(
        R13D_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    a = json.loads(
        R13FA_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_d = (
        "R13D_FROZEN_DETECTOR_BYTETRACK_BOTSORT_EXECUTION_COMPLETE_RAW_OUTPUTS_LOCKED_READY_FOR_R13E_TRACKING_METRICS"
    )

    expected_a = (
        "R13F_A_DETECTOR_TRACKER_BOTTLENECK_DECOMPOSITION_COMPLETE_REVIEW_BEFORE_FORECAST_PROPAGATION"
    )

    if d.get("status") != expected_d:
        raise RuntimeError(
            f"Unexpected R13-D status: {d.get('status')}"
        )

    if a.get("status") != expected_a:
        raise RuntimeError(
            f"Unexpected R13-F-A status: {a.get('status')}"
        )

    summary = {
        x["component"]: x
        for x in d["summary"]
    }

    for comp, path in [
        ("bytetrack", BYTE_BOXES),
        ("botsort", BOT_BOXES),
    ]:
        if sha256_file(path) != summary[comp]["boxes_sha256"]:
            raise RuntimeError(
                f"Frozen tracker output hash mismatch: {path}"
            )

    return d, a


def prepare_tracker(path, tracker_name):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df["sequence"] = (
        df["sequence"]
        .astype(str)
        .str.strip()
    )

    df["frame"] = pd.to_numeric(
        df["frame"],
        errors="raise",
    ).astype(int)

    df["track_id"] = pd.to_numeric(
        df["track_id"],
        errors="coerce",
    )

    df = df.loc[
        df["track_id"].notna()
        & (df["track_id"] >= 0)
    ].copy()

    df["track_id"] = df["track_id"].astype(int)

    # One-class detector was locked as Elephant.
    if len(df):
        classes = sorted(
            df["class_name"]
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        if len(classes) > 1:
            elephant = [
                c
                for c in classes
                if "elephant" in c.lower()
            ]

            if len(elephant) != 1:
                raise RuntimeError(
                    f"{tracker_name}: ambiguous class set {classes}"
                )

            df = df.loc[
                df["class_name"].astype(str)
                ==
                elephant[0]
            ].copy()

    for c in [
        "cx",
        "cy",
        "w",
        "h",
        "confidence",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "cx",
            "cy",
            "w",
            "h",
        ]
    ).copy()

    df["tracker"] = tracker_name

    return df


def split_track_into_segments(track_df):
    t = (
        track_df
        .sort_values(
            "frame"
        )
        .drop_duplicates(
            subset=["frame"],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    if len(t) == 0:
        return []

    frames = t["frame"].to_numpy(int)

    # missing = difference - 1
    missing = np.diff(frames) - 1

    split_after = np.where(
        missing > MAX_SMALL_GAP
    )[0]

    boundaries = [0]

    for idx in split_after:
        boundaries.append(
            idx + 1
        )

    boundaries.append(
        len(t)
    )

    segments = []

    for i in range(
        len(boundaries) - 1
    ):
        seg = t.iloc[
            boundaries[i]:
            boundaries[i + 1]
        ].copy()

        if len(seg):
            segments.append(
                seg
            )

    return segments


def interpolate_small_gaps(seg):
    s = seg.sort_values(
        "frame"
    ).copy()

    start = int(
        s["frame"].min()
    )

    end = int(
        s["frame"].max()
    )

    full_frames = pd.DataFrame(
        {
            "frame":
                np.arange(
                    start,
                    end + 1,
                    dtype=int,
                )
        }
    )

    cols = [
        "frame",
        "cx",
        "cy",
        "w",
        "h",
        "confidence",
    ]

    merged = full_frames.merge(
        s[cols],
        how="left",
        on="frame",
    )

    observed = merged[
        "cx"
    ].notna()

    # Since large gaps were already split, any remaining missing span is <= 3.
    for c in [
        "cx",
        "cy",
        "w",
        "h",
        "confidence",
    ]:
        merged[c] = merged[c].interpolate(
            method="linear",
            limit_direction="both",
        )

    if merged[
        [
            "cx",
            "cy",
            "w",
            "h",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Unexpected NaN after small-gap interpolation."
        )

    merged["observed"] = observed.astype(int)

    return merged


def resample_10hz(interpolated):
    # Frozen revision rule:
    # choose every 3rd 30-Hz frame beginning at the segment's first frame.
    if len(interpolated) == 0:
        return interpolated.copy()

    start = int(
        interpolated["frame"].iloc[0]
    )

    mask = (
        (
            interpolated["frame"]
            -
            start
        )
        %
        RESAMPLE_STEP
        ==
        0
    )

    out = interpolated.loc[
        mask
    ].copy()

    out = out.reset_index(
        drop=True
    )

    return out


def count_windows(n_points, future_points):
    total_needed = (
        HISTORY_10HZ
        +
        future_points
    )

    if n_points < total_needed:
        return 0

    return (
        (
            n_points
            -
            total_needed
        )
        //
        WINDOW_STRIDE_10HZ
        +
        1
    )


def analyze_tracker(df, tracker_name):
    segment_rows = []
    per_sequence_rows = []

    segment_counter = 0

    for (
        sequence,
        track_id,
    ), group in df.groupby(
        [
            "sequence",
            "track_id",
        ],
        sort=True,
    ):
        segments = split_track_into_segments(
            group
        )

        for local_segment_id, seg in enumerate(
            segments,
            start=1,
        ):
            segment_counter += 1

            dense = interpolate_small_gaps(
                seg
            )

            sampled = resample_10hz(
                dense
            )

            n30 = int(
                len(
                    dense
                )
            )

            n10 = int(
                len(
                    sampled
                )
            )

            n_obs = int(
                sampled[
                    "observed"
                ].sum()
            )

            w2 = int(
                count_windows(
                    n10,
                    FUTURE_2S_10HZ,
                )
            )

            w5 = int(
                count_windows(
                    n10,
                    FUTURE_5S_10HZ,
                )
            )

            segment_rows.append(
                {
                    "tracker":
                        tracker_name,

                    "sequence":
                        sequence,

                    "track_id":
                        int(
                            track_id
                        ),

                    "segment_id":
                        local_segment_id,

                    "segment_key":
                        (
                            f"{sequence}|"
                            f"{int(track_id)}|"
                            f"{local_segment_id}"
                        ),

                    "start_frame":
                        int(
                            dense[
                                "frame"
                            ].min()
                        ),

                    "end_frame":
                        int(
                            dense[
                                "frame"
                            ].max()
                        ),

                    "dense_30hz_frames":
                        n30,

                    "resampled_10hz_points":
                        n10,

                    "observed_10hz_points":
                        n_obs,

                    "interpolated_10hz_points":
                        n10
                        -
                        n_obs,

                    "windows_2s_to_2s":
                        w2,

                    "windows_2s_to_5s":
                        w5,

                    "forecastable_2s":
                        w2 > 0,

                    "forecastable_5s":
                        w5 > 0,
                }
            )

    segments_df = pd.DataFrame(
        segment_rows
    )

    sequences = sorted(
        df[
            "sequence"
        ]
        .unique()
        .tolist()
    )

    for seq in sequences:
        s = segments_df.loc[
            segments_df[
                "sequence"
            ]
            ==
            seq
        ].copy()

        raw_tracks = int(
            df.loc[
                df[
                    "sequence"
                ]
                ==
                seq,
                "track_id",
            ]
            .nunique()
        )

        per_sequence_rows.append(
            {
                "tracker":
                    tracker_name,

                "sequence":
                    seq,

                "raw_tracker_ids":
                    raw_tracks,

                "segments_after_gap_rule":
                    int(
                        len(
                            s
                        )
                    ),

                "forecastable_segments_2s":
                    int(
                        s[
                            "forecastable_2s"
                        ].sum()
                    )
                    if len(
                        s
                    )
                    else 0,

                "forecastable_segments_5s":
                    int(
                        s[
                            "forecastable_5s"
                        ].sum()
                    )
                    if len(
                        s
                    )
                    else 0,

                "windows_2s_to_2s":
                    int(
                        s[
                            "windows_2s_to_2s"
                        ].sum()
                    )
                    if len(
                        s
                    )
                    else 0,

                "windows_2s_to_5s":
                    int(
                        s[
                            "windows_2s_to_5s"
                        ].sum()
                    )
                    if len(
                        s
                    )
                    else 0,
            }
        )

    per_seq_df = pd.DataFrame(
        per_sequence_rows
    )

    aggregate = {
        "tracker":
            tracker_name,

        "raw_tracker_ids_sum_across_sequences":
            int(
                df.groupby(
                    "sequence"
                )[
                    "track_id"
                ]
                .nunique()
                .sum()
            ),

        "segments_after_gap_rule":
            int(
                len(
                    segments_df
                )
            ),

        "forecastable_segments_2s":
            int(
                segments_df[
                    "forecastable_2s"
                ].sum()
            )
            if len(
                segments_df
            )
            else 0,

        "forecastable_segments_5s":
            int(
                segments_df[
                    "forecastable_5s"
                ].sum()
            )
            if len(
                segments_df
            )
            else 0,

        "windows_2s_to_2s":
            int(
                segments_df[
                    "windows_2s_to_2s"
                ].sum()
            )
            if len(
                segments_df
            )
            else 0,

        "windows_2s_to_5s":
            int(
                segments_df[
                    "windows_2s_to_5s"
                ].sum()
            )
            if len(
                segments_df
            )
            else 0,
    }

    aggregate[
        "window_coverage_vs_gt_final_2s"
    ] = (
        aggregate[
            "windows_2s_to_2s"
        ]
        /
        REFERENCE_GT_FINAL[
            "2s"
        ][
            "windows"
        ]
    )

    aggregate[
        "window_coverage_vs_gt_final_5s"
    ] = (
        aggregate[
            "windows_2s_to_5s"
        ]
        /
        REFERENCE_GT_FINAL[
            "5s"
        ][
            "windows"
        ]
    )

    return (
        segments_df,
        per_seq_df,
        aggregate,
    )


def main():
    print(
        "\nR13-F-B — TRACKER-DERIVED FORECASTABILITY AUDIT"
    )

    print(
        "AUDIT ONLY. NO forecasting model inference and NO tuning. "
        "This step asks whether the frozen tracker outputs contain enough temporally dense observations "
        "to construct the same basic 2-s history with 2-s or 5-s future forecasting tasks. "
        "Tracker tracks are split when >3 source frames are missing, <=3-frame gaps are linearly interpolated, "
        "then resampled from 30 Hz to 10 Hz by taking every third frame from each segment start. "
        "At 10 Hz, history=20 points, future=20/50 points, stride=10 points. "
        "This is a frozen NEW revision diagnostic and does not replace the historical GT-based pipeline.\n"
    )

    verify_upstream()

    byte = prepare_tracker(
        BYTE_BOXES,
        "ByteTrack",
    )

    bot = prepare_tracker(
        BOT_BOXES,
        "BoT-SORT",
    )

    outputs = []

    all_segments = []
    all_per_seq = []

    for name, df in [
        (
            "ByteTrack",
            byte,
        ),
        (
            "BoT-SORT",
            bot,
        ),
    ]:
        seg, seq, agg = analyze_tracker(
            df,
            name,
        )

        all_segments.append(
            seg
        )

        all_per_seq.append(
            seq
        )

        outputs.append(
            agg
        )

    segments_df = pd.concat(
        all_segments,
        ignore_index=True,
        sort=False,
    )

    per_seq_df = pd.concat(
        all_per_seq,
        ignore_index=True,
        sort=False,
    )

    aggregate_df = pd.DataFrame(
        outputs
    )

    conclusion_rows = []

    for _, r in aggregate_df.iterrows():
        tracker = str(
            r[
                "tracker"
            ]
        )

        w2 = int(
            r[
                "windows_2s_to_2s"
            ]
        )

        w5 = int(
            r[
                "windows_2s_to_5s"
            ]
        )

        c2 = float(
            r[
                "window_coverage_vs_gt_final_2s"
            ]
        )

        c5 = float(
            r[
                "window_coverage_vs_gt_final_5s"
            ]
        )

        if w2 == 0 and w5 == 0:
            interpretation = (
                "No tracker-derived forecasting windows; direct downstream forecast evaluation is infeasible."
            )
        elif c2 < 0.10 or c5 < 0.10:
            interpretation = (
                "Only sparse tracker-derived forecasting coverage; downstream forecasting can be exploratory only, "
                "not a representative end-to-end performance estimate."
            )
        else:
            interpretation = (
                "Non-trivial tracker-derived forecasting coverage exists; downstream propagation analysis is feasible "
                "with explicit coverage reporting."
            )

        conclusion_rows.append(
            {
                "tracker":
                    tracker,

                "2s_windows":
                    w2,

                "5s_windows":
                    w5,

                "2s_coverage_vs_gt_final":
                    c2,

                "5s_coverage_vs_gt_final":
                    c5,

                "interpretation":
                    interpretation,
            }
        )

    conclusion_df = pd.DataFrame(
        conclusion_rows
    )

    segments_path = (
        OUT
        /
        "R13F_B_TRACKER_SEGMENT_FORECASTABILITY.csv"
    )

    per_seq_path = (
        OUT
        /
        "R13F_B_PER_SEQUENCE_FORECASTABILITY.csv"
    )

    aggregate_path = (
        OUT
        /
        "R13F_B_AGGREGATE_FORECASTABILITY.csv"
    )

    conclusion_path = (
        OUT
        /
        "R13F_B_CONCLUSION.csv"
    )

    segments_df.to_csv(
        segments_path,
        index=False,
    )

    per_seq_df.to_csv(
        per_seq_path,
        index=False,
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    conclusion_df.to_csv(
        conclusion_path,
        index=False,
    )

    status = (
        "R13F_B_TRACKER_DERIVED_FORECASTABILITY_AUDIT_COMPLETE_REVIEW_COVERAGE_BEFORE_FORECAST_INFERENCE"
    )

    manifest = {
        "step":
            "R13F_B_TRACKER_DERIVED_FORECASTABILITY_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "forecast_inference":
            False,

        "tuning":
            False,

        "construction_rule": {
            "source_fps":
                SOURCE_FPS,

            "target_fps":
                TARGET_FPS,

            "max_small_gap":
                MAX_SMALL_GAP,

            "resample_step":
                RESAMPLE_STEP,

            "history_points_10hz":
                HISTORY_10HZ,

            "future_2s_points_10hz":
                FUTURE_2S_10HZ,

            "future_5s_points_10hz":
                FUTURE_5S_10HZ,

            "window_stride_10hz":
                WINDOW_STRIDE_10HZ,
        },

        "reference_historical_final_windows": {
            "2s":
                REFERENCE_GT_FINAL[
                    "2s"
                ][
                    "windows"
                ],

            "5s":
                REFERENCE_GT_FINAL[
                    "5s"
                ][
                    "windows"
                ],
        },

        "aggregate_results":
            aggregate_df.to_dict(
                orient="records"
            ),

        "outputs": {
            "segments":
                str(
                    segments_path
                ),

            "per_sequence":
                str(
                    per_seq_path
                ),

            "aggregate":
                str(
                    aggregate_path
                ),

            "conclusion":
                str(
                    conclusion_path
                ),
        },
    }

    (
        OUT
        /
        "R13F_B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "AGGREGATE FORECASTABILITY:"
    )

    print(
        aggregate_df.to_string(
            index=False
        )
    )

    print(
        "\nPER-SEQUENCE FORECASTABILITY:"
    )

    print(
        per_seq_df.to_string(
            index=False
        )
    )

    print(
        "\nCONCLUSION:"
    )

    print(
        conclusion_df.to_string(
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
        "\nR13-F-B COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
