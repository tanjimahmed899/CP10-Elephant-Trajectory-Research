
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R13_ROOT = REVISION / "R13_end_to_end_pipeline"

R13D_DIR = R13_ROOT / "R13D_frozen_detector_tracker_execution"
R13D_MANIFEST = R13D_DIR / "R13D_MANIFEST.json"
SEQ_MANIFEST_DIR = R13D_DIR / "sequence_manifests"

R13FC_DIR = R13_ROOT / "R13F_C_end_to_end_feasibility_and_claim_lock"
R13FC_MANIFEST = R13FC_DIR / "R13F_C_MANIFEST.json"

DET_FRAMES = R13D_DIR / "R13D_DETECTIONS_ALL_FRAMES.csv"
BYTE_FRAMES = R13D_DIR / "R13D_BYTETRACK_ALL_FRAMES.csv"
BOT_FRAMES = R13D_DIR / "R13D_BOTSORT_ALL_FRAMES.csv"

OUT = (
    REVISION
    / "R16_latency"
    / "R16A_computational_latency_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_FPS = 30.0
TARGET_FPS = 10.0

SOURCE_FRAME_BUDGET_MS = 1000.0 / SOURCE_FPS
TARGET_FRAME_BUDGET_MS = 1000.0 / TARGET_FPS

WARNING_HORIZONS_S = [2.0, 5.0]


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
        R13FC_MANIFEST,
        DET_FRAMES,
        BYTE_FRAMES,
        BOT_FRAMES,
        SEQ_MANIFEST_DIR,
    ]:
        require(p)

    d = json.loads(
        R13D_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    c = json.loads(
        R13FC_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_d = (
        "R13D_FROZEN_DETECTOR_BYTETRACK_BOTSORT_EXECUTION_COMPLETE_RAW_OUTPUTS_LOCKED_READY_FOR_R13E_TRACKING_METRICS"
    )

    expected_c = (
        "R13_END_TO_END_FEASIBILITY_FULLY_ASSESSED_DETECTOR_BOTTLENECK_LOCKED_REPRESENTATIVE_FORECAST_PROPAGATION_NOT_SUPPORTED_READY_FOR_NEXT_STRENGTHENING_STEP"
    )

    if d.get("status") != expected_d:
        raise RuntimeError(
            f"Unexpected R13-D status: {d.get('status')}"
        )

    if c.get("status") != expected_c:
        raise RuntimeError(
            f"Unexpected R13-F-C status: {c.get('status')}"
        )

    summary = {
        x["component"]: x
        for x in d["summary"]
    }

    for comp, path in [
        ("detections", DET_FRAMES),
        ("bytetrack", BYTE_FRAMES),
        ("botsort", BOT_FRAMES),
    ]:
        actual = sha256_file(path)
        expected = summary[comp]["frames_sha256"]

        if actual != expected:
            raise RuntimeError(
                f"Frozen R13-D frame output hash mismatch: {path}"
            )

    return d, c


def load_sequence_manifests(
    sequences,
    frozen_hashes,
):
    rows = []

    component_labels = {
        "detections": "Detector-only",
        "bytetrack": "Detector+ByteTrack",
        "botsort": "Detector+BoT-SORT",
    }

    for sequence in sequences:
        for component in [
            "detections",
            "bytetrack",
            "botsort",
        ]:
            path = (
                SEQ_MANIFEST_DIR
                / f"{sequence}_{component}_COMPLETE.json"
            )

            require(path)

            m = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            if m.get("status") != "COMPLETE":
                raise RuntimeError(
                    f"Incomplete sequence manifest: {path}"
                )

            if m.get("frozen_hashes") != frozen_hashes:
                raise RuntimeError(
                    f"Frozen hash mismatch in sequence manifest: {path}"
                )

            frames = int(
                m["frames_processed"]
            )

            elapsed = float(
                m["elapsed_seconds"]
            )

            if frames <= 0 or elapsed <= 0:
                raise RuntimeError(
                    f"Invalid timing values in {path}"
                )

            avg_ms = (
                elapsed
                * 1000.0
                / frames
            )

            fps = (
                frames
                / elapsed
            )

            rows.append(
                {
                    "component":
                        component,

                    "component_label":
                        component_labels[
                            component
                        ],

                    "sequence":
                        sequence,

                    "frames":
                        frames,

                    "elapsed_seconds":
                        elapsed,

                    "avg_wall_ms_per_frame":
                        avg_ms,

                    "effective_fps":
                        fps,

                    "meets_30hz_budget":
                        avg_ms
                        <=
                        SOURCE_FRAME_BUDGET_MS,

                    "meets_10hz_budget":
                        avg_ms
                        <=
                        TARGET_FRAME_BUDGET_MS,

                    "realtime_factor_vs_30hz":
                        avg_ms
                        /
                        SOURCE_FRAME_BUDGET_MS,

                    "realtime_factor_vs_10hz":
                        avg_ms
                        /
                        TARGET_FRAME_BUDGET_MS,

                    "box_rows":
                        int(
                            m.get(
                                "box_rows",
                                0,
                            )
                        ),

                    "assigned_track_rows":
                        int(
                            m.get(
                                "assigned_track_rows",
                                0,
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def aggregate_wallclock(seq_df):
    rows = []

    for component, g in seq_df.groupby(
        "component",
        sort=False,
    ):
        frames = int(
            g["frames"].sum()
        )

        elapsed = float(
            g[
                "elapsed_seconds"
            ].sum()
        )

        weighted_ms = (
            elapsed
            * 1000.0
            / frames
        )

        fps = (
            frames
            / elapsed
        )

        rows.append(
            {
                "component":
                    component,

                "component_label":
                    str(
                        g[
                            "component_label"
                        ].iloc[
                            0
                        ]
                    ),

                "frames":
                    frames,

                "total_elapsed_seconds":
                    elapsed,

                "weighted_wall_ms_per_frame":
                    weighted_ms,

                "effective_fps":
                    fps,

                "sequence_avg_ms_min":
                    float(
                        g[
                            "avg_wall_ms_per_frame"
                        ].min()
                    ),

                "sequence_avg_ms_median":
                    float(
                        g[
                            "avg_wall_ms_per_frame"
                        ].median()
                    ),

                "sequence_avg_ms_max":
                    float(
                        g[
                            "avg_wall_ms_per_frame"
                        ].max()
                    ),

                "meets_30hz_budget":
                    weighted_ms
                    <=
                    SOURCE_FRAME_BUDGET_MS,

                "meets_10hz_budget":
                    weighted_ms
                    <=
                    TARGET_FRAME_BUDGET_MS,

                "realtime_factor_vs_30hz":
                    weighted_ms
                    /
                    SOURCE_FRAME_BUDGET_MS,

                "realtime_factor_vs_10hz":
                    weighted_ms
                    /
                    TARGET_FRAME_BUDGET_MS,
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_reported_model_times(
    path,
    component,
):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    numeric_cols = [
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df[
        "reported_model_stage_ms"
    ] = (
        df[
            numeric_cols
        ]
        .sum(
            axis=1,
            min_count=1,
        )
    )

    rows = []

    for metric in (
        numeric_cols
        +
        [
            "reported_model_stage_ms"
        ]
    ):
        s = (
            df[
                metric
            ]
            .dropna()
            .astype(
                float
            )
        )

        if len(s) == 0:
            continue

        rows.append(
            {
                "component":
                    component,

                "metric":
                    metric,

                "n_frames":
                    int(
                        len(
                            s
                        )
                    ),

                "mean_ms":
                    float(
                        s.mean()
                    ),

                "median_ms":
                    float(
                        s.median()
                    ),

                "p95_ms":
                    float(
                        s.quantile(
                            0.95
                        )
                    ),

                "max_ms":
                    float(
                        s.max()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_horizon_context(
    aggregate_df,
):
    rows = []

    for _, r in aggregate_df.iterrows():
        for horizon in WARNING_HORIZONS_S:
            ms = float(
                r[
                    "weighted_wall_ms_per_frame"
                ]
            )

            horizon_ms = (
                horizon
                * 1000.0
            )

            rows.append(
                {
                    "component":
                        r[
                            "component"
                        ],

                    "component_label":
                        r[
                            "component_label"
                        ],

                    "warning_horizon_seconds":
                        horizon,

                    "weighted_wall_ms_per_frame":
                        ms,

                    "compute_fraction_of_horizon":
                        ms
                        /
                        horizon_ms,

                    "remaining_horizon_after_one_frame_compute_ms":
                        horizon_ms
                        -
                        ms,

                    "interpretation":
                        (
                            "Computational processing is below the warning horizon, "
                            "but this does NOT establish adequate operational or human response time."
                        )
                        if ms < horizon_ms
                        else
                        (
                            "Per-frame computational processing exceeds the warning horizon."
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR16-A — COMPUTATIONAL LATENCY AUDIT"
    )

    print(
        "NO NEW INFERENCE. NO RETRAINING. NO TUNING. "
        "Uses frozen R13-D per-sequence wall-clock manifests and Ultralytics-reported frame timings. "
        "Detector+ByteTrack and Detector+BoT-SORT timings are treated as complete detector+tracker calls because "
        "model.track() performs detection and tracking together. Wall-clock timing includes image loading, Python overhead, "
        "model inference and tracker execution inside the per-frame loop, but excludes model construction and final CSV writing. "
        "Ultralytics 'speed' fields are reported separately because they may not include all tracker/IO overhead.\n"
    )

    d, c = verify_upstream()

    sequences = list(
        d[
            "sequences"
        ]
    )

    frozen_hashes = d[
        "frozen_hashes"
    ]

    seq_df = load_sequence_manifests(
        sequences,
        frozen_hashes,
    )

    agg_df = aggregate_wallclock(
        seq_df
    )

    model_time_df = pd.concat(
        [
            summarize_reported_model_times(
                DET_FRAMES,
                "detections",
            ),

            summarize_reported_model_times(
                BYTE_FRAMES,
                "bytetrack",
            ),

            summarize_reported_model_times(
                BOT_FRAMES,
                "botsort",
            ),
        ],
        ignore_index=True,
    )

    horizon_df = build_horizon_context(
        agg_df
    )

    byte = agg_df.loc[
        agg_df[
            "component"
        ]
        ==
        "bytetrack"
    ].iloc[
        0
    ]

    bot = agg_df.loc[
        agg_df[
            "component"
        ]
        ==
        "botsort"
    ].iloc[
        0
    ]

    claim_matrix = pd.DataFrame(
        [
            {
                "claim":
                    "The frozen CPU detector+ByteTrack pipeline meets 30 Hz real-time processing.",
                "status":
                    (
                        "SUPPORTED"
                        if bool(
                            byte[
                                "meets_30hz_budget"
                            ]
                        )
                        else
                        "REJECTED"
                    ),
                "basis":
                    (
                        f"weighted wall-clock={float(byte['weighted_wall_ms_per_frame']):.3f} ms/frame, "
                        f"effective FPS={float(byte['effective_fps']):.3f}; 30-Hz budget={SOURCE_FRAME_BUDGET_MS:.3f} ms."
                    ),
            },
            {
                "claim":
                    "The frozen CPU detector+ByteTrack pipeline meets 10 Hz processing.",
                "status":
                    (
                        "SUPPORTED"
                        if bool(
                            byte[
                                "meets_10hz_budget"
                            ]
                        )
                        else
                        "REJECTED"
                    ),
                "basis":
                    (
                        f"weighted wall-clock={float(byte['weighted_wall_ms_per_frame']):.3f} ms/frame; "
                        f"10-Hz budget={TARGET_FRAME_BUDGET_MS:.3f} ms."
                    ),
            },
            {
                "claim":
                    "The frozen CPU detector+BoT-SORT pipeline meets 30 Hz real-time processing.",
                "status":
                    (
                        "SUPPORTED"
                        if bool(
                            bot[
                                "meets_30hz_budget"
                            ]
                        )
                        else
                        "REJECTED"
                    ),
                "basis":
                    (
                        f"weighted wall-clock={float(bot['weighted_wall_ms_per_frame']):.3f} ms/frame, "
                        f"effective FPS={float(bot['effective_fps']):.3f}; 30-Hz budget={SOURCE_FRAME_BUDGET_MS:.3f} ms."
                    ),
            },
            {
                "claim":
                    "The frozen CPU detector+BoT-SORT pipeline meets 10 Hz processing.",
                "status":
                    (
                        "SUPPORTED"
                        if bool(
                            bot[
                                "meets_10hz_budget"
                            ]
                        )
                        else
                        "REJECTED"
                    ),
                "basis":
                    (
                        f"weighted wall-clock={float(bot['weighted_wall_ms_per_frame']):.3f} ms/frame; "
                        f"10-Hz budget={TARGET_FRAME_BUDGET_MS:.3f} ms."
                    ),
            },
            {
                "claim":
                    "Computational latency alone proves that a 2-s or 5-s warning horizon is operationally sufficient.",
                "status":
                    "REJECTED",
                "basis":
                    "Compute time is only one component of total response latency; communications, drone dispatch/intervention, and human response are not measured here.",
            },
            {
                "claim":
                    "This timing experiment measures GPU deployment performance.",
                "status":
                    "REJECTED",
                "basis":
                    "R13-D was executed on CPU.",
            },
        ]
    )

    seq_path = (
        OUT
        /
        "R16A_PER_SEQUENCE_WALLCLOCK_LATENCY.csv"
    )

    agg_path = (
        OUT
        /
        "R16A_AGGREGATE_WALLCLOCK_LATENCY.csv"
    )

    model_path = (
        OUT
        /
        "R16A_ULTRALYTICS_REPORTED_MODEL_TIMES.csv"
    )

    horizon_path = (
        OUT
        /
        "R16A_WARNING_HORIZON_COMPUTE_CONTEXT.csv"
    )

    claim_path = (
        OUT
        /
        "R16A_LATENCY_CLAIM_MATRIX.csv"
    )

    seq_df.to_csv(
        seq_path,
        index=False,
    )

    agg_df.to_csv(
        agg_path,
        index=False,
    )

    model_time_df.to_csv(
        model_path,
        index=False,
    )

    horizon_df.to_csv(
        horizon_path,
        index=False,
    )

    claim_matrix.to_csv(
        claim_path,
        index=False,
    )

    manuscript = f"""R16-A computational latency — locked interpretation

The frozen end-to-end detector/tracker stress test was executed on CPU. Sequence-level wall-clock timing was recovered from the frozen R13-D manifests. These timings represent the per-frame loop and include image loading, Python overhead, detector inference, and tracker execution for model.track() runs, while excluding model construction and final CSV serialization.

Detector+ByteTrack:
- weighted wall-clock latency: {float(byte['weighted_wall_ms_per_frame']):.3f} ms/frame
- effective throughput: {float(byte['effective_fps']):.3f} FPS
- 30-Hz budget met: {bool(byte['meets_30hz_budget'])}
- 10-Hz budget met: {bool(byte['meets_10hz_budget'])}

Detector+BoT-SORT:
- weighted wall-clock latency: {float(bot['weighted_wall_ms_per_frame']):.3f} ms/frame
- effective throughput: {float(bot['effective_fps']):.3f} FPS
- 30-Hz budget met: {bool(bot['meets_30hz_budget'])}
- 10-Hz budget met: {bool(bot['meets_10hz_budget'])}

Reviewer-safe interpretation:
These measurements quantify computational throughput only. They must not be interpreted as evidence that a 2-s or 5-s warning horizon provides adequate operational response time. End-to-end field response would also include communication delay, decision logic, drone availability/dispatch, intervention latency, and human response time, none of which were measured in this experiment.
"""

    manuscript_path = (
        OUT
        /
        "R16A_MANUSCRIPT_READY_LATENCY_INTERPRETATION.txt"
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8",
    )

    status = (
        "R16A_COMPUTATIONAL_LATENCY_AUDIT_COMPLETE_CPU_TIMING_LOCKED_READY_FOR_R16B_OPERATIONAL_RESPONSE_LATENCY_CONTEXT"
    )

    manifest = {
        "step":
            "R16A_COMPUTATIONAL_LATENCY_AUDIT",

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

        "device":
            d.get(
                "device_name",
                d.get(
                    "device",
                    "CPU",
                ),
            ),

        "source_fps":
            SOURCE_FPS,

        "target_fps":
            TARGET_FPS,

        "source_frame_budget_ms":
            SOURCE_FRAME_BUDGET_MS,

        "target_frame_budget_ms":
            TARGET_FRAME_BUDGET_MS,

        "byte_track_weighted_ms_per_frame":
            float(
                byte[
                    "weighted_wall_ms_per_frame"
                ]
            ),

        "byte_track_effective_fps":
            float(
                byte[
                    "effective_fps"
                ]
            ),

        "botsort_weighted_ms_per_frame":
            float(
                bot[
                    "weighted_wall_ms_per_frame"
                ]
            ),

        "botsort_effective_fps":
            float(
                bot[
                    "effective_fps"
                ]
            ),

        "operational_response_time_claim_allowed":
            False,

        "outputs": {
            "per_sequence_wallclock":
                str(
                    seq_path
                ),

            "aggregate_wallclock":
                str(
                    agg_path
                ),

            "ultralytics_reported_times":
                str(
                    model_path
                ),

            "warning_horizon_compute_context":
                str(
                    horizon_path
                ),

            "claim_matrix":
                str(
                    claim_path
                ),

            "manuscript_interpretation":
                str(
                    manuscript_path
                ),
        },
    }

    (
        OUT
        /
        "R16A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "AGGREGATE WALL-CLOCK LATENCY:"
    )

    print(
        agg_df.to_string(
            index=False
        )
    )

    print(
        "\nULTRALYTICS REPORTED MODEL-STAGE TIMES:"
    )

    print(
        model_time_df.to_string(
            index=False
        )
    )

    print(
        "\nWARNING-HORIZON COMPUTE CONTEXT:"
    )

    print(
        horizon_df.to_string(
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
        "\nR16-A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
