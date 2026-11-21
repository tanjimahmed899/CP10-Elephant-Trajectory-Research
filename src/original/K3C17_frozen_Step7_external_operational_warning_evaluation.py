from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

PRELOCK = (
    BASE
    / "k3_operational_warning"
    / "step7_external_prelock"
    / "STEP6D_K3C10_BAMBI_STEP7_PRELOCK.json"
)

EXPECTED_PRELOCK_SHA256 = (
    "8cc687c3f91ecbfb2a3c0f491427d91ccfdca959659a52e0836a98889c890bd1"
)

PRED_DIR = (
    BASE
    / "k3_operational_warning"
    / "secondary_frozen_inference_v2"
)

PRED_MANIFEST = (
    PRED_DIR
    / "STEP6D_K3C15_SECONDARY_FROZEN_INFERENCE_LOCK.json"
)

EXPECTED_PRED_MANIFEST_SHA256 = (
    "06a863cf9d86d1cf9faa3416d7fbfe41f0eadcb3d9ffba25845e7cf801a65c33"
)

K3C16_DIR = (
    BASE
    / "k3_operational_warning"
    / "secondary_fixed_conformal_audit_v3"
)

K3C16_SUMMARY = (
    K3C16_DIR
    / "STEP6D_K3C16_SUMMARY.json"
)

K3C11_DIR = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_reconstruction_v1"
)

K3C13_DIR = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_canonical_lock_v1"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "frozen_step7_external_evaluation_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

TASK_CONFIG = {
    "2s_to_2s": {
        "response_thresholds_sec":
            [
                0.5,
                1.0,
                1.5,
                2.0,
            ],
    },

    "2s_to_5s": {
        "response_thresholds_sec":
            [
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
    },
}

BOUNDARIES = [
    0.25,
    0.50,
    0.75,
]

AXES = [
    ("x", 0),
    ("y", 1),
]

SAMPLING_HZ = 10.0
PERSISTENCE_WINDOWS = 2
PRIMARY_ALPHA = 0.10


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
            block = f.read(
                chunk
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def verify_frozen_inputs():
    require(
        PRELOCK
    )

    prelock_hash = sha256_file(
        PRELOCK
    )

    if prelock_hash != EXPECTED_PRELOCK_SHA256:
        raise RuntimeError(
            "K3C10 prelock hash mismatch. "
            f"Expected {EXPECTED_PRELOCK_SHA256}, got {prelock_hash}"
        )

    prelock = json.loads(
        PRELOCK.read_text(
            encoding="utf-8"
        )
    )

    fixed = prelock.get(
        "fixed_rule",
        {}
    )

    required_rule_checks = {
        "primary_conformal_level":
            0.90,

        "persistence_windows":
            2,

        "coordinate_system":
            "camera-corrected normalized image-plane coordinates",
    }

    for key, expected in required_rule_checks.items():
        actual = fixed.get(
            key
        )

        if actual != expected:
            raise RuntimeError(
                f"K3C10 frozen rule mismatch for {key}: "
                f"expected {expected!r}, got {actual!r}"
            )

    if fixed.get(
        "step18_virtual_boundaries"
    ) != BOUNDARIES:
        raise RuntimeError(
            "K3C10 Step18 boundary lock does not match expected [0.25,0.5,0.75]."
        )

    if fixed.get(
        "step18_axes"
    ) != [
        "x",
        "y",
    ]:
        raise RuntimeError(
            "K3C10 Step18 axis lock does not match x/y."
        )

    require(
        PRED_MANIFEST
    )

    pred_hash = sha256_file(
        PRED_MANIFEST
    )

    if pred_hash != EXPECTED_PRED_MANIFEST_SHA256:
        raise RuntimeError(
            "K3C15 prediction manifest hash mismatch. "
            f"Expected {EXPECTED_PRED_MANIFEST_SHA256}, got {pred_hash}"
        )

    pred_manifest = json.loads(
        PRED_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        pred_manifest.get(
            "status"
        )
        !=
        "SECONDARY_FROZEN_INFERENCE_COMPLETE_PREDICTIONS_LOCKED_NO_STEP7_YET"
    ):
        raise RuntimeError(
            "K3C15 manifest status is not the expected frozen pre-Step7 state."
        )

    require(
        K3C16_SUMMARY
    )

    k3c16_hash = sha256_file(
        K3C16_SUMMARY
    )

    k3c16 = json.loads(
        K3C16_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    if (
        k3c16.get(
            "status"
        )
        !=
        "FIXED_CONFORMAL_TRANSFER_AUDIT_COMPLETE_NO_RECALIBRATION_READY_FOR_FROZEN_STEP7_EVALUATION"
    ):
        raise RuntimeError(
            f"K3C16 status unexpected: {k3c16.get('status')}"
        )

    if not bool(
        k3c16.get(
            "no_bambi_recalibration"
        )
    ):
        raise RuntimeError(
            "K3C16 does not prove no BAMBI recalibration."
        )

    fixed_q_records = k3c16.get(
        "fixed_q_values",
        []
    )

    q90 = {}

    for item in fixed_q_records:
        try:
            task = str(
                item[
                    "task"
                ]
            )

            alpha = float(
                item[
                    "alpha"
                ]
            )

            q = float(
                item[
                    "q"
                ]
            )

        except Exception:
            continue

        if abs(
            alpha
            -
            PRIMARY_ALPHA
        ) < 1e-12:
            q90[
                task
            ] = q

    missing_tasks = (
        set(
            TASK_CONFIG.keys()
        )
        -
        set(
            q90.keys()
        )
    )

    if missing_tasks:
        raise RuntimeError(
            f"K3C16 summary lacks primary q90 for tasks: {sorted(missing_tasks)}"
        )

    return {
        "prelock_hash":
            prelock_hash,

        "pred_manifest_hash":
            pred_hash,

        "k3c16_summary_hash":
            k3c16_hash,

        "q90":
            q90,
    }


def load_task_data(
    task,
):
    pred_npz = (
        PRED_DIR
        /
        f"{task}_BAMBI_Flight6_secondary_frozen_ensemble_predictions_LOCKED.npz"
    )

    metadata_csv = (
        PRED_DIR
        /
        f"{task}_BAMBI_Flight6_secondary_frozen_inference_metadata_LOCKED.csv"
    )

    history_npz = (
        K3C11_DIR
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_windows.npz"
    )

    canonical_npz = (
        K3C13_DIR
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_canonical_LOCKED.npz"
    )

    for p in [
        pred_npz,
        metadata_csv,
        history_npz,
        canonical_npz,
    ]:
        require(
            p
        )

    pred = np.load(
        pred_npz,
        allow_pickle=False,
    )

    history_arrays = np.load(
        history_npz,
        allow_pickle=False,
    )

    canonical = np.load(
        canonical_npz,
        allow_pickle=False,
    )

    metadata = pd.read_csv(
        metadata_csv
    )

    prediction = np.asarray(
        pred[
            "prediction_anchor_norm"
        ],
        dtype=np.float64,
    )

    truth = np.asarray(
        pred[
            "ground_truth_anchor_norm"
        ],
        dtype=np.float64,
    )

    history = np.asarray(
        history_arrays[
            "history_anchor_norm"
        ],
        dtype=np.float64,
    )

    scale = np.asarray(
        canonical[
            "scale"
        ],
        dtype=np.float64,
    )

    n = prediction.shape[
        0
    ]

    if truth.shape != prediction.shape:
        raise RuntimeError(
            f"{task}: prediction/truth shape mismatch."
        )

    if history.shape != (
        n,
        20,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected history shape {history.shape}"
        )

    if scale.shape != (
        n,
    ):
        raise RuntimeError(
            f"{task}: scale shape mismatch {scale.shape}"
        )

    if len(
        metadata
    ) != n:
        raise RuntimeError(
            f"{task}: metadata rows {len(metadata)} != samples {n}"
        )

    sample_index = pd.to_numeric(
        metadata[
            "sample_index"
        ],
        errors="raise",
    ).astype(
        int
    ).to_numpy()

    if not np.array_equal(
        sample_index,
        np.arange(
            n,
            dtype=int,
        ),
    ):
        raise RuntimeError(
            f"{task}: metadata sample_index does not match NPZ order."
        )

    if not all(
        np.isfinite(
            arr
        ).all()
        for arr in [
            prediction,
            truth,
            history,
            scale,
        ]
    ):
        raise RuntimeError(
            f"{task}: nonfinite values in frozen Step7 inputs."
        )

    return {
        "prediction":
            prediction,

        "truth":
            truth,

        "history":
            history,

        "scale":
            scale,

        "metadata":
            metadata,

        "pred_npz":
            pred_npz,

        "metadata_csv":
            metadata_csv,

        "history_npz":
            history_npz,

        "canonical_npz":
            canonical_npz,
    }


def first_crossing_tte(
    signed_future,
):
    hits = np.flatnonzero(
        signed_future
        >=
        0.0
    )

    if len(
        hits
    ) == 0:
        return np.nan

    return float(
        (
            int(
                hits[
                    0
                ]
            )
            +
            1
        )
        /
        SAMPLING_HZ
    )


def build_base_geofence_instances(
    task,
    data,
    q90,
):
    prediction = data[
        "prediction"
    ]

    truth = data[
        "truth"
    ]

    history = data[
        "history"
    ]

    scale = data[
        "scale"
    ]

    metadata = data[
        "metadata"
    ]

    rows = []

    for sample_index in range(
        prediction.shape[
            0
        ]
    ):
        current = history[
            sample_index,
            -1,
            :
        ]

        pred_future = prediction[
            sample_index
        ]

        true_future = truth[
            sample_index
        ]

        absolute_conformal_radius = float(
            q90
            *
            scale[
                sample_index
            ]
        )

        meta = metadata.iloc[
            sample_index
        ]

        for axis_name, axis_idx in AXES:
            current_value = float(
                current[
                    axis_idx
                ]
            )

            for boundary in BOUNDARIES:
                # Step18: one directional instance except exact tie.
                if current_value == float(
                    boundary
                ):
                    continue

                if current_value < float(
                    boundary
                ):
                    direction = "high"
                    sign = 1.0
                else:
                    direction = "low"
                    sign = -1.0

                true_signed = (
                    sign
                    *
                    (
                        true_future[
                            :,
                            axis_idx
                        ]
                        -
                        float(
                            boundary
                        )
                    )
                )

                pred_signed = (
                    sign
                    *
                    (
                        pred_future[
                            :,
                            axis_idx
                        ]
                        -
                        float(
                            boundary
                        )
                    )
                )

                true_score = float(
                    np.max(
                        true_signed
                    )
                )

                pred_score = float(
                    np.max(
                        pred_signed
                    )
                )

                true_cross = bool(
                    true_score
                    >=
                    0.0
                )

                pred_cross = bool(
                    pred_score
                    >=
                    0.0
                )

                true_tte = (
                    first_crossing_tte(
                        true_signed
                    )
                    if true_cross
                    else
                    np.nan
                )

                pred_tte = (
                    first_crossing_tte(
                        pred_signed
                    )
                    if pred_cross
                    else
                    np.nan
                )

                uncertainty_risk_score = float(
                    pred_score
                    +
                    absolute_conformal_radius
                )

                uncertainty_intersects = bool(
                    uncertainty_risk_score
                    >=
                    0.0
                )

                row = {
                    "task":
                        task,

                    "sample_index":
                        int(
                            sample_index
                        ),

                    "track_id":
                        int(
                            meta[
                                "track_id"
                            ]
                        ),

                    "window_start_time_sec":
                        float(
                            meta[
                                "window_start_time_sec"
                            ]
                        ),

                    "axis":
                        axis_name,

                    "axis_index":
                        int(
                            axis_idx
                        ),

                    "boundary":
                        float(
                            boundary
                        ),

                    "direction":
                        direction,

                    "sign":
                        float(
                            sign
                        ),

                    "current_position":
                        current_value,

                    "distance_to_boundary":
                        float(
                            abs(
                                current_value
                                -
                                float(
                                    boundary
                                )
                            )
                        ),

                    "true_oriented_risk_score":
                        true_score,

                    "oriented_risk_score":
                        pred_score,

                    "true_crossing":
                        true_cross,

                    "true_tte_sec":
                        true_tte,

                    "predicted_crossing":
                        pred_cross,

                    "predicted_tte_sec":
                        pred_tte,

                    "q90_fixed_step6c":
                        float(
                            q90
                        ),

                    "canonical_scale":
                        float(
                            scale[
                                sample_index
                            ]
                        ),

                    "absolute_conformal_radius":
                        absolute_conformal_radius,

                    "uncertainty_risk_score":
                        uncertainty_risk_score,

                    "uncertainty_intersects_geofence":
                        uncertainty_intersects,
                }

                for optional_col in [
                    "global_segment_id",
                    "gap_segment_id",
                ]:
                    if optional_col in metadata.columns:
                        row[
                            optional_col
                        ] = int(
                            meta[
                                optional_col
                            ]
                        )

                rows.append(
                    row
                )

    base = pd.DataFrame(
        rows
    )

    if len(
        base
    ) == 0:
        raise RuntimeError(
            f"{task}: no Step18 geofence instances were constructed."
        )

    return base


def add_persistence(
    base,
):
    df = base.copy()

    group_cols = [
        "task",
        "track_id",
    ]

    for optional_col in [
        "global_segment_id",
        "gap_segment_id",
    ]:
        if optional_col in df.columns:
            group_cols.append(
                optional_col
            )

    group_cols.extend(
        [
            "axis",
            "boundary",
            "direction",
        ]
    )

    df = df.sort_values(
        group_cols
        +
        [
            "window_start_time_sec",
            "sample_index",
        ]
    ).reset_index(
        drop=True
    )

    previous = (
        df
        .groupby(
            group_cols,
            sort=False,
        )[
            "uncertainty_intersects_geofence"
        ]
        .shift(
            1
        )
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    df[
        "previous_available_uncertainty_intersection"
    ] = previous

    df[
        "persistent_uncertainty_n2"
    ] = (
        df[
            "uncertainty_intersects_geofence"
        ].astype(
            bool
        )
        &
        previous
    )

    return df


def evaluate_response_threshold(
    persisted_base,
    response_sec,
):
    df = persisted_base.copy()

    pred_cross = df[
        "predicted_crossing"
    ].astype(
        bool
    )

    pred_tte = pd.to_numeric(
        df[
            "predicted_tte_sec"
        ],
        errors="coerce",
    )

    distance = pd.to_numeric(
        df[
            "distance_to_boundary"
        ],
        errors="coerce",
    )

    persistent_uncertainty = df[
        "persistent_uncertainty_n2"
    ].astype(
        bool
    )

    # Exact Step7H RED logic.
    red = (
        (
            pred_cross
            &
            pred_tte.notna()
            &
            (
                pred_tte
                <=
                float(
                    response_sec
                )
            )
        )
        |
        (
            pred_cross
            &
            pred_tte.isna()
            &
            distance.notna()
            &
            (
                distance
                <=
                0.0
            )
        )
    )

    yellow = (
        ~red
        &
        (
            pred_cross
            |
            persistent_uncertainty
        )
    )

    alarm = np.where(
        red,
        "RED",
        np.where(
            yellow,
            "YELLOW",
            "GREEN",
        ),
    )

    true_cross = df[
        "true_crossing"
    ].astype(
        bool
    )

    true_tte = pd.to_numeric(
        df[
            "true_tte_sec"
        ],
        errors="coerce",
    )

    urgent = (
        true_cross
        &
        true_tte.notna()
        &
        (
            true_tte
            <=
            float(
                response_sec
            )
        )
    )

    alert = (
        alarm
        !=
        "GREEN"
    )

    red_bool = (
        alarm
        ==
        "RED"
    )

    df[
        "response_threshold_sec"
    ] = float(
        response_sec
    )

    df[
        "alarm"
    ] = alarm

    df[
        "combined_alert"
    ] = alert

    df[
        "true_urgent_event"
    ] = urgent

    df[
        "red_alert"
    ] = red_bool

    return df


def safe_rate(
    numerator_mask,
    denominator_mask,
):
    denominator = int(
        np.sum(
            denominator_mask
        )
    )

    if denominator == 0:
        return np.nan

    numerator = int(
        np.sum(
            numerator_mask
            &
            denominator_mask
        )
    )

    return float(
        numerator
        /
        denominator
    )


def compute_metrics(
    df,
):
    true_cross = df[
        "true_crossing"
    ].to_numpy(
        dtype=bool
    )

    non_event = ~true_cross

    urgent = df[
        "true_urgent_event"
    ].to_numpy(
        dtype=bool
    )

    nonurgent = ~urgent

    alert = df[
        "combined_alert"
    ].to_numpy(
        dtype=bool
    )

    red = df[
        "red_alert"
    ].to_numpy(
        dtype=bool
    )

    yellow = (
        df[
            "alarm"
        ].to_numpy()
        ==
        "YELLOW"
    )

    green = (
        df[
            "alarm"
        ].to_numpy()
        ==
        "GREEN"
    )

    return {
        "instances":
            int(
                len(
                    df
                )
            ),

        "true_crossing_instances":
            int(
                true_cross.sum()
            ),

        "urgent_instances":
            int(
                urgent.sum()
            ),

        "combined_alert_instances":
            int(
                alert.sum()
            ),

        "red_instances":
            int(
                red.sum()
            ),

        "yellow_instances":
            int(
                yellow.sum()
            ),

        "green_instances":
            int(
                green.sum()
            ),

        "combined_alert_sensitivity":
            safe_rate(
                alert,
                true_cross,
            ),

        "combined_false_alert_rate":
            safe_rate(
                alert,
                non_event,
            ),

        "red_urgent_sensitivity":
            safe_rate(
                red,
                urgent,
            ),

        "red_nonurgent_false_positive_rate":
            safe_rate(
                red,
                nonurgent,
            ),

        "red_precision_for_urgent":
            (
                float(
                    np.sum(
                        red
                        &
                        urgent
                    )
                    /
                    np.sum(
                        red
                    )
                )
                if
                np.sum(
                    red
                )
                >
                0
                else
                np.nan
            ),

        "combined_alert_precision_for_crossing":
            (
                float(
                    np.sum(
                        alert
                        &
                        true_cross
                    )
                    /
                    np.sum(
                        alert
                    )
                )
                if
                np.sum(
                    alert
                )
                >
                0
                else
                np.nan
            ),
    }


def summarize_instances(
    instances,
):
    summary_rows = []

    boundary_rows = []

    for (
        task,
        response_sec
    ), group in instances.groupby(
        [
            "task",
            "response_threshold_sec",
        ],
        sort=False,
    ):
        metrics = compute_metrics(
            group
        )

        row = {
            "task":
                task,

            "response_threshold_sec":
                float(
                    response_sec
                ),
        }

        row.update(
            metrics
        )

        summary_rows.append(
            row
        )

        for (
            axis,
            boundary,
        ), bgroup in group.groupby(
            [
                "axis",
                "boundary",
            ],
            sort=False,
        ):
            bmetrics = compute_metrics(
                bgroup
            )

            brow = {
                "task":
                    task,

                "response_threshold_sec":
                    float(
                        response_sec
                    ),

                "axis":
                    axis,

                "boundary":
                    float(
                        boundary
                    ),
            }

            brow.update(
                bmetrics
            )

            boundary_rows.append(
                brow
            )

    summary = pd.DataFrame(
        summary_rows
    )

    per_boundary = pd.DataFrame(
        boundary_rows
    )

    macro_metric_cols = [
        "combined_alert_sensitivity",
        "combined_false_alert_rate",
        "red_urgent_sensitivity",
        "red_nonurgent_false_positive_rate",
        "red_precision_for_urgent",
        "combined_alert_precision_for_crossing",
    ]

    macro_rows = []

    for (
        task,
        response_sec
    ), group in per_boundary.groupby(
        [
            "task",
            "response_threshold_sec",
        ],
        sort=False,
    ):
        row = {
            "task":
                task,

            "response_threshold_sec":
                float(
                    response_sec
                ),

            "boundaries":
                int(
                    len(
                        group
                    )
                ),
        }

        for col in macro_metric_cols:
            values = pd.to_numeric(
                group[
                    col
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(
                    values
                )
            ]

            row[
                f"boundary_macro_{col}"
            ] = (
                float(
                    finite.mean()
                )
                if len(
                    finite
                )
                >
                0
                else
                np.nan
            )

        macro_rows.append(
            row
        )

    boundary_macro = pd.DataFrame(
        macro_rows
    )

    return (
        summary,
        per_boundary,
        boundary_macro,
    )


def main():
    print(
        "\nSTEP 6D-K3C17 — FROZEN STEP7 EXTERNAL OPERATIONAL WARNING EVALUATION"
    )

    print(
        "Secondary BAMBI Flight 6 stress test using the prelocked Step7H rule."
    )

    print(
        "NO q recalibration, NO threshold tuning, NO persistence tuning, "
        "NO response-threshold tuning, and NO model tuning.\n"
    )

    provenance = verify_frozen_inputs()

    print(
        "VERIFIED K3C10 PRELOCK SHA256:"
    )
    print(
        provenance[
            "prelock_hash"
        ]
    )

    print(
        "\nVERIFIED K3C15 PREDICTION MANIFEST SHA256:"
    )
    print(
        provenance[
            "pred_manifest_hash"
        ]
    )

    print(
        "\nK3C16 SUMMARY SHA256:"
    )
    print(
        provenance[
            "k3c16_summary_hash"
        ]
    )

    print(
        "\nFIXED PRIMARY q90 VALUES:"
    )
    print(
        json.dumps(
            provenance[
                "q90"
            ],
            indent=2,
        )
    )

    all_instances = []

    base_files = []

    for task, config in TASK_CONFIG.items():
        data = load_task_data(
            task
        )

        q90 = float(
            provenance[
                "q90"
            ][
                task
            ]
        )

        base = build_base_geofence_instances(
            task,
            data,
            q90,
        )

        persisted = add_persistence(
            base
        )

        base_path = (
            OUT
            /
            f"{task}_STEP18_base_geofence_instances_with_persistence.csv"
        )

        persisted.to_csv(
            base_path,
            index=False,
        )

        base_files.append({
            "task":
                task,

            "path":
                str(
                    base_path
                ),

            "sha256":
                sha256_file(
                    base_path
                ),
        })

        print(
            f"\n{task} BASE GEOFENCE INSTANCES:"
        )

        print(
            json.dumps(
                {
                    "rows":
                        int(
                            len(
                                persisted
                            )
                        ),

                    "samples":
                        int(
                            persisted[
                                "sample_index"
                            ].nunique()
                        ),

                    "tracks":
                        int(
                            persisted[
                                "track_id"
                            ].nunique()
                        ),

                    "q90_fixed_step6c":
                        q90,

                    "true_crossing_fraction":
                        float(
                            persisted[
                                "true_crossing"
                            ].mean()
                        ),

                    "predicted_crossing_fraction":
                        float(
                            persisted[
                                "predicted_crossing"
                            ].mean()
                        ),

                    "uncertainty_intersection_fraction":
                        float(
                            persisted[
                                "uncertainty_intersects_geofence"
                            ].mean()
                        ),

                    "persistent_uncertainty_n2_fraction":
                        float(
                            persisted[
                                "persistent_uncertainty_n2"
                            ].mean()
                        ),
                },
                indent=2,
            )
        )

        for response_sec in config[
            "response_thresholds_sec"
        ]:
            evaluated = evaluate_response_threshold(
                persisted,
                response_sec,
            )

            all_instances.append(
                evaluated
            )

    instances = pd.concat(
        all_instances,
        ignore_index=True,
    )

    summary, per_boundary, boundary_macro = summarize_instances(
        instances
    )

    instances_path = (
        OUT
        / "step6dk3c17_frozen_step7_instances.csv"
    )

    summary_path = (
        OUT
        / "TABLE_step6dk3c17_frozen_step7_metrics.csv"
    )

    per_boundary_path = (
        OUT
        / "TABLE_step6dk3c17_metrics_per_boundary.csv"
    )

    boundary_macro_path = (
        OUT
        / "TABLE_step6dk3c17_boundary_macro_metrics.csv"
    )

    instances.to_csv(
        instances_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    per_boundary.to_csv(
        per_boundary_path,
        index=False,
    )

    boundary_macro.to_csv(
        boundary_macro_path,
        index=False,
    )

    print(
        "\nINSTANCE-WEIGHTED METRICS:"
    )

    print(
        summary.to_string(
            index=False,
        )
    )

    print(
        "\nBOUNDARY-MACRO METRICS:"
    )

    print(
        boundary_macro.to_string(
            index=False,
        )
    )

    # Scientific interpretation flag: 5s q90 is known from K3C16 to be badly
    # under-covered. Preserve this explicitly in the manifest.
    k3c16 = json.loads(
        K3C16_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    k3c16_summary_csv = Path(
        k3c16[
            "summary_csv"
        ]
    )

    require(
        k3c16_summary_csv
    )

    cov = pd.read_csv(
        k3c16_summary_csv
    )

    q90_cov = cov[
        np.isclose(
            pd.to_numeric(
                cov[
                    "alpha"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            ),
            PRIMARY_ALPHA,
        )
    ].copy()

    q90_coverage_by_task = {}

    for _, row in q90_cov.iterrows():
        q90_coverage_by_task[
            str(
                row[
                    "task"
                ]
            )
        ] = float(
            row[
                "window_coverage"
            ]
        )

    manifest = {
        "step":
            "STEP_6D_K3C17_FROZEN_STEP7_EXTERNAL_OPERATIONAL_WARNING_EVALUATION",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "FROZEN_STEP7_EXTERNAL_EVALUATION_COMPLETE_SECONDARY_STRESS_TEST",

        "analysis_role":
            (
                "Secondary/exploratory frozen operational-transfer stress test "
                "on BAMBI Flight 6. Not field validation and not a pristine "
                "confirmatory uncertainty holdout."
            ),

        "verified_provenance":
            provenance,

        "rules": {
            "primary_conformal_level":
                0.90,

            "persistence_windows":
                PERSISTENCE_WINDOWS,

            "persistence_semantics":
                "two consecutive available forecast windows for the same track/segment/axis/boundary/direction",

            "boundaries":
                BOUNDARIES,

            "axes":
                [
                    "x",
                    "y",
                ],

            "sampling_hz":
                SAMPLING_HZ,

            "response_thresholds_sec":
                {
                    task:
                        config[
                            "response_thresholds_sec"
                        ]
                    for task, config in TASK_CONFIG.items()
                },

            "red":
                (
                    "Nominal predicted crossing at/before response threshold, "
                    "or predicted crossing with unavailable TTE while already at/inside boundary."
                ),

            "yellow":
                (
                    "Not RED and (nominal predicted crossing after threshold OR "
                    "90% conformal intersection with same virtual geofence for "
                    "two consecutive available windows)."
                ),

            "green":
                "Neither RED nor YELLOW.",
        },

        "fixed_q90_window_coverage_from_k3c16":
            q90_coverage_by_task,

        "critical_interpretation":
            {
                "2s_to_2s":
                    (
                        "The secondary Step18-compatible q90 transfer was fully covered "
                        "in K3C16 but highly conservative."
                    ),

                "2s_to_5s":
                    (
                        "The secondary Step18-compatible q90 transfer severely under-covered "
                        "in K3C16. Therefore 5s Step7 uncertainty-assisted warning results "
                        "must NOT be described as calibrated at 90%."
                    ),
            },

        "base_geofence_files":
            base_files,

        "instances_csv":
            str(
                instances_path
            ),

        "instances_csv_sha256":
            sha256_file(
                instances_path
            ),

        "summary_csv":
            str(
                summary_path
            ),

        "summary_csv_sha256":
            sha256_file(
                summary_path
            ),

        "per_boundary_csv":
            str(
                per_boundary_path
            ),

        "per_boundary_csv_sha256":
            sha256_file(
                per_boundary_path
            ),

        "boundary_macro_csv":
            str(
                boundary_macro_path
            ),

        "boundary_macro_csv_sha256":
            sha256_file(
                boundary_macro_path
            ),

        "frozen_constraints_respected": {
            "model_tuning":
                False,

            "q_recalibration":
                False,

            "threshold_tuning":
                False,

            "persistence_tuning":
                False,

            "response_threshold_tuning":
                False,
        },

        "nonclaims": [
            "BAMBI Flight 6 is a red-deer/human cross-domain dataset, not elephant-specific validation.",
            "The 50 degree Flight-6 FOVY remains an assumed/default BAMBI reference-pipeline value.",
            "Virtual geofences are normalized synthetic boundaries, not measured village/farm boundaries.",
            "Response thresholds are scenario assumptions, not measured team response times.",
            "This is not field validation.",
            "The 5s q90 uncertainty layer is not calibrated on this secondary representation because K3C16 showed severe undercoverage.",
            "No result was tuned or repaired using BAMBI.",
        ],
    }

    manifest_path = (
        OUT
        / "STEP6D_K3C17_FROZEN_STEP7_EXTERNAL_EVALUATION_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nMANIFEST:"
    )
    print(
        manifest_path
    )

    print(
        "MANIFEST SHA256:"
    )
    print(
        sha256_file(
            manifest_path
        )
    )

    print(
        "\nSTATUS:"
    )
    print(
        "FROZEN_STEP7_EXTERNAL_EVALUATION_COMPLETE_SECONDARY_STRESS_TEST"
    )

    print(
        "\nSTEP 6D-K3C17 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
