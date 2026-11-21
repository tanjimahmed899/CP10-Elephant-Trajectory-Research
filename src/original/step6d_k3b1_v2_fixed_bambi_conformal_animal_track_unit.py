from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# READ-ONLY / FROZEN INPUTS
# ---------------------------------------------------------------------
STEP6C_DIR = Path(
    r"C:\Users\HP\Downloads\step6c_crossfit_conformal"
)

CALIBRATION_SUMMARY = (
    STEP6C_DIR
    / "track_conformal_calibration_summary.csv"
)

STEPWISE_RADII = (
    STEP6C_DIR
    / "track_conformal_stepwise_radii.csv"
)

STEP6C_SOURCE = Path(
    r"C:\Users\HP\Downloads\step6c_crossfit_conformal.py"
)

BASE = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

CANONICAL_DIR = (
    BASE
    / "k2b3_final_canonical"
)

INFERENCE_DIR = (
    BASE
    / "k3_frozen_ensemble"
    / "first_inference"
)

CANONICAL_LOCK = (
    CANONICAL_DIR
    / "STEP6D_K2B3_CANONICAL_DATASET_LOCK.json"
)

MODEL_LOCK = (
    BASE
    / "k3_frozen_ensemble"
    / "model_lock"
    / "STEP6D_K3A1_FROZEN_TRANSFORMER_MODEL_LOCK.json"
)

OUT = (
    BASE
    / "k3_fixed_conformal"
    / "k3b1_v2_fixed_no_recalibration_animal_track_unit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


TASKS = {
    "2s_to_2s": {
        "future_steps": 20,

        "canonical_npz":
            CANONICAL_DIR
            / "2s_to_2s_bambi_flight6_canonical_LOCKED.npz",

        "metadata":
            CANONICAL_DIR
            / "2s_to_2s_bambi_flight6_canonical_metadata_LOCKED.csv",

        "inference_npz":
            INFERENCE_DIR
            / "2s_to_2s_BAMBI_Flight6_first_locked_inference.npz",
    },

    "2s_to_5s": {
        "future_steps": 50,

        "canonical_npz":
            CANONICAL_DIR
            / "2s_to_5s_bambi_flight6_canonical_LOCKED.npz",

        "metadata":
            CANONICAL_DIR
            / "2s_to_5s_bambi_flight6_canonical_metadata_LOCKED.csv",

        "inference_npz":
            INFERENCE_DIR
            / "2s_to_5s_BAMBI_Flight6_first_locked_inference.npz",
    },
}


CONSISTENCY_TOL_M = 1e-4


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def resolve_column(
    df,
    exact_candidates,
    contains_candidates=()
):
    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for name in exact_candidates:
        if name.lower() in lower_map:
            return lower_map[
                name.lower()
            ]

    for c in df.columns:
        low = c.lower()

        if all(
            token.lower() in low
            for token in contains_candidates
        ):
            return c

    raise RuntimeError(
        "Could not resolve required column. "
        f"Exact candidates={exact_candidates}; "
        f"contains={contains_candidates}; "
        f"available={list(df.columns)}"
    )


def build_track_key(meta):
    """
    Confirmatory aggregation unit = original BAMBI animal track_id.

    Important:
    K2A preprocessing may split one animal trajectory into multiple
    contiguous blocks around the two prelocked global discontinuities.
    Those blocks are preprocessing segments, NOT new animals. Therefore
    they must not replace the Step6C track-level calibration/evaluation
    unit.

    We intentionally ignore any metadata column named `track_key` when
    it encodes a block/segment identity and force aggregation by the
    original animal `track_id`.
    """
    if "track_id" not in meta.columns:
        raise RuntimeError(
            "BAMBI metadata has no track_id; cannot preserve the "
            "original animal-level conformal aggregation unit."
        )

    return np.asarray(
        [
            f"BAMBI6_track_{int(x)}"
            for x in meta[
                "track_id"
            ].to_numpy()
        ],
        dtype=object
    )


def exact_track_metrics(
    track_key,
    window_covered
):
    temp = pd.DataFrame({
        "track_key":
            np.asarray(
                track_key,
                dtype=str
            ),

        "covered":
            np.asarray(
                window_covered,
                dtype=float
            ),
    })

    by_track = (
        temp
        .groupby(
            "track_key",
            as_index=False
        )
        .agg(
            covered_windows=(
                "covered",
                "sum"
            ),

            windows=(
                "covered",
                "size"
            ),

            track_macro_window_coverage=(
                "covered",
                "mean"
            ),

            track_fully_covered=(
                "covered",
                "min"
            ),
        )
    )

    return {
        "window_coverage":
            float(
                np.asarray(
                    window_covered,
                    dtype=float
                ).mean()
            ),

        "track_macro_window_coverage":
            float(
                by_track[
                    "track_macro_window_coverage"
                ].mean()
            ),

        "fully_covered_track_fraction":
            float(
                by_track[
                    "track_fully_covered"
                ].mean()
            ),

        "by_track":
            by_track,
    }


def load_primary_q_table():
    cal = pd.read_csv(
        CALIBRATION_SUMMARY
    )

    required = [
        "task",
        "alpha",
        "target_coverage",
        "calibration_tracks",
        "normalized_track_simultaneous_radius",
        "order_statistic_k",
        "calibration_track_simultaneous_coverage",
    ]

    missing = [
        c
        for c in required
        if c not in cal.columns
    ]

    if missing:
        raise RuntimeError(
            "Calibration summary missing columns: "
            f"{missing}"
        )

    return cal.copy()


def load_stepwise_q_table():
    df = pd.read_csv(
        STEPWISE_RADII
    )

    task_col = resolve_column(
        df,
        [
            "task"
        ]
    )

    alpha_col = resolve_column(
        df,
        [
            "alpha"
        ]
    )

    step_col = resolve_column(
        df,
        [
            "forecast_step",
            "step"
        ],
        [
            "step"
        ]
    )

    time_col = resolve_column(
        df,
        [
            "forecast_time_sec",
            "time_sec"
        ],
        [
            "time"
        ]
    )

    q_col = resolve_column(
        df,
        [
            "normalized_track_step_radius",
            "normalized_step_radius"
        ],
        [
            "normalized",
            "radius"
        ]
    )

    return (
        df.copy(),
        {
            "task":
                task_col,

            "alpha":
                alpha_col,

            "forecast_step":
                step_col,

            "forecast_time_sec":
                time_col,

            "q_step":
                q_col,
        }
    )


def main():
    print(
        "\nSTEP 6D-K3B1-V2 — FIXED NO-RECALIBRATION BAMBI CONFORMAL EVALUATION"
    )

    print(
        "Uses original Step6C q-values and exact coverage definitions."
    )

    print(
        "NO BAMBI recalibration. NO q tuning. NO model tuning. NO Step-7.\n"
    )

    for p in [
        CALIBRATION_SUMMARY,
        STEPWISE_RADII,
        STEP6C_SOURCE,
        CANONICAL_LOCK,
        MODEL_LOCK,
    ]:
        require(
            p
        )

    canonical_lock = json.loads(
        CANONICAL_LOCK.read_text(
            encoding="utf-8"
        )
    )

    model_lock = json.loads(
        MODEL_LOCK.read_text(
            encoding="utf-8"
        )
    )

    if (
        canonical_lock.get(
            "status"
        )
        !=
        "LOCKED_BEFORE_ANY_BAMBI_MODEL_PREDICTIONS"
    ):
        raise RuntimeError(
            "Unexpected canonical lock status."
        )

    if (
        model_lock.get(
            "status"
        )
        !=
        "LOCKED_BEFORE_ANY_BAMBI_MODEL_PREDICTIONS"
    ):
        raise RuntimeError(
            "Unexpected model lock status."
        )

    primary_q = load_primary_q_table()

    (
        stepwise_q,
        step_cols
    ) = load_stepwise_q_table()

    print(
        "FIXED STEP6C PRIMARY CALIBRATION TABLE:"
    )

    print(
        primary_q.to_string(
            index=False
        )
    )

    all_primary_rows = []
    all_timewise_rows = []
    all_track_rows = []
    all_failure_rows = []

    input_hash_rows = []

    for p in [
        CALIBRATION_SUMMARY,
        STEPWISE_RADII,
        STEP6C_SOURCE,
        CANONICAL_LOCK,
        MODEL_LOCK,
    ]:
        input_hash_rows.append({
            "file":
                str(
                    p
                ),

            "sha256":
                sha256_file(
                    p
                ),
        })

    for task, cfg in TASKS.items():
        print(
            f"\n{'=' * 80}"
        )

        print(
            f"BAMBI CONFIRMATORY COVERAGE: {task}"
        )

        print(
            f"{'=' * 80}"
        )

        for key in [
            "canonical_npz",
            "metadata",
            "inference_npz",
        ]:
            require(
                cfg[
                    key
                ]
            )

            input_hash_rows.append({
                "file":
                    str(
                        cfg[
                            key
                        ]
                    ),

                "sha256":
                    sha256_file(
                        cfg[
                            key
                        ]
                    ),
            })

        with np.load(
            cfg[
                "canonical_npz"
            ],
            allow_pickle=False
        ) as z:
            gt_can = np.asarray(
                z[
                    "future_canonical"
                ],
                dtype=np.float64
            )

            scale = np.asarray(
                z[
                    "scale"
                ],
                dtype=np.float64
            )

            gt_abs_locked = np.asarray(
                z[
                    "future_abs_xy"
                ],
                dtype=np.float64
            )

        with np.load(
            cfg[
                "inference_npz"
            ],
            allow_pickle=False
        ) as z:
            pred_can = np.asarray(
                z[
                    "prediction_canonical"
                ],
                dtype=np.float64
            )

            pred_abs = np.asarray(
                z[
                    "prediction_abs_xy"
                ],
                dtype=np.float64
            )

            gt_abs_inf = np.asarray(
                z[
                    "ground_truth_abs_xy"
                ],
                dtype=np.float64
            )

        meta = pd.read_csv(
            cfg[
                "metadata"
            ]
        )

        if not (
            len(
                pred_can
            )
            ==
            len(
                gt_can
            )
            ==
            len(
                scale
            )
            ==
            len(
                meta
            )
        ):
            raise RuntimeError(
                f"{task}: sample-count mismatch."
            )

        if (
            pred_can.shape
            !=
            gt_can.shape
        ):
            raise RuntimeError(
                f"{task}: canonical prediction/GT shape mismatch: "
                f"{pred_can.shape} vs {gt_can.shape}"
            )

        if (
            pred_abs.shape
            !=
            gt_abs_locked.shape
            or
            pred_abs.shape
            !=
            gt_abs_inf.shape
        ):
            raise RuntimeError(
                f"{task}: absolute-array shape mismatch."
            )

        gt_abs_diff = float(
            np.max(
                np.abs(
                    gt_abs_locked
                    -
                    gt_abs_inf
                )
            )
        )

        if gt_abs_diff > 1e-9:
            raise RuntimeError(
                f"{task}: inference GT differs from locked canonical GT. "
                f"max diff={gt_abs_diff}"
            )

        err_can = np.linalg.norm(
            pred_can
            -
            gt_can,
            axis=2
        )

        err_abs = np.linalg.norm(
            pred_abs
            -
            gt_abs_locked,
            axis=2
        )

        reconstructed_abs = (
            err_can
            *
            scale[
                :,
                None
            ]
        )

        consistency_diff = float(
            np.max(
                np.abs(
                    reconstructed_abs
                    -
                    err_abs
                )
            )
        )

        print(
            "Canonical-to-world error consistency max diff (m):",
            consistency_diff
        )

        if consistency_diff > CONSISTENCY_TOL_M:
            raise RuntimeError(
                f"{task}: canonical/world error consistency diff "
                f"{consistency_diff} m exceeds {CONSISTENCY_TOL_M} m."
            )

        sample_max_can = err_can.max(
            axis=1
        )

        sample_max_abs = err_abs.max(
            axis=1
        )

        track_key = build_track_key(
            meta
        )

        cal_task = (
            primary_q[
                primary_q[
                    "task"
                ]
                .astype(str)
                ==
                task
            ]
            .copy()
            .sort_values(
                "alpha",
                ascending=False
            )
        )

        if len(
            cal_task
        ) != 2:
            raise RuntimeError(
                f"{task}: expected two fixed alpha rows, got {len(cal_task)}."
            )

        for _, cal_row in cal_task.iterrows():
            alpha = float(
                cal_row[
                    "alpha"
                ]
            )

            target = float(
                cal_row[
                    "target_coverage"
                ]
            )

            q_track = float(
                cal_row[
                    "normalized_track_simultaneous_radius"
                ]
            )

            absolute_radius = (
                q_track
                *
                scale
            )

            window_covered = (
                sample_max_can
                <=
                q_track
            )

            track_metrics = exact_track_metrics(
                track_key,
                window_covered
            )

            sq = stepwise_q[
                (
                    stepwise_q[
                        step_cols[
                            "task"
                        ]
                    ]
                    .astype(str)
                    ==
                    task
                )
                &
                (
                    np.isclose(
                        pd.to_numeric(
                            stepwise_q[
                                step_cols[
                                    "alpha"
                                ]
                            ],
                            errors="coerce"
                        ),
                        alpha,
                        rtol=0,
                        atol=1e-12
                    )
                )
            ].copy()

            if len(
                sq
            ) != cfg[
                "future_steps"
            ]:
                raise RuntimeError(
                    f"{task} alpha={alpha}: expected "
                    f"{cfg['future_steps']} stepwise radii, got {len(sq)}."
                )

            sq[
                "__step"
            ] = pd.to_numeric(
                sq[
                    step_cols[
                        "forecast_step"
                    ]
                ],
                errors="raise"
            ).astype(int)

            sq = sq.sort_values(
                "__step"
            )

            q_steps = pd.to_numeric(
                sq[
                    step_cols[
                        "q_step"
                    ]
                ],
                errors="raise"
            ).to_numpy(
                dtype=np.float64
            )

            time_steps = pd.to_numeric(
                sq[
                    step_cols[
                        "forecast_time_sec"
                    ]
                ],
                errors="raise"
            ).to_numpy(
                dtype=np.float64
            )

            step_radius_abs = (
                scale[
                    :,
                    None
                ]
                *
                q_steps[
                    None,
                    :
                ]
            )

            step_covered = (
                err_abs
                <=
                step_radius_abs
            )

            stepwise_window_coverage = float(
                step_covered.mean()
            )

            endpoint_coverage = float(
                step_covered[
                    :,
                    -1
                ].mean()
            )

            endpoint_q = float(
                q_steps[
                    -1
                ]
            )

            all_primary_rows.append({
                "task":
                    task,

                "species":
                    "Red deer (Cervus elaphus)",

                "dataset":
                    "BAMBI_FLIGHT6",

                "dataset_role_for_uncertainty":
                    "PRELOCKED_CROSS_SPECIES_EXTERNAL_CONFIRMATORY_STRESS_TEST",

                "alpha":
                    alpha,

                "target_coverage":
                    target,

                "samples":
                    int(
                        len(
                            meta
                        )
                    ),

                "tracks":
                    int(
                        len(
                            np.unique(
                                track_key
                            )
                        )
                    ),

                "normalized_track_simultaneous_radius":
                    q_track,

                "calibration_tracks":
                    int(
                        cal_row[
                            "calibration_tracks"
                        ]
                    ),

                "calibration_order_statistic_k":
                    int(
                        cal_row[
                            "order_statistic_k"
                        ]
                    ),

                "calibration_track_simultaneous_coverage":
                    float(
                        cal_row[
                            "calibration_track_simultaneous_coverage"
                        ]
                    ),

                "mean_absolute_radius_m":
                    float(
                        absolute_radius.mean()
                    ),

                "median_absolute_radius_m":
                    float(
                        np.median(
                            absolute_radius
                        )
                    ),

                "p95_absolute_radius_m":
                    float(
                        np.quantile(
                            absolute_radius,
                            0.95
                        )
                    ),

                "window_simultaneous_coverage":
                    track_metrics[
                        "window_coverage"
                    ],

                "window_coverage_gap_pp":
                    float(
                        100.0
                        *
                        (
                            track_metrics[
                                "window_coverage"
                            ]
                            -
                            target
                        )
                    ),

                "track_macro_window_coverage":
                    track_metrics[
                        "track_macro_window_coverage"
                    ],

                "fully_covered_track_fraction":
                    track_metrics[
                        "fully_covered_track_fraction"
                    ],

                "stepwise_window_coverage":
                    stepwise_window_coverage,

                "endpoint_normalized_radius":
                    endpoint_q,

                "endpoint_window_coverage":
                    endpoint_coverage,

                "mean_max_canonical_error":
                    float(
                        sample_max_can.mean()
                    ),

                "median_max_canonical_error":
                    float(
                        np.median(
                            sample_max_can
                        )
                    ),

                "p95_max_canonical_error":
                    float(
                        np.quantile(
                            sample_max_can,
                            0.95
                        )
                    ),

                "mean_max_absolute_error_m":
                    float(
                        sample_max_abs.mean()
                    ),

                "p95_max_absolute_error_m":
                    float(
                        np.quantile(
                            sample_max_abs,
                            0.95
                        )
                    ),

                "max_abs_vs_scaled_canonical_error_diff_m":
                    consistency_diff,

                "recalibrated_on_bambi":
                    False,
            })

            bt = track_metrics[
                "by_track"
            ].copy()

            bt.insert(
                0,
                "task",
                task
            )

            bt.insert(
                1,
                "alpha",
                alpha
            )

            bt.insert(
                2,
                "target_coverage",
                target
            )

            all_track_rows.append(
                bt
            )

            for j in range(
                cfg[
                    "future_steps"
                ]
            ):
                all_timewise_rows.append({
                    "task":
                        task,

                    "alpha":
                        alpha,

                    "target_coverage":
                        target,

                    "forecast_step":
                        int(
                            sq[
                                "__step"
                            ].iloc[
                                j
                            ]
                        ),

                    "forecast_time_sec":
                        float(
                            time_steps[
                                j
                            ]
                        ),

                    "normalized_track_step_radius":
                        float(
                            q_steps[
                                j
                            ]
                        ),

                    "mean_absolute_step_radius_m":
                        float(
                            step_radius_abs[
                                :,
                                j
                            ].mean()
                        ),

                    "window_coverage":
                        float(
                            step_covered[
                                :,
                                j
                            ].mean()
                        ),

                    "coverage_gap_pp":
                        float(
                            100.0
                            *
                            (
                                step_covered[
                                    :,
                                    j
                                ].mean()
                                -
                                target
                            )
                        ),

                    "mean_error_absolute_m":
                        float(
                            err_abs[
                                :,
                                j
                            ].mean()
                        ),
                })

            failures = np.where(
                ~window_covered
            )[0]

            if len(
                failures
            ) > 0:
                excess = (
                    sample_max_can[
                        failures
                    ]
                    -
                    q_track
                )

                order = failures[
                    np.argsort(
                        -excess
                    )
                ]

                top = order[
                    :min(
                        30,
                        len(
                            order
                        )
                    )
                ]

                for idx in top:
                    row = {
                        "task":
                            task,

                        "alpha":
                            alpha,

                        "target_coverage":
                            target,

                        "sample_index":
                            int(
                                idx
                            ),

                        "track_key":
                            str(
                                track_key[
                                    idx
                                ]
                            ),

                        "scale":
                            float(
                                scale[
                                    idx
                                ]
                            ),

                        "max_canonical_error":
                            float(
                                sample_max_can[
                                    idx
                                ]
                            ),

                        "normalized_radius":
                            q_track,

                        "canonical_excess":
                            float(
                                sample_max_can[
                                    idx
                                ]
                                -
                                q_track
                            ),

                        "max_absolute_error_m":
                            float(
                                sample_max_abs[
                                    idx
                                ]
                            ),

                        "absolute_radius_m":
                            float(
                                absolute_radius[
                                    idx
                                ]
                            ),
                    }

                    for c in [
                        "track_id",
                        "block_key",
                        "start_time_sec",
                        "end_time_sec",
                        "window_id",
                        "sample_id",
                    ]:
                        if c in meta.columns:
                            v = meta.iloc[
                                idx
                            ][
                                c
                            ]

                            row[
                                c
                            ] = (
                                v.item()
                                if hasattr(
                                    v,
                                    "item"
                                )
                                else v
                            )

                    all_failure_rows.append(
                        row
                    )

            print(
                f"alpha={alpha:.2f} target={target:.2f} "
                f"q={q_track:.9f} | "
                f"window={track_metrics['window_coverage']:.6f} | "
                f"track_macro={track_metrics['track_macro_window_coverage']:.6f} | "
                f"fully_track={track_metrics['fully_covered_track_fraction']:.6f} | "
                f"stepwise={stepwise_window_coverage:.6f} | "
                f"endpoint={endpoint_coverage:.6f}"
            )

    primary_df = pd.DataFrame(
        all_primary_rows
    )

    timewise_df = pd.DataFrame(
        all_timewise_rows
    )

    track_df = pd.concat(
        all_track_rows,
        ignore_index=True
    )

    failures_df = pd.DataFrame(
        all_failure_rows
    )

    hashes_df = pd.DataFrame(
        input_hash_rows
    ).drop_duplicates(
        subset=[
            "file"
        ]
    )

    primary_path = (
        OUT
        / "TABLE_step6dk3b1v2_fixed_bambi_primary_coverage.csv"
    )

    timewise_path = (
        OUT
        / "TABLE_step6dk3b1v2_fixed_bambi_timewise_coverage.csv"
    )

    track_path = (
        OUT
        / "TABLE_step6dk3b1v2_fixed_bambi_track_coverage.csv"
    )

    failures_path = (
        OUT
        / "step6dk3b1v2_fixed_bambi_failure_cases.csv"
    )

    hashes_path = (
        OUT
        / "STEP6D_K3B1_V2_INPUT_HASHES.csv"
    )

    primary_df.to_csv(
        primary_path,
        index=False
    )

    timewise_df.to_csv(
        timewise_path,
        index=False
    )

    track_df.to_csv(
        track_path,
        index=False
    )

    failures_df.to_csv(
        failures_path,
        index=False
    )

    hashes_df.to_csv(
        hashes_path,
        index=False
    )

    output_hashes = []

    for p in [
        primary_path,
        timewise_path,
        track_path,
        failures_path,
        hashes_path,
    ]:
        output_hashes.append({
            "file":
                str(
                    p
                ),

            "sha256":
                sha256_file(
                    p
                ),
        })

    output_hashes_df = pd.DataFrame(
        output_hashes
    )

    output_hashes_df.to_csv(
        OUT
        / "STEP6D_K3B1_V2_OUTPUT_HASHES.csv",
        index=False
    )

    summary = {
        "step":
            "STEP_6D_K3B1_V2_FIXED_NO_RECALIBRATION_BAMBI_CONFORMAL_EVALUATION",

        "status":
            "FIXED_STEP6C_BAMBI_CONFIRMATORY_COVERAGE_COMPLETE",

        "species":
            "Red deer (Cervus elaphus)",

        "dataset":
            "BAMBI Flight 6",

        "dataset_role":
            "cross-species external confirmatory robustness/stress test",

        "recalibration_on_bambi":
            False,

        "q_source":
            str(
                CALIBRATION_SUMMARY
            ),

        "stepwise_q_source":
            str(
                STEPWISE_RADII
            ),

        "step6c_source":
            str(
                STEP6C_SOURCE
            ),

        "primary_rule":
            "window covered iff max canonical radial error over all future steps <= fixed Step6C q_track",

        "absolute_radius_rule":
            "absolute radius in BAMBI world metres = fixed normalized q * per-window BAMBI history-path scale",

        "track_macro_rule":
            "mean of each track's window coverage",

        "fully_covered_track_rule":
            "fraction of original BAMBI animal track_ids for which every evaluation window is simultaneously covered",

        "results":
            primary_df.to_dict(
                orient="records"
            ),

        "important_interpretation": [
            "No BAMBI outcome was used to estimate or modify q90/q95.",
            "No BAMBI-specific recalibration, threshold tuning, model tuning, or checkpoint substitution is performed.",
            "The fixed Step6C normalized radii are transferred unchanged into BAMBI canonical space.",
            "Track-level aggregation is forced to original BAMBI animal track_id; preprocessing block/segment identities are not counted as separate tracks.",
            "World-metre radii are obtained only by the original Step6C scale mapping q * scale.",
            "BAMBI is a red-deer/human cross-species external stress test, not elephant-specific validation.",
            "Coverage results must be reported even if they miss nominal 90% or 95% targets.",
            "The Flight-6 georeferencing uses the BAMBI reference-pipeline 50-degree FOV assumption rather than independently measured Flight-6 camera calibration.",
        ],
    }

    (
        OUT
        / "STEP6D_K3B1_V2_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    with open(
        OUT
        / "STEP6D_K3B1_V2_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K3B1-V2 — FIXED NO-RECALIBRATION BAMBI CONFORMAL EVALUATION\n"
        )

        f.write(
            "=" * 94
            +
            "\n\n"
        )

        f.write(
            "PRIMARY COVERAGE RESULTS\n"
        )

        f.write(
            "------------------------\n"
        )

        f.write(
            primary_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nSTATUS\n"
        )

        f.write(
            "------\n"
        )

        f.write(
            summary[
                "status"
            ]
            +
            "\n"
        )

    print(
        "\nFINAL FIXED BAMBI CONFORMAL COVERAGE:"
    )

    print(
        primary_df.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        summary[
            "status"
        ]
    )

    print(
        "\nSTEP 6D-K3B1 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
