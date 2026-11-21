
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"

R11B_DIR = (
    REVISION_ROOT
    / "R11_noise_floor_analysis"
    / "R11B_empirical_geometric_residual_floor_protocol_lock"
)
R11B_MANIFEST = R11B_DIR / "R11B_MANIFEST.json"

R10A_DIR = (
    REVISION_ROOT
    / "R10_ensemble_analysis"
    / "R10A_frozen_transformer_ensemble_evidence_audit"
)
R10A_PRED_INVENTORY = (
    R10A_DIR
    / "R10A_FROZEN_TRANSFORMER_PREDICTION_INVENTORY.csv"
)

R10B_DIR = (
    REVISION_ROOT
    / "R10_ensemble_analysis"
    / "R10B_testreal_transformer_ensemble_value"
)
R10B_SAMPLE = (
    R10B_DIR
    / "R10B_TESTREAL_ENSEMBLE_SAMPLE_METRICS.csv"
)

GEOMETRY_CSV = (
    REVISION_ROOT
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3B_corrected_isotropic_rebuild"
    / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"
)

TESTREAL_DIR = MAIN1 / "final_neural_trajectory_dataset"

OUT = (
    REVISION_ROOT
    / "R11_noise_floor_analysis"
    / "R11C_testreal_forecast_error_vs_geometric_residual_floor"
)
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "samples": 1053,
        "segments": 89,
        "sequences": 6,
        "future_length": 20,
    },
    "2s_to_5s": {
        "samples": 826,
        "segments": 63,
        "sequences": 5,
        "future_length": 50,
    },
}

SEEDS = [42, 52, 62]
BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20261029


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


def verify_r11b():
    require(R11B_MANIFEST)

    m = json.loads(
        R11B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R11B_TESTREAL_EMPIRICAL_GEOMETRIC_RESIDUAL_F50_F95_LOCKED_READY_FOR_R11C_FORECAST_ERROR_ABOVE_FLOOR_ANALYSIS"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R11-B status: {m.get('status')}"
        )

    f50 = float(m["F50_px"])
    f95 = float(m["F95_px"])

    if not (
        np.isfinite(f50)
        and
        np.isfinite(f95)
        and
        f50 > 0
        and
        f95 > f50
    ):
        raise RuntimeError(
            f"Invalid locked residual references: F50={f50}, F95={f95}"
        )

    return m, f50, f95


def load_geometry():
    require(GEOMETRY_CSV)

    df = pd.read_csv(
        GEOMETRY_CSV,
        dtype={
            "sequence": str,
        }
    )

    required = {
        "sequence",
        "width_px",
        "height_px",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Geometry lock missing columns: {sorted(missing)}"
        )

    return {
        str(row["sequence"]): {
            "W": float(row["width_px"]),
            "H": float(row["height_px"]),
        }
        for _, row in df.iterrows()
    }


def load_testreal_metadata(task):
    p = (
        TESTREAL_DIR
        /
        f"{task}_birdsai_test_metadata.csv"
    )

    require(p)

    df = pd.read_csv(
        p,
        dtype={
            "sample_id": str,
            "sequence": str,
            "source_group": str,
            "segment_id": str,
            "track_key": str,
        }
    )

    cfg = TASKS[task]

    if len(df) != cfg["samples"]:
        raise RuntimeError(
            f"{task}: metadata rows={len(df)}, expected={cfg['samples']}"
        )

    required = {
        "sample_id",
        "sequence",
        "segment_id",
        "track_key",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{task}: metadata missing columns {sorted(missing)}"
        )

    if df["sample_id"].duplicated().any():
        raise RuntimeError(
            f"{task}: duplicate sample_id."
        )

    return df, p


def load_seed_prediction_paths():
    require(R10A_PRED_INVENTORY)

    df = pd.read_csv(
        R10A_PRED_INVENTORY,
        dtype={
            "task": str,
            "dataset_label": str,
            "prediction_path": str,
            "prediction_sha256": str,
        }
    )

    df = df[
        df["dataset_label"] == "BIRDSAI_test"
    ].copy()

    mapping = {}

    for _, row in df.iterrows():
        key = (
            str(row["task"]),
            int(row["seed"]),
        )

        p = Path(
            str(row["prediction_path"])
        )

        require(p)

        current_hash = sha256_file(p)

        if current_hash != str(
            row["prediction_sha256"]
        ):
            raise RuntimeError(
                f"Prediction hash changed: {p}"
            )

        mapping[key] = p

    expected = {
        (task, seed)
        for task in TASKS
        for seed in SEEDS
    }

    if set(mapping) != expected:
        raise RuntimeError(
            f"Unexpected TestReal prediction map keys: {sorted(mapping)}"
        )

    return mapping


def load_prediction(path):
    arrays = np.load(
        path,
        allow_pickle=False
    )

    required = {
        "truth_abs",
        "prediction_abs",
    }

    missing = required - set(
        arrays.files
    )

    if missing:
        raise RuntimeError(
            f"{path.name}: missing keys {sorted(missing)}"
        )

    truth = np.asarray(
        arrays["truth_abs"],
        dtype=np.float64
    )

    pred = np.asarray(
        arrays["prediction_abs"],
        dtype=np.float64
    )

    return truth, pred


def pointwise_pixel_errors(
    truth,
    pred,
    meta,
    geometry,
):
    seqs = meta["sequence"].astype(str)

    missing = sorted(
        set(seqs)
        -
        set(geometry)
    )

    if missing:
        raise RuntimeError(
            f"Geometry missing sequences: {missing}"
        )

    W = np.asarray(
        [
            geometry[s]["W"]
            for s in seqs
        ],
        dtype=np.float64
    )

    H = np.asarray(
        [
            geometry[s]["H"]
            for s in seqs
        ],
        dtype=np.float64
    )

    delta = pred - truth

    dx = (
        delta[:, :, 0]
        *
        W[:, None]
    )

    dy = (
        delta[:, :, 1]
        *
        H[:, None]
    )

    return np.sqrt(
        dx ** 2
        +
        dy ** 2
    )


def percentile_ci(values):
    return (
        float(
            np.percentile(
                values,
                2.5
            )
        ),
        float(
            np.percentile(
                values,
                97.5
            )
        ),
    )


def segment_macro_table(
    sample_df,
    value_col,
):
    return (
        sample_df.groupby(
            [
                "sequence",
                "segment_id",
            ],
            dropna=False,
            as_index=False
        )
        .agg(
            segment_value=(
                value_col,
                "mean"
            ),
            windows=(
                "sample_id",
                "size"
            ),
        )
    )


def bootstrap_segment_macro_mean(
    sample_df,
    value_col,
    rng_seed,
):
    seg = segment_macro_table(
        sample_df,
        value_col,
    )

    values = seg[
        "segment_value"
    ].to_numpy(
        dtype=np.float64
    )

    n = len(values)

    point = float(
        values.mean()
    )

    rng = np.random.default_rng(
        rng_seed
    )

    boot = np.empty(
        BOOTSTRAP_B,
        dtype=np.float64
    )

    for b in range(
        BOOTSTRAP_B
    ):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        boot[b] = float(
            values[
                idx
            ].mean()
        )

    low, high = percentile_ci(
        boot
    )

    return {
        "segments": int(n),
        "point": point,
        "ci95_low": low,
        "ci95_high": high,
    }


def leave_one_sequence_out(
    sample_df,
    value_col,
):
    rows = []

    for held_out in sorted(
        sample_df[
            "sequence"
        ].astype(str).unique()
    ):
        kept = sample_df[
            sample_df[
                "sequence"
            ].astype(str)
            !=
            held_out
        ].copy()

        seg = segment_macro_table(
            kept,
            value_col,
        )

        rows.append(
            {
                "held_out_sequence":
                    held_out,

                "remaining_segments":
                    int(
                        len(seg)
                    ),

                "segment_macro_mean":
                    float(
                        seg[
                            "segment_value"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def analyze_task(
    task,
    pred_paths,
    geometry,
    f50,
    f95,
):
    meta, meta_path = load_testreal_metadata(
        task
    )

    truths = []
    preds = []

    for seed in SEEDS:
        truth, pred = load_prediction(
            pred_paths[
                (
                    task,
                    seed,
                )
            ]
        )

        cfg = TASKS[task]

        expected_shape = (
            cfg["samples"],
            cfg["future_length"],
            2,
        )

        if truth.shape != expected_shape:
            raise RuntimeError(
                f"{task}/seed{seed}: truth shape={truth.shape}, expected={expected_shape}"
            )

        if pred.shape != expected_shape:
            raise RuntimeError(
                f"{task}/seed{seed}: pred shape={pred.shape}, expected={expected_shape}"
            )

        truths.append(
            truth
        )

        preds.append(
            pred
        )

    truth0 = truths[0]

    for truth in truths[1:]:
        if not np.array_equal(
            truth,
            truth0
        ):
            raise RuntimeError(
                f"{task}: truth differs across seeds."
            )

    ensemble_pred = np.mean(
        np.stack(
            preds,
            axis=0
        ),
        axis=0
    )

    point_err = pointwise_pixel_errors(
        truth0,
        ensemble_pred,
        meta,
        geometry,
    )

    ade = point_err.mean(
        axis=1
    )

    fde = point_err[
        :,
        -1
    ]

    point_fraction_le_f50 = (
        point_err <= f50
    ).mean(
        axis=1
    )

    point_fraction_le_f95 = (
        point_err <= f95
    ).mean(
        axis=1
    )

    point_fraction_gt_f50 = 1.0 - point_fraction_le_f50
    point_fraction_gt_f95 = 1.0 - point_fraction_le_f95

    sample = meta.copy()

    sample[
        "task"
    ] = task

    sample[
        "ensemble_ADE_px"
    ] = ade

    sample[
        "ensemble_FDE_px"
    ] = fde

    sample[
        "ADE_over_F50"
    ] = ade / f50

    sample[
        "ADE_over_F95"
    ] = ade / f95

    sample[
        "FDE_over_F50"
    ] = fde / f50

    sample[
        "FDE_over_F95"
    ] = fde / f95

    sample[
        "ADE_above_F50"
    ] = (
        ade > f50
    ).astype(float)

    sample[
        "ADE_above_F95"
    ] = (
        ade > f95
    ).astype(float)

    sample[
        "FDE_above_F50"
    ] = (
        fde > f50
    ).astype(float)

    sample[
        "FDE_above_F95"
    ] = (
        fde > f95
    ).astype(float)

    sample[
        "point_fraction_above_F50"
    ] = point_fraction_gt_f50

    sample[
        "point_fraction_above_F95"
    ] = point_fraction_gt_f95

    sample[
        "ADE_excess_over_F50_px"
    ] = np.maximum(
        ade - f50,
        0.0
    )

    sample[
        "ADE_excess_over_F95_px"
    ] = np.maximum(
        ade - f95,
        0.0
    )

    sample[
        "FDE_excess_over_F50_px"
    ] = np.maximum(
        fde - f50,
        0.0
    )

    sample[
        "FDE_excess_over_F95_px"
    ] = np.maximum(
        fde - f95,
        0.0
    )

    # Save full pointwise error matrix in long form.
    point_rows = []

    for i in range(
        len(sample)
    ):
        sid = str(
            sample.iloc[
                i
            ][
                "sample_id"
            ]
        )

        seq = str(
            sample.iloc[
                i
            ][
                "sequence"
            ]
        )

        seg = str(
            sample.iloc[
                i
            ][
                "segment_id"
            ]
        )

        trk = str(
            sample.iloc[
                i
            ][
                "track_key"
            ]
        )

        for t in range(
            point_err.shape[
                1
            ]
        ):
            e = float(
                point_err[
                    i,
                    t
                ]
            )

            point_rows.append(
                {
                    "task":
                        task,

                    "sample_id":
                        sid,

                    "sequence":
                        seq,

                    "segment_id":
                        seg,

                    "track_key":
                        trk,

                    "future_step_index":
                        int(
                            t
                        ),

                    "error_px":
                        e,

                    "above_F50":
                        bool(
                            e > f50
                        ),

                    "above_F95":
                        bool(
                            e > f95
                        ),

                    "error_over_F50":
                        e / f50,

                    "error_over_F95":
                        e / f95,
                }
            )

    point_df = pd.DataFrame(
        point_rows
    )

    summary = {
        "task":
            task,

        "samples":
            int(
                len(
                    sample
                )
            ),

        "segments":
            int(
                sample.groupby(
                    [
                        "sequence",
                        "segment_id",
                    ],
                    dropna=False
                ).ngroups
            ),

        "sequences":
            int(
                sample[
                    "sequence"
                ].nunique()
            ),

        "F50_px":
            f50,

        "F95_px":
            f95,

        "ensemble_ADE_px_window_mean":
            float(
                ade.mean()
            ),

        "ensemble_FDE_px_window_mean":
            float(
                fde.mean()
            ),

        "window_mean_ADE_over_F50":
            float(
                ade.mean()
                /
                f50
            ),

        "window_mean_ADE_over_F95":
            float(
                ade.mean()
                /
                f95
            ),

        "window_mean_FDE_over_F50":
            float(
                fde.mean()
                /
                f50
            ),

        "window_mean_FDE_over_F95":
            float(
                fde.mean()
                /
                f95
            ),

        "all_future_points_above_F50_pct":
            float(
                100.0
                *
                (
                    point_err > f50
                ).mean()
            ),

        "all_future_points_above_F95_pct":
            float(
                100.0
                *
                (
                    point_err > f95
                ).mean()
            ),

        "samples_ADE_above_F50_pct":
            float(
                100.0
                *
                (
                    ade > f50
                ).mean()
            ),

        "samples_ADE_above_F95_pct":
            float(
                100.0
                *
                (
                    ade > f95
                ).mean()
            ),

        "samples_FDE_above_F50_pct":
            float(
                100.0
                *
                (
                    fde > f50
                ).mean()
            ),

        "samples_FDE_above_F95_pct":
            float(
                100.0
                *
                (
                    fde > f95
                ).mean()
            ),

        "metadata_path":
            str(
                meta_path
            ),

        "metadata_sha256":
            sha256_file(
                meta_path
            ),
    }

    return (
        sample,
        point_df,
        summary,
    )


def main():
    print(
        "\nR11-C — BIRDSAI TESTREAL FORECAST ERROR VS EMPIRICAL GEOMETRIC RESIDUAL FLOOR"
    )

    print(
        "NO training and NO new inference. This step reconstructs the already-frozen "
        "3-seed Transformer ensemble on BIRDSAI TestReal from the locked seed prediction "
        "files and converts every future-point error to pixels using the locked sequence "
        "geometry. Errors are compared only to the held-out BIRDSAI TestReal residual "
        "references locked in R11-B: F50 and F95. Primary outputs include pointwise fractions "
        "above each reference, sample ADE/FDE ratios, segment-cluster bootstrap confidence "
        "intervals, and leave-one-sequence-out sensitivity. No residual reference is subtracted "
        "from model error as if variances were known to be additive.\n"
    )

    r11b, f50, f95 = verify_r11b()

    require(R10B_SAMPLE)

    geometry = load_geometry()
    pred_paths = load_seed_prediction_paths()

    sample_frames = []
    point_frames = []
    summaries = []
    bootstrap_rows = []
    loo_rows = []

    metrics_to_bootstrap = [
        (
            "ADE_above_F50",
            "fraction_of_samples_ADE_above_F50",
        ),
        (
            "ADE_above_F95",
            "fraction_of_samples_ADE_above_F95",
        ),
        (
            "FDE_above_F50",
            "fraction_of_samples_FDE_above_F50",
        ),
        (
            "FDE_above_F95",
            "fraction_of_samples_FDE_above_F95",
        ),
        (
            "point_fraction_above_F50",
            "mean_sample_fraction_future_points_above_F50",
        ),
        (
            "point_fraction_above_F95",
            "mean_sample_fraction_future_points_above_F95",
        ),
        (
            "ADE_over_F95",
            "ADE_over_F95_ratio",
        ),
        (
            "FDE_over_F95",
            "FDE_over_F95_ratio",
        ),
    ]

    for task_index, task in enumerate(
        TASKS
    ):
        (
            sample,
            point_df,
            summary,
        ) = analyze_task(
            task,
            pred_paths,
            geometry,
            f50,
            f95,
        )

        if summary["segments"] != TASKS[task]["segments"]:
            raise RuntimeError(
                f"{task}: segments={summary['segments']}, "
                f"expected={TASKS[task]['segments']}"
            )

        sample_frames.append(
            sample
        )

        point_frames.append(
            point_df
        )

        summaries.append(
            summary
        )

        for metric_index, (
            value_col,
            metric_name,
        ) in enumerate(
            metrics_to_bootstrap
        ):
            result = bootstrap_segment_macro_mean(
                sample,
                value_col,
                BOOTSTRAP_SEED
                +
                task_index
                *
                1000
                +
                metric_index,
            )

            bootstrap_rows.append(
                {
                    "task":
                        task,

                    "metric":
                        metric_name,

                    "segments":
                        result[
                            "segments"
                        ],

                    "estimate":
                        result[
                            "point"
                        ],

                    "ci95_low":
                        result[
                            "ci95_low"
                        ],

                    "ci95_high":
                        result[
                            "ci95_high"
                        ],

                    "dependency_unit":
                        "(sequence, segment_id)",

                    "within_segment_reduction":
                        "mean across overlapping windows",
                }
            )

            loo = leave_one_sequence_out(
                sample,
                value_col,
            )

            loo[
                "task"
            ] = task

            loo[
                "metric"
            ] = metric_name

            loo_rows.append(
                loo
            )

    sample_all = pd.concat(
        sample_frames,
        ignore_index=True
    )

    point_all = pd.concat(
        point_frames,
        ignore_index=True
    )

    summary_df = pd.DataFrame(
        summaries
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    loo_df = pd.concat(
        loo_rows,
        ignore_index=True
    )

    # Sanity check against R10-B sample ADE/FDE.
    r10b = pd.read_csv(
        R10B_SAMPLE,
        dtype={
            "task": str,
            "sample_id": str,
        }
    )

    compare_rows = []

    for task in TASKS:
        ours = sample_all[
            sample_all[
                "task"
            ]
            ==
            task
        ][
            [
                "sample_id",
                "ensemble_ADE_px",
                "ensemble_FDE_px",
            ]
        ]

        ref = r10b[
            r10b[
                "task"
            ]
            ==
            task
        ][
            [
                "sample_id",
                "ensemble_ADE_px",
                "ensemble_FDE_px",
            ]
        ]

        merged = ours.merge(
            ref,
            on="sample_id",
            suffixes=(
                "_R11C",
                "_R10B",
            ),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        if not (
            merged[
                "_merge"
            ]
            ==
            "both"
        ).all():
            raise RuntimeError(
                f"{task}: R10-B sample identity mismatch."
            )

        ade_diff = np.abs(
            merged[
                "ensemble_ADE_px_R11C"
            ].to_numpy(
                dtype=np.float64
            )
            -
            merged[
                "ensemble_ADE_px_R10B"
            ].to_numpy(
                dtype=np.float64
            )
        )

        fde_diff = np.abs(
            merged[
                "ensemble_FDE_px_R11C"
            ].to_numpy(
                dtype=np.float64
            )
            -
            merged[
                "ensemble_FDE_px_R10B"
            ].to_numpy(
                dtype=np.float64
            )
        )

        compare_rows.append(
            {
                "task":
                    task,

                "samples":
                    int(
                        len(
                            merged
                        )
                    ),

                "max_abs_ADE_difference_px":
                    float(
                        ade_diff.max()
                    ),

                "max_abs_FDE_difference_px":
                    float(
                        fde_diff.max()
                    ),

                "mean_abs_ADE_difference_px":
                    float(
                        ade_diff.mean()
                    ),

                "mean_abs_FDE_difference_px":
                    float(
                        fde_diff.mean()
                    ),
            }
        )

    reproducibility_df = pd.DataFrame(
        compare_rows
    )

    if (
        reproducibility_df[
            "max_abs_ADE_difference_px"
        ].max()
        >
        1e-9
        or
        reproducibility_df[
            "max_abs_FDE_difference_px"
        ].max()
        >
        1e-9
    ):
        raise RuntimeError(
            "R11-C reconstructed ensemble errors do not exactly reproduce R10-B."
        )

    sample_path = (
        OUT
        /
        "R11C_TESTREAL_SAMPLE_ERROR_VS_RESIDUAL_FLOOR.csv"
    )

    point_path = (
        OUT
        /
        "R11C_TESTREAL_POINTWISE_PIXEL_ERRORS.csv"
    )

    summary_path = (
        OUT
        /
        "R11C_TESTREAL_NOISE_FLOOR_SUMMARY.csv"
    )

    bootstrap_path = (
        OUT
        /
        "R11C_SEGMENT_CLUSTER_BOOTSTRAP.csv"
    )

    loo_path = (
        OUT
        /
        "R11C_LEAVE_ONE_SEQUENCE_OUT_SENSITIVITY.csv"
    )

    reproducibility_path = (
        OUT
        /
        "R11C_R10B_REPRODUCIBILITY_CHECK.csv"
    )

    sample_all.to_csv(
        sample_path,
        index=False
    )

    point_all.to_csv(
        point_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False
    )

    loo_df.to_csv(
        loo_path,
        index=False
    )

    reproducibility_df.to_csv(
        reproducibility_path,
        index=False
    )

    status = (
        "R11C_TESTREAL_FORECAST_ERROR_RELATIVE_TO_EMPIRICAL_RESIDUAL_FLOOR_QUANTIFIED_READY_FOR_R11D_NOISE_FLOOR_CONCLUSION"
    )

    manifest = {
        "step":
            "R11C_TESTREAL_FORECAST_ERROR_VS_EMPIRICAL_GEOMETRIC_RESIDUAL_FLOOR",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "new_inference":
            False,

        "F50_px":
            f50,

        "F95_px":
            f95,

        "forecast_output":
            "frozen 3-seed Transformer prediction ensemble",

        "primary_domain":
            "BIRDSAI TestReal",

        "metric_space":
            "pixel",

        "dependency_unit":
            "(sequence, segment_id)",

        "bootstrap_replicates":
            BOOTSTRAP_B,

        "bootstrap_seed_base":
            BOOTSTRAP_SEED,

        "important_interpretation_rule":
            (
                "Forecast error is compared to empirical residual references by ratios and "
                "fractions above the reference. Residuals are not subtracted from forecast error "
                "as if a known additive-noise variance model had been established."
            ),

        "claim_guardrails": [
            "F50/F95 are BIRDSAI TestReal post-correction geometric residual references.",
            "Do not call F50/F95 universal irreducible annotation noise.",
            "Do not transfer these pixel references directly to Fresh WildBox or BAMBI.",
            "A forecast error above F95 indicates it exceeds the measured residual envelope, not that all excess is model error.",
        ],

        "outputs": {
            "sample_metrics":
                str(
                    sample_path
                ),

            "pointwise_errors":
                str(
                    point_path
                ),

            "summary":
                str(
                    summary_path
                ),

            "segment_cluster_bootstrap":
                str(
                    bootstrap_path
                ),

            "leave_one_sequence_out":
                str(
                    loo_path
                ),

            "r10b_reproducibility":
                str(
                    reproducibility_path
                ),
        },
    }

    (
        OUT
        /
        "R11C_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "TESTREAL FORECAST ERROR VS RESIDUAL-FLOOR SUMMARY:"
    )

    print(
        summary_df[
            [
                "task",
                "samples",
                "segments",
                "sequences",
                "F50_px",
                "F95_px",
                "ensemble_ADE_px_window_mean",
                "ensemble_FDE_px_window_mean",
                "window_mean_ADE_over_F95",
                "window_mean_FDE_over_F95",
                "all_future_points_above_F95_pct",
                "samples_ADE_above_F95_pct",
                "samples_FDE_above_F95_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nPRIMARY SEGMENT-CLUSTER BOOTSTRAP:"
    )

    primary = bootstrap_df[
        bootstrap_df[
            "metric"
        ].isin(
            [
                "fraction_of_samples_ADE_above_F95",
                "fraction_of_samples_FDE_above_F95",
                "mean_sample_fraction_future_points_above_F95",
                "ADE_over_F95_ratio",
                "FDE_over_F95_ratio",
            ]
        )
    ]

    print(
        primary.to_string(
            index=False
        )
    )

    print(
        "\nR10-B REPRODUCIBILITY CHECK:"
    )

    print(
        reproducibility_df.to_string(
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
        "\nR11-C COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
