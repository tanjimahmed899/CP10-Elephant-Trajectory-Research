
from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"

R11B_DIR = (
    REVISION_ROOT
    / "R11_noise_floor_analysis"
    / "R11B_empirical_geometric_residual_floor_protocol_lock"
)
R11B_MANIFEST = R11B_DIR / "R11B_MANIFEST.json"

R11C_DIR = (
    REVISION_ROOT
    / "R11_noise_floor_analysis"
    / "R11C_testreal_forecast_error_vs_geometric_residual_floor"
)
R11C_MANIFEST = R11C_DIR / "R11C_MANIFEST.json"

R10B_DIR = (
    REVISION_ROOT
    / "R10_ensemble_analysis"
    / "R10B_testreal_transformer_ensemble_value"
)
R10B_SAMPLE = R10B_DIR / "R10B_TESTREAL_ENSEMBLE_SAMPLE_METRICS.csv"
R10B_BOOT = R10B_DIR / "R10B_PAIRED_SEGMENT_CLUSTER_BOOTSTRAP.csv"

OUT = (
    REVISION_ROOT
    / "R11_noise_floor_analysis"
    / "R11D_ensemble_gain_vs_residual_scale"
)
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "samples": 1053,
        "segments": 89,
        "sequences": 6,
    },
    "2s_to_5s": {
        "samples": 826,
        "segments": 63,
        "sequences": 5,
    },
}

BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20261029


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def verify_upstream():
    for p in [
        R11B_MANIFEST,
        R11C_MANIFEST,
        R10B_SAMPLE,
        R10B_BOOT,
    ]:
        require(p)

    b = json.loads(
        R11B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    c = json.loads(
        R11C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_b = (
        "R11B_TESTREAL_EMPIRICAL_GEOMETRIC_RESIDUAL_F50_F95_LOCKED_READY_FOR_R11C_FORECAST_ERROR_ABOVE_FLOOR_ANALYSIS"
    )

    expected_c = (
        "R11C_TESTREAL_FORECAST_ERROR_RELATIVE_TO_EMPIRICAL_RESIDUAL_FLOOR_QUANTIFIED_READY_FOR_R11D_NOISE_FLOOR_CONCLUSION"
    )

    if b.get("status") != expected_b:
        raise RuntimeError(
            f"Unexpected R11-B status: {b.get('status')}"
        )

    if c.get("status") != expected_c:
        raise RuntimeError(
            f"Unexpected R11-C status: {c.get('status')}"
        )

    return b, c


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


def segment_gain_table(df, gain_col):
    return (
        df.groupby(
            [
                "sequence",
                "segment_id",
            ],
            dropna=False,
            as_index=False
        )
        .agg(
            segment_gain_px=(
                gain_col,
                "median"
            ),
            windows=(
                "sample_id",
                "size"
            ),
        )
    )


def cluster_bootstrap_gain(
    df,
    gain_col,
    rng_seed,
):
    seg = segment_gain_table(
        df,
        gain_col
    )

    vals = seg[
        "segment_gain_px"
    ].to_numpy(
        dtype=np.float64
    )

    n = len(vals)

    point = float(
        vals.mean()
    )

    rng = np.random.default_rng(
        rng_seed
    )

    boot = np.empty(
        BOOTSTRAP_B,
        dtype=np.float64
    )

    for i in range(
        BOOTSTRAP_B
    ):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        boot[i] = float(
            vals[idx].mean()
        )

    low, high = percentile_ci(
        boot
    )

    return {
        "segments": int(n),
        "gain_px": point,
        "ci95_low": low,
        "ci95_high": high,
    }


def leave_one_sequence_out(
    df,
    gain_col,
):
    rows = []

    for held_out in sorted(
        df["sequence"].astype(str).unique()
    ):
        kept = df[
            df["sequence"].astype(str)
            !=
            held_out
        ].copy()

        seg = segment_gain_table(
            kept,
            gain_col
        )

        rows.append(
            {
                "held_out_sequence":
                    held_out,

                "remaining_segments":
                    int(
                        len(seg)
                    ),

                "gain_px_segment_median_macro":
                    float(
                        seg[
                            "segment_gain_px"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR11-D — ENSEMBLE GAIN MAGNITUDE VS EMPIRICAL GEOMETRIC RESIDUAL SCALE"
    )

    print(
        "NO training and NO inference. This step asks a different question from R11-C: "
        "even though absolute forecasting error is far above the measured residual envelope, "
        "are the statistically significant R10 ensemble gains themselves large or small relative "
        "to the same BIRDSAI TestReal residual scale? Signed paired gain is defined as "
        "comparator error - ensemble error, so positive values favor the ensemble. "
        "The BIRDSAI F50/F95 references are used only for scale contextualization, not as "
        "significance thresholds and not as quantities to subtract from model error.\n"
    )

    b, _ = verify_upstream()

    f50 = float(
        b["F50_px"]
    )

    f95 = float(
        b["F95_px"]
    )

    sample = pd.read_csv(
        R10B_SAMPLE,
        dtype={
            "task": str,
            "sample_id": str,
            "sequence": str,
            "segment_id": str,
            "track_key": str,
        }
    )

    required = {
        "task",
        "sample_id",
        "sequence",
        "segment_id",
        "ensemble_ADE_px",
        "ensemble_FDE_px",
        "mean_individual_seed_ADE_px",
        "mean_individual_seed_FDE_px",
        "validation_selected_seed_ADE_px",
        "validation_selected_seed_FDE_px",
    }

    missing = required - set(
        sample.columns
    )

    if missing:
        raise RuntimeError(
            f"R10-B sample file missing columns: {sorted(missing)}"
        )

    comparisons = [
        (
            "mean_individual_seed",
            "ADE",
            "mean_individual_seed_ADE_px",
            "ensemble_ADE_px",
        ),
        (
            "mean_individual_seed",
            "FDE",
            "mean_individual_seed_FDE_px",
            "ensemble_FDE_px",
        ),
        (
            "validation_selected_seed",
            "ADE",
            "validation_selected_seed_ADE_px",
            "ensemble_ADE_px",
        ),
        (
            "validation_selected_seed",
            "FDE",
            "validation_selected_seed_FDE_px",
            "ensemble_FDE_px",
        ),
    ]

    summary_rows = []
    bootstrap_rows = []
    loo_rows = []
    sample_rows = []

    for task_index, task in enumerate(
        TASKS
    ):
        tdf = sample[
            sample["task"] == task
        ].copy()

        cfg = TASKS[task]

        if len(tdf) != cfg["samples"]:
            raise RuntimeError(
                f"{task}: samples={len(tdf)}, expected={cfg['samples']}"
            )

        segments = tdf.groupby(
            [
                "sequence",
                "segment_id",
            ],
            dropna=False
        ).ngroups

        if int(segments) != cfg["segments"]:
            raise RuntimeError(
                f"{task}: segments={segments}, expected={cfg['segments']}"
            )

        for comp_index, (
            comparator,
            metric,
            comparator_col,
            ensemble_col,
        ) in enumerate(
            comparisons
        ):
            gain_col = (
                f"gain_{comparator}_{metric}_px"
            )

            tdf[
                gain_col
            ] = (
                pd.to_numeric(
                    tdf[
                        comparator_col
                    ],
                    errors="raise"
                )
                -
                pd.to_numeric(
                    tdf[
                        ensemble_col
                    ],
                    errors="raise"
                )
            )

            gain = tdf[
                gain_col
            ].to_numpy(
                dtype=np.float64
            )

            result = cluster_bootstrap_gain(
                tdf,
                gain_col,
                BOOTSTRAP_SEED
                +
                task_index
                *
                1000
                +
                comp_index,
            )

            summary_rows.append(
                {
                    "task":
                        task,

                    "comparator":
                        comparator,

                    "metric":
                        metric,

                    "samples":
                        int(
                            len(
                                tdf
                            )
                        ),

                    "segments":
                        int(
                            segments
                        ),

                    "window_mean_signed_gain_px":
                        float(
                            gain.mean()
                        ),

                    "window_median_signed_gain_px":
                        float(
                            np.median(
                                gain
                            )
                        ),

                    "segment_median_macro_gain_px":
                        result[
                            "gain_px"
                        ],

                    "gain_over_F50":
                        result[
                            "gain_px"
                        ]
                        /
                        f50,

                    "gain_over_F95":
                        result[
                            "gain_px"
                        ]
                        /
                        f95,

                    "samples_gain_positive_pct":
                        float(
                            100.0
                            *
                            (
                                gain > 0
                            ).mean()
                        ),

                    "samples_gain_exceeds_F50_pct":
                        float(
                            100.0
                            *
                            (
                                gain > f50
                            ).mean()
                        ),

                    "samples_gain_exceeds_F95_pct":
                        float(
                            100.0
                            *
                            (
                                gain > f95
                            ).mean()
                        ),

                    "samples_worsening_exceeds_F50_pct":
                        float(
                            100.0
                            *
                            (
                                gain < -f50
                            ).mean()
                        ),

                    "samples_worsening_exceeds_F95_pct":
                        float(
                            100.0
                            *
                            (
                                gain < -f95
                            ).mean()
                        ),
                }
            )

            bootstrap_rows.append(
                {
                    "task":
                        task,

                    "comparator":
                        comparator,

                    "metric":
                        metric,

                    "segments":
                        result[
                            "segments"
                        ],

                    "gain_px":
                        result[
                            "gain_px"
                        ],

                    "ci95_low":
                        result[
                            "ci95_low"
                        ],

                    "ci95_high":
                        result[
                            "ci95_high"
                        ],

                    "gain_over_F50":
                        result[
                            "gain_px"
                        ]
                        /
                        f50,

                    "gain_over_F95":
                        result[
                            "gain_px"
                        ]
                        /
                        f95,

                    "ci95_low_over_F50":
                        result[
                            "ci95_low"
                        ]
                        /
                        f50,

                    "ci95_high_over_F50":
                        result[
                            "ci95_high"
                        ]
                        /
                        f50,

                    "ci95_low_over_F95":
                        result[
                            "ci95_low"
                        ]
                        /
                        f95,

                    "ci95_high_over_F95":
                        result[
                            "ci95_high"
                        ]
                        /
                        f95,

                    "direction":
                        (
                            "ENSEMBLE_GAIN_CI_EXCLUDES_ZERO"
                            if result[
                                "ci95_low"
                            ]
                            >
                            0
                            else
                            "NO_CLEAR_ENSEMBLE_GAIN"
                        ),
                }
            )

            loo = leave_one_sequence_out(
                tdf,
                gain_col,
            )

            loo[
                "task"
            ] = task

            loo[
                "comparator"
            ] = comparator

            loo[
                "metric"
            ] = metric

            loo[
                "gain_over_F50"
            ] = (
                loo[
                    "gain_px_segment_median_macro"
                ]
                /
                f50
            )

            loo[
                "gain_over_F95"
            ] = (
                loo[
                    "gain_px_segment_median_macro"
                ]
                /
                f95
            )

            loo_rows.append(
                loo
            )

            keep_cols = [
                "task",
                "sample_id",
                "sequence",
                "segment_id",
                "track_key",
            ]

            tmp = tdf[
                keep_cols
            ].copy()

            tmp[
                "comparator"
            ] = comparator

            tmp[
                "metric"
            ] = metric

            tmp[
                "signed_gain_px"
            ] = gain

            tmp[
                "gain_over_F50"
            ] = gain / f50

            tmp[
                "gain_over_F95"
            ] = gain / f95

            sample_rows.append(
                tmp
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    loo_df = pd.concat(
        loo_rows,
        ignore_index=True
    )

    long_sample_df = pd.concat(
        sample_rows,
        ignore_index=True
    )

    # Cross-check R10-B paired effect signs/magnitudes.
    r10b_boot = pd.read_csv(
        R10B_BOOT
    )

    crosscheck_rows = []

    for _, row in bootstrap_df.iterrows():
        comp_name = (
            "ensemble_vs_mean_individual_seed"
            if row["comparator"]
            ==
            "mean_individual_seed"
            else
            "ensemble_vs_validation_selected_seed"
        )

        ref = r10b_boot[
            (
                r10b_boot[
                    "task"
                ].astype(str)
                ==
                str(
                    row[
                        "task"
                    ]
                )
            )
            &
            (
                r10b_boot[
                    "comparator"
                ].astype(str)
                ==
                comp_name
            )
            &
            (
                r10b_boot[
                    "metric"
                ].astype(str)
                ==
                str(
                    row[
                        "metric"
                    ]
                )
            )
        ]

        if len(ref) != 1:
            raise RuntimeError(
                f"Missing R10-B reference for {row['task']} {comp_name} {row['metric']}"
            )

        ref_row = ref.iloc[0]

        # R10-B uses ensemble - comparator; R11-D uses comparator - ensemble.
        expected_gain = -float(
            ref_row[
                "delta_px"
            ]
        )

        crosscheck_rows.append(
            {
                "task":
                    row[
                        "task"
                    ],

                "comparator":
                    row[
                        "comparator"
                    ],

                "metric":
                    row[
                        "metric"
                    ],

                "R11D_gain_px":
                    float(
                        row[
                            "gain_px"
                        ]
                    ),

                "negative_R10B_delta_px":
                    expected_gain,

                "abs_difference":
                    abs(
                        float(
                            row[
                                "gain_px"
                            ]
                        )
                        -
                        expected_gain
                    ),
            }
        )

    crosscheck_df = pd.DataFrame(
        crosscheck_rows
    )

    if crosscheck_df[
        "abs_difference"
    ].max() > 1e-12:
        raise RuntimeError(
            "R11-D gain does not exactly reproduce the sign-reversed R10-B paired effect."
        )

    summary_path = (
        OUT
        /
        "R11D_ENSEMBLE_GAIN_VS_RESIDUAL_SCALE_SUMMARY.csv"
    )

    bootstrap_path = (
        OUT
        /
        "R11D_SEGMENT_CLUSTER_BOOTSTRAP_GAIN_SCALE.csv"
    )

    loo_path = (
        OUT
        /
        "R11D_LEAVE_ONE_SEQUENCE_OUT_GAIN_SCALE.csv"
    )

    sample_path = (
        OUT
        /
        "R11D_SAMPLE_LEVEL_GAIN_VS_RESIDUAL_SCALE.csv"
    )

    crosscheck_path = (
        OUT
        /
        "R11D_R10B_EFFECT_REPRODUCIBILITY.csv"
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

    long_sample_df.to_csv(
        sample_path,
        index=False
    )

    crosscheck_df.to_csv(
        crosscheck_path,
        index=False
    )

    status = (
        "R11D_ENSEMBLE_GAIN_CONTEXTUALIZED_AGAINST_F50_F95_READY_FOR_R11E_FINAL_NOISE_FLOOR_CONCLUSION"
    )

    manifest = {
        "step":
            "R11D_ENSEMBLE_GAIN_MAGNITUDE_VS_EMPIRICAL_GEOMETRIC_RESIDUAL_SCALE",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "F50_px":
            f50,

        "F95_px":
            f95,

        "gain_definition":
            "comparator error - ensemble error; positive favors ensemble",

        "primary_dependency_unit":
            "(sequence, segment_id)",

        "within_segment_reduction":
            "median paired gain across overlapping windows",

        "bootstrap_replicates":
            BOOTSTRAP_B,

        "important_interpretation_rule":
            (
                "F50/F95 are contextual scales, not statistical significance thresholds. "
                "A gain below F95 can still be statistically reproducible; it should simply be "
                "described as small relative to the conservative residual envelope."
            ),

        "outputs": {
            "summary":
                str(
                    summary_path
                ),

            "bootstrap":
                str(
                    bootstrap_path
                ),

            "leave_one_sequence_out":
                str(
                    loo_path
                ),

            "sample_level":
                str(
                    sample_path
                ),

            "r10b_reproducibility":
                str(
                    crosscheck_path
                ),
        },
    }

    (
        OUT
        /
        "R11D_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "ENSEMBLE GAIN VS RESIDUAL SCALE SUMMARY:"
    )

    print(
        summary_df[
            [
                "task",
                "comparator",
                "metric",
                "segment_median_macro_gain_px",
                "gain_over_F50",
                "gain_over_F95",
                "samples_gain_positive_pct",
                "samples_gain_exceeds_F50_pct",
                "samples_gain_exceeds_F95_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nPRIMARY SEGMENT-CLUSTER BOOTSTRAP GAIN SCALE:"
    )

    print(
        bootstrap_df.to_string(
            index=False
        )
    )

    print(
        "\nLEAVE-ONE-SEQUENCE-OUT GAIN SCALE:"
    )

    print(
        loo_df.to_string(
            index=False
        )
    )

    print(
        "\nR10-B EFFECT REPRODUCIBILITY:"
    )

    print(
        crosscheck_df.to_string(
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
        "\nR11-D COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
