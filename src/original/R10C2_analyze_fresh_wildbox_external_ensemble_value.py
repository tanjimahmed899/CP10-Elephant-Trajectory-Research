
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R10_ROOT = REVISION_ROOT / "R10_ensemble_analysis"

R10B_DIR = R10_ROOT / "R10B_testreal_transformer_ensemble_value"
R10B_MANIFEST = R10B_DIR / "R10B_MANIFEST.json"
R10B_SUMMARY = R10B_DIR / "R10B_TESTREAL_ENSEMBLE_SUMMARY.csv"

R10C1_DIR = (
    R10_ROOT
    / "R10C1_fresh_wildbox_frozen_transformer_reinference"
)
R10C1_MANIFEST = R10C1_DIR / "R10C1_MANIFEST.json"
R10C1_SEED_METRICS = (
    R10C1_DIR
    / "R10C1_FRESH_WILDBOX_SEED_LEVEL_METRICS.csv"
)

OUT = (
    R10_ROOT
    / "R10C2_fresh_wildbox_external_ensemble_value"
)
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "samples": 865,
        "tracks": 78,
        "sources": 3,
    },
    "2s_to_5s": {
        "samples": 637,
        "tracks": 71,
        "sources": 3,
    },
}

SEEDS = [42, 52, 62]
BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20261029


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


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
        R10B_MANIFEST,
        R10B_SUMMARY,
        R10C1_MANIFEST,
        R10C1_SEED_METRICS,
    ]:
        require(p)

    b = json.loads(
        R10B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    c1 = json.loads(
        R10C1_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_b = (
        "R10B_TESTREAL_TRANSFORMER_ENSEMBLE_VALUE_QUANTIFIED_READY_FOR_R10C_FRESH_WILDBOX_TRANSFORMER_REINFERENCE"
    )

    allowed_c1 = {
        "R10C1_FRESH_WILDBOX_6_FROZEN_TRANSFORMER_RUNS_COMPLETE_ENSEMBLE_REPRODUCED_READY_FOR_R10C2_EXTERNAL_ENSEMBLE_VALUE_BOOTSTRAP",
        "R10C1_FRESH_WILDBOX_6_FROZEN_TRANSFORMER_RUNS_COMPLETE_R2_ENSEMBLE_ALIGNED_WITH_SMALL_NUMERICAL_DIFFERENCE_READY_FOR_R10C2",
    }

    if b.get("status") != expected_b:
        raise RuntimeError(
            f"Unexpected R10-B status: {b.get('status')}"
        )

    if c1.get("status") not in allowed_c1:
        raise RuntimeError(
            f"Unexpected R10-C1 status: {c1.get('status')}"
        )

    if int(c1.get("runs", -1)) != 6:
        raise RuntimeError(
            "R10-C1 did not complete exactly 6 Transformer runs."
        )

    return b, c1


def load_seed_run_map(c1_manifest):
    mapping = {}

    seed_csv = pd.read_csv(
        R10C1_SEED_METRICS,
        dtype={
            "task": str,
            "sample_metrics_csv": str,
            "prediction_npz": str,
        }
    )

    if len(seed_csv) != 6:
        raise RuntimeError(
            f"Expected 6 seed rows, got {len(seed_csv)}."
        )

    for _, row in seed_csv.iterrows():
        key = (
            str(row["task"]),
            int(row["seed"]),
        )

        sm = Path(
            str(row["sample_metrics_csv"])
        )

        pred = Path(
            str(row["prediction_npz"])
        )

        require(sm)
        require(pred)

        if (
            "sample_metrics_sha256" in row.index
            and
            pd.notna(
                row["sample_metrics_sha256"]
            )
        ):
            if sha256_file(sm) != str(
                row["sample_metrics_sha256"]
            ):
                raise RuntimeError(
                    f"Sample metrics hash changed: {sm}"
                )

        if (
            "prediction_npz_sha256" in row.index
            and
            pd.notna(
                row["prediction_npz_sha256"]
            )
        ):
            if sha256_file(pred) != str(
                row["prediction_npz_sha256"]
            ):
                raise RuntimeError(
                    f"Prediction hash changed: {pred}"
                )

        mapping[key] = {
            "sample_metrics_csv": sm,
            "prediction_npz": pred,
            "best_validation_track_macro_ADE": float(
                row[
                    "best_validation_track_macro_ADE"
                ]
            ),
            "best_epoch": int(
                row[
                    "best_epoch"
                ]
            ),
        }

    return mapping


def load_sample_metrics(path):
    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "source_group": str,
            "track_key": str,
            "segment": str,
        }
    )

    required = {
        "sample_id",
        "source_group",
        "track_key",
        "ADE",
        "FDE",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"{path.name}: missing columns {sorted(missing)}"
        )

    return df


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


def build_task_sample_table(
    task,
    run_map,
):
    sample_tables = {}
    predictions = {}
    truths = {}

    validation_rows = []

    for seed in SEEDS:
        info = run_map[
            (
                task,
                seed,
            )
        ]

        sm = load_sample_metrics(
            info["sample_metrics_csv"]
        )

        expected = TASKS[
            task
        ]["samples"]

        if len(sm) != expected:
            raise RuntimeError(
                f"{task}/seed{seed}: sample rows={len(sm)}, expected={expected}"
            )

        sample_tables[
            seed
        ] = sm

        truth, pred = load_prediction(
            info["prediction_npz"]
        )

        if truth.shape[0] != expected:
            raise RuntimeError(
                f"{task}/seed{seed}: prediction samples mismatch."
            )

        truths[
            seed
        ] = truth

        predictions[
            seed
        ] = pred

        validation_rows.append(
            {
                "task": task,
                "seed": seed,
                "best_epoch": info["best_epoch"],
                "best_validation_track_macro_ADE":
                    info[
                        "best_validation_track_macro_ADE"
                    ],
            }
        )

    base = sample_tables[
        SEEDS[0]
    ][
        [
            "sample_id",
            "source_group",
            "track_key",
        ]
        +
        (
            ["segment"]
            if "segment" in sample_tables[
                SEEDS[0]
            ].columns
            else
            []
        )
    ].copy()

    if base["sample_id"].duplicated().any():
        raise RuntimeError(
            f"{task}: duplicate sample_id."
        )

    for seed in SEEDS:
        sm = sample_tables[
            seed
        ]

        if not np.array_equal(
            sm["sample_id"].astype(str).to_numpy(),
            base["sample_id"].astype(str).to_numpy(),
        ):
            raise RuntimeError(
                f"{task}: sample order differs across seeds."
            )

    truth0 = truths[
        SEEDS[0]
    ]

    for seed in SEEDS[1:]:
        if not np.array_equal(
            truths[seed],
            truth0
        ):
            raise RuntimeError(
                f"{task}: truth differs across seeds."
            )

    validation_selected_seed = min(
        validation_rows,
        key=lambda r:
            r[
                "best_validation_track_macro_ADE"
            ],
    )[
        "seed"
    ]

    ensemble_pred = np.mean(
        np.stack(
            [
                predictions[
                    seed
                ]
                for seed in SEEDS
            ],
            axis=0
        ),
        axis=0
    )

    ensemble_err = np.linalg.norm(
        ensemble_pred
        -
        truth0,
        axis=2
    )

    ensemble_ade = ensemble_err.mean(
        axis=1
    )

    ensemble_fde = ensemble_err[
        :,
        -1
    ]

    for seed in SEEDS:
        base[
            f"seed{seed}_ADE"
        ] = sample_tables[
            seed
        ][
            "ADE"
        ].to_numpy(
            dtype=np.float64
        )

        base[
            f"seed{seed}_FDE"
        ] = sample_tables[
            seed
        ][
            "FDE"
        ].to_numpy(
            dtype=np.float64
        )

    base[
        "mean_individual_seed_ADE"
    ] = base[
        [
            f"seed{seed}_ADE"
            for seed in SEEDS
        ]
    ].mean(
        axis=1
    )

    base[
        "mean_individual_seed_FDE"
    ] = base[
        [
            f"seed{seed}_FDE"
            for seed in SEEDS
        ]
    ].mean(
        axis=1
    )

    base[
        "validation_selected_seed"
    ] = validation_selected_seed

    base[
        "validation_selected_seed_ADE"
    ] = base[
        f"seed{validation_selected_seed}_ADE"
    ]

    base[
        "validation_selected_seed_FDE"
    ] = base[
        f"seed{validation_selected_seed}_FDE"
    ]

    base[
        "ensemble_ADE"
    ] = ensemble_ade

    base[
        "ensemble_FDE"
    ] = ensemble_fde

    base[
        "delta_ensemble_minus_mean_seed_ADE"
    ] = (
        base[
            "ensemble_ADE"
        ]
        -
        base[
            "mean_individual_seed_ADE"
        ]
    )

    base[
        "delta_ensemble_minus_mean_seed_FDE"
    ] = (
        base[
            "ensemble_FDE"
        ]
        -
        base[
            "mean_individual_seed_FDE"
        ]
    )

    base[
        "delta_ensemble_minus_validation_seed_ADE"
    ] = (
        base[
            "ensemble_ADE"
        ]
        -
        base[
            "validation_selected_seed_ADE"
        ]
    )

    base[
        "delta_ensemble_minus_validation_seed_FDE"
    ] = (
        base[
            "ensemble_FDE"
        ]
        -
        base[
            "validation_selected_seed_FDE"
        ]
    )

    for seed in SEEDS:
        base[
            f"delta_ensemble_minus_seed{seed}_ADE"
        ] = (
            base[
                "ensemble_ADE"
            ]
            -
            base[
                f"seed{seed}_ADE"
            ]
        )

        base[
            f"delta_ensemble_minus_seed{seed}_FDE"
        ] = (
            base[
                "ensemble_FDE"
            ]
            -
            base[
                f"seed{seed}_FDE"
            ]
        )

    # Native-coordinate pairwise prediction diversity.
    pairwise_ade = []
    pairwise_fde = []

    diversity_rows = []

    for a, b in [
        (42, 52),
        (42, 62),
        (52, 62),
    ]:
        dist = np.linalg.norm(
            predictions[a]
            -
            predictions[b],
            axis=2
        )

        d_ade = dist.mean(
            axis=1
        )

        d_fde = dist[
            :,
            -1
        ]

        pairwise_ade.append(
            d_ade
        )

        pairwise_fde.append(
            d_fde
        )

        diversity_rows.append(
            {
                "task": task,
                "seed_a": a,
                "seed_b": b,
                "mean_pairwise_trajectory_disagreement":
                    float(
                        d_ade.mean()
                    ),
                "mean_pairwise_endpoint_disagreement":
                    float(
                        d_fde.mean()
                    ),
            }
        )

    base[
        "mean_pairwise_trajectory_disagreement"
    ] = np.mean(
        np.stack(
            pairwise_ade,
            axis=0
        ),
        axis=0
    )

    base[
        "mean_pairwise_endpoint_disagreement"
    ] = np.mean(
        np.stack(
            pairwise_fde,
            axis=0
        ),
        axis=0
    )

    return (
        base,
        pd.DataFrame(
            validation_rows
        ),
        pd.DataFrame(
            diversity_rows
        ),
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


def track_cluster_bootstrap(
    df,
    value_col,
    seed,
):
    track = (
        df.groupby(
            "track_key",
            dropna=False,
            as_index=False
        )
        .agg(
            track_effect=(
                value_col,
                "median"
            ),
            windows=(
                "sample_id",
                "size"
            ),
            source_group=(
                "source_group",
                "first"
            ),
        )
    )

    n = len(
        track
    )

    values = track[
        "track_effect"
    ].to_numpy(
        dtype=np.float64
    )

    point = float(
        values.mean()
    )

    rng = np.random.default_rng(
        seed
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

    if high < 0:
        direction = (
            "ENSEMBLE_BETTER_CI_EXCLUDES_ZERO"
        )

    elif low > 0:
        direction = (
            "ENSEMBLE_WORSE_CI_EXCLUDES_ZERO"
        )

    else:
        direction = (
            "NO_CLEAR_DIFFERENCE_CI_INCLUDES_ZERO"
        )

    return {
        "tracks": int(
            n
        ),
        "delta": point,
        "ci95_low": low,
        "ci95_high": high,
        "direction": direction,
    }


def leave_one_source_out(
    df,
    value_col,
):
    rows = []

    sources = sorted(
        df[
            "source_group"
        ].astype(str).unique()
    )

    for held_out in sources:
        kept = df[
            df[
                "source_group"
            ].astype(str)
            !=
            held_out
        ].copy()

        track = (
            kept.groupby(
                "track_key",
                dropna=False,
                as_index=False
            )
            .agg(
                track_effect=(
                    value_col,
                    "median"
                )
            )
        )

        rows.append(
            {
                "held_out_source":
                    held_out,

                "remaining_tracks":
                    int(
                        len(
                            track
                        )
                    ),

                "delta_track_median_macro":
                    float(
                        track[
                            "track_effect"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def spearman_via_ranks(x, y):
    sx = pd.Series(
        x
    )

    sy = pd.Series(
        y
    )

    if (
        sx.nunique(
            dropna=True
        )
        <=
        1
        or
        sy.nunique(
            dropna=True
        )
        <=
        1
    ):
        return np.nan

    return float(
        sx.rank(
            method="average"
        ).corr(
            sy.rank(
                method="average"
            )
        )
    )


def percent_gain(
    ensemble,
    comparator,
):
    if comparator == 0:
        return np.nan

    return float(
        100.0
        *
        (
            comparator
            -
            ensemble
        )
        /
        comparator
    )


def main():
    print(
        "\nR10-C2 — FRESH WILDBOX EXTERNAL TRANSFORMER ENSEMBLE VALUE"
    )

    print(
        "NO training and NO new inference. This step uses the six frozen fresh-WildBox "
        "Transformer predictions produced in R10-C1. The actual 3-seed prediction ensemble "
        "is compared against (1) the mean individual-seed error and (2) the validation-selected "
        "single seed. Because WildBox windows overlap within tracks, the primary dependency "
        "unit is track_key: paired sample differences are collapsed to within-track medians, "
        "then bootstrapped by track for 20,000 replicates. Leave-one-source-out sensitivity "
        "checks whether the conclusion depends on any one of the three fresh WildBox videos. "
        "Delta = ensemble - comparator; negative favors the ensemble.\n"
    )

    _, c1_manifest = verify_upstream()

    run_map = load_seed_run_map(
        c1_manifest
    )

    all_sample = []
    all_validation = []
    all_diversity = []
    bootstrap_rows = []
    loo_rows = []
    summary_rows = []
    correlation_rows = []

    comparison_specs = [
        (
            "ensemble_vs_mean_individual_seed",
            "ADE",
            "delta_ensemble_minus_mean_seed_ADE",
        ),
        (
            "ensemble_vs_mean_individual_seed",
            "FDE",
            "delta_ensemble_minus_mean_seed_FDE",
        ),
        (
            "ensemble_vs_validation_selected_seed",
            "ADE",
            "delta_ensemble_minus_validation_seed_ADE",
        ),
        (
            "ensemble_vs_validation_selected_seed",
            "FDE",
            "delta_ensemble_minus_validation_seed_FDE",
        ),
    ]

    for task_index, task in enumerate(
        TASKS
    ):
        (
            sample,
            validation,
            diversity,
        ) = build_task_sample_table(
            task,
            run_map,
        )

        if int(
            sample[
                "track_key"
            ].nunique()
        ) != TASKS[
            task
        ][
            "tracks"
        ]:
            raise RuntimeError(
                f"{task}: track count mismatch."
            )

        if int(
            sample[
                "source_group"
            ].nunique()
        ) != TASKS[
            task
        ][
            "sources"
        ]:
            raise RuntimeError(
                f"{task}: source count mismatch."
            )

        sample[
            "task"
        ] = task

        all_sample.append(
            sample
        )

        all_validation.append(
            validation
        )

        all_diversity.append(
            diversity
        )

        validation_seed = int(
            sample[
                "validation_selected_seed"
            ].iloc[
                0
            ]
        )

        ensemble_ade = float(
            sample[
                "ensemble_ADE"
            ].mean()
        )

        ensemble_fde = float(
            sample[
                "ensemble_FDE"
            ].mean()
        )

        mean_seed_ade = float(
            sample[
                "mean_individual_seed_ADE"
            ].mean()
        )

        mean_seed_fde = float(
            sample[
                "mean_individual_seed_FDE"
            ].mean()
        )

        val_ade = float(
            sample[
                "validation_selected_seed_ADE"
            ].mean()
        )

        val_fde = float(
            sample[
                "validation_selected_seed_FDE"
            ].mean()
        )

        summary_rows.append(
            {
                "task":
                    task,

                "samples":
                    int(
                        len(
                            sample
                        )
                    ),

                "tracks":
                    int(
                        sample[
                            "track_key"
                        ].nunique()
                    ),

                "sources":
                    int(
                        sample[
                            "source_group"
                        ].nunique()
                    ),

                "validation_selected_seed":
                    validation_seed,

                "ensemble_ADE_window_mean":
                    ensemble_ade,

                "ensemble_FDE_window_mean":
                    ensemble_fde,

                "mean_individual_seed_ADE_window_mean":
                    mean_seed_ade,

                "mean_individual_seed_FDE_window_mean":
                    mean_seed_fde,

                "validation_selected_seed_ADE_window_mean":
                    val_ade,

                "validation_selected_seed_FDE_window_mean":
                    val_fde,

                "ensemble_gain_vs_mean_seed_ADE":
                    ensemble_ade
                    -
                    mean_seed_ade,

                "ensemble_gain_vs_mean_seed_FDE":
                    ensemble_fde
                    -
                    mean_seed_fde,

                "ensemble_gain_vs_validation_seed_ADE":
                    ensemble_ade
                    -
                    val_ade,

                "ensemble_gain_vs_validation_seed_FDE":
                    ensemble_fde
                    -
                    val_fde,

                "percent_improvement_vs_mean_seed_ADE":
                    percent_gain(
                        ensemble_ade,
                        mean_seed_ade
                    ),

                "percent_improvement_vs_mean_seed_FDE":
                    percent_gain(
                        ensemble_fde,
                        mean_seed_fde
                    ),

                "percent_improvement_vs_validation_seed_ADE":
                    percent_gain(
                        ensemble_ade,
                        val_ade
                    ),

                "percent_improvement_vs_validation_seed_FDE":
                    percent_gain(
                        ensemble_fde,
                        val_fde
                    ),
            }
        )

        for spec_index, (
            comparator,
            metric,
            value_col,
        ) in enumerate(
            comparison_specs
        ):
            result = track_cluster_bootstrap(
                sample,
                value_col,
                BOOTSTRAP_SEED
                +
                task_index
                *
                1000
                +
                spec_index,
            )

            bootstrap_rows.append(
                {
                    "task":
                        task,

                    "comparator":
                        comparator,

                    "metric":
                        metric,

                    "dependency_unit":
                        "track_key",

                    "within_track_reduction":
                        "median paired difference",

                    "tracks":
                        result[
                            "tracks"
                        ],

                    "delta":
                        result[
                            "delta"
                        ],

                    "ci95_low":
                        result[
                            "ci95_low"
                        ],

                    "ci95_high":
                        result[
                            "ci95_high"
                        ],

                    "direction":
                        result[
                            "direction"
                        ],
                }
            )

            loo = leave_one_source_out(
                sample,
                value_col,
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

            loo_rows.append(
                loo
            )

        correlation_rows.extend(
            [
                {
                    "task":
                        task,

                    "gain_target":
                        "ensemble_minus_mean_seed_ADE",

                    "spearman_rho_diversity_vs_gain":
                        spearman_via_ranks(
                            sample[
                                "mean_pairwise_trajectory_disagreement"
                            ],
                            sample[
                                "delta_ensemble_minus_mean_seed_ADE"
                            ],
                        ),

                    "note":
                        (
                            "Negative gain favors ensemble; negative rho means greater "
                            "seed disagreement tends to coincide with larger ensemble benefit."
                        ),
                },
                {
                    "task":
                        task,

                    "gain_target":
                        "ensemble_minus_mean_seed_FDE",

                    "spearman_rho_diversity_vs_gain":
                        spearman_via_ranks(
                            sample[
                                "mean_pairwise_endpoint_disagreement"
                            ],
                            sample[
                                "delta_ensemble_minus_mean_seed_FDE"
                            ],
                        ),

                    "note":
                        (
                            "Negative gain favors ensemble; negative rho means greater "
                            "seed disagreement tends to coincide with larger ensemble benefit."
                        ),
                },
            ]
        )

    sample_df = pd.concat(
        all_sample,
        ignore_index=True
    )

    validation_df = pd.concat(
        all_validation,
        ignore_index=True
    )

    diversity_df = pd.concat(
        all_diversity,
        ignore_index=True
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    loo_df = pd.concat(
        loo_rows,
        ignore_index=True
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    sample_path = (
        OUT
        /
        "R10C2_FRESH_WILDBOX_ENSEMBLE_SAMPLE_METRICS.csv"
    )

    validation_path = (
        OUT
        /
        "R10C2_VALIDATION_SELECTED_SEED_LOCK.csv"
    )

    diversity_path = (
        OUT
        /
        "R10C2_PAIRWISE_SEED_DIVERSITY.csv"
    )

    bootstrap_path = (
        OUT
        /
        "R10C2_TRACK_CLUSTER_BOOTSTRAP.csv"
    )

    loo_path = (
        OUT
        /
        "R10C2_LEAVE_ONE_SOURCE_OUT_SENSITIVITY.csv"
    )

    summary_path = (
        OUT
        /
        "R10C2_EXTERNAL_ENSEMBLE_SUMMARY.csv"
    )

    correlation_path = (
        OUT
        /
        "R10C2_DIVERSITY_GAIN_ASSOCIATION.csv"
    )

    sample_df.to_csv(
        sample_path,
        index=False
    )

    validation_df.to_csv(
        validation_path,
        index=False
    )

    diversity_df.to_csv(
        diversity_path,
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

    summary_df.to_csv(
        summary_path,
        index=False
    )

    correlation_df.to_csv(
        correlation_path,
        index=False
    )

    status = (
        "R10C2_FRESH_WILDBOX_EXTERNAL_ENSEMBLE_VALUE_QUANTIFIED_READY_FOR_R10D_CROSS_DOMAIN_ENSEMBLE_CONCLUSION"
    )

    manifest = {
        "step":
            "R10C2_FRESH_WILDBOX_EXTERNAL_TRANSFORMER_ENSEMBLE_VALUE",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "new_inference":
            False,

        "representation":
            "original frozen anisotropic canonical representation",

        "metric_space":
            "native normalized coordinate space; comparisons are within representation",

        "primary_comparators":
            [
                "mean individual-seed error",
                "validation-selected single seed",
            ],

        "primary_dependency_unit":
            "track_key",

        "within_track_reduction":
            "median paired difference",

        "bootstrap_replicates":
            BOOTSTRAP_B,

        "bootstrap_seed_base":
            BOOTSTRAP_SEED,

        "source_sensitivity":
            "leave-one-source-out over the 3 fresh WildBox videos",

        "delta_definition":
            "ensemble - comparator; negative favors ensemble",

        "claim_guardrail":
            (
                "Do not generalize fresh WildBox ensemble benefit unless track-cluster CIs "
                "and leave-one-source-out sensitivity are directionally consistent. "
                "Do not call the 3-video fresh holdout broad ecological generalization."
            ),

        "outputs": {
            "sample_metrics":
                str(
                    sample_path
                ),

            "validation_seed_lock":
                str(
                    validation_path
                ),

            "pairwise_diversity":
                str(
                    diversity_path
                ),

            "track_cluster_bootstrap":
                str(
                    bootstrap_path
                ),

            "leave_one_source_out":
                str(
                    loo_path
                ),

            "summary":
                str(
                    summary_path
                ),

            "diversity_gain_association":
                str(
                    correlation_path
                ),
        },
    }

    (
        OUT
        /
        "R10C2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "FRESH WILDBOX ENSEMBLE SUMMARY:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nPRIMARY TRACK-CLUSTER BOOTSTRAP:"
    )

    print(
        bootstrap_df[
            [
                "task",
                "comparator",
                "metric",
                "tracks",
                "delta",
                "ci95_low",
                "ci95_high",
                "direction",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nLEAVE-ONE-SOURCE-OUT SENSITIVITY:"
    )

    print(
        loo_df.to_string(
            index=False
        )
    )

    print(
        "\nPAIRWISE SEED DIVERSITY:"
    )

    print(
        diversity_df.to_string(
            index=False
        )
    )

    print(
        "\nDIVERSITY vs ENSEMBLE-GAIN ASSOCIATION:"
    )

    print(
        correlation_df.to_string(
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
        "\nR10-C2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
