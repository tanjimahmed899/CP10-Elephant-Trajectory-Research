
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

C1_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C1_anisotropy_geometry_distortion"
)

C1_MANIFEST = C1_DIR / "R9H_C1_MANIFEST.json"
C1_PER_SAMPLE = C1_DIR / "R9H_C1_PER_SAMPLE_GEOMETRY_DISTORTION.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C2_clustered_ci_and_representation_decision"
)

OUT.mkdir(parents=True, exist_ok=True)

BOOTSTRAPS = 20000
SEED = 20261029
ALPHA = 0.05

TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]

CONTINUOUS_METRICS = {
    "heading_abs_difference_deg":
        "median",

    "path_length_percent_difference_aniso_minus_iso":
        "median",

    "scale_percent_difference_aniso_minus_iso":
        "median",

    "history_canonical_mean_point_l2_difference":
        "median",

    "future_canonical_mean_point_l2_difference":
        "median",

    "future_canonical_endpoint_l2_difference":
        "median",
}

RATE_METRICS = {
    "floor_activation_changed":
        "mean",
}


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


def verify_c1():
    require(
        C1_MANIFEST
    )

    require(
        C1_PER_SAMPLE
    )

    manifest = json.loads(
        C1_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R9H_C1_ANISOTROPY_CANONICALIZATION_GEOMETRY_DISTORTION_MEASURED_READY_FOR_R9H_C2_CLUSTERED_CI_AND_REPRESENTATION_DECISION"
    )

    if manifest.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R9-H-C1 status: {manifest.get('status')}"
        )

    if (
        manifest.get("per_sample_sha256")
        !=
        sha256_file(
            C1_PER_SAMPLE
        )
    ):
        raise RuntimeError(
            "R9-H-C1 per-sample CSV hash mismatch."
        )

    return manifest


def load_data():
    df = pd.read_csv(
        C1_PER_SAMPLE,
        dtype={
            "task": str,
            "sequence": str,
            "segment_id": str,
        }
    )

    required = {
        "task",
        "sample_id",
        "sequence",
        "segment_id",
        *CONTINUOUS_METRICS.keys(),
        *RATE_METRICS.keys(),
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"C1 per-sample CSV missing columns: {sorted(missing)}"
        )

    if not set(
        df["task"].unique()
    ).issuperset(
        TASKS
    ):
        raise RuntimeError(
            f"Unexpected task coverage: {sorted(df['task'].unique())}"
        )

    return df


def make_segment_table(df):
    """
    Primary dependency-aware unit:
      segment = (sequence, segment_id)

    Each continuous metric is reduced to the within-segment median.
    Each binary/rate metric is reduced to the within-segment mean.
    This prevents long, heavily overlapping tracks from dominating the primary inference.
    """
    rows = []

    for (
        task,
        sequence,
        segment_id,
    ), g in df.groupby(
        [
            "task",
            "sequence",
            "segment_id",
        ],
        sort=True,
        dropna=False,
    ):
        row = {
            "task":
                task,

            "sequence":
                sequence,

            "segment_id":
                segment_id,

            "windows":
                int(
                    len(
                        g
                    )
                ),
        }

        for metric in CONTINUOUS_METRICS:
            vals = pd.to_numeric(
                g[
                    metric
                ],
                errors="coerce"
            ).to_numpy(
                dtype=np.float64
            )

            row[
                metric
            ] = float(
                np.nanmedian(
                    vals
                )
            )

        for metric in RATE_METRICS:
            vals = (
                g[
                    metric
                ]
                .astype(float)
                .to_numpy(
                    dtype=np.float64
                )
            )

            row[
                metric
            ] = float(
                np.nanmean(
                    vals
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def primary_stat(
    values,
    stat_type
):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    if stat_type == "median":
        return float(
            np.nanmedian(
                values
            )
        )

    if stat_type == "mean":
        return float(
            np.nanmean(
                values
            )
        )

    raise ValueError(
        stat_type
    )


def bootstrap_task(
    segment_df,
    task,
    rng
):
    g = (
        segment_df[
            segment_df[
                "task"
            ]
            ==
            task
        ]
        .reset_index(
            drop=True
        )
    )

    n_clusters = len(
        g
    )

    if n_clusters < 2:
        raise RuntimeError(
            f"{task}: insufficient segment clusters ({n_clusters})."
        )

    metric_specs = {
        **CONTINUOUS_METRICS,
        **RATE_METRICS,
    }

    observed = {
        metric:
            primary_stat(
                g[
                    metric
                ].to_numpy(
                    dtype=np.float64
                ),
                stat_type
            )
        for metric, stat_type
        in metric_specs.items()
    }

    boot = {
        metric:
            np.empty(
                BOOTSTRAPS,
                dtype=np.float64
            )
        for metric in metric_specs
    }

    n = n_clusters

    for b in range(
        BOOTSTRAPS
    ):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        sampled = g.iloc[
            idx
        ]

        for metric, stat_type in metric_specs.items():
            boot[
                metric
            ][
                b
            ] = primary_stat(
                sampled[
                    metric
                ].to_numpy(
                    dtype=np.float64
                ),
                stat_type
            )

    rows = []

    for metric, stat_type in metric_specs.items():
        arr = boot[
            metric
        ]

        lo = float(
            np.quantile(
                arr,
                ALPHA / 2.0
            )
        )

        hi = float(
            np.quantile(
                arr,
                1.0 - ALPHA / 2.0
            )
        )

        obs = observed[
            metric
        ]

        if metric in RATE_METRICS:
            display_scale = 100.0
            display_unit = "percent"
        elif metric == "heading_abs_difference_deg":
            display_scale = 1.0
            display_unit = "degrees"
        elif "percent_difference" in metric:
            display_scale = 1.0
            display_unit = "percent"
        else:
            display_scale = 1.0
            display_unit = "canonical_L2"

        rows.append(
            {
                "task":
                    task,

                "metric":
                    metric,

                "primary_segment_statistic":
                    stat_type,

                "segments":
                    int(
                        n_clusters
                    ),

                "sequences":
                    int(
                        g[
                            "sequence"
                        ].nunique()
                    ),

                "bootstrap_replicates":
                    BOOTSTRAPS,

                "seed":
                    SEED,

                "estimate":
                    obs
                    *
                    display_scale,

                "ci95_low":
                    lo
                    *
                    display_scale,

                "ci95_high":
                    hi
                    *
                    display_scale,

                "display_unit":
                    display_unit,
            }
        )

    return pd.DataFrame(
        rows
    )


def sequence_leave_one_out(
    segment_df
):
    metric_specs = {
        **CONTINUOUS_METRICS,
        **RATE_METRICS,
    }

    rows = []

    for task in TASKS:
        g = segment_df[
            segment_df[
                "task"
            ]
            ==
            task
        ].copy()

        sequences = sorted(
            g[
                "sequence"
            ].unique()
        )

        for held_out in sequences:
            keep = g[
                g[
                    "sequence"
                ]
                !=
                held_out
            ]

            if len(
                keep
            ) == 0:
                continue

            row = {
                "task":
                    task,

                "held_out_sequence":
                    held_out,

                "remaining_sequences":
                    int(
                        keep[
                            "sequence"
                        ].nunique()
                    ),

                "remaining_segments":
                    int(
                        len(
                            keep
                        )
                    ),
            }

            for metric, stat_type in metric_specs.items():
                value = primary_stat(
                    keep[
                        metric
                    ].to_numpy(
                        dtype=np.float64
                    ),
                    stat_type
                )

                if metric in RATE_METRICS:
                    value *= 100.0

                row[
                    metric
                ] = value

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def sequence_consistency(
    segment_df
):
    rows = []

    for (
        task,
        sequence,
    ), g in segment_df.groupby(
        [
            "task",
            "sequence",
        ],
        sort=True
    ):
        rows.append(
            {
                "task":
                    task,

                "sequence":
                    sequence,

                "segments":
                    int(
                        len(
                            g
                        )
                    ),

                "heading_segment_median_deg":
                    float(
                        np.median(
                            g[
                                "heading_abs_difference_deg"
                            ]
                        )
                    ),

                "path_length_pct_segment_median":
                    float(
                        np.median(
                            g[
                                "path_length_percent_difference_aniso_minus_iso"
                            ]
                        )
                    ),

                "scale_pct_segment_median":
                    float(
                        np.median(
                            g[
                                "scale_percent_difference_aniso_minus_iso"
                            ]
                        )
                    ),

                "future_canonical_mean_l2_segment_median":
                    float(
                        np.median(
                            g[
                                "future_canonical_mean_point_l2_difference"
                            ]
                        )
                    ),

                "floor_activation_changed_segment_mean_pct":
                    float(
                        g[
                            "floor_activation_changed"
                        ].mean()
                        *
                        100.0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_representation_decision(
    ci_df,
    consistency_df
):
    rows = []

    for task in TASKS:
        ci = ci_df[
            ci_df[
                "task"
            ]
            ==
            task
        ].set_index(
            "metric"
        )

        seq = consistency_df[
            consistency_df[
                "task"
            ]
            ==
            task
        ]

        heading_lo = float(
            ci.loc[
                "heading_abs_difference_deg",
                "ci95_low"
            ]
        )

        path_lo = float(
            ci.loc[
                "path_length_percent_difference_aniso_minus_iso",
                "ci95_low"
            ]
        )

        future_lo = float(
            ci.loc[
                "future_canonical_mean_point_l2_difference",
                "ci95_low"
            ]
        )

        floor_lo = float(
            ci.loc[
                "floor_activation_changed",
                "ci95_low"
            ]
        )

        all_seq_heading_positive = bool(
            (
                seq[
                    "heading_segment_median_deg"
                ]
                >
                0.0
            ).all()
        )

        all_seq_path_positive = bool(
            (
                seq[
                    "path_length_pct_segment_median"
                ]
                >
                0.0
            ).all()
        )

        # This is a protocol/action decision, NOT a model-superiority test.
        # We require the cluster-aware lower CI to be above zero for:
        # - heading distortion magnitude,
        # - path-length change,
        # - canonical future change,
        # plus cross-sequence consistency.
        material_geometry_change = bool(
            heading_lo > 0.0
            and path_lo > 0.0
            and future_lo > 0.0
            and all_seq_heading_positive
            and all_seq_path_positive
        )

        if material_geometry_change:
            decision = (
                "MATCHED_ISOTROPIC_NEURAL_RETRAINING_ABLATION_WARRANTED"
            )

            interpretation = (
                "Cluster-aware evidence shows that anisotropic normalization "
                "materially changes the frozen canonicalizer geometry across "
                "segments and sequences. Proceed to a matched isotropic neural "
                "retraining ablation using identical splits, architectures, seeds "
                "and hyperparameters, with the isotropic scale floor re-derived "
                "from isotropic TrainReal. Do NOT claim isotropic forecasting "
                "superiority until that matched retraining is evaluated."
            )

        else:
            decision = (
                "GEOMETRY_CHANGE_NOT_ROBUST_ENOUGH_TO_JUSTIFY_RETRAINING_FROM_THIS_AUDIT"
            )

            interpretation = (
                "This geometry audit alone does not provide robust enough "
                "cluster-aware evidence to mandate isotropic retraining."
            )

        rows.append(
            {
                "task":
                    task,

                "heading_ci95_low_deg":
                    heading_lo,

                "path_length_pct_ci95_low":
                    path_lo,

                "future_canonical_mean_l2_ci95_low":
                    future_lo,

                "floor_activation_change_ci95_low_pct":
                    floor_lo,

                "all_sequences_heading_median_positive":
                    all_seq_heading_positive,

                "all_sequences_path_change_median_positive":
                    all_seq_path_positive,

                "decision":
                    decision,

                "interpretation":
                    interpretation,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-H-C2 — SEGMENT/TRACK-CLUSTERED CI + REPRESENTATION DECISION"
    )

    print(
        "NO training and NO model inference. Overlapping windows are dependency-aware: "
        "the primary unit is (sequence, segment_id). Continuous metrics are reduced "
        "to within-segment medians and floor-change to within-segment rates; 20,000 "
        "segment-cluster bootstrap resamples produce 95% percentile CIs. "
        "Leave-one-sequence-out sensitivity is reported separately.\n"
    )

    verify_c1()

    per_sample = load_data()

    segment_df = make_segment_table(
        per_sample
    )

    segment_path = (
        OUT
        / "R9H_C2_SEGMENT_MACRO_TABLE.csv"
    )

    segment_df.to_csv(
        segment_path,
        index=False
    )

    rng = np.random.default_rng(
        SEED
    )

    ci_frames = []

    for task in TASKS:
        ci_frames.append(
            bootstrap_task(
                segment_df,
                task,
                rng
            )
        )

    ci_df = pd.concat(
        ci_frames,
        ignore_index=True
    )

    loo_df = sequence_leave_one_out(
        segment_df
    )

    consistency_df = sequence_consistency(
        segment_df
    )

    decision_df = build_representation_decision(
        ci_df,
        consistency_df
    )

    ci_path = (
        OUT
        / "R9H_C2_SEGMENT_CLUSTER_BOOTSTRAP_CI.csv"
    )

    loo_path = (
        OUT
        / "R9H_C2_LEAVE_ONE_SEQUENCE_OUT.csv"
    )

    consistency_path = (
        OUT
        / "R9H_C2_SEQUENCE_CONSISTENCY.csv"
    )

    decision_path = (
        OUT
        / "R9H_C2_REPRESENTATION_DECISION.csv"
    )

    ci_df.to_csv(
        ci_path,
        index=False
    )

    loo_df.to_csv(
        loo_path,
        index=False
    )

    consistency_df.to_csv(
        consistency_path,
        index=False
    )

    decision_df.to_csv(
        decision_path,
        index=False
    )

    all_warranted = bool(
        (
            decision_df[
                "decision"
            ]
            ==
            "MATCHED_ISOTROPIC_NEURAL_RETRAINING_ABLATION_WARRANTED"
        ).all()
    )

    if all_warranted:
        status = (
            "R9H_C2_CLUSTERED_CI_CONFIRMS_MATERIAL_GEOMETRY_CHANGE_READY_FOR_R9I_MATCHED_ISOTROPIC_NEURAL_RETRAINING_PROTOCOL"
        )
    else:
        status = (
            "R9H_C2_CLUSTERED_CI_COMPLETE_REPRESENTATION_DECISION_REQUIRES_REVIEW"
        )

    manifest = {
        "step":
            "R9H_C2_SEGMENT_TRACK_CLUSTERED_CI_AND_REPRESENTATION_DECISION",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "bootstrap_replicates":
            BOOTSTRAPS,

        "seed":
            SEED,

        "confidence_level":
            0.95,

        "primary_dependency_unit":
            "(sequence, segment_id)",

        "primary_statistic_policy":
            (
                "Within-segment median for continuous distortion metrics; "
                "within-segment mean for floor-change; bootstrap segments "
                "with replacement and aggregate segments equally."
            ),

        "sequence_sensitivity":
            "leave-one-sequence-out",

        "decision_scope":
            (
                "Whether matched isotropic neural retraining ablation is warranted. "
                "This is NOT a forecasting-superiority conclusion."
            ),

        "important_nonclaim":
            (
                "No claim that isotropic normalization improves forecasting "
                "performance is permitted until matched neural retraining/evaluation."
            ),

        "new_training":
            False,

        "model_inference":
            False,

        "frozen_results_modified":
            False,

        "segment_macro_csv":
            str(
                segment_path
            ),

        "segment_macro_sha256":
            sha256_file(
                segment_path
            ),

        "cluster_bootstrap_ci_csv":
            str(
                ci_path
            ),

        "cluster_bootstrap_ci_sha256":
            sha256_file(
                ci_path
            ),

        "leave_one_sequence_out_csv":
            str(
                loo_path
            ),

        "sequence_consistency_csv":
            str(
                consistency_path
            ),

        "representation_decision_csv":
            str(
                decision_path
            ),

        "decisions":
            decision_df[
                [
                    "task",
                    "decision",
                ]
            ].to_dict(
                orient="records"
            ),
    }

    (
        OUT
        / "R9H_C2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "SEGMENT-CLUSTER BOOTSTRAP 95% CI:"
    )

    print(
        ci_df.to_string(
            index=False
        )
    )

    print(
        "\nSEQUENCE CONSISTENCY:"
    )

    print(
        consistency_df.to_string(
            index=False
        )
    )

    print(
        "\nLEAVE-ONE-SEQUENCE-OUT RANGE:"
    )

    if len(
        loo_df
    ):
        metric_cols = [
            "heading_abs_difference_deg",
            "path_length_percent_difference_aniso_minus_iso",
            "future_canonical_mean_point_l2_difference",
            "floor_activation_changed",
        ]

        rows = []

        for task in TASKS:
            g = loo_df[
                loo_df[
                    "task"
                ]
                ==
                task
            ]

            row = {
                "task":
                    task
            }

            for metric in metric_cols:
                row[
                    f"{metric}_min"
                ] = float(
                    g[
                        metric
                    ].min()
                )

                row[
                    f"{metric}_max"
                ] = float(
                    g[
                        metric
                    ].max()
                )

            rows.append(
                row
            )

        print(
            pd.DataFrame(
                rows
            ).to_string(
                index=False
            )
        )

    print(
        "\nREPRESENTATION DECISION:"
    )

    print(
        decision_df.to_string(
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
        "\nR9-H-C2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
