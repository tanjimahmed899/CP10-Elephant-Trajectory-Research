from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

R3A_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R3_clustered_bootstrap"
    / "R3A_track_cluster_recovery"
)

R3A_MANIFEST = R3A_DIR / "R3A_MANIFEST.json"
R3A_SAMPLE = R3A_DIR / "R3A_six_model_same_sample_metrics_with_clusters.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R3_clustered_bootstrap"
    / "R3B_paired_clustered_bootstrap_CI"
)

OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Stationary",
    "ConstantVelocity",
    "KalmanCV",
    "GRU",
    "LSTM",
    "TransformerEnsemble",
]

METRICS = ["ade", "fde"]

PRIMARY_PAIRS = [
    ("TransformerEnsemble", "Stationary"),
    ("TransformerEnsemble", "ConstantVelocity"),
    ("TransformerEnsemble", "KalmanCV"),
    ("TransformerEnsemble", "GRU"),
    ("TransformerEnsemble", "LSTM"),
    ("LSTM", "GRU"),
]

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20261029


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


def verify_r3a():
    require(R3A_MANIFEST)
    require(R3A_SAMPLE)

    manifest = json.loads(
        R3A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("status")
        !=
        "R3A_TRACK_SOURCE_CLUSTERS_RECOVERED_AND_PAIRED_READY_FOR_BOOTSTRAP"
    ):
        raise RuntimeError(
            f"Unexpected R3A status: {manifest.get('status')}"
        )

    expected_hash = manifest.get("cluster_ready_sample_sha256")

    if expected_hash and sha256_file(R3A_SAMPLE) != expected_hash:
        raise RuntimeError(
            "R3A cluster-ready sample CSV hash mismatch."
        )

    return manifest


def validate_input(df):
    required = {
        "dataset",
        "task",
        "model",
        "sample_id",
        "track_id",
        "source_id",
        "ade",
        "fde",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"R3A sample table missing columns: {sorted(missing)}"
        )

    if df[["ade", "fde"]].isna().any().any():
        raise RuntimeError(
            "R3A sample table contains missing ADE/FDE values."
        )

    if df[["track_id", "source_id"]].isna().any().any():
        raise RuntimeError(
            "R3A sample table contains missing track/source IDs."
        )

    # Strict one row per model/sample.
    dup = df.duplicated(
        subset=[
            "dataset",
            "task",
            "model",
            "sample_id",
        ],
        keep=False,
    )

    if dup.any():
        raise RuntimeError(
            "Duplicate dataset/task/model/sample rows found in R3A input."
        )

    # Paired sample sets and cluster labels must be identical across models.
    audits = []

    for (dataset, task), group in df.groupby(
        ["dataset", "task"],
        sort=True,
    ):
        ref = (
            group[group["model"] == MODELS[0]]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )

        if len(ref) == 0:
            raise RuntimeError(
                f"Missing reference model {MODELS[0]} for {dataset} {task}"
            )

        for model in MODELS:
            cur = (
                group[group["model"] == model]
                .sort_values("sample_id")
                .reset_index(drop=True)
            )

            same_ids = (
                cur["sample_id"].astype(str).tolist()
                ==
                ref["sample_id"].astype(str).tolist()
            )

            same_tracks = (
                cur["track_id"].astype(str).tolist()
                ==
                ref["track_id"].astype(str).tolist()
            )

            same_sources = (
                cur["source_id"].astype(str).tolist()
                ==
                ref["source_id"].astype(str).tolist()
            )

            audits.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "model": model,
                    "samples": len(cur),
                    "same_sample_ids": same_ids,
                    "same_track_sequence": same_tracks,
                    "same_source_sequence": same_sources,
                }
            )

            if not (same_ids and same_tracks and same_sources):
                raise RuntimeError(
                    f"Pairing failed for {dataset} {task} {model}"
                )

    return pd.DataFrame(audits)


def make_cluster_model_table(
    group,
    cluster_col,
    metric,
):
    """
    Equal-cluster estimand:
      1) mean metric within each cluster/model;
      2) pair models by the exact same clusters.
    """
    table = (
        group.groupby(
            [cluster_col, "model"],
            as_index=False,
        )[metric]
        .mean()
        .pivot(
            index=cluster_col,
            columns="model",
            values=metric,
        )
    )

    missing_models = [
        model
        for model in MODELS
        if model not in table.columns
    ]

    if missing_models:
        raise RuntimeError(
            f"{cluster_col}/{metric}: missing models in cluster table: {missing_models}"
        )

    if table[MODELS].isna().any().any():
        raise RuntimeError(
            f"{cluster_col}/{metric}: at least one cluster is missing a model."
        )

    return table[MODELS].copy()


def paired_bootstrap_difference(
    diff_by_cluster,
    reps,
    rng,
):
    diff_by_cluster = np.asarray(
        diff_by_cluster,
        dtype=np.float64,
    )

    n = len(diff_by_cluster)

    if n < 2:
        raise RuntimeError(
            f"Need at least 2 clusters, got {n}"
        )

    point = float(
        diff_by_cluster.mean()
    )

    # Chunked bootstrap to keep memory bounded.
    chunk = 2000
    boots = np.empty(
        reps,
        dtype=np.float64,
    )

    pos = 0

    while pos < reps:
        k = min(
            chunk,
            reps - pos,
        )

        idx = rng.integers(
            0,
            n,
            size=(k, n),
        )

        boots[pos:pos + k] = (
            diff_by_cluster[idx]
            .mean(axis=1)
        )

        pos += k

    ci_low, ci_high = np.quantile(
        boots,
        [0.025, 0.975],
    )

    prob_lt_zero = float(
        np.mean(
            boots < 0
        )
    )

    prob_gt_zero = float(
        np.mean(
            boots > 0
        )
    )

    excludes_zero = bool(
        ci_high < 0
        or
        ci_low > 0
    )

    return {
        "point_difference_A_minus_B": point,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "ci95_excludes_zero": excludes_zero,
        "bootstrap_prob_A_better": prob_lt_zero,
        "bootstrap_prob_A_worse": prob_gt_zero,
        "bootstrap_reps": int(reps),
        "clusters": int(n),
    }


def compute_all(
    df,
):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    rows = []
    cluster_tables = []

    for (dataset, task), group in df.groupby(
        ["dataset", "task"],
        sort=True,
    ):
        for cluster_level, cluster_col in [
            ("track", "track_id"),
            ("source", "source_id"),
        ]:
            for metric in METRICS:
                table = make_cluster_model_table(
                    group,
                    cluster_col,
                    metric,
                )

                cluster_tables.append(
                    table.reset_index()
                    .assign(
                        dataset=dataset,
                        task=task,
                        cluster_level=cluster_level,
                        metric=metric,
                    )
                )

                for model_a, model_b in PRIMARY_PAIRS:
                    diff = (
                        table[model_a]
                        -
                        table[model_b]
                    ).to_numpy(
                        dtype=np.float64
                    )

                    result = paired_bootstrap_difference(
                        diff,
                        BOOTSTRAP_REPS,
                        rng,
                    )

                    mean_a = float(
                        table[model_a].mean()
                    )

                    mean_b = float(
                        table[model_b].mean()
                    )

                    rel = np.nan

                    if mean_b != 0:
                        rel = float(
                            100.0
                            *
                            (
                                mean_a
                                -
                                mean_b
                            )
                            /
                            mean_b
                        )

                    rows.append(
                        {
                            "dataset": dataset,
                            "task": task,
                            "cluster_level": cluster_level,
                            "metric": metric.upper(),
                            "model_A": model_a,
                            "model_B": model_b,
                            "cluster_macro_A": mean_a,
                            "cluster_macro_B": mean_b,
                            "relative_difference_pct_A_vs_B": rel,
                            **result,
                        }
                    )

    return (
        pd.DataFrame(rows),
        pd.concat(
            cluster_tables,
            ignore_index=True,
        ),
    )


def interpretation_table(results):
    rows = []

    for _, row in results.iterrows():
        diff = float(
            row["point_difference_A_minus_B"]
        )

        low = float(
            row["ci95_low"]
        )

        high = float(
            row["ci95_high"]
        )

        if high < 0:
            conclusion = (
                "A lower than B; 95% paired cluster-bootstrap CI excludes zero"
            )
        elif low > 0:
            conclusion = (
                "A higher than B; 95% paired cluster-bootstrap CI excludes zero"
            )
        else:
            conclusion = (
                "Difference uncertain at 95% level; CI includes zero"
            )

        # Source-level Fresh has only 3 sources: explicitly label sensitivity only.
        inferential_role = (
            "primary"
            if row["cluster_level"] == "track"
            else
            "sensitivity_only"
        )

        small_cluster_warning = bool(
            int(row["clusters"]) < 10
        )

        rows.append(
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "cluster_level": row["cluster_level"],
                "metric": row["metric"],
                "model_A": row["model_A"],
                "model_B": row["model_B"],
                "difference_A_minus_B": diff,
                "ci95_low": low,
                "ci95_high": high,
                "conclusion": conclusion,
                "inferential_role": inferential_role,
                "small_cluster_warning": small_cluster_warning,
            }
        )

    return pd.DataFrame(rows)


def focused_transformer_table(results):
    return results[
        results["model_A"]
        ==
        "TransformerEnsemble"
    ].copy()


def main():
    print(
        "\nR3-B — PAIRED TRACK/SOURCE-CLUSTERED BOOTSTRAP 95% CI"
    )

    print(
        "Primary inference: track-cluster paired bootstrap. "
        "Source-cluster bootstrap is reported as sensitivity analysis only. "
        "Difference = model_A - model_B, so negative values favor model_A.\n"
    )

    verify_r3a()

    df = pd.read_csv(
        R3A_SAMPLE
    )

    pairing_audit = validate_input(
        df
    )

    results, cluster_table = compute_all(
        df
    )

    interpretation = interpretation_table(
        results
    )

    transformer = focused_transformer_table(
        results
    )

    pairing_path = OUT / "R3B_pairing_audit.csv"
    results_path = OUT / "R3B_paired_cluster_bootstrap_results.csv"
    cluster_path = OUT / "R3B_cluster_model_means.csv"
    interpretation_path = OUT / "R3B_interpretation.csv"
    transformer_path = OUT / "R3B_transformer_focused_CI.csv"

    pairing_audit.to_csv(
        pairing_path,
        index=False,
    )

    results.to_csv(
        results_path,
        index=False,
    )

    cluster_table.to_csv(
        cluster_path,
        index=False,
    )

    interpretation.to_csv(
        interpretation_path,
        index=False,
    )

    transformer.to_csv(
        transformer_path,
        index=False,
    )

    status = (
        "R3B_PAIRED_TRACK_SOURCE_CLUSTERED_BOOTSTRAP_COMPLETE_READY_FOR_R3C_INTERPRETATION"
    )

    manifest = {
        "step": "R3B_PAIRED_CLUSTERED_BOOTSTRAP_CI",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "primary_inference": "paired equal-track cluster bootstrap",
        "source_analysis_role": "sensitivity_only",
        "difference_definition": "model_A - model_B; negative favors model_A",
        "source_r3a": str(R3A_SAMPLE),
        "source_r3a_sha256": sha256_file(R3A_SAMPLE),
        "results_csv": str(results_path),
        "results_sha256": sha256_file(results_path),
        "transformer_focused_csv": str(transformer_path),
        "interpretation_csv": str(interpretation_path),
        "pairing_audit_csv": str(pairing_path),
    }

    manifest_path = OUT / "R3B_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("PAIRING AUDIT:")
    print(
        pairing_audit.to_string(
            index=False
        )
    )

    print(
        "\nPRIMARY TRACK-CLUSTER RESULTS — TRANSFORMER COMPARISONS:"
    )

    primary = transformer[
        transformer["cluster_level"]
        ==
        "track"
    ].copy()

    print(
        primary[
            [
                "dataset",
                "task",
                "metric",
                "model_A",
                "model_B",
                "clusters",
                "cluster_macro_A",
                "cluster_macro_B",
                "point_difference_A_minus_B",
                "relative_difference_pct_A_vs_B",
                "ci95_low",
                "ci95_high",
                "ci95_excludes_zero",
                "bootstrap_prob_A_better",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSOURCE-CLUSTER SENSITIVITY — TRANSFORMER COMPARISONS:"
    )

    source = transformer[
        transformer["cluster_level"]
        ==
        "source"
    ].copy()

    print(
        source[
            [
                "dataset",
                "task",
                "metric",
                "model_A",
                "model_B",
                "clusters",
                "point_difference_A_minus_B",
                "ci95_low",
                "ci95_high",
                "ci95_excludes_zero",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nLSTM vs GRU — PRIMARY TRACK-CLUSTER RESULTS:"
    )

    lstm_gru = results[
        (results["cluster_level"] == "track")
        &
        (results["model_A"] == "LSTM")
        &
        (results["model_B"] == "GRU")
    ].copy()

    print(
        lstm_gru[
            [
                "dataset",
                "task",
                "metric",
                "clusters",
                "point_difference_A_minus_B",
                "ci95_low",
                "ci95_high",
                "ci95_excludes_zero",
                "bootstrap_prob_A_better",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR3-B COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
