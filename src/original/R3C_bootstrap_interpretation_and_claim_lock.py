from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

R3B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R3_clustered_bootstrap"
    / "R3B_paired_clustered_bootstrap_CI"
)

R3B_MANIFEST = R3B_DIR / "R3B_MANIFEST.json"
R3B_RESULTS = R3B_DIR / "R3B_paired_cluster_bootstrap_results.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R3_clustered_bootstrap"
    / "R3C_interpretation_claim_lock"
)

OUT.mkdir(parents=True, exist_ok=True)


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


def verify():
    require(R3B_MANIFEST)
    require(R3B_RESULTS)

    manifest = json.loads(
        R3B_MANIFEST.read_text(encoding="utf-8")
    )

    if (
        manifest.get("status")
        !=
        "R3B_PAIRED_TRACK_SOURCE_CLUSTERED_BOOTSTRAP_COMPLETE_READY_FOR_R3C_INTERPRETATION"
    ):
        raise RuntimeError(
            f"Unexpected R3B status: {manifest.get('status')}"
        )

    expected_hash = manifest.get("results_sha256")
    if expected_hash and sha256_file(R3B_RESULTS) != expected_hash:
        raise RuntimeError("R3B results hash mismatch.")

    return manifest


def classify(row):
    diff = float(row["point_difference_A_minus_B"])
    low = float(row["ci95_low"])
    high = float(row["ci95_high"])

    if high < 0:
        relation = "A_better"
    elif low > 0:
        relation = "A_worse"
    else:
        relation = "uncertain"

    return relation


def make_primary_transformer_table(df):
    primary = df[
        (df["cluster_level"] == "track")
        &
        (df["model_A"] == "TransformerEnsemble")
    ].copy()

    primary["relation"] = primary.apply(classify, axis=1)

    primary["claim_text"] = primary.apply(
        lambda r:
            (
                f"Transformer significantly lower {r['metric']} than {r['model_B']}"
                if r["relation"] == "A_better"
                else
                f"Transformer significantly higher {r['metric']} than {r['model_B']}"
                if r["relation"] == "A_worse"
                else
                f"No clear 95% track-cluster evidence of a {r['metric']} difference "
                f"between Transformer and {r['model_B']}"
            ),
        axis=1,
    )

    return primary


def make_task_level_conclusions(primary):
    rows = []

    for (dataset, task), g in primary.groupby(
        ["dataset", "task"],
        sort=True,
    ):
        def rel(model, metric):
            x = g[
                (g["model_B"] == model)
                &
                (g["metric"].str.upper() == metric)
            ]
            if len(x) != 1:
                return None
            return str(x.iloc[0]["relation"])

        rows.append(
            {
                "dataset": dataset,
                "task": task,

                "Transformer_vs_Stationary_ADE":
                    rel("Stationary", "ADE"),

                "Transformer_vs_CV_ADE":
                    rel("ConstantVelocity", "ADE"),

                "Transformer_vs_Kalman_ADE":
                    rel("KalmanCV", "ADE"),

                "Transformer_vs_GRU_ADE":
                    rel("GRU", "ADE"),

                "Transformer_vs_LSTM_ADE":
                    rel("LSTM", "ADE"),

                "Transformer_vs_Stationary_FDE":
                    rel("Stationary", "FDE"),

                "Transformer_vs_CV_FDE":
                    rel("ConstantVelocity", "FDE"),

                "Transformer_vs_Kalman_FDE":
                    rel("KalmanCV", "FDE"),

                "Transformer_vs_GRU_FDE":
                    rel("GRU", "FDE"),

                "Transformer_vs_LSTM_FDE":
                    rel("LSTM", "FDE"),
            }
        )

    return pd.DataFrame(rows)


def make_claim_lock(primary, all_results):
    rows = []

    rows.append({
        "claim_id": "R3C_CLAIM_01",
        "status": "SUPPORTED",
        "claim": (
            "Forecast-model performance is task- and dataset-dependent; "
            "no single model dominates all TestReal and Fresh WildBox conditions."
        ),
        "basis": (
            "Paired track-cluster bootstrap shows mixed directions and significance "
            "patterns across horizons/datasets."
        ),
    })

    rows.append({
        "claim_id": "R3C_CLAIM_02",
        "status": "NOT_SUPPORTED",
        "claim": (
            "The frozen Transformer ensemble is the overall best forecasting model."
        ),
        "basis": (
            "On BIRDSAI TestReal 2s→2s, Transformer has significantly higher "
            "ADE/FDE than GRU and LSTM; on 2s→5s it is significantly worse than GRU."
        ),
    })

    rows.append({
        "claim_id": "R3C_CLAIM_03",
        "status": "SUPPORTED",
        "claim": (
            "On BIRDSAI TestReal, Transformer clearly improves over constant-velocity "
            "forecasting at both 2s and 5s horizons."
        ),
        "basis": (
            "Track-cluster 95% CIs for Transformer − ConstantVelocity are below zero "
            "for ADE and FDE at both horizons."
        ),
    })

    rows.append({
        "claim_id": "R3C_CLAIM_04",
        "status": "SUPPORTED_WITH_LIMITATION",
        "claim": (
            "On Fresh WildBox 2s→2s, Transformer improves ADE over Stationary, "
            "ConstantVelocity and KalmanCV, but is not clearly different from GRU/LSTM."
        ),
        "basis": (
            "Track-cluster ADE CIs exclude zero versus Stationary/CV/KalmanCV, "
            "but include zero versus GRU/LSTM; only 10 tracks."
        ),
    })

    rows.append({
        "claim_id": "R3C_CLAIM_05",
        "status": "SUPPORTED_WITH_LIMITATION",
        "claim": (
            "On Fresh WildBox 2s→5s, Transformer FDE is slightly lower than GRU/LSTM, "
            "but ADE differences versus GRU/LSTM are uncertain."
        ),
        "basis": (
            "Track-cluster FDE CIs are narrowly below zero versus GRU/LSTM; "
            "ADE CIs include zero; only 10 tracks."
        ),
    })

    rows.append({
        "claim_id": "R3C_CLAIM_06",
        "status": "SUPPORTED",
        "claim": (
            "LSTM outperforms GRU on BIRDSAI TestReal 2s→2s, but their differences "
            "are uncertain for TestReal 2s→5s and Fresh WildBox."
        ),
        "basis": (
            "Paired track-cluster LSTM−GRU CIs are below zero only for TestReal 2s→2s "
            "ADE/FDE."
        ),
    })

    rows.append({
        "claim_id": "R3C_CLAIM_07",
        "status": "METHOD_NOTE",
        "claim": (
            "Track-cluster bootstrap is the primary inferential analysis; "
            "source-cluster bootstrap is sensitivity analysis only."
        ),
        "basis": (
            "Source counts are small, especially Fresh WildBox with only 3 sources, "
            "so source-level intervals are too coarse for primary claims."
        ),
    })

    rows.append({
        "claim_id": "R3C_CLAIM_08",
        "status": "NOT_SUPPORTED",
        "claim": (
            "Transformer superiority is statistically established across datasets."
        ),
        "basis": (
            "Multiple comparisons are uncertain or favor recurrent models; "
            "there is no cross-condition dominance."
        ),
    })

    return pd.DataFrame(rows)


def make_manuscript_summary(primary):
    # Compact, model-comparison-focused table.
    keep = primary[
        [
            "dataset",
            "task",
            "metric",
            "model_B",
            "clusters",
            "cluster_macro_A",
            "cluster_macro_B",
            "point_difference_A_minus_B",
            "relative_difference_pct_A_vs_B",
            "ci95_low",
            "ci95_high",
            "ci95_excludes_zero",
        ]
    ].copy()

    keep = keep.rename(
        columns={
            "model_B": "Comparator",
            "cluster_macro_A": "Transformer_cluster_macro",
            "cluster_macro_B": "Comparator_cluster_macro",
            "point_difference_A_minus_B": "Difference_Transformer_minus_Comparator",
            "relative_difference_pct_A_vs_B": "Relative_difference_pct",
        }
    )

    return keep


def main():
    print(
        "\nR3-C — BOOTSTRAP INTERPRETATION + CLAIM LOCK"
    )
    print(
        "NO new statistical test is run. This step converts the frozen R3B "
        "track-cluster bootstrap results into manuscript-safe conclusions.\n"
    )

    verify()

    df = pd.read_csv(R3B_RESULTS)

    primary = make_primary_transformer_table(df)
    task_conclusions = make_task_level_conclusions(primary)
    claim_lock = make_claim_lock(primary, df)
    manuscript = make_manuscript_summary(primary)

    primary_path = OUT / "R3C_transformer_primary_track_interpretation.csv"
    task_path = OUT / "R3C_task_level_conclusions.csv"
    claim_path = OUT / "R3C_CLAIM_LOCK.csv"
    manuscript_path = OUT / "R3C_manuscript_comparison_table.csv"

    primary.to_csv(primary_path, index=False)
    task_conclusions.to_csv(task_path, index=False)
    claim_lock.to_csv(claim_path, index=False)
    manuscript.to_csv(manuscript_path, index=False)

    status = (
        "R3C_BOOTSTRAP_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R4_CONFORMAL_COVERAGE"
    )

    manifest = {
        "step": "R3C_BOOTSTRAP_INTERPRETATION_CLAIM_LOCK",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "new_statistics_generated": False,
        "primary_inference": "paired track-cluster bootstrap from R3B",
        "source_analysis_role": "sensitivity_only",
        "source_r3b": str(R3B_RESULTS),
        "source_r3b_sha256": sha256_file(R3B_RESULTS),
        "claim_lock_csv": str(claim_path),
        "claim_lock_sha256": sha256_file(claim_path),
        "manuscript_table_csv": str(manuscript_path),
    }

    manifest_path = OUT / "R3C_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("TASK-LEVEL TRANSFORMER CONCLUSIONS:")
    print(task_conclusions.to_string(index=False))

    print("\nCLAIM LOCK:")
    print(claim_lock.to_string(index=False))

    print("\nMANUSCRIPT-SAFE TRANSFORMER COMPARISON TABLE:")
    print(manuscript.to_string(index=False))

    print("\nSTATUS:")
    print(status)

    print("\nR3-C COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
