from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

R2E_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2E_six_model_fair_benchmark"
)

R2E_MANIFEST = R2E_DIR / "R2E_MANIFEST.json"
R2E_SAMPLE = R2E_DIR / "R2E_six_model_same_sample_metrics.csv"
R2E_SUMMARY = R2E_DIR / "R2E_six_model_fair_summary.csv"

B1_INPUT_PROV = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B1_stationary_cv_transformer"
    / "R2D_B1_input_provenance.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R3_clustered_bootstrap"
    / "R3A_track_cluster_recovery"
)

OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    ("BIRDSAI_TestReal", "2s_to_2s"): 1053,
    ("BIRDSAI_TestReal", "2s_to_5s"): 826,
    ("Fresh_WildBox", "2s_to_2s"): 865,
    ("Fresh_WildBox", "2s_to_5s"): 637,
}

MODELS = [
    "Stationary",
    "ConstantVelocity",
    "KalmanCV",
    "GRU",
    "LSTM",
    "TransformerEnsemble",
]


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


def find_col(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def verify_r2e():
    for p in [
        R2E_MANIFEST,
        R2E_SAMPLE,
        R2E_SUMMARY,
        B1_INPUT_PROV,
    ]:
        require(p)

    manifest = json.loads(
        R2E_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("status")
        !=
        "R2E_SIX_MODEL_FAIR_SAME_SAMPLE_BENCHMARK_COMPLETE_READY_FOR_R3_CLUSTERED_BOOTSTRAP"
    ):
        raise RuntimeError(
            f"Unexpected R2E status: {manifest.get('status')}"
        )

    expected_hash = manifest.get("final_sample_metrics_sha256")
    if expected_hash and sha256_file(R2E_SAMPLE) != expected_hash:
        raise RuntimeError("R2E sample table hash mismatch.")

    return manifest


def metadata_path_for_task(task):
    prov = pd.read_csv(B1_INPUT_PROV)

    hit = prov[
        (prov["dataset"].astype(str) == "BIRDSAI_TestReal")
        &
        (prov["task"].astype(str) == task)
    ].copy()

    if len(hit) != 1:
        raise RuntimeError(
            f"Expected one TestReal B1 provenance row for {task}, got {len(hit)}"
        )

    row = hit.iloc[0]

    path = Path(str(row["metadata"]))
    require(path)

    if sha256_file(path) != str(row["metadata_sha256"]):
        raise RuntimeError(
            f"Metadata hash changed for {task}: {path}"
        )

    return path


def recover_testreal_track_ids(r2e):
    patched_parts = []
    audit = []

    for task in ["2s_to_2s", "2s_to_5s"]:
        expected = EXPECTED[
            ("BIRDSAI_TestReal", task)
        ]

        meta_path = metadata_path_for_task(task)
        meta = pd.read_csv(meta_path)

        if len(meta) != expected:
            raise RuntimeError(
                f"{task}: metadata rows {len(meta)} != expected {expected}"
            )

        # Prefer the explicit original track_key used by the frozen Kalman/
        # recurrent evaluator. Otherwise fall back to another track-like field.
        track_col = find_col(
            meta,
            [
                "track_key",
                "tracklet_id",
                "track_id",
                "derived_track_id",
                "track",
            ],
        )

        source_col = find_col(
            meta,
            [
                "source_group",
                "source_id",
                "source",
                "video_id",
                "video",
                "sequence_id",
                "sequence",
            ],
        )

        if track_col is None:
            raise RuntimeError(
                f"{task}: no track-like column in metadata {list(meta.columns)}"
            )

        if source_col is None:
            raise RuntimeError(
                f"{task}: no source-like column in metadata {list(meta.columns)}"
            )

        track_seq = meta[track_col].astype(str).tolist()
        source_seq = meta[source_col].astype(str).tolist()

        task_rows = r2e[
            (r2e["dataset"] == "BIRDSAI_TestReal")
            &
            (r2e["task"] == task)
        ].copy()

        for model in MODELS:
            sub = (
                task_rows[
                    task_rows["model"] == model
                ]
                .copy()
                .reset_index(drop=True)
            )

            if len(sub) != expected:
                raise RuntimeError(
                    f"{task} {model}: count {len(sub)} != {expected}"
                )

            if sub["source_id"].astype(str).tolist() != source_seq:
                raise RuntimeError(
                    f"{task} {model}: source order does not match frozen metadata."
                )

            # Only replace missing/non-informative canonical track namespace.
            old_unique = int(
                sub["track_id"]
                .astype(str)
                .nunique()
            )

            sub["track_id"] = track_seq
            sub["source_id"] = source_seq

            patched_parts.append(sub)

            audit.append(
                {
                    "dataset": "BIRDSAI_TestReal",
                    "task": task,
                    "model": model,
                    "metadata": str(meta_path),
                    "metadata_sha256": sha256_file(meta_path),
                    "track_column_used": track_col,
                    "source_column_used": source_col,
                    "rows": expected,
                    "old_track_unique": old_unique,
                    "new_track_unique": int(pd.Series(track_seq).nunique()),
                    "source_unique": int(pd.Series(source_seq).nunique()),
                    "row_order_verified": True,
                }
            )

    fresh = r2e[
        r2e["dataset"] == "Fresh_WildBox"
    ].copy()

    patched = pd.concat(
        patched_parts + [fresh],
        ignore_index=True,
    )

    return patched, pd.DataFrame(audit)


def validate_clusters(df):
    rows = []

    for (dataset, task), expected in EXPECTED.items():
        sub = df[
            (df["dataset"] == dataset)
            &
            (df["task"] == task)
        ].copy()

        for model in MODELS:
            m = sub[
                sub["model"] == model
            ].copy()

            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "model": model,
                    "samples": int(len(m)),
                    "expected_samples": expected,
                    "tracks": int(m["track_id"].astype(str).nunique()),
                    "sources": int(m["source_id"].astype(str).nunique()),
                    "missing_track_ids": int(m["track_id"].isna().sum()),
                    "missing_source_ids": int(m["source_id"].isna().sum()),
                    "sample_count_pass": bool(len(m) == expected),
                    "cluster_ids_pass": bool(
                        m["track_id"].notna().all()
                        and
                        m["source_id"].notna().all()
                        and
                        m["track_id"].astype(str).nunique() > 1
                        and
                        m["source_id"].astype(str).nunique() > 0
                    ),
                }
            )

    out = pd.DataFrame(rows)

    # Within each dataset/task, all models must have identical clustering vectors.
    pair_rows = []

    for dataset, task in EXPECTED:
        sub = df[
            (df["dataset"] == dataset)
            &
            (df["task"] == task)
        ].copy()

        ref = (
            sub[sub["model"] == MODELS[0]]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )

        for model in MODELS[1:]:
            cur = (
                sub[sub["model"] == model]
                .sort_values("sample_id")
                .reset_index(drop=True)
            )

            pair_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "model": model,
                    "same_sample_ids_as_reference": bool(
                        cur["sample_id"].astype(str).tolist()
                        ==
                        ref["sample_id"].astype(str).tolist()
                    ),
                    "same_track_sequence_as_reference": bool(
                        cur["track_id"].astype(str).tolist()
                        ==
                        ref["track_id"].astype(str).tolist()
                    ),
                    "same_source_sequence_as_reference": bool(
                        cur["source_id"].astype(str).tolist()
                        ==
                        ref["source_id"].astype(str).tolist()
                    ),
                }
            )

    pair = pd.DataFrame(pair_rows)

    return out, pair


def recompute_cluster_summary(df):
    instance = (
        df.groupby(
            ["dataset", "task", "model"],
            as_index=False,
        )
        .agg(
            samples=("ade", "size"),
            ADE=("ade", "mean"),
            FDE=("fde", "mean"),
        )
    )

    track_table = (
        df.groupby(
            ["dataset", "task", "model", "track_id"],
            as_index=False,
        )
        .agg(
            ADE_track=("ade", "mean"),
            FDE_track=("fde", "mean"),
            windows=("ade", "size"),
        )
    )

    track_macro = (
        track_table.groupby(
            ["dataset", "task", "model"],
            as_index=False,
        )
        .agg(
            tracks=("track_id", "nunique"),
            ADE_track_macro=("ADE_track", "mean"),
            FDE_track_macro=("FDE_track", "mean"),
        )
    )

    source_table = (
        df.groupby(
            ["dataset", "task", "model", "source_id"],
            as_index=False,
        )
        .agg(
            ADE_source=("ade", "mean"),
            FDE_source=("fde", "mean"),
            windows=("ade", "size"),
        )
    )

    source_macro = (
        source_table.groupby(
            ["dataset", "task", "model"],
            as_index=False,
        )
        .agg(
            sources=("source_id", "nunique"),
            ADE_source_macro=("ADE_source", "mean"),
            FDE_source_macro=("FDE_source", "mean"),
        )
    )

    return (
        instance
        .merge(
            track_macro,
            on=["dataset", "task", "model"],
            how="left",
        )
        .merge(
            source_macro,
            on=["dataset", "task", "model"],
            how="left",
        )
    )


def main():
    print(
        "\nR3-A — TRACK-CLUSTER RECOVERY PREFLIGHT"
    )

    print(
        "Repairs only the clustering namespace needed for paired bootstrap. "
        "ADE/FDE values are not changed.\n"
    )

    verify_r2e()

    r2e = pd.read_csv(R2E_SAMPLE)

    patched, audit = recover_testreal_track_ids(r2e)

    checks, paired_checks = validate_clusters(patched)

    if not checks["sample_count_pass"].all():
        raise RuntimeError(
            "Sample-count validation failed after track recovery."
        )

    if not checks["cluster_ids_pass"].all():
        raise RuntimeError(
            "Cluster ID validation failed after track recovery."
        )

    if not paired_checks[
        [
            "same_sample_ids_as_reference",
            "same_track_sequence_as_reference",
            "same_source_sequence_as_reference",
        ]
    ].all().all():
        raise RuntimeError(
            "Paired clustering namespace is not identical across models."
        )

    summary = recompute_cluster_summary(patched)

    sample_path = OUT / "R3A_six_model_same_sample_metrics_with_clusters.csv"
    audit_path = OUT / "R3A_testreal_track_recovery_audit.csv"
    checks_path = OUT / "R3A_cluster_readiness_checks.csv"
    pair_path = OUT / "R3A_paired_cluster_alignment_checks.csv"
    summary_path = OUT / "R3A_cluster_ready_summary.csv"

    patched.to_csv(sample_path, index=False)
    audit.to_csv(audit_path, index=False)
    checks.to_csv(checks_path, index=False)
    paired_checks.to_csv(pair_path, index=False)
    summary.to_csv(summary_path, index=False)

    status = (
        "R3A_TRACK_SOURCE_CLUSTERS_RECOVERED_AND_PAIRED_READY_FOR_BOOTSTRAP"
    )

    manifest = {
        "step": "R3A_TRACK_CLUSTER_RECOVERY_PREFLIGHT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "metric_values_modified": False,
        "source_r2e": str(R2E_SAMPLE),
        "source_r2e_sha256": sha256_file(R2E_SAMPLE),
        "cluster_ready_sample_csv": str(sample_path),
        "cluster_ready_sample_sha256": sha256_file(sample_path),
        "audit_csv": str(audit_path),
        "readiness_csv": str(checks_path),
        "paired_alignment_csv": str(pair_path),
        "summary_csv": str(summary_path),
    }

    manifest_path = OUT / "R3A_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("TESTREAL TRACK RECOVERY AUDIT:")
    print(audit.to_string(index=False))

    print("\nCLUSTER READINESS CHECKS:")
    print(checks.to_string(index=False))

    print("\nPAIRED CLUSTER ALIGNMENT CHECKS:")
    print(paired_checks.to_string(index=False))

    print("\nCLUSTER-READY SUMMARY:")
    print(summary.to_string(index=False))

    print("\nSTATUS:")
    print(status)

    print("\nR3-A COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
