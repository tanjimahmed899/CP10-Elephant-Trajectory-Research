from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

B1_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B1_stationary_cv_transformer"
)

B2_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2_testreal_kalman_gru_lstm"
)

B1_MANIFEST = B1_DIR / "R2D_B1_MANIFEST.json"
B1_SAMPLE = B1_DIR / "R2D_B1_sample_metrics.csv"

B2_MANIFEST = B2_DIR / "R2D_B2_MANIFEST.json"
B2_SAMPLE = B2_DIR / "R2D_B2_sample_metrics.csv"


B1_INPUT_PROVENANCE = B1_DIR / "R2D_B1_input_provenance.csv"
B2_INPUT_PROVENANCE = B2_DIR / "R2D_B2_input_provenance.csv"

FRESH_PRIMARY = (
    MAIN_RESULT_2
    / "step3_fresh_wildbox_full_comparison"
    / "step3_new_model_sample_metrics.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2E_six_model_fair_benchmark"
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

FRESH_NEEDED = [
    "KalmanCV",
    "GRU",
    "LSTM",
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


def normalize_model(x):
    s = str(x).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    if "stationary" in s:
        return "Stationary"
    if "kalman" in s:
        return "KalmanCV"
    if "constantvelocity" in s or s in {"cv", "constantvel"}:
        return "ConstantVelocity"
    if "lstm" in s:
        return "LSTM"
    if "gru" in s:
        return "GRU"
    if "transformer" in s:
        return "TransformerEnsemble"

    return None


def normalize_task(x):
    s = str(x).strip().lower().replace(" ", "")

    if any(v in s for v in ["2s_to_2s", "2s-to-2s", "2to2", "2s2s", "2→2"]):
        return "2s_to_2s"

    if any(v in s for v in ["2s_to_5s", "2s-to-5s", "2to5", "2s5s", "2→5"]):
        return "2s_to_5s"

    return None


def find_col(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def verify_inputs():
    for p in [
        B1_MANIFEST,
        B1_SAMPLE,
        B2_MANIFEST,
        B2_SAMPLE,
        B1_INPUT_PROVENANCE,
        B2_INPUT_PROVENANCE,
        FRESH_PRIMARY,
    ]:
        require(p)

    b1 = json.loads(B1_MANIFEST.read_text(encoding="utf-8"))
    b2 = json.loads(B2_MANIFEST.read_text(encoding="utf-8"))

    if (
        b1.get("status")
        !=
        "R2D_B1_TARGETED_RECOMPUTATION_COMPLETE_REFERENCE_SANITY_PASS"
    ):
        raise RuntimeError(f"Unexpected B1 status: {b1.get('status')}")

    if (
        b2.get("status")
        !=
        "R2D_B2_TESTREAL_KALMAN_GRU_LSTM_SAMPLE_METRICS_COMPLETE_REFERENCE_SANITY_PASS"
    ):
        raise RuntimeError(f"Unexpected B2 status: {b2.get('status')}")

    # Verify sample files against their manifests if hashes exist.
    b1_hash = b1.get("sample_metrics_sha256")
    if b1_hash and sha256_file(B1_SAMPLE) != b1_hash:
        raise RuntimeError("B1 sample metrics hash mismatch.")

    b2_hash = b2.get("sample_metrics_sha256")
    if b2_hash and sha256_file(B2_SAMPLE) != b2_hash:
        raise RuntimeError("B2 sample metrics hash mismatch.")

    return b1, b2


def canonicalize_revision_df(df, expected_dataset=None):
    required = {
        "dataset",
        "task",
        "model",
        "sample_id",
        "ade",
        "fde",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Revision sample table missing columns: {sorted(missing)}"
        )

    out = df.copy()

    out["dataset"] = out["dataset"].astype(str)
    out["task"] = out["task"].astype(str)
    out["model"] = out["model"].map(normalize_model)

    if out["model"].isna().any():
        bad = sorted(
            set(
                df.loc[out["model"].isna(), "model"].astype(str)
            )
        )
        raise RuntimeError(f"Unrecognized model labels: {bad}")

    out["sample_id"] = out["sample_id"].astype(str)

    if "track_id" not in out.columns:
        out["track_id"] = None

    if "source_id" not in out.columns:
        out["source_id"] = None

    out["track_id"] = out["track_id"].astype(str)
    out["source_id"] = out["source_id"].astype(str)

    if expected_dataset is not None:
        out = out[out["dataset"] == expected_dataset].copy()

    return out[
        [
            "dataset",
            "task",
            "model",
            "sample_id",
            "track_id",
            "source_id",
            "ade",
            "fde",
        ]
    ]


def canonicalize_fresh_existing(df):
    model_col = find_col(
        df,
        ["model", "method", "model_name", "baseline", "predictor"],
    )
    task_col = find_col(
        df,
        ["task", "forecast_task", "horizon"],
    )
    ade_col = find_col(
        df,
        ["ADE", "ade", "sample_ADE", "sample_ade"],
    )
    fde_col = find_col(
        df,
        ["FDE", "fde", "sample_FDE", "sample_fde"],
    )

    if model_col is None or task_col is None or ade_col is None or fde_col is None:
        raise RuntimeError(
            "Fresh existing sample metrics missing model/task/ADE/FDE columns. "
            f"Columns={list(df.columns)}"
        )

    sample_col = find_col(
        df,
        [
            "sample_id",
            "sample_index",
            "window_id",
            "window_index",
            "index",
        ],
    )

    track_col = find_col(
        df,
        [
            "track_id",
            "tracklet_id",
            "track_key",
            "track",
        ],
    )

    source_col = find_col(
        df,
        [
            "source_id",
            "source_group",
            "source",
            "video_id",
            "video",
            "sequence_id",
            "sequence",
        ],
    )

    seed_col = find_col(
        df,
        ["seed", "random_seed"],
    )

    out = pd.DataFrame(
        {
            "task": df[task_col].map(normalize_task),
            "model": df[model_col].map(normalize_model),
            "ade": pd.to_numeric(df[ade_col], errors="coerce"),
            "fde": pd.to_numeric(df[fde_col], errors="coerce"),
        }
    )

    if out["task"].isna().any():
        bad = sorted(
            set(
                df.loc[out["task"].isna(), task_col].astype(str)
            )
        )
        raise RuntimeError(f"Unrecognized Fresh task labels: {bad}")

    if out["model"].isna().any():
        bad = sorted(
            set(
                df.loc[out["model"].isna(), model_col].astype(str)
            )
        )
        raise RuntimeError(f"Unrecognized Fresh model labels: {bad}")

    if sample_col is not None:
        out["raw_sample_id"] = df[sample_col].astype(str)
    else:
        out["raw_sample_id"] = None

    if track_col is not None:
        out["track_id"] = df[track_col].astype(str)
    else:
        out["track_id"] = None

    if source_col is not None:
        out["source_id"] = df[source_col].astype(str)
    else:
        out["source_id"] = None

    if seed_col is not None:
        out["seed"] = df[seed_col]
    else:
        out["seed"] = np.nan

    out["dataset"] = "Fresh_WildBox"

    out = out[
        out["model"].isin(FRESH_NEEDED)
    ].copy()

    if out["ade"].isna().any() or out["fde"].isna().any():
        raise RuntimeError("Fresh existing metrics contain nonnumeric ADE/FDE.")

    return out, {
        "model_col": model_col,
        "task_col": task_col,
        "ade_col": ade_col,
        "fde_col": fde_col,
        "sample_col": sample_col,
        "track_col": track_col,
        "source_col": source_col,
        "seed_col": seed_col,
    }


def build_fresh_model_table(existing, b1_fresh):
    """
    Align Fresh Kalman/GRU/LSTM to the exact B1 Fresh sample order.

    Preferred path:
      1. exact raw sample_id if it yields a 1:1 mapping;
      2. otherwise row-order within task/model, but ONLY if every required model
         has exactly the expected sample count and metadata track/source sequences
         agree with B1 where both are available.

    Seed-level GRU/LSTM rows are reduced to per-sample mean metrics across the
    frozen seeds only if multiple rows per sample are present.
    """
    rows = []
    audit = []

    for task in ["2s_to_2s", "2s_to_5s"]:
        expected = EXPECTED[("Fresh_WildBox", task)]

        ref = (
            b1_fresh[
                (b1_fresh["task"] == task)
                &
                (b1_fresh["model"] == "TransformerEnsemble")
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(ref) != expected:
            raise RuntimeError(
                f"B1 Fresh Transformer reference count mismatch for {task}: "
                f"{len(ref)} != {expected}"
            )

        for model in FRESH_NEEDED:
            sub = existing[
                (existing["task"] == task)
                &
                (existing["model"] == model)
            ].copy()

            if len(sub) == 0:
                raise RuntimeError(
                    f"No existing Fresh rows for {task} {model}"
                )

            # If seed-level rows exist, aggregate only when repeated sample IDs or
            # exact multiple-of-N row counts prove a seed-expanded table.
            if len(sub) != expected:
                if len(sub) % expected != 0:
                    raise RuntimeError(
                        f"Fresh {task} {model}: row count {len(sub)} is not "
                        f"expected {expected} or a whole multiple of it."
                    )

                repeat = len(sub) // expected

                if repeat <= 1:
                    raise RuntimeError(
                        f"Fresh {task} {model}: unexpected row structure."
                    )

                # Prefer raw_sample_id grouping when it gives exactly expected groups.
                if sub["raw_sample_id"].notna().all():
                    grouped = (
                        sub.groupby(
                            "raw_sample_id",
                            sort=False,
                            as_index=False,
                        )
                        .agg(
                            ade=("ade", "mean"),
                            fde=("fde", "mean"),
                            track_id=("track_id", "first"),
                            source_id=("source_id", "first"),
                        )
                    )

                    if len(grouped) == expected:
                        sub = grouped.copy()
                        sub["raw_sample_id"] = sub["raw_sample_id"].astype(str)

                    else:
                        # Fall back to positional seed blocks only if each block has N.
                        sub = sub.reset_index(drop=True)
                        block_metrics = []

                        for block in range(repeat):
                            block_df = sub.iloc[
                                block * expected:
                                (block + 1) * expected
                            ].reset_index(drop=True)

                            block_metrics.append(
                                block_df[["ade", "fde"]].to_numpy(dtype=float)
                            )

                        mean_metrics = np.mean(
                            np.stack(block_metrics, axis=0),
                            axis=0,
                        )

                        sub = sub.iloc[:expected].reset_index(drop=True).copy()
                        sub["ade"] = mean_metrics[:, 0]
                        sub["fde"] = mean_metrics[:, 1]

                else:
                    sub = sub.reset_index(drop=True)
                    block_metrics = []

                    for block in range(repeat):
                        block_df = sub.iloc[
                            block * expected:
                            (block + 1) * expected
                        ].reset_index(drop=True)

                        block_metrics.append(
                            block_df[["ade", "fde"]].to_numpy(dtype=float)
                        )

                    mean_metrics = np.mean(
                        np.stack(block_metrics, axis=0),
                        axis=0,
                    )

                    sub = sub.iloc[:expected].reset_index(drop=True).copy()
                    sub["ade"] = mean_metrics[:, 0]
                    sub["fde"] = mean_metrics[:, 1]

            sub = sub.reset_index(drop=True)

            if len(sub) != expected:
                raise RuntimeError(
                    f"Fresh {task} {model}: final count {len(sub)} != {expected}"
                )

            alignment_method = None

            # Exact sample ID mapping, if raw IDs equal B1 IDs as sets.
            if sub["raw_sample_id"].notna().all():
                ref_ids = set(ref["sample_id"].astype(str))
                sub_ids = set(sub["raw_sample_id"].astype(str))

                if (
                    len(ref_ids) == expected
                    and
                    len(sub_ids) == expected
                    and
                    ref_ids == sub_ids
                ):
                    sub_indexed = (
                        sub.set_index(
                            sub["raw_sample_id"].astype(str)
                        )
                    )

                    aligned = sub_indexed.loc[
                        ref["sample_id"].astype(str)
                    ].reset_index(drop=True)

                    alignment_method = "exact_sample_id"

                else:
                    aligned = sub.copy()
            else:
                aligned = sub.copy()

            # If exact IDs did not work, require strong positional/metadata evidence.
            if alignment_method is None:
                aligned = aligned.reset_index(drop=True)

                track_match = None
                source_match = None

                if (
                    aligned["track_id"].notna().all()
                    and
                    ref["track_id"].notna().all()
                ):
                    track_match = (
                        aligned["track_id"].astype(str).tolist()
                        ==
                        ref["track_id"].astype(str).tolist()
                    )

                if (
                    aligned["source_id"].notna().all()
                    and
                    ref["source_id"].notna().all()
                ):
                    source_match = (
                        aligned["source_id"].astype(str).tolist()
                        ==
                        ref["source_id"].astype(str).tolist()
                    )

                # Require at least one metadata sequence match if IDs are unavailable.
                if track_match is not True and source_match is not True:
                    raise RuntimeError(
                        f"Fresh {task} {model}: cannot prove positional sample "
                        "alignment to B1 reference. Exact sample IDs did not match, "
                        f"track_match={track_match}, source_match={source_match}."
                    )

                alignment_method = "row_order_with_metadata_sequence_match"

            final = pd.DataFrame(
                {
                    "dataset": "Fresh_WildBox",
                    "task": task,
                    "model": model,
                    "sample_id": ref["sample_id"].astype(str).values,
                    "track_id": ref["track_id"].astype(str).values,
                    "source_id": ref["source_id"].astype(str).values,
                    "ade": aligned["ade"].astype(float).values,
                    "fde": aligned["fde"].astype(float).values,
                }
            )

            rows.append(final)

            audit.append(
                {
                    "dataset": "Fresh_WildBox",
                    "task": task,
                    "model": model,
                    "expected_samples": expected,
                    "aligned_samples": len(final),
                    "alignment_method": alignment_method,
                }
            )

    return pd.concat(rows, ignore_index=True), pd.DataFrame(audit)



def remap_testreal_b2_to_b1_ids(
    b1,
    b2,
):
    """
    B1 and B2 were generated from the same frozen TestReal samples but used
    different sample-id namespaces. Remap B2 sample IDs to B1 canonical IDs
    only after proving identical input metadata provenance and identical
    track/source row sequences.

    No metric value is modified.
    """
    b1_prov = pd.read_csv(
        B1_INPUT_PROVENANCE
    )

    b2_prov = pd.read_csv(
        B2_INPUT_PROVENANCE
    )

    audit = []
    remapped_parts = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        expected = EXPECTED[
            (
                "BIRDSAI_TestReal",
                task,
            )
        ]

        # ------------------------------------------------------------
        # Provenance proof: same metadata path + SHA256.
        # ------------------------------------------------------------
        p1 = b1_prov[
            (
                b1_prov["dataset"].astype(str)
                ==
                "BIRDSAI_TestReal"
            )
            &
            (
                b1_prov["task"].astype(str)
                ==
                task
            )
        ].copy()

        p2 = b2_prov[
            b2_prov["task"].astype(str)
            ==
            task
        ].copy()

        if len(p1) != 1 or len(p2) != 1:
            raise RuntimeError(
                f"TestReal {task}: expected one B1 and one B2 provenance row; "
                f"got B1={len(p1)}, B2={len(p2)}"
            )

        p1 = p1.iloc[0]
        p2 = p2.iloc[0]

        metadata_path_match = (
            str(
                Path(
                    str(
                        p1["metadata"]
                    )
                )
            ).lower()
            ==
            str(
                Path(
                    str(
                        p2["metadata"]
                    )
                )
            ).lower()
        )

        metadata_hash_match = (
            str(
                p1["metadata_sha256"]
            )
            ==
            str(
                p2["metadata_sha256"]
            )
        )

        npz_path_match = (
            str(
                Path(
                    str(
                        p1["npz"]
                    )
                )
            ).lower()
            ==
            str(
                Path(
                    str(
                        p2["npz"]
                    )
                )
            ).lower()
        )

        npz_hash_match = (
            str(
                p1["npz_sha256"]
            )
            ==
            str(
                p2["npz_sha256"]
            )
        )

        if not (
            metadata_path_match
            and
            metadata_hash_match
            and
            npz_path_match
            and
            npz_hash_match
        ):
            raise RuntimeError(
                f"TestReal {task}: B1/B2 frozen input provenance mismatch. "
                f"metadata_path_match={metadata_path_match}, "
                f"metadata_hash_match={metadata_hash_match}, "
                f"npz_path_match={npz_path_match}, "
                f"npz_hash_match={npz_hash_match}"
            )

        # Use B1 Transformer row order as canonical sample identity.
        ref = (
            b1[
                (
                    b1["dataset"]
                    ==
                    "BIRDSAI_TestReal"
                )
                &
                (
                    b1["task"]
                    ==
                    task
                )
                &
                (
                    b1["model"]
                    ==
                    "TransformerEnsemble"
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if len(ref) != expected:
            raise RuntimeError(
                f"TestReal {task}: B1 canonical reference count "
                f"{len(ref)} != {expected}"
            )

        # Also prove B1 Stationary/CV share exact same row-order metadata.
        for model in [
            "Stationary",
            "ConstantVelocity",
        ]:
            chk = (
                b1[
                    (
                        b1["dataset"]
                        ==
                        "BIRDSAI_TestReal"
                    )
                    &
                    (
                        b1["task"]
                        ==
                        task
                    )
                    &
                    (
                        b1["model"]
                        ==
                        model
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            if len(chk) != expected:
                raise RuntimeError(
                    f"TestReal {task} {model}: count {len(chk)} != {expected}"
                )

            if (
                chk["track_id"].astype(str).tolist()
                !=
                ref["track_id"].astype(str).tolist()
            ):
                raise RuntimeError(
                    f"TestReal {task} {model}: B1 track sequence differs "
                    "from Transformer canonical reference."
                )

            if (
                chk["source_id"].astype(str).tolist()
                !=
                ref["source_id"].astype(str).tolist()
            ):
                raise RuntimeError(
                    f"TestReal {task} {model}: B1 source sequence differs "
                    "from Transformer canonical reference."
                )

        for model in [
            "KalmanCV",
            "GRU",
            "LSTM",
        ]:
            sub = (
                b2[
                    (
                        b2["dataset"]
                        ==
                        "BIRDSAI_TestReal"
                    )
                    &
                    (
                        b2["task"]
                        ==
                        task
                    )
                    &
                    (
                        b2["model"]
                        ==
                        model
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            if len(sub) != expected:
                raise RuntimeError(
                    f"TestReal {task} {model}: B2 count {len(sub)} != {expected}"
                )

            track_match = (
                sub["track_id"].astype(str).tolist()
                ==
                ref["track_id"].astype(str).tolist()
            )

            source_match = (
                sub["source_id"].astype(str).tolist()
                ==
                ref["source_id"].astype(str).tolist()
            )

            if not (
                track_match
                and
                source_match
            ):
                raise RuntimeError(
                    f"TestReal {task} {model}: cannot safely remap sample IDs; "
                    f"track_match={track_match}, source_match={source_match}"
                )

            sub["sample_id"] = (
                ref["sample_id"]
                .astype(str)
                .values
            )

            remapped_parts.append(
                sub
            )

            audit.append(
                {
                    "dataset":
                        "BIRDSAI_TestReal",

                    "task":
                        task,

                    "model":
                        model,

                    "samples":
                        expected,

                    "metadata_path_match":
                        True,

                    "metadata_hash_match":
                        True,

                    "npz_path_match":
                        True,

                    "npz_hash_match":
                        True,

                    "track_sequence_match":
                        True,

                    "source_sequence_match":
                        True,

                    "sample_id_action":
                        "B2 IDs remapped to B1 canonical IDs by proven identical row order",
                }
            )

    remapped = pd.concat(
        remapped_parts,
        ignore_index=True,
    )

    return remapped, pd.DataFrame(
        audit
    )

def same_sample_check(df):
    audit = []

    for (dataset, task), expected in EXPECTED.items():
        sub = df[
            (df["dataset"] == dataset)
            &
            (df["task"] == task)
        ].copy()

        model_sets = {}

        for model in MODELS:
            m = sub[sub["model"] == model].copy()
            ids = set(m["sample_id"].astype(str))
            model_sets[model] = ids

        counts = {
            model: len(model_sets[model])
            for model in MODELS
        }

        intersection = set.intersection(
            *[
                model_sets[m]
                for m in MODELS
            ]
        )

        all_equal = all(
            model_sets[m] == model_sets[MODELS[0]]
            for m in MODELS[1:]
        )

        audit.append(
            {
                "dataset": dataset,
                "task": task,
                "expected_samples": expected,
                "same_sample_intersection": len(intersection),
                "all_model_sample_sets_equal": bool(all_equal),
                "all_expected_samples_aligned": bool(
                    all_equal
                    and
                    len(intersection) == expected
                    and
                    all(counts[m] == expected for m in MODELS)
                ),
                **{
                    f"{model}_samples": counts[model]
                    for model in MODELS
                },
            }
        )

    return pd.DataFrame(audit)


def summary_tables(df):
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

    track = (
        df.groupby(
            ["dataset", "task", "model", "track_id"],
            as_index=False,
        )
        .agg(
            ADE_track=("ade", "mean"),
            FDE_track=("fde", "mean"),
        )
        .groupby(
            ["dataset", "task", "model"],
            as_index=False,
        )
        .agg(
            ADE_track_macro=("ADE_track", "mean"),
            FDE_track_macro=("FDE_track", "mean"),
            tracks=("track_id", "nunique"),
        )
    )

    source = (
        df.groupby(
            ["dataset", "task", "model", "source_id"],
            as_index=False,
        )
        .agg(
            ADE_source=("ade", "mean"),
            FDE_source=("fde", "mean"),
        )
        .groupby(
            ["dataset", "task", "model"],
            as_index=False,
        )
        .agg(
            ADE_source_macro=("ADE_source", "mean"),
            FDE_source_macro=("FDE_source", "mean"),
            sources=("source_id", "nunique"),
        )
    )

    combined = (
        instance
        .merge(
            track,
            on=["dataset", "task", "model"],
            how="left",
        )
        .merge(
            source,
            on=["dataset", "task", "model"],
            how="left",
        )
    )

    order = {
        model: i
        for i, model in enumerate(MODELS)
    }

    combined["_model_order"] = combined["model"].map(order)

    combined = combined.sort_values(
        ["dataset", "task", "_model_order"]
    ).drop(columns="_model_order")

    return combined


def main():
    print(
        "\nR2E-V2 — SIX-MODEL FAIR SAME-SAMPLE BENCHMARK CONSOLIDATION"
    )

    print(
        "Combines recomputed B1/B2 rows with existing frozen Fresh "
        "Kalman/GRU/LSTM rows. No retraining and no new model selection.\n"
    )

    verify_inputs()

    b1 = canonicalize_revision_df(
        pd.read_csv(B1_SAMPLE)
    )

    b2 = canonicalize_revision_df(
        pd.read_csv(B2_SAMPLE),
        expected_dataset="BIRDSAI_TestReal",
    )

    b2, testreal_alignment = remap_testreal_b2_to_b1_ids(
        b1,
        b2,
    )

    fresh_existing_raw = pd.read_csv(FRESH_PRIMARY)

    fresh_existing, schema = canonicalize_fresh_existing(
        fresh_existing_raw
    )

    b1_fresh = b1[
        b1["dataset"] == "Fresh_WildBox"
    ].copy()

    fresh_extra, fresh_alignment = build_fresh_model_table(
        fresh_existing,
        b1_fresh,
    )

    # B1 contributes Stationary/CV/Transformer for both datasets.
    # B2 contributes Kalman/GRU/LSTM for TestReal.
    # Existing Fresh contributes Kalman/GRU/LSTM for Fresh.
    final = pd.concat(
        [
            b1,
            b2,
            fresh_extra,
        ],
        ignore_index=True,
    )

    # Strict uniqueness: one row per dataset/task/model/sample_id.
    dup = final.duplicated(
        subset=[
            "dataset",
            "task",
            "model",
            "sample_id",
        ],
        keep=False,
    )

    if dup.any():
        bad = final.loc[
            dup,
            [
                "dataset",
                "task",
                "model",
                "sample_id",
            ],
        ].head(50)

        raise RuntimeError(
            "Duplicate model/sample rows after merge:\n"
            + bad.to_string(index=False)
        )

    alignment = same_sample_check(final)

    if not alignment["all_expected_samples_aligned"].all():
        raise RuntimeError(
            "Final six-model same-sample alignment failed:\n"
            + alignment.to_string(index=False)
        )

    summary = summary_tables(final)

    final_path = OUT / "R2E_six_model_same_sample_metrics.csv"
    summary_path = OUT / "R2E_six_model_fair_summary.csv"
    alignment_path = OUT / "R2E_same_sample_alignment.csv"
    fresh_alignment_path = OUT / "R2E_fresh_existing_alignment_audit.csv"
    testreal_alignment_path = OUT / "R2E_testreal_B1_B2_alignment_audit.csv"

    final.to_csv(final_path, index=False)
    summary.to_csv(summary_path, index=False)
    alignment.to_csv(alignment_path, index=False)
    fresh_alignment.to_csv(fresh_alignment_path, index=False)
    testreal_alignment.to_csv(testreal_alignment_path, index=False)

    provenance = {
        "step": "R2E_SIX_MODEL_FAIR_SAME_SAMPLE_BENCHMARK",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "R2E_SIX_MODEL_FAIR_SAME_SAMPLE_BENCHMARK_COMPLETE_READY_FOR_R3_CLUSTERED_BOOTSTRAP",
        "retraining": False,
        "model_selection": False,
        "fresh_existing_source": str(FRESH_PRIMARY),
        "fresh_existing_source_sha256": sha256_file(FRESH_PRIMARY),
        "fresh_existing_schema": schema,
        "b1_source": str(B1_SAMPLE),
        "b1_source_sha256": sha256_file(B1_SAMPLE),
        "b2_source": str(B2_SAMPLE),
        "b2_source_sha256": sha256_file(B2_SAMPLE),
        "final_sample_metrics": str(final_path),
        "final_sample_metrics_sha256": sha256_file(final_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "alignment": str(alignment_path),
        "alignment_sha256": sha256_file(alignment_path),
        "testreal_alignment_audit": str(testreal_alignment_path),
        "testreal_alignment_audit_sha256": sha256_file(testreal_alignment_path),
    }

    manifest_path = OUT / "R2E_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    print("TESTREAL B1/B2 ALIGNMENT AUDIT:")
    print(
        testreal_alignment.to_string(index=False)
    )

    print("\nFRESH EXISTING ALIGNMENT AUDIT:")
    print(
        fresh_alignment.to_string(index=False)
    )

    print("\nSAME-SAMPLE ALIGNMENT:")
    print(
        alignment.to_string(index=False)
    )

    print("\nFAIR SIX-MODEL SUMMARY:")
    print(
        summary.to_string(index=False)
    )

    print("\nSTATUS:")
    print(
        provenance["status"]
    )

    print("\nR2E COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
