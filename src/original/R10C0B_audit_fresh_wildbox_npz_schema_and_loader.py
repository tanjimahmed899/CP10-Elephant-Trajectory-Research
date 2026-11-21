
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R2_ROOT = REVISION_ROOT / "R2_fair_forecasting"
R10_ROOT = REVISION_ROOT / "R10_ensemble_analysis"

R10C0_DIR = R10_ROOT / "R10C0_fresh_wildbox_input_provenance_lock"
R10C0_MANIFEST = R10C0_DIR / "R10C0_MANIFEST.json"

OUT = (
    R10_ROOT
    / "R10C0B_fresh_wildbox_schema_and_loader_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "2s_to_2s": {
        "npz": (
            MAIN1
            / "wildbox_fresh_confirmatory_results"
            / "2s_to_2s_fresh_dataset.npz"
        ),
        "metadata": (
            MAIN1
            / "wildbox_fresh_confirmatory_results"
            / "2s_to_2s_fresh_metadata.csv"
        ),
        "expected_samples": 865,
        "expected_future_length": 20,
    },
    "2s_to_5s": {
        "npz": (
            MAIN1
            / "wildbox_fresh_confirmatory_results"
            / "2s_to_5s_fresh_dataset.npz"
        ),
        "metadata": (
            MAIN1
            / "wildbox_fresh_confirmatory_results"
            / "2s_to_5s_fresh_metadata.csv"
        ),
        "expected_samples": 637,
        "expected_future_length": 50,
    },
}

TEXT_ROOTS = [
    MAIN1,
    R2_ROOT,
]

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".txt",
    ".md",
}

# Selected CSVs are small/provenance-oriented and useful for source recovery.
SELECTED_CSV_NAME_PARTS = [
    "provenance",
    "manifest",
    "selected",
    "source",
    "readiness",
    "alignment",
    "audit",
    "summary",
    "inventory",
]


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


def verify_r10c0():
    require(R10C0_MANIFEST)

    m = json.loads(
        R10C0_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R10C0_FRESH_WILDBOX_INPUTS_NOT_UNIQUELY_RESOLVED_REVIEW_CANDIDATES_BEFORE_INFERENCE"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R10-C0 status: {m.get('status')}"
        )

    return m


def numeric_summary(arr):
    a = np.asarray(arr)

    out = {
        "shape": str(tuple(a.shape)),
        "dtype": str(a.dtype),
        "ndim": int(a.ndim),
        "finite_fraction": None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }

    if np.issubdtype(a.dtype, np.number):
        af = a.astype(np.float64, copy=False)
        finite = np.isfinite(af)

        out["finite_fraction"] = float(
            finite.mean()
        )

        if finite.any():
            vals = af[finite]

            out["min"] = float(vals.min())
            out["max"] = float(vals.max())
            out["mean"] = float(vals.mean())
            out["std"] = float(vals.std())

    return out


def audit_npz_and_metadata():
    array_rows = []
    dataset_rows = []
    metadata_rows = []

    for task, cfg in FILES.items():
        npz_path = cfg["npz"]
        meta_path = cfg["metadata"]

        require(npz_path)
        require(meta_path)

        arrays = np.load(
            npz_path,
            allow_pickle=False
        )

        for key in arrays.files:
            s = numeric_summary(
                arrays[key]
            )

            array_rows.append(
                {
                    "task": task,
                    "npz_path": str(npz_path),
                    "npz_sha256": sha256_file(npz_path),
                    "array_key": key,
                    **s,
                }
            )

        meta = pd.read_csv(
            meta_path,
            dtype={
                "sample_id": str,
                "source_group": str,
                "track_key": str,
                "segment": str,
            }
        )

        if len(meta) != cfg["expected_samples"]:
            raise RuntimeError(
                f"{task}: metadata rows={len(meta)}, "
                f"expected={cfg['expected_samples']}"
            )

        metadata_rows.append(
            {
                "task": task,
                "metadata_path": str(meta_path),
                "metadata_sha256": sha256_file(meta_path),
                "rows": int(len(meta)),
                "columns": "|".join(
                    str(c)
                    for c in meta.columns
                ),
                "unique_sample_id": (
                    int(meta["sample_id"].nunique())
                    if "sample_id" in meta.columns
                    else None
                ),
                "unique_track_key": (
                    int(meta["track_key"].nunique())
                    if "track_key" in meta.columns
                    else None
                ),
                "unique_source_group": (
                    int(meta["source_group"].nunique())
                    if "source_group" in meta.columns
                    else None
                ),
                "unique_segment": (
                    int(meta["segment"].nunique())
                    if "segment" in meta.columns
                    else None
                ),
            }
        )

        candidate_sample_axes = []

        for key in arrays.files:
            a = np.asarray(
                arrays[key]
            )

            if (
                a.ndim >= 1
                and
                a.shape[0] == cfg["expected_samples"]
            ):
                candidate_sample_axes.append(
                    key
                )

        future_length_matches = []

        for key in arrays.files:
            a = np.asarray(
                arrays[key]
            )

            if (
                a.ndim >= 2
                and
                a.shape[0] == cfg["expected_samples"]
                and
                a.shape[1] == cfg["expected_future_length"]
            ):
                future_length_matches.append(
                    key
                )

        dataset_rows.append(
            {
                "task": task,
                "expected_samples": cfg["expected_samples"],
                "expected_future_length": cfg["expected_future_length"],
                "npz_keys": "|".join(arrays.files),
                "arrays_with_expected_sample_axis":
                    "|".join(candidate_sample_axes),
                "arrays_matching_expected_future_length":
                    "|".join(future_length_matches),
                "metadata_rows": int(len(meta)),
                "metadata_sample_id_unique": (
                    bool(
                        meta["sample_id"].nunique()
                        ==
                        len(meta)
                    )
                    if "sample_id" in meta.columns
                    else False
                ),
            }
        )

    return (
        pd.DataFrame(array_rows),
        pd.DataFrame(dataset_rows),
        pd.DataFrame(metadata_rows),
    )


def should_scan_text_file(path):
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return True

    if suffix == ".csv":
        name = path.name.lower()

        return any(
            token in name
            for token in SELECTED_CSV_NAME_PARTS
        )

    return False


def search_source_references():
    target_names = [
        "2s_to_2s_fresh_dataset.npz",
        "2s_to_5s_fresh_dataset.npz",
        "2s_to_2s_fresh_metadata.csv",
        "2s_to_5s_fresh_metadata.csv",
        "wildbox_fresh_confirmatory_results",
    ]

    rows = []
    seen = set()

    for root in TEXT_ROOTS:
        if not root.exists():
            continue

        for p in root.glob("**/*"):
            if not p.is_file():
                continue

            if not should_scan_text_file(p):
                continue

            # Skip giant text-like artifacts.
            try:
                if p.stat().st_size > 5_000_000:
                    continue
            except Exception:
                continue

            resolved = str(
                p.resolve()
            ).lower()

            if resolved in seen:
                continue

            seen.add(resolved)

            try:
                lines = p.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).splitlines()
            except Exception:
                continue

            for i, line in enumerate(
                lines,
                start=1
            ):
                lower = line.lower()

                if any(
                    target.lower() in lower
                    for target in target_names
                ):
                    rows.append(
                        {
                            "source_file": str(p),
                            "line_number": i,
                            "line": line,
                        }
                    )

    return pd.DataFrame(rows)


def search_loader_patterns():
    """
    Find source code likely responsible for loading fresh WildBox NPZs and
    adapting them for frozen neural models.
    """
    needles = [
        "np.load(",
        "history_canonical",
        "future_canonical",
        "history_abs",
        "future_abs",
        "wildbox_fresh",
        "fresh_dataset",
        "prediction_abs",
        "evaluate(",
        "predict_model(",
        "canonical",
    ]

    rows = []

    roots = [
        MAIN1,
        R2_ROOT,
    ]

    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for p in root.glob("**/*.py"):
            resolved = str(
                p.resolve()
            ).lower()

            if resolved in seen:
                continue

            seen.add(resolved)

            try:
                if p.stat().st_size > 2_000_000:
                    continue

                lines = p.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).splitlines()
            except Exception:
                continue

            for i, line in enumerate(
                lines,
                start=1
            ):
                low = line.lower()

                if any(
                    needle.lower() in low
                    for needle in needles
                ):
                    rows.append(
                        {
                            "source_file": str(p),
                            "line_number": i,
                            "line": line,
                        }
                    )

    return pd.DataFrame(rows)


def inspect_r2_fresh_sample_metrics():
    candidates = [
        (
            R2_ROOT
            / "R2D_B1_stationary_cv_transformer"
            / "R2D_B1_sample_metrics.csv"
        ),
        (
            R2_ROOT
            / "R2E_six_model_fair_benchmark"
            / "R2E_six_model_same_sample_metrics.csv"
        ),
        (
            R2_ROOT
            / "R2E_six_model_fair_benchmark"
            / "R2E_fresh_existing_alignment_audit.csv"
        ),
        (
            R2_ROOT
            / "R2E_six_model_fair_benchmark"
            / "R2E_same_sample_alignment.csv"
        ),
    ]

    rows = []

    for p in candidates:
        if not p.exists():
            continue

        try:
            df = pd.read_csv(
                p
            )
        except Exception as exc:
            rows.append(
                {
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "rows": None,
                    "columns": f"<read error: {exc}>",
                    "fresh_2s2_rows_detected": None,
                    "fresh_2s5_rows_detected": None,
                }
            )
            continue

        cols_lower = {
            str(c).lower(): c
            for c in df.columns
        }

        task_col = None
        for candidate in [
            "task",
            "horizon",
        ]:
            if candidate in cols_lower:
                task_col = cols_lower[candidate]
                break

        dataset_col = None
        for candidate in [
            "dataset",
            "dataset_name",
            "split",
            "domain",
            "source",
        ]:
            if candidate in cols_lower:
                dataset_col = cols_lower[candidate]
                break

        fresh2 = None
        fresh5 = None

        if task_col is not None:
            work = df.copy()

            if dataset_col is not None:
                mask = (
                    work[dataset_col]
                    .astype(str)
                    .str.lower()
                    .str.contains("wildbox|fresh", regex=True)
                )

                work = work[mask]

            fresh2 = int(
                (
                    work[task_col]
                    .astype(str)
                    ==
                    "2s_to_2s"
                ).sum()
            )

            fresh5 = int(
                (
                    work[task_col]
                    .astype(str)
                    ==
                    "2s_to_5s"
                ).sum()
            )

        rows.append(
            {
                "path": str(p),
                "sha256": sha256_file(p),
                "rows": int(len(df)),
                "columns": "|".join(
                    str(c)
                    for c in df.columns
                ),
                "fresh_2s2_rows_detected": fresh2,
                "fresh_2s5_rows_detected": fresh5,
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR10-C0B — FRESH WILDBOX NPZ SCHEMA + LOADER AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R10-C0 uniquely identified the 865/637 fresh WildBox NPZ and metadata files, "
        "but the NPZs did not match the assumed final_neural_trajectory_dataset key schema. "
        "This step inspects every NPZ key/shape/dtype, verifies metadata identity, and "
        "searches R2/source code for the exact loader or adapter previously used for these "
        "fresh WildBox files. The goal is to determine whether direct neural inference is "
        "possible or whether a deterministic schema adapter must be reproduced.\n"
    )

    verify_r10c0()

    (
        array_df,
        dataset_df,
        metadata_df,
    ) = audit_npz_and_metadata()

    source_ref_df = search_source_references()
    loader_df = search_loader_patterns()
    r2_metrics_df = inspect_r2_fresh_sample_metrics()

    array_path = (
        OUT
        /
        "R10C0B_NPZ_ARRAY_SCHEMA.csv"
    )

    dataset_path = (
        OUT
        /
        "R10C0B_DATASET_LEVEL_SCHEMA_SUMMARY.csv"
    )

    metadata_path = (
        OUT
        /
        "R10C0B_METADATA_IDENTITY_SUMMARY.csv"
    )

    source_ref_path = (
        OUT
        /
        "R10C0B_EXACT_FRESH_FILE_SOURCE_REFERENCES.csv"
    )

    loader_path = (
        OUT
        /
        "R10C0B_LOADER_PATTERN_EVIDENCE.csv"
    )

    metrics_path = (
        OUT
        /
        "R10C0B_R2_FRESH_METRIC_ARTIFACT_SUMMARY.csv"
    )

    array_df.to_csv(
        array_path,
        index=False
    )

    dataset_df.to_csv(
        dataset_path,
        index=False
    )

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    source_ref_df.to_csv(
        source_ref_path,
        index=False
    )

    loader_df.to_csv(
        loader_path,
        index=False
    )

    r2_metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # Decision is intentionally conservative.
    has_expected_sample_axes = True
    has_expected_future_axes = True

    for _, row in dataset_df.iterrows():
        if not str(
            row[
                "arrays_with_expected_sample_axis"
            ]
        ).strip():
            has_expected_sample_axes = False

        if not str(
            row[
                "arrays_matching_expected_future_length"
            ]
        ).strip():
            has_expected_future_axes = False

    loader_evidence_found = (
        len(
            source_ref_df
        )
        >
        0
    )

    if (
        has_expected_sample_axes
        and
        has_expected_future_axes
        and
        loader_evidence_found
    ):
        status = (
            "R10C0B_FRESH_WILDBOX_SCHEMA_AUDITED_LOADER_EVIDENCE_FOUND_READY_FOR_R10C1_ADAPTER_OR_DIRECT_INFERENCE_BUILD"
        )
    else:
        status = (
            "R10C0B_FRESH_WILDBOX_SCHEMA_AUDITED_MORE_LOADER_RECOVERY_NEEDED_BEFORE_INFERENCE"
        )

    manifest = {
        "step":
            "R10C0B_FRESH_WILDBOX_NPZ_SCHEMA_AND_LOADER_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "fresh_files": {
            task: {
                "npz":
                    str(
                        cfg[
                            "npz"
                        ]
                    ),

                "npz_sha256":
                    sha256_file(
                        cfg[
                            "npz"
                        ]
                    ),

                "metadata":
                    str(
                        cfg[
                            "metadata"
                        ]
                    ),

                "metadata_sha256":
                    sha256_file(
                        cfg[
                            "metadata"
                        ]
                    ),

                "expected_samples":
                    cfg[
                        "expected_samples"
                    ],

                "expected_future_length":
                    cfg[
                        "expected_future_length"
                    ],
            }
            for task, cfg in FILES.items()
        },

        "claim_guardrail":
            (
                "Do not call the fresh NPZ unusable merely because its keys differ from "
                "final_neural_trajectory_dataset. Recover the original R2 loader/adapter "
                "and reproduce that deterministic transformation before inference."
            ),

        "outputs": {
            "npz_array_schema":
                str(
                    array_path
                ),

            "dataset_level_schema":
                str(
                    dataset_path
                ),

            "metadata_identity":
                str(
                    metadata_path
                ),

            "exact_file_source_references":
                str(
                    source_ref_path
                ),

            "loader_pattern_evidence":
                str(
                    loader_path
                ),

            "r2_metric_artifacts":
                str(
                    metrics_path
                ),
        },
    }

    (
        OUT
        /
        "R10C0B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "DATASET-LEVEL SCHEMA SUMMARY:"
    )

    print(
        dataset_df.to_string(
            index=False
        )
    )

    print(
        "\nNPZ ARRAY SCHEMA:"
    )

    print(
        array_df[
            [
                "task",
                "array_key",
                "shape",
                "dtype",
                "finite_fraction",
                "min",
                "max",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nMETADATA IDENTITY SUMMARY:"
    )

    print(
        metadata_df.to_string(
            index=False
        )
    )

    print(
        "\nEXACT FRESH-FILE SOURCE REFERENCES:"
    )

    if len(
        source_ref_df
    ):
        print(
            source_ref_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No exact fresh-file filename references found in scanned source/provenance artifacts."
        )

    print(
        "\nR2 FRESH METRIC ARTIFACT SUMMARY:"
    )

    print(
        r2_metrics_df.to_string(
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
        "\nR10-C0B COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
