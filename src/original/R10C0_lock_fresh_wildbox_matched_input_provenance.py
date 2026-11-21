
from pathlib import Path
import hashlib
import json
import ast
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R2_ROOT = REVISION_ROOT / "R2_fair_forecasting"
R10_ROOT = REVISION_ROOT / "R10_ensemble_analysis"

R10B_DIR = R10_ROOT / "R10B_testreal_transformer_ensemble_value"
R10B_MANIFEST = R10B_DIR / "R10B_MANIFEST.json"

R2E_DIR = R2_ROOT / "R2E_six_model_fair_benchmark"
R2D_B1_DIR = R2_ROOT / "R2D_B1_stationary_cv_transformer"
R2A_V2_DIR = R2_ROOT / "R2A_preflight_v2"
R2D_A_DIR = R2_ROOT / "R2D_targeted_recompute_preflight"

CANDIDATE_FILES = [
    R2E_DIR / "R2E_MANIFEST.json",
    R2E_DIR / "R2E_fresh_existing_alignment_audit.csv",
    R2E_DIR / "R2E_same_sample_alignment.csv",
    R2E_DIR / "R2E_six_model_same_sample_metrics.csv",
    R2D_B1_DIR / "R2D_B1_input_provenance.csv",
    R2D_B1_DIR / "R2D_B1_sample_metrics.csv",
    R2D_B1_DIR / "R2D_B1_summary_metrics.csv",
    R2A_V2_DIR / "R2A_V2_dataset_artifact_inventory.csv",
    R2A_V2_DIR / "R2A_V2_fresh_source_content_evidence.csv",
    R2D_A_DIR / "R2D_A_metadata_candidates.csv",
    R2D_A_DIR / "R2D_A_npz_candidate_scores.csv",
    R2D_A_DIR / "R2D_A_selected_npz.csv",
]

OUT = (
    R10_ROOT
    / "R10C0_fresh_wildbox_input_provenance_lock"
)
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "expected_samples": 865,
        "future_length": 20,
    },
    "2s_to_5s": {
        "expected_samples": 637,
        "future_length": 50,
    },
}

LIKELY_REQUIRED_NPZ_KEYS = {
    "history_canonical",
    "future_canonical",
    "history_abs",
    "future_abs",
    "origin",
    "heading_angle",
    "scale",
    "raw_path_length",
}


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


def verify_r10b():
    require(R10B_MANIFEST)

    m = json.loads(
        R10B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R10B_TESTREAL_TRANSFORMER_ENSEMBLE_VALUE_QUANTIFIED_READY_FOR_R10C_FRESH_WILDBOX_TRANSFORMER_REINFERENCE"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R10-B status: {m.get('status')}"
        )

    return m


def read_text_evidence(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    rows = []

    needles = [
        "865",
        "637",
        "WildBox",
        "wildbox",
        "fresh",
        "canonical",
        ".npz",
        "metadata",
        "2s_to_2s",
        "2s_to_5s",
    ]

    for i, line in enumerate(
        text.splitlines(),
        start=1
    ):
        if any(
            needle in line
            for needle in needles
        ):
            rows.append(
                {
                    "source_file": str(path),
                    "line_number": i,
                    "line": line,
                }
            )

    return rows


def scan_known_r2_artifacts():
    rows = []

    for path in CANDIDATE_FILES:
        if not path.exists():
            continue

        row = {
            "path": str(path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "rows": None,
            "columns": "",
        }

        if path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path)
                row["rows"] = int(len(df))
                row["columns"] = "|".join(
                    str(c)
                    for c in df.columns
                )
            except Exception as exc:
                row["columns"] = f"<read error: {exc}>"

        rows.append(row)

    return pd.DataFrame(rows)


def extract_all_path_like_strings():
    """
    Search the known R2 text/CSV/JSON artifacts for strings ending in .npz/.csv
    that may identify the exact fresh WildBox inputs.
    """
    rows = []

    for path in CANDIDATE_FILES:
        if not path.exists():
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue

        # Split broadly; preserve Windows paths with spaces when possible by scanning
        # each line for suffix anchors and collecting surrounding text fragments.
        for line_no, line in enumerate(
            text.splitlines(),
            start=1
        ):
            lower = line.lower()

            if (
                ".npz" not in lower
                and
                ".csv" not in lower
            ):
                continue

            rows.append(
                {
                    "source_file": str(path),
                    "line_number": line_no,
                    "line": line,
                    "mentions_wildbox": "wildbox" in lower,
                    "mentions_fresh": "fresh" in lower,
                    "mentions_865": "865" in line,
                    "mentions_637": "637" in line,
                    "mentions_2s_to_2s": "2s_to_2s" in lower,
                    "mentions_2s_to_5s": "2s_to_5s" in lower,
                }
            )

    return pd.DataFrame(rows)


def candidate_npz_files_from_revision():
    rows = []
    seen = set()

    roots = [
        R2_ROOT,
        MAIN1,
        DOWNLOADS,
    ]

    patterns = [
        "**/*wildbox*.npz",
        "**/*WildBox*.npz",
        "**/*fresh*.npz",
        "**/*external*.npz",
    ]

    for root in roots:
        if not root.exists():
            continue

        for pattern in patterns:
            try:
                iterable = root.glob(pattern)
            except Exception:
                continue

            for p in iterable:
                if not p.is_file():
                    continue

                # Avoid scanning our newly generated revision outputs from R9/R10
                # unless they are under R2, because those could be representation-specific.
                pstr = str(p)

                key = str(p.resolve()).lower()
                if key in seen:
                    continue

                seen.add(key)

                info = {
                    "path": pstr,
                    "name": p.name,
                    "sha256": sha256_file(p),
                    "size_bytes": int(p.stat().st_size),
                    "keys": "",
                    "shape_history_canonical": "",
                    "shape_future_canonical": "",
                    "shape_history_abs": "",
                    "shape_future_abs": "",
                    "samples_inferred": None,
                    "future_length_inferred": None,
                    "usable_canonical_signature": False,
                    "read_error": "",
                }

                try:
                    arr = np.load(
                        p,
                        allow_pickle=False
                    )

                    info["keys"] = "|".join(
                        arr.files
                    )

                    for key_name in [
                        "history_canonical",
                        "future_canonical",
                        "history_abs",
                        "future_abs",
                    ]:
                        if key_name in arr.files:
                            shape = tuple(
                                np.asarray(
                                    arr[key_name]
                                ).shape
                            )
                            info[
                                f"shape_{key_name}"
                            ] = str(shape)

                    if "history_canonical" in arr.files:
                        h = np.asarray(
                            arr["history_canonical"]
                        )
                        if h.ndim >= 1:
                            info["samples_inferred"] = int(
                                h.shape[0]
                            )

                    elif "history_abs" in arr.files:
                        h = np.asarray(
                            arr["history_abs"]
                        )
                        if h.ndim >= 1:
                            info["samples_inferred"] = int(
                                h.shape[0]
                            )

                    if "future_canonical" in arr.files:
                        f = np.asarray(
                            arr["future_canonical"]
                        )
                        if f.ndim >= 2:
                            info["future_length_inferred"] = int(
                                f.shape[1]
                            )

                    elif "future_abs" in arr.files:
                        f = np.asarray(
                            arr["future_abs"]
                        )
                        if f.ndim >= 2:
                            info["future_length_inferred"] = int(
                                f.shape[1]
                            )

                    info["usable_canonical_signature"] = (
                        LIKELY_REQUIRED_NPZ_KEYS
                        .issubset(
                            set(arr.files)
                        )
                    )

                except Exception as exc:
                    info["read_error"] = repr(exc)

                rows.append(info)

    return pd.DataFrame(rows)


def candidate_metadata_files():
    rows = []
    seen = set()

    roots = [
        R2_ROOT,
        MAIN1,
        DOWNLOADS,
    ]

    patterns = [
        "**/*wildbox*metadata*.csv",
        "**/*WildBox*metadata*.csv",
        "**/*fresh*metadata*.csv",
        "**/*external*metadata*.csv",
    ]

    for root in roots:
        if not root.exists():
            continue

        for pattern in patterns:
            try:
                iterable = root.glob(pattern)
            except Exception:
                continue

            for p in iterable:
                if not p.is_file():
                    continue

                key = str(p.resolve()).lower()

                if key in seen:
                    continue

                seen.add(key)

                row = {
                    "path": str(p),
                    "name": p.name,
                    "sha256": sha256_file(p),
                    "size_bytes": int(p.stat().st_size),
                    "rows": None,
                    "columns": "",
                    "read_error": "",
                }

                try:
                    df = pd.read_csv(
                        p
                    )

                    row["rows"] = int(
                        len(df)
                    )

                    row["columns"] = "|".join(
                        str(c)
                        for c in df.columns
                    )

                except Exception as exc:
                    row["read_error"] = repr(
                        exc
                    )

                rows.append(row)

    return pd.DataFrame(rows)


def resolve_candidates(npz_df, meta_df):
    result_rows = []

    for task, cfg in TASKS.items():
        n = cfg["expected_samples"]
        fut = cfg["future_length"]

        npz_candidates = npz_df[
            (
                pd.to_numeric(
                    npz_df[
                        "samples_inferred"
                    ],
                    errors="coerce"
                )
                ==
                n
            )
            &
            (
                pd.to_numeric(
                    npz_df[
                        "future_length_inferred"
                    ],
                    errors="coerce"
                )
                ==
                fut
            )
        ].copy()

        # Prefer true canonical NPZs with all required arrays.
        canonical_candidates = npz_candidates[
            npz_candidates[
                "usable_canonical_signature"
            ]
            ==
            True
        ].copy()

        metadata_candidates = meta_df[
            pd.to_numeric(
                meta_df[
                    "rows"
                ],
                errors="coerce"
            )
            ==
            n
        ].copy()

        canonical_paths = canonical_candidates[
            "path"
        ].tolist()

        metadata_paths = metadata_candidates[
            "path"
        ].tolist()

        result_rows.append(
            {
                "task":
                    task,

                "expected_samples":
                    n,

                "expected_future_length":
                    fut,

                "matching_npz_candidates":
                    int(
                        len(
                            npz_candidates
                        )
                    ),

                "matching_canonical_npz_candidates":
                    int(
                        len(
                            canonical_candidates
                        )
                    ),

                "matching_metadata_candidates":
                    int(
                        len(
                            metadata_candidates
                        )
                    ),

                "canonical_npz_paths":
                    " || ".join(
                        canonical_paths
                    ),

                "metadata_paths":
                    " || ".join(
                        metadata_paths
                    ),
            }
        )

    return pd.DataFrame(
        result_rows
    )


def main():
    print(
        "\nR10-C0 — FRESH WILDBOX MATCHED INPUT PROVENANCE LOCK"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R10-A proved that legacy WildBox_external predictions contain only 93/60 samples "
        "and must not be confused with the later fresh matched WildBox evaluation (865/637). "
        "This audit searches the already-completed R2 fair-comparison artifacts and local "
        "experiment outputs for the exact fresh WildBox canonical NPZ and metadata files, "
        "verifies their sample/future shapes, and records all provenance before any R10 "
        "fresh-WildBox Transformer reinference is allowed.\n"
    )

    verify_r10b()

    known_df = scan_known_r2_artifacts()
    path_evidence_df = extract_all_path_like_strings()
    npz_df = candidate_npz_files_from_revision()
    meta_df = candidate_metadata_files()

    if len(npz_df) == 0:
        raise RuntimeError(
            "No WildBox/fresh/external NPZ candidates were found."
        )

    if len(meta_df) == 0:
        raise RuntimeError(
            "No WildBox/fresh/external metadata CSV candidates were found."
        )

    resolution_df = resolve_candidates(
        npz_df,
        meta_df,
    )

    known_path = (
        OUT
        /
        "R10C0_KNOWN_R2_ARTIFACT_INVENTORY.csv"
    )

    evidence_path = (
        OUT
        /
        "R10C0_R2_PATH_TEXT_EVIDENCE.csv"
    )

    npz_path = (
        OUT
        /
        "R10C0_WILDBOX_NPZ_CANDIDATE_INVENTORY.csv"
    )

    meta_path = (
        OUT
        /
        "R10C0_WILDBOX_METADATA_CANDIDATE_INVENTORY.csv"
    )

    resolution_path = (
        OUT
        /
        "R10C0_TASK_INPUT_RESOLUTION.csv"
    )

    known_df.to_csv(
        known_path,
        index=False
    )

    path_evidence_df.to_csv(
        evidence_path,
        index=False
    )

    npz_df.to_csv(
        npz_path,
        index=False
    )

    meta_df.to_csv(
        meta_path,
        index=False
    )

    resolution_df.to_csv(
        resolution_path,
        index=False
    )

    uniquely_resolved = True

    for _, row in resolution_df.iterrows():
        if int(
            row[
                "matching_canonical_npz_candidates"
            ]
        ) != 1:
            uniquely_resolved = False

        if int(
            row[
                "matching_metadata_candidates"
            ]
        ) != 1:
            uniquely_resolved = False

    if uniquely_resolved:
        status = (
            "R10C0_FRESH_WILDBOX_865_637_CANONICAL_INPUTS_UNIQUELY_LOCKED_READY_FOR_R10C1_FROZEN_TRANSFORMER_REINFERENCE"
        )
    else:
        status = (
            "R10C0_FRESH_WILDBOX_INPUTS_NOT_UNIQUELY_RESOLVED_REVIEW_CANDIDATES_BEFORE_INFERENCE"
        )

    manifest = {
        "step":
            "R10C0_FRESH_WILDBOX_MATCHED_INPUT_PROVENANCE_LOCK",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "target_sample_counts":
            {
                "2s_to_2s":
                    865,

                "2s_to_5s":
                    637,
            },

        "legacy_wildbox_external_counts_not_allowed":
            {
                "2s_to_2s":
                    93,

                "2s_to_5s":
                    60,
            },

        "claim_guardrail":
            (
                "No fresh-WildBox ensemble claim is allowed until exact canonical NPZ and metadata "
                "for 865/637 matched samples are uniquely resolved and hash-locked."
            ),

        "outputs": {
            "known_r2_artifacts":
                str(
                    known_path
                ),

            "path_text_evidence":
                str(
                    evidence_path
                ),

            "npz_candidates":
                str(
                    npz_path
                ),

            "metadata_candidates":
                str(
                    meta_path
                ),

            "task_resolution":
                str(
                    resolution_path
                ),
        },
    }

    (
        OUT
        /
        "R10C0_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "TASK INPUT RESOLUTION:"
    )

    print(
        resolution_df.to_string(
            index=False
        )
    )

    print(
        "\nTOP MATCHING NPZ CANDIDATES:"
    )

    top_npz = npz_df[
        pd.to_numeric(
            npz_df[
                "samples_inferred"
            ],
            errors="coerce"
        ).isin(
            [
                865,
                637,
            ]
        )
    ].copy()

    if len(
        top_npz
    ):
        print(
            top_npz[
                [
                    "path",
                    "samples_inferred",
                    "future_length_inferred",
                    "usable_canonical_signature",
                    "sha256",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No NPZ candidate with 865/637 samples."
        )

    print(
        "\nTOP MATCHING METADATA CANDIDATES:"
    )

    top_meta = meta_df[
        pd.to_numeric(
            meta_df[
                "rows"
            ],
            errors="coerce"
        ).isin(
            [
                865,
                637,
            ]
        )
    ].copy()

    if len(
        top_meta
    ):
        print(
            top_meta[
                [
                    "path",
                    "rows",
                    "columns",
                    "sha256",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No metadata candidate with 865/637 rows."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR10-C0 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
