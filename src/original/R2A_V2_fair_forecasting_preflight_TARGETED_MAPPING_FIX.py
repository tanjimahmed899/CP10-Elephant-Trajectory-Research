from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

SEARCH_ROOTS = [
    MAIN_RESULT_2,
    MAIN_RESULT,
    DOWNLOADS,
]

R1_RECEIPT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R1_protocol_lock_v3"
    / "R1_V3_REVISION_PROTOCOL_RECEIPT.json"
)

EXPECTED_R1_PROTOCOL_SHA256 = (
    "4af15bd17f57d4c9932f482dcc047f3b344be4b810d0a4c7b48942f8d17406a8"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2A_preflight_v2"
)

OUT.mkdir(parents=True, exist_ok=True)

SCRIPT_TARGETS = {
    "constant_velocity":
        [
            "run_constant_velocity_baseline.py",
        ],

    "kalman":
        [
            "run_kalman_baseline.py",
        ],

    "lstm_gru":
        [
            "train_lstm_gru.py",
        ],

    "transformer_train":
        [
            "train_transformer.py",
        ],

    "testreal_frozen_evaluator":
        [
            "evaluate_birdsai_testreal_frozen_ensemble.py",
        ],

    "fresh_holdout_evaluator":
        [
            "evaluate_fresh_wildbox_holdout.py",
        ],

    "fresh_holdout_selector":
        [
            "select_fresh_wildbox_holdout.py",
        ],

    "final_neural_dataset_builder":
        [
            "build_final_neural_dataset.py",
            "build_final_neural_trajectory_dataset.py",
        ],
}

MODEL_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
}

DATA_EXTENSIONS = {
    ".npz",
    ".npy",
    ".csv",
    ".json",
    ".txt",
    ".md",
}

FRESH_SOURCES = [
    "DJI_20240119152335_0005_V",
    "DJI_20240626142859_0001_V",
    "DJI_20260227084135_0002_V",
]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)


def verify_r1():
    require(R1_RECEIPT)

    receipt = json.loads(
        R1_RECEIPT.read_text(
            encoding="utf-8"
        )
    )

    if receipt.get("status") != "R1_REVISION_PROTOCOL_LOCKED_READY_FOR_R2":
        raise RuntimeError(
            f"Unexpected R1 status: {receipt.get('status')}"
        )

    if receipt.get("protocol_sha256") != EXPECTED_R1_PROTOCOL_SHA256:
        raise RuntimeError(
            "R1 protocol hash mismatch.\n"
            f"Expected: {EXPECTED_R1_PROTOCOL_SHA256}\n"
            f"Found:    {receipt.get('protocol_sha256')}"
        )

    return receipt


def root_rank(path):
    low = str(path).lower()

    if str(MAIN_RESULT_2).lower() in low:
        return 0

    if str(MAIN_RESULT).lower() in low:
        return 1

    if str(DOWNLOADS).lower() in low:
        return 2

    return 9


def find_exact_names(names):
    hits = []
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for name in names:
            for p in root.rglob(name):
                if not p.is_file():
                    continue

                rp = str(p.resolve()).lower()

                if rp in seen:
                    continue

                seen.add(rp)
                hits.append(p)

    return sorted(
        hits,
        key=lambda p: (
            root_rank(p),
            str(p).lower(),
        )
    )


def discover_scripts():
    rows = []

    for role, names in SCRIPT_TARGETS.items():
        hits = find_exact_names(names)

        selected = hits[0] if hits else None

        rows.append({
            "role":
                role,

            "accepted_names":
                " | ".join(names),

            "selected_name":
                (
                    selected.name
                    if selected is not None
                    else None
                ),

            "selected_path":
                (
                    str(selected)
                    if selected is not None
                    else None
                ),

            "selected_sha256":
                (
                    sha256_file(selected)
                    if selected is not None
                    else None
                ),

            "copies_found":
                len(hits),
        })

    return pd.DataFrame(rows)


def classify_model(path):
    """
    Use filename first, not parent folder, to avoid classifying LSTM checkpoints
    inside a folder named 'lstm_gru_results' as GRU.
    """
    name = path.name.lower()
    full = str(path).lower()

    # Filename has priority.
    if "lstm" in name:
        return "LSTM"

    if "gru" in name:
        return "GRU"

    if "transformer" in name:
        return "Transformer"

    if "kalman" in name:
        return "Kalman"

    # Fallback only when filename itself is uninformative.
    if "lstm" in full and "gru" not in full:
        return "LSTM"

    if "gru" in full and "lstm" not in full:
        return "GRU"

    if "transformer" in full:
        return "Transformer"

    if "kalman" in full:
        return "Kalman"

    return None


def classify_dataset(path):
    full = str(path).lower()

    if (
        "fresh" in full
        and
        "wildbox" in full
    ):
        return "Fresh_WildBox"

    if (
        "testreal" in full
        or
        "birdsai_testreal" in full
    ):
        return "BIRDSAI_TestReal"

    return None


def classify_task(path):
    full = str(path).lower()

    patterns_22 = [
        "2s_to_2s",
        "2s-to-2s",
        "2to2",
        "2s2s",
    ]

    patterns_25 = [
        "2s_to_5s",
        "2s-to-5s",
        "2to5",
        "2s5s",
    ]

    if any(token in full for token in patterns_22):
        return "2s_to_2s"

    if any(token in full for token in patterns_25):
        return "2s_to_5s"

    return None


def discover_model_files():
    rows = []
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if (
                not p.is_file()
                or
                p.suffix.lower()
                not in
                MODEL_EXTENSIONS
            ):
                continue

            rp = str(p.resolve()).lower()

            if rp in seen:
                continue

            seen.add(rp)

            model = classify_model(p)

            if model is None:
                continue

            try:
                size_bytes = int(p.stat().st_size)
            except Exception:
                size_bytes = None

            rows.append({
                "root_rank":
                    root_rank(p),

                "path":
                    str(p),

                "name":
                    p.name,

                "model":
                    model,

                "dataset_hint":
                    classify_dataset(p),

                "task_hint":
                    classify_task(p),

                "size_bytes":
                    size_bytes,

                "sha256":
                    sha256_file(p),
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No neural model checkpoint candidates found."
        )

    return df.sort_values(
        [
            "model",
            "task_hint",
            "root_rank",
            "name",
        ],
        na_position="last",
    ).reset_index(drop=True)


def discover_dataset_artifacts():
    rows = []
    seen = set()

    interesting_tokens = [
        "testreal",
        "fresh",
        "wildbox",
        "trajectory",
        "neural_dataset",
        "canonical",
        "ensemble",
    ]

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if (
                not p.is_file()
                or
                p.suffix.lower()
                not in
                DATA_EXTENSIONS
            ):
                continue

            full = str(p).lower()

            if not any(
                token in full
                for token in interesting_tokens
            ):
                continue

            rp = str(p.resolve()).lower()

            if rp in seen:
                continue

            seen.add(rp)

            try:
                size_bytes = int(p.stat().st_size)
            except Exception:
                size_bytes = None

            rows.append({
                "root_rank":
                    root_rank(p),

                "path":
                    str(p),

                "name":
                    p.name,

                "suffix":
                    p.suffix.lower(),

                "dataset_hint":
                    classify_dataset(p),

                "task_hint":
                    classify_task(p),

                "size_bytes":
                    size_bytes,
            })

    return pd.DataFrame(rows).sort_values(
        [
            "dataset_hint",
            "task_hint",
            "root_rank",
            "name",
        ],
        na_position="last",
    ).reset_index(drop=True)


def content_search_files():
    """
    Search source IDs inside text/config/script contents, not only filenames.
    This proves the preregistered holdout sources even when source IDs are not
    present in the artifact path.
    """
    rows = []

    candidate_suffixes = {
        ".py",
        ".json",
        ".csv",
        ".txt",
        ".md",
    }

    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if (
                not p.is_file()
                or
                p.suffix.lower()
                not in
                candidate_suffixes
            ):
                continue

            # Avoid enormous irrelevant text/config inventories.
            low_path = str(p).lower()

            if not any(
                token
                in
                low_path
                for token in [
                    "fresh",
                    "wildbox",
                    "holdout",
                    "selection",
                    "ensemble",
                ]
            ):
                continue

            rp = str(p.resolve()).lower()

            if rp in seen:
                continue

            seen.add(rp)

            try:
                raw = p.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            for source in FRESH_SOURCES:
                if source.lower() not in raw.lower():
                    continue

                # Collect short line evidence.
                evidence_lines = []

                for line_no, line in enumerate(
                    raw.splitlines(),
                    start=1,
                ):
                    if source.lower() in line.lower():
                        evidence_lines.append(
                            f"L{line_no}: {line.strip()[:400]}"
                        )

                rows.append({
                    "source":
                        source,

                    "evidence_file":
                        str(p),

                    "evidence_sha256":
                        sha256_file(p),

                    "evidence_lines":
                        " || ".join(
                            evidence_lines[:10]
                        ),

                    "root_rank":
                        root_rank(p),
                })

    return pd.DataFrame(rows)


def fresh_source_proof(content_df):
    rows = []

    for source in FRESH_SOURCES:
        if len(content_df) == 0:
            subset = pd.DataFrame()
        else:
            subset = content_df[
                content_df[
                    "source"
                ]
                ==
                source
            ].copy()

        rows.append({
            "source":
                source,

            "content_evidence_files":
                int(
                    len(
                        subset
                    )
                ),

            "proved":
                bool(
                    len(
                        subset
                    )
                    >
                    0
                ),

            "best_evidence":
                (
                    subset.sort_values(
                        [
                            "root_rank",
                            "evidence_file",
                        ]
                    ).iloc[
                        0
                    ][
                        "evidence_file"
                    ]
                    if len(
                        subset
                    )
                    >
                    0
                    else None
                ),

            "evidence_preview":
                (
                    subset.sort_values(
                        [
                            "root_rank",
                            "evidence_file",
                        ]
                    ).iloc[
                        0
                    ][
                        "evidence_lines"
                    ]
                    if len(
                        subset
                    )
                    >
                    0
                    else None
                ),
        })

    return pd.DataFrame(rows)


def selected_model_counts(model_df):
    rows = []

    for model in [
        "GRU",
        "LSTM",
        "Transformer",
    ]:
        subset = model_df[
            model_df[
                "model"
            ]
            ==
            model
        ].copy()

        rows.append({
            "model":
                model,

            "checkpoint_candidates":
                int(
                    len(
                        subset
                    )
                ),

            "example_files":
                " | ".join(
                    subset[
                        "path"
                    ].head(
                        12
                    ).astype(
                        str
                    ).tolist()
                ),
        })

    return pd.DataFrame(rows)


def readiness(
    scripts_df,
    model_counts,
    dataset_df,
    fresh_proof_df,
):
    checks = []

    required_roles = [
        "constant_velocity",
        "kalman",
        "lstm_gru",
        "testreal_frozen_evaluator",
        "fresh_holdout_evaluator",
        "fresh_holdout_selector",
        "final_neural_dataset_builder",
    ]

    for role in required_roles:
        row = scripts_df[
            scripts_df[
                "role"
            ]
            ==
            role
        ]

        found = (
            len(
                row
            )
            >
            0
            and
            pd.notna(
                row.iloc[
                    0
                ][
                    "selected_path"
                ]
            )
        )

        checks.append({
            "check":
                f"script::{role}",

            "pass":
                bool(found),

            "note":
                (
                    str(
                        row.iloc[
                            0
                        ][
                            "selected_path"
                        ]
                    )
                    if found
                    else
                    "not found"
                ),
        })

    for model in [
        "GRU",
        "LSTM",
        "Transformer",
    ]:
        row = model_counts[
            model_counts[
                "model"
            ]
            ==
            model
        ]

        count = (
            int(
                row.iloc[
                    0
                ][
                    "checkpoint_candidates"
                ]
            )
            if len(
                row
            )
            >
            0
            else
            0
        )

        checks.append({
            "check":
                f"model::{model}",

            "pass":
                bool(
                    count
                    >
                    0
                ),

            "note":
                f"{count} checkpoint candidates",
        })

    for dataset in [
        "BIRDSAI_TestReal",
        "Fresh_WildBox",
    ]:
        count = int(
            (
                dataset_df[
                    "dataset_hint"
                ]
                ==
                dataset
            ).sum()
        )

        checks.append({
            "check":
                f"dataset::{dataset}",

            "pass":
                bool(
                    count
                    >
                    0
                ),

            "note":
                f"{count} dataset/result artifacts",
        })

    for _, row in fresh_proof_df.iterrows():
        checks.append({
            "check":
                f"fresh_source::{row['source']}",

            "pass":
                bool(
                    row[
                        "proved"
                    ]
                ),

            "note":
                (
                    str(
                        row[
                            "best_evidence"
                        ]
                    )
                    if bool(
                        row[
                            "proved"
                        ]
                    )
                    else
                    "source ID not found in relevant file contents"
                ),
        })

    return pd.DataFrame(checks)


def main():
    print(
        "\nR2A-V2 — FAIR FORECASTING PREFLIGHT / TARGETED SOURCE-MAPPING PATCH"
    )

    print(
        "NO TestReal or Fresh benchmark metric is generated. "
        "This only fixes exact source/model/dataset provenance discovery.\n"
    )

    receipt = verify_r1()

    print(
        "VERIFIED R1 PROTOCOL SHA256:"
    )
    print(
        receipt[
            "protocol_sha256"
        ]
    )

    scripts_df = discover_scripts()
    model_df = discover_model_files()
    model_counts = selected_model_counts(
        model_df
    )
    dataset_df = discover_dataset_artifacts()
    fresh_content_df = content_search_files()
    fresh_proof_df = fresh_source_proof(
        fresh_content_df
    )

    readiness_df = readiness(
        scripts_df,
        model_counts,
        dataset_df,
        fresh_proof_df,
    )

    scripts_path = (
        OUT
        /
        "R2A_V2_source_scripts.csv"
    )

    model_path = (
        OUT
        /
        "R2A_V2_model_checkpoint_inventory.csv"
    )

    model_counts_path = (
        OUT
        /
        "R2A_V2_model_checkpoint_counts.csv"
    )

    dataset_path = (
        OUT
        /
        "R2A_V2_dataset_artifact_inventory.csv"
    )

    fresh_content_path = (
        OUT
        /
        "R2A_V2_fresh_source_content_evidence.csv"
    )

    fresh_proof_path = (
        OUT
        /
        "R2A_V2_fresh_source_proof.csv"
    )

    readiness_path = (
        OUT
        /
        "R2A_V2_readiness_checks.csv"
    )

    scripts_df.to_csv(
        scripts_path,
        index=False,
    )

    model_df.to_csv(
        model_path,
        index=False,
    )

    model_counts.to_csv(
        model_counts_path,
        index=False,
    )

    dataset_df.to_csv(
        dataset_path,
        index=False,
    )

    fresh_content_df.to_csv(
        fresh_content_path,
        index=False,
    )

    fresh_proof_df.to_csv(
        fresh_proof_path,
        index=False,
    )

    readiness_df.to_csv(
        readiness_path,
        index=False,
    )

    all_pass = bool(
        readiness_df[
            "pass"
        ].all()
    )

    status = (
        "R2A_V2_PREFLIGHT_COMPLETE_READY_FOR_R2B_FAIR_BENCHMARK"
        if all_pass
        else
        "R2A_V2_PREFLIGHT_COMPLETE_ADDITIONAL_TARGETED_MAPPING_REQUIRED"
    )

    summary = {
        "step":
            "R2A_V2_FAIR_FORECASTING_PREFLIGHT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "r1_protocol_sha256":
            receipt[
                "protocol_sha256"
            ],

        "new_test_metrics_generated":
            False,

        "fixes": [
            "accept build_final_neural_dataset.py as the exact dataset builder",
            "classify LSTM/GRU checkpoints from filename before parent folder",
            "prove fresh holdout source IDs from file contents rather than path names",
        ],

        "outputs": {
            "scripts":
                str(scripts_path),

            "models":
                str(model_path),

            "model_counts":
                str(model_counts_path),

            "datasets":
                str(dataset_path),

            "fresh_content_evidence":
                str(fresh_content_path),

            "fresh_source_proof":
                str(fresh_proof_path),

            "readiness":
                str(readiness_path),
        },
    }

    summary_path = (
        OUT
        /
        "R2A_V2_PREFLIGHT_SUMMARY.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSELECTED SOURCE SCRIPTS:"
    )
    print(
        scripts_df.to_string(
            index=False
        )
    )

    print(
        "\nMODEL CHECKPOINT COUNTS:"
    )
    print(
        model_counts.to_string(
            index=False
        )
    )

    print(
        "\nFRESH HOLDOUT SOURCE PROOF:"
    )
    print(
        fresh_proof_df.to_string(
            index=False
        )
    )

    print(
        "\nREADINESS CHECKS:"
    )
    print(
        readiness_df.to_string(
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
        "\nR2A-V2 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
