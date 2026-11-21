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
    / "R2A_preflight"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET_SCRIPT_NAMES = [
    "run_constant_velocity_baseline.py",
    "run_kalman_baseline.py",
    "train_lstm_gru.py",
    "train_transformer.py",
    "evaluate_birdsai_testreal_frozen_ensemble.py",
    "evaluate_fresh_wildbox_holdout.py",
    "select_fresh_wildbox_holdout.py",
    "build_final_neural_trajectory_dataset.py",
]

KEYWORDS = [
    "testreal",
    "fresh",
    "wildbox",
    "transformer",
    "lstm",
    "gru",
    "kalman",
    "constant_velocity",
    "stationary",
    "trajectory",
    "dataset",
]

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
}


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def require(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)


def verify_r1():
    require(
        R1_RECEIPT
    )

    receipt = json.loads(
        R1_RECEIPT.read_text(
            encoding="utf-8"
        )
    )

    if (
        receipt.get(
            "status"
        )
        !=
        "R1_REVISION_PROTOCOL_LOCKED_READY_FOR_R2"
    ):
        raise RuntimeError(
            f"Unexpected R1 status: {receipt.get('status')}"
        )

    actual_protocol_hash = receipt.get(
        "protocol_sha256"
    )

    if actual_protocol_hash != EXPECTED_R1_PROTOCOL_SHA256:
        raise RuntimeError(
            "R1 protocol hash mismatch.\n"
            f"Expected: {EXPECTED_R1_PROTOCOL_SHA256}\n"
            f"Receipt:  {actual_protocol_hash}"
        )

    return receipt


def root_rank(path):
    s = str(path).lower()

    if str(MAIN_RESULT_2).lower() in s:
        return 0

    if str(MAIN_RESULT).lower() in s:
        return 1

    if str(DOWNLOADS).lower() in s:
        return 2

    return 9


def discover_exact_name(filename):
    hits = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob(
            filename
        ):
            if p.is_file():
                hits.append(
                    p
                )

    unique = []

    seen = set()

    for p in hits:
        rp = str(
            p.resolve()
        ).lower()

        if rp in seen:
            continue

        seen.add(
            rp
        )

        unique.append(
            p
        )

    return sorted(
        unique,
        key=lambda p: (
            root_rank(
                p
            ),
            str(
                p
            ).lower(),
        )
    )


def safe_ast(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(
            text
        )

    except Exception:
        return (
            text,
            None,
        )

    return (
        text,
        tree,
    )


def collect_ast_inventory(path):
    text, tree = safe_ast(
        path
    )

    result = {
        "script":
            str(
                path
            ),

        "sha256":
            sha256_file(
                path
            ),

        "functions":
            [],

        "classes":
            [],

        "interesting_literals":
            [],
    }

    if tree is None:
        result[
            "parse_error"
        ] = True

        return result

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            try:
                sig = ast.unparse(
                    node.args
                )

            except Exception:
                sig = "<unparse_failed>"

            result[
                "functions"
            ].append({
                "name":
                    node.name,

                "line":
                    int(
                        node.lineno
                    ),

                "args":
                    sig,
            })

        elif isinstance(
            node,
            ast.ClassDef
        ):
            result[
                "classes"
            ].append({
                "name":
                    node.name,

                "line":
                    int(
                        node.lineno
                    ),
            })

    for node in ast.walk(
        tree
    ):
        if not (
            isinstance(
                node,
                ast.Constant
            )
            and
            isinstance(
                node.value,
                str
            )
        ):
            continue

        value = node.value

        low = value.lower()

        if (
            any(
                ext in low
                for ext in [
                    ".npz",
                    ".npy",
                    ".csv",
                    ".json",
                    ".pt",
                    ".pth",
                    ".ckpt",
                ]
            )
            or
            any(
                keyword
                in
                low
                for keyword in KEYWORDS
            )
        ):
            result[
                "interesting_literals"
            ].append({
                "line":
                    int(
                        getattr(
                            node,
                            "lineno",
                            -1
                        )
                    ),

                "value":
                    value,
            })

    return result


def discover_scripts():
    rows = []

    audit = []

    for filename in TARGET_SCRIPT_NAMES:
        hits = discover_exact_name(
            filename
        )

        if not hits:
            rows.append({
                "target_name":
                    filename,

                "selected_path":
                    None,

                "selected_sha256":
                    None,

                "copies_found":
                    0,
            })

            continue

        selected = hits[
            0
        ]

        rows.append({
            "target_name":
                filename,

            "selected_path":
                str(
                    selected
                ),

            "selected_sha256":
                sha256_file(
                    selected
                ),

            "copies_found":
                len(
                    hits
                ),
        })

        audit.append(
            collect_ast_inventory(
                selected
            )
        )

    return (
        pd.DataFrame(
            rows
        ),
        audit,
    )


def classify_candidate(path):
    name = path.name.lower()
    full = str(path).lower()

    dataset = None
    task = None
    role = None
    model = None

    if "testreal" in full or "birdsai_testreal" in full:
        dataset = "BIRDSAI_TestReal"

    if (
        "fresh" in full
        and
        "wildbox" in full
    ):
        dataset = "Fresh_WildBox"

    if any(
        token
        in
        full
        for token in [
            "2s_to_2s",
            "2s-to-2s",
            "2to2",
            "2s2s",
        ]
    ):
        task = "2s_to_2s"

    if any(
        token
        in
        full
        for token in [
            "2s_to_5s",
            "2s-to-5s",
            "2to5",
            "2s5s",
        ]
    ):
        task = "2s_to_5s"

    if path.suffix.lower() in MODEL_EXTENSIONS:
        role = "model_checkpoint"

    elif path.suffix.lower() == ".npz":
        role = "npz_dataset_or_predictions"

    elif path.suffix.lower() == ".csv":
        role = "metadata_or_results"

    elif path.suffix.lower() == ".json":
        role = "manifest_or_config"

    if "gru" in full:
        model = "GRU"

    elif "lstm" in full:
        model = "LSTM"

    elif "transformer" in full:
        model = "Transformer"

    elif "kalman" in full:
        model = "Kalman"

    return {
        "dataset":
            dataset,

        "task":
            task,

        "role":
            role,

        "model":
            model,
    }


def discover_data_and_models():
    rows = []

    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob(
            "*"
        ):
            if not p.is_file():
                continue

            suffix = p.suffix.lower()

            if (
                suffix
                not in
                MODEL_EXTENSIONS
                |
                DATA_EXTENSIONS
            ):
                continue

            full = str(
                p
            ).lower()

            # Keep only likely R2-relevant artifacts.
            if not any(
                keyword
                in
                full
                for keyword in KEYWORDS
            ):
                continue

            rp = str(
                p.resolve()
            ).lower()

            if rp in seen:
                continue

            seen.add(
                rp
            )

            meta = classify_candidate(
                p
            )

            try:
                size_bytes = int(
                    p.stat().st_size
                )

            except Exception:
                size_bytes = None

            # Hash model files and reasonably small metadata/config files.
            # Skip hashing huge dataset files during preflight.
            actual_hash = None

            if (
                suffix
                in
                MODEL_EXTENSIONS
                or
                (
                    size_bytes
                    is not None
                    and
                    size_bytes
                    <=
                    50
                    *
                    1024
                    *
                    1024
                )
            ):
                try:
                    actual_hash = sha256_file(
                        p
                    )
                except Exception:
                    actual_hash = None

            rows.append({
                "root_rank":
                    root_rank(
                        p
                    ),

                "path":
                    str(
                        p
                    ),

                "name":
                    p.name,

                "suffix":
                    suffix,

                "size_bytes":
                    size_bytes,

                "sha256_if_computed":
                    actual_hash,

                **meta,
            })

    df = pd.DataFrame(
        rows
    )

    if len(
        df
    ) == 0:
        raise RuntimeError(
            "No R2-relevant data/model candidates were discovered."
        )

    return df.sort_values(
        [
            "root_rank",
            "dataset",
            "task",
            "role",
            "name",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )


def find_expected_fresh_sources(candidate_df):
    locked_sources = [
        "DJI_20240119152335_0005_V",
        "DJI_20240626142859_0001_V",
        "DJI_20260227084135_0002_V",
    ]

    rows = []

    for source in locked_sources:
        matches = candidate_df[
            candidate_df[
                "path"
            ].str.contains(
                source,
                case=False,
                regex=False,
            )
        ].copy()

        rows.append({
            "source":
                source,

            "artifact_matches":
                int(
                    len(
                        matches
                    )
                ),

            "example_paths":
                " | ".join(
                    matches[
                        "path"
                    ].head(
                        5
                    ).astype(
                        str
                    ).tolist()
                ),
        })

    return pd.DataFrame(
        rows
    )


def readiness_summary(
    scripts_df,
    candidate_df,
    fresh_df,
):
    checks = []

    required_scripts = [
        "run_constant_velocity_baseline.py",
        "run_kalman_baseline.py",
        "train_lstm_gru.py",
        "evaluate_birdsai_testreal_frozen_ensemble.py",
        "build_final_neural_trajectory_dataset.py",
    ]

    for name in required_scripts:
        row = scripts_df[
            scripts_df[
                "target_name"
            ]
            ==
            name
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
                f"script::{name}",

            "pass":
                bool(
                    found
                ),

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

    # Need both frozen evaluation datasets represented somewhere.
    for dataset in [
        "BIRDSAI_TestReal",
        "Fresh_WildBox",
    ]:
        count = int(
            (
                candidate_df[
                    "dataset"
                ]
                ==
                dataset
            ).sum()
        )

        checks.append({
            "check":
                f"dataset_artifacts::{dataset}",

            "pass":
                bool(
                    count
                    >
                    0
                ),

            "note":
                f"{count} candidate artifacts",
        })

    # Need frozen models visible.
    for model in [
        "GRU",
        "LSTM",
        "Transformer",
    ]:
        count = int(
            (
                candidate_df[
                    "model"
                ]
                ==
                model
            ).sum()
        )

        checks.append({
            "check":
                f"model_artifacts::{model}",

            "pass":
                bool(
                    count
                    >
                    0
                ),

            "note":
                f"{count} candidate artifacts",
        })

    for _, row in fresh_df.iterrows():
        checks.append({
            "check":
                f"fresh_source::{row['source']}",

            "pass":
                bool(
                    int(
                        row[
                            "artifact_matches"
                        ]
                    )
                    >
                    0
                ),

            "note":
                f"{int(row['artifact_matches'])} matching artifacts",
        })

    return pd.DataFrame(
        checks
    )


def main():
    print(
        "\nR2A — FAIR FORECASTING BENCHMARK PREFLIGHT + EXACT SOURCE RECOVERY"
    )

    print(
        "NO new TestReal/Fresh metric is generated. "
        "This step only recovers exact frozen datasets, scripts, models, and provenance.\n"
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

    print(
        "\nSEARCH ROOTS:"
    )

    for root in SEARCH_ROOTS:
        print(
            "-",
            root,
            "| exists =",
            root.exists(),
        )

    scripts_df, script_audit = discover_scripts()

    candidates_df = discover_data_and_models()

    fresh_df = find_expected_fresh_sources(
        candidates_df
    )

    readiness_df = readiness_summary(
        scripts_df,
        candidates_df,
        fresh_df,
    )

    scripts_path = (
        OUT
        / "R2A_selected_source_scripts.csv"
    )

    candidates_path = (
        OUT
        / "R2A_data_model_candidate_inventory.csv"
    )

    fresh_path = (
        OUT
        / "R2A_fresh_holdout_source_presence.csv"
    )

    readiness_path = (
        OUT
        / "R2A_readiness_checks.csv"
    )

    audit_path = (
        OUT
        / "R2A_source_ast_inventory.json"
    )

    scripts_df.to_csv(
        scripts_path,
        index=False,
    )

    candidates_df.to_csv(
        candidates_path,
        index=False,
    )

    fresh_df.to_csv(
        fresh_path,
        index=False,
    )

    readiness_df.to_csv(
        readiness_path,
        index=False,
    )

    audit_path.write_text(
        json.dumps(
            script_audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    all_pass = bool(
        readiness_df[
            "pass"
        ].all()
    )

    status = (
        "R2A_PREFLIGHT_COMPLETE_READY_TO_BUILD_FAIR_BENCHMARK"
        if all_pass
        else
        "R2A_PREFLIGHT_COMPLETE_TARGETED_SOURCE_MAPPING_PATCH_REQUIRED"
    )

    summary = {
        "step":
            "R2A_FAIR_FORECASTING_BENCHMARK_PREFLIGHT",

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

        "scripts_csv":
            str(
                scripts_path
            ),

        "candidate_inventory_csv":
            str(
                candidates_path
            ),

        "fresh_source_presence_csv":
            str(
                fresh_path
            ),

        "readiness_csv":
            str(
                readiness_path
            ),

        "source_ast_inventory_json":
            str(
                audit_path
            ),
    }

    summary_path = (
        OUT
        / "R2A_PREFLIGHT_SUMMARY.json"
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
        "\nFRESH HOLDOUT SOURCE PRESENCE:"
    )
    print(
        fresh_df.to_string(
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
        "\nR2A COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
