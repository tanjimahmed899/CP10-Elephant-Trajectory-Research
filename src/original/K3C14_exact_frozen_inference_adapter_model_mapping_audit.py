from pathlib import Path
import ast
import hashlib
import json
import re
import textwrap

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

K3C13_MANIFEST = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_canonical_lock_v1"
    / "STEP6D_K3C13_CANONICAL_DATASET_LOCK.json"
)

EXPECTED_K3C13_SHA256 = (
    "71d83c774233538342f4b410a789750943e61a8513fe983249c8f8f5a2aee605"
)

MODEL_LOCK_DIR = (
    BASE
    / "k3_frozen_ensemble"
    / "model_lock"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "frozen_inference_adapter_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


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
    if not Path(path).exists():
        raise FileNotFoundError(path)


def find_named_python(filename):
    candidates = []

    for root in [
        MAIN_RESULT,
        DOWNLOADS,
    ]:
        if not root.exists():
            continue

        for p in root.rglob(filename):
            if p.is_file():
                score = 0
                low = str(p).lower()

                if str(MAIN_RESULT).lower() in low:
                    score += 1000

                if "copy" in low:
                    score -= 100

                candidates.append(
                    (
                        score,
                        p,
                    )
                )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find {filename} under Main Result or Downloads."
        )

    candidates = sorted(
        candidates,
        key=lambda x: (
            -x[
                0
            ],
            str(
                x[
                    1
                ]
            ).lower(),
        )
    )

    return (
        candidates[
            0
        ][
            1
        ],
        [
            p
            for _,
            p
            in candidates
        ],
    )


def find_k3a2_script():
    patterns = [
        "step6d_k3a2_v4_first_locked_bambi_inference.py",
        "*k3a2*first*locked*bambi*inference*.py",
        "*k3a2*inference*.py",
    ]

    hits = []

    for pattern in patterns:
        for p in DOWNLOADS.rglob(
            pattern
        ):
            if p.is_file():
                hits.append(
                    p
                )

        if hits:
            break

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

    if not unique:
        return (
            None,
            []
        )

    unique = sorted(
        unique,
        key=lambda p:
            str(
                p
            ).lower()
    )

    preferred = [
        p
        for p in unique
        if p.name.lower()
        ==
        "step6d_k3a2_v4_first_locked_bambi_inference.py"
    ]

    selected = (
        preferred[
            0
        ]
        if preferred
        else unique[
            0
        ]
    )

    return (
        selected,
        unique
    )


def extract_function_source(
    path,
    function_name,
):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text
    )

    nodes = [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef
        )
        and node.name
        ==
        function_name
    ]

    if len(
        nodes
    ) != 1:
        raise RuntimeError(
            f"{path}: expected exactly one def {function_name}, found {len(nodes)}"
        )

    node = nodes[
        0
    ]

    source = ast.get_source_segment(
        text,
        node
    )

    if not source:
        raise RuntimeError(
            f"Could not extract {function_name} source."
        )

    args = []

    positional = (
        list(
            node.args.posonlyargs
        )
        +
        list(
            node.args.args
        )
    )

    defaults = list(
        node.args.defaults
    )

    default_offset = (
        len(
            positional
        )
        -
        len(
            defaults
        )
    )

    for idx, arg in enumerate(
        positional
    ):
        default_repr = None

        if idx >= default_offset:
            default_node = defaults[
                idx
                -
                default_offset
            ]

            try:
                default_repr = ast.unparse(
                    default_node
                )
            except Exception:
                default_repr = "<unparse_failed>"

        args.append({
            "name":
                arg.arg,

            "default":
                default_repr,
        })

    return (
        text,
        tree,
        node,
        source,
        args,
    )


def collect_calls_to_function(
    path,
    function_name,
):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text
    )

    rows = []

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call
        ):
            continue

        called = None

        if isinstance(
            node.func,
            ast.Name
        ):
            called = node.func.id

        elif isinstance(
            node.func,
            ast.Attribute
        ):
            called = node.func.attr

        if called != function_name:
            continue

        try:
            expression = ast.get_source_segment(
                text,
                node
            )

        except Exception:
            expression = None

        rows.append({
            "line":
                int(
                    getattr(
                        node,
                        "lineno",
                        -1
                    )
                ),

            "expression":
                expression,
        })

    return rows


def extract_string_literals(
    path,
    extensions=(
        ".pt",
        ".pth",
        ".ckpt",
    ),
):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text
    )

    rows = []

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Constant
        ):
            continue

        if not isinstance(
            node.value,
            str
        ):
            continue

        value = node.value

        if any(
            ext.lower()
            in
            value.lower()
            for ext in extensions
        ):
            rows.append({
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

    return rows


def read_lock_files():
    require(
        MODEL_LOCK_DIR
    )

    rows = []

    for p in sorted(
        MODEL_LOCK_DIR.rglob(
            "*"
        )
    ):
        if not p.is_file():
            continue

        row = {
            "file":
                str(
                    p
                ),

            "name":
                p.name,

            "suffix":
                p.suffix.lower(),

            "size_bytes":
                int(
                    p.stat().st_size
                ),

            "sha256":
                sha256_file(
                    p
                ),
        }

        if p.suffix.lower() in {
            ".json",
            ".txt",
            ".csv",
        }:
            try:
                preview = p.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                row[
                    "text_preview"
                ] = preview[
                    :5000
                ]

            except Exception as e:
                row[
                    "text_preview"
                ] = (
                    f"<read failed: {type(e).__name__}: {e}>"
                )

        else:
            row[
                "text_preview"
            ] = None

        rows.append(
            row
        )

    if not rows:
        raise RuntimeError(
            f"Model lock directory is empty: {MODEL_LOCK_DIR}"
        )

    return pd.DataFrame(
        rows
    )


def recursively_find_checkpoint_records(obj, context="root"):
    records = []

    if isinstance(
        obj,
        dict
    ):
        normalized_keys = {
            str(
                k
            ).lower():
                k
            for k in obj.keys()
        }

        path_keys = [
            k
            for low,
            k
            in normalized_keys.items()
            if any(
                token
                in
                low
                for token in [
                    "path",
                    "checkpoint",
                    "model",
                    "weight",
                    "ckpt",
                ]
            )
        ]

        hash_keys = [
            k
            for low,
            k
            in normalized_keys.items()
            if "sha256" in low
            or
            (
                "hash"
                in
                low
            )
        ]

        seed_keys = [
            k
            for low,
            k
            in normalized_keys.items()
            if "seed" in low
        ]

        task_keys = [
            k
            for low,
            k
            in normalized_keys.items()
            if any(
                token
                in
                low
                for token in [
                    "task",
                    "horizon",
                    "future",
                ]
            )
        ]

        candidate_paths = []

        for key in path_keys:
            value = obj.get(
                key
            )

            if isinstance(
                value,
                str
            ) and any(
                ext in value.lower()
                for ext in [
                    ".pt",
                    ".pth",
                    ".ckpt",
                ]
            ):
                candidate_paths.append(
                    (
                        key,
                        value,
                    )
                )

        for key, value in candidate_paths:
            records.append({
                "context":
                    context,

                "path_key":
                    str(
                        key
                    ),

                "checkpoint_path":
                    value,

                "hash_values":
                    [
                        str(
                            obj.get(
                                hk
                            )
                        )
                        for hk in hash_keys
                    ],

                "seed_values":
                    [
                        str(
                            obj.get(
                                sk
                            )
                        )
                        for sk in seed_keys
                    ],

                "task_values":
                    [
                        str(
                            obj.get(
                                tk
                            )
                        )
                        for tk in task_keys
                    ],
            })

        for key, value in obj.items():
            records.extend(
                recursively_find_checkpoint_records(
                    value,
                    context=f"{context}.{key}",
                )
            )

    elif isinstance(
        obj,
        list
    ):
        for idx, value in enumerate(
            obj
        ):
            records.extend(
                recursively_find_checkpoint_records(
                    value,
                    context=f"{context}[{idx}]",
                )
            )

    return records


def parse_lock_jsons(
    lock_df,
):
    rows = []

    for _, row in lock_df.iterrows():
        path = Path(
            row[
                "file"
            ]
        )

        if path.suffix.lower() != ".json":
            continue

        try:
            obj = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            continue

        found = recursively_find_checkpoint_records(
            obj,
            context=path.name,
        )

        for item in found:
            item[
                "lock_json"
            ] = str(
                path
            )

            rows.append(
                item
            )

    return pd.DataFrame(
        rows
    )


def resolve_candidate_checkpoint(
    raw_path,
):
    p = Path(
        raw_path
    )

    candidates = []

    if p.is_absolute():
        candidates.append(
            p
        )

    else:
        for root in [
            MODEL_LOCK_DIR,
            DOWNLOADS,
            MAIN_RESULT,
        ]:
            candidates.append(
                root
                /
                p
            )

    existing = [
        x
        for x in candidates
        if x.exists()
        and x.is_file()
    ]

    if len(
        existing
    ) == 1:
        return existing[
            0
        ]

    if len(
        existing
    ) > 1:
        return existing[
            0
        ]

    # Final targeted filename-only search.
    basename = p.name

    if basename:
        matches = []

        for root in [
            DOWNLOADS,
            MAIN_RESULT,
        ]:
            if not root.exists():
                continue

            for x in root.rglob(
                basename
            ):
                if x.is_file():
                    matches.append(
                        x
                    )

        if len(
            matches
        ) == 1:
            return matches[
                0
            ]

    return None


def infer_task_and_seed(
    checkpoint_path,
    context_text,
):
    text = (
        str(
            checkpoint_path
        )
        +
        " "
        +
        context_text
    ).lower()

    seed = None

    seed_match = re.search(
        r"(?:seed[_\-\s]*)(42|52|62)\b",
        text
    )

    if seed_match:
        seed = int(
            seed_match.group(
                1
            )
        )
    else:
        standalone = re.findall(
            r"\b(42|52|62)\b",
            text
        )

        if len(
            set(
                standalone
            )
        ) == 1:
            seed = int(
                standalone[
                    0
                ]
            )

    task = None

    if any(
        token in text
        for token in [
            "2s_to_2s",
            "2to2",
            "2_to_2",
            "h20",
            "future20",
            "pred20",
        ]
    ):
        task = "2s_to_2s"

    if any(
        token in text
        for token in [
            "2s_to_5s",
            "2to5",
            "2_to_5",
            "h50",
            "future50",
            "pred50",
        ]
    ):
        if task is not None:
            task = "AMBIGUOUS"
        else:
            task = "2s_to_5s"

    return (
        task,
        seed,
    )


def main():
    print(
        "\nSTEP 6D-K3C14 — EXACT FROZEN INFERENCE ADAPTER + MODEL-MAPPING AUDIT"
    )

    print(
        "Read-only. NO model prediction, NO q recalibration, NO geofence metric, and NO Step7 alarm is generated.\n"
    )

    require(
        K3C13_MANIFEST
    )

    actual_k3c13_hash = sha256_file(
        K3C13_MANIFEST
    )

    if actual_k3c13_hash != EXPECTED_K3C13_SHA256:
        raise RuntimeError(
            "K3C13 lock hash mismatch. "
            f"Expected {EXPECTED_K3C13_SHA256}, got {actual_k3c13_hash}"
        )

    evaluator, evaluator_candidates = find_named_python(
        "evaluate_birdsai_testreal_frozen_ensemble.py"
    )

    (
        eval_text,
        eval_tree,
        eval_node,
        load_source,
        load_signature,
    ) = extract_function_source(
        evaluator,
        "load_transformer",
    )

    eval_calls = collect_calls_to_function(
        evaluator,
        "load_transformer",
    )

    k3a2_script, k3a2_candidates = find_k3a2_script()

    k3a2_calls = []

    k3a2_literals = []

    if k3a2_script is not None:
        k3a2_calls = collect_calls_to_function(
            k3a2_script,
            "load_transformer",
        )

        k3a2_literals = extract_string_literals(
            k3a2_script
        )

    lock_df = read_lock_files()

    lock_df.to_csv(
        OUT
        / "TABLE_step6dk3c14_model_lock_inventory.csv",
        index=False,
    )

    parsed = parse_lock_jsons(
        lock_df
    )

    if len(
        parsed
    ) > 0:
        parsed.to_csv(
            OUT
            / "TABLE_step6dk3c14_checkpoint_records_from_lock_json.csv",
            index=False,
        )

    resolved_rows = []

    if len(
        parsed
    ) > 0:
        for _, row in parsed.iterrows():
            raw = row[
                "checkpoint_path"
            ]

            resolved = resolve_candidate_checkpoint(
                raw
            )

            context_text = (
                str(
                    row.get(
                        "context",
                        ""
                    )
                )
                +
                " "
                +
                " ".join(
                    row.get(
                        "seed_values",
                        []
                    )
                    if isinstance(
                        row.get(
                            "seed_values",
                            []
                        ),
                        list
                    )
                    else []
                )
                +
                " "
                +
                " ".join(
                    row.get(
                        "task_values",
                        []
                    )
                    if isinstance(
                        row.get(
                            "task_values",
                            []
                        ),
                        list
                    )
                    else []
                )
            )

            task, seed = infer_task_and_seed(
                raw,
                context_text,
            )

            resolved_rows.append({
                "lock_json":
                    row[
                        "lock_json"
                    ],

                "context":
                    row[
                        "context"
                    ],

                "raw_checkpoint_path":
                    raw,

                "resolved_checkpoint":
                    (
                        str(
                            resolved
                        )
                        if resolved is not None
                        else None
                    ),

                "resolved_exists":
                    bool(
                        resolved is not None
                    ),

                "resolved_sha256":
                    (
                        sha256_file(
                            resolved
                        )
                        if resolved is not None
                        else None
                    ),

                "inferred_task":
                    task,

                "inferred_seed":
                    seed,

                "lock_hash_values":
                    json.dumps(
                        row.get(
                            "hash_values",
                            []
                        )
                    ),
            })

    resolved_df = pd.DataFrame(
        resolved_rows
    )

    if len(
        resolved_df
    ) > 0:
        resolved_df.to_csv(
            OUT
            / "TABLE_step6dk3c14_resolved_checkpoint_mapping.csv",
            index=False,
        )

    (
        OUT
        / "STEP6D_K3C14_LOAD_TRANSFORMER_SOURCE.txt"
    ).write_text(
        (
            f"SOURCE FILE: {evaluator}\n"
            f"SHA256: {sha256_file(evaluator)}\n\n"
            f"{load_source}\n"
        ),
        encoding="utf-8",
    )

    call_bundle = {
        "evaluator_source":
            str(
                evaluator
            ),

        "evaluator_sha256":
            sha256_file(
                evaluator
            ),

        "load_transformer_signature":
            load_signature,

        "load_transformer_calls_in_evaluator":
            eval_calls,

        "k3a2_source":
            (
                str(
                    k3a2_script
                )
                if k3a2_script is not None
                else None
            ),

        "k3a2_sha256":
            (
                sha256_file(
                    k3a2_script
                )
                if k3a2_script is not None
                else None
            ),

        "load_transformer_calls_in_k3a2":
            k3a2_calls,

        "checkpoint_string_literals_in_k3a2":
            k3a2_literals,

        "evaluator_candidates":
            [
                str(
                    p
                )
                for p in evaluator_candidates
            ],

        "k3a2_candidates":
            [
                str(
                    p
                )
                for p in k3a2_candidates
            ],
    }

    (
        OUT
        / "STEP6D_K3C14_CALL_AND_SOURCE_AUDIT.json"
    ).write_text(
        json.dumps(
            call_bundle,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "VERIFIED K3C13 MANIFEST SHA256:"
    )
    print(
        actual_k3c13_hash
    )

    print(
        "\nSELECTED FROZEN EVALUATOR SOURCE:"
    )
    print(
        evaluator
    )

    print(
        "EVALUATOR SHA256:"
    )
    print(
        sha256_file(
            evaluator
        )
    )

    print(
        "\nload_transformer AST SIGNATURE:"
    )
    print(
        json.dumps(
            load_signature,
            indent=2,
        )
    )

    print(
        "\nload_transformer CALLS IN ORIGINAL EVALUATOR:"
    )
    print(
        json.dumps(
            eval_calls,
            indent=2,
        )
    )

    print(
        "\nSELECTED K3A2 FIRST-INFERENCE SOURCE:"
    )
    print(
        k3a2_script
    )

    print(
        "\nload_transformer CALLS IN K3A2:"
    )
    print(
        json.dumps(
            k3a2_calls,
            indent=2,
        )
    )

    print(
        "\nMODEL LOCK FILE INVENTORY:"
    )
    print(
        lock_df[
            [
                "name",
                "suffix",
                "size_bytes",
                "sha256",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRESOLVED CHECKPOINT MAPPING:"
    )

    if len(
        resolved_df
    ) > 0:
        cols = [
            "resolved_checkpoint",
            "resolved_exists",
            "resolved_sha256",
            "inferred_task",
            "inferred_seed",
            "lock_hash_values",
        ]

        print(
            resolved_df[
                cols
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "NO_CHECKPOINT_RECORDS_PARSED_FROM_LOCK_JSON"
        )

    # Decide readiness without guessing.
    ready = False
    reason = []

    if k3a2_script is None:
        reason.append(
            "K3A2 first-inference source not found."
        )

    if len(
        eval_calls
    ) == 0:
        reason.append(
            "No load_transformer call found in original evaluator."
        )

    if len(
        k3a2_calls
    ) == 0:
        reason.append(
            "No load_transformer call found in K3A2 source."
        )

    if len(
        resolved_df
    ) > 0:
        existing = resolved_df[
            resolved_df[
                "resolved_exists"
            ]
            ==
            True
        ].copy()

        unique_checkpoints = existing[
            "resolved_checkpoint"
        ].dropna().unique()

        task_seed_pairs = (
            existing[
                [
                    "inferred_task",
                    "inferred_seed",
                ]
            ]
            .dropna()
            .drop_duplicates()
        )

        expected_pairs = {
            (
                "2s_to_2s",
                42,
            ),
            (
                "2s_to_2s",
                52,
            ),
            (
                "2s_to_2s",
                62,
            ),
            (
                "2s_to_5s",
                42,
            ),
            (
                "2s_to_5s",
                52,
            ),
            (
                "2s_to_5s",
                62,
            ),
        }

        found_pairs = {
            (
                str(
                    r[
                        "inferred_task"
                    ]
                ),
                int(
                    r[
                        "inferred_seed"
                    ]
                ),
            )
            for _,
            r
            in task_seed_pairs.iterrows()
            if pd.notna(
                r[
                    "inferred_task"
                ]
            )
            and
            pd.notna(
                r[
                    "inferred_seed"
                ]
            )
        }

        if (
            len(
                unique_checkpoints
            )
            >=
            6
            and
            expected_pairs.issubset(
                found_pairs
            )
        ):
            ready = True

        else:
            reason.append(
                "Could not prove complete 6-checkpoint task/seed mapping from lock JSON alone."
            )

    else:
        reason.append(
            "No checkpoint paths could be parsed from model-lock JSON."
        )

    if ready:
        status = (
            "EXACT_FROZEN_INFERENCE_ADAPTER_AND_6_CHECKPOINT_MAPPING_PROVEN_"
            "READY_FOR_SECONDARY_LOCKED_INFERENCE"
        )
    else:
        status = (
            "FROZEN_INFERENCE_SOURCE_RECOVERED_BUT_CHECKPOINT_MAPPING_REQUIRES_"
            "TARGETED_ADAPTER_PATCH_BEFORE_INFERENCE"
        )

    summary = {
        "step":
            "STEP_6D_K3C14_EXACT_FROZEN_INFERENCE_ADAPTER_MODEL_MAPPING_AUDIT",

        "status":
            status,

        "k3c13_manifest":
            str(
                K3C13_MANIFEST
            ),

        "k3c13_manifest_sha256":
            actual_k3c13_hash,

        "evaluator_source":
            str(
                evaluator
            ),

        "evaluator_sha256":
            sha256_file(
                evaluator
            ),

        "load_transformer_signature":
            load_signature,

        "k3a2_source":
            (
                str(
                    k3a2_script
                )
                if k3a2_script is not None
                else None
            ),

        "readiness_reasons":
            reason,

        "no_prediction_generated":
            True,

        "no_q_recalibration":
            True,

        "no_step7_metric_generated":
            True,
    }

    (
        OUT
        / "STEP6D_K3C14_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSTATUS:"
    )
    print(
        status
    )

    if reason:
        print(
            "\nREADINESS NOTES:"
        )

        for item in reason:
            print(
                "-",
                item
            )

    print(
        "\nSTEP 6D-K3C14 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
