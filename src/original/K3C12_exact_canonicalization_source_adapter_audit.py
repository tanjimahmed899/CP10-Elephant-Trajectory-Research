from pathlib import Path
import ast
import hashlib
import json
import re
import types

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

K3C11_DIR = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_reconstruction_v1"
)

K2B3_DIR = (
    BASE
    / "k2b3_final_canonical"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "exact_canonicalization_adapter_audit"
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


def find_python_files_with_function(function_name):
    hits = []

    pattern = re.compile(
        rf"^\s*def\s+{re.escape(function_name)}\s*\(",
        flags=re.MULTILINE,
    )

    for root in [
        DOWNLOADS,
        Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"),
    ]:
        if not root.exists():
            continue

        for p in root.rglob("*.py"):
            try:
                text = p.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
            except Exception:
                continue

            if pattern.search(text):
                score = 0
                low = str(p).lower()

                if p.name.lower() == "build_final_neural_dataset.py":
                    score += 1000

                if "build_final_neural_dataset" in p.name.lower():
                    score += 500

                if "evaluate_fresh_wildbox_holdout" in p.name.lower():
                    score += 300

                if "main result" in low:
                    score += 200

                if "copy" in low:
                    score -= 100

                hits.append({
                    "file": p,
                    "score": score,
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })

    hits = sorted(
        hits,
        key=lambda x: (
            -x["score"],
            str(x["file"]).lower(),
        )
    )

    return hits


def extract_function_source(path, function_name):
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
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        )
        and node.name == function_name
    ]

    if len(nodes) != 1:
        raise RuntimeError(
            f"{path}: expected exactly one {function_name}, found {len(nodes)}"
        )

    node = nodes[0]

    source = ast.get_source_segment(
        text,
        node
    )

    if not source:
        raise RuntimeError(
            f"Could not extract {function_name} source from {path}"
        )

    return (
        text,
        tree,
        node,
        source,
    )


def safe_namespace_from_source(
    full_text,
    tree,
    target_function_name
):
    """
    Execute only imports, function definitions, class definitions, and simple
    literal assignments from the source file. Never execute main blocks or
    top-level data processing.
    """
    safe_body = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.ClassDef,
            )
        ):
            safe_body.append(
                node
            )
            continue

        if isinstance(
            node,
            ast.Assign
        ):
            try:
                ast.literal_eval(
                    node.value
                )
            except Exception:
                continue

            safe_body.append(
                node
            )

        elif isinstance(
            node,
            ast.AnnAssign
        ):
            if node.value is None:
                continue

            try:
                ast.literal_eval(
                    node.value
                )
            except Exception:
                continue

            safe_body.append(
                node
            )

    module = ast.Module(
        body=safe_body,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module
    )

    ns = {
        "__name__":
            "_canonicalization_audit_module_",

        "np":
            np,

        "numpy":
            np,
    }

    try:
        exec(
            compile(
                module,
                "<canonicalization_safe_extract>",
                "exec",
            ),
            ns,
            ns,
        )
    except Exception as e:
        # Fall back to target function only with np namespace.
        _, _, node, source = extract_function_source_from_text(
            full_text,
            target_function_name
        )

        ns = {
            "__name__":
                "_canonicalization_audit_module_",

            "np":
                np,

            "numpy":
                np,
        }

        exec(
            compile(
                source,
                "<canonicalization_target_only>",
                "exec",
            ),
            ns,
            ns,
        )

    if target_function_name not in ns:
        raise RuntimeError(
            f"Could not construct callable {target_function_name}"
        )

    return ns


def extract_function_source_from_text(
    text,
    function_name
):
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
        and node.name == function_name
    ]

    if len(nodes) != 1:
        raise RuntimeError(
            f"Expected one function {function_name}, found {len(nodes)}"
        )

    node = nodes[0]

    source = ast.get_source_segment(
        text,
        node
    )

    return (
        tree,
        node,
        source,
    )


def function_signature_from_ast(node):
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

    return args


def return_statements(node):
    rows = []

    for sub in ast.walk(
        node
    ):
        if isinstance(
            sub,
            ast.Return
        ):
            try:
                expression = ast.unparse(
                    sub.value
                )
            except Exception:
                expression = repr(
                    sub.value
                )

            rows.append({
                "line":
                    int(
                        getattr(
                            sub,
                            "lineno",
                            -1
                        )
                    ),

                "expression":
                    expression,
            })

    return rows


def load_one_k3c11_sample(task):
    npz_path = (
        K3C11_DIR
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_windows.npz"
    )

    require(
        npz_path
    )

    arrays = np.load(
        npz_path,
        allow_pickle=False
    )

    history = np.asarray(
        arrays[
            "history_anchor_norm"
        ],
        dtype=float,
    )

    future = np.asarray(
        arrays[
            "future_anchor_norm"
        ],
        dtype=float,
    )

    return (
        npz_path,
        history[
            0
        ],
        future[
            0
        ],
    )


def locked_scale_floor(task):
    matches = sorted(
        p
        for p in K2B3_DIR.glob(
            "*metadata_LOCKED.csv"
        )
        if task.lower()
        in
        p.name.lower()
    )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Expected one locked K2B3 metadata file for {task}:\n"
            +
            "\n".join(
                str(p)
                for p in matches
            )
        )

    df = pd.read_csv(
        matches[
            0
        ],
        usecols=[
            "scale_floor",
        ],
    )

    values = pd.to_numeric(
        df[
            "scale_floor"
        ],
        errors="coerce"
    ).dropna().unique()

    if len(
        values
    ) != 1:
        raise RuntimeError(
            f"{task}: scale_floor is not uniquely locked: {values}"
        )

    return (
        matches[
            0
        ],
        float(
            values[
                0
            ]
        ),
    )


def call_canonicalize_sample(
    fn,
    history,
    future,
    scale_floor,
):
    import inspect

    sig = inspect.signature(
        fn
    )

    params = list(
        sig.parameters.keys()
    )

    mapping = {
        "history":
            history,

        "hist":
            history,

        "x":
            history,

        "obs":
            history,

        "future":
            future,

        "fut":
            future,

        "y":
            future,

        "target":
            future,

        "scale_floor":
            scale_floor,

        "floor":
            scale_floor,
    }

    if all(
        p in mapping
        for p in params
    ):
        kwargs = {
            p:
                mapping[
                    p
                ]
            for p in params
        }

        return (
            sig,
            fn(
                **kwargs
            ),
        )

    # Common positional form.
    if len(
        params
    ) == 3:
        return (
            sig,
            fn(
                history,
                future,
                scale_floor,
            ),
        )

    raise RuntimeError(
        f"Unsupported canonicalize_sample signature: {sig}"
    )


def describe_result(result):
    rows = []

    if isinstance(
        result,
        tuple
    ):
        iterable = list(
            enumerate(
                result
            )
        )

    elif isinstance(
        result,
        list
    ):
        iterable = list(
            enumerate(
                result
            )
        )

    elif isinstance(
        result,
        dict
    ):
        iterable = list(
            result.items()
        )

    else:
        iterable = [
            (
                "return",
                result,
            )
        ]

    for key, value in iterable:
        arr = None

        try:
            arr = np.asarray(
                value
            )
        except Exception:
            pass

        rows.append({
            "key":
                str(
                    key
                ),

            "python_type":
                type(
                    value
                ).__name__,

            "shape":
                (
                    str(
                        tuple(
                            arr.shape
                        )
                    )
                    if
                    arr is not None
                    else
                    None
                ),

            "dtype":
                (
                    str(
                        arr.dtype
                    )
                    if
                    arr is not None
                    else
                    None
                ),

            "preview":
                repr(
                    value
                )[
                    :500
                ],
        })

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nSTEP 6D-K3C12 — EXACT CANONICALIZATION SOURCE + ADAPTER AUDIT"
    )

    print(
        "Read-only. NO derived-dataset lock, NO model inference, NO q recalibration, and NO Step7 metrics are generated.\n"
    )

    candidates = find_python_files_with_function(
        "canonicalize_sample"
    )

    if not candidates:
        raise RuntimeError(
            "Could not find any Python source containing def canonicalize_sample."
        )

    candidate_df = pd.DataFrame(
        [
            {
                "rank":
                    i + 1,

                "file":
                    str(
                        item[
                            "file"
                        ]
                    ),

                "score":
                    item[
                        "score"
                    ],

                "size_bytes":
                    item[
                        "size_bytes"
                    ],

                "sha256":
                    item[
                        "sha256"
                    ],
            }
            for i, item in enumerate(
                candidates
            )
        ]
    )

    candidate_df.to_csv(
        OUT
        / "TABLE_step6dk3c12_canonicalize_sample_candidates.csv",
        index=False
    )

    selected = candidates[
        0
    ][
        "file"
    ]

    full_text, tree, node, source = extract_function_source(
        selected,
        "canonicalize_sample",
    )

    signature_ast = function_signature_from_ast(
        node
    )

    returns = return_statements(
        node
    )

    (
        OUT
        / "STEP6D_K3C12_CANONICALIZE_SAMPLE_SOURCE.txt"
    ).write_text(
        (
            f"SOURCE FILE: {selected}\n"
            f"SHA256: {sha256_file(selected)}\n\n"
            f"{source}\n"
        ),
        encoding="utf-8"
    )

    ns = safe_namespace_from_source(
        full_text,
        tree,
        "canonicalize_sample",
    )

    fn = ns[
        "canonicalize_sample"
    ]

    dry_rows = []
    result_descriptions = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        npz_path, history, future = load_one_k3c11_sample(
            task
        )

        floor_path, floor = locked_scale_floor(
            task
        )

        sig, result = call_canonicalize_sample(
            fn,
            history,
            future,
            floor,
        )

        desc = describe_result(
            result
        )

        desc.insert(
            0,
            "task",
            task,
        )

        result_descriptions.append(
            desc
        )

        dry_rows.append({
            "task":
                task,

            "source_file":
                str(
                    selected
                ),

            "source_sha256":
                sha256_file(
                    selected
                ),

            "call_signature":
                str(
                    sig
                ),

            "scale_floor":
                floor,

            "scale_floor_source":
                str(
                    floor_path
                ),

            "input_history_shape":
                str(
                    tuple(
                        history.shape
                    )
                ),

            "input_future_shape":
                str(
                    tuple(
                        future.shape
                    )
                ),

            "input_all_finite":
                bool(
                    np.isfinite(
                        history
                    ).all()
                    and
                    np.isfinite(
                        future
                    ).all()
                ),

            "dry_run_success":
                True,

            "k3c11_npz":
                str(
                    npz_path
                ),
        })

    dry_df = pd.DataFrame(
        dry_rows
    )

    dry_df.to_csv(
        OUT
        / "TABLE_step6dk3c12_dry_run.csv",
        index=False
    )

    result_df = pd.concat(
        result_descriptions,
        ignore_index=True,
    )

    result_df.to_csv(
        OUT
        / "TABLE_step6dk3c12_return_structure.csv",
        index=False
    )

    summary = {
        "step":
            "STEP_6D_K3C12_EXACT_CANONICALIZATION_SOURCE_ADAPTER_AUDIT",

        "selected_source":
            str(
                selected
            ),

        "selected_source_sha256":
            sha256_file(
                selected
            ),

        "ast_signature":
            signature_ast,

        "return_statements":
            returns,

        "candidate_count":
            len(
                candidates
            ),

        "dry_run_tasks":
            dry_rows,

        "status":
            "EXACT_CANONICALIZE_SAMPLE_SOURCE_RECOVERED_AND_DRY_RUN_SUCCESS_READY_FOR_DERIVED_DATASET_LOCK",

        "nonclaims": [
            "No anchor-normalized canonical dataset was locked in this step.",
            "No secondary BAMBI Transformer inference was run.",
            "No conformal q value was recalibrated.",
            "No Step7 geofence or alarm metric was generated.",
        ],
    }

    (
        OUT
        / "STEP6D_K3C12_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "SELECTED CANONICALIZATION SOURCE:"
    )
    print(
        selected
    )

    print(
        "\nSOURCE SHA256:"
    )
    print(
        sha256_file(
            selected
        )
    )

    print(
        "\nAST SIGNATURE:"
    )
    print(
        json.dumps(
            signature_ast,
            indent=2
        )
    )

    print(
        "\nRETURN STATEMENTS:"
    )
    print(
        json.dumps(
            returns,
            indent=2
        )
    )

    print(
        "\nDRY RUN:"
    )
    print(
        dry_df.to_string(
            index=False
        )
    )

    print(
        "\nRETURN STRUCTURE:"
    )
    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )
    print(
        "EXACT_CANONICALIZE_SAMPLE_SOURCE_RECOVERED_AND_DRY_RUN_SUCCESS_READY_FOR_DERIVED_DATASET_LOCK"
    )

    print(
        "\nSTEP 6D-K3C12 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
