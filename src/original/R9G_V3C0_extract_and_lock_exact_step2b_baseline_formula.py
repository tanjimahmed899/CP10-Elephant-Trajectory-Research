
from pathlib import Path
import ast
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd


MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

STEP2B = MAIN2 / "step2b_raw_vs_corrected_forecasting.py"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3C0_exact_baseline_formula_lock"
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


def load_source():
    require(STEP2B)

    text = STEP2B.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(STEP2B),
    )

    return text, lines, tree


def function_index(tree):
    rows = []

    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            rows.append(
                {
                    "function":
                        node.name,

                    "start_line":
                        int(
                            node.lineno
                        ),

                    "end_line":
                        int(
                            getattr(
                                node,
                                "end_lineno",
                                node.lineno,
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def names_called(node):
    out = set()

    for child in ast.walk(node):
        if isinstance(
            child,
            ast.Call,
        ):
            f = child.func

            if isinstance(
                f,
                ast.Name,
            ):
                out.add(
                    f.id
                )

    return out


def names_loaded(node):
    out = set()

    for child in ast.walk(node):
        if (
            isinstance(
                child,
                ast.Name,
            )
            and
            isinstance(
                child.ctx,
                ast.Load,
            )
        ):
            out.add(
                child.id
            )

    return out


def get_function_map(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
    }


def trace_dependencies(
    root_name,
    function_map,
):
    if root_name not in function_map:
        raise RuntimeError(
            f"Required function not found: {root_name}"
        )

    visited = set()
    queue = [
        root_name
    ]

    edges = []

    while queue:
        name = queue.pop(
            0
        )

        if name in visited:
            continue

        visited.add(
            name
        )

        node = function_map[
            name
        ]

        calls = sorted(
            names_called(
                node
            )
        )

        for called in calls:
            if called in function_map:
                edges.append(
                    {
                        "caller":
                            name,

                        "callee":
                            called,
                    }
                )

                if called not in visited:
                    queue.append(
                        called
                    )

    return visited, pd.DataFrame(
        edges
    )


def source_block(
    lines,
    node,
):
    start = int(
        node.lineno
    )

    end = int(
        getattr(
            node,
            "end_lineno",
            node.lineno,
        )
    )

    return "\n".join(
        f"{i}: {lines[i-1]}"
        for i in range(
            start,
            end + 1,
        )
    )


def collect_relevant_functions(
    lines,
    function_map,
    names,
):
    rows = []

    for name in sorted(
        names,
        key=lambda n: function_map[n].lineno,
    ):
        node = function_map[
            name
        ]

        rows.append(
            {
                "function":
                    name,

                "start_line":
                    int(
                        node.lineno
                    ),

                "end_line":
                    int(
                        getattr(
                            node,
                            "end_lineno",
                            node.lineno,
                        )
                    ),

                "source":
                    source_block(
                        lines,
                        node,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def collect_referenced_constants(
    tree,
    relevant_function_names,
    function_map,
):
    loaded = set()

    for name in relevant_function_names:
        loaded |= names_loaded(
            function_map[
                name
            ]
        )

    rows = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue

        targets = []

        if isinstance(
            node,
            ast.Assign,
        ):
            for t in node.targets:
                if isinstance(
                    t,
                    ast.Name,
                ):
                    targets.append(
                        t.id
                    )
        else:
            if isinstance(
                node.target,
                ast.Name,
            ):
                targets.append(
                    node.target.id
                )

        for name in targets:
            if name not in loaded:
                continue

            rows.append(
                {
                    "name":
                        name,

                    "line_number":
                        int(
                            node.lineno
                        ),

                    "source":
                        ast.unparse(
                            node
                        )
                        if hasattr(
                            ast,
                            "unparse",
                        )
                        else
                        "",
                }
            )

    return pd.DataFrame(
        rows
    )


def find_model_branches(
    lines,
    eval_node,
):
    start = int(
        eval_node.lineno
    )

    end = int(
        getattr(
            eval_node,
            "end_lineno",
            eval_node.lineno,
        )
    )

    rows = []

    keywords = [
        "Stationary",
        "ConstantVelocity",
        "velocity",
        "prediction",
        "pred",
        "ADE",
        "FDE",
        "history",
        "future",
    ]

    for i in range(
        start,
        end + 1,
    ):
        line = lines[
            i - 1
        ]

        if any(
            k.lower()
            in
            line.lower()
            for k in keywords
        ):
            rows.append(
                {
                    "line_number":
                        i,

                    "line":
                        line,
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-G-V3C0 — EXACT STEP2B BASELINE FORMULA LOCK"
    )

    print(
        "DIAGNOSTIC ONLY. NO baseline evaluation, NO isotropic comparison, "
        "NO bootstrap, NO training, NO frozen modification. "
        "This step extracts the exact Stationary / ConstantVelocity evaluation "
        "function and every local helper it depends on from the original Step2b source.\n"
    )

    text, lines, tree = (
        load_source()
    )

    funcs = function_index(
        tree
    )

    fmap = get_function_map(
        tree
    )

    if "evaluate_representation" not in fmap:
        raise RuntimeError(
            "evaluate_representation not found in Step2b source."
        )

    relevant_names, edges = (
        trace_dependencies(
            "evaluate_representation",
            fmap,
        )
    )

    relevant = (
        collect_relevant_functions(
            lines,
            fmap,
            relevant_names,
        )
    )

    constants = (
        collect_referenced_constants(
            tree,
            relevant_names,
            fmap,
        )
    )

    branches = (
        find_model_branches(
            lines,
            fmap[
                "evaluate_representation"
            ],
        )
    )

    func_index_path = (
        OUT
        / "R9G_V3C0_FUNCTION_INDEX.csv"
    )

    relevant_path = (
        OUT
        / "R9G_V3C0_RELEVANT_BASELINE_FUNCTIONS.csv"
    )

    edges_path = (
        OUT
        / "R9G_V3C0_FUNCTION_DEPENDENCIES.csv"
    )

    constants_path = (
        OUT
        / "R9G_V3C0_REFERENCED_CONSTANTS.csv"
    )

    branches_path = (
        OUT
        / "R9G_V3C0_MODEL_BRANCH_LINES.csv"
    )

    funcs.to_csv(
        func_index_path,
        index=False,
    )

    relevant.to_csv(
        relevant_path,
        index=False,
    )

    edges.to_csv(
        edges_path,
        index=False,
    )

    constants.to_csv(
        constants_path,
        index=False,
    )

    branches.to_csv(
        branches_path,
        index=False,
    )

    status = (
        "R9G_V3C0_EXACT_STEP2B_BASELINE_FORMULA_LOCK_COMPLETE_READY_FOR_R9G_V3C_CLEAN_COMPARISON"
    )

    manifest = {
        "step":
            "R9G_V3C0_EXACT_STEP2B_BASELINE_FORMULA_LOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "step2b_source":
            str(
                STEP2B
            ),

        "step2b_source_sha256":
            sha256_file(
                STEP2B
            ),

        "root_function":
            "evaluate_representation",

        "relevant_functions":
            sorted(
                relevant_names
            ),

        "new_training":
            False,

        "comparison_performed":
            False,

        "bootstrap_performed":
            False,

        "frozen_results_modified":
            False,

        "relevant_functions_csv":
            str(
                relevant_path
            ),

        "dependencies_csv":
            str(
                edges_path
            ),

        "constants_csv":
            str(
                constants_path
            ),

        "model_branch_lines_csv":
            str(
                branches_path
            ),
    }

    (
        OUT
        / "R9G_V3C0_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "RELEVANT FUNCTION DEPENDENCIES:"
    )

    if len(
        edges
    ):
        print(
            edges.to_string(
                index=False
            )
        )
    else:
        print(
            "evaluate_representation uses no local helper functions."
        )

    print(
        "\nREFERENCED CONSTANTS:"
    )

    if len(
        constants
    ):
        print(
            constants.to_string(
                index=False
            )
        )
    else:
        print(
            "No module-level constants referenced."
        )

    print(
        "\nMODEL BRANCH LINES:"
    )

    print(
        branches.to_string(
            index=False
        )
    )

    print(
        "\nEXACT RELEVANT FUNCTION SOURCE:"
    )

    for _, row in relevant.iterrows():
        print(
            "\n"
            +
            "=" * 90
        )

        print(
            f"FUNCTION: {row['function']} "
            f"(lines {row['start_line']}-{row['end_line']})"
        )

        print(
            "=" * 90
        )

        print(
            row[
                "source"
            ]
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-G-V3C0 COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
