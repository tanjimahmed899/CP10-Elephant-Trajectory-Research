
from pathlib import Path
import ast
import hashlib
import inspect
import json
import re
from datetime import datetime, timezone

import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

EQUIV_SRC = (
    MAIN2
    / "step6d_k2b2_executable_canonicalizer_equivalence.py"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_B_exact_geometry_formula_lock"
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


def hash_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def eval_path_expr(node, env):
    """
    Safely evaluate the small subset of Python expressions typically used
    for pathlib source-path constants:
      Path("...")
      NAME
      NAME / "child"
      NAME.joinpath(...)
      str literals
    """
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise KeyError(node.id)

    if isinstance(node, ast.Call):
        # Path("...")
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and len(node.args) == 1
        ):
            return Path(
                eval_path_expr(
                    node.args[0],
                    env
                )
            )

        # something.joinpath(...)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "joinpath"
        ):
            base = eval_path_expr(
                node.func.value,
                env
            )
            base = Path(base)

            parts = [
                str(
                    eval_path_expr(
                        a,
                        env
                    )
                )
                for a in node.args
            ]

            return base.joinpath(
                *parts
            )

        raise ValueError(
            f"Unsupported call: {ast.dump(node)}"
        )

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = eval_path_expr(
            node.left,
            env
        )
        right = eval_path_expr(
            node.right,
            env
        )
        return Path(left) / str(right)

    raise ValueError(
        f"Unsupported expression: {ast.dump(node)}"
    )


def resolve_top_level_paths(source_path):
    text = source_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text,
        filename=str(source_path)
    )

    env = {}

    for node in tree.body:
        target_name = None
        value_node = None

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
                value_node = node.value

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target_name = node.target.id
                value_node = node.value

        if target_name is None or value_node is None:
            continue

        try:
            value = eval_path_expr(
                value_node,
                env
            )
        except Exception:
            continue

        env[target_name] = value

    return env


def find_all_named_files(filename):
    found = []

    for root in [MAIN1, MAIN2]:
        if not root.exists():
            continue

        for p in root.rglob(filename):
            if p.is_file():
                found.append(
                    p.resolve()
                )

    # stable unique
    uniq = []
    seen = set()

    for p in found:
        s = str(p).lower()
        if s not in seen:
            seen.add(s)
            uniq.append(p)

    return uniq


def get_function_node(tree, name):
    candidates = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef
            )
        )
        and node.name == name
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level function {name!r}; "
            f"found {len(candidates)}"
        )

    return candidates[0]


def get_source(lines, node, numbered=False):
    start = int(node.lineno)
    end = int(
        getattr(
            node,
            "end_lineno",
            node.lineno
        )
    )

    if numbered:
        return "\n".join(
            f"{i}: {lines[i-1]}"
            for i in range(
                start,
                end + 1
            )
        )

    return "\n".join(
        lines[
            start - 1:
            end
        ]
    )


def local_function_map(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef
            )
        )
    }


def directly_called_names(node):
    out = set()

    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                out.add(
                    sub.func.id
                )

    return out


def recursive_local_dependencies(root_name, fmap):
    visited = set()
    order = []

    def visit(name):
        if name in visited:
            return

        visited.add(name)

        if name not in fmap:
            return

        node = fmap[name]

        for callee in sorted(
            directly_called_names(node)
        ):
            if callee in fmap:
                visit(callee)

        order.append(name)

    visit(root_name)

    # root last due DFS; put root first for readability,
    # preserve dependency order after it.
    deps = [
        n
        for n in order
        if n != root_name
    ]

    return [
        root_name
    ] + deps


def referenced_global_names(node):
    local_names = set()

    # function args
    for arg in list(node.args.args) + list(node.args.kwonlyargs):
        local_names.add(
            arg.arg
        )

    if node.args.vararg:
        local_names.add(
            node.args.vararg.arg
        )

    if node.args.kwarg:
        local_names.add(
            node.args.kwarg.arg
        )

    # local stores
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            local_names.add(
                sub.id
            )

    loads = {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name)
        and isinstance(sub.ctx, ast.Load)
    }

    builtins_like = {
        "np",
        "math",
        "float",
        "int",
        "len",
        "max",
        "min",
        "range",
        "enumerate",
        "zip",
        "abs",
        "dict",
        "list",
        "tuple",
        "set",
        "bool",
        "str",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "Exception",
    }

    return sorted(
        loads
        -
        local_names
        -
        builtins_like
    )


def top_level_constant_assignments(tree, lines):
    rows = []

    for node in tree.body:
        target = None

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target = node.target.id

        if target is None:
            continue

        start = int(node.lineno)
        end = int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        )

        rows.append(
            {
                "name": target,
                "start_line": start,
                "end_line": end,
                "source": "\n".join(
                    f"{i}: {lines[i-1]}"
                    for i in range(
                        start,
                        end + 1
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def formula_lines_for_nodes(lines, nodes):
    patterns = {
        "origin_anchor": [
            r"\borigin\b",
            r"\banchor\b",
            r"history\s*\[\s*-1",
        ],
        "heading_angle": [
            r"arctan2",
            r"\bheading\b",
            r"\bangle\b",
            r"\btheta\b",
        ],
        "rotation": [
            r"np\.cos",
            r"np\.sin",
            r"\brotation\b",
            r"\brotate\b",
            r"\bR\s*=",
            r"@\s*R",
            r"@\s*R\.T",
        ],
        "scale": [
            r"np\.linalg\.norm",
            r"\bscale\b",
            r"path_length",
            r"displacement",
            r"floor",
        ],
        "canonical_output": [
            r"history_canonical",
            r"future_canonical",
            r"canonical",
        ],
    }

    selected_ranges = []

    for node in nodes:
        selected_ranges.append(
            (
                int(node.lineno),
                int(
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno
                    )
                ),
                node.name,
            )
        )

    rows = []

    for start, end, fname in selected_ranges:
        for i in range(start, end + 1):
            line = lines[i - 1]

            for category, pats in patterns.items():
                if any(
                    re.search(
                        p,
                        line,
                        flags=re.IGNORECASE
                    )
                    for p in pats
                ):
                    rows.append(
                        {
                            "function": fname,
                            "line_number": i,
                            "category": category,
                            "line": line,
                        }
                    )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR9-H-B — EXACT FROZEN CANONICALIZATION GEOMETRY FORMULA LOCK"
    )

    print(
        "DIAGNOSTIC / PROTOCOL LOCK ONLY. "
        "This step resolves the exact BUILD_SRC used by the prior executable "
        "canonicalizer-equivalence audit, extracts build_final_neural_dataset.py::"
        "canonicalize_sample and all local function dependencies, records referenced "
        "constants, and locks the exact origin/heading/rotation/scale formula source. "
        "NO geometry comparison yet, NO model training, NO bootstrap, NO frozen overwrite.\n"
    )

    require(
        EQUIV_SRC
    )

    env = resolve_top_level_paths(
        EQUIV_SRC
    )

    build_src = env.get(
        "BUILD_SRC"
    )

    fallback_used = False
    candidate_files = find_all_named_files(
        "build_final_neural_dataset.py"
    )

    if build_src is not None:
        build_src = Path(
            build_src
        )

    # If static resolution cannot recover BUILD_SRC, allow a single unambiguous
    # build_final_neural_dataset.py copy. If there are several, compare hashes
    # before proceeding.
    if build_src is None or not build_src.exists():
        fallback_used = True

        if len(candidate_files) == 0:
            raise RuntimeError(
                "Could not resolve BUILD_SRC and no build_final_neural_dataset.py "
                "was found under Main Result / Main Result 2.0."
            )

        if len(candidate_files) == 1:
            build_src = candidate_files[0]

        else:
            # Compare canonicalize_sample bodies across all copies.
            body_hashes = []

            for p in candidate_files:
                txt = p.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
                tree = ast.parse(
                    txt,
                    filename=str(p)
                )
                lines = txt.splitlines()
                fn = get_function_node(
                    tree,
                    "canonicalize_sample"
                )
                body = get_source(
                    lines,
                    fn,
                    numbered=False
                )
                body_hashes.append(
                    (
                        p,
                        hash_text(body)
                    )
                )

            unique_hashes = sorted(
                set(
                    h
                    for _, h in body_hashes
                )
            )

            if len(unique_hashes) != 1:
                details = "\n".join(
                    f"{p} -> {h}"
                    for p, h in body_hashes
                )

                raise RuntimeError(
                    "Multiple differing build_final_neural_dataset.py copies found "
                    "and BUILD_SRC could not be statically resolved. "
                    "Refusing to guess.\n"
                    +
                    details
                )

            # All copies have same canonicalize_sample body.
            # Prefer Main Result 2.0, then lexicographic path.
            build_src = sorted(
                candidate_files,
                key=lambda p: (
                    0 if str(MAIN2).lower() in str(p).lower() else 1,
                    str(p).lower()
                )
            )[0]

    require(
        build_src
    )

    text = build_src.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(build_src)
    )

    fmap = local_function_map(
        tree
    )

    if "canonicalize_sample" not in fmap:
        raise RuntimeError(
            f"{build_src} does not contain canonicalize_sample()."
        )

    dependency_order = recursive_local_dependencies(
        "canonicalize_sample",
        fmap
    )

    relevant_nodes = [
        fmap[name]
        for name in dependency_order
        if name in fmap
    ]

    # Referenced globals/constants over canonicalizer + recursive local deps.
    referenced = set()

    for node in relevant_nodes:
        referenced.update(
            referenced_global_names(
                node
            )
        )

    constants = top_level_constant_assignments(
        tree,
        lines
    )

    if len(constants):
        constants_used = constants[
            constants["name"].isin(
                sorted(
                    referenced
                )
            )
        ].copy()
    else:
        constants_used = pd.DataFrame()

    # Exact source bundle.
    function_rows = []

    for node in relevant_nodes:
        src_plain = get_source(
            lines,
            node,
            numbered=False
        )

        src_numbered = get_source(
            lines,
            node,
            numbered=True
        )

        function_rows.append(
            {
                "function": node.name,
                "start_line": int(
                    node.lineno
                ),
                "end_line": int(
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno
                    )
                ),
                "source_sha256": hash_text(
                    src_plain
                ),
                "source": src_numbered,
            }
        )

    function_df = pd.DataFrame(
        function_rows
    )

    formulas = formula_lines_for_nodes(
        lines,
        relevant_nodes
    )

    # Direct dependency table.
    dep_rows = []

    for node in relevant_nodes:
        for callee in sorted(
            directly_called_names(
                node
            )
        ):
            if callee in fmap:
                dep_rows.append(
                    {
                        "caller": node.name,
                        "callee": callee,
                    }
                )

    dep_df = pd.DataFrame(
        dep_rows
    )

    # Inventory all discovered copies for audit.
    candidate_rows = []

    for p in candidate_files:
        candidate_rows.append(
            {
                "path": str(p),
                "file_sha256": sha256_file(
                    p
                ),
                "is_selected": (
                    p.resolve()
                    ==
                    build_src.resolve()
                ),
            }
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    selected_function = function_df[
        function_df["function"]
        ==
        "canonicalize_sample"
    ].iloc[0]

    lock_summary = pd.DataFrame(
        [
            {
                "resolved_equivalence_source":
                    str(EQUIV_SRC),

                "resolved_build_src":
                    str(build_src),

                "build_src_file_sha256":
                    sha256_file(
                        build_src
                    ),

                "canonicalize_sample_start_line":
                    int(
                        selected_function[
                            "start_line"
                        ]
                    ),

                "canonicalize_sample_end_line":
                    int(
                        selected_function[
                            "end_line"
                        ]
                    ),

                "canonicalize_sample_source_sha256":
                    str(
                        selected_function[
                            "source_sha256"
                        ]
                    ),

                "recursive_local_function_count":
                    int(
                        len(
                            relevant_nodes
                        )
                    ),

                "referenced_global_name_count":
                    int(
                        len(
                            referenced
                        )
                    ),

                "resolved_by_fallback":
                    bool(
                        fallback_used
                    ),
            }
        ]
    )

    # Save.
    candidate_path = (
        OUT
        / "R9H_B_BUILD_FINAL_SOURCE_CANDIDATES.csv"
    )

    summary_path = (
        OUT
        / "R9H_B_EXACT_FORMULA_LOCK_SUMMARY.csv"
    )

    funcs_path = (
        OUT
        / "R9H_B_EXACT_CANONICALIZER_FUNCTIONS.csv"
    )

    deps_path = (
        OUT
        / "R9H_B_FUNCTION_DEPENDENCIES.csv"
    )

    const_path = (
        OUT
        / "R9H_B_REFERENCED_CONSTANTS.csv"
    )

    formula_path = (
        OUT
        / "R9H_B_GEOMETRY_FORMULA_LINES.csv"
    )

    bundle_path = (
        OUT
        / "R9H_B_EXACT_CANONICALIZER_SOURCE_BUNDLE.txt"
    )

    candidate_df.to_csv(
        candidate_path,
        index=False
    )

    lock_summary.to_csv(
        summary_path,
        index=False
    )

    function_df.to_csv(
        funcs_path,
        index=False
    )

    dep_df.to_csv(
        deps_path,
        index=False
    )

    constants_used.to_csv(
        const_path,
        index=False
    )

    formulas.to_csv(
        formula_path,
        index=False
    )

    with open(
        bundle_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "R9-H-B EXACT FROZEN CANONICALIZER SOURCE BUNDLE\n\n"
        )

        f.write(
            f"Selected source: {build_src}\n"
        )

        f.write(
            f"Selected source SHA256: {sha256_file(build_src)}\n\n"
        )

        for _, row in function_df.iterrows():
            f.write(
                "=" * 100
                +
                "\n"
            )

            f.write(
                f"FUNCTION: {row['function']} "
                f"(lines {row['start_line']}-{row['end_line']})\n"
            )

            f.write(
                f"SOURCE SHA256: {row['source_sha256']}\n"
            )

            f.write(
                "=" * 100
                +
                "\n"
            )

            f.write(
                row[
                    "source"
                ]
            )

            f.write(
                "\n\n"
            )

    status = (
        "R9H_B_EXACT_FROZEN_CANONICALIZATION_FORMULA_LOCKED_READY_FOR_R9H_C_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
    )

    manifest = {
        "step":
            "R9H_B_EXACT_FROZEN_CANONICALIZATION_GEOMETRY_FORMULA_LOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "equivalence_source":
            str(
                EQUIV_SRC
            ),

        "equivalence_source_sha256":
            sha256_file(
                EQUIV_SRC
            ),

        "selected_build_src":
            str(
                build_src
            ),

        "selected_build_src_sha256":
            sha256_file(
                build_src
            ),

        "canonicalize_sample_source_sha256":
            str(
                selected_function[
                    "source_sha256"
                ]
            ),

        "recursive_local_dependencies":
            dependency_order,

        "referenced_global_names":
            sorted(
                referenced
            ),

        "fallback_used":
            bool(
                fallback_used
            ),

        "new_training":
            False,

        "geometry_metrics_run":
            False,

        "bootstrap_performed":
            False,

        "frozen_results_modified":
            False,

        "outputs": {
            "candidate_inventory":
                str(
                    candidate_path
                ),

            "formula_lock_summary":
                str(
                    summary_path
                ),

            "function_sources":
                str(
                    funcs_path
                ),

            "function_dependencies":
                str(
                    deps_path
                ),

            "referenced_constants":
                str(
                    const_path
                ),

            "formula_lines":
                str(
                    formula_path
                ),

            "exact_source_bundle":
                str(
                    bundle_path
                ),
        },
    }

    (
        OUT
        / "R9H_B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "RESOLVED FROZEN CANONICALIZER SOURCE:"
    )

    print(
        lock_summary.to_string(
            index=False
        )
    )

    print(
        "\nFUNCTION DEPENDENCIES:"
    )

    if len(dep_df):
        print(
            dep_df.to_string(
                index=False
            )
        )
    else:
        print(
            "canonicalize_sample has no project-local function dependencies."
        )

    print(
        "\nREFERENCED CONSTANTS / TOP-LEVEL ASSIGNMENTS:"
    )

    if len(constants_used):
        print(
            constants_used.to_string(
                index=False
            )
        )
    else:
        print(
            "No referenced top-level constant assignments were detected."
        )

    print(
        "\nGEOMETRY FORMULA LINES:"
    )

    if len(formulas):
        print(
            formulas.to_string(
                index=False
            )
        )
    else:
        print(
            "No formula lines detected — inspect exact source bundle."
        )

    print(
        "\nEXACT FROZEN CANONICALIZER SOURCE:"
    )

    for _, row in function_df.iterrows():
        print(
            "\n"
            +
            "=" * 100
        )

        print(
            f"{row['function']} "
            f"(lines {row['start_line']}-{row['end_line']})"
        )

        print(
            "=" * 100
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
        "\nR9-H-B COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
