from pathlib import Path
import ast
import json
import re
import hashlib

READ_ONLY_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k2b_representation_audit"
    / "exact_formula"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    (
        READ_ONLY_ROOT / "build_final_neural_dataset.py",
        "canonicalize_sample",
    ),
    (
        READ_ONLY_ROOT / "evaluate_fresh_wildbox_holdout.py",
        "canonicalize_window",
    ),
]

EXTRA_TARGETS = [
    (
        READ_ONLY_ROOT / "audit_canonical_motion.py",
        "rotate_to_heading",
    ),
    (
        READ_ONLY_ROOT / "train_transformer.py",
        "inverse_transform_batch",
    ),
    (
        READ_ONLY_ROOT / "evaluate_transformer_ensemble.py",
        "inverse_transform_batch",
    ),
]

KEY_TERMS = [
    "origin",
    "scale",
    "theta",
    "angle",
    "heading",
    "atan2",
    "cos",
    "sin",
    "norm",
    "history",
    "future",
    "canonical",
    "rotate",
    "eps",
    "floor",
]


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_tree(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )
    tree = ast.parse(text)
    return text, tree


def find_function(tree, name):
    matches = [
        n
        for n in ast.walk(tree)
        if isinstance(
            n,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and n.name == name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one function {name}, found {len(matches)}"
        )

    return matches[0]


def get_source(path, func_name):
    text, tree = load_tree(path)
    node = find_function(
        tree,
        func_name
    )

    src = ast.get_source_segment(
        text,
        node
    )

    if not src:
        lines = text.splitlines()
        src = "\n".join(
            lines[
                node.lineno - 1:
                getattr(
                    node,
                    "end_lineno",
                    node.lineno
                )
            ]
        )

    return {
        "path": path,
        "function": func_name,
        "source": src,
        "line_start": int(node.lineno),
        "line_end": int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        ),
        "node": node,
        "tree": tree,
        "full_text": text,
    }


def normalized_ast_dump(node):
    # Drop line-number metadata so structural comparison is stable.
    return ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False
    )


def collect_called_names(node):
    names = []

    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                names.append(
                    n.func.id
                )
            elif isinstance(n.func, ast.Attribute):
                parts = []
                cur = n.func

                while isinstance(cur, ast.Attribute):
                    parts.append(
                        cur.attr
                    )
                    cur = cur.value

                if isinstance(cur, ast.Name):
                    parts.append(
                        cur.id
                    )

                names.append(
                    ".".join(
                        reversed(parts)
                    )
                )

    return sorted(
        set(names)
    )


def extract_assignments(node, source_text):
    rows = []

    for n in ast.walk(node):
        if not isinstance(
            n,
            (ast.Assign, ast.AnnAssign, ast.AugAssign)
        ):
            continue

        seg = ast.get_source_segment(
            source_text,
            n
        )

        if not seg:
            continue

        low = seg.lower()

        if any(
            term in low
            for term in KEY_TERMS
        ):
            rows.append(
                (
                    int(
                        getattr(
                            n,
                            "lineno",
                            -1
                        )
                    ),
                    " ".join(
                        seg.strip().split()
                    )
                )
            )

    rows.sort(
        key=lambda x: x[0]
    )

    return rows


def extract_returns(node, source_text):
    rows = []

    for n in ast.walk(node):
        if not isinstance(
            n,
            ast.Return
        ):
            continue

        seg = ast.get_source_segment(
            source_text,
            n
        )

        if seg:
            rows.append(
                (
                    int(
                        n.lineno
                    ),
                    " ".join(
                        seg.strip().split()
                    )
                )
            )

    rows.sort(
        key=lambda x: x[0]
    )

    return rows


def find_local_helper_sources(
    tree,
    full_text,
    called_names
):
    helper_rows = []

    top_level_funcs = {
        n.name: n
        for n in tree.body
        if isinstance(
            n,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        )
    }

    for call in called_names:
        base = call.split(".")[-1]

        if base not in top_level_funcs:
            continue

        node = top_level_funcs[base]
        src = ast.get_source_segment(
            full_text,
            node
        )

        helper_rows.append({
            "name":
                base,

            "line_start":
                int(
                    node.lineno
                ),

            "line_end":
                int(
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno
                    )
                ),

            "source":
                src,
        })

    return helper_rows


def signature_summary(info):
    node = info["node"]

    args = [
        a.arg
        for a in node.args.args
    ]

    assignments = extract_assignments(
        node,
        info["full_text"]
    )

    returns = extract_returns(
        node,
        info["full_text"]
    )

    calls = collect_called_names(
        node
    )

    helpers = find_local_helper_sources(
        info["tree"],
        info["full_text"],
        calls
    )

    return {
        "file":
            str(
                info["path"]
            ),

        "function":
            info["function"],

        "line_start":
            info["line_start"],

        "line_end":
            info["line_end"],

        "args":
            args,

        "sha256_source":
            sha256_text(
                info["source"]
            ),

        "calls":
            calls,

        "key_assignments":
            [
                {
                    "line":
                        line,

                    "code":
                        code,
                }
                for line, code
                in assignments
            ],

        "returns":
            [
                {
                    "line":
                        line,

                    "code":
                        code,
                }
                for line, code
                in returns
            ],

        "local_helpers":
            [
                {
                    "name":
                        h["name"],

                    "line_start":
                        h["line_start"],

                    "line_end":
                        h["line_end"],
                }
                for h in helpers
            ],
    }, helpers


def write_source_bundle(
    label,
    info,
    summary,
    helpers
):
    with open(
        OUT
        / f"{label}_FUNCTION_SOURCE.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            f"FILE: {info['path']}\n"
            f"FUNCTION: {info['function']}\n"
            f"LINES: {info['line_start']}-{info['line_end']}\n"
            f"SHA256_SOURCE: {summary['sha256_source']}\n\n"
        )

        f.write(
            info["source"]
        )

        f.write(
            "\n\n"
            +
            "=" * 84
            +
            "\nLOCAL HELPERS CALLED DIRECTLY\n"
            +
            "=" * 84
            +
            "\n"
        )

        if helpers:
            for h in helpers:
                f.write(
                    f"\nHELPER: {h['name']} "
                    f"LINES {h['line_start']}-{h['line_end']}\n"
                )

                f.write(
                    "-" * 84
                    +
                    "\n"
                )

                f.write(
                    h["source"]
                    or ""
                )

                f.write(
                    "\n"
                )
        else:
            f.write(
                "NONE\n"
            )


def compact_formula_lines(summary):
    out = []

    for item in summary[
        "key_assignments"
    ]:
        out.append(
            f"L{item['line']}: {item['code']}"
        )

    for item in summary[
        "returns"
    ]:
        out.append(
            f"L{item['line']}: {item['code']}"
        )

    return out


def main():
    print(
        "\nSTEP 6D-K2B1 — EXACT CANONICAL FORMULA EXTRACTION"
    )

    print(
        "Read-only source extraction. No BAMBI transform or evaluation.\n"
    )

    all_infos = []
    all_summaries = []

    for i, (
        path,
        func_name
    ) in enumerate(
        TARGETS
        +
        EXTRA_TARGETS,
        start=1
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

        info = get_source(
            path,
            func_name
        )

        summary, helpers = signature_summary(
            info
        )

        label = (
            f"{i:02d}_"
            +
            path.stem
            +
            "_"
            +
            func_name
        )

        write_source_bundle(
            label,
            info,
            summary,
            helpers
        )

        all_infos.append(
            info
        )

        all_summaries.append(
            summary
        )

    # Main two canonical functions.
    a = all_infos[0]
    b = all_infos[1]

    a_summary = all_summaries[0]
    b_summary = all_summaries[1]

    exact_source_equal = (
        a["source"].strip()
        ==
        b["source"].strip()
    )

    exact_ast_equal = (
        normalized_ast_dump(
            a["node"]
        )
        ==
        normalized_ast_dump(
            b["node"]
        )
    )

    # Compare compact math-relevant statements after replacing function names.
    a_lines = compact_formula_lines(
        a_summary
    )

    b_lines = compact_formula_lines(
        b_summary
    )

    report = {
        "step":
            "STEP_6D_K2B1_EXACT_CANONICAL_FORMULA_EXTRACTION",

        "main_candidate_1":
            {
                "file":
                    str(
                        a["path"]
                    ),

                "function":
                    a["function"],

                "lines":
                    [
                        a["line_start"],
                        a["line_end"]
                    ],
            },

        "main_candidate_2":
            {
                "file":
                    str(
                        b["path"]
                    ),

                "function":
                    b["function"],

                "lines":
                    [
                        b["line_start"],
                        b["line_end"]
                    ],
            },

        "exact_source_equal":
            bool(
                exact_source_equal
            ),

        "exact_ast_equal":
            bool(
                exact_ast_equal
            ),

        "candidate_1_key_formula_lines":
            a_lines,

        "candidate_2_key_formula_lines":
            b_lines,

        "status":
            "FORMULAS_EXTRACTED_PENDING_HUMAN_CONFIRMATION",
    }

    with open(
        OUT
        / "STEP6D_K2B1_EXACT_FORMULA_REPORT.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    with open(
        OUT
        / "STEP6D_K2B1_EXACT_FORMULA_REPORT.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K2B1 — EXACT CANONICAL FORMULA EXTRACTION\n"
        )

        f.write(
            "=" * 82
            +
            "\n\n"
        )

        f.write(
            "MAIN CANDIDATE 1\n"
        )

        f.write(
            "----------------\n"
        )

        f.write(
            f"{a['path']}\n"
            f"{a['function']} "
            f"lines {a['line_start']}-{a['line_end']}\n\n"
        )

        for line in a_lines:
            f.write(
                line
                +
                "\n"
            )

        f.write(
            "\nMAIN CANDIDATE 2\n"
        )

        f.write(
            "----------------\n"
        )

        f.write(
            f"{b['path']}\n"
            f"{b['function']} "
            f"lines {b['line_start']}-{b['line_end']}\n\n"
        )

        for line in b_lines:
            f.write(
                line
                +
                "\n"
            )

        f.write(
            "\nSTRUCTURAL COMPARISON\n"
        )

        f.write(
            "---------------------\n"
        )

        f.write(
            f"exact_source_equal = {exact_source_equal}\n"
        )

        f.write(
            f"exact_ast_equal = {exact_ast_equal}\n"
        )

        f.write(
            "\nINVERSE-TRANSFORM / HELPER SOURCES\n"
        )

        f.write(
            "----------------------------------\n"
        )

        for s in all_summaries[2:]:
            f.write(
                f"\nFILE: {s['file']}\n"
                f"FUNCTION: {s['function']}\n"
                f"LINES: {s['line_start']}-{s['line_end']}\n"
            )

            for line in compact_formula_lines(
                s
            ):
                f.write(
                    line
                    +
                    "\n"
                )

        f.write(
            "\nSTATUS\n"
        )

        f.write(
            "------\n"
        )

        f.write(
            report[
                "status"
            ]
            +
            "\n"
        )

    print(
        "MAIN CANDIDATE 1 KEY FORMULA LINES:"
    )

    for line in a_lines:
        print(
            line
        )

    print(
        "\nMAIN CANDIDATE 2 KEY FORMULA LINES:"
    )

    for line in b_lines:
        print(
            line
        )

    print(
        "\nSTRUCTURAL COMPARISON:"
    )

    print(
        "exact_source_equal =",
        exact_source_equal
    )

    print(
        "exact_ast_equal =",
        exact_ast_equal
    )

    print(
        "\nINVERSE / HELPER FORMULA LINES:"
    )

    for s in all_summaries[2:]:
        print(
            f"\n{s['file']} :: {s['function']}"
        )

        for line in compact_formula_lines(
            s
        ):
            print(
                line
            )

    print(
        "\nSTATUS:"
    )

    print(
        report[
            "status"
        ]
    )

    print(
        "\nSTEP 6D-K2B1 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
