
from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R12_ROOT = REVISION_ROOT / "R12_track_accounting"

V6_DIR = (
    R12_ROOT
    / "R12E_v6_exact_final_filter_source_provenance"
)
V6_MANIFEST = V6_DIR / "R12E_V6_MANIFEST.json"

BUILDER = MAIN1 / "build_final_neural_dataset.py"

OUT = (
    R12_ROOT
    / "R12E_v7_direct_final_builder_ast_audit"
)
OUT.mkdir(parents=True, exist_ok=True)


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


def verify_v6():
    require(V6_MANIFEST)

    m = json.loads(
        V6_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R12E_V6_EXACT_FINAL_FILTER_RULE_NOT_YET_LOCATED_REVIEW_TOP_SOURCE_CANDIDATES"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R12-E-V6 status: {m.get('status')}"
        )

    return m


def containing_function(node, parents):
    cur = node

    while cur in parents:
        cur = parents[cur]

        if isinstance(
            cur,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            return cur.name

    return "<module>"


def context_block(
    lines,
    start_line,
    end_line,
    radius=18,
):
    start = max(
        1,
        start_line - radius,
    )

    end = min(
        len(lines),
        end_line + radius,
    )

    return (
        start,
        end,
        "\n".join(
            f"{i}: {lines[i-1]}"
            for i in range(
                start,
                end + 1,
            )
        ),
    )


def main():
    print(
        "\nR12-E-V7 — DIRECT FINAL-BUILDER AST / EXECUTION-CONTEXT AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R12-E-V6 identified build_final_neural_dataset.py as the real final-dataset "
        "builder but did not recover the implemented filter condition. This step audits "
        "that single source file directly using Python AST plus exact source context. "
        "It extracts every expression involving the literal 'core', every .loc[...] "
        "selection, dataframe read/write statements, split/TestReal logic, and their "
        "execution order. No original file is modified.\n"
    )

    verify_v6()
    require(BUILDER)

    text = BUILDER.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(BUILDER)
    )

    parents = {}

    for parent in ast.walk(
        tree
    ):
        for child in ast.iter_child_nodes(
            parent
        ):
            parents[child] = parent

    evidence_rows = []

    # 1) AST nodes that literally involve "core".
    for node in ast.walk(
        tree
    ):
        has_core = False

        for sub in ast.walk(
            node
        ):
            if (
                isinstance(
                    sub,
                    ast.Constant
                )
                and
                isinstance(
                    sub.value,
                    str
                )
                and
                sub.value.strip().lower()
                ==
                "core"
            ):
                has_core = True
                break

        if not has_core:
            continue

        if not isinstance(
            node,
            (
                ast.Compare,
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,
                ast.If,
                ast.Return,
                ast.Subscript,
                ast.Call,
            )
        ):
            continue

        start = int(
            getattr(
                node,
                "lineno",
                1,
            )
        )

        end = int(
            getattr(
                node,
                "end_lineno",
                start,
            )
        )

        src = ast.get_source_segment(
            text,
            node,
        ) or ""

        cstart, cend, context = context_block(
            lines,
            start,
            end,
            radius=20,
        )

        evidence_rows.append(
            {
                "evidence_type":
                    "AST_LITERAL_CORE",

                "function":
                    containing_function(
                        node,
                        parents,
                    ),

                "node_type":
                    type(
                        node
                    ).__name__,

                "start_line":
                    start,

                "end_line":
                    end,

                "source_expression":
                    src,

                "context_start":
                    cstart,

                "context_end":
                    cend,

                "context":
                    context,
            }
        )

    # 2) Every .loc[...] subscript.
    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Subscript,
        ):
            continue

        value = node.value

        is_loc = (
            isinstance(
                value,
                ast.Attribute,
            )
            and
            value.attr
            ==
            "loc"
        )

        if not is_loc:
            continue

        start = int(
            node.lineno
        )

        end = int(
            getattr(
                node,
                "end_lineno",
                start,
            )
        )

        src = ast.get_source_segment(
            text,
            node,
        ) or ""

        cstart, cend, context = context_block(
            lines,
            start,
            end,
            radius=12,
        )

        evidence_rows.append(
            {
                "evidence_type":
                    "AST_LOC_SELECTION",

                "function":
                    containing_function(
                        node,
                        parents,
                    ),

                "node_type":
                    type(
                        node
                    ).__name__,

                "start_line":
                    start,

                "end_line":
                    end,

                "source_expression":
                    src,

                "context_start":
                    cstart,

                "context_end":
                    cend,

                "context":
                    context,
            }
        )

    # 3) Key lexical fingerprints, with generous context.
    lexical_terms = [
        "core",
        "corrected_windows_2s_to_2s",
        "corrected_windows_2s_to_5s",
        "corrected_trajectory_dataset",
        "final_neural_trajectory_dataset",
        "birdsai_test_metadata",
        "test_real",
        "testreal",
        "split",
        "read_csv",
        "to_csv",
        "np.savez",
        "savez",
        "track_key",
    ]

    for i, line in enumerate(
        lines,
        start=1,
    ):
        ll = line.lower()

        matched = [
            term
            for term in lexical_terms
            if term.lower() in ll
        ]

        if not matched:
            continue

        cstart, cend, context = context_block(
            lines,
            i,
            i,
            radius=12,
        )

        evidence_rows.append(
            {
                "evidence_type":
                    "LEXICAL_CONTEXT",

                "function":
                    "",

                "node_type":
                    "",

                "start_line":
                    i,

                "end_line":
                    i,

                "source_expression":
                    line,

                "context_start":
                    cstart,

                "context_end":
                    cend,

                "context":
                    context,

                "matched_terms":
                    "|".join(
                        matched
                    ),
            }
        )

    evidence_df = pd.DataFrame(
        evidence_rows
    )

    if len(
        evidence_df
    ):
        evidence_df = (
            evidence_df
            .sort_values(
                [
                    "start_line",
                    "evidence_type",
                ]
            )
            .drop_duplicates(
                subset=[
                    "evidence_type",
                    "start_line",
                    "end_line",
                    "source_expression",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # Extract compact high-value candidate expressions.
    candidate_rows = []

    for _, row in evidence_df.iterrows():
        src = str(
            row.get(
                "source_expression",
                ""
            )
        )

        low = src.lower()

        high_value = (
            "core"
            in
            low
            or
            ".loc["
            in
            low
            or
            "test_real"
            in
            low
            or
            "testreal"
            in
            low
        )

        if not high_value:
            continue

        candidate_rows.append(
            {
                "line":
                    int(
                        row[
                            "start_line"
                        ]
                    ),

                "evidence_type":
                    row[
                        "evidence_type"
                    ],

                "function":
                    row.get(
                        "function",
                        "",
                    ),

                "expression":
                    src,
            }
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    # Conservative rule lock:
    # If a Compare/If AST expression itself contains literal core, preserve the
    # exact source expression instead of normalizing it into our own guessed rule.
    explicit_core_ast = []

    for _, row in evidence_df.iterrows():
        if row[
            "evidence_type"
        ] != "AST_LITERAL_CORE":
            continue

        node_type = str(
            row[
                "node_type"
            ]
        )

        src = str(
            row[
                "source_expression"
            ]
        )

        if node_type in {
            "Compare",
            "If",
            "Subscript",
            "Assign",
        }:
            explicit_core_ast.append(
                {
                    "line":
                        int(
                            row[
                                "start_line"
                            ]
                        ),

                    "node_type":
                        node_type,

                    "function":
                        row[
                            "function"
                        ],

                    "exact_source_expression":
                        src,
                }
            )

    explicit_df = pd.DataFrame(
        explicit_core_ast
    )

    # A source-locked rule is ready if at least one executable AST expression
    # contains literal core and is a comparison/filter-like expression.
    lockable = False

    if len(
        explicit_df
    ):
        for _, row in explicit_df.iterrows():
            src = str(
                row[
                    "exact_source_expression"
                ]
            ).lower()

            if (
                "core"
                in
                src
                and
                (
                    "=="
                    in
                    src
                    or
                    ".isin"
                    in
                    src
                    or
                    ".loc["
                    in
                    src
                    or
                    "query"
                    in
                    src
                )
            ):
                lockable = True
                break

    if lockable:
        status = (
            "R12E_V7_EXACT_CORE_FILTER_EXPRESSION_SOURCE_LOCKED_READY_FOR_R12E_V8_FINAL_RECONSTRUCTION"
        )

    else:
        status = (
            "R12E_V7_BUILDER_CONTEXT_EXTRACTED_BUT_FILTER_EXPRESSION_STILL_NEEDS_MANUAL_LINE_REVIEW"
        )

    evidence_path = (
        OUT
        /
        "R12E_V7_FULL_BUILDER_AST_CONTEXT_EVIDENCE.csv"
    )

    candidate_path = (
        OUT
        /
        "R12E_V7_HIGH_VALUE_EXECUTION_LINES.csv"
    )

    explicit_path = (
        OUT
        /
        "R12E_V7_EXPLICIT_CORE_AST_EXPRESSIONS.csv"
    )

    evidence_df.to_csv(
        evidence_path,
        index=False,
    )

    candidate_df.to_csv(
        candidate_path,
        index=False,
    )

    explicit_df.to_csv(
        explicit_path,
        index=False,
    )

    manifest = {
        "step":
            "R12E_V7_DIRECT_FINAL_BUILDER_AST_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "builder_path":
            str(
                BUILDER
            ),

        "builder_sha256":
            sha256_file(
                BUILDER
            ),

        "explicit_core_ast_expression_count":
            int(
                len(
                    explicit_df
                )
            ),

        "source_rule_lockable":
            lockable,

        "outputs": {
            "full_ast_context_evidence":
                str(
                    evidence_path
                ),

            "high_value_execution_lines":
                str(
                    candidate_path
                ),

            "explicit_core_ast_expressions":
                str(
                    explicit_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_V7_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "BUILDER LOCK:"
    )

    print(
        "  Path:",
        BUILDER
    )

    print(
        "  SHA256:",
        sha256_file(
            BUILDER
        )
    )

    print(
        "\nEXPLICIT CORE AST EXPRESSIONS:"
    )

    if len(
        explicit_df
    ):
        print(
            explicit_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No executable AST expression containing literal 'core' was found."
        )

    print(
        "\nHIGH-VALUE EXECUTION LINES:"
    )

    if len(
        candidate_df
    ):
        print(
            candidate_df.head(
                120
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No high-value lines found."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-E-V7 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
