
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

V9_DIR = (
    R12_ROOT
    / "R12E_v9_exact_core10hz_read_write_dataflow_provenance"
)
V9_MANIFEST = V9_DIR / "R12E_V9_MANIFEST.json"

CANDIDATE = MAIN1 / "build_unified_10hz_dataset.py"

OUT = (
    R12_ROOT
    / "R12E_v10_direct_unified10hz_builder_audit"
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


def verify_v9():
    require(V9_MANIFEST)

    m = json.loads(
        V9_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R12E_V9_NO_TARGET_CORE10HZ_WRITER_FOUND_SOURCE_MAY_BE_MISSING"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R12-E-V9 status: {m.get('status')}"
        )

    return m


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return (
            f"{base}.{node.attr}"
            if base
            else node.attr
        )

    return ""


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


def context(lines, start, end, radius=10):
    s = max(
        1,
        start - radius,
    )

    e = min(
        len(lines),
        end + radius,
    )

    return (
        s,
        e,
        "\n".join(
            f"{i}: {lines[i-1]}"
            for i in range(
                s,
                e + 1,
            )
        ),
    )


def main():
    print(
        "\nR12-E-V10 — DIRECT AUDIT OF build_unified_10hz_dataset.py"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R12-E-V9 did not resolve dynamic target-file writes. This step directly audits "
        "build_unified_10hz_dataset.py, the strongest upstream producer candidate, and "
        "extracts executable expressions involving core/quality/acceptance/repaired-motion, "
        "10-Hz resampling, output filename construction, and save operations. "
        "No original file is modified.\n"
    )

    verify_v9()
    require(CANDIDATE)

    text = CANDIDATE.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(CANDIDATE)
    )

    parents = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(
            parent
        ):
            parents[child] = parent

    evidence_rows = []

    key_terms = [
        "quality_tier",
        "repaired_motion_pct",
        "accepted",
        "core",
        "extended",
        "reject_quality",
        "invalid_motion",
        "10hz",
        "10 hz",
        "resample",
        "downsample",
        "stride",
        "3",
        "np.savez",
        "np.savez_compressed",
        "to_csv",
        "metadata",
        "output",
        "filename",
        "test_real",
        "train_real",
    ]

    # Lexical contexts
    for i, line in enumerate(
        lines,
        start=1,
    ):
        ll = line.lower()

        matched = [
            term
            for term in key_terms
            if term.lower() in ll
        ]

        if not matched:
            continue

        s, e, ctx = context(
            lines,
            i,
            i,
            radius=10,
        )

        evidence_rows.append(
            {
                "evidence_type":
                    "LEXICAL",

                "function":
                    "",

                "node_type":
                    "",

                "start_line":
                    i,

                "end_line":
                    i,

                "matched_terms":
                    "|".join(
                        matched
                    ),

                "expression":
                    line,

                "context_start":
                    s,

                "context_end":
                    e,

                "context":
                    ctx,
            }
        )

    # AST comparisons / if conditions / assignments / calls
    interesting_node_types = (
        ast.Compare,
        ast.If,
        ast.Assign,
        ast.AnnAssign,
        ast.Call,
        ast.Subscript,
        ast.Return,
    )

    for node in ast.walk(tree):
        if not isinstance(
            node,
            interesting_node_types,
        ):
            continue

        src = ast.get_source_segment(
            text,
            node,
        ) or ""

        low = src.lower()

        matched = [
            term
            for term in key_terms
            if term.lower() in low
        ]

        if not matched:
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

        s, e, ctx = context(
            lines,
            start,
            end,
            radius=8,
        )

        evidence_rows.append(
            {
                "evidence_type":
                    "AST",

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

                "matched_terms":
                    "|".join(
                        matched
                    ),

                "expression":
                    src,

                "context_start":
                    s,

                "context_end":
                    e,

                "context":
                    ctx,
            }
        )

    evidence = pd.DataFrame(
        evidence_rows
    )

    if len(
        evidence
    ):
        evidence = (
            evidence
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
                    "expression",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # High-value executable expressions
    executable = []

    if len(
        evidence
    ):
        for _, row in evidence.iterrows():
            if row[
                "evidence_type"
            ] != "AST":
                continue

            src = str(
                row[
                    "expression"
                ]
            )

            low = src.lower()

            high = (
                (
                    "quality_tier"
                    in low
                    or
                    "repaired_motion_pct"
                    in low
                    or
                    "accepted"
                    in low
                )
                and
                (
                    "=="
                    in src
                    or
                    "<="
                    in src
                    or
                    "<"
                    in src
                    or
                    ".isin"
                    in low
                    or
                    ".loc["
                    in low
                    or
                    "query"
                    in low
                )
            ) or (
                "np.savez"
                in low
                or
                ".to_csv"
                in low
            ) or (
                "10hz"
                in low
                and
                "="
                in src
            )

            if not high:
                continue

            executable.append(
                {
                    "line":
                        int(
                            row[
                                "start_line"
                            ]
                        ),

                    "function":
                        row[
                            "function"
                        ],

                    "node_type":
                        row[
                            "node_type"
                        ],

                    "expression":
                        src,
                }
            )

    executable_df = pd.DataFrame(
        executable
    )

    # Source-level rule classification, conservative.
    text_low = text.lower()

    producer_signal = (
        (
            "np.savez"
            in text_low
            or
            "np.savez_compressed"
            in text_low
        )
        and
        "to_csv"
        in text_low
        and
        "10hz"
        in text_low
    )

    filter_signal = (
        "quality_tier"
        in text_low
        or
        "repaired_motion_pct"
        in text_low
        or
        "accepted"
        in text_low
    )

    rule_candidates = []

    for _, row in executable_df.iterrows():
        src = str(
            row[
                "expression"
            ]
        )

        low = src.lower()

        if (
            "quality_tier"
            in low
            and
            "core"
            in low
        ):
            rule_candidates.append(
                {
                    "rule_type":
                        "quality_tier_core",

                    "line":
                        int(
                            row[
                                "line"
                            ]
                        ),

                    "expression":
                        src,
                }
            )

        if (
            "accepted"
            in low
            and
            (
                "== 1"
                in low
                or
                "==1"
                in low
                or
                "true"
                in low
            )
        ):
            rule_candidates.append(
                {
                    "rule_type":
                        "accepted_true",

                    "line":
                        int(
                            row[
                                "line"
                            ]
                        ),

                    "expression":
                        src,
                }
            )

        if (
            "repaired_motion_pct"
            in low
            and
            (
                "<= 10"
                in low
                or
                "<=10"
                in low
                or
                "< 10"
                in low
                or
                "<10"
                in low
            )
        ):
            rule_candidates.append(
                {
                    "rule_type":
                        "repaired_motion_pct_le_10",

                    "line":
                        int(
                            row[
                                "line"
                            ]
                        ),

                    "expression":
                        src,
                }
            )

    rule_df = pd.DataFrame(
        rule_candidates
    )

    unique_rule_types = (
        sorted(
            rule_df[
                "rule_type"
            ]
            .unique()
            .tolist()
        )
        if len(
            rule_df
        )
        else []
    )

    if (
        producer_signal
        and
        len(
            unique_rule_types
        )
        ==
        1
    ):
        status = (
            "R12E_V10_UNIFIED10HZ_PRODUCER_AND_SINGLE_EXACT_FILTER_RULE_LOCKED_READY_FOR_R12E_V11_FULL_RECONSTRUCTION"
        )

    elif producer_signal and len(
        unique_rule_types
    ) > 1:
        status = (
            "R12E_V10_UNIFIED10HZ_PRODUCER_CONFIRMED_MULTIPLE_FILTER_EXPRESSIONS_FOUND_REVIEW_EXECUTION_ORDER"
        )

    elif producer_signal:
        status = (
            "R12E_V10_UNIFIED10HZ_PRODUCER_CONFIRMED_FILTER_RULE_NOT_YET_EXPLICITLY_PARSED"
        )

    else:
        status = (
            "R12E_V10_BUILD_UNIFIED_10HZ_IS_NOT_CONFIRMED_AS_TARGET_PRODUCER_REVIEW_CONTEXT"
        )

    evidence_path = (
        OUT
        /
        "R12E_V10_FULL_SOURCE_EVIDENCE.csv"
    )

    executable_path = (
        OUT
        /
        "R12E_V10_HIGH_VALUE_EXECUTABLE_EXPRESSIONS.csv"
    )

    rule_path = (
        OUT
        /
        "R12E_V10_FILTER_RULE_CANDIDATES.csv"
    )

    evidence.to_csv(
        evidence_path,
        index=False,
    )

    executable_df.to_csv(
        executable_path,
        index=False,
    )

    rule_df.to_csv(
        rule_path,
        index=False,
    )

    manifest = {
        "step":
            "R12E_V10_DIRECT_UNIFIED10HZ_BUILDER_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "candidate_path":
            str(
                CANDIDATE
            ),

        "candidate_sha256":
            sha256_file(
                CANDIDATE
            ),

        "producer_signal":
            bool(
                producer_signal
            ),

        "filter_signal":
            bool(
                filter_signal
            ),

        "unique_rule_types":
            unique_rule_types,

        "outputs": {
            "full_source_evidence":
                str(
                    evidence_path
                ),

            "high_value_expressions":
                str(
                    executable_path
                ),

            "filter_rule_candidates":
                str(
                    rule_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_V10_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "CANDIDATE BUILDER LOCK:"
    )

    print(
        "  Path:",
        CANDIDATE
    )

    print(
        "  SHA256:",
        sha256_file(
            CANDIDATE
        )
    )

    print(
        "  Producer signal:",
        producer_signal
    )

    print(
        "  Filter-field signal:",
        filter_signal
    )

    print(
        "\nFILTER RULE CANDIDATES:"
    )

    if len(
        rule_df
    ):
        print(
            rule_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No explicit filter candidate parsed."
        )

    print(
        "\nHIGH-VALUE EXECUTABLE EXPRESSIONS:"
    )

    if len(
        executable_df
    ):
        print(
            executable_df.head(
                160
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No high-value executable expressions found."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-E-V10 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
