
from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_A_V2_focused_python_canonicalizer_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    MAIN2 / "step6d_k2b3_build_and_lock_final_canonical_dataset.py",
    MAIN2 / "step6d_k2b2_executable_canonicalizer_equivalence.py",
    MAIN2 / "step6d_k2b2_v4_float64_and_exact_inverse_roundtrip.py",
    MAIN2 / "step6d_k2b2_v3_task_floor_equivalence_roundtrip.py",
    MAIN2 / "step6b_scale_adaptive_conformal.py",
    MAIN2 / "K3C13_full_anchor_normalized_canonical_dataset_lock.py",
    MAIN1 / "audit_canonical_motion.py",
]

FUNCTION_TERMS = [
    "canonical",
    "canonicalize",
    "heading",
    "rotate",
    "rotation",
    "angle",
    "theta",
    "anchor",
    "normalize",
    "inverse",
    "roundtrip",
]

SOURCE_TERMS = [
    "np.arctan2",
    "arctan2",
    "np.cos",
    "np.sin",
    "heading",
    "rotation",
    "rotate",
    "canonical",
    "origin",
    "anchor",
    "history",
    "future",
    "scale",
    "norm",
]


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


def source_hash(text):
    compact = re.sub(
        r"\s+",
        "",
        text,
    )

    return hashlib.sha256(
        compact.encode(
            "utf-8"
        )
    ).hexdigest()


def load_source(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(path),
    )

    return text, lines, tree


def function_nodes(tree):
    out = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            out.append(
                node
            )

    return sorted(
        out,
        key=lambda n: n.lineno,
    )


def get_source_block(lines, node):
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
        lines[
            start - 1:
            end
        ]
    )


def numbered_source_block(lines, node):
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
        f"{i}: {lines[i - 1]}"
        for i in range(
            start,
            end + 1,
        )
    )


def relevant_function(
    node,
    source,
):
    name = node.name.lower()
    low = source.lower()

    name_hit = any(
        t in name
        for t in FUNCTION_TERMS
    )

    source_hits = sum(
        low.count(
            t.lower()
        )
        for t in SOURCE_TERMS
    )

    geometry_formula_hit = any(
        t in low
        for t in [
            "arctan2",
            "np.cos",
            "np.sin",
            "rotation",
            "rotate",
            "heading",
        ]
    )

    canonical_hit = (
        "canonical"
        in
        low
    )

    return bool(
        name_hit
        or
        (
            canonical_hit
            and
            source_hits >= 2
        )
        or
        (
            geometry_formula_hit
            and
            source_hits >= 3
        )
    )


def extract_functions(path):
    text, lines, tree = (
        load_source(
            path
        )
    )

    rows = []

    for node in function_nodes(
        tree
    ):
        source = get_source_block(
            lines,
            node,
        )

        if not relevant_function(
            node,
            source,
        ):
            continue

        rows.append(
            {
                "file":
                    path.name,

                "path":
                    str(
                        path
                    ),

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

                "source_hash_ws_ignored":
                    source_hash(
                        source
                    ),

                "source":
                    numbered_source_block(
                        lines,
                        node,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def extract_formula_lines(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    patterns = {
        "heading":
            [
                r"arctan2",
                r"heading",
                r"theta",
                r"angle",
            ],

        "origin_anchor":
            [
                r"origin",
                r"anchor",
                r"history\[-1",
                r"history\[:,\s*-1",
            ],

        "rotation":
            [
                r"np\.cos",
                r"np\.sin",
                r"rotation",
                r"rotate",
                r"\bR\s*=",
            ],

        "scale":
            [
                r"np\.linalg\.norm",
                r"\bscale\b",
                r"distance",
                r"displacement",
            ],

        "canonical":
            [
                r"canonical",
                r"history_can",
                r"future_can",
                r"history_canonical",
                r"future_canonical",
            ],

        "inverse":
            [
                r"inverse",
                r"decanonical",
                r"roundtrip",
            ],
    }

    rows = []

    for i, line in enumerate(
        lines,
        start=1,
    ):
        for category, pats in patterns.items():
            if any(
                re.search(
                    p,
                    line,
                    flags=re.IGNORECASE,
                )
                for p in pats
            ):
                rows.append(
                    {
                        "file":
                            path.name,

                        "path":
                            str(
                                path
                            ),

                        "line_number":
                            i,

                        "category":
                            category,

                        "line":
                            line,
                    }
                )

    return pd.DataFrame(
        rows
    )


def build_context(path, formula_df, radius=8):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()
    selected = set()

    if len(
        formula_df
    ) == 0:
        return pd.DataFrame()

    for line_number in formula_df[
        "line_number"
    ].astype(int):
        for j in range(
            max(
                1,
                line_number - radius,
            ),
            min(
                len(lines),
                line_number + radius,
            )
            +
            1,
        ):
            selected.add(
                j
            )

    return pd.DataFrame(
        [
            {
                "file":
                    path.name,

                "path":
                    str(
                        path
                    ),

                "line_number":
                    j,

                "line":
                    lines[
                        j - 1
                    ],
            }
            for j in sorted(
                selected
            )
        ]
    )


def compare_function_hashes(function_df):
    if len(
        function_df
    ) == 0:
        return pd.DataFrame()

    rows = []

    grouped = function_df.groupby(
        "source_hash_ws_ignored",
        dropna=False,
    )

    for hash_value, g in grouped:
        if len(
            g
        ) < 2:
            continue

        rows.append(
            {
                "source_hash_ws_ignored":
                    hash_value,

                "occurrences":
                    int(
                        len(
                            g
                        )
                    ),

                "functions":
                    " | ".join(
                        (
                            g["file"]
                            +
                            "::"
                            +
                            g["function"]
                        ).tolist()
                    ),

                "files":
                    " | ".join(
                        sorted(
                            g["file"]
                            .unique()
                            .tolist()
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-H-A-V2 — FOCUSED PYTHON CANONICALIZER SOURCE AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. JSON pose/georef files are excluded. "
        "Only forecasting/canonicalization Python sources identified by R9-H-A are audited. "
        "This step extracts exact canonicalization-related functions, formula lines, "
        "source context, and checks whether identical function bodies are reused across files. "
        "NO geometry metrics, NO retraining, NO bootstrap, NO frozen modification.\n"
    )

    missing = [
        str(p)
        for p in CANDIDATES
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Expected canonicalization candidate file(s) missing:\n"
            +
            "\n".join(
                missing
            )
        )

    inventory_rows = []
    function_frames = []
    formula_frames = []
    context_frames = []

    for path in CANDIDATES:
        inventory_rows.append(
            {
                "file":
                    path.name,

                "path":
                    str(
                        path
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),

                "size_bytes":
                    int(
                        path.stat().st_size
                    ),
            }
        )

        fdf = extract_functions(
            path
        )

        ldf = extract_formula_lines(
            path
        )

        cdf = build_context(
            path,
            ldf,
            radius=8,
        )

        if len(
            fdf
        ):
            function_frames.append(
                fdf
            )

        if len(
            ldf
        ):
            formula_frames.append(
                ldf
            )

        if len(
            cdf
        ):
            context_frames.append(
                cdf
            )

    inventory = pd.DataFrame(
        inventory_rows
    )

    functions = (
        pd.concat(
            function_frames,
            ignore_index=True,
        )
        if function_frames
        else
        pd.DataFrame()
    )

    formulas = (
        pd.concat(
            formula_frames,
            ignore_index=True,
        )
        if formula_frames
        else
        pd.DataFrame()
    )

    contexts = (
        pd.concat(
            context_frames,
            ignore_index=True,
        )
        if context_frames
        else
        pd.DataFrame()
    )

    duplicates = (
        compare_function_hashes(
            functions
        )
    )

    inventory_path = (
        OUT
        / "R9H_A_V2_PYTHON_SOURCE_INVENTORY.csv"
    )

    functions_path = (
        OUT
        / "R9H_A_V2_RELEVANT_FUNCTIONS.csv"
    )

    formulas_path = (
        OUT
        / "R9H_A_V2_FORMULA_LINES.csv"
    )

    contexts_path = (
        OUT
        / "R9H_A_V2_FORMULA_CONTEXT.csv"
    )

    duplicates_path = (
        OUT
        / "R9H_A_V2_IDENTICAL_FUNCTION_BODY_GROUPS.csv"
    )

    inventory.to_csv(
        inventory_path,
        index=False,
    )

    functions.to_csv(
        functions_path,
        index=False,
    )

    formulas.to_csv(
        formulas_path,
        index=False,
    )

    contexts.to_csv(
        contexts_path,
        index=False,
    )

    duplicates.to_csv(
        duplicates_path,
        index=False,
    )

    status = (
        "R9H_A_V2_FOCUSED_PYTHON_CANONICALIZER_AUDIT_COMPLETE_READY_FOR_R9H_B_EXACT_GEOMETRY_FORMULA_LOCK"
    )

    manifest = {
        "step":
            "R9H_A_V2_FOCUSED_PYTHON_CANONICALIZER_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "json_pose_files_excluded":
            True,

        "new_training":
            False,

        "geometry_metrics_run":
            False,

        "bootstrap_performed":
            False,

        "frozen_results_modified":
            False,

        "candidate_count":
            int(
                len(
                    inventory
                )
            ),

        "relevant_function_count":
            int(
                len(
                    functions
                )
            ),

        "formula_line_count":
            int(
                len(
                    formulas
                )
            ),

        "identical_function_body_group_count":
            int(
                len(
                    duplicates
                )
            ),

        "source_inventory_csv":
            str(
                inventory_path
            ),

        "relevant_functions_csv":
            str(
                functions_path
            ),

        "formula_lines_csv":
            str(
                formulas_path
            ),

        "formula_context_csv":
            str(
                contexts_path
            ),

        "identical_function_groups_csv":
            str(
                duplicates_path
            ),
    }

    (
        OUT
        / "R9H_A_V2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "PYTHON SOURCE INVENTORY:"
    )

    print(
        inventory.to_string(
            index=False
        )
    )

    print(
        "\nRELEVANT CANONICALIZATION FUNCTIONS:"
    )

    if len(
        functions
    ):
        print(
            functions[
                [
                    "file",
                    "function",
                    "start_line",
                    "end_line",
                    "source_hash_ws_ignored",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No relevant functions found."
        )

    print(
        "\nFORMULA LINES:"
    )

    if len(
        formulas
    ):
        print(
            formulas.to_string(
                index=False
            )
        )
    else:
        print(
            "No formula lines found."
        )

    print(
        "\nIDENTICAL FUNCTION BODY GROUPS:"
    )

    if len(
        duplicates
    ):
        print(
            duplicates.to_string(
                index=False
            )
        )
    else:
        print(
            "No exact whitespace-insensitive duplicate function bodies found."
        )

    print(
        "\nEXACT RELEVANT FUNCTION SOURCE:"
    )

    if len(
        functions
    ):
        for _, row in functions.iterrows():
            print(
                "\n"
                +
                "=" * 100
            )

            print(
                f"{row['file']} :: {row['function']} "
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
        "\nR9-H-A-V2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
