from pathlib import Path
import ast
import json
import re
import hashlib
import numpy as np
import pandas as pd

STEP6C_SRC = Path(
    r"C:\Users\HP\Downloads\step6c_crossfit_conformal.py"
)

STEP6C_DIR = Path(
    r"C:\Users\HP\Downloads\step6c_crossfit_conformal"
)

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_fixed_conformal"
    / "source_audit_v3"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGET_FUNCTIONS = [
    "conformal_quantile",
    "build_track_calibration",
    "bootstrap_track_metrics",
    "evaluate_dataset",
    "main",
]

TARGET_FILES = [
    STEP6C_DIR / "track_conformal_calibration_summary.csv",
    STEP6C_DIR / "crossfit_track_calibration_scores.csv",
    STEP6C_DIR / "crossfit_oof_sample_scores.csv",
    STEP6C_DIR / "track_conformal_retrospective_evaluation.csv",
    STEP6C_DIR / "track_conformal_timewise_evaluation.csv",
    STEP6C_DIR / "track_conformal_failure_cases.csv",
    STEP6C_DIR / "track_conformal_stepwise_radii.csv",
    STEP6C_DIR / "SUMMARY.txt",
]

KEYWORDS = [
    "score",
    "radial",
    "error",
    "max",
    "endpoint",
    "q90",
    "q95",
    "quantile",
    "scale",
    "covered",
    "coverage",
    "fully_covered",
    "track_macro",
    "track",
    "groupby",
    "alpha",
    "radius",
]


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get_function_source(text, tree, name):
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
        return None

    node = matches[0]

    src = ast.get_source_segment(
        text,
        node
    ) or ""

    return {
        "name": name,
        "line_start": int(node.lineno),
        "line_end": int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        ),
        "source": src,
        "node": node,
    }


def relevant_statements(info, full_text):
    rows = []

    if info is None:
        return rows

    for node in ast.walk(info["node"]):
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.If,
                ast.Return,
            )
        ):
            continue

        seg = ast.get_source_segment(
            full_text,
            node
        )

        if not seg:
            continue

        low = seg.lower()

        if not any(
            k in low
            for k in KEYWORDS
        ):
            continue

        rows.append({
            "function": info["name"],
            "line": int(
                getattr(
                    node,
                    "lineno",
                    -1
                )
            ),
            "statement": " ".join(
                seg.strip().split()
            ),
        })

    rows.sort(
        key=lambda x: (
            x["function"],
            x["line"]
        )
    )

    return rows


def inspect_csv(path):
    df = pd.read_csv(path)

    return {
        "file": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "head": df.head(5).to_dict(
            orient="records"
        ),
    }


def find_q_columns(df):
    cols = []

    for c in df.columns:
        low = c.lower()

        if any(
            k in low
            for k in [
                "q",
                "radius",
                "scale",
                "score",
                "coverage",
                "covered",
                "endpoint",
                "alpha",
            ]
        ):
            cols.append(c)

    return cols


def numeric_relation_audit():
    """
    Use the retrospective failure-case table to test the hypothesized
    conversion:
        world_radius = canonical_q * sample_scale
    without computing any BAMBI coverage.
    """
    path = STEP6C_DIR / "track_conformal_failure_cases.csv"
    df = pd.read_csv(path)

    result = {
        "file": str(path),
        "columns": list(df.columns),
        "candidate_relations": [],
    }

    numeric_cols = []

    for c in df.columns:
        s = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        if s.notna().mean() > 0.95:
            numeric_cols.append(c)

    # Search every triple A*B ≈ C across numeric columns.
    best = []

    for a in numeric_cols:
        av = pd.to_numeric(
            df[a],
            errors="coerce"
        ).to_numpy(dtype=float)

        for b in numeric_cols:
            if b == a:
                continue

            bv = pd.to_numeric(
                df[b],
                errors="coerce"
            ).to_numpy(dtype=float)

            prod = av * bv

            for c in numeric_cols:
                if c in {a, b}:
                    continue

                cv = pd.to_numeric(
                    df[c],
                    errors="coerce"
                ).to_numpy(dtype=float)

                mask = (
                    np.isfinite(prod)
                    &
                    np.isfinite(cv)
                )

                if mask.sum() < 5:
                    continue

                abs_diff = np.abs(
                    prod[mask]
                    -
                    cv[mask]
                )

                med = float(
                    np.median(abs_diff)
                )

                mx = float(
                    np.max(abs_diff)
                )

                if med < 1e-8:
                    best.append({
                        "A": a,
                        "B": b,
                        "C": c,
                        "median_abs_diff": med,
                        "max_abs_diff": mx,
                    })

    best = sorted(
        best,
        key=lambda x: (
            x["median_abs_diff"],
            x["max_abs_diff"]
        )
    )

    result[
        "candidate_relations"
    ] = best[:30]

    return result


def main():
    print(
        "\nSTEP 6D-K3B0-V3 — EXACT STEP6C FORMULA / UNIT / COVERAGE EXTRACTION"
    )

    print(
        "Read-only audit. No BAMBI conformal coverage is computed.\n"
    )

    require(
        STEP6C_SRC
    )

    for p in TARGET_FILES:
        require(
            p
        )

    text = STEP6C_SRC.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text
    )

    function_infos = []
    statement_rows = []

    for name in TARGET_FUNCTIONS:
        info = get_function_source(
            text,
            tree,
            name
        )

        if info is None:
            continue

        function_infos.append(
            info
        )

        statement_rows.extend(
            relevant_statements(
                info,
                text
            )
        )

    # Write exact function source bundle.
    with open(
        OUT / "STEP6D_K3B0_V3_EXACT_FUNCTION_SOURCES.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            f"SOURCE: {STEP6C_SRC}\n"
            f"SHA256: {sha256_file(STEP6C_SRC)}\n\n"
        )

        for info in function_infos:
            f.write(
                "=" * 100
                +
                "\n"
            )

            f.write(
                f"FUNCTION: {info['name']}\n"
                f"LINES: {info['line_start']}-{info['line_end']}\n"
            )

            f.write(
                "=" * 100
                +
                "\n"
            )

            f.write(
                info["source"]
            )

            f.write(
                "\n\n"
            )

    statements_df = pd.DataFrame(
        statement_rows
    )

    statements_df.to_csv(
        OUT / "TABLE_step6dk3b0v3_relevant_formula_statements.csv",
        index=False
    )

    # Inventory exact CSV schemas.
    csv_inventory = []

    for path in TARGET_FILES:
        if path.suffix.lower() == ".csv":
            csv_inventory.append(
                inspect_csv(
                    path
                )
            )

    with open(
        OUT / "STEP6D_K3B0_V3_CSV_SCHEMA_INVENTORY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            csv_inventory,
            f,
            indent=2
        )

    schema_rows = []

    for item in csv_inventory:
        for idx, c in enumerate(
            item[
                "columns"
            ]
        ):
            schema_rows.append({
                "file": item["file"],
                "column_index": idx,
                "column": c,
            })

    pd.DataFrame(
        schema_rows
    ).to_csv(
        OUT / "TABLE_step6dk3b0v3_csv_columns.csv",
        index=False
    )

    # Pull exact calibration summary.
    cal = pd.read_csv(
        STEP6C_DIR
        / "track_conformal_calibration_summary.csv"
    )

    retrospective = pd.read_csv(
        STEP6C_DIR
        / "track_conformal_retrospective_evaluation.csv"
    )

    timewise = pd.read_csv(
        STEP6C_DIR
        / "track_conformal_timewise_evaluation.csv"
    )

    relation_audit = numeric_relation_audit()

    with open(
        OUT / "STEP6D_K3B0_V3_NUMERIC_RELATION_AUDIT.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            relation_audit,
            f,
            indent=2
        )

    # Extract source statements most likely to answer the five key questions.
    key_statement_patterns = [
        "np.linalg.norm",
        ".max(",
        "groupby",
        "conformal_quantile",
        "scale",
        "radius",
        "covered",
        "fully_covered",
        "track_macro",
        "endpoint",
    ]

    key_rows = []

    for row in statement_rows:
        low = row[
            "statement"
        ].lower()

        if any(
            p.lower() in low
            for p in key_statement_patterns
        ):
            key_rows.append(
                row
            )

    key_df = pd.DataFrame(
        key_rows
    )

    key_df.to_csv(
        OUT / "TABLE_step6dk3b0v3_key_formula_statements.csv",
        index=False
    )

    # Print compact information.
    print(
        "CALIBRATION SUMMARY:"
    )

    print(
        cal.to_string(
            index=False
        )
    )

    print(
        "\nRETROSPECTIVE EVALUATION COLUMNS:"
    )

    print(
        list(
            retrospective.columns
        )
    )

    print(
        "\nTIMEWISE EVALUATION COLUMNS:"
    )

    print(
        list(
            timewise.columns
        )
    )

    print(
        "\nKEY SOURCE FORMULA STATEMENTS:"
    )

    if len(
        key_df
    ):
        print(
            key_df.to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nNUMERIC RELATION AUDIT:"
    )

    print(
        json.dumps(
            relation_audit[
                "candidate_relations"
            ][
                :15
            ],
            indent=2
        )
    )

    # Status stays pending human/source interpretation;
    # this script deliberately does not decide coverage itself.
    summary = {
        "step":
            "STEP_6D_K3B0_V3_EXACT_STEP6C_FORMULA_UNIT_COVERAGE_EXTRACTION",

        "step6c_source":
            str(
                STEP6C_SRC
            ),

        "step6c_source_sha256":
            sha256_file(
                STEP6C_SRC
            ),

        "calibration_rows":
            int(
                len(
                    cal
                )
            ),

        "retrospective_rows":
            int(
                len(
                    retrospective
                )
            ),

        "timewise_rows":
            int(
                len(
                    timewise
                )
            ),

        "formula_statement_count":
            int(
                len(
                    key_df
                )
            ),

        "status":
            "EXACT_STEP6C_FORMULAS_EXTRACTED_READY_FOR_UNIT_AND_COVERAGE_CONFIRMATION",
    }

    (
        OUT
        / "STEP6D_K3B0_V3_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nSTATUS:"
    )

    print(
        summary[
            "status"
        ]
    )

    print(
        "\nSTEP 6D-K3B0-V3 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
