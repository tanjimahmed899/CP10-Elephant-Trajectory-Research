from pathlib import Path
import ast
import hashlib
import json
import pandas as pd

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

STEP18A = ROOT / "step18a_lock_geofence_protocol.py"
STEP18B = ROOT / "step18b_evaluate_locked_geofence.py"

LOCK_JSON = (
    ROOT
    / "step18_geofence"
    / "STEP18_PROTOCOL_LOCKED.json"
)

LOCK_TXT = (
    ROOT
    / "step18_geofence"
    / "STEP18_PROTOCOL_LOCKED.txt"
)

EVAL_CSV = (
    ROOT
    / "step18_geofence"
    / "step18b_evaluation"
    / "step18b_all_geofence_instances.csv"
)

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_operational_warning"
    / "step18_geometry_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

KEYWORDS = [
    "boundary",
    "direction",
    "axis",
    "risk_score",
    "oriented_risk_score",
    "distance_to_boundary",
    "true_crossing",
    "predicted_crossing",
    "true_tte",
    "predicted_tte",
    "tte",
    "crossing",
    "geofence",
    "scale",
    "origin",
    "canonical",
    "absolute",
    "inside",
    "outside",
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


def get_function_infos(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(text)

    rows = []
    statements = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            continue

        src = ast.get_source_segment(
            text,
            node
        ) or ""

        low = src.lower()

        hits = [
            k
            for k in KEYWORDS
            if k.lower() in low
        ]

        if not hits:
            continue

        rows.append({
            "file":
                str(path),

            "function":
                node.name,

            "line_start":
                int(node.lineno),

            "line_end":
                int(
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno
                    )
                ),

            "score":
                int(len(set(hits))),

            "hits":
                ";".join(
                    sorted(
                        set(hits)
                    )
                ),

            "source":
                src,
        })

        for child in ast.walk(node):
            if not isinstance(
                child,
                (
                    ast.Assign,
                    ast.AnnAssign,
                    ast.AugAssign,
                    ast.If,
                    ast.For,
                    ast.Return,
                    ast.Expr,
                )
            ):
                continue

            seg = ast.get_source_segment(
                text,
                child
            )

            if not seg:
                continue

            seg_low = seg.lower()

            seg_hits = [
                k
                for k in KEYWORDS
                if k.lower() in seg_low
            ]

            if not seg_hits:
                continue

            statements.append({
                "file":
                    str(path),

                "function":
                    node.name,

                "line":
                    int(
                        getattr(
                            child,
                            "lineno",
                            -1
                        )
                    ),

                "hits":
                    ";".join(
                        sorted(
                            set(seg_hits)
                        )
                    ),

                "statement":
                    " ".join(
                        seg.strip().split()
                    ),
            })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(statements),
        text,
        tree,
    )


def top_level_assignments(path, text, tree):
    rows = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            continue

        if isinstance(node, ast.Assign):
            names = [
                t.id
                for t in node.targets
                if isinstance(
                    t,
                    ast.Name
                )
            ]
            value_node = node.value
        else:
            names = (
                [
                    node.target.id
                ]
                if isinstance(
                    node.target,
                    ast.Name
                )
                else []
            )
            value_node = node.value

        if not names:
            continue

        seg = ast.get_source_segment(
            text,
            node
        ) or ""

        low = seg.lower()

        if not any(
            k.lower() in low
            for k in KEYWORDS
        ):
            continue

        try:
            literal = ast.literal_eval(
                value_node
            )
        except Exception:
            literal = None

        rows.append({
            "file":
                str(path),

            "line":
                int(
                    getattr(
                        node,
                        "lineno",
                        -1
                    )
                ),

            "targets":
                ";".join(
                    names
                ),

            "literal_value":
                repr(
                    literal
                ),

            "source":
                " ".join(
                    seg.strip().split()
                ),
        })

    return pd.DataFrame(
        rows
    )


def lexical_contexts(path, text):
    lines = text.splitlines()

    rows = []

    for i, line in enumerate(
        lines,
        start=1
    ):
        low = line.lower()

        hits = [
            k
            for k in KEYWORDS
            if k.lower() in low
        ]

        if not hits:
            continue

        lo = max(
            1,
            i - 4
        )

        hi = min(
            len(lines),
            i + 4
        )

        context = "\n".join(
            f"{j:05d}: {lines[j-1]}"
            for j in range(
                lo,
                hi + 1
            )
        )

        rows.append({
            "file":
                str(path),

            "line":
                int(i),

            "hits":
                ";".join(
                    sorted(
                        set(hits)
                    )
                ),

            "text":
                line.strip(),

            "context":
                context,
        })

    return pd.DataFrame(
        rows
    )


def sample_schema():
    df = pd.read_csv(
        EVAL_CSV,
        nrows=500
    )

    return {
        "rows_sampled":
            int(len(df)),

        "columns":
            list(df.columns),

        "dtypes":
            {
                c: str(df[c].dtype)
                for c in df.columns
            },

        "head":
            df.head(5).to_dict(
                orient="records"
            ),
    }


def numeric_relation_audit():
    df = pd.read_csv(
        EVAL_CSV
    )

    result = {
        "rows":
            int(len(df)),

        "relations":
            [],
    }

    needed = [
        "scale",
        "risk_score",
        "oriented_risk_score",
        "distance_to_boundary",
    ]

    result["present"] = {
        c: c in df.columns
        for c in needed
    }

    if (
        "oriented_risk_score" in df.columns
        and
        "distance_to_boundary" in df.columns
    ):
        a = pd.to_numeric(
            df["oriented_risk_score"],
            errors="coerce"
        ).to_numpy(dtype=float)

        b = pd.to_numeric(
            df["distance_to_boundary"],
            errors="coerce"
        ).to_numpy(dtype=float)

        mask = (
            pd.notna(a)
            &
            pd.notna(b)
        )

        if mask.sum() > 0:
            same = (
                (a[mask] >= 0)
                ==
                (b[mask] >= 0)
            )

            opposite = (
                (a[mask] >= 0)
                ==
                (b[mask] <= 0)
            )

            result[
                "relations"
            ].append({
                "name":
                    "sign_agreement_oriented_risk_vs_distance",

                "same_sign_fraction":
                    float(
                        same.mean()
                    ),

                "opposite_sign_fraction":
                    float(
                        opposite.mean()
                    ),

                "n":
                    int(
                        mask.sum()
                    ),
            })

    if (
        "risk_score" in df.columns
        and
        "oriented_risk_score" in df.columns
    ):
        r = pd.to_numeric(
            df["risk_score"],
            errors="coerce"
        ).to_numpy(dtype=float)

        o = pd.to_numeric(
            df["oriented_risk_score"],
            errors="coerce"
        ).to_numpy(dtype=float)

        mask = (
            pd.notna(r)
            &
            pd.notna(o)
        )

        if mask.sum() > 0:
            diff = abs(r[mask]) - abs(o[mask])

            result[
                "relations"
            ].append({
                "name":
                    "abs_risk_vs_oriented_risk",

                "median_abs_difference":
                    float(
                        pd.Series(
                            diff
                        )
                        .abs()
                        .median()
                    ),

                "max_abs_difference":
                    float(
                        pd.Series(
                            diff
                        )
                        .abs()
                        .max()
                    ),

                "n":
                    int(
                        mask.sum()
                    ),
            })

    return result


def main():
    print(
        "\nSTEP 6D-K3C3 — EXACT STEP18 GEOFENCE GEOMETRY / ORIENTATION / TTE AUDIT"
    )

    print(
        "Read-only audit. NO BAMBI geofence instances or alarms are generated.\n"
    )

    for p in [
        STEP18A,
        STEP18B,
        LOCK_JSON,
        LOCK_TXT,
        EVAL_CSV,
    ]:
        require(
            p
        )

    fn_frames = []
    st_frames = []
    assign_frames = []
    lex_frames = []

    for path in [
        STEP18A,
        STEP18B,
    ]:
        (
            fn_df,
            st_df,
            text,
            tree,
        ) = get_function_infos(
            path
        )

        fn_frames.append(
            fn_df
        )

        st_frames.append(
            st_df
        )

        assign_frames.append(
            top_level_assignments(
                path,
                text,
                tree
            )
        )

        lex_frames.append(
            lexical_contexts(
                path,
                text
            )
        )

    functions = pd.concat(
        fn_frames,
        ignore_index=True
    )

    statements = pd.concat(
        st_frames,
        ignore_index=True
    )

    assignments = pd.concat(
        assign_frames,
        ignore_index=True
    )

    lexical = pd.concat(
        lex_frames,
        ignore_index=True
    )

    if len(
        functions
    ):
        functions = functions.sort_values(
            [
                "score",
                "file",
                "line_start",
            ],
            ascending=[
                False,
                True,
                True,
            ]
        )

    functions.to_csv(
        OUT
        / "TABLE_step6dk3c3_ranked_step18_functions.csv",
        index=False
    )

    statements.to_csv(
        OUT
        / "TABLE_step6dk3c3_step18_formula_statements.csv",
        index=False
    )

    assignments.to_csv(
        OUT
        / "TABLE_step6dk3c3_step18_top_level_assignments.csv",
        index=False
    )

    lexical.to_csv(
        OUT
        / "step6dk3c3_step18_lexical_contexts.csv",
        index=False
    )

    lock = json.loads(
        LOCK_JSON.read_text(
            encoding="utf-8"
        )
    )

    schema = sample_schema()

    relation = numeric_relation_audit()

    with open(
        OUT
        / "STEP6D_K3C3_STEP18_PROTOCOL_LOCK_COPY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            lock,
            f,
            indent=2
        )

    with open(
        OUT
        / "STEP6D_K3C3_STEP18_INSTANCE_SCHEMA.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            schema,
            f,
            indent=2
        )

    with open(
        OUT
        / "STEP6D_K3C3_NUMERIC_RELATION_AUDIT.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            relation,
            f,
            indent=2
        )

    with open(
        OUT
        / "STEP6D_K3C3_EXACT_STEP18_SOURCE_BUNDLE.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            f"STEP18A: {STEP18A}\n"
            f"SHA256: {sha256_file(STEP18A)}\n\n"
            f"STEP18B: {STEP18B}\n"
            f"SHA256: {sha256_file(STEP18B)}\n\n"
        )

        for _, row in functions.iterrows():
            f.write("=" * 100 + "\n")
            f.write(
                f"FILE: {row['file']}\n"
                f"FUNCTION: {row['function']}\n"
                f"LINES: {row['line_start']}-{row['line_end']}\n"
                f"HITS: {row['hits']}\n"
            )
            f.write("=" * 100 + "\n")
            f.write(row["source"])
            f.write("\n\n")

    print("STEP18 LOCKED PROTOCOL:")
    print(
        json.dumps(
            lock,
            indent=2
        )
    )

    print("\nTOP STEP18 FUNCTIONS:")
    print(
        functions.head(30)[
            [
                "file",
                "function",
                "line_start",
                "line_end",
                "score",
                "hits",
            ]
        ].to_string(index=False)
        if len(functions)
        else "NONE"
    )

    print("\nKEY STEP18 FORMULA STATEMENTS:")
    print(
        statements.head(120).to_string(index=False)
        if len(statements)
        else "NONE"
    )

    print("\nSTEP18 INSTANCE SCHEMA:")
    print(
        json.dumps(
            schema,
            indent=2
        )[:12000]
    )

    print("\nNUMERIC ORIENTATION/UNIT RELATION AUDIT:")
    print(
        json.dumps(
            relation,
            indent=2
        )
    )

    summary = {
        "step":
            "STEP_6D_K3C3_EXACT_STEP18_GEOFENCE_GEOMETRY_ORIENTATION_TTE_AUDIT",

        "step18a_sha256":
            sha256_file(
                STEP18A
            ),

        "step18b_sha256":
            sha256_file(
                STEP18B
            ),

        "protocol_lock_sha256":
            sha256_file(
                LOCK_JSON
            ),

        "geofence_instance_table_sha256":
            sha256_file(
                EVAL_CSV
            ),

        "candidate_functions":
            int(
                len(
                    functions
                )
            ),

        "formula_statements":
            int(
                len(
                    statements
                )
            ),

        "status":
            "EXACT_STEP18_GEOMETRY_FORMULAS_EXTRACTED_READY_FOR_BAMBI_GEOFENCE_RECONSTRUCTION_DECISION",

        "important_rule": [
            "Do not infer orientation or units from sign agreement alone.",
            "BAMBI base geofence reconstruction may proceed only after source code confirms boundary construction, risk orientation, crossing, TTE, and scale mapping.",
            "No original Main Result file is modified by this audit.",
        ],
    }

    (
        OUT
        / "STEP6D_K3C3_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSTATUS:")
    print(
        summary[
            "status"
        ]
    )

    print("\nSTEP 6D-K3C3 COMPLETE")
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
