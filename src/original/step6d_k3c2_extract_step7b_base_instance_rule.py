from pathlib import Path
import ast
import hashlib
import json
import re
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")

STEP7B_SRC = (
    DOWNLOADS
    / "step7b_v2_response_aware_green_yellow_red.py"
)

STEP7A_LOCK = (
    DOWNLOADS
    / "step7_response_aware_warning"
    / "STEP7A_PROTOCOL_LOCK.json"
)

STEP7H_SRC = (
    DOWNLOADS
    / "step7h_final_response_aware_with_locked_persistence.py"
)

STEP7H_MANIFEST = (
    DOWNLOADS
    / "step7_response_aware_warning"
    / "STEP7_FINAL_DEVELOPMENT_MANIFEST.json"
)

BAMBI_BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

BAMBI_CANONICAL = (
    BAMBI_BASE
    / "k2b3_final_canonical"
)

BAMBI_INFERENCE = (
    BAMBI_BASE
    / "k3_frozen_ensemble"
    / "first_inference"
)

K3B1 = (
    BAMBI_BASE
    / "k3_fixed_conformal"
    / "k3b1_v2_fixed_no_recalibration_animal_track_unit"
)

OUT = (
    BAMBI_BASE
    / "k3_operational_warning"
    / "base_instance_rule_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_FUNCTIONS = [
    "classify_instances",
    "aggregate_sample_level",
    "metric_row",
    "track_macro",
    "main",
]

KEYWORDS = [
    "geofence",
    "boundary",
    "axis",
    "direction",
    "response",
    "threshold",
    "crossing",
    "predicted_crossing",
    "true_crossing",
    "tte",
    "time_to",
    "distance_to_boundary",
    "uncertainty_intersects_geofence",
    "conformal",
    "q90",
    "radius",
    "scale",
    "alpha",
    "inside",
    "intersect",
    "scenario",
    "offset",
    "x",
    "y",
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
        "node": node,
        "source": ast.get_source_segment(
            text,
            node
        ) or "",
    }


def relevant_statements(info, full_text):
    rows = []

    if info is None:
        return rows

    allowed = (
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.If,
        ast.For,
        ast.Return,
        ast.Expr,
    )

    for node in ast.walk(info["node"]):
        if not isinstance(node, allowed):
            continue

        seg = ast.get_source_segment(
            full_text,
            node
        )

        if not seg:
            continue

        low = seg.lower()

        hits = [
            k
            for k in KEYWORDS
            if k.lower() in low
        ]

        if not hits:
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
            "hits": ";".join(
                sorted(
                    set(
                        hits
                    )
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


def top_level_assignments(text, tree):
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
            targets = [
                t.id
                for t in node.targets
                if isinstance(
                    t,
                    ast.Name
                )
            ]
            value_node = node.value
        else:
            targets = (
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

        if not targets:
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
                    targets
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

    return rows


def exact_string_contexts(text):
    lines = text.splitlines()

    patterns = [
        "response_times",
        "response_time",
        "boundaries",
        "boundary",
        "axis",
        "direction",
        "q90",
        "conformal",
        "predicted_crossing_bool",
        "true_crossing_bool",
        "uncertainty_intersects_geofence",
        "predicted_tte_sec",
        "true_tte_sec",
        "distance_to_boundary",
        "scale",
    ]

    rows = []

    for i, line in enumerate(
        lines,
        start=1
    ):
        low = line.lower()

        hits = [
            p
            for p in patterns
            if p.lower() in low
        ]

        if not hits:
            continue

        lo = max(
            1,
            i - 4
        )

        hi = min(
            len(
                lines
            ),
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
            "line":
                int(
                    i
                ),

            "hits":
                ";".join(
                    hits
                ),

            "text":
                line.strip(),

            "context":
                context,
        })

    return rows


def bambi_schema():
    rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        meta_path = (
            BAMBI_CANONICAL
            / f"{task}_bambi_flight6_canonical_metadata_LOCKED.csv"
        )

        pred_path = (
            BAMBI_INFERENCE
            / f"{task}_BAMBI_Flight6_first_locked_inference.npz"
        )

        require(meta_path)
        require(pred_path)

        meta = pd.read_csv(
            meta_path
        )

        rows.append({
            "task":
                task,

            "metadata_rows":
                int(
                    len(
                        meta
                    )
                ),

            "tracks":
                int(
                    meta[
                        "track_id"
                    ].nunique()
                )
                if
                "track_id"
                in
                meta.columns
                else None,

            "metadata_columns":
                ";".join(
                    meta.columns.tolist()
                ),
        })

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nSTEP 6D-K3C2 — EXACT STEP7B-V2 BASE INSTANCE / GEOFENCE RULE EXTRACTION"
    )

    print(
        "Read-only audit. NO BAMBI alarm evaluation is computed.\n"
    )

    for p in [
        STEP7B_SRC,
        STEP7A_LOCK,
        STEP7H_SRC,
        STEP7H_MANIFEST,
        BAMBI_CANONICAL,
        BAMBI_INFERENCE,
        K3B1,
    ]:
        require(
            p
        )

    text = STEP7B_SRC.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text
    )

    infos = []
    statement_rows = []

    for name in TARGET_FUNCTIONS:
        info = get_function_source(
            text,
            tree,
            name
        )

        if info is not None:
            infos.append(
                info
            )

            statement_rows.extend(
                relevant_statements(
                    info,
                    text
                )
            )

    assignments = top_level_assignments(
        text,
        tree
    )

    lexical = exact_string_contexts(
        text
    )

    stmt_df = pd.DataFrame(
        statement_rows
    )

    assign_df = pd.DataFrame(
        assignments
    )

    lex_df = pd.DataFrame(
        lexical
    )

    schema_df = bambi_schema()

    stmt_df.to_csv(
        OUT
        / "TABLE_step6dk3c2_step7b_exact_statements.csv",
        index=False
    )

    assign_df.to_csv(
        OUT
        / "TABLE_step6dk3c2_step7b_top_level_assignments.csv",
        index=False
    )

    lex_df.to_csv(
        OUT
        / "step6dk3c2_step7b_lexical_contexts.csv",
        index=False
    )

    schema_df.to_csv(
        OUT
        / "TABLE_step6dk3c2_bambi_schema.csv",
        index=False
    )

    with open(
        OUT
        / "STEP6D_K3C2_EXACT_STEP7B_FUNCTIONS.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            f"SOURCE: {STEP7B_SRC}\n"
            f"SHA256: {sha256_file(STEP7B_SRC)}\n\n"
        )

        for info in infos:
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
                info[
                    "source"
                ]
            )

            f.write(
                "\n\n"
            )

    protocol = json.loads(
        STEP7A_LOCK.read_text(
            encoding="utf-8"
        )
    )

    manifest = json.loads(
        STEP7H_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    with open(
        OUT
        / "STEP6D_K3C2_STEP7A_PROTOCOL_LOCK_COPY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            protocol,
            f,
            indent=2
        )

    with open(
        OUT
        / "STEP6D_K3C2_STEP7H_MANIFEST_COPY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2
        )

    print(
        "STEP7A PROTOCOL LOCK:"
    )

    print(
        json.dumps(
            protocol,
            indent=2
        )
    )

    print(
        "\nKEY STEP7B-V2 SOURCE STATEMENTS:"
    )

    if len(
        stmt_df
    ):
        print(
            stmt_df.to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nSTEP7B-V2 TOP-LEVEL ASSIGNMENTS:"
    )

    if len(
        assign_df
    ):
        print(
            assign_df.to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nBAMBI INPUT SCHEMA:"
    )

    print(
        schema_df.to_string(
            index=False
        )
    )

    summary = {
        "step":
            "STEP_6D_K3C2_EXACT_STEP7B_V2_BASE_INSTANCE_RULE_EXTRACTION",

        "step7b_source":
            str(
                STEP7B_SRC
            ),

        "step7b_source_sha256":
            sha256_file(
                STEP7B_SRC
            ),

        "step7a_protocol_lock_sha256":
            sha256_file(
                STEP7A_LOCK
            ),

        "step7h_manifest_sha256":
            sha256_file(
                STEP7H_MANIFEST
            ),

        "statement_count":
            int(
                len(
                    stmt_df
                )
            ),

        "top_level_assignment_count":
            int(
                len(
                    assign_df
                )
            ),

        "status":
            "EXACT_STEP7B_BASE_INSTANCE_RULE_EXTRACTED_READY_FOR_BAMBI_RECONSTRUCTION_DECISION",

        "rules": [
            "No BAMBI alarms are generated in K3C2.",
            "No virtual-geofence definition, response-time assumption, conformal level, or persistence rule is modified.",
            "The next step may reconstruct BAMBI Step7 base instances only if all required formulas and scenario parameters are source-confirmed.",
        ],
    }

    (
        OUT
        / "STEP6D_K3C2_SUMMARY.json"
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
        "\nSTEP 6D-K3C2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
