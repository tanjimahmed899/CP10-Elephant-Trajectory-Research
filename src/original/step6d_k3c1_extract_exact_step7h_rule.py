from pathlib import Path
import ast
import hashlib
import json
import re
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")

STEP7H_SRC = (
    DOWNLOADS
    / "step7h_final_response_aware_with_locked_persistence.py"
)

STEP7_MANIFEST = (
    DOWNLOADS
    / "step7_response_aware_warning"
    / "STEP7_FINAL_DEVELOPMENT_MANIFEST.json"
)

STEP7A_LOCK = (
    DOWNLOADS
    / "step7_response_aware_warning"
    / "STEP7A_PROTOCOL_LOCK.json"
)

K3B1_DIR = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_fixed_conformal"
    / "k3b1_v2_fixed_no_recalibration_animal_track_unit"
)

BAMBI_META_DIR = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k2b3_final_canonical"
)

BAMBI_INFERENCE_DIR = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_frozen_ensemble"
    / "first_inference"
)

OUT = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_operational_warning"
    / "exact_rule_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGET_FUNCTIONS = [
    "evaluate",
    "main",
]

KEYWORDS = [
    "red",
    "yellow",
    "green",
    "response",
    "threshold",
    "geofence",
    "boundary",
    "cross",
    "crossing",
    "tte",
    "time_to",
    "persistence",
    "n_persist",
    "conformal",
    "q90",
    "radius",
    "inside",
    "intersect",
    "track",
    "sample",
    "window",
    "scale",
    "direction",
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
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == name
    ]

    if len(matches) != 1:
        return None

    node = matches[0]
    src = ast.get_source_segment(text, node) or ""

    return {
        "name": name,
        "line_start": int(node.lineno),
        "line_end": int(getattr(node, "end_lineno", node.lineno)),
        "node": node,
        "source": src,
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

        seg = ast.get_source_segment(full_text, node)

        if not seg:
            continue

        low = seg.lower()

        hits = [
            k for k in KEYWORDS
            if k in low
        ]

        if not hits:
            continue

        rows.append({
            "function": info["name"],
            "line": int(getattr(node, "lineno", -1)),
            "hits": ";".join(sorted(set(hits))),
            "statement": " ".join(seg.strip().split()),
        })

    rows.sort(key=lambda x: (x["function"], x["line"]))
    return rows


def literal_assignments(text, tree):
    rows = []

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        if isinstance(node, ast.Assign):
            targets = [
                t.id for t in node.targets
                if isinstance(t, ast.Name)
            ]
            value_node = node.value
        else:
            targets = (
                [node.target.id]
                if isinstance(node.target, ast.Name)
                else []
            )
            value_node = node.value

        if not targets:
            continue

        try:
            value = ast.literal_eval(value_node)
        except Exception:
            value = None

        seg = ast.get_source_segment(text, node) or ""

        low = seg.lower()

        if any(k in low for k in KEYWORDS):
            rows.append({
                "line": int(getattr(node, "lineno", -1)),
                "targets": ";".join(targets),
                "literal_value": repr(value),
                "source": " ".join(seg.strip().split()),
            })

    return rows


def input_schema_inventory():
    rows = []

    # BAMBI canonical metadata.
    for task in ["2s_to_2s", "2s_to_5s"]:
        meta_path = (
            BAMBI_META_DIR
            / f"{task}_bambi_flight6_canonical_metadata_LOCKED.csv"
        )
        require(meta_path)

        df = pd.read_csv(meta_path)

        rows.append({
            "task": task,
            "kind": "canonical_metadata",
            "file": str(meta_path),
            "rows": int(len(df)),
            "columns": ";".join(df.columns.astype(str).tolist()),
        })

        # Forecast/inference error table.
        err_path = (
            BAMBI_INFERENCE_DIR
            / f"{task}_ensemble_per_window_errors.csv"
        )
        require(err_path)

        edf = pd.read_csv(err_path)

        rows.append({
            "task": task,
            "kind": "forecast_error_table",
            "file": str(err_path),
            "rows": int(len(edf)),
            "columns": ";".join(edf.columns.astype(str).tolist()),
        })

    return pd.DataFrame(rows)


def main():
    print(
        "\nSTEP 6D-K3C1 — EXACT FROZEN STEP7H RULE / INPUT COMPATIBILITY EXTRACTION"
    )
    print(
        "Read-only audit. NO BAMBI Step7 warning evaluation is computed.\n"
    )

    for p in [
        STEP7H_SRC,
        STEP7_MANIFEST,
        STEP7A_LOCK,
        K3B1_DIR,
        BAMBI_META_DIR,
        BAMBI_INFERENCE_DIR,
    ]:
        require(p)

    text = STEP7H_SRC.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(text)

    infos = []
    statements = []

    for name in TARGET_FUNCTIONS:
        info = get_function_source(
            text,
            tree,
            name
        )

        if info is not None:
            infos.append(info)
            statements.extend(
                relevant_statements(
                    info,
                    text
                )
            )

    literals = literal_assignments(
        text,
        tree
    )

    # Exact source bundle.
    with open(
        OUT / "STEP6D_K3C1_EXACT_STEP7H_SOURCES.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            f"SOURCE: {STEP7H_SRC}\n"
            f"SHA256: {sha256_file(STEP7H_SRC)}\n\n"
        )

        for info in infos:
            f.write("=" * 100 + "\n")
            f.write(
                f"FUNCTION: {info['name']}\n"
                f"LINES: {info['line_start']}-{info['line_end']}\n"
            )
            f.write("=" * 100 + "\n")
            f.write(info["source"])
            f.write("\n\n")

    stmt_df = pd.DataFrame(statements)
    lit_df = pd.DataFrame(literals)

    stmt_df.to_csv(
        OUT / "TABLE_step6dk3c1_step7h_relevant_statements.csv",
        index=False
    )

    lit_df.to_csv(
        OUT / "TABLE_step6dk3c1_step7h_literals.csv",
        index=False
    )

    # Search exact strings/expressions likely defining the rule.
    lines = text.splitlines()

    patterns = [
        "RED",
        "YELLOW",
        "GREEN",
        "response",
        "threshold",
        "persistence",
        "N_PERSIST",
        "q90",
        "conformal",
        "geofence",
        "boundary",
        "crossing",
        "tte",
        "inside",
        "intersect",
    ]

    lexical_rows = []

    for i, line in enumerate(lines, start=1):
        low = line.lower()

        hits = [
            p for p in patterns
            if p.lower() in low
        ]

        if not hits:
            continue

        lo = max(1, i - 3)
        hi = min(len(lines), i + 3)

        context = "\n".join(
            f"{j:05d}: {lines[j-1]}"
            for j in range(lo, hi + 1)
        )

        lexical_rows.append({
            "line": int(i),
            "hits": ";".join(hits),
            "text": line.strip(),
            "context": context,
        })

    lexical_df = pd.DataFrame(
        lexical_rows
    )

    lexical_df.to_csv(
        OUT / "step6dk3c1_step7h_lexical_contexts.csv",
        index=False
    )

    # Existing final manifest / protocol lock.
    manifest = json.loads(
        STEP7_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    protocol = json.loads(
        STEP7A_LOCK.read_text(
            encoding="utf-8"
        )
    )

    with open(
        OUT / "STEP6D_K3C1_STEP7_FINAL_MANIFEST_COPY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2
        )

    with open(
        OUT / "STEP6D_K3C1_STEP7A_PROTOCOL_LOCK_COPY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            protocol,
            f,
            indent=2
        )

    schema_df = input_schema_inventory()

    schema_df.to_csv(
        OUT / "TABLE_step6dk3c1_bambi_input_schema.csv",
        index=False
    )

    # Compact compatibility facts from actual BAMBI metadata.
    compat_rows = []

    for task in ["2s_to_2s", "2s_to_5s"]:
        meta = pd.read_csv(
            BAMBI_META_DIR
            / f"{task}_bambi_flight6_canonical_metadata_LOCKED.csv"
        )

        compat_rows.append({
            "task":
                task,

            "rows":
                int(len(meta)),

            "tracks":
                int(meta["track_id"].nunique())
                if "track_id" in meta.columns
                else None,

            "has_track_id":
                bool("track_id" in meta.columns),

            "has_window_start_time_sec":
                bool("window_start_time_sec" in meta.columns),

            "has_window_end_time_sec":
                bool("window_end_time_sec" in meta.columns),

            "has_scale":
                bool("scale" in meta.columns),

            "has_heading_angle":
                bool("heading_angle_rad" in meta.columns),

            "has_origin_xy":
                bool(
                    "origin_x" in meta.columns
                    and
                    "origin_y" in meta.columns
                ),
        })

    compat_df = pd.DataFrame(
        compat_rows
    )

    compat_df.to_csv(
        OUT / "TABLE_step6dk3c1_bambi_core_compatibility.csv",
        index=False
    )

    print(
        "FINAL LOCKED MANIFEST:"
    )
    print(
        json.dumps(
            manifest,
            indent=2
        )
    )

    print(
        "\nKEY STEP7H SOURCE STATEMENTS:"
    )

    if len(stmt_df):
        print(
            stmt_df.to_string(
                index=False
            )
        )
    else:
        print("NONE")

    print(
        "\nSTEP7H LITERAL/CONFIG ASSIGNMENTS:"
    )

    if len(lit_df):
        print(
            lit_df.to_string(
                index=False
            )
        )
    else:
        print("NONE")

    print(
        "\nBAMBI CORE COMPATIBILITY:"
    )

    print(
        compat_df.to_string(
            index=False
        )
    )

    # We deliberately do not auto-declare compatibility based only on
    # keyword presence. The next step should inspect exact statements.
    summary = {
        "step":
            "STEP_6D_K3C1_EXACT_FROZEN_STEP7H_RULE_INPUT_COMPATIBILITY_EXTRACTION",

        "step7h_source":
            str(STEP7H_SRC),

        "step7h_source_sha256":
            sha256_file(STEP7H_SRC),

        "step7_manifest_sha256":
            sha256_file(STEP7_MANIFEST),

        "step7a_protocol_lock_sha256":
            sha256_file(STEP7A_LOCK),

        "relevant_statement_count":
            int(len(stmt_df)),

        "literal_assignment_count":
            int(len(lit_df)),

        "status":
            "EXACT_STEP7H_RULE_EXTRACTED_READY_FOR_FINAL_BAMBI_COMPATIBILITY_DECISION",

        "nonclaims": [
            "No BAMBI Step7 warning evaluation is performed here.",
            "No Step7H threshold, persistence rule, geofence definition, or response assumption is modified.",
            "N=2 remains two consecutive available forecast windows, not seconds.",
            "The virtual geofence remains a normalized/virtual scenario, not a real village/farm boundary.",
            "Response thresholds remain assumed sensitivity-analysis values, not measured field response times.",
        ],
    }

    (
        OUT / "STEP6D_K3C1_SUMMARY.json"
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
        summary["status"]
    )

    print(
        "\nSTEP 6D-K3C1 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
