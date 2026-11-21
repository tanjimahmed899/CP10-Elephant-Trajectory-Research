from pathlib import Path
import ast
import hashlib
import json
import re
import pandas as pd

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

WILDBOX_BUILD = ROOT / "build_wildbox_trajectories.py"
FRESH_EVAL = ROOT / "evaluate_fresh_wildbox_holdout.py"

BAMBI_REPO = (
    Path(r"C:\Users\HP\Downloads")
    / "Dataset-main"
    / "Dataset-main"
)

BAMBI_GT = BAMBI_REPO / "6_gt.txt"
BAMBI_META = BAMBI_REPO / "6_metadata.json"
BAMBI_POSES = BAMBI_REPO / "6_matched_poses.json"

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_operational_warning"
    / "exact_normalized_mapping_confirmation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

KEYWORDS = [
    "x_norm",
    "y_norm",
    "u",
    "v",
    "width",
    "height",
    "image_width",
    "image_height",
    "anchor_camera",
    "corrected",
    "transform",
    "homography",
    "project",
    "projection",
    "camera",
    "pixel",
    "center",
    "bbox",
    "normalize",
    "normalized",
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


def read_text(path):
    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def collect_functions(path):
    text = read_text(path)
    tree = ast.parse(text)

    rows = []

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

        score = len(
            set(hits)
        )

        if "x_norm" in low:
            score += 8
        if "y_norm" in low:
            score += 8
        if "anchor_camera" in low:
            score += 6
        if "corrected" in low:
            score += 5
        if "image_width" in low:
            score += 4
        if "image_height" in low:
            score += 4
        if "homography" in low:
            score += 5
        if "project" in low:
            score += 4

        rows.append({
            "file": str(path),
            "function": node.name,
            "line_start": int(node.lineno),
            "line_end": int(
                getattr(
                    node,
                    "end_lineno",
                    node.lineno
                )
            ),
            "score": int(score),
            "hits": ";".join(
                sorted(
                    set(hits)
                )
            ),
            "source": src,
        })

    return pd.DataFrame(rows), text


def relevant_statements(path, text):
    tree = ast.parse(text)
    rows = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.Return,
                ast.If,
                ast.For,
                ast.Expr,
            )
        ):
            continue

        seg = ast.get_source_segment(
            text,
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

        strong = any(
            token in low
            for token in [
                "x_norm",
                "y_norm",
                "anchor_camera",
                "image_width",
                "image_height",
                "corrected",
                "homography",
                "project",
                " / ",
            ]
        )

        if not hits or not strong:
            continue

        rows.append({
            "file": str(path),
            "line": int(
                getattr(
                    node,
                    "lineno",
                    -1
                )
            ),
            "hits": ";".join(
                sorted(
                    set(hits)
                )
            ),
            "statement": " ".join(
                seg.strip().split()
            ),
        })

    return pd.DataFrame(rows)


def lexical_contexts(path, text):
    lines = text.splitlines()
    rows = []

    patterns = [
        "x_norm",
        "y_norm",
        "anchor_camera",
        "image_width",
        "image_height",
        "corrected",
        "homography",
        "project",
        "transform",
        "u =",
        "v =",
        "center",
        "bbox",
    ]

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

        lo = max(1, i - 6)
        hi = min(len(lines), i + 6)

        context = "\n".join(
            f"{j:05d}: {lines[j-1]}"
            for j in range(
                lo,
                hi + 1
            )
        )

        rows.append({
            "file": str(path),
            "line": int(i),
            "hits": ";".join(hits),
            "text": line.strip(),
            "context": context,
        })

    return pd.DataFrame(rows)


def inspect_bambi_gt():
    """
    Read MOT-style text without assuming exact semantic column names beyond
    the conventional first six numeric columns. We only inventory shape/range.
    """
    df = pd.read_csv(
        BAMBI_GT,
        header=None
    )

    out = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "first_10_rows": df.head(10).values.tolist(),
    }

    if df.shape[1] >= 6:
        numeric = df.iloc[:, :6].apply(
            pd.to_numeric,
            errors="coerce"
        )

        out["first_six_column_min"] = [
            None if pd.isna(x) else float(x)
            for x in numeric.min().tolist()
        ]

        out["first_six_column_max"] = [
            None if pd.isna(x) else float(x)
            for x in numeric.max().tolist()
        ]

    return out


def inspect_bambi_json(path):
    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    summary = {
        "type": type(raw).__name__,
    }

    if isinstance(raw, dict):
        summary["top_level_keys"] = list(raw.keys())[:100]

        scalar_like = {}
        for k, v in raw.items():
            if isinstance(
                v,
                (
                    str,
                    int,
                    float,
                    bool,
                )
            ) or v is None:
                scalar_like[k] = v

        summary["top_level_scalar_values"] = scalar_like

    elif isinstance(raw, list):
        summary["length"] = len(raw)

        if raw:
            summary["first_item_type"] = type(
                raw[0]
            ).__name__

            if isinstance(raw[0], dict):
                summary["first_item_keys"] = list(
                    raw[0].keys()
                )[:100]

    return summary


def main():
    print(
        "\nSTEP 6D-K3C6 — EXACT NORMALIZED IMAGE MAPPING FORMULA CONFIRMATION"
    )

    print(
        "Read-only audit. NO BAMBI coordinate conversion or Step7 alarms are generated.\n"
    )

    for p in [
        WILDBOX_BUILD,
        FRESH_EVAL,
        BAMBI_GT,
        BAMBI_META,
        BAMBI_POSES,
    ]:
        require(p)

    fn_frames = []
    stmt_frames = []
    lex_frames = []

    for path in [
        WILDBOX_BUILD,
        FRESH_EVAL,
    ]:
        fn_df, text = collect_functions(
            path
        )

        fn_frames.append(
            fn_df
        )

        stmt_frames.append(
            relevant_statements(
                path,
                text
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
        stmt_frames,
        ignore_index=True
    )

    lexical = pd.concat(
        lex_frames,
        ignore_index=True
    )

    if len(functions):
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
        / "TABLE_step6dk3c6_ranked_mapping_functions.csv",
        index=False
    )

    statements.to_csv(
        OUT
        / "TABLE_step6dk3c6_mapping_statements.csv",
        index=False
    )

    lexical.to_csv(
        OUT
        / "TABLE_step6dk3c6_mapping_contexts.csv",
        index=False
    )

    with open(
        OUT
        / "STEP6D_K3C6_EXACT_MAPPING_SOURCE_BUNDLE.txt",
        "w",
        encoding="utf-8"
    ) as f:
        for _, row in functions.head(
            30
        ).iterrows():
            f.write("=" * 100 + "\n")
            f.write(
                f"FILE: {row['file']}\n"
                f"FUNCTION: {row['function']}\n"
                f"LINES: {row['line_start']}-{row['line_end']}\n"
                f"SCORE: {row['score']}\n"
                f"HITS: {row['hits']}\n"
            )
            f.write("=" * 100 + "\n")
            f.write(row["source"])
            f.write("\n\n")

    bambi_gt = inspect_bambi_gt()
    bambi_meta = inspect_bambi_json(
        BAMBI_META
    )
    bambi_poses = inspect_bambi_json(
        BAMBI_POSES
    )

    with open(
        OUT
        / "STEP6D_K3C6_BAMBI_RAW_INPUT_INVENTORY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "gt": bambi_gt,
                "metadata": bambi_meta,
                "matched_poses": bambi_poses,
            },
            f,
            indent=2
        )

    print(
        "TOP EXACT MAPPING FUNCTIONS:"
    )

    print(
        functions.head(
            30
        )[
            [
                "file",
                "function",
                "line_start",
                "line_end",
                "score",
                "hits",
            ]
        ].to_string(
            index=False
        )
        if len(functions)
        else "NONE"
    )

    print(
        "\nKEY MAPPING STATEMENTS:"
    )

    print(
        statements.head(
            120
        ).to_string(
            index=False
        )
        if len(statements)
        else "NONE"
    )

    print(
        "\nMAPPING CONTEXTS:"
    )

    print(
        lexical.head(
            120
        ).to_string(
            index=False
        )
        if len(lexical)
        else "NONE"
    )

    print(
        "\nBAMBI RAW GT INVENTORY:"
    )

    print(
        json.dumps(
            bambi_gt,
            indent=2
        )[:8000]
    )

    print(
        "\nBAMBI METADATA INVENTORY:"
    )

    print(
        json.dumps(
            bambi_meta,
            indent=2
        )[:8000]
    )

    print(
        "\nBAMBI MATCHED POSES INVENTORY:"
    )

    print(
        json.dumps(
            bambi_poses,
            indent=2
        )[:8000]
    )

    combined = " ".join(
        statements[
            "statement"
        ].astype(str).tolist()
    ).lower() if len(
        statements
    ) else ""

    formula_xy_norm = (
        "x_norm"
        in
        combined
        and
        "y_norm"
        in
        combined
        and
        "width"
        in
        combined
        and
        "height"
        in
        combined
    )

    corrected_evidence = (
        "corrected"
        in
        combined
        or
        "anchor_camera"
        in
        combined
        or
        "homography"
        in
        combined
        or
        "project"
        in
        combined
    )

    # Conservative status only. We do not auto-run any conversion.
    if (
        formula_xy_norm
        and
        corrected_evidence
    ):
        status = (
            "NORMALIZED_MAPPING_FORMULA_AND_PRENORMALIZATION_"
            "CORRECTION_EVIDENCE_RECOVERED_PENDING_BAMBI_EQUIVALENCE_CHECK"
        )
    elif formula_xy_norm:
        status = (
            "NORMALIZED_MAPPING_FORMULA_RECOVERED_BUT_"
            "PRENORMALIZATION_CORRECTION_NOT_PROVEN"
        )
    else:
        status = (
            "EXACT_NORMALIZED_MAPPING_NOT_FULLY_CONFIRMED_"
            "DO_NOT_RUN_BAMBI_STEP7"
        )

    summary = {
        "step":
            "STEP_6D_K3C6_EXACT_NORMALIZED_IMAGE_MAPPING_FORMULA_CONFIRMATION",

        "wildbox_build_sha256":
            sha256_file(
                WILDBOX_BUILD
            ),

        "fresh_eval_sha256":
            sha256_file(
                FRESH_EVAL
            ),

        "formula_xy_norm_evidence":
            bool(
                formula_xy_norm
            ),

        "prenormalization_correction_evidence":
            bool(
                corrected_evidence
            ),

        "status":
            status,

        "rule":
            (
                "Do not treat BAMBI raw MOT pixels divided by 1024 as equivalent "
                "to Step18 coordinates unless source proves the same pre-normalization "
                "camera-correction/anchor-frame convention."
            ),
    }

    (
        OUT
        / "STEP6D_K3C6_SUMMARY.json"
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
        status
    )

    print(
        "\nSTEP 6D-K3C6 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
