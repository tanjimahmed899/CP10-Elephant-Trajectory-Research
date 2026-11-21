from pathlib import Path
import ast
import hashlib
import json
import re
import pandas as pd
import numpy as np

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

BUILD_10HZ = ROOT / "build_unified_10hz_dataset.py"
BUILD_FINAL = ROOT / "build_final_neural_dataset.py"

CORRECTED_ROOT = ROOT / "corrected_trajectory_dataset"

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_operational_warning"
    / "coordinate_mapping_provenance"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_FUNCTIONS = [
    "load_birdsai",
    "make_relative",
    "canonicalize_sample",
    "transform_dataset",
]

DIMENSION_TERMS = [
    "width",
    "height",
    "image_width",
    "image_height",
    "frame_width",
    "frame_height",
    "resolution",
    "normalize",
    "normalized",
    "corrected_x",
    "corrected_y",
    "x_norm",
    "y_norm",
    "x_corrected",
    "y_corrected",
    "bbox",
    "center_x",
    "center_y",
]

COMMON_DIMS = [
    320, 480, 512, 576, 640, 720, 768, 960,
    1024, 1080, 1280, 1920, 2160, 2560, 3840, 4096
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


def get_named_function(path, function_name):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(text)

    matches = [
        n for n in ast.walk(tree)
        if isinstance(
            n,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and n.name == function_name
    ]

    if len(matches) != 1:
        return None

    node = matches[0]

    return {
        "file": str(path),
        "function": function_name,
        "line_start": int(node.lineno),
        "line_end": int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        ),
        "source": ast.get_source_segment(
            text,
            node
        ) or "",
    }


def find_dimension_contexts(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()
    rows = []

    for i, line in enumerate(lines, start=1):
        low = line.lower()

        term_hits = [
            t for t in DIMENSION_TERMS
            if t in low
        ]

        number_hits = [
            d for d in COMMON_DIMS
            if re.search(
                rf"(?<!\d){d}(?!\d)",
                line
            )
        ]

        division_like = (
            "/" in line
            and any(
                token in low
                for token in [
                    "width",
                    "height",
                    "norm",
                    "correct",
                    "center",
                    "bbox",
                    "x",
                    "y",
                ]
            )
        )

        if not (
            term_hits
            or number_hits
            or division_like
        ):
            continue

        lo = max(1, i - 5)
        hi = min(len(lines), i + 5)

        context = "\n".join(
            f"{j:05d}: {lines[j-1]}"
            for j in range(lo, hi + 1)
        )

        rows.append({
            "file": str(path),
            "line": int(i),
            "term_hits": ";".join(term_hits),
            "number_hits": ";".join(
                str(x) for x in number_hits
            ),
            "division_like": bool(division_like),
            "text": line.strip(),
            "context": context,
        })

    return rows


def inspect_corrected_csvs():
    rows = []
    stats = []

    if not CORRECTED_ROOT.exists():
        return (
            pd.DataFrame(rows),
            pd.DataFrame(stats)
        )

    for path in CORRECTED_ROOT.rglob("*.csv"):
        try:
            df = pd.read_csv(
                path,
                nrows=1000
            )
        except Exception:
            continue

        cols = list(df.columns)

        relevant_cols = [
            c for c in cols
            if any(
                token in c.lower()
                for token in [
                    "x",
                    "y",
                    "width",
                    "height",
                    "norm",
                    "correct",
                    "center",
                    "bbox",
                    "frame",
                ]
            )
        ]

        rows.append({
            "file": str(path),
            "columns": ";".join(cols),
            "relevant_columns": ";".join(
                relevant_cols
            ),
        })

        for c in relevant_cols:
            s = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            if s.notna().sum() == 0:
                continue

            stats.append({
                "file": str(path),
                "column": c,
                "n_numeric": int(
                    s.notna().sum()
                ),
                "min": float(
                    s.min()
                ),
                "max": float(
                    s.max()
                ),
                "median": float(
                    s.median()
                ),
            })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(stats)
    )


def ratio_candidate_audit(csv_stats):
    """
    Only identify plausible normalized/raw pairs within the same file.
    This does NOT declare a mapping; it flags candidates for provenance.
    """
    rows = []

    if csv_stats.empty:
        return pd.DataFrame(rows)

    for file, g in csv_stats.groupby("file"):
        recs = g.to_dict(
            orient="records"
        )

        normalized_candidates = [
            r for r in recs
            if (
                r["min"] >= -0.25
                and
                r["max"] <= 1.25
            )
        ]

        raw_candidates = [
            r for r in recs
            if r["max"] > 10
        ]

        for n in normalized_candidates:
            for r in raw_candidates:
                if n["column"] == r["column"]:
                    continue

                # Only lexical x/x or y/y pairing.
                nlow = n["column"].lower()
                rlow = r["column"].lower()

                axis_match = (
                    (
                        "x" in nlow
                        and
                        "x" in rlow
                    )
                    or
                    (
                        "y" in nlow
                        and
                        "y" in rlow
                    )
                )

                if not axis_match:
                    continue

                rows.append({
                    "file": file,
                    "normalized_candidate": n["column"],
                    "raw_candidate": r["column"],
                    "normalized_min": n["min"],
                    "normalized_max": n["max"],
                    "raw_min": r["min"],
                    "raw_max": r["max"],
                })

    return pd.DataFrame(rows)


def search_all_python_for_mapping():
    rows = []

    for path in ROOT.rglob("*.py"):
        try:
            if path.stat().st_size > 2_000_000:
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue

        low = text.lower()

        strong_terms = [
            "image_width",
            "image_height",
            "frame_width",
            "frame_height",
            "x_norm",
            "y_norm",
            "normalized_x",
            "normalized_y",
            "corrected_x",
            "corrected_y",
        ]

        hits = [
            t for t in strong_terms
            if t in low
        ]

        if not hits:
            continue

        lines = text.splitlines()

        for i, line in enumerate(
            lines,
            start=1
        ):
            ll = line.lower()

            line_hits = [
                t for t in strong_terms
                if t in ll
            ]

            if not line_hits:
                continue

            lo = max(1, i - 5)
            hi = min(len(lines), i + 5)

            context = "\n".join(
                f"{j:05d}: {lines[j-1]}"
                for j in range(lo, hi + 1)
            )

            rows.append({
                "file": str(path),
                "line": int(i),
                "hits": ";".join(line_hits),
                "text": line.strip(),
                "context": context,
            })

    return pd.DataFrame(rows)


def main():
    print(
        "\nSTEP 6D-K3C5 — TARGETED NORMALIZED IMAGE-PLANE MAPPING PROVENANCE RECOVERY"
    )

    print(
        "Read-only audit. NO BAMBI coordinate conversion or Step7 alarms are generated.\n"
    )

    for p in [
        BUILD_10HZ,
        BUILD_FINAL,
    ]:
        require(p)

    function_infos = []

    for path, name in [
        (BUILD_10HZ, "load_birdsai"),
        (BUILD_10HZ, "make_relative"),
        (BUILD_FINAL, "canonicalize_sample"),
        (BUILD_FINAL, "transform_dataset"),
    ]:
        info = get_named_function(
            path,
            name
        )

        if info is not None:
            function_infos.append(
                info
            )

    with open(
        OUT
        / "STEP6D_K3C5_TARGET_FUNCTION_SOURCES.txt",
        "w",
        encoding="utf-8"
    ) as f:
        for info in function_infos:
            f.write("=" * 100 + "\n")
            f.write(
                f"FILE: {info['file']}\n"
                f"FUNCTION: {info['function']}\n"
                f"LINES: {info['line_start']}-{info['line_end']}\n"
            )
            f.write("=" * 100 + "\n")
            f.write(info["source"])
            f.write("\n\n")

    context_rows = []

    for path in [
        BUILD_10HZ,
        BUILD_FINAL,
    ]:
        context_rows.extend(
            find_dimension_contexts(
                path
            )
        )

    context_df = pd.DataFrame(
        context_rows
    )

    context_df.to_csv(
        OUT
        / "TABLE_step6dk3c5_target_dimension_contexts.csv",
        index=False
    )

    (
        csv_inventory,
        csv_stats
    ) = inspect_corrected_csvs()

    csv_inventory.to_csv(
        OUT
        / "TABLE_step6dk3c5_corrected_csv_inventory.csv",
        index=False
    )

    csv_stats.to_csv(
        OUT
        / "TABLE_step6dk3c5_corrected_csv_numeric_stats.csv",
        index=False
    )

    ratio_candidates = ratio_candidate_audit(
        csv_stats
    )

    ratio_candidates.to_csv(
        OUT
        / "TABLE_step6dk3c5_possible_raw_normalized_pairs.csv",
        index=False
    )

    all_py = search_all_python_for_mapping()

    all_py.to_csv(
        OUT
        / "TABLE_step6dk3c5_all_python_mapping_hits.csv",
        index=False
    )

    print(
        "TARGET FUNCTIONS RECOVERED:"
    )

    for info in function_infos:
        print(
            f"{info['file']} :: {info['function']} "
            f"(lines {info['line_start']}-{info['line_end']})"
        )

    print(
        "\nTARGET DIMENSION/NORMALIZATION CONTEXTS:"
    )

    if len(
        context_df
    ):
        print(
            context_df.head(
                100
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nCORRECTED TRAJECTORY CSV INVENTORY:"
    )

    if len(
        csv_inventory
    ):
        print(
            csv_inventory.head(
                50
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nCORRECTED TRAJECTORY NUMERIC STATS:"
    )

    if len(
        csv_stats
    ):
        print(
            csv_stats.head(
                100
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nPOSSIBLE RAW/NORMALIZED COLUMN PAIRS:"
    )

    if len(
        ratio_candidates
    ):
        print(
            ratio_candidates.head(
                100
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nALL-PYTHON STRONG MAPPING HITS:"
    )

    if len(
        all_py
    ):
        print(
            all_py.head(
                100
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    # Conservative evidence check only.
    combined = ""

    for df in [
        context_df,
        all_py,
    ]:
        if len(df):
            combined += " ".join(
                df.astype(
                    str
                ).fillna(
                    ""
                ).values.flatten()
            ).lower()

    exact_source_formula_evidence = any(
        token in combined
        for token in [
            "/ image_width",
            "/ image_height",
            "/ frame_width",
            "/ frame_height",
            "normalized_x =",
            "normalized_y =",
            "x_norm =",
            "y_norm =",
        ]
    )

    if exact_source_formula_evidence:
        status = (
            "EXACT_NORMALIZED_IMAGE_MAPPING_CANDIDATE_FOUND_"
            "PENDING_FORMULA_CONFIRMATION"
        )
    else:
        status = (
            "NO_EXACT_NORMALIZED_IMAGE_MAPPING_RECOVERED_"
            "DO_NOT_RUN_BAMBI_STEP7"
        )

    summary = {
        "step":
            "STEP_6D_K3C5_TARGETED_NORMALIZED_IMAGE_PLANE_MAPPING_PROVENANCE_RECOVERY",

        "build_unified_10hz_sha256":
            sha256_file(
                BUILD_10HZ
            ),

        "build_final_neural_sha256":
            sha256_file(
                BUILD_FINAL
            ),

        "target_functions_recovered":
            [
                {
                    "file": x["file"],
                    "function": x["function"],
                    "line_start": x["line_start"],
                    "line_end": x["line_end"],
                }
                for x in function_infos
            ],

        "corrected_csv_files":
            int(
                len(
                    csv_inventory
                )
            ),

        "strong_python_mapping_hits":
            int(
                len(
                    all_py
                )
            ),

        "exact_source_formula_evidence":
            bool(
                exact_source_formula_evidence
            ),

        "status":
            status,

        "decision_rule": (
            "If the exact frozen normalized image-plane mapping cannot be "
            "proven from source/data provenance, do not convert BAMBI UTM "
            "coordinates ad hoc and do not claim BAMBI Step7 operational confirmation."
        ),
    }

    (
        OUT
        / "STEP6D_K3C5_SUMMARY.json"
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
        "\nSTEP 6D-K3C5 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
