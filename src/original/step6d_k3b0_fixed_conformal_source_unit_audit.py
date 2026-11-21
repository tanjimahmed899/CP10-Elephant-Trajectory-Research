from pathlib import Path
import ast
import json
import re
import hashlib
import pandas as pd

READ_ONLY_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

BAMBI_INFERENCE_DIR = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_frozen_ensemble"
    / "first_inference"
)

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_fixed_conformal"
    / "source_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

KEYWORDS = [
    "q90",
    "q95",
    "conformal",
    "coverage",
    "track",
    "max_radial",
    "endpoint",
    "nonconform",
    "score",
    "quantile",
    "calibration",
    "fully_covered",
    "track_macro",
]

LIKELY_FILES = [
    "step6c_crossfit_conformal.py",
    "step6c_crossfit_conformal_tracklevel.py",
    "step6c_crossfit.py",
    "step6c_conformal.py",
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


def score_text(text):
    low = text.lower()
    score = 0

    for k in KEYWORDS:
        if k.lower() in low:
            score += 1

    # Strong signals.
    if "2.158768" in text:
        score += 20
    if "2.560773" in text:
        score += 20
    if "1.975130" in text:
        score += 20
    if "2.950976" in text:
        score += 20

    return score


def function_rows(path, text):
    rows = []

    try:
        tree = ast.parse(text)
    except Exception:
        return rows

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue

        src = ast.get_source_segment(text, node) or ""
        s = score_text(src)

        if s <= 0:
            continue

        rows.append({
            "file": str(path),
            "function": node.name,
            "line_start": int(node.lineno),
            "line_end": int(getattr(node, "end_lineno", node.lineno)),
            "score": int(s),
            "source": src,
        })

    return rows


def lexical_rows(path, text):
    rows = []
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        low = line.lower()

        if not any(k.lower() in low for k in KEYWORDS):
            continue

        lo = max(1, i - 3)
        hi = min(len(lines), i + 3)

        context = "\n".join(
            f"{j:05d}: {lines[j-1]}"
            for j in range(lo, hi + 1)
        )

        rows.append({
            "file": str(path),
            "line": int(i),
            "text": line.strip(),
            "context": context,
        })

    return rows


def numeric_literal_rows(path, text):
    rows = []

    try:
        tree = ast.parse(text)
    except Exception:
        return rows

    targets = {
        2.158768,
        2.560773,
        1.975130,
        2.950976,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            v = float(node.value)

            for target in targets:
                if abs(v - target) < 1e-4:
                    rows.append({
                        "file": str(path),
                        "line": int(getattr(node, "lineno", -1)),
                        "value": v,
                        "target": target,
                    })

    return rows


def main():
    print(
        "\nSTEP 6D-K3B0 — FIXED STEP6C CONFORMAL SOURCE/UNIT AUDIT"
    )
    print(
        "Read-only audit only. No BAMBI conformal coverage is computed.\n"
    )

    require(READ_ONLY_ROOT)
    require(BAMBI_INFERENCE_DIR)

    py_files = list(READ_ONLY_ROOT.rglob("*.py"))

    file_scores = []
    func_all = []
    lexical_all = []
    numeric_all = []

    for path in py_files:
        try:
            if path.stat().st_size > 5_000_000:
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue

        s = score_text(text)

        if s > 0:
            file_scores.append({
                "file": str(path),
                "name": path.name,
                "score": int(s),
                "sha256": sha256_file(path),
            })

        func_all.extend(
            function_rows(path, text)
        )

        if s >= 4 or path.name in LIKELY_FILES:
            lexical_all.extend(
                lexical_rows(path, text)
            )

        numeric_all.extend(
            numeric_literal_rows(path, text)
        )

    files_df = pd.DataFrame(file_scores)
    funcs_df = pd.DataFrame(func_all)
    lexical_df = pd.DataFrame(lexical_all)
    numeric_df = pd.DataFrame(numeric_all)

    if len(files_df):
        files_df = files_df.sort_values(
            ["score", "file"],
            ascending=[False, True]
        )

    if len(funcs_df):
        funcs_df = funcs_df.sort_values(
            ["score", "file", "line_start"],
            ascending=[False, True, True]
        )

    files_df.to_csv(
        OUT / "TABLE_step6dk3b0_ranked_conformal_source_files.csv",
        index=False
    )

    funcs_df.to_csv(
        OUT / "TABLE_step6dk3b0_ranked_conformal_functions.csv",
        index=False
    )

    lexical_df.to_csv(
        OUT / "step6dk3b0_conformal_lexical_evidence.csv",
        index=False
    )

    numeric_df.to_csv(
        OUT / "step6dk3b0_qvalue_literal_evidence.csv",
        index=False
    )

    # Also inspect first-inference NPZ keys/shapes/dtypes to ensure
    # canonical and world-space errors are both available for the next step.
    inference_inventory = []

    for path in sorted(
        BAMBI_INFERENCE_DIR.glob(
            "*_BAMBI_Flight6_first_locked_inference.npz"
        )
    ):
        import numpy as np

        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                arr = z[key]

                inference_inventory.append({
                    "file": str(path),
                    "key": key,
                    "shape": str(tuple(arr.shape)),
                    "dtype": str(arr.dtype),
                })

    inference_df = pd.DataFrame(
        inference_inventory
    )

    inference_df.to_csv(
        OUT / "TABLE_step6dk3b0_bambi_inference_array_inventory.csv",
        index=False
    )

    top_files = (
        files_df.head(15).to_dict(orient="records")
        if len(files_df)
        else []
    )

    top_funcs = (
        funcs_df.head(20)[
            [
                "file",
                "function",
                "line_start",
                "line_end",
                "score",
            ]
        ].to_dict(orient="records")
        if len(funcs_df)
        else []
    )

    summary = {
        "step":
            "STEP_6D_K3B0_FIXED_STEP6C_CONFORMAL_SOURCE_UNIT_AUDIT",

        "python_files_scanned":
            int(len(py_files)),

        "keyword_positive_files":
            int(len(files_df)),

        "candidate_functions":
            int(len(funcs_df)),

        "qvalue_literal_hits":
            int(len(numeric_df)),

        "bambi_inference_arrays":
            int(len(inference_df)),

        "status":
            (
                "CONFORMAL_SOURCE_CANDIDATES_FOUND_PENDING_EXACT_SCORE_UNIT_CONFIRMATION"
                if len(files_df) and len(funcs_df)
                else
                "CONFORMAL_SOURCE_NOT_RESOLVED_DO_NOT_COMPUTE_BAMBI_COVERAGE"
            ),

        "next_required_confirmation": [
            "Exact Step6C primary nonconformity score formula.",
            "Whether q90/q95 are canonical-distance units or absolute-distance units.",
            "Exact track-level aggregation used for primary coverage.",
            "Exact window-level and fully-covered-track definitions.",
            "Task-specific fixed q90/q95 values and whether endpoint q differs from primary max-error q.",
        ],
    }

    (
        OUT / "STEP6D_K3B0_SUMMARY.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    with open(
        OUT / "STEP6D_K3B0_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K3B0 — FIXED STEP6C CONFORMAL SOURCE/UNIT AUDIT\n"
        )
        f.write("=" * 84 + "\n\n")

        f.write("TOP RANKED CONFORMAL SOURCE FILES\n")
        f.write("---------------------------------\n")
        f.write(
            files_df.head(20).to_string(index=False)
            if len(files_df)
            else "NONE"
        )

        f.write("\n\nTOP CONFORMAL FUNCTIONS\n")
        f.write("-----------------------\n")
        f.write(
            funcs_df.head(30)[
                [
                    "file",
                    "function",
                    "line_start",
                    "line_end",
                    "score",
                ]
            ].to_string(index=False)
            if len(funcs_df)
            else "NONE"
        )

        f.write("\n\nQ-VALUE LITERAL EVIDENCE\n")
        f.write("------------------------\n")
        f.write(
            numeric_df.to_string(index=False)
            if len(numeric_df)
            else "NONE"
        )

        f.write("\n\nBAMBI INFERENCE ARRAY INVENTORY\n")
        f.write("-------------------------------\n")
        f.write(
            inference_df.to_string(index=False)
            if len(inference_df)
            else "NONE"
        )

        f.write("\n\nSTATUS\n------\n")
        f.write(summary["status"] + "\n")

    print("TOP RANKED CONFORMAL SOURCE FILES:")
    print(
        files_df.head(20).to_string(index=False)
        if len(files_df)
        else "NONE"
    )

    print("\nTOP CONFORMAL FUNCTIONS:")
    print(
        funcs_df.head(30)[
            [
                "file",
                "function",
                "line_start",
                "line_end",
                "score",
            ]
        ].to_string(index=False)
        if len(funcs_df)
        else "NONE"
    )

    print("\nQ-VALUE LITERAL EVIDENCE:")
    print(
        numeric_df.to_string(index=False)
        if len(numeric_df)
        else "NONE"
    )

    print("\nBAMBI INFERENCE ARRAY INVENTORY:")
    print(
        inference_df.to_string(index=False)
        if len(inference_df)
        else "NONE"
    )

    print("\nSTATUS:")
    print(summary["status"])

    print("\nSTEP 6D-K3B0 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
