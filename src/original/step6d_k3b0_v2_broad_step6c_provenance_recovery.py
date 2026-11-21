from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import math
import pandas as pd

# Wider READ-ONLY search root. Nothing under this tree is modified.
READ_ONLY_ROOTS = [
    Path(r"E:\AIUB\Capstone Project\Journal Based Results"),
]

# Existing generated outputs may contain the Step6C lock/results even if the
# original source script was not copied into Main Result.
DOWNLOAD_ROOTS = [
    Path(r"C:\Users\HP\Downloads"),
]

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_fixed_conformal"
    / "source_audit_v2"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGET_Q = {
    "2s_to_2s_primary_q90": 2.158768,
    "2s_to_2s_primary_q95": 2.560773,
    "2s_to_2s_endpoint_q90": 2.076248,
    "2s_to_2s_endpoint_q95": 2.446776,
    "2s_to_5s_primary_q90": 1.975130,
    "2s_to_5s_primary_q95": 2.950976,
    "2s_to_5s_endpoint_q90": 1.975130,
    "2s_to_5s_endpoint_q95": 2.950976,
}

TEXT_EXTS = {
    ".py", ".json", ".txt", ".csv", ".md", ".yaml", ".yml",
    ".log", ".tsv"
}

FILENAME_TERMS = [
    "step6",
    "step_6",
    "conformal",
    "crossfit",
    "cross_fit",
    "calibration",
    "coverage",
    "uncertainty",
]

CONTENT_TERMS = [
    "conformal",
    "nonconform",
    "q90",
    "q95",
    "coverage",
    "fully covered",
    "fully_covered",
    "track macro",
    "track_macro",
    "max radial",
    "max_radial",
    "endpoint",
    "calibration",
    "quantile",
    "track-level",
    "track_level",
    "cross-fitted",
    "cross_fitted",
    "crossfit",
]

# Strong source-definition patterns.
FORMULA_TERMS = [
    "np.linalg.norm",
    "max_radial",
    "error.max",
    ".max(axis",
    "groupby",
    "track_id",
    "quantile",
    "ceil",
    "covered",
    "coverage",
]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_read_text(path, max_bytes=20_000_000):
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def all_candidate_files():
    seen = set()
    rows = []

    for root_type, roots in [
        ("journal_read_only", READ_ONLY_ROOTS),
        ("downloads_existing", DOWNLOAD_ROOTS),
    ]:
        for root in roots:
            if not root.exists():
                continue

            for path in root.rglob("*"):
                if not path.is_file():
                    continue

                # Avoid scanning huge unrelated binary trees.
                if path.suffix.lower() not in TEXT_EXTS:
                    continue

                # For Downloads, strongly restrict to relevant project names
                # unless a file is small and explicitly conformal/step6 related.
                if root_type == "downloads_existing":
                    low_path = str(path).lower()
                    if not any(
                        term in low_path
                        for term in [
                            "step6",
                            "conformal",
                            "uncert",
                            "journal",
                            "trajectory",
                            "bambi",
                            "transformer",
                        ]
                    ):
                        continue

                key = str(path).lower()
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    (root_type, path)
                )

    return rows


def q_hits(path, text):
    rows = []
    lines = text.splitlines()

    # Match target values using looser decimal prefixes because summaries may
    # store more digits or rounded values.
    for label, target in TARGET_Q.items():
        prefixes = {
            f"{target:.6f}",
            f"{target:.5f}",
            f"{target:.4f}",
        }

        for i, line in enumerate(lines, start=1):
            if any(p in line for p in prefixes):
                lo = max(1, i - 3)
                hi = min(len(lines), i + 3)

                context = "\n".join(
                    f"{j:05d}: {lines[j-1]}"
                    for j in range(lo, hi + 1)
                )

                rows.append({
                    "file": str(path),
                    "line": int(i),
                    "label": label,
                    "target_value": float(target),
                    "text": line.strip(),
                    "context": context,
                })

    return rows


def keyword_hits(path, text):
    rows = []
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        low = line.lower()

        hit_terms = [
            t for t in CONTENT_TERMS
            if t in low
        ]

        if not hit_terms:
            continue

        # Keep only higher-value contexts.
        nearby = "\n".join(
            lines[
                max(0, i - 5):
                min(len(lines), i + 4)
            ]
        ).lower()

        strength = len(hit_terms)

        if any(
            t in nearby
            for t in FORMULA_TERMS
        ):
            strength += 3

        if any(
            q in nearby
            for q in [
                "2.158",
                "2.560",
                "1.975",
                "2.950",
                "2.076",
                "2.446",
            ]
        ):
            strength += 10

        if strength < 2:
            continue

        lo = max(1, i - 4)
        hi = min(len(lines), i + 4)

        context = "\n".join(
            f"{j:05d}: {lines[j-1]}"
            for j in range(lo, hi + 1)
        )

        rows.append({
            "file": str(path),
            "line": int(i),
            "strength": int(strength),
            "hit_terms": ";".join(hit_terms),
            "text": line.strip(),
            "context": context,
        })

    return rows


def ast_formula_functions(path, text):
    rows = []

    if path.suffix.lower() != ".py":
        return rows

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
        low = src.lower()

        hits = [
            t for t in CONTENT_TERMS + FORMULA_TERMS
            if t in low
        ]

        if len(hits) < 3:
            continue

        # Strongly prioritize actual conformal-ish code.
        score = len(set(hits))

        if "conformal" in low:
            score += 5
        if "quantile" in low:
            score += 3
        if "track_id" in low:
            score += 2
        if "coverage" in low:
            score += 2
        if "max_radial" in low or "error.max" in low:
            score += 2

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
            "hits": ";".join(sorted(set(hits))),
            "source": src,
        })

    return rows


def filename_score(path):
    low = path.name.lower()
    score = 0

    for t in FILENAME_TERMS:
        if t in low:
            score += 3

    parent_low = str(path.parent).lower()

    for t in FILENAME_TERMS:
        if t in parent_low:
            score += 1

    return score


def main():
    print(
        "\nSTEP 6D-K3B0-V2 — BROAD STEP6C CONFORMAL PROVENANCE RECOVERY"
    )

    print(
        "Read-only search. No BAMBI coverage is computed.\n"
    )

    candidates = all_candidate_files()

    file_rows = []
    q_rows = []
    kw_rows = []
    fn_rows = []

    for root_type, path in candidates:
        text = safe_read_text(path)

        if text is None:
            continue

        low = text.lower()

        content_score = sum(
            1 for term in CONTENT_TERMS
            if term in low
        )

        q_count_before = len(q_rows)
        q_rows.extend(
            q_hits(path, text)
        )
        q_count = len(q_rows) - q_count_before

        kw_count_before = len(kw_rows)
        kw_rows.extend(
            keyword_hits(path, text)
        )
        kw_count = len(kw_rows) - kw_count_before

        fn_count_before = len(fn_rows)
        fn_rows.extend(
            ast_formula_functions(path, text)
        )
        fn_count = len(fn_rows) - fn_count_before

        total_score = (
            filename_score(path)
            + content_score
            + 20 * q_count
            + 2 * kw_count
            + 3 * fn_count
        )

        if total_score <= 0:
            continue

        try:
            digest = sha256_file(path)
            size_bytes = int(path.stat().st_size)
        except Exception:
            digest = "ERROR"
            size_bytes = -1

        file_rows.append({
            "root_type": root_type,
            "file": str(path),
            "name": path.name,
            "size_bytes": size_bytes,
            "score": int(total_score),
            "q_hits": int(q_count),
            "keyword_context_hits": int(kw_count),
            "formula_functions": int(fn_count),
            "sha256": digest,
        })

    files_df = pd.DataFrame(file_rows)
    q_df = pd.DataFrame(q_rows)
    kw_df = pd.DataFrame(kw_rows)
    fn_df = pd.DataFrame(fn_rows)

    if len(files_df):
        files_df = files_df.sort_values(
            ["score", "q_hits", "file"],
            ascending=[False, False, True]
        )

    if len(q_df):
        q_df = q_df.sort_values(
            ["label", "file", "line"]
        )

    if len(kw_df):
        kw_df = kw_df.sort_values(
            ["strength", "file", "line"],
            ascending=[False, True, True]
        )

    if len(fn_df):
        fn_df = fn_df.sort_values(
            ["score", "file", "line_start"],
            ascending=[False, True, True]
        )

    files_df.to_csv(
        OUT / "TABLE_step6dk3b0v2_ranked_files.csv",
        index=False
    )

    q_df.to_csv(
        OUT / "TABLE_step6dk3b0v2_qvalue_hits.csv",
        index=False
    )

    kw_df.to_csv(
        OUT / "step6dk3b0v2_keyword_contexts.csv",
        index=False
    )

    fn_df.to_csv(
        OUT / "TABLE_step6dk3b0v2_formula_functions.csv",
        index=False
    )

    # Determine whether every expected q label has at least one source hit.
    q_labels_found = (
        sorted(q_df["label"].unique().tolist())
        if len(q_df)
        else []
    )

    q_labels_expected = sorted(
        TARGET_Q.keys()
    )

    all_q_values_recovered = (
        q_labels_found
        ==
        q_labels_expected
    )

    # Candidate files with both q hits and conformal formula evidence are
    # the strongest provenance sources.
    strong_files = pd.DataFrame()

    if len(files_df):
        strong_files = files_df[
            (files_df["q_hits"] > 0)
            |
            (files_df["formula_functions"] > 0)
        ].copy()

    status = (
        "STEP6C_PROVENANCE_RECOVERED_PENDING_EXACT_FORMULA_EXTRACTION"
        if all_q_values_recovered and len(fn_df)
        else
        "STEP6C_PROVENANCE_PARTIAL_REQUIRES_TARGETED_EXTRACTION"
    )

    summary = {
        "step":
            "STEP_6D_K3B0_V2_BROAD_STEP6C_PROVENANCE_RECOVERY",

        "candidate_text_files_scanned":
            int(len(candidates)),

        "ranked_relevant_files":
            int(len(files_df)),

        "q_value_hits":
            int(len(q_df)),

        "q_labels_found":
            q_labels_found,

        "q_labels_expected":
            q_labels_expected,

        "all_expected_q_labels_recovered":
            bool(all_q_values_recovered),

        "candidate_formula_functions":
            int(len(fn_df)),

        "status":
            status,

        "rule":
            "Do not compute BAMBI conformal coverage until exact Step6C score/unit/aggregation definitions are source-confirmed.",
    }

    (
        OUT
        / "STEP6D_K3B0_V2_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    with open(
        OUT
        / "STEP6D_K3B0_V2_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K3B0-V2 — BROAD STEP6C CONFORMAL PROVENANCE RECOVERY\n"
        )
        f.write("=" * 92 + "\n\n")

        f.write("TOP RANKED FILES\n")
        f.write("----------------\n")
        f.write(
            files_df.head(40).to_string(index=False)
            if len(files_df)
            else "NONE"
        )

        f.write("\n\nQ-VALUE HITS\n")
        f.write("------------\n")
        f.write(
            q_df.head(120).to_string(index=False)
            if len(q_df)
            else "NONE"
        )

        f.write("\n\nTOP FORMULA FUNCTIONS\n")
        f.write("---------------------\n")
        f.write(
            fn_df.head(40)[
                [
                    "file",
                    "function",
                    "line_start",
                    "line_end",
                    "score",
                    "hits",
                ]
            ].to_string(index=False)
            if len(fn_df)
            else "NONE"
        )

        f.write("\n\nSTRONG PROVENANCE FILES\n")
        f.write("-----------------------\n")
        f.write(
            strong_files.head(40).to_string(index=False)
            if len(strong_files)
            else "NONE"
        )

        f.write("\n\nSTATUS\n")
        f.write("------\n")
        f.write(status + "\n")

    print("TOP RANKED FILES:")
    print(
        files_df.head(40).to_string(index=False)
        if len(files_df)
        else "NONE"
    )

    print("\nQ-VALUE HITS:")
    print(
        q_df.head(120)[
            [
                "file",
                "line",
                "label",
                "target_value",
                "text",
            ]
        ].to_string(index=False)
        if len(q_df)
        else "NONE"
    )

    print("\nTOP FORMULA FUNCTIONS:")
    print(
        fn_df.head(40)[
            [
                "file",
                "function",
                "line_start",
                "line_end",
                "score",
                "hits",
            ]
        ].to_string(index=False)
        if len(fn_df)
        else "NONE"
    )

    print("\nSTRONG PROVENANCE FILES:")
    print(
        strong_files.head(40)[
            [
                "root_type",
                "file",
                "score",
                "q_hits",
                "formula_functions",
            ]
        ].to_string(index=False)
        if len(strong_files)
        else "NONE"
    )

    print("\nSTATUS:")
    print(status)

    print("\nSTEP 6D-K3B0-V2 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
