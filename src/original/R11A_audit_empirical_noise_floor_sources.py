
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"

R10D_MANIFEST = (
    REVISION_ROOT
    / "R10_ensemble_analysis"
    / "R10D_cross_domain_ensemble_conclusion"
    / "R10D_MANIFEST.json"
)

OUT = (
    REVISION_ROOT
    / "R11_noise_floor_analysis"
    / "R11A_empirical_noise_floor_source_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

# Known high-value roots from prior completed work.
SEARCH_ROOTS = [
    REVISION_ROOT,
    MAIN1,
    MAIN2,
    DOWNLOADS,
]

# Strong filename hints for empirical observation / correction noise.
NAME_HINTS = [
    "ego",
    "motion",
    "residual",
    "landmark",
    "stationary",
    "jitter",
    "noise",
    "correction",
    "homography",
    "flow",
    "track",
    "trajectory",
]

# Avoid huge broad scans in unrelated areas.
SKIP_DIR_TOKENS = [
    "site-packages",
    ".git",
    "__pycache__",
    "node_modules",
]

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".txt",
    ".md",
}

MAX_TEXT_BYTES = 5_000_000


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


def verify_r10_complete():
    require(R10D_MANIFEST)

    m = json.loads(
        R10D_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R10D_CROSS_DOMAIN_ENSEMBLE_CONCLUSION_LOCKED_R10_COMPLETE_READY_FOR_R11_NOISE_FLOOR_ANALYSIS"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R10-D status: {m.get('status')}"
        )

    return m


def should_skip(path):
    lower = str(path).lower()

    return any(
        token.lower() in lower
        for token in SKIP_DIR_TOKENS
    )


def candidate_files():
    rows = []
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.glob("**/*"):
            if not p.is_file():
                continue

            if should_skip(p):
                continue

            suffix = p.suffix.lower()

            if suffix not in TEXT_SUFFIXES:
                continue

            name_lower = p.name.lower()

            if not any(
                hint in name_lower
                for hint in NAME_HINTS
            ):
                continue

            key = str(p.resolve()).lower()

            if key in seen:
                continue

            seen.add(key)

            try:
                size = int(p.stat().st_size)
            except Exception:
                continue

            if size > MAX_TEXT_BYTES:
                continue

            rows.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "suffix": suffix,
                    "size_bytes": size,
                    "sha256": sha256_file(p),
                }
            )

    return pd.DataFrame(rows)


def scan_keyword_contexts(file_df):
    patterns = [
        r"raw.*correct",
        r"correct.*raw",
        r"residual",
        r"stationary.*landmark",
        r"landmark",
        r"median.*px",
        r"p95",
        r"ego.?motion",
        r"background.*motion",
        r"noise.?floor",
        r"jitter",
        r"pixel",
        r"quantiz",
    ]

    regexes = [
        re.compile(
            pat,
            flags=re.IGNORECASE
        )
        for pat in patterns
    ]

    rows = []

    for _, meta in file_df.iterrows():
        p = Path(
            str(meta["path"])
        )

        try:
            text = p.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue

        for i, line in enumerate(
            text.splitlines(),
            start=1
        ):
            matched = [
                patterns[idx]
                for idx, rgx in enumerate(regexes)
                if rgx.search(line)
            ]

            if not matched:
                continue

            rows.append(
                {
                    "source_file": str(p),
                    "line_number": i,
                    "matched_patterns": "|".join(matched),
                    "line": line,
                }
            )

    return pd.DataFrame(rows)


def numeric_csv_inventory(file_df):
    rows = []

    for _, meta in file_df.iterrows():
        if str(meta["suffix"]) != ".csv":
            continue

        p = Path(
            str(meta["path"])
        )

        try:
            df = pd.read_csv(p)
        except Exception as exc:
            rows.append(
                {
                    "path": str(p),
                    "rows": None,
                    "columns": f"<read error: {exc}>",
                    "numeric_columns": "",
                    "candidate_metric_columns": "",
                }
            )
            continue

        numeric_cols = [
            str(c)
            for c in df.columns
            if pd.api.types.is_numeric_dtype(
                df[c]
            )
        ]

        candidate_metric_cols = [
            str(c)
            for c in df.columns
            if any(
                token in str(c).lower()
                for token in [
                    "resid",
                    "motion",
                    "error",
                    "err",
                    "px",
                    "raw",
                    "correct",
                    "jitter",
                    "flow",
                    "disp",
                ]
            )
        ]

        rows.append(
            {
                "path": str(p),
                "rows": int(len(df)),
                "columns": "|".join(
                    str(c)
                    for c in df.columns
                ),
                "numeric_columns": "|".join(
                    numeric_cols
                ),
                "candidate_metric_columns": "|".join(
                    candidate_metric_cols
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_promising_csvs(csv_inventory):
    rows = []

    for _, row in csv_inventory.iterrows():
        metric_cols = str(
            row[
                "candidate_metric_columns"
            ]
        ).strip()

        if not metric_cols:
            continue

        p = Path(
            str(
                row[
                    "path"
                ]
            )
        )

        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        for col in metric_cols.split("|"):
            if col not in df.columns:
                continue

            vals = pd.to_numeric(
                df[col],
                errors="coerce"
            ).dropna()

            if len(vals) == 0:
                continue

            rows.append(
                {
                    "path": str(p),
                    "column": col,
                    "n": int(len(vals)),
                    "min": float(vals.min()),
                    "median": float(vals.median()),
                    "mean": float(vals.mean()),
                    "p90": float(vals.quantile(0.90)),
                    "p95": float(vals.quantile(0.95)),
                    "max": float(vals.max()),
                }
            )

    return pd.DataFrame(rows)


def classify_sources(summary_df):
    rows = []

    for _, row in summary_df.iterrows():
        path = str(
            row[
                "path"
            ]
        ).lower()

        col = str(
            row[
                "column"
            ]
        ).lower()

        score = 0
        reasons = []

        if "ego" in path or "motion" in path:
            score += 2
            reasons.append(
                "ego/motion artifact"
            )

        if "landmark" in path or "stationary" in path:
            score += 3
            reasons.append(
                "stationary-landmark evidence"
            )

        if "resid" in col or "resid" in path:
            score += 3
            reasons.append(
                "residual metric"
            )

        if "correct" in col or "correct" in path:
            score += 2
            reasons.append(
                "corrected-motion metric"
            )

        if "px" in col or "pixel" in col:
            score += 2
            reasons.append(
                "pixel-space metric"
            )

        if "warning" in path or "geofence" in path:
            score -= 2
            reasons.append(
                "downstream warning artifact"
            )

        if "forecast" in path and "resid" not in path:
            score -= 1
            reasons.append(
                "forecast result rather than measurement noise"
            )

        rows.append(
            {
                **row.to_dict(),
                "noise_floor_relevance_score":
                    score,
                "relevance_reasons":
                    "|".join(reasons),
            }
        )

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            [
                "noise_floor_relevance_score",
                "path",
                "column",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).reset_index(
            drop=True
        )

    return out


def main():
    print(
        "\nR11-A — EMPIRICAL NOISE-FLOOR SOURCE AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO training, NO inference, NO threshold tuning. "
        "R10 must already be locked. This step searches completed experiment artifacts "
        "for empirical observation/correction-noise evidence (especially stationary-landmark "
        "ego-motion residuals and other pixel-space residual measures). The goal is to avoid "
        "inventing an arbitrary 'noise floor'. Forecast errors, warning thresholds, and model "
        "performance are not themselves treated as measurement noise unless separately justified.\n"
    )

    verify_r10_complete()

    files_df = candidate_files()

    if len(files_df) == 0:
        raise RuntimeError(
            "No candidate empirical-noise artifacts found."
        )

    contexts_df = scan_keyword_contexts(
        files_df
    )

    csv_inventory_df = numeric_csv_inventory(
        files_df
    )

    numeric_summary_df = summarize_promising_csvs(
        csv_inventory_df
    )

    ranked_df = classify_sources(
        numeric_summary_df
    )

    files_path = (
        OUT
        /
        "R11A_CANDIDATE_NOISE_ARTIFACTS.csv"
    )

    contexts_path = (
        OUT
        /
        "R11A_KEYWORD_CONTEXTS.csv"
    )

    csv_inventory_path = (
        OUT
        /
        "R11A_NUMERIC_CSV_INVENTORY.csv"
    )

    numeric_summary_path = (
        OUT
        /
        "R11A_CANDIDATE_METRIC_SUMMARY.csv"
    )

    ranked_path = (
        OUT
        /
        "R11A_RANKED_EMPIRICAL_NOISE_SOURCES.csv"
    )

    files_df.to_csv(
        files_path,
        index=False
    )

    contexts_df.to_csv(
        contexts_path,
        index=False
    )

    csv_inventory_df.to_csv(
        csv_inventory_path,
        index=False
    )

    numeric_summary_df.to_csv(
        numeric_summary_path,
        index=False
    )

    ranked_df.to_csv(
        ranked_path,
        index=False
    )

    strong = ranked_df[
        ranked_df[
            "noise_floor_relevance_score"
        ]
        >=
        5
    ].copy() if len(ranked_df) else pd.DataFrame()

    if len(strong):
        status = (
            "R11A_EMPIRICAL_NOISE_SOURCES_FOUND_READY_FOR_R11B_NOISE_FLOOR_PROTOCOL_LOCK"
        )
    else:
        status = (
            "R11A_NO_STRONG_EMPIRICAL_NOISE_SOURCE_FOUND_REVIEW_BEFORE_R11B"
        )

    manifest = {
        "step":
            "R11A_EMPIRICAL_NOISE_FLOOR_SOURCE_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "threshold_tuning":
            False,

        "primary_scientific_rule":
            (
                "Noise floor must come from an empirical measurement/correction-noise source "
                "or a clearly justified lower-bound process; model forecast error itself is "
                "not used to define the noise floor."
            ),

        "strong_candidate_count":
            int(
                len(
                    strong
                )
            ),

        "outputs": {
            "candidate_artifacts":
                str(
                    files_path
                ),

            "keyword_contexts":
                str(
                    contexts_path
                ),

            "numeric_csv_inventory":
                str(
                    csv_inventory_path
                ),

            "candidate_metric_summary":
                str(
                    numeric_summary_path
                ),

            "ranked_empirical_noise_sources":
                str(
                    ranked_path
                ),
        },
    }

    (
        OUT
        /
        "R11A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "TOP EMPIRICAL NOISE-FLOOR CANDIDATES:"
    )

    if len(ranked_df):
        print(
            ranked_df[
                [
                    "noise_floor_relevance_score",
                    "path",
                    "column",
                    "n",
                    "median",
                    "p90",
                    "p95",
                    "max",
                    "relevance_reasons",
                ]
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )
    else:
        print(
            "No numeric residual candidates were summarized."
        )

    print(
        "\nSTRONG CANDIDATE COUNT:"
    )

    print(
        int(
            len(
                strong
            )
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR11-A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
