
from pathlib import Path
import json
import hashlib
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
    / "R9H_A_canonicalization_source_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
]

TEXT_SUFFIXES = {
    ".py",
    ".txt",
    ".md",
    ".json",
}

PRIMARY_TERMS = [
    "canonical",
    "canonicalize",
    "heading",
    "rotation",
    "rotate",
    "angle",
    "atan2",
    "theta",
    "relative",
    "origin",
    "last observed",
    "last observation",
    "history_rel",
    "future_rel",
    "trajectory",
]

STRONG_TERMS = [
    "canonical",
    "canonicalize",
    "atan2",
    "heading",
    "rotation",
    "rotate",
]

MODEL_TERMS = [
    "transformer",
    "gru",
    "lstm",
    "trajectory",
    "forecast",
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


def scan_files():
    rows = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if (
                not p.is_file()
                or p.suffix.lower() not in TEXT_SUFFIXES
            ):
                continue

            try:
                text = p.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            low = text.lower()

            strong_hits = {
                term: low.count(term.lower())
                for term in STRONG_TERMS
                if low.count(term.lower()) > 0
            }

            model_hits = {
                term: low.count(term.lower())
                for term in MODEL_TERMS
                if low.count(term.lower()) > 0
            }

            primary_hits = {
                term: low.count(term.lower())
                for term in PRIMARY_TERMS
                if low.count(term.lower()) > 0
            }

            # Require at least one strong geometry/canonicalization term.
            if not strong_hits:
                continue

            # Prefer forecasting/model preprocessing files.
            score = (
                10 * sum(strong_hits.values())
                + 2 * sum(model_hits.values())
                + sum(primary_hits.values())
            )

            lines = text.splitlines()
            snippets = []

            for i, line in enumerate(lines, start=1):
                l = line.lower()

                if any(
                    term.lower() in l
                    for term in STRONG_TERMS
                ):
                    a = max(1, i - 8)
                    b = min(len(lines), i + 12)

                    snippets.append(
                        "\n".join(
                            f"{j}: {lines[j-1]}"
                            for j in range(a, b + 1)
                        )
                    )

                    if len(snippets) >= 12:
                        break

            rows.append(
                {
                    "filename": p.name,
                    "path": str(p),
                    "root": root.name,
                    "score": int(score),
                    "strong_hits_json": json.dumps(
                        strong_hits,
                        sort_keys=True,
                    ),
                    "model_hits_json": json.dumps(
                        model_hits,
                        sort_keys=True,
                    ),
                    "all_hits_json": json.dumps(
                        primary_hits,
                        sort_keys=True,
                    ),
                    "sha256": sha256_file(p),
                    "snippets": "\n---\n".join(snippets),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        [
            "score",
            "filename",
            "path",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort",
    )


def classify_candidate_lines(path):
    text = Path(path).read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    categories = {
        "origin_translation": [
            r"origin",
            r"last[_ ]?observ",
            r"history_rel",
            r"future_rel",
        ],
        "heading_estimation": [
            r"heading",
            r"atan2",
            r"theta",
            r"angle",
        ],
        "rotation_matrix": [
            r"cos\(",
            r"sin\(",
            r"rotation",
            r"rotate",
            r"@",
        ],
        "scale_normalization": [
            r"scale",
            r"norm\(",
            r"distance",
            r"displacement",
        ],
        "canonical_output": [
            r"canonical",
            r"history_can",
            r"future_can",
            r"history_canonical",
            r"future_canonical",
        ],
    }

    rows = []

    for i, line in enumerate(lines, start=1):
        for category, pats in categories.items():
            if any(
                re.search(
                    pat,
                    line,
                    flags=re.IGNORECASE,
                )
                for pat in pats
            ):
                rows.append(
                    {
                        "file": Path(path).name,
                        "line_number": i,
                        "category": category,
                        "line": line,
                    }
                )

    return pd.DataFrame(rows)


def build_context(path, radius=10):
    text = Path(path).read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()
    selected = set()

    for i, line in enumerate(lines, start=1):
        low = line.lower()

        if any(
            term.lower() in low
            for term in STRONG_TERMS
        ):
            for j in range(
                max(1, i - radius),
                min(len(lines), i + radius) + 1,
            ):
                selected.add(j)

    rows = []

    for j in sorted(selected):
        rows.append(
            {
                "file": Path(path).name,
                "line_number": j,
                "line": lines[j - 1],
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR9-H-A — CANONICALIZATION / HEADING / ROTATION SOURCE AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO geometry metrics, NO retraining, NO bootstrap, "
        "NO frozen modification. This step locates and locks the exact forecasting "
        "preprocessing code that defines trajectory origin, heading, rotation and "
        "canonical coordinates before measuring anisotropic-vs-isotropic distortion.\n"
    )

    candidates = scan_files()

    if len(candidates) == 0:
        raise RuntimeError(
            "No canonicalization/heading/rotation source candidates found."
        )

    # Save all candidate inventory.
    inventory_path = (
        OUT
        / "R9H_A_CANONICALIZATION_SOURCE_CANDIDATES.csv"
    )

    candidates.to_csv(
        inventory_path,
        index=False,
    )

    # Audit top candidates in detail.
    top = candidates.head(12).copy()

    classified_frames = []
    context_frames = []

    for _, row in top.iterrows():
        p = Path(row["path"])

        classified = classify_candidate_lines(
            p
        )

        if len(classified):
            classified["path"] = str(p)
            classified_frames.append(
                classified
            )

        context = build_context(
            p,
            radius=10,
        )

        if len(context):
            context["path"] = str(p)
            context_frames.append(
                context
            )

    classified_all = (
        pd.concat(
            classified_frames,
            ignore_index=True,
        )
        if classified_frames
        else
        pd.DataFrame()
    )

    context_all = (
        pd.concat(
            context_frames,
            ignore_index=True,
        )
        if context_frames
        else
        pd.DataFrame()
    )

    classified_path = (
        OUT
        / "R9H_A_TOP_CANDIDATE_CLASSIFIED_LINES.csv"
    )

    context_path = (
        OUT
        / "R9H_A_TOP_CANDIDATE_CONTEXT.csv"
    )

    classified_all.to_csv(
        classified_path,
        index=False,
    )

    context_all.to_csv(
        context_path,
        index=False,
    )

    status = (
        "R9H_A_CANONICALIZATION_SOURCE_AUDIT_COMPLETE_READY_FOR_R9H_B_EXACT_GEOMETRY_FORMULA_LOCK"
    )

    manifest = {
        "step":
            "R9H_A_CANONICALIZATION_SOURCE_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

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
                len(candidates)
            ),

        "candidate_inventory_csv":
            str(
                inventory_path
            ),

        "candidate_inventory_sha256":
            sha256_file(
                inventory_path
            ),

        "classified_lines_csv":
            str(
                classified_path
            ),

        "context_csv":
            str(
                context_path
            ),

        "top_candidate_files":
            [
                {
                    "filename": str(r["filename"]),
                    "path": str(r["path"]),
                    "score": int(r["score"]),
                    "sha256": str(r["sha256"]),
                }
                for _, r in top.iterrows()
            ],
    }

    (
        OUT
        / "R9H_A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "TOP CANONICALIZATION SOURCE CANDIDATES:"
    )

    print(
        top[
            [
                "filename",
                "score",
                "strong_hits_json",
                "model_hits_json",
                "path",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nTOP CANDIDATE CLASSIFIED LINES:"
    )

    if len(classified_all):
        print(
            classified_all.head(
                500
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No classified lines found."
        )

    print(
        "\nTOP CANDIDATE CONTEXT:"
    )

    if len(context_all):
        print(
            context_all.head(
                700
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No context lines found."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-H-A COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
