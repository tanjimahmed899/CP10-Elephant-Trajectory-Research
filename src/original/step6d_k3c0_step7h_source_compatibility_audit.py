from pathlib import Path
import ast
import hashlib
import json
import re

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

STEP7_DIR = (
    DOWNLOADS
    / "step7_response_aware_warning"
)

FINAL_MANIFEST = (
    STEP7_DIR
    / "STEP7_FINAL_DEVELOPMENT_MANIFEST.json"
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

OUT = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_operational_warning"
    / "source_compatibility_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

KEYWORDS = [
    "N=2",
    "persistence",
    "RED",
    "YELLOW",
    "GREEN",
    "response",
    "threshold",
    "geofence",
    "boundary",
    "q90",
    "conformal",
    "crossing",
    "tte",
    "time_to",
    "inside",
    "intersect",
    "nominal",
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


def safe_text(path):
    try:
        if path.stat().st_size > 10_000_000:
            return None

        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:
        return None


def function_candidates(path, text):
    rows = []

    try:
        tree = ast.parse(
            text
        )

    except Exception:
        return rows

    for node in ast.walk(
        tree
    ):
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

        if len(
            hits
        ) < 2:
            continue

        score = len(
            hits
        )

        if "red" in low:
            score += 2

        if "yellow" in low:
            score += 2

        if "persistence" in low:
            score += 2

        if "geofence" in low:
            score += 2

        if "tte" in low:
            score += 2

        rows.append({
            "file":
                str(
                    path
                ),

            "function":
                node.name,

            "line_start":
                int(
                    node.lineno
                ),

            "line_end":
                int(
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno
                    )
                ),

            "score":
                int(
                    score
                ),

            "hits":
                ";".join(
                    hits
                ),

            "source":
                src,
        })

    return rows


def lexical_contexts(path, text):
    rows = []
    lines = text.splitlines()

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
            i - 3
        )

        hi = min(
            len(
                lines
            ),
            i + 3
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
                str(
                    path
                ),

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


def main():
    print(
        "\nSTEP 6D-K3C0 — FROZEN STEP7H OPERATIONAL RULE SOURCE/COMPATIBILITY AUDIT"
    )

    print(
        "Read-only audit only. NO BAMBI Step-7 warning evaluation is computed.\n"
    )

    require(
        STEP7_DIR
    )

    require(
        FINAL_MANIFEST
    )

    require(
        K3B1_DIR
    )

    manifest = json.loads(
        FINAL_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    print(
        "FINAL STEP7 MANIFEST:"
    )

    print(
        json.dumps(
            manifest,
            indent=2
        )[
            :12000
        ]
    )

    # ------------------------------------------------------------
    # Search exact Step7 source / lock files.
    # ------------------------------------------------------------
    candidate_rows = []
    fn_rows = []
    lex_rows = []

    roots = [
        STEP7_DIR,
        DOWNLOADS,
    ]

    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            low_path = str(
                path
            ).lower()

            if root == DOWNLOADS:
                if "step7" not in low_path:
                    continue

            if path.suffix.lower() not in {
                ".py",
                ".json",
                ".txt",
                ".csv",
                ".md",
            }:
                continue

            key = str(
                path
            ).lower()

            if key in seen:
                continue

            seen.add(
                key
            )

            text = safe_text(
                path
            )

            if text is None:
                continue

            low = text.lower()

            hits = [
                k
                for k in KEYWORDS
                if k.lower() in low
            ]

            if not hits:
                continue

            score = len(
                hits
            )

            name_low = path.name.lower()

            if "step7h" in name_low:
                score += 20

            if "final" in name_low:
                score += 10

            if "manifest" in name_low:
                score += 8

            if "lock" in name_low:
                score += 5

            candidate_rows.append({
                "file":
                    str(
                        path
                    ),

                "name":
                    path.name,

                "score":
                    int(
                        score
                    ),

                "hits":
                    ";".join(
                        hits
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            })

            if path.suffix.lower() == ".py":
                fn_rows.extend(
                    function_candidates(
                        path,
                        text
                    )
                )

                lex_rows.extend(
                    lexical_contexts(
                        path,
                        text
                    )
                )

    candidates = pd.DataFrame(
        candidate_rows
    )

    functions = pd.DataFrame(
        fn_rows
    )

    lexical = pd.DataFrame(
        lex_rows
    )

    if len(
        candidates
    ):
        candidates = candidates.sort_values(
            [
                "score",
                "file",
            ],
            ascending=[
                False,
                True,
            ]
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

    candidates.to_csv(
        OUT
        / "TABLE_step6dk3c0_ranked_step7_sources.csv",
        index=False
    )

    functions.to_csv(
        OUT
        / "TABLE_step6dk3c0_ranked_step7_functions.csv",
        index=False
    )

    lexical.to_csv(
        OUT
        / "step6dk3c0_step7_lexical_contexts.csv",
        index=False
    )

    # ------------------------------------------------------------
    # BAMBI compatibility inventory.
    # ------------------------------------------------------------
    compatibility_rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        meta_path = (
            BAMBI_META_DIR
            / f"{task}_bambi_flight6_canonical_metadata_LOCKED.csv"
        )

        require(
            meta_path
        )

        meta = pd.read_csv(
            meta_path
        )

        columns = set(
            meta.columns
        )

        time_candidates = [
            c
            for c in meta.columns
            if any(
                token in c.lower()
                for token in [
                    "time",
                    "start",
                    "end",
                ]
            )
        ]

        identity_candidates = [
            c
            for c in meta.columns
            if any(
                token in c.lower()
                for token in [
                    "track",
                    "block",
                    "window",
                    "sample",
                ]
            )
        ]

        compatibility_rows.append({
            "task":
                task,

            "metadata_file":
                str(
                    meta_path
                ),

            "rows":
                int(
                    len(
                        meta
                    )
                ),

            "has_track_id":
                bool(
                    "track_id"
                    in
                    columns
                ),

            "has_block_key":
                bool(
                    "block_key"
                    in
                    columns
                ),

            "time_columns":
                ";".join(
                    time_candidates
                ),

            "identity_columns":
                ";".join(
                    identity_candidates
                ),
        })

    compatibility = pd.DataFrame(
        compatibility_rows
    )

    compatibility.to_csv(
        OUT
        / "TABLE_step6dk3c0_bambi_compatibility_inventory.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Confirm K3B1 fixed coverage result exists.
    # ------------------------------------------------------------
    primary_coverage = (
        K3B1_DIR
        / "TABLE_step6dk3b1v2_fixed_bambi_primary_coverage.csv"
    )

    require(
        primary_coverage
    )

    coverage = pd.read_csv(
        primary_coverage
    )

    # Step7H final rule uses 90% conformal.
    coverage90 = coverage[
        np.isclose(
            pd.to_numeric(
                coverage[
                    "alpha"
                ],
                errors="coerce"
            ),
            0.10,
            atol=1e-12,
            rtol=0
        )
    ].copy()

    coverage90.to_csv(
        OUT
        / "TABLE_step6dk3c0_bambi_fixed_q90_coverage_context.csv",
        index=False
    )

    print(
        "\nTOP STEP7 SOURCE CANDIDATES:"
    )

    print(
        candidates.head(
            30
        ).to_string(
            index=False
        )
        if len(
            candidates
        )
        else
        "NONE"
    )

    print(
        "\nTOP STEP7 FUNCTIONS:"
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
        if len(
            functions
        )
        else
        "NONE"
    )

    print(
        "\nBAMBI COMPATIBILITY INVENTORY:"
    )

    print(
        compatibility.to_string(
            index=False
        )
    )

    print(
        "\nBAMBI FIXED q90 COVERAGE CONTEXT:"
    )

    print(
        coverage90.to_string(
            index=False
        )
    )

    status = (
        "STEP7H_SOURCE_AND_BAMBI_COMPATIBILITY_CANDIDATES_FOUND_PENDING_EXACT_RULE_EXTRACTION"
        if len(
            candidates
        )
        and
        len(
            functions
        )
        else
        "STEP7H_SOURCE_NOT_RESOLVED_DO_NOT_RUN_BAMBI_WARNING_EVALUATION"
    )

    summary = {
        "step":
            "STEP_6D_K3C0_FROZEN_STEP7H_SOURCE_COMPATIBILITY_AUDIT",

        "step7_manifest":
            str(
                FINAL_MANIFEST
            ),

        "step7_manifest_sha256":
            sha256_file(
                FINAL_MANIFEST
            ),

        "candidate_source_files":
            int(
                len(
                    candidates
                )
            ),

        "candidate_functions":
            int(
                len(
                    functions
                )
            ),

        "bambi_q90_context_rows":
            int(
                len(
                    coverage90
                )
            ),

        "status":
            status,

        "rules": [
            "No Step7H tuning or warning evaluation is performed in K3C0.",
            "If BAMBI Step7 evaluation proceeds, the final locked 90% conformal + N=2 persistence + per-geofence + response-escalation rule must remain unchanged.",
            "N=2 means two consecutive available forecast windows, not two seconds.",
            "BAMBI remains a cross-species external stress test; any Step7 result is operational stress testing, not field validation.",
        ],
    }

    (
        OUT
        / "STEP6D_K3C0_SUMMARY.json"
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
        "\nSTEP 6D-K3C0 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
