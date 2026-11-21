
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R12_ROOT = REVISION_ROOT / "R12_track_accounting"

R12E_V5_DIR = (
    R12_ROOT
    / "R12E_v5_exact_testreal_filter_rule_reconstruction"
)
R12E_V5_MANIFEST = R12E_V5_DIR / "R12E_V5_MANIFEST.json"

OUT = (
    R12_ROOT
    / "R12E_v6_exact_final_filter_source_provenance"
)
OUT.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
    REVISION_ROOT,
]

TEXT_SUFFIXES = {
    ".py",
    ".txt",
    ".md",
    ".json",
}

MAX_BYTES = 15_000_000

# Strong fingerprints of the final BIRDSAI neural dataset builder.
FINGERPRINTS = [
    "final_neural_trajectory_dataset",
    "2s_to_2s_birdsai_test_metadata.csv",
    "2s_to_5s_birdsai_test_metadata.csv",
    "birdsai_test_metadata",
    "history_canonical",
    "future_canonical",
    "track_key",
    "quality_tier",
    "repaired_motion_pct",
    "accepted",
]

FILTER_TERMS = [
    "quality_tier",
    "repaired_motion_pct",
    "accepted",
    "core",
    "extended",
    "reject_quality",
    "invalid_motion",
    "<= 10",
    "<10",
    "<=10",
    "10.0",
    "20.0",
    "filter",
    "query",
    "loc[",
    "drop",
    "continue",
    "test_real",
    "testreal",
    "split",
]

SKIP_TOKENS = [
    "site-packages",
    "__pycache__",
    ".git",
    "node_modules",
]


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


def verify_v5():
    require(R12E_V5_MANIFEST)

    m = json.loads(
        R12E_V5_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R12E_V5_FINAL_FILTER_NOT_YET_EXACTLY_RESOLVED_REVIEW_CANDIDATES"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R12-E-V5 status: {m.get('status')}"
        )

    return m


def should_skip(path):
    low = str(path).lower()

    return any(
        token.lower() in low
        for token in SKIP_TOKENS
    )


def scan_sources():
    ranked_rows = []
    context_rows = []

    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.glob("**/*"):
            if not p.is_file():
                continue

            if should_skip(p):
                continue

            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue

            try:
                if p.stat().st_size > MAX_BYTES:
                    continue
            except Exception:
                continue

            key = str(p.resolve()).lower()

            if key in seen:
                continue

            seen.add(key)

            try:
                text = p.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            low = text.lower()

            fp_hits = [
                token
                for token in FINGERPRINTS
                if token.lower() in low
            ]

            filter_hits = [
                token
                for token in FILTER_TERMS
                if token.lower() in low
            ]

            # Ignore weak generic-only hits.
            if not fp_hits:
                continue

            score = 0
            reasons = []

            for token in fp_hits:
                if token in {
                    "2s_to_2s_birdsai_test_metadata.csv",
                    "2s_to_5s_birdsai_test_metadata.csv",
                }:
                    score += 100
                    reasons.append(
                        "exact_output_filename"
                    )

                elif token == "final_neural_trajectory_dataset":
                    score += 50
                    reasons.append(
                        "final_dataset_directory"
                    )

                elif token in {
                    "history_canonical",
                    "future_canonical",
                }:
                    score += 25
                    reasons.append(
                        "canonical_array_schema"
                    )

                elif token in {
                    "quality_tier",
                    "repaired_motion_pct",
                    "accepted",
                }:
                    score += 20
                    reasons.append(
                        f"filter_field:{token}"
                    )

                elif token == "track_key":
                    score += 15
                    reasons.append(
                        "track_key_schema"
                    )

                else:
                    score += 10

            if p.suffix.lower() == ".py":
                score += 20
                reasons.append(
                    "executable_python"
                )

            if "final" in p.name.lower():
                score += 10
                reasons.append(
                    "final_in_filename"
                )

            ranked_rows.append(
                {
                    "path":
                        str(p),

                    "sha256":
                        sha256_file(p),

                    "suffix":
                        p.suffix.lower(),

                    "size_bytes":
                        int(
                            p.stat().st_size
                        ),

                    "score":
                        int(
                            score
                        ),

                    "fingerprint_hits":
                        "|".join(
                            fp_hits
                        ),

                    "filter_term_hits":
                        "|".join(
                            filter_hits
                        ),

                    "reasons":
                        "|".join(
                            reasons
                        ),
                }
            )

            lines = text.splitlines()

            for i, line in enumerate(
                lines,
                start=1,
            ):
                ll = line.lower()

                if not any(
                    term.lower() in ll
                    for term in (
                        FINGERPRINTS
                        +
                        FILTER_TERMS
                    )
                ):
                    continue

                start = max(
                    1,
                    i - 7,
                )

                end = min(
                    len(lines),
                    i + 7,
                )

                context_rows.append(
                    {
                        "source_file":
                            str(p),

                        "source_sha256":
                            sha256_file(p),

                        "anchor_line":
                            i,

                        "anchor_text":
                            line,

                        "context_start":
                            start,

                        "context_end":
                            end,

                        "context":
                            "\n".join(
                                f"{j}: {lines[j-1]}"
                                for j in range(
                                    start,
                                    end + 1,
                                )
                            ),
                    }
                )

    ranked = pd.DataFrame(
        ranked_rows
    )

    if len(ranked):
        ranked = ranked.sort_values(
            [
                "score",
                "path",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(
            drop=True
        )

    contexts = pd.DataFrame(
        context_rows
    )

    return ranked, contexts


def extract_filter_like_lines(ranked):
    rows = []

    if len(ranked) == 0:
        return pd.DataFrame()

    top = ranked.head(
        20
    )

    for _, rec in top.iterrows():
        p = Path(
            str(
                rec[
                    "path"
                ]
            )
        )

        if p.suffix.lower() != ".py":
            continue

        try:
            lines = p.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except Exception:
            continue

        for i, line in enumerate(
            lines,
            start=1,
        ):
            ll = line.lower()

            # High-value lines only.
            if not any(
                term in ll
                for term in [
                    "quality_tier",
                    "repaired_motion_pct",
                    "accepted",
                    "core",
                    "extended",
                    "reject_quality",
                    "invalid_motion",
                    "test_real",
                    "final_neural_trajectory_dataset",
                    "birdsai_test_metadata",
                ]
            ):
                continue

            assignment_or_condition = (
                "=" in line
                or
                "if " in ll
                or
                "query(" in ll
                or
                ".loc[" in ll
                or
                "continue" in ll
                or
                "drop" in ll
            )

            if not assignment_or_condition:
                continue

            rows.append(
                {
                    "source_file":
                        str(p),

                    "source_sha256":
                        str(
                            rec[
                                "sha256"
                            ]
                        ),

                    "line_number":
                        i,

                    "line":
                        line,
                }
            )

    return pd.DataFrame(
        rows
    )


def classify_source_rule(
    ranked,
    rule_lines,
):
    """
    Conservative source-level decision.
    Do not infer an exact rule unless an executable source has both:
      - final dataset/output fingerprint
      - explicit filtering condition involving one of the candidate fields.
    """
    evidence = []

    if len(ranked) == 0:
        return pd.DataFrame(), ""

    py = ranked[
        ranked[
            "suffix"
        ]
        ==
        ".py"
    ].copy()

    for _, rec in py.iterrows():
        path = Path(
            str(
                rec[
                    "path"
                ]
            )
        )

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        low = text.lower()

        output_fp = any(
            token in low
            for token in [
                "final_neural_trajectory_dataset",
                "2s_to_2s_birdsai_test_metadata.csv",
                "2s_to_5s_birdsai_test_metadata.csv",
                "birdsai_test_metadata",
            ]
        )

        if not output_fp:
            continue

        candidates = []

        # Explicit semantic patterns only.
        patterns = [
            (
                "quality_tier_core",
                r"quality_tier[^\\n]{0,120}(?:==|isin|query|core)[^\\n]{0,120}core",
            ),
            (
                "accepted_equals_1",
                r"accepted[^\\n]{0,120}(?:==|isin|query)[^\\n]{0,80}(?:1|true)",
            ),
            (
                "repaired_motion_pct_le_10",
                r"repaired_motion_pct[^\\n]{0,120}(?:<=|<)[^\\n]{0,40}10(?:\\.0)?",
            ),
        ]

        for rule_name, pattern in patterns:
            matches = list(
                re.finditer(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                )
            )

            for m in matches:
                line_num = (
                    text[
                        :m.start()
                    ].count(
                        "\n"
                    )
                    +
                    1
                )

                line = text.splitlines()[
                    line_num
                    -
                    1
                ]

                candidates.append(
                    {
                        "source_file":
                            str(path),

                        "source_sha256":
                            sha256_file(
                                path
                            ),

                        "rule_name":
                            rule_name,

                        "line_number":
                            line_num,

                        "line":
                            line,

                        "pattern":
                            pattern,
                    }
                )

        evidence.extend(
            candidates
        )

    ev = pd.DataFrame(
        evidence
    )

    exact_rule = ""

    if len(ev):
        # If the same source explicitly contains one candidate rule and not
        # competing final-filter rules, we can lock it. Otherwise remain diagnostic.
        rule_names = sorted(
            ev[
                "rule_name"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(
            rule_names
        ) == 1:
            exact_rule = rule_names[
                0
            ]

    return ev, exact_rule


def main():
    print(
        "\nR12-E-V6 — EXACT FINAL-FILTER SOURCE PROVENANCE AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R12-E-V5 showed that multiple candidate filters are empirically equivalent on "
        "BIRDSAI TestReal, so data alone cannot identify the implemented filter. This step "
        "searches the original experiment source tree using exact final-dataset output/schema "
        "fingerprints and extracts executable filtering conditions. It does not infer a rule "
        "from equivalent data subsets.\n"
    )

    verify_v5()

    ranked, contexts = scan_sources()

    rule_lines = extract_filter_like_lines(
        ranked
    )

    explicit_evidence, exact_rule = classify_source_rule(
        ranked,
        rule_lines,
    )

    ranked_path = (
        OUT
        /
        "R12E_V6_RANKED_FINAL_DATASET_BUILDER_CANDIDATES.csv"
    )

    contexts_path = (
        OUT
        /
        "R12E_V6_FINAL_FILTER_CONTEXT_EVIDENCE.csv"
    )

    rule_lines_path = (
        OUT
        /
        "R12E_V6_TOP_EXECUTABLE_FILTER_LINES.csv"
    )

    explicit_path = (
        OUT
        /
        "R12E_V6_EXPLICIT_FINAL_FILTER_RULE_EVIDENCE.csv"
    )

    ranked.to_csv(
        ranked_path,
        index=False,
    )

    contexts.to_csv(
        contexts_path,
        index=False,
    )

    rule_lines.to_csv(
        rule_lines_path,
        index=False,
    )

    explicit_evidence.to_csv(
        explicit_path,
        index=False,
    )

    if exact_rule:
        status = (
            "R12E_V6_EXACT_IMPLEMENTED_FINAL_FILTER_RULE_LOCATED_READY_FOR_R12E_V7_FULL_RECONSTRUCTION"
        )

    elif len(
        explicit_evidence
    ):
        status = (
            "R12E_V6_MULTIPLE_EXPLICIT_FILTER_RULES_FOUND_REVIEW_EXECUTION_ORDER_BEFORE_R12E_V7"
        )

    else:
        status = (
            "R12E_V6_EXACT_FINAL_FILTER_RULE_NOT_YET_LOCATED_REVIEW_TOP_SOURCE_CANDIDATES"
        )

    manifest = {
        "step":
            "R12E_V6_EXACT_FINAL_FILTER_SOURCE_PROVENANCE_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "empirical_equivalence_not_used_as_proof":
            True,

        "exact_rule":
            exact_rule,

        "explicit_rule_evidence_rows":
            int(
                len(
                    explicit_evidence
                )
            ),

        "outputs": {
            "ranked_builder_candidates":
                str(
                    ranked_path
                ),

            "context_evidence":
                str(
                    contexts_path
                ),

            "top_executable_filter_lines":
                str(
                    rule_lines_path
                ),

            "explicit_filter_rule_evidence":
                str(
                    explicit_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_V6_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "TOP FINAL-DATASET BUILDER CANDIDATES:"
    )

    if len(
        ranked
    ):
        print(
            ranked[
                [
                    "score",
                    "path",
                    "sha256",
                    "fingerprint_hits",
                    "filter_term_hits",
                    "reasons",
                ]
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )
    else:
        print(
            "No candidate source files found."
        )

    print(
        "\nTOP EXECUTABLE FILTER LINES:"
    )

    if len(
        rule_lines
    ):
        print(
            rule_lines.head(
                120
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No high-value executable filter lines extracted."
        )

    print(
        "\nEXPLICIT FINAL-FILTER RULE EVIDENCE:"
    )

    if len(
        explicit_evidence
    ):
        print(
            explicit_evidence.to_string(
                index=False
            )
        )
    else:
        print(
            "No explicit final-filter condition identified yet."
        )

    print(
        "\nEXACT IMPLEMENTED RULE:"
    )

    print(
        exact_rule
        if exact_rule
        else
        "Not uniquely source-locked yet."
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-E-V6 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
