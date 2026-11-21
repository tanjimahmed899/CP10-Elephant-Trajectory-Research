
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

R12C_V2_DIR = (
    R12_ROOT
    / "R12C_v2_resolve_45_total_tracks_with_canonical_id_mapping"
)
R12C_V2_MANIFEST = R12C_V2_DIR / "R12C_V2_MANIFEST.json"

OUT = (
    R12_ROOT
    / "R12D_segment_build_rule_source_provenance_audit"
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

MAX_BYTES = 10_000_000

# Exact metadata column names are powerful fingerprints for the builder.
STRONG_FINGERPRINTS = [
    "segment_id",
    "track_key",
    "obs_start_frame",
    "obs_end_frame",
    "future_end_frame",
    "accepted",
    "quality_tier",
    "repaired_motion_pct",
]

# Continuity / segmentation concepts we want to recover from executable code.
RULE_TERMS = [
    "segment_id",
    "track_key",
    "np.diff",
    "diff(",
    "frame_gap",
    "max_gap",
    "gap",
    "contiguous",
    "continuity",
    "consecutive",
    "resample",
    "10hz",
    "10 hz",
    "history",
    "future",
    "horizon",
    "obs_start_frame",
    "obs_end_frame",
    "future_end_frame",
    "min_len",
    "min_length",
    "window",
    "stride",
    "accepted",
    "quality_tier",
    "repaired_motion_pct",
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


def verify_r12c_v2():
    require(R12C_V2_MANIFEST)

    m = json.loads(
        R12C_V2_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R12C_V2_45_TOTAL_TRACKS_37_2S_ELIGIBLE_31_5S_ELIGIBLE_EXACTLY_RESOLVED_READY_FOR_R12D_SEGMENT_BUILD_RULE_LOCK"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R12-C-V2 status: {m.get('status')}"
        )

    return m


def should_skip(path):
    low = str(path).lower()

    return any(
        token.lower() in low
        for token in SKIP_TOKENS
    )


def scan_source_files():
    rows = []
    evidence_rows = []

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

            strong_hits = [
                token
                for token in STRONG_FINGERPRINTS
                if token.lower() in low
            ]

            rule_hits = [
                token
                for token in RULE_TERMS
                if token.lower() in low
            ]

            if not strong_hits and not rule_hits:
                continue

            score = 0
            reasons = []

            score += 10 * len(strong_hits)

            if "segment_id" in low:
                score += 20
                reasons.append("segment_id")

            if "track_key" in low:
                score += 20
                reasons.append("track_key")

            if "obs_start_frame" in low:
                score += 15
                reasons.append("obs_start_frame")

            if "future_end_frame" in low:
                score += 15
                reasons.append("future_end_frame")

            if "np.diff" in low or "diff(" in low:
                score += 10
                reasons.append("frame_difference_logic")

            if "gap" in low:
                score += 10
                reasons.append("gap_logic")

            if "contiguous" in low or "continuity" in low or "consecutive" in low:
                score += 10
                reasons.append("continuity_logic")

            if "resample" in low or "10hz" in low or "10 hz" in low:
                score += 5
                reasons.append("resampling_logic")

            if p.suffix.lower() == ".py":
                score += 10
                reasons.append("executable_python")

            if "final_neural" in p.name.lower():
                score += 10
                reasons.append("final_neural_name")

            if "trajectory" in p.name.lower():
                score += 5
                reasons.append("trajectory_name")

            rows.append(
                {
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "suffix": p.suffix.lower(),
                    "size_bytes": int(p.stat().st_size),
                    "score": score,
                    "strong_fingerprint_hits": "|".join(strong_hits),
                    "rule_term_hits": "|".join(rule_hits),
                    "reasons": "|".join(reasons),
                }
            )

            lines = text.splitlines()

            interesting_indices = []

            for i, line in enumerate(lines):
                ll = line.lower()

                if any(
                    term.lower() in ll
                    for term in RULE_TERMS
                ):
                    interesting_indices.append(i)

            # Save compact context blocks around every matching line.
            covered = set()

            for i in interesting_indices:
                start = max(0, i - 4)
                end = min(len(lines), i + 5)

                block_key = (
                    str(p),
                    start,
                    end,
                )

                if block_key in covered:
                    continue

                covered.add(block_key)

                block = "\n".join(
                    f"{j + 1}: {lines[j]}"
                    for j in range(start, end)
                )

                evidence_rows.append(
                    {
                        "source_file": str(p),
                        "context_start_line": start + 1,
                        "context_end_line": end,
                        "anchor_line": i + 1,
                        "anchor_text": lines[i],
                        "context": block,
                    }
                )

    source_df = pd.DataFrame(rows)

    if len(source_df):
        source_df = source_df.sort_values(
            [
                "score",
                "path",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(drop=True)

    evidence_df = pd.DataFrame(
        evidence_rows
    )

    return source_df, evidence_df


def scan_csv_metadata_provenance():
    """
    Search CSVs for builder-like fields and values that can help identify
    segmentation rule consequences without assuming the rule.
    """
    rows = []
    seen = set()

    for root in [
        MAIN1,
        MAIN2,
    ]:
        if not root.exists():
            continue

        for p in root.glob("**/*.csv"):
            if should_skip(p):
                continue

            key = str(p.resolve()).lower()

            if key in seen:
                continue

            seen.add(key)

            try:
                if p.stat().st_size > 100_000_000:
                    continue

                head = pd.read_csv(
                    p,
                    nrows=20,
                    low_memory=False,
                )
            except Exception:
                continue

            cols = [
                str(c)
                for c in head.columns
            ]

            low_cols = {
                str(c).lower()
                for c in head.columns
            }

            fingerprint_count = sum(
                1
                for token in STRONG_FINGERPRINTS
                if token.lower() in low_cols
            )

            if fingerprint_count == 0:
                continue

            rows.append(
                {
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "size_bytes": int(p.stat().st_size),
                    "fingerprint_column_count": fingerprint_count,
                    "columns": "|".join(cols),
                }
            )

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            [
                "fingerprint_column_count",
                "path",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(drop=True)

    return out


def inspect_top_python_candidates(source_df, evidence_df):
    if len(source_df) == 0:
        return pd.DataFrame()

    py = source_df[
        source_df["suffix"] == ".py"
    ].head(15)

    rows = []

    for _, cand in py.iterrows():
        p = Path(
            str(cand["path"])
        )

        try:
            lines = p.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except Exception:
            continue

        # Collect explicit assignments/constants near segmentation words.
        for i, line in enumerate(lines, start=1):
            ll = line.lower()

            if not any(
                token in ll
                for token in [
                    "gap",
                    "segment",
                    "history",
                    "future",
                    "horizon",
                    "stride",
                    "resample",
                    "fps",
                    "hz",
                    "min_len",
                    "min_length",
                    "window",
                ]
            ):
                continue

            # Assignment-like lines are especially useful.
            if (
                "=" in line
                or
                line.lstrip().startswith("def ")
                or
                line.lstrip().startswith("for ")
                or
                line.lstrip().startswith("if ")
            ):
                rows.append(
                    {
                        "source_file": str(p),
                        "line_number": i,
                        "line": line,
                    }
                )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR12-D — SEGMENT BUILD-RULE SOURCE PROVENANCE AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R12-C-V2 has already resolved the population accounting: 45 total TestReal "
        "tracks, 37 represented at 2s, 31 represented at 5s, producing 89/63 forecast "
        "segments. This step does NOT infer the segmentation rule from counts. Instead, "
        "it searches the original experiment code/provenance for the exact executable "
        "logic that creates segment_id / track_key and applies continuity, gap, resampling, "
        "history, and future-window constraints.\n"
    )

    verify_r12c_v2()

    source_df, evidence_df = scan_source_files()
    csv_df = scan_csv_metadata_provenance()
    assignment_df = inspect_top_python_candidates(
        source_df,
        evidence_df,
    )

    source_path = (
        OUT
        /
        "R12D_RANKED_SEGMENT_BUILDER_SOURCE_CANDIDATES.csv"
    )

    evidence_path = (
        OUT
        /
        "R12D_SEGMENT_RULE_CONTEXT_EVIDENCE.csv"
    )

    csv_path = (
        OUT
        /
        "R12D_METADATA_PROVENANCE_CSV_CANDIDATES.csv"
    )

    assignment_path = (
        OUT
        /
        "R12D_TOP_PYTHON_RULE_ASSIGNMENTS.csv"
    )

    source_df.to_csv(
        source_path,
        index=False
    )

    evidence_df.to_csv(
        evidence_path,
        index=False
    )

    csv_df.to_csv(
        csv_path,
        index=False
    )

    assignment_df.to_csv(
        assignment_path,
        index=False
    )

    # Conservative readiness rule:
    # require at least one Python file containing the exact metadata fingerprints
    # plus segment_id and track_key.
    ready = False
    top_exact = pd.DataFrame()

    if len(source_df):
        py = source_df[
            source_df["suffix"] == ".py"
        ].copy()

        if len(py):
            mask = (
                py[
                    "strong_fingerprint_hits"
                ]
                .astype(str)
                .str.contains(
                    "segment_id",
                    regex=False
                )
                &
                py[
                    "strong_fingerprint_hits"
                ]
                .astype(str)
                .str.contains(
                    "track_key",
                    regex=False
                )
                &
                py[
                    "strong_fingerprint_hits"
                ]
                .astype(str)
                .str.contains(
                    "obs_start_frame",
                    regex=False
                )
                &
                py[
                    "strong_fingerprint_hits"
                ]
                .astype(str)
                .str.contains(
                    "future_end_frame",
                    regex=False
                )
            )

            top_exact = py[
                mask
            ].copy()

            ready = len(
                top_exact
            ) > 0

    if ready:
        status = (
            "R12D_SEGMENT_BUILDER_SOURCE_LOCATED_READY_FOR_R12E_EXACT_RULE_EXTRACTION_AND_RECONSTRUCTION"
        )
    else:
        status = (
            "R12D_SEGMENT_BUILDER_SOURCE_NOT_UNIQUELY_LOCATED_REVIEW_TOP_CANDIDATES_BEFORE_RECONSTRUCTION"
        )

    top_exact_path = (
        OUT
        /
        "R12D_EXACT_BUILDER_CANDIDATES.csv"
    )

    top_exact.to_csv(
        top_exact_path,
        index=False
    )

    manifest = {
        "step":
            "R12D_SEGMENT_BUILD_RULE_SOURCE_PROVENANCE_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "rule_inferred_from_counts":
            False,

        "exact_builder_candidate_count":
            int(
                len(
                    top_exact
                )
            ),

        "scientific_guardrail":
            (
                "Do not describe the continuity/gap rule until executable source "
                "evidence is recovered. Counts alone are insufficient to infer the rule."
            ),

        "outputs": {
            "ranked_source_candidates":
                str(
                    source_path
                ),

            "rule_context_evidence":
                str(
                    evidence_path
                ),

            "metadata_csv_candidates":
                str(
                    csv_path
                ),

            "top_python_assignments":
                str(
                    assignment_path
                ),

            "exact_builder_candidates":
                str(
                    top_exact_path
                ),
        },
    }

    (
        OUT
        /
        "R12D_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "TOP SEGMENT-BUILDER SOURCE CANDIDATES:"
    )

    if len(source_df):
        print(
            source_df[
                [
                    "score",
                    "path",
                    "strong_fingerprint_hits",
                    "rule_term_hits",
                    "reasons",
                ]
            ]
            .head(
                25
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
        "\nEXACT BUILDER CANDIDATES:"
    )

    if len(top_exact):
        print(
            top_exact[
                [
                    "score",
                    "path",
                    "sha256",
                    "strong_fingerprint_hits",
                    "rule_term_hits",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No Python file yet satisfies the exact builder fingerprint."
        )

    print(
        "\nTOP PYTHON RULE ASSIGNMENTS / CONDITIONS:"
    )

    if len(assignment_df):
        print(
            assignment_df.head(
                80
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No assignment/condition evidence extracted."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-D COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
