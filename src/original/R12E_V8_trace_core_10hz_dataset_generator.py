
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

V7_DIR = (
    R12_ROOT
    / "R12E_v7_direct_final_builder_ast_audit"
)
V7_MANIFEST = V7_DIR / "R12E_V7_MANIFEST.json"

OUT = (
    R12_ROOT
    / "R12E_v8_trace_core_10hz_dataset_generator"
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

MAX_BYTES = 20_000_000

TARGET_FILENAMES = [
    "birdsai_test_real_core_2s_to_2s_10hz.npz",
    "birdsai_test_real_core_2s_to_2s_10hz_metadata.csv",
    "birdsai_test_real_core_2s_to_5s_10hz.npz",
    "birdsai_test_real_core_2s_to_5s_10hz_metadata.csv",
]

TARGET_STEMS = [
    "birdsai_test_real_core",
    "birdsai_train_real_core",
    "core_2s_to_2s_10hz",
    "core_2s_to_5s_10hz",
    "10hz_metadata",
]

FILTER_TERMS = [
    "quality_tier",
    "repaired_motion_pct",
    "accepted",
    "core",
    "extended",
    "reject_quality",
    "invalid_motion",
    "10hz",
    "10 hz",
    "resample",
    "interpol",
    "split",
    "test_real",
    "testreal",
]

WRITE_TERMS = [
    "np.savez",
    "np.savez_compressed",
    "to_csv",
    "write_text",
    "open(",
]

READ_TERMS = [
    "np.load",
    "read_csv",
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


def verify_v7():
    require(V7_MANIFEST)

    m = json.loads(
        V7_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R12E_V7_BUILDER_CONTEXT_EXTRACTED_BUT_FILTER_EXPRESSION_STILL_NEEDS_MANUAL_LINE_REVIEW"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R12-E-V7 status: {m.get('status')}"
        )

    return m


def should_skip(path):
    low = str(path).lower()

    return any(
        token.lower() in low
        for token in SKIP_TOKENS
    )


def scan_sources():
    file_rows = []
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

            exact_hits = [
                name
                for name in TARGET_FILENAMES
                if name.lower() in low
            ]

            stem_hits = [
                stem
                for stem in TARGET_STEMS
                if stem.lower() in low
            ]

            if not exact_hits and not stem_hits:
                continue

            write_hits = [
                term
                for term in WRITE_TERMS
                if term.lower() in low
            ]

            read_hits = [
                term
                for term in READ_TERMS
                if term.lower() in low
            ]

            filter_hits = [
                term
                for term in FILTER_TERMS
                if term.lower() in low
            ]

            score = 0
            reasons = []

            score += 100 * len(
                exact_hits
            )

            score += 25 * len(
                stem_hits
            )

            if write_hits:
                score += 80
                reasons.append(
                    "contains_write_operation"
                )

            if read_hits:
                score += 20
                reasons.append(
                    "contains_read_operation"
                )

            if p.suffix.lower() == ".py":
                score += 30
                reasons.append(
                    "executable_python"
                )

            if filter_hits:
                score += 5 * len(
                    filter_hits
                )

            file_rows.append(
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

                    "exact_filename_hits":
                        "|".join(
                            exact_hits
                        ),

                    "stem_hits":
                        "|".join(
                            stem_hits
                        ),

                    "write_hits":
                        "|".join(
                            write_hits
                        ),

                    "read_hits":
                        "|".join(
                            read_hits
                        ),

                    "filter_hits":
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

                matched_target = [
                    token
                    for token in (
                        TARGET_FILENAMES
                        +
                        TARGET_STEMS
                    )
                    if token.lower() in ll
                ]

                matched_rule = [
                    token
                    for token in (
                        FILTER_TERMS
                        +
                        WRITE_TERMS
                    )
                    if token.lower() in ll
                ]

                if not matched_target and not matched_rule:
                    continue

                start = max(
                    1,
                    i - 12,
                )

                end = min(
                    len(lines),
                    i + 12,
                )

                context_rows.append(
                    {
                        "source_file":
                            str(p),

                        "source_sha256":
                            sha256_file(p),

                        "anchor_line":
                            i,

                        "matched_target_terms":
                            "|".join(
                                matched_target
                            ),

                        "matched_rule_terms":
                            "|".join(
                                matched_rule
                            ),

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

    files_df = pd.DataFrame(
        file_rows
    )

    if len(
        files_df
    ):
        files_df = (
            files_df
            .sort_values(
                [
                    "score",
                    "path",
                ],
                ascending=[
                    False,
                    True,
                ]
            )
            .reset_index(
                drop=True
            )
        )

    context_df = pd.DataFrame(
        context_rows
    )

    return files_df, context_df


def locate_writer_candidates(
    files_df,
):
    if len(
        files_df
    ) == 0:
        return pd.DataFrame()

    rows = []

    for _, row in files_df.iterrows():
        if row[
            "suffix"
        ] != ".py":
            continue

        p = Path(
            str(
                row[
                    "path"
                ]
            )
        )

        try:
            text = p.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        low = text.lower()

        target_signal = any(
            token.lower() in low
            for token in TARGET_STEMS
        )

        write_signal = any(
            token.lower() in low
            for token in [
                "np.savez",
                "np.savez_compressed",
                ".to_csv",
            ]
        )

        if not (
            target_signal
            and
            write_signal
        ):
            continue

        # Strong candidate if same script also contains filtering vocabulary.
        filter_signal = any(
            token.lower() in low
            for token in [
                "quality_tier",
                "repaired_motion_pct",
                "accepted",
                "core",
            ]
        )

        rows.append(
            {
                "path":
                    str(p),

                "sha256":
                    sha256_file(p),

                "score":
                    int(
                        row[
                            "score"
                        ]
                    ),

                "target_signal":
                    target_signal,

                "write_signal":
                    write_signal,

                "filter_signal":
                    filter_signal,

                "exact_filename_hits":
                    row[
                        "exact_filename_hits"
                    ],

                "stem_hits":
                    row[
                        "stem_hits"
                    ],

                "filter_hits":
                    row[
                        "filter_hits"
                    ],
            }
        )

    out = pd.DataFrame(
        rows
    )

    if len(
        out
    ):
        out = out.sort_values(
            [
                "filter_signal",
                "score",
                "path",
            ],
            ascending=[
                False,
                False,
                True,
            ]
        ).reset_index(
            drop=True
        )

    return out


def extract_writer_rule_lines(
    writers,
):
    rows = []

    if len(
        writers
    ) == 0:
        return pd.DataFrame()

    for _, rec in writers.head(
        15
    ).iterrows():
        p = Path(
            str(
                rec[
                    "path"
                ]
            )
        )

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

            interesting = any(
                term.lower() in ll
                for term in [
                    "quality_tier",
                    "repaired_motion_pct",
                    "accepted",
                    "core",
                    "extended",
                    "reject_quality",
                    "invalid_motion",
                    "np.savez",
                    "np.savez_compressed",
                    "to_csv",
                    "10hz",
                    "10 hz",
                    "resample",
                ]
            )

            if not interesting:
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


def writer_confidence(
    writers,
):
    if len(
        writers
    ) == 0:
        return pd.DataFrame(), ""

    best = writers.iloc[
        0
    ]

    exact_rule_source = ""

    # Conservative: require a Python writer with target + write + filter signals.
    if bool(
        best[
            "target_signal"
        ]
    ) and bool(
        best[
            "write_signal"
        ]
    ) and bool(
        best[
            "filter_signal"
        ]
    ):
        exact_rule_source = str(
            best[
                "path"
            ]
        )

    return (
        pd.DataFrame(
            [
                best.to_dict()
            ]
        ),
        exact_rule_source,
    )


def main():
    print(
        "\nR12-E-V8 — TRACE GENERATOR OF PRE-FILTERED 'CORE' 10-Hz DATASETS"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R12-E-V7 showed that build_final_neural_dataset.py consumes files already named "
        "birdsai_*_core_*_10hz rather than applying the core filter itself. This step traces "
        "those exact/stem filenames upstream to the Python writer that generated them, then "
        "extracts the writer's core/quality/acceptance/resampling logic. No original file is modified.\n"
    )

    verify_v7()

    files_df, context_df = scan_sources()

    writers = locate_writer_candidates(
        files_df
    )

    rule_lines = extract_writer_rule_lines(
        writers
    )

    best_df, exact_rule_source = writer_confidence(
        writers
    )

    files_path = (
        OUT
        /
        "R12E_V8_RANKED_CORE_10HZ_PROVENANCE_SOURCES.csv"
    )

    contexts_path = (
        OUT
        /
        "R12E_V8_CORE_10HZ_CONTEXT_EVIDENCE.csv"
    )

    writers_path = (
        OUT
        /
        "R12E_V8_CORE_10HZ_WRITER_CANDIDATES.csv"
    )

    rule_lines_path = (
        OUT
        /
        "R12E_V8_WRITER_CORE_FILTER_RULE_LINES.csv"
    )

    best_path = (
        OUT
        /
        "R12E_V8_TOP_WRITER_LOCK.csv"
    )

    files_df.to_csv(
        files_path,
        index=False
    )

    context_df.to_csv(
        contexts_path,
        index=False
    )

    writers.to_csv(
        writers_path,
        index=False
    )

    rule_lines.to_csv(
        rule_lines_path,
        index=False
    )

    best_df.to_csv(
        best_path,
        index=False
    )

    if exact_rule_source:
        status = (
            "R12E_V8_CORE_10HZ_WRITER_LOCATED_READY_FOR_R12E_V9_EXACT_FILTER_EXPRESSION_LOCK"
        )

    elif len(
        writers
    ):
        status = (
            "R12E_V8_CORE_10HZ_WRITER_CANDIDATES_FOUND_BUT_NOT_UNIQUELY_LOCKED_REVIEW_TOP_WRITER"
        )

    else:
        status = (
            "R12E_V8_CORE_10HZ_WRITER_NOT_FOUND_REVIEW_GENERATION_PROVENANCE"
        )

    manifest = {
        "step":
            "R12E_V8_TRACE_CORE_10HZ_DATASET_GENERATOR",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "final_builder_is_filter_stage":
            False,

        "core_filter_occurs_upstream":
            True,

        "top_writer_source":
            exact_rule_source,

        "outputs": {
            "ranked_sources":
                str(
                    files_path
                ),

            "context_evidence":
                str(
                    contexts_path
                ),

            "writer_candidates":
                str(
                    writers_path
                ),

            "writer_rule_lines":
                str(
                    rule_lines_path
                ),

            "top_writer_lock":
                str(
                    best_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_V8_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8"
    )

    print(
        "TOP CORE-10HZ PROVENANCE SOURCES:"
    )

    if len(
        files_df
    ):
        print(
            files_df[
                [
                    "score",
                    "path",
                    "sha256",
                    "exact_filename_hits",
                    "stem_hits",
                    "write_hits",
                    "read_hits",
                    "filter_hits",
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
            "No matching provenance source found."
        )

    print(
        "\nCORE-10HZ WRITER CANDIDATES:"
    )

    if len(
        writers
    ):
        print(
            writers.to_string(
                index=False
            )
        )
    else:
        print(
            "No Python writer candidate found."
        )

    print(
        "\nTOP WRITER RULE LINES:"
    )

    if len(
        rule_lines
    ):
        print(
            rule_lines.head(
                160
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No writer rule lines extracted."
        )

    print(
        "\nTOP WRITER SOURCE:"
    )

    print(
        exact_rule_source
        if exact_rule_source
        else
        "Not uniquely locked yet."
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-E-V8 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
