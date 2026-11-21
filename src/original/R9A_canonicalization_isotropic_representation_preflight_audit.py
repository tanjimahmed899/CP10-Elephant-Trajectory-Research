
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import re

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
]

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9A_representation_preflight_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
}

CSV_HINTS = [
    "x",
    "y",
    "width",
    "height",
    "frame_width",
    "frame_height",
    "image_width",
    "image_height",
    "bbox",
    "center",
    "cx",
    "cy",
    "normalized",
    "canonical",
    "corrected",
    "raw",
    "trajectory",
]

TEXT_PATTERNS = {
    "x_div_width": [
        r"\bx\s*/\s*(?:w|width|frame_width|image_width)\b",
        r"\bcx\s*/\s*(?:w|width|frame_width|image_width)\b",
    ],
    "y_div_height": [
        r"\by\s*/\s*(?:h|height|frame_height|image_height)\b",
        r"\bcy\s*/\s*(?:h|height|frame_height|image_height)\b",
    ],
    "normalized_xy_terms": [
        r"normalized[_\s-]*x",
        r"normalized[_\s-]*y",
        r"canonical[_\s-]*(?:x|y|trajectory)",
        r"normalize(?:d|ation)?",
    ],
    "camera_corrected_terms": [
        r"camera[_\s-]*corrected",
        r"ego[_\s-]*motion",
        r"homograph",
        r"reference[_\s-]*frame",
    ],
    "isotropic_terms": [
        r"isotropic",
        r"common[_\s-]*scale",
        r"max\s*\(\s*(?:width|w)\s*,\s*(?:height|h)\s*\)",
        r"min\s*\(\s*(?:width|w)\s*,\s*(?:height|h)\s*\)",
    ],
}


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
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def audit_text_file(path, root_label):
    text = safe_text(path)

    if not text:
        return None

    lower = text.lower()

    hits = {}

    for group, patterns in TEXT_PATTERNS.items():
        group_hits = []

        for pat in patterns:
            for m in re.finditer(
                pat,
                lower,
                flags=re.IGNORECASE,
            ):
                start = max(
                    0,
                    m.start() - 120,
                )

                end = min(
                    len(text),
                    m.end() + 180,
                )

                snippet = (
                    text[start:end]
                    .replace("\r", " ")
                    .replace("\n", " ")
                )

                group_hits.append(
                    snippet
                )

                if len(group_hits) >= 5:
                    break

            if len(group_hits) >= 5:
                break

        hits[group] = group_hits

    total = sum(
        len(v)
        for v in hits.values()
    )

    if total == 0:
        return None

    return {
        "root_label":
            root_label,

        "path":
            str(path),

        "filename":
            path.name,

        "suffix":
            path.suffix.lower(),

        "sha256":
            sha256_file(path),

        "x_div_width_hits":
            len(
                hits[
                    "x_div_width"
                ]
            ),

        "y_div_height_hits":
            len(
                hits[
                    "y_div_height"
                ]
            ),

        "normalized_xy_hits":
            len(
                hits[
                    "normalized_xy_terms"
                ]
            ),

        "camera_corrected_hits":
            len(
                hits[
                    "camera_corrected_terms"
                ]
            ),

        "isotropic_hits":
            len(
                hits[
                    "isotropic_terms"
                ]
            ),

        "example_snippets":
            " || ".join(
                [
                    x
                    for group in hits.values()
                    for x in group[:2]
                ][
                    :6
                ]
            ),
    }


def audit_csv(path, root_label):
    try:
        df = pd.read_csv(
            path,
            nrows=200,
        )
    except Exception as e:
        return {
            "root_label":
                root_label,

            "path":
                str(path),

            "filename":
                path.name,

            "sha256":
                sha256_file(path),

            "rows_previewed":
                None,

            "interesting_columns":
                None,

            "has_width":
                False,

            "has_height":
                False,

            "has_xy":
                False,

            "has_corrected":
                False,

            "has_raw":
                False,

            "read_error":
                f"{type(e).__name__}: {e}",
        }

    cols = [
        str(c)
        for c in df.columns
    ]

    lower = [
        c.lower()
        for c in cols
    ]

    interesting = []

    for c in cols:
        lc = c.lower()

        if any(
            hint in lc
            for hint in CSV_HINTS
        ):
            interesting.append(c)

    has_width = any(
        any(
            token in c
            for token in [
                "width",
                "frame_w",
                "image_w",
            ]
        )
        for c in lower
    )

    has_height = any(
        any(
            token in c
            for token in [
                "height",
                "frame_h",
                "image_h",
            ]
        )
        for c in lower
    )

    xy_tokens = {
        "x",
        "y",
        "cx",
        "cy",
        "center_x",
        "center_y",
        "corrected_x",
        "corrected_y",
        "raw_x",
        "raw_y",
    }

    has_xy = any(
        c in xy_tokens
        or c.endswith("_x")
        or c.endswith("_y")
        for c in lower
    )

    has_corrected = any(
        "corrected" in c
        for c in lower
    )

    has_raw = any(
        "raw" in c
        for c in lower
    )

    if not (
        interesting
        or
        has_width
        or
        has_height
        or
        has_xy
        or
        has_corrected
        or
        has_raw
    ):
        return None

    return {
        "root_label":
            root_label,

        "path":
            str(path),

        "filename":
            path.name,

        "sha256":
            sha256_file(path),

        "rows_previewed":
            int(
                len(df)
            ),

        "interesting_columns":
            " | ".join(
                interesting
            ),

        "has_width":
            bool(
                has_width
            ),

        "has_height":
            bool(
                has_height
            ),

        "has_xy":
            bool(
                has_xy
            ),

        "has_corrected":
            bool(
                has_corrected
            ),

        "has_raw":
            bool(
                has_raw
            ),

        "read_error":
            None,
    }


def main():
    print(
        "\nR9-A — CANONICALIZATION / SCALE / ISOTROPIC-COORDINATE PREFLIGHT AUDIT"
    )

    print(
        "NO model training and NO frozen-result overwrite. "
        "This step identifies the exact existing coordinate representation, "
        "the source artifacts needed for an isotropic ablation, and whether "
        "width/height metadata are available.\n"
    )

    text_rows = []
    csv_rows = []
    found_roots = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        found_roots.append(
            str(root)
        )

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            try:
                if path.suffix.lower() in TEXT_SUFFIXES:
                    row = audit_text_file(
                        path,
                        root.name,
                    )

                    if row is not None:
                        text_rows.append(
                            row
                        )

                elif path.suffix.lower() == ".csv":
                    row = audit_csv(
                        path,
                        root.name,
                    )

                    if row is not None:
                        csv_rows.append(
                            row
                        )

            except Exception:
                # Audit must be robust to irrelevant/broken artifacts.
                continue

    if not found_roots:
        raise RuntimeError(
            "Neither Main Result nor Main Result 2.0 exists."
        )

    text_df = pd.DataFrame(
        text_rows
    )

    csv_df = pd.DataFrame(
        csv_rows
    )

    if len(text_df):
        text_df["anisotropic_evidence_score"] = (
            text_df[
                "x_div_width_hits"
            ]
            +
            text_df[
                "y_div_height_hits"
            ]
            +
            text_df[
                "normalized_xy_hits"
            ]
        )

        text_df = text_df.sort_values(
            [
                "anisotropic_evidence_score",
                "camera_corrected_hits",
                "isotropic_hits",
                "filename",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        )

    if len(csv_df):
        csv_df["candidate_score"] = (
            csv_df[
                [
                    "has_width",
                    "has_height",
                    "has_xy",
                    "has_corrected",
                    "has_raw",
                ]
            ]
            .astype(int)
            .sum(
                axis=1
            )
        )

        csv_df = csv_df.sort_values(
            [
                "candidate_score",
                "filename",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )

    text_path = (
        OUT
        / "R9A_TEXT_REPRESENTATION_EVIDENCE.csv"
    )

    csv_path = (
        OUT
        / "R9A_COORDINATE_SOURCE_CANDIDATES.csv"
    )

    text_df.to_csv(
        text_path,
        index=False,
    )

    csv_df.to_csv(
        csv_path,
        index=False,
    )

    anisotropic_hits = 0

    if len(text_df):
        anisotropic_hits = int(
            (
                text_df[
                    "x_div_width_hits"
                ]
                +
                text_df[
                    "y_div_height_hits"
                ]
            ).sum()
        )

    isotropic_hits = 0

    if len(text_df):
        isotropic_hits = int(
            text_df[
                "isotropic_hits"
            ].sum()
        )

    width_height_candidates = 0

    if len(csv_df):
        width_height_candidates = int(
            (
                csv_df[
                    "has_width"
                ]
                &
                csv_df[
                    "has_height"
                ]
            ).sum()
        )

    corrected_xy_candidates = 0

    if len(csv_df):
        corrected_xy_candidates = int(
            (
                csv_df[
                    "has_xy"
                ]
                &
                csv_df[
                    "has_corrected"
                ]
            ).sum()
        )

    status = (
        "R9A_CANONICALIZATION_REPRESENTATION_PREFLIGHT_COMPLETE_READY_FOR_R9B_SOURCE_LOCK"
    )

    manifest = {
        "step":
            "R9A_CANONICALIZATION_REPRESENTATION_PREFLIGHT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "frozen_results_modified":
            False,

        "search_roots":
            found_roots,

        "text_files_with_representation_evidence":
            int(
                len(
                    text_df
                )
            ),

        "csv_coordinate_candidates":
            int(
                len(
                    csv_df
                )
            ),

        "anisotropic_xW_yH_pattern_hits":
            anisotropic_hits,

        "existing_isotropic_pattern_hits":
            isotropic_hits,

        "width_height_metadata_candidate_tables":
            width_height_candidates,

        "corrected_xy_candidate_tables":
            corrected_xy_candidates,

        "text_evidence_csv":
            str(
                text_path
            ),

        "coordinate_candidates_csv":
            str(
                csv_path
            ),
    }

    (
        OUT
        / "R9A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "TOP REPRESENTATION EVIDENCE:"
    )

    if len(text_df):
        print(
            text_df[
                [
                    "root_label",
                    "filename",
                    "anisotropic_evidence_score",
                    "x_div_width_hits",
                    "y_div_height_hits",
                    "normalized_xy_hits",
                    "camera_corrected_hits",
                    "isotropic_hits",
                    "path",
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
            "No text representation evidence found."
        )

    print(
        "\nTOP COORDINATE SOURCE CANDIDATES:"
    )

    if len(csv_df):
        print(
            csv_df[
                [
                    "root_label",
                    "filename",
                    "candidate_score",
                    "has_width",
                    "has_height",
                    "has_xy",
                    "has_corrected",
                    "has_raw",
                    "interesting_columns",
                    "path",
                ]
            ]
            .head(
                40
            )
            .to_string(
                index=False
            )
        )
    else:
        print(
            "No CSV coordinate-source candidates found."
        )

    print(
        "\nPREFLIGHT SUMMARY:"
    )

    print(
        json.dumps(
            {
                "anisotropic_xW_yH_pattern_hits":
                    anisotropic_hits,

                "existing_isotropic_pattern_hits":
                    isotropic_hits,

                "width_height_metadata_candidate_tables":
                    width_height_candidates,

                "corrected_xy_candidate_tables":
                    corrected_xy_candidates,
            },
            indent=2,
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-A COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
