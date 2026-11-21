
from pathlib import Path
import json
import hashlib
import re
from datetime import datetime, timezone

import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BUILDER = MAIN1 / "build_corrected_trajectories.py"
UNIFIED = MAIN1 / "build_unified_10hz_dataset.py"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3A_corrected_builder_formula_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGET_TERMS = [
    "corrected",
    "homography",
    "ego",
    "motion",
    "cx_n",
    "cy_n",
    "img_w",
    "img_h",
    "width",
    "height",
    "history",
    "future",
    "np.savez",
    "savez_compressed",
    "feature",
    "columns",
    "accepted",
    "quality_tier",
    "segment_id",
]


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


def read_lines(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )
    return text.splitlines()


def extract_context(path, max_radius=6):
    lines = read_lines(path)
    hit_lines = set()

    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(term.lower() in low for term in TARGET_TERMS):
            for j in range(
                max(1, i - max_radius),
                min(len(lines), i + max_radius) + 1,
            ):
                hit_lines.add(j)

    rows = []

    for j in sorted(hit_lines):
        rows.append(
            {
                "file": path.name,
                "line_number": j,
                "line": lines[j - 1],
            }
        )

    return pd.DataFrame(rows)


def extract_functions(path):
    lines = read_lines(path)

    funcs = []
    current = None
    start = None

    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()

        if stripped.startswith("def "):
            if current is not None:
                funcs.append(
                    {
                        "function": current,
                        "start_line": start,
                        "end_line": i - 1,
                    }
                )

            current = stripped.split(
                "def ",
                1,
            )[1].split(
                "(",
                1,
            )[0].strip()

            start = i

    if current is not None:
        funcs.append(
            {
                "function": current,
                "start_line": start,
                "end_line": len(lines),
            }
        )

    return pd.DataFrame(funcs)


def classify_lines(path):
    lines = read_lines(path)

    patterns = {
        "normalization_assignment": [
            r"cx_n",
            r"cy_n",
            r"/\s*img_w",
            r"/\s*img_h",
            r"/\s*width",
            r"/\s*height",
        ],
        "pixel_coordinate_assignment": [
            r"\bcx\b",
            r"\bcy\b",
            r"corrected_x",
            r"corrected_y",
            r"corrected_cx",
            r"corrected_cy",
        ],
        "homography_or_transform": [
            r"homograph",
            r"perspective",
            r"transform",
            r"warp",
            r"affine",
            r"matrix",
        ],
        "feature_stack_or_schema": [
            r"stack",
            r"column_stack",
            r"concatenate",
            r"features",
            r"feature_cols",
            r"np\.savez",
            r"savez_compressed",
        ],
        "width_height_source": [
            r"img_w",
            r"img_h",
            r"width",
            r"height",
        ],
    }

    rows = []

    for i, line in enumerate(lines, start=1):
        for category, pats in patterns.items():
            if any(
                re.search(
                    p,
                    line,
                    flags=re.IGNORECASE,
                )
                for p in pats
            ):
                rows.append(
                    {
                        "file": path.name,
                        "line_number": i,
                        "category": category,
                        "line": line,
                    }
                )

    return pd.DataFrame(rows)


def exact_block(path, start, end):
    lines = read_lines(path)
    rows = []

    for i in range(
        max(1, start),
        min(len(lines), end) + 1,
    ):
        rows.append(
            {
                "file": path.name,
                "line_number": i,
                "line": lines[i - 1],
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR9-G-V3A — EXACT CORRECTED-TRAJECTORY BUILDER FORMULA AUDIT / LOCK"
    )

    print(
        "DIAGNOSTIC ONLY. NO isotropic rebuild, NO baseline comparison, "
        "NO bootstrap, NO training, NO frozen modification. "
        "This step recovers the exact order of ego-motion correction, "
        "pixel/normalized coordinate conversion, width/height usage, "
        "and the 7-feature NPZ schema.\n"
    )

    require(BUILDER)
    require(UNIFIED)

    builder_context = extract_context(
        BUILDER,
        max_radius=7,
    )

    unified_context = extract_context(
        UNIFIED,
        max_radius=5,
    )

    builder_funcs = extract_functions(
        BUILDER
    )

    builder_classified = classify_lines(
        BUILDER
    )

    unified_classified = classify_lines(
        UNIFIED
    )

    # Full source blocks are useful because the builder is likely short enough
    # and exact order matters for scientific provenance.
    builder_full = exact_block(
        BUILDER,
        1,
        2000,
    )

    unified_full = exact_block(
        UNIFIED,
        1,
        2000,
    )

    paths = {
        "builder_context":
            OUT / "R9G_V3A_BUILDER_RELEVANT_CONTEXT.csv",

        "unified_context":
            OUT / "R9G_V3A_UNIFIED_RELEVANT_CONTEXT.csv",

        "builder_functions":
            OUT / "R9G_V3A_BUILDER_FUNCTION_INDEX.csv",

        "builder_classified":
            OUT / "R9G_V3A_BUILDER_CLASSIFIED_LINES.csv",

        "unified_classified":
            OUT / "R9G_V3A_UNIFIED_CLASSIFIED_LINES.csv",

        "builder_full":
            OUT / "R9G_V3A_BUILDER_FULL_SOURCE_LINES.csv",

        "unified_full":
            OUT / "R9G_V3A_UNIFIED_FULL_SOURCE_LINES.csv",
    }

    builder_context.to_csv(
        paths["builder_context"],
        index=False,
    )

    unified_context.to_csv(
        paths["unified_context"],
        index=False,
    )

    builder_funcs.to_csv(
        paths["builder_functions"],
        index=False,
    )

    builder_classified.to_csv(
        paths["builder_classified"],
        index=False,
    )

    unified_classified.to_csv(
        paths["unified_classified"],
        index=False,
    )

    builder_full.to_csv(
        paths["builder_full"],
        index=False,
    )

    unified_full.to_csv(
        paths["unified_full"],
        index=False,
    )

    status = (
        "R9G_V3A_CORRECTED_BUILDER_FORMULA_AUDIT_COMPLETE_READY_FOR_R9G_V3B_VALID_ISOTROPIC_REBUILD"
    )

    manifest = {
        "step":
            "R9G_V3A_CORRECTED_BUILDER_FORMULA_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "isotropic_rebuild_performed":
            False,

        "baseline_comparison_performed":
            False,

        "bootstrap_performed":
            False,

        "frozen_results_modified":
            False,

        "corrected_builder":
            str(BUILDER),

        "corrected_builder_sha256":
            sha256_file(BUILDER),

        "unified_builder":
            str(UNIFIED),

        "unified_builder_sha256":
            sha256_file(UNIFIED),

        "outputs":
            {
                k: str(v)
                for k, v in paths.items()
            },
    }

    (
        OUT
        / "R9G_V3A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "BUILDER FUNCTION INDEX:"
    )

    print(
        builder_funcs.to_string(
            index=False
        )
    )

    print(
        "\nBUILDER CLASSIFIED FORMULA LINES:"
    )

    if len(
        builder_classified
    ):
        print(
            builder_classified.to_string(
                index=False
            )
        )
    else:
        print(
            "No classified builder lines found."
        )

    print(
        "\nBUILDER RELEVANT CONTEXT:"
    )

    print(
        builder_context.to_string(
            index=False
        )
    )

    print(
        "\nUNIFIED-DATASET RELEVANT CONTEXT:"
    )

    print(
        unified_context.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-G-V3A COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
