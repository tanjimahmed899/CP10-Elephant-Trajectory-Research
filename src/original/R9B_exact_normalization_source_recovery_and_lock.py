
from pathlib import Path
import json
import hashlib
import re
from datetime import datetime, timezone

import pandas as pd
import numpy as np


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9B_exact_normalization_source_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

CLEANED = (
    MAIN1
    / "trajectory_dataset"
    / "cleaned_track_points.csv"
)

KEY_SCRIPT_CANDIDATES = [
    MAIN1 / "build_final_neural_dataset.py",
    MAIN1 / "train_rmpgu_v2.py",
    MAIN1 / "train_rmpgu_v2_fixed.py",
    MAIN2 / "step5_ablation.py",
    MAIN2 / "step6d_k3c6_confirm_exact_normalized_mapping.py",
    MAIN2 / "step6d_k3c6_v2_confirm_exact_normalized_mapping_autodiscover.py",
]

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
]

TARGET_TOKENS = [
    "cx_n",
    "cy_n",
    "normalized_x",
    "normalized_y",
    "cx /",
    "cy /",
    "/ width",
    "/ height",
    "frame_width",
    "frame_height",
    "image_width",
    "image_height",
    "img_w",
    "img_h",
    "video_width",
    "video_height",
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


def read_text(path):
    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def snippets_for_token(text, token, radius=220, max_hits=20):
    out = []
    lower = text.lower()
    token_lower = token.lower()

    start = 0
    while True:
        i = lower.find(
            token_lower,
            start,
        )

        if i < 0:
            break

        a = max(
            0,
            i - radius,
        )

        b = min(
            len(text),
            i + len(token) + radius,
        )

        snippet = (
            text[a:b]
            .replace("\r", " ")
            .replace("\n", " ")
        )

        out.append(
            snippet
        )

        if len(out) >= max_hits:
            break

        start = i + len(token)

    return out


def scan_scripts():
    rows = []
    visited = set()

    files = []

    for p in KEY_SCRIPT_CANDIDATES:
        if p.exists():
            files.append(p)

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*.py"):
            if p not in visited:
                files.append(p)

    for path in files:
        if path in visited:
            continue

        visited.add(path)

        text = read_text(
            path
        )

        if not text:
            continue

        found = False

        token_map = {}

        for token in TARGET_TOKENS:
            hits = snippets_for_token(
                text,
                token,
            )

            if hits:
                found = True
                token_map[
                    token
                ] = hits

        if not found:
            continue

        rows.append(
            {
                "path":
                    str(path),

                "filename":
                    path.name,

                "sha256":
                    sha256_file(path),

                "cx_n_hits":
                    len(
                        token_map.get(
                            "cx_n",
                            [],
                        )
                    ),

                "cy_n_hits":
                    len(
                        token_map.get(
                            "cy_n",
                            [],
                        )
                    ),

                "frame_width_hits":
                    len(
                        token_map.get(
                            "frame_width",
                            [],
                        )
                    ),

                "frame_height_hits":
                    len(
                        token_map.get(
                            "frame_height",
                            [],
                        )
                    ),

                "width_division_hits":
                    len(
                        token_map.get(
                            "/ width",
                            [],
                        )
                    ),

                "height_division_hits":
                    len(
                        token_map.get(
                            "/ height",
                            [],
                        )
                    ),

                "example_snippets":
                    " || ".join(
                        [
                            s
                            for token in TARGET_TOKENS
                            for s in token_map.get(
                                token,
                                [],
                            )[:2]
                        ][:12]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def inspect_cleaned_points():
    if not CLEANED.exists():
        raise FileNotFoundError(
            CLEANED
        )

    df = pd.read_csv(
        CLEANED
    )

    required = {
        "cx",
        "cy",
        "cx_n",
        "cy_n",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"cleaned_track_points missing {sorted(missing)}"
        )

    stats = []

    for col in [
        "x",
        "y",
        "cx",
        "cy",
        "cx_n",
        "cy_n",
        "vx_n",
        "vy_n",
    ]:
        if col not in df.columns:
            continue

        s = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        finite = s[
            np.isfinite(
                s
            )
        ]

        if len(finite) == 0:
            continue

        stats.append(
            {
                "column":
                    col,

                "count":
                    int(
                        len(finite)
                    ),

                "min":
                    float(
                        finite.min()
                    ),

                "median":
                    float(
                        finite.median()
                    ),

                "max":
                    float(
                        finite.max()
                    ),

                "mean":
                    float(
                        finite.mean()
                    ),

                "std":
                    float(
                        finite.std(
                            ddof=0
                        )
                    ),
            }
        )

    # Recover candidate denominators implied by cx/cx_n and cy/cy_n.
    valid_x = (
        pd.to_numeric(
            df["cx_n"],
            errors="coerce",
        ).abs()
        >
        1e-12
    )

    valid_y = (
        pd.to_numeric(
            df["cy_n"],
            errors="coerce",
        ).abs()
        >
        1e-12
    )

    implied_w = (
        pd.to_numeric(
            df.loc[
                valid_x,
                "cx",
            ],
            errors="coerce",
        )
        /
        pd.to_numeric(
            df.loc[
                valid_x,
                "cx_n",
            ],
            errors="coerce",
        )
    )

    implied_h = (
        pd.to_numeric(
            df.loc[
                valid_y,
                "cy",
            ],
            errors="coerce",
        )
        /
        pd.to_numeric(
            df.loc[
                valid_y,
                "cy_n",
            ],
            errors="coerce",
        )
    )

    implied_w = implied_w[
        np.isfinite(
            implied_w
        )
        &
        (
            implied_w > 0
        )
    ]

    implied_h = implied_h[
        np.isfinite(
            implied_h
        )
        &
        (
            implied_h > 0
        )
    ]

    denom = {
        "implied_width_count":
            int(
                len(
                    implied_w
                )
            ),

        "implied_width_median":
            float(
                implied_w.median()
            )
            if len(
                implied_w
            )
            else None,

        "implied_width_q01":
            float(
                implied_w.quantile(
                    0.01
                )
            )
            if len(
                implied_w
            )
            else None,

        "implied_width_q99":
            float(
                implied_w.quantile(
                    0.99
                )
            )
            if len(
                implied_w
            )
            else None,

        "implied_height_count":
            int(
                len(
                    implied_h
                )
            ),

        "implied_height_median":
            float(
                implied_h.median()
            )
            if len(
                implied_h
            )
            else None,

        "implied_height_q01":
            float(
                implied_h.quantile(
                    0.01
                )
            )
            if len(
                implied_h
            )
            else None,

        "implied_height_q99":
            float(
                implied_h.quantile(
                    0.99
                )
            )
            if len(
                implied_h
            )
            else None,
    }

    return (
        df,
        pd.DataFrame(
            stats
        ),
        denom,
    )


def unique_implied_denominators(df):
    rows = []

    for axis, raw_col, norm_col in [
        (
            "x",
            "cx",
            "cx_n",
        ),
        (
            "y",
            "cy",
            "cy_n",
        ),
    ]:
        raw = pd.to_numeric(
            df[
                raw_col
            ],
            errors="coerce",
        )

        norm = pd.to_numeric(
            df[
                norm_col
            ],
            errors="coerce",
        )

        mask = (
            np.isfinite(
                raw
            )
            &
            np.isfinite(
                norm
            )
            &
            (
                np.abs(
                    norm
                )
                >
                1e-12
            )
        )

        ratio = (
            raw[
                mask
            ]
            /
            norm[
                mask
            ]
        )

        rounded = ratio.round(
            6
        )

        counts = (
            rounded.value_counts()
            .head(
                30
            )
        )

        for value, count in counts.items():
            rows.append(
                {
                    "axis":
                        axis,

                    "implied_denominator":
                        float(
                            value
                        ),

                    "count":
                        int(
                            count
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def decide_representation(denom, script_df):
    w = denom[
        "implied_width_median"
    ]

    h = denom[
        "implied_height_median"
    ]

    if (
        w is not None
        and
        h is not None
        and
        np.isfinite(
            w
        )
        and
        np.isfinite(
            h
        )
    ):
        rel_diff = abs(
            w - h
        ) / max(
            abs(
                w
            ),
            abs(
                h
            ),
            1e-12,
        )

        if rel_diff > 0.01:
            empirical = (
                "EMPIRICALLY_CONSISTENT_WITH_SEPARATE_X_Y_DENOMINATORS"
            )
        else:
            empirical = (
                "EMPIRICALLY_CONSISTENT_WITH_COMMON_X_Y_SCALE"
            )
    else:
        rel_diff = None
        empirical = (
            "INSUFFICIENT_EMPIRICAL_DENOMINATOR_EVIDENCE"
        )

    script_support = False

    if len(
        script_df
    ):
        script_support = bool(
            (
                script_df[
                    "cx_n_hits"
                ]
                >
                0
            ).any()
            and
            (
                script_df[
                    "cy_n_hits"
                ]
                >
                0
            ).any()
        )

    return {
        "empirical_representation_assessment":
            empirical,

        "implied_x_y_scale_relative_difference":
            rel_diff,

        "script_contains_cx_n_and_cy_n_references":
            script_support,
    }


def main():
    print(
        "\nR9-B — EXACT NORMALIZATION SOURCE RECOVERY / SOURCE LOCK"
    )

    print(
        "NO training and NO frozen-result modification. "
        "This step recovers the definition of cx_n/cy_n from code and from "
        "the observed raw-to-normalized coordinate ratios before any isotropic ablation is allowed.\n"
    )

    scripts = scan_scripts()

    df, stats, denom = inspect_cleaned_points()

    top_denoms = unique_implied_denominators(
        df
    )

    assessment = decide_representation(
        denom,
        scripts,
    )

    scripts_path = (
        OUT
        / "R9B_SCRIPT_NORMALIZATION_EVIDENCE.csv"
    )

    stats_path = (
        OUT
        / "R9B_CLEANED_TRACK_POINT_RANGE_AUDIT.csv"
    )

    denoms_path = (
        OUT
        / "R9B_TOP_IMPLIED_NORMALIZATION_DENOMINATORS.csv"
    )

    scripts.to_csv(
        scripts_path,
        index=False,
    )

    stats.to_csv(
        stats_path,
        index=False,
    )

    top_denoms.to_csv(
        denoms_path,
        index=False,
    )

    source_lock = {
        "coordinate_source":
            str(
                CLEANED
            ),

        "coordinate_source_sha256":
            sha256_file(
                CLEANED
            ),

        "required_columns":
            [
                "cx",
                "cy",
                "cx_n",
                "cy_n",
            ],

        "implied_denominator_summary":
            denom,

        "representation_assessment":
            assessment,

        "isotropic_ablation_allowed_yet":
            False,

        "reason":
            (
                "R9B is a source-definition lock only. "
                "R9C must explicitly verify the exact formula and define a non-destructive "
                "isotropic transform before any model/baseline comparison."
            ),
    }

    source_lock_path = (
        OUT
        / "R9B_SOURCE_LOCK.json"
    )

    source_lock_path.write_text(
        json.dumps(
            source_lock,
            indent=2,
        ),
        encoding="utf-8",
    )

    status = (
        "R9B_EXACT_NORMALIZATION_SOURCE_RECOVERY_COMPLETE_READY_FOR_R9C_TRANSFORM_LOCK"
    )

    manifest = {
        "step":
            "R9B_EXACT_NORMALIZATION_SOURCE_RECOVERY",

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

        "source_lock_json":
            str(
                source_lock_path
            ),

        "source_lock_sha256":
            sha256_file(
                source_lock_path
            ),

        "script_evidence_csv":
            str(
                scripts_path
            ),

        "range_audit_csv":
            str(
                stats_path
            ),

        "denominator_audit_csv":
            str(
                denoms_path
            ),
    }

    (
        OUT
        / "R9B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "CLEANED TRACK POINT RANGE AUDIT:"
    )

    print(
        stats.to_string(
            index=False
        )
    )

    print(
        "\nIMPLIED NORMALIZATION DENOMINATOR SUMMARY:"
    )

    print(
        json.dumps(
            denom,
            indent=2,
        )
    )

    print(
        "\nTOP IMPLIED DENOMINATORS:"
    )

    print(
        top_denoms.to_string(
            index=False
        )
    )

    print(
        "\nTOP SCRIPT NORMALIZATION EVIDENCE:"
    )

    if len(
        scripts
    ):
        show = scripts.copy()

        show["score"] = (
            show[
                "cx_n_hits"
            ]
            +
            show[
                "cy_n_hits"
            ]
            +
            show[
                "frame_width_hits"
            ]
            +
            show[
                "frame_height_hits"
            ]
            +
            show[
                "width_division_hits"
            ]
            +
            show[
                "height_division_hits"
            ]
        )

        show = show.sort_values(
            [
                "score",
                "filename",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )

        print(
            show[
                [
                    "filename",
                    "score",
                    "cx_n_hits",
                    "cy_n_hits",
                    "frame_width_hits",
                    "frame_height_hits",
                    "width_division_hits",
                    "height_division_hits",
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
            "No script evidence found."
        )

    print(
        "\nREPRESENTATION ASSESSMENT:"
    )

    print(
        json.dumps(
            assessment,
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
        "\nR9-B COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
