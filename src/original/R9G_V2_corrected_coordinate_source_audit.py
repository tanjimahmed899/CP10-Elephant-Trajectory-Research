
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

CORR_DIR = MAIN1 / "corrected_trajectory_dataset"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V2_corrected_coordinate_source_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
]

TEXT_SUFFIXES = {
    ".py",
    ".txt",
    ".json",
    ".md",
}

KEY_TERMS = [
    "corrected_trajectory_dataset",
    "corrected_windows_",
    "corrected_2s_to_2s",
    "corrected_2s_to_5s",
    "np.savez",
    "savez_compressed",
    "history",
    "future",
    "homography",
    "ego_motion",
    "cx_n",
    "cy_n",
    "img_w",
    "img_h",
    "width",
    "height",
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


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def scan_scripts():
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

            score = 0
            hits = []

            for term in KEY_TERMS:
                count = low.count(
                    term.lower()
                )

                if count:
                    score += count
                    hits.append(
                        f"{term}:{count}"
                    )

            if score == 0:
                continue

            # Need stronger evidence that file is about corrected trajectory construction.
            strong = any(
                x in low
                for x in [
                    "corrected_trajectory_dataset",
                    "corrected_windows_",
                    "corrected_2s_to_2s",
                    "corrected_2s_to_5s",
                ]
            )

            if not strong:
                continue

            snippets = []

            lines = text.splitlines()

            for i, line in enumerate(
                lines,
                start=1,
            ):
                l = line.lower()

                if any(
                    t.lower() in l
                    for t in [
                        "corrected_trajectory_dataset",
                        "np.savez",
                        "savez_compressed",
                        "corrected_windows_",
                        "history =",
                        "future =",
                        "cx_n",
                        "cy_n",
                        "homography",
                        "ego_motion",
                    ]
                ):
                    a = max(
                        1,
                        i - 3,
                    )

                    b = min(
                        len(lines),
                        i + 5,
                    )

                    snippet = "\n".join(
                        f"{j}: {lines[j-1]}"
                        for j in range(
                            a,
                            b + 1,
                        )
                    )

                    snippets.append(
                        snippet
                    )

                    if len(snippets) >= 10:
                        break

            rows.append(
                {
                    "root":
                        root.name,

                    "filename":
                        p.name,

                    "path":
                        str(p),

                    "score":
                        int(score),

                    "term_hits":
                        " | ".join(
                            hits
                        ),

                    "sha256":
                        sha256_file(p),

                    "snippets":
                        "\n---\n".join(
                            snippets
                        ),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
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


def inspect_metadata():
    rows = []

    for task in TASKS:
        p = CORR_DIR / f"corrected_windows_{task}.csv"
        require(p)

        df = pd.read_csv(
            p,
            nrows=50,
        )

        cols = list(
            map(
                str,
                df.columns,
            )
        )

        interesting = [
            c
            for c in cols
            if any(
                token in c.lower()
                for token in [
                    "width",
                    "height",
                    "img",
                    "frame",
                    "scale",
                    "homography",
                    "motion",
                    "corrected",
                    "cx",
                    "cy",
                    "x",
                    "y",
                    "sample",
                    "segment",
                    "sequence",
                ]
            )
        ]

        rows.append(
            {
                "task":
                    task,

                "path":
                    str(p),

                "sha256":
                    sha256_file(p),

                "all_columns":
                    " | ".join(
                        cols
                    ),

                "interesting_columns":
                    " | ".join(
                        interesting
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def inspect_npz():
    rows = []

    for task in TASKS:
        p = CORR_DIR / f"corrected_{task}.npz"
        require(p)

        arr = np.load(
            p
        )

        for key in arr.files:
            a = np.asarray(
                arr[
                    key
                ]
            )

            if a.ndim < 2:
                continue

            row = {
                "task":
                    task,

                "path":
                    str(p),

                "sha256":
                    sha256_file(p),

                "key":
                    key,

                "shape":
                    str(
                        tuple(
                            a.shape
                        )
                    ),

                "dtype":
                    str(
                        a.dtype
                    ),

                "finite":
                    bool(
                        np.isfinite(
                            a
                        ).all()
                    ),
            }

            # Per feature diagnostics for 3D tensors.
            if a.ndim == 3:
                feature_stats = {}

                for k in range(
                    a.shape[
                        2
                    ]
                ):
                    col = a[
                        ...,
                        k
                    ].astype(
                        np.float64
                    )

                    feature_stats[
                        str(k)
                    ] = {
                        "min":
                            float(
                                np.nanmin(
                                    col
                                )
                            ),

                        "median":
                            float(
                                np.nanmedian(
                                    col
                                )
                            ),

                        "max":
                            float(
                                np.nanmax(
                                    col
                                )
                            ),

                        "mean":
                            float(
                                np.nanmean(
                                    col
                                )
                            ),

                        "std":
                            float(
                                np.nanstd(
                                    col
                                )
                            ),
                    }

                row[
                    "feature_stats_json"
                ] = json.dumps(
                    feature_stats,
                    sort_keys=True,
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def discover_corrected_point_tables():
    rows = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob(
            "*.csv"
        ):
            name = p.name.lower()

            if not any(
                token in name
                for token in [
                    "corrected",
                    "ego",
                    "trajectory",
                    "point",
                ]
            ):
                continue

            try:
                df = pd.read_csv(
                    p,
                    nrows=30,
                )
            except Exception:
                continue

            cols = list(
                map(
                    str,
                    df.columns,
                )
            )

            lc = [
                c.lower()
                for c in cols
            ]

            has_xy = any(
                c in [
                    "x",
                    "y",
                    "cx",
                    "cy",
                    "cx_n",
                    "cy_n",
                    "corrected_x",
                    "corrected_y",
                    "corrected_cx",
                    "corrected_cy",
                ]
                or c.endswith(
                    "_x"
                )
                or c.endswith(
                    "_y"
                )
                for c in lc
            )

            has_identity = any(
                c in lc
                for c in [
                    "sequence",
                    "segment_id",
                    "track_key",
                    "frame",
                    "sample_id",
                ]
            )

            if not (
                has_xy
                and
                has_identity
            ):
                continue

            rows.append(
                {
                    "path":
                        str(
                            p
                        ),

                    "filename":
                        p.name,

                    "rows_previewed":
                        int(
                            len(
                                df
                            )
                        ),

                    "columns":
                        " | ".join(
                            cols
                        ),

                    "sha256":
                        sha256_file(
                            p
                        ),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "filename",
            "path",
        ],
        kind="mergesort",
    )


def main():
    print(
        "\nR9-G-V2 — CORRECTED-COORDINATE SOURCE AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO bootstrap, NO new comparison, NO model training, "
        "NO frozen modification. R9G showed a large apparent degradation, but the "
        "isotropic branch was built from cleaned_track_points while the anisotropic "
        "branch came from the camera-corrected trajectory NPZ. This audit recovers "
        "the exact corrected-coordinate construction source before a valid normalization ablation.\n"
    )

    scripts = scan_scripts()
    metadata = inspect_metadata()
    npz = inspect_npz()
    point_tables = discover_corrected_point_tables()

    scripts_path = (
        OUT
        / "R9G_V2_CORRECTED_BUILD_SCRIPT_AUDIT.csv"
    )

    metadata_path = (
        OUT
        / "R9G_V2_CORRECTED_METADATA_SCHEMA.csv"
    )

    npz_path = (
        OUT
        / "R9G_V2_CORRECTED_NPZ_FEATURE_AUDIT.csv"
    )

    points_path = (
        OUT
        / "R9G_V2_CORRECTED_POINT_TABLE_CANDIDATES.csv"
    )

    scripts.to_csv(
        scripts_path,
        index=False,
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    npz.to_csv(
        npz_path,
        index=False,
    )

    point_tables.to_csv(
        points_path,
        index=False,
    )

    status = (
        "R9G_V2_CORRECTED_COORDINATE_SOURCE_AUDIT_COMPLETE_READY_FOR_R9G_V3_VALID_ISOTROPIC_REBUILD"
    )

    manifest = {
        "step":
            "R9G_V2_CORRECTED_COORDINATE_SOURCE_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "r9g_previous_comparison_valid_for_claim":
            False,

        "reason_previous_comparison_invalid":
            (
                "Previous R9G compared camera-corrected anisotropic trajectories "
                "against isotropic windows rebuilt from cleaned_track_points. "
                "That mixes ego-motion correction/source differences with normalization."
            ),

        "new_training":
            False,

        "bootstrap_run":
            False,

        "frozen_results_modified":
            False,

        "script_candidate_count":
            int(
                len(
                    scripts
                )
            ),

        "corrected_point_table_candidate_count":
            int(
                len(
                    point_tables
                )
            ),

        "script_audit_csv":
            str(
                scripts_path
            ),

        "metadata_schema_csv":
            str(
                metadata_path
            ),

        "npz_feature_audit_csv":
            str(
                npz_path
            ),

        "point_table_candidates_csv":
            str(
                points_path
            ),
    }

    (
        OUT
        / "R9G_V2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "CORRECTED BUILD SCRIPT CANDIDATES:"
    )

    if len(
        scripts
    ):
        print(
            scripts[
                [
                    "filename",
                    "score",
                    "term_hits",
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
            "No corrected-build script candidates found."
        )

    print(
        "\nCORRECTED METADATA SCHEMA:"
    )

    print(
        metadata.to_string(
            index=False
        )
    )

    print(
        "\nCORRECTED NPZ FEATURE AUDIT:"
    )

    print(
        npz.to_string(
            index=False
        )
    )

    print(
        "\nCORRECTED POINT-TABLE CANDIDATES:"
    )

    if len(
        point_tables
    ):
        print(
            point_tables.head(
                50
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No corrected point-level CSV candidates found."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-G-V2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
