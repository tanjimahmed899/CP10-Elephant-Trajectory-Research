
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

STEP2B_SCRIPT = (
    MAIN2
    / "step2b_raw_vs_corrected_forecasting.py"
)

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
    DOWNLOADS,
]

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9F_V3A_exact_corrected_source_mapping_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

TASKS = [
    "2s_to_2s",
    "2s_to_5s",
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


def extract_script_context():
    require(STEP2B_SCRIPT)

    text = STEP2B_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    ranges = [
        (1, 120),
        (680, 1030),
    ]

    rows = []

    for a, b in ranges:
        for i in range(
            a,
            min(
                b,
                len(lines),
            )
            + 1,
        ):
            rows.append(
                {
                    "line_number":
                        i,

                    "line":
                        lines[
                            i - 1
                        ],
                }
            )

    return pd.DataFrame(rows)


def discover_metadata():
    rows = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for task in TASKS:
            names = [
                f"corrected_windows_{task}.csv",
                f"*{task}*corrected*.csv",
                f"*corrected*{task}*.csv",
            ]

            seen = set()

            for pattern in names:
                try:
                    candidates = root.rglob(
                        pattern
                    )
                except Exception:
                    continue

                for p in candidates:
                    if p in seen:
                        continue

                    seen.add(
                        p
                    )

                    try:
                        df = pd.read_csv(
                            p
                        )
                    except Exception as e:
                        rows.append(
                            {
                                "task":
                                    task,

                                "path":
                                    str(
                                        p
                                    ),

                                "rows":
                                    None,

                                "columns":
                                    None,

                                "sample_id_column":
                                    None,

                                "sample_id_min":
                                    None,

                                "sample_id_max":
                                    None,

                                "sample_id_nunique":
                                    None,

                                "read_error":
                                    f"{type(e).__name__}: {e}",

                                "sha256":
                                    sha256_file(
                                        p
                                    ),
                            }
                        )

                        continue

                    cols = list(
                        map(
                            str,
                            df.columns,
                        )
                    )

                    sample_col = next(
                        (
                            c
                            for c in [
                                "sample_id",
                                "sample_index",
                                "window_id",
                            ]
                            if c in df.columns
                        ),
                        None,
                    )

                    row = {
                        "task":
                            task,

                        "path":
                            str(
                                p
                            ),

                        "rows":
                            int(
                                len(
                                    df
                                )
                            ),

                        "columns":
                            " | ".join(
                                cols
                            ),

                        "sample_id_column":
                            sample_col,

                        "sample_id_min":
                            None,

                        "sample_id_max":
                            None,

                        "sample_id_nunique":
                            None,

                        "read_error":
                            None,

                        "sha256":
                            sha256_file(
                                p
                            ),
                    }

                    if sample_col is not None:
                        s = pd.to_numeric(
                            df[
                                sample_col
                            ],
                            errors="coerce",
                        )

                        finite = s[
                            np.isfinite(
                                s
                            )
                        ]

                        if len(
                            finite
                        ):
                            row[
                                "sample_id_min"
                            ] = int(
                                finite.min()
                            )

                            row[
                                "sample_id_max"
                            ] = int(
                                finite.max()
                            )

                            row[
                                "sample_id_nunique"
                            ] = int(
                                finite.nunique()
                            )

                    # Compact categorical distribution audit.
                    cat_summary = {}

                    for c in [
                        "dataset",
                        "role",
                        "split",
                        "subset",
                        "source_group",
                        "sequence",
                        "valid",
                        "status",
                        "task",
                    ]:
                        if c in df.columns:
                            vc = (
                                df[
                                    c
                                ]
                                .astype(str)
                                .value_counts(
                                    dropna=False
                                )
                                .head(
                                    30
                                )
                                .to_dict()
                            )

                            cat_summary[
                                c
                            ] = {
                                str(k):
                                    int(v)
                                for k, v in vc.items()
                            }

                    row[
                        "categorical_summary_json"
                    ] = json.dumps(
                        cat_summary,
                        sort_keys=True,
                    )

                    rows.append(
                        row
                    )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(
        rows
    )

    return out.sort_values(
        [
            "task",
            "rows",
            "path",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="mergesort",
    )


def discover_npz():
    rows = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for task in TASKS:
            patterns = [
                f"corrected_{task}.npz",
                f"*{task}*corrected*.npz",
                f"*corrected*{task}*.npz",
            ]

            seen = set()

            for pattern in patterns:
                try:
                    candidates = root.rglob(
                        pattern
                    )
                except Exception:
                    continue

                for p in candidates:
                    if p in seen:
                        continue

                    seen.add(
                        p
                    )

                    row = {
                        "task":
                            task,

                        "path":
                            str(
                                p
                            ),

                        "sha256":
                            sha256_file(
                                p
                            ),

                        "keys":
                            None,

                        "shapes_json":
                            None,

                        "dtypes_json":
                            None,

                        "read_error":
                            None,
                    }

                    try:
                        arr = np.load(
                            p
                        )

                        shapes = {}
                        dtypes = {}

                        for k in arr.files:
                            a = np.asarray(
                                arr[
                                    k
                                ]
                            )

                            shapes[
                                k
                            ] = list(
                                a.shape
                            )

                            dtypes[
                                k
                            ] = str(
                                a.dtype
                            )

                        row[
                            "keys"
                        ] = " | ".join(
                            arr.files
                        )

                        row[
                            "shapes_json"
                        ] = json.dumps(
                            shapes,
                            sort_keys=True,
                        )

                        row[
                            "dtypes_json"
                        ] = json.dumps(
                            dtypes,
                            sort_keys=True,
                        )

                    except Exception as e:
                        row[
                            "read_error"
                        ] = f"{type(e).__name__}: {e}"

                    rows.append(
                        row
                    )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "task",
            "path",
        ],
        kind="mergesort",
    )


def inspect_primary_pair(metadata_df, npz_df):
    rows = []

    for task in TASKS:
        meta_candidates = metadata_df[
            metadata_df[
                "task"
            ]
            ==
            task
        ]

        npz_candidates = npz_df[
            npz_df[
                "task"
            ]
            ==
            task
        ]

        for _, mr in meta_candidates.iterrows():
            for _, nr in npz_candidates.iterrows():
                mp = Path(
                    mr[
                        "path"
                    ]
                )

                np = Path(
                    nr[
                        "path"
                    ]
                )

                same_parent = (
                    mp.parent
                    ==
                    np.parent
                )

                rows.append(
                    {
                        "task":
                            task,

                        "metadata_path":
                            str(
                                mp
                            ),

                        "npz_path":
                            str(
                                np
                            ),

                        "same_parent":
                            bool(
                                same_parent
                            ),

                        "metadata_rows":
                            mr[
                                "rows"
                            ],

                        "sample_id_column":
                            mr[
                                "sample_id_column"
                            ],

                        "sample_id_min":
                            mr[
                                "sample_id_min"
                            ],

                        "sample_id_max":
                            mr[
                                "sample_id_max"
                            ],

                        "sample_id_nunique":
                            mr[
                                "sample_id_nunique"
                            ],

                        "npz_shapes_json":
                            nr[
                                "shapes_json"
                            ],

                        "metadata_sha256":
                            mr[
                                "sha256"
                            ],

                        "npz_sha256":
                            nr[
                                "sha256"
                            ],
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "task",
            "same_parent",
            "metadata_path",
            "npz_path",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        kind="mergesort",
    )


def main():
    print(
        "\nR9-F-V3A — EXACT CORRECTED-SOURCE / SAMPLE-MAPPING AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO isotropic windows, NO model training, "
        "NO frozen-output modification. This step inspects the exact Step2b "
        "source code and all corrected metadata/NPZ candidates to recover the "
        "filtering, indexing and feature-slicing logic that produced the frozen "
        "1053/826 evaluation windows.\n"
    )

    script_context = extract_script_context()

    metadata = discover_metadata()

    npz = discover_npz()

    pair_audit = inspect_primary_pair(
        metadata,
        npz,
    )

    script_path = (
        OUT
        / "R9F_V3A_STEP2B_SCRIPT_CONTEXT.csv"
    )

    metadata_path = (
        OUT
        / "R9F_V3A_CORRECTED_METADATA_INVENTORY.csv"
    )

    npz_path = (
        OUT
        / "R9F_V3A_CORRECTED_NPZ_INVENTORY.csv"
    )

    pair_path = (
        OUT
        / "R9F_V3A_METADATA_NPZ_PAIR_AUDIT.csv"
    )

    script_context.to_csv(
        script_path,
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

    pair_audit.to_csv(
        pair_path,
        index=False,
    )

    status = (
        "R9F_V3A_EXACT_CORRECTED_SOURCE_MAPPING_AUDIT_COMPLETE_READY_FOR_R9F_V3B_MAPPING_RECOVERY"
    )

    manifest = {
        "step":
            "R9F_V3A_EXACT_CORRECTED_SOURCE_MAPPING_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "isotropic_windows_built":
            False,

        "frozen_results_modified":
            False,

        "step2b_script":
            str(
                STEP2B_SCRIPT
            ),

        "step2b_script_sha256":
            sha256_file(
                STEP2B_SCRIPT
            ),

        "metadata_candidate_count":
            int(
                len(
                    metadata
                )
            ),

        "npz_candidate_count":
            int(
                len(
                    npz
                )
            ),

        "pair_audit_count":
            int(
                len(
                    pair_audit
                )
            ),

        "script_context_csv":
            str(
                script_path
            ),

        "metadata_inventory_csv":
            str(
                metadata_path
            ),

        "npz_inventory_csv":
            str(
                npz_path
            ),

        "pair_audit_csv":
            str(
                pair_path
            ),
    }

    (
        OUT
        / "R9F_V3A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "STEP2B SOURCE CONTEXT — TOP:"
    )

    print(
        script_context.head(
            120
        ).to_string(
            index=False
        )
    )

    print(
        "\nSTEP2B SOURCE CONTEXT — CORRECTED LOADING / LOOP:"
    )

    print(
        script_context[
            script_context[
                "line_number"
            ]
            >=
            680
        ]
        .head(
            360
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nCORRECTED METADATA INVENTORY:"
    )

    if len(
        metadata
    ):
        show_cols = [
            "task",
            "rows",
            "sample_id_column",
            "sample_id_min",
            "sample_id_max",
            "sample_id_nunique",
            "categorical_summary_json",
            "path",
        ]

        print(
            metadata[
                [
                    c
                    for c in show_cols
                    if c in metadata.columns
                ]
            ]
            .to_string(
                index=False
            )
        )
    else:
        print(
            "No corrected metadata candidates found."
        )

    print(
        "\nCORRECTED NPZ INVENTORY:"
    )

    if len(
        npz
    ):
        print(
            npz[
                [
                    "task",
                    "keys",
                    "shapes_json",
                    "dtypes_json",
                    "path",
                ]
            ]
            .to_string(
                index=False
            )
        )
    else:
        print(
            "No corrected NPZ candidates found."
        )

    print(
        "\nMETADATA / NPZ PAIR AUDIT:"
    )

    if len(
        pair_audit
    ):
        print(
            pair_audit[
                [
                    "task",
                    "same_parent",
                    "metadata_rows",
                    "sample_id_column",
                    "sample_id_min",
                    "sample_id_max",
                    "sample_id_nunique",
                    "npz_shapes_json",
                    "metadata_path",
                    "npz_path",
                ]
            ]
            .to_string(
                index=False
            )
        )
    else:
        print(
            "No metadata/NPZ pairs available."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-F-V3A COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
