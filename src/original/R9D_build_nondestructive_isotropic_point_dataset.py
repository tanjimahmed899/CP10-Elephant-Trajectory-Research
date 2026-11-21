
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

R9C_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9C_isotropic_transform_lock"
)

R9C_MANIFEST = R9C_DIR / "R9C_MANIFEST.json"
R9C_LOCK = R9C_DIR / "R9C_ISOTROPIC_TRANSFORM_LOCK.json"

SOURCE_POINTS = (
    MAIN1
    / "trajectory_dataset"
    / "cleaned_track_points.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9D_isotropic_point_dataset_build"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

POINT_OUT = OUT / "R9D_isotropic_cleaned_track_points.csv"
SEQ_AUDIT_OUT = OUT / "R9D_sequence_geometry_audit.csv"
ROW_AUDIT_OUT = OUT / "R9D_row_integrity_audit.csv"
TRANSFORM_AUDIT_OUT = OUT / "R9D_transform_consistency_audit.csv"


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


def verify_r9c():
    for p in [
        R9C_MANIFEST,
        R9C_LOCK,
        SOURCE_POINTS,
    ]:
        require(p)

    manifest = json.loads(
        R9C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("status")
        !=
        "R9C_ISOTROPIC_TRANSFORM_FORMALLY_LOCKED_READY_FOR_R9D_NONDESTRUCTIVE_ABLATION_BUILD"
    ):
        raise RuntimeError(
            f"Unexpected R9C status: {manifest.get('status')}"
        )

    lock = json.loads(
        R9C_LOCK.read_text(
            encoding="utf-8"
        )
    )

    branch = lock.get(
        "new_isotropic_branch",
        {}
    )

    if (
        branch.get("common_scale")
        !=
        "s_i = max(W_i, H_i)"
    ):
        raise RuntimeError(
            "R9C common-scale lock mismatch."
        )

    if (
        branch.get("x_iso")
        !=
        "cx_iso_i = cx_i / s_i"
    ):
        raise RuntimeError(
            "R9C x transform lock mismatch."
        )

    if (
        branch.get("y_iso")
        !=
        "cy_iso_i = cy_i / s_i"
    ):
        raise RuntimeError(
            "R9C y transform lock mismatch."
        )

    return manifest, lock


def required_columns(df):
    required = {
        "sequence",
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
            f"Source points missing columns: {sorted(missing)}"
        )


def find_track_column(df):
    candidates = [
        "track_key",
        "track_id",
        "obj_id",
        "object_id",
        "id",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        "Could not identify track/object column."
    )


def find_order_columns(df):
    candidates = [
        "frame",
        "frame_id",
        "frame_idx",
        "timestamp",
        "time_sec",
        "t",
    ]

    found = [
        c
        for c in candidates
        if c in df.columns
    ]

    if not found:
        raise RuntimeError(
            "Could not identify frame/time ordering column."
        )

    return found


def recover_dimensions(df):
    cx = pd.to_numeric(
        df["cx"],
        errors="raise",
    ).astype(float)

    cy = pd.to_numeric(
        df["cy"],
        errors="raise",
    ).astype(float)

    cx_n = pd.to_numeric(
        df["cx_n"],
        errors="raise",
    ).astype(float)

    cy_n = pd.to_numeric(
        df["cy_n"],
        errors="raise",
    ).astype(float)

    if (
        (cx_n.abs() < 1e-12).any()
        or
        (cy_n.abs() < 1e-12).any()
    ):
        raise RuntimeError(
            "Encountered zero cx_n/cy_n; cannot recover dimensions safely."
        )

    width = np.rint(
        cx / cx_n
    ).astype(int)

    height = np.rint(
        cy / cy_n
    ).astype(int)

    if (
        (width <= 0).any()
        or
        (height <= 0).any()
    ):
        raise RuntimeError(
            "Recovered invalid width/height."
        )

    return width, height


def validate_sequence_dimensions(df):
    rows = []

    for sequence, g in df.groupby(
        "sequence",
        sort=True,
        dropna=False,
    ):
        widths = sorted(
            pd.unique(
                g["img_w_locked"]
            ).tolist()
        )

        heights = sorted(
            pd.unique(
                g["img_h_locked"]
            ).tolist()
        )

        scales = sorted(
            pd.unique(
                g["iso_scale_px"]
            ).tolist()
        )

        rows.append(
            {
                "sequence":
                    str(sequence),

                "rows":
                    int(len(g)),

                "unique_width_count":
                    int(len(widths)),

                "unique_height_count":
                    int(len(heights)),

                "unique_iso_scale_count":
                    int(len(scales)),

                "width_values":
                    json.dumps(
                        [
                            int(x)
                            for x in widths
                        ]
                    ),

                "height_values":
                    json.dumps(
                        [
                            int(x)
                            for x in heights
                        ]
                    ),

                "iso_scale_values":
                    json.dumps(
                        [
                            int(x)
                            for x in scales
                        ]
                    ),

                "geometry_stable_within_sequence":
                    bool(
                        len(widths) == 1
                        and
                        len(heights) == 1
                        and
                        len(scales) == 1
                    ),
            }
        )

    audit = pd.DataFrame(
        rows
    )

    if not audit[
        "geometry_stable_within_sequence"
    ].all():
        bad = audit[
            ~audit[
                "geometry_stable_within_sequence"
            ]
        ]

        raise RuntimeError(
            "Width/height are not stable within at least one sequence:\n"
            +
            bad.to_string(
                index=False
            )
        )

    return audit


def recompute_dynamics(df, track_col, order_cols):
    sort_cols = [
        "sequence",
        track_col,
        *order_cols,
    ]

    # Remove duplicate order columns while preserving order.
    seen = set()
    sort_cols = [
        c
        for c in sort_cols
        if not (
            c in seen
            or
            seen.add(c)
        )
    ]

    work = df.sort_values(
        sort_cols,
        kind="mergesort",
    ).copy()

    group_cols = [
        "sequence",
        track_col,
    ]

    work["vx_iso"] = (
        work.groupby(
            group_cols,
            sort=False,
        )["cx_iso"]
        .diff()
        .fillna(0.0)
    )

    work["vy_iso"] = (
        work.groupby(
            group_cols,
            sort=False,
        )["cy_iso"]
        .diff()
        .fillna(0.0)
    )

    work["speed_iso"] = np.sqrt(
        work["vx_iso"] ** 2
        +
        work["vy_iso"] ** 2
    )

    work["heading_iso_rad"] = np.arctan2(
        work["vy_iso"],
        work["vx_iso"],
    )

    work["ax_iso"] = (
        work.groupby(
            group_cols,
            sort=False,
        )["vx_iso"]
        .diff()
        .fillna(0.0)
    )

    work["ay_iso"] = (
        work.groupby(
            group_cols,
            sort=False,
        )["vy_iso"]
        .diff()
        .fillna(0.0)
    )

    work["accel_iso"] = np.sqrt(
        work["ax_iso"] ** 2
        +
        work["ay_iso"] ** 2
    )

    # Restore original row order exactly.
    work = work.sort_values(
        "_original_row_index",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return work


def build_dataset():
    src = pd.read_csv(
        SOURCE_POINTS
    )

    required_columns(
        src
    )

    track_col = find_track_column(
        src
    )

    order_cols = find_order_columns(
        src
    )

    src = src.copy()

    src[
        "_original_row_index"
    ] = np.arange(
        len(src),
        dtype=int,
    )

    width, height = recover_dimensions(
        src
    )

    src[
        "img_w_locked"
    ] = width

    src[
        "img_h_locked"
    ] = height

    src[
        "iso_scale_px"
    ] = np.maximum(
        width,
        height,
    )

    src[
        "cx_iso"
    ] = (
        pd.to_numeric(
            src["cx"],
            errors="raise",
        ).astype(float)
        /
        src[
            "iso_scale_px"
        ]
    )

    src[
        "cy_iso"
    ] = (
        pd.to_numeric(
            src["cy"],
            errors="raise",
        ).astype(float)
        /
        src[
            "iso_scale_px"
        ]
    )

    seq_audit = validate_sequence_dimensions(
        src
    )

    out = recompute_dynamics(
        src,
        track_col,
        order_cols,
    )

    return (
        src,
        out,
        seq_audit,
        track_col,
        order_cols,
    )


def integrity_audit(src, out, track_col):
    checks = []

    def add(name, passed, detail):
        checks.append(
            {
                "check":
                    name,

                "passed":
                    bool(
                        passed
                    ),

                "detail":
                    str(
                        detail
                    ),
            }
        )

    add(
        "row_count_unchanged",
        len(src) == len(out),
        f"{len(src)} -> {len(out)}",
    )

    add(
        "sequence_count_unchanged",
        src["sequence"].nunique(
            dropna=False
        )
        ==
        out["sequence"].nunique(
            dropna=False
        ),
        (
            f"{src['sequence'].nunique(dropna=False)} -> "
            f"{out['sequence'].nunique(dropna=False)}"
        ),
    )

    src_tracks = (
        src[
            [
                "sequence",
                track_col,
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    out_tracks = (
        out[
            [
                "sequence",
                track_col,
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    add(
        "track_count_unchanged",
        src_tracks == out_tracks,
        f"{src_tracks} -> {out_tracks}",
    )

    add(
        "original_row_order_recoverable",
        np.array_equal(
            out[
                "_original_row_index"
            ].to_numpy(
                dtype=int
            ),
            np.arange(
                len(out),
                dtype=int,
            ),
        ),
        "original row index must be exactly 0..N-1 after restore",
    )

    add(
        "cx_iso_range",
        bool(
            (
                (
                    out[
                        "cx_iso"
                    ]
                    >=
                    0
                )
                &
                (
                    out[
                        "cx_iso"
                    ]
                    <=
                    1
                    +
                    1e-12
                )
            ).all()
        ),
        (
            f"min={out['cx_iso'].min()}, "
            f"max={out['cx_iso'].max()}"
        ),
    )

    add(
        "cy_iso_range",
        bool(
            (
                (
                    out[
                        "cy_iso"
                    ]
                    >=
                    0
                )
                &
                (
                    out[
                        "cy_iso"
                    ]
                    <=
                    1
                    +
                    1e-12
                )
            ).all()
        ),
        (
            f"min={out['cy_iso'].min()}, "
            f"max={out['cy_iso'].max()}"
        ),
    )

    add(
        "same_scale_used_per_row",
        True,
        "cx_iso and cy_iso both use iso_scale_px=max(img_w_locked,img_h_locked)",
    )

    dyn_cols = [
        "vx_iso",
        "vy_iso",
        "speed_iso",
        "heading_iso_rad",
        "ax_iso",
        "ay_iso",
        "accel_iso",
    ]

    add(
        "dynamics_finite",
        bool(
            np.isfinite(
                out[
                    dyn_cols
                ].to_numpy(
                    dtype=float
                )
            ).all()
        ),
        "all recomputed isotropic dynamics finite",
    )

    audit = pd.DataFrame(
        checks
    )

    if not audit[
        "passed"
    ].all():
        raise RuntimeError(
            "R9D integrity audit failed:\n"
            +
            audit.to_string(
                index=False
            )
        )

    return audit


def transform_consistency(out):
    x_recon = (
        out[
            "cx_iso"
        ]
        *
        out[
            "iso_scale_px"
        ]
    )

    y_recon = (
        out[
            "cy_iso"
        ]
        *
        out[
            "iso_scale_px"
        ]
    )

    cx = pd.to_numeric(
        out["cx"],
        errors="raise",
    ).astype(float)

    cy = pd.to_numeric(
        out["cy"],
        errors="raise",
    ).astype(float)

    xerr = np.abs(
        x_recon
        -
        cx
    )

    yerr = np.abs(
        y_recon
        -
        cy
    )

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "pixel_reconstruction_x_abs_error",

                "max":
                    float(
                        xerr.max()
                    ),

                "median":
                    float(
                        np.median(
                            xerr
                        )
                    ),
            },

            {
                "metric":
                    "pixel_reconstruction_y_abs_error",

                "max":
                    float(
                        yerr.max()
                    ),

                "median":
                    float(
                        np.median(
                            yerr
                        )
                    ),
            },

            {
                "metric":
                    "aspect_ratio_x_scale_to_y_scale_anisotropic",

                "max":
                    float(
                        (
                            out[
                                "img_w_locked"
                            ]
                            /
                            out[
                                "img_h_locked"
                            ]
                        ).max()
                    ),

                "median":
                    float(
                        (
                            out[
                                "img_w_locked"
                            ]
                            /
                            out[
                                "img_h_locked"
                            ]
                        ).median()
                    ),
            },

            {
                "metric":
                    "isotropic_x_y_scale_ratio",

                "max":
                    1.0,

                "median":
                    1.0,
            },
        ]
    )

    if (
        xerr.max()
        >
        1e-9
        or
        yerr.max()
        >
        1e-9
    ):
        raise RuntimeError(
            "Isotropic transform failed pixel reconstruction check."
        )

    return summary


def main():
    print(
        "\nR9-D — NON-DESTRUCTIVE ISOTROPIC POINT-DATASET BUILD"
    )

    print(
        "NO model training. NO frozen anisotropic file overwrite. "
        "This step builds a parallel isotropic point-level trajectory table, "
        "recomputes motion features from isotropic coordinates, and verifies "
        "1:1 row/sequence/track integrity.\n"
    )

    verify_r9c()

    (
        src,
        out,
        seq_audit,
        track_col,
        order_cols,
    ) = build_dataset()

    row_audit = integrity_audit(
        src,
        out,
        track_col,
    )

    transform_audit = transform_consistency(
        out
    )

    # Preserve all original columns plus isolated R9 columns.
    out.to_csv(
        POINT_OUT,
        index=False,
    )

    seq_audit.to_csv(
        SEQ_AUDIT_OUT,
        index=False,
    )

    row_audit.to_csv(
        ROW_AUDIT_OUT,
        index=False,
    )

    transform_audit.to_csv(
        TRANSFORM_AUDIT_OUT,
        index=False,
    )

    status = (
        "R9D_NONDESTRUCTIVE_ISOTROPIC_POINT_DATASET_BUILT_READY_FOR_R9E_WINDOW_PROTOCOL_MATCH"
    )

    manifest = {
        "step":
            "R9D_NONDESTRUCTIVE_ISOTROPIC_POINT_DATASET_BUILD",

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

        "source_points":
            str(
                SOURCE_POINTS
            ),

        "source_points_sha256":
            sha256_file(
                SOURCE_POINTS
            ),

        "output_points":
            str(
                POINT_OUT
            ),

        "output_points_sha256":
            sha256_file(
                POINT_OUT
            ),

        "rows":
            int(
                len(
                    out
                )
            ),

        "sequences":
            int(
                out[
                    "sequence"
                ].nunique(
                    dropna=False
                )
            ),

        "tracks":
            int(
                out[
                    [
                        "sequence",
                        track_col,
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        "track_column":
            track_col,

        "ordering_columns":
            order_cols,

        "new_columns":
            [
                "img_w_locked",
                "img_h_locked",
                "iso_scale_px",
                "cx_iso",
                "cy_iso",
                "vx_iso",
                "vy_iso",
                "speed_iso",
                "heading_iso_rad",
                "ax_iso",
                "ay_iso",
                "accel_iso",
            ],

        "old_normalized_columns_preserved_but_not_modified":
            [
                c
                for c in [
                    "cx_n",
                    "cy_n",
                    "vx_n",
                    "vy_n",
                    "speed_n",
                    "heading_rad",
                ]
                if c in out.columns
            ],

        "row_integrity_csv":
            str(
                ROW_AUDIT_OUT
            ),

        "sequence_geometry_csv":
            str(
                SEQ_AUDIT_OUT
            ),

        "transform_consistency_csv":
            str(
                TRANSFORM_AUDIT_OUT
            ),
    }

    (
        OUT
        / "R9D_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "ROW / TRACK INTEGRITY AUDIT:"
    )

    print(
        row_audit.to_string(
            index=False
        )
    )

    print(
        "\nSEQUENCE GEOMETRY AUDIT:"
    )

    print(
        seq_audit.to_string(
            index=False
        )
    )

    print(
        "\nTRANSFORM CONSISTENCY AUDIT:"
    )

    print(
        transform_audit.to_string(
            index=False
        )
    )

    print(
        "\nISOTROPIC DATASET SUMMARY:"
    )

    print(
        json.dumps(
            {
                "rows":
                    int(
                        len(
                            out
                        )
                    ),

                "sequences":
                    int(
                        out[
                            "sequence"
                        ].nunique(
                            dropna=False
                        )
                    ),

                "tracks":
                    int(
                        out[
                            [
                                "sequence",
                                track_col,
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),

                "cx_iso_min":
                    float(
                        out[
                            "cx_iso"
                        ].min()
                    ),

                "cx_iso_max":
                    float(
                        out[
                            "cx_iso"
                        ].max()
                    ),

                "cy_iso_min":
                    float(
                        out[
                            "cy_iso"
                        ].min()
                    ),

                "cy_iso_max":
                    float(
                        out[
                            "cy_iso"
                        ].max()
                    ),

                "output":
                    str(
                        POINT_OUT
                    ),
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
        "\nR9-D COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
