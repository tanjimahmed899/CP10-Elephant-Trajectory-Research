
from pathlib import Path
import json
import hashlib
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

R9B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9B_exact_normalization_source_lock"
)

R9B_MANIFEST = R9B_DIR / "R9B_MANIFEST.json"
R9B_SOURCE_LOCK = R9B_DIR / "R9B_SOURCE_LOCK.json"

CLEANED = (
    MAIN1
    / "trajectory_dataset"
    / "cleaned_track_points.csv"
)

SOURCE_SCRIPT = (
    MAIN1
    / "build_elephant_trajectory_dataset.py"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9C_isotropic_transform_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

RECON_TOL = 5e-4


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


def verify_r9b():
    for p in [
        R9B_MANIFEST,
        R9B_SOURCE_LOCK,
        CLEANED,
        SOURCE_SCRIPT,
    ]:
        require(p)

    manifest = json.loads(
        R9B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("status")
        !=
        "R9B_EXACT_NORMALIZATION_SOURCE_RECOVERY_COMPLETE_READY_FOR_R9C_TRANSFORM_LOCK"
    ):
        raise RuntimeError(
            f"Unexpected R9B status: {manifest.get('status')}"
        )

    source_lock = json.loads(
        R9B_SOURCE_LOCK.read_text(
            encoding="utf-8"
        )
    )

    assessment = (
        source_lock
        .get("representation_assessment", {})
        .get("empirical_representation_assessment")
    )

    if (
        assessment
        !=
        "EMPIRICALLY_CONSISTENT_WITH_SEPARATE_X_Y_DENOMINATORS"
    ):
        raise RuntimeError(
            f"R9C requires separate X/Y denominator evidence; got {assessment}"
        )

    return manifest, source_lock


def extract_source_context():
    text = SOURCE_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    rows = []

    for i, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        if (
            "cx_n" in lower
            or "cy_n" in lower
            or "vx_n" in lower
            or "vy_n" in lower
        ):
            a = max(
                1,
                i - 3,
            )

            b = min(
                len(lines),
                i + 3,
            )

            context = "\n".join(
                f"{j}: {lines[j-1]}"
                for j in range(
                    a,
                    b + 1,
                )
            )

            rows.append(
                {
                    "line_number":
                        i,

                    "line":
                        line,

                    "context":
                        context,
                }
            )

    if not rows:
        raise RuntimeError(
            "No cx_n/cy_n/vx_n/vy_n definitions found in source script."
        )

    return pd.DataFrame(
        rows
    )


def infer_integer_denominators(df):
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
            "Zero normalized coordinate encountered; denominator recovery needs explicit handling."
        )

    w_float = (
        cx
        /
        cx_n
    )

    h_float = (
        cy
        /
        cy_n
    )

    w = np.rint(
        w_float
    ).astype(int)

    h = np.rint(
        h_float
    ).astype(int)

    if (
        (w <= 0).any()
        or
        (h <= 0).any()
    ):
        raise RuntimeError(
            "Recovered non-positive width/height denominator."
        )

    recon_x = (
        cx
        /
        w
    )

    recon_y = (
        cy
        /
        h
    )

    err_x = np.abs(
        recon_x
        -
        cx_n
    )

    err_y = np.abs(
        recon_y
        -
        cy_n
    )

    audit = pd.DataFrame(
        {
            "implied_width_float":
                w_float,

            "locked_width_px":
                w,

            "cx_n_abs_reconstruction_error":
                err_x,

            "implied_height_float":
                h_float,

            "locked_height_px":
                h,

            "cy_n_abs_reconstruction_error":
                err_y,
        }
    )

    return audit


def find_group_columns(df):
    candidates = [
        "sequence",
        "sequence_id",
        "source",
        "source_group",
        "video",
        "video_id",
        "seq",
        "seq_id",
        "track_key",
        "track_id",
        "obj_id",
    ]

    return [
        c
        for c in candidates
        if c in df.columns
    ]


def denominator_stability(df, audit):
    merged = df.copy()

    merged["locked_width_px"] = (
        audit["locked_width_px"]
        .to_numpy()
    )

    merged["locked_height_px"] = (
        audit["locked_height_px"]
        .to_numpy()
    )

    groups = find_group_columns(
        merged
    )

    rows = []

    for c in groups:
        agg = (
            merged.groupby(
                c,
                dropna=False,
            )
            .agg(
                rows=(
                    "locked_width_px",
                    "size",
                ),
                unique_widths=(
                    "locked_width_px",
                    "nunique",
                ),
                unique_heights=(
                    "locked_height_px",
                    "nunique",
                ),
            )
            .reset_index()
        )

        rows.append(
            {
                "group_column":
                    c,

                "groups":
                    int(
                        len(
                            agg
                        )
                    ),

                "groups_with_single_width":
                    int(
                        (
                            agg[
                                "unique_widths"
                            ]
                            ==
                            1
                        ).sum()
                    ),

                "groups_with_single_height":
                    int(
                        (
                            agg[
                                "unique_heights"
                            ]
                            ==
                            1
                        ).sum()
                    ),

                "groups_with_single_width_and_height":
                    int(
                        (
                            (
                                agg[
                                    "unique_widths"
                                ]
                                ==
                                1
                            )
                            &
                            (
                                agg[
                                    "unique_heights"
                                ]
                                ==
                                1
                            )
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_transform_preview(df, audit):
    out = pd.DataFrame(
        index=df.index
    )

    for c in [
        "sequence",
        "sequence_id",
        "source",
        "source_group",
        "track_key",
        "track_id",
        "frame",
        "frame_id",
        "cx",
        "cy",
        "cx_n",
        "cy_n",
    ]:
        if c in df.columns:
            out[c] = df[c]

    out[
        "locked_width_px"
    ] = audit[
        "locked_width_px"
    ].to_numpy()

    out[
        "locked_height_px"
    ] = audit[
        "locked_height_px"
    ].to_numpy()

    out[
        "isotropic_scale_px"
    ] = np.maximum(
        out[
            "locked_width_px"
        ],
        out[
            "locked_height_px"
        ],
    )

    out[
        "cx_iso"
    ] = (
        pd.to_numeric(
            df["cx"],
            errors="raise",
        ).astype(float)
        /
        out[
            "isotropic_scale_px"
        ]
    )

    out[
        "cy_iso"
    ] = (
        pd.to_numeric(
            df["cy"],
            errors="raise",
        ).astype(float)
        /
        out[
            "isotropic_scale_px"
        ]
    )

    out[
        "aspect_ratio_width_over_height"
    ] = (
        out[
            "locked_width_px"
        ]
        /
        out[
            "locked_height_px"
        ]
    )

    return out


def validate_isotropic_geometry(preview):
    sx = (
        1.0
        /
        preview[
            "isotropic_scale_px"
        ].astype(float)
    )

    sy = (
        1.0
        /
        preview[
            "isotropic_scale_px"
        ].astype(float)
    )

    equal_scale = np.allclose(
        sx,
        sy,
        atol=0.0,
        rtol=0.0,
    )

    in_range = bool(
        (
            (
                preview[
                    "cx_iso"
                ]
                >=
                0
            )
            &
            (
                preview[
                    "cx_iso"
                ]
                <=
                1.0
                +
                1e-9
            )
            &
            (
                preview[
                    "cy_iso"
                ]
                >=
                0
            )
            &
            (
                preview[
                    "cy_iso"
                ]
                <=
                1.0
                +
                1e-9
            )
        ).all()
    )

    return {
        "same_scale_applied_to_x_and_y":
            bool(
                equal_scale
            ),

        "all_isotropic_coordinates_within_0_1":
            in_range,

        "scale_definition":
            "s_i = max(W_i, H_i)",

        "x_definition":
            "cx_iso_i = cx_i / s_i",

        "y_definition":
            "cy_iso_i = cy_i / s_i",

        "velocity_rule":
            (
                "Do NOT rescale frozen vx_n/vy_n independently. "
                "Recompute velocity/displacement from cx_iso/cy_iso after transformation."
            ),

        "canonicalization_rule":
            (
                "Any translation/rotation/path-length canonicalization in the R9 branch "
                "must be recomputed from isotropic coordinates; do not reuse anisotropic "
                "canonical tensors."
            ),
    }


def main():
    print(
        "\nR9-C — EXACT ANISOTROPIC FORMULA VERIFICATION + ISOTROPIC TRANSFORM LOCK"
    )

    print(
        "NO model training. NO frozen-output modification. "
        "This step verifies the existing normalized coordinates, locks per-row W/H "
        "denominators, and defines the isolated isotropic branch using one common pixel scale.\n"
    )

    verify_r9b()

    source_context = extract_source_context()

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
            f"Missing columns in cleaned points: {sorted(missing)}"
        )

    audit = infer_integer_denominators(
        df
    )

    max_x_err = float(
        audit[
            "cx_n_abs_reconstruction_error"
        ].max()
    )

    max_y_err = float(
        audit[
            "cy_n_abs_reconstruction_error"
        ].max()
    )

    if (
        max_x_err
        >
        RECON_TOL
        or
        max_y_err
        >
        RECON_TOL
    ):
        raise RuntimeError(
            f"Rounded W/H do not reconstruct normalized coordinates within tolerance. "
            f"max_x_err={max_x_err}, max_y_err={max_y_err}"
        )

    stability = denominator_stability(
        df,
        audit,
    )

    preview = build_transform_preview(
        df,
        audit,
    )

    geometry = validate_isotropic_geometry(
        preview
    )

    unique_pairs = (
        preview[
            [
                "locked_width_px",
                "locked_height_px",
            ]
        ]
        .value_counts()
        .reset_index(
            name="rows"
        )
        .sort_values(
            "rows",
            ascending=False,
        )
    )

    source_context_path = (
        OUT
        / "R9C_SOURCE_CODE_NORMALIZATION_CONTEXT.csv"
    )

    denom_audit_path = (
        OUT
        / "R9C_DENOMINATOR_RECONSTRUCTION_AUDIT.csv"
    )

    stability_path = (
        OUT
        / "R9C_DENOMINATOR_GROUP_STABILITY.csv"
    )

    pair_path = (
        OUT
        / "R9C_WIDTH_HEIGHT_PAIR_COUNTS.csv"
    )

    preview_path = (
        OUT
        / "R9C_ISOTROPIC_TRANSFORM_PREVIEW.csv"
    )

    source_context.to_csv(
        source_context_path,
        index=False,
    )

    audit.to_csv(
        denom_audit_path,
        index=False,
    )

    stability.to_csv(
        stability_path,
        index=False,
    )

    unique_pairs.to_csv(
        pair_path,
        index=False,
    )

    preview.head(
        5000
    ).to_csv(
        preview_path,
        index=False,
    )

    transform_lock = {
        "source_points":
            str(
                CLEANED
            ),

        "source_points_sha256":
            sha256_file(
                CLEANED
            ),

        "source_builder_script":
            str(
                SOURCE_SCRIPT
            ),

        "source_builder_script_sha256":
            sha256_file(
                SOURCE_SCRIPT
            ),

        "existing_representation":
            {
                "x":
                    "cx_n uses an X-specific image-scale denominator W_i",

                "y":
                    "cy_n uses a Y-specific image-scale denominator H_i",

                "geometry_implication":
                    (
                        "When W_i != H_i, Euclidean distances/angles in (cx_n, cy_n) "
                        "are anisotropically scaled relative to pixel geometry."
                    ),
            },

        "denominator_reconstruction":
            {
                "width":
                    "W_i = round(cx_i / cx_n_i)",

                "height":
                    "H_i = round(cy_i / cy_n_i)",

                "max_cx_n_reconstruction_error":
                    max_x_err,

                "max_cy_n_reconstruction_error":
                    max_y_err,

                "tolerance":
                    RECON_TOL,
            },

        "new_isotropic_branch":
            {
                "status":
                    "LOCKED_NON_DESTRUCTIVE_ABLATION_BRANCH",

                "common_scale":
                    "s_i = max(W_i, H_i)",

                "x_iso":
                    "cx_iso_i = cx_i / s_i",

                "y_iso":
                    "cy_iso_i = cy_i / s_i",

                "why_max_dimension":
                    (
                        "A single denominator preserves equal x/y metric scaling, "
                        "preserves image aspect ratio, and keeps both coordinates "
                        "within [0,1] without clipping."
                    ),

                "velocity_rule":
                    (
                        "Recompute velocities/displacements from isotropic coordinates; "
                        "do not reuse vx_n/vy_n."
                    ),

                "canonicalization_rule":
                    (
                        "Recompute all translation/rotation/path-length canonicalization "
                        "from isotropic coordinates."
                    ),

                "frozen_branch_rule":
                    (
                        "Existing cx_n/cy_n datasets, checkpoints, predictions, conformal q, "
                        "warning results and manuscript locks remain unchanged."
                    ),
            },

        "geometry_validation":
            geometry,
    }

    transform_lock_path = (
        OUT
        / "R9C_ISOTROPIC_TRANSFORM_LOCK.json"
    )

    transform_lock_path.write_text(
        json.dumps(
            transform_lock,
            indent=2,
        ),
        encoding="utf-8",
    )

    status = (
        "R9C_ISOTROPIC_TRANSFORM_FORMALLY_LOCKED_READY_FOR_R9D_NONDESTRUCTIVE_ABLATION_BUILD"
    )

    manifest = {
        "step":
            "R9C_ISOTROPIC_TRANSFORM_LOCK",

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

        "reconstruction_tolerance":
            RECON_TOL,

        "max_cx_n_reconstruction_error":
            max_x_err,

        "max_cy_n_reconstruction_error":
            max_y_err,

        "transform_lock_json":
            str(
                transform_lock_path
            ),

        "transform_lock_sha256":
            sha256_file(
                transform_lock_path
            ),

        "source_context_csv":
            str(
                source_context_path
            ),

        "width_height_pairs_csv":
            str(
                pair_path
            ),
    }

    (
        OUT
        / "R9C_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "SOURCE NORMALIZATION CODE CONTEXT:"
    )

    print(
        source_context.to_string(
            index=False
        )
    )

    print(
        "\nDENOMINATOR RECONSTRUCTION CHECK:"
    )

    print(
        json.dumps(
            {
                "rows":
                    int(
                        len(
                            audit
                        )
                    ),

                "max_cx_n_abs_reconstruction_error":
                    max_x_err,

                "max_cy_n_abs_reconstruction_error":
                    max_y_err,

                "tolerance":
                    RECON_TOL,
            },
            indent=2,
        )
    )

    print(
        "\nTOP WIDTH/HEIGHT PAIRS:"
    )

    print(
        unique_pairs.head(
            20
        ).to_string(
            index=False
        )
    )

    print(
        "\nDENOMINATOR GROUP STABILITY:"
    )

    if len(
        stability
    ):
        print(
            stability.to_string(
                index=False
            )
        )
    else:
        print(
            "No known grouping columns available; row-level reconstruction remains locked."
        )

    print(
        "\nISOTROPIC TRANSFORM LOCK:"
    )

    print(
        json.dumps(
            transform_lock[
                "new_isotropic_branch"
            ],
            indent=2,
        )
    )

    print(
        "\nGEOMETRY VALIDATION:"
    )

    print(
        json.dumps(
            geometry,
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
        "\nR9-C COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
