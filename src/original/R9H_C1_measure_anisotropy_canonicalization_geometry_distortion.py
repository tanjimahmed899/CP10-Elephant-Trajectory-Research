
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

CORR_DIR = MAIN1 / "corrected_trajectory_dataset"
BUILD_SRC = MAIN1 / "build_final_neural_dataset.py"

R9H_B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_B_exact_geometry_formula_lock"
)

R9H_B_MANIFEST = R9H_B_DIR / "R9H_B_MANIFEST.json"

FLOOR_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C0_V3_authoritative_scale_floor_lock"
)

FLOOR_MANIFEST = FLOOR_DIR / "R9H_C0_V3_MANIFEST.json"

ISO_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3B_corrected_isotropic_rebuild"
)

ISO_GEOMETRY = ISO_DIR / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C1_anisotropy_geometry_distortion"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "expected_samples": 1053,
        "future_len": 20,
    },
    "2s_to_5s": {
        "expected_samples": 826,
        "future_len": 50,
    },
}

RECENT_HEADING_FRAMES = 10

EXPECTED_BUILD_SRC_SHA256 = (
    "7156eeef6f91f6affe7067adc37fa8d20aafc34d82d53155c47019085aeb25e4"
)

PIXEL_IDENTITY_TOL = 1e-9


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


def verify_protocol_locks():
    for p in [
        BUILD_SRC,
        R9H_B_MANIFEST,
        FLOOR_MANIFEST,
        ISO_GEOMETRY,
    ]:
        require(p)

    build_hash = sha256_file(
        BUILD_SRC
    )

    if build_hash != EXPECTED_BUILD_SRC_SHA256:
        raise RuntimeError(
            "Frozen canonicalizer source hash changed. "
            f"Expected {EXPECTED_BUILD_SRC_SHA256}, got {build_hash}."
        )

    r9h_b = json.loads(
        R9H_B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        r9h_b.get("status")
        !=
        "R9H_B_EXACT_FROZEN_CANONICALIZATION_FORMULA_LOCKED_READY_FOR_R9H_C_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
    ):
        raise RuntimeError(
            f"Unexpected R9H-B status: {r9h_b.get('status')}"
        )

    if (
        r9h_b.get("selected_build_src_sha256")
        !=
        build_hash
    ):
        raise RuntimeError(
            "R9H-B locked build-source hash does not match current source."
        )

    floors = json.loads(
        FLOOR_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        floors.get("status")
        !=
        "R9H_C0_V3_AUTHORITATIVE_TASK_SCALE_FLOORS_LOCKED_READY_FOR_R9H_C1_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
    ):
        raise RuntimeError(
            f"Unexpected scale-floor lock status: {floors.get('status')}"
        )

    locked_floors = {
        str(k): float(v)
        for k, v in floors[
            "locked_scale_floors"
        ].items()
    }

    for task in TASKS:
        if task not in locked_floors:
            raise RuntimeError(
                f"Scale floor missing for {task}."
            )

    return (
        r9h_b,
        floors,
        locked_floors,
    )


# =====================================================================
# EXACT FROZEN CANONICALIZER FORMULA FROM build_final_neural_dataset.py
# Locked by R9-H-B. Do not change independently.
# =====================================================================

def history_path_length(
    history
):
    return float(
        np.linalg.norm(
            np.diff(
                history,
                axis=0
            ),
            axis=1
        ).sum()
    )


def estimate_heading(
    history
):
    recent = history[
        -RECENT_HEADING_FRAMES:
    ]

    t = np.arange(
        len(recent),
        dtype=np.float64
    )

    tc = (
        t
        -
        t.mean()
    )

    denom = np.sum(
        tc ** 2
    )

    if denom <= 1e-12:
        return 0.0

    centered = (
        recent
        -
        recent.mean(
            axis=0,
            keepdims=True
        )
    )

    velocity = (
        tc[:, None]
        *
        centered
    ).sum(
        axis=0
    ) / denom

    magnitude = np.linalg.norm(
        velocity
    )

    if magnitude <= 1e-10:
        return 0.0

    return float(
        np.arctan2(
            velocity[1],
            velocity[0]
        )
    )


def rotation_to_heading(
    angle
):
    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )

    return np.array(
        [
            [c, s],
            [-s, c]
        ],
        dtype=np.float64
    )


def canonicalize_sample_exact(
    history,
    future,
    scale_floor
):
    history = np.asarray(
        history,
        dtype=np.float64
    )

    future = np.asarray(
        future,
        dtype=np.float64
    )

    origin = history[
        -1
    ].copy()

    angle = estimate_heading(
        history
    )

    R = rotation_to_heading(
        angle
    )

    path_length = history_path_length(
        history
    )

    scale = max(
        path_length,
        scale_floor
    )

    h_local = (
        history
        -
        origin
    )

    f_local = (
        future
        -
        origin
    )

    h_rot = (
        h_local
        @
        R.T
    )

    f_rot = (
        f_local
        @
        R.T
    )

    h_can = (
        h_rot
        /
        scale
    )

    f_can = (
        f_rot
        /
        scale
    )

    return {
        "history_canonical":
            h_can,

        "future_canonical":
            f_can,

        "origin":
            origin,

        "angle":
            float(
                angle
            ),

        "rotation":
            R,

        "scale":
            float(
                scale
            ),

        "raw_path_length":
            float(
                path_length
            ),

        "floor_applied":
            bool(
                path_length
                <
                scale_floor
            ),
    }


def load_geometry():
    df = pd.read_csv(
        ISO_GEOMETRY,
        dtype={
            "sequence": str,
        }
    )

    required = {
        "sequence",
        "width_px",
        "height_px",
        "iso_scale_px",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"Geometry lock missing columns: {sorted(missing)}"
        )

    return {
        str(r["sequence"]): {
            "W": float(
                r["width_px"]
            ),
            "H": float(
                r["height_px"]
            ),
            "S": float(
                r["iso_scale_px"]
            ),
        }
        for _, r in df.iterrows()
    }


def load_anisotropic(task):
    meta_path = (
        CORR_DIR
        /
        f"corrected_windows_{task}.csv"
    )

    npz_path = (
        CORR_DIR
        /
        f"corrected_{task}.npz"
    )

    require(
        meta_path
    )

    require(
        npz_path
    )

    meta = pd.read_csv(
        meta_path,
        dtype={
            "split": str,
            "sequence": str,
            "segment_id": str,
        }
    )

    arrays = np.load(
        npz_path
    )

    core = meta[
        (
            pd.to_numeric(
                meta["accepted"],
                errors="coerce"
            )
            ==
            1
        )
        &
        (
            meta[
                "quality_tier"
            ]
            .astype(str)
            .str.lower()
            ==
            "core"
        )
        &
        (
            meta[
                "split"
            ]
            .astype(str)
            ==
            "test_real"
        )
    ].copy()

    core = (
        core
        .sort_values(
            "sample_id",
            kind="mergesort"
        )
        .reset_index(
            drop=True
        )
    )

    ids = pd.to_numeric(
        core[
            "sample_id"
        ],
        errors="raise"
    ).astype(
        int
    )

    history = np.asarray(
        arrays[
            "history"
        ][
            ids.to_numpy(),
            :,
            :2
        ],
        dtype=np.float64
    )[
        :,
        ::3,
        :
    ]

    future = np.asarray(
        arrays[
            "future"
        ][
            ids.to_numpy(),
            :,
            :2
        ],
        dtype=np.float64
    )[
        :,
        ::3,
        :
    ]

    return (
        core,
        history,
        future,
        meta_path,
        npz_path,
    )


def load_isotropic(task):
    meta_path = (
        ISO_DIR
        /
        f"R9G_V3B_corrected_isotropic_windows_{task}.csv"
    )

    npz_path = (
        ISO_DIR
        /
        f"R9G_V3B_corrected_isotropic_{task}.npz"
    )

    require(
        meta_path
    )

    require(
        npz_path
    )

    meta = pd.read_csv(
        meta_path,
        dtype={
            "sequence": str,
            "segment_id": str,
        }
    )

    arrays = np.load(
        npz_path
    )

    history = np.asarray(
        arrays[
            "history"
        ],
        dtype=np.float64
    )

    future = np.asarray(
        arrays[
            "future"
        ],
        dtype=np.float64
    )

    return (
        meta,
        history,
        future,
        meta_path,
        npz_path,
    )


def periodic_angle_difference(
    a,
    b
):
    return float(
        np.arctan2(
            np.sin(
                a - b
            ),
            np.cos(
                a - b
            )
        )
    )


def pointwise_l2_summary(
    a,
    b
):
    d = np.linalg.norm(
        np.asarray(
            a,
            dtype=np.float64
        )
        -
        np.asarray(
            b,
            dtype=np.float64
        ),
        axis=1
    )

    return (
        float(
            d.mean()
        ),
        float(
            np.median(
                d
            )
        ),
        float(
            d.max()
        ),
        float(
            d[-1]
        ),
    )


def origin_to_pixel(
    origin,
    representation,
    geom
):
    p = np.asarray(
        origin,
        dtype=np.float64
    ).copy()

    if representation == "anisotropic":
        p[0] *= geom["W"]
        p[1] *= geom["H"]

        return p

    if representation == "isotropic":
        p *= geom["S"]

        return p

    raise ValueError(
        representation
    )


def compare_task(
    task,
    scale_floor,
    geometry
):
    (
        ani_meta,
        ani_h,
        ani_f,
        ani_meta_path,
        ani_npz_path,
    ) = load_anisotropic(
        task
    )

    (
        iso_meta,
        iso_h,
        iso_f,
        iso_meta_path,
        iso_npz_path,
    ) = load_isotropic(
        task
    )

    expected = TASKS[
        task
    ][
        "expected_samples"
    ]

    if (
        len(
            ani_meta
        )
        !=
        expected
        or
        len(
            iso_meta
        )
        !=
        expected
    ):
        raise RuntimeError(
            f"{task}: expected {expected} samples; "
            f"got anisotropic={len(ani_meta)}, isotropic={len(iso_meta)}."
        )

    keys = [
        "sample_id",
        "sequence",
        "segment_id",
    ]

    for key in keys:
        a = (
            ani_meta[
                key
            ]
            .astype(str)
            .to_numpy()
        )

        b = (
            iso_meta[
                key
            ]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(
            a,
            b
        ):
            raise RuntimeError(
                f"{task}: metadata mismatch in {key}."
            )

    if ani_h.shape != iso_h.shape:
        raise RuntimeError(
            f"{task}: history shape mismatch."
        )

    if ani_f.shape != iso_f.shape:
        raise RuntimeError(
            f"{task}: future shape mismatch."
        )

    rows = []

    for i in range(
        expected
    ):
        row = ani_meta.iloc[
            i
        ]

        sample_id = int(
            row[
                "sample_id"
            ]
        )

        sequence = str(
            row[
                "sequence"
            ]
        )

        segment_id = str(
            row[
                "segment_id"
            ]
        )

        if sequence not in geometry:
            raise RuntimeError(
                f"{task}: missing geometry for {sequence}."
            )

        geom = geometry[
            sequence
        ]

        a = canonicalize_sample_exact(
            ani_h[
                i
            ],
            ani_f[
                i
            ],
            scale_floor
        )

        z = canonicalize_sample_exact(
            iso_h[
                i
            ],
            iso_f[
                i
            ],
            scale_floor
        )

        angle_signed = periodic_angle_difference(
            a[
                "angle"
            ],
            z[
                "angle"
            ]
        )

        angle_abs_deg = float(
            np.degrees(
                abs(
                    angle_signed
                )
            )
        )

        angle_signed_deg = float(
            np.degrees(
                angle_signed
            )
        )

        rotation_fro = float(
            np.linalg.norm(
                a[
                    "rotation"
                ]
                -
                z[
                    "rotation"
                ],
                ord="fro"
            )
        )

        path_iso = z[
            "raw_path_length"
        ]

        path_aniso = a[
            "raw_path_length"
        ]

        scale_iso = z[
            "scale"
        ]

        scale_aniso = a[
            "scale"
        ]

        path_ratio = (
            path_aniso
            /
            path_iso
            if path_iso > 1e-15
            else np.nan
        )

        scale_ratio = (
            scale_aniso
            /
            scale_iso
            if scale_iso > 1e-15
            else np.nan
        )

        (
            h_mean,
            h_median,
            h_max,
            h_endpoint,
        ) = pointwise_l2_summary(
            a[
                "history_canonical"
            ],
            z[
                "history_canonical"
            ]
        )

        (
            f_mean,
            f_median,
            f_max,
            f_endpoint,
        ) = pointwise_l2_summary(
            a[
                "future_canonical"
            ],
            z[
                "future_canonical"
            ]
        )

        origin_a_px = origin_to_pixel(
            a[
                "origin"
            ],
            "anisotropic",
            geom
        )

        origin_z_px = origin_to_pixel(
            z[
                "origin"
            ],
            "isotropic",
            geom
        )

        origin_px_max_abs_diff = float(
            np.max(
                np.abs(
                    origin_a_px
                    -
                    origin_z_px
                )
            )
        )

        rows.append(
            {
                "task":
                    task,

                "sample_id":
                    sample_id,

                "sequence":
                    sequence,

                "segment_id":
                    segment_id,

                "aspect_ratio_W_over_H":
                    geom[
                        "W"
                    ]
                    /
                    geom[
                        "H"
                    ],

                "scale_floor":
                    scale_floor,

                "heading_anisotropic_rad":
                    a[
                        "angle"
                    ],

                "heading_isotropic_rad":
                    z[
                        "angle"
                    ],

                "heading_signed_difference_deg_aniso_minus_iso":
                    angle_signed_deg,

                "heading_abs_difference_deg":
                    angle_abs_deg,

                "rotation_matrix_frobenius_difference":
                    rotation_fro,

                "path_length_anisotropic":
                    path_aniso,

                "path_length_isotropic":
                    path_iso,

                "path_length_ratio_aniso_over_iso":
                    path_ratio,

                "path_length_percent_difference_aniso_minus_iso":
                    (
                        (
                            path_aniso
                            -
                            path_iso
                        )
                        /
                        path_iso
                        *
                        100.0
                    )
                    if path_iso > 1e-15
                    else np.nan,

                "scale_anisotropic":
                    scale_aniso,

                "scale_isotropic_fixed_floor":
                    scale_iso,

                "scale_ratio_aniso_over_iso":
                    scale_ratio,

                "scale_percent_difference_aniso_minus_iso":
                    (
                        (
                            scale_aniso
                            -
                            scale_iso
                        )
                        /
                        scale_iso
                        *
                        100.0
                    )
                    if scale_iso > 1e-15
                    else np.nan,

                "floor_applied_anisotropic":
                    a[
                        "floor_applied"
                    ],

                "floor_applied_isotropic_fixed_floor":
                    z[
                        "floor_applied"
                    ],

                "floor_activation_changed":
                    bool(
                        a[
                            "floor_applied"
                        ]
                        !=
                        z[
                            "floor_applied"
                        ]
                    ),

                "history_canonical_mean_point_l2_difference":
                    h_mean,

                "history_canonical_median_point_l2_difference":
                    h_median,

                "history_canonical_max_point_l2_difference":
                    h_max,

                "history_canonical_endpoint_l2_difference":
                    h_endpoint,

                "future_canonical_mean_point_l2_difference":
                    f_mean,

                "future_canonical_median_point_l2_difference":
                    f_median,

                "future_canonical_max_point_l2_difference":
                    f_max,

                "future_canonical_endpoint_l2_difference":
                    f_endpoint,

                "origin_pixel_max_abs_difference":
                    origin_px_max_abs_diff,
            }
        )

    provenance = {
        "anisotropic_metadata":
            str(
                ani_meta_path
            ),

        "anisotropic_metadata_sha256":
            sha256_file(
                ani_meta_path
            ),

        "anisotropic_npz":
            str(
                ani_npz_path
            ),

        "anisotropic_npz_sha256":
            sha256_file(
                ani_npz_path
            ),

        "corrected_isotropic_metadata":
            str(
                iso_meta_path
            ),

        "corrected_isotropic_metadata_sha256":
            sha256_file(
                iso_meta_path
            ),

        "corrected_isotropic_npz":
            str(
                iso_npz_path
            ),

        "corrected_isotropic_npz_sha256":
            sha256_file(
                iso_npz_path
            ),
    }

    return (
        pd.DataFrame(
            rows
        ),
        provenance,
    )


def percentile(
    series,
    q
):
    return float(
        np.nanpercentile(
            pd.to_numeric(
                series,
                errors="coerce"
            ).to_numpy(
                dtype=np.float64
            ),
            q
        )
    )


def summarize_task(
    g
):
    heading = g[
        "heading_abs_difference_deg"
    ]

    path_pct = g[
        "path_length_percent_difference_aniso_minus_iso"
    ]

    scale_pct = g[
        "scale_percent_difference_aniso_minus_iso"
    ]

    future_can = g[
        "future_canonical_mean_point_l2_difference"
    ]

    history_can = g[
        "history_canonical_mean_point_l2_difference"
    ]

    return {
        "samples":
            int(
                len(
                    g
                )
            ),

        "segments":
            int(
                g[
                    [
                        "sequence",
                        "segment_id",
                    ]
                ]
                .drop_duplicates()
                .shape[
                    0
                ]
            ),

        "sequences":
            int(
                g[
                    "sequence"
                ].nunique()
            ),

        "heading_abs_diff_median_deg":
            percentile(
                heading,
                50
            ),

        "heading_abs_diff_q90_deg":
            percentile(
                heading,
                90
            ),

        "heading_abs_diff_q95_deg":
            percentile(
                heading,
                95
            ),

        "heading_abs_diff_max_deg":
            float(
                heading.max()
            ),

        "heading_diff_gt_5deg_pct":
            float(
                (
                    heading
                    >
                    5.0
                ).mean()
                *
                100.0
            ),

        "heading_diff_gt_10deg_pct":
            float(
                (
                    heading
                    >
                    10.0
                ).mean()
                *
                100.0
            ),

        "heading_diff_gt_20deg_pct":
            float(
                (
                    heading
                    >
                    20.0
                ).mean()
                *
                100.0
            ),

        "rotation_fro_diff_median":
            percentile(
                g[
                    "rotation_matrix_frobenius_difference"
                ],
                50
            ),

        "path_length_pct_diff_median":
            percentile(
                path_pct,
                50
            ),

        "path_length_pct_diff_q05":
            percentile(
                path_pct,
                5
            ),

        "path_length_pct_diff_q95":
            percentile(
                path_pct,
                95
            ),

        "scale_pct_diff_median":
            percentile(
                scale_pct,
                50
            ),

        "scale_pct_diff_q05":
            percentile(
                scale_pct,
                5
            ),

        "scale_pct_diff_q95":
            percentile(
                scale_pct,
                95
            ),

        "floor_applied_anisotropic_pct":
            float(
                g[
                    "floor_applied_anisotropic"
                ].mean()
                *
                100.0
            ),

        "floor_applied_isotropic_fixed_floor_pct":
            float(
                g[
                    "floor_applied_isotropic_fixed_floor"
                ].mean()
                *
                100.0
            ),

        "floor_activation_changed_pct":
            float(
                g[
                    "floor_activation_changed"
                ].mean()
                *
                100.0
            ),

        "history_canonical_mean_l2_median":
            percentile(
                history_can,
                50
            ),

        "history_canonical_mean_l2_q90":
            percentile(
                history_can,
                90
            ),

        "future_canonical_mean_l2_median":
            percentile(
                future_can,
                50
            ),

        "future_canonical_mean_l2_q90":
            percentile(
                future_can,
                90
            ),

        "future_canonical_endpoint_l2_median":
            percentile(
                g[
                    "future_canonical_endpoint_l2_difference"
                ],
                50
            ),

        "future_canonical_endpoint_l2_q90":
            percentile(
                g[
                    "future_canonical_endpoint_l2_difference"
                ],
                90
            ),

        "max_origin_pixel_abs_difference":
            float(
                g[
                    "origin_pixel_max_abs_difference"
                ].max()
            ),
    }


def build_task_summary(
    df
):
    rows = []

    for task, g in df.groupby(
        "task",
        sort=True
    ):
        row = {
            "task":
                task
        }

        row.update(
            summarize_task(
                g
            )
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_sequence_summary(
    df
):
    rows = []

    for (
        task,
        sequence,
    ), g in df.groupby(
        [
            "task",
            "sequence",
        ],
        sort=True
    ):
        rows.append(
            {
                "task":
                    task,

                "sequence":
                    sequence,

                "samples":
                    int(
                        len(
                            g
                        )
                    ),

                "segments":
                    int(
                        g[
                            "segment_id"
                        ].nunique()
                    ),

                "aspect_ratio_W_over_H":
                    float(
                        g[
                            "aspect_ratio_W_over_H"
                        ].iloc[
                            0
                        ]
                    ),

                "heading_abs_diff_median_deg":
                    percentile(
                        g[
                            "heading_abs_difference_deg"
                        ],
                        50
                    ),

                "heading_abs_diff_q90_deg":
                    percentile(
                        g[
                            "heading_abs_difference_deg"
                        ],
                        90
                    ),

                "path_length_pct_diff_median":
                    percentile(
                        g[
                            "path_length_percent_difference_aniso_minus_iso"
                        ],
                        50
                    ),

                "future_canonical_mean_l2_median":
                    percentile(
                        g[
                            "future_canonical_mean_point_l2_difference"
                        ],
                        50
                    ),

                "floor_activation_changed_pct":
                    float(
                        g[
                            "floor_activation_changed"
                        ].mean()
                        *
                        100.0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-H-C1 — ANISOTROPIC NORMALIZATION CANONICALIZATION-GEOMETRY DISTORTION AUDIT"
    )

    print(
        "NO training and NO model inference. The exact frozen canonicalizer formula "
        "locked in R9-H-B is applied to the exact same camera-corrected TestReal windows "
        "in two coordinate representations: existing anisotropic x/W,y/H and "
        "aspect-preserving isotropic x/S,y/S with S=max(W,H). "
        "Task-specific frozen scale floors are taken from the R9-H-C0-V3 authoritative lock.\n"
    )

    (
        r9h_b,
        floor_manifest,
        floors,
    ) = verify_protocol_locks()

    geometry = load_geometry()

    dfs = []
    provenance = {}

    for task in TASKS:
        df, prov = compare_task(
            task,
            floors[
                task
            ],
            geometry
        )

        dfs.append(
            df
        )

        provenance[
            task
        ] = prov

    per_sample = pd.concat(
        dfs,
        ignore_index=True
    )

    if (
        per_sample[
            "origin_pixel_max_abs_difference"
        ].max()
        >
        PIXEL_IDENTITY_TOL
    ):
        raise RuntimeError(
            "Pixel-equivalent origin identity check failed. "
            "Representation branches may not refer to the same corrected coordinates."
        )

    task_summary = build_task_summary(
        per_sample
    )

    sequence_summary = build_sequence_summary(
        per_sample
    )

    per_sample_path = (
        OUT
        /
        "R9H_C1_PER_SAMPLE_GEOMETRY_DISTORTION.csv"
    )

    task_summary_path = (
        OUT
        /
        "R9H_C1_TASK_SUMMARY.csv"
    )

    sequence_summary_path = (
        OUT
        /
        "R9H_C1_SEQUENCE_SUMMARY.csv"
    )

    per_sample.to_csv(
        per_sample_path,
        index=False
    )

    task_summary.to_csv(
        task_summary_path,
        index=False
    )

    sequence_summary.to_csv(
        sequence_summary_path,
        index=False
    )

    status = (
        "R9H_C1_ANISOTROPY_CANONICALIZATION_GEOMETRY_DISTORTION_MEASURED_READY_FOR_R9H_C2_CLUSTERED_CI_AND_REPRESENTATION_DECISION"
    )

    manifest = {
        "step":
            "R9H_C1_ANISOTROPY_CANONICALIZATION_GEOMETRY_DISTORTION_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "scientific_scope":
            (
                "Geometry/canonicalization diagnostic only. "
                "No neural-model performance claim is made."
            ),

        "anisotropic_representation":
            "x/W, y/H",

        "isotropic_representation":
            "x/S, y/S where S=max(W,H)",

        "same_camera_corrected_coordinates":
            True,

        "same_samples":
            True,

        "same_frozen_canonicalizer_formula":
            True,

        "fixed_frozen_scale_floors":
            floors,

        "important_interpretation":
            [
                (
                    "Heading-angle and raw path-length differences quantify the "
                    "geometric effect of axis-anisotropic normalization."
                ),
                (
                    "Canonical-tensor differences quantify how much the frozen "
                    "canonicalizer input/output representation changes under "
                    "aspect-preserving coordinates."
                ),
                (
                    "The isotropic branch uses the existing frozen numeric scale floor "
                    "for this diagnostic. A future isotropic neural-model retraining "
                    "would need its scale floor re-derived from isotropic TrainReal data."
                ),
                (
                    "This step does not evaluate frozen neural checkpoints on isotropic "
                    "inputs and does not claim improved forecasting performance."
                ),
            ],

        "canonicalizer_source":
            str(
                BUILD_SRC
            ),

        "canonicalizer_source_sha256":
            sha256_file(
                BUILD_SRC
            ),

        "r9h_b_manifest":
            str(
                R9H_B_MANIFEST
            ),

        "scale_floor_manifest":
            str(
                FLOOR_MANIFEST
            ),

        "new_training":
            False,

        "model_inference":
            False,

        "bootstrap_performed":
            False,

        "frozen_results_modified":
            False,

        "per_sample_csv":
            str(
                per_sample_path
            ),

        "per_sample_sha256":
            sha256_file(
                per_sample_path
            ),

        "task_summary_csv":
            str(
                task_summary_path
            ),

        "task_summary_sha256":
            sha256_file(
                task_summary_path
            ),

        "sequence_summary_csv":
            str(
                sequence_summary_path
            ),

        "sequence_summary_sha256":
            sha256_file(
                sequence_summary_path
            ),

        "provenance":
            provenance,
    }

    (
        OUT
        /
        "R9H_C1_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "TASK-LEVEL GEOMETRY DISTORTION SUMMARY:"
    )

    print(
        task_summary.to_string(
            index=False
        )
    )

    print(
        "\nSEQUENCE-LEVEL SUMMARY:"
    )

    print(
        sequence_summary.to_string(
            index=False
        )
    )

    print(
        "\nKEY SANITY CHECKS:"
    )

    print(
        "  max pixel-equivalent origin difference:",
        float(
            per_sample[
                "origin_pixel_max_abs_difference"
            ].max()
        )
    )

    print(
        "  samples:",
        {
            task:
                int(
                    (
                        per_sample[
                            "task"
                        ]
                        ==
                        task
                    ).sum()
                )
            for task in TASKS
        }
    )

    print(
        "  locked scale floors:",
        json.dumps(
            floors
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-H-C1 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
