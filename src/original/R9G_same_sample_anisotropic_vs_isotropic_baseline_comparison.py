
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

R9D_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9D_isotropic_point_dataset_build"
)

R9F_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9F_V3C_exact_same_sample_isotropic_windows"
)

ISO_POINTS = (
    R9D_DIR
    / "R9D_isotropic_cleaned_track_points.csv"
)

R9F_MANIFEST = (
    R9F_DIR
    / "R9F_V3C_MANIFEST.json"
)

CORR_DIR = (
    MAIN1
    / "corrected_trajectory_dataset"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_same_sample_baseline_comparison"
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

MODELS = [
    "Stationary",
    "ConstantVelocity",
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


def verify_inputs():
    require(R9F_MANIFEST)
    require(ISO_POINTS)

    m = json.loads(
        R9F_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        m.get("status")
        !=
        "R9F_V3C_EXACT_SAME_SAMPLE_ISOTROPIC_WINDOWS_BUILT_READY_FOR_R9G_BASELINE_COMPARISON"
    ):
        raise RuntimeError(
            f"Unexpected R9F-V3C status: {m.get('status')}"
        )

    return m


def predict_stationary(history, future_len):
    return np.repeat(
        history[-1:, :],
        future_len,
        axis=0,
    )


def estimate_velocity_linear_regression(history, fps=10.0):
    h = np.asarray(
        history,
        dtype=np.float64,
    )

    t = np.arange(
        len(h),
        dtype=np.float64,
    ) / float(fps)

    tc = t - t.mean()
    denom = np.sum(
        tc ** 2
    )

    if denom <= 1e-12:
        return np.zeros(
            2,
            dtype=np.float64,
        )

    hc = (
        h
        -
        h.mean(
            axis=0,
            keepdims=True,
        )
    )

    vel = np.sum(
        tc[:, None]
        *
        hc,
        axis=0,
    ) / denom

    return vel


def predict_constant_velocity(
    history,
    future_len,
    fps=10.0,
):
    velocity = (
        estimate_velocity_linear_regression(
            history,
            fps=fps,
        )
    )

    future_times = (
        np.arange(
            1,
            future_len + 1,
            dtype=np.float64,
        )
        /
        float(fps)
    )

    return (
        history[-1][None, :]
        +
        future_times[:, None]
        *
        velocity[None, :]
    )


def predict(model, history, future_len):
    if model == "Stationary":
        return predict_stationary(
            history,
            future_len,
        )

    if model == "ConstantVelocity":
        return predict_constant_velocity(
            history,
            future_len,
            fps=10.0,
        )

    raise ValueError(
        f"Unknown model: {model}"
    )


def load_isotropic(task):
    npz_path = (
        R9F_DIR
        / f"R9F_V3C_isotropic_{task}.npz"
    )

    meta_path = (
        R9F_DIR
        / f"R9F_V3C_isotropic_windows_{task}.csv"
    )

    require(npz_path)
    require(meta_path)

    arr = np.load(
        npz_path
    )

    history = np.asarray(
        arr[
            "history"
        ],
        dtype=np.float64,
    )

    future = np.asarray(
        arr[
            "future"
        ],
        dtype=np.float64,
    )

    meta = pd.read_csv(
        meta_path,
        dtype={
            "sequence": str,
            "segment_id": str,
        },
    )

    return (
        history,
        future,
        meta,
        npz_path,
        meta_path,
    )


def load_anisotropic(task):
    meta_path = (
        CORR_DIR
        / f"corrected_windows_{task}.csv"
    )

    npz_path = (
        CORR_DIR
        / f"corrected_{task}.npz"
    )

    require(meta_path)
    require(npz_path)

    metadata = pd.read_csv(
        meta_path,
        dtype={
            "split": str,
            "sequence": str,
            "segment_id": str,
        },
    )

    arrays = np.load(
        npz_path
    )

    history_all = np.asarray(
        arrays[
            "history"
        ]
    )

    future_all = np.asarray(
        arrays[
            "future"
        ]
    )

    core = metadata[
        (
            pd.to_numeric(
                metadata[
                    "accepted"
                ],
                errors="coerce",
            )
            ==
            1
        )
        &
        (
            metadata[
                "quality_tier"
            ]
            .astype(str)
            .str.lower()
            ==
            "core"
        )
        &
        (
            metadata[
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
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    ids = pd.to_numeric(
        core[
            "sample_id"
        ],
        errors="raise",
    ).astype(int)

    history = np.asarray(
        history_all[
            ids.to_numpy(),
            :,
            :2,
        ],
        dtype=np.float64,
    )[
        :,
        ::3,
        :,
    ]

    future = np.asarray(
        future_all[
            ids.to_numpy(),
            :,
            :2,
        ],
        dtype=np.float64,
    )[
        :,
        ::3,
        :,
    ]

    return (
        history,
        future,
        core,
        npz_path,
        meta_path,
    )


def build_sequence_geometry_map():
    df = pd.read_csv(
        ISO_POINTS,
        dtype={
            "sequence": str,
        },
    )

    required = {
        "sequence",
        "img_w_locked",
        "img_h_locked",
        "iso_scale_px",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing geometry columns: {sorted(missing)}"
        )

    agg = (
        df.groupby(
            "sequence",
            dropna=False,
        )
        .agg(
            width_px=(
                "img_w_locked",
                "first",
            ),
            height_px=(
                "img_h_locked",
                "first",
            ),
            iso_scale_px=(
                "iso_scale_px",
                "first",
            ),
            n_width=(
                "img_w_locked",
                "nunique",
            ),
            n_height=(
                "img_h_locked",
                "nunique",
            ),
            n_scale=(
                "iso_scale_px",
                "nunique",
            ),
        )
        .reset_index()
    )

    if not (
        (
            agg[
                [
                    "n_width",
                    "n_height",
                    "n_scale",
                ]
            ]
            ==
            1
        ).all().all()
    ):
        raise RuntimeError(
            "Sequence geometry is not stable."
        )

    return {
        str(r["sequence"]): {
            "width_px": float(r["width_px"]),
            "height_px": float(r["height_px"]),
            "iso_scale_px": float(r["iso_scale_px"]),
        }
        for _, r in agg.iterrows()
    }


def to_pixel_equivalent(
    coords,
    representation,
    geometry,
):
    out = np.asarray(
        coords,
        dtype=np.float64,
    ).copy()

    if representation == "anisotropic":
        out[
            ...,
            0
        ] *= geometry[
            "width_px"
        ]

        out[
            ...,
            1
        ] *= geometry[
            "height_px"
        ]

        return out

    if representation == "isotropic":
        out *= geometry[
            "iso_scale_px"
        ]

        return out

    raise ValueError(
        representation
    )


def ade_fde(
    pred_px,
    truth_px,
):
    err = np.linalg.norm(
        pred_px
        -
        truth_px,
        axis=1,
    )

    return (
        float(
            np.mean(
                err
            )
        ),
        float(
            err[-1]
        ),
    )


def evaluate_task(
    task,
    geometry_map,
):
    (
        iso_h,
        iso_f,
        iso_meta,
        iso_npz,
        iso_meta_path,
    ) = load_isotropic(
        task
    )

    (
        ani_h,
        ani_f,
        ani_meta,
        ani_npz,
        ani_meta_path,
    ) = load_anisotropic(
        task
    )

    expected = TASKS[
        task
    ][
        "expected_samples"
    ]

    if (
        len(
            iso_meta
        )
        !=
        expected
        or
        len(
            ani_meta
        )
        !=
        expected
    ):
        raise RuntimeError(
            f"{task}: metadata count mismatch."
        )

    if not np.array_equal(
        pd.to_numeric(
            iso_meta[
                "sample_id"
            ],
            errors="raise",
        ).to_numpy(
            dtype=int
        ),
        pd.to_numeric(
            ani_meta[
                "sample_id"
            ],
            errors="raise",
        ).to_numpy(
            dtype=int
        ),
    ):
        raise RuntimeError(
            f"{task}: sample order mismatch."
        )

    if not np.array_equal(
        iso_meta[
            "sequence"
        ].astype(str)
        .to_numpy(),
        ani_meta[
            "sequence"
        ].astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            f"{task}: sequence order mismatch."
        )

    rows = []

    for i in range(
        expected
    ):
        sequence = str(
            iso_meta.iloc[
                i
            ][
                "sequence"
            ]
        )

        if sequence not in geometry_map:
            raise RuntimeError(
                f"{task}: geometry missing for {sequence}"
            )

        geometry = geometry_map[
            sequence
        ]

        sample_id = int(
            iso_meta.iloc[
                i
            ][
                "sample_id"
            ]
        )

        segment_id = str(
            iso_meta.iloc[
                i
            ][
                "segment_id"
            ]
        )

        for model in MODELS:
            pred_ani = predict(
                model,
                ani_h[
                    i
                ],
                ani_f.shape[
                    1
                ],
            )

            pred_iso = predict(
                model,
                iso_h[
                    i
                ],
                iso_f.shape[
                    1
                ],
            )

            pred_ani_px = to_pixel_equivalent(
                pred_ani,
                "anisotropic",
                geometry,
            )

            true_ani_px = to_pixel_equivalent(
                ani_f[
                    i
                ],
                "anisotropic",
                geometry,
            )

            pred_iso_px = to_pixel_equivalent(
                pred_iso,
                "isotropic",
                geometry,
            )

            true_iso_px = to_pixel_equivalent(
                iso_f[
                    i
                ],
                "isotropic",
                geometry,
            )

            ani_ade, ani_fde = ade_fde(
                pred_ani_px,
                true_ani_px,
            )

            iso_ade, iso_fde = ade_fde(
                pred_iso_px,
                true_iso_px,
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

                    "model":
                        model,

                    "width_px":
                        geometry[
                            "width_px"
                        ],

                    "height_px":
                        geometry[
                            "height_px"
                        ],

                    "iso_scale_px":
                        geometry[
                            "iso_scale_px"
                        ],

                    "anisotropic_ADE_px":
                        ani_ade,

                    "anisotropic_FDE_px":
                        ani_fde,

                    "isotropic_ADE_px":
                        iso_ade,

                    "isotropic_FDE_px":
                        iso_fde,

                    "ADE_change_px":
                        iso_ade
                        -
                        ani_ade,

                    "FDE_change_px":
                        iso_fde
                        -
                        ani_fde,

                    "ADE_improvement_pct":
                        (
                            (
                                ani_ade
                                -
                                iso_ade
                            )
                            /
                            ani_ade
                            *
                            100.0
                        )
                        if ani_ade
                        >
                        1e-12
                        else np.nan,

                    "FDE_improvement_pct":
                        (
                            (
                                ani_fde
                                -
                                iso_fde
                            )
                            /
                            ani_fde
                            *
                            100.0
                        )
                        if ani_fde
                        >
                        1e-12
                        else np.nan,
                }
            )

    provenance = {
        "isotropic_npz":
            str(
                iso_npz
            ),

        "isotropic_npz_sha256":
            sha256_file(
                iso_npz
            ),

        "isotropic_metadata":
            str(
                iso_meta_path
            ),

        "isotropic_metadata_sha256":
            sha256_file(
                iso_meta_path
            ),

        "anisotropic_npz":
            str(
                ani_npz
            ),

        "anisotropic_npz_sha256":
            sha256_file(
                ani_npz
            ),

        "anisotropic_metadata":
            str(
                ani_meta_path
            ),

        "anisotropic_metadata_sha256":
            sha256_file(
                ani_meta_path
            ),
    }

    return (
        pd.DataFrame(
            rows
        ),
        provenance,
    )


def summarize(sample_df):
    rows = []

    for (
        task,
        model,
    ), g in sample_df.groupby(
        [
            "task",
            "model",
        ],
        sort=True,
    ):
        ani_ade = float(
            g[
                "anisotropic_ADE_px"
            ].mean()
        )

        ani_fde = float(
            g[
                "anisotropic_FDE_px"
            ].mean()
        )

        iso_ade = float(
            g[
                "isotropic_ADE_px"
            ].mean()
        )

        iso_fde = float(
            g[
                "isotropic_FDE_px"
            ].mean()
        )

        # Segment macro = dependency-aware descriptive companion.
        segment_macro = (
            g.groupby(
                [
                    "sequence",
                    "segment_id",
                ],
                dropna=False,
            )[
                [
                    "anisotropic_ADE_px",
                    "anisotropic_FDE_px",
                    "isotropic_ADE_px",
                    "isotropic_FDE_px",
                ]
            ]
            .mean()
        )

        source_macro = (
            g.groupby(
                "sequence",
                dropna=False,
            )[
                [
                    "anisotropic_ADE_px",
                    "anisotropic_FDE_px",
                    "isotropic_ADE_px",
                    "isotropic_FDE_px",
                ]
            ]
            .mean()
        )

        rows.append(
            {
                "task":
                    task,

                "model":
                    model,

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

                "anisotropic_mean_ADE_px":
                    ani_ade,

                "isotropic_mean_ADE_px":
                    iso_ade,

                "mean_ADE_improvement_pct":
                    (
                        (
                            ani_ade
                            -
                            iso_ade
                        )
                        /
                        ani_ade
                        *
                        100.0
                    ),

                "anisotropic_mean_FDE_px":
                    ani_fde,

                "isotropic_mean_FDE_px":
                    iso_fde,

                "mean_FDE_improvement_pct":
                    (
                        (
                            ani_fde
                            -
                            iso_fde
                        )
                        /
                        ani_fde
                        *
                        100.0
                    ),

                "anisotropic_segment_macro_ADE_px":
                    float(
                        segment_macro[
                            "anisotropic_ADE_px"
                        ].mean()
                    ),

                "isotropic_segment_macro_ADE_px":
                    float(
                        segment_macro[
                            "isotropic_ADE_px"
                        ].mean()
                    ),

                "anisotropic_segment_macro_FDE_px":
                    float(
                        segment_macro[
                            "anisotropic_FDE_px"
                        ].mean()
                    ),

                "isotropic_segment_macro_FDE_px":
                    float(
                        segment_macro[
                            "isotropic_FDE_px"
                        ].mean()
                    ),

                "anisotropic_source_macro_ADE_px":
                    float(
                        source_macro[
                            "anisotropic_ADE_px"
                        ].mean()
                    ),

                "isotropic_source_macro_ADE_px":
                    float(
                        source_macro[
                            "isotropic_ADE_px"
                        ].mean()
                    ),

                "anisotropic_source_macro_FDE_px":
                    float(
                        source_macro[
                            "anisotropic_FDE_px"
                        ].mean()
                    ),

                "isotropic_source_macro_FDE_px":
                    float(
                        source_macro[
                            "isotropic_FDE_px"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_geometry_consistency(sample_df):
    rows = []

    for task, g in sample_df.groupby(
        "task",
        sort=True,
    ):
        rows.append(
            {
                "task":
                    task,

                "unique_sequences":
                    int(
                        g[
                            "sequence"
                        ].nunique()
                    ),

                "width_min_px":
                    float(
                        g[
                            "width_px"
                        ].min()
                    ),

                "width_max_px":
                    float(
                        g[
                            "width_px"
                        ].max()
                    ),

                "height_min_px":
                    float(
                        g[
                            "height_px"
                        ].min()
                    ),

                "height_max_px":
                    float(
                        g[
                            "height_px"
                        ].max()
                    ),

                "aspect_ratio_median":
                    float(
                        (
                            g[
                                "width_px"
                            ]
                            /
                            g[
                                "height_px"
                            ]
                        ).median()
                    ),

                "comparison_space":
                    "pixel-equivalent",

                "anisotropic_backtransform":
                    "x*W, y*H",

                "isotropic_backtransform":
                    "x*S, y*S where S=max(W,H)",
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-G — SAME-SAMPLE ANISOTROPIC vs ISOTROPIC BASELINE COMPARISON"
    )

    print(
        "NO training. Stationary and Constant-Velocity are evaluated on the exact "
        "same frozen TestReal samples. To avoid scale-confounded normalized-unit "
        "comparisons, predictions and truths are converted to a common pixel-equivalent "
        "geometry before ADE/FDE are computed.\n"
    )

    verify_inputs()

    geometry_map = (
        build_sequence_geometry_map()
    )

    dfs = []
    provenance = {}

    for task in TASKS:
        df, prov = evaluate_task(
            task,
            geometry_map,
        )

        dfs.append(
            df
        )

        provenance[
            task
        ] = prov

    sample_df = pd.concat(
        dfs,
        ignore_index=True,
    )

    summary = summarize(
        sample_df
    )

    geometry_audit = (
        build_geometry_consistency(
            sample_df
        )
    )

    sample_path = (
        OUT
        / "R9G_SAME_SAMPLE_PER_SAMPLE_METRICS.csv"
    )

    summary_path = (
        OUT
        / "R9G_SAME_SAMPLE_BASELINE_SUMMARY.csv"
    )

    geometry_path = (
        OUT
        / "R9G_COMMON_GEOMETRY_AUDIT.csv"
    )

    sample_df.to_csv(
        sample_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    geometry_audit.to_csv(
        geometry_path,
        index=False,
    )

    status = (
        "R9G_SAME_SAMPLE_ANISOTROPIC_VS_ISOTROPIC_BASELINE_COMPARISON_COMPLETE_READY_FOR_R9H_PAIRED_CLUSTERED_BOOTSTRAP"
    )

    manifest = {
        "step":
            "R9G_SAME_SAMPLE_ANISOTROPIC_VS_ISOTROPIC_BASELINE_COMPARISON",

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

        "models":
            MODELS,

        "comparison_space":
            "pixel-equivalent",

        "why_not_compare_raw_normalized_ADE_FDE":
            (
                "Anisotropic and isotropic normalized coordinates use different scales. "
                "Direct normalized-unit ADE/FDE differences would confound geometry with scale. "
                "Both branches are therefore back-transformed to the same pixel-equivalent space."
            ),

        "anisotropic_backtransform":
            "x_px = x_aniso * W; y_px = y_aniso * H",

        "isotropic_backtransform":
            "x_px = x_iso * S; y_px = y_iso * S; S=max(W,H)",

        "sample_metrics_csv":
            str(
                sample_path
            ),

        "sample_metrics_sha256":
            sha256_file(
                sample_path
            ),

        "summary_csv":
            str(
                summary_path
            ),

        "summary_sha256":
            sha256_file(
                summary_path
            ),

        "geometry_audit_csv":
            str(
                geometry_path
            ),

        "provenance":
            provenance,
    }

    (
        OUT
        / "R9G_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "COMMON GEOMETRY AUDIT:"
    )

    print(
        geometry_audit.to_string(
            index=False
        )
    )

    print(
        "\nSAME-SAMPLE BASELINE SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nPROVENANCE:"
    )

    print(
        json.dumps(
            provenance,
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
        "\nR9-G COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
