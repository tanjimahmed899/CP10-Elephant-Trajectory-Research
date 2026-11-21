
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

CORR_DIR = MAIN1 / "corrected_trajectory_dataset"

V3B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3B_corrected_isotropic_rebuild"
)

V3C0_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3C0_exact_baseline_formula_lock"
)

V3B_MANIFEST = V3B_DIR / "R9G_V3B_MANIFEST.json"
V3C0_MANIFEST = V3C0_DIR / "R9G_V3C0_MANIFEST.json"

GEOMETRY_CSV = V3B_DIR / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3C_clean_baseline_comparison"
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

TARGET_FPS = 10.0
EQUIVALENCE_TOL_PX = 1e-9


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


def verify_manifests():
    for p in [
        V3B_MANIFEST,
        V3C0_MANIFEST,
        GEOMETRY_CSV,
    ]:
        require(p)

    v3b = json.loads(
        V3B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    v3c0 = json.loads(
        V3C0_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        v3b.get("status")
        !=
        "R9G_V3B_VALID_CAMERA_CORRECTED_ISOTROPIC_BRANCH_REBUILT_READY_FOR_R9G_V3C_CLEAN_BASELINE_COMPARISON"
    ):
        raise RuntimeError(
            f"Unexpected V3B status: {v3b.get('status')}"
        )

    if (
        v3c0.get("status")
        !=
        "R9G_V3C0_EXACT_STEP2B_BASELINE_FORMULA_LOCK_COMPLETE_READY_FOR_R9G_V3C_CLEAN_COMPARISON"
    ):
        raise RuntimeError(
            f"Unexpected V3C0 status: {v3c0.get('status')}"
        )

    return v3b, v3c0


def estimate_velocity_linear_regression(
    history,
    fps=10,
):
    history = np.asarray(
        history,
        dtype=np.float64,
    )

    T = len(
        history
    )

    times = np.arange(
        T,
        dtype=np.float64,
    ) / float(
        fps
    )

    times_centered = (
        times
        -
        times.mean()
    )

    denom = np.sum(
        times_centered ** 2
    )

    if denom <= 1e-12:
        return np.zeros(
            2,
            dtype=np.float64,
        )

    positions_centered = (
        history
        -
        history.mean(
            axis=0,
            keepdims=True,
        )
    )

    velocity = (
        times_centered[:, None]
        *
        positions_centered
    ).sum(
        axis=0
    ) / denom

    return velocity


def predict_constant_velocity(
    history,
    future_length,
    fps=10,
):
    velocity = (
        estimate_velocity_linear_regression(
            history,
            fps,
        )
    )

    future_times = (
        np.arange(
            1,
            future_length + 1,
            dtype=np.float64,
        )
        /
        float(
            fps
        )
    )

    prediction = (
        history[-1][None, :]
        +
        future_times[:, None]
        *
        velocity[None, :]
    )

    return prediction


def predict_stationary(
    history,
    future_length,
):
    return np.repeat(
        history[-1:, :],
        future_length,
        axis=0,
    )


def metrics(
    prediction,
    truth,
):
    errors = np.linalg.norm(
        np.asarray(
            prediction
        )
        -
        np.asarray(
            truth
        ),
        axis=1,
    )

    ade = float(
        errors.mean()
    )

    fde = float(
        errors[-1]
    )

    return ade, fde


def evaluate_representation(
    history,
    future,
    model_name,
):
    if model_name == "Stationary":
        prediction = (
            predict_stationary(
                history,
                len(
                    future
                ),
            )
        )

    elif model_name == "ConstantVelocity":
        prediction = (
            predict_constant_velocity(
                history,
                len(
                    future
                ),
                TARGET_FPS,
            )
        )

    else:
        raise ValueError(
            model_name
        )

    return prediction, metrics(
        prediction,
        future,
    )


def load_geometry():
    g = pd.read_csv(
        GEOMETRY_CSV,
        dtype={
            "sequence": str,
        },
    )

    required = {
        "sequence",
        "width_px",
        "height_px",
        "iso_scale_px",
    }

    missing = required - set(g.columns)

    if missing:
        raise RuntimeError(
            f"Geometry CSV missing: {sorted(missing)}"
        )

    return {
        str(r["sequence"]): {
            "W": float(r["width_px"]),
            "H": float(r["height_px"]),
            "S": float(r["iso_scale_px"]),
        }
        for _, r in g.iterrows()
    }


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

    meta = pd.read_csv(
        meta_path,
        dtype={
            "split": str,
            "sequence": str,
            "segment_id": str,
        },
    )

    arr = np.load(
        npz_path
    )

    core = meta[
        (
            pd.to_numeric(
                meta["accepted"],
                errors="coerce",
            )
            ==
            1
        )
        &
        (
            meta["quality_tier"]
            .astype(str)
            .str.lower()
            ==
            "core"
        )
        &
        (
            meta["split"].astype(str)
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
        core["sample_id"],
        errors="raise",
    ).astype(int)

    history = np.asarray(
        arr["history"][
            ids.to_numpy(),
            :,
            :2,
        ],
        dtype=np.float64,
    )[
        :,
        ::3,
        :
    ]

    future = np.asarray(
        arr["future"][
            ids.to_numpy(),
            :,
            :2,
        ],
        dtype=np.float64,
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
        V3B_DIR
        / f"R9G_V3B_corrected_isotropic_windows_{task}.csv"
    )

    npz_path = (
        V3B_DIR
        / f"R9G_V3B_corrected_isotropic_{task}.npz"
    )

    require(meta_path)
    require(npz_path)

    meta = pd.read_csv(
        meta_path,
        dtype={
            "sequence": str,
            "segment_id": str,
        },
    )

    arr = np.load(
        npz_path
    )

    history = np.asarray(
        arr["history"],
        dtype=np.float64,
    )

    future = np.asarray(
        arr["future"],
        dtype=np.float64,
    )

    return (
        meta,
        history,
        future,
        meta_path,
        npz_path,
    )


def backtransform_to_px(
    coords,
    representation,
    geom,
):
    c = np.asarray(
        coords,
        dtype=np.float64,
    ).copy()

    if representation == "anisotropic":
        c[..., 0] *= geom["W"]
        c[..., 1] *= geom["H"]
        return c

    if representation == "isotropic":
        c *= geom["S"]
        return c

    raise ValueError(
        representation
    )


def compare_task(
    task,
    geometry,
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

    n = TASKS[task]["expected_samples"]

    if len(ani_meta) != n or len(iso_meta) != n:
        raise RuntimeError(
            f"{task}: metadata count mismatch."
        )

    ids_ani = pd.to_numeric(
        ani_meta["sample_id"],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    ids_iso = pd.to_numeric(
        iso_meta["sample_id"],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    if not np.array_equal(
        ids_ani,
        ids_iso,
    ):
        raise RuntimeError(
            f"{task}: sample IDs/order mismatch."
        )

    if not np.array_equal(
        ani_meta["sequence"]
        .astype(str)
        .to_numpy(),
        iso_meta["sequence"]
        .astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            f"{task}: sequence order mismatch."
        )

    if not np.array_equal(
        ani_meta["segment_id"]
        .astype(str)
        .to_numpy(),
        iso_meta["segment_id"]
        .astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            f"{task}: segment order mismatch."
        )

    if ani_h.shape != iso_h.shape:
        raise RuntimeError(
            f"{task}: history shape mismatch {ani_h.shape} vs {iso_h.shape}"
        )

    if ani_f.shape != iso_f.shape:
        raise RuntimeError(
            f"{task}: future shape mismatch {ani_f.shape} vs {iso_f.shape}"
        )

    rows = []

    for i in range(
        n
    ):
        seq = str(
            ani_meta.iloc[i]["sequence"]
        )

        seg = str(
            ani_meta.iloc[i]["segment_id"]
        )

        sample_id = int(
            ani_meta.iloc[i]["sample_id"]
        )

        if seq not in geometry:
            raise RuntimeError(
                f"{task}: geometry missing for {seq}"
            )

        geom = geometry[
            seq
        ]

        for model in MODELS:
            pred_ani, (
                ade_ani_norm,
                fde_ani_norm,
            ) = evaluate_representation(
                ani_h[i],
                ani_f[i],
                model,
            )

            pred_iso, (
                ade_iso_norm,
                fde_iso_norm,
            ) = evaluate_representation(
                iso_h[i],
                iso_f[i],
                model,
            )

            pred_ani_px = backtransform_to_px(
                pred_ani,
                "anisotropic",
                geom,
            )

            truth_ani_px = backtransform_to_px(
                ani_f[i],
                "anisotropic",
                geom,
            )

            pred_iso_px = backtransform_to_px(
                pred_iso,
                "isotropic",
                geom,
            )

            truth_iso_px = backtransform_to_px(
                iso_f[i],
                "isotropic",
                geom,
            )

            ade_ani_px, fde_ani_px = metrics(
                pred_ani_px,
                truth_ani_px,
            )

            ade_iso_px, fde_iso_px = metrics(
                pred_iso_px,
                truth_iso_px,
            )

            ade_diff = (
                ade_iso_px
                -
                ade_ani_px
            )

            fde_diff = (
                fde_iso_px
                -
                fde_ani_px
            )

            max_pred_px_abs_diff = float(
                np.max(
                    np.abs(
                        pred_ani_px
                        -
                        pred_iso_px
                    )
                )
            )

            max_truth_px_abs_diff = float(
                np.max(
                    np.abs(
                        truth_ani_px
                        -
                        truth_iso_px
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
                        seq,

                    "segment_id":
                        seg,

                    "model":
                        model,

                    "anisotropic_ADE_normalized":
                        ade_ani_norm,

                    "isotropic_ADE_normalized":
                        ade_iso_norm,

                    "anisotropic_FDE_normalized":
                        fde_ani_norm,

                    "isotropic_FDE_normalized":
                        fde_iso_norm,

                    "anisotropic_ADE_px":
                        ade_ani_px,

                    "isotropic_ADE_px":
                        ade_iso_px,

                    "ADE_px_difference_iso_minus_aniso":
                        ade_diff,

                    "anisotropic_FDE_px":
                        fde_ani_px,

                    "isotropic_FDE_px":
                        fde_iso_px,

                    "FDE_px_difference_iso_minus_aniso":
                        fde_diff,

                    "abs_ADE_px_difference":
                        abs(
                            ade_diff
                        ),

                    "abs_FDE_px_difference":
                        abs(
                            fde_diff
                        ),

                    "max_prediction_px_coordinate_abs_difference":
                        max_pred_px_abs_diff,

                    "max_truth_px_coordinate_abs_difference":
                        max_truth_px_abs_diff,

                    "pixel_equivalent_within_tolerance":
                        bool(
                            max(
                                abs(
                                    ade_diff
                                ),
                                abs(
                                    fde_diff
                                ),
                                max_pred_px_abs_diff,
                                max_truth_px_abs_diff,
                            )
                            <=
                            EQUIVALENCE_TOL_PX
                        ),
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


def summarize(df):
    rows = []

    for (
        task,
        model,
    ), g in df.groupby(
        [
            "task",
            "model",
        ],
        sort=True,
    ):
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
                        g["sequence"]
                        .nunique()
                    ),

                "anisotropic_mean_ADE_normalized":
                    float(
                        g[
                            "anisotropic_ADE_normalized"
                        ].mean()
                    ),

                "isotropic_mean_ADE_normalized":
                    float(
                        g[
                            "isotropic_ADE_normalized"
                        ].mean()
                    ),

                "anisotropic_mean_FDE_normalized":
                    float(
                        g[
                            "anisotropic_FDE_normalized"
                        ].mean()
                    ),

                "isotropic_mean_FDE_normalized":
                    float(
                        g[
                            "isotropic_FDE_normalized"
                        ].mean()
                    ),

                "anisotropic_mean_ADE_px":
                    float(
                        g[
                            "anisotropic_ADE_px"
                        ].mean()
                    ),

                "isotropic_mean_ADE_px":
                    float(
                        g[
                            "isotropic_ADE_px"
                        ].mean()
                    ),

                "mean_ADE_px_difference_iso_minus_aniso":
                    float(
                        g[
                            "ADE_px_difference_iso_minus_aniso"
                        ].mean()
                    ),

                "max_abs_ADE_px_difference":
                    float(
                        g[
                            "abs_ADE_px_difference"
                        ].max()
                    ),

                "anisotropic_mean_FDE_px":
                    float(
                        g[
                            "anisotropic_FDE_px"
                        ].mean()
                    ),

                "isotropic_mean_FDE_px":
                    float(
                        g[
                            "isotropic_FDE_px"
                        ].mean()
                    ),

                "mean_FDE_px_difference_iso_minus_aniso":
                    float(
                        g[
                            "FDE_px_difference_iso_minus_aniso"
                        ].mean()
                    ),

                "max_abs_FDE_px_difference":
                    float(
                        g[
                            "abs_FDE_px_difference"
                        ].max()
                    ),

                "max_prediction_px_coordinate_abs_difference":
                    float(
                        g[
                            "max_prediction_px_coordinate_abs_difference"
                        ].max()
                    ),

                "max_truth_px_coordinate_abs_difference":
                    float(
                        g[
                            "max_truth_px_coordinate_abs_difference"
                        ].max()
                    ),

                "all_samples_pixel_equivalent_within_tolerance":
                    bool(
                        g[
                            "pixel_equivalent_within_tolerance"
                        ].all()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-G-V3C — CLEAN SAME-SAMPLE BASELINE REPRESENTATION SANITY CHECK"
    )

    print(
        "NO training. Exact Step2b Stationary and ConstantVelocity formulas are used. "
        "Frozen camera-corrected anisotropic and rebuilt camera-corrected isotropic "
        "representations are compared on identical samples. Primary comparison is in "
        "common pixel-equivalent geometry; normalized-unit metrics are descriptive only "
        "and are not interpreted across representations.\n"
    )

    verify_manifests()

    geometry = load_geometry()

    dfs = []
    provenance = {}

    for task in TASKS:
        df, prov = compare_task(
            task,
            geometry,
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

    if not summary[
        "all_samples_pixel_equivalent_within_tolerance"
    ].all():
        bad = sample_df[
            ~sample_df[
                "pixel_equivalent_within_tolerance"
            ]
        ]

        raise RuntimeError(
            "Clean representation sanity check FAILED: "
            f"{len(bad)} sample-model rows exceed "
            f"{EQUIVALENCE_TOL_PX} px tolerance."
        )

    sample_path = (
        OUT
        / "R9G_V3C_PER_SAMPLE_CLEAN_BASELINE_COMPARISON.csv"
    )

    summary_path = (
        OUT
        / "R9G_V3C_CLEAN_BASELINE_SUMMARY.csv"
    )

    sample_df.to_csv(
        sample_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    status = (
        "R9G_V3C_CLEAN_BASELINE_PIXEL_EQUIVALENCE_CONFIRMED_READY_FOR_R9H_CANONICALIZATION_GEOMETRY_AUDIT"
    )

    manifest = {
        "step":
            "R9G_V3C_CLEAN_BASELINE_REPRESENTATION_SANITY_CHECK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "bootstrap_performed":
            False,

        "frozen_results_modified":
            False,

        "models":
            MODELS,

        "primary_comparison_space":
            "pixel-equivalent",

        "normalized_metrics_interpretation":
            "descriptive_only_not_cross_representation_comparable",

        "equivalence_tolerance_px":
            EQUIVALENCE_TOL_PX,

        "expected_result":
            (
                "Stationary and ConstantVelocity are affine-equivariant to the "
                "reversible anisotropic-to-isotropic axis scaling. Therefore their "
                "predictions should be pixel-equivalent when source trajectories and "
                "camera correction are identical."
            ),

        "why_no_bootstrap_if_equivalent":
            (
                "A paired bootstrap is not informative for a deterministic algebraic "
                "equivalence whose observed per-sample differences are only numerical "
                "roundoff."
            ),

        "per_sample_csv":
            str(
                sample_path
            ),

        "per_sample_sha256":
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

        "provenance":
            provenance,
    }

    (
        OUT
        / "R9G_V3C_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "CLEAN BASELINE SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nMAXIMUM EQUIVALENCE DEVIATIONS:"
    )

    cols = [
        "task",
        "model",
        "max_abs_ADE_px_difference",
        "max_abs_FDE_px_difference",
        "max_prediction_px_coordinate_abs_difference",
        "max_truth_px_coordinate_abs_difference",
        "all_samples_pixel_equivalent_within_tolerance",
    ]

    print(
        summary[
            cols
        ].to_string(
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
        "\nR9-G-V3C COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
