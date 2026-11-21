
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

CORR_DIR = MAIN1 / "corrected_trajectory_dataset"

R9D_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9D_isotropic_point_dataset_build"
)

R9G_V3A_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3A_corrected_builder_formula_lock"
)

R9G_V3A_MANIFEST = (
    R9G_V3A_DIR
    / "R9G_V3A_MANIFEST.json"
)

ISO_POINT_SOURCE = (
    R9D_DIR
    / "R9D_isotropic_cleaned_track_points.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3B_corrected_isotropic_rebuild"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "expected_samples": 1053,
        "future_10hz": 20,
    },
    "2s_to_5s": {
        "expected_samples": 826,
        "future_10hz": 50,
    },
}

DOWNSAMPLE = 3


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


def verify_v3a():
    require(R9G_V3A_MANIFEST)

    m = json.loads(
        R9G_V3A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        m.get("status")
        !=
        "R9G_V3A_CORRECTED_BUILDER_FORMULA_AUDIT_COMPLETE_READY_FOR_R9G_V3B_VALID_ISOTROPIC_REBUILD"
    ):
        raise RuntimeError(
            f"Unexpected R9G-V3A status: {m.get('status')}"
        )

    return m


def build_sequence_geometry():
    require(ISO_POINT_SOURCE)

    df = pd.read_csv(
        ISO_POINT_SOURCE,
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

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"R9D geometry source missing columns: {sorted(missing)}"
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
            "Sequence W/H/S are not stable."
        )

    geometry = {
        str(r["sequence"]): {
            "W": float(r["width_px"]),
            "H": float(r["height_px"]),
            "S": float(r["iso_scale_px"]),
        }
        for _, r in agg.iterrows()
    }

    return geometry, agg


def load_core(task):
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

    arrays = np.load(
        npz_path
    )

    hist_all = np.asarray(
        arrays["history"],
        dtype=np.float64,
    )

    fut_all = np.asarray(
        arrays["future"],
        dtype=np.float64,
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

    if len(core) != TASKS[task]["expected_samples"]:
        raise RuntimeError(
            f"{task}: core count {len(core)} != expected {TASKS[task]['expected_samples']}"
        )

    ids = pd.to_numeric(
        core["sample_id"],
        errors="raise",
    ).astype(int)

    if ids.duplicated().any():
        raise RuntimeError(
            f"{task}: duplicate sample IDs."
        )

    if (ids < 0).any() or (ids >= len(hist_all)).any():
        raise RuntimeError(
            f"{task}: invalid sample IDs for NPZ indexing."
        )

    hist_aniso = hist_all[
        ids.to_numpy(),
        :,
        :2,
    ][
        :,
        ::DOWNSAMPLE,
        :
    ]

    fut_aniso = fut_all[
        ids.to_numpy(),
        :,
        :2,
    ][
        :,
        ::DOWNSAMPLE,
        :
    ]

    expected_h = (
        TASKS[task]["expected_samples"],
        20,
        2,
    )

    expected_f = (
        TASKS[task]["expected_samples"],
        TASKS[task]["future_10hz"],
        2,
    )

    if hist_aniso.shape != expected_h:
        raise RuntimeError(
            f"{task}: anisotropic history shape {hist_aniso.shape} != {expected_h}"
        )

    if fut_aniso.shape != expected_f:
        raise RuntimeError(
            f"{task}: anisotropic future shape {fut_aniso.shape} != {expected_f}"
        )

    return (
        core,
        hist_aniso,
        fut_aniso,
        meta_path,
        npz_path,
    )


def rebuild_corrected_isotropic(
    core,
    hist_aniso,
    fut_aniso,
    geometry,
):
    hist_iso = np.empty_like(
        hist_aniso,
        dtype=np.float64,
    )

    fut_iso = np.empty_like(
        fut_aniso,
        dtype=np.float64,
    )

    audit_rows = []

    for i, row in core.iterrows():
        seq = str(
            row["sequence"]
        )

        if seq not in geometry:
            raise RuntimeError(
                f"Missing geometry for sequence {seq}"
            )

        W = geometry[seq]["W"]
        H = geometry[seq]["H"]
        S = geometry[seq]["S"]

        if S != max(W, H):
            raise RuntimeError(
                f"{seq}: locked S != max(W,H)"
            )

        # Frozen corrected anisotropic coordinates are:
        # x_aniso = x_corrected_px / W
        # y_aniso = y_corrected_px / H
        # Recover corrected pixel coordinates exactly.
        hist_px = np.empty_like(
            hist_aniso[i],
            dtype=np.float64,
        )

        fut_px = np.empty_like(
            fut_aniso[i],
            dtype=np.float64,
        )

        hist_px[:, 0] = (
            hist_aniso[i, :, 0]
            *
            W
        )

        hist_px[:, 1] = (
            hist_aniso[i, :, 1]
            *
            H
        )

        fut_px[:, 0] = (
            fut_aniso[i, :, 0]
            *
            W
        )

        fut_px[:, 1] = (
            fut_aniso[i, :, 1]
            *
            H
        )

        # One common denominator for both axes.
        hist_iso[i] = (
            hist_px
            /
            S
        )

        fut_iso[i] = (
            fut_px
            /
            S
        )

        # Exact round-trip back to the frozen anisotropic coordinates.
        hist_back = np.empty_like(
            hist_iso[i],
            dtype=np.float64,
        )

        fut_back = np.empty_like(
            fut_iso[i],
            dtype=np.float64,
        )

        hist_back[:, 0] = (
            hist_iso[i, :, 0]
            *
            S
            /
            W
        )

        hist_back[:, 1] = (
            hist_iso[i, :, 1]
            *
            S
            /
            H
        )

        fut_back[:, 0] = (
            fut_iso[i, :, 0]
            *
            S
            /
            W
        )

        fut_back[:, 1] = (
            fut_iso[i, :, 1]
            *
            S
            /
            H
        )

        hist_err = np.abs(
            hist_back
            -
            hist_aniso[i]
        )

        fut_err = np.abs(
            fut_back
            -
            fut_aniso[i]
        )

        audit_rows.append(
            {
                "sample_id":
                    int(row["sample_id"]),

                "sequence":
                    seq,

                "segment_id":
                    str(row["segment_id"]),

                "W":
                    W,

                "H":
                    H,

                "S":
                    S,

                "max_history_roundtrip_abs_error":
                    float(
                        hist_err.max()
                    ),

                "max_future_roundtrip_abs_error":
                    float(
                        fut_err.max()
                    ),
            }
        )

    return (
        hist_iso,
        fut_iso,
        pd.DataFrame(audit_rows),
    )


def main():
    print(
        "\nR9-G-V3B — VALID CAMERA-CORRECTED ISOTROPIC REBUILD"
    )

    print(
        "NO model training. NO ego-motion rerun. NO frozen overwrite. "
        "This step reconstructs corrected pixel coordinates directly from the frozen "
        "camera-corrected anisotropic NPZ (x*W, y*H), then applies the locked common "
        "scale S=max(W,H) to create a clean corrected-isotropic branch.\n"
    )

    verify_v3a()

    geometry, geometry_table = (
        build_sequence_geometry()
    )

    summary_rows = []
    provenance = {}

    all_roundtrip = []

    for task in TASKS:
        (
            core,
            hist_aniso,
            fut_aniso,
            meta_path,
            npz_path,
        ) = load_core(task)

        (
            hist_iso,
            fut_iso,
            roundtrip,
        ) = rebuild_corrected_isotropic(
            core,
            hist_aniso,
            fut_aniso,
            geometry,
        )

        roundtrip.insert(
            0,
            "task",
            task,
        )

        all_roundtrip.append(
            roundtrip
        )

        max_hist_err = float(
            roundtrip[
                "max_history_roundtrip_abs_error"
            ].max()
        )

        max_fut_err = float(
            roundtrip[
                "max_future_roundtrip_abs_error"
            ].max()
        )

        if max_hist_err > 1e-12:
            raise RuntimeError(
                f"{task}: history round-trip error too high: {max_hist_err}"
            )

        if max_fut_err > 1e-12:
            raise RuntimeError(
                f"{task}: future round-trip error too high: {max_fut_err}"
            )

        if not np.isfinite(hist_iso).all():
            raise RuntimeError(
                f"{task}: non-finite isotropic history."
            )

        if not np.isfinite(fut_iso).all():
            raise RuntimeError(
                f"{task}: non-finite isotropic future."
            )

        npz_out = (
            OUT
            / f"R9G_V3B_corrected_isotropic_{task}.npz"
        )

        meta_out = (
            OUT
            / f"R9G_V3B_corrected_isotropic_windows_{task}.csv"
        )

        np.savez_compressed(
            npz_out,
            history=hist_iso,
            future=fut_iso,
        )

        core.to_csv(
            meta_out,
            index=False,
        )

        summary_rows.append(
            {
                "task":
                    task,

                "samples":
                    int(
                        hist_iso.shape[0]
                    ),

                "history_shape":
                    str(
                        tuple(hist_iso.shape)
                    ),

                "future_shape":
                    str(
                        tuple(fut_iso.shape)
                    ),

                "history_all_finite":
                    bool(
                        np.isfinite(hist_iso).all()
                    ),

                "future_all_finite":
                    bool(
                        np.isfinite(fut_iso).all()
                    ),

                "max_history_roundtrip_abs_error":
                    max_hist_err,

                "max_future_roundtrip_abs_error":
                    max_fut_err,

                "history_min":
                    float(
                        hist_iso.min()
                    ),

                "history_max":
                    float(
                        hist_iso.max()
                    ),

                "future_min":
                    float(
                        fut_iso.min()
                    ),

                "future_max":
                    float(
                        fut_iso.max()
                    ),

                "output_npz":
                    str(npz_out),

                "output_npz_sha256":
                    sha256_file(npz_out),

                "output_metadata":
                    str(meta_out),

                "output_metadata_sha256":
                    sha256_file(meta_out),
            }
        )

        provenance[task] = {
            "frozen_corrected_metadata":
                str(meta_path),

            "frozen_corrected_metadata_sha256":
                sha256_file(meta_path),

            "frozen_corrected_npz":
                str(npz_path),

            "frozen_corrected_npz_sha256":
                sha256_file(npz_path),

            "formula":
                {
                    "corrected_pixel_x":
                        "x_corr_px = x_corr_aniso * W",

                    "corrected_pixel_y":
                        "y_corr_px = y_corr_aniso * H",

                    "common_scale":
                        "S = max(W,H)",

                    "corrected_isotropic_x":
                        "x_corr_iso = x_corr_px / S",

                    "corrected_isotropic_y":
                        "y_corr_iso = y_corr_px / S",
                },
        }

    summary = pd.DataFrame(
        summary_rows
    )

    roundtrip_all = pd.concat(
        all_roundtrip,
        ignore_index=True,
    )

    geometry_path = (
        OUT
        / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"
    )

    summary_path = (
        OUT
        / "R9G_V3B_REBUILD_SUMMARY.csv"
    )

    roundtrip_path = (
        OUT
        / "R9G_V3B_ROUNDTRIP_AUDIT.csv"
    )

    geometry_table.to_csv(
        geometry_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    roundtrip_all.to_csv(
        roundtrip_path,
        index=False,
    )

    status = (
        "R9G_V3B_VALID_CAMERA_CORRECTED_ISOTROPIC_BRANCH_REBUILT_READY_FOR_R9G_V3C_CLEAN_BASELINE_COMPARISON"
    )

    manifest = {
        "step":
            "R9G_V3B_VALID_CAMERA_CORRECTED_ISOTROPIC_REBUILD",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "new_training":
            False,

        "ego_motion_rerun":
            False,

        "frozen_results_modified":
            False,

        "source_of_corrected_geometry":
            "Frozen corrected NPZ first two channels",

        "scientific_control":
            (
                "Both anisotropic and isotropic branches now represent the same "
                "camera-corrected coordinates; only the normalization geometry differs."
            ),

        "sequence_geometry_csv":
            str(geometry_path),

        "summary_csv":
            str(summary_path),

        "roundtrip_audit_csv":
            str(roundtrip_path),

        "provenance":
            provenance,
    }

    (
        OUT
        / "R9G_V3B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "SEQUENCE GEOMETRY LOCK:"
    )

    print(
        geometry_table.to_string(
            index=False
        )
    )

    print(
        "\nCORRECTED-ISOTROPIC REBUILD SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nROUND-TRIP AUDIT SUMMARY:"
    )

    print(
        roundtrip_all.groupby(
            "task",
            sort=True,
        )[
            [
                "max_history_roundtrip_abs_error",
                "max_future_roundtrip_abs_error",
            ]
        ]
        .max()
        .reset_index()
        .to_string(
            index=False
        )
    )

    print(
        "\nPROVENANCE / FORMULA LOCK:"
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

    print(status)

    print(
        "\nR9-G-V3B COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
