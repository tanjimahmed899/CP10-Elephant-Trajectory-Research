
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

FINAL_NEURAL_DIR = MAIN1 / "final_neural_trajectory_dataset"
CORR_DIR = MAIN1 / "corrected_trajectory_dataset"

R9I_B_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_B_exact_model_hyperparameter_and_split_lock"
    / "R9I_B_MANIFEST.json"
)

R9I_C0_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_C0_exact_scale_floor_population_audit"
    / "R9I_C0_MANIFEST.json"
)

FROZEN_FLOOR_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C0_V3_authoritative_scale_floor_lock"
    / "R9H_C0_V3_MANIFEST.json"
)

R9G_V3B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3B_corrected_isotropic_rebuild"
)

GEOMETRY_CSV = (
    R9G_V3B_DIR
    / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_C_V2_matched_isotropic_trainreal_dataset"
)

DATASET_OUT = OUT / "dataset"

OUT.mkdir(parents=True, exist_ok=True)
DATASET_OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "expected_rows": 531,
        "expected_train": 434,
        "expected_validation": 97,
        "future_len": 20,
    },
    "2s_to_5s": {
        "expected_rows": 382,
        "expected_train": 294,
        "expected_validation": 88,
        "future_len": 50,
    },
}

DOWNSAMPLE = 3
RECENT_HEADING_FRAMES = 10
SCALE_FLOOR_QUANTILE = 0.05
QUANTILE_METHOD = "linear"

ANISOTROPIC_FLOOR_REPRO_TOL = 1e-15
PIXEL_ROUNDTRIP_TOL = 1e-9


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


def verify_locks():
    for p in [
        R9I_B_MANIFEST,
        R9I_C0_MANIFEST,
        FROZEN_FLOOR_MANIFEST,
        GEOMETRY_CSV,
    ]:
        require(p)

    r9i_b = json.loads(
        R9I_B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    r9i_c0 = json.loads(
        R9I_C0_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    floor_m = json.loads(
        FROZEN_FLOOR_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_b = (
        "R9I_B_EXACT_MODEL_HYPERPARAMETER_SPLIT_AND_CHECKPOINT_LOCK_COMPLETE_READY_FOR_R9I_C_ISOTROPIC_TRAINREAL_DATASET_BUILD"
    )

    expected_c0 = (
        "R9I_C0_EXACT_SCALE_FLOOR_DERIVATION_POPULATION_RECOVERED_READY_FOR_R9I_C_V2_MATCHED_ISOTROPIC_DATASET_BUILD"
    )

    expected_floor = (
        "R9H_C0_V3_AUTHORITATIVE_TASK_SCALE_FLOORS_LOCKED_READY_FOR_R9H_C1_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
    )

    if r9i_b.get("status") != expected_b:
        raise RuntimeError(
            f"Unexpected R9-I-B status: {r9i_b.get('status')}"
        )

    if r9i_c0.get("status") != expected_c0:
        raise RuntimeError(
            f"Unexpected R9-I-C0 status: {r9i_c0.get('status')}"
        )

    if floor_m.get("status") != expected_floor:
        raise RuntimeError(
            f"Unexpected frozen-floor status: {floor_m.get('status')}"
        )

    frozen_floors = {
        str(k): float(v)
        for k, v
        in floor_m[
            "locked_scale_floors"
        ].items()
    }

    return (
        r9i_b,
        r9i_c0,
        floor_m,
        frozen_floors,
    )


def load_geometry():
    df = pd.read_csv(
        GEOMETRY_CSV,
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

    geometry = {}

    for _, row in df.iterrows():
        sequence = str(
            row[
                "sequence"
            ]
        )

        W = float(
            row[
                "width_px"
            ]
        )

        H = float(
            row[
                "height_px"
            ]
        )

        S = float(
            row[
                "iso_scale_px"
            ]
        )

        if not np.isclose(
            S,
            max(W, H),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{sequence}: S != max(W,H)."
            )

        geometry[
            sequence
        ] = {
            "W": W,
            "H": H,
            "S": S,
        }

    return geometry


def history_path_length(history):
    history = np.asarray(
        history,
        dtype=np.float64
    )

    return float(
        np.linalg.norm(
            np.diff(
                history,
                axis=0
            ),
            axis=1
        ).sum()
    )


def estimate_heading(history):
    history = np.asarray(
        history,
        dtype=np.float64
    )

    recent = history[
        -RECENT_HEADING_FRAMES:
    ]

    t = np.arange(
        len(recent),
        dtype=np.float64
    )

    tc = t - t.mean()

    denom = float(
        np.sum(
            tc ** 2
        )
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

    magnitude = float(
        np.linalg.norm(
            velocity
        )
    )

    if magnitude <= 1e-10:
        return 0.0

    return float(
        np.arctan2(
            velocity[
                1
            ],
            velocity[
                0
            ]
        )
    )


def rotation_to_heading(angle):
    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )

    return np.array(
        [
            [c, s],
            [-s, c],
        ],
        dtype=np.float64
    )


def canonicalize_sample(
    history,
    future,
    scale_floor,
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
        float(
            scale_floor
        )
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
            h_can.astype(
                np.float32
            ),

        "future_canonical":
            f_can.astype(
                np.float32
            ),

        "history_abs":
            history.astype(
                np.float32
            ),

        "future_abs":
            future.astype(
                np.float32
            ),

        "origin":
            origin.astype(
                np.float32
            ),

        "heading_angle":
            np.float32(
                angle
            ),

        "scale":
            np.float32(
                scale
            ),

        "raw_path_length":
            np.float32(
                path_length
            ),
    }


def transform_dataset(
    history,
    future,
    scale_floor,
):
    hc = []
    fc = []
    ha = []
    fa = []
    origins = []
    angles = []
    scales = []
    raw_paths = []

    for i in range(
        len(
            history
        )
    ):
        result = canonicalize_sample(
            history[
                i
            ],
            future[
                i
            ],
            scale_floor
        )

        hc.append(
            result[
                "history_canonical"
            ]
        )

        fc.append(
            result[
                "future_canonical"
            ]
        )

        ha.append(
            result[
                "history_abs"
            ]
        )

        fa.append(
            result[
                "future_abs"
            ]
        )

        origins.append(
            result[
                "origin"
            ]
        )

        angles.append(
            result[
                "heading_angle"
            ]
        )

        scales.append(
            result[
                "scale"
            ]
        )

        raw_paths.append(
            result[
                "raw_path_length"
            ]
        )

    return {
        "history_canonical":
            np.stack(
                hc
            ),

        "future_canonical":
            np.stack(
                fc
            ),

        "history_abs":
            np.stack(
                ha
            ),

        "future_abs":
            np.stack(
                fa
            ),

        "origin":
            np.stack(
                origins
            ),

        "heading_angle":
            np.asarray(
                angles,
                dtype=np.float32
            ),

        "scale":
            np.asarray(
                scales,
                dtype=np.float32
            ),

        "raw_path_length":
            np.asarray(
                raw_paths,
                dtype=np.float32
            ),
    }


def load_original_final(task):
    meta_path = (
        FINAL_NEURAL_DIR
        /
        f"{task}_birdsai_trainval_metadata.csv"
    )

    npz_path = (
        FINAL_NEURAL_DIR
        /
        f"{task}_birdsai_trainval_canonical.npz"
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
            "sample_id": str,
            "sequence": str,
            "source_group": str,
            "segment_id": str,
            "track_key": str,
            "neural_split": str,
        }
    )

    arrays = np.load(
        npz_path,
        allow_pickle=False
    )

    return (
        meta,
        arrays,
        meta_path,
        npz_path,
    )


def load_corrected_trainreal(task):
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
            "sample_id": str,
            "sequence": str,
            "segment_id": str,
            "split": str,
            "quality_tier": str,
        }
    )

    arrays = np.load(
        npz_path,
        allow_pickle=False
    )

    core = meta[
        (
            pd.to_numeric(
                meta[
                    "accepted"
                ],
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
            "train_real"
        )
    ].copy()

    if core[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{task}: duplicate TrainReal sample_id."
        )

    return (
        core,
        arrays,
        meta_path,
        npz_path,
    )


def map_final_to_corrected(
    task,
    final_meta,
    corrected_core,
):
    lookup = (
        corrected_core
        .set_index(
            corrected_core[
                "sample_id"
            ].astype(str)
        )
    )

    rows = []

    for _, row in final_meta.iterrows():
        sid = str(
            row[
                "sample_id"
            ]
        )

        if sid not in lookup.index:
            raise RuntimeError(
                f"{task}: final sample_id={sid} absent from corrected TrainReal."
            )

        corr = lookup.loc[
            sid
        ]

        if isinstance(
            corr,
            pd.DataFrame
        ):
            raise RuntimeError(
                f"{task}: sample_id={sid} maps to multiple corrected rows."
            )

        for col in [
            "sequence",
            "segment_id",
            "obs_start_frame",
            "obs_end_frame",
            "future_end_frame",
        ]:
            if (
                col in final_meta.columns
                and
                col in corrected_core.columns
            ):
                a = str(
                    row[
                        col
                    ]
                )

                b = str(
                    corr[
                        col
                    ]
                )

                if a != b:
                    raise RuntimeError(
                        f"{task}: identity mismatch sample_id={sid}, "
                        f"{col}: final={a}, corrected={b}"
                    )

        rows.append(
            corr
        )

    mapped = pd.DataFrame(
        rows
    ).reset_index(
        drop=True
    )

    return mapped


def extract_anisotropic_windows(
    task,
    mapped,
    arrays,
):
    histories = []
    futures = []

    for _, row in mapped.iterrows():
        sid = int(
            row[
                "sample_id"
            ]
        )

        h = np.asarray(
            arrays[
                "history"
            ][
                sid,
                :,
                :2
            ],
            dtype=np.float64
        )[
            ::DOWNSAMPLE
        ]

        f = np.asarray(
            arrays[
                "future"
            ][
                sid,
                :,
                :2
            ],
            dtype=np.float64
        )[
            ::DOWNSAMPLE
        ]

        histories.append(
            h
        )

        futures.append(
            f
        )

    history = np.stack(
        histories,
        axis=0
    )

    future = np.stack(
        futures,
        axis=0
    )

    expected_future = TASKS[
        task
    ][
        "future_len"
    ]

    if history.shape != (
        len(
            mapped
        ),
        20,
        2,
    ):
        raise RuntimeError(
            f"{task}: history shape={history.shape}"
        )

    if future.shape != (
        len(
            mapped
        ),
        expected_future,
        2,
    ):
        raise RuntimeError(
            f"{task}: future shape={future.shape}"
        )

    return (
        history,
        future,
    )


def to_isotropic(
    task,
    mapped,
    history_aniso,
    future_aniso,
    geometry,
):
    history_iso = np.empty_like(
        history_aniso,
        dtype=np.float64
    )

    future_iso = np.empty_like(
        future_aniso,
        dtype=np.float64
    )

    max_roundtrip = 0.0

    for i, row in mapped.iterrows():
        sequence = str(
            row[
                "sequence"
            ]
        )

        if sequence not in geometry:
            raise RuntimeError(
                f"{task}: missing geometry for {sequence}."
            )

        W = geometry[
            sequence
        ][
            "W"
        ]

        H = geometry[
            sequence
        ][
            "H"
        ]

        S = geometry[
            sequence
        ][
            "S"
        ]

        h_px = history_aniso[
            i
        ].copy()

        h_px[
            :,
            0
        ] *= W

        h_px[
            :,
            1
        ] *= H

        f_px = future_aniso[
            i
        ].copy()

        f_px[
            :,
            0
        ] *= W

        f_px[
            :,
            1
        ] *= H

        history_iso[
            i
        ] = (
            h_px
            /
            S
        )

        future_iso[
            i
        ] = (
            f_px
            /
            S
        )

        h_rt = (
            history_iso[
                i
            ]
            *
            S
        )

        f_rt = (
            future_iso[
                i
            ]
            *
            S
        )

        max_roundtrip = max(
            max_roundtrip,
            float(
                np.max(
                    np.abs(
                        h_rt
                        -
                        h_px
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        f_rt
                        -
                        f_px
                    )
                )
            ),
        )

    if max_roundtrip > PIXEL_ROUNDTRIP_TOL:
        raise RuntimeError(
            f"{task}: pixel roundtrip max error={max_roundtrip}"
        )

    return (
        history_iso,
        future_iso,
        max_roundtrip,
    )


def derive_floors(
    task,
    history_aniso,
    history_iso,
    frozen_floor,
):
    aniso_lengths = np.asarray(
        [
            history_path_length(
                h
            )
            for h in history_aniso
        ],
        dtype=np.float64
    )

    iso_lengths = np.asarray(
        [
            history_path_length(
                h
            )
            for h in history_iso
        ],
        dtype=np.float64
    )

    reproduced = float(
        np.quantile(
            aniso_lengths,
            SCALE_FLOOR_QUANTILE,
            method=QUANTILE_METHOD
        )
    )

    if not np.isclose(
        reproduced,
        frozen_floor,
        rtol=0.0,
        atol=ANISOTROPIC_FLOOR_REPRO_TOL,
    ):
        raise RuntimeError(
            f"{task}: exact original floor reproduction failed. "
            f"reproduced={reproduced}, frozen={frozen_floor}"
        )

    isotropic_floor = float(
        np.quantile(
            iso_lengths,
            SCALE_FLOOR_QUANTILE,
            method=QUANTILE_METHOD
        )
    )

    return (
        aniso_lengths,
        iso_lengths,
        reproduced,
        isotropic_floor,
    )


def validate_split_and_weights(
    task,
    final_meta,
    old_arrays,
):
    cfg = TASKS[
        task
    ]

    if len(
        final_meta
    ) != cfg[
        "expected_rows"
    ]:
        raise RuntimeError(
            f"{task}: expected rows={cfg['expected_rows']}, "
            f"got={len(final_meta)}"
        )

    train_mask = (
        final_meta[
            "neural_split"
        ]
        .astype(str)
        .to_numpy()
        ==
        "train"
    )

    val_mask = (
        final_meta[
            "neural_split"
        ]
        .astype(str)
        .to_numpy()
        ==
        "validation"
    )

    if int(
        train_mask.sum()
    ) != cfg[
        "expected_train"
    ]:
        raise RuntimeError(
            f"{task}: train count mismatch."
        )

    if int(
        val_mask.sum()
    ) != cfg[
        "expected_validation"
    ]:
        raise RuntimeError(
            f"{task}: validation count mismatch."
        )

    required = {
        "train_indices",
        "validation_indices",
        "train_sample_weights",
    }

    missing = required - set(
        old_arrays.files
    )

    if missing:
        raise RuntimeError(
            f"{task}: old NPZ missing critical keys: {sorted(missing)}"
        )

    train_idx = np.asarray(
        old_arrays[
            "train_indices"
        ],
        dtype=np.int64
    )

    val_idx = np.asarray(
        old_arrays[
            "validation_indices"
        ],
        dtype=np.int64
    )

    expected_train_idx = np.flatnonzero(
        train_mask
    )

    expected_val_idx = np.flatnonzero(
        val_mask
    )

    if not np.array_equal(
        np.sort(
            train_idx
        ),
        expected_train_idx,
    ):
        raise RuntimeError(
            f"{task}: train_indices do not match metadata split."
        )

    if not np.array_equal(
        np.sort(
            val_idx
        ),
        expected_val_idx,
    ):
        raise RuntimeError(
            f"{task}: validation_indices do not match metadata split."
        )

    return (
        train_idx,
        val_idx,
        np.asarray(
            old_arrays[
                "train_sample_weights"
            ]
        ),
    )


def build_output_payload(
    task,
    old_arrays,
    transformed,
):
    expected_geometry_keys = {
        "history_canonical",
        "future_canonical",
        "history_abs",
        "future_abs",
        "origin",
        "heading_angle",
        "scale",
        "raw_path_length",
    }

    old_keys = set(
        old_arrays.files
    )

    missing = expected_geometry_keys - old_keys

    if missing:
        raise RuntimeError(
            f"{task}: original final NPZ missing geometry keys: {sorted(missing)}"
        )

    payload = {}

    for key in old_arrays.files:
        if key in transformed:
            old = np.asarray(
                old_arrays[
                    key
                ]
            )

            new = np.asarray(
                transformed[
                    key
                ]
            )

            if old.shape != new.shape:
                raise RuntimeError(
                    f"{task}: shape mismatch for {key}: "
                    f"old={old.shape}, new={new.shape}"
                )

            payload[
                key
            ] = new.astype(
                old.dtype,
                copy=False
            )

        else:
            payload[
                key
            ] = np.asarray(
                old_arrays[
                    key
                ]
            )

    return payload


def update_metadata(
    final_meta,
    iso_lengths,
    transformed,
    isotropic_floor,
):
    out = final_meta.copy()

    if "scale_floor" in out.columns:
        out[
            "scale_floor"
        ] = isotropic_floor

    out[
        "r9_coordinate_representation"
    ] = "isotropic_x_over_S_y_over_S"

    out[
        "r9_isotropic_scale_floor"
    ] = isotropic_floor

    out[
        "r9_isotropic_raw_path_length"
    ] = iso_lengths

    out[
        "r9_isotropic_canonical_scale"
    ] = transformed[
        "scale"
    ]

    out[
        "r9_isotropic_heading_angle_rad"
    ] = transformed[
        "heading_angle"
    ]

    out[
        "r9_isotropic_origin_x"
    ] = transformed[
        "origin"
    ][
        :,
        0
    ]

    out[
        "r9_isotropic_origin_y"
    ] = transformed[
        "origin"
    ][
        :,
        1
    ]

    return out


def process_task(
    task,
    geometry,
    frozen_floor,
):
    (
        final_meta,
        old_arrays,
        final_meta_path,
        old_npz_path,
    ) = load_original_final(
        task
    )

    (
        corrected_core,
        corrected_arrays,
        corrected_meta_path,
        corrected_npz_path,
    ) = load_corrected_trainreal(
        task
    )

    if len(
        corrected_core
    ) != TASKS[
        task
    ][
        "expected_rows"
    ]:
        raise RuntimeError(
            f"{task}: corrected TrainReal pool rows={len(corrected_core)}, "
            f"expected={TASKS[task]['expected_rows']}"
        )

    mapped = map_final_to_corrected(
        task,
        final_meta,
        corrected_core
    )

    (
        history_aniso,
        future_aniso,
    ) = extract_anisotropic_windows(
        task,
        mapped,
        corrected_arrays
    )

    (
        history_iso,
        future_iso,
        pixel_roundtrip,
    ) = to_isotropic(
        task,
        mapped,
        history_aniso,
        future_aniso,
        geometry
    )

    (
        aniso_lengths,
        iso_lengths,
        reproduced_aniso_floor,
        isotropic_floor,
    ) = derive_floors(
        task,
        history_aniso,
        history_iso,
        frozen_floor
    )

    (
        train_idx,
        val_idx,
        old_train_weights,
    ) = validate_split_and_weights(
        task,
        final_meta,
        old_arrays
    )

    transformed = transform_dataset(
        history_iso,
        future_iso,
        isotropic_floor
    )

    payload = build_output_payload(
        task,
        old_arrays,
        transformed
    )

    out_meta = update_metadata(
        final_meta,
        iso_lengths,
        transformed,
        isotropic_floor
    )

    out_npz = (
        DATASET_OUT
        /
        f"{task}_birdsai_trainval_canonical.npz"
    )

    out_meta_path = (
        DATASET_OUT
        /
        f"{task}_birdsai_trainval_metadata.csv"
    )

    np.savez_compressed(
        out_npz,
        **payload
    )

    out_meta.to_csv(
        out_meta_path,
        index=False
    )

    # Re-open and verify critical split/weight arrays stayed identical.
    new_arrays = np.load(
        out_npz,
        allow_pickle=False
    )

    for key in [
        "train_indices",
        "validation_indices",
        "train_sample_weights",
    ]:
        if not np.array_equal(
            np.asarray(
                new_arrays[
                    key
                ]
            ),
            np.asarray(
                old_arrays[
                    key
                ]
            ),
        ):
            raise RuntimeError(
                f"{task}: critical array {key} changed."
            )

    # Ensure canonical arrays are finite.
    for key in [
        "history_canonical",
        "future_canonical",
        "history_abs",
        "future_abs",
        "origin",
        "heading_angle",
        "scale",
        "raw_path_length",
    ]:
        arr = np.asarray(
            new_arrays[
                key
            ]
        )

        if not np.isfinite(
            arr
        ).all():
            raise RuntimeError(
                f"{task}: non-finite values in {key}."
            )

    summary = {
        "task":
            task,

        "all_trainreal_rows_used_for_floor":
            int(
                len(
                    history_iso
                )
            ),

        "neural_train_rows":
            int(
                len(
                    train_idx
                )
            ),

        "neural_validation_rows":
            int(
                len(
                    val_idx
                )
            ),

        "frozen_anisotropic_floor":
            float(
                frozen_floor
            ),

        "reproduced_anisotropic_floor":
            float(
                reproduced_aniso_floor
            ),

        "new_isotropic_all_trainreal_floor":
            float(
                isotropic_floor
            ),

        "isotropic_over_anisotropic_floor_ratio":
            float(
                isotropic_floor
                /
                frozen_floor
            ),

        "anisotropic_path_q05":
            float(
                np.quantile(
                    aniso_lengths,
                    SCALE_FLOOR_QUANTILE,
                    method=QUANTILE_METHOD
                )
            ),

        "isotropic_path_q05":
            float(
                np.quantile(
                    iso_lengths,
                    SCALE_FLOOR_QUANTILE,
                    method=QUANTILE_METHOD
                )
            ),

        "anisotropic_path_median":
            float(
                np.median(
                    aniso_lengths
                )
            ),

        "isotropic_path_median":
            float(
                np.median(
                    iso_lengths
                )
            ),

        "pixel_roundtrip_max_abs_error":
            float(
                pixel_roundtrip
            ),

        "same_train_indices":
            True,

        "same_validation_indices":
            True,

        "same_train_sample_weights":
            True,

        "output_npz":
            str(
                out_npz
            ),

        "output_npz_sha256":
            sha256_file(
                out_npz
            ),

        "output_metadata":
            str(
                out_meta_path
            ),

        "output_metadata_sha256":
            sha256_file(
                out_meta_path
            ),
    }

    provenance = {
        "task":
            task,

        "original_final_metadata":
            str(
                final_meta_path
            ),

        "original_final_metadata_sha256":
            sha256_file(
                final_meta_path
            ),

        "original_final_npz":
            str(
                old_npz_path
            ),

        "original_final_npz_sha256":
            sha256_file(
                old_npz_path
            ),

        "corrected_trainreal_metadata":
            str(
                corrected_meta_path
            ),

        "corrected_trainreal_metadata_sha256":
            sha256_file(
                corrected_meta_path
            ),

        "corrected_trainreal_npz":
            str(
                corrected_npz_path
            ),

        "corrected_trainreal_npz_sha256":
            sha256_file(
                corrected_npz_path
            ),
    }

    return (
        summary,
        provenance,
    )


def main():
    print(
        "\nR9-I-C-V2 — MATCHED ISOTROPIC TRAINREAL CANONICAL DATASET BUILD"
    )

    print(
        "NO neural training and NO model inference. "
        "This version uses the EXACT original scale-floor derivation protocol recovered "
        "in R9-I-C0: the 5th percentile of path lengths across the ENTIRE BIRDSAI "
        "TrainReal pool BEFORE the track-disjoint neural train/validation split. "
        "The anisotropic floor must reproduce exactly. Then the same rule is applied "
        "to isotropic coordinates. Existing train_indices, validation_indices and "
        "train_sample_weights are copied unchanged. TestReal and WildBox are not used.\n"
    )

    (
        r9i_b,
        r9i_c0,
        floor_m,
        frozen_floors,
    ) = verify_locks()

    geometry = load_geometry()

    summaries = []
    provenance = []

    for task in TASKS:
        summary, prov = process_task(
            task,
            geometry,
            frozen_floors[
                task
            ],
        )

        summaries.append(
            summary
        )

        provenance.append(
            prov
        )

    summary_df = pd.DataFrame(
        summaries
    )

    provenance_df = pd.DataFrame(
        provenance
    )

    summary_path = (
        OUT
        /
        "R9I_C_V2_TASK_SUMMARY.csv"
    )

    provenance_path = (
        OUT
        /
        "R9I_C_V2_PROVENANCE.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    provenance_df.to_csv(
        provenance_path,
        index=False
    )

    status = (
        "R9I_C_V2_MATCHED_ISOTROPIC_TRAINREAL_CANONICAL_DATASETS_BUILT_READY_FOR_R9I_D_MATCHED_NEURAL_RETRAINING_SCRIPT"
    )

    manifest = {
        "step":
            "R9I_C_V2_MATCHED_ISOTROPIC_TRAINREAL_CANONICAL_DATASET_BUILD",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "coordinate_representation":
            "x/S,y/S where S=max(width_px,height_px)",

        "scale_floor_protocol":
            {
                "source_population":
                    "all BIRDSAI TrainReal rows before neural train/validation split",

                "quantile":
                    SCALE_FLOOR_QUANTILE,

                "quantile_method":
                    QUANTILE_METHOD,

                "matches_original_pipeline":
                    True,

                "testreal_used":
                    False,

                "wildbox_used":
                    False,
            },

        "matched_ablation_locks":
            {
                "same_final_trainreal_rows":
                    True,

                "same_train_indices":
                    True,

                "same_validation_indices":
                    True,

                "same_train_sample_weights":
                    True,

                "same_neural_split_membership":
                    True,
            },

        "important_method_note":
            (
                "The original pipeline derives scale_floor from the full BIRDSAI "
                "TrainReal pool before creating the neural track-disjoint train/validation "
                "split. R9-I-C-V2 intentionally reproduces that same preprocessing "
                "protocol for a matched representation ablation. TestReal and WildBox "
                "remain excluded."
            ),

        "new_training":
            False,

        "model_inference":
            False,

        "frozen_results_modified":
            False,

        "dataset_directory":
            str(
                DATASET_OUT
            ),

        "task_summary_csv":
            str(
                summary_path
            ),

        "task_summary_sha256":
            sha256_file(
                summary_path
            ),

        "provenance_csv":
            str(
                provenance_path
            ),

        "provenance_sha256":
            sha256_file(
                provenance_path
            ),

        "tasks":
            summary_df.to_dict(
                orient="records"
            ),
    }

    (
        OUT
        /
        "R9I_C_V2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "MATCHED ISOTROPIC TRAINREAL DATASET SUMMARY:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nPROVENANCE:"
    )

    print(
        provenance_df.to_string(
            index=False
        )
    )

    print(
        "\nCRITICAL LOCKS:"
    )

    for _, row in summary_df.iterrows():
        print(
            f"  {row['task']}: "
            f"all TrainReal rows for floor={int(row['all_trainreal_rows_used_for_floor'])}; "
            f"anisotropic frozen={row['frozen_anisotropic_floor']:.17g}; "
            f"reproduced={row['reproduced_anisotropic_floor']:.17g}; "
            f"new isotropic floor={row['new_isotropic_all_trainreal_floor']:.17g}; "
            f"train/val={int(row['neural_train_rows'])}/{int(row['neural_validation_rows'])}"
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-I-C-V2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
