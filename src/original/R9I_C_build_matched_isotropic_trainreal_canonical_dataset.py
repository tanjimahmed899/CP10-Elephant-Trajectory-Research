
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

R9I_B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_B_exact_model_hyperparameter_and_split_lock"
)

R9I_B_MANIFEST = R9I_B_DIR / "R9I_B_MANIFEST.json"

R9H_C0_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C0_V3_authoritative_scale_floor_lock"
)

R9H_C0_MANIFEST = R9H_C0_DIR / "R9H_C0_V3_MANIFEST.json"

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
    / "R9I_C_isotropic_trainreal_canonical_dataset"
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

ANISOTROPIC_FLOOR_REPRO_TOL = 1e-10
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


def verify_protocol_locks():
    for p in [
        R9I_B_MANIFEST,
        R9H_C0_MANIFEST,
        GEOMETRY_CSV,
    ]:
        require(p)

    r9i_b = json.loads(
        R9I_B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_b = (
        "R9I_B_EXACT_MODEL_HYPERPARAMETER_SPLIT_AND_CHECKPOINT_LOCK_COMPLETE_READY_FOR_R9I_C_ISOTROPIC_TRAINREAL_DATASET_BUILD"
    )

    if r9i_b.get("status") != expected_b:
        raise RuntimeError(
            f"Unexpected R9-I-B status: {r9i_b.get('status')}"
        )

    floor_lock = json.loads(
        R9H_C0_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_floor = (
        "R9H_C0_V3_AUTHORITATIVE_TASK_SCALE_FLOORS_LOCKED_READY_FOR_R9H_C1_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
    )

    if floor_lock.get("status") != expected_floor:
        raise RuntimeError(
            f"Unexpected frozen floor-lock status: {floor_lock.get('status')}"
        )

    frozen_aniso_floors = {
        str(k): float(v)
        for k, v
        in floor_lock[
            "locked_scale_floors"
        ].items()
    }

    for task in TASKS:
        if task not in frozen_aniso_floors:
            raise RuntimeError(
                f"Frozen anisotropic scale floor missing for {task}."
            )

    return (
        r9i_b,
        floor_lock,
        frozen_aniso_floors,
    )


def load_geometry():
    require(GEOMETRY_CSV)

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

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Geometry lock missing columns: {sorted(missing)}"
        )

    geom = {}

    for _, r in df.iterrows():
        seq = str(r["sequence"])

        W = float(r["width_px"])
        H = float(r["height_px"])
        S = float(r["iso_scale_px"])

        if not np.isclose(
            S,
            max(W, H),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{seq}: iso_scale_px != max(W,H)."
            )

        geom[seq] = {
            "W": W,
            "H": H,
            "S": S,
        }

    return geom


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
            velocity[1],
            velocity[0]
        )
    )


def rotation_to_heading(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [c, s],
            [-s, c],
        ],
        dtype=np.float64,
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

    origin = history[-1].copy()

    angle = estimate_heading(
        history
    )

    R = rotation_to_heading(
        angle
    )

    raw_path_length = history_path_length(
        history
    )

    scale = max(
        raw_path_length,
        float(scale_floor)
    )

    history_canonical = (
        (history - origin)
        @
        R.T
    ) / scale

    future_canonical = (
        (future - origin)
        @
        R.T
    ) / scale

    return {
        "history_canonical":
            history_canonical,

        "future_canonical":
            future_canonical,

        "origin":
            origin,

        "angle":
            float(angle),

        "scale":
            float(scale),

        "raw_path_length":
            float(raw_path_length),
    }


def load_original_final_dataset(task):
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

    require(meta_path)
    require(npz_path)

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


def load_corrected_source(task):
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

    require(meta_path)
    require(npz_path)

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

    if "history" not in arrays.files:
        raise RuntimeError(
            f"{npz_path} missing history array."
        )

    if "future" not in arrays.files:
        raise RuntimeError(
            f"{npz_path} missing future array."
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
            "train_real"
        )
    ].copy()

    if core["sample_id"].duplicated().any():
        dups = (
            core.loc[
                core["sample_id"].duplicated(
                    keep=False
                ),
                "sample_id"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"{task}: corrected TrainReal sample_id duplicates: {dups[:10]}"
        )

    return (
        core,
        arrays,
        meta_path,
        npz_path,
    )


def map_final_rows_to_corrected(
    task,
    final_meta,
    corrected_core,
):
    if "sample_id" not in final_meta.columns:
        raise RuntimeError(
            f"{task}: final metadata missing sample_id."
        )

    if final_meta["sample_id"].duplicated().any():
        raise RuntimeError(
            f"{task}: final metadata sample_id is not unique."
        )

    lookup = (
        corrected_core
        .set_index(
            corrected_core[
                "sample_id"
            ].astype(str)
        )
    )

    mapped_rows = []

    for _, row in final_meta.iterrows():
        sid = str(
            row[
                "sample_id"
            ]
        )

        if sid not in lookup.index:
            raise RuntimeError(
                f"{task}: final sample_id {sid} not found in corrected TrainReal core."
            )

        corr_row = lookup.loc[
            sid
        ]

        if isinstance(
            corr_row,
            pd.DataFrame
        ):
            raise RuntimeError(
                f"{task}: sample_id {sid} maps to multiple corrected rows."
            )

        # Validate common identity columns when available.
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
                    corr_row[
                        col
                    ]
                )

                if a != b:
                    raise RuntimeError(
                        f"{task}: identity mismatch for sample_id={sid}, "
                        f"column={col}, final={a}, corrected={b}"
                    )

        mapped_rows.append(
            corr_row
        )

    mapped = pd.DataFrame(
        mapped_rows
    ).reset_index(
        drop=True
    )

    return mapped


def extract_corrected_windows(
    task,
    mapped,
    corrected_arrays,
):
    histories = []
    futures = []

    for _, row in mapped.iterrows():
        sid = int(
            row[
                "sample_id"
            ]
        )

        if not (
            0
            <=
            sid
            <
            len(
                corrected_arrays[
                    "history"
                ]
            )
        ):
            raise RuntimeError(
                f"{task}: sample_id {sid} outside corrected NPZ history range."
            )

        if not (
            0
            <=
            sid
            <
            len(
                corrected_arrays[
                    "future"
                ]
            )
        ):
            raise RuntimeError(
                f"{task}: sample_id {sid} outside corrected NPZ future range."
            )

        h = np.asarray(
            corrected_arrays[
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
            corrected_arrays[
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

        histories.append(h)
        futures.append(f)

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
        len(mapped),
        20,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected history shape {history.shape}."
        )

    if future.shape != (
        len(mapped),
        expected_future,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected future shape {future.shape}."
        )

    return (
        history,
        future,
    )


def corrected_aniso_to_iso(
    task,
    mapped,
    history_aniso,
    future_aniso,
    geometry,
):
    h_iso = np.empty_like(
        history_aniso,
        dtype=np.float64
    )

    f_iso = np.empty_like(
        future_aniso,
        dtype=np.float64
    )

    pixel_roundtrip_max = 0.0

    for i, row in mapped.iterrows():
        sequence = str(
            row[
                "sequence"
            ]
        )

        if sequence not in geometry:
            raise RuntimeError(
                f"{task}: missing sequence geometry for {sequence}."
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

        h_iso[
            i
        ] = (
            h_px
            /
            S
        )

        f_iso[
            i
        ] = (
            f_px
            /
            S
        )

        h_px_rt = (
            h_iso[
                i
            ]
            *
            S
        )

        f_px_rt = (
            f_iso[
                i
            ]
            *
            S
        )

        pixel_roundtrip_max = max(
            pixel_roundtrip_max,
            float(
                np.max(
                    np.abs(
                        h_px_rt
                        -
                        h_px
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        f_px_rt
                        -
                        f_px
                    )
                )
            ),
        )

    if pixel_roundtrip_max > PIXEL_ROUNDTRIP_TOL:
        raise RuntimeError(
            f"{task}: pixel roundtrip failed, max={pixel_roundtrip_max}"
        )

    return (
        h_iso,
        f_iso,
        pixel_roundtrip_max,
    )


def derive_floor_and_verify_rule(
    task,
    final_meta,
    history_aniso,
    history_iso,
    frozen_aniso_floor,
):
    split = (
        final_meta[
            "neural_split"
        ]
        .astype(str)
        .to_numpy()
    )

    train_mask = (
        split
        ==
        "train"
    )

    validation_mask = (
        split
        ==
        "validation"
    )

    expected_train = TASKS[
        task
    ][
        "expected_train"
    ]

    expected_validation = TASKS[
        task
    ][
        "expected_validation"
    ]

    if int(
        train_mask.sum()
    ) != expected_train:
        raise RuntimeError(
            f"{task}: train count mismatch."
        )

    if int(
        validation_mask.sum()
    ) != expected_validation:
        raise RuntimeError(
            f"{task}: validation count mismatch."
        )

    aniso_lengths = np.array(
        [
            history_path_length(
                h
            )
            for h in history_aniso
        ],
        dtype=np.float64
    )

    iso_lengths = np.array(
        [
            history_path_length(
                h
            )
            for h in history_iso
        ],
        dtype=np.float64
    )

    reproduced_aniso_floor = float(
        np.quantile(
            aniso_lengths[
                train_mask
            ],
            SCALE_FLOOR_QUANTILE
        )
    )

    if not np.isclose(
        reproduced_aniso_floor,
        frozen_aniso_floor,
        rtol=0.0,
        atol=ANISOTROPIC_FLOOR_REPRO_TOL,
    ):
        raise RuntimeError(
            f"{task}: scale-floor rule reproduction failed. "
            f"Derived anisotropic train floor={reproduced_aniso_floor}, "
            f"frozen={frozen_aniso_floor}."
        )

    isotropic_floor = float(
        np.quantile(
            iso_lengths[
                train_mask
            ],
            SCALE_FLOOR_QUANTILE
        )
    )

    return {
        "train_mask":
            train_mask,

        "validation_mask":
            validation_mask,

        "anisotropic_path_lengths":
            aniso_lengths,

        "isotropic_path_lengths":
            iso_lengths,

        "reproduced_anisotropic_floor":
            reproduced_aniso_floor,

        "isotropic_floor":
            isotropic_floor,
    }


def canonicalize_all(
    history_iso,
    future_iso,
    scale_floor,
):
    n = len(
        history_iso
    )

    history_can = np.empty_like(
        history_iso,
        dtype=np.float64
    )

    future_can = np.empty_like(
        future_iso,
        dtype=np.float64
    )

    origins = np.empty(
        (
            n,
            2
        ),
        dtype=np.float64
    )

    angles = np.empty(
        n,
        dtype=np.float64
    )

    scales = np.empty(
        n,
        dtype=np.float64
    )

    path_lengths = np.empty(
        n,
        dtype=np.float64
    )

    for i in range(n):
        c = canonicalize_sample(
            history_iso[
                i
            ],
            future_iso[
                i
            ],
            scale_floor
        )

        history_can[
            i
        ] = c[
            "history_canonical"
        ]

        future_can[
            i
        ] = c[
            "future_canonical"
        ]

        origins[
            i
        ] = c[
            "origin"
        ]

        angles[
            i
        ] = c[
            "angle"
        ]

        scales[
            i
        ] = c[
            "scale"
        ]

        path_lengths[
            i
        ] = c[
            "raw_path_length"
        ]

    return {
        "history_canonical":
            history_can,

        "future_canonical":
            future_can,

        "origins":
            origins,

        "angles":
            angles,

        "scales":
            scales,

        "path_lengths":
            path_lengths,
    }


def validate_and_clone_npz(
    task,
    old_arrays,
    final_meta,
    canonical,
    isotropic_floor,
):
    required = {
        "history_canonical",
        "future_canonical",
        "train_indices",
        "validation_indices",
        "train_sample_weights",
    }

    missing = required - set(
        old_arrays.files
    )

    if missing:
        raise RuntimeError(
            f"{task}: old final NPZ missing required keys: {sorted(missing)}"
        )

    old_h = np.asarray(
        old_arrays[
            "history_canonical"
        ]
    )

    old_f = np.asarray(
        old_arrays[
            "future_canonical"
        ]
    )

    if old_h.shape != canonical[
        "history_canonical"
    ].shape:
        raise RuntimeError(
            f"{task}: history_canonical shape mismatch: "
            f"old={old_h.shape}, new={canonical['history_canonical'].shape}"
        )

    if old_f.shape != canonical[
        "future_canonical"
    ].shape:
        raise RuntimeError(
            f"{task}: future_canonical shape mismatch: "
            f"old={old_f.shape}, new={canonical['future_canonical'].shape}"
        )

    train_indices = np.asarray(
        old_arrays[
            "train_indices"
        ],
        dtype=np.int64
    )

    val_indices = np.asarray(
        old_arrays[
            "validation_indices"
        ],
        dtype=np.int64
    )

    metadata_train = np.flatnonzero(
        final_meta[
            "neural_split"
        ]
        .astype(str)
        .to_numpy()
        ==
        "train"
    )

    metadata_val = np.flatnonzero(
        final_meta[
            "neural_split"
        ]
        .astype(str)
        .to_numpy()
        ==
        "validation"
    )

    if not np.array_equal(
        np.sort(
            train_indices
        ),
        metadata_train
    ):
        raise RuntimeError(
            f"{task}: old NPZ train_indices do not exactly match metadata neural_split."
        )

    if not np.array_equal(
        np.sort(
            val_indices
        ),
        metadata_val
    ):
        raise RuntimeError(
            f"{task}: old NPZ validation_indices do not exactly match metadata neural_split."
        )

    replacement_map = {
        "history_canonical":
            canonical[
                "history_canonical"
            ].astype(
                old_h.dtype,
                copy=False
            ),

        "future_canonical":
            canonical[
                "future_canonical"
            ].astype(
                old_f.dtype,
                copy=False
            ),
    }

    # Replace known coordinate-dependent auxiliary arrays only when present.
    name_to_new = {
        "origins":
            canonical[
                "origins"
            ],

        "origin":
            canonical[
                "origins"
            ],

        "angles":
            canonical[
                "angles"
            ],

        "angle":
            canonical[
                "angles"
            ],

        "heading_angles":
            canonical[
                "angles"
            ],

        "scales":
            canonical[
                "scales"
            ],

        "scale":
            canonical[
                "scales"
            ],

        "path_lengths":
            canonical[
                "path_lengths"
            ],

        "raw_path_lengths":
            canonical[
                "path_lengths"
            ],
    }

    suspicious = []

    for key in old_arrays.files:
        low = key.lower()

        if key in replacement_map:
            continue

        if key in name_to_new:
            old = np.asarray(
                old_arrays[
                    key
                ]
            )

            new = np.asarray(
                name_to_new[
                    key
                ]
            )

            if old.shape != new.shape:
                raise RuntimeError(
                    f"{task}: auxiliary key {key} shape mismatch: "
                    f"old={old.shape}, new={new.shape}"
                )

            replacement_map[
                key
            ] = new.astype(
                old.dtype,
                copy=False
            )

            continue

        if low == "scale_floor":
            old = np.asarray(
                old_arrays[
                    key
                ]
            )

            replacement_map[
                key
            ] = np.asarray(
                isotropic_floor,
                dtype=old.dtype
            )

            continue

        # Refuse silent copying of unknown geometry-dependent arrays.
        if any(
            token in low
            for token in [
                "canonical",
                "origin",
                "angle",
                "heading",
                "scale",
                "path_length",
            ]
        ):
            suspicious.append(
                key
            )

    if suspicious:
        raise RuntimeError(
            f"{task}: unrecognized potentially geometry-dependent NPZ keys: "
            f"{suspicious}. Refusing silent copy."
        )

    out_payload = {}

    for key in old_arrays.files:
        if key in replacement_map:
            out_payload[
                key
            ] = replacement_map[
                key
            ]
        else:
            out_payload[
                key
            ] = np.asarray(
                old_arrays[
                    key
                ]
            )

    return (
        out_payload,
        sorted(
            replacement_map.keys()
        ),
    )


def update_metadata(
    final_meta,
    mapped,
    floor_info,
    canonical,
    isotropic_floor,
):
    out = final_meta.copy()

    # Preserve all original columns and update existing scale_floor if present.
    if "scale_floor" in out.columns:
        out[
            "scale_floor"
        ] = isotropic_floor

    out[
        "r9_isotropic_coordinate_representation"
    ] = "x/S,y/S;S=max(width_px,height_px)"

    out[
        "r9_isotropic_scale_floor"
    ] = isotropic_floor

    out[
        "r9_isotropic_raw_path_length"
    ] = floor_info[
        "isotropic_path_lengths"
    ]

    out[
        "r9_isotropic_canonical_scale"
    ] = canonical[
        "scales"
    ]

    out[
        "r9_isotropic_heading_angle_rad"
    ] = canonical[
        "angles"
    ]

    out[
        "r9_isotropic_origin_x"
    ] = canonical[
        "origins"
    ][
        :,
        0
    ]

    out[
        "r9_isotropic_origin_y"
    ] = canonical[
        "origins"
    ][
        :,
        1
    ]

    out[
        "r9_corrected_source_sequence"
    ] = mapped[
        "sequence"
    ].astype(str).to_numpy()

    return out


def process_task(
    task,
    geometry,
    frozen_aniso_floor,
):
    cfg = TASKS[
        task
    ]

    (
        final_meta,
        old_arrays,
        final_meta_path,
        old_npz_path,
    ) = load_original_final_dataset(
        task
    )

    if len(
        final_meta
    ) != cfg[
        "expected_rows"
    ]:
        raise RuntimeError(
            f"{task}: expected {cfg['expected_rows']} final rows, "
            f"got {len(final_meta)}."
        )

    (
        corrected_core,
        corrected_arrays,
        corrected_meta_path,
        corrected_npz_path,
    ) = load_corrected_source(
        task
    )

    mapped = map_final_rows_to_corrected(
        task,
        final_meta,
        corrected_core
    )

    (
        history_aniso,
        future_aniso,
    ) = extract_corrected_windows(
        task,
        mapped,
        corrected_arrays
    )

    (
        history_iso,
        future_iso,
        pixel_roundtrip_max,
    ) = corrected_aniso_to_iso(
        task,
        mapped,
        history_aniso,
        future_aniso,
        geometry
    )

    floor_info = derive_floor_and_verify_rule(
        task,
        final_meta,
        history_aniso,
        history_iso,
        frozen_aniso_floor
    )

    isotropic_floor = floor_info[
        "isotropic_floor"
    ]

    canonical = canonicalize_all(
        history_iso,
        future_iso,
        isotropic_floor
    )

    (
        payload,
        replaced_keys,
    ) = validate_and_clone_npz(
        task,
        old_arrays,
        final_meta,
        canonical,
        isotropic_floor
    )

    out_meta = update_metadata(
        final_meta,
        mapped,
        floor_info,
        canonical,
        isotropic_floor
    )

    out_npz_path = (
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
        out_npz_path,
        **payload
    )

    out_meta.to_csv(
        out_meta_path,
        index=False
    )

    # Re-open and verify training-critical arrays exactly survived.
    new_arrays = np.load(
        out_npz_path,
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
                f"{task}: training-critical key {key} changed unexpectedly."
            )

    summary = {
        "task":
            task,

        "rows":
            int(
                len(
                    final_meta
                )
            ),

        "train_rows":
            int(
                (
                    final_meta[
                        "neural_split"
                    ].astype(str)
                    ==
                    "train"
                ).sum()
            ),

        "validation_rows":
            int(
                (
                    final_meta[
                        "neural_split"
                    ].astype(str)
                    ==
                    "validation"
                ).sum()
            ),

        "frozen_anisotropic_floor":
            float(
                frozen_aniso_floor
            ),

        "reproduced_anisotropic_floor":
            float(
                floor_info[
                    "reproduced_anisotropic_floor"
                ]
            ),

        "new_isotropic_train_floor":
            float(
                isotropic_floor
            ),

        "isotropic_over_anisotropic_floor_ratio":
            float(
                isotropic_floor
                /
                frozen_aniso_floor
            ),

        "train_isotropic_path_length_q05":
            float(
                np.quantile(
                    floor_info[
                        "isotropic_path_lengths"
                    ][
                        floor_info[
                            "train_mask"
                        ]
                    ],
                    0.05
                )
            ),

        "train_isotropic_path_length_median":
            float(
                np.median(
                    floor_info[
                        "isotropic_path_lengths"
                    ][
                        floor_info[
                            "train_mask"
                        ]
                    ]
                )
            ),

        "validation_isotropic_path_length_median":
            float(
                np.median(
                    floor_info[
                        "isotropic_path_lengths"
                    ][
                        floor_info[
                            "validation_mask"
                        ]
                    ]
                )
            ),

        "pixel_roundtrip_max_abs_error":
            float(
                pixel_roundtrip_max
            ),

        "npz_keys":
            "|".join(
                old_arrays.files
            ),

        "replaced_npz_keys":
            "|".join(
                replaced_keys
            ),

        "output_npz":
            str(
                out_npz_path
            ),

        "output_npz_sha256":
            sha256_file(
                out_npz_path
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

        "corrected_metadata":
            str(
                corrected_meta_path
            ),

        "corrected_metadata_sha256":
            sha256_file(
                corrected_meta_path
            ),

        "corrected_npz":
            str(
                corrected_npz_path
            ),

        "corrected_npz_sha256":
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
        "\nR9-I-C — ISOTROPIC TRAINREAL SCALE-FLOOR RE-DERIVATION + MATCHED CANONICAL DATASET BUILD"
    )

    print(
        "NO neural training and NO model inference. "
        "Exact final BIRDSAI TrainReal train/validation membership is preserved. "
        "Camera-corrected anisotropic coordinates are converted to aspect-preserving "
        "isotropic coordinates x/S,y/S with S=max(W,H). The original 5th-percentile "
        "scale-floor derivation rule is first reproduced on anisotropic TRAIN rows "
        "and must match the frozen floor; then a NEW isotropic scale floor is derived "
        "from isotropic TRAIN rows only. Validation/TestReal/WildBox are excluded from "
        "floor derivation. Exact frozen canonicalizer geometry is then applied.\n"
    )

    (
        r9i_b,
        floor_lock,
        frozen_aniso_floors,
    ) = verify_protocol_locks()

    geometry = load_geometry()

    summaries = []
    provenance = []

    for task in TASKS:
        summary, prov = process_task(
            task,
            geometry,
            frozen_aniso_floors[
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
        "R9I_C_TASK_SUMMARY.csv"
    )

    provenance_path = (
        OUT
        /
        "R9I_C_PROVENANCE.csv"
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
        "R9I_C_ISOTROPIC_TRAINREAL_SCALE_FLOORS_DERIVED_AND_MATCHED_CANONICAL_DATASETS_BUILT_READY_FOR_R9I_D_MATCHED_NEURAL_RETRAINING_SCRIPT"
    )

    manifest = {
        "step":
            "R9I_C_ISOTROPIC_TRAINREAL_SCALE_FLOOR_AND_CANONICAL_DATASET_BUILD",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "coordinate_representation":
            "x/S,y/S with S=max(width_px,height_px)",

        "scale_floor_rule":
            {
                "unit":
                    "window",

                "source_split":
                    "BIRDSAI TrainReal neural_split=train only",

                "quantile":
                    SCALE_FLOOR_QUANTILE,

                "validation_used":
                    False,

                "testreal_used":
                    False,

                "wildbox_used":
                    False,
            },

        "same_split_membership":
            True,

        "same_train_sample_weights":
            True,

        "same_train_indices":
            True,

        "same_validation_indices":
            True,

        "new_training":
            False,

        "model_inference":
            False,

        "frozen_results_modified":
            False,

        "output_dataset_directory":
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

        "tasks":
            summary_df.to_dict(
                orient="records"
            ),

        "important_nonclaims": [
            "No isotropic neural model has been trained in this step.",
            "No forecasting-performance improvement is claimed.",
            "Old anisotropic checkpoints are not evaluated on isotropic tensors.",
            "TestReal and WildBox are not used to derive the isotropic scale floor.",
        ],
    }

    (
        OUT
        /
        "R9I_C_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "ISOTROPIC TRAINREAL DATASET SUMMARY:"
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
            f"anisotropic frozen floor={row['frozen_anisotropic_floor']:.17g}, "
            f"reproduced={row['reproduced_anisotropic_floor']:.17g}, "
            f"new isotropic train-only floor={row['new_isotropic_train_floor']:.17g}, "
            f"train/val={int(row['train_rows'])}/{int(row['validation_rows'])}"
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-I-C COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
