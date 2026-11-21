
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

R9I_C_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_C_V2_matched_isotropic_trainreal_dataset"
    / "R9I_C_V2_MANIFEST.json"
)

R9I_D1_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_D1_matched_isotropic_training_only"
    / "R9I_D1_MANIFEST.json"
)

GEOMETRY_CSV = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3B_corrected_isotropic_rebuild"
    / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_E0_matched_testreal_isotropic_dataset"
)

DATASET_OUT = OUT / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
DATASET_OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "expected_rows": 1053,
        "future_len": 20,
    },
    "2s_to_5s": {
        "expected_rows": 826,
        "future_len": 50,
    },
}

DOWNSAMPLE = 3
RECENT_HEADING_FRAMES = 10
PIXEL_ROUNDTRIP_TOL = 1e-9
ANISO_ABS_MATCH_TOL = 2e-6

EXPECTED_TEST_NPZ_KEYS = {
    "history_canonical",
    "future_canonical",
    "history_abs",
    "future_abs",
    "origin",
    "heading_angle",
    "scale",
    "raw_path_length",
}


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


def verify_upstream():
    for p in [
        R9I_C_MANIFEST,
        R9I_D1_MANIFEST,
        GEOMETRY_CSV,
    ]:
        require(p)

    c = json.loads(
        R9I_C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    d1 = json.loads(
        R9I_D1_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        c.get("status")
        !=
        "R9I_C_V2_MATCHED_ISOTROPIC_TRAINREAL_CANONICAL_DATASETS_BUILT_READY_FOR_R9I_D_MATCHED_NEURAL_RETRAINING_SCRIPT"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-C status: {c.get('status')}"
        )

    if (
        d1.get("status")
        !=
        "R9I_D1_ALL_18_MATCHED_ISOTROPIC_NEURAL_RUNS_TRAINED_READY_FOR_R9I_E_MATCHED_TESTREAL_ISOTROPIC_EVALUATION_BUILD"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-D1 status: {d1.get('status')}"
        )

    if not d1.get(
        "all_18_runs_complete",
        False
    ):
        raise RuntimeError(
            "R9-I-D1 manifest does not confirm all 18 runs complete."
        )

    floors = {}

    for row in c[
        "tasks"
    ]:
        floors[
            row[
                "task"
            ]
        ] = float(
            row[
                "new_isotropic_all_trainreal_floor"
            ]
        )

    for task in TASKS:
        if task not in floors:
            raise RuntimeError(
                f"Missing isotropic TrainReal floor for {task}."
            )

    return (
        c,
        d1,
        floors,
    )


def load_geometry():
    df = pd.read_csv(
        GEOMETRY_CSV,
        dtype={
            "sequence": str,
        }
    )

    geometry = {}

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
            f"Geometry CSV missing columns: {sorted(missing)}"
        )

    for _, row in df.iterrows():
        seq = str(
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
                f"{seq}: iso scale mismatch."
            )

        geometry[
            seq
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

    if float(
        np.linalg.norm(
            velocity
        )
    ) <= 1e-10:
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


def transform_dataset(
    history,
    future,
    scale_floor,
):
    hc = []
    fc = []
    origins = []
    angles = []
    scales = []
    raw_paths = []

    for i in range(
        len(
            history
        )
    ):
        h = np.asarray(
            history[
                i
            ],
            dtype=np.float64
        )

        f = np.asarray(
            future[
                i
            ],
            dtype=np.float64
        )

        origin = h[
            -1
        ].copy()

        angle = estimate_heading(
            h
        )

        R = rotation_to_heading(
            angle
        )

        path_length = history_path_length(
            h
        )

        scale = max(
            path_length,
            float(
                scale_floor
            )
        )

        h_can = (
            (
                h
                -
                origin
            )
            @
            R.T
        ) / scale

        f_can = (
            (
                f
                -
                origin
            )
            @
            R.T
        ) / scale

        hc.append(
            h_can.astype(
                np.float32
            )
        )

        fc.append(
            f_can.astype(
                np.float32
            )
        )

        origins.append(
            origin.astype(
                np.float32
            )
        )

        angles.append(
            np.float32(
                angle
            )
        )

        scales.append(
            np.float32(
                scale
            )
        )

        raw_paths.append(
            np.float32(
                path_length
            )
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
            np.asarray(
                history,
                dtype=np.float32
            ),

        "future_abs":
            np.asarray(
                future,
                dtype=np.float32
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


def load_original_test(task):
    meta_path = (
        FINAL_NEURAL_DIR
        /
        f"{task}_birdsai_test_metadata.csv"
    )

    npz_path = (
        FINAL_NEURAL_DIR
        /
        f"{task}_birdsai_test_canonical.npz"
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
        }
    )

    arrays = np.load(
        npz_path,
        allow_pickle=False
    )

    if set(
        arrays.files
    ) != EXPECTED_TEST_NPZ_KEYS:
        raise RuntimeError(
            f"{task}: unexpected original TestReal NPZ keys: {arrays.files}"
        )

    return (
        meta,
        arrays,
        meta_path,
        npz_path,
    )


def load_corrected_testreal(task):
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
            "test_real"
        )
    ].copy()

    if core[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{task}: duplicate TestReal sample_id."
        )

    return (
        core,
        arrays,
        meta_path,
        npz_path,
    )


def map_test_rows(
    task,
    final_meta,
    corrected_core,
):
    if final_meta[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{task}: final TestReal sample_id duplicate."
        )

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
                f"{task}: final TestReal sample_id={sid} absent from corrected test_real core."
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

    return pd.DataFrame(
        rows
    ).reset_index(
        drop=True
    )


def extract_anisotropic_windows(
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

        histories.append(
            h
        )

        futures.append(
            f
        )

    h = np.stack(
        histories
    )

    f = np.stack(
        futures
    )

    cfg = TASKS[
        task
    ]

    if h.shape != (
        cfg[
            "expected_rows"
        ],
        20,
        2,
    ):
        raise RuntimeError(
            f"{task}: history shape={h.shape}"
        )

    if f.shape != (
        cfg[
            "expected_rows"
        ],
        cfg[
            "future_len"
        ],
        2,
    ):
        raise RuntimeError(
            f"{task}: future shape={f.shape}"
        )

    return (
        h,
        f,
    )


def validate_anisotropic_source(
    task,
    history_aniso,
    future_aniso,
    original_arrays,
):
    old_h = np.asarray(
        original_arrays[
            "history_abs"
        ],
        dtype=np.float64
    )

    old_f = np.asarray(
        original_arrays[
            "future_abs"
        ],
        dtype=np.float64
    )

    if old_h.shape != history_aniso.shape:
        raise RuntimeError(
            f"{task}: original history_abs shape mismatch."
        )

    if old_f.shape != future_aniso.shape:
        raise RuntimeError(
            f"{task}: original future_abs shape mismatch."
        )

    h_err = float(
        np.max(
            np.abs(
                old_h
                -
                history_aniso
            )
        )
    )

    f_err = float(
        np.max(
            np.abs(
                old_f
                -
                future_aniso
            )
        )
    )

    if h_err > ANISO_ABS_MATCH_TOL:
        raise RuntimeError(
            f"{task}: original TestReal history_abs does not match reconstructed corrected source. max={h_err}"
        )

    if f_err > ANISO_ABS_MATCH_TOL:
        raise RuntimeError(
            f"{task}: original TestReal future_abs does not match reconstructed corrected source. max={f_err}"
        )

    return (
        h_err,
        f_err,
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
        seq = str(
            row[
                "sequence"
            ]
        )

        if seq not in geometry:
            raise RuntimeError(
                f"{task}: missing geometry for sequence {seq}."
            )

        W = geometry[
            seq
        ][
            "W"
        ]

        H = geometry[
            seq
        ][
            "H"
        ]

        S = geometry[
            seq
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

        max_roundtrip = max(
            max_roundtrip,
            float(
                np.max(
                    np.abs(
                        history_iso[
                            i
                        ]
                        *
                        S
                        -
                        h_px
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        future_iso[
                            i
                        ]
                        *
                        S
                        -
                        f_px
                    )
                )
            ),
        )

    if max_roundtrip > PIXEL_ROUNDTRIP_TOL:
        raise RuntimeError(
            f"{task}: pixel roundtrip failed, max={max_roundtrip}"
        )

    return (
        history_iso,
        future_iso,
        max_roundtrip,
    )


def update_metadata(
    meta,
    transformed,
    scale_floor,
):
    out = meta.copy()

    if "scale_floor" in out.columns:
        out[
            "scale_floor"
        ] = scale_floor

    out[
        "r9_coordinate_representation"
    ] = "isotropic_x_over_S_y_over_S"

    out[
        "r9_isotropic_scale_floor"
    ] = scale_floor

    out[
        "r9_isotropic_raw_path_length"
    ] = transformed[
        "raw_path_length"
    ]

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
    scale_floor,
):
    cfg = TASKS[
        task
    ]

    (
        final_meta,
        original_arrays,
        final_meta_path,
        original_npz_path,
    ) = load_original_test(
        task
    )

    if len(
        final_meta
    ) != cfg[
        "expected_rows"
    ]:
        raise RuntimeError(
            f"{task}: final TestReal rows={len(final_meta)}, expected={cfg['expected_rows']}"
        )

    (
        corrected_core,
        corrected_arrays,
        corrected_meta_path,
        corrected_npz_path,
    ) = load_corrected_testreal(
        task
    )

    if len(
        corrected_core
    ) != cfg[
        "expected_rows"
    ]:
        raise RuntimeError(
            f"{task}: corrected TestReal core rows={len(corrected_core)}, expected={cfg['expected_rows']}"
        )

    mapped = map_test_rows(
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
        aniso_h_maxerr,
        aniso_f_maxerr,
    ) = validate_anisotropic_source(
        task,
        history_aniso,
        future_aniso,
        original_arrays
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

    transformed = transform_dataset(
        history_iso,
        future_iso,
        scale_floor
    )

    out_npz = (
        DATASET_OUT
        /
        f"{task}_birdsai_test_canonical.npz"
    )

    out_meta_path = (
        DATASET_OUT
        /
        f"{task}_birdsai_test_metadata.csv"
    )

    np.savez_compressed(
        out_npz,
        **transformed
    )

    out_meta = update_metadata(
        final_meta,
        transformed,
        scale_floor
    )

    out_meta.to_csv(
        out_meta_path,
        index=False
    )

    new_arrays = np.load(
        out_npz,
        allow_pickle=False
    )

    if set(
        new_arrays.files
    ) != EXPECTED_TEST_NPZ_KEYS:
        raise RuntimeError(
            f"{task}: output TestReal NPZ key mismatch."
        )

    for key in EXPECTED_TEST_NPZ_KEYS:
        if not np.isfinite(
            np.asarray(
                new_arrays[
                    key
                ]
            )
        ).all():
            raise RuntimeError(
                f"{task}: non-finite values in {key}."
            )

    summary = {
        "task":
            task,

        "samples":
            int(
                len(
                    final_meta
                )
            ),

        "tracks":
            int(
                final_meta[
                    "track_key"
                ].nunique()
            )
            if "track_key" in final_meta.columns
            else None,

        "sequences":
            int(
                final_meta[
                    "source_group"
                ].nunique()
            )
            if "source_group" in final_meta.columns
            else int(
                final_meta[
                    "sequence"
                ].nunique()
            ),

        "trainreal_derived_isotropic_scale_floor":
            float(
                scale_floor
            ),

        "original_anisotropic_history_abs_max_error":
            aniso_h_maxerr,

        "original_anisotropic_future_abs_max_error":
            aniso_f_maxerr,

        "pixel_roundtrip_max_abs_error":
            float(
                pixel_roundtrip
            ),

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

        "original_anisotropic_test_metadata":
            str(
                final_meta_path
            ),

        "original_anisotropic_test_metadata_sha256":
            sha256_file(
                final_meta_path
            ),

        "original_anisotropic_test_npz":
            str(
                original_npz_path
            ),

        "original_anisotropic_test_npz_sha256":
            sha256_file(
                original_npz_path
            ),

        "corrected_source_metadata":
            str(
                corrected_meta_path
            ),

        "corrected_source_metadata_sha256":
            sha256_file(
                corrected_meta_path
            ),

        "corrected_source_npz":
            str(
                corrected_npz_path
            ),

        "corrected_source_npz_sha256":
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
        "\nR9-I-E0 — MATCHED BIRDSAI TESTREAL ISOTROPIC CANONICAL DATASET BUILD"
    )

    print(
        "NO training and NO model inference. The already-frozen matched isotropic "
        "checkpoints are left untouched. This step rebuilds the exact BIRDSAI TestReal "
        "sample set in aspect-preserving isotropic coordinates, using the camera-corrected "
        "TestReal trajectories, per-sequence S=max(W,H), the exact frozen canonicalizer, "
        "and the isotropic scale floor derived ONLY from BIRDSAI TrainReal in R9-I-C. "
        "No TestReal tuning or recalibration occurs.\n"
    )

    (
        c,
        d1,
        floors,
    ) = verify_upstream()

    geometry = load_geometry()

    summaries = []
    provenance = []

    for task in TASKS:
        summary, prov = process_task(
            task,
            geometry,
            floors[
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
        "R9I_E0_TESTREAL_ISOTROPIC_DATASET_SUMMARY.csv"
    )

    provenance_path = (
        OUT
        /
        "R9I_E0_PROVENANCE.csv"
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
        "R9I_E0_MATCHED_TESTREAL_ISOTROPIC_CANONICAL_DATASETS_BUILT_READY_FOR_R9I_E1_FROZEN_18_CHECKPOINT_TESTREAL_INFERENCE"
    )

    manifest = {
        "step":
            "R9I_E0_MATCHED_BIRDSAI_TESTREAL_ISOTROPIC_DATASET_BUILD",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "dataset":
            "BIRDSAI TestReal",

        "coordinate_representation":
            "x/S,y/S where S=max(width_px,height_px)",

        "scale_floor_source":
            "BIRDSAI TrainReal only, exact original pre-split 5th-percentile rule",

        "testreal_used_for_scale_floor":
            False,

        "training":
            False,

        "model_inference":
            False,

        "frozen_isotropic_checkpoints_modified":
            False,

        "original_frozen_anisotropic_results_modified":
            False,

        "dataset_directory":
            str(
                DATASET_OUT
            ),

        "task_summary_csv":
            str(
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
    }

    (
        OUT
        /
        "R9I_E0_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "TESTREAL ISOTROPIC DATASET SUMMARY:"
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
            f"  {row['task']}: samples={int(row['samples'])}; "
            f"tracks={int(row['tracks']) if pd.notna(row['tracks']) else 'NA'}; "
            f"sequences={int(row['sequences'])}; "
            f"TrainReal-derived isotropic floor={row['trainreal_derived_isotropic_scale_floor']:.17g}; "
            f"aniso source maxerr history={row['original_anisotropic_history_abs_max_error']:.3e}, "
            f"future={row['original_anisotropic_future_abs_max_error']:.3e}; "
            f"pixel roundtrip={row['pixel_roundtrip_max_abs_error']:.3e}"
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-I-E0 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
