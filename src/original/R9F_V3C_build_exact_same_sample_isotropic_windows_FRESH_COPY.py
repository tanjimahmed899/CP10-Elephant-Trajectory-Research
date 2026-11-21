
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

R9F_V3B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9F_V3B_exact_mapping_recovery"
)

ISO_POINTS = (
    R9D_DIR
    / "R9D_isotropic_cleaned_track_points.csv"
)

R9D_MANIFEST = (
    R9D_DIR
    / "R9D_MANIFEST.json"
)

V3B_MANIFEST = (
    R9F_V3B_DIR
    / "R9F_V3B_MANIFEST.json"
)

FRAME_LOCK = (
    R9F_V3B_DIR
    / "R9F_V3B_EXACT_CORE_SAMPLE_FRAME_LOCK.csv"
)

CORR_DIR = (
    MAIN1
    / "corrected_trajectory_dataset"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9F_V3C_exact_same_sample_isotropic_windows"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

TASKS = {
    "2s_to_2s": {
        "expected_samples": 1053,
        "history_raw": 60,
        "future_raw": 60,
        "history_10hz": 20,
        "future_10hz": 20,
    },

    "2s_to_5s": {
        "expected_samples": 826,
        "history_raw": 60,
        "future_raw": 150,
        "history_10hz": 20,
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


def verify_inputs():
    for p in [
        ISO_POINTS,
        R9D_MANIFEST,
        V3B_MANIFEST,
        FRAME_LOCK,
    ]:
        require(p)

    r9d = json.loads(
        R9D_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    v3b = json.loads(
        V3B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        r9d.get("status")
        !=
        "R9D_NONDESTRUCTIVE_ISOTROPIC_POINT_DATASET_BUILT_READY_FOR_R9E_WINDOW_PROTOCOL_MATCH"
    ):
        raise RuntimeError(
            f"Unexpected R9D status: {r9d.get('status')}"
        )

    if (
        v3b.get("status")
        !=
        "R9F_V3B_EXACT_STEP2B_MAPPING_RECOVERED_AND_LOCKED_READY_FOR_R9F_V3C_ISOTROPIC_WINDOW_BUILD"
    ):
        raise RuntimeError(
            f"Unexpected R9F-V3B status: {v3b.get('status')}"
        )

    return r9d, v3b


def identify_iso_columns(df):
    required = {
        "split",
        "sequence",
        "segment_id",
        "frame",
        "cx_iso",
        "cy_iso",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            "R9D isotropic point dataset is missing exact Step2b lookup columns: "
            f"{sorted(missing)}"
        )

    return {
        "split": "split",
        "sequence": "sequence",
        "segment": "segment_id",
        "frame": "frame",
        "x": "cx_iso",
        "y": "cy_iso",
    }


def build_track_lookup(df, cols):
    lookup = {}

    group_cols = [
        cols["split"],
        cols["sequence"],
        cols["segment"],
    ]

    for key, g in df.groupby(
        group_cols,
        sort=False,
        dropna=False,
    ):
        key = tuple(
            str(x)
            for x in key
        )

        gg = g.copy()

        frames = pd.to_numeric(
            gg[
                cols["frame"]
            ],
            errors="raise",
        ).astype(int)

        if frames.duplicated().any():
            dup = frames[
                frames.duplicated()
            ].unique().tolist()

            raise RuntimeError(
                f"Duplicate frame values for lookup key {key}: {dup[:20]}"
            )

        gg[
            "_frame_int"
        ] = frames

        lookup[
            key
        ] = (
            gg.set_index(
                "_frame_int"
            )
            .sort_index()
        )

    return lookup


def load_frozen_corrected_reference(task):
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

    sample_ids = pd.to_numeric(
        core[
            "sample_id"
        ],
        errors="raise",
    ).astype(int)

    corrected_history = (
        np.asarray(
            history_all[
                sample_ids.to_numpy(),
                :,
                :2,
            ],
            dtype=np.float64,
        )
        [
            :,
            ::DOWNSAMPLE,
            :,
        ]
    )

    corrected_future = (
        np.asarray(
            future_all[
                sample_ids.to_numpy(),
                :,
                :2,
            ],
            dtype=np.float64,
        )
        [
            :,
            ::DOWNSAMPLE,
            :,
        ]
    )

    return (
        core,
        corrected_history,
        corrected_future,
        meta_path,
        npz_path,
    )


def verify_frame_lock_against_core(
    task,
    core,
    lock,
):
    lock_task = lock[
        lock[
            "task"
        ].astype(str)
        ==
        task
    ].copy()

    lock_task = (
        lock_task
        .sort_values(
            "sample_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    core_cmp = core[
        [
            "sample_id",
            "sequence",
            "segment_id",
            "obs_start_frame",
            "obs_end_frame",
            "future_end_frame",
        ]
    ].copy()

    core_cmp = (
        core_cmp
        .sort_values(
            "sample_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    for c in [
        "sample_id",
        "obs_start_frame",
        "obs_end_frame",
        "future_end_frame",
    ]:
        lock_task[
            c
        ] = pd.to_numeric(
            lock_task[
                c
            ],
            errors="raise",
        ).astype(int)

        core_cmp[
            c
        ] = pd.to_numeric(
            core_cmp[
                c
            ],
            errors="raise",
        ).astype(int)

    for c in [
        "sequence",
        "segment_id",
    ]:
        lock_task[
            c
        ] = lock_task[
            c
        ].astype(str)

        core_cmp[
            c
        ] = core_cmp[
            c
        ].astype(str)

    if len(
        lock_task
    ) != len(
        core_cmp
    ):
        raise RuntimeError(
            f"{task}: frame-lock row count differs from exact filtered core."
        )

    if not lock_task[
        [
            "sample_id",
            "sequence",
            "segment_id",
            "obs_start_frame",
            "obs_end_frame",
            "future_end_frame",
        ]
    ].equals(
        core_cmp
    ):
        raise RuntimeError(
            f"{task}: R9F-V3B frame lock no longer matches exact filtered core metadata."
        )

    return lock_task


def extract_isotropic_window(
    track,
    obs_start_frame,
    obs_end_frame,
    future_end_frame,
    cfg,
):
    expected_obs_end = (
        int(
            obs_start_frame
        )
        +
        cfg[
            "history_raw"
        ]
        -
        1
    )

    if (
        int(
            obs_end_frame
        )
        !=
        expected_obs_end
    ):
        raise RuntimeError(
            f"History frame bound mismatch: obs_start={obs_start_frame}, "
            f"obs_end={obs_end_frame}, expected={expected_obs_end}"
        )

    expected_future_end = (
        int(
            obs_start_frame
        )
        +
        cfg[
            "history_raw"
        ]
        +
        cfg[
            "future_raw"
        ]
        -
        1
    )

    if (
        int(
            future_end_frame
        )
        !=
        expected_future_end
    ):
        raise RuntimeError(
            f"Future frame bound mismatch: start={obs_start_frame}, "
            f"future_end={future_end_frame}, expected={expected_future_end}"
        )

    needed_frames = list(
        range(
            int(
                obs_start_frame
            ),
            int(
                future_end_frame
            )
            +
            1,
        )
    )

    missing = [
        f
        for f in needed_frames
        if f not in track.index
    ]

    if missing:
        raise RuntimeError(
            f"Missing frames in isotropic source; first missing={missing[0]}"
        )

    raw = (
        track.loc[
            needed_frames,
            [
                "cx_iso",
                "cy_iso",
            ],
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    expected_length = (
        cfg[
            "history_raw"
        ]
        +
        cfg[
            "future_raw"
        ]
    )

    if len(
        raw
    ) != expected_length:
        raise RuntimeError(
            f"Unexpected isotropic raw length {len(raw)} != {expected_length}"
        )

    history_raw = raw[
        :
        cfg[
            "history_raw"
        ]
    ]

    future_raw = raw[
        cfg[
            "history_raw"
        ]:
    ]

    history_10hz = history_raw[
        ::DOWNSAMPLE
    ]

    future_10hz = future_raw[
        ::DOWNSAMPLE
    ]

    return (
        history_10hz,
        future_10hz,
    )


def build_task(
    task,
    cfg,
    lookup,
    frame_lock,
):
    (
        core,
        corrected_history,
        corrected_future,
        meta_path,
        npz_path,
    ) = load_frozen_corrected_reference(
        task
    )

    lock_task = verify_frame_lock_against_core(
        task,
        core,
        frame_lock,
    )

    if len(
        core
    ) != cfg[
        "expected_samples"
    ]:
        raise RuntimeError(
            f"{task}: exact core count {len(core)} "
            f"!= expected {cfg['expected_samples']}"
        )

    histories = []
    futures = []
    rows = []

    for _, row in lock_task.iterrows():
        key = (
            "test_real",
            str(
                row[
                    "sequence"
                ]
            ),
            str(
                row[
                    "segment_id"
                ]
            ),
        )

        if key not in lookup:
            raise RuntimeError(
                f"{task}: isotropic Step2b lookup key missing: {key}"
            )

        history, future = extract_isotropic_window(
            lookup[
                key
            ],
            int(
                row[
                    "obs_start_frame"
                ]
            ),
            int(
                row[
                    "obs_end_frame"
                ]
            ),
            int(
                row[
                    "future_end_frame"
                ]
            ),
            cfg,
        )

        if history.shape != (
            cfg[
                "history_10hz"
            ],
            2,
        ):
            raise RuntimeError(
                f"{task} sample {row['sample_id']}: "
                f"history shape {history.shape}"
            )

        if future.shape != (
            cfg[
                "future_10hz"
            ],
            2,
        ):
            raise RuntimeError(
                f"{task} sample {row['sample_id']}: "
                f"future shape {future.shape}"
            )

        histories.append(
            history
        )

        futures.append(
            future
        )

        rows.append(
            {
                "task":
                    task,

                "sample_id":
                    int(
                        row[
                            "sample_id"
                        ]
                    ),

                "split":
                    "test_real",

                "sequence":
                    str(
                        row[
                            "sequence"
                        ]
                    ),

                "segment_id":
                    str(
                        row[
                            "segment_id"
                        ]
                    ),

                "obs_start_frame":
                    int(
                        row[
                            "obs_start_frame"
                        ]
                    ),

                "obs_end_frame":
                    int(
                        row[
                            "obs_end_frame"
                        ]
                    ),

                "future_end_frame":
                    int(
                        row[
                            "future_end_frame"
                        ]
                    ),
            }
        )

    history_arr = np.stack(
        histories,
        axis=0,
    )

    future_arr = np.stack(
        futures,
        axis=0,
    )

    if history_arr.shape != corrected_history.shape:
        raise RuntimeError(
            f"{task}: isotropic history shape {history_arr.shape} "
            f"!= frozen corrected history shape {corrected_history.shape}"
        )

    if future_arr.shape != corrected_future.shape:
        raise RuntimeError(
            f"{task}: isotropic future shape {future_arr.shape} "
            f"!= frozen corrected future shape {corrected_future.shape}"
        )

    return (
        history_arr,
        future_arr,
        pd.DataFrame(
            rows
        ),
        corrected_history,
        corrected_future,
        meta_path,
        npz_path,
    )


def geometry_audit(
    task,
    iso_history,
    iso_future,
    corrected_history,
    corrected_future,
):
    # This is not an equality test: the point is to quantify representation
    # differences while ensuring same sample/time shapes.
    hist_delta = np.linalg.norm(
        iso_history
        -
        corrected_history,
        axis=2,
    )

    fut_delta = np.linalg.norm(
        iso_future
        -
        corrected_future,
        axis=2,
    )

    return {
        "task":
            task,

        "same_history_shape":
            bool(
                iso_history.shape
                ==
                corrected_history.shape
            ),

        "same_future_shape":
            bool(
                iso_future.shape
                ==
                corrected_future.shape
            ),

        "isotropic_history_all_finite":
            bool(
                np.isfinite(
                    iso_history
                ).all()
            ),

        "isotropic_future_all_finite":
            bool(
                np.isfinite(
                    iso_future
                ).all()
            ),

        "median_history_representation_delta":
            float(
                np.median(
                    hist_delta
                )
            ),

        "max_history_representation_delta":
            float(
                np.max(
                    hist_delta
                )
            ),

        "median_future_representation_delta":
            float(
                np.median(
                    fut_delta
                )
            ),

        "max_future_representation_delta":
            float(
                np.max(
                    fut_delta
                )
            ),

        "iso_history_min":
            float(
                iso_history.min()
            ),

        "iso_history_max":
            float(
                iso_history.max()
            ),

        "iso_future_min":
            float(
                iso_future.min()
            ),

        "iso_future_max":
            float(
                iso_future.max()
            ),
    }


def main():
    print(
        "\nR9-F-V3C — EXACT SAME-SAMPLE ISOTROPIC WINDOW BUILD"
    )

    print(
        "NO model training and NO frozen overwrite. "
        "This step uses the exact Step2b key=(split,sequence,segment_id), "
        "the locked sample IDs, and exact obs_start/obs_end/future_end frame bounds "
        "to build parallel isotropic 10 Hz windows.\n"
    )

    verify_inputs()

    iso = pd.read_csv(
        ISO_POINTS,
        dtype={
            "split": str,
            "sequence": str,
            "segment_id": str,
        },
    )

    frame_lock = pd.read_csv(
        FRAME_LOCK,
        dtype={
            "sequence": str,
            "segment_id": str,
        },
    )

    cols = identify_iso_columns(
        iso
    )

    lookup = build_track_lookup(
        iso,
        cols,
    )

    summary_rows = []
    geometry_rows = []
    provenance = {}

    for task, cfg in TASKS.items():
        (
            history,
            future,
            metadata,
            corrected_history,
            corrected_future,
            corr_meta_path,
            corr_npz_path,
        ) = build_task(
            task,
            cfg,
            lookup,
            frame_lock,
        )

        npz_out = (
            OUT
            / f"R9F_V3C_isotropic_{task}.npz"
        )

        meta_out = (
            OUT
            / f"R9F_V3C_isotropic_windows_{task}.csv"
        )

        np.savez_compressed(
            npz_out,
            history=history,
            future=future,
        )

        metadata.to_csv(
            meta_out,
            index=False,
        )

        summary_rows.append(
            {
                "task":
                    task,

                "samples":
                    int(
                        history.shape[
                            0
                        ]
                    ),

                "history_shape":
                    str(
                        tuple(
                            history.shape
                        )
                    ),

                "future_shape":
                    str(
                        tuple(
                            future.shape
                        )
                    ),

                "expected_samples":
                    int(
                        cfg[
                            "expected_samples"
                        ]
                    ),

                "sample_count_match":
                    bool(
                        history.shape[
                            0
                        ]
                        ==
                        cfg[
                            "expected_samples"
                        ]
                    ),

                "history_all_finite":
                    bool(
                        np.isfinite(
                            history
                        ).all()
                    ),

                "future_all_finite":
                    bool(
                        np.isfinite(
                            future
                        ).all()
                    ),

                "output_npz":
                    str(
                        npz_out
                    ),

                "output_npz_sha256":
                    sha256_file(
                        npz_out
                    ),

                "output_metadata":
                    str(
                        meta_out
                    ),

                "output_metadata_sha256":
                    sha256_file(
                        meta_out
                    ),
            }
        )

        geometry_rows.append(
            geometry_audit(
                task,
                history,
                future,
                corrected_history,
                corrected_future,
            )
        )

        provenance[
            task
        ] = {
            "frozen_corrected_metadata":
                str(
                    corr_meta_path
                ),

            "frozen_corrected_metadata_sha256":
                sha256_file(
                    corr_meta_path
                ),

            "frozen_corrected_npz":
                str(
                    corr_npz_path
                ),

            "frozen_corrected_npz_sha256":
                sha256_file(
                    corr_npz_path
                ),

            "mapping_rule":
                {
                    "lookup_key":
                        "(split, sequence, segment_id)",

                    "window_start":
                        "obs_start_frame",

                    "history_end":
                        "obs_end_frame",

                    "window_end":
                        "future_end_frame",

                    "coordinate_columns":
                        [
                            "cx_iso",
                            "cy_iso",
                        ],

                    "downsample":
                        "::3",
                },
        }

    summary = pd.DataFrame(
        summary_rows
    )

    geometry = pd.DataFrame(
        geometry_rows
    )

    if not summary[
        "sample_count_match"
    ].all():
        raise RuntimeError(
            "R9F-V3C sample-count mismatch."
        )

    if not summary[
        "history_all_finite"
    ].all():
        raise RuntimeError(
            "R9F-V3C history contains non-finite values."
        )

    if not summary[
        "future_all_finite"
    ].all():
        raise RuntimeError(
            "R9F-V3C future contains non-finite values."
        )

    if not geometry[
        "same_history_shape"
    ].all():
        raise RuntimeError(
            "R9F-V3C history shape mismatch versus frozen corrected reference."
        )

    if not geometry[
        "same_future_shape"
    ].all():
        raise RuntimeError(
            "R9F-V3C future shape mismatch versus frozen corrected reference."
        )

    summary_path = (
        OUT
        / "R9F_V3C_WINDOW_BUILD_SUMMARY.csv"
    )

    geometry_path = (
        OUT
        / "R9F_V3C_GEOMETRY_REPRESENTATION_AUDIT.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    geometry.to_csv(
        geometry_path,
        index=False,
    )

    status = (
        "R9F_V3C_EXACT_SAME_SAMPLE_ISOTROPIC_WINDOWS_BUILT_READY_FOR_R9G_BASELINE_COMPARISON"
    )

    manifest = {
        "step":
            "R9F_V3C_EXACT_SAME_SAMPLE_ISOTROPIC_WINDOW_BUILD",

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

        "isotropic_source":
            str(
                ISO_POINTS
            ),

        "isotropic_source_sha256":
            sha256_file(
                ISO_POINTS
            ),

        "exact_frame_lock":
            str(
                FRAME_LOCK
            ),

        "exact_frame_lock_sha256":
            sha256_file(
                FRAME_LOCK
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

        "geometry_audit_sha256":
            sha256_file(
                geometry_path
            ),

        "provenance":
            provenance,
    }

    (
        OUT
        / "R9F_V3C_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "ISOTROPIC WINDOW BUILD SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nGEOMETRY / REPRESENTATION AUDIT:"
    )

    print(
        geometry.to_string(
            index=False
        )
    )

    print(
        "\nLOCKED PROVENANCE:"
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
        "\nR9-F-V3C COMPLETE"
    )

    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
