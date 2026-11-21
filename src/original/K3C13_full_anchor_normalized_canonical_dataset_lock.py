from pathlib import Path
import ast
import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

ORIGINAL_SOURCE = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result\build_final_neural_dataset.py"
)

EXPECTED_SOURCE_SHA256 = (
    "7156eeef6f91f6affe7067adc37fa8d20aafc34d82d53155c47019085aeb25e4"
)

PRELOCK = (
    BASE
    / "k3_operational_warning"
    / "step7_external_prelock"
    / "STEP6D_K3C10_BAMBI_STEP7_PRELOCK.json"
)

EXPECTED_PRELOCK_SHA256 = (
    "8cc687c3f91ecbfb2a3c0f491427d91ccfdca959659a52e0836a98889c890bd1"
)

K3C11_DIR = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_reconstruction_v1"
)

K3C11_SUMMARY = (
    K3C11_DIR
    / "STEP6D_K3C11_SUMMARY.json"
)

K3C12_SUMMARY = (
    BASE
    / "k3_operational_warning"
    / "exact_canonicalization_adapter_audit"
    / "STEP6D_K3C12_SUMMARY.json"
)

K2B3_DIR = (
    BASE
    / "k2b3_final_canonical"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_canonical_lock_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


def require(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def extract_safe_namespace(
    source_path,
    target_function="canonicalize_sample",
):
    text = source_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(
        text
    )

    safe_body = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.ClassDef,
            ),
        ):
            safe_body.append(
                node
            )
            continue

        if isinstance(
            node,
            ast.Assign,
        ):
            try:
                ast.literal_eval(
                    node.value
                )
            except Exception:
                continue

            safe_body.append(
                node
            )
            continue

        if isinstance(
            node,
            ast.AnnAssign,
        ):
            if node.value is None:
                continue

            try:
                ast.literal_eval(
                    node.value
                )
            except Exception:
                continue

            safe_body.append(
                node
            )

    module = ast.Module(
        body=safe_body,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module
    )

    ns = {
        "__name__":
            "_k3c13_exact_canonicalization_",

        "np":
            np,

        "numpy":
            np,

        "math":
            math,
    }

    exec(
        compile(
            module,
            str(
                source_path
            ),
            "exec",
        ),
        ns,
        ns,
    )

    if target_function not in ns:
        raise RuntimeError(
            f"Exact source namespace does not contain {target_function}"
        )

    return ns


def locked_scale_floor(task):
    matches = sorted(
        p
        for p in K2B3_DIR.glob(
            "*metadata_LOCKED.csv"
        )
        if task.lower()
        in
        p.name.lower()
    )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Expected exactly one locked K2B3 metadata CSV for {task}:\n"
            +
            "\n".join(
                str(p)
                for p in matches
            )
        )

    df = pd.read_csv(
        matches[
            0
        ],
        usecols=[
            "scale_floor",
        ],
    )

    values = pd.to_numeric(
        df[
            "scale_floor"
        ],
        errors="coerce",
    ).dropna().unique()

    if len(
        values
    ) != 1:
        raise RuntimeError(
            f"{task}: scale_floor is not unique: {values}"
        )

    return (
        matches[
            0
        ],
        float(
            values[
                0
            ]
        ),
    )


def verify_k3c11_input(
    task,
):
    npz_path = (
        K3C11_DIR
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_windows.npz"
    )

    metadata_path = (
        K3C11_DIR
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_metadata.csv"
    )

    require(
        npz_path
    )

    require(
        metadata_path
    )

    arrays = np.load(
        npz_path,
        allow_pickle=False,
    )

    required_keys = {
        "history_anchor_norm",
        "future_anchor_norm",
        "history_world_xyz",
        "future_world_xyz",
    }

    missing = (
        required_keys
        -
        set(
            arrays.files
        )
    )

    if missing:
        raise RuntimeError(
            f"{task}: K3C11 NPZ missing arrays: {sorted(missing)}"
        )

    history = np.asarray(
        arrays[
            "history_anchor_norm"
        ],
        dtype=np.float64,
    )

    future = np.asarray(
        arrays[
            "future_anchor_norm"
        ],
        dtype=np.float64,
    )

    metadata = pd.read_csv(
        metadata_path
    )

    if len(
        metadata
    ) != history.shape[
        0
    ]:
        raise RuntimeError(
            f"{task}: metadata rows {len(metadata)} != history samples {history.shape[0]}"
        )

    if future.shape[
        0
    ] != history.shape[
        0
    ]:
        raise RuntimeError(
            f"{task}: history/future sample count mismatch."
        )

    if history.shape[
        1:
    ] != (
        20,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected history shape {history.shape}"
        )

    expected_future_steps = (
        20
        if task
        ==
        "2s_to_2s"
        else
        50
    )

    if future.shape[
        1:
    ] != (
        expected_future_steps,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected future shape {future.shape}"
        )

    if not (
        np.isfinite(
            history
        ).all()
        and
        np.isfinite(
            future
        ).all()
    ):
        raise RuntimeError(
            f"{task}: K3C11 normalized arrays contain nonfinite values."
        )

    required_meta = {
        "sample_index",
        "track_id",
        "window_start_time_sec",
        "anchor_frame",
        "anchor_time_offset_sec",
    }

    missing_meta = (
        required_meta
        -
        set(
            metadata.columns
        )
    )

    if missing_meta:
        raise RuntimeError(
            f"{task}: K3C11 metadata missing columns: {sorted(missing_meta)}"
        )

    sample_index = pd.to_numeric(
        metadata[
            "sample_index"
        ],
        errors="raise",
    ).astype(
        int
    ).to_numpy()

    if not np.array_equal(
        sample_index,
        np.arange(
            len(
                metadata
            ),
            dtype=int,
        ),
    ):
        raise RuntimeError(
            f"{task}: sample_index is not exactly 0..N-1."
        )

    return {
        "npz_path":
            npz_path,

        "metadata_path":
            metadata_path,

        "history":
            history,

        "future":
            future,

        "metadata":
            metadata,
    }


def canonicalize_task(
    task,
    fn,
):
    source = verify_k3c11_input(
        task
    )

    floor_path, scale_floor = locked_scale_floor(
        task
    )

    history = source[
        "history"
    ]

    future = source[
        "future"
    ]

    n = history.shape[
        0
    ]

    h_can = np.empty(
        history.shape,
        dtype=np.float32,
    )

    f_can = np.empty(
        future.shape,
        dtype=np.float32,
    )

    origin = np.empty(
        (
            n,
            2,
        ),
        dtype=np.float32,
    )

    angle = np.empty(
        n,
        dtype=np.float32,
    )

    scale = np.empty(
        n,
        dtype=np.float32,
    )

    raw_path_length = np.empty(
        n,
        dtype=np.float32,
    )

    for i in range(
        n
    ):
        result = fn(
            history[
                i
            ],
            future[
                i
            ],
            scale_floor,
        )

        required = {
            "history_canonical",
            "future_canonical",
            "origin",
            "angle",
            "scale",
            "raw_path_length",
        }

        missing = (
            required
            -
            set(
                result.keys()
            )
        )

        if missing:
            raise RuntimeError(
                f"{task} sample {i}: canonicalize_sample missing keys {sorted(missing)}"
            )

        h = np.asarray(
            result[
                "history_canonical"
            ]
        )

        f = np.asarray(
            result[
                "future_canonical"
            ]
        )

        o = np.asarray(
            result[
                "origin"
            ]
        )

        a = np.asarray(
            result[
                "angle"
            ]
        )

        s = np.asarray(
            result[
                "scale"
            ]
        )

        r = np.asarray(
            result[
                "raw_path_length"
            ]
        )

        if h.shape != (
            20,
            2,
        ):
            raise RuntimeError(
                f"{task} sample {i}: bad canonical history shape {h.shape}"
            )

        if f.shape != future.shape[
            1:
        ]:
            raise RuntimeError(
                f"{task} sample {i}: bad canonical future shape {f.shape}"
            )

        if o.shape != (
            2,
        ):
            raise RuntimeError(
                f"{task} sample {i}: bad origin shape {o.shape}"
            )

        values_to_check = [
            h,
            f,
            o,
            np.atleast_1d(
                a
            ),
            np.atleast_1d(
                s
            ),
            np.atleast_1d(
                r
            ),
        ]

        if not all(
            np.isfinite(
                x
            ).all()
            for x in values_to_check
        ):
            raise RuntimeError(
                f"{task} sample {i}: nonfinite canonicalization result."
            )

        h_can[
            i
        ] = h.astype(
            np.float32,
            copy=False,
        )

        f_can[
            i
        ] = f.astype(
            np.float32,
            copy=False,
        )

        origin[
            i
        ] = o.astype(
            np.float32,
            copy=False,
        )

        angle[
            i
        ] = np.float32(
            a
        )

        scale[
            i
        ] = np.float32(
            s
        )

        raw_path_length[
            i
        ] = np.float32(
            r
        )

    floor_applied = (
        raw_path_length
        <
        np.float32(
            scale_floor
        )
    )

    if np.any(
        scale
        <
        np.float32(
            scale_floor
        )
        -
        np.float32(
            1e-7
        )
    ):
        raise RuntimeError(
            f"{task}: at least one scale is below the locked scale floor."
        )

    out_npz = (
        OUT
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_canonical_LOCKED.npz"
    )

    np.savez_compressed(
        out_npz,
        history_canonical=h_can,
        future_canonical=f_can,
        origin=origin,
        angle=angle,
        scale=scale,
        raw_path_length=raw_path_length,
        scale_floor=np.asarray(
            scale_floor,
            dtype=np.float32,
        ),
    )

    metadata = source[
        "metadata"
    ].copy()

    metadata[
        "canonical_origin_x"
    ] = origin[
        :,
        0
    ]

    metadata[
        "canonical_origin_y"
    ] = origin[
        :,
        1
    ]

    metadata[
        "canonical_angle_rad"
    ] = angle

    metadata[
        "canonical_scale"
    ] = scale

    metadata[
        "canonical_raw_path_length"
    ] = raw_path_length

    metadata[
        "canonical_scale_floor"
    ] = float(
        scale_floor
    )

    metadata[
        "canonical_scale_floor_applied"
    ] = floor_applied

    out_metadata = (
        OUT
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_canonical_metadata_LOCKED.csv"
    )

    metadata.to_csv(
        out_metadata,
        index=False,
    )

    summary = {
        "task":
            task,

        "samples":
            int(
                n
            ),

        "tracks":
            int(
                metadata[
                    "track_id"
                ].nunique()
            ),

        "history_shape":
            list(
                h_can.shape
            ),

        "future_shape":
            list(
                f_can.shape
            ),

        "history_dtype":
            str(
                h_can.dtype
            ),

        "future_dtype":
            str(
                f_can.dtype
            ),

        "origin_dtype":
            str(
                origin.dtype
            ),

        "angle_dtype":
            str(
                angle.dtype
            ),

        "scale_dtype":
            str(
                scale.dtype
            ),

        "scale_floor":
            float(
                scale_floor
            ),

        "scale_floor_source":
            str(
                floor_path
            ),

        "scale_floor_source_sha256":
            sha256_file(
                floor_path
            ),

        "scale_floor_applied_samples":
            int(
                floor_applied.sum()
            ),

        "raw_path_length_min":
            float(
                raw_path_length.min()
            ),

        "raw_path_length_median":
            float(
                np.median(
                    raw_path_length
                )
            ),

        "raw_path_length_max":
            float(
                raw_path_length.max()
            ),

        "scale_min":
            float(
                scale.min()
            ),

        "scale_median":
            float(
                np.median(
                    scale
                )
            ),

        "scale_max":
            float(
                scale.max()
            ),

        "canonical_history_abs_max":
            float(
                np.abs(
                    h_can
                ).max()
            ),

        "canonical_future_abs_max":
            float(
                np.abs(
                    f_can
                ).max()
            ),

        "source_k3c11_npz":
            str(
                source[
                    "npz_path"
                ]
            ),

        "source_k3c11_npz_sha256":
            sha256_file(
                source[
                    "npz_path"
                ]
            ),

        "source_k3c11_metadata":
            str(
                source[
                    "metadata_path"
                ]
            ),

        "source_k3c11_metadata_sha256":
            sha256_file(
                source[
                    "metadata_path"
                ]
            ),

        "locked_npz":
            str(
                out_npz
            ),

        "locked_npz_sha256":
            sha256_file(
                out_npz
            ),

        "locked_metadata":
            str(
                out_metadata
            ),

        "locked_metadata_sha256":
            sha256_file(
                out_metadata
            ),
    }

    return summary


def main():
    print(
        "\nSTEP 6D-K3C13 — FULL ANCHOR-NORMALIZED CANONICAL DATASET + HASH LOCK"
    )

    print(
        "Uses the exact original canonicalize_sample() source recovered in K3C12."
    )

    print(
        "NO Transformer inference, NO q recalibration, NO geofence metric, and NO Step7 alarm is generated.\n"
    )

    for path in [
        ORIGINAL_SOURCE,
        PRELOCK,
        K3C11_SUMMARY,
        K3C12_SUMMARY,
    ]:
        require(
            path
        )

    actual_source_hash = sha256_file(
        ORIGINAL_SOURCE
    )

    if actual_source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Original canonicalization source hash mismatch. "
            f"Expected {EXPECTED_SOURCE_SHA256}, got {actual_source_hash}"
        )

    actual_prelock_hash = sha256_file(
        PRELOCK
    )

    if actual_prelock_hash != EXPECTED_PRELOCK_SHA256:
        raise RuntimeError(
            "K3C10 prelock hash mismatch. "
            f"Expected {EXPECTED_PRELOCK_SHA256}, got {actual_prelock_hash}"
        )

    k3c11 = json.loads(
        K3C11_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    if not str(
        k3c11.get(
            "status",
            ""
        )
    ).startswith(
        "ANCHOR_NORMALIZED_RECONSTRUCTION_COMPLETE"
    ):
        raise RuntimeError(
            f"K3C11 is not in a completed state: {k3c11.get('status')}"
        )

    k3c12 = json.loads(
        K3C12_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    if (
        k3c12.get(
            "status"
        )
        !=
        "EXACT_CANONICALIZE_SAMPLE_SOURCE_RECOVERED_AND_DRY_RUN_SUCCESS_READY_FOR_DERIVED_DATASET_LOCK"
    ):
        raise RuntimeError(
            f"K3C12 status unexpected: {k3c12.get('status')}"
        )

    if (
        Path(
            k3c12.get(
                "selected_source",
                ""
            )
        )
        !=
        ORIGINAL_SOURCE
    ):
        raise RuntimeError(
            "K3C12 selected source is not the expected original source path."
        )

    if (
        k3c12.get(
            "selected_source_sha256"
        )
        !=
        EXPECTED_SOURCE_SHA256
    ):
        raise RuntimeError(
            "K3C12 selected source hash does not match the expected original source."
        )

    ns = extract_safe_namespace(
        ORIGINAL_SOURCE
    )

    canonicalize_sample = ns[
        "canonicalize_sample"
    ]

    print(
        "EXACT CANONICALIZATION SOURCE:"
    )
    print(
        ORIGINAL_SOURCE
    )

    print(
        "SOURCE SHA256:"
    )
    print(
        actual_source_hash
    )

    print(
        "\nVERIFIED K3C10 PRELOCK SHA256:"
    )
    print(
        actual_prelock_hash
    )

    task_summaries = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        summary = canonicalize_task(
            task,
            canonicalize_sample,
        )

        task_summaries.append(
            summary
        )

        print(
            f"\n{task} LOCK SUMMARY:"
        )

        print(
            json.dumps(
                summary,
                indent=2
            )
        )

    manifest = {
        "step":
            "STEP_6D_K3C13_FULL_ANCHOR_NORMALIZED_CANONICAL_DATASET_HASH_LOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "DERIVED_ANCHOR_NORMALIZED_CANONICAL_DATASET_LOCKED_BEFORE_SECONDARY_FROZEN_INFERENCE",

        "analysis_role":
            (
                "Secondary frozen operational-transfer preparation using BAMBI Flight 6. "
                "This is not a new untouched confirmatory uncertainty holdout."
            ),

        "exact_canonicalization_source":
            str(
                ORIGINAL_SOURCE
            ),

        "exact_canonicalization_source_sha256":
            actual_source_hash,

        "k3c10_prelock":
            str(
                PRELOCK
            ),

        "k3c10_prelock_sha256":
            actual_prelock_hash,

        "k3c11_summary":
            str(
                K3C11_SUMMARY
            ),

        "k3c11_summary_sha256":
            sha256_file(
                K3C11_SUMMARY
            ),

        "k3c12_summary":
            str(
                K3C12_SUMMARY
            ),

        "k3c12_summary_sha256":
            sha256_file(
                K3C12_SUMMARY
            ),

        "tasks":
            task_summaries,

        "immutable_next_step_constraints": {
            "model_retraining":
                False,

            "model_substitution":
                False,

            "bambi_q_recalibration":
                False,

            "bambi_threshold_tuning":
                False,

            "persistence_tuning":
                False,

            "response_threshold_tuning":
                False,

            "primary_conformal_level":
                0.90,

            "persistence_windows":
                2,

            "step7_development_rule":
                "frozen after Step7H",
        },

        "nonclaims": [
            "No secondary BAMBI Transformer inference has been run in this step.",
            "No Step7 BAMBI operational metric or alarm has been generated in this step.",
            "The 50 degree BAMBI FOVY remains an assumed/default reference-pipeline value.",
            "Virtual geofences remain normalized synthetic boundaries rather than measured field boundaries.",
        ],
    }

    manifest_path = (
        OUT
        / "STEP6D_K3C13_CANONICAL_DATASET_LOCK.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8",
    )

    receipt = {
        "manifest":
            str(
                manifest_path
            ),

        "manifest_sha256":
            sha256_file(
                manifest_path
            ),

        "status":
            "K3C13_CANONICAL_DATASET_LOCK_COMPLETE_NO_SECONDARY_INFERENCE_RUN",
    }

    receipt_path = (
        OUT
        / "STEP6D_K3C13_LOCK_RECEIPT.json"
    )

    receipt_path.write_text(
        json.dumps(
            receipt,
            indent=2
        ),
        encoding="utf-8",
    )

    print(
        "\nMANIFEST:"
    )
    print(
        manifest_path
    )

    print(
        "MANIFEST SHA256:"
    )
    print(
        receipt[
            "manifest_sha256"
        ]
    )

    print(
        "\nSTATUS:"
    )
    print(
        "K3C13_CANONICAL_DATASET_LOCK_COMPLETE_NO_SECONDARY_INFERENCE_RUN"
    )

    print(
        "\nSTEP 6D-K3C13 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
