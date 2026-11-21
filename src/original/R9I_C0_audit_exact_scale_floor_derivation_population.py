
from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BUILD_SRC = MAIN1 / "build_final_neural_dataset.py"
FINAL_NEURAL_DIR = MAIN1 / "final_neural_trajectory_dataset"
CORR_DIR = MAIN1 / "corrected_trajectory_dataset"

FLOOR_LOCK = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C0_V3_authoritative_scale_floor_lock"
    / "R9H_C0_V3_MANIFEST.json"
)

R9I_B_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_B_exact_model_hyperparameter_and_split_lock"
    / "R9I_B_MANIFEST.json"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_C0_exact_scale_floor_population_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "future_len": 20,
    },
    "2s_to_5s": {
        "future_len": 50,
    },
}

DOWNSAMPLE = 3
Q = 0.05

MATCH_TOLS = [
    1e-15,
    1e-12,
    1e-10,
    1e-8,
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


def verify_locks():
    for p in [
        BUILD_SRC,
        FLOOR_LOCK,
        R9I_B_MANIFEST,
    ]:
        require(p)

    floor_m = json.loads(
        FLOOR_LOCK.read_text(
            encoding="utf-8"
        )
    )

    r9i_b = json.loads(
        R9I_B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_floor = (
        "R9H_C0_V3_AUTHORITATIVE_TASK_SCALE_FLOORS_LOCKED_READY_FOR_R9H_C1_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
    )

    expected_b = (
        "R9I_B_EXACT_MODEL_HYPERPARAMETER_SPLIT_AND_CHECKPOINT_LOCK_COMPLETE_READY_FOR_R9I_C_ISOTROPIC_TRAINREAL_DATASET_BUILD"
    )

    if floor_m.get("status") != expected_floor:
        raise RuntimeError(
            f"Unexpected floor-lock status: {floor_m.get('status')}"
        )

    if r9i_b.get("status") != expected_b:
        raise RuntimeError(
            f"Unexpected R9-I-B status: {r9i_b.get('status')}"
        )

    frozen = {
        str(k): float(v)
        for k, v in floor_m[
            "locked_scale_floors"
        ].items()
    }

    return frozen


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


def extract_scale_floor_source_context():
    text = BUILD_SRC.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()
    selected = set()

    needles = [
        "scale_floor",
        "SCALE_FLOOR",
        "quantile",
        "percentile",
        "path_length",
        "train_indices",
        "validation_indices",
        "neural_split",
    ]

    for i, line in enumerate(
        lines,
        start=1
    ):
        low = line.lower()

        if (
            "scale_floor" in low
            or (
                "quantile" in low
                and "path" in low
            )
            or (
                "percentile" in low
                and "path" in low
            )
        ):
            for j in range(
                max(1, i - 18),
                min(len(lines), i + 24) + 1
            ):
                selected.add(j)

    rows = [
        {
            "line_number": j,
            "line": lines[j - 1],
        }
        for j in sorted(selected)
    ]

    return pd.DataFrame(rows)


def extract_scale_floor_functions():
    text = BUILD_SRC.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(BUILD_SRC)
    )

    rows = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            continue

        start = int(node.lineno)
        end = int(
            getattr(
                node,
                "end_lineno",
                node.lineno
            )
        )

        src = "\n".join(
            lines[
                start - 1:
                end
            ]
        )

        low = src.lower()

        if (
            "scale_floor" not in low
            and "quantile" not in low
            and "percentile" not in low
        ):
            continue

        rows.append(
            {
                "function":
                    node.name,

                "start_line":
                    start,

                "end_line":
                    end,

                "source":
                    "\n".join(
                        f"{i}: {lines[i-1]}"
                        for i in range(
                            start,
                            end + 1
                        )
                    ),
            }
        )

    return pd.DataFrame(rows)


def load_final_metadata(task):
    p = (
        FINAL_NEURAL_DIR
        /
        f"{task}_birdsai_trainval_metadata.csv"
    )

    require(p)

    df = pd.read_csv(
        p,
        dtype={
            "sample_id": str,
            "sequence": str,
            "segment_id": str,
            "track_key": str,
            "neural_split": str,
        }
    )

    return (
        df,
        p,
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
        raise RuntimeError(
            f"{task}: duplicate corrected TrainReal sample_id."
        )

    return (
        core,
        arrays,
        meta_path,
        npz_path,
    )


def get_history_by_sample_ids(
    task,
    sample_ids,
    corrected_arrays,
):
    histories = []

    for sid_str in sample_ids:
        sid = int(
            sid_str
        )

        if not (
            0
            <= sid
            <
            len(
                corrected_arrays[
                    "history"
                ]
            )
        ):
            raise RuntimeError(
                f"{task}: sample_id {sid} outside corrected history range."
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

        if h.shape != (
            20,
            2,
        ):
            raise RuntimeError(
                f"{task}: sample_id={sid}, bad history shape={h.shape}"
            )

        histories.append(
            h
        )

    return np.stack(
        histories,
        axis=0
    )


def compute_lengths(histories):
    return np.array(
        [
            history_path_length(
                h
            )
            for h in histories
        ],
        dtype=np.float64
    )


def quantile_methods(values):
    """
    Compare NumPy quantile methods because an exact frozen value may reflect
    a particular interpolation convention.
    """
    methods = [
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ]

    rows = []

    for method in methods:
        value = float(
            np.quantile(
                values,
                Q,
                method=method
            )
        )

        rows.append(
            (
                method,
                value
            )
        )

    return rows


def best_match_info(value, frozen):
    diff = abs(
        float(value)
        -
        float(frozen)
    )

    tol_match = ""

    for tol in MATCH_TOLS:
        if diff <= tol:
            tol_match = f"{tol:.0e}"
            break

    return (
        diff,
        tol_match,
    )


def add_candidate_rows(
    rows,
    task,
    frozen,
    population_name,
    values,
    population_detail,
):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return

    for method, qvalue in quantile_methods(
        values
    ):
        diff, tol = best_match_info(
            qvalue,
            frozen
        )

        rows.append(
            {
                "task":
                    task,

                "population":
                    population_name,

                "population_detail":
                    population_detail,

                "n":
                    int(
                        len(
                            values
                        )
                    ),

                "quantile":
                    Q,

                "quantile_method":
                    method,

                "candidate_floor":
                    qvalue,

                "frozen_floor":
                    frozen,

                "abs_difference":
                    diff,

                "match_tolerance":
                    tol,

                "mean_path_length":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "median_path_length":
                    float(
                        np.median(
                            values
                        )
                    ),

                "min_path_length":
                    float(
                        np.min(
                            values
                        )
                    ),

                "max_path_length":
                    float(
                        np.max(
                            values
                        )
                    ),
            }
        )


def build_population_audit(
    task,
    frozen,
):
    final_meta, final_meta_path = (
        load_final_metadata(
            task
        )
    )

    (
        corrected_core,
        corrected_arrays,
        corrected_meta_path,
        corrected_npz_path,
    ) = load_corrected_trainreal(
        task
    )

    corrected_lookup = set(
        corrected_core[
            "sample_id"
        ].astype(str)
    )

    final_ids = final_meta[
        "sample_id"
    ].astype(str).tolist()

    missing = [
        sid
        for sid in final_ids
        if sid not in corrected_lookup
    ]

    if missing:
        raise RuntimeError(
            f"{task}: final metadata contains sample_ids not in corrected TrainReal core: "
            f"{missing[:10]}"
        )

    # A) all final trainval rows
    h_final_all = get_history_by_sample_ids(
        task,
        final_ids,
        corrected_arrays
    )

    len_final_all = compute_lengths(
        h_final_all
    )

    # B) exact neural train / validation
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

    val_mask = (
        split
        ==
        "validation"
    )

    # C) all corrected accepted/core TrainReal pool
    core_ids = corrected_core[
        "sample_id"
    ].astype(str).tolist()

    h_core_all = get_history_by_sample_ids(
        task,
        core_ids,
        corrected_arrays
    )

    len_core_all = compute_lengths(
        h_core_all
    )

    rows = []

    add_candidate_rows(
        rows,
        task,
        frozen,
        "final_trainval_all_rows",
        len_final_all,
        "All rows in final neural trainval metadata before neural_split restriction.",
    )

    add_candidate_rows(
        rows,
        task,
        frozen,
        "final_neural_train_rows",
        len_final_all[
            train_mask
        ],
        "Rows with neural_split=train.",
    )

    add_candidate_rows(
        rows,
        task,
        frozen,
        "final_neural_validation_rows",
        len_final_all[
            val_mask
        ],
        "Rows with neural_split=validation.",
    )

    add_candidate_rows(
        rows,
        task,
        frozen,
        "corrected_core_trainreal_all_rows",
        len_core_all,
        "All accepted=1, quality_tier=core, split=train_real rows from corrected metadata.",
    )

    # D) track-level candidate populations from final trainval
    track_col = None

    for c in [
        "track_key",
        "segment_id",
    ]:
        if c in final_meta.columns:
            track_col = c
            break

    if track_col is not None:
        tmp = final_meta[
            [
                track_col,
                "neural_split",
            ]
        ].copy()

        tmp[
            "path_length"
        ] = len_final_all

        track_median_all = (
            tmp.groupby(
                track_col,
                dropna=False
            )[
                "path_length"
            ]
            .median()
            .to_numpy(
                dtype=np.float64
            )
        )

        add_candidate_rows(
            rows,
            task,
            frozen,
            "final_trainval_track_medians_all",
            track_median_all,
            f"One median path length per {track_col} across all final trainval rows.",
        )

        train_tmp = tmp[
            tmp[
                "neural_split"
            ].astype(str)
            ==
            "train"
        ]

        track_median_train = (
            train_tmp.groupby(
                track_col,
                dropna=False
            )[
                "path_length"
            ]
            .median()
            .to_numpy(
                dtype=np.float64
            )
        )

        add_candidate_rows(
            rows,
            task,
            frozen,
            "final_train_track_medians",
            track_median_train,
            f"One median path length per {track_col} within neural_split=train.",
        )

    audit = pd.DataFrame(
        rows
    )

    audit = audit.sort_values(
        [
            "abs_difference",
            "population",
            "quantile_method",
        ],
        kind="mergesort"
    ).reset_index(
        drop=True
    )

    provenance = {
        "task":
            task,

        "final_metadata":
            str(
                final_meta_path
            ),

        "final_metadata_sha256":
            sha256_file(
                final_meta_path
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

        "final_rows":
            int(
                len(
                    final_meta
                )
            ),

        "train_rows":
            int(
                train_mask.sum()
            ),

        "validation_rows":
            int(
                val_mask.sum()
            ),

        "corrected_core_trainreal_rows":
            int(
                len(
                    corrected_core
                )
            ),
    }

    return (
        audit,
        provenance,
    )


def main():
    print(
        "\nR9-I-C0 — EXACT SCALE-FLOOR DERIVATION POPULATION AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. The first R9-I-C attempt safely stopped because "
        "the frozen anisotropic scale floor was NOT reproduced from neural_split=train "
        "rows. This audit does not build a new dataset and does not train any model. "
        "It extracts the exact source context around scale-floor derivation and compares "
        "candidate source populations/interpolation methods against the authoritative "
        "frozen floors.\n"
    )

    frozen = verify_locks()

    source_context = (
        extract_scale_floor_source_context()
    )

    source_functions = (
        extract_scale_floor_functions()
    )

    audit_frames = []
    prov_rows = []

    for task in TASKS:
        audit, prov = build_population_audit(
            task,
            frozen[
                task
            ]
        )

        audit_frames.append(
            audit
        )

        prov_rows.append(
            prov
        )

    audit_df = pd.concat(
        audit_frames,
        ignore_index=True
    )

    provenance_df = pd.DataFrame(
        prov_rows
    )

    exactish = audit_df[
        audit_df[
            "abs_difference"
        ]
        <=
        1e-10
    ].copy()

    best = (
        audit_df
        .sort_values(
            [
                "task",
                "abs_difference",
            ],
            kind="mergesort"
        )
        .groupby(
            "task",
            as_index=False
        )
        .first()
    )

    source_context_path = (
        OUT
        /
        "R9I_C0_BUILD_FINAL_SCALE_FLOOR_SOURCE_CONTEXT.csv"
    )

    source_functions_path = (
        OUT
        /
        "R9I_C0_SCALE_FLOOR_RELEVANT_FUNCTIONS.csv"
    )

    audit_path = (
        OUT
        /
        "R9I_C0_POPULATION_QUANTILE_AUDIT.csv"
    )

    exactish_path = (
        OUT
        /
        "R9I_C0_EXACTISH_MATCHES.csv"
    )

    best_path = (
        OUT
        /
        "R9I_C0_BEST_MATCH_BY_TASK.csv"
    )

    provenance_path = (
        OUT
        /
        "R9I_C0_PROVENANCE.csv"
    )

    source_context.to_csv(
        source_context_path,
        index=False
    )

    source_functions.to_csv(
        source_functions_path,
        index=False
    )

    audit_df.to_csv(
        audit_path,
        index=False
    )

    exactish.to_csv(
        exactish_path,
        index=False
    )

    best.to_csv(
        best_path,
        index=False
    )

    provenance_df.to_csv(
        provenance_path,
        index=False
    )

    all_exactish = bool(
        set(
            exactish[
                "task"
            ].unique()
        )
        ==
        set(
            TASKS.keys()
        )
    )

    if all_exactish:
        status = (
            "R9I_C0_EXACT_SCALE_FLOOR_DERIVATION_POPULATION_RECOVERED_READY_FOR_R9I_C_V2_MATCHED_ISOTROPIC_DATASET_BUILD"
        )
    else:
        status = (
            "R9I_C0_SCALE_FLOOR_POPULATION_AUDIT_COMPLETE_EXACT_DERIVATION_STILL_NEEDS_SOURCE_RESOLUTION"
        )

    manifest = {
        "step":
            "R9I_C0_EXACT_SCALE_FLOOR_DERIVATION_POPULATION_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "frozen_scale_floors":
            frozen,

        "new_dataset_built":
            False,

        "new_training":
            False,

        "model_inference":
            False,

        "frozen_results_modified":
            False,

        "reason_for_audit":
            (
                "R9-I-C attempt derived 2s_to_2s q05=0.06784638218230425 "
                "from neural_split=train, which did not match frozen "
                "0.06820264416877736. Therefore the derivation population "
                "must be recovered exactly before isotropic floor derivation."
            ),

        "outputs": {
            "source_context":
                str(
                    source_context_path
                ),

            "relevant_functions":
                str(
                    source_functions_path
                ),

            "population_quantile_audit":
                str(
                    audit_path
                ),

            "exactish_matches":
                str(
                    exactish_path
                ),

            "best_match_by_task":
                str(
                    best_path
                ),

            "provenance":
                str(
                    provenance_path
                ),
        },
    }

    (
        OUT
        /
        "R9I_C0_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "BUILD_FINAL SCALE-FLOOR SOURCE CONTEXT:"
    )

    print(
        source_context.to_string(
            index=False
        )
    )

    print(
        "\nSCALE-FLOOR RELEVANT FUNCTIONS:"
    )

    if len(
        source_functions
    ):
        print(
            source_functions.to_string(
                index=False
            )
        )
    else:
        print(
            "No top-level function containing scale_floor/quantile was isolated."
        )

    print(
        "\nBEST POPULATION / QUANTILE MATCH BY TASK:"
    )

    print(
        best[
            [
                "task",
                "population",
                "n",
                "quantile_method",
                "candidate_floor",
                "frozen_floor",
                "abs_difference",
                "match_tolerance",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nEXACTISH MATCHES (abs diff <= 1e-10):"
    )

    if len(
        exactish
    ):
        print(
            exactish[
                [
                    "task",
                    "population",
                    "n",
                    "quantile_method",
                    "candidate_floor",
                    "frozen_floor",
                    "abs_difference",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "None."
        )

    print(
        "\nALL POPULATION CANDIDATES:"
    )

    print(
        audit_df[
            [
                "task",
                "population",
                "n",
                "quantile_method",
                "candidate_floor",
                "frozen_floor",
                "abs_difference",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-I-C0 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
