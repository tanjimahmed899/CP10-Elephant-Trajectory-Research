
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R10_ROOT = REVISION_ROOT / "R10_ensemble_analysis"

R10A_DIR = (
    R10_ROOT
    / "R10A_frozen_transformer_ensemble_evidence_audit"
)

R10A_MANIFEST = R10A_DIR / "R10A_MANIFEST.json"
R10A_PRED_INVENTORY = (
    R10A_DIR
    / "R10A_FROZEN_TRANSFORMER_PREDICTION_INVENTORY.csv"
)

TESTREAL_DIR = (
    MAIN1
    / "final_neural_trajectory_dataset"
)

GEOMETRY_CSV = (
    REVISION_ROOT
    / "R9_canonicalization_isotropic_ablation"
    / "R9G_V3B_corrected_isotropic_rebuild"
    / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"
)

OUT = (
    R10_ROOT
    / "R10B_testreal_transformer_ensemble_value"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

TASKS = {
    "2s_to_2s": {
        "samples": 1053,
        "segments": 89,
        "sequences": 6,
    },
    "2s_to_5s": {
        "samples": 826,
        "segments": 63,
        "sequences": 5,
    },
}

SEEDS = [42, 52, 62]

BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20261029


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


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


def verify_r10a():
    require(
        R10A_MANIFEST
    )

    require(
        R10A_PRED_INVENTORY
    )

    require(
        GEOMETRY_CSV
    )

    m = json.loads(
        R10A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R10A_PRIMARY_TESTREAL_READY_EXISTING_WILDBOX_NOT_FRESH_READY_FOR_R10B_TESTREAL_AND_R10C_FRESH_WILDBOX_REINFERENCE"
    )

    if m.get(
        "status"
    ) != expected:
        raise RuntimeError(
            f"Unexpected R10-A status: {m.get('status')}"
        )

    if not bool(
        m.get(
            "testreal_ready",
            False
        )
    ):
        raise RuntimeError(
            "R10-A did not lock TestReal as ready."
        )

    return m


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
    }

    missing = (
        required
        -
        set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Geometry lock missing columns: {sorted(missing)}"
        )

    return {
        str(
            row[
                "sequence"
            ]
        ):
            {
                "W":
                    float(
                        row[
                            "width_px"
                        ]
                    ),

                "H":
                    float(
                        row[
                            "height_px"
                        ]
                    ),
            }
        for _, row in df.iterrows()
    }


def load_testreal_metadata(task):
    p = (
        TESTREAL_DIR
        /
        f"{task}_birdsai_test_metadata.csv"
    )

    require(
        p
    )

    df = pd.read_csv(
        p,
        dtype={
            "sample_id": str,
            "sequence": str,
            "source_group": str,
            "segment_id": str,
            "track_key": str,
        }
    )

    expected = TASKS[
        task
    ][
        "samples"
    ]

    if len(
        df
    ) != expected:
        raise RuntimeError(
            f"{task}: metadata rows={len(df)}, expected={expected}"
        )

    if "segment_id" not in df.columns:
        raise RuntimeError(
            f"{task}: segment_id missing from metadata."
        )

    if "track_key" not in df.columns:
        raise RuntimeError(
            f"{task}: track_key missing from metadata."
        )

    return (
        df,
        p,
    )


def source_sequence_series(meta):
    if "sequence" in meta.columns:
        return meta[
            "sequence"
        ].astype(str)

    if "source_group" in meta.columns:
        return meta[
            "source_group"
        ].astype(str)

    raise RuntimeError(
        "Metadata has neither sequence nor source_group."
    )


def load_r10a_testreal_inventory():
    df = pd.read_csv(
        R10A_PRED_INVENTORY,
        dtype={
            "task": str,
            "dataset_label": str,
            "prediction_path": str,
            "prediction_sha256": str,
            "checkpoint_path": str,
            "checkpoint_sha256": str,
        }
    )

    df = df[
        df[
            "dataset_label"
        ]
        ==
        "BIRDSAI_test"
    ].copy()

    if len(
        df
    ) != 6:
        raise RuntimeError(
            f"Expected 6 TestReal Transformer prediction rows, found {len(df)}."
        )

    mapping = {}

    for _, row in df.iterrows():
        key = (
            str(
                row[
                    "task"
                ]
            ),
            int(
                row[
                    "seed"
                ]
            ),
        )

        pred = Path(
            str(
                row[
                    "prediction_path"
                ]
            )
        )

        ckpt = Path(
            str(
                row[
                    "checkpoint_path"
                ]
            )
        )

        require(
            pred
        )

        require(
            ckpt
        )

        if sha256_file(
            pred
        ) != str(
            row[
                "prediction_sha256"
            ]
        ):
            raise RuntimeError(
                f"Prediction hash changed: {pred}"
            )

        if sha256_file(
            ckpt
        ) != str(
            row[
                "checkpoint_sha256"
            ]
        ):
            raise RuntimeError(
                f"Checkpoint hash changed: {ckpt}"
            )

        mapping[
            key
        ] = {
            "prediction_path":
                pred,

            "prediction_sha256":
                str(
                    row[
                        "prediction_sha256"
                    ]
                ),

            "checkpoint_path":
                ckpt,

            "checkpoint_sha256":
                str(
                    row[
                        "checkpoint_sha256"
                    ]
                ),
        }

    if len(
        mapping
    ) != 6:
        raise RuntimeError(
            "TestReal prediction identity map is not exactly 6."
        )

    return mapping


def load_checkpoint_validation_metric(path):
    import torch

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "task",
        "seed",
        "best_epoch",
        "best_validation_track_macro_ADE",
    }

    missing = (
        required
        -
        set(
            ckpt.keys()
        )
    )

    if missing:
        raise RuntimeError(
            f"{path.name}: checkpoint missing {sorted(missing)}"
        )

    return {
        "task":
            str(
                ckpt[
                    "task"
                ]
            ),

        "seed":
            int(
                ckpt[
                    "seed"
                ]
            ),

        "best_epoch":
            int(
                ckpt[
                    "best_epoch"
                ]
            ),

        "best_validation_track_macro_ADE":
            float(
                ckpt[
                    "best_validation_track_macro_ADE"
                ]
            ),
    }


def load_prediction(path):
    arrays = np.load(
        path,
        allow_pickle=False
    )

    required = {
        "truth_abs",
        "prediction_abs",
    }

    missing = (
        required
        -
        set(
            arrays.files
        )
    )

    if missing:
        raise RuntimeError(
            f"{path.name}: missing keys {sorted(missing)}"
        )

    truth = np.asarray(
        arrays[
            "truth_abs"
        ],
        dtype=np.float64
    )

    pred = np.asarray(
        arrays[
            "prediction_abs"
        ],
        dtype=np.float64
    )

    return truth, pred


def pixel_errors(
    truth,
    pred,
    meta,
    geometry,
):
    if truth.shape != pred.shape:
        raise RuntimeError(
            f"truth/pred shape mismatch: {truth.shape} vs {pred.shape}"
        )

    seqs = source_sequence_series(
        meta
    )

    missing = sorted(
        set(
            seqs
        )
        -
        set(
            geometry
        )
    )

    if missing:
        raise RuntimeError(
            f"Geometry missing sequences: {missing}"
        )

    W = np.asarray(
        [
            geometry[
                seq
            ][
                "W"
            ]
            for seq in seqs
        ],
        dtype=np.float64
    )

    H = np.asarray(
        [
            geometry[
                seq
            ][
                "H"
            ]
            for seq in seqs
        ],
        dtype=np.float64
    )

    delta = (
        pred
        -
        truth
    )

    dx = (
        delta[
            :,
            :,
            0
        ]
        *
        W[
            :,
            None
        ]
    )

    dy = (
        delta[
            :,
            :,
            1
        ]
        *
        H[
            :,
            None
        ]
    )

    err = np.sqrt(
        dx ** 2
        +
        dy ** 2
    )

    return (
        err.mean(
            axis=1
        ),
        err[
            :,
            -1
        ],
    )


def prediction_disagreement_px(
    pred_a,
    pred_b,
    meta,
    geometry,
):
    seqs = source_sequence_series(
        meta
    )

    W = np.asarray(
        [
            geometry[
                seq
            ][
                "W"
            ]
            for seq in seqs
        ],
        dtype=np.float64
    )

    H = np.asarray(
        [
            geometry[
                seq
            ][
                "H"
            ]
            for seq in seqs
        ],
        dtype=np.float64
    )

    delta = (
        pred_a
        -
        pred_b
    )

    dx = (
        delta[
            :,
            :,
            0
        ]
        *
        W[
            :,
            None
        ]
    )

    dy = (
        delta[
            :,
            :,
            1
        ]
        *
        H[
            :,
            None
        ]
    )

    dist = np.sqrt(
        dx ** 2
        +
        dy ** 2
    )

    return (
        dist.mean(
            axis=1
        ),
        dist[
            :,
            -1
        ],
    )


def cluster_cols(meta):
    if "sequence" in meta.columns:
        source_col = "sequence"
    elif "source_group" in meta.columns:
        source_col = "source_group"
    else:
        raise RuntimeError(
            "No sequence/source_group for clustering."
        )

    return [
        source_col,
        "segment_id",
    ]


def percentile_ci(values):
    return (
        float(
            np.percentile(
                values,
                2.5
            )
        ),
        float(
            np.percentile(
                values,
                97.5
            )
        ),
    )


def cluster_bootstrap(
    sample_df,
    value_col,
    rng_seed,
):
    cols = cluster_cols(
        sample_df
    )

    segment_df = (
        sample_df.groupby(
            cols,
            dropna=False,
            as_index=False
        )
        .agg(
            segment_effect=(
                value_col,
                "median"
            ),

            windows=(
                "sample_id",
                "size"
            ),
        )
    )

    n = len(
        segment_df
    )

    values = segment_df[
        "segment_effect"
    ].to_numpy(
        dtype=np.float64
    )

    point = float(
        values.mean()
    )

    rng = np.random.default_rng(
        rng_seed
    )

    boot = np.empty(
        BOOTSTRAP_B,
        dtype=np.float64
    )

    for b in range(
        BOOTSTRAP_B
    ):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        boot[
            b
        ] = float(
            values[
                idx
            ].mean()
        )

    low, high = percentile_ci(
        boot
    )

    if high < 0:
        direction = (
            "ENSEMBLE_BETTER_CI_EXCLUDES_ZERO"
        )
    elif low > 0:
        direction = (
            "ENSEMBLE_WORSE_CI_EXCLUDES_ZERO"
        )
    else:
        direction = (
            "NO_CLEAR_DIFFERENCE_CI_INCLUDES_ZERO"
        )

    return {
        "segments":
            int(
                n
            ),

        "point":
            point,

        "ci95_low":
            low,

        "ci95_high":
            high,

        "direction":
            direction,
    }


def spearman_via_ranks(x, y):
    sx = pd.Series(
        x
    )

    sy = pd.Series(
        y
    )

    if (
        sx.nunique(
            dropna=True
        )
        <=
        1
        or
        sy.nunique(
            dropna=True
        )
        <=
        1
    ):
        return np.nan

    return float(
        sx.rank(
            method="average"
        ).corr(
            sy.rank(
                method="average"
            ),
            method="pearson"
        )
    )


def analyze_task(
    task,
    inventory,
    geometry,
):
    meta, meta_path = load_testreal_metadata(
        task
    )

    seed_preds = {}
    seed_truth = {}
    seed_metrics = []
    checkpoint_rows = []

    for seed in SEEDS:
        info = inventory[
            (
                task,
                seed,
            )
        ]

        truth, pred = load_prediction(
            info[
                "prediction_path"
            ]
        )

        expected = TASKS[
            task
        ][
            "samples"
        ]

        if truth.shape[
            0
        ] != expected:
            raise RuntimeError(
                f"{task}/seed{seed}: samples={truth.shape[0]}, expected={expected}"
            )

        seed_preds[
            seed
        ] = pred

        seed_truth[
            seed
        ] = truth

        ade, fde = pixel_errors(
            truth,
            pred,
            meta,
            geometry,
        )

        seed_metrics.append(
            {
                "task":
                    task,

                "seed":
                    seed,

                "ADE_px_window_mean":
                    float(
                        ade.mean()
                    ),

                "FDE_px_window_mean":
                    float(
                        fde.mean()
                    ),
            }
        )

        ck = load_checkpoint_validation_metric(
            info[
                "checkpoint_path"
            ]
        )

        if (
            ck[
                "task"
            ]
            !=
            task
            or
            ck[
                "seed"
            ]
            !=
            seed
        ):
            raise RuntimeError(
                f"{task}/seed{seed}: checkpoint identity mismatch."
            )

        checkpoint_rows.append(
            {
                "task":
                    task,

                "seed":
                    seed,

                "best_epoch":
                    ck[
                        "best_epoch"
                    ],

                "best_validation_track_macro_ADE":
                    ck[
                        "best_validation_track_macro_ADE"
                    ],

                "checkpoint_path":
                    str(
                        info[
                            "checkpoint_path"
                        ]
                    ),

                "checkpoint_sha256":
                    info[
                        "checkpoint_sha256"
                    ],
            }
        )

    truth0 = seed_truth[
        SEEDS[
            0
        ]
    ]

    for seed in SEEDS[
        1:
    ]:
        if not np.array_equal(
            seed_truth[
                seed
            ],
            truth0
        ):
            raise RuntimeError(
                f"{task}: truth differs across seeds."
            )

    validation_best_seed = min(
        checkpoint_rows,
        key=lambda r:
            r[
                "best_validation_track_macro_ADE"
            ],
    )[
        "seed"
    ]

    ensemble_pred = np.mean(
        np.stack(
            [
                seed_preds[
                    seed
                ]
                for seed in SEEDS
            ],
            axis=0
        ),
        axis=0
    )

    ens_ade, ens_fde = pixel_errors(
        truth0,
        ensemble_pred,
        meta,
        geometry,
    )

    per_seed_ade = {}
    per_seed_fde = {}

    for seed in SEEDS:
        ade, fde = pixel_errors(
            truth0,
            seed_preds[
                seed
            ],
            meta,
            geometry,
        )

        per_seed_ade[
            seed
        ] = ade

        per_seed_fde[
            seed
        ] = fde

    mean_seed_ade = np.mean(
        np.stack(
            [
                per_seed_ade[
                    seed
                ]
                for seed in SEEDS
            ],
            axis=0
        ),
        axis=0
    )

    mean_seed_fde = np.mean(
        np.stack(
            [
                per_seed_fde[
                    seed
                ]
                for seed in SEEDS
            ],
            axis=0
        ),
        axis=0
    )

    val_seed_ade = per_seed_ade[
        validation_best_seed
    ]

    val_seed_fde = per_seed_fde[
        validation_best_seed
    ]

    sample = meta.copy()

    sample[
        "task"
    ] = task

    sample[
        "ensemble_ADE_px"
    ] = ens_ade

    sample[
        "ensemble_FDE_px"
    ] = ens_fde

    sample[
        "mean_individual_seed_ADE_px"
    ] = mean_seed_ade

    sample[
        "mean_individual_seed_FDE_px"
    ] = mean_seed_fde

    sample[
        "validation_selected_seed"
    ] = validation_best_seed

    sample[
        "validation_selected_seed_ADE_px"
    ] = val_seed_ade

    sample[
        "validation_selected_seed_FDE_px"
    ] = val_seed_fde

    sample[
        "delta_ensemble_minus_mean_seed_ADE_px"
    ] = (
        ens_ade
        -
        mean_seed_ade
    )

    sample[
        "delta_ensemble_minus_mean_seed_FDE_px"
    ] = (
        ens_fde
        -
        mean_seed_fde
    )

    sample[
        "delta_ensemble_minus_validation_seed_ADE_px"
    ] = (
        ens_ade
        -
        val_seed_ade
    )

    sample[
        "delta_ensemble_minus_validation_seed_FDE_px"
    ] = (
        ens_fde
        -
        val_seed_fde
    )

    for seed in SEEDS:
        sample[
            f"seed{seed}_ADE_px"
        ] = per_seed_ade[
            seed
        ]

        sample[
            f"seed{seed}_FDE_px"
        ] = per_seed_fde[
            seed
        ]

        sample[
            f"delta_ensemble_minus_seed{seed}_ADE_px"
        ] = (
            ens_ade
            -
            per_seed_ade[
                seed
            ]
        )

        sample[
            f"delta_ensemble_minus_seed{seed}_FDE_px"
        ] = (
            ens_fde
            -
            per_seed_fde[
                seed
            ]
        )

    pairwise_ades = []
    pairwise_fdes = []

    diversity_rows = []

    pairs = [
        (
            42,
            52,
        ),
        (
            42,
            62,
        ),
        (
            52,
            62,
        ),
    ]

    for a, b in pairs:
        d_ade, d_fde = prediction_disagreement_px(
            seed_preds[
                a
            ],
            seed_preds[
                b
            ],
            meta,
            geometry,
        )

        pairwise_ades.append(
            d_ade
        )

        pairwise_fdes.append(
            d_fde
        )

        diversity_rows.append(
            {
                "task":
                    task,

                "seed_a":
                    a,

                "seed_b":
                    b,

                "mean_pairwise_trajectory_disagreement_px":
                    float(
                        d_ade.mean()
                    ),

                "mean_pairwise_endpoint_disagreement_px":
                    float(
                        d_fde.mean()
                    ),
            }
        )

    sample[
        "mean_pairwise_trajectory_disagreement_px"
    ] = np.mean(
        np.stack(
            pairwise_ades,
            axis=0
        ),
        axis=0
    )

    sample[
        "mean_pairwise_endpoint_disagreement_px"
    ] = np.mean(
        np.stack(
            pairwise_fdes,
            axis=0
        ),
        axis=0
    )

    comparison_specs = [
        (
            "ensemble_vs_mean_individual_seed",
            "ADE",
            "delta_ensemble_minus_mean_seed_ADE_px",
        ),
        (
            "ensemble_vs_mean_individual_seed",
            "FDE",
            "delta_ensemble_minus_mean_seed_FDE_px",
        ),
        (
            "ensemble_vs_validation_selected_seed",
            "ADE",
            "delta_ensemble_minus_validation_seed_ADE_px",
        ),
        (
            "ensemble_vs_validation_selected_seed",
            "FDE",
            "delta_ensemble_minus_validation_seed_FDE_px",
        ),
    ]

    for seed in SEEDS:
        comparison_specs.extend(
            [
                (
                    f"ensemble_vs_seed{seed}",
                    "ADE",
                    f"delta_ensemble_minus_seed{seed}_ADE_px",
                ),
                (
                    f"ensemble_vs_seed{seed}",
                    "FDE",
                    f"delta_ensemble_minus_seed{seed}_FDE_px",
                ),
            ]
        )

    bootstrap_rows = []

    for idx, (
        comparator,
        metric,
        col,
    ) in enumerate(
        comparison_specs
    ):
        result = cluster_bootstrap(
            sample,
            col,
            BOOTSTRAP_SEED
            +
            (
                1000
                if task
                ==
                "2s_to_5s"
                else
                0
            )
            +
            idx,
        )

        bootstrap_rows.append(
            {
                "task":
                    task,

                "comparator":
                    comparator,

                "metric":
                    metric,

                "delta_definition":
                    "ensemble - comparator; negative favors ensemble",

                "segments":
                    result[
                        "segments"
                    ],

                "delta_px":
                    result[
                        "point"
                    ],

                "ci95_low":
                    result[
                        "ci95_low"
                    ],

                "ci95_high":
                    result[
                        "ci95_high"
                    ],

                "direction":
                    result[
                        "direction"
                    ],
            }
        )

    diversity_gain_rows = [
        {
            "task":
                task,

            "gain_target":
                "ensemble_minus_mean_seed_ADE",

            "spearman_rho_diversity_vs_gain":
                spearman_via_ranks(
                    sample[
                        "mean_pairwise_trajectory_disagreement_px"
                    ],
                    sample[
                        "delta_ensemble_minus_mean_seed_ADE_px"
                    ],
                ),

            "note":
                (
                    "Negative gain is better for ensemble; a negative rho means "
                    "greater seed diversity tends to coincide with larger ensemble benefit."
                ),
        },
        {
            "task":
                task,

            "gain_target":
                "ensemble_minus_mean_seed_FDE",

            "spearman_rho_diversity_vs_gain":
                spearman_via_ranks(
                    sample[
                        "mean_pairwise_endpoint_disagreement_px"
                    ],
                    sample[
                        "delta_ensemble_minus_mean_seed_FDE_px"
                    ],
                ),

            "note":
                (
                    "Negative gain is better for ensemble; a negative rho means "
                    "greater seed diversity tends to coincide with larger ensemble benefit."
                ),
        },
    ]

    summary = {
        "task":
            task,

        "samples":
            int(
                len(
                    sample
                )
            ),

        "segments":
            int(
                sample.groupby(
                    cluster_cols(
                        sample
                    ),
                    dropna=False
                ).ngroups
            ),

        "sequences":
            int(
                source_sequence_series(
                    sample
                ).nunique()
            ),

        "validation_selected_seed":
            int(
                validation_best_seed
            ),

        "ensemble_ADE_px_window_mean":
            float(
                ens_ade.mean()
            ),

        "ensemble_FDE_px_window_mean":
            float(
                ens_fde.mean()
            ),

        "mean_individual_seed_ADE_px_window_mean":
            float(
                mean_seed_ade.mean()
            ),

        "mean_individual_seed_FDE_px_window_mean":
            float(
                mean_seed_fde.mean()
            ),

        "validation_selected_seed_ADE_px_window_mean":
            float(
                val_seed_ade.mean()
            ),

        "validation_selected_seed_FDE_px_window_mean":
            float(
                val_seed_fde.mean()
            ),

        "ensemble_gain_vs_mean_seed_ADE_px":
            float(
                ens_ade.mean()
                -
                mean_seed_ade.mean()
            ),

        "ensemble_gain_vs_mean_seed_FDE_px":
            float(
                ens_fde.mean()
                -
                mean_seed_fde.mean()
            ),

        "ensemble_gain_vs_validation_seed_ADE_px":
            float(
                ens_ade.mean()
                -
                val_seed_ade.mean()
            ),

        "ensemble_gain_vs_validation_seed_FDE_px":
            float(
                ens_fde.mean()
                -
                val_seed_fde.mean()
            ),
    }

    return (
        sample,
        pd.DataFrame(
            seed_metrics
        ),
        pd.DataFrame(
            checkpoint_rows
        ),
        pd.DataFrame(
            diversity_rows
        ),
        pd.DataFrame(
            bootstrap_rows
        ),
        pd.DataFrame(
            diversity_gain_rows
        ),
        summary,
        meta_path,
    )


def main():
    print(
        "\nR10-B — BIRDSAI TESTREAL FROZEN TRANSFORMER ENSEMBLE VALUE ANALYSIS"
    )

    print(
        "NO training and NO new model inference. This step uses the already-frozen "
        "primary anisotropic Transformer seed predictions on exact BIRDSAI TestReal. "
        "The actual 3-seed prediction ensemble is formed by averaging prediction_abs "
        "across seeds 42/52/62. Primary comparisons are pixel-space ADE/FDE versus "
        "(1) the per-sample mean individual-seed error and (2) the single seed selected "
        "ONLY by validation track-macro ADE. Paired dependence is handled by a "
        "20,000-replicate segment-cluster bootstrap. Pairwise seed disagreement is "
        "reported as ensemble diversity; test-selected 'best seed' is not used as a "
        "primary comparator.\n"
    )

    verify_r10a()
    geometry = load_geometry()
    inventory = load_r10a_testreal_inventory()

    all_sample = []
    all_seed = []
    all_ckpt = []
    all_diversity = []
    all_bootstrap = []
    all_corr = []
    summaries = []
    provenance_rows = []

    for task in TASKS:
        (
            sample_df,
            seed_df,
            ckpt_df,
            diversity_df,
            bootstrap_df,
            corr_df,
            summary,
            meta_path,
        ) = analyze_task(
            task,
            inventory,
            geometry,
        )

        sample_df[
            "task"
        ] = task

        all_sample.append(
            sample_df
        )

        all_seed.append(
            seed_df
        )

        all_ckpt.append(
            ckpt_df
        )

        all_diversity.append(
            diversity_df
        )

        all_bootstrap.append(
            bootstrap_df
        )

        all_corr.append(
            corr_df
        )

        summaries.append(
            summary
        )

        provenance_rows.append(
            {
                "task":
                    task,

                "metadata_path":
                    str(
                        meta_path
                    ),

                "metadata_sha256":
                    sha256_file(
                        meta_path
                    ),
            }
        )

    sample_all = pd.concat(
        all_sample,
        ignore_index=True
    )

    seed_all = pd.concat(
        all_seed,
        ignore_index=True
    )

    ckpt_all = pd.concat(
        all_ckpt,
        ignore_index=True
    )

    diversity_all = pd.concat(
        all_diversity,
        ignore_index=True
    )

    bootstrap_all = pd.concat(
        all_bootstrap,
        ignore_index=True
    )

    corr_all = pd.concat(
        all_corr,
        ignore_index=True
    )

    summary_df = pd.DataFrame(
        summaries
    )

    provenance_df = pd.DataFrame(
        provenance_rows
    )

    sample_path = (
        OUT
        /
        "R10B_TESTREAL_ENSEMBLE_SAMPLE_METRICS.csv"
    )

    seed_path = (
        OUT
        /
        "R10B_TESTREAL_INDIVIDUAL_SEED_METRICS.csv"
    )

    ckpt_path = (
        OUT
        /
        "R10B_VALIDATION_SEED_SELECTION_LOCK.csv"
    )

    diversity_path = (
        OUT
        /
        "R10B_PAIRWISE_SEED_DIVERSITY.csv"
    )

    bootstrap_path = (
        OUT
        /
        "R10B_PAIRED_SEGMENT_CLUSTER_BOOTSTRAP.csv"
    )

    corr_path = (
        OUT
        /
        "R10B_DIVERSITY_GAIN_ASSOCIATION.csv"
    )

    summary_path = (
        OUT
        /
        "R10B_TESTREAL_ENSEMBLE_SUMMARY.csv"
    )

    provenance_path = (
        OUT
        /
        "R10B_PROVENANCE.csv"
    )

    sample_all.to_csv(
        sample_path,
        index=False
    )

    seed_all.to_csv(
        seed_path,
        index=False
    )

    ckpt_all.to_csv(
        ckpt_path,
        index=False
    )

    diversity_all.to_csv(
        diversity_path,
        index=False
    )

    bootstrap_all.to_csv(
        bootstrap_path,
        index=False
    )

    corr_all.to_csv(
        corr_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    provenance_df.to_csv(
        provenance_path,
        index=False
    )

    primary = bootstrap_all[
        bootstrap_all[
            "comparator"
        ].isin(
            [
                "ensemble_vs_mean_individual_seed",
                "ensemble_vs_validation_selected_seed",
            ]
        )
    ].copy()

    primary_both_tasks_ready = (
        len(
            primary
        )
        ==
        8
    )

    if not primary_both_tasks_ready:
        raise RuntimeError(
            f"Expected 8 primary bootstrap rows, found {len(primary)}."
        )

    status = (
        "R10B_TESTREAL_TRANSFORMER_ENSEMBLE_VALUE_QUANTIFIED_READY_FOR_R10C_FRESH_WILDBOX_TRANSFORMER_REINFERENCE"
    )

    manifest = {
        "step":
            "R10B_BIRDSAI_TESTREAL_FROZEN_TRANSFORMER_ENSEMBLE_VALUE_ANALYSIS",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "new_model_inference":
            False,

        "representation":
            "original frozen anisotropic primary pipeline",

        "ensemble":
            "arithmetic mean of prediction_abs across Transformer seeds 42,52,62",

        "primary_metric_space":
            "pixel",

        "primary_comparators":
            [
                "per-sample mean individual-seed error",
                "validation-selected single seed using best_validation_track_macro_ADE only",
            ],

        "test_selected_best_seed_used_as_primary_comparator":
            False,

        "dependency_unit":
            "(sequence, segment_id)",

        "within_segment_reduction":
            "median paired difference",

        "bootstrap_replicates":
            BOOTSTRAP_B,

        "bootstrap_seed_base":
            BOOTSTRAP_SEED,

        "delta_definition":
            "ensemble - comparator; negative favors ensemble",

        "outputs": {
            "sample_metrics":
                str(
                    sample_path
                ),

            "individual_seed_metrics":
                str(
                    seed_path
                ),

            "validation_seed_selection_lock":
                str(
                    ckpt_path
                ),

            "pairwise_seed_diversity":
                str(
                    diversity_path
                ),

            "paired_bootstrap":
                str(
                    bootstrap_path
                ),

            "diversity_gain_association":
                str(
                    corr_path
                ),

            "summary":
                str(
                    summary_path
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
        "R10B_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "TESTREAL ENSEMBLE SUMMARY:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nVALIDATION-SELECTED SEED LOCK:"
    )

    print(
        ckpt_all.to_string(
            index=False
        )
    )

    print(
        "\nPRIMARY PAIRED SEGMENT-CLUSTER BOOTSTRAP:"
    )

    print(
        primary[
            [
                "task",
                "comparator",
                "metric",
                "segments",
                "delta_px",
                "ci95_low",
                "ci95_high",
                "direction",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nPAIRWISE SEED DIVERSITY:"
    )

    print(
        diversity_all.to_string(
            index=False
        )
    )

    print(
        "\nDIVERSITY vs ENSEMBLE-GAIN ASSOCIATION:"
    )

    print(
        corr_all.to_string(
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
        "\nR10-B COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
