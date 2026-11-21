
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
)

E1B_DIR = BASE / "R9I_E1B_frozen_18_checkpoint_testreal_inference"
E2B_DIR = BASE / "R9I_E2B_frozen_old_anisotropic_reinference"

R9I_E1B_MANIFEST = E1B_DIR / "R9I_E1B_MANIFEST.json"
R9I_E2B_MANIFEST = E2B_DIR / "R9I_E2B_MANIFEST.json"

GEOMETRY_CSV = (
    BASE
    / "R9G_V3B_corrected_isotropic_rebuild"
    / "R9G_V3B_SEQUENCE_GEOMETRY_LOCK.csv"
)

OUT = (
    BASE
    / "R9I_E2C_paired_segment_cluster_bootstrap_anisotropic_vs_isotropic"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "expected_samples": 1053,
        "expected_segments": 89,
        "expected_sequences": 6,
        "future_length": 20,
    },
    "2s_to_5s": {
        "expected_samples": 826,
        "expected_segments": 63,
        "expected_sequences": 5,
        "future_length": 50,
    },
}

MODELS = ["Transformer", "LSTM", "GRU"]
SEEDS = [42, 52, 62]

BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20261029

IDENTITY_CANDIDATES = [
    "sample_id",
    "sequence",
    "source_group",
    "segment_id",
    "track_key",
    "obs_start_frame",
    "obs_end_frame",
    "future_end_frame",
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_manifests():
    for p in [
        R9I_E1B_MANIFEST,
        R9I_E2B_MANIFEST,
        GEOMETRY_CSV,
    ]:
        require(p)

    iso = json.loads(
        R9I_E1B_MANIFEST.read_text(encoding="utf-8")
    )
    aniso = json.loads(
        R9I_E2B_MANIFEST.read_text(encoding="utf-8")
    )

    if (
        iso.get("status")
        !=
        "R9I_E1B_ALL_18_FROZEN_ISOTROPIC_CHECKPOINTS_EVALUATED_ON_MATCHED_TESTREAL_READY_FOR_R9I_E2_PAIRED_ANISOTROPIC_VS_ISOTROPIC_COMPARISON"
    ):
        raise RuntimeError(
            f"Unexpected E1B status: {iso.get('status')}"
        )

    if (
        aniso.get("status")
        !=
        "R9I_E2B_ALL_18_OLD_ANISOTROPIC_CHECKPOINTS_REINFERRED_READY_FOR_R9I_E2C_PAIRED_SEGMENT_CLUSTER_BOOTSTRAP_COMPARISON"
    ):
        raise RuntimeError(
            f"Unexpected E2B status: {aniso.get('status')}"
        )

    if int(iso.get("runs", -1)) != 18:
        raise RuntimeError("E1B manifest does not contain 18 runs.")

    if int(aniso.get("runs", -1)) != 18:
        raise RuntimeError("E2B manifest does not contain 18 runs.")

    return iso, aniso


def run_map(manifest):
    mapping = {}

    for row in manifest["runs_detail"]:
        key = (
            str(row["task"]),
            str(row["model"]),
            int(row["seed"]),
        )

        if key in mapping:
            raise RuntimeError(
                f"Duplicate run identity in manifest: {key}"
            )

        mapping[key] = row

    return mapping


def load_geometry():
    df = pd.read_csv(
        GEOMETRY_CSV,
        dtype={"sequence": str}
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

    out = {}

    for _, row in df.iterrows():
        seq = str(row["sequence"])
        W = float(row["width_px"])
        H = float(row["height_px"])
        S = float(row["iso_scale_px"])

        if not np.isclose(
            S,
            max(W, H),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{seq}: iso scale != max(W,H)."
            )

        out[seq] = {
            "W": W,
            "H": H,
            "S": S,
        }

    return out


def normalize_identity_dtypes(df):
    out = df.copy()

    for col in IDENTITY_CANDIDATES:
        if col in out.columns:
            out[col] = out[col].astype(str)

    return out


def identity_columns(a, b):
    preferred = [
        "sample_id",
        "sequence",
        "segment_id",
        "track_key",
        "obs_start_frame",
        "obs_end_frame",
        "future_end_frame",
    ]

    cols = [
        c
        for c in preferred
        if c in a.columns and c in b.columns
    ]

    if "sample_id" not in cols:
        raise RuntimeError(
            "sample_id is required for exact pairing."
        )

    if "segment_id" not in cols:
        raise RuntimeError(
            "segment_id is required for segment-cluster bootstrap."
        )

    return cols


def load_sample_metrics(run, representation):
    path = Path(run["sample_metrics_csv"])
    require(path)

    expected_hash = run["sample_metrics_sha256"]

    if sha256_file(path) != expected_hash:
        raise RuntimeError(
            f"{representation}: sample metrics hash changed: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "sequence": str,
            "source_group": str,
            "segment_id": str,
            "track_key": str,
        },
    )

    return normalize_identity_dtypes(df)


def pair_seed_run(
    task,
    model,
    seed,
    iso_run,
    aniso_run,
):
    iso = load_sample_metrics(
        iso_run,
        "isotropic"
    )

    aniso = load_sample_metrics(
        aniso_run,
        "anisotropic"
    )

    expected = TASKS[task]["expected_samples"]

    if len(iso) != expected:
        raise RuntimeError(
            f"{task}/{model}/{seed}: isotropic rows={len(iso)}, expected={expected}"
        )

    if len(aniso) != expected:
        raise RuntimeError(
            f"{task}/{model}/{seed}: anisotropic rows={len(aniso)}, expected={expected}"
        )

    keys = identity_columns(
        iso,
        aniso
    )

    if iso[keys].duplicated().any():
        raise RuntimeError(
            f"{task}/{model}/{seed}: duplicate isotropic identities."
        )

    if aniso[keys].duplicated().any():
        raise RuntimeError(
            f"{task}/{model}/{seed}: duplicate anisotropic identities."
        )

    iso_keep = iso[
        keys
        +
        [
            "ADE_px",
            "FDE_px",
        ]
    ].copy()

    aniso_keep = aniso[
        keys
        +
        [
            "ADE_px",
            "FDE_px",
        ]
    ].copy()

    iso_keep = iso_keep.rename(
        columns={
            "ADE_px": "ADE_px_iso",
            "FDE_px": "FDE_px_iso",
        }
    )

    aniso_keep = aniso_keep.rename(
        columns={
            "ADE_px": "ADE_px_aniso",
            "FDE_px": "FDE_px_aniso",
        }
    )

    paired = aniso_keep.merge(
        iso_keep,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )

    if not (
        paired["_merge"] == "both"
    ).all():
        bad = paired[
            paired["_merge"] != "both"
        ]

        raise RuntimeError(
            f"{task}/{model}/{seed}: non-matched rows={len(bad)}"
        )

    paired = paired.drop(
        columns=["_merge"]
    )

    if len(paired) != expected:
        raise RuntimeError(
            f"{task}/{model}/{seed}: paired rows={len(paired)}, expected={expected}"
        )

    paired["task"] = task
    paired["model"] = model
    paired["seed"] = seed

    paired["delta_ADE_px_iso_minus_aniso"] = (
        paired["ADE_px_iso"]
        -
        paired["ADE_px_aniso"]
    )

    paired["delta_FDE_px_iso_minus_aniso"] = (
        paired["FDE_px_iso"]
        -
        paired["FDE_px_aniso"]
    )

    paired["pct_ADE_change_iso_vs_aniso"] = (
        100.0
        *
        paired["delta_ADE_px_iso_minus_aniso"]
        /
        paired["ADE_px_aniso"].replace(0.0, np.nan)
    )

    paired["pct_FDE_change_iso_vs_aniso"] = (
        100.0
        *
        paired["delta_FDE_px_iso_minus_aniso"]
        /
        paired["FDE_px_aniso"].replace(0.0, np.nan)
    )

    return paired


def build_seed_paired_table(
    iso_map,
    aniso_map,
):
    frames = []

    for task in TASKS:
        for model in MODELS:
            for seed in SEEDS:
                key = (
                    task,
                    model,
                    seed,
                )

                if key not in iso_map:
                    raise RuntimeError(
                        f"Missing isotropic run: {key}"
                    )

                if key not in aniso_map:
                    raise RuntimeError(
                        f"Missing anisotropic run: {key}"
                    )

                frames.append(
                    pair_seed_run(
                        task,
                        model,
                        seed,
                        iso_map[key],
                        aniso_map[key],
                    )
                )

    return pd.concat(
        frames,
        ignore_index=True
    )


def three_seed_sample_mean(seed_paired):
    identity = [
        c
        for c in [
            "task",
            "model",
            "sample_id",
            "sequence",
            "source_group",
            "segment_id",
            "track_key",
            "obs_start_frame",
            "obs_end_frame",
            "future_end_frame",
        ]
        if c in seed_paired.columns
    ]

    numeric = [
        "ADE_px_iso",
        "FDE_px_iso",
        "ADE_px_aniso",
        "FDE_px_aniso",
        "delta_ADE_px_iso_minus_aniso",
        "delta_FDE_px_iso_minus_aniso",
    ]

    counts = (
        seed_paired.groupby(
            identity,
            dropna=False
        )["seed"]
        .nunique()
        .reset_index(
            name="n_seeds"
        )
    )

    bad = counts[
        counts["n_seeds"] != 3
    ]

    if len(bad):
        raise RuntimeError(
            f"Some sample identities do not have exactly 3 seeds: {len(bad)}"
        )

    out = (
        seed_paired.groupby(
            identity,
            dropna=False,
            as_index=False
        )[numeric]
        .mean()
    )

    out["n_seeds"] = 3

    out["pct_ADE_change_iso_vs_aniso"] = (
        100.0
        *
        out["delta_ADE_px_iso_minus_aniso"]
        /
        out["ADE_px_aniso"].replace(0.0, np.nan)
    )

    out["pct_FDE_change_iso_vs_aniso"] = (
        100.0
        *
        out["delta_FDE_px_iso_minus_aniso"]
        /
        out["FDE_px_aniso"].replace(0.0, np.nan)
    )

    return out


def cluster_columns(df):
    cols = []

    if "sequence" in df.columns:
        cols.append("sequence")
    elif "source_group" in df.columns:
        cols.append("source_group")

    if "segment_id" in df.columns:
        cols.append("segment_id")

    if len(cols) != 2:
        raise RuntimeError(
            f"Could not form (sequence, segment_id) cluster key. cols={cols}"
        )

    return cols


def segment_summary(sample_mean):
    cluster_cols = cluster_columns(
        sample_mean
    )

    rows = []

    for (
        task,
        model
    ), group in sample_mean.groupby(
        [
            "task",
            "model",
        ],
        sort=False
    ):
        expected_segments = TASKS[
            task
        ][
            "expected_segments"
        ]

        seg = (
            group.groupby(
                cluster_cols,
                dropna=False,
                as_index=False
            )
            .agg(
                windows=(
                    "sample_id",
                    "size"
                ),

                delta_ADE_px_segment_median=(
                    "delta_ADE_px_iso_minus_aniso",
                    "median"
                ),

                delta_FDE_px_segment_median=(
                    "delta_FDE_px_iso_minus_aniso",
                    "median"
                ),

                delta_ADE_px_segment_mean=(
                    "delta_ADE_px_iso_minus_aniso",
                    "mean"
                ),

                delta_FDE_px_segment_mean=(
                    "delta_FDE_px_iso_minus_aniso",
                    "mean"
                ),

                ADE_px_iso_segment_mean=(
                    "ADE_px_iso",
                    "mean"
                ),

                ADE_px_aniso_segment_mean=(
                    "ADE_px_aniso",
                    "mean"
                ),

                FDE_px_iso_segment_mean=(
                    "FDE_px_iso",
                    "mean"
                ),

                FDE_px_aniso_segment_mean=(
                    "FDE_px_aniso",
                    "mean"
                ),
            )
        )

        if len(seg) != expected_segments:
            raise RuntimeError(
                f"{task}/{model}: segments={len(seg)}, expected={expected_segments}"
            )

        seg["task"] = task
        seg["model"] = model

        rows.append(
            seg
        )

    return pd.concat(
        rows,
        ignore_index=True
    )


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


def bootstrap_segment_effects(segment_df):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    rows = []

    for (
        task,
        model
    ), group in segment_df.groupby(
        [
            "task",
            "model",
        ],
        sort=False
    ):
        group = group.reset_index(
            drop=True
        )

        n = len(group)

        ade_primary = group[
            "delta_ADE_px_segment_median"
        ].to_numpy(
            dtype=np.float64
        )

        fde_primary = group[
            "delta_FDE_px_segment_median"
        ].to_numpy(
            dtype=np.float64
        )

        ade_mean_sens = group[
            "delta_ADE_px_segment_mean"
        ].to_numpy(
            dtype=np.float64
        )

        fde_mean_sens = group[
            "delta_FDE_px_segment_mean"
        ].to_numpy(
            dtype=np.float64
        )

        boot_ade = np.empty(
            BOOTSTRAP_B,
            dtype=np.float64
        )

        boot_fde = np.empty(
            BOOTSTRAP_B,
            dtype=np.float64
        )

        boot_ade_mean_sens = np.empty(
            BOOTSTRAP_B,
            dtype=np.float64
        )

        boot_fde_mean_sens = np.empty(
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

            boot_ade[b] = float(
                ade_primary[idx].mean()
            )

            boot_fde[b] = float(
                fde_primary[idx].mean()
            )

            boot_ade_mean_sens[b] = float(
                ade_mean_sens[idx].mean()
            )

            boot_fde_mean_sens[b] = float(
                fde_mean_sens[idx].mean()
            )

        ade_lo, ade_hi = percentile_ci(
            boot_ade
        )

        fde_lo, fde_hi = percentile_ci(
            boot_fde
        )

        ade_s_lo, ade_s_hi = percentile_ci(
            boot_ade_mean_sens
        )

        fde_s_lo, fde_s_hi = percentile_ci(
            boot_fde_mean_sens
        )

        point_ade = float(
            ade_primary.mean()
        )

        point_fde = float(
            fde_primary.mean()
        )

        point_ade_sens = float(
            ade_mean_sens.mean()
        )

        point_fde_sens = float(
            fde_mean_sens.mean()
        )

        if ade_hi < 0:
            ade_direction = "ISOTROPIC_BETTER_CI_EXCLUDES_ZERO"
        elif ade_lo > 0:
            ade_direction = "ISOTROPIC_WORSE_CI_EXCLUDES_ZERO"
        else:
            ade_direction = "NO_CLEAR_DIFFERENCE_CI_INCLUDES_ZERO"

        if fde_hi < 0:
            fde_direction = "ISOTROPIC_BETTER_CI_EXCLUDES_ZERO"
        elif fde_lo > 0:
            fde_direction = "ISOTROPIC_WORSE_CI_EXCLUDES_ZERO"
        else:
            fde_direction = "NO_CLEAR_DIFFERENCE_CI_INCLUDES_ZERO"

        rows.append(
            {
                "task":
                    task,

                "model":
                    model,

                "segments":
                    int(
                        n
                    ),

                "bootstrap_replicates":
                    BOOTSTRAP_B,

                "bootstrap_seed":
                    BOOTSTRAP_SEED,

                "primary_estimand":
                    "mean across segments of within-segment median paired pixel error difference (isotropic - anisotropic)",

                "ADE_delta_px_primary":
                    point_ade,

                "ADE_delta_px_ci95_low":
                    ade_lo,

                "ADE_delta_px_ci95_high":
                    ade_hi,

                "ADE_direction":
                    ade_direction,

                "FDE_delta_px_primary":
                    point_fde,

                "FDE_delta_px_ci95_low":
                    fde_lo,

                "FDE_delta_px_ci95_high":
                    fde_hi,

                "FDE_direction":
                    fde_direction,

                "ADE_delta_px_segment_mean_sensitivity":
                    point_ade_sens,

                "ADE_delta_px_segment_mean_sensitivity_ci95_low":
                    ade_s_lo,

                "ADE_delta_px_segment_mean_sensitivity_ci95_high":
                    ade_s_hi,

                "FDE_delta_px_segment_mean_sensitivity":
                    point_fde_sens,

                "FDE_delta_px_segment_mean_sensitivity_ci95_low":
                    fde_s_lo,

                "FDE_delta_px_segment_mean_sensitivity_ci95_high":
                    fde_s_hi,
            }
        )

    return pd.DataFrame(
        rows
    )


def seed_sensitivity(seed_paired):
    rows = []

    for (
        task,
        model,
        seed
    ), group in seed_paired.groupby(
        [
            "task",
            "model",
            "seed",
        ],
        sort=False
    ):
        cluster_cols = cluster_columns(
            group
        )

        seg = (
            group.groupby(
                cluster_cols,
                dropna=False,
                as_index=False
            )
            .agg(
                delta_ADE=(
                    "delta_ADE_px_iso_minus_aniso",
                    "median"
                ),

                delta_FDE=(
                    "delta_FDE_px_iso_minus_aniso",
                    "median"
                ),
            )
        )

        rows.append(
            {
                "task":
                    task,

                "model":
                    model,

                "seed":
                    int(
                        seed
                    ),

                "segments":
                    int(
                        len(
                            seg
                        )
                    ),

                "ADE_delta_px_segment_median_macro":
                    float(
                        seg[
                            "delta_ADE"
                        ].mean()
                    ),

                "FDE_delta_px_segment_median_macro":
                    float(
                        seg[
                            "delta_FDE"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def loo_sequence_sensitivity(sample_mean):
    source_col = (
        "sequence"
        if "sequence" in sample_mean.columns
        else "source_group"
    )

    rows = []

    for (
        task,
        model
    ), group in sample_mean.groupby(
        [
            "task",
            "model",
        ],
        sort=False
    ):
        sequences = sorted(
            group[
                source_col
            ].astype(str).unique()
        )

        for held_out in sequences:
            kept = group[
                group[
                    source_col
                ].astype(str)
                !=
                held_out
            ].copy()

            cluster_cols = cluster_columns(
                kept
            )

            seg = (
                kept.groupby(
                    cluster_cols,
                    dropna=False,
                    as_index=False
                )
                .agg(
                    delta_ADE=(
                        "delta_ADE_px_iso_minus_aniso",
                        "median"
                    ),

                    delta_FDE=(
                        "delta_FDE_px_iso_minus_aniso",
                        "median"
                    ),
                )
            )

            rows.append(
                {
                    "task":
                        task,

                    "model":
                        model,

                    "held_out_sequence":
                        held_out,

                    "remaining_segments":
                        int(
                            len(
                                seg
                            )
                        ),

                    "ADE_delta_px_primary":
                        float(
                            seg[
                                "delta_ADE"
                            ].mean()
                        ),

                    "FDE_delta_px_primary":
                        float(
                            seg[
                                "delta_FDE"
                            ].mean()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def ensemble_sample_metrics(
    task,
    model,
    representation,
    runs,
    geometry,
):
    prediction_arrays = []
    truth_arrays = []
    metadata_path = None

    for seed in SEEDS:
        key = (
            task,
            model,
            seed,
        )

        run = runs[
            key
        ]

        pred_path = Path(
            run[
                "prediction_npz"
            ]
        )

        require(
            pred_path
        )

        if sha256_file(
            pred_path
        ) != run[
            "prediction_npz_sha256"
        ]:
            raise RuntimeError(
                f"{representation}/{key}: prediction hash changed."
            )

        arrays = np.load(
            pred_path,
            allow_pickle=False
        )

        prediction_arrays.append(
            np.asarray(
                arrays[
                    "prediction_abs"
                ],
                dtype=np.float64
            )
        )

        truth_arrays.append(
            np.asarray(
                arrays[
                    "truth_abs"
                ],
                dtype=np.float64
            )
        )

        sm_path = Path(
            run[
                "sample_metrics_csv"
            ]
        )

        if metadata_path is None:
            sample_df = pd.read_csv(
                sm_path,
                dtype={
                    "sample_id": str,
                    "sequence": str,
                    "source_group": str,
                    "segment_id": str,
                    "track_key": str,
                },
            )

    truth0 = truth_arrays[0]

    for truth in truth_arrays[1:]:
        if not np.array_equal(
            truth,
            truth0
        ):
            raise RuntimeError(
                f"{representation}/{task}/{model}: truth arrays differ across seeds."
            )

    ensemble_pred = np.mean(
        np.stack(
            prediction_arrays,
            axis=0
        ),
        axis=0
    )

    delta = ensemble_pred - truth0

    source_col = (
        "sequence"
        if "sequence" in sample_df.columns
        else "source_group"
    )

    seqs = sample_df[
        source_col
    ].astype(str).tolist()

    if representation == "isotropic":
        scales = np.asarray(
            [
                geometry[
                    seq
                ][
                    "S"
                ]
                for seq in seqs
            ],
            dtype=np.float64
        )

        err_px = (
            np.linalg.norm(
                delta,
                axis=2
            )
            *
            scales[
                :,
                None
            ]
        )

    else:
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

        dx = delta[
            :,
            :,
            0
        ] * W[
            :,
            None
        ]

        dy = delta[
            :,
            :,
            1
        ] * H[
            :,
            None
        ]

        err_px = np.sqrt(
            dx ** 2
            +
            dy ** 2
        )

    out = normalize_identity_dtypes(
        sample_df[
            [
                c
                for c in IDENTITY_CANDIDATES
                if c in sample_df.columns
            ]
        ].copy()
    )

    out[
        f"ADE_px_{representation}"
    ] = err_px.mean(
        axis=1
    )

    out[
        f"FDE_px_{representation}"
    ] = err_px[
        :,
        -1
    ]

    return out


def transformer_ensemble_comparison(
    iso_map,
    aniso_map,
    geometry,
):
    rows = []

    for task in TASKS:
        iso = ensemble_sample_metrics(
            task,
            "Transformer",
            "isotropic",
            iso_map,
            geometry,
        )

        aniso = ensemble_sample_metrics(
            task,
            "Transformer",
            "anisotropic",
            aniso_map,
            geometry,
        )

        keys = identity_columns(
            iso,
            aniso
        )

        paired = aniso.merge(
            iso,
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        if not (
            paired[
                "_merge"
            ]
            ==
            "both"
        ).all():
            raise RuntimeError(
                f"{task}: Transformer ensemble sample pairing failed."
            )

        paired = paired.drop(
            columns=[
                "_merge"
            ]
        )

        paired[
            "delta_ADE_px_iso_minus_aniso"
        ] = (
            paired[
                "ADE_px_isotropic"
            ]
            -
            paired[
                "ADE_px_anisotropic"
            ]
        )

        paired[
            "delta_FDE_px_iso_minus_aniso"
        ] = (
            paired[
                "FDE_px_isotropic"
            ]
            -
            paired[
                "FDE_px_anisotropic"
            ]
        )

        cluster_cols = cluster_columns(
            paired
        )

        seg = (
            paired.groupby(
                cluster_cols,
                dropna=False,
                as_index=False
            )
            .agg(
                delta_ADE=(
                    "delta_ADE_px_iso_minus_aniso",
                    "median"
                ),

                delta_FDE=(
                    "delta_FDE_px_iso_minus_aniso",
                    "median"
                ),
            )
        )

        rng = np.random.default_rng(
            BOOTSTRAP_SEED
            +
            (
                0
                if task == "2s_to_2s"
                else 1000
            )
        )

        n = len(
            seg
        )

        ade = seg[
            "delta_ADE"
        ].to_numpy(
            dtype=np.float64
        )

        fde = seg[
            "delta_FDE"
        ].to_numpy(
            dtype=np.float64
        )

        boot_ade = np.empty(
            BOOTSTRAP_B,
            dtype=np.float64
        )

        boot_fde = np.empty(
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

            boot_ade[
                b
            ] = ade[
                idx
            ].mean()

            boot_fde[
                b
            ] = fde[
                idx
            ].mean()

        ade_lo, ade_hi = percentile_ci(
            boot_ade
        )

        fde_lo, fde_hi = percentile_ci(
            boot_fde
        )

        row = {
            "task":
                task,

            "model":
                "Transformer_3Seed_Prediction_Ensemble",

            "segments":
                n,

            "ADE_px_anisotropic_window_mean":
                float(
                    paired[
                        "ADE_px_anisotropic"
                    ].mean()
                ),

            "ADE_px_isotropic_window_mean":
                float(
                    paired[
                        "ADE_px_isotropic"
                    ].mean()
                ),

            "FDE_px_anisotropic_window_mean":
                float(
                    paired[
                        "FDE_px_anisotropic"
                    ].mean()
                ),

            "FDE_px_isotropic_window_mean":
                float(
                    paired[
                        "FDE_px_isotropic"
                    ].mean()
                ),

            "ADE_delta_px_primary":
                float(
                    ade.mean()
                ),

            "ADE_delta_px_ci95_low":
                ade_lo,

            "ADE_delta_px_ci95_high":
                ade_hi,

            "FDE_delta_px_primary":
                float(
                    fde.mean()
                ),

            "FDE_delta_px_ci95_low":
                fde_lo,

            "FDE_delta_px_ci95_high":
                fde_hi,
        }

        rows.append(
            row
        )

        paired.to_csv(
            OUT
            /
            f"R9I_E2C_{task}_TRANSFORMER_3SEED_ENSEMBLE_PAIRED_SAMPLE_METRICS.csv",
            index=False
        )

    return pd.DataFrame(
        rows
    )


def decision_table(bootstrap_df):
    rows = []

    for _, row in bootstrap_df.iterrows():
        ade_lo = float(
            row[
                "ADE_delta_px_ci95_low"
            ]
        )

        ade_hi = float(
            row[
                "ADE_delta_px_ci95_high"
            ]
        )

        fde_lo = float(
            row[
                "FDE_delta_px_ci95_low"
            ]
        )

        fde_hi = float(
            row[
                "FDE_delta_px_ci95_high"
            ]
        )

        if (
            ade_hi < 0
            and fde_hi < 0
        ):
            decision = (
                "ISOTROPIC_IMPROVES_BOTH_ADE_AND_FDE_WITH_CLUSTERED_CI_EXCLUDING_ZERO"
            )

        elif (
            ade_lo > 0
            and fde_lo > 0
        ):
            decision = (
                "ISOTROPIC_WORSENS_BOTH_ADE_AND_FDE_WITH_CLUSTERED_CI_EXCLUDING_ZERO"
            )

        else:
            decision = (
                "MIXED_OR_INCONCLUSIVE_NO_GLOBAL_SUPERIORITY_CLAIM"
            )

        rows.append(
            {
                "task":
                    row[
                        "task"
                    ],

                "model":
                    row[
                        "model"
                    ],

                "decision":
                    decision,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-I-E2C — PAIRED ANISOTROPIC VS ISOTROPIC SEGMENT-CLUSTER BOOTSTRAP"
    )

    print(
        "NO training and NO inference. This step compares the already-frozen anisotropic "
        "and isotropic neural results on EXACT matched BIRDSAI TestReal samples. "
        "Primary fair metrics are pixel-space ADE/FDE. For each task/model, the same-seed "
        "paired sample errors are first averaged across seeds; overlapping windows are then "
        "collapsed to within-segment median paired differences, and the mean segment effect "
        "is bootstrapped by segment (20,000 replicates, seed 20261029). "
        "Delta = isotropic - anisotropic, so negative is better for isotropic. "
        "A within-segment mean sensitivity analysis, per-seed sensitivity, leave-one-sequence-out "
        "sensitivity, and actual 3-seed Transformer prediction-ensemble comparison are also saved.\n"
    )

    iso_manifest, aniso_manifest = verify_manifests()
    iso_map = run_map(
        iso_manifest
    )
    aniso_map = run_map(
        aniso_manifest
    )
    geometry = load_geometry()

    seed_paired = build_seed_paired_table(
        iso_map,
        aniso_map,
    )

    seed_paired_path = (
        OUT
        /
        "R9I_E2C_ALL_SEED_PAIRED_SAMPLE_METRICS.csv"
    )

    seed_paired.to_csv(
        seed_paired_path,
        index=False
    )

    sample_mean = three_seed_sample_mean(
        seed_paired
    )

    sample_mean_path = (
        OUT
        /
        "R9I_E2C_3SEED_MEAN_PAIRED_SAMPLE_METRICS.csv"
    )

    sample_mean.to_csv(
        sample_mean_path,
        index=False
    )

    segment_df = segment_summary(
        sample_mean
    )

    segment_path = (
        OUT
        /
        "R9I_E2C_SEGMENT_LEVEL_PAIRED_EFFECTS.csv"
    )

    segment_df.to_csv(
        segment_path,
        index=False
    )

    bootstrap_df = bootstrap_segment_effects(
        segment_df
    )

    bootstrap_path = (
        OUT
        /
        "R9I_E2C_PRIMARY_SEGMENT_CLUSTER_BOOTSTRAP_RESULTS.csv"
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False
    )

    seed_sens_df = seed_sensitivity(
        seed_paired
    )

    seed_sens_path = (
        OUT
        /
        "R9I_E2C_SEED_SENSITIVITY.csv"
    )

    seed_sens_df.to_csv(
        seed_sens_path,
        index=False
    )

    loo_df = loo_sequence_sensitivity(
        sample_mean
    )

    loo_path = (
        OUT
        /
        "R9I_E2C_LEAVE_ONE_SEQUENCE_OUT_SENSITIVITY.csv"
    )

    loo_df.to_csv(
        loo_path,
        index=False
    )

    ensemble_df = transformer_ensemble_comparison(
        iso_map,
        aniso_map,
        geometry,
    )

    ensemble_path = (
        OUT
        /
        "R9I_E2C_TRANSFORMER_3SEED_PREDICTION_ENSEMBLE_COMPARISON.csv"
    )

    ensemble_df.to_csv(
        ensemble_path,
        index=False
    )

    decision_df = decision_table(
        bootstrap_df
    )

    decision_path = (
        OUT
        /
        "R9I_E2C_REPRESENTATION_DECISION_TABLE.csv"
    )

    decision_df.to_csv(
        decision_path,
        index=False
    )

    status = (
        "R9I_E2C_PAIRED_SEGMENT_CLUSTER_BOOTSTRAP_COMPLETE_READY_FOR_R9I_F_CANONICALIZATION_ABLATION_CONCLUSION"
    )

    manifest = {
        "step":
            "R9I_E2C_PAIRED_ANISOTROPIC_VS_ISOTROPIC_SEGMENT_CLUSTER_BOOTSTRAP",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "paired_samples":
            True,

        "primary_metric_space":
            "pixel",

        "delta_definition":
            "isotropic - anisotropic; negative favors isotropic",

        "seed_handling":
            "same-seed sample errors paired, then averaged across seeds before primary segment analysis",

        "dependency_unit":
            "(sequence, segment_id)",

        "primary_within_segment_reduction":
            "median paired pixel error difference",

        "primary_across_segment_estimand":
            "mean of segment-level medians",

        "bootstrap_replicates":
            BOOTSTRAP_B,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "ci":
            "95% percentile cluster bootstrap",

        "multiple_testing_adjustment":
            False,

        "claim_guardrail":
            (
                "Do not claim universal isotropic superiority unless the direction is "
                "consistent and the paired clustered CI supports it. Native normalized "
                "ADE/FDE are not used for cross-representation superiority claims."
            ),

        "outputs": {
            "all_seed_paired_sample_metrics":
                str(
                    seed_paired_path
                ),

            "three_seed_mean_paired_sample_metrics":
                str(
                    sample_mean_path
                ),

            "segment_level_effects":
                str(
                    segment_path
                ),

            "primary_bootstrap":
                str(
                    bootstrap_path
                ),

            "seed_sensitivity":
                str(
                    seed_sens_path
                ),

            "leave_one_sequence_out":
                str(
                    loo_path
                ),

            "transformer_3seed_prediction_ensemble":
                str(
                    ensemble_path
                ),

            "decision_table":
                str(
                    decision_path
                ),
        },
    }

    (
        OUT
        /
        "R9I_E2C_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "PRIMARY SEGMENT-CLUSTER BOOTSTRAP RESULTS:"
    )

    display_cols = [
        "task",
        "model",
        "segments",
        "ADE_delta_px_primary",
        "ADE_delta_px_ci95_low",
        "ADE_delta_px_ci95_high",
        "ADE_direction",
        "FDE_delta_px_primary",
        "FDE_delta_px_ci95_low",
        "FDE_delta_px_ci95_high",
        "FDE_direction",
    ]

    print(
        bootstrap_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\nREPRESENTATION DECISION TABLE:"
    )

    print(
        decision_df.to_string(
            index=False
        )
    )

    print(
        "\nTRANSFORMER 3-SEED PREDICTION ENSEMBLE COMPARISON:"
    )

    print(
        ensemble_df.to_string(
            index=False
        )
    )

    print(
        "\nSEED SENSITIVITY:"
    )

    print(
        seed_sens_df.to_string(
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
        "\nR9-I-E2C COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
