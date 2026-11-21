from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

SEARCH_ROOTS = [
    MAIN_RESULT_2,
    MAIN_RESULT,
]

R1_RECEIPT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R1_protocol_lock_v3"
    / "R1_V3_REVISION_PROTOCOL_RECEIPT.json"
)

R2A_SUMMARY = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2A_preflight_v2"
    / "R2A_V2_PREFLIGHT_SUMMARY.json"
)

EXPECTED_R1_PROTOCOL_SHA256 = (
    "4af15bd17f57d4c9932f482dcc047f3b344be4b810d0a4c7b48942f8d17406a8"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2B_fair_same_sample_benchmark"
)

OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SAMPLES = {
    ("BIRDSAI_TestReal", "2s_to_2s"):
        1053,

    ("BIRDSAI_TestReal", "2s_to_5s"):
        826,

    ("Fresh_WildBox", "2s_to_2s"):
        865,

    ("Fresh_WildBox", "2s_to_5s"):
        637,
}

PRIMARY_MODELS = [
    "Stationary",
    "ConstantVelocity",
    "KalmanCV",
    "GRU",
    "LSTM",
    "TransformerEnsemble",
]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def verify_prerequisites():
    if not R1_RECEIPT.exists():
        raise FileNotFoundError(R1_RECEIPT)

    r1 = json.loads(
        R1_RECEIPT.read_text(
            encoding="utf-8"
        )
    )

    if r1.get("protocol_sha256") != EXPECTED_R1_PROTOCOL_SHA256:
        raise RuntimeError(
            "R1 protocol SHA256 mismatch."
        )

    if not R2A_SUMMARY.exists():
        raise FileNotFoundError(R2A_SUMMARY)

    r2a = json.loads(
        R2A_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    if (
        r2a.get("status")
        !=
        "R2A_V2_PREFLIGHT_COMPLETE_READY_FOR_R2B_FAIR_BENCHMARK"
    ):
        raise RuntimeError(
            f"R2A is not ready for R2B: {r2a.get('status')}"
        )

    return r1, r2a


def normalize_task(value):
    text = str(value).lower().replace(" ", "")

    if any(
        token in text
        for token in [
            "2s_to_2s",
            "2s→2s",
            "2s->2s",
            "2to2",
            "2s2s",
        ]
    ):
        return "2s_to_2s"

    if any(
        token in text
        for token in [
            "2s_to_5s",
            "2s→5s",
            "2s->5s",
            "2to5",
            "2s5s",
        ]
    ):
        return "2s_to_5s"

    return None


def normalize_dataset_from_text(value):
    text = str(value).lower()

    if (
        "fresh"
        in
        text
        and
        "wildbox"
        in
        text
    ):
        return "Fresh_WildBox"

    if (
        "testreal"
        in
        text
        or
        "birdsai_testreal"
        in
        text
    ):
        return "BIRDSAI_TestReal"

    return None


def normalize_model(value, seed=None, path_text=""):
    raw = str(value).strip()
    low = raw.lower().replace(" ", "").replace("-", "").replace("_", "")

    if "stationary" in low:
        return "Stationary"

    if (
        "kalman"
        in
        low
    ):
        return "KalmanCV"

    if (
        low in {
            "cv",
            "constantvelocity",
            "constantvel",
        }
        or
        (
            "constant"
            in
            low
            and
            "velocity"
            in
            low
        )
    ):
        return "ConstantVelocity"

    if "gru" in low:
        if seed is not None and pd.notna(seed):
            try:
                return f"GRU_seed{int(float(seed))}"
            except Exception:
                pass
        return "GRU"

    if "lstm" in low:
        if seed is not None and pd.notna(seed):
            try:
                return f"LSTM_seed{int(float(seed))}"
            except Exception:
                pass
        return "LSTM"

    if "transformer" in low:
        if "ensemble" in low or "frozenensemble" in low or "3seed" in low:
            return "TransformerEnsemble"

        if seed is not None and pd.notna(seed):
            try:
                return f"Transformer_seed{int(float(seed))}"
            except Exception:
                pass

        # Path-level hint.
        path_low = str(path_text).lower()

        if "ensemble" in path_low:
            return "TransformerEnsemble"

        return "Transformer"

    return raw


def first_existing(columns, candidates):
    lookup = {
        str(c).lower():
            c
        for c in columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    return None


def discover_candidate_csvs():
    candidates = []
    seen = set()

    path_tokens = [
        "step3_fresh_wildbox_full_comparison",
        "fresh_wildbox",
        "birdsai_testreal",
        "frozen_ensemble",
        "lstm_gru_results",
        "baseline",
        "kalman",
        "constant_velocity",
        "transformer",
        "sample_metrics",
    ]

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*.csv"):
            if not p.is_file():
                continue

            full = str(p).lower()

            if not any(
                token in full
                for token in path_tokens
            ):
                continue

            rp = str(p.resolve()).lower()

            if rp in seen:
                continue

            seen.add(rp)

            candidates.append(p)

    return sorted(
        candidates,
        key=lambda p: (
            0
            if str(MAIN_RESULT_2).lower()
            in
            str(p).lower()
            else
            1,
            str(p).lower(),
        )
    )


def extract_metric_rows(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if len(df) == 0:
        return pd.DataFrame()

    ade_col = first_existing(
        df.columns,
        [
            "ade",
            "sample_ade",
            "ade_mean",
            "mean_ade",
        ],
    )

    fde_col = first_existing(
        df.columns,
        [
            "fde",
            "sample_fde",
            "fde_mean",
            "mean_fde",
        ],
    )

    if ade_col is None or fde_col is None:
        return pd.DataFrame()

    model_col = first_existing(
        df.columns,
        [
            "model",
            "model_name",
            "method",
            "baseline",
            "predictor",
        ],
    )

    task_col = first_existing(
        df.columns,
        [
            "task",
            "horizon",
            "forecast_task",
        ],
    )

    seed_col = first_existing(
        df.columns,
        [
            "seed",
            "model_seed",
        ],
    )

    sample_col = first_existing(
        df.columns,
        [
            "sample_index",
            "sample_id",
            "window_index",
            "window_id",
            "index",
        ],
    )

    track_col = first_existing(
        df.columns,
        [
            "track_id",
            "tracklet_id",
            "track",
            "derived_track_id",
        ],
    )

    source_col = first_existing(
        df.columns,
        [
            "source",
            "source_id",
            "video",
            "video_id",
            "sequence",
            "sequence_id",
        ],
    )

    dataset_col = first_existing(
        df.columns,
        [
            "dataset",
            "split",
            "evaluation_set",
        ],
    )

    path_dataset = normalize_dataset_from_text(
        path
    )

    path_task = normalize_task(
        path
    )

    rows = []

    for idx, row in df.iterrows():
        try:
            ade = float(
                row[
                    ade_col
                ]
            )

            fde = float(
                row[
                    fde_col
                ]
            )
        except Exception:
            continue

        if not (
            np.isfinite(ade)
            and
            np.isfinite(fde)
        ):
            continue

        dataset = path_dataset

        if dataset is None and dataset_col is not None:
            dataset = normalize_dataset_from_text(
                row[
                    dataset_col
                ]
            )

        # Fresh source IDs prove Fresh even if path lacks the word.
        if dataset is None and source_col is not None:
            source_value = str(
                row[
                    source_col
                ]
            )

            if source_value.startswith(
                "DJI_"
            ):
                dataset = "Fresh_WildBox"

        task = path_task

        if task is None and task_col is not None:
            task = normalize_task(
                row[
                    task_col
                ]
            )

        if dataset not in {
            "BIRDSAI_TestReal",
            "Fresh_WildBox",
        }:
            continue

        if task not in {
            "2s_to_2s",
            "2s_to_5s",
        }:
            continue

        seed = (
            row[
                seed_col
            ]
            if seed_col is not None
            else None
        )

        if model_col is not None:
            model = normalize_model(
                row[
                    model_col
                ],
                seed=seed,
                path_text=path,
            )

        else:
            # Infer from path only when explicit model column is absent.
            model = normalize_model(
                path.name,
                seed=seed,
                path_text=path,
            )

        sample_id = (
            str(
                row[
                    sample_col
                ]
            )
            if sample_col is not None
            else
            str(
                idx
            )
        )

        source_id = (
            str(
                row[
                    source_col
                ]
            )
            if source_col is not None
            else
            None
        )

        track_id = (
            str(
                row[
                    track_col
                ]
            )
            if track_col is not None
            else
            None
        )

        rows.append({
            "dataset":
                dataset,

            "task":
                task,

            "model_raw":
                (
                    str(
                        row[
                            model_col
                        ]
                    )
                    if model_col is not None
                    else
                    path.name
                ),

            "model":
                model,

            "seed":
                seed,

            "sample_id":
                sample_id,

            "track_id":
                track_id,

            "source_id":
                source_id,

            "ade":
                ade,

            "fde":
                fde,

            "source_csv":
                str(
                    path
                ),

            "source_csv_sha256":
                sha256_file(
                    path
                ),

            "source_row":
                int(
                    idx
                ),
        })

    return pd.DataFrame(
        rows
    )


def collect_all_metrics():
    candidate_files = discover_candidate_csvs()

    audit_rows = []

    metric_frames = []

    for path in candidate_files:
        extracted = extract_metric_rows(
            path
        )

        audit_rows.append({
            "path":
                str(
                    path
                ),

            "sha256":
                sha256_file(
                    path
                ),

            "metric_rows_extracted":
                int(
                    len(
                        extracted
                    )
                ),
        })

        if len(
            extracted
        ) > 0:
            metric_frames.append(
                extracted
            )

    if not metric_frames:
        raise RuntimeError(
            "No sample-level ADE/FDE rows could be recovered from candidate CSV files."
        )

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    audit = pd.DataFrame(
        audit_rows
    )

    return (
        metrics,
        audit,
    )


def deduplicate_rows(metrics):
    """
    Prefer Main Result 2.0 over Main Result for byte-equivalent/repeated result rows.
    Duplicate identity is dataset/task/model/sample/source/track + ADE/FDE.
    """
    df = metrics.copy()

    df[
        "path_rank"
    ] = df[
        "source_csv"
    ].map(
        lambda p:
            0
            if str(MAIN_RESULT_2).lower()
            in
            str(p).lower()
            else
            1
    )

    dedup_cols = [
        "dataset",
        "task",
        "model",
        "sample_id",
        "track_id",
        "source_id",
        "ade",
        "fde",
    ]

    df = (
        df
        .sort_values(
            [
                "path_rank",
                "source_csv",
                "source_row",
            ]
        )
        .drop_duplicates(
            subset=dedup_cols,
            keep="first",
        )
        .drop(
            columns=[
                "path_rank"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


def collapse_seed_metrics(metrics):
    """
    For GRU/LSTM seed rows, report the mean metric across the 3 frozen seeds per sample.
    This is NOT a trajectory ensemble and is labelled SeedMeanMetric.
    TransformerEnsemble remains the true frozen trajectory ensemble when present.
    """
    df = metrics.copy()

    output = []

    for (
        dataset,
        task,
    ), group in df.groupby(
        [
            "dataset",
            "task",
        ],
        sort=False,
    ):
        # Pass through non-GRU/LSTM-seed rows.
        seed_regex = re.compile(
            r"^(GRU|LSTM)_seed\d+$"
        )

        normal = group[
            ~group[
                "model"
            ].astype(
                str
            ).str.match(
                seed_regex
            )
        ].copy()

        output.append(
            normal
        )

        for family in [
            "GRU",
            "LSTM",
        ]:
            family_rows = group[
                group[
                    "model"
                ].astype(
                    str
                ).str.startswith(
                    family
                    +
                    "_seed"
                )
            ].copy()

            if len(
                family_rows
            ) == 0:
                # If a direct family-level metric already exists, do nothing.
                continue

            key_cols = [
                "dataset",
                "task",
                "sample_id",
                "track_id",
                "source_id",
            ]

            agg = (
                family_rows
                .groupby(
                    key_cols,
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    ade=(
                        "ade",
                        "mean"
                    ),

                    fde=(
                        "fde",
                        "mean"
                    ),

                    seed_count=(
                        "model",
                        "nunique"
                    ),
                )
            )

            agg[
                "model_raw"
            ] = (
                family
                +
                " 3-seed mean metric"
            )

            agg[
                "model"
            ] = family

            agg[
                "seed"
            ] = None

            agg[
                "source_csv"
            ] = "derived from frozen seed-level sample metrics"

            agg[
                "source_csv_sha256"
            ] = None

            agg[
                "source_row"
            ] = None

            output.append(
                agg[
                    [
                        "dataset",
                        "task",
                        "model_raw",
                        "model",
                        "seed",
                        "sample_id",
                        "track_id",
                        "source_id",
                        "ade",
                        "fde",
                        "source_csv",
                        "source_csv_sha256",
                        "source_row",
                    ]
                ]
            )

    return pd.concat(
        output,
        ignore_index=True,
    )


def coverage_table(metrics):
    rows = []

    for (
        dataset,
        task,
        model,
    ), group in metrics.groupby(
        [
            "dataset",
            "task",
            "model",
        ],
        sort=False,
    ):
        rows.append({
            "dataset":
                dataset,

            "task":
                task,

            "model":
                model,

            "rows":
                int(
                    len(
                        group
                    )
                ),

            "unique_samples":
                int(
                    group[
                        "sample_id"
                    ].nunique()
                ),

            "unique_tracks":
                int(
                    group[
                        "track_id"
                    ].dropna().nunique()
                ),

            "unique_sources":
                int(
                    group[
                        "source_id"
                    ].dropna().nunique()
                ),
        })

    return pd.DataFrame(
        rows
    )


def select_primary_rows(metrics):
    return metrics[
        metrics[
            "model"
        ].isin(
            PRIMARY_MODELS
        )
    ].copy()


def same_sample_intersection(primary):
    rows = []

    aligned_frames = []

    for (
        dataset,
        task,
    ), group in primary.groupby(
        [
            "dataset",
            "task",
        ],
        sort=False,
    ):
        model_sets = {}

        for model in PRIMARY_MODELS:
            subset = group[
                group[
                    "model"
                ]
                ==
                model
            ]

            model_sets[
                model
            ] = set(
                subset[
                    "sample_id"
                ].astype(
                    str
                )
            )

        if any(
            len(
                s
            )
            ==
            0
            for s in model_sets.values()
        ):
            intersection = set()

        else:
            intersection = set.intersection(
                *model_sets.values()
            )

        expected = EXPECTED_SAMPLES[
            (
                dataset,
                task,
            )
        ]

        row = {
            "dataset":
                dataset,

            "task":
                task,

            "expected_samples":
                int(
                    expected
                ),

            "same_sample_intersection":
                int(
                    len(
                        intersection
                    )
                ),

            "all_expected_samples_aligned":
                bool(
                    len(
                        intersection
                    )
                    ==
                    expected
                ),
        }

        for model in PRIMARY_MODELS:
            row[
                f"{model}_samples"
            ] = int(
                len(
                    model_sets[
                        model
                    ]
                )
            )

        rows.append(
            row
        )

        if intersection:
            aligned = group[
                group[
                    "sample_id"
                ].astype(
                    str
                ).isin(
                    intersection
                )
            ].copy()

            aligned_frames.append(
                aligned
            )

    aligned_all = (
        pd.concat(
            aligned_frames,
            ignore_index=True,
        )
        if aligned_frames
        else
        pd.DataFrame()
    )

    return (
        pd.DataFrame(
            rows
        ),
        aligned_all,
    )


def summary_metrics(aligned):
    rows = []

    if len(
        aligned
    ) == 0:
        return pd.DataFrame()

    for (
        dataset,
        task,
        model,
    ), group in aligned.groupby(
        [
            "dataset",
            "task",
            "model",
        ],
        sort=False,
    ):
        row = {
            "dataset":
                dataset,

            "task":
                task,

            "model":
                model,

            "samples":
                int(
                    group[
                        "sample_id"
                    ].nunique()
                ),

            "ADE_mean":
                float(
                    group[
                        "ade"
                    ].mean()
                ),

            "FDE_mean":
                float(
                    group[
                        "fde"
                    ].mean()
                ),
        }

        valid_track = group[
            "track_id"
        ].notna()

        if valid_track.any():
            track = (
                group[
                    valid_track
                ]
                .groupby(
                    "track_id",
                    as_index=False,
                )
                .agg(
                    ADE=(
                        "ade",
                        "mean"
                    ),

                    FDE=(
                        "fde",
                        "mean"
                    ),
                )
            )

            row[
                "track_macro_ADE"
            ] = float(
                track[
                    "ADE"
                ].mean()
            )

            row[
                "track_macro_FDE"
            ] = float(
                track[
                    "FDE"
                ].mean()
            )

            row[
                "tracks"
            ] = int(
                len(
                    track
                )
            )

        else:
            row[
                "track_macro_ADE"
            ] = np.nan

            row[
                "track_macro_FDE"
            ] = np.nan

            row[
                "tracks"
            ] = 0

        valid_source = group[
            "source_id"
        ].notna()

        if valid_source.any():
            source = (
                group[
                    valid_source
                ]
                .groupby(
                    "source_id",
                    as_index=False,
                )
                .agg(
                    ADE=(
                        "ade",
                        "mean"
                    ),

                    FDE=(
                        "fde",
                        "mean"
                    ),
                )
            )

            row[
                "source_macro_ADE"
            ] = float(
                source[
                    "ADE"
                ].mean()
            )

            row[
                "source_macro_FDE"
            ] = float(
                source[
                    "FDE"
                ].mean()
            )

            row[
                "sources"
            ] = int(
                len(
                    source
                )
            )

        else:
            row[
                "source_macro_ADE"
            ] = np.nan

            row[
                "source_macro_FDE"
            ] = np.nan

            row[
                "sources"
            ] = 0

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR2B — FAIR SAME-SAMPLE FORECASTING BENCHMARK CONSOLIDATION"
    )

    print(
        "Uses existing frozen sample-level outputs where available. "
        "NO retraining and NO post-test model selection.\n"
    )

    r1, r2a = verify_prerequisites()

    raw_metrics, file_audit = collect_all_metrics()

    dedup = deduplicate_rows(
        raw_metrics
    )

    collapsed = collapse_seed_metrics(
        dedup
    )

    coverage = coverage_table(
        collapsed
    )

    primary = select_primary_rows(
        collapsed
    )

    alignment, aligned = same_sample_intersection(
        primary
    )

    summary = summary_metrics(
        aligned
    )

    raw_path = (
        OUT
        /
        "R2B_recovered_sample_metrics_long.csv"
    )

    audit_path = (
        OUT
        /
        "R2B_source_csv_audit.csv"
    )

    coverage_path = (
        OUT
        /
        "R2B_model_sample_coverage.csv"
    )

    alignment_path = (
        OUT
        /
        "R2B_same_sample_alignment.csv"
    )

    aligned_path = (
        OUT
        /
        "R2B_aligned_primary_model_sample_metrics.csv"
    )

    summary_path = (
        OUT
        /
        "R2B_fair_primary_model_summary.csv"
    )

    dedup.to_csv(
        raw_path,
        index=False,
    )

    file_audit.to_csv(
        audit_path,
        index=False,
    )

    coverage.to_csv(
        coverage_path,
        index=False,
    )

    alignment.to_csv(
        alignment_path,
        index=False,
    )

    aligned.to_csv(
        aligned_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    full_alignment = bool(
        len(
            alignment
        )
        ==
        len(
            EXPECTED_SAMPLES
        )
        and
        alignment[
            "all_expected_samples_aligned"
        ].all()
    )

    status = (
        "R2B_FAIR_SAME_SAMPLE_BENCHMARK_COMPLETE_READY_FOR_R3_CLUSTERED_BOOTSTRAP"
        if full_alignment
        else
        "R2B_EXISTING_OUTPUTS_AUDITED_TARGETED_RECOMPUTATION_REQUIRED_FOR_MISSING_MODEL_SAMPLE_ROWS"
    )

    result = {
        "step":
            "R2B_FAIR_SAME_SAMPLE_FORECASTING_BENCHMARK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "r1_protocol_sha256":
            r1[
                "protocol_sha256"
            ],

        "r2a_status":
            r2a[
                "status"
            ],

        "retraining":
            False,

        "post_test_model_selection":
            False,

        "primary_models": PRIMARY_MODELS,

        "important_note":
            (
                "GRU and LSTM primary rows are 3-seed mean metrics when seed-level "
                "sample metrics were recovered. They are not trajectory ensembles. "
                "TransformerEnsemble refers to the frozen arithmetic-mean trajectory ensemble."
            ),

        "outputs": {
            "sample_metrics":
                str(
                    raw_path
                ),

            "source_audit":
                str(
                    audit_path
                ),

            "coverage":
                str(
                    coverage_path
                ),

            "alignment":
                str(
                    alignment_path
                ),

            "aligned_primary_metrics":
                str(
                    aligned_path
                ),

            "summary":
                str(
                    summary_path
                ),
        },
    }

    result_path = (
        OUT
        /
        "R2B_SUMMARY.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "SAME-SAMPLE ALIGNMENT:"
    )
    print(
        alignment.to_string(
            index=False
        )
    )

    print(
        "\nFAIR PRIMARY MODEL SUMMARY:"
    )

    if len(
        summary
    ) > 0:
        print(
            summary.to_string(
                index=False
            )
        )

    else:
        print(
            "No complete primary-model same-sample benchmark could be assembled."
        )

    print(
        "\nSTATUS:"
    )
    print(
        status
    )

    print(
        "\nR2B COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
