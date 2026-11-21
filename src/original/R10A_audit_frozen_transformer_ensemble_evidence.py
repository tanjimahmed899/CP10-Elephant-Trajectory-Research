
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"

R9_FINAL_MANIFEST = (
    REVISION_ROOT
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_F_canonicalization_ablation_conclusion"
    / "R9I_F_MANIFEST.json"
)

TRANSFORMER_RESULTS = MAIN1 / "transformer_results"
FINAL_NEURAL_DIR = MAIN1 / "final_neural_trajectory_dataset"

OUT = (
    REVISION_ROOT
    / "R10_ensemble_analysis"
    / "R10A_frozen_transformer_ensemble_evidence_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "expected_testreal_samples": 1053,
        "expected_fresh_wildbox_samples": 865,
    },
    "2s_to_5s": {
        "expected_testreal_samples": 826,
        "expected_fresh_wildbox_samples": 637,
    },
}

SEEDS = [42, 52, 62]

REQUIRED_PRED_KEYS = {
    "truth_abs",
    "prediction_abs",
    "prediction_canonical",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def verify_r9_complete():
    require(R9_FINAL_MANIFEST)

    m = json.loads(
        R9_FINAL_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R9I_F_CANONICALIZATION_ABLATION_CONCLUSION_LOCKED_R9_COMPLETE_READY_FOR_R10_ENSEMBLE_ANALYSIS"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R9 final status: {m.get('status')}"
        )

    return m


def prediction_file(task, dataset_label, seed):
    p = (
        TRANSFORMER_RESULTS
        /
        f"{task}_{dataset_label}_Transformer_seed{seed}_predictions.npz"
    )
    require(p)
    return p


def checkpoint_file(task, seed):
    p = (
        TRANSFORMER_RESULTS
        /
        f"{task}_Transformer_seed{seed}.pt"
    )
    require(p)
    return p


def inspect_prediction(path):
    arrays = np.load(
        path,
        allow_pickle=False
    )

    missing = REQUIRED_PRED_KEYS - set(
        arrays.files
    )

    if missing:
        raise RuntimeError(
            f"{path.name}: missing prediction keys {sorted(missing)}"
        )

    truth = np.asarray(
        arrays["truth_abs"]
    )

    pred = np.asarray(
        arrays["prediction_abs"]
    )

    pred_can = np.asarray(
        arrays["prediction_canonical"]
    )

    if truth.shape != pred.shape:
        raise RuntimeError(
            f"{path.name}: truth/pred shape mismatch: {truth.shape} vs {pred.shape}"
        )

    if truth.shape != pred_can.shape:
        raise RuntimeError(
            f"{path.name}: truth/pred_can shape mismatch: {truth.shape} vs {pred_can.shape}"
        )

    if truth.ndim != 3 or truth.shape[-1] != 2:
        raise RuntimeError(
            f"{path.name}: unexpected trajectory shape {truth.shape}"
        )

    if not np.isfinite(
        truth.astype(np.float64)
    ).all():
        raise RuntimeError(
            f"{path.name}: non-finite truth."
        )

    if not np.isfinite(
        pred.astype(np.float64)
    ).all():
        raise RuntimeError(
            f"{path.name}: non-finite prediction."
        )

    return {
        "samples": int(
            truth.shape[0]
        ),

        "future_length": int(
            truth.shape[1]
        ),

        "truth_shape": str(
            truth.shape
        ),

        "prediction_shape": str(
            pred.shape
        ),

        "truth_dtype": str(
            truth.dtype
        ),

        "prediction_dtype": str(
            pred.dtype
        ),

        "truth_abs":
            truth,

        "prediction_abs":
            pred,
    }


def audit_primary_predictions():
    rows = []
    truth_equality_rows = []

    for task in TASKS:
        for dataset_label in [
            "BIRDSAI_test",
            "WildBox_external",
        ]:
            seed_records = []

            for seed in SEEDS:
                p = prediction_file(
                    task,
                    dataset_label,
                    seed,
                )

                info = inspect_prediction(
                    p
                )

                ckpt = checkpoint_file(
                    task,
                    seed
                )

                rows.append(
                    {
                        "task":
                            task,

                        "dataset_label":
                            dataset_label,

                        "seed":
                            seed,

                        "prediction_path":
                            str(
                                p
                            ),

                        "prediction_sha256":
                            sha256_file(
                                p
                            ),

                        "prediction_size_bytes":
                            int(
                                p.stat().st_size
                            ),

                        "checkpoint_path":
                            str(
                                ckpt
                            ),

                        "checkpoint_sha256":
                            sha256_file(
                                ckpt
                            ),

                        "samples":
                            info[
                                "samples"
                            ],

                        "future_length":
                            info[
                                "future_length"
                            ],

                        "truth_shape":
                            info[
                                "truth_shape"
                            ],

                        "prediction_shape":
                            info[
                                "prediction_shape"
                            ],

                        "truth_dtype":
                            info[
                                "truth_dtype"
                            ],

                        "prediction_dtype":
                            info[
                                "prediction_dtype"
                            ],
                    }
                )

                seed_records.append(
                    (
                        seed,
                        info,
                    )
                )

            truth0 = seed_records[
                0
            ][
                1
            ][
                "truth_abs"
            ]

            all_exact = True
            max_diff = 0.0

            for seed, info in seed_records[
                1:
            ]:
                truth = info[
                    "truth_abs"
                ]

                if truth.shape != truth0.shape:
                    all_exact = False
                    max_diff = float(
                        "inf"
                    )
                    break

                exact = np.array_equal(
                    truth,
                    truth0
                )

                diff = float(
                    np.max(
                        np.abs(
                            truth.astype(
                                np.float64
                            )
                            -
                            truth0.astype(
                                np.float64
                            )
                        )
                    )
                )

                all_exact = (
                    all_exact
                    and exact
                )

                max_diff = max(
                    max_diff,
                    diff
                )

            n = int(
                truth0.shape[
                    0
                ]
            )

            if dataset_label == "BIRDSAI_test":
                expected = TASKS[
                    task
                ][
                    "expected_testreal_samples"
                ]

                dataset_classification = (
                    "MATCHED_PRIMARY_TESTREAL"
                    if n == expected
                    else
                    "UNEXPECTED_TESTREAL_SAMPLE_COUNT"
                )

            else:
                fresh_expected = TASKS[
                    task
                ][
                    "expected_fresh_wildbox_samples"
                ]

                if n == fresh_expected:
                    dataset_classification = (
                        "MATCHES_FRESH_WILDBOX_SAMPLE_COUNT"
                    )
                else:
                    dataset_classification = (
                        "DOES_NOT_MATCH_FRESH_WILDBOX_SAMPLE_COUNT"
                    )

            truth_equality_rows.append(
                {
                    "task":
                        task,

                    "dataset_label":
                        dataset_label,

                    "samples":
                        n,

                    "future_length":
                        int(
                            truth0.shape[
                                1
                            ]
                        ),

                    "truth_exact_across_3_seeds":
                        bool(
                            all_exact
                        ),

                    "truth_max_abs_difference_across_seeds":
                        max_diff,

                    "dataset_classification":
                        dataset_classification,
                }
            )

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            truth_equality_rows
        ),
    )


def inspect_metadata_candidate(path):
    try:
        df = pd.read_csv(
            path
        )

        return {
            "readable":
                True,

            "rows":
                int(
                    len(
                        df
                    )
                ),

            "columns":
                "|".join(
                    str(
                        c
                    )
                    for c in df.columns
                ),
        }

    except Exception as exc:
        return {
            "readable":
                False,

            "rows":
                None,

            "columns":
                f"<read error: {exc}>",
        }


def audit_metadata_candidates():
    rows = []

    patterns = [
        "*birdsai*test*metadata*.csv",
        "*wildbox*metadata*.csv",
        "*WildBox*metadata*.csv",
        "*external*metadata*.csv",
    ]

    seen = set()

    roots = [
        FINAL_NEURAL_DIR,
        MAIN1,
    ]

    for root in roots:
        if not root.exists():
            continue

        for pattern in patterns:
            for p in root.glob(
                f"**/{pattern}"
            ):
                resolved = str(
                    p.resolve()
                ).lower()

                if resolved in seen:
                    continue

                seen.add(
                    resolved
                )

                info = inspect_metadata_candidate(
                    p
                )

                rows.append(
                    {
                        "path":
                            str(
                                p
                            ),

                        "name":
                            p.name,

                        "sha256":
                            sha256_file(
                                p
                            ),

                        "size_bytes":
                            int(
                                p.stat().st_size
                            ),

                        **info,
                    }
                )

    return pd.DataFrame(
        rows
    )


def audit_r2_revision_candidates():
    rows = []

    if not REVISION_ROOT.exists():
        return pd.DataFrame(
            rows
        )

    candidate_files = []

    for pattern in [
        "**/*R2*.json",
        "**/*R2*.csv",
        "**/*fair*.json",
        "**/*fair*.csv",
        "**/*Fresh*WildBox*.json",
        "**/*Fresh*WildBox*.csv",
        "**/*fresh*wildbox*.json",
        "**/*fresh*wildbox*.csv",
    ]:
        try:
            candidate_files.extend(
                REVISION_ROOT.glob(
                    pattern
                )
            )
        except Exception:
            pass

    unique = []
    seen = set()

    for p in candidate_files:
        if not p.is_file():
            continue

        key = str(
            p.resolve()
        ).lower()

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            p
        )

    for p in unique:
        row = {
            "path":
                str(
                    p
                ),

            "name":
                p.name,

            "suffix":
                p.suffix.lower(),

            "sha256":
                sha256_file(
                    p
                ),

            "size_bytes":
                int(
                    p.stat().st_size
                ),

            "csv_rows":
                None,

            "mentions_865":
                False,

            "mentions_637":
                False,

            "mentions_wildbox":
                False,

            "mentions_fresh":
                False,
        }

        try:
            text = p.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            row[
                "mentions_865"
            ] = "865" in text

            row[
                "mentions_637"
            ] = "637" in text

            row[
                "mentions_wildbox"
            ] = "wildbox" in text.lower()

            row[
                "mentions_fresh"
            ] = "fresh" in text.lower()

        except Exception:
            pass

        if p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(
                    p
                )

                row[
                    "csv_rows"
                ] = int(
                    len(
                        df
                    )
                )

            except Exception:
                pass

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def summarize_dataset_status(truth_df):
    rows = []

    for _, row in truth_df.iterrows():
        task = str(
            row[
                "task"
            ]
        )

        label = str(
            row[
                "dataset_label"
            ]
        )

        samples = int(
            row[
                "samples"
            ]
        )

        if label == "BIRDSAI_test":
            expected = TASKS[
                task
            ][
                "expected_testreal_samples"
            ]

            ready = (
                samples
                ==
                expected
                and
                bool(
                    row[
                        "truth_exact_across_3_seeds"
                    ]
                )
            )

            recommendation = (
                "READY_FOR_PRIMARY_ENSEMBLE_ANALYSIS"
                if ready
                else
                "NOT_READY_INVESTIGATE_TESTREAL_IDENTITY"
            )

        else:
            expected = TASKS[
                task
            ][
                "expected_fresh_wildbox_samples"
            ]

            ready = (
                samples
                ==
                expected
                and
                bool(
                    row[
                        "truth_exact_across_3_seeds"
                    ]
                )
            )

            recommendation = (
                "READY_AS_FRESH_WILDBOX_ENSEMBLE_INPUT"
                if ready
                else
                "DO_NOT_TREAT_AS_FRESH_WILDBOX_FIND_MATCHED_R2_DATASET_AND_REINFER"
            )

        rows.append(
            {
                "task":
                    task,

                "dataset_label":
                    label,

                "observed_samples":
                    samples,

                "expected_samples_for_target_analysis":
                    expected,

                "truth_exact_across_3_seeds":
                    bool(
                        row[
                            "truth_exact_across_3_seeds"
                        ]
                    ),

                "ready_for_target_ensemble_analysis":
                    bool(
                        ready
                    ),

                "recommendation":
                    recommendation,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR10-A — FROZEN TRANSFORMER ENSEMBLE EVIDENCE AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R9 must be fully locked before R10 begins. This audit verifies the frozen "
        "primary anisotropic Transformer checkpoints and prediction files, checks that "
        "truth arrays are identical across seeds, confirms BIRDSAI TestReal sample counts, "
        "and determines whether the existing WildBox_external predictions correspond to "
        "the later fresh matched WildBox sample set (865 / 637) or to an older external set. "
        "It also inventories candidate metadata and R2/fair-comparison artifacts so R10 "
        "does not silently mix evaluation protocols.\n"
    )

    r9 = verify_r9_complete()

    (
        prediction_df,
        truth_df,
    ) = audit_primary_predictions()

    metadata_df = audit_metadata_candidates()

    r2_df = audit_r2_revision_candidates()

    status_df = summarize_dataset_status(
        truth_df
    )

    prediction_path = (
        OUT
        /
        "R10A_FROZEN_TRANSFORMER_PREDICTION_INVENTORY.csv"
    )

    truth_path = (
        OUT
        /
        "R10A_TRUTH_IDENTITY_ACROSS_SEEDS.csv"
    )

    metadata_path = (
        OUT
        /
        "R10A_METADATA_CANDIDATE_INVENTORY.csv"
    )

    r2_path = (
        OUT
        /
        "R10A_R2_FAIR_WILDBOX_CANDIDATES.csv"
    )

    status_path = (
        OUT
        /
        "R10A_DATASET_READINESS_DECISION.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    truth_df.to_csv(
        truth_path,
        index=False
    )

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    r2_df.to_csv(
        r2_path,
        index=False
    )

    status_df.to_csv(
        status_path,
        index=False
    )

    testreal_ready = bool(
        status_df[
            status_df[
                "dataset_label"
            ]
            ==
            "BIRDSAI_test"
        ][
            "ready_for_target_ensemble_analysis"
        ].all()
    )

    wildbox_rows = status_df[
        status_df[
            "dataset_label"
        ]
        ==
        "WildBox_external"
    ]

    fresh_wildbox_ready = bool(
        len(
            wildbox_rows
        )
        ==
        2
        and
        wildbox_rows[
            "ready_for_target_ensemble_analysis"
        ].all()
    )

    if not testreal_ready:
        final_status = (
            "R10A_PRIMARY_TESTREAL_ENSEMBLE_INPUT_NOT_READY_INVESTIGATE_BEFORE_R10B"
        )

    elif fresh_wildbox_ready:
        final_status = (
            "R10A_PRIMARY_TESTREAL_AND_FRESH_WILDBOX_ENSEMBLE_INPUTS_LOCKED_READY_FOR_R10B_ENSEMBLE_VALUE_ANALYSIS"
        )

    else:
        final_status = (
            "R10A_PRIMARY_TESTREAL_READY_EXISTING_WILDBOX_NOT_FRESH_READY_FOR_R10B_TESTREAL_AND_R10C_FRESH_WILDBOX_REINFERENCE"
        )

    manifest = {
        "step":
            "R10A_FROZEN_TRANSFORMER_ENSEMBLE_EVIDENCE_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            final_status,

        "training":
            False,

        "inference":
            False,

        "r9_final_status":
            r9.get(
                "status"
            ),

        "primary_representation":
            "original frozen anisotropic pipeline",

        "testreal_ready":
            testreal_ready,

        "existing_wildbox_matches_fresh_sample_counts":
            fresh_wildbox_ready,

        "fresh_wildbox_target_counts":
            {
                "2s_to_2s":
                    865,

                "2s_to_5s":
                    637,
            },

        "claim_guardrail":
            (
                "Do not treat an older WildBox_external prediction set as the later fresh "
                "matched WildBox evaluation merely because the filename contains WildBox_external."
            ),

        "outputs":
            {
                "prediction_inventory":
                    str(
                        prediction_path
                    ),

                "truth_identity":
                    str(
                        truth_path
                    ),

                "metadata_candidates":
                    str(
                        metadata_path
                    ),

                "r2_fair_wildbox_candidates":
                    str(
                        r2_path
                    ),

                "dataset_readiness":
                    str(
                        status_path
                    ),
            },
    }

    manifest_path = (
        OUT
        /
        "R10A_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "FROZEN TRANSFORMER PREDICTION INVENTORY:"
    )

    print(
        prediction_df[
            [
                "task",
                "dataset_label",
                "seed",
                "samples",
                "future_length",
                "prediction_sha256",
                "checkpoint_sha256",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nTRUTH IDENTITY / DATASET CLASSIFICATION:"
    )

    print(
        truth_df.to_string(
            index=False
        )
    )

    print(
        "\nDATASET READINESS DECISION:"
    )

    print(
        status_df.to_string(
            index=False
        )
    )

    print(
        "\nR2 / FAIR WILDBOX CANDIDATE ARTIFACTS:"
    )

    if len(
        r2_df
    ):
        print(
            r2_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No R2/fair/fresh-WildBox candidate artifacts found by filename scan."
        )

    print(
        "\nSTATUS:"
    )

    print(
        final_status
    )

    print(
        "\nR10-A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
