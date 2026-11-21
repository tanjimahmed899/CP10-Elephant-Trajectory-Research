
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

R9I_B_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_B_exact_model_hyperparameter_and_split_lock"
    / "R9I_B_MANIFEST.json"
)

R9I_E0_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_E0_matched_testreal_isotropic_dataset"
    / "R9I_E0_MANIFEST.json"
)

R9I_E1B_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_E1B_frozen_18_checkpoint_testreal_inference"
    / "R9I_E1B_MANIFEST.json"
)

ORIGINAL_TEST_DIR = MAIN1 / "final_neural_trajectory_dataset"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9I_E2A_old_anisotropic_checkpoint_and_testreal_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["2s_to_2s", "2s_to_5s"]
SEEDS = [42, 52, 62]
MODELS = ["Transformer", "LSTM", "GRU"]

# Authoritative hashes recovered and corrected in R9-I-A/R9-I-B.
EXPECTED_OLD_CHECKPOINT_HASHES = {
    ("2s_to_2s", "GRU", 42): "582ba748563fb493ef102b5b6a7e65f0b0536cb8a5b459c98cdb6c2fe7832f68",
    ("2s_to_2s", "LSTM", 42): "9c895c9be57785ea34b1201237a5416265a88262d52ace3cce6bbba76453e3c9",
    ("2s_to_2s", "GRU", 52): "82a2e3c3d2034703e1ef4d01d8c8fb9b02e2fd613cdf8df2b357aacf918c6f24",
    ("2s_to_2s", "LSTM", 52): "8a7efcc9f9a2fc7111481ffa94571d141753aa7a990ad84dc6596693fa751078",
    ("2s_to_2s", "GRU", 62): "f8225457542ba49ad07e9059756551ee63a07ffe3a35df3ef92c67979f147c97",
    ("2s_to_2s", "LSTM", 62): "4b82fbe5a513bde5ad9f2e01c820fc08264be168f22f5d9e26763567611a8bcb",
    ("2s_to_5s", "GRU", 42): "ed7194336b374002994b788d1a8503d6fff852d1ac9fe5321b18ad19f6409aea",
    ("2s_to_5s", "LSTM", 42): "52bc9ffbdb2f68d9873fcbb4d77465c095350e8c2910a2463fb9b775b0aee59f",
    ("2s_to_5s", "GRU", 52): "fcc8ec3bcc333af622121d19cef9fdf5330591b9199712b58eee4ab4376721e4",
    ("2s_to_5s", "LSTM", 52): "0d6dde4bd78cf14d318d01d1279d51d149444cef6e35d7a083ba63ddca7367cb",
    ("2s_to_5s", "GRU", 62): "2e095e248947c71343554337abef4f5cd94ace64fa383b1c4aec37ade4c1a9c4",
    ("2s_to_5s", "LSTM", 62): "62c5ce1189852d91faa2f70724618d8f82e7eb3581961a5feb906e7e91061cb8",
    ("2s_to_2s", "Transformer", 42): "8244d6dc60be34d5c7f3dcbb1a50036152f8c5e96d69f3b3f5a93ac6e4a7f14f",
    ("2s_to_2s", "Transformer", 52): "9264f2d81945601149cc2f6a970a2765942cb9a45e711c4d552434e9feb75eb5",
    ("2s_to_2s", "Transformer", 62): "0257eb576a78a6a40da51d8bbf728df30d5d8b77cd6ca7e21be81fa0fd3cded2",
    ("2s_to_5s", "Transformer", 42): "846e1398eb8354e871ea80b7a90888f99d02ce10dac56f0787bf3205588bbb0a",
    ("2s_to_5s", "Transformer", 52): "e8316f3db5a60e72d1fee28384726b5353ffaf8c68ffeec64abb2326b4098958",
    ("2s_to_5s", "Transformer", 62): "44d1d6d0e15af6844667b44e2bf6a4cb574b90884df8250af138e7841ec82b3a",
}

EXPECTED_OLD_TEST_HASHES = {
    "2s_to_2s": {
        "metadata": "06327381b239fe81417b21e0d6925008e5f648012379af9d1d10284ed4af98ac",
        "npz": "2b4a6fe06e5c33220013f8d56c0fef5eb9f55740f98ea1bf14a69b0d98802c5c",
    },
    "2s_to_5s": {
        "metadata": "807b9ad7e1e0a5e56900990dab13319ab423b05703cf5dde58ddbb5d2871c435",
        "npz": "7d0dcf8ecc85de11128887dcddb5a87fca1170833ed4563474d3478170966551",
    },
}

CANDIDATE_DIRS = [
    MAIN1 / "transformer_results",
    MAIN1 / "lstm_gru_results",
    MAIN2 / "transformer_results",
    MAIN2 / "lstm_gru_results",
    DOWNLOADS / "transformer_results",
    DOWNLOADS / "lstm_gru_results",
    MAIN1 / "final_neural_trajectory_dataset",
    MAIN2 / "final_neural_trajectory_dataset",
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


def verify_upstream():
    for p in [
        R9I_B_MANIFEST,
        R9I_E0_MANIFEST,
        R9I_E1B_MANIFEST,
    ]:
        require(p)

    b = json.loads(
        R9I_B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    e0 = json.loads(
        R9I_E0_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    e1b = json.loads(
        R9I_E1B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        b.get("status")
        !=
        "R9I_B_EXACT_MODEL_HYPERPARAMETER_SPLIT_AND_CHECKPOINT_LOCK_COMPLETE_READY_FOR_R9I_C_ISOTROPIC_TRAINREAL_DATASET_BUILD"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-B status: {b.get('status')}"
        )

    if (
        e0.get("status")
        !=
        "R9I_E0_MATCHED_TESTREAL_ISOTROPIC_CANONICAL_DATASETS_BUILT_READY_FOR_R9I_E1_FROZEN_18_CHECKPOINT_TESTREAL_INFERENCE"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-E0 status: {e0.get('status')}"
        )

    if (
        e1b.get("status")
        !=
        "R9I_E1B_ALL_18_FROZEN_ISOTROPIC_CHECKPOINTS_EVALUATED_ON_MATCHED_TESTREAL_READY_FOR_R9I_E2_PAIRED_ANISOTROPIC_VS_ISOTROPIC_COMPARISON"
    ):
        raise RuntimeError(
            f"Unexpected R9-I-E1B status: {e1b.get('status')}"
        )

    return (
        b,
        e0,
        e1b,
    )


def checkpoint_filename(
    task,
    model,
    seed,
):
    return f"{task}_{model}_seed{seed}.pt"


def locate_checkpoint(
    task,
    model,
    seed,
    expected_hash,
):
    name = checkpoint_filename(
        task,
        model,
        seed,
    )

    candidates = []

    for directory in CANDIDATE_DIRS:
        if not directory.exists():
            continue

        direct = directory / name

        if direct.exists():
            candidates.append(
                direct
            )

        # A shallow recursive fallback inside only known result directories.
        try:
            for p in directory.glob(
                f"**/{name}"
            ):
                if p.exists() and p not in candidates:
                    candidates.append(
                        p
                    )
        except Exception:
            pass

    matches = []

    inspected = []

    for p in candidates:
        current_hash = sha256_file(
            p
        )

        inspected.append(
            {
                "path":
                    str(
                        p
                    ),

                "sha256":
                    current_hash,

                "matches_expected":
                    current_hash == expected_hash,
            }
        )

        if current_hash == expected_hash:
            matches.append(
                p
            )

    unique_matches = []

    seen = set()

    for p in matches:
        resolved = str(
            p.resolve()
        ).lower()

        if resolved not in seen:
            seen.add(
                resolved
            )

            unique_matches.append(
                p
            )

    if len(
        unique_matches
    ) != 1:
        return (
            None,
            inspected,
        )

    return (
        unique_matches[
            0
        ],
        inspected,
    )


def audit_checkpoints():
    rows = []
    search_rows = []

    for task in TASKS:
        for model in MODELS:
            for seed in SEEDS:
                expected_hash = EXPECTED_OLD_CHECKPOINT_HASHES[
                    (
                        task,
                        model,
                        seed,
                    )
                ]

                found, inspected = locate_checkpoint(
                    task,
                    model,
                    seed,
                    expected_hash,
                )

                for item in inspected:
                    search_rows.append(
                        {
                            "task":
                                task,

                            "model":
                                model,

                            "seed":
                                seed,

                            "expected_sha256":
                                expected_hash,

                            **item,
                        }
                    )

                rows.append(
                    {
                        "task":
                            task,

                        "model":
                            model,

                        "seed":
                            seed,

                        "expected_sha256":
                            expected_hash,

                        "resolved_path":
                            (
                                str(
                                    found
                                )
                                if found is not None
                                else ""
                            ),

                        "resolved":
                            found is not None,

                        "resolved_sha256":
                            (
                                sha256_file(
                                    found
                                )
                                if found is not None
                                else ""
                            ),
                    }
                )

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            search_rows
        ),
    )


def audit_original_testreal():
    rows = []

    for task in TASKS:
        meta_path = (
            ORIGINAL_TEST_DIR
            /
            f"{task}_birdsai_test_metadata.csv"
        )

        npz_path = (
            ORIGINAL_TEST_DIR
            /
            f"{task}_birdsai_test_canonical.npz"
        )

        require(
            meta_path
        )

        require(
            npz_path
        )

        meta_hash = sha256_file(
            meta_path
        )

        npz_hash = sha256_file(
            npz_path
        )

        if (
            meta_hash
            !=
            EXPECTED_OLD_TEST_HASHES[
                task
            ][
                "metadata"
            ]
        ):
            raise RuntimeError(
                f"{task}: original anisotropic TestReal metadata hash mismatch."
            )

        if (
            npz_hash
            !=
            EXPECTED_OLD_TEST_HASHES[
                task
            ][
                "npz"
            ]
        ):
            raise RuntimeError(
                f"{task}: original anisotropic TestReal NPZ hash mismatch."
            )

        meta = pd.read_csv(
            meta_path,
            dtype={
                "sample_id": str,
                "track_key": str,
                "source_group": str,
                "sequence": str,
            }
        )

        rows.append(
            {
                "task":
                    task,

                "samples":
                    int(
                        len(
                            meta
                        )
                    ),

                "tracks":
                    int(
                        meta[
                            "track_key"
                        ].nunique()
                    )
                    if "track_key" in meta.columns
                    else None,

                "sequences":
                    int(
                        meta[
                            "source_group"
                        ].nunique()
                    )
                    if "source_group" in meta.columns
                    else int(
                        meta[
                            "sequence"
                        ].nunique()
                    ),

                "metadata":
                    str(
                        meta_path
                    ),

                "metadata_sha256":
                    meta_hash,

                "npz":
                    str(
                        npz_path
                    ),

                "npz_sha256":
                    npz_hash,
            }
        )

    return pd.DataFrame(
        rows
    )


def audit_existing_old_predictions(
    checkpoint_df,
):
    rows = []

    result_dirs = sorted(
        {
            str(
                Path(
                    p
                ).parent
            )
            for p in checkpoint_df[
                "resolved_path"
            ].tolist()
            if str(
                p
            )
        }
    )

    for directory_str in result_dirs:
        directory = Path(
            directory_str
        )

        if not directory.exists():
            continue

        for p in directory.glob(
            "*predictions.npz"
        ):
            name = p.name

            if not any(
                task in name
                for task in TASKS
            ):
                continue

            model = None

            for candidate in MODELS:
                if candidate in name:
                    model = candidate
                    break

            seed_match = re.search(
                r"seed(42|52|62)",
                name,
                flags=re.IGNORECASE,
            )

            task = next(
                (
                    t
                    for t in TASKS
                    if t in name
                ),
                None,
            )

            if (
                model is None
                or seed_match is None
                or task is None
            ):
                continue

            rows.append(
                {
                    "task":
                        task,

                    "model":
                        model,

                    "seed":
                        int(
                            seed_match.group(
                                1
                            )
                        ),

                    "path":
                        str(
                            p
                        ),

                    "sha256":
                        sha256_file(
                            p
                        ),

                    "size_bytes":
                        int(
                            p.stat().st_size
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR9-I-E2A — OLD ANISOTROPIC CHECKPOINT + TESTREAL LOCK"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "Before the paired anisotropic-vs-isotropic comparison, this step resolves the "
        "18 original anisotropic frozen checkpoints by their authoritative R9-I-B hashes, "
        "verifies the original anisotropic BIRDSAI TestReal files by frozen hashes, and "
        "inventories any already-existing old prediction NPZ files. No old result is modified.\n"
    )

    verify_upstream()

    (
        checkpoint_df,
        search_df,
    ) = audit_checkpoints()

    testreal_df = audit_original_testreal()

    old_prediction_df = audit_existing_old_predictions(
        checkpoint_df
    )

    checkpoint_path = (
        OUT
        /
        "R9I_E2A_OLD_ANISOTROPIC_CHECKPOINT_LOCK.csv"
    )

    search_path = (
        OUT
        /
        "R9I_E2A_CHECKPOINT_SEARCH_DETAIL.csv"
    )

    testreal_path = (
        OUT
        /
        "R9I_E2A_OLD_ANISOTROPIC_TESTREAL_LOCK.csv"
    )

    pred_path = (
        OUT
        /
        "R9I_E2A_EXISTING_OLD_PREDICTION_INVENTORY.csv"
    )

    checkpoint_df.to_csv(
        checkpoint_path,
        index=False
    )

    search_df.to_csv(
        search_path,
        index=False
    )

    testreal_df.to_csv(
        testreal_path,
        index=False
    )

    old_prediction_df.to_csv(
        pred_path,
        index=False
    )

    resolved_count = int(
        checkpoint_df[
            "resolved"
        ].sum()
    )

    if resolved_count == 18:
        status = (
            "R9I_E2A_ALL_18_OLD_ANISOTROPIC_CHECKPOINTS_AND_TESTREAL_LOCKED_READY_FOR_R9I_E2B_FROZEN_ANISOTROPIC_REINFERENCE"
        )
    else:
        status = (
            "R9I_E2A_OLD_ANISOTROPIC_CHECKPOINT_LOCK_INCOMPLETE_NEEDS_PATH_RESOLUTION"
        )

    manifest = {
        "step":
            "R9I_E2A_OLD_ANISOTROPIC_CHECKPOINT_AND_TESTREAL_LOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "checkpoints_expected":
            18,

        "checkpoints_resolved":
            resolved_count,

        "original_testreal_tasks_verified":
            int(
                len(
                    testreal_df
                )
            ),

        "existing_prediction_files_found":
            int(
                len(
                    old_prediction_df
                )
            ),

        "training":
            False,

        "inference":
            False,

        "old_results_modified":
            False,

        "outputs": {
            "checkpoint_lock":
                str(
                    checkpoint_path
                ),

            "checkpoint_search_detail":
                str(
                    search_path
                ),

            "original_testreal_lock":
                str(
                    testreal_path
                ),

            "existing_prediction_inventory":
                str(
                    pred_path
                ),
        },
    }

    (
        OUT
        /
        "R9I_E2A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "OLD ANISOTROPIC CHECKPOINT LOCK:"
    )

    print(
        checkpoint_df.to_string(
            index=False
        )
    )

    print(
        "\nORIGINAL ANISOTROPIC TESTREAL LOCK:"
    )

    print(
        testreal_df.to_string(
            index=False
        )
    )

    print(
        "\nEXISTING OLD PREDICTION INVENTORY:"
    )

    if len(
        old_prediction_df
    ):
        print(
            old_prediction_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No existing old prediction NPZ files found in the resolved checkpoint result directories."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-I-E2A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
