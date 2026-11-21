from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 6D-A: UNTOUCHED UNCERTAINTY HOLDOUT AUDIT + PRE-LOCK
#
# IMPORTANT:
# This script does NOT run any model and does NOT compute any forecast error.
# It only inventories data structure / metadata and looks for an eligible
# holdout that has not already been used in Steps 1-6C.
#
# If exactly one eligible READY canonical dataset exists, it creates a
# cryptographic pre-lock. If none exists, it reports raw candidate assets
# (especially NESTLER) that could be converted into a new holdout later.
#
# All new outputs go to Downloads.
# ============================================================

PROJECT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

DOWNLOADS = Path(
    r"C:\Users\HP\Downloads"
)

OUT = DOWNLOADS / "step6d_uncertainty_holdout"

OUT.mkdir(
    parents=True,
    exist_ok=True
)

REQUIRED_KEYS = {
    "history_canonical",
    "future_canonical",
    "history_abs",
    "future_abs",
    "origin",
    "heading_angle",
    "scale",
}

# Anything under these paths is already used for training/evaluation/development
# and is therefore NOT eligible as a new confirmatory uncertainty holdout.
USED_DIR_NAMES = {
    "final_neural_trajectory_dataset",
    "wildbox_fresh_confirmatory_results",
    "transformer_results",
    "transformer_ensemble_validation",
    "step2_ego_motion_validation",
    "step2b_raw_vs_corrected_baseline",
    "step2c_ego_motion_geofence_ablation",
    "step3_fresh_wildbox_full_comparison",
    "step4_clustered_bootstrap",
    "step5_ablation",
    "step6_conformal_uncertainty",
    "step6b_scale_adaptive_conformal",
    "step6c_crossfit_conformal",
}

# Filename/path terms known to correspond to already-used model-development sets.
USED_TERMS = {
    "birdsai_train",
    "birdsai_test",
    "trainreal",
    "testreal",
    "wildbox_external",
    "fresh_wildbox",
    "fresh_holdout",
    "fresh_confirmatory",
}

RAW_CANDIDATE_TERMS = {
    "nestler",
    "wildbox",
    "bird",
    "elephant",
}


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_under_used_dir(path):
    names = {p.lower() for p in path.parts}
    return any(name.lower() in names for name in USED_DIR_NAMES)


def contains_used_term(path):
    s = str(path).lower()
    return any(term in s for term in USED_TERMS)


def inspect_npz(path):
    row = {
        "path": str(path),
        "filename": path.name,
        "ready_canonical": False,
        "eligible_pre_result": False,
        "reason": "",
        "samples": None,
        "history_steps": None,
        "future_steps": None,
        "has_required_keys": False,
        "metadata_path": "",
    }

    try:
        arrays = np.load(path, allow_pickle=False)
        keys = set(arrays.files)

        row["has_required_keys"] = REQUIRED_KEYS.issubset(keys)

        if row["has_required_keys"]:
            h = arrays["history_canonical"]
            f = arrays["future_canonical"]

            if (
                h.ndim == 3
                and f.ndim == 3
                and h.shape[-1] == 2
                and f.shape[-1] == 2
                and len(h) == len(f)
            ):
                row["ready_canonical"] = True
                row["samples"] = int(h.shape[0])
                row["history_steps"] = int(h.shape[1])
                row["future_steps"] = int(f.shape[1])

        # Search sibling metadata candidates without opening outcome columns.
        stem = path.stem
        possible = [
            path.with_name(stem.replace("_dataset", "_metadata") + ".csv"),
            path.with_name(stem.replace("_canonical", "_metadata") + ".csv"),
            path.with_name(stem + "_metadata.csv"),
        ]

        for p in possible:
            if p.exists():
                row["metadata_path"] = str(p)
                break

        if not row["ready_canonical"]:
            row["reason"] = "not a ready canonical trajectory dataset"

        elif is_under_used_dir(path):
            row["reason"] = "already-used directory"

        elif contains_used_term(path):
            row["reason"] = "path/name matches already-used dataset"

        else:
            row["eligible_pre_result"] = True
            row["reason"] = "eligible by structure/path audit only"

    except Exception as e:
        row["reason"] = f"inspection_error: {type(e).__name__}: {e}"

    return row


def raw_candidate_inventory(search_root, max_files=5000):
    rows = []
    count = 0

    if not search_root.exists():
        return rows

    for p in search_root.rglob("*"):
        if not p.is_file():
            continue

        count += 1
        if count > max_files:
            break

        s = str(p).lower()

        if any(term in s for term in RAW_CANDIDATE_TERMS):
            if p.suffix.lower() in {
                ".zip", ".mp4", ".avi", ".mov", ".mkv",
                ".csv", ".json", ".txt", ".npz"
            }:
                rows.append({
                    "path": str(p),
                    "suffix": p.suffix.lower(),
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
                    "contains_nestler": "nestler" in s,
                    "contains_wildbox": "wildbox" in s,
                    "contains_birdsai": "birdsai" in s or "bird" in s,
                })

    return rows


def main():
    npz_rows = []

    for root in [PROJECT, DOWNLOADS]:
        if not root.exists():
            continue

        for p in root.rglob("*.npz"):
            # Do not recursively inspect Step 6D outputs themselves.
            if OUT in p.parents:
                continue

            npz_rows.append(
                inspect_npz(p)
            )

    npz_df = pd.DataFrame(npz_rows)

    if len(npz_df) == 0:
        npz_df = pd.DataFrame(columns=[
            "path", "filename", "ready_canonical",
            "eligible_pre_result", "reason", "samples",
            "history_steps", "future_steps", "has_required_keys",
            "metadata_path"
        ])

    npz_df.to_csv(
        OUT / "canonical_dataset_audit.csv",
        index=False
    )

    eligible = npz_df[
        npz_df["eligible_pre_result"] == True
    ].copy()

    eligible.to_csv(
        OUT / "eligible_ready_holdout_candidates.csv",
        index=False
    )

    raw_rows = []
    raw_rows.extend(
        raw_candidate_inventory(PROJECT)
    )
    raw_rows.extend(
        raw_candidate_inventory(DOWNLOADS)
    )

    raw_df = pd.DataFrame(raw_rows)

    if len(raw_df):
        raw_df = raw_df.drop_duplicates(
            subset=["path"]
        ).sort_values(
            ["contains_nestler", "size_mb"],
            ascending=[False, False]
        )

    raw_df.to_csv(
        OUT / "raw_candidate_assets.csv",
        index=False
    )

    lock = None

    # Automatically pre-lock only when there is exactly one structurally
    # eligible READY dataset. No result/performance information is used.
    if len(eligible) == 1:
        candidate_path = Path(
            eligible.iloc[0]["path"]
        )

        metadata_path = str(
            eligible.iloc[0]["metadata_path"]
        ).strip()

        lock = {
            "lock_stage": "STEP_6D_PRE_RESULT_LOCK",
            "selection_rule": (
                "the only ready canonical trajectory dataset outside "
                "all known training/evaluation/development paths"
            ),
            "model_or_error_results_inspected_before_lock": False,
            "dataset_path": str(candidate_path),
            "dataset_sha256": sha256_file(candidate_path),
            "metadata_path": metadata_path if metadata_path else None,
            "metadata_sha256": (
                sha256_file(Path(metadata_path))
                if metadata_path and Path(metadata_path).exists()
                else None
            ),
            "samples": int(
                eligible.iloc[0]["samples"]
            ),
            "history_steps": int(
                eligible.iloc[0]["history_steps"]
            ),
            "future_steps": int(
                eligible.iloc[0]["future_steps"]
            ),
        }

        with open(
            OUT / "STEP6D_HOLDOUT_PRELOCK.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                lock,
                f,
                indent=2
            )

    with open(
        OUT / "SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-A: UNTOUCHED UNCERTAINTY HOLDOUT AUDIT\n"
        )
        f.write(
            "=" * 72 + "\n\n"
        )
        f.write(
            "No model predictions or forecast errors were computed.\n"
        )
        f.write(
            "Eligibility used file structure/path history only.\n\n"
        )

        f.write(
            f"Ready eligible canonical candidates: {len(eligible)}\n\n"
        )

        if len(eligible):
            f.write(
                eligible[
                    [
                        "path",
                        "samples",
                        "history_steps",
                        "future_steps",
                        "metadata_path",
                    ]
                ].to_string(index=False)
            )
            f.write("\n\n")

        if lock is not None:
            f.write(
                "PRE-LOCK CREATED BEFORE EVALUATION:\n"
            )
            f.write(
                json.dumps(lock, indent=2)
            )
            f.write("\n")
        else:
            f.write(
                "No automatic ready-dataset pre-lock was created.\n"
            )
            if len(eligible) == 0:
                f.write(
                    "A new holdout must be built from an unused raw asset "
                    "before Step 6D evaluation.\n"
                )
            else:
                f.write(
                    "More than one eligible dataset exists; choose/lock "
                    "using a predeclared non-performance rule before evaluation.\n"
                )

        f.write("\nRAW CANDIDATE ASSETS (names/metadata only):\n")
        if len(raw_df):
            f.write(
                raw_df.head(100).to_string(index=False)
            )
        else:
            f.write("None found by filename/path scan.")

    print("\nSTEP 6D-A AUDIT COMPLETE")
    print("Outputs:", OUT)

    print("\nREADY ELIGIBLE CANONICAL HOLDOUTS:")
    if len(eligible):
        print(
            eligible[
                [
                    "path",
                    "samples",
                    "history_steps",
                    "future_steps",
                    "metadata_path"
                ]
            ].to_string(index=False)
        )
    else:
        print("NONE")

    if lock is not None:
        print("\nPRE-RESULT HOLDOUT LOCK CREATED:")
        print(
            OUT / "STEP6D_HOLDOUT_PRELOCK.json"
        )
    else:
        print("\nNO READY HOLDOUT WAS AUTO-LOCKED.")

    if len(raw_df):
        print("\nTOP RAW CANDIDATE ASSETS:")
        cols = [
            "path",
            "suffix",
            "size_mb",
            "contains_nestler",
            "contains_wildbox",
            "contains_birdsai"
        ]
        print(
            raw_df[cols].head(30).to_string(index=False)
        )


if __name__ == "__main__":
    main()
