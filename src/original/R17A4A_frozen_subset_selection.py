
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

R17A3B_DIR = R17_ROOT / "R17A3B_coco_animal_cooccurrence_audit"
R17A3B_MANIFEST = R17A3B_DIR / "R17A3B_MANIFEST.json"
COCO_POSITIVE = R17A3B_DIR / "R17A3B_ELEPHANT_PLUS_ANIMAL_CANDIDATES.csv"
COCO_NEGATIVE = R17A3B_DIR / "R17A3B_NEGATIVE_MULTI_ANIMAL_CANDIDATES.csv"
COCO_CONTROL = R17A3B_DIR / "R17A3B_ELEPHANT_ONLY_CONTROLS.csv"

R17A2_DIR = R17_ROOT / "R17A2_existing_dataset_cooccurrence_audit"
R17A2_MANIFEST = R17A2_DIR / "R17A2_MANIFEST.json"
R17A2_IMAGE_LEVEL = R17A2_DIR / "R17A2_IMAGE_LEVEL_COOCCURRENCE_AUDIT.csv"

OUT = R17_ROOT / "R17A4A_frozen_subset_selection"
OUT.mkdir(parents=True, exist_ok=True)

SELECTION_SALT = "R17A4A_COCO_MAIN_V1_20261030"
MAIN_GROUP_N = 151


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


def stable_key(split, image_id, group):
    text = (
        f"{SELECTION_SALT}|{group}|{split}|{image_id}"
    ).encode("utf-8")

    return hashlib.sha256(
        text
    ).hexdigest()


def verify_upstream():
    for p in [
        R17A3B_MANIFEST,
        COCO_POSITIVE,
        COCO_NEGATIVE,
        COCO_CONTROL,
        R17A2_MANIFEST,
        R17A2_IMAGE_LEVEL,
    ]:
        require(p)

    b = json.loads(
        R17A3B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    a2 = json.loads(
        R17A2_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_b = (
        "R17A3B_COCO_ANIMAL_AMONG_ANIMALS_POOL_VERIFIED_READY_FOR_R17A4_FROZEN_SUBSET_SELECTION_AND_PIXEL_DOWNLOAD"
    )

    expected_a2 = (
        "R17A2_EXISTING_COOCCURRENCE_INSUFFICIENT_EXTERNAL_MIXED_SPECIES_EVALUATION_DATASET_REQUIRED_BEFORE_R17B_PROTOCOL_LOCK"
    )

    if b.get("status") != expected_b:
        raise RuntimeError(
            f"Unexpected R17-A3B status: {b.get('status')}"
        )

    if a2.get("status") != expected_a2:
        raise RuntimeError(
            f"Unexpected R17-A2 status: {a2.get('status')}"
        )

    return b, a2


def normalize_image_id(df):
    out = df.copy()

    out["split"] = (
        out["split"]
        .astype(str)
        .str.strip()
    )

    out["image_id"] = pd.to_numeric(
        out["image_id"],
        errors="raise",
    ).astype(int)

    out["file_name"] = (
        out["file_name"]
        .astype(str)
        .str.strip()
    )

    out["coco_identity"] = (
        out["split"]
        + ":"
        + out["image_id"].astype(str)
    )

    return out


def freeze_main_subset():
    positive = normalize_image_id(
        pd.read_csv(
            COCO_POSITIVE,
            low_memory=False,
        )
    )

    negative = normalize_image_id(
        pd.read_csv(
            COCO_NEGATIVE,
            low_memory=False,
        )
    )

    control = normalize_image_id(
        pd.read_csv(
            COCO_CONTROL,
            low_memory=False,
        )
    )

    # Primary positive policy: use ALL available elephant+animal scenes.
    positive = (
        positive
        .drop_duplicates(
            subset=["coco_identity"]
        )
        .copy()
    )

    if len(positive) != MAIN_GROUP_N:
        raise RuntimeError(
            f"Expected exactly {MAIN_GROUP_N} unique positive COCO scenes, found {len(positive)}."
        )

    positive["evaluation_group"] = (
        "POSITIVE_ELEPHANT_PLUS_ANIMAL"
    )

    positive["selection_policy"] = (
        "ALL_AVAILABLE_UNIQUE_POSITIVE_SCENES"
    )

    positive["selection_rank"] = range(
        1,
        len(positive) + 1,
    )

    # Negative policy: deterministic hash selection from all no-elephant,
    # multi-animal candidates.
    negative = (
        negative
        .drop_duplicates(
            subset=["coco_identity"]
        )
        .copy()
    )

    negative["stable_hash"] = [
        stable_key(
            split,
            image_id,
            "NEGATIVE",
        )
        for split, image_id in zip(
            negative["split"],
            negative["image_id"],
        )
    ]

    negative = (
        negative
        .sort_values(
            [
                "stable_hash",
                "split",
                "image_id",
            ]
        )
        .head(
            MAIN_GROUP_N
        )
        .copy()
    )

    if len(negative) != MAIN_GROUP_N:
        raise RuntimeError(
            f"Insufficient negative candidates: {len(negative)}"
        )

    negative["evaluation_group"] = (
        "NEGATIVE_MULTI_ANIMAL_NO_ELEPHANT"
    )

    negative["selection_policy"] = (
        "DETERMINISTIC_SHA256_HASH_SAMPLE"
    )

    negative["selection_rank"] = range(
        1,
        len(negative) + 1,
    )

    # Clean elephant-only controls: prioritize no-person scenes, then
    # deterministic fill if needed.
    control = (
        control
        .drop_duplicates(
            subset=["coco_identity"]
        )
        .copy()
    )

    control["person_present"] = (
        control["person_present"]
        .astype(str)
        .str.lower()
        .isin(
            ["true", "1", "yes"]
        )
    )

    control["stable_hash"] = [
        stable_key(
            split,
            image_id,
            "CONTROL",
        )
        for split, image_id in zip(
            control["split"],
            control["image_id"],
        )
    ]

    clean_control = (
        control.loc[
            ~control["person_present"]
        ]
        .sort_values(
            [
                "stable_hash",
                "split",
                "image_id",
            ]
        )
        .copy()
    )

    if len(clean_control) >= MAIN_GROUP_N:
        control_sel = clean_control.head(
            MAIN_GROUP_N
        ).copy()

        control_policy = (
            "DETERMINISTIC_HASH_SAMPLE_PERSON_ABSENT_PRIORITY"
        )
    else:
        remaining = control.loc[
            ~control[
                "coco_identity"
            ].isin(
                set(
                    clean_control[
                        "coco_identity"
                    ]
                )
            )
        ].sort_values(
            [
                "stable_hash",
                "split",
                "image_id",
            ]
        )

        needed = (
            MAIN_GROUP_N
            -
            len(clean_control)
        )

        control_sel = pd.concat(
            [
                clean_control,
                remaining.head(
                    needed
                ),
            ],
            ignore_index=True,
        )

        control_policy = (
            "PERSON_ABSENT_PRIORITY_THEN_DETERMINISTIC_FILL"
        )

    if len(control_sel) != MAIN_GROUP_N:
        raise RuntimeError(
            f"Insufficient elephant-only controls: {len(control_sel)}"
        )

    control_sel["evaluation_group"] = (
        "CONTROL_ELEPHANT_ONLY_AMONG_ANIMAL_TARGETS"
    )

    control_sel["selection_policy"] = (
        control_policy
    )

    control_sel["selection_rank"] = range(
        1,
        len(control_sel) + 1,
    )

    # No identity overlap across the three primary groups.
    groups = {
        "positive": set(
            positive[
                "coco_identity"
            ]
        ),
        "negative": set(
            negative[
                "coco_identity"
            ]
        ),
        "control": set(
            control_sel[
                "coco_identity"
            ]
        ),
    }

    if groups["positive"] & groups["negative"]:
        raise RuntimeError(
            "Positive/negative identity overlap."
        )

    if groups["positive"] & groups["control"]:
        raise RuntimeError(
            "Positive/control identity overlap."
        )

    if groups["negative"] & groups["control"]:
        raise RuntimeError(
            "Negative/control identity overlap."
        )

    primary = pd.concat(
        [
            positive,
            negative,
            control_sel,
        ],
        ignore_index=True,
        sort=False,
    )

    return (
        positive,
        negative,
        control_sel,
        primary,
    )


def freeze_existing_in_vocab_supplement():
    df = pd.read_csv(
        R17A2_IMAGE_LEVEL,
        low_memory=False,
    )

    flag = (
        df["elephant_plus_other"]
        .astype(str)
        .str.lower()
        .isin(
            ["true", "1", "yes"]
        )
    )

    sub = df.loc[
        flag
    ].copy()

    if len(sub) == 0:
        return pd.DataFrame()

    sub["image_path"] = (
        sub["image_path"]
        .astype(str)
        .str.strip()
    )

    rows = []

    for _, r in sub.iterrows():
        path = Path(
            str(
                r["image_path"]
            )
        )

        if not path.exists():
            continue

        digest = sha256_file(
            path
        )

        rows.append(
            {
                "image_path":
                    str(path),

                "image_name":
                    path.name,

                "image_stem":
                    path.stem,

                "sha256":
                    digest,

                "dataset_root":
                    str(
                        r.get(
                            "dataset_root",
                            "",
                        )
                    ),

                "split":
                    str(
                        r.get(
                            "split",
                            "",
                        )
                    ),

                "classes_present":
                    str(
                        r.get(
                            "classes_present",
                            "",
                        )
                    ),

                "num_boxes":
                    int(
                        r.get(
                            "num_boxes",
                            0,
                        )
                    ),
            }
        )

    if not rows:
        return pd.DataFrame()

    files = pd.DataFrame(
        rows
    )

    # Deduplicate actual repeated copies by content hash, not by dataset row count.
    files = (
        files
        .sort_values(
            [
                "sha256",
                "image_path",
            ]
        )
        .drop_duplicates(
            subset=["sha256"],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    files["evaluation_group"] = (
        "SUPPLEMENTARY_IN_VOCAB_ELEPHANT_PLUS_LION"
    )

    files["selection_policy"] = (
        "ALL_UNIQUE_EXISTING_COOCCURRENCE_FILES_DEDUPLICATED_BY_SHA256"
    )

    files["selection_rank"] = range(
        1,
        len(files) + 1,
    )

    return files


def main():
    print(
        "\nR17-A4A — FROZEN MIXED-SPECIES EVALUATION SUBSET SELECTION"
    )

    print(
        "SELECTION LOCK ONLY. NO image download, NO model inference, and NO tuning. "
        "The primary COCO evaluation uses all 151 available elephant+animal positive scenes, "
        "151 deterministic no-elephant multi-animal negatives, and 151 deterministic elephant-only controls. "
        "Existing elephant+lion scenes are retained separately as a deduplicated in-vocabulary supplementary set.\n"
    )

    b, a2 = verify_upstream()

    (
        positive,
        negative,
        control,
        primary,
    ) = freeze_main_subset()

    supplementary = (
        freeze_existing_in_vocab_supplement()
    )

    positive_path = (
        OUT
        / "R17A4A_PRIMARY_POSITIVE_151.csv"
    )

    negative_path = (
        OUT
        / "R17A4A_PRIMARY_NEGATIVE_151.csv"
    )

    control_path = (
        OUT
        / "R17A4A_PRIMARY_CONTROL_151.csv"
    )

    primary_path = (
        OUT
        / "R17A4A_PRIMARY_BALANCED_453.csv"
    )

    supplementary_path = (
        OUT
        / "R17A4A_SUPPLEMENTARY_IN_VOCAB_UNIQUE.csv"
    )

    positive.to_csv(
        positive_path,
        index=False,
    )

    negative.to_csv(
        negative_path,
        index=False,
    )

    control.to_csv(
        control_path,
        index=False,
    )

    primary.to_csv(
        primary_path,
        index=False,
    )

    supplementary.to_csv(
        supplementary_path,
        index=False,
    )

    pair_counts = []

    for confuser in [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "bear",
        "zebra",
        "giraffe",
    ]:
        mask = (
            positive[
                "animal_confusers"
            ]
            .fillna("")
            .astype(str)
            .str.split("|")
            .apply(
                lambda xs:
                    confuser
                    in xs
            )
        )

        pair_counts.append(
            {
                "pair":
                    f"elephant+{confuser}",

                "selected_positive_images":
                    int(
                        mask.sum()
                    ),
            }
        )

    pair_df = pd.DataFrame(
        pair_counts
    )

    pair_path = (
        OUT
        / "R17A4A_SELECTED_POSITIVE_PAIR_COUNTS.csv"
    )

    pair_df.to_csv(
        pair_path,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "group":
                    "PRIMARY_POSITIVE_ELEPHANT_PLUS_ANIMAL",

                "n_images":
                    len(
                        positive
                    ),

                "source":
                    "COCO 2017 train+val",

                "policy":
                    "all unique available positives",
            },
            {
                "group":
                    "PRIMARY_NEGATIVE_MULTI_ANIMAL_NO_ELEPHANT",

                "n_images":
                    len(
                        negative
                    ),

                "source":
                    "COCO 2017 train+val",

                "policy":
                    "deterministic SHA256 sample",
            },
            {
                "group":
                    "PRIMARY_CONTROL_ELEPHANT_ONLY",

                "n_images":
                    len(
                        control
                    ),

                "source":
                    "COCO 2017 train+val",

                "policy":
                    "person-absent priority + deterministic SHA256 sample",
            },
            {
                "group":
                    "SUPPLEMENTARY_IN_VOCAB_ELEPHANT_PLUS_LION",

                "n_images":
                    len(
                        supplementary
                    ),

                "source":
                    "existing project V5/exported datasets",

                "policy":
                    "all unique files deduplicated by SHA256",
            },
        ]
    )

    summary_path = (
        OUT
        / "R17A4A_SELECTION_SUMMARY.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    status = (
        "R17A4A_FROZEN_EVALUATION_SUBSET_LOCKED_453_COCO_PRIMARY_PLUS_DEDUPLICATED_IN_VOCAB_SUPPLEMENT_READY_FOR_R17A4B_PIXEL_DOWNLOAD"
    )

    manifest = {
        "step":
            "R17A4A_FROZEN_MIXED_SPECIES_SUBSET_SELECTION",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "model_inference":
            False,

        "pixel_download":
            False,

        "selection_salt":
            SELECTION_SALT,

        "primary_group_size_each":
            MAIN_GROUP_N,

        "primary_total_images":
            int(
                len(
                    primary
                )
            ),

        "positive_policy":
            "all 151 unique COCO elephant+animal scenes",

        "negative_policy":
            "151 deterministic SHA256-selected COCO multi-animal no-elephant scenes",

        "control_policy":
            "151 deterministic elephant-only controls, prioritizing person-absent scenes",

        "supplementary_in_vocab_unique_images":
            int(
                len(
                    supplementary
                )
            ),

        "identity_overlap_between_primary_groups":
            0,

        "source_manifest_sha256": {
            "R17A3B":
                sha256_file(
                    R17A3B_MANIFEST
                ),

            "R17A2":
                sha256_file(
                    R17A2_MANIFEST
                ),
        },

        "output_sha256": {
            "positive":
                sha256_file(
                    positive_path
                ),

            "negative":
                sha256_file(
                    negative_path
                ),

            "control":
                sha256_file(
                    control_path
                ),

            "primary":
                sha256_file(
                    primary_path
                ),

            "supplementary":
                sha256_file(
                    supplementary_path
                ),

            "pair_counts":
                sha256_file(
                    pair_path
                ),
        },

        "outputs": {
            "positive":
                str(
                    positive_path
                ),

            "negative":
                str(
                    negative_path
                ),

            "control":
                str(
                    control_path
                ),

            "primary_453":
                str(
                    primary_path
                ),

            "supplementary_in_vocab":
                str(
                    supplementary_path
                ),

            "positive_pair_counts":
                str(
                    pair_path
                ),

            "summary":
                str(
                    summary_path
                ),
        },
    }

    (
        OUT
        / "R17A4A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "SELECTION SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSELECTED POSITIVE PAIR COUNTS:"
    )

    print(
        pair_df.to_string(
            index=False
        )
    )

    print(
        "\nSUPPLEMENTARY IN-VOCAB DEDUPLICATION:"
    )

    print(
        f"unique existing elephant+lion files = {len(supplementary)}"
    )

    if len(
        supplementary
    ):
        print(
            supplementary[
                [
                    "image_name",
                    "classes_present",
                    "sha256",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nPRIMARY GROUP IDENTITY OVERLAP: 0"
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR17-A4A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
