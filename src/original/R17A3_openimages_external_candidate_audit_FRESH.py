
from pathlib import Path
import csv
import hashlib
import json
import urllib.request
from collections import defaultdict, Counter
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

R17A2_MANIFEST = (
    R17_ROOT
    / "R17A2_existing_dataset_cooccurrence_audit"
    / "R17A2_MANIFEST.json"
)

OUT = (
    R17_ROOT
    / "R17A3_openimages_external_candidate_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

META = OUT / "openimages_metadata"
META.mkdir(parents=True, exist_ok=True)

URLS = {
    "classes": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "validation_boxes": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test_boxes": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}

FILES = {
    "classes": META / "oidv7-class-descriptions-boxable.csv",
    "validation_boxes": META / "validation-annotations-bbox.csv",
    "test_boxes": META / "test-annotations-bbox.csv",
}

ELEPHANT = "Elephant"

IN_VOCAB_CONFUSERS = [
    "Lion",
    "Tiger",
    "Crocodile",
]

OOD_CONFUSERS = [
    "Cattle",
    "Deer",
    "Horse",
    "Zebra",
    "Giraffe",
    "Goat",
    "Sheep",
    "Pig",
    "Dog",
    "Person",
    "Rhinoceros",
    "Hippopotamus",
    "Leopard",
    "Cheetah",
    "Bear",
]

ALL_TARGETS = (
    [ELEPHANT]
    + IN_VOCAB_CONFUSERS
    + OOD_CONFUSERS
)


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


def verify_r17a2():
    require(R17A2_MANIFEST)

    m = json.loads(
        R17A2_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R17A2_EXISTING_COOCCURRENCE_INSUFFICIENT_EXTERNAL_MIXED_SPECIES_EVALUATION_DATASET_REQUIRED_BEFORE_R17B_PROTOCOL_LOCK"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R17-A2 status: {m.get('status')}"
        )

    if not bool(
        m.get(
            "new_external_dataset_required_for_main_test",
            False,
        )
    ):
        raise RuntimeError(
            "R17-A2 does not require an external main evaluation set."
        )

    return m


def download_if_needed(key):
    path = FILES[key]
    url = URLS[key]

    if path.exists() and path.stat().st_size > 0:
        print(
            f"USING EXISTING: {path}"
        )
        return

    print(
        f"DOWNLOADING: {url}"
    )
    print(
        f"TO: {path}"
    )

    tmp = path.with_suffix(
        path.suffix + ".part"
    )

    if tmp.exists():
        tmp.unlink()

    urllib.request.urlretrieve(
        url,
        tmp,
    )

    tmp.replace(
        path
    )


def load_class_map():
    mid_to_name = {}
    name_to_mid = {}

    with open(
        FILES["classes"],
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 2:
                continue

            mid = row[0].strip()
            name = row[1].strip()

            mid_to_name[mid] = name
            name_to_mid[name] = mid

    missing = [
        x
        for x in ALL_TARGETS
        if x not in name_to_mid
    ]

    if missing:
        raise RuntimeError(
            "Required Open Images classes missing: "
            + ", ".join(missing)
        )

    return mid_to_name, name_to_mid


def audit_split(
    split_name,
    bbox_path,
    mid_to_name,
    name_to_mid,
):
    target_mids = {
        name_to_mid[x]
        for x in ALL_TARGETS
    }

    image_classes = defaultdict(set)
    image_box_counts = Counter()
    image_target_box_counts = Counter()

    total_rows = 0
    relevant_rows = 0

    with open(
        bbox_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "ImageID",
            "LabelName",
        }

        if not required.issubset(
            reader.fieldnames or []
        ):
            raise RuntimeError(
                f"{bbox_path} missing required columns."
            )

        for row in reader:
            total_rows += 1

            image_id = row["ImageID"].strip()
            mid = row["LabelName"].strip()

            image_box_counts[
                image_id
            ] += 1

            if mid not in target_mids:
                continue

            relevant_rows += 1
            image_target_box_counts[
                image_id
            ] += 1

            image_classes[
                image_id
            ].add(
                mid_to_name[mid]
            )

    candidate_rows = []
    pair_counts = Counter()

    for image_id, classes in image_classes.items():
        cls = sorted(classes)

        elephant = (
            ELEPHANT in classes
        )

        iv_present = sorted(
            classes.intersection(
                IN_VOCAB_CONFUSERS
            )
        )

        ood_present = sorted(
            classes.intersection(
                OOD_CONFUSERS
            )
        )

        other_targets = sorted(
            set(iv_present)
            |
            set(ood_present)
        )

        positive_mixed = bool(
            elephant
            and other_targets
        )

        negative_mixed = bool(
            (not elephant)
            and len(
                other_targets
            )
            >= 2
        )

        if elephant:
            for other in other_targets:
                pair_counts[
                    f"Elephant+{other}"
                ] += 1

        category = "OTHER"

        if positive_mixed:
            if iv_present and ood_present:
                category = (
                    "ELEPHANT_PLUS_IV_AND_OOD"
                )
            elif iv_present:
                category = (
                    "ELEPHANT_PLUS_IN_VOCAB"
                )
            elif ood_present:
                category = (
                    "ELEPHANT_PLUS_OOD"
                )

        elif negative_mixed:
            category = (
                "NEGATIVE_MULTI_ANIMAL_NO_ELEPHANT"
            )

        elif elephant:
            category = (
                "ELEPHANT_ONLY_AMONG_TARGET_CLASSES"
            )

        candidate_rows.append(
            {
                "split":
                    split_name,

                "image_id":
                    image_id,

                "category":
                    category,

                "elephant_present":
                    elephant,

                "in_vocab_confusers":
                    "|".join(
                        iv_present
                    ),

                "ood_confusers":
                    "|".join(
                        ood_present
                    ),

                "all_target_classes":
                    "|".join(
                        cls
                    ),

                "num_target_classes":
                    len(
                        cls
                    ),

                "total_boxes_all_openimages_classes":
                    int(
                        image_box_counts[
                            image_id
                        ]
                    ),

                "target_class_boxes":
                    int(
                        image_target_box_counts[
                            image_id
                        ]
                    ),

                "official_download_id":
                    f"{split_name}/{image_id}",
            }
        )

    candidates = pd.DataFrame(
        candidate_rows
    )

    pair_df = pd.DataFrame(
        [
            {
                "split":
                    split_name,

                "pair":
                    pair,

                "image_count":
                    count,
            }
            for pair, count in sorted(
                pair_counts.items()
            )
        ]
    )

    if len(candidates):
        category_counts = (
            candidates.groupby(
                "category"
            )
            .size()
            .rename(
                "image_count"
            )
            .reset_index()
        )

        category_counts[
            "split"
        ] = split_name

        category_counts = (
            category_counts[
                [
                    "split",
                    "category",
                    "image_count",
                ]
            ]
        )
    else:
        category_counts = pd.DataFrame(
            columns=[
                "split",
                "category",
                "image_count",
            ]
        )

    stats = {
        "split":
            split_name,

        "total_bbox_rows_scanned":
            total_rows,

        "target_bbox_rows":
            relevant_rows,

        "images_with_any_target_class":
            len(
                candidates
            ),

        "elephant_images":
            int(
                candidates[
                    "elephant_present"
                ].sum()
            )
            if len(
                candidates
            )
            else 0,

        "elephant_plus_any_confuser":
            int(
                candidates[
                    "category"
                ].isin(
                    [
                        "ELEPHANT_PLUS_IN_VOCAB",
                        "ELEPHANT_PLUS_OOD",
                        "ELEPHANT_PLUS_IV_AND_OOD",
                    ]
                ).sum()
            )
            if len(
                candidates
            )
            else 0,

        "elephant_plus_in_vocab":
            int(
                candidates[
                    "category"
                ].isin(
                    [
                        "ELEPHANT_PLUS_IN_VOCAB",
                        "ELEPHANT_PLUS_IV_AND_OOD",
                    ]
                ).sum()
            )
            if len(
                candidates
            )
            else 0,

        "elephant_plus_ood":
            int(
                candidates[
                    "category"
                ].isin(
                    [
                        "ELEPHANT_PLUS_OOD",
                        "ELEPHANT_PLUS_IV_AND_OOD",
                    ]
                ).sum()
            )
            if len(
                candidates
            )
            else 0,

        "negative_multi_animal_no_elephant":
            int(
                (
                    candidates[
                        "category"
                    ]
                    ==
                    "NEGATIVE_MULTI_ANIMAL_NO_ELEPHANT"
                ).sum()
            )
            if len(
                candidates
            )
            else 0,
    }

    return (
        candidates,
        pair_df,
        category_counts,
        stats,
    )


def main():
    print(
        "\nR17-A3 — OPEN IMAGES V7 EXTERNAL MIXED-SPECIES CANDIDATE AUDIT"
    )

    print(
        "METADATA SELECTION ONLY. NO model inference and NO image download. "
        "Uses official Open Images V7 boxable-class metadata plus official validation/test bounding-box annotations. "
        "The purpose is to verify that a real external evaluation pool exists before any image pixels are downloaded.\n"
    )

    verify_r17a2()

    for key in [
        "classes",
        "validation_boxes",
        "test_boxes",
    ]:
        download_if_needed(
            key
        )

    mid_to_name, name_to_mid = (
        load_class_map()
    )

    class_rows = []

    for name in ALL_TARGETS:
        class_rows.append(
            {
                "class_name":
                    name,

                "mid":
                    name_to_mid[
                        name
                    ],

                "role":
                    (
                        "TARGET"
                        if name == ELEPHANT
                        else
                        (
                            "IN_VOCAB_CONFUSER"
                            if name
                            in IN_VOCAB_CONFUSERS
                            else
                            "OOD_CONFUSER"
                        )
                    ),
            }
        )

    class_df = pd.DataFrame(
        class_rows
    )

    all_candidates = []
    all_pairs = []
    all_categories = []
    stats_rows = []

    for split_name, key in [
        (
            "validation",
            "validation_boxes",
        ),
        (
            "test",
            "test_boxes",
        ),
    ]:
        print(
            f"SCANNING {split_name.upper()} ..."
        )

        c, p, cat, stats = audit_split(
            split_name,
            FILES[key],
            mid_to_name,
            name_to_mid,
        )

        all_candidates.append(
            c
        )

        all_pairs.append(
            p
        )

        all_categories.append(
            cat
        )

        stats_rows.append(
            stats
        )

    candidates = pd.concat(
        all_candidates,
        ignore_index=True,
        sort=False,
    )

    pairs = pd.concat(
        all_pairs,
        ignore_index=True,
        sort=False,
    )

    categories = pd.concat(
        all_categories,
        ignore_index=True,
        sort=False,
    )

    stats_df = pd.DataFrame(
        stats_rows
    )

    positive_mask = candidates[
        "category"
    ].isin(
        [
            "ELEPHANT_PLUS_IN_VOCAB",
            "ELEPHANT_PLUS_OOD",
            "ELEPHANT_PLUS_IV_AND_OOD",
        ]
    )

    positive = candidates.loc[
        positive_mask
    ].copy()

    negative = candidates.loc[
        candidates[
            "category"
        ]
        ==
        "NEGATIVE_MULTI_ANIMAL_NO_ELEPHANT"
    ].copy()

    total_positive = int(
        len(
            positive
        )
    )

    total_iv = int(
        positive[
            "category"
        ].isin(
            [
                "ELEPHANT_PLUS_IN_VOCAB",
                "ELEPHANT_PLUS_IV_AND_OOD",
            ]
        ).sum()
    )

    total_ood = int(
        positive[
            "category"
        ].isin(
            [
                "ELEPHANT_PLUS_OOD",
                "ELEPHANT_PLUS_IV_AND_OOD",
            ]
        ).sum()
    )

    total_negative = int(
        len(
            negative
        )
    )

    # Conservative readiness thresholds for a useful external evaluation pool.
    main_pool_ready = bool(
        total_positive >= 50
        and total_negative >= 50
        and total_ood >= 25
    )

    iv_specific_ready = bool(
        total_iv >= 20
    )

    decision = pd.DataFrame(
        [
            {
                "external_source":
                    "Open Images V7 validation + test bounding-box annotations",

                "positive_mixed_elephant_images":
                    total_positive,

                "positive_with_in_vocab_confuser":
                    total_iv,

                "positive_with_ood_confuser":
                    total_ood,

                "negative_multi_animal_no_elephant_images":
                    total_negative,

                "main_external_pool_ready":
                    main_pool_ready,

                "in_vocab_specific_pool_ready":
                    iv_specific_ready,

                "decision":
                    (
                        "OPENIMAGES_EXTERNAL_POOL_READY_FOR_FREEZE_AND_PIXEL_DOWNLOAD"
                        if main_pool_ready
                        else
                        "OPENIMAGES_POOL_INSUFFICIENT_NEED_ADDITIONAL_EXTERNAL_SOURCE"
                    ),
            }
        ]
    )

    class_path = (
        OUT
        / "R17A3_TARGET_CLASS_MID_MAP.csv"
    )

    candidates_path = (
        OUT
        / "R17A3_ALL_TARGET_CLASS_CANDIDATES.csv"
    )

    positive_path = (
        OUT
        / "R17A3_POSITIVE_ELEPHANT_MIXED_CANDIDATES.csv"
    )

    negative_path = (
        OUT
        / "R17A3_NEGATIVE_MULTI_ANIMAL_CANDIDATES.csv"
    )

    pair_path = (
        OUT
        / "R17A3_ELEPHANT_PAIR_COUNTS.csv"
    )

    category_path = (
        OUT
        / "R17A3_CATEGORY_COUNTS.csv"
    )

    stats_path = (
        OUT
        / "R17A3_SPLIT_SCAN_STATS.csv"
    )

    decision_path = (
        OUT
        / "R17A3_DECISION_SUMMARY.csv"
    )

    class_df.to_csv(
        class_path,
        index=False,
    )

    candidates.to_csv(
        candidates_path,
        index=False,
    )

    positive.to_csv(
        positive_path,
        index=False,
    )

    negative.to_csv(
        negative_path,
        index=False,
    )

    pairs.to_csv(
        pair_path,
        index=False,
    )

    categories.to_csv(
        category_path,
        index=False,
    )

    stats_df.to_csv(
        stats_path,
        index=False,
    )

    decision.to_csv(
        decision_path,
        index=False,
    )

    if main_pool_ready:
        status = (
            "R17A3_OPENIMAGES_EXTERNAL_MIXED_SPECIES_POOL_VERIFIED_READY_FOR_R17A4_FROZEN_SUBSET_SELECTION_AND_DOWNLOAD"
        )
    else:
        status = (
            "R17A3_OPENIMAGES_POOL_INSUFFICIENT_ADDITIONAL_EXTERNAL_SOURCE_REQUIRED"
        )

    manifest = {
        "step":
            "R17A3_OPENIMAGES_EXTERNAL_MIXED_SPECIES_CANDIDATE_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "model_inference":
            False,

        "image_pixels_downloaded":
            False,

        "source":
            "Open Images V7",

        "source_urls":
            URLS,

        "metadata_sha256": {
            k:
                sha256_file(
                    v
                )
            for k, v in FILES.items()
        },

        "elephant_class":
            ELEPHANT,

        "in_vocab_confusers":
            IN_VOCAB_CONFUSERS,

        "ood_confusers":
            OOD_CONFUSERS,

        "positive_mixed_elephant_images":
            total_positive,

        "positive_with_in_vocab_confuser":
            total_iv,

        "positive_with_ood_confuser":
            total_ood,

        "negative_multi_animal_no_elephant_images":
            total_negative,

        "main_external_pool_ready":
            main_pool_ready,

        "in_vocab_specific_pool_ready":
            iv_specific_ready,

        "outputs": {
            "class_map":
                str(
                    class_path
                ),

            "all_candidates":
                str(
                    candidates_path
                ),

            "positive_candidates":
                str(
                    positive_path
                ),

            "negative_candidates":
                str(
                    negative_path
                ),

            "pair_counts":
                str(
                    pair_path
                ),

            "category_counts":
                str(
                    category_path
                ),

            "split_stats":
                str(
                    stats_path
                ),

            "decision_summary":
                str(
                    decision_path
                ),
        },
    }

    (
        OUT
        / "R17A3_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "CLASS MAP:"
    )
    print(
        class_df.to_string(
            index=False
        )
    )

    print(
        "\nSPLIT SCAN STATS:"
    )
    print(
        stats_df.to_string(
            index=False
        )
    )

    print(
        "\nCATEGORY COUNTS:"
    )
    print(
        categories.to_string(
            index=False
        )
    )

    print(
        "\nELEPHANT + CONFUSER PAIR COUNTS:"
    )
    if len(pairs):
        print(
            pairs.to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nDECISION SUMMARY:"
    )
    print(
        decision.to_string(
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
        "\nR17-A3 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
