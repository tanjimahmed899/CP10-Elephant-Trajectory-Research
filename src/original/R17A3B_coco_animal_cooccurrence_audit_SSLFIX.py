
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

R17A3_MANIFEST = (
    R17_ROOT
    / "R17A3_openimages_external_candidate_audit"
    / "R17A3_MANIFEST.json"
)

OUT = (
    R17_ROOT
    / "R17A3B_coco_animal_cooccurrence_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

META = OUT / "coco_metadata"
META.mkdir(parents=True, exist_ok=True)

ANNOT_ZIP_URL = (
    "https://storage.googleapis.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

# Published checksum for annotations_trainval2017.zip (TensorFlow Datasets catalog/checksum file).
EXPECTED_ANNOT_ZIP_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
EXPECTED_ANNOT_ZIP_SIZE = 252907541

ANNOT_ZIP = META / "annotations_trainval2017.zip"
TRAIN_JSON = META / "annotations" / "instances_train2017.json"
VAL_JSON = META / "annotations" / "instances_val2017.json"

TARGET = "elephant"

ANIMAL_CONFUSERS = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "bear",
    "zebra",
    "giraffe",
]

# Person is tracked separately as a practical non-animal confuser,
# but does NOT count toward the primary animal-among-animals readiness decision.
PRACTICAL_NON_ANIMAL = [
    "person",
]


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


def verify_r17a3():
    require(R17A3_MANIFEST)

    m = json.loads(
        R17A3_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R17A3_OPENIMAGES_EXTERNAL_MIXED_SPECIES_POOL_VERIFIED_READY_FOR_R17A4_FROZEN_SUBSET_SELECTION_AND_DOWNLOAD"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R17-A3 status: {m.get('status')}"
        )

    return m


def download_annotations():
    def verify_zip_integrity(path):
        size = path.stat().st_size
        digest = sha256_file(path)

        if size != EXPECTED_ANNOT_ZIP_SIZE:
            raise RuntimeError(
                f"COCO annotation ZIP size mismatch: got {size}, expected {EXPECTED_ANNOT_ZIP_SIZE}"
            )

        if digest != EXPECTED_ANNOT_ZIP_SHA256:
            raise RuntimeError(
                "COCO annotation ZIP SHA256 mismatch: "
                f"got {digest}, expected {EXPECTED_ANNOT_ZIP_SHA256}"
            )

        print(
            "COCO ANNOTATION ZIP INTEGRITY PASS:"
        )
        print(
            f"  size={size} bytes"
        )
        print(
            f"  sha256={digest}"
        )

    if ANNOT_ZIP.exists() and ANNOT_ZIP.stat().st_size > 0:
        try:
            verify_zip_integrity(ANNOT_ZIP)
            print(
                f"USING VERIFIED EXISTING COCO ANNOTATION ZIP: {ANNOT_ZIP}"
            )
        except Exception:
            print(
                "Existing COCO annotation ZIP is incomplete/invalid; deleting and re-downloading."
            )
            ANNOT_ZIP.unlink()

    if not ANNOT_ZIP.exists():
        print(
            f"DOWNLOADING COCO 2017 ANNOTATIONS VIA GOOGLE CLOUD STORAGE:\n{ANNOT_ZIP_URL}"
        )
        print(
            f"TO: {ANNOT_ZIP}"
        )

        tmp = ANNOT_ZIP.with_suffix(
            ".zip.part"
        )

        if tmp.exists():
            tmp.unlink()

        try:
            urllib.request.urlretrieve(
                ANNOT_ZIP_URL,
                tmp,
            )
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        verify_zip_integrity(tmp)
        tmp.replace(
            ANNOT_ZIP
        )

    if not TRAIN_JSON.exists() or not VAL_JSON.exists():
        print(
            "EXTRACTING instances_train2017.json and instances_val2017.json ..."
        )

        with zipfile.ZipFile(
            ANNOT_ZIP,
            "r",
        ) as zf:
            wanted = {
                "annotations/instances_train2017.json",
                "annotations/instances_val2017.json",
            }

            members = set(
                zf.namelist()
            )

            missing = wanted - members

            if missing:
                raise RuntimeError(
                    f"COCO annotation ZIP missing expected files: {sorted(missing)}"
                )

            for member in sorted(wanted):
                zf.extract(
                    member,
                    META,
                )


def audit_split(split_name, json_path):
    print(
        f"LOADING {split_name.upper()} ANNOTATIONS ..."
    )

    data = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    categories = {
        int(c["id"]): str(c["name"]).strip().lower()
        for c in data["categories"]
    }

    name_to_id = {
        name: cid
        for cid, name in categories.items()
    }

    required_names = (
        [TARGET]
        + ANIMAL_CONFUSERS
        + PRACTICAL_NON_ANIMAL
    )

    missing = [
        x
        for x in required_names
        if x not in name_to_id
    ]

    if missing:
        raise RuntimeError(
            f"COCO split missing expected categories: {missing}"
        )

    image_meta = {
        int(i["id"]): i
        for i in data["images"]
    }

    image_classes = {}
    image_box_count = Counter()

    target_ids = {
        name_to_id[x]
        for x in required_names
    }

    for ann in data["annotations"]:
        image_id = int(
            ann["image_id"]
        )
        cid = int(
            ann["category_id"]
        )

        if cid not in target_ids:
            continue

        image_classes.setdefault(
            image_id,
            set(),
        ).add(
            categories[cid]
        )

        image_box_count[
            image_id
        ] += 1

    rows = []
    pair_counts = Counter()

    for image_id, cls_set in image_classes.items():
        elephant_present = (
            TARGET in cls_set
        )

        animal_present = sorted(
            set(
                ANIMAL_CONFUSERS
            ).intersection(
                cls_set
            )
        )

        person_present = (
            "person" in cls_set
        )

        primary_animal_mixed = bool(
            elephant_present
            and len(
                animal_present
            )
            > 0
        )

        practical_mixed = bool(
            elephant_present
            and person_present
        )

        negative_multi_animal = bool(
            (not elephant_present)
            and len(
                animal_present
            )
            >= 2
        )

        elephant_only_among_animal_targets = bool(
            elephant_present
            and len(
                animal_present
            )
            == 0
        )

        for other in animal_present:
            if elephant_present:
                pair_counts[
                    f"elephant+{other}"
                ] += 1

        meta = image_meta[
            image_id
        ]

        rows.append(
            {
                "split":
                    split_name,

                "image_id":
                    image_id,

                "file_name":
                    str(
                        meta.get(
                            "file_name",
                            "",
                        )
                    ),

                "width":
                    int(
                        meta.get(
                            "width",
                            0,
                        )
                    ),

                "height":
                    int(
                        meta.get(
                            "height",
                            0,
                        )
                    ),

                "elephant_present":
                    elephant_present,

                "animal_confusers":
                    "|".join(
                        animal_present
                    ),

                "num_animal_confuser_classes":
                    len(
                        animal_present
                    ),

                "person_present":
                    person_present,

                "primary_elephant_plus_animal":
                    primary_animal_mixed,

                "elephant_plus_person":
                    practical_mixed,

                "negative_multi_animal_no_elephant":
                    negative_multi_animal,

                "elephant_only_among_animal_targets":
                    elephant_only_among_animal_targets,

                "target_related_box_count":
                    int(
                        image_box_count[
                            image_id
                        ]
                    ),

                "image_url":
                    (
                        f"https://images.cocodataset.org/"
                        f"{'train2017' if split_name == 'train2017' else 'val2017'}/"
                        f"{meta.get('file_name','')}"
                    ),
            }
        )

    df = pd.DataFrame(
        rows
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

    stats = {
        "split":
            split_name,

        "total_images_in_split":
            len(
                data["images"]
            ),

        "images_with_target_related_classes":
            len(
                df
            ),

        "elephant_images":
            int(
                df[
                    "elephant_present"
                ].sum()
            )
            if len(
                df
            )
            else 0,

        "elephant_plus_animal_images":
            int(
                df[
                    "primary_elephant_plus_animal"
                ].sum()
            )
            if len(
                df
            )
            else 0,

        "elephant_plus_person_images":
            int(
                df[
                    "elephant_plus_person"
                ].sum()
            )
            if len(
                df
            )
            else 0,

        "negative_multi_animal_no_elephant":
            int(
                df[
                    "negative_multi_animal_no_elephant"
                ].sum()
            )
            if len(
                df
            )
            else 0,

        "elephant_only_among_animal_targets":
            int(
                df[
                    "elephant_only_among_animal_targets"
                ].sum()
            )
            if len(
                df
            )
            else 0,
    }

    return (
        df,
        pair_df,
        stats,
        categories,
    )


def main():
    print(
        "\nR17-A3B — COCO 2017 ANIMAL–ANIMAL CO-OCCURRENCE AUDIT"
    )

    print(
        "METADATA ONLY. NO model inference and NO image pixel download. "
        "Uses official COCO 2017 train/validation instance annotations to find same-image scenes "
        "where elephant co-occurs with another animal class. Person is reported separately and "
        "does not count toward the primary animal-among-animals readiness decision.\n"
    )

    verify_r17a3()
    download_annotations()

    all_rows = []
    all_pairs = []
    stats_rows = []
    category_reference = None

    for split_name, path in [
        (
            "train2017",
            TRAIN_JSON,
        ),
        (
            "val2017",
            VAL_JSON,
        ),
    ]:
        df, pairs, stats, categories = audit_split(
            split_name,
            path,
        )

        all_rows.append(
            df
        )

        all_pairs.append(
            pairs
        )

        stats_rows.append(
            stats
        )

        if category_reference is None:
            category_reference = categories

    candidates = pd.concat(
        all_rows,
        ignore_index=True,
        sort=False,
    )

    pair_df = pd.concat(
        all_pairs,
        ignore_index=True,
        sort=False,
    )

    stats_df = pd.DataFrame(
        stats_rows
    )

    positive = candidates.loc[
        candidates[
            "primary_elephant_plus_animal"
        ].astype(bool)
    ].copy()

    negative = candidates.loc[
        candidates[
            "negative_multi_animal_no_elephant"
        ].astype(bool)
    ].copy()

    elephant_only = candidates.loc[
        candidates[
            "elephant_only_among_animal_targets"
        ].astype(bool)
    ].copy()

    total_positive = int(
        len(
            positive
        )
    )

    total_negative = int(
        len(
            negative
        )
    )

    total_elephant_only = int(
        len(
            elephant_only
        )
    )

    unique_pairs = int(
        pair_df[
            "pair"
        ].nunique()
    ) if len(
        pair_df
    ) else 0

    # Conservative readiness:
    # >=50 elephant+animal scenes, >=5 distinct animal-confuser pairs,
    # >=50 negative multi-animal scenes, >=50 elephant-only controls.
    main_ready = bool(
        total_positive >= 50
        and unique_pairs >= 5
        and total_negative >= 50
        and total_elephant_only >= 50
    )

    decision = pd.DataFrame(
        [
            {
                "external_source":
                    "COCO 2017 train+val public instance annotations",

                "positive_elephant_plus_animal_images":
                    total_positive,

                "distinct_elephant_animal_pairs":
                    unique_pairs,

                "negative_multi_animal_no_elephant_images":
                    total_negative,

                "elephant_only_control_images":
                    total_elephant_only,

                "main_animal_among_animals_pool_ready":
                    main_ready,

                "decision":
                    (
                        "COCO_ANIMAL_COOCCURRENCE_POOL_READY_FOR_FROZEN_SUBSET_SELECTION"
                        if main_ready
                        else
                        "COCO_POOL_INSUFFICIENT_NEED_ADDITIONAL_ANIMAL_COOCCURRENCE_SOURCE"
                    ),
            }
        ]
    )

    class_rows = []

    for cid, name in sorted(
        category_reference.items()
    ):
        if (
            name == TARGET
            or name in ANIMAL_CONFUSERS
            or name in PRACTICAL_NON_ANIMAL
        ):
            class_rows.append(
                {
                    "category_id":
                        cid,

                    "class_name":
                        name,

                    "role":
                        (
                            "TARGET"
                            if name == TARGET
                            else
                            (
                                "ANIMAL_CONFUSER"
                                if name
                                in ANIMAL_CONFUSERS
                                else
                                "PRACTICAL_NON_ANIMAL"
                            )
                        ),
                }
            )

    class_df = pd.DataFrame(
        class_rows
    )

    class_path = (
        OUT
        /
        "R17A3B_COCO_TARGET_CLASS_MAP.csv"
    )

    candidates_path = (
        OUT
        /
        "R17A3B_ALL_CANDIDATES.csv"
    )

    positive_path = (
        OUT
        /
        "R17A3B_ELEPHANT_PLUS_ANIMAL_CANDIDATES.csv"
    )

    negative_path = (
        OUT
        /
        "R17A3B_NEGATIVE_MULTI_ANIMAL_CANDIDATES.csv"
    )

    controls_path = (
        OUT
        /
        "R17A3B_ELEPHANT_ONLY_CONTROLS.csv"
    )

    pair_path = (
        OUT
        /
        "R17A3B_ELEPHANT_ANIMAL_PAIR_COUNTS.csv"
    )

    stats_path = (
        OUT
        /
        "R17A3B_SPLIT_STATS.csv"
    )

    decision_path = (
        OUT
        /
        "R17A3B_DECISION_SUMMARY.csv"
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

    elephant_only.to_csv(
        controls_path,
        index=False,
    )

    pair_df.to_csv(
        pair_path,
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

    if main_ready:
        status = (
            "R17A3B_COCO_ANIMAL_AMONG_ANIMALS_POOL_VERIFIED_READY_FOR_R17A4_FROZEN_SUBSET_SELECTION_AND_PIXEL_DOWNLOAD"
        )
    else:
        status = (
            "R17A3B_COCO_ANIMAL_COOCCURRENCE_POOL_INSUFFICIENT_ADDITIONAL_SOURCE_REQUIRED"
        )

    manifest = {
        "step":
            "R17A3B_COCO_ANIMAL_COOCCURRENCE_AUDIT",

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
            "COCO 2017",

        "annotation_zip_url":
            ANNOT_ZIP_URL,

        "annotation_zip_expected_sha256":
            EXPECTED_ANNOT_ZIP_SHA256,

        "annotation_zip_expected_size_bytes":
            EXPECTED_ANNOT_ZIP_SIZE,

        "annotation_zip_sha256":
            sha256_file(
                ANNOT_ZIP
            ),

        "train_json_sha256":
            sha256_file(
                TRAIN_JSON
            ),

        "val_json_sha256":
            sha256_file(
                VAL_JSON
            ),

        "target":
            TARGET,

        "animal_confusers":
            ANIMAL_CONFUSERS,

        "person_reported_separately":
            True,

        "positive_elephant_plus_animal_images":
            total_positive,

        "distinct_elephant_animal_pairs":
            unique_pairs,

        "negative_multi_animal_no_elephant_images":
            total_negative,

        "elephant_only_control_images":
            total_elephant_only,

        "main_pool_ready":
            main_ready,

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

            "elephant_only_controls":
                str(
                    controls_path
                ),

            "pair_counts":
                str(
                    pair_path
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
        /
        "R17A3B_MANIFEST.json"
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
        "\nSPLIT STATS:"
    )
    print(
        stats_df.to_string(
            index=False
        )
    )

    print(
        "\nELEPHANT + ANIMAL PAIR COUNTS:"
    )
    if len(
        pair_df
    ):
        print(
            pair_df.to_string(
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
        "\nR17-A3B COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
