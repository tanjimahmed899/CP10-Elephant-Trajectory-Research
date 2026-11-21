
from pathlib import Path
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd

HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
PROJECT = Path(r"E:\AIUB\Capstone Project\capstone\wildlife_project")

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

R17A_MANIFEST = (
    R17_ROOT
    / "R17A_mixed_species_asset_provenance_audit"
    / "R17A_MANIFEST.json"
)

OUT = (
    R17_ROOT
    / "R17A2_existing_dataset_cooccurrence_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

# Main datasets found by R17-A.
DATASET_ROOTS = [
    PROJECT / "V5",
    PROJECT / "exported_dataset" / "project-2-dataset-instance1786346684.783013-ultralytics-_yolo-_detection-10",
]

CHECKPOINT_CLASS_NAMES = [
    "tiger",
    "lion",
    "elephant",
    "crocodile",
]

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
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


def verify_r17a():
    require(R17A_MANIFEST)

    m = json.loads(
        R17A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R17A_MIXED_SPECIES_CORE_ASSETS_FOUND_CHECKPOINT_CLASSES_AND_MEDIA_AUDITED_READY_FOR_R17B_FROZEN_ROBUSTNESS_PROTOCOL_LOCK"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R17-A status: {m.get('status')}"
        )

    classes = [
        x.strip().lower()
        for x in str(
            m.get(
                "selected_model_classes",
                "",
            )
        ).split("|")
        if x.strip()
    ]

    if classes and classes != CHECKPOINT_CLASS_NAMES:
        raise RuntimeError(
            "R17-A selected checkpoint class order differs from expected "
            f"{CHECKPOINT_CLASS_NAMES}: {classes}"
        )

    return m


def image_index(images_dir):
    idx = {}

    if not images_dir.exists():
        return idx

    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        idx[p.stem] = p

    return idx


def parse_label_file(label_path):
    class_ids = []
    n_boxes = 0
    malformed = 0

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    except Exception:
        return [], 0, 1

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            malformed += 1
            continue

        try:
            cid = int(float(parts[0]))
        except Exception:
            malformed += 1
            continue

        class_ids.append(cid)
        n_boxes += 1

    return class_ids, n_boxes, malformed


def class_name(cid):
    if 0 <= cid < len(CHECKPOINT_CLASS_NAMES):
        return CHECKPOINT_CLASS_NAMES[cid]

    return f"UNKNOWN_CLASS_{cid}"


def audit_dataset(dataset_root):
    rows = []
    split_summaries = []
    pair_counter = Counter()

    if not dataset_root.exists():
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    possible_splits = [
        ("Train", "Train"),
        ("Validation", "Validation"),
        ("Test", "Test"),
        ("train", "train"),
        ("valid", "valid"),
        ("val", "val"),
        ("test", "test"),
    ]

    seen_split_keys = set()

    for image_split, label_split in possible_splits:
        images_dir = dataset_root / "images" / image_split
        labels_dir = dataset_root / "labels" / label_split

        if not images_dir.exists() and not labels_dir.exists():
            continue

        key = (
            str(images_dir).lower(),
            str(labels_dir).lower(),
        )

        if key in seen_split_keys:
            continue

        seen_split_keys.add(key)

        images = image_index(images_dir)

        label_files = (
            sorted(labels_dir.rglob("*.txt"))
            if labels_dir.exists()
            else []
        )

        all_stems = sorted(
            set(images.keys())
            |
            {p.stem for p in label_files}
        )

        label_map = {
            p.stem: p
            for p in label_files
        }

        split_counts = {
            "dataset_root": str(dataset_root),
            "split": image_split,
            "image_count": len(images),
            "label_file_count": len(label_files),
            "labeled_images": 0,
            "empty_label_images": 0,
            "total_boxes": 0,
            "multi_object_images": 0,
            "multi_class_images": 0,
            "elephant_images": 0,
            "elephant_only_images": 0,
            "elephant_plus_other_images": 0,
            "non_elephant_only_images": 0,
            "malformed_label_rows": 0,
        }

        for stem in all_stems:
            label_path = label_map.get(stem)
            image_path = images.get(stem)

            if label_path is None:
                class_ids = []
                n_boxes = 0
                malformed = 0
            else:
                class_ids, n_boxes, malformed = parse_label_file(
                    label_path
                )

            unique_ids = sorted(set(class_ids))
            names = [
                class_name(x)
                for x in unique_ids
            ]

            unique_names = sorted(set(names))

            elephant_present = (
                "elephant" in unique_names
            )

            other_present = any(
                x != "elephant"
                for x in unique_names
            )

            is_multi_object = n_boxes > 1
            is_multi_class = len(unique_names) > 1
            is_elephant_plus_other = (
                elephant_present
                and other_present
            )

            if label_path is not None:
                split_counts["labeled_images"] += 1

            if label_path is not None and n_boxes == 0:
                split_counts["empty_label_images"] += 1

            split_counts["total_boxes"] += n_boxes
            split_counts["malformed_label_rows"] += malformed

            if is_multi_object:
                split_counts["multi_object_images"] += 1

            if is_multi_class:
                split_counts["multi_class_images"] += 1

            if elephant_present:
                split_counts["elephant_images"] += 1

            if elephant_present and not other_present:
                split_counts["elephant_only_images"] += 1

            if is_elephant_plus_other:
                split_counts["elephant_plus_other_images"] += 1

            if (
                not elephant_present
                and len(unique_names) > 0
            ):
                split_counts["non_elephant_only_images"] += 1

            if is_elephant_plus_other:
                for other in unique_names:
                    if other == "elephant":
                        continue
                    pair_counter[
                        (
                            str(dataset_root),
                            image_split,
                            f"elephant+{other}",
                        )
                    ] += 1

            rows.append(
                {
                    "dataset_root": str(dataset_root),
                    "split": image_split,
                    "image_stem": stem,
                    "image_path": (
                        str(image_path)
                        if image_path is not None
                        else ""
                    ),
                    "label_path": (
                        str(label_path)
                        if label_path is not None
                        else ""
                    ),
                    "num_boxes": n_boxes,
                    "num_unique_classes": len(unique_names),
                    "classes_present": "|".join(unique_names),
                    "elephant_present": elephant_present,
                    "non_elephant_present": other_present,
                    "multi_object": is_multi_object,
                    "multi_class": is_multi_class,
                    "elephant_plus_other": is_elephant_plus_other,
                    "malformed_rows": malformed,
                }
            )

        split_summaries.append(
            split_counts
        )

    pair_rows = [
        {
            "dataset_root": dataset,
            "split": split,
            "pair": pair,
            "image_count": count,
        }
        for (
            dataset,
            split,
            pair,
        ), count in sorted(
            pair_counter.items()
        )
    ]

    return (
        pd.DataFrame(rows),
        pd.DataFrame(split_summaries),
        pd.DataFrame(pair_rows),
    )


def main():
    print(
        "\nR17-A2 — EXISTING DATASET CO-OCCURRENCE AUDIT"
    )

    print(
        "AUDIT ONLY. NO training and NO inference. "
        "This step checks whether the existing YOLO datasets actually contain same-image mixed-species scenes, "
        "especially elephant + another detector class. It does not assume that a multi-class dataset automatically "
        "contains co-occurrence scenes.\n"
    )

    r17a = verify_r17a()

    all_rows = []
    all_summaries = []
    all_pairs = []

    for root in DATASET_ROOTS:
        rows, summary, pairs = audit_dataset(
            root
        )

        if len(rows):
            all_rows.append(rows)

        if len(summary):
            all_summaries.append(summary)

        if len(pairs):
            all_pairs.append(pairs)

    if all_rows:
        image_df = pd.concat(
            all_rows,
            ignore_index=True,
        )
    else:
        image_df = pd.DataFrame()

    if all_summaries:
        split_df = pd.concat(
            all_summaries,
            ignore_index=True,
        )
    else:
        split_df = pd.DataFrame()

    if all_pairs:
        pair_df = pd.concat(
            all_pairs,
            ignore_index=True,
        )
    else:
        pair_df = pd.DataFrame(
            columns=[
                "dataset_root",
                "split",
                "pair",
                "image_count",
            ]
        )

    total_images = int(
        split_df[
            "image_count"
        ].sum()
    ) if len(split_df) else 0

    total_multi_class = int(
        split_df[
            "multi_class_images"
        ].sum()
    ) if len(split_df) else 0

    total_elephant_images = int(
        split_df[
            "elephant_images"
        ].sum()
    ) if len(split_df) else 0

    total_elephant_plus_other = int(
        split_df[
            "elephant_plus_other_images"
        ].sum()
    ) if len(split_df) else 0

    cooccurrence_rate_among_elephant = (
        total_elephant_plus_other
        /
        total_elephant_images
        if total_elephant_images > 0
        else 0.0
    )

    # Conservative decision:
    # >=20 genuine same-image elephant+other scenes = minimally usable main evaluation pool.
    # 5-19 = too small for main claim, but useful supplementary.
    # <5 = effectively absent.
    if total_elephant_plus_other >= 20:
        dataset_decision = (
            "EXISTING_DATASET_HAS_USABLE_ELEPHANT_PLUS_OTHER_COOCCURRENCE"
        )
        new_external_dataset_required_for_main_test = False
    elif total_elephant_plus_other >= 5:
        dataset_decision = (
            "EXISTING_COOCCURRENCE_IS_SMALL_SUPPLEMENTARY_ONLY"
        )
        new_external_dataset_required_for_main_test = True
    else:
        dataset_decision = (
            "EXISTING_COOCCURRENCE_EFFECTIVELY_ABSENT"
        )
        new_external_dataset_required_for_main_test = True

    summary = pd.DataFrame(
        [
            {
                "checkpoint_classes": "|".join(
                    CHECKPOINT_CLASS_NAMES
                ),
                "audited_dataset_roots": len(DATASET_ROOTS),
                "total_images": total_images,
                "total_multi_class_images": total_multi_class,
                "total_elephant_images": total_elephant_images,
                "total_elephant_plus_other_images": total_elephant_plus_other,
                "elephant_plus_other_rate_among_elephant_images": cooccurrence_rate_among_elephant,
                "decision": dataset_decision,
                "new_external_dataset_required_for_main_test": new_external_dataset_required_for_main_test,
            }
        ]
    )

    image_path = (
        OUT
        / "R17A2_IMAGE_LEVEL_COOCCURRENCE_AUDIT.csv"
    )
    split_path = (
        OUT
        / "R17A2_SPLIT_LEVEL_COOCCURRENCE_SUMMARY.csv"
    )
    pair_path = (
        OUT
        / "R17A2_ELEPHANT_PAIR_COUNTS.csv"
    )
    summary_path = (
        OUT
        / "R17A2_DECISION_SUMMARY.csv"
    )

    image_df.to_csv(
        image_path,
        index=False,
    )
    split_df.to_csv(
        split_path,
        index=False,
    )
    pair_df.to_csv(
        pair_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    examples = (
        image_df.loc[
            image_df[
                "elephant_plus_other"
            ].astype(bool)
        ].copy()
        if len(image_df)
        else pd.DataFrame()
    )

    examples_path = (
        OUT
        / "R17A2_ELEPHANT_PLUS_OTHER_EXAMPLES.csv"
    )

    examples.head(
        100
    ).to_csv(
        examples_path,
        index=False,
    )

    if new_external_dataset_required_for_main_test:
        status = (
            "R17A2_EXISTING_COOCCURRENCE_INSUFFICIENT_EXTERNAL_MIXED_SPECIES_EVALUATION_DATASET_REQUIRED_BEFORE_R17B_PROTOCOL_LOCK"
        )
    else:
        status = (
            "R17A2_EXISTING_COOCCURRENCE_SUFFICIENT_READY_FOR_R17B_FROZEN_MIXED_SPECIES_PROTOCOL_LOCK"
        )

    manifest = {
        "step":
            "R17A2_EXISTING_DATASET_COOCCURRENCE_AUDIT",
        "created_utc":
            utc_now(),
        "status":
            status,
        "training":
            False,
        "inference":
            False,
        "checkpoint_classes":
            CHECKPOINT_CLASS_NAMES,
        "total_elephant_images":
            total_elephant_images,
        "total_elephant_plus_other_images":
            total_elephant_plus_other,
        "cooccurrence_rate_among_elephant_images":
            cooccurrence_rate_among_elephant,
        "decision":
            dataset_decision,
        "new_external_dataset_required_for_main_test":
            new_external_dataset_required_for_main_test,
        "important_interpretation":
            [
                "A multi-class checkpoint does not prove same-image mixed-species robustness.",
                "Cow/deer/dog/horse and BAMBI deer/human are out-of-class negative confusers for this 4-class checkpoint, not named supported classes.",
                "If existing elephant+other co-occurrence is insufficient, acquire/freeze an external mixed-species evaluation dataset before robustness inference.",
            ],
        "outputs": {
            "image_level":
                str(image_path),
            "split_level":
                str(split_path),
            "pair_counts":
                str(pair_path),
            "examples":
                str(examples_path),
            "decision_summary":
                str(summary_path),
        },
    }

    (
        OUT
        / "R17A2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "DECISION SUMMARY:"
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSPLIT-LEVEL SUMMARY:"
    )
    if len(split_df):
        print(
            split_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No auditable dataset split found."
        )

    print(
        "\nELEPHANT + OTHER CLASS PAIR COUNTS:"
    )
    if len(pair_df):
        print(
            pair_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No elephant+other same-image co-occurrence found."
        )

    print(
        "\nEXAMPLE COOCCURRENCE IMAGES:"
    )
    if len(examples):
        cols = [
            "dataset_root",
            "split",
            "image_path",
            "num_boxes",
            "classes_present",
        ]
        print(
            examples[
                cols
            ].head(
                30
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nSTATUS:"
    )
    print(
        status
    )

    print(
        "\nR17-A2 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
