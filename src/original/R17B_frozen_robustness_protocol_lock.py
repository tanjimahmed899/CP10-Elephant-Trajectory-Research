
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

# Upstream frozen artifacts
A4A_DIR = R17_ROOT / "R17A4A_frozen_subset_selection"
A4A_MANIFEST = A4A_DIR / "R17A4A_MANIFEST.json"
PRIMARY_453 = A4A_DIR / "R17A4A_PRIMARY_BALANCED_453.csv"
SUPP_9 = A4A_DIR / "R17A4A_SUPPLEMENTARY_IN_VOCAB_UNIQUE.csv"

A4B_DIR = R17_ROOT / "R17A4B_frozen_pixel_download"
A4B_MANIFEST = A4B_DIR / "R17A4B_MANIFEST.json"
A4B_AUDIT = A4B_DIR / "R17A4B_PIXEL_DOWNLOAD_AUDIT.csv"

A4C_DIR = R17_ROOT / "R17A4C_detector_dataset_leakage_audit"
A4C_MANIFEST = A4C_DIR / "R17A4C_MANIFEST.json"

A3B_DIR = R17_ROOT / "R17A3B_coco_animal_cooccurrence_audit"
A3B_MANIFEST = A3B_DIR / "R17A3B_MANIFEST.json"
TRAIN_JSON = A3B_DIR / "coco_metadata" / "annotations" / "instances_train2017.json"
VAL_JSON = A3B_DIR / "coco_metadata" / "annotations" / "instances_val2017.json"

A2_DIR = R17_ROOT / "R17A2_existing_dataset_cooccurrence_audit"
A2_IMAGE_LEVEL = A2_DIR / "R17A2_IMAGE_LEVEL_COOCCURRENCE_AUDIT.csv"

# Detectors
SELECTIVE_CKPT = Path(
    r"E:\AIUB\Capstone Project\capstone\wildlife_project\wildlife_training\four_animals_v5_final_50\weights\best.pt"
)

ELEPHANT_ONLY_CKPT = Path(
    r"E:\AIUB\Capstone Project\Final 1.0\Elephant Detection.v2-resized640.yolov8 (2)\Elephant Detection.v2-resized640.yolov8\runs\detect\elephant_final_clean\weights\best.pt"
)

OUT = R17_ROOT / "R17B_frozen_robustness_protocol_lock"
OUT.mkdir(parents=True, exist_ok=True)

# Frozen inference/evaluation settings
IMGSZ = 640
INFERENCE_CONF_FLOOR = 0.05
PRIMARY_CONF = 0.25
SENSITIVITY_CONFS = [0.10, 0.25, 0.50]
NMS_IOU = 0.70
MAX_DET = 300
AGNOSTIC_NMS = False
AUGMENT = False

MATCH_IOU_PRIMARY = 0.50
AP_IOU_GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20261030

SELECTIVE_EXPECTED_CLASSES = [
    "tiger",
    "lion",
    "elephant",
    "crocodile",
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


def read_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def verify_upstream():
    for p in [
        A4A_MANIFEST,
        PRIMARY_453,
        SUPP_9,
        A4B_MANIFEST,
        A4B_AUDIT,
        A4C_MANIFEST,
        A3B_MANIFEST,
        TRAIN_JSON,
        VAL_JSON,
        A2_IMAGE_LEVEL,
        SELECTIVE_CKPT,
        ELEPHANT_ONLY_CKPT,
    ]:
        require(p)

    a4a = read_json(A4A_MANIFEST)
    a4b = read_json(A4B_MANIFEST)
    a4c = read_json(A4C_MANIFEST)
    a3b = read_json(A3B_MANIFEST)

    expected_a4a = (
        "R17A4A_FROZEN_EVALUATION_SUBSET_LOCKED_453_COCO_PRIMARY_PLUS_DEDUPLICATED_IN_VOCAB_SUPPLEMENT_READY_FOR_R17A4B_PIXEL_DOWNLOAD"
    )
    expected_a4b = (
        "R17A4B_FROZEN_453_COCO_PIXELS_DOWNLOADED_VERIFIED_AND_SHA256_LOCKED_READY_FOR_R17B_ROBUSTNESS_PROTOCOL_LOCK"
    )
    expected_a4c = (
        "R17A4C_NO_EXACT_OR_FLAGGED_NEAR_DUPLICATE_OVERLAP_READY_FOR_R17B_FROZEN_ROBUSTNESS_PROTOCOL_LOCK"
    )
    expected_a3b = (
        "R17A3B_COCO_ANIMAL_AMONG_ANIMALS_POOL_VERIFIED_READY_FOR_R17A4_FROZEN_SUBSET_SELECTION_AND_PIXEL_DOWNLOAD"
    )

    checks = [
        ("R17-A4A", a4a.get("status"), expected_a4a),
        ("R17-A4B", a4b.get("status"), expected_a4b),
        ("R17-A4C", a4c.get("status"), expected_a4c),
        ("R17-A3B", a3b.get("status"), expected_a3b),
    ]

    for name, got, expected in checks:
        if got != expected:
            raise RuntimeError(
                f"{name} status mismatch: {got}"
            )

    if sha256_file(PRIMARY_453) != a4a["output_sha256"]["primary"]:
        raise RuntimeError(
            "Frozen 453-row subset hash mismatch."
        )

    if int(a4b.get("valid_pixel_files", -1)) != 453:
        raise RuntimeError(
            "R17-A4B does not have 453 valid pixel files."
        )

    if int(a4b.get("failed_pixel_files", -1)) != 0:
        raise RuntimeError(
            "R17-A4B contains failed downloads."
        )

    if int(a4c.get("exact_sha256_overlap_pairs", -1)) != 0:
        raise RuntimeError(
            "External-set exact leakage was found."
        )

    if int(a4c.get("perceptual_candidate_pairs", -1)) != 0:
        raise RuntimeError(
            "External-set perceptual candidates remain unresolved."
        )

    return a4a, a4b, a4c, a3b


def load_model_metadata(checkpoint):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required only to read checkpoint metadata."
        ) from exc

    model = YOLO(str(checkpoint))

    names = model.names

    if isinstance(names, dict):
        ordered = [
            str(names[k]).strip()
            for k in sorted(names)
        ]
    else:
        ordered = [
            str(x).strip()
            for x in names
        ]

    return {
        "checkpoint": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "num_classes": len(ordered),
        "class_names": ordered,
    }


def load_coco_subset_gt():
    selected = pd.read_csv(
        PRIMARY_453,
        low_memory=False,
    )

    selected["image_id"] = pd.to_numeric(
        selected["image_id"],
        errors="raise",
    ).astype(int)

    selected["split"] = (
        selected["split"]
        .astype(str)
        .str.strip()
    )

    if len(selected) != 453:
        raise RuntimeError(
            f"Expected 453 frozen COCO rows, found {len(selected)}."
        )

    selected_key = {
        (
            str(r["split"]),
            int(r["image_id"]),
        )
        for _, r in selected.iterrows()
    }

    pixel_audit = pd.read_csv(
        A4B_AUDIT,
        low_memory=False,
    )

    pixel_audit["image_id"] = pd.to_numeric(
        pixel_audit["image_id"],
        errors="raise",
    ).astype(int)

    pixel_lookup = {
        (
            str(r["split"]),
            int(r["image_id"]),
        ): str(r["local_path"])
        for _, r in pixel_audit.iterrows()
        if str(r["status"])
        in {"DOWNLOADED_VALID", "EXISTING_VALID"}
    }

    gt_rows = []

    for split, path in [
        ("train2017", TRAIN_JSON),
        ("val2017", VAL_JSON),
    ]:
        data = read_json(path)

        categories = {
            int(c["id"]): str(c["name"]).strip().lower()
            for c in data["categories"]
        }

        images = {
            int(i["id"]): i
            for i in data["images"]
        }

        for ann in data["annotations"]:
            image_id = int(ann["image_id"])

            if (split, image_id) not in selected_key:
                continue

            category_id = int(
                ann["category_id"]
            )

            category_name = categories[
                category_id
            ]

            if category_name not in {
                "elephant",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "bear",
                "zebra",
                "giraffe",
                "person",
            }:
                continue

            x, y, w, h = [
                float(v)
                for v in ann["bbox"]
            ]

            meta = images[image_id]

            gt_rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "file_name": str(
                        meta["file_name"]
                    ),
                    "local_path": pixel_lookup[
                        (split, image_id)
                    ],
                    "image_width": int(
                        meta["width"]
                    ),
                    "image_height": int(
                        meta["height"]
                    ),
                    "annotation_id": int(
                        ann["id"]
                    ),
                    "category_id": category_id,
                    "category_name": category_name,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "bbox_x2": x + w,
                    "bbox_y2": y + h,
                    "area": float(
                        ann.get(
                            "area",
                            w * h,
                        )
                    ),
                    "iscrowd": int(
                        ann.get(
                            "iscrowd",
                            0,
                        )
                    ),
                }
            )

    gt = pd.DataFrame(
        gt_rows
    )

    if len(gt) == 0:
        raise RuntimeError(
            "No COCO GT rows were extracted."
        )

    represented = set(
        zip(
            gt["split"],
            gt["image_id"],
        )
    )

    missing_gt = selected_key - represented

    if missing_gt:
        raise RuntimeError(
            f"{len(missing_gt)} frozen COCO images have no target/confuser GT rows."
        )

    return selected, gt


def load_supplement_gt():
    supp = pd.read_csv(
        SUPP_9,
        low_memory=False,
    )

    source = pd.read_csv(
        A2_IMAGE_LEVEL,
        low_memory=False,
    )

    source = source.loc[
        source[
            "elephant_plus_other"
        ]
        .astype(str)
        .str.lower()
        .isin(
            ["true", "1", "yes"]
        )
    ].copy()

    source_lookup = {}

    for _, r in source.iterrows():
        p = Path(
            str(
                r["image_path"]
            )
        )

        if not p.exists():
            continue

        digest = sha256_file(p)

        source_lookup.setdefault(
            digest,
            r,
        )

    class_map = {
        0: "tiger",
        1: "lion",
        2: "elephant",
        3: "crocodile",
    }

    rows = []

    for _, s in supp.iterrows():
        digest = str(
            s["sha256"]
        ).strip()

        if digest not in source_lookup:
            raise RuntimeError(
                f"Supplementary image SHA not found in R17-A2 source audit: {digest}"
            )

        src = source_lookup[
            digest
        ]

        image_path = Path(
            str(
                src["image_path"]
            )
        )

        label_path = Path(
            str(
                src["label_path"]
            )
        )

        require(image_path)
        require(label_path)

        from PIL import Image

        with Image.open(image_path) as im:
            width, height = im.size

        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

        for line_index, line in enumerate(
            lines,
            start=1,
        ):
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cid = int(
                float(
                    parts[0]
                )
            )

            xc, yc, bw, bh = [
                float(v)
                for v in parts[1:5]
            ]

            x = (
                xc - bw / 2.0
            ) * width

            y = (
                yc - bh / 2.0
            ) * height

            w = bw * width
            h = bh * height

            rows.append(
                {
                    "image_sha256": digest,
                    "image_path": str(
                        image_path
                    ),
                    "label_path": str(
                        label_path
                    ),
                    "image_width": int(
                        width
                    ),
                    "image_height": int(
                        height
                    ),
                    "line_index": int(
                        line_index
                    ),
                    "class_id": cid,
                    "class_name": class_map.get(
                        cid,
                        f"unknown_{cid}",
                    ),
                    "bbox_x": float(
                        x
                    ),
                    "bbox_y": float(
                        y
                    ),
                    "bbox_w": float(
                        w
                    ),
                    "bbox_h": float(
                        h
                    ),
                    "bbox_x2": float(
                        x + w
                    ),
                    "bbox_y2": float(
                        y + h
                    ),
                }
            )

    gt = pd.DataFrame(
        rows
    )

    if len(gt) == 0:
        raise RuntimeError(
            "No supplementary GT boxes extracted."
        )

    return supp, gt


def main():
    print(
        "\nR17-B — FROZEN MIXED-SPECIES ROBUSTNESS PROTOCOL LOCK"
    )

    print(
        "PROTOCOL LOCK ONLY. NO detector inference, NO threshold tuning, and NO result-dependent rule changes. "
        "This locks the primary 4-class selective detector, the original elephant-only comparator, "
        "the frozen external COCO evaluation set, the 9-image in-vocabulary elephant+lion supplement, "
        "all operating points, GT matching rules, metrics, and statistical comparisons before test inference.\n"
    )

    a4a, a4b, a4c, a3b = verify_upstream()

    selective_meta = load_model_metadata(
        SELECTIVE_CKPT
    )

    comparator_meta = load_model_metadata(
        ELEPHANT_ONLY_CKPT
    )

    selective_classes_lower = [
        x.lower()
        for x in selective_meta[
            "class_names"
        ]
    ]

    if selective_classes_lower != SELECTIVE_EXPECTED_CLASSES:
        raise RuntimeError(
            "Selective checkpoint class order mismatch: "
            f"{selective_meta['class_names']}"
        )

    comparator_classes_lower = [
        x.lower()
        for x in comparator_meta[
            "class_names"
        ]
    ]

    if (
        len(comparator_classes_lower) != 1
        or comparator_classes_lower[0] != "elephant"
    ):
        raise RuntimeError(
            "Elephant-only comparator checkpoint is not single-class Elephant: "
            f"{comparator_meta['class_names']}"
        )

    selected, coco_gt = load_coco_subset_gt()
    supp, supp_gt = load_supplement_gt()

    coco_gt_path = (
        OUT
        / "R17B_FROZEN_COCO_GT_BOXES.csv"
    )

    supp_gt_path = (
        OUT
        / "R17B_FROZEN_SUPPLEMENT_ELEPHANT_LION_GT_BOXES.csv"
    )

    coco_gt.to_csv(
        coco_gt_path,
        index=False,
    )

    supp_gt.to_csv(
        supp_gt_path,
        index=False,
    )

    detector_df = pd.DataFrame(
        [
            {
                "detector_id":
                    "SELECTIVE_4CLASS_V5",

                "role":
                    "PRIMARY_SELECTIVE_DETECTOR",

                "checkpoint":
                    selective_meta[
                        "checkpoint"
                    ],

                "sha256":
                    selective_meta[
                        "sha256"
                    ],

                "num_classes":
                    selective_meta[
                        "num_classes"
                    ],

                "class_names":
                    "|".join(
                        selective_meta[
                            "class_names"
                        ]
                    ),
            },
            {
                "detector_id":
                    "ELEPHANT_ONLY_ORIGINAL",

                "role":
                    "COMPARATOR",

                "checkpoint":
                    comparator_meta[
                        "checkpoint"
                    ],

                "sha256":
                    comparator_meta[
                        "sha256"
                    ],

                "num_classes":
                    comparator_meta[
                        "num_classes"
                    ],

                "class_names":
                    "|".join(
                        comparator_meta[
                            "class_names"
                        ]
                    ),
            },
        ]
    )

    detector_path = (
        OUT
        / "R17B_DETECTOR_LOCK.csv"
    )

    detector_df.to_csv(
        detector_path,
        index=False,
    )

    settings_df = pd.DataFrame(
        [
            {
                "parameter":
                    "imgsz",
                "value":
                    IMGSZ,
                "role":
                    "inference",
            },
            {
                "parameter":
                    "inference_conf_floor",
                "value":
                    INFERENCE_CONF_FLOOR,
                "role":
                    "store predictions once, evaluate pre-locked confidence thresholds",
            },
            {
                "parameter":
                    "primary_confidence_threshold",
                "value":
                    PRIMARY_CONF,
                "role":
                    "PRIMARY operating point",
            },
            {
                "parameter":
                    "sensitivity_confidence_thresholds",
                "value":
                    "|".join(
                        str(x)
                        for x in SENSITIVITY_CONFS
                    ),
                "role":
                    "SECONDARY fixed threshold sensitivity; no selection by test performance",
            },
            {
                "parameter":
                    "nms_iou",
                "value":
                    NMS_IOU,
                "role":
                    "inference",
            },
            {
                "parameter":
                    "max_det",
                "value":
                    MAX_DET,
                "role":
                    "inference",
            },
            {
                "parameter":
                    "agnostic_nms",
                "value":
                    AGNOSTIC_NMS,
                "role":
                    "inference",
            },
            {
                "parameter":
                    "augment",
                "value":
                    AUGMENT,
                "role":
                    "inference",
            },
            {
                "parameter":
                    "primary_gt_match_iou",
                "value":
                    MATCH_IOU_PRIMARY,
                "role":
                    "object-level TP/FP/FN",
            },
            {
                "parameter":
                    "ap_iou_grid",
                "value":
                    "|".join(
                        f"{x:.2f}"
                        for x in AP_IOU_GRID
                    ),
                "role":
                    "AP50 and AP50:95",
            },
            {
                "parameter":
                    "bootstrap_reps",
                "value":
                    BOOTSTRAP_REPS,
                "role":
                    "paired image-clustered CI",
            },
            {
                "parameter":
                    "bootstrap_seed",
                "value":
                    BOOTSTRAP_SEED,
                "role":
                    "reproducibility",
            },
        ]
    )

    settings_path = (
        OUT
        / "R17B_INFERENCE_EVALUATION_SETTINGS_LOCK.csv"
    )

    settings_df.to_csv(
        settings_path,
        index=False,
    )

    metric_rows = [
        {
            "metric_id":
                "M1",
            "metric":
                "Elephant object precision/recall/F1 @ IoU=0.50, conf=0.25",
            "scope":
                "Primary COCO 453; report overall and by group",
            "primary":
                True,
        },
        {
            "metric_id":
                "M2",
            "metric":
                "Elephant AP50 and AP50:95",
            "scope":
                "Primary COCO 453",
            "primary":
                True,
        },
        {
            "metric_id":
                "M3",
            "metric":
                "Negative-image false-elephant rate",
            "scope":
                "151 no-elephant multi-animal images",
            "primary":
                True,
        },
        {
            "metric_id":
                "M4",
            "metric":
                "Elephant-positive image sensitivity",
            "scope":
                "151 elephant+animal images",
            "primary":
                True,
        },
        {
            "metric_id":
                "M5",
            "metric":
                "Mixed-scene elephant recall minus elephant-only-control recall",
            "scope":
                "151 positive mixed vs 151 elephant-only controls",
            "primary":
                True,
        },
        {
            "metric_id":
                "M6",
            "metric":
                "Confuser-to-elephant overlap error rate @ IoU>=0.50",
            "scope":
                "GT bird/cat/dog/horse/sheep/cow/bear/zebra/giraffe boxes",
            "primary":
                True,
        },
        {
            "metric_id":
                "M7",
            "metric":
                "Per-confuser-pair elephant recall and false-elephant errors",
            "scope":
                "9 COCO animal-pair categories",
            "primary":
                False,
        },
        {
            "metric_id":
                "M8",
            "metric":
                "In-vocabulary elephant+lion selective-filter performance",
            "scope":
                "9 unique existing elephant+lion images",
            "primary":
                False,
        },
        {
            "metric_id":
                "M9",
            "metric":
                "Unfiltered-any-class alarm vs elephant-class-filtered alarm",
            "scope":
                "4-class detector only; secondary operational ablation",
            "primary":
                False,
        },
        {
            "metric_id":
                "M10",
            "metric":
                "Fixed confidence sensitivity at 0.10/0.25/0.50",
            "scope":
                "No post-test operating-point selection",
            "primary":
                False,
        },
    ]

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_path = (
        OUT
        / "R17B_METRIC_LOCK.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    claim_rows = [
        {
            "claim":
                "The 4-class detector selectively identifies elephants in mixed-animal scenes.",
            "allowed_if_supported":
                True,
            "constraint":
                "Must be supported by frozen COCO metrics; do not generalize beyond evaluated image domains/classes.",
        },
        {
            "claim":
                "The 4-class detector is more selective than the original elephant-only detector.",
            "allowed_if_supported":
                True,
            "constraint":
                "Requires paired image-clustered CI on pre-locked metrics; do not claim if CI includes no improvement.",
        },
        {
            "claim":
                "The 4-class detector is robust to unseen animal species.",
            "allowed_if_supported":
                False,
            "constraint":
                "COCO confusers are OOD stress tests only; success does not establish universal unseen-species robustness.",
        },
        {
            "claim":
                "The 9-image elephant+lion supplement proves broad in-vocabulary robustness.",
            "allowed_if_supported":
                False,
            "constraint":
                "Supplement is small and lion-only; report as supplementary challenge evidence.",
        },
        {
            "claim":
                "Mixed-species image results validate detector-to-tracker-to-forecast end-to-end performance.",
            "allowed_if_supported":
                False,
            "constraint":
                "R17 image evaluation is detector/selective-filter robustness only; historical end-to-end limitation remains.",
        },
        {
            "claim":
                "BAMBI deer/human is elephant-specific validation.",
            "allowed_if_supported":
                False,
            "constraint":
                "BAMBI remains cross-species/cross-domain negative stress only.",
        },
        {
            "claim":
                "Threshold 0.25 was chosen from test performance.",
            "allowed_if_supported":
                False,
            "constraint":
                "0.25 is pre-locked before inference; test-dependent tuning is prohibited.",
        },
    ]

    claims_df = pd.DataFrame(
        claim_rows
    )

    claims_path = (
        OUT
        / "R17B_CLAIM_BOUNDARY_LOCK.csv"
    )

    claims_df.to_csv(
        claims_path,
        index=False,
    )

    protocol = {
        "step":
            "R17B_FROZEN_MIXED_SPECIES_ROBUSTNESS_PROTOCOL_LOCK",

        "created_utc":
            utc_now(),

        "status":
            "R17B_FROZEN_ROBUSTNESS_PROTOCOL_LOCKED_NO_TEST_TUNING_READY_FOR_R17C_FROZEN_DUAL_DETECTOR_INFERENCE",

        "training":
            False,

        "inference":
            False,

        "tuning":
            False,

        "primary_question":
            (
                "Can the frozen 4-class detector selectively retain elephant detections "
                "in real mixed-animal scenes while reducing false-elephant errors relative "
                "to the original single-class elephant detector?"
            ),

        "primary_dataset": {
            "source":
                "COCO 2017 frozen external subset",

            "total_images":
                453,

            "groups": {
                "positive_elephant_plus_animal":
                    151,

                "negative_multi_animal_no_elephant":
                    151,

                "elephant_only_control":
                    151,
            },

            "leakage_audit":
                "0 exact SHA256 overlaps; 0 flagged perceptual near-duplicate candidates against detector datasets",
        },

        "supplementary_dataset": {
            "source":
                "existing project V5/exported test material",

            "unique_images":
                int(
                    len(
                        supp
                    )
                ),

            "content":
                "elephant+lion",

            "role":
                "small in-vocabulary challenge only",
        },

        "detectors": {
            "primary_selective":
                selective_meta,

            "comparator_elephant_only":
                comparator_meta,
        },

        "selective_operational_rule":
            (
                "Run the 4-class detector without pre-filtering classes; preserve all predicted classes. "
                "Operational elephant output is obtained only by retaining predictions whose checkpoint class name is 'elephant'. "
                "Secondary ablation compares this elephant-filtered output with an intentionally unfiltered any-class alarm."
            ),

        "confidence_protocol": {
            "inference_floor":
                INFERENCE_CONF_FLOOR,

            "primary":
                PRIMARY_CONF,

            "sensitivity":
                SENSITIVITY_CONFS,

            "rule":
                "No threshold may be selected or changed based on R17 test outcomes.",
        },

        "matching_protocol": {
            "primary_iou":
                MATCH_IOU_PRIMARY,

            "assignment":
                "Greedy one-to-one confidence-descending matching within each image and target category; one GT box matched at most once.",

            "confuser_error":
                (
                    "A predicted elephant box overlapping a non-elephant GT animal box at IoU>=0.50 "
                    "without being matched to an elephant GT is counted as a confuser-to-elephant error."
                ),
        },

        "statistics": {
            "paired_unit":
                "image",

            "bootstrap_reps":
                BOOTSTRAP_REPS,

            "seed":
                BOOTSTRAP_SEED,

            "ci":
                "95% percentile paired image-clustered bootstrap",

            "comparison_direction":
                (
                    "For false-error metrics: elephant-only comparator minus selective detector; "
                    "positive favors selective detector. For recall/sensitivity metrics: selective detector minus comparator; "
                    "positive favors selective detector."
                ),
        },

        "frozen_hashes": {
            "primary_453_csv":
                sha256_file(
                    PRIMARY_453
                ),

            "supplement_9_csv":
                sha256_file(
                    SUPP_9
                ),

            "coco_gt_boxes_csv":
                sha256_file(
                    coco_gt_path
                ),

            "supplement_gt_boxes_csv":
                sha256_file(
                    supp_gt_path
                ),

            "selective_checkpoint":
                selective_meta[
                    "sha256"
                ],

            "elephant_only_checkpoint":
                comparator_meta[
                    "sha256"
                ],
        },

        "outputs": {
            "detector_lock":
                str(
                    detector_path
                ),

            "settings_lock":
                str(
                    settings_path
                ),

            "metrics_lock":
                str(
                    metrics_path
                ),

            "claim_boundary_lock":
                str(
                    claims_path
                ),

            "coco_gt":
                str(
                    coco_gt_path
                ),

            "supplement_gt":
                str(
                    supp_gt_path
                ),
        },
    }

    manifest_path = (
        OUT
        / "R17B_PROTOCOL_LOCK.json"
    )

    manifest_path.write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "DETECTOR LOCK:"
    )
    print(
        detector_df.to_string(
            index=False
        )
    )

    print(
        "\nDATASET / GT LOCK:"
    )
    print(
        f"COCO primary images = {len(selected)}"
    )
    print(
        f"COCO frozen GT boxes = {len(coco_gt)}"
    )
    print(
        f"Supplementary unique images = {len(supp)}"
    )
    print(
        f"Supplementary GT boxes = {len(supp_gt)}"
    )

    print(
        "\nOPERATING POINT LOCK:"
    )
    print(
        f"inference conf floor = {INFERENCE_CONF_FLOOR}"
    )
    print(
        f"primary conf = {PRIMARY_CONF}"
    )
    print(
        f"sensitivity confs = {SENSITIVITY_CONFS}"
    )
    print(
        f"NMS IoU = {NMS_IOU}"
    )
    print(
        f"GT match IoU = {MATCH_IOU_PRIMARY}"
    )

    print(
        "\nPRIMARY METRICS:"
    )
    print(
        metrics_df.loc[
            metrics_df[
                "primary"
            ].astype(bool)
        ].to_string(
            index=False
        )
    )

    print(
        "\nCLAIM BOUNDARIES:"
    )
    print(
        claims_df.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )
    print(
        protocol[
            "status"
        ]
    )

    print(
        "\nR17-B COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
