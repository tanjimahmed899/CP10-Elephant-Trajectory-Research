
from pathlib import Path
import hashlib
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

R17B_DIR = R17_ROOT / "R17B_frozen_robustness_protocol_lock"
PROTOCOL_PATH = R17B_DIR / "R17B_PROTOCOL_LOCK.json"
COCO_GT_PATH = R17B_DIR / "R17B_FROZEN_COCO_GT_BOXES.csv"
SUPP_GT_PATH = R17B_DIR / "R17B_FROZEN_SUPPLEMENT_ELEPHANT_LION_GT_BOXES.csv"

A4A_DIR = R17_ROOT / "R17A4A_frozen_subset_selection"
PRIMARY_453_PATH = A4A_DIR / "R17A4A_PRIMARY_BALANCED_453.csv"
SUPP_9_PATH = A4A_DIR / "R17A4A_SUPPLEMENTARY_IN_VOCAB_UNIQUE.csv"

R17C_DIR = R17_ROOT / "R17C_frozen_dual_detector_inference"
R17C_MANIFEST_PATH = R17C_DIR / "R17C_MANIFEST.json"
RAW_PRED_PATH = R17C_DIR / "R17C_RAW_PREDICTIONS.csv"
IMAGE_AUDIT_PATH = R17C_DIR / "R17C_IMAGE_INFERENCE_AUDIT.csv"

OUT = R17_ROOT / "R17D_locked_metric_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

SELECTIVE_ID = "SELECTIVE_4CLASS_V5"
COMPARATOR_ID = "ELEPHANT_ONLY_ORIGINAL"

PRIMARY_CONF = 0.25
SENSITIVITY_CONFS = [0.10, 0.25, 0.50]
MATCH_IOU = 0.50
AP_IOUS = [round(0.50 + 0.05 * i, 2) for i in range(10)]

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20261030

ANIMAL_CONFUSERS = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "bear",
    "zebra",
    "giraffe",
}


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


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def f1_score(p, r):
    if not np.isfinite(p) or not np.isfinite(r):
        return float("nan")
    return safe_div(2.0 * p * r, p + r)


def iou_xyxy(box, boxes):
    if len(boxes) == 0:
        return np.zeros(
            0,
            dtype=np.float64,
        )

    x1 = np.maximum(
        box[0],
        boxes[:, 0],
    )
    y1 = np.maximum(
        box[1],
        boxes[:, 1],
    )
    x2 = np.minimum(
        box[2],
        boxes[:, 2],
    )
    y2 = np.minimum(
        box[3],
        boxes[:, 3],
    )

    iw = np.maximum(
        0.0,
        x2 - x1,
    )
    ih = np.maximum(
        0.0,
        y2 - y1,
    )

    inter = iw * ih

    area_a = max(
        0.0,
        box[2] - box[0],
    ) * max(
        0.0,
        box[3] - box[1],
    )

    area_b = (
        np.maximum(
            0.0,
            boxes[:, 2] - boxes[:, 0],
        )
        *
        np.maximum(
            0.0,
            boxes[:, 3] - boxes[:, 1],
        )
    )

    union = (
        area_a
        +
        area_b
        -
        inter
    )

    out = np.zeros_like(
        inter,
        dtype=np.float64,
    )

    good = union > 0

    out[good] = (
        inter[good]
        /
        union[good]
    )

    return out


def verify_inputs():
    for p in [
        PROTOCOL_PATH,
        COCO_GT_PATH,
        SUPP_GT_PATH,
        PRIMARY_453_PATH,
        SUPP_9_PATH,
        R17C_MANIFEST_PATH,
        RAW_PRED_PATH,
        IMAGE_AUDIT_PATH,
    ]:
        require(p)

    protocol = read_json(
        PROTOCOL_PATH
    )

    r17c = read_json(
        R17C_MANIFEST_PATH
    )

    expected_b = (
        "R17B_FROZEN_ROBUSTNESS_PROTOCOL_LOCKED_NO_TEST_TUNING_READY_FOR_R17C_FROZEN_DUAL_DETECTOR_INFERENCE"
    )

    expected_c = (
        "R17C_FROZEN_DUAL_DETECTOR_INFERENCE_COMPLETE_RAW_PREDICTIONS_LOCKED_READY_FOR_R17D_METRIC_EVALUATION"
    )

    if protocol.get("status") != expected_b:
        raise RuntimeError(
            f"Unexpected R17-B status: {protocol.get('status')}"
        )

    if r17c.get("status") != expected_c:
        raise RuntimeError(
            f"Unexpected R17-C status: {r17c.get('status')}"
        )

    if sha256_file(
        RAW_PRED_PATH
    ) != r17c["output_sha256"]["raw_predictions"]:
        raise RuntimeError(
            "R17-C raw prediction file hash mismatch."
        )

    if sha256_file(
        IMAGE_AUDIT_PATH
    ) != r17c["output_sha256"]["image_inference_audit"]:
        raise RuntimeError(
            "R17-C image audit file hash mismatch."
        )

    cp = protocol[
        "confidence_protocol"
    ]

    if float(
        cp["primary"]
    ) != PRIMARY_CONF:
        raise RuntimeError(
            "Primary confidence differs from frozen R17-B."
        )

    if [
        float(x)
        for x in cp["sensitivity"]
    ] != SENSITIVITY_CONFS:
        raise RuntimeError(
            "Sensitivity thresholds differ from frozen R17-B."
        )

    if float(
        protocol[
            "matching_protocol"
        ][
            "primary_iou"
        ]
    ) != MATCH_IOU:
        raise RuntimeError(
            "Primary matching IoU differs from frozen R17-B."
        )

    stats = protocol[
        "statistics"
    ]

    if int(
        stats["bootstrap_reps"]
    ) != BOOTSTRAP_REPS:
        raise RuntimeError(
            "Bootstrap reps differ from frozen R17-B."
        )

    if int(
        stats["seed"]
    ) != BOOTSTRAP_SEED:
        raise RuntimeError(
            "Bootstrap seed differs from frozen R17-B."
        )

    return protocol, r17c


def load_data():
    primary = pd.read_csv(
        PRIMARY_453_PATH,
        low_memory=False,
    )

    primary["image_id"] = pd.to_numeric(
        primary["image_id"],
        errors="raise",
    ).astype(int)

    coco_gt = pd.read_csv(
        COCO_GT_PATH,
        low_memory=False,
    )

    coco_gt["image_id"] = pd.to_numeric(
        coco_gt["image_id"],
        errors="raise",
    ).astype(int)

    coco_gt["iscrowd"] = pd.to_numeric(
        coco_gt["iscrowd"],
        errors="coerce",
    ).fillna(0).astype(int)

    supp = pd.read_csv(
        SUPP_9_PATH,
        low_memory=False,
    )

    supp_gt = pd.read_csv(
        SUPP_GT_PATH,
        low_memory=False,
    )

    raw = pd.read_csv(
        RAW_PRED_PATH,
        low_memory=False,
    )

    raw["image_id"] = pd.to_numeric(
        raw["image_id"],
        errors="raise",
    ).astype(int)

    raw["confidence"] = pd.to_numeric(
        raw["confidence"],
        errors="raise",
    ).astype(float)

    for col in [
        "x1",
        "y1",
        "x2",
        "y2",
    ]:
        raw[col] = pd.to_numeric(
            raw[col],
            errors="raise",
        ).astype(float)

    return (
        primary,
        coco_gt,
        supp,
        supp_gt,
        raw,
    )


def primary_image_keys(primary):
    return [
        (
            str(r["split"]),
            int(r["image_id"]),
        )
        for _, r in primary.iterrows()
    ]


def build_gt_maps(coco_gt):
    elephant = coco_gt.loc[
        (
            coco_gt[
                "category_name"
            ]
            .astype(str)
            .str.lower()
            ==
            "elephant"
        )
        &
        (
            coco_gt[
                "iscrowd"
            ]
            ==
            0
        )
    ].copy()

    elephant_crowd = coco_gt.loc[
        (
            coco_gt[
                "category_name"
            ]
            .astype(str)
            .str.lower()
            ==
            "elephant"
        )
        &
        (
            coco_gt[
                "iscrowd"
            ]
            !=
            0
        )
    ].copy()

    confusers = coco_gt.loc[
        coco_gt[
            "category_name"
        ]
        .astype(str)
        .str.lower()
        .isin(
            ANIMAL_CONFUSERS
        )
        &
        (
            coco_gt[
                "iscrowd"
            ]
            ==
            0
        )
    ].copy()

    ele_map = defaultdict(
        list
    )
    crowd_map = defaultdict(
        list
    )
    conf_map = defaultdict(
        list
    )

    for _, r in elephant.iterrows():
        key = (
            str(
                r["split"]
            ),
            int(
                r["image_id"]
            ),
        )

        ele_map[
            key
        ].append(
            [
                float(
                    r["bbox_x"]
                ),
                float(
                    r["bbox_y"]
                ),
                float(
                    r["bbox_x2"]
                ),
                float(
                    r["bbox_y2"]
                ),
            ]
        )

    for _, r in elephant_crowd.iterrows():
        key = (
            str(
                r["split"]
            ),
            int(
                r["image_id"]
            ),
        )

        crowd_map[
            key
        ].append(
            [
                float(
                    r["bbox_x"]
                ),
                float(
                    r["bbox_y"]
                ),
                float(
                    r["bbox_x2"]
                ),
                float(
                    r["bbox_y2"]
                ),
            ]
        )

    for _, r in confusers.iterrows():
        key = (
            str(
                r["split"]
            ),
            int(
                r["image_id"]
            ),
        )

        conf_map[
            key
        ].append(
            (
                str(
                    r["category_name"]
                ).lower(),
                [
                    float(
                        r["bbox_x"]
                    ),
                    float(
                        r["bbox_y"]
                    ),
                    float(
                        r["bbox_x2"]
                    ),
                    float(
                        r["bbox_y2"]
                    ),
                ],
            )
        )

    return (
        ele_map,
        crowd_map,
        conf_map,
        elephant,
        elephant_crowd,
        confusers,
    )


def build_pred_map(raw, detector_id, dataset_scope, conf_threshold):
    p = raw.loc[
        (
            raw[
                "detector_id"
            ]
            ==
            detector_id
        )
        &
        (
            raw[
                "dataset_scope"
            ]
            ==
            dataset_scope
        )
        &
        (
            raw[
                "confidence"
            ]
            >=
            conf_threshold
        )
    ].copy()

    # Frozen operational rule:
    # primary selective detector keeps only class=elephant.
    p = p.loc[
        p[
            "class_name"
        ]
        .astype(str)
        .str.lower()
        ==
        "elephant"
    ].copy()

    pred_map = defaultdict(
        list
    )

    for _, r in p.iterrows():
        key = (
            str(
                r["split"]
            ),
            int(
                r["image_id"]
            ),
        )

        pred_map[
            key
        ].append(
            {
                "confidence":
                    float(
                        r["confidence"]
                    ),

                "box":
                    np.array(
                        [
                            float(
                                r["x1"]
                            ),
                            float(
                                r["y1"]
                            ),
                            float(
                                r["x2"]
                            ),
                            float(
                                r["y2"]
                            ),
                        ],
                        dtype=np.float64,
                    ),
            }
        )

    for key in pred_map:
        pred_map[key].sort(
            key=lambda x:
                -x["confidence"]
        )

    return pred_map


def evaluate_image(
    key,
    preds,
    gt_boxes,
    crowd_boxes,
    confuser_items,
    iou_threshold,
):
    gt = np.asarray(
        gt_boxes,
        dtype=np.float64,
    ).reshape(
        -1,
        4,
    )

    crowd = np.asarray(
        crowd_boxes,
        dtype=np.float64,
    ).reshape(
        -1,
        4,
    )

    conf_boxes = np.asarray(
        [
            x[1]
            for x in confuser_items
        ],
        dtype=np.float64,
    ).reshape(
        -1,
        4,
    )

    matched = np.zeros(
        len(gt),
        dtype=bool,
    )

    tp = 0
    fp = 0
    ignored_crowd_pred = 0
    confuser_error_preds = 0
    pred_records = []

    for pred in preds:
        box = pred[
            "box"
        ]

        score = pred[
            "confidence"
        ]

        gt_ious = iou_xyxy(
            box,
            gt,
        )

        best_idx = -1
        best_iou = 0.0

        if len(
            gt_ious
        ):
            order = np.argsort(
                -gt_ious
            )

            for idx in order:
                if matched[
                    idx
                ]:
                    continue

                if gt_ious[
                    idx
                ] >= iou_threshold:
                    best_idx = int(
                        idx
                    )
                    best_iou = float(
                        gt_ious[
                            idx
                        ]
                    )
                    break

        if best_idx >= 0:
            matched[
                best_idx
            ] = True
            tp += 1

            pred_records.append(
                {
                    "score":
                        score,
                    "tp":
                        1,
                    "fp":
                        0,
                    "ignored":
                        0,
                    "matched_iou":
                        best_iou,
                    "confuser_error":
                        0,
                }
            )

            continue

        # Standard crowd-style handling:
        # prediction overlapping an elephant crowd box is ignored.
        crowd_ious = iou_xyxy(
            box,
            crowd,
        )

        if (
            len(
                crowd_ious
            )
            and
            float(
                np.max(
                    crowd_ious
                )
            )
            >=
            iou_threshold
        ):
            ignored_crowd_pred += 1

            pred_records.append(
                {
                    "score":
                        score,
                    "tp":
                        0,
                    "fp":
                        0,
                    "ignored":
                        1,
                    "matched_iou":
                        0.0,
                    "confuser_error":
                        0,
                }
            )

            continue

        is_confuser_error = 0

        if len(
            conf_boxes
        ):
            conf_ious = iou_xyxy(
                box,
                conf_boxes,
            )

            if (
                len(
                    conf_ious
                )
                and
                float(
                    np.max(
                        conf_ious
                    )
                )
                >=
                iou_threshold
            ):
                is_confuser_error = 1
                confuser_error_preds += 1

        fp += 1

        pred_records.append(
            {
                "score":
                    score,
                "tp":
                    0,
                "fp":
                    1,
                "ignored":
                    0,
                "matched_iou":
                    0.0,
                "confuser_error":
                    is_confuser_error,
            }
        )

    fn = int(
        len(
            gt
        )
        -
        tp
    )

    return {
        "tp":
            int(
                tp
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "gt_count":
            int(
                len(
                    gt
                )
            ),

        "pred_count_scored":
            int(
                tp
                +
                fp
            ),

        "ignored_crowd_pred":
            int(
                ignored_crowd_pred
            ),

        "confuser_error_preds":
            int(
                confuser_error_preds
            ),

        "confuser_gt_count":
            int(
                len(
                    conf_boxes
                )
            ),

        "image_has_elephant_gt":
            int(
                len(
                    gt
                )
                > 0
            ),

        "image_has_correct_elephant_detection":
            int(
                tp
                > 0
            ),

        "image_has_false_elephant_prediction":
            int(
                fp
                > 0
            ),

        "image_has_any_elephant_prediction":
            int(
                (
                    tp
                    +
                    fp
                )
                > 0
            ),

        "image_has_confuser_error":
            int(
                confuser_error_preds
                > 0
            ),

        "prediction_records":
            pred_records,
    }


def evaluate_primary_threshold(
    primary,
    raw,
    ele_map,
    crowd_map,
    conf_map,
    conf_threshold,
):
    pred_maps = {
        detector_id:
            build_pred_map(
                raw,
                detector_id,
                "PRIMARY_COCO_453",
                conf_threshold,
            )
        for detector_id in [
            SELECTIVE_ID,
            COMPARATOR_ID,
        ]
    }

    rows = []

    for _, r in primary.iterrows():
        key = (
            str(
                r["split"]
            ),
            int(
                r["image_id"]
            ),
        )

        group = str(
            r[
                "evaluation_group"
            ]
        )

        for detector_id in [
            SELECTIVE_ID,
            COMPARATOR_ID,
        ]:
            metrics = evaluate_image(
                key=key,
                preds=pred_maps[
                    detector_id
                ].get(
                    key,
                    [],
                ),
                gt_boxes=ele_map.get(
                    key,
                    [],
                ),
                crowd_boxes=crowd_map.get(
                    key,
                    [],
                ),
                confuser_items=conf_map.get(
                    key,
                    [],
                ),
                iou_threshold=MATCH_IOU,
            )

            row = {
                "confidence_threshold":
                    float(
                        conf_threshold
                    ),

                "detector_id":
                    detector_id,

                "evaluation_group":
                    group,

                "split":
                    key[0],

                "image_id":
                    key[1],
            }

            for k, v in metrics.items():
                if k == "prediction_records":
                    continue
                row[k] = v

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def aggregate_object_metrics(df, scope_name):
    tp = int(
        df[
            "tp"
        ].sum()
    )
    fp = int(
        df[
            "fp"
        ].sum()
    )
    fn = int(
        df[
            "fn"
        ].sum()
    )

    p = safe_div(
        tp,
        tp + fp,
    )
    r = safe_div(
        tp,
        tp + fn,
    )
    f1 = f1_score(
        p,
        r,
    )

    return {
        "scope":
            scope_name,

        "images":
            int(
                len(
                    df
                )
            ),

        "elephant_gt":
            int(
                df[
                    "gt_count"
                ].sum()
            ),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "precision":
            p,

        "recall":
            r,

        "f1":
            f1,

        "image_sensitivity":
            safe_div(
                int(
                    df[
                        "image_has_correct_elephant_detection"
                    ].sum()
                ),
                int(
                    (
                        df[
                            "image_has_elephant_gt"
                        ]
                        >
                        0
                    ).sum()
                ),
            ),

        "false_elephant_image_rate":
            safe_div(
                int(
                    df[
                        "image_has_false_elephant_prediction"
                    ].sum()
                ),
                int(
                    len(
                        df
                    )
                ),
            ),

        "confuser_error_predictions":
            int(
                df[
                    "confuser_error_preds"
                ].sum()
            ),

        "confuser_gt_boxes":
            int(
                df[
                    "confuser_gt_count"
                ].sum()
            ),

        "confuser_error_per_confuser_gt":
            safe_div(
                int(
                    df[
                        "confuser_error_preds"
                    ].sum()
                ),
                int(
                    df[
                        "confuser_gt_count"
                    ].sum()
                ),
            ),

        "images_with_confuser_error":
            int(
                df[
                    "image_has_confuser_error"
                ].sum()
            ),
    }


def group_metric_table(image_metrics):
    rows = []

    for conf in SENSITIVITY_CONFS:
        dconf = image_metrics.loc[
            np.isclose(
                image_metrics[
                    "confidence_threshold"
                ],
                conf,
            )
        ]

        for detector_id in [
            SELECTIVE_ID,
            COMPARATOR_ID,
        ]:
            dd = dconf.loc[
                dconf[
                    "detector_id"
                ]
                ==
                detector_id
            ]

            agg = aggregate_object_metrics(
                dd,
                "ALL_PRIMARY_453",
            )

            rows.append(
                {
                    "confidence_threshold":
                        conf,
                    "detector_id":
                        detector_id,
                    **agg,
                }
            )

            for group in sorted(
                dd[
                    "evaluation_group"
                ].unique()
            ):
                dg = dd.loc[
                    dd[
                        "evaluation_group"
                    ]
                    ==
                    group
                ]

                agg = aggregate_object_metrics(
                    dg,
                    group,
                )

                rows.append(
                    {
                        "confidence_threshold":
                            conf,
                        "detector_id":
                            detector_id,
                        **agg,
                    }
                )

    return pd.DataFrame(
        rows
    )


def ap_for_detector(
    primary,
    raw,
    ele_map,
    crowd_map,
    detector_id,
    iou_threshold,
):
    p = raw.loc[
        (
            raw[
                "detector_id"
            ]
            ==
            detector_id
        )
        &
        (
            raw[
                "dataset_scope"
            ]
            ==
            "PRIMARY_COCO_453"
        )
        &
        (
            raw[
                "class_name"
            ]
            .astype(str)
            .str.lower()
            ==
            "elephant"
        )
    ].copy()

    p = p.sort_values(
        [
            "confidence",
        ],
        ascending=False,
    )

    valid_gt_total = sum(
        len(
            ele_map.get(
                (
                    str(r["split"]),
                    int(r["image_id"]),
                ),
                [],
            )
        )
        for _, r in primary.iterrows()
    )

    matched_by_image = {}

    for _, r in primary.iterrows():
        key = (
            str(
                r["split"]
            ),
            int(
                r["image_id"]
            ),
        )

        matched_by_image[
            key
        ] = np.zeros(
            len(
                ele_map.get(
                    key,
                    [],
                )
            ),
            dtype=bool,
        )

    tps = []
    fps = []
    ignored = []

    for _, r in p.iterrows():
        key = (
            str(
                r["split"]
            ),
            int(
                r["image_id"]
            ),
        )

        box = np.array(
            [
                float(
                    r["x1"]
                ),
                float(
                    r["y1"]
                ),
                float(
                    r["x2"]
                ),
                float(
                    r["y2"]
                ),
            ],
            dtype=np.float64,
        )

        gt = np.asarray(
            ele_map.get(
                key,
                [],
            ),
            dtype=np.float64,
        ).reshape(
            -1,
            4,
        )

        crowd = np.asarray(
            crowd_map.get(
                key,
                [],
            ),
            dtype=np.float64,
        ).reshape(
            -1,
            4,
        )

        matched = matched_by_image[
            key
        ]

        gt_ious = iou_xyxy(
            box,
            gt,
        )

        best = -1

        if len(
            gt_ious
        ):
            order = np.argsort(
                -gt_ious
            )

            for idx in order:
                if matched[
                    idx
                ]:
                    continue

                if gt_ious[
                    idx
                ] >= iou_threshold:
                    best = int(
                        idx
                    )
                    break

        if best >= 0:
            matched[
                best
            ] = True
            tps.append(
                1
            )
            fps.append(
                0
            )
            ignored.append(
                0
            )
            continue

        crowd_ious = iou_xyxy(
            box,
            crowd,
        )

        if (
            len(
                crowd_ious
            )
            and
            float(
                np.max(
                    crowd_ious
                )
            )
            >=
            iou_threshold
        ):
            tps.append(
                0
            )
            fps.append(
                0
            )
            ignored.append(
                1
            )
            continue

        tps.append(
            0
        )
        fps.append(
            1
        )
        ignored.append(
            0
        )

    tps = np.asarray(
        tps,
        dtype=np.float64,
    )
    fps = np.asarray(
        fps,
        dtype=np.float64,
    )
    ignored = np.asarray(
        ignored,
        dtype=bool,
    )

    tps = tps[
        ~ignored
    ]
    fps = fps[
        ~ignored
    ]

    if valid_gt_total <= 0:
        return float(
            "nan"
        )

    if len(
        tps
    ) == 0:
        return 0.0

    cum_tp = np.cumsum(
        tps
    )
    cum_fp = np.cumsum(
        fps
    )

    recall = (
        cum_tp
        /
        valid_gt_total
    )

    precision = (
        cum_tp
        /
        np.maximum(
            cum_tp
            +
            cum_fp,
            1e-12,
        )
    )

    # Monotonic precision envelope.
    precision_envelope = precision.copy()

    for i in range(
        len(
            precision_envelope
        )
        -
        2,
        -1,
        -1,
    ):
        precision_envelope[
            i
        ] = max(
            precision_envelope[
                i
            ],
            precision_envelope[
                i
                +
                1
            ],
        )

    recall_grid = np.linspace(
        0.0,
        1.0,
        101,
    )

    interp = []

    for rt in recall_grid:
        mask = recall >= rt

        if mask.any():
            interp.append(
                float(
                    np.max(
                        precision_envelope[
                            mask
                        ]
                    )
                )
            )
        else:
            interp.append(
                0.0
            )

    return float(
        np.mean(
            interp
        )
    )


def compute_ap_table(
    primary,
    raw,
    ele_map,
    crowd_map,
):
    rows = []

    for detector_id in [
        SELECTIVE_ID,
        COMPARATOR_ID,
    ]:
        aps = []

        for iou_t in AP_IOUS:
            ap = ap_for_detector(
                primary,
                raw,
                ele_map,
                crowd_map,
                detector_id,
                iou_t,
            )

            aps.append(
                ap
            )

            rows.append(
                {
                    "detector_id":
                        detector_id,

                    "metric":
                        f"AP@IoU={iou_t:.2f}",

                    "value":
                        ap,
                }
            )

        rows.append(
            {
                "detector_id":
                    detector_id,

                "metric":
                    "AP50",

                "value":
                    aps[
                        0
                    ],
            }
        )

        rows.append(
            {
                "detector_id":
                    detector_id,

                "metric":
                    "AP50:95",

                "value":
                    float(
                        np.mean(
                            aps
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def confuser_pair_metrics(
    primary,
    coco_gt,
    raw,
):
    pairs = [
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

    rows = []

    conf = PRIMARY_CONF

    image_metrics = None

    # image-level base metrics already computed elsewhere,
    # but rebuild only if needed outside.
    for pair in pairs:
        image_ids = set(
            zip(
                coco_gt.loc[
                    (
                        coco_gt[
                            "category_name"
                        ]
                        .astype(str)
                        .str.lower()
                        ==
                        pair
                    )
                    &
                    (
                        coco_gt[
                            "iscrowd"
                        ]
                        ==
                        0
                    ),
                    "split",
                ].astype(
                    str
                ),
                coco_gt.loc[
                    (
                        coco_gt[
                            "category_name"
                        ]
                        .astype(str)
                        .str.lower()
                        ==
                        pair
                    )
                    &
                    (
                        coco_gt[
                            "iscrowd"
                        ]
                        ==
                        0
                    ),
                    "image_id",
                ].astype(
                    int
                ),
            )
        )

        pos_primary = primary.loc[
            [
                (
                    str(r["split"]),
                    int(r["image_id"]),
                )
                in image_ids
                and
                str(
                    r[
                        "evaluation_group"
                    ]
                )
                ==
                "POSITIVE_ELEPHANT_PLUS_ANIMAL"
                for _, r in primary.iterrows()
            ]
        ]

        if len(
            pos_primary
        ) == 0:
            continue

        rows.append(
            {
                "pair":
                    f"elephant+{pair}",
                "images":
                    int(
                        len(
                            pos_primary
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def bootstrap_ci(
    image_metrics_primary,
):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    d = image_metrics_primary.loc[
        np.isclose(
            image_metrics_primary[
                "confidence_threshold"
            ],
            PRIMARY_CONF,
        )
    ].copy()

    keys = [
        (
            str(r["split"]),
            int(r["image_id"]),
        )
        for _, r in d.loc[
            d[
                "detector_id"
            ]
            ==
            SELECTIVE_ID
        ].iterrows()
    ]

    sel = d.loc[
        d[
            "detector_id"
        ]
        ==
        SELECTIVE_ID
    ].copy()

    cmp = d.loc[
        d[
            "detector_id"
        ]
        ==
        COMPARATOR_ID
    ].copy()

    sel["key"] = list(
        zip(
            sel[
                "split"
            ].astype(
                str
            ),
            sel[
                "image_id"
            ].astype(
                int
            ),
        )
    )

    cmp["key"] = list(
        zip(
            cmp[
                "split"
            ].astype(
                str
            ),
            cmp[
                "image_id"
            ].astype(
                int
            ),
        )
    )

    sel = sel.set_index(
        "key"
    ).loc[
        keys
    ]

    cmp = cmp.set_index(
        "key"
    ).loc[
        keys
    ]

    if len(sel) != 453 or len(cmp) != 453:
        raise RuntimeError(
            "Paired bootstrap alignment failed."
        )

    groups = np.array(
        sel[
            "evaluation_group"
        ].astype(
            str
        ).tolist()
    )

    def metric_from_indices(
        frame,
        indices,
        metric_name,
    ):
        x = frame.iloc[
            indices
        ]

        if metric_name == "precision":
            tp = x[
                "tp"
            ].sum()
            fp = x[
                "fp"
            ].sum()
            return safe_div(
                tp,
                tp + fp,
            )

        if metric_name == "recall":
            tp = x[
                "tp"
            ].sum()
            fn = x[
                "fn"
            ].sum()
            return safe_div(
                tp,
                tp + fn,
            )

        if metric_name == "f1":
            tp = x[
                "tp"
            ].sum()
            fp = x[
                "fp"
            ].sum()
            fn = x[
                "fn"
            ].sum()

            p = safe_div(
                tp,
                tp + fp,
            )
            r = safe_div(
                tp,
                tp + fn,
            )

            return f1_score(
                p,
                r,
            )

        if metric_name == "image_sensitivity":
            denom = (
                x[
                    "image_has_elephant_gt"
                ]
                >
                0
            ).sum()

            return safe_div(
                x[
                    "image_has_correct_elephant_detection"
                ].sum(),
                denom,
            )

        if metric_name == "false_elephant_image_rate":
            return float(
                x[
                    "image_has_false_elephant_prediction"
                ].mean()
            )

        if metric_name == "confuser_error_rate":
            denom = x[
                "confuser_gt_count"
            ].sum()

            return safe_div(
                x[
                    "confuser_error_preds"
                ].sum(),
                denom,
            )

        raise KeyError(
            metric_name
        )

    comparisons = []

    def run_bootstrap(
        label,
        mask,
        metric_name,
        direction,
    ):
        idx_pool = np.flatnonzero(
            mask
        )

        if len(
            idx_pool
        ) == 0:
            return

        sel_point = metric_from_indices(
            sel,
            idx_pool,
            metric_name,
        )

        cmp_point = metric_from_indices(
            cmp,
            idx_pool,
            metric_name,
        )

        if direction == "selective_minus_comparator":
            point_delta = (
                sel_point
                -
                cmp_point
            )
        else:
            point_delta = (
                cmp_point
                -
                sel_point
            )

        deltas = np.empty(
            BOOTSTRAP_REPS,
            dtype=np.float64,
        )

        n = len(
            idx_pool
        )

        for b in range(
            BOOTSTRAP_REPS
        ):
            sampled = rng.choice(
                idx_pool,
                size=n,
                replace=True,
            )

            s = metric_from_indices(
                sel,
                sampled,
                metric_name,
            )

            c = metric_from_indices(
                cmp,
                sampled,
                metric_name,
            )

            if direction == "selective_minus_comparator":
                deltas[
                    b
                ] = (
                    s - c
                )
            else:
                deltas[
                    b
                ] = (
                    c - s
                )

        finite = deltas[
            np.isfinite(
                deltas
            )
        ]

        if len(
            finite
        ) == 0:
            lo = hi = float(
                "nan"
            )
        else:
            lo, hi = np.percentile(
                finite,
                [
                    2.5,
                    97.5,
                ],
            )

        comparisons.append(
            {
                "comparison":
                    label,

                "metric":
                    metric_name,

                "direction":
                    direction,

                "selective_value":
                    sel_point,

                "comparator_value":
                    cmp_point,

                "point_delta_positive_favors_selective":
                    point_delta,

                "ci95_low":
                    float(
                        lo
                    ),

                "ci95_high":
                    float(
                        hi
                    ),

                "ci_excludes_zero":
                    bool(
                        np.isfinite(
                            lo
                        )
                        and
                        np.isfinite(
                            hi
                        )
                        and
                        (
                            lo > 0
                            or
                            hi < 0
                        )
                    ),

                "n_images":
                    int(
                        len(
                            idx_pool
                        )
                    ),

                "bootstrap_reps":
                    BOOTSTRAP_REPS,

                "seed":
                    BOOTSTRAP_SEED,
            }
        )

    all_mask = np.ones(
        len(
            sel
        ),
        dtype=bool,
    )

    positive_mask = (
        groups
        ==
        "POSITIVE_ELEPHANT_PLUS_ANIMAL"
    )

    negative_mask = (
        groups
        ==
        "NEGATIVE_MULTI_ANIMAL_NO_ELEPHANT"
    )

    control_mask = (
        groups
        ==
        "CONTROL_ELEPHANT_ONLY_AMONG_ANIMAL_TARGETS"
    )

    run_bootstrap(
        "ALL_PRIMARY_453_PRECISION",
        all_mask,
        "precision",
        "selective_minus_comparator",
    )

    run_bootstrap(
        "ALL_PRIMARY_453_RECALL",
        all_mask,
        "recall",
        "selective_minus_comparator",
    )

    run_bootstrap(
        "ALL_PRIMARY_453_F1",
        all_mask,
        "f1",
        "selective_minus_comparator",
    )

    run_bootstrap(
        "NEGATIVE_FALSE_ELEPHANT_IMAGE_RATE",
        negative_mask,
        "false_elephant_image_rate",
        "comparator_minus_selective",
    )

    run_bootstrap(
        "POSITIVE_IMAGE_SENSITIVITY",
        positive_mask,
        "image_sensitivity",
        "selective_minus_comparator",
    )

    run_bootstrap(
        "POSITIVE_MIXED_OBJECT_RECALL",
        positive_mask,
        "recall",
        "selective_minus_comparator",
    )

    run_bootstrap(
        "CONTROL_ELEPHANT_OBJECT_RECALL",
        control_mask,
        "recall",
        "selective_minus_comparator",
    )

    run_bootstrap(
        "POSITIVE_CONFUSER_ERROR_RATE",
        positive_mask,
        "confuser_error_rate",
        "comparator_minus_selective",
    )

    return pd.DataFrame(
        comparisons
    )


def m5_gap_table(
    group_metrics,
):
    d = group_metrics.loc[
        np.isclose(
            group_metrics[
                "confidence_threshold"
            ],
            PRIMARY_CONF,
        )
    ].copy()

    rows = []

    for detector_id in [
        SELECTIVE_ID,
        COMPARATOR_ID,
    ]:
        dd = d.loc[
            d[
                "detector_id"
            ]
            ==
            detector_id
        ]

        mixed = dd.loc[
            dd[
                "scope"
            ]
            ==
            "POSITIVE_ELEPHANT_PLUS_ANIMAL",
            "recall",
        ]

        control = dd.loc[
            dd[
                "scope"
            ]
            ==
            "CONTROL_ELEPHANT_ONLY_AMONG_ANIMAL_TARGETS",
            "recall",
        ]

        if len(
            mixed
        ) != 1 or len(
            control
        ) != 1:
            raise RuntimeError(
                "Could not resolve M5 group recalls."
            )

        mixed_v = float(
            mixed.iloc[
                0
            ]
        )
        control_v = float(
            control.iloc[
                0
            ]
        )

        rows.append(
            {
                "detector_id":
                    detector_id,

                "mixed_scene_recall":
                    mixed_v,

                "elephant_only_control_recall":
                    control_v,

                "mixed_minus_control_recall":
                    mixed_v
                    -
                    control_v,
            }
        )

    return pd.DataFrame(
        rows
    )


def pair_detailed_metrics(
    primary,
    coco_gt,
    image_metrics,
):
    d = image_metrics.loc[
        np.isclose(
            image_metrics[
                "confidence_threshold"
            ],
            PRIMARY_CONF,
        )
    ].copy()

    image_pairs = defaultdict(
        set
    )

    animal_gt = coco_gt.loc[
        (
            coco_gt[
                "category_name"
            ]
            .astype(str)
            .str.lower()
            .isin(
                ANIMAL_CONFUSERS
            )
        )
        &
        (
            coco_gt[
                "iscrowd"
            ]
            ==
            0
        )
    ]

    for _, r in animal_gt.iterrows():
        key = (
            str(
                r["split"]
            ),
            int(
                r["image_id"]
            ),
        )

        image_pairs[
            key
        ].add(
            str(
                r[
                    "category_name"
                ]
            ).lower()
        )

    rows = []

    for pair in sorted(
        ANIMAL_CONFUSERS
    ):
        keys = {
            key
            for key, classes in image_pairs.items()
            if pair in classes
        }

        if not keys:
            continue

        pair_positive_mask = pd.Series(
            [
                (
                    (
                        str(r["split"]),
                        int(r["image_id"]),
                    )
                    in keys
                )
                and
                (
                    str(
                        r[
                            "evaluation_group"
                        ]
                    )
                    ==
                    "POSITIVE_ELEPHANT_PLUS_ANIMAL"
                )
                for _, r in d.iterrows()
            ],
            index=d.index,
            dtype=bool,
        )

        for detector_id in [
            SELECTIVE_ID,
            COMPARATOR_ID,
        ]:
            dd = d.loc[
                (
                    d[
                        "detector_id"
                    ]
                    ==
                    detector_id
                )
                &
                pair_positive_mask
            ]

            if len(
                dd
            ) == 0:
                continue

            agg = aggregate_object_metrics(
                dd,
                f"elephant+{pair}",
            )

            rows.append(
                {
                    "pair":
                        f"elephant+{pair}",

                    "detector_id":
                        detector_id,

                    **agg,
                }
            )

    return pd.DataFrame(
        rows
    )


def supplementary_metrics(
    supp_gt,
    raw,
):
    # Build image identity by SHA for 9 supplementary images.
    supp_raw = raw.loc[
        raw[
            "dataset_scope"
        ]
        ==
        "SUPPLEMENTARY_IN_VOCAB_9"
    ].copy()

    # Map image_path to GT boxes because supplementary image_id in R17-C
    # is an internal 1..9 identity.
    gt_by_path_ele = defaultdict(
        list
    )
    gt_by_path_lion = defaultdict(
        list
    )

    for _, r in supp_gt.iterrows():
        p = str(
            r["image_path"]
        )

        box = [
            float(
                r["bbox_x"]
            ),
            float(
                r["bbox_y"]
            ),
            float(
                r["bbox_x2"]
            ),
            float(
                r["bbox_y2"]
            ),
        ]

        name = str(
            r[
                "class_name"
            ]
        ).lower()

        if name == "elephant":
            gt_by_path_ele[
                p
            ].append(
                box
            )

        elif name == "lion":
            gt_by_path_lion[
                p
            ].append(
                box
            )

    rows = []

    all_paths = sorted(
        set(
            supp_raw[
                "image_path"
            ].astype(
                str
            )
        )
    )

    for detector_id in [
        SELECTIVE_ID,
        COMPARATOR_ID,
    ]:
        tp = fp = fn = 0
        lion_confusion_preds = 0
        lion_gt_total = 0
        correct_images = 0
        any_elephant_images = 0

        for pth in all_paths:
            preds_df = supp_raw.loc[
                (
                    supp_raw[
                        "detector_id"
                    ]
                    ==
                    detector_id
                )
                &
                (
                    supp_raw[
                        "image_path"
                    ]
                    .astype(
                        str
                    )
                    ==
                    pth
                )
                &
                (
                    supp_raw[
                        "class_name"
                    ]
                    .astype(
                        str
                    )
                    .str.lower()
                    ==
                    "elephant"
                )
                &
                (
                    supp_raw[
                        "confidence"
                    ]
                    >=
                    PRIMARY_CONF
                )
            ].sort_values(
                "confidence",
                ascending=False,
            )

            preds = [
                {
                    "confidence":
                        float(
                            r[
                                "confidence"
                            ]
                        ),

                    "box":
                        np.array(
                            [
                                float(
                                    r[
                                        "x1"
                                    ]
                                ),
                                float(
                                    r[
                                        "y1"
                                    ]
                                ),
                                float(
                                    r[
                                        "x2"
                                    ]
                                ),
                                float(
                                    r[
                                        "y2"
                                    ]
                                ),
                            ],
                            dtype=np.float64,
                        ),
                }
                for _, r in preds_df.iterrows()
            ]

            lion_items = [
                (
                    "lion",
                    box,
                )
                for box in gt_by_path_lion.get(
                    pth,
                    [],
                )
            ]

            m = evaluate_image(
                key=(
                    "supp",
                    pth,
                ),
                preds=preds,
                gt_boxes=gt_by_path_ele.get(
                    pth,
                    [],
                ),
                crowd_boxes=[],
                confuser_items=lion_items,
                iou_threshold=MATCH_IOU,
            )

            tp += m["tp"]
            fp += m["fp"]
            fn += m["fn"]
            lion_confusion_preds += m[
                "confuser_error_preds"
            ]
            lion_gt_total += m[
                "confuser_gt_count"
            ]
            correct_images += m[
                "image_has_correct_elephant_detection"
            ]
            any_elephant_images += m[
                "image_has_any_elephant_prediction"
            ]

        precision = safe_div(
            tp,
            tp + fp,
        )
        recall = safe_div(
            tp,
            tp + fn,
        )

        rows.append(
            {
                "detector_id":
                    detector_id,

                "images":
                    len(
                        all_paths
                    ),

                "elephant_tp":
                    tp,

                "elephant_fp":
                    fp,

                "elephant_fn":
                    fn,

                "elephant_precision":
                    precision,

                "elephant_recall":
                    recall,

                "elephant_f1":
                    f1_score(
                        precision,
                        recall,
                    ),

                "elephant_positive_image_sensitivity":
                    safe_div(
                        correct_images,
                        len(
                            all_paths
                        ),
                    ),

                "images_with_any_elephant_prediction":
                    any_elephant_images,

                "lion_gt_boxes":
                    lion_gt_total,

                "lion_to_elephant_overlap_error_predictions":
                    lion_confusion_preds,

                "lion_confusion_error_per_lion_gt":
                    safe_div(
                        lion_confusion_preds,
                        lion_gt_total,
                    ),
            }
        )

    # Selective detector lion-class detection summary at conf=.25
    lion_pred = supp_raw.loc[
        (
            supp_raw[
                "detector_id"
            ]
            ==
            SELECTIVE_ID
        )
        &
        (
            supp_raw[
                "class_name"
            ]
            .astype(
                str
            )
            .str.lower()
            ==
            "lion"
        )
        &
        (
            supp_raw[
                "confidence"
            ]
            >=
            PRIMARY_CONF
        )
    ]

    extra = {
        "selective_lion_predictions_conf025":
            int(
                len(
                    lion_pred
                )
            ),
    }

    return (
        pd.DataFrame(
            rows
        ),
        extra,
    )


def any_class_alarm_ablation(
    primary,
    raw,
):
    neg = primary.loc[
        primary[
            "evaluation_group"
        ]
        ==
        "NEGATIVE_MULTI_ANIMAL_NO_ELEPHANT"
    ]

    pos = primary.loc[
        primary[
            "evaluation_group"
        ]
        ==
        "POSITIVE_ELEPHANT_PLUS_ANIMAL"
    ]

    def keys_of(df):
        return {
            (
                str(r["split"]),
                int(r["image_id"]),
            )
            for _, r in df.iterrows()
        }

    neg_keys = keys_of(
        neg
    )
    pos_keys = keys_of(
        pos
    )

    sel = raw.loc[
        (
            raw[
                "detector_id"
            ]
            ==
            SELECTIVE_ID
        )
        &
        (
            raw[
                "dataset_scope"
            ]
            ==
            "PRIMARY_COCO_453"
        )
        &
        (
            raw[
                "confidence"
            ]
            >=
            PRIMARY_CONF
        )
    ].copy()

    sel["key"] = list(
        zip(
            sel[
                "split"
            ].astype(
                str
            ),
            sel[
                "image_id"
            ].astype(
                int
            ),
        )
    )

    any_class_keys = set(
        sel[
            "key"
        ]
    )

    elephant_keys = set(
        sel.loc[
            sel[
                "class_name"
            ]
            .astype(
                str
            )
            .str.lower()
            ==
            "elephant",
            "key",
        ]
    )

    rows = []

    for mode, alarm_keys in [
        (
            "UNFILTERED_ANY_CLASS_ALARM",
            any_class_keys,
        ),
        (
            "ELEPHANT_CLASS_FILTERED_ALARM",
            elephant_keys,
        ),
    ]:
        rows.append(
            {
                "alarm_mode":
                    mode,

                "negative_images":
                    len(
                        neg_keys
                    ),

                "negative_alarm_images":
                    len(
                        neg_keys
                        &
                        alarm_keys
                    ),

                "negative_alarm_rate":
                    safe_div(
                        len(
                            neg_keys
                            &
                            alarm_keys
                        ),
                        len(
                            neg_keys
                        ),
                    ),

                "positive_mixed_images":
                    len(
                        pos_keys
                    ),

                "positive_alarm_images":
                    len(
                        pos_keys
                        &
                        alarm_keys
                    ),

                "positive_alarm_rate":
                    safe_div(
                        len(
                            pos_keys
                            &
                            alarm_keys
                        ),
                        len(
                            pos_keys
                        ),
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def decision_summary(
    group_metrics,
    bootstrap,
    ap_table,
):
    primary = group_metrics.loc[
        np.isclose(
            group_metrics[
                "confidence_threshold"
            ],
            PRIMARY_CONF,
        )
        &
        (
            group_metrics[
                "scope"
            ]
            ==
            "ALL_PRIMARY_453"
        )
    ]

    def get(det, metric):
        x = primary.loc[
            primary[
                "detector_id"
            ]
            ==
            det,
            metric,
        ]

        return float(
            x.iloc[
                0
            ]
        )

    def boot(name):
        x = bootstrap.loc[
            bootstrap[
                "comparison"
            ]
            ==
            name
        ]

        if len(
            x
        ) != 1:
            return None

        return x.iloc[
            0
        ]

    false_row = boot(
        "NEGATIVE_FALSE_ELEPHANT_IMAGE_RATE"
    )

    sens_row = boot(
        "POSITIVE_IMAGE_SENSITIVITY"
    )

    f1_row = boot(
        "ALL_PRIMARY_453_F1"
    )

    ap50 = {}

    ap5095 = {}

    for det in [
        SELECTIVE_ID,
        COMPARATOR_ID,
    ]:
        ap50[
            det
        ] = float(
            ap_table.loc[
                (
                    ap_table[
                        "detector_id"
                    ]
                    ==
                    det
                )
                &
                (
                    ap_table[
                        "metric"
                    ]
                    ==
                    "AP50"
                ),
                "value",
            ].iloc[
                0
            ]
        )

        ap5095[
            det
        ] = float(
            ap_table.loc[
                (
                    ap_table[
                        "detector_id"
                    ]
                    ==
                    det
                )
                &
                (
                    ap_table[
                        "metric"
                    ]
                    ==
                    "AP50:95"
                ),
                "value",
            ].iloc[
                0
            ]
        )

    more_selective_supported = bool(
        false_row is not None
        and
        float(
            false_row[
                "ci95_low"
            ]
        )
        >
        0
    )

    sensitivity_noninferior_descriptive = bool(
        sens_row is not None
        and
        float(
            sens_row[
                "ci95_low"
            ]
        )
        >=
        0
    )

    rows = [
        {
            "question":
                "Does selective 4-class filtering reduce false-elephant negative-image alarms vs elephant-only comparator?",

            "prelocked_decision_rule":
                "paired bootstrap delta = comparator - selective; support only if 95% CI lower bound > 0",

            "result_supported":
                more_selective_supported,

            "point_delta":
                (
                    float(
                        false_row[
                            "point_delta_positive_favors_selective"
                        ]
                    )
                    if false_row is not None
                    else float(
                        "nan"
                    )
                ),

            "ci95_low":
                (
                    float(
                        false_row[
                            "ci95_low"
                        ]
                    )
                    if false_row is not None
                    else float(
                        "nan"
                    )
                ),

            "ci95_high":
                (
                    float(
                        false_row[
                            "ci95_high"
                        ]
                    )
                    if false_row is not None
                    else float(
                        "nan"
                    )
                ),
        },
        {
            "question":
                "Does selective detector improve/maintain positive-image sensitivity vs comparator?",

            "prelocked_decision_rule":
                "paired bootstrap delta = selective - comparator; descriptive support only if CI lower bound >= 0; no formal non-inferiority margin was pre-specified",

            "result_supported":
                sensitivity_noninferior_descriptive,

            "point_delta":
                (
                    float(
                        sens_row[
                            "point_delta_positive_favors_selective"
                        ]
                    )
                    if sens_row is not None
                    else float(
                        "nan"
                    )
                ),

            "ci95_low":
                (
                    float(
                        sens_row[
                            "ci95_low"
                        ]
                    )
                    if sens_row is not None
                    else float(
                        "nan"
                    )
                ),

            "ci95_high":
                (
                    float(
                        sens_row[
                            "ci95_high"
                        ]
                    )
                    if sens_row is not None
                    else float(
                        "nan"
                    )
                ),
        },
        {
            "question":
                "Overall F1 comparison",

            "prelocked_decision_rule":
                "paired bootstrap delta = selective - comparator; CI reported",

            "result_supported":
                (
                    bool(
                        float(
                            f1_row[
                                "ci95_low"
                            ]
                        )
                        >
                        0
                    )
                    if f1_row is not None
                    else False
                ),

            "point_delta":
                (
                    float(
                        f1_row[
                            "point_delta_positive_favors_selective"
                        ]
                    )
                    if f1_row is not None
                    else float(
                        "nan"
                    )
                ),

            "ci95_low":
                (
                    float(
                        f1_row[
                            "ci95_low"
                        ]
                    )
                    if f1_row is not None
                    else float(
                        "nan"
                    )
                ),

            "ci95_high":
                (
                    float(
                        f1_row[
                            "ci95_high"
                        ]
                    )
                    if f1_row is not None
                    else float(
                        "nan"
                    )
                ),
        },
        {
            "question":
                "AP50 / AP50:95 descriptive comparison",

            "prelocked_decision_rule":
                "report both detectors; no bootstrap significance rule pre-locked for AP",

            "result_supported":
                None,

            "point_delta":
                ap50[
                    SELECTIVE_ID
                ]
                -
                ap50[
                    COMPARATOR_ID
                ],

            "ci95_low":
                float(
                    "nan"
                ),

            "ci95_high":
                float(
                    "nan"
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR17-D — LOCKED MIXED-SPECIES METRIC EVALUATION"
    )

    print(
        "EVALUATION ONLY. Uses the frozen R17-B protocol and frozen R17-C raw predictions. "
        "No model rerun, no threshold tuning, and no result-dependent operating-point changes. "
        "Primary scoring uses conf=0.25 and IoU=0.50. The 0.10 and 0.50 thresholds are reported only as pre-locked sensitivity analyses.\n"
    )

    protocol, r17c = verify_inputs()

    (
        primary,
        coco_gt,
        supp,
        supp_gt,
        raw,
    ) = load_data()

    (
        ele_map,
        crowd_map,
        conf_map,
        elephant_gt,
        elephant_crowd_gt,
        confuser_gt,
    ) = build_gt_maps(
        coco_gt
    )

    print(
        "GT AUDIT:"
    )
    print(
        f"  scored non-crowd elephant GT boxes = {len(elephant_gt)}"
    )
    print(
        f"  excluded elephant crowd GT boxes = {len(elephant_crowd_gt)}"
    )
    print(
        f"  scored non-crowd animal-confuser GT boxes = {len(confuser_gt)}"
    )

    all_image_metrics = []

    for conf in SENSITIVITY_CONFS:
        print(
            f"\nEVALUATING FIXED CONFIDENCE {conf:.2f} ..."
        )

        im = evaluate_primary_threshold(
            primary,
            raw,
            ele_map,
            crowd_map,
            conf_map,
            conf,
        )

        all_image_metrics.append(
            im
        )

    image_metrics = pd.concat(
        all_image_metrics,
        ignore_index=True,
    )

    image_metrics_path = (
        OUT
        / "R17D_IMAGE_LEVEL_METRICS.csv"
    )

    image_metrics.to_csv(
        image_metrics_path,
        index=False,
    )

    group_metrics = group_metric_table(
        image_metrics
    )

    group_metrics_path = (
        OUT
        / "R17D_GROUP_METRICS.csv"
    )

    group_metrics.to_csv(
        group_metrics_path,
        index=False,
    )

    m5 = m5_gap_table(
        group_metrics
    )

    m5_path = (
        OUT
        / "R17D_M5_MIXED_VS_CONTROL_RECALL_GAP.csv"
    )

    m5.to_csv(
        m5_path,
        index=False,
    )

    print(
        "\nCOMPUTING AP50 / AP50:95 ..."
    )

    ap_table = compute_ap_table(
        primary,
        raw,
        ele_map,
        crowd_map,
    )

    ap_path = (
        OUT
        / "R17D_AP_METRICS.csv"
    )

    ap_table.to_csv(
        ap_path,
        index=False,
    )

    pair_metrics = pair_detailed_metrics(
        primary,
        coco_gt,
        image_metrics,
    )

    pair_path = (
        OUT
        / "R17D_PER_CONFUSER_PAIR_METRICS.csv"
    )

    pair_metrics.to_csv(
        pair_path,
        index=False,
    )

    supp_metrics, supp_extra = (
        supplementary_metrics(
            supp_gt,
            raw,
        )
    )

    supp_path = (
        OUT
        / "R17D_SUPPLEMENTARY_IN_VOCAB_METRICS.csv"
    )

    supp_metrics.to_csv(
        supp_path,
        index=False,
    )

    alarm_ablation = any_class_alarm_ablation(
        primary,
        raw,
    )

    alarm_path = (
        OUT
        / "R17D_SELECTIVE_FILTER_ALARM_ABLATION.csv"
    )

    alarm_ablation.to_csv(
        alarm_path,
        index=False,
    )

    print(
        "\nRUNNING PRE-LOCKED 20,000-REPLICATE PAIRED IMAGE BOOTSTRAP ..."
    )

    bootstrap = bootstrap_ci(
        image_metrics
    )

    bootstrap_path = (
        OUT
        / "R17D_PAIRED_BOOTSTRAP_CI.csv"
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    decisions = decision_summary(
        group_metrics,
        bootstrap,
        ap_table,
    )

    decision_path = (
        OUT
        / "R17D_DECISION_SUMMARY.csv"
    )

    decisions.to_csv(
        decision_path,
        index=False,
    )

    primary025 = group_metrics.loc[
        np.isclose(
            group_metrics[
                "confidence_threshold"
            ],
            PRIMARY_CONF,
        )
    ].copy()

    primary025_path = (
        OUT
        / "R17D_PRIMARY_CONF025_METRICS.csv"
    )

    primary025.to_csv(
        primary025_path,
        index=False,
    )

    sensitivity_path = (
        OUT
        / "R17D_FIXED_THRESHOLD_SENSITIVITY.csv"
    )

    group_metrics.to_csv(
        sensitivity_path,
        index=False,
    )

    status = (
        "R17D_LOCKED_MIXED_SPECIES_METRICS_BOOTSTRAP_AND_FILTER_ABLATION_COMPLETE_READY_FOR_R17E_CLAIM_INTERPRETATION_LOCK"
    )

    manifest = {
        "step":
            "R17D_LOCKED_MIXED_SPECIES_METRIC_EVALUATION",

        "status":
            status,

        "model_rerun":
            False,

        "threshold_tuning":
            False,

        "primary_confidence":
            PRIMARY_CONF,

        "sensitivity_confidences":
            SENSITIVITY_CONFS,

        "primary_match_iou":
            MATCH_IOU,

        "ap_iou_grid":
            AP_IOUS,

        "crowd_handling":
            (
                "iscrowd=1 elephant GT excluded from scored denominator; "
                "predictions overlapping elephant crowd GT at scoring IoU are ignored"
            ),

        "bootstrap_reps":
            BOOTSTRAP_REPS,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "supplementary_selective_lion_predictions_conf025":
            supp_extra[
                "selective_lion_predictions_conf025"
            ],

        "input_hashes": {
            "r17c_raw_predictions":
                sha256_file(
                    RAW_PRED_PATH
                ),

            "r17b_coco_gt":
                sha256_file(
                    COCO_GT_PATH
                ),

            "r17b_supp_gt":
                sha256_file(
                    SUPP_GT_PATH
                ),
        },

        "output_hashes": {
            "image_metrics":
                sha256_file(
                    image_metrics_path
                ),

            "group_metrics":
                sha256_file(
                    group_metrics_path
                ),

            "ap_metrics":
                sha256_file(
                    ap_path
                ),

            "pair_metrics":
                sha256_file(
                    pair_path
                ),

            "supplementary_metrics":
                sha256_file(
                    supp_path
                ),

            "alarm_ablation":
                sha256_file(
                    alarm_path
                ),

            "bootstrap_ci":
                sha256_file(
                    bootstrap_path
                ),

            "decision_summary":
                sha256_file(
                    decision_path
                ),
        },

        "important_interpretation_limits": [
            "R17 evaluates image-level detector/selective-filter robustness, not detector-to-tracker-to-forecast end-to-end performance.",
            "COCO animal classes are external OOD stressors relative to the 4-class checkpoint; success does not imply universal unseen-species robustness.",
            "The 9-image elephant+lion set is a small supplementary in-vocabulary challenge only.",
            "No formal non-inferiority margin was pre-specified for recall/sensitivity; absence of a significant loss is not proof of non-inferiority.",
            "AP values are descriptive comparisons; no AP bootstrap significance rule was pre-locked.",
        ],

        "outputs": {
            "image_level_metrics":
                str(
                    image_metrics_path
                ),

            "primary_conf025_metrics":
                str(
                    primary025_path
                ),

            "group_metrics":
                str(
                    group_metrics_path
                ),

            "m5_gap":
                str(
                    m5_path
                ),

            "ap_metrics":
                str(
                    ap_path
                ),

            "per_confuser_pair":
                str(
                    pair_path
                ),

            "supplementary_metrics":
                str(
                    supp_path
                ),

            "selective_filter_ablation":
                str(
                    alarm_path
                ),

            "bootstrap_ci":
                str(
                    bootstrap_path
                ),

            "decision_summary":
                str(
                    decision_path
                ),
        },
    }

    manifest_path = (
        OUT
        / "R17D_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nPRIMARY CONF=0.25 METRICS:"
    )

    print(
        primary025.to_string(
            index=False
        )
    )

    print(
        "\nAP SUMMARY:"
    )

    print(
        ap_table.loc[
            ap_table[
                "metric"
            ].isin(
                [
                    "AP50",
                    "AP50:95",
                ]
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\nM5 MIXED VS CONTROL RECALL GAP:"
    )

    print(
        m5.to_string(
            index=False
        )
    )

    print(
        "\nPAIRED BOOTSTRAP CI:"
    )

    print(
        bootstrap.to_string(
            index=False
        )
    )

    print(
        "\nSUPPLEMENTARY IN-VOCAB METRICS:"
    )

    print(
        supp_metrics.to_string(
            index=False
        )
    )

    print(
        f"\nselective lion-class predictions @ conf=0.25 = {supp_extra['selective_lion_predictions_conf025']}"
    )

    print(
        "\nSELECTIVE FILTER ALARM ABLATION:"
    )

    print(
        alarm_ablation.to_string(
            index=False
        )
    )

    print(
        "\nDECISION SUMMARY:"
    )

    print(
        decisions.to_string(
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
        "\nR17-D COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
