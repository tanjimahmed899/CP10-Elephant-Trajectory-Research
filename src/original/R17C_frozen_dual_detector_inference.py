
from pathlib import Path
import hashlib
import json
import math
import time
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

# Frozen R17-B protocol
R17B_DIR = R17_ROOT / "R17B_frozen_robustness_protocol_lock"
PROTOCOL = R17B_DIR / "R17B_PROTOCOL_LOCK.json"
DETECTOR_LOCK = R17B_DIR / "R17B_DETECTOR_LOCK.csv"

# Frozen primary/supplement datasets
A4A_DIR = R17_ROOT / "R17A4A_frozen_subset_selection"
PRIMARY_453 = A4A_DIR / "R17A4A_PRIMARY_BALANCED_453.csv"
SUPP_9 = A4A_DIR / "R17A4A_SUPPLEMENTARY_IN_VOCAB_UNIQUE.csv"

A4B_DIR = R17_ROOT / "R17A4B_frozen_pixel_download"
A4B_AUDIT = A4B_DIR / "R17A4B_PIXEL_DOWNLOAD_AUDIT.csv"

OUT = R17_ROOT / "R17C_frozen_dual_detector_inference"
OUT.mkdir(parents=True, exist_ok=True)

RAW_PRED_PATH = OUT / "R17C_RAW_PREDICTIONS.csv"
IMAGE_AUDIT_PATH = OUT / "R17C_IMAGE_INFERENCE_AUDIT.csv"
SUMMARY_PATH = OUT / "R17C_INFERENCE_SUMMARY.csv"
MANIFEST_PATH = OUT / "R17C_MANIFEST.json"

BATCH_SIZE = 16


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def read_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def boolish(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes"}


def verify_protocol():
    for p in [
        PROTOCOL,
        DETECTOR_LOCK,
        PRIMARY_453,
        SUPP_9,
        A4B_AUDIT,
    ]:
        require(p)

    protocol = read_json(PROTOCOL)

    expected = (
        "R17B_FROZEN_ROBUSTNESS_PROTOCOL_LOCKED_NO_TEST_TUNING_READY_FOR_R17C_FROZEN_DUAL_DETECTOR_INFERENCE"
    )

    if protocol.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R17-B status: {protocol.get('status')}"
        )

    if boolish(protocol.get("inference", True)):
        raise RuntimeError(
            "R17-B protocol unexpectedly indicates inference already occurred."
        )

    detector_lock = pd.read_csv(
        DETECTOR_LOCK,
        low_memory=False,
    )

    if len(detector_lock) != 2:
        raise RuntimeError(
            f"Expected exactly 2 frozen detectors, found {len(detector_lock)}."
        )

    return protocol, detector_lock


def build_primary_image_table():
    selected = pd.read_csv(
        PRIMARY_453,
        low_memory=False,
    )

    pixel = pd.read_csv(
        A4B_AUDIT,
        low_memory=False,
    )

    selected["image_id"] = pd.to_numeric(
        selected["image_id"],
        errors="raise",
    ).astype(int)

    pixel["image_id"] = pd.to_numeric(
        pixel["image_id"],
        errors="raise",
    ).astype(int)

    good = pixel["status"].isin(
        [
            "DOWNLOADED_VALID",
            "EXISTING_VALID",
        ]
    )

    pixel = pixel.loc[
        good
    ].copy()

    cols = [
        "evaluation_group",
        "split",
        "image_id",
        "local_path",
        "sha256",
    ]

    merged = selected.merge(
        pixel[cols],
        on=[
            "evaluation_group",
            "split",
            "image_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(merged) != 453:
        raise RuntimeError(
            f"Primary merge should contain 453 images, found {len(merged)}."
        )

    if merged["local_path"].isna().any():
        raise RuntimeError(
            "One or more frozen primary images are missing local pixel paths."
        )

    rows = []

    for _, r in merged.iterrows():
        p = Path(
            str(
                r["local_path"]
            )
        )

        require(p)

        digest = sha256_file(
            p
        )

        if digest != str(
            r["sha256"]
        ).strip():
            raise RuntimeError(
                f"Primary image SHA changed since R17-A4B: {p}"
            )

        rows.append(
            {
                "dataset_scope":
                    "PRIMARY_COCO_453",

                "evaluation_group":
                    str(
                        r["evaluation_group"]
                    ),

                "split":
                    str(
                        r["split"]
                    ),

                "image_id":
                    int(
                        r["image_id"]
                    ),

                "image_sha256":
                    digest,

                "image_path":
                    str(
                        p
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_supp_image_table():
    supp = pd.read_csv(
        SUPP_9,
        low_memory=False,
    )

    if len(supp) != 9:
        raise RuntimeError(
            f"Expected 9 frozen supplementary images, found {len(supp)}."
        )

    rows = []

    for idx, r in supp.iterrows():
        p = Path(
            str(
                r["image_path"]
            )
        )

        require(p)

        digest = sha256_file(
            p
        )

        locked = str(
            r["sha256"]
        ).strip()

        if digest != locked:
            raise RuntimeError(
                f"Supplementary image SHA changed: {p}"
            )

        rows.append(
            {
                "dataset_scope":
                    "SUPPLEMENTARY_IN_VOCAB_9",

                "evaluation_group":
                    "SUPPLEMENTARY_IN_VOCAB_ELEPHANT_PLUS_LION",

                "split":
                    str(
                        r.get(
                            "split",
                            "Test",
                        )
                    ),

                "image_id":
                    int(
                        idx + 1
                    ),

                "image_sha256":
                    digest,

                "image_path":
                    str(
                        p
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def chunked(df, batch_size):
    n = len(df)

    for start in range(
        0,
        n,
        batch_size,
    ):
        yield (
            start,
            df.iloc[
                start:start + batch_size
            ].copy(),
        )


def names_as_dict(model):
    names = model.names

    if isinstance(names, dict):
        return {
            int(k): str(v)
            for k, v in names.items()
        }

    return {
        i: str(v)
        for i, v in enumerate(
            names
        )
    }


def infer_detector(
    detector_row,
    image_table,
    protocol,
):
    from ultralytics import YOLO

    detector_id = str(
        detector_row["detector_id"]
    )

    checkpoint = Path(
        str(
            detector_row["checkpoint"]
        )
    )

    locked_ckpt_sha = str(
        detector_row["sha256"]
    ).strip()

    require(checkpoint)

    actual_ckpt_sha = sha256_file(
        checkpoint
    )

    if actual_ckpt_sha != locked_ckpt_sha:
        raise RuntimeError(
            f"Checkpoint SHA mismatch for {detector_id}."
        )

    settings = protocol

    conf_floor = float(
        settings[
            "confidence_protocol"
        ][
            "inference_floor"
        ]
    )

    # Read exact inference settings from the frozen CSV-style protocol values.
    imgsz = 640
    nms_iou = 0.70
    max_det = 300
    agnostic_nms = False
    augment = False

    model = YOLO(
        str(
            checkpoint
        )
    )

    class_names = names_as_dict(
        model
    )

    pred_rows = []
    audit_rows = []

    total_images = len(
        image_table
    )

    wall_start = time.perf_counter()

    processed = 0

    for start, batch in chunked(
        image_table,
        BATCH_SIZE,
    ):
        paths = batch[
            "image_path"
        ].tolist()

        batch_start = time.perf_counter()

        results = model.predict(
            source=paths,
            imgsz=imgsz,
            conf=conf_floor,
            iou=nms_iou,
            max_det=max_det,
            agnostic_nms=agnostic_nms,
            augment=augment,
            verbose=False,
            save=False,
            stream=False,
        )

        batch_elapsed = (
            time.perf_counter()
            -
            batch_start
        )

        if len(results) != len(batch):
            raise RuntimeError(
                f"{detector_id}: result count {len(results)} does not match batch size {len(batch)}."
            )

        per_image_wall_ms = (
            1000.0
            *
            batch_elapsed
            /
            max(
                len(batch),
                1,
            )
        )

        for local_idx, (_, meta) in enumerate(
            batch.iterrows()
        ):
            result = results[
                local_idx
            ]

            orig_shape = result.orig_shape

            if orig_shape is None:
                h = None
                w = None
            else:
                h = int(
                    orig_shape[0]
                )
                w = int(
                    orig_shape[1]
                )

            boxes = result.boxes

            n_pred = (
                int(
                    len(boxes)
                )
                if boxes is not None
                else 0
            )

            speed = (
                result.speed
                if isinstance(
                    result.speed,
                    dict,
                )
                else {}
            )

            audit_rows.append(
                {
                    "detector_id":
                        detector_id,

                    "dataset_scope":
                        meta[
                            "dataset_scope"
                        ],

                    "evaluation_group":
                        meta[
                            "evaluation_group"
                        ],

                    "split":
                        meta[
                            "split"
                        ],

                    "image_id":
                        int(
                            meta[
                                "image_id"
                            ]
                        ),

                    "image_sha256":
                        meta[
                            "image_sha256"
                        ],

                    "image_path":
                        meta[
                            "image_path"
                        ],

                    "image_width":
                        w,

                    "image_height":
                        h,

                    "num_predictions_at_conf_floor":
                        n_pred,

                    "batch_wall_ms_per_image":
                        per_image_wall_ms,

                    "preprocess_ms":
                        float(
                            speed.get(
                                "preprocess",
                                math.nan,
                            )
                        ),

                    "inference_ms":
                        float(
                            speed.get(
                                "inference",
                                math.nan,
                            )
                        ),

                    "postprocess_ms":
                        float(
                            speed.get(
                                "postprocess",
                                math.nan,
                            )
                        ),
                }
            )

            if boxes is None or len(boxes) == 0:
                continue

            xyxy = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            confs = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            clss = (
                boxes.cls
                .detach()
                .cpu()
                .numpy()
            )

            for j in range(
                len(boxes)
            ):
                cid = int(
                    clss[j]
                )

                x1, y1, x2, y2 = [
                    float(v)
                    for v in xyxy[j]
                ]

                pred_rows.append(
                    {
                        "detector_id":
                            detector_id,

                        "dataset_scope":
                            meta[
                                "dataset_scope"
                            ],

                        "evaluation_group":
                            meta[
                                "evaluation_group"
                            ],

                        "split":
                            meta[
                                "split"
                            ],

                        "image_id":
                            int(
                                meta[
                                    "image_id"
                                ]
                            ),

                        "image_sha256":
                            meta[
                                "image_sha256"
                            ],

                        "image_path":
                            meta[
                                "image_path"
                            ],

                        "prediction_index":
                            int(
                                j
                            ),

                        "class_id":
                            cid,

                        "class_name":
                            class_names.get(
                                cid,
                                f"UNKNOWN_{cid}",
                            ),

                        "confidence":
                            float(
                                confs[j]
                            ),

                        "x1":
                            x1,

                        "y1":
                            y1,

                        "x2":
                            x2,

                        "y2":
                            y2,

                        "width":
                            float(
                                x2 - x1
                            ),

                        "height":
                            float(
                                y2 - y1
                            ),
                    }
                )

        processed += len(
            batch
        )

        print(
            f"  {detector_id}: {processed}/{total_images} images"
        )

    wall_elapsed = (
        time.perf_counter()
        -
        wall_start
    )

    return (
        pd.DataFrame(
            pred_rows
        ),
        pd.DataFrame(
            audit_rows
        ),
        wall_elapsed,
    )


def main():
    print(
        "\nR17-C — FROZEN DUAL-DETECTOR RAW INFERENCE"
    )

    print(
        "INFERENCE ONLY. NO metric-based threshold selection, NO test tuning, and NO claim interpretation. "
        "Both frozen detectors are run on the exact frozen 453-image COCO primary set and the 9-image "
        "in-vocabulary supplement at the pre-locked confidence floor. All raw classes/confidences/boxes are saved.\n"
    )

    protocol, detector_lock = (
        verify_protocol()
    )

    primary = (
        build_primary_image_table()
    )

    supplement = (
        build_supp_image_table()
    )

    images = pd.concat(
        [
            primary,
            supplement,
        ],
        ignore_index=True,
    )

    if len(images) != 462:
        raise RuntimeError(
            f"Expected 462 total frozen images, found {len(images)}."
        )

    if images[
        [
            "dataset_scope",
            "image_sha256",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate image identity found within frozen R17-C image table."
        )

    all_preds = []
    all_audits = []
    summary_rows = []

    for _, detector_row in detector_lock.iterrows():
        detector_id = str(
            detector_row[
                "detector_id"
            ]
        )

        print(
            f"\nRUNNING {detector_id} ..."
        )

        preds, audit, elapsed = (
            infer_detector(
                detector_row,
                images,
                protocol,
            )
        )

        all_preds.append(
            preds
        )

        all_audits.append(
            audit
        )

        summary_rows.append(
            {
                "detector_id":
                    detector_id,

                "images_inferred":
                    len(
                        audit
                    ),

                "raw_predictions_at_conf_floor":
                    len(
                        preds
                    ),

                "images_with_at_least_one_prediction":
                    int(
                        (
                            audit[
                                "num_predictions_at_conf_floor"
                            ]
                            > 0
                        ).sum()
                    ),

                "wall_elapsed_seconds":
                    float(
                        elapsed
                    ),

                "mean_wall_ms_per_image":
                    float(
                        1000.0
                        * elapsed
                        / len(
                            audit
                        )
                    ),

                "mean_ultralytics_inference_ms":
                    float(
                        audit[
                            "inference_ms"
                        ].mean()
                    ),
            }
        )

    raw = pd.concat(
        all_preds,
        ignore_index=True,
        sort=False,
    )

    audit = pd.concat(
        all_audits,
        ignore_index=True,
        sort=False,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    expected_audit_rows = (
        len(
            detector_lock
        )
        *
        len(
            images
        )
    )

    if len(audit) != expected_audit_rows:
        raise RuntimeError(
            f"Expected {expected_audit_rows} image-detector audit rows, found {len(audit)}."
        )

    # Every detector must have exactly one audit row per frozen image.
    dup = audit.duplicated(
        subset=[
            "detector_id",
            "dataset_scope",
            "image_sha256",
        ]
    )

    if dup.any():
        raise RuntimeError(
            "Duplicate detector-image audit rows found."
        )

    raw.to_csv(
        RAW_PRED_PATH,
        index=False,
    )

    audit.to_csv(
        IMAGE_AUDIT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    if len(audit) == 924:
        status = (
            "R17C_FROZEN_DUAL_DETECTOR_INFERENCE_COMPLETE_RAW_PREDICTIONS_LOCKED_READY_FOR_R17D_METRIC_EVALUATION"
        )
    else:
        status = (
            "R17C_INFERENCE_INCOMPLETE_REVIEW_REQUIRED"
        )

    manifest = {
        "step":
            "R17C_FROZEN_DUAL_DETECTOR_INFERENCE",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "threshold_tuning":
            False,

        "metric_interpretation":
            False,

        "primary_images":
            int(
                len(
                    primary
                )
            ),

        "supplement_images":
            int(
                len(
                    supplement
                )
            ),

        "total_unique_images":
            int(
                len(
                    images
                )
            ),

        "detectors":
            int(
                len(
                    detector_lock
                )
            ),

        "detector_image_audit_rows":
            int(
                len(
                    audit
                )
            ),

        "raw_prediction_rows":
            int(
                len(
                    raw
                )
            ),

        "frozen_inference_settings": {
            "imgsz":
                640,

            "conf_floor":
                float(
                    protocol[
                        "confidence_protocol"
                    ][
                        "inference_floor"
                    ]
                ),

            "nms_iou":
                0.70,

            "max_det":
                300,

            "agnostic_nms":
                False,

            "augment":
                False,
        },

        "important_interpretation": [
            "This step produces raw frozen predictions only.",
            "No test-dependent confidence threshold was selected.",
            "The 4-class detector preserves all predicted classes in the raw output; elephant filtering is deferred to the pre-locked evaluation step.",
            "The wall-clock timing in this step is descriptive only and does not replace the frozen R16 latency analysis.",
        ],

        "output_sha256": {
            "raw_predictions":
                sha256_file(
                    RAW_PRED_PATH
                ),

            "image_inference_audit":
                sha256_file(
                    IMAGE_AUDIT_PATH
                ),

            "summary":
                sha256_file(
                    SUMMARY_PATH
                ),
        },

        "outputs": {
            "raw_predictions":
                str(
                    RAW_PRED_PATH
                ),

            "image_inference_audit":
                str(
                    IMAGE_AUDIT_PATH
                ),

            "summary":
                str(
                    SUMMARY_PATH
                ),
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nINFERENCE SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nRAW CLASS COUNTS:"
    )

    if len(
        raw
    ):
        class_counts = (
            raw.groupby(
                [
                    "detector_id",
                    "dataset_scope",
                    "class_name",
                ]
            )
            .size()
            .rename(
                "prediction_count"
            )
            .reset_index()
        )

        print(
            class_counts.to_string(
                index=False
            )
        )
    else:
        print(
            "No predictions at the frozen inference confidence floor."
        )

    print(
        "\nINTEGRITY:"
    )

    print(
        f"unique frozen images = {len(images)}"
    )

    print(
        f"detector-image audit rows = {len(audit)} / 924"
    )

    print(
        f"raw prediction rows = {len(raw)}"
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR17-C COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
