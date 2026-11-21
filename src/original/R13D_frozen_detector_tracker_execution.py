
from pathlib import Path
import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R13_ROOT = REVISION / "R13_end_to_end_pipeline"

R13C_DIR = R13_ROOT / "R13C_frozen_end_to_end_protocol_lock"
R13C_MANIFEST = R13C_DIR / "R13C_MANIFEST.json"
R13C_PROTOCOL = R13C_DIR / "R13C_FROZEN_PROTOCOL.json"
R13C_FRAME_MAP = R13C_DIR / "R13C_SEQUENCE_FRAME_MAPPING_LOCK.csv"

OUT = R13_ROOT / "R13D_frozen_detector_tracker_execution"
DET_DIR = OUT / "detections"
BYTE_DIR = OUT / "bytetrack"
BOT_DIR = OUT / "botsort"
LOG_DIR = OUT / "sequence_manifests"

for d in [OUT, DET_DIR, BYTE_DIR, BOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BOX_COLUMNS = [
    "sequence",
    "frame",
    "image_name",
    "image_path",
    "image_width",
    "image_height",
    "detection_index",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "w",
    "h",
]

FRAME_COLUMNS = [
    "sequence",
    "frame",
    "image_name",
    "image_path",
    "image_width",
    "image_height",
    "num_boxes",
    "num_assigned_track_ids",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
]


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


def parse_frame_number(path):
    nums = re.findall(r"(\d+)", Path(path).stem)
    if not nums:
        raise RuntimeError(
            f"Could not parse frame number from image filename: {path}"
        )
    return int(nums[-1])


def sorted_images(image_dir):
    imgs = [
        p for p in Path(image_dir).iterdir()
        if p.is_file()
        and p.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]

    return sorted(
        imgs,
        key=lambda p: (
            parse_frame_number(p),
            p.name,
        ),
    )


def set_reproducibility(seed=20261030):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def load_and_verify_protocol():
    for p in [
        R13C_MANIFEST,
        R13C_PROTOCOL,
        R13C_FRAME_MAP,
    ]:
        require(p)

    manifest = json.loads(
        R13C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R13C_FROZEN_END_TO_END_PROTOCOL_LOCKED_NO_TEST_TUNING_READY_FOR_R13D_DETECTOR_TRACKER_EXECUTION"
    )

    if manifest.get("status") != expected:
        raise RuntimeError(
            f"R13-C not in locked PASS state: {manifest.get('status')}"
        )

    protocol = json.loads(
        R13C_PROTOCOL.read_text(
            encoding="utf-8"
        )
    )

    if sha256_file(R13C_PROTOCOL) != manifest["protocol_sha256"]:
        raise RuntimeError(
            "Frozen R13-C protocol hash mismatch."
        )

    detector = Path(
        protocol["detector"]["checkpoint"]
    )
    byte_yaml = Path(
        protocol["tracking"]["primary_tracker_yaml"]
    )
    bot_yaml = Path(
        protocol["tracking"]["sensitivity_tracker_yaml"]
    )

    for p in [
        detector,
        byte_yaml,
        bot_yaml,
    ]:
        require(p)

    if sha256_file(detector) != protocol["detector"]["checkpoint_sha256"]:
        raise RuntimeError(
            "Frozen detector checkpoint hash mismatch."
        )

    if sha256_file(byte_yaml) != protocol["tracking"]["primary_tracker_yaml_sha256"]:
        raise RuntimeError(
            "Frozen ByteTrack YAML hash mismatch."
        )

    if sha256_file(bot_yaml) != protocol["tracking"]["sensitivity_tracker_yaml_sha256"]:
        raise RuntimeError(
            "Frozen BoT-SORT YAML hash mismatch."
        )

    return manifest, protocol


def device_lock():
    if torch.cuda.is_available():
        return 0, torch.cuda.get_device_name(0)

    return "cpu", "CPU"


def extract_result_rows(
    result,
    sequence,
    frame_no,
    image_path,
    allow_track_ids,
):
    h, w = result.orig_shape

    speed = getattr(
        result,
        "speed",
        {},
    ) or {}

    frame_row = {
        "sequence": sequence,
        "frame": frame_no,
        "image_name": image_path.name,
        "image_path": str(image_path),
        "image_width": int(w),
        "image_height": int(h),
        "num_boxes": 0,
        "num_assigned_track_ids": 0,
        "preprocess_ms": float(speed.get("preprocess", np.nan)),
        "inference_ms": float(speed.get("inference", np.nan)),
        "postprocess_ms": float(speed.get("postprocess", np.nan)),
    }

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return [], frame_row

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)

    if allow_track_ids and boxes.id is not None:
        ids = boxes.id.detach().cpu().numpy()
    else:
        ids = np.full(
            len(boxes),
            np.nan,
            dtype=float,
        )

    frame_row["num_boxes"] = int(
        len(boxes)
    )

    frame_row["num_assigned_track_ids"] = int(
        np.isfinite(ids).sum()
    )

    names = result.names

    rows = []

    for i in range(
        len(boxes)
    ):
        x1, y1, x2, y2 = [
            float(v)
            for v in xyxy[i]
        ]

        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        class_id = int(
            cls[i]
        )

        if isinstance(
            names,
            dict,
        ):
            class_name = str(
                names.get(
                    class_id,
                    class_id,
                )
            )
        else:
            try:
                class_name = str(
                    names[
                        class_id
                    ]
                )
            except Exception:
                class_name = str(
                    class_id
                )

        track_id = (
            int(
                ids[i]
            )
            if np.isfinite(
                ids[i]
            )
            else
            -1
        )

        rows.append(
            {
                "sequence": sequence,
                "frame": frame_no,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "image_width": int(w),
                "image_height": int(h),
                "detection_index": int(i),
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(conf[i]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "w": bw,
                "h": bh,
            }
        )

    return rows, frame_row


def component_paths(
    component,
    sequence,
):
    if component == "detections":
        base = DET_DIR
    elif component == "bytetrack":
        base = BYTE_DIR
    elif component == "botsort":
        base = BOT_DIR
    else:
        raise ValueError(
            component
        )

    return {
        "boxes": base / f"{sequence}_{component}_boxes.csv",
        "frames": base / f"{sequence}_{component}_frames.csv",
        "manifest": LOG_DIR / f"{sequence}_{component}_COMPLETE.json",
    }


def is_component_complete(
    component,
    sequence,
    expected_frames,
    frozen_hashes,
):
    paths = component_paths(
        component,
        sequence,
    )

    for p in paths.values():
        if not p.exists():
            return False

    try:
        m = json.loads(
            paths["manifest"].read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    if m.get("status") != "COMPLETE":
        return False

    if int(
        m.get(
            "frames_processed",
            -1,
        )
    ) != int(
        expected_frames
    ):
        return False

    if m.get("frozen_hashes") != frozen_hashes:
        return False

    return True


def run_sequence_component(
    component,
    sequence,
    image_dir,
    detector_ckpt,
    tracker_yaml,
    settings,
    device,
    frozen_hashes,
):
    images = sorted_images(
        image_dir
    )

    if not images:
        raise RuntimeError(
            f"No images in {image_dir}"
        )

    paths = component_paths(
        component,
        sequence,
    )

    if is_component_complete(
        component,
        sequence,
        len(images),
        frozen_hashes,
    ):
        print(
            f"  SKIP {component}: already complete and hash-verified."
        )
        return

    # New model object per sequence/component.
    # This guarantees tracker reset at sequence boundaries.
    model = YOLO(
        str(
            detector_ckpt
        )
    )

    all_boxes = []
    all_frames = []

    start = time.time()

    for idx, img in enumerate(
        images,
        start=1,
    ):
        frame_no = parse_frame_number(
            img
        )

        kwargs = {
            "source": str(img),
            "imgsz": int(settings["imgsz"]),
            "conf": float(settings["conf"]),
            "iou": float(settings["iou"]),
            "max_det": int(settings["max_det"]),
            "agnostic_nms": bool(settings["agnostic_nms"]),
            "augment": bool(settings["augment"]),
            "device": device,
            "half": False,
            "verbose": False,
        }

        classes = settings.get(
            "classes",
            None,
        )

        if classes is not None:
            kwargs["classes"] = classes

        if component == "detections":
            result_list = model.predict(
                **kwargs
            )
            allow_track_ids = False
        else:
            result_list = model.track(
                **kwargs,
                tracker=str(
                    tracker_yaml
                ),
                persist=True,
            )
            allow_track_ids = True

        if len(
            result_list
        ) != 1:
            raise RuntimeError(
                f"Expected one result for {img}, got {len(result_list)}"
            )

        box_rows, frame_row = extract_result_rows(
            result_list[0],
            sequence,
            frame_no,
            img,
            allow_track_ids,
        )

        all_boxes.extend(
            box_rows
        )
        all_frames.append(
            frame_row
        )

        if (
            idx == 1
            or idx % 100 == 0
            or idx == len(images)
        ):
            print(
                f"    {component} {sequence}: {idx}/{len(images)} frames"
            )

    boxes_df = pd.DataFrame(
        all_boxes,
        columns=BOX_COLUMNS,
    )

    frames_df = pd.DataFrame(
        all_frames,
        columns=FRAME_COLUMNS,
    )

    boxes_df.to_csv(
        paths["boxes"],
        index=False,
    )

    frames_df.to_csv(
        paths["frames"],
        index=False,
    )

    elapsed = float(
        time.time()
        -
        start
    )

    seq_manifest = {
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "component": component,
        "sequence": sequence,
        "frames_processed": int(
            len(images)
        ),
        "box_rows": int(
            len(boxes_df)
        ),
        "assigned_track_rows": int(
            (
                boxes_df["track_id"]
                >=
                0
            ).sum()
        )
        if len(
            boxes_df
        )
        else 0,
        "elapsed_seconds": elapsed,
        "frozen_hashes": frozen_hashes,
        "outputs": {
            "boxes": str(
                paths["boxes"]
            ),
            "frames": str(
                paths["frames"]
            ),
        },
    }

    paths["manifest"].write_text(
        json.dumps(
            seq_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


def combine_component(
    component,
    sequences,
):
    box_frames = []
    frame_frames = []

    for seq in sequences:
        paths = component_paths(
            component,
            seq,
        )

        require(
            paths["boxes"]
        )
        require(
            paths["frames"]
        )

        box_frames.append(
            pd.read_csv(
                paths["boxes"],
                low_memory=False,
            )
        )

        frame_frames.append(
            pd.read_csv(
                paths["frames"],
                low_memory=False,
            )
        )

    boxes = pd.concat(
        box_frames,
        ignore_index=True,
        sort=False,
    )

    frames = pd.concat(
        frame_frames,
        ignore_index=True,
        sort=False,
    )

    boxes_out = OUT / f"R13D_{component.upper()}_ALL_BOXES.csv"
    frames_out = OUT / f"R13D_{component.upper()}_ALL_FRAMES.csv"

    boxes.to_csv(
        boxes_out,
        index=False,
    )

    frames.to_csv(
        frames_out,
        index=False,
    )

    summary = {
        "component": component,
        "frames": int(
            len(
                frames
            )
        ),
        "boxes": int(
            len(
                boxes
            )
        ),
        "assigned_track_rows": int(
            (
                boxes["track_id"]
                >=
                0
            ).sum()
        )
        if len(
            boxes
        )
        else 0,
        "unique_track_ids_within_sequence_sum": int(
            boxes.loc[
                boxes["track_id"]
                >=
                0
            ]
            .groupby(
                "sequence"
            )[
                "track_id"
            ]
            .nunique()
            .sum()
        )
        if len(
            boxes
        )
        else 0,
        "boxes_path": str(
            boxes_out
        ),
        "frames_path": str(
            frames_out
        ),
        "boxes_sha256": sha256_file(
            boxes_out
        ),
        "frames_sha256": sha256_file(
            frames_out
        ),
    }

    return summary


def main():
    print(
        "\nR13-D — FROZEN DETECTOR + TRACKER EXECUTION"
    )
    print(
        "ACTUAL INFERENCE STEP. NO training and NO TestReal tuning. "
        "Runs the frozen detector on the six locked TestReal sequences, then runs "
        "the frozen ByteTrack primary tracker and BoT-SORT sensitivity tracker. "
        "Raw per-box and per-frame outputs are saved before any forecasting or warning analysis. "
        "The run is resumable: completed sequence/components are hash-checked and skipped.\n"
    )

    set_reproducibility()

    manifest, protocol = load_and_verify_protocol()

    detector_ckpt = Path(
        protocol["detector"]["checkpoint"]
    )

    settings = protocol["detector"]["settings"]

    byte_yaml = Path(
        protocol["tracking"]["primary_tracker_yaml"]
    )

    bot_yaml = Path(
        protocol["tracking"]["sensitivity_tracker_yaml"]
    )

    sequences = list(
        protocol["data"]["sequences"]
    )

    frame_root = Path(
        protocol["data"]["frame_root"]
    )

    require(
        frame_root
    )

    device, device_name = device_lock()

    frozen_hashes = {
        "protocol_sha256": manifest["protocol_sha256"],
        "detector_checkpoint_sha256": protocol["detector"]["checkpoint_sha256"],
        "bytetrack_yaml_sha256": protocol["tracking"]["primary_tracker_yaml_sha256"],
        "botsort_yaml_sha256": protocol["tracking"]["sensitivity_tracker_yaml_sha256"],
    }

    print(
        "DEVICE:",
        device_name,
    )
    print(
        "DETECTOR:",
        detector_ckpt,
    )
    print(
        "PRIMARY TRACKER:",
        byte_yaml,
    )
    print(
        "SENSITIVITY TRACKER:",
        bot_yaml,
    )

    for seq in sequences:
        seq_dir = frame_root / seq
        require(
            seq_dir
        )

        print(
            f"\nSEQUENCE {seq}"
        )

        run_sequence_component(
            "detections",
            seq,
            seq_dir,
            detector_ckpt,
            None,
            settings,
            device,
            frozen_hashes,
        )

        run_sequence_component(
            "bytetrack",
            seq,
            seq_dir,
            detector_ckpt,
            byte_yaml,
            settings,
            device,
            frozen_hashes,
        )

        run_sequence_component(
            "botsort",
            seq,
            seq_dir,
            detector_ckpt,
            bot_yaml,
            settings,
            device,
            frozen_hashes,
        )

    summaries = []

    for component in [
        "detections",
        "bytetrack",
        "botsort",
    ]:
        summaries.append(
            combine_component(
                component,
                sequences,
            )
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = OUT / "R13D_EXECUTION_SUMMARY.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    status = (
        "R13D_FROZEN_DETECTOR_BYTETRACK_BOTSORT_EXECUTION_COMPLETE_RAW_OUTPUTS_LOCKED_READY_FOR_R13E_TRACKING_METRICS"
    )

    final_manifest = {
        "step": "R13D_FROZEN_DETECTOR_TRACKER_EXECUTION",
        "created_utc": utc_now(),
        "status": status,
        "training": False,
        "testreal_tuning": False,
        "detector_inference": True,
        "tracking_inference": True,
        "forecast_inference": False,
        "warning_analysis": False,
        "original_files_modified": False,
        "device": str(
            device
        ),
        "device_name": device_name,
        "frozen_hashes": frozen_hashes,
        "sequences": sequences,
        "summary": summaries,
        "outputs": {
            "execution_summary": str(
                summary_path
            ),
        },
    }

    manifest_path = OUT / "R13D_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            final_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nEXECUTION SUMMARY:"
    )
    print(
        summary_df.to_string(
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
        "\nR13-D COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
