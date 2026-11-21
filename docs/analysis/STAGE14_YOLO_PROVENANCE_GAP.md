# Original YOLO elephant detector — unresolved training provenance

**Status:** NOT VERIFIED by the uploaded Stage 13 filename searches. This file is a research evidence-gap record, not a negative assertion that the original records do not exist.

## Evidence currently found

The targeted inventory returned 43 candidate entries: 24 CSV, 14 Python, 4 JSON, and 1 PNG. None is identified in the inventory as an original YOLO training checkpoint, and none was verified to be the training record associated with the original reported performance. Several matches came from R13 detector *evaluation*, compatibility and pipeline-audit artifacts, which are scientifically different from the original training run. Two candidate records came from Downloads; filename matching can produce unrelated false positives.

## What is required to claim full provenance

1. **Training datasets:** 3,280-image dataset version, split counts, image IDs/hashes, label schema and exact data source/version.
2. **Training command/config:** `data.yaml`, model architecture, starting weights, `imgsz`, epochs, batch size, seed, augmentation and threshold settings, and the exact Ultralytics/PyTorch versions.
3. **Original training artifacts:** `results.csv`, command/run log, validation curves and confusion matrix, plus any relevant experiment run ID.
4. **Checkpoint identity:** SHA-256 (and if applicable path/basename) of the final detector checkpoint actually used to compute the original figures.
5. **Independent evaluation linkage:** exact test images, IoU matching rule, confidence threshold, class mapping and script version used to produce the 91.8% precision, 88.4% recall, 94.0% mAP50 (and 69.8% mAP50:95 where reported).

These metrics refer to the original elephant detection test cohort; they must not be attributed to BIRDSAI TestReal or Stage 7C. Stage 7B TrainReal adaptation produces different training and validation records and is not a substitute for primary baseline provenance.

## Present wording for a journal manuscript

'Historical detector-only performance on the original elephant image cohort is reported from the project records; complete original training configuration, data-split provenance and checkpoint-to-evaluation linkage have not yet been independently verified. Downstream BIRDSAI TestReal detector and tracker measurements are reported separately using their own frozen protocols.'

This limitation should be updated only after verifying primary original records; do not infer missing training configuration from a later experiment's defaults.
