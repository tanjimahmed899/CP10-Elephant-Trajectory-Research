# Stage 21 — Frozen detector on recovered 329-image candidate test split

**Scientific classification: NEW frozen-checkpoint candidate-test evaluation with agreement at historical reporting precision. This is NOT confirmation of the original training dataset version or a reconstruction of the historical evaluation command.**

## Identity, cohort, and controls

- Source: recovered `elephant_final_clean/weights/best.pt` referred to by the Stage 7B adaptation training configuration, **not** the Stage 7B adapted checkpoint.
- Frozen source checkpoint SHA-256: `1d2eb19d22da78e82a92fcfb3a44ab87b78c5690d7d9caef6e0f55fba26d1055`.
- Recovered candidate dataset YAML SHA-256: `dbc64441fd4112d6e8690c2fbaebaaeb0da86816d2309c2106d6240f9668cf48`.
- Test image-plus-label content snapshot SHA-256: `0d395d4edc956e54c5bc20d566b393f8999d8337527ae2d008234d0c183e1b08`.
- Candidate dataset: 3,280 images and 4,477 one-class elephant boxes in total; candidate test split: **329 images, 432 elephant boxes, 56 negative images**.
- No cross-split byte-identical image duplicates found in the prior 3,280-image SHA-256 scan; this does not exclude near-duplicates or source/video leakage.
- Model inference on frozen trusted local checkpoint; no retraining, no source-file modifications. Validation cache isolated from original source.

## Test results

| Metric | Historical rounded value (context only) | New frozen candidate test | Difference |
|---|---:|---:|---:|
| Precision | 91.8% | 91.8217% | +0.0217 pp |
| Recall | 88.4% | 88.3651% | -0.0349 pp |
| mAP@50 | 94.0% | 94.0420% | +0.0420 pp |
| mAP@50:95 | 69.8% | 69.8322% | +0.0322 pp |

All four metrics agree with historical values **when rounded to one decimal percent**. This is strong numerical corroboration, **not** proof that the original historical command and dataset versions were identical. The historical claims are not independently established as a reconstruction of the original validation protocol.

## Frozen evaluation protocol

Ultralytics `model.val` on candidate `test` split: image size `640`, batch `8`, device `cpu`, workers `0`, prediction confidence floor `0.001`, NMS IoU `0.7`, maximum detections `300`, augmentation `false`, `rect=true`. NMS IoU is **not** the mAP@50 matching threshold. The reported Ultralytics P and R should not be interpreted automatically as precision/recall at a fixed inference threshold of 0.001.

Environment: Python `3.12.3`, Ultralytics `8.4.216`, torch `2.15.0+cpu`, torchvision `0.29.2+cpu`, NumPy `2.5.4`, OpenCV `5.1.0.95`.

## Evidence integrity and remaining gaps

The Stage 21 sanitized metrics record is `experiments/CP10_Stage21_Frozen_Detector_Test_Results/results/STAGE21_FROZEN_CANDIDATE_TEST_METRICS.json`. Its input evaluation-summary SHA-256 was `0951358d967f83ffa79bedd7dc3b9fbea10aa3050467c3b7489554a4fe1a2020`. All checkpoint, candidate YAML, and private test image/label fingerprints were reconciled against Stage 16/19/20 records before release preparation. Ultralytics printed 329 images and 432 instances in the user-supplied terminal transcript; automatic structured re-reading of validator cohort counts was not implemented in Stage 20.

**Still unresolved:** the original run-declared `data.yaml` is not available, the historical evaluation command and test split provenance are not independently recovered, and visual near-duplicate or correlated source-video leakage has not been excluded. Consequently, do not describe this as proof of the original study's held-out testing protocol or of downstream BIRDSAI TestReal performance. Do not extrapolate these numbers to field video or full deployed pipeline performance.

This document and the sanitized numerical record contain no raw images, original `.pt` weights, private E-drive paths, row-level wildlife coordinates or source footage. Public release remains subject to author/license review.
