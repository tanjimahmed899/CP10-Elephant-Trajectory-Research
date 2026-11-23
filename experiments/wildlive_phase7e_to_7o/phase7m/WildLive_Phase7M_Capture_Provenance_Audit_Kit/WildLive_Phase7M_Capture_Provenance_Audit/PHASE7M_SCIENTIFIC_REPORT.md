# CP 1.0 — Phase 7M: Capture-event provenance and independent validation eligibility

**Audit date:** 22 November 2026  
**Scope:** Four locally held full elephant-annotation archives; four sparse original JPEG sample ZIPs; two saved locked-evaluation ZIPs; original Phase 7K protocol and its released code hashes; official WildLive README and landing pages.  
**Decision:** **Provenance evidence assembled, official capture-event independence NOT established.** This is not a new predictive-model experiment.

## Executive finding

The four inspected elephant-labeled WildLive folders comprise 3,008 annotated frames, 7,230 elephant boxes, 14 annotated track IDs (track IDs are local to each recording), and 12,948 overlapping image-plane forecasting windows (20-observed + 20/50-future). **These are not 12,948 independent experiments**, nor 14 independent biological animals across flights, nor four certified capture events. The official dataset describes 22 UAV-acquired video sequences, not 22 independent flights.

`DJI_0204_video1` and the method-development recording `DJI_0204_video2` exhibit strong same-landscape evidence in RGB background matching. Hence `DJI_0204_video1` is a **related-scene exploratory** recording, not an independent external confirmation. The held-out-labelled `DJI_0161_video1` has already been evaluated and its scores inspected; its source flight/session relationship is unverified. **No author-confirmed independent capture event has yet been established from supplied evidence.**

## Evidence inventory — full annotation re-audit

| Folder | Frames | Elephant boxes | Track IDs | 20→20 windows | 20→50 windows | Role |
|---|---:|---:|---:|---:|---:|---|
| DJI_0207 | 868 | 2,604 | 3 | 2,487 | 2,397 | Used in development |
| DJI_0204_video2 | 357 | 1,428 | 4 | 1,272 | 1,152 | Used in development |
| DJI_0204_video1 | 283 | 1,698 | 6 | 1,464 | 1,284 | Related scene; exploratory evaluation |
| DJI_0161_video1 | 1,500 | 1,500 | 1 | 1,461 | 1,431 | Evaluated, capture independence unresolved |
| **TOTAL (descriptive only)** | **3,008** | **7,230** | **14 local track IDs** | **6,684** | **6,264** | **0 independently confirmed flights** |

All four archives had valid ZIP CRC; frame indices were continuous within each archive; elephant bounding-box geometry check reported 0 invalid boxes. Source SHA-256, missing scores, frame metadata field inventories and per-archive counts are recorded in `evidence/generated/annotation_inventory.csv`. Missing scores are not equivalent to failed geometry checks.

## Direct search for capture metadata

For all 3,008 JSON files, the observed top-level keys were `description`, `flags`, `imageData`, `imageHeight`, `imagePath`, `imageWidth`, `shapes` and `version`. No authoritative flight ID, capture timestamp, per-frame timestamp, recording session mapping, camera pose, or native capture FPS appeared among the predefined metadata keys examined. Each `frame_N.json` is **an indexed annotation**, not a timestamp.

In all **25 sampled original 3840×2160 JPEGs** inspected from the provided ZIPs, standard EXIF `DateTime`, `DateTimeOriginal`, `DateTimeDigitized` and `GPSInfo` fields were absent. This absence in selected JPEGs does not rule out metadata in unprovided original MP4s, external flight logs, XMP sidecars or author-held files. In particular, sample images' filesystem modification date cannot establish capture date.

**Timing:** 20 and 50 are frame horizons. Do not call them 2 s and 5 s for this dataset without verified capture FPS and temporal subsampling metadata. WildLive project processing speed (e.g., 7.53 fps tracking throughput on 4K) is **not** the original video's capture FPS.

## Image-content cross-sequence comparison

Six preselected RGB pairs were checked using locally available original JPEGs, grayscale resize to 1280×720, OpenCV ORB (`nfeatures=5000`, `fastThreshold=20`), Hamming KNN Lowe ratio 0.75, RANSAC homography 4-pixel threshold. This is scene evidence, not animal identity or temporal alignment proof.

| Compared samples | Filtered feature matches | Homography inliers | Conclusion |
|---|---:|---:|---|
| DJI_0204_video1 frame 200 vs DJI_0204_video2 frame 200 | 1,351 | **1,320** | **Strong common-scene/background evidence** |
| DJI_0204_video1 frame 200 vs DJI_0161_video1 frame 200 | 13 | 4 | Inconclusive |
| DJI_0204_video2 frame 200 vs DJI_0161_video1 frame 200 | 7 | 4 | Inconclusive |
| DJI_0207 frame 389 vs DJI_0204_video1 frame 200 | 13 | 5 | Inconclusive |
| DJI_0207 frame 389 vs DJI_0204_video2 frame 200 | 16 | 5 | Inconclusive |
| DJI_0207 frame 389 vs DJI_0161_video1 frame 200 | 9 | 4 | Inconclusive |

One positive common-scene pair is strong evidence that clips share a physical background; it does **not** establish a single flight, contiguous frames, the same animals or exact video lineage. Conversely, no strong match in one sparse pair is **not** evidence of different flights; appearance, zoom, viewing angle and capture time can prevent matches. Match counts can vary by OpenCV build and codec.

For conservative group accounting, keep both `DJI_0204` clips under one **provisional related-scene family**. The other two folders retain *unresolved* group membership. Three provisional labels in a spreadsheet are **not three verified independent groups**.

## Lock integrity and existing forecasting outputs

The original Phase 7K protocol is retained unchanged (SHA-256 `408022abb8923eb61afdf1eb665db8ba7bb02ce4258b21ae4a3fd89dab4abb5c`); all **8/8** released protocol/code/model-gate file hashes matched the original local Phase 7K release. The two saved Phase 7K/7L evaluation ZIP manifests matched their corresponding annotation SHA-256 and recorded all **6/6 original Transformer checkpoint hashes** plus **2/2 gate model hashes**. This is **manifest/output provenance**, not independent full neural forward inference (checkpoint binaries are not embedded in these output archives).

Past forecasting outcomes remain:

- `DJI_0161_video1`: Adaptive versus locked CV-OLS: 13.49% lower mean ADE at the 20-frame horizon, 23.86% lower at the 50-frame horizon, but not better than the strongest individual expert; capture independence unresolved.
- `DJI_0204_video1`: Adaptive versus locked CV-OLS: about 0.61% higher ADE at 20 frames and about 0.053% higher at 50 frames; relates spatially to development `DJI_0204_video2`.
- `DJI_0207` and `DJI_0204_video2`: development, **cannot** be re-labelled as untouched external evaluation.

These are previously analysed outcomes, not newly generated experiments. Do not select, re-tune, or justify the gate based on this audit.

## Official documentation findings

The University of Bristol dataset landing page reports 22 UAV-acquired sequences and 19,139 frames. The official `Readme.txt` explicitly says the release **does not provide predefined train/test splits**. The checked landing page, README and project page do not give sufficient per-sequence flight IDs, original-video parent mapping, authoritative timestamp and capture cadence to certify independence; the authors may still have additional information.

The repository's 22 sequences do **not** demonstrate 22 different flights. `evidence/complete_22_folder_status_register.csv` consolidates the full four-sequence audits and two batches of sparse 6-candidate screens. Zebra/giraffe labels in sampled frames do not prove a sequence contains zero elephants outside those samples. Confirm species by full annotation census before exclusion if further elephant recordings are needed.

**Public sources:**
- https://data.bristol.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69
- https://data.bris.ac.uk/datasets/ewnuoroebuae20vei0cu6zu69/Readme.txt
- https://dat-nguyenvn.github.io/WildLive/
- https://research-information.bris.ac.uk/en/datasets/wildlive/

## Actionable next phase: obtain author-backed flight lineage

1. Send `templates/DATASET_AUTHOR_METADATA_REQUEST.md` to the WildLive dataset team via a verified contact route. **No message has been sent**, and no email address was guessed.
2. Obtain an anonymized per-sequence **parent video ID, flight/session ID, source clip offsets, verified FPS/annotation sampling cadence, approximate recording dates, relation of the 0204 clips**, and confirmation whether 0161/0207 share an encounter. Precise elephant GPS coordinates are neither requested nor necessary.
3. Archive the author response; enter facts in `templates/author_capture_event_reply_template.csv`. Use `templates/PROVENANCE_REPLY_REVIEW_CHECKLIST.md` to version the group decisions. Do not retroactively call an already-inspected result untouched/pre-registered.
4. For additional validation, **prefer genuinely new author-verified elephant capture events**, with models/checkpoints/primary comparator unchanged; score only after complete species/visual QA and group assignment. Use capture-event-level descriptive averages, no pseudoreplication-based significance claims.
5. If reliable flight metadata cannot be obtained, state **capture independence unverified** explicitly in the paper. The current adaptive method is not demonstrated to consistently outperform the strongest baselines.

## Reproducibility and integrity

- Source: `src/phase7m.py` includes `audit`, `verify-lock`, and `verify-release` commands.
- `src/test_phase7m.py`: **14 offline unit/regression tests passed** (includes negative controls, gap-aware windows, tamper rejection and withheld group status).
- Original checkpoint and gate files are not modified or included in this package.
- The ZIP package does not include native source annotation ZIPs, original JPEGs or precise geolocations.
- Scripts parse locally supplied archives, record content hashes, and reproduce the generated CSV/JSON outputs. A sparse visual matching result is not a certificate of independent capture events.

**Final Phase 7M status: provenance registry and reproducible local audit PASS; author-backed flight lineage PENDING; independent capture-event validation NOT ESTABLISHED.**
