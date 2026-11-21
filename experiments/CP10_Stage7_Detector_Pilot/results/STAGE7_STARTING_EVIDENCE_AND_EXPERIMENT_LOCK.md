# Stage 7 — detector domain-transfer bottleneck and frozen sensitivity pilot

**Status (2026-11-21):** R13/R17 original stored evidence re-audited; new Stage 7 *development-only* experiment designed and software-tested. No new real-detector inference has yet been performed. Do not present code smoke tests as experimental evidence.

## Baseline evidence from prior experiments

Frozen R13: six BIRDSAI TestReal sequences / 9,085 frames / 48,055 observed elephant GT instances. Single-class elephant YOLOv8 checkpoint (sha256 `1d2eb19d22da78e82a92fcfb3a44ab87b78c5690d7d9caef6e0f55fba26d1055`), `imgsz=640`, `conf=0.10`, NMS IoU=0.70; no TestReal tuning.

At IoU >=0.50: detector: 4,288 predictions, 2,431 TP, 1,857 FP, 45,624 FN; precision 56.69%, recall 5.06%. **Count-only recall upper bound** at current number of detections: 4,288 / 48,055 = 8.92%; this is **not** an upper bound under a *new* detector inference configuration, which may produce more boxes.

ByteTrack ID-assigned observation recall 2.59%, HOTA 2.97%, IDF1 2.46%; BoT-SORT assigned recall 2.80%, HOTA 3.37%, IDF1 2.92%. Only two valid 2-second tracker-derived forecasting windows and zero 5-second windows per tracker. Historical 2s/5s forecast accuracy remains based on curated GT-derived trajectories, not validated end-to-end detections.

Four out of six evaluated TestReal sequences yielded zero detector true-positive boxes at IoU >=0.50 (55, 58, 349, 357); sequence 349 yielded zero detection boxes. Sequence 353 accounted for most matched instances; severe camera/scene/domain dependence is plausible but no causal explanation established.

R17 on the separate 453-image external COCO evaluation: elephant-only comparator recall=62.05%, F1=0.6438; selective four-class model recall=12.86%, F1=0.2036. The lower no-elephant negative-image false alarm rate of the four-class detector does not justify replacing it. **R17 COCO numbers are not BIRDSAI TestReal numbers.**

## Stage 7 experimental question

Can the **same frozen elephant-only detector** increase TrainReal detection recall (especially for small elephant bounding boxes) using lower confidence or higher input resolution *without intolerable FP inflation*, and does the dev-selected setting subsequently yield better detector/tracker/forecast-window coverage on previously examined TestReal data?

## Development-only first experiment (this package)

1. Locate original R13 elephant detector checkpoint and verify the exact sha256. Use only `train_real` rows in original cleaned BIRDSAI ground-truth CSV; exclude interpolated rows and non-elephant species, if encoded. Look up matching **TrainReal** images. Never use TestReal images in development selection. The preflight fails closed if these assets are missing.
2. Select up to six TrainReal sequences and 24 evenly spaced *annotated* frames per sequence, using seed `20261121`, with no exposure to model outcomes. All images chosen have annotated elephant GT; this sample does **not** support negative-image false-alarm rates.
3. Evaluate frozen checkpoint under predeclared configurations: `baseline_640_conf010` (original R13), `highres_960_conf010`, `lowconf_640_conf003`, `highres_960_conf003`. NMS IoU 0.70, no augmentation. Optional tiled config is in the code but NOT in default comparison. It requires a separate explicitly labeled pilot protocol.
4. At IoU >=0.50, perform unique frame-level GT/box matching, preserving frame and sequence identities. Report recall, precision, F1, FP **per annotated positive image**, small-GT recall (GT box occupies <0.5% image area), source-level recall, and wall-clock latency. No mAP or tracking metric is claimed from this pilot.
5. **Predeclared selection gate:** candidate must increase recall by at least 0.03 absolute, precision >= max(0.10, baseline precision - 0.10), FP per annotated image <= baseline + 0.5. Among eligible candidates choose maximal recall, then lower FP, then lower latency. If none passes, retain baseline. This is a *pilot decision rule*, not a formal hypothesis test.
6. Freeze selected settings *before* further retrospective TestReal stress testing. Record per-sequence failures, HOTA/IDF1/coverage, FPS, and 2s/5s forecastable windows if/when the full experiment is run. Do not tune again on TestReal.

**Limits:** TestReal results would be a retrospective re-evaluation of a dataset already inspected in R13, not untouched independent confirmation. Original detector checkpoint may have unknown pretraining exposure; no claim of fully independent development data without provenance checks. Changes from new inference settings are not yet empirically established.

## Review decision

- The scientifically defensible next step is a frozen detector **development pilot**, not relabeling four-class COCO results or reporting detector-only illustrations as operational end-to-end accuracy.
- If TrainReal image sequences are missing, stop and collect/locate them. Do not silently tune on R13 TestReal. Dataset-driven fine-tuning would be a separate, clearly logged intervention requiring an independent evaluation split.
- Reusable metric unit: observed GT elephant instances with one-to-one IoU >=0.50 matching; track-level HOTA/IDF1 and forecastable windows belong to the subsequent locked test stage.
