# Stage 7B — TrainReal-only elephant detector domain-adaptation pilot

## Why a new stage?
All Stage 7A frozen-configuration candidates failed predeclared precision/recall/FP acceptance gates. The 640/0.10 R13 detector stays the comparator; **do not label high-resolution/low-confidence a successful remedy**. This separately declared experiment tests supervised *thermal-domain adaptation* and is **not** a continuation of Stage 7A's fixed-configuration sweep.

## Data isolation
Only BIRDSAI `train_real` rows of `Main Result/trajectory_dataset/cleaned_track_points.csv` and corresponding images are read. Elephant class `species=1`, non-interpolated detections only. No test_real images, labels, or checkpoint selection are used. Original WildBox-trained R13 `best.pt` must match SHA-256 `1d2eb19d22da78e82a92fcfb3a44ab87b78c5690d7d9caef6e0f55fba26d1055`.

Train sequences (3): `0000000367_0000000000`, `0000000351_0000000000`, `0000000056_0000000000`; held-out development validation sequences (2): `0000000010_0000000000`, `0000000054_0000000000`. Distinct sequence does not imply unrelated recording session or unseen ecological environment.

Sample limit, selected by evenly spaced frame ID **before any model outcomes**: up to 1,200 images per training sequence, 150 per validation sequence. Positive annotated frames only (no negative-frame specificity). Near-adjacent samples are correlated. The original checkpoint may have unknown prior data exposure. TestReal was already explored in R13, and any subsequent TestReal analysis is retrospective, not a new independent study.

## Training protocol (predeclared)
YOLO original R13 checkpoint, 8 epochs, 640 pixels, 8 batch, seed 20261121, AdamW lr0 1e-4, workers 0, CPU, no mosaic/mixup/flips. Select `best.pt` using the Ultralytics default validation fitness. Compare original/fine-tuned mAP on the same two TrainReal validation sequences using conf=0.001, IoU 0.7. This is validation data used for checkpoint selection, **not an independent test**. No tracker, HOTA/IDF1 or end-to-end gain can be claimed yet.

## Run in WSL
1. Download `CP10_Stage7B_TrainReal_Adaptation.zip` to `C:\Users\HP\Downloads`.
2. Run:

```bash
mkdir -p ~/cp10_stage7b_tools
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage7B_TrainReal_Adaptation.zip" ~/cp10_stage7b_tools
bash ~/cp10_stage7b_tools/CP10_Stage7B_TrainReal_Adaptation/run_stage7b_wsl.sh
```

3. Upload `C:\Users\HP\Downloads\CP10_Stage7B_TrainReal_Review.zip` here. This review includes the new `best.pt` checkpoint and original checkpoint hash, not raw frames.

If an execution step fails, send the terminal error along with `~/CP10_Stage7B_TrainReal_Adaptation/CP10_Stage7B_Prepare.log` or `CP10_Stage7B_TrainReal_Adaptation/CP10_Stage7B_Train.log` as available. If the output path is already prepared, the script **refuses to overwrite** it. Use a new output path only after recording the original experiment identity.

## Next stage only after audit
After checking trained model and held-out TrainReal validation, freeze a **single** model and operating threshold. Then evaluate on previously inspected BIRDSAI TestReal once, with detector IoU50 P/R/F1, per-sequence results, tracker IDF1/HOTA, 2s/5s usable windows, negative-image FP and CPU latency. Never use TestReal metrics to select among additional checkpoints.
