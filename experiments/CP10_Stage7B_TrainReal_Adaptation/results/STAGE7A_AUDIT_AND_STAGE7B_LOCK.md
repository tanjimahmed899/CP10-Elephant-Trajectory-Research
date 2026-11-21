# Stage 7A checked results and Stage 7B next experiment

## 7A: verified uploaded real inference (not smoke-test output)

- The uploaded `CP10_Stage7_Dev_RealInference_Review.zip` includes 13 nonempty evidence files, four inference configurations, actual bounding boxes and timings, and model provenance. The separate diagnostics ZIP reports all five process steps with exit code zero.
- TrainReal development set: 120 annotated images, 521 elephant GT boxes across five sequences, 480 image/config inference calls, original R13 checkpoint SHA-256 `1d2eb19d22da78e82a92fcfb3a44ab87b78c5690d7d9caef6e0f55fba26d1055`.
- IoU ≥ 0.50 matching, (TP,FP,FN): baseline 640/conf .10 = (10,9,511); 960/.10 = (14,45,507); 640/.03 = (18,24,503); 960/.03 = (32,72,489).
- Recalls, respectively: **1.92%, 2.69%, 3.45%, 6.14%**. Precisions: **52.63%, 23.73%, 42.86%, 30.77%**. FP per annotated positive image: **0.075, 0.375, 0.200, 0.600**.
- Aggressive 960/.03: +4.223 percentage-point recall, but precision 0.3077 is below the preregistered floor (baseline precision minus .10 = 0.4263) and FP/image 0.600 exceeds the allowed 0.575. 640/.03 misses the +3 percentage-point recall gate. **No candidate meets all gates; retain 640/.10.**
- **No true positives in the following TrainReal sequences under any tested setting:** `0000000010_0000000000`, `0000000054_0000000000`, `0000000056_0000000000`. This does not prove no elephants exist; all these frames have GT elephants.
- Of 137 small GT elephants (<0.5% image area), 960/.03 matched just 3; the other settings matched zero.
- The above are **development-only detection metrics**; not TestReal, not mAP, not tracking IDF1/HOTA, not end-to-end forecasting, and not specificity on unannotated negative frames. The R13 TestReal baseline has separately already been reported (5.06% detector recall), so its repetition without a new model would add little information.

## Stage 7B: new, deliberately separate intervention

The next meaningful hypothesis is that **supervised fine-tuning on BIRDSAI TrainReal thermal-domain elephant annotations** can improve the original RGB-trained detector's cross-domain localization. Stage 7A demonstrates an inference-configuration limitation, **not proof that fine-tuning will succeed**.

A fixed, sequence-disjoint exploratory TrainReal pilot is packaged. Training: sequences `367`, `351`, `56`; validation: `10`, `54` (full exact IDs in code). Up to 1200 selected positive frames per train sequence and 150 per validation sequence. Train on the *same* original R13 checkpoint for 8 epochs, AdamW lr0=0.0001, imgsz=640, batch=8, seed 20261121, CPU, no mosaic/mixup/flips. Checkpoint choice is Ultralytics `best.pt` by validation fitness. Report both baseline and adapted detector performance on the two TrainReal **development validation** sequences, with no TestReal access.

**This stage is an exploratory pilot, not external confirmation.** TrainReal images in the original GT CSV may differ from full raw annotation sources, and there may be sequence/session dependence. Validation drives checkpoint selection. The original detector's prior training data may overlap unknown sources. The subsequent TestReal dataset has already been inspected in R13 and cannot be newly independent.

### After Stage 7B

Only if TrainReal validation indicates improvement should one freeze **one checkpoint** and run a one-time retrospective TestReal comparison with per-sequence IoU50, negative-frame false positives, IDF1/HOTA and usable tracker-derived 2s/5s forecasting windows. The predeclared operational gate should include recall, precision/FP, and tracker continuity. Do not tune model selection on TestReal.
