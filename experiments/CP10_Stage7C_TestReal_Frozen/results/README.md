# CP10 Stage 7C — Frozen BIRDSAI TestReal detector comparison

The Stage 7B model was fine-tuned on TrainReal only and chosen using TrainReal validation. **Do not modify model weights, confidence, image size, NMS, or test sequence list in Stage 7C.** TestReal was already analyzed in R13, so this test is retrospective, not an independent holdout.

## What this actually measures

It compares the original R13 YOLO and Stage 7B adapted YOLO on the same six TestReal sequences, using 640-pixel inference, confidence 0.10, NMS IoU 0.7, one-to-one detection matching at IoU >= 0.50, source GT species=elephant, interpolated rows excluded. It also audits original R13 baseline parity and computes metrics by sequence. Frames without annotated elephants are included. No tracker is executed by this script.

## How to run in WSL

Download `CP10_Stage7C_TestReal_Frozen.zip` to `C:\Users\HP\Downloads`, then:

```bash
mkdir -p ~/cp10_stage7c_tools
python3 -m zipfile -e '/mnt/c/Users/HP/Downloads/CP10_Stage7C_TestReal_Frozen.zip' ~/cp10_stage7c_tools
bash ~/cp10_stage7c_tools/CP10_Stage7C_TestReal_Frozen/run_stage7c_wsl.sh
```

The runner activates `~/cp10_detection_venv` if present, writes TestReal outputs outside original E-drive research files, and creates `~/CP10_Stage7C_TestReal_Review.zip`. It also copies that ZIP to `C:\Users\HP\Downloads`. If inference fails, the package contains the log and available preflight evidence. Upload `CP10_Stage7C_TestReal_Review.zip`.

## Checkpoint provenance

The adapted checkpoint `best_Stage7B_trainreal.pt` was copied byte-for-byte from your Stage 7B review archive. Both checkpoint SHA-256 digests are verified before inference. Original R13 detector is loaded from the unchanged E-drive path.

## Interpretation rules

- The 0% original and 3.78% adapted recall from Stage 7B are from positive-frame-only TrainReal **validation**, with confidence-swept Ultralytics validation. They cannot be equated directly with fixed-confidence TestReal.
- The original historical R13 baseline (2,431 TP from 48,055 GT at IoU 0.5) must be checked against the new environment's original-checkpoint replay. If results differ, report both; do not silently conflate.
- Precision and false boxes/frame are based on the selected elephant GT annotations; absence of labels for other species cannot be assumed equivalent to curated fully negative examples.
- Sequence coverage, matched instance recall, small elephant recall, and false boxes/frame must be considered together. No detector-only result establishes improved ByteTrack/HOTA/IDF1 or valid forecast windows.
- If Stage 7C improves TestReal detection materially, the next experiment is detector-to-tracker continuity using the same locked tracker YAML, then actual forecastable windows.
