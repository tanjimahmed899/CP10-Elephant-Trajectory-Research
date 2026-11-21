# Stage 7C audit and rationale for Stage 7D

Input: `CP10_Stage7C_TestReal_Review.zip` SHA-256 6e3b5e691d814da5a636f828c2c7520e8b71d2513cea1891e31e80a08ddd1176.

- Original R13 checkpoint SHA-256: 1d2eb19d22da78e82a92fcfb3a44ab87b78c5690d7d9caef6e0f55fba26d1055.
- Stage 7B adapted checkpoint SHA-256: 71671b4175509e96bb6e4cd7e803b82c9c4b606061438a009d98d9f8b8e8d3d1.
- Both on six BIRDSAI TestReal sequences: 9,085 frames, 48,055 observed GT elephant instances.
- Locked detection settings: img 640, conf 0.1, IoU NMS 0.7; detection match IoU>=0.5.
- Original: TP=2,453.0, FP=1,835.0, FN=45,602.0, recall=5.105%, precision=57.206%, F1=9.373%, FP/frame=0.202.
- Adapted: TP=27,382.0, FP=52,684.0, FN=20,673.0, recall=56.981%, precision=34.199%, F1=42.744%, FP/frame=5.799.
- Adapted recall ratio: 11.16×. False box ratio: 28.71×. Both true-positive gain and increased unmatched detection burden must be reported.
- Small GT 7,331: original TP=0.0, adapted TP=1915.0 (recall=26.12%).
- Historical R13 baseline detector 2,431 TP; this replay 2,453 TP, 4,288 predicted boxes unchanged. Exact parity **FALSE**; do not silently collapse those baselines.
- Sequence 0055: 0 matched TP even after adaptation. Sequence 0353: adapted recall 84.18%; sequence 0352: adapted false boxes/frame 7.48.

## Next experiment

Use saved detections with byte-for-byte frozen `bytetrack_FROZEN.yaml`, offline ByteTrack stateful sequence replay, and original R13E HOTA/IDF1 evaluator plus R13F_B track-window construction. Avoid TestReal tuning. Nonparity is anticipated because R13 used Ultralytics 8.3.75 and Stage7C used 8.4.216. A detector-only recall increase is not evidence that IDF1/HOTA or true forecastable track windows increase.

Important: `Stage7C_AnyMatchedElephant_Frame_Diagnostics.csv` reports only **any-target** matched frames, not identity-specific temporal continuity; no tracking metric can be inferred from it.
