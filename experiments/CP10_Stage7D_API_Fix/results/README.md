# Stage 7D: Ultralytics 8.4.216 BYTETracker API fix

The uploaded log shows that the original R13 frozen ByteTrack YAML was verified successfully, but the Stage 7D script stopped before tracking due to `BYTETracker.__init__() got an unexpected keyword argument 'frame_rate'`.

The exact call is updated from `BYTETracker(args=SimpleNamespace(**cfg),frame_rate=30)` to `BYTETracker(args=SimpleNamespace(**cfg))`. Current Ultralytics 8.4.216 exposes a one-argument constructor, with the existing YAML `track_buffer: 30` still in frame-count units. No threshold, model checkpoint, YOLO prediction, evaluation metric, or original research data is modified.

Download this ZIP to `C:\Users\HP\Downloads`, then in WSL:

```bash
mkdir -p ~/cp10_stage7d_api_fix
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage7D_API_Fix.zip" ~/cp10_stage7d_api_fix
bash ~/cp10_stage7d_api_fix/fix_stage7d_and_rerun.sh
```

The repair is idempotent; it backs up the old script, checks the runtime signature, reruns tracking and creates an updated `CP10_Stage7D_Tracker_Review.zip` in Windows Downloads. Upload that ZIP whether the run succeeds or fails. The previous failed traceback is saved as `~/CP10_Stage7D_Previous_API_Failure.txt` rather than included in a successful new results archive.

No claim about HOTA/IDF1 or downstream forecast-window yield should be made before the new results are audited.
