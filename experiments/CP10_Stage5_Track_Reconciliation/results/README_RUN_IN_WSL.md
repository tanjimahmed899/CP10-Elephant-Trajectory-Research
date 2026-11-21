# CP10 Stage 5: reconcile generic tracked objects with KITTI elephant ground truth

This script inspects the **three candidate WildBox ZIPs in place**. It resolves `tracking_summary.json` generic `object` tracks using per-frame, one-to-one IoU >= 0.5 matches to `kitti_labels` elephants. It infers the tracker's box image scale from candidate resolutions and reports track runs / estimated 2s (40 frames) and 5s (70 frames) windows.

It **does not train or evaluate a forecaster**, claim a new held-out recording group, label every generic object as elephant, or change source files. Existing source video 0006 was already inspected. The two 0004 videos share already-inspected `elep2` / `elep3` recording groups, so treat future analysis as retrospective within-group evidence.

## Windows WSL steps

Save `CP10_Stage5_Track_Reconciliation.zip` in `C:\Users\HP\Downloads`. Then:

```bash
mkdir -p ~/cp10_stage5_reconcile
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage5_Track_Reconciliation.zip" ~/cp10_stage5_reconcile
python3 ~/cp10_stage5_reconcile/CP10_Stage5_Track_Reconciliation/stage5_reconcile_semantic_tracks_wsl.py \
 --root "/mnt/e/AIUB/Capstone Project/Journal Based Results" \
 --out ~/CP10_Stage5_Semantic_Track_Audit
```

Then copy `~/CP10_Stage5_Semantic_Track_Audit/CP10_Stage5_Semantic_Audit_Results.zip` to your Windows Downloads and upload it. The script scans prior research report text for mentions of candidate filenames, but such mentions are **not** proof that the videos were used as test data. The run is read-only. Python standard library only.

## Files produced
- `CP10_Stage5_Semantic_Segments.csv`: GT-matched object tracks and upper-bound observed/strict windows
- `CP10_Stage5_Semantic_Tracks.csv`: per-track evidence
- `CP10_Stage5_Prior_Usage_Mentions.csv`: source usage references for manual inspection
- `CP10_Stage5_Semantic_Audit.json`, `CP10_Stage5_Semantic_Decision.txt`

For rigor, track-frame association is based on the IoU of *tracking_summary* boxes (not model detections) with source KITTI annotations. The track-box spatial resolution may differ from source 1920x1080, so the code estimates the scale. It never changes the raw semantic class annotations.
