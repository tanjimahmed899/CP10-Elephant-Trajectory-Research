# CP10 Stage 7E — GT-linked tracking-window audit

## Why this audit is necessary

Stage 7D completed on the same six BIRDSAI TestReal sequences (9,085 frames, 48,055 observed GT elephant boxes) with the *frozen* ByteTrack YAML and Ultralytics 8.4.216. The adapted detector increased aggregate HOTA **0.025220 → 0.204086**, IDF1 **0.021390 → 0.193310** and MOTA **0.011945 → 0.147643**. However, it also produced **1,307 identity switches**, **3,597 fragmentation events**, and **1,582 tracker IDs** for only **45 GT elephant IDs**. These are not reliable end-to-end trajectories yet.

Stage7D's **371 potential 2s-to-2s** and **191 potential 2s-to-5s** candidate windows are based on tracker segment lengths, short-gap interpolation and 30→10 Hz decimation. They were *not* checked for GT identity purity or future GT availability. Of the 191 long-horizon windows, **180 came from sequence `0000000353_0000000000`**. These numbers cannot be equated with successful Transformer inference windows or actual elephants.

Stage 7E checks the original unaltered GT file and Stage7D's saved boxes. It performs one-to-one 0.50-IoU framewise matching, then repeats the original candidate window enumeration and evaluates two explicitly declared **ORACLE diagnostic eligibility gates**, without any training/inference or TestReal tuning:

* **Strict**: all 20 sampled 10 Hz observed-history tracking boxes match the same GT elephant at IoU≥0.50, with all 20/50 future GT frames available.
* **Sensitivity check**: at least 16/20 observed-history boxes match the same GT, with no observed match to another GT elephant and all target future GT positions available. This 80% gate is a diagnostic sensitivity analysis, not a performance-optimized threshold.

Ground truth is used **only for evaluation and linkage**, never as an online track label or to select detector/ByteTrack configurations. These thresholds evaluate sample quality retrospectively. Even passing the strict gate does not imply Transformer inference has been performed or that a detector could autonomously know the correct track identity.

## WSL usage

1. Download `CP10_Stage7E_GT_Linkage_Audit.zip` to `C:\Users\HP\Downloads`.
2. Ensure the completed `CP10_Stage7D_Tracker_Review.zip` remains in `C:\Users\HP\Downloads`. If not, copy it there first. Avoid stale earlier failure-only review ZIPs.
3. In WSL:

```bash
mkdir -p ~/cp10_stage7e_tools
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage7E_GT_Linkage_Audit.zip" ~/cp10_stage7e_tools
bash ~/cp10_stage7e_tools/CP10_Stage7E_GT_Linkage_Audit/run_stage7e_wsl.sh
```

Results are written to `~/CP10_Stage7E_GT_Linkage_Results/`, and `~/CP10_Stage7E_GT_Linkage_Review.zip` is copied to Windows Downloads. Upload the latter. The original E: drive research tree and all frozen configurations remain unchanged. The script requires only the existing `cp10_detection_venv` with numpy/pandas/scipy; it does **not** import Ultralytics or Torch.

**Failure gate:** GT must contain exactly 48,055 non-interpolated TestReal elephant boxes in the six original sequences. Stage7D output must contain exactly 40,509 tracker boxes and the four expected potential window totals (original 2/0, adapted 371/191). Otherwise, the script fails loudly and still packages a diagnostic log.

## Output fields

- `Stage7E_Oracle_Frame_Associations.csv`: every tracked box associated (or not) with the unique matching GT elephant ID at IoU≥0.50.
- `Stage7E_Tracker_Segment_GT_Purity.csv`: purity and GT association by tracker segment.
- `Stage7E_All_Tracked_Window_GT_Linkage.csv`: per-window oracle linkage details, matching counts, missing tracks and availability of GT future.
- `Stage7E_GT_Linked_Window_Summary.csv`: potential, strict and 80%-gate candidate totals by model and horizon.
- `Stage7E_PerSequence_GT_Linked_Windows.csv`: source sensitivity and concentration.
- `Stage7E_PerSequence_Frame_GTAudit.csv`: matched tracker box count by sequence.
- `Stage7E_Provenance.json`: hashes, methods, count gates, limitations.
- `Stage7E_Console.log`: execution details.

## Manuscript caution

This is retrospective evaluation on already-inspected BIRDSAI TestReal. Historical R13 used Ultralytics 8.3.75 while Stage7D used 8.4.216; do not describe new metrics as exact original R13 reproduction. If oracle linkage remains weak, we should report detection-to-tracking bottleneck rather than claim an end-to-end improvement. A separate **GT-blind** Transformer inference/evaluation on tracker histories is necessary to report trajectory ADE/FDE.
