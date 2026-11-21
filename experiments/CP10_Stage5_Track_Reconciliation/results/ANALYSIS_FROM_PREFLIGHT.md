# Stage 5 decision from the supplied archive feasibility audit

The 3 candidate ZIPs comprise 918 frame–label pairs (459 + 89 + 370), 1,379 KITTI elephant 2D boxes (459 + 178 + 742), and 259 frames containing multiple elephants (89 + 170). Their `tracking_summary.json` tracks are all named generic `object`; this *does not* establish an absence of elephants. Treat KITTI object labels and tracker class names as different annotation namespaces until a geometric correspondence has been verified.

| Candidate | Frames | KITTI elephant boxes | Generic 3D tracks | 2 s / 5 s potential windows (ambiguous label) |
| --- | ---: | ---: | ---: | ---: |
| elep1 / DJI_20240119152552_0006_V | 459 | 459 | 3 across three segments | 342 / 262 |
| elep2 / DJI_20240626143832_0004_V | 89 | 178 | 2 | 100 / 40 |
| elep3 / DJI_20260227084907_0004_V | 370 | 742 | 6 across two segments | 424 / 248 |
| Total | 918 | 1,379 | 11 | 866 / 550 |

Counts are upper-bound overlapping windows based on contiguous per-track frame indices at the preflight's assumed 10 Hz. They are not semantic-qualified forecast windows and do not constitute independent evidence units. The first video's frames were previously used for qualitative examination; the latter two videos' full prior use has not been conclusively ruled out. All three belong to previously used recording groups, so a new-group confirmatory claim would be incorrect.

The accompanying script checks per-frame IoU of tracker boxes with KITTI elephant annotations after mapping tracking-space coordinates into image-space dimensions. For confidence, it uses one-to-one assignments; only tracks with at least 10 observed frames and >=80% elephant-box match rate are *provisionally* resolved. It also reports strict contiguous windows with GT elephant matches in every frame. This is a semantic and geometric data audit, not a model performance metric.

On earlier EXP3 sampled video segments with elephant GT, 96 out of 96 GT elephant boxes were matched at IoU>=0.5 after 512x288 -> 1920x1080 rescaling. This is a test of the association code on *prior* sources, not a result for the 3 candidate ZIPs. The candidate ZIPs are only on the user's WSL disk, and must be audited locally.
