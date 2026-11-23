# Phase 7H — WildLive DJI_0204_video2 annotation-only intake

**Status:** `ANNOTATION_INTAKE_PASS; VISUAL_TRACK_ID_QA_PENDING; INDEPENDENT_VALIDATION_UNVERIFIED`

## Data inspected

Original user-supplied `DJI_0204_video2_boxid.zip`, 245,018 bytes, SHA-256 `e196356fb231aa520b63e709c9698b497bb0fa22edbbd94aacbe37b3a3ecd689`. This audit actually parsed all `frame_N.json` files and independently reproduced Phase 7H's original `intake` command. No video/JPEG frames were supplied for this recording; therefore no species/ID/box visual audit was performed, no model evaluated.

## Full inventory

- 357 JSON annotation frames, indices 0 through 356 inclusive, no missing indices.
- 3840 × 2160 annotation geometry throughout. All 1,428 `shapes` have `label=elephant`.
- Four labels `group_id=0,1,2,3`, each in every frame (357 annotations per track), one continuous index run per track.
- Every box has four 2D points; all coordinates finite and within annotated image bounds; no duplicate elephant IDs within a frame.
- `score=None` in 649 of the 1,428 annotations; **not automatically a bad annotation** and no score-based removal justified.
- 20-frame history followed by 20-frame target: (357 − 20 − 20 + 1) × 4 = **1,272** stride-one windows.
- 20-frame history followed by 50-frame target: (357 − 20 − 50 + 1) × 4 = **1,152** stride-one windows.
- Together these are **2,424 highly overlapping windows from ONE named recording**, not independent capture groups or independent study samples.

| Track ID | Box annotations | Continuous runs | 20→20 windows | 20→50 windows |
|---|---:|---:|---:|---:|
| 0 | 357 | 1 | 318 | 288 |
| 1 | 357 | 1 | 318 | 288 |
| 2 | 357 | 1 | 318 | 288 |
| 3 | 357 | 1 | 318 | 288 |

## Important targeted visual QA risks

1. **Track ID 1 and ID 3 bounding boxes overlap in all 357 frames.** Their median pixel-box IoU is **0.253608** (maximum **0.335462**), and their median box-centre separation is **13.91 px** at 3840 × 2160. This could reflect two legitimately close/occluded elephants (e.g., adult/calf) OR duplicated/misaligned annotations. **This numeric result cannot distinguish those hypotheses.** Do not count four *visually verified distinct elephants* yet.
2. ID 0 has the largest raw normalized centre step at frames **303→304** (0.004690), and an additional candidate at **10→11** (0.003337). ID 2's largest adjacent step is **33→34** (0.002163). Larger motion is not necessarily a bad label; camera panning or actual motion can cause it.
3. All positions are raw bounding-box centres on the image plane. There is no source camera-motion correction, altitude scale, confirmed physical ground-plane position, original FPS, frame timestamps or independent flight/session verification.

**Priority image frames for visual inspection:** `0, 33, 34, 100, 200, 303, 304, 356` (four IDs including overlap at the beginning/middle/end and flagged motion pairs). Use original frames at source resolution and render annotation overlays without modifying the original JSON.

## Research gate

- **PASS:** elephant-labelled JSON availability, CRC/integrity, index continuity, machine-readable bounding boxes, frame-count eligibility.
- **PENDING:** visual species and independent track ID verification (especially ID 1 vs ID 3), frame timing/FPS, WildBox and DJI_0207 recording-event overlap check.
- **NOT RUN:** new recording's baseline, frozen Transformer, uncertainty/conformal, early-warning metrics. Do not advertise any yet.

After manual review, keep original IDs and labels intact. If an annotation appears wrong, keep a separate review ledger and do not silently revise test ground truth. The Phase 7H frozen checkpoint lock should remain unchanged. If evaluation proceeds, report it as **exploratory frame-index transfer**, unless/source provenance and timing evidence change its status. Results must not be used to retune the locked model and reclassify the same data as untouched confirmatory evaluation.

## Reproducible output files

- `results/annotation_intake_manifest.json` — machine-readable provenance, eligibility and limitations.
- `results/track_availability.csv` — four track lengths and expected windows.
- `results/elephant_bbox_centres.csv` — all 1,428 original annotation centres, box coordinates in original pixel space and normalized image plane.
- `results/pairwise_bbox_overlap.csv` — 2,142 pairwise box-IoU rows over all six ID pairs per frame.
- `results/pairwise_overlap_summary.csv` — pair-level IoU/center-distance statistics.
- `results/adjacent_frame_centre_jumps.csv` and `results/top_20_motion_QA_pairs.csv` — targeted annotation dynamics for visual review; not evidence of real-world speed.
- `figures/elephant_box_centre_tracks.png` and `figures/ID1_ID3_box_overlap.png` — annotation-only diagnostic figures.
- `scripts/audit_dji0204.py` and `scripts/make_visual_overlays.py` — reproducible standalone Python tooling; `scripts/test_intake_scripts.py` has three passing tests.
- `phase7h_reference_intake` — original Phase 7H generalized intake manifest and CSV reproduced against this uploaded ZIP.
