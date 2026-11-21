# Stage 7D tracking audit: verified from uploaded original report

**Dataset:** BIRDSAI TestReal, 6 sequences, 9,085 frames, 48,055 observed GT elephant boxes, 45 GT identities. Saved Stage7C detections; no new YOLO inference; matched frozen ByteTrack YAML; Ultralytics 8.4.216.

| Metric | Original | Adapted |
|---|---:|---:|
| HOTA | 0.025220 | **0.204086** |
| IDF1 | 0.021390 | **0.193310** |
| MOTA | 0.011945 | **0.147643** |
| ID switches | 43 | 1,307 |
| Fragmentations | 145 | 3,597 |
| Tracked boxes | 1,595 | 38,914 |
| Tracker identities | 74 | 1,582 |
| CLR true positives | 1,106 | 23,658 |
| CLR false positives | 489 | 15,256 |
| Potential 2s-to-2s windows | 2 | 371 |
| Potential 2s-to-5s windows | 0 | 191 |

**Interpretation:** Accuracy and associations improved markedly, but substantial identity errors, fragmentation and false tracks remain. The adapted model has MOTA −0.186217 on sequence `0000000352_0000000000`. Of 191 potential long-horizon windows, 180 (94.2%) are from `0000000353_0000000000`. Thus potential window counts are not independently GT-validated nor broad across recordings.

**Limitations:** This is retrospective TestReal analysis. The new replay is comparable within Ultralytics 8.4.216, but not an exact historical Ultralytics 8.3.75 R13 reproduction. Tracker-derived potential windows do not demonstrate correct GT elephant ID, true future supervision, successful Transformer inference, independent holdout performance or real-time deployment.

**Next:** Apply GT-linked quality audit in Stage7E, then freeze a GT-blind downstream Transformer evaluation for valid tracker histories only. No threshold tuning on TestReal.
