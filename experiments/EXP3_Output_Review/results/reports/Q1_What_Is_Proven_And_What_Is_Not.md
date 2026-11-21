# Q1 evidence-and-claim gate — November 16, 2026

## Established from uploaded files and prior replay

- Historical YOLOv8n detection baseline: 94.0% mAP50 on its held-out detection split. Not a WildBox transfer detector score.
- Transformer ensemble: TestReal 2 s ADE 0.019076, 5 s 0.035590; earlier three-source Fresh WildBox 2 s 0.028814, 5 s 0.050132 (normalized image coordinates).
- R8 translation-aligned **CV only** matched evaluation: 2 s ADE 0.053522 raw vs 0.020148 corrected, 5 s 0.123675 vs 0.040174, same corrected future target. Full projective equivalence NOT established.
- R8 Stationary matched improvement exactly zero. Avoid native-frame stationarity improvement claims.
- Step7H frozen Transformer warning retrospective replay: 20,286 geofence cases; Fresh 2 s nominal recall/FWR 0.285/0.005, q90 N=2 0.868/0.371, with RED urgent recall 0.250. Individual six-boundary cases are correlated, not separate real events.
- Step7G JSON confirms N=2 selected post hoc from TestReal; not independent TestReal confirmation.
- 6 checkpoints load and have expected forward shapes; historical checkpoint authenticity separately requires training-time hash comparison.
- 6 additional video source IDs selected without outcome inspection; exact publisher SHA256 metadata now preserved.

## Unresolved physical/empirical gates

1. No six selected new video ZIPs in working session; no legitimate fresh EXP3 ADE/FDE, q90 coverage, Step7H RED/YELLOW/GREEN numbers.
2. No frame-wise source-to-anchor camera transforms or original full image inputs; no full R8 projective geometry match.
3. End-to-end detector→tracker→forecaster propagation too sparse (2 usable 2 s windows; 0 5 s); no real-time Raspberry Pi validation.
4. No new elephant-specific independent ecological site/safari group or actual surveyed geofence.
5. Original Step7H event-level / false alerts per hour cannot be inferred from boundary-instance FPR.

## Scientific publication rule

Do not substitute fake estimates, redefine the same data as independent confirmation, drop zero-window videos, tune parameters on the held-out set, or assert human-elephant conflict reduction. Manuscript is a working draft until data gates pass.
