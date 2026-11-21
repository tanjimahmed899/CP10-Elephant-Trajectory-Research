# CP10 — Stage 9 manuscript integration, audit, and editorial handover

Date: 21 November 2026

## Deliverable

- `CP10_Stage9_Revised_Journal_Manuscript.docx`: a **20-page**, editable, A4, single-column scientific review draft, rendered and inspected after generation.
- 10 embedded empirical figures (original saved drone RGB/forecast overlays and Stage 4–8 numerical plots).
- 10 numbered summary tables; 17 references retained/normalized from the 25-page source paper; substantive methods, results, limitations, and appendices.
- Original base source: `/mnt/data/Demo paper.pdf` (25 pages). This is a newly composed, structurally revised draft using that source and uploaded Stage 4–8 evidence archives, **not** a track-changes redline of an original DOCX.

## Main changes and exact evaluation populations

1. Stage 4 R8 corrected-history **input-only, same-target** learned Transformer comparison: TestReal 2s N=1,053, ADE raw 0.058217 vs corrected 0.019500 (66.51% reduction); 5s N=826, ADE raw 0.078495 vs corrected 0.036340 (53.62% reduction), three matched seeds. Numerical replay of **estimated affine** preprocessing across 2,792 CORE windows: max coordinate discrepancy 5.96e-8; not exact 3D physical pose.
2. EXP3 WildBox six-video frozen forecasts: 754/572 windows, mean ADE 0.050488/0.112798; original frozen q90 pathwise coverage 96.55%/69.76%. These are six video files from three previously seen recording groups, not independent sites.
3. Stage 5 calibration-transfer **negative result**: 2s development-only q90 covers 3/53 and 5s 2/28 retrospective evaluation windows; frozen TrainReal q90 covers 53/53 and 24/28. This is *not* improved calibration or a conformal guarantee.
4. Stage 6 replay: 1,471 windows from nine WildBox videos; fixed segment-reference GT matching to 1.19e-7, altered crossing labels in 299/4,524 (2s) and 327/3,432 (5s) original cases; frozen N=2 matches 116/133 and 109/124 3-frame-stable synthetic crossing events, with 348 and 237 unassociated alert runs. This is not a GPS or ground-surveyed geofence.
5. Stage 7C adapted thermal detector: TestReal 9,085 frames, recall 5.10%→56.98% and false positive boxes 1,835→52,684. Important precision cost documented.
6. Stage 7D frozen ByteTrack: HOTA 0.025220→0.204086, IDF1 0.021390→0.193310; identity switches 43→1,307 and fragmentation 145→3,597. No claim of stable operational track identity.
7. Stage 7E strict offline GT linkage: 187 2s and 124 5s windows, all from one TestReal sequence (`0353`), with original detector yielding zero strict windows. Offline selection must not be described as a GT-blind throughput result.
8. Stage 7F same-input tracker-history frozen ensemble: 2s ADE 0.010664 vs stationary 0.013623 (21.72%); 5s 0.023050 vs 0.028285 (18.51%). Only one previously studied video; not fully end-to-end forecasting across all elephants.
9. Stage 8 bootstrap of complete GT identities within this one video: 2s absolute ADE gain interval [0.00187,0.00390]; 5s [0.00212,0.00720]. Descriptive within-video uncertainty only.

## Source artifacts (supplied by the researcher)

- `Demo paper.pdf` — original 25-page manuscript, dataset descriptions, historical comparisons, and reference list.
- `CP10_Stage4_R8_Learned_Model_Results.zip` — learned camera-motion ablation and reconstructions.
- `CP10_Stage5_Calibration_Audit_Results.zip` — frozen residual comparisons and transfer failure.
- `CP10_Stage6_FixedReference_Warning_Episode_Results.zip` — 3D annotation pose replay, fixed-reference warning cases and event matches.
- `CP10_Stage7C_TestReal_Review.zip` — fixed-protocol detector evaluation.
- `CP10_Stage7D_Tracker_Review.zip` — frozen tracker replay.
- `CP10_Stage7E_GT_Linkage_Review.zip` — GT linkage and track history feasibility.
- `CP10_Stage7F_Tracker_Forecast_Review.zip` — frozen model forecasts.
- `CP10_Stage8_Statistical_Evidence_Audit_Results.zip` — final consistency and descriptive intervals.

## Editor's scientific guardrails

- No author-flown/drone-captured new video is included; source-provider UAV footage and BIRDSAI thermal imagery are separate domains.
- No normalized image-space metric should be presented in metres.
- No claim of independent site/video-group generalization for previously inspected test recordings.
- Do not conflate source-specific TestReal, Fresh WildBox, EXP3, Stage 5, and conditional Stage 7F population estimates.
- Full field alarm performance, source-independent conformal coverage, a surveyed physical fence, and safe operational false-warning rates remain **unestablished**.
- Bibliography was derived from original manuscript and partly spot-checked against public publisher/arXiv/official dataset listings (Elchik 2025, Rastogi 2025, WildBox 2026, Samad 2026). Remaining references and full author lists **require independent reference verification** and target-journal formatting. The WildBox dataset resource reference was changed to the publicly verifiable `3DOMFBK/WildBox` resource.
- Page count is **20** on the local LibreOffice A4 render. Word font substitutions or journal layout requirements may change pagination.
- Figures were originally generated by the project experiments, but the manuscript itself has not undergone independent peer review. The draft is not presented as accepted or journal submission-ready without the remaining verification.

## Formatting and QA

The DOCX was generated with `python-docx`, rendered with `/home/oai/skills/docx/render_docx.py`, and visually inspected using thumbnails of each page. No pages had text bounding boxes outside the sheet margins in the rendered PDF. The final draft is single-column, editable, A4, Times New Roman, 11 pt body, with repeating page footers and numbered figures/tables.

## Supervisor review checklist

1. Confirm all data-use permissions and dataset-version identities, especially WildBox archive origin and metadata license.
2. Independently verify every bibliography entry, spelling, author order, DOI, publication status, and URL.
3. Specify the journal and apply its article type, word limit, references, figure count, and graphical abstract requirements.
4. Add author names, affiliations, corresponding author, ethics and data-availability statements, contributions, and funding disclosures as applicable.
5. Decide whether the journal permits a long retrospective negative-results-and-methodology paper; if not, reduce tables/appendix or submit a narrower Stage 4 + Stage 7 methodological focus.
6. Keep the original archived results and code alongside the manuscript; publish only after checking possible identity or site privacy risks in field footage and source metadata.
