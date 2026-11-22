# EXP4 Phase 6 — Independent Recording-Group Validation Protocol

**Version:** 0.1-DRAFT · **Prepared:** 22 November 2026 · **Status:** NOT PREREGISTERED / NO NEW DATA / NO CONFIRMATORY RESULTS.

## Executive decision

The existing Phase 5 clearance policy is **an exploratory heuristic**, not a validated new conformal method. The only changed decision threshold worth confirming is **2-second** `rho=0.75` versus the unchanged **2-second** `rho=1.00` baseline. At **5 seconds**, the selected `rho=1.00` equals baseline, so it cannot establish improvement. The frozen `TrainReal` q90 values must remain unchanged in this primary comparison; otherwise calibration changes will confound the decision-rule effect.

Phase 6 has two **separate** goals: (A) independent confirmatory validation of the historical decision-rule contrast on equivalent eligible *moving-anchor* synthetic-boundary instances; and (B) diagnostic transfer testing of existing 2/5 s forecasting and conformal coverage on genuinely new groups. A prospective *fixed-segment-reference* **event-level** warning study is a separate, stronger endpoint and requires new case annotations and validated coordinate transport. Do not substitute moving-anchor warning labels for fixed-reference crossing events.

## 1. Original evidence and exact frozen rules

**Phase 5 historical evidence (previously studied, not confirmation)**:

| Dataset/task | Policy | Recall | False-alert rate | Precision |
|---|---|---:|---:|---:|
| BIRDSAI TestReal, 2 s | N=2 original | 82.35% | 35.07% | 8.99% |
| BIRDSAI TestReal, 2 s | N=2, rho=0.75 | 80.00% | 26.31% | 11.34% |
| WildBox Fresh, 2 s | N=2 original | 86.81% | 37.11% | 12.08% |
| WildBox Fresh, 2 s | N=2, rho=0.75 | 82.64% | 28.03% | 14.76% |
| BIRDSAI TestReal, 5 s | N=2 original/candidate | 83.33% | 30.56% | 18.64% |

Source: user-supplied `EXP4_Phase5_Method/phase5_selected_comparison.csv` and `PHASE5_EXPLORATORY_POLICY_LOCK.json`. Each count represents a **virtual geofence instance**, not an independent wildlife incident. Synthetic reference cases are not surveyed physical boundaries.

**Frozen calibration from Step6C:** `q90(2s)=2.158767888280567`; `q90(5s)=1.975129668255982` using track-level calibration scores. Uploaded source-score audit reports 62 two-second calibration tracks (52 from sequence `0367`) and 34 five-second calibration tracks (32 from `0367`). This source concentration does not justify an arbitrary new-domain 90% coverage claim.

**Candidate definition**, using the same forecast frame and target geometry:

- Let `z` be signed maximum nominal future-path penetration toward the boundary: `z>=0` means nominal crossing.
- Let `r = q90 * s` where `s` is the frozen history-derived scale in the original coordinate system.
- A candidate uncertainty alert at threshold `rho` requires **both** the saved uncertainty intersection condition and `z >= -rho*r`.
- After two consecutive **available forecast rows** for the same recording/track/task/axis/boundary/direction, an uncertainty YELLOW candidate is persistent.
- Combined alarm = nominal crossing OR persistent YELLOW candidate. Nominal/RED crossing is never suppressed.
- Baseline uses `rho=1`; experimental 2 s candidate `rho=0.75`; **5 s remains `rho=1`**.
- Do **not** select another `rho`, model weight, detector, or threshold by looking at prospective test outcomes.

## 2. Scientific hypotheses and falsification criteria

**Primary H1 (2 s only, source-equivalent moving-anchor scenario):** Compared with original `N=2`, `rho=0.75` reduces the **group-macro** false-positive/false-warning fraction across new capture groups, without a practically unacceptable absolute recall reduction. Proposed *planning* recall noninferiority margin: **5 percentage points**. This margin and minimum sample-size must receive supervisor and domain-stakeholder approval *before data unsealing*. Do not call the margin a published regulatory safety requirement.

**H1 fails or remains inconclusive** when the group-level paired FPR difference is not convincingly below zero, the recall penalty may exceed the pre-approved margin, the number of independent groups or positive episodes is insufficient, geometry is invalid, or pipeline eligibility is too narrow to characterize deployment.

**Secondary H2 (2/5 s):** Frozen TrainReal q90 pathwise coverage and normalized radius will be estimated separately on unseen groups; the aim is to **measure** transfer (including failure), not guarantee 90% under arbitrary shift. Five-second undercoverage remains an unresolved weakness. Predefine pathwise coverage as *all future normalized 2-D target positions within the existing sample-specific q90 region*.

**Secondary H3 (fixed-reference event evaluation, separate from H1):** Compare alert-run burden and sensitivity on *deduplicated* three-before/three-after stable annotated crossing events in one fixed segment-reference image frame. These are simulated crossing events unless independent physical geofence meaning is established. Episode matching windows and event tolerances must be approved prior to evaluation; **the current package does not invent them**.

## 3. Prospective dataset acquisition, provenance and split isolation

- Use **new elephant recordings** from collection groups not represented by existing WildBox `elep1/elep2/elep3` or previously inspected sources, and not used anywhere in development or calibration. A second file from the same flight/event does not create a new group.
- Record `recording_id`, `collection_group_id`, `capture_event_id`, `subject_group_id` (when identifiable), recording date/time, site code (coarsened for safety), camera/sensor ID, native frame rate, duration, original bytes SHA-256, annotation SHA-256, and split assignment.
- Define a collection group conservatively as connected recordings sharing an acquisition session/site, camera flight, or known elephant herd/identity. If independence is uncertain, merge into the same group; do not split correlated frames across partitions.
- Proposed feasibility target: **at least 3 independent groups for a separately evaluated adaptation/calibration arm and at least 8 new, untouched confirmation groups**; these are **design targets, not a statistical power calculation**. Adjust only before pre-registration with formal sample-size and ethical/logistical review; if unavailable, label the study exploratory or inconclusive.
- Main application domain is **elephant RGB aerial recordings** with appropriate annotations and camera data. BIRDSAI thermal experiments and BAMBI red-deer/human cannot be pooled as equivalent elephant RGB confirmation. Domain-specific calibration arms require genuinely separate sets.
- Keep raw media, camera intrinsics/extrinsics, annotation source, individual/track identities and verification logs. No inferred metres: independent scale and geometric calibration needed before physical distance/geofence claims.
- Before any user-operated drone collection, obtain applicable institutional/wildlife/aviation permissions, assess animal disturbance, privacy and safe handling. Published responsible-drone guidance calls for adjusting or abandoning flights if wildlife shows adverse responses (University of Bristol, 2023; see literature section).

## 4. Eligibility and non-leakage gates

**Before accessing test labels:** freeze source list and checksums, methods, weights, eligibility, calibration groups, parameters, software/environment manifest, and thresholds. Archive a third-party-verifiable timestamp or a supervisor-signed dated pre-registration. A local JSON file alone **does not prove prospective registration**. Historical `TestReal` and WildBox recording groups are development evidence only.

Input: 20 observed positions at canonical 10 Hz; future 20 (2 s) or 50 (5 s). For primary forecasting estimates, report both annotation-conditioned and unfiltered GT-blind pipeline populations separately. Strict 20/20 GT-linked histories are **offline evaluation selection**; never present strict conditional ADE/FDE as population-wide end-to-end accuracy. All detection failures, lost tracks, unforecastable sequences and false tracks must appear in attrition/yield accounting.

**Geometry gate:** distinguish (i) moving-anchor synthetic line (comparable to Phase 5 historical warning instances), (ii) fixed segment-reference line with coordinate-consistent GT and forecast, and (iii) physically surveyed geofence. The confirmatory H1 comparison uses (i) on completely new groups to preserve the source-equivalent Phase 5 task; its result concerns a synthetic camera-relative line only. Only (ii), after validated mapping and uncertainty-region transport, qualifies for the separate proposed fixed-reference event study; (iii) needs additional independent field geometry. Ground-truth future depth may support retrospective truth re-projection, but not future-time prediction inputs.

**Timing gate:** the historical N=2 policy counts consecutive *available* forecast windows, not fixed seconds. Record actual UTC forecast timestamps, compute gaps and compare the frozen `available` rule with the **separately labeled** clock-aware sensitivity (`<= 1.1 s` between rows for planned 1 s output cadence). This clock-aware branch is not the originally selected policy and must not quietly replace the primary baseline.

## 5. Hierarchy of evaluation endpoints

| Endpoint | Unit / denominator | Critical interpretation |
|---|---|---|
| Detector recall | GT elephant box instances, IoU >=0.50; per source | All failures included; report FP/frame |
| Tracker ID continuity | Videos/GT IDs; HOTA/IDF1, ID switches, fragments | No identity-ground-truth oracle at runtime |
| Forecastability yield | Eligible tracked segments or distinct original GT identities / defined upstream eligible population | Denominator depends on stage; report flow diagram |
| Forecast ADE/FDE | normalized image-plane; pooled, track-macro and capture-group macro | No metres or speed without calibrated geometry |
| Pathwise q90 coverage | Each full future path; per group, track-macro, and pooled | Nominal q90 is not a guaranteed cross-domain coverage rate |
| q90 radius/efficiency | sample-specific `q*s` in same normalized geometry | More coverage with inflated radius is not unqualified success |
| Instance-level FPR | FP/(FP+TN) on matched synthetic directional boundary cases | Not 1 - precision; not field false alerts/hour |
| Instance-level recall | TP/(TP+FN) on matched eligible cases | Case prevalence and overlapping windows matter |
| Instance precision | TP/(TP+FP) | Report numerator and denominator |
| Event sensitivity | matched true stable crossing events / all eligible events | Deduplicate multiple overlapping forecast windows |
| Alert-run burden | unassociated runs / all runs, plus absolute counts | Not necessarily real-world false notifications |
| Alerts per monitoring hour | distinct alert runs / verified valid monitored hours | Only if continuous capture and valid timestamps establish denominator |
| End-to-end latency | capture to detector, tracker, forecaster, decision, dissemination | Offline FPS or forecast horizon is not total response time |

All primary comparisons must use the **same eligible instances** for baseline/candidate. If geometry correction changes the label definition, report it as a separate protocol rather than mixing its event counts with moving-anchor confusion matrices.

## 6. Statistical analysis and interpretation

1. Generate paired candidate/baseline predictions per eligible instance without looking at truth when computing the alarm. Preserve all output rows, including negative examples, and list absent files/detections as attrition.
2. Aggregate confusion matrices **within capture group** and then take group-macro averages; also provide pooled counts clearly labeled descriptive. Some groups may have zero positive events (undefined recall) or no warnings (undefined precision); report `NA`, never manufacture zeros.
3. Main paired effect estimates: group-macro difference in instance FPR and group-macro difference in recall (candidate minus baseline). Report every group's delta, number of evaluable groups and uncertainty. Only perform inferential paired group-bootstrap/permutation procedures when the group count/assumptions and prespecified sample-size plan justify them. Highly correlated frames/instances are **not** independent Bernoulli trials.
4. A 5 percentage-point recall margin is a **proposed planning margin**, not an accepted safety margin. Formal noninferiority needs a prespecified one-sided confidence bound and adequate independent groups/event support; otherwise conclude inconclusive.
5. For pathwise q90, report coverage numerator, denominator, per-group range, pointwise/radius distributions and diagnostic coverage by history-only motion regimes where sample sizes permit. Never tune those regimes from new test residuals.
6. Report source-specific failure cases and a separate shift table; avoid treating BIRDSAI, RGB WildBox, BAMBI, and newly collected data as a homogeneous performance population.
7. Source-specific qualitative examples may be shown only if not identifiable sensitive location data and if actual predictions/annotations are available. No fictional trajectories or confidence graphs.

## 7. End-to-end process and go/no-go gates

**Gate 0 — Supervisor-approved registration:** Approve research question, ethical approvals, geometry, group definition, sample/power plan, recall margin, fixed event matching, selected model checkpoints, and no-retuning rule; timestamp/lock BEFORE new test outcomes. Current status: **NOT MET**.

**Gate 1 — Independent source availability:** At least a feasible number of truly new groups and complete source manifests; check metadata, duplication, licensing, capture overlap and exposure history. Current status: **NO NEW GROUPS SUPPLIED**.

**Gate 2 — Reproducible extraction:** Verify RGB annotations, camera reference transforms, timestamps, original 10 Hz mapping, detector and tracker logs, all predicted and true future paths. Do not compute geofence truth in a different frame from the prediction.

**Gate 3 — Preflight execution:** Run `phase6_preflight.py`; no out-of-split or reused groups, duplicate keys, absent labels or unverified/mismatched geometry. The task-equivalent H1 rows must use `moving_anchor`, and fixed-reference rows require independently validated uncertainty transport. All preflight failures logged; never silently drop a difficult video.

**Gate 4 — Frozen replay:** Run `phase6_eval.py --geometry moving_anchor` for source-equivalent baseline vs 2 s candidate, plus 5 s unchanged control; `--geometry segment_fixed_reference` is a distinct diagnostic and cannot be pooled with moving-anchor outcomes; recompute per-group and pooled counts. For conformal and event-level metrics, use separately audited future-path and event-reference inputs; the current evaluator intentionally does not claim to compute them.

**Gate 5 — Adjudication:** Two reviewers/independent audit of uncertain event labels; per-group negative findings; full dependency/attrition table; prespecified statistical analysis. If primary objective fails, report failure without a new threshold search on confirmation footage.

**Gate 6 — Manuscript update:** Only then insert confirmed results into the final journal manuscript with exact denominators, software/artifact hashes and substantive limitations. Any second-round method after test inspection requires **another disjoint holdout**.

## 8. What is not currently validated

- No new independent capture group was supplied, hence **no confirmatory Phase 6 result** exists.
- No verified new fixed-reference forecast/GT CSV exists; the old `step7b_v2_all_geofence_instance_alarms.csv` uses earlier camera-relative warning cases and cannot be renamed into independent fixed-reference data.
- No event-level matching tolerance is independently approved, no field geofence or actual notification dataset exists.
- No new trained forecasting model or recalibrated 5 s conformal method has been evaluated in Phase 6.
- The prior 5 s undercoverage, limited Stage 7F sequence diversity, false positives and depth ambiguity remain material.

## 9. Literature grounding (verified external context, not capstone data)

- Zhou, Y., Lindemann, L., & Sesia, M. (2024). *Conformalized Adaptive Forecasting of Heterogeneous Trajectories*. ICML/PMLR 235. https://proceedings.mlr.press/v235/zhou24l.html . Establishes that simultaneous trajectory conformal prediction and heterogeneity-aware methods already exist; a new method must be compared against appropriate literature.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). *Conformal prediction beyond exchangeability*. *Annals of Statistics*, 51(2), 816–845. https://doi.org/10.1214/23-AOS2276 . Context for limits of standard exchangeability claims under shift.
- University of Bristol (2023). *Responsible drone use in biodiversity conservation: guidelines*. https://www.bristol.ac.uk/policybristol/policy-briefings/conservation-drones-guidelines/ . Animal-welfare considerations for a prospective collection protocol; not a substitute for local permissions.

## 10. Primary authentic sources used in this design

- `EXP4_Phase5_Method/PHASE5_EXPLORATORY_POLICY_LOCK.json` — actual post-hoc policy and limitations.
- `EXP4_Phase5_Method/phase5_selected_comparison.csv` — exact baseline/candidate sample counts and metrics.
- `STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json` — historical N=2 available-window selection and timing caveats.
- `crossfit_track_calibration_scores.csv` and `EXP4_Phase1_Report/calibration_source_concentration.csv` — exact q and sequence distribution.
- `Experimental Results (Part-4).pdf` — fixed-reference Stage 6 diagnostics, Stage 7 detector-to-forecast attrition and BAMBI caveats.
- `Final Demo paper.docx` / Phase 4 corrected manuscript — current claim boundaries; not proof of new test evidence.
