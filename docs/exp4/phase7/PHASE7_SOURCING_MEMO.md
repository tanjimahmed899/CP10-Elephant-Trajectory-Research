# Phase 7 — Source acquisition, provenance checks, and external-validation readiness

**Research:** Elephant trajectory forecasting, conformal uncertainty, and virtual-boundary warning evaluation.  
**Date:** 22 November 2026.  
**Status:** DATA SOURCING IN PROGRESS — **NO NEW RECORDINGS ACQUIRED OR PROSPECTIVE EVALUATIONS EXECUTED**.

## Decision

The Phase 6 plan cannot legitimately be called experimentally validated from the existing BIRDSAI, WildBox elep1/elep2/elep3, or BAMBI evidence. The next executable action is controlled acquisition of source-disjoint elephant RGB videos, *not* further tuning on TestReal or the previously inspected WildBox source groups.

**Phase 5 hypothesis retained unchanged:** two-second moving-anchor synthetic-geofence policy `N=2`, frozen TrainReal q90 `2.158767888280567`, baseline clearance `rho=1.0`, candidate `rho=0.75`. Five-second candidate is `rho=1.0`, **identical to baseline** (q90 `1.975129668255982`); therefore the 5-second branch is a transfer/coverage diagnostic, not a claimed policy improvement. The original two-available-forecast-windows persistence remains the primary historical rule.

## Verified external source screening (public metadata, not newly acquired experiment data)

| Source | What the official records support | Main barrier | Screening decision |
|---|---|---|---|
| **WildLive** — University of Bristol, DOI 10.5523/bris.ewnuoroebuae20vei0cu6zu69 | 22 RGB UAV 4K sequences, 19,139 frames, 215,800 mixed-species boxes and masks, 291 mixed-species tracklets and sparse point tracks; elephant included; footage from Ol Pejeta, Kenya | Full 204.6 GiB dataset, non-commercial license, elephant-specific group count and flight-pose availability unverified. **WildBox also used Ol Pejeta** and has overlapping contributing investigators; source overlap must be investigated | **Priority 1 candidate, not approved as independent** |
| **Elchik et al. elephant drone tracking** (2025, DOI 10.1139/dsa-2025-0032) | Eight 4K MP4 clips in Welgevonden, South Africa; original research describes tracking annotations. Published repository makes software available | All eight clips are from the **same day and the same herd**; raw clips/GT annotations and licence must be requested or confirmed. Treat conservatively as one capture group | **Priority 2 candidate, not eight groups** |
| **Dryad elephant-drone habituation** (2026, DOI 10.5061/dryad.c866t1gjb) | 35 trials of 14 identified groups; behavioural tabular records including flight IDs | Public release is ~97 KB of CSV and R code, not labelled 10-Hz elephant trajectories or video footage | **Useful prospective partner contact; not forecast data as released** |
| **WildBox** | Elephant groups `elep1`, `elep2`, `elep3` with per-frame cameras/track data | Already used repeatedly in development / EXP3 | **Exclude from new-group confirmation** |
| **UgaWildlifeVision** | Updated annotated still frames derived from NESTLER, small elephant subset | NESTLER context already studied; still-image annotations do not provide guaranteed longitudinal tracks or pose | **Detection-only candidate, not primary trajectory confirmation** |
| **BAMBI** | Cross-species dataset with red deer/human pose/track context | Not elephant-specific; previously evaluated | **Exclude from elephant confirmation** |

### Evidence and source links

- WildLive official record: https://data.bris.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69 ; project: https://dat-nguyenvn.github.io/WildLive/ ; DOI https://doi.org/10.5523/bris.ewnuoroebuae20vei0cu6zu69 . Bristol's resource catalogue states the **Non-Commercial Government Licence** and complete archive size **204.6 GiB**. This is not proof of permission for all intended downstream uses; review exact terms.
- WildBox README: https://huggingface.co/datasets/3DOMFBK/WildBox/blob/main/README.md ; WildBox public metadata and NeurIPS Croissant validation report identify Ol Pejeta among source sites, giving **plausible but unproven** overlap with WildLive.
- Welgevonden original study: https://researchonline.ljmu.ac.uk/id/eprint/28795/ ; official implementation: https://github.com/ConservationDronesAI/ElephantDetectionAndTracking . Article's methodology states **eight videos from the same day and herd**. Code MIT licensing does **not** automatically grant raw-video redistribution rights.
- Dryad elephant behavioural metadata: https://datadryad.org/dataset/doi:10.5061/dryad.c866t1gjb .
- UgaWildlifeVision source: https://zenodo.org/records/23171670 .

## Acquisition-first work order

1. **Contact WildLive curators:** request an elephant-only video inventory, exact capture dates/flight IDs, point/box temporal continuity, camera-intrinsic/extrinsic or IMU/GPS/pose metadata, use conditions, and a determination of whether frames overlap the WildBox elep1/2/3 sources. A video with only an annotation track ID, but without 20+50 contiguous sampled positions, cannot support the 5-second forecast task. If no valid camera geometry exists, moving-anchor H1 may still be tested with clearly bounded interpretation; fixed-reference H3 cannot be claimed.
2. **Contact Welgevonden study curators:** request raw eight clips, native frame timestamps, track-ID labels, metadata and permitted research use. Treat the eight clips as a single cluster until source-level evidence proves distinct capture events/elephants (the published description says one herd).
3. **Conduct metadata-only intake:** do not inspect performance on candidate new groups. Fill the template `recording_intake_TEMPLATE.csv`; first compile hashes, capture event/date, subject group, and file-level metadata. Use `scripts/phase7_intake_gate.py`; this detects obvious aliasing and group leakage **but does not itself establish source independence**.
4. **Supervisor review:** approve H1 5-percentage-point recall loss margin (or revise **before opening test outcomes**), minimum independent group/event support, target population, rights, geometry, and any numerical power analysis. Phase 6's suggested 3 calibration / 8 confirmation groups are planning targets **not empirically justified sample sizes**. Capture a dated external registration or signed protocol.
5. **Freeze and run:** only after authorized raw media, GT, and provenance exist, use Phase 6's `phase6_preflight.py` and `phase6_eval.py` on new-group instances. Keep all model checkpoints, eligibility, thresholds, and group splits frozen. Do not use GT future data to determine live prediction behavior.

## Scientific stop / proceed gates

- **STOP:** provenance evidence absent; source overlap unresolved; same herd/flight or previously inspected group marketed as multiple independent groups; unusable track-ID chronology; missing exact 20-history + 20/50-future path support; incompatible sensor without a stated cross-modal protocol; unapproved reuse rights; annotation future leakage into model or alert decisions.
- **PROCEED TO SEPARATE, LABELLED EXPLORATORY PILOT:** if elephant RGB video with track annotations exists but camera poses or sufficient independent groups are absent. Pilot may assess feasibility, not confirmatory independent generalization.
- **PROCEED TO SUPERVISOR REVIEW FOR PROSPECTIVE CONFIRMATION:** when actual disjoint collection groups, annotation, rights, consent/wildlife governance as applicable, geometry, protocol approval, power and timestamp evidence are documented. The script can screen these prerequisites but can never verify them by assertion alone.

## Reproducibility and claims

The current source-screening inventory and local software checks are NOT measurements of forecast coverage, ADE/FDE, warning precision or any field outcome. No data have been downloaded, no new test groups analysed, and no new manuscript results are warranted. Source identities and overlaps require curator/original metadata confirmation. Discovery of a differently named publicly released dataset is insufficient proof of fresh, independent data.
