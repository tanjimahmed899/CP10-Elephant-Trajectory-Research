# Phase 7O: valid and invalid claims

| Claim | Status | Basis |
|---|---|---|
| Input ZIP SHA-256 identities checked | Verified in generated run | `source_file_hashes.csv` and lock |
| 8 targeted image pairs, 24 transform fits | Verified | `heldout_registration_models.csv` |
| Homography lower median held-out BG fit error in 8/8 sampled pairs | Supported, retrospective | Per-pair matched ORB held-out evaluation |
| 5 event pairs exceed 30-native-px post-registration residual | Supported as diagnostic flags | `high_residual_QA_events_NOT_confirmed_errors.csv` |
| Five events are confirmed annotation errors | NOT supported | Needs manual identity/pose/box review |
| High-residual events cause model errors | NOT supported | No causal manipulation |
| Full corrected forecast ADE improves | NOT evaluated | Insufficient full RGB/coordinate predictions |
| Flight-level independence | NOT established | Missing source capture-event metadata |
| Locked adaptive outperforms best fixed expert | NOT supported | Phase 7N archived results |
| Operational field-ready safety alerts | NOT established | No verified physical geofence/ground-truth deployment |

Post hoc window stratification is not a new independent experiment. Keep the full unfiltered primary result and publish all caveats.
