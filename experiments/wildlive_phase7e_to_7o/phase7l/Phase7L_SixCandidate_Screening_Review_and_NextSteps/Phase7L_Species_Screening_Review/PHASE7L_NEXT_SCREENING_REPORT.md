# CP 1.0 — Phase 7L: Six-candidate species-screen review

Input: `selected_candidate_species_samples.csv` (SHA-256 `2c1f5ab6b2045b7f5dd607a7b4d903069fadc31aae53dc03594edfa03a7c1f9f`).

## Observed sample-only findings

- All six candidates had `OTHER_LABELS_ONLY_IN_SAMPLES` status, so none had an elephant label in the sampled, readable JSON.
- `DJI_0142_video1`: giraffe, 5/5 sampled frames readable.
- `DJI_0117_video1`: zebra, 4/5; frame 600 failed with HTTPError.
- `DJI_0117_video4`, `DJI_0601_video4`, `DJI_0601_video5`, and `DJI_20230719145816_0003_V_video2`: zebra, 5/5 each.
- No negative sample result establishes whole-sequence absence of elephants.
- No candidate has verified source-flight or herd independence from this evidence.

## Exact next action

Use `screen_remaining_six.sh` under Ubuntu/WSL where the original Phase 7K kit resides. It probes five annotation frames in each of the remaining six minimally screened folders without looking at model errors or changing locked gates. It outputs `remaining_six_candidate_species_samples.csv` and copies it to Windows Downloads.

Other previously sampled elephant-positive folder `DJI_0204_video1` belongs to the same filename family as development `DJI_0204_video2`. It may be useful for additional exploratory analysis but is *not* an independent validation unit until original video/flight provenance and overlap are assessed.

**No new validation data or improved forecasting result is claimed in this report.**
