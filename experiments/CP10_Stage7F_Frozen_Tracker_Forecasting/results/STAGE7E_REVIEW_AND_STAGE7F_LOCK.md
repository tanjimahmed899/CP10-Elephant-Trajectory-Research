# Stage7E scientific interpretation and Stage7F predeclared continuation

Data provenance: `CP10_Stage7E_GT_Linkage_Review.zip` (original user-submitted completed WSL audit), with per-frame Hungarian GT↔track IoU≥0.50 and model-level counts reconstructed from the CSV. The audit did not execute Transformer inference.

| Pipeline / horizon | Potential windows | Strict GT-linked (20/20) | Diagnostic ≥16/20, no conflict |
|---|---:|---:|---:|
| Original YOLO + frozen ByteTrack, 2s | 2 | 0 | 0 |
| Original YOLO + frozen ByteTrack, 5s | 0 | 0 | 0 |
| Adapted YOLO + frozen ByteTrack, 2s | 371 | 187 | 264 |
| Adapted YOLO + frozen ByteTrack, 5s | 191 | 124 | 164 |

Crucial clustering: all 187 strict 2s and 124 strict 5s cases come from TestReal sequence `0000000353_0000000000`, and represent **13 distinct GT elephant identities** at 2s and **10 identities** at 5s. Other sequences produce zero strictly eligible windows. Each window is overlapping/correlated with others. A sample-level bootstrap treating all windows as independent is inappropriate; inferential claims must acknowledge single-sequence support.

Most importantly, 'forecastable' means a tracker sequence has a full-length sequence of observations; 'GT linked' means a retrospective matching rule verifies which elephant it corresponds to. Neither means the Transformer forecast was measured. This is *not* a fully deployable, GT-blind end-to-end success rate because track eligibility requires ground truth.

Stage7F *frozen evaluation* design:

1. Take Stage7D tracker bounding-box centers for strict 20/20 histories; **never** substitute GT history into this arm.
2. Apply original R8 affine image-motion transforms from the observed-window start to every observed point; original GT future is registered to the same frame solely for evaluation.
3. Canonicalize history exactly as in the completed Stage5/R8 Transformer pipeline (10-step linear-slope heading, path-length scale floor); freeze original 2s/5s Transformer seeds 42, 52, 62.
4. Predict actual 2s/5s trajectories and record ADE, FDE, per-elephant clustered errors. Include three-seed ensemble plus individual seeds.
5. Separately compute a *non-deployable oracle-history diagnostic*: identical frozen model with GT-derived observed histories, evaluated on precisely the same future targets. This quantifies the downstream cost of imperfect tracking inputs. Oracle is not used for model selection and will be labeled explicitly.
6. Maintain 187 / 124 exact window eligibility counts, original Stage7D SHA verification, source GT SHA verification and experiment hashes. Any discrepancy halts the experiment.

**Claims not supported by Stage7E**: successful end-to-end ADE/FDE, fully GT-blind usable track yield, all-sequence downstream improvement, independent holdout, or field-deployable elephant warning. Further confirmation needs additional sequences and a truly GT-blind operational yield metric.
