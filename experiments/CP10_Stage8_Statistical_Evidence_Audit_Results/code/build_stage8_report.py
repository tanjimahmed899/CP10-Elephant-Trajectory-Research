#!/usr/bin/env python3
"""Cross-archive claim audit. Keeps exploratory vs independent evidence separate."""
import io, json, hashlib, zipfile
from pathlib import Path
import pandas as pd
root=Path('/mnt/data'); out=root/'CP10_Stage8_Statistical_Evidence_Audit';out.mkdir(exist_ok=True)

def table(zipname, csvname):
    with zipfile.ZipFile(root/zipname) as z: return pd.read_csv(io.BytesIO(z.read(csvname)))

def close(x,y,tol=1e-8):
    assert abs(x-y)<tol, (x,y)

s4=table('CP10_Stage4_R8_Learned_Model_Results.zip','CP10_Stage4_R8_Evidence/R8_SameTarget_InputOnly_Seed_Comparisons.csv')
s5=table('CP10_Stage5_Calibration_Audit_Results.zip','CP10_Stage5_Calibration_Audit/Stage5_Calibration_Coverage_Summary.csv')
s6=table('CP10_Stage6_FixedReference_Warning_Episode_Results.zip','CP10_Stage6_FixedReference_Audit/Stage6_FixedReference_Exploratory_Episode_Metrics.csv')
s7c=table('CP10_Stage7C_TestReal_Review.zip','Stage7C_Overall_IoU50.csv')
s7e=table('CP10_Stage7E_GT_Linkage_Review.zip','Stage7E_GT_Linked_Window_Summary.csv')
s8=pd.read_csv(out/'Stage8_GT_Clustered_Summary.csv')
with zipfile.ZipFile(root/'CP10_Stage7D_Tracker_Review.zip') as z:
    r=z.read('Stage7D_Readout.txt').decode()
    assert '0.204086' in r and '0.19331' in r
rows=[]
def add(stage,metric,estimate,scope,status,caveat,source):
    rows.append(dict(stage=stage,metric=metric,estimate=estimate,scope=scope,claim_status=status,key_limitations=caveat,source_archive=source))
for task,hz in [('2s_to_2s','2s'),('2s_to_5s','5s')]:
    d=s4[s4.task==task]
    raw=d.raw_input_ADE.mean(); corr=d.corrected_input_ADE.mean();gain=100*(raw-corr)/raw
    add('R8 / Stage 4',hz+' input-only ADE',f'raw {raw:.5f} vs corrected {corr:.5f} ({gain:.1f}% lower)',
        'Three seed-matched Transformer runs on historical TestReal','Positive retrospective ablation',
        'Same corrected-frame target; recording groups previously studied; no external holdout',
        'CP10_Stage4_R8_Learned_Model_Results.zip')
for task,hz in [('2s_to_2s','2s'),('2s_to_5s','5s')]:
    evals=s5[(s5.split=='evaluation')&(s5.horizon==task)]
    fr=evals[evals.method=='Frozen TrainReal q90'].iloc[0]
    dv=evals[evals.method=='Development-only split-conformal q90'].iloc[0]
    add('Stage 5',hz+' retrospective evaluation pathwise coverage',f'frozen {fr.covered}/{fr.n} ({fr.pathwise_coverage*100:.1f}%) vs development-only {dv.covered}/{dv.n} ({dv.pathwise_coverage*100:.1f}%)',
        'Two prior-group evaluation videos','Negative calibration-transfer result',
        'Overlapping windows; development and evaluation are not independent recording groups; conditional coverage not guaranteed',
        'CP10_Stage5_Calibration_Audit_Results.zip')
for hz in ['2s','5s']:
    subset=s6[(s6.source=='EXP3')&(s6.horizon==hz)&(s6.persistence_k==3)]
    nom=subset[subset.policy=='nominal'].iloc[0]
    n2=subset[subset.policy=='fixed_N2_gapaware'].iloc[0]
    add('Stage 6',hz+' fixed-reference episode recall',f'nominal {nom.events_detected}/{nom.eligible_annotated_crossing_events}, N2 {n2.events_detected}/{n2.eligible_annotated_crossing_events}; unassociated N2 alert runs={n2.event_unassociated_alert_runs}',
        'Six EXP3 videos; simulated 3-frame-stable 2D virtual-line crossings','Exploratory episode diagnostics',
        'Predicted future depth approximated, not a surveyed world geofence; high alert-run burden',
        'CP10_Stage6_FixedReference_Warning_Episode_Results.zip')
original=s7c[s7c.model=='original'].iloc[0];adapted=s7c[s7c.model=='adapted'].iloc[0]
close(original.recall,2453/48055)
add('Stage 7C','BIRDSAI TestReal detector recall IoU>=0.50',f'{original.recall*100:.2f}% original vs {adapted.recall*100:.2f}% adapted; FP {int(original.FP)} to {int(adapted.FP)}',
    '9,085 frames, 48,055 annotated objects, fixed confidence .10','Positive detector improvement with cost',
    'Non-independent, high false positives (5.80 boxes/frame); historical R13 count non-exact parity',
    'CP10_Stage7C_TestReal_Review.zip')
add('Stage 7D','ByteTrack HOTA / IDF1', 'HOTA 0.0252 to 0.2041; IDF1 0.0214 to 0.1933',
    'Six TestReal sequences; same frozen replay settings','Positive tracking improvement with cost',
    'Identity switches 43 to 1307; source software version differs from historical R13',
    'CP10_Stage7D_Tracker_Review.zip')
for hz in ['2s','5s']:
    row=s7e[(s7e.model=='adapted')&(s7e.horizon==hz)].iloc[0]
    add('Stage 7E',hz+' strictly GT-linked forecastable windows',f'{int(row.GT_linked_strict)} / {int(row.potential)} potential',
        'Offline GT association; all strict windows from sequence 0353','Conditional evaluation availability',
        'GT-based eligibility not deployable; original detector has zero strict windows',
        'CP10_Stage7E_GT_Linkage_Review.zip')
for _,g in s8.iterrows():
    add('Stage 7F / Stage 8',g.horizon+' conditional tracker-to-Transformer ADE',
        f'Transformer {g.ADE_transformer_window_mean:.5f} vs stationary {g.ADE_stationary_window_mean:.5f} (gain {g.ADE_reduction_window_pct:.1f}%; identity-resampling gain interval [{g.bootstrap_2p5_ADE_gain:.5f},{g.bootstrap_97p5_ADE_gain:.5f}])',
        f'One TestReal sequence, {int(g.n_windows)} overlapping windows, {int(g.n_gt_identities)} GT elephants',
        'Positive within-sequence conditional result',
        'Cluster bootstrap within one video only; cannot infer to other videos or GT-blind operation',
        'CP10_Stage7F_Tracker_Forecast_Review.zip')
ledger=pd.DataFrame(rows)
ledger.to_csv(out/'Stage8_Cross_Experiment_Claim_Ledger.csv',index=False)
manifest={}
for file in sorted(root.glob('CP10_Stage*.zip')):
    if file.name in set(ledger.source_archive):
        manifest[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
(out/'Stage8_Upstream_Archive_Hashes.json').write_text(json.dumps(manifest,indent=2))
fmt=lambda x:f'{x:.5f}'
a=s8.loc[s8.horizon=='2s'].iloc[0];b=s8.loc[s8.horizon=='5s'].iloc[0]
report=f'''# CP10 — Stage 8 Statistical Evidence Audit (21 November 2026)

## Status

The corrected Stage 7F review ZIP completed without the previous `summarize()` error. All six frozen Transformer checkpoints (three seeds, two horizons) were evaluated. The final archive contains two prediction NPZ files, 3,110 per-predictor metric rows, 311 unique eligible windows, provenance records, and per-elephant summaries. Numerical replay from the NPZ arrays reproduced the reported sample ADE/FDE and stationary errors to floating-point precision (<1e-12). The source ZIP passed CRC verification.

**Scope:** these are retrospective, GT-eligibility-conditioned forecasts from one BIRDSAI TestReal sequence (`0000000353_0000000000`). Window timestamps overlap, and the source recording was used in earlier research; this is not a new external holdout or a GT-blind production score.

## Stage 7F paired ADE results

| Measure | 2 s | 5 s |
|---|---:|---:|
| GT-linked strict evaluation windows | {int(a.n_windows)} | {int(b.n_windows)} |
| Distinct GT elephants | {int(a.n_gt_identities)} | {int(b.n_gt_identities)} |
| Tracker-history Transformer ensemble ADE | {fmt(a.ADE_transformer_window_mean)} | {fmt(b.ADE_transformer_window_mean)} |
| Tracker-history stationary ADE | {fmt(a.ADE_stationary_window_mean)} | {fmt(b.ADE_stationary_window_mean)} |
| Window-weighted ADE reduction | {a.ADE_reduction_window_pct:.2f}% | {b.ADE_reduction_window_pct:.2f}% |
| Window-weighted absolute ADE gain | {fmt(a.ADE_gain_window_mean)} | {fmt(b.ADE_gain_window_mean)} |
| GT-identity macro ADE reduction | {a.ADE_reduction_gt_macro_pct:.2f}% | {b.ADE_reduction_gt_macro_pct:.2f}% |
| GT identities with positive mean ADE gain | {int(a.GT_identities_with_positive_ADE_gain)}/{int(a.n_gt_identities)} | {int(b.GT_identities_with_positive_ADE_gain)}/{int(b.n_gt_identities)} |
| GT-identity bootstrap 2.5% to 97.5% interval for absolute ADE gain | [{fmt(a.bootstrap_2p5_ADE_gain)}, {fmt(a.bootstrap_97p5_ADE_gain)}] | [{fmt(b.bootstrap_2p5_ADE_gain)}, {fmt(b.bootstrap_97p5_ADE_gain)}] |
| GT-identity bootstrap interval for relative ADE reduction | [{a.bootstrap_2p5_ADE_reduction_pct:.1f}%, {a.bootstrap_97p5_ADE_reduction_pct:.1f}%] | [{b.bootstrap_2p5_ADE_reduction_pct:.1f}%, {b.bootstrap_97p5_ADE_reduction_pct:.1f}%] |
| Transformer FDE | {fmt(a.FDE_transformer_window_mean)} | {fmt(b.FDE_transformer_window_mean)} |
| Stationary FDE | {fmt(a.FDE_stationary_window_mean)} | {fmt(b.FDE_stationary_window_mean)} |

### Statistical design

We calculate **paired differences** for each identical anchor window: `stationary ADE – frozen Transformer ADE`. Instead of pretending 187/124 overlapping windows are independent, we resample GT elephants **with replacement** ({20000:,} replicates, seed 20261121), carrying each selected elephant's complete set of windows and recomputing the pooled paired means each replicate. This returns within-video *identity-resampling descriptive intervals*. The intervals DO NOT reflect between-video shift or between-recording variability: there is one sequence and elephant motion/recording effects can correlate across identities. Do not present these intervals as cross-domain guarantees or an independent-significance test.

Window-weighted and identity-macro reductions differ (especially 2s) because elephants contribute unequal counts of overlapping windows. The manuscript should report **both**. Three model seeds are not three independent datasets; ensemble metrics and bootstrap samples should not be confused with independent replication.

### Eligibility and distribution caution

Offline ground truth was used to admit only histories with 20/20 GT-associated tracked positions and complete future annotations. Neither future annotations nor GT IDs were inputs to the frozen Transformer, but the evaluation population was selected with oracle information. Other TestReal sequences contributed **zero strict windows**. Therefore neither detector-level improvement nor HOTA enhancement implies a reliable online warning system. In particular, a lower tracker-history error than the GT-history diagnostic on this selected population does **not** prove more accurate elephant localization than ground truth.

## Cross-stage evidence synthesis

| Experiment | Result that is supported | What is **not** established |
|---|---|---|
| R8 / Stage 4 | Corrected-history Transformers outperform raw-history alternatives under same-target retrospective TestReal ablation (2s ~66.5% ADE reduction, 5s ~53.6%) | An independent dataset, exact 3D camera-movement correction, or metre accuracy |
| Stage 5 | Development-only conformal q90 transfers poorly (2s 3/53, 5s 2/28 pathwise coverage on selected evaluation videos), while frozen TrainReal q90 obtains 53/53 and 24/28 | Restored 90% conditional coverage, a universal calibration law, or independent-group generalization |
| Stage 6 | Fixed-reference GT crossing-event simulation detects reference inconsistencies and quantifies recall/false-run trade-offs; EXP3 N2 detects 116/133 (2s) and 109/124 (5s) stable events | GPS-anchored ground fence, verified future-depth projection, or deployment safety |
| Stage 7C | Fixed-protocol TestReal detector recall rises 5.10%→56.98%, with much higher false boxes/frame | Precision-preserving detector replacement or cross-dataset independence |
| Stage 7D | Same-protocol tracker replay HOTA rises 0.0252→0.2041; IDF1 rises 0.0214→0.1933 | Stable operational identities; ID switches also rise sharply |
| Stage 7E | Adapted tracker yields 187 strict 2s and 124 strict 5s GT-linked evaluation windows | GT-blind track eligibility or multi-sequence strict forecast availability |
| Stage 7F | Frozen Transformer improves conditional ADE over stationary predictions on one sequence | Deployment-grade end-to-end forecasting or generalizable risk alerts |

## Manuscript-ready conservative wording

"On a retrospective BIRDSAI TestReal sequence, the frozen three-seed Transformer ensemble achieved mean ADE of **0.01066** and **0.02305** normalized image-coordinate units for 2-s and 5-s horizons, respectively, when supplied with detector–ByteTrack histories passing a strict offline ground-truth association gate. These correspond to **21.7%** and **18.5%** lower mean ADE than stationary forecasts using the same tracked histories. Across 13 and 10 annotated elephant identities, respectively, the per-identity mean ADE advantage was positive for 13/13 and 9/10 animals. The analysis remains limited to one previously studied recording, overlapping forecast windows, and ground-truth-conditional eligibility; it does not establish fully automated field performance."

## Decision and next experiments

1. **Lock the positive results** (R8, Stage7C/7D, conditional Stage7F), retaining exact data and evaluation protocol definitions; do not generalize across distinct TestReal/Fresh/EXP3 populations.
2. **Report the failed or weak aspects** (Stage5 domain-shift calibration, Stage6 high false-run burden and assumed depth, Stage7C high FP/frame, only one sequence supporting strict downstream forecasts). These are scientifically necessary, not presentation errors.
3. A genuine next major confirmation requires **new, preregistered recording groups** and a fixed physical boundary/reference protocol. Thresholds, detector models and inclusion criteria must be locked beforehand. Without additional independent scenes, no statistical device can manufacture an external holdout.
4. For manuscript finalization, include tables clearly marked *annotation-conditioned retrospective*, *fixed-protocol previously studied TestReal*, and *exploratory geometry simulation*. Do not claim clinical/field/operational readiness.

## Files and reproducibility

- `Stage8_Prediction_Array_to_CSV_Verification.csv` — numerical NPZ-to-CSV verification for both arms, stationary and ensemble.
- `Stage8_Report_to_Samples_Consistency.csv` — corrected reporter outputs vs independently recomputed sample aggregates.
- `Stage8_GT_Clustered_Summary.csv` — paired means, identity-macro effects and bootstrap intervals.
- `Stage8_Per_GT_Identity_Differences.csv` and `Stage8_Paired_Window_Differences.csv` — audit and replot data.
- `Stage8_Cross_Experiment_Claim_Ledger.csv` — study-level claim status and source archives.
- `Stage8_Audit_Manifest.json` / `Stage8_Upstream_Archive_Hashes.json` — source provenance.
- `audit_stage8.py` — standalone rerun script: `python audit_stage8.py --input CP10_Stage7F_Tracker_Forecast_Review.zip --out ./Stage8_Rerun --replicates 20000`.
'''
(out/'STAGE8_STATISTICAL_AUDIT_AND_PAPER_CLAIMS.md').write_text(report,encoding='utf8')
print('Ledger rows',len(rows));print(ledger.to_string(index=False,max_colwidth=34))
