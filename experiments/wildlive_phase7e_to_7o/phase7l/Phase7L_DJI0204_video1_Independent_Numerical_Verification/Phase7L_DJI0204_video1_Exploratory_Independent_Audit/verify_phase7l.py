#!/usr/bin/env python3
"""Independent numerical verification of Phase 7L DJI_0204_video1 uploaded evaluation.
No model forward pass: checkpoint binaries/prediction trajectories absent.
"""
from pathlib import Path
import zipfile, json, hashlib, io, sys, argparse
import numpy as np, pandas as pd
parser=argparse.ArgumentParser(description='Verify Phase7L DJI0204_video1 report, source, gate choices and per-window ADE/FDE')
parser.add_argument('--evaluation',type=Path,required=True)
parser.add_argument('--annotations',type=Path,required=True)
parser.add_argument('--kit',type=Path,required=True,help='Extracted original Phase7K validation kit containing prospective_lock.json and src/')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
EVAL=args.evaluation.resolve()
ANN=args.annotations.resolve()
KIT=args.kit.resolve()
OUT=args.out.resolve()
OUT.mkdir(parents=True,exist_ok=True)
BASE='results/DJI_0204_video1_RELATED_SCENE_EXPLORATORY/'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_csv(z,path):
    return pd.read_csv(io.BytesIO(z.read(BASE+path)),dtype={'track_id':str})

with zipfile.ZipFile(EVAL) as z:
    assert z.testzip() is None,'Broken ZIP/CRC'
    manifest=json.loads(z.read(BASE+'evaluation_manifest.json'))
    detail=load_csv(z,'all_experts_and_gate_per_window.csv')
    choices=load_csv(z,'gate_choices_OBSERVATION_ONLY.csv')
    summary=load_csv(z,'per_track_and_pooled_descriptive.csv')
    floor=load_csv(z,'scale_floor_rates.csv')
    availability=load_csv(z,'track_availability.csv')
assert manifest['sequence_id']=='DJI_0204_video1'
assert manifest['file_sha256']==sha(ANN) and manifest['file_bytes']==ANN.stat().st_size
lock=json.loads((KIT/'prospective_lock.json').read_text())
for h in (20,50):
    fn=f'phase7j_dev_gate_h{h}.json'
    expected=lock['release_files_sha256'][f'locked_models/{fn}']
    actual=sha(KIT/'locked_models'/fn)
    assert expected==actual==manifest['gate_sha256'][fn],f'Gate hash mismatch: {h}'
for k,v in lock['model_checkpoint_sha256'].items():
    val=manifest['checkpoint_hashes_verified'][k]
    assert val['sha256']==v['sha256'] and val['bytes']==v['size_bytes'],f'Checkpoint hash mismatch: {k}'

assert manifest['n_scored_windows']==len(detail)==len(choices)==2748
assert set(detail.horizon_frames.unique())=={20,50}
keys=['sequence','horizon_frames','track_id','segment_id','first_observed_frame']
assert not detail.duplicated(keys).any() and not choices.duplicated(keys).any()
assert set(detail.sequence)=={'DJI_0204_video1'}
joined=detail.merge(choices,on=keys,validate='one_to_one',suffixes=('','_choice'),indicator=True)
assert len(joined)==len(detail) and (joined['_merge']=='both').all()
assert joined.selected_expert.eq(joined.selected_expert_choice).all()
expected_models={'stationary','cv_ols','transformer_ensemble'}
assert set(detail.selected_expert).issubset(expected_models)
for model in ['stationary','cv_ols','transformer_ensemble','cv20','adaptive']:
    for metric in ['ade','fde']:
        col=f'{model}_{metric}_norm'
        vals=detail[col].to_numpy(dtype=float)
        assert np.isfinite(vals).all() and (vals>=0).all(),f'Invalid {col}'
for model in expected_models:
    block=detail.loc[detail.selected_expert.eq(model)]
    for metric in ['ade','fde']:
        assert np.allclose(block[f'adaptive_{metric}_norm'],block[f'{model}_{metric}_norm'],atol=1e-12,rtol=1e-10)
assert np.allclose(detail.adaptive_minus_cv_ols_ADE,detail.adaptive_ade_norm-detail.cv_ols_ade_norm,atol=1e-12,rtol=1e-10)

models=['stationary','cv20','cv_ols','transformer_ensemble','adaptive']
result=[];track_result=[];offsets=[];selection=[];paired=[]
for h in (20,50):
    d=detail.query('horizon_frames == @h').copy()
    target_windows=1464 if h==20 else 1284
    max_start=243 if h==20 else 213
    assert len(d)==target_windows
    assert set(d.track_id)==set(map(str,range(6)))
    assert (d.last_target_frame==d.first_observed_frame+20+h-1).all()
    for t in map(str,range(6)):
        dt=d.query('track_id==@t')
        assert len(dt)==max_start+1
        assert np.array_equal(np.sort(dt.first_observed_frame.to_numpy()),np.arange(max_start+1))
    for track, dt in [('pooled',d)]+[(t,d[d.track_id==t]) for t in map(str,range(6))]:
        rec=summary[(summary.horizon_frames==h)&(summary.track_id==track)]
        assert len(rec)==1
        rec=rec.iloc[0]
        assert int(rec.windows)==len(dt)
        for model in models:
            for metric in ['ade','fde']:
                c=f'{model}_{metric}_norm'
                assert np.isclose(dt[c].mean(),rec[c],atol=1e-12,rtol=1e-10),f'{h},{track},{c}'
        for model in expected_models:
            c=f'selected_{model}_fraction'
            assert np.isclose(dt.selected_expert.eq(model).mean(),rec[c],atol=1e-12,rtol=1e-10)
        assert np.isclose(dt.adaptive_minus_cv_ols_ADE.mean(),rec.adaptive_minus_cv_ols_ADE,atol=1e-12)
        row={'horizon_frames':h,'track_id':track,'windows':len(dt)}
        for model in models:
            row[f'{model}_ADE']=dt[f'{model}_ade_norm'].mean()
            row[f'{model}_FDE']=dt[f'{model}_fde_norm'].mean()
        row['adaptive_minus_cv_ols_ADE']=dt.adaptive_minus_cv_ols_ADE.mean()
        row['adaptive_relative_ADE_change_percent_vs_cv_ols']=100*(row['adaptive_ADE']/row['cv_ols_ADE']-1)
        row['adaptive_minus_cv_ols_FDE']=(dt.adaptive_fde_norm-dt.cv_ols_fde_norm).mean()
        (result if track=='pooled' else track_result).append(row)
    for mod in ['stationary','cv_ols','transformer_ensemble']:
        selection.append({'horizon_frames':h,'selected_expert':mod,'count':int(d.selected_expert.eq(mod).sum()),'fraction':float(d.selected_expert.eq(mod).mean())})
    pair=d.adaptive_ade_norm-d.cv_ols_ade_norm
    paired.append({'horizon_frames':h,'windows':len(d),'adaptive_better_window_fraction':float((pair < -1e-12).mean()),'adaptive_worse_window_fraction':float((pair > 1e-12).mean()),'adaptive_equal_window_fraction':float((np.abs(pair)<=1e-12).mean()),'median_delta_ADE':float(pair.median()),'p95_delta_ADE':float(pair.quantile(.95))})
    for offset in range(20+h):
        sample=d[d.first_observed_frame%(20+h)==offset]
        assert len(sample)>0
        delta=(sample.adaptive_ade_norm-sample.cv_ols_ade_norm).mean()
        best=min(sample[f'{m}_ade_norm'].mean() for m in models[:-1])
        offsets.append({'horizon_frames':h,'offset':offset,'number_of_windows_all_tracks':len(sample),'adaptive_minus_cv_ols_ADE':float(delta),'adaptive_minus_best_fixed_model_ADE':float(sample.adaptive_ade_norm.mean()-best)})
assert availability.windows_20.sum()==1464 and availability.windows_50.sum()==1284
assert np.allclose(floor.scale_floor_active_fraction,1.0)

# Rebuild *all* frozen gate decisions from original annotation histories.
sys.path.insert(0,str(KIT/'src'))
import phase7k
inv,tracks,segments=phase7k.inventory(ANN,'DJI_0204_video1')
assert inv['n_20_frame_windows']==1464 and inv['n_50_frame_windows']==1284
obs=phase7k.build_observation_features(segments,'DJI_0204_video1')
regenerated=phase7k.predict_gate_from_features(obs)
regenerated['track_id']=regenerated.track_id.astype(str)
choices['track_id']=choices.track_id.astype(str)
recompare=regenerated.merge(choices,on=keys,validate='one_to_one',indicator=True,suffixes=('_regenerated','_archive'))
assert len(recompare)==2748 and recompare['_merge'].eq('both').all()
assert recompare.selected_expert_regenerated.eq(recompare.selected_expert_archive).all(), 'Mismatch in gate re-generation'

for name,df in [('verified_pooled_summary',pd.DataFrame(result)),('verified_per_elephant_summary',pd.DataFrame(track_result)),('selection_counts',pd.DataFrame(selection)),('nonoverlap_start_offset_sensitivity',pd.DataFrame(offsets)),('paired_window_diagnostics',pd.DataFrame(paired))]:
    df.to_csv(OUT/f'{name}.csv',index=False,float_format='%.12g')
text={
    'upload_zip_CRC_PASS':True,'source_sha256_verified':True,
    'released_gate_hashes_verified':2,
    'checkpoint_hashes_equal_to_locked_manifest':6,
    'checkpoint_binaries_recomputed':False,
    'transformer_forward_inference_recomputed':False,
    'gate_decisions_from_original_annotations_matched':2748,
    'unique_window_keys':2748,
    'source_independence_verified':False,
    'source_capture_group_status':'DJI_0204 family; scene overlap with DJI_0204_video2; exploratory only',
    'numeric_summary_rows_matched':len(summary),
    'scale_floor_activation_all_windows':True,
}
(OUT/'verification_status.json').write_text(json.dumps(text,indent=2)+'\n')
print('PASS',text)
print('\nPooled ADE/FDE:');print(pd.DataFrame(result).to_string(index=False))
print('\nGATE:');print(pd.DataFrame(selection).to_string(index=False))
print('\nPAIRED:');print(pd.DataFrame(paired).to_string(index=False))
print('\nOFFSETS:')
o=pd.DataFrame(offsets)
for h in (20,50):
 x=o[o.horizon_frames==h];print(h,len(x),'adaptive better CV',sum(x.adaptive_minus_cv_ols_ADE<0),'adaptive better best-fixed',sum(x.adaptive_minus_best_fixed_model_ADE<0),'range',x.adaptive_minus_cv_ols_ADE.min(),x.adaptive_minus_cv_ols_ADE.max())
