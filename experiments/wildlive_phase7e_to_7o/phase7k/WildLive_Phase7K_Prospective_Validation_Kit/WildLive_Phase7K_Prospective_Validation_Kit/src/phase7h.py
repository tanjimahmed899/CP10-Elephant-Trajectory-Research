#!/usr/bin/env python3
"""Phase 7H: frame-index-only WildLive elephant sequence intake and locked frozen evaluation.

Dataset files stay local. A passing computational test NEVER proves independent flights.
No FPS guesses, no re-training, no thresholds tuned from candidate test results.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re,sys,zipfile
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from phase7f_locked_reference import (
    TASKS,SEEDS,OBS,canonicalize,reconstruct,stationary,cv_ols,model_factory,
    load_checkpoint,predict,sha256_file,ade_fde
)

BASELINES=('stationary','cv20','cv_ols')
ALL_MODELS=(*BASELINES,'transformer_ensemble')
KNOWN_WILDBOX_VIDEOS=[
    'DJI_20240119151437_0002_V','DJI_20240119151048_0001_V',
    'DJI_20240626143443_0003_V','DJI_20240626143248_0002_V',
    'DJI_20260227084724_0003_V','DJI_20260227082707_0003_V']
FRAME_RE=re.compile(r'(?:^|/)frame_(\d+)\.json$',re.I)


def write_csv(path, rows):
    if not rows:return
    with open(path,'w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def parse_archive(path):
    """Use full bounding-box geometry, not score; support two/four corner rectangles."""
    frames={}; labels=Counter(); missing_score=0; dimensions=None
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'ZIP CRC error: {bad}')
        for name in z.namelist():
            m=FRAME_RE.search(name)
            if not m:continue
            idx=int(m.group(1))
            if idx in frames:raise ValueError(f'duplicate frame number {idx}; one ZIP must contain ONE sequence')
            j=json.loads(z.read(name))
            w,h=j.get('imageWidth'),j.get('imageHeight')
            if not isinstance(w,int) or not isinstance(h,int) or w<=0 or h<=0:
                raise ValueError(f'invalid image dimensions: {name}')
            if dimensions is not None and dimensions!=(w,h):
                raise ValueError('mixed annotation resolutions in one recording')
            dimensions=(w,h)
            seen={}
            for s in j.get('shapes',[]):
                label=str(s.get('label','')).strip().lower()
                labels[label]+=1
                if label!='elephant':continue
                if s.get('group_id') is None:
                    raise ValueError(f'elephant without identity in {name}; no invented tracking IDs')
                tid=str(s['group_id'])
                if tid in seen:raise ValueError(f'duplicate elephant group_id {tid} in {name}')
                pts=np.asarray(s.get('points',[]),dtype=float)
                if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<2 or not np.isfinite(pts).all():
                    raise ValueError(f'invalid box points in {name} id={tid}')
                lo=pts.min(axis=0);hi=pts.max(axis=0)
                if (lo<0).any() or hi[0]>w or hi[1]>h or (hi<=lo).any():
                    raise ValueError(f'out-of-bounds rectangle in {name} id={tid}')
                if s.get('score') is None: missing_score+=1
                seen[tid]=((lo+hi)/np.array([2*w,2*h],dtype=float)).tolist()
            frames[idx]=seen
    if not frames:raise ValueError('No frame_N.json annotations in ZIP')
    tracks=defaultdict(list)
    for t in sorted(frames):
        for ident,xy in frames[t].items(): tracks[ident].append((t,xy))
    return frames,tracks,{'dimensions':dimensions,'labels':dict(labels),'missing_elephant_scores':missing_score}


def contiguous_runs(track):
    out=[];run=[]
    for idx,xy in sorted(track):
        if run and idx!=run[-1][0]+1:
            out.append(run);run=[]
        run.append((idx,xy))
    if run:out.append(run)
    return out


def inventory(path,sequence):
    frames,tracks,meta=parse_archive(path)
    rows=[];segments=[]
    for tid,track in sorted(tracks.items()):
        runs=contiguous_runs(track)
        for n,run in enumerate(runs):
            arr=np.array([p for _,p in run],dtype=np.float64)
            start,end=run[0][0],run[-1][0]
            segments.append({'track_id':tid,'segment_id':n,'start_frame':start,'end_frame':end,'coords':arr})
        lengths=[len(x) for x in runs]
        rows.append({'track_id':tid,'observed_annotations':len(track),'continuous_runs':len(runs),
            'longest_run_frames':max(lengths),'windows_20':sum(max(0,L-OBS-20+1) for L in lengths),
            'windows_50':sum(max(0,L-OBS-50+1) for L in lengths)})
    frame_ids=sorted(frames)
    result={'phase':'7H','sequence_id':sequence,'file_sha256':sha256_file(path),
        'file_bytes':path.stat().st_size,'min_frame_index':min(frame_ids),'max_frame_index':max(frame_ids),
        'n_annotation_frames':len(frames),'nonconsecutive_frame_indices':
            len(frame_ids)!=max(frame_ids)-min(frame_ids)+1,
        'elephant_track_ids':len(tracks),'labels':meta['labels'],
        'source_dimensions_px':meta['dimensions'],'missing_elephant_scores':meta['missing_elephant_scores'],
        'n_20_frame_windows':sum(x['windows_20'] for x in rows),
        'n_50_frame_windows':sum(x['windows_50'] for x in rows),
        'historical_wildbox_exact_id_match':sequence in KNOWN_WILDBOX_VIDEOS,
        'known_wildbox_sources_checked':KNOWN_WILDBOX_VIDEOS,
        'native_fps_verified':False,'released_frame_timestamps_verified':False,
        'source_independence_verified':False,'new_independent_confirmation':False,
        'recording_isolation':'UNVERIFIED; cross-sequence duplicates and same flight/herd must be reviewed',
        'coordinate_units':'normalized image-plane box centre; no camera correction',
        'review_requirements':['Visual species verification','Check nearby sequences share same flight/herd',
            'Compare frames and source IDs against historical WildBox',
            'Verify timestamps before claiming seconds-based performance']}
    return result,rows,segments


def make_windows(seg,h):
    coords=seg['coords'];n=len(coords)-OBS-h+1
    if n<1:return None
    # strided views copied to prevent accidental in-place changes
    x=np.stack([coords[i:i+OBS] for i in range(n)]).astype(np.float64)
    y=np.stack([coords[i+OBS:i+OBS+h] for i in range(n)]).astype(np.float64)
    start=np.arange(seg['start_frame'],seg['start_frame']+n,dtype=int)
    return x,y,start


def evaluate(segments,checkpoint_dir,lock,batch_size):
    if batch_size<1:raise ValueError('batch_size must be positive')
    if not checkpoint_dir:raise ValueError('Frozen checkpoints are mandatory')
    required={}
    for h in TASKS:
        for seed in SEEDS:
            name=f'{TASKS[h]["name"]}_Transformer_seed{seed}.pt'
            path=checkpoint_dir/name
            if not path.is_file():raise FileNotFoundError(f'Missing frozen checkpoint: {path}')
            actual=sha256_file(path)
            expected=lock['model_checkpoints'][name]['sha256']
            if actual!=expected:
                raise ValueError(f'CHECKPOINT HASH MISMATCH: {name} (possibly ablation or changed weights)')
            required[name]={'sha256':actual,'bytes':path.stat().st_size}
    if not any(len(s['coords'])>=OBS+20 for s in segments):
        raise ValueError('No 20-frame task window in this elephant recording')
    details=[];summary=[];scales=[]
    for h in TASKS:
        # All inference uses original training canonicalization and untouched checkpoints.
        models=[load_checkpoint(checkpoint_dir/f'{TASKS[h]["name"]}_Transformer_seed{seed}.pt',
                                TASKS[h]['name'],seed,h) for seed in SEEDS]
        for seg in segments:
            windows=make_windows(seg,h)
            if windows is None:continue
            x,y,starts=windows
            can=canonicalize(x,TASKS[h]['scale_floor'])
            scales.append({'horizon_frames':h,'track_id':seg['track_id'],
                           'segment_id':seg['segment_id'],'windows':len(starts),
                           'scale_floor_active_fraction':float(np.mean(can['path']<TASKS[h]['scale_floor']))})
            pred_can=[predict(m,can['history_canonical'],batch_size) for m in models]
            trans=np.mean([reconstruct(pc,can) for pc in pred_can],axis=0)
            velocity=(x[:,-1,:]-x[:,0,:])/(OBS-1)
            times=np.arange(1,h+1)[None,:,None]
            preds={'stationary':stationary(x,h),
                   'cv20':stationary(x,h)+times*velocity[:,None,:],
                   'cv_ols':cv_ols(x,h),'transformer_ensemble':trans}
            errors={k:ade_fde(v,y) for k,v in preds.items()}
            for i,start in enumerate(starts):
                row={'horizon_frames':h,'track_id':seg['track_id'],
                     'segment_id':seg['segment_id'],'first_observed_frame':int(start),
                     'last_target_frame':int(start+OBS+h-1)}
                for model,(ade,fde) in errors.items():
                    row[f'{model}_ade_norm']=float(ade[i])
                    row[f'{model}_fde_norm']=float(fde[i])
                details.append(row)
    # Store both per-track and pooled values. Pooled is window-weighted and descriptive.
    for h in TASKS:
        for tid in ('pooled',*sorted({r['track_id'] for r in details if r['horizon_frames']==h})):
            subset=[r for r in details if r['horizon_frames']==h and (tid=='pooled' or r['track_id']==tid)]
            if not subset:continue
            for m in ALL_MODELS:
                summary.append({'horizon_frames':h,'track_id':tid,'model':m,
                    'windows':len(subset),'ade_norm':float(np.mean([r[m+'_ade_norm'] for r in subset])),
                    'fde_norm':float(np.mean([r[m+'_fde_norm'] for r in subset]))})
    return summary,details,scales,required


def command_intake(args):
    args.output.mkdir(parents=True,exist_ok=True)
    result,tracks,segments=inventory(args.zip,args.sequence)
    (args.output/'intake_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    write_csv(args.output/'track_availability.csv',tracks)
    print(json.dumps({k:result[k] for k in ('sequence_id','n_annotation_frames','labels',
        'elephant_track_ids','n_20_frame_windows','n_50_frame_windows',
        'source_independence_verified')},indent=2))
    if result['elephant_track_ids']==0:print('INELIGIBLE: no label=elephant found')
    if result['n_50_frame_windows']==0:print('INELIGIBLE for 50-frame task: no >=70-frame continuous elephant track')
    print('ANNOTATION INTAKE ONLY: flight independence, timing, and visual labels UNVERIFIED')


def command_evaluate(args):
    args.output.mkdir(parents=True,exist_ok=True)
    result,tracks,segments=inventory(args.zip,args.sequence)
    lock=json.loads(args.lock.read_text(encoding='utf8'))
    summary,details,scales,ck=evaluate(segments,args.checkpoint_dir,lock,args.batch_size)
    write_csv(args.output/'locked_model_per_window.csv',details)
    write_csv(args.output/'locked_model_per_track_and_pooled.csv',summary)
    write_csv(args.output/'canonicalization_scale_floor_rates.csv',scales)
    write_csv(args.output/'track_availability.csv',tracks)
    result.update({'locked_model_checkpoint_hashes_verified':ck,
      'model_weights_retrained':False,'forecast_errors_exploratory_only':True,
      'any_statistical_significance_claim':False,
      'new_confirmatory_evidence':False,
      'status':'LOCKED_FRAME_EVALUATION_COMPLETE__INDEPENDENCE_UNVERIFIED'})
    (args.output/'evaluation_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for row in summary:
        if row['track_id']=='pooled':print(row)
    print('EXPLORATORY ONLY: provenance, frame timing and cross-source overlap not independently verified')


def command_aggregate(args):
    results=[]; hashes=set(); seen=set()
    for p in args.inputs:
        m=json.loads((p/'evaluation_manifest.json').read_text())
        if m['sequence_id'] in seen:raise ValueError('Duplicate sequence ID')
        if m['file_sha256'] in hashes:raise ValueError('Same annotation ZIP included twice')
        hashes.add(m['file_sha256']);seen.add(m['sequence_id'])
        if m['sequence_id']=='DJI_0207':
            raise ValueError('DJI_0207 is previously inspected; exclude it from new-group evaluation cohort')
        with (p/'locked_model_per_track_and_pooled.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                if r['track_id']!='pooled':continue
                results.append({'sequence_id':m['sequence_id'],'horizon_frames':r['horizon_frames'],
                    'model':r['model'],'windows':r['windows'],'ade_norm':r['ade_norm'],
                    'fde_norm':r['fde_norm'],'capture_group_independence':'UNVERIFIED'})
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'per_sequence_descriptive_scores.csv',results)
    aggregate={'unique_candidate_sequence_names':len(seen),
        'independent_capture_groups_confirmed':0,
        'new_independent_confirmation':'NOT ESTABLISHED',
        'notes':'Each sequence may share a flight/herd/session; filename and file hashes do not prove independence. No p-values or group CIs.',
        'sequences':sorted(seen)}
    (args.output/'aggregate_status.json').write_text(json.dumps(aggregate,indent=2),encoding='utf8')
    print(json.dumps(aggregate,indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('intake');a.add_argument('--zip',type=Path,required=True)
    a.add_argument('--sequence',required=True);a.add_argument('--output',type=Path,required=True)
    a.set_defaults(func=command_intake)
    b=sub.add_parser('evaluate');b.add_argument('--zip',type=Path,required=True)
    b.add_argument('--sequence',required=True);b.add_argument('--checkpoint-dir',type=Path,required=True)
    b.add_argument('--lock',type=Path,default=Path(__file__).resolve().parents[1]/'checkpoint_lock.json')
    b.add_argument('--output',type=Path,required=True);b.add_argument('--batch-size',type=int,default=128)
    b.set_defaults(func=command_evaluate)
    c=sub.add_parser('aggregate');c.add_argument('--inputs',type=Path,nargs='+',required=True)
    c.add_argument('--output',type=Path,required=True);c.set_defaults(func=command_aggregate)
    args=p.parse_args(argv);args.func(args)

if __name__=='__main__':main()
