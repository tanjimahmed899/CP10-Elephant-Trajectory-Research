#!/usr/bin/env python3
"""Stage7E: retrospective GT-linked tracker-window feasibility audit.

No training, detection inference, tracking rerun, or TestReal parameter tuning.
Ground-truth identity is used for ORACLE VERIFICATION ONLY, never online inference.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SEQS = ['0000000055_0000000000','0000000058_0000000000',
        '0000000349_0000000000','0000000352_0000000000',
        '0000000353_0000000000','0000000357_0000000000']
BASE='/mnt/e/AIUB/Capstone Project/Journal Based Results'
STAGE7D_REQUIRED=['Stage7D_ByteTrack_All_Boxes.csv','Stage7D_Forecastable_Segments.csv',
                 'Stage7D_Provenance.json','Stage7D_Aggregate_Tracking.csv']


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def canon_id(v):
    if pd.isna(v):return ''
    s=str(v).strip()
    try:
        v=float(s)
        if np.isfinite(v) and v.is_integer():return str(int(v))
    except (ValueError,TypeError):pass
    return s


def iou_matrix(gt_boxes, pr_boxes):
    if len(gt_boxes)==0 or len(pr_boxes)==0:return np.zeros((len(gt_boxes),len(pr_boxes)))
    g=np.asarray(gt_boxes,float);p=np.asarray(pr_boxes,float)
    x1=np.maximum(g[:,None,0],p[None,:,0]);y1=np.maximum(g[:,None,1],p[None,:,1])
    x2=np.minimum(g[:,None,2],p[None,:,2]);y2=np.minimum(g[:,None,3],p[None,:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    ga=np.maximum(0,g[:,2]-g[:,0])*np.maximum(0,g[:,3]-g[:,1])
    pa=np.maximum(0,p[:,2]-p[:,0])*np.maximum(0,p[:,3]-p[:,1])
    den=ga[:,None]+pa[None,:]-inter
    return np.divide(inter,den,out=np.zeros_like(inter),where=den>1e-12)


def match_one_frame(gt, pr, threshold=0.5):
    """Assign each tracker box to at most one GT elephant via IoU/Hungarian."""
    if len(gt)==0 or len(pr)==0:return {}
    g=np.column_stack([gt.x.to_numpy(float),gt.y.to_numpy(float),
                       (gt.x+gt.w).to_numpy(float),(gt.y+gt.h).to_numpy(float)])
    p=pr[['x1','y1','x2','y2']].to_numpy(float)
    iou=iou_matrix(g,p)
    # Prefer the maximum number of valid matches, then the highest total IoU.
    # Matching below the threshold receives no reward.
    score=np.where(iou>=threshold,1.0+iou,0.0)
    gi,pi=linear_sum_assignment(-score)
    return {int(pr.index[j]):(str(gt.iloc[i].gt_id),float(iou[i,j]))
            for i,j in zip(gi,pi) if iou[i,j]>=threshold}


def get_input(zip_path,gt_path,allow_partial):
    with zipfile.ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'Invalid stage7D review ZIP member {bad}')
        for n in STAGE7D_REQUIRED:
            if n not in z.namelist():raise ValueError('Missing Stage7D source file '+n)
        src=json.loads(z.read('Stage7D_Provenance.json'))
        if src.get('status')!='TRACKER_METRICS_COMPLETE':
            raise ValueError('Stage7D provenance is not COMPLETE')
        tracks=pd.read_csv(io.BytesIO(z.read('Stage7D_ByteTrack_All_Boxes.csv')),
                           dtype={'sequence':str})
        segments=pd.read_csv(io.BytesIO(z.read('Stage7D_Forecastable_Segments.csv')),
                            dtype={'sequence':str})
    if len(tracks)!=40509:raise ValueError(f'Unexpected tracker output rows: {len(tracks)} != 40509')
    if set(tracks.model.unique())!={'original','adapted'}:raise ValueError('Unexpected model arms')
    if set(tracks.sequence.unique()) != set(SEQS):
        raise ValueError('Unexpected sequence coverage in tracker boxes')
    gt=pd.read_csv(gt_path,low_memory=False,dtype={'sequence':str})
    if not {'sequence','split','frame','object_id','x','y','w','h'}.issubset(gt.columns):
        raise ValueError('GT file missing required track/bbox fields')
    split=gt['split'].astype(str).str.strip().str.lower()
    splitmask=split.isin(['test_real','testreal','test-real'])
    if not splitmask.any(): splitmask=split.str.contains('test',regex=False)
    gt=gt.loc[splitmask & gt.sequence.isin(SEQS)].copy()
    if 'interpolated' in gt.columns:
        gt=gt.loc[pd.to_numeric(gt.interpolated,errors='coerce').fillna(0)==0].copy()
    if len(gt)!=48055 and not allow_partial:raise ValueError(f'Expected exactly 48055 GT instances, got {len(gt)}')
    gt['frame']=pd.to_numeric(gt.frame,errors='raise').astype(int)
    gt['gt_id']=gt.object_id.map(canon_id)
    gt['x']=pd.to_numeric(gt.x,errors='raise');gt['y']=pd.to_numeric(gt.y,errors='raise')
    gt['w']=pd.to_numeric(gt.w,errors='raise');gt['h']=pd.to_numeric(gt.h,errors='raise')
    if (gt[['w','h']]<=0).any().any():raise ValueError('Invalid ground-truth bbox dimensions')
    tracks['frame']=pd.to_numeric(tracks.frame,errors='raise').astype(int)
    tracks['track_id']=pd.to_numeric(tracks.track_id,errors='raise').astype(int)
    if tracks.duplicated(['model','sequence','frame','track_id']).any():
        raise ValueError('Duplicate tracker boxes with same model/sequence/frame/track-id')
    return tracks,segments,gt,src


def run(args):
    zp=Path(args.stage7d).expanduser(); gp=Path(args.gt).expanduser()
    out=Path(args.out).expanduser();out.mkdir(parents=True,exist_ok=True)
    tracks,segments,gt,prov=get_input(zp,gp,args.allow_partial_gt)
    # One-to-one framewise association; no GT information is passed to ByteTrack.
    truth={(seq,int(frame)):g for (seq,frame),g in gt.groupby(['sequence','frame'],sort=False)}
    matched_gt=np.full(len(tracks),'',dtype=object)
    matched_iou=np.zeros(len(tracks),dtype=np.float32)
    n_frames=0
    for (model,seq,frame),prs in tracks.groupby(['model','sequence','frame'],sort=False):
        pairs=match_one_frame(truth.get((seq,int(frame)),gt.iloc[0:0]),prs)
        for ix,(obj,iou) in pairs.items():matched_gt[ix]=obj; matched_iou[ix]=iou
        n_frames+=1
    tracks['oracle_GT_id']=matched_gt;tracks['oracle_IoU']=matched_iou
    tracks['oracle_matched']=tracks.oracle_GT_id!=''
    trcols=['model','sequence','frame','track_id','oracle_GT_id','oracle_IoU','oracle_matched','confidence']
    tracks[trcols].to_csv(out/'Stage7E_Oracle_Frame_Associations.csv',index=False)
    # Exact same R13F-compatible candidate generation as Stage7D.
    trdict={(model,seq,int(tid)): g.set_index('frame',drop=False).sort_index()
            for (model,seq,tid),g in tracks.groupby(['model','sequence','track_id'],sort=False)}
    gtids={(seq,int(frame)):set(g.gt_id.astype(str)) for (seq,frame),g in truth.items()}
    windows=[];segment_details=[]
    for _,seg in segments.iterrows():
        model,seq,tid=str(seg.model),str(seg.sequence),int(seg.track_id)
        key=(model,seq,tid)
        group=trdict[key]
        start,end=int(seg.start_frame),int(seg.end_frame)
        obs=group.loc[(group.index>=start)&(group.index<=end)]
        fs=np.sort(obs.frame.unique())
        assert len(fs)==int(seg.frames_observed) and fs[0]==start and fs[-1]==end,(key,start,end)
        assert np.max(np.diff(fs),initial=0)<=4,(key,start,end)
        sampled=np.arange(start,end+1,3,dtype=int)
        assert len(sampled)==int(seg.n10),(key,start,end,len(sampled),int(seg.n10))
        segment_ids=obs.loc[obs.oracle_matched,'oracle_GT_id'].tolist()
        cnt=Counter(segment_ids)
        lead_obj=cnt.most_common(1)[0][0] if cnt else ''
        segment_details.append({'model':model,'sequence':seq,'track_id':tid,'segment':int(seg.segment),
            'frames_observed':len(obs),'oracle_matched_frames':int(obs.oracle_matched.sum()),
            'oracle_matched_fraction':float(obs.oracle_matched.mean()),'distinct_GT_ids':len(cnt),
            'dominant_GT_id_DIAGNOSTIC':lead_obj,'dominant_GT_frames_DIAGNOSTIC':cnt[lead_obj],
            'windows_2s':int(seg.windows_2s),'windows_5s':int(seg.windows_5s)})
        for horizon,n_future,n_expected in [('2s',20,int(seg.windows_2s)),('5s',50,int(seg.windows_5s))]:
            total=20+n_future
            starts=list(range(0,len(sampled)-total+1,10))
            assert len(starts)==n_expected,(key,horizon,len(starts),n_expected)
            for k in starts:
                hist=sampled[k:k+20]
                fut=sampled[k+20:k+total]
                lookups=[]
                for f in hist:
                    if f in obs.index:
                        row=obs.loc[f]
                        if isinstance(row,pd.DataFrame):raise ValueError('Duplicate frame in segment')
                        lookups.append((int(f),str(row.oracle_GT_id),True))
                    else:lookups.append((int(f),'',False))
                matched=[x[1] for x in lookups if x[1]]
                votes=Counter(matched)
                # Reject ties: no arbitrary identity assignment.
                order=votes.most_common()
                tied=len(order)>1 and order[0][1]==order[1][1]
                major=order[0][0] if order and not tied else ''
                same=sum(x[1]==major for x in lookups) if major else 0
                conflicting=sum(bool(x[1]) and x[1]!=major for x in lookups) if major else 0
                missing=sum(not x[2] for x in lookups)
                no_iou=sum(x[2] and not x[1] for x in lookups)
                gt_future=sum(major in gtids.get((seq,int(f)),set()) for f in fut) if major else 0
                # Predeclared diagnostic gates, not TestReal-tuned selection.
                strict=bool(major and same==20 and gt_future==n_future)
                permissive=bool(major and same>=16 and conflicting==0 and gt_future==n_future)
                windows.append({'model':model,'sequence':seq,'track_id':tid,'segment':int(seg.segment),
                    'horizon':horizon,'window_start_30hz':int(sampled[k]),
                    'anchor_frame_30hz':int(hist[-1]),'future_end_30hz':int(fut[-1]),
                    'GT_id_ORACLE_DIAGNOSTIC':major,'observed_history_with_GTIoU50':int(same),
                    'history_other_GT_IoU50':int(conflicting),
                    'history_missing_tracker_output':int(missing),'history_unmatched_tracker_output':int(no_iou),
                    'GT_future_available_steps':int(gt_future),'GT_future_required_steps':n_future,
                    'gt_linked_strict_20of20':strict,'gt_linked_80pct_no_conflict':permissive,
                    'oracle_GT_used_only_for_audit':True})
    win=pd.DataFrame(windows)
    if len(win)==0:raise RuntimeError('No stage7D candidate windows; cannot audit')
    win.to_csv(out/'Stage7E_All_Tracked_Window_GT_Linkage.csv',index=False)
    pd.DataFrame(segment_details).to_csv(out/'Stage7E_Tracker_Segment_GT_Purity.csv',index=False)
    by_model=win.groupby(['model','horizon']).agg(
      potential=('horizon','size'),GT_linked_strict=('gt_linked_strict_20of20','sum'),
      GT_linked_80pct=('gt_linked_80pct_no_conflict','sum'),
      candidates_with_any_GT=('GT_id_ORACLE_DIAGNOSTIC',lambda c:int((c!='').sum())),
    ).reset_index()
    win['GT_future_complete']=win.GT_future_available_steps==win.GT_future_required_steps
    complete=win.groupby(['model','horizon'])['GT_future_complete'].sum().reset_index(name='future_GT_complete')
    by_model=by_model.merge(complete,on=['model','horizon'])
    arms=pd.MultiIndex.from_product([['original','adapted'],['2s','5s']],names=['model','horizon']).to_frame(index=False)
    by_model=arms.merge(by_model,on=['model','horizon'],how='left').fillna(0)
    for col in ['potential','GT_linked_strict','GT_linked_80pct','candidates_with_any_GT','future_GT_complete']:
        by_model[col]=by_model[col].astype(int)
    by_model['strict_fraction_of_potential']=np.divide(by_model.GT_linked_strict,by_model.potential,out=np.zeros(len(by_model)),where=by_model.potential>0)
    by_model['gate80_fraction_of_potential']=np.divide(by_model.GT_linked_80pct,by_model.potential,out=np.zeros(len(by_model)),where=by_model.potential>0)
    by_model.to_csv(out/'Stage7E_GT_Linked_Window_Summary.csv',index=False)
    group=win.groupby(['model','horizon','sequence']).agg(
        potential=('horizon','size'),strict=('gt_linked_strict_20of20','sum'),
        gate80=('gt_linked_80pct_no_conflict','sum'),future_GT_complete=('GT_future_complete','sum')).reset_index()
    group.to_csv(out/'Stage7E_PerSequence_GT_Linked_Windows.csv',index=False)
    frame_stats=tracks.groupby(['model','sequence']).agg(
        track_boxes=('frame','size'),gt_matched_boxes=('oracle_matched','sum'),
        distinct_tracker_ids=('track_id','nunique')).reset_index()
    frame_stats['GT_matched_fraction_of_track_boxes']=frame_stats.gt_matched_boxes/frame_stats.track_boxes
    frame_stats.to_csv(out/'Stage7E_PerSequence_Frame_GTAudit.csv',index=False)
    counts=win.groupby(['model','horizon']).size().to_dict()
    expected={('original','2s'):2,('original','5s'):0,('adapted','2s'):371,('adapted','5s'):191}
    if any(counts.get(k,0)!=v for k,v in expected.items()):
        raise ValueError(f'Window parity failure: {counts}; expected {expected}')
    status={'status':'GT_LINKAGE_AUDIT_COMPLETE','stage7d_zip_sha256':sha256(zp),
      'source_gt_sha256':sha256(gp),'source_gt_rows':len(gt),'source_tracker_rows':len(tracks),
      'frame_groups_with_tracker_boxes':n_frames,'source_sequences':SEQS,
      'matching_rule':'per-frame one-to-one Hungarian IoU >=0.50; GT identity used only for offline audit',
      'strict_gate':'20/20 10-Hz history tracker boxes matched same GT id, future GT all present',
      'diagnostic_gate80':'at least 16/20 10-Hz history boxes matched same GT id, zero conflict, full future GT',
      'oracle_labels_not_deployable':True,'actual_transformer_predictions_run':False,
      'not_independent_holdout':True,'sampling':'Segment start frame plus 3-frame 30Hz steps; stride=10 samples',
      'selection_note':'strict and diagnostic 80% gates are preregistered in code and not tuned on TestReal'}
    (out/'Stage7E_Provenance.json').write_text(json.dumps(status,indent=2)+'\n')
    header=['Stage 7E — GT-linked tracker-window audit (retrospective, annotation-assisted diagnostic)',
       'Training: NONE. New YOLO/ByteTrack/Transformer inference: NONE.',
       'GT identities used ONLY to verify candidate tracks; they are not available to deployed inference.',
       'Stage7D raw window totals include unmatched/false tracks and are NOT actual forecasting sample counts.',
       '\n'+by_model.to_string(index=False),
       '\nSource-level:\n'+group.to_string(index=False)]
    (out/'Stage7E_Readout.txt').write_text('\n'.join(header)+'\n')
    print('\n'.join(header),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage7d',required=True)
    ap.add_argument('--gt',default=str(Path(BASE)/'Main Result/trajectory_dataset/cleaned_track_points.csv'))
    ap.add_argument('--out',required=True)
    ap.add_argument('--allow-partial-gt',action='store_true',help='FOR LOCAL SOFTWARE TESTS ONLY; never for paper metrics')
    args=ap.parse_args();out=Path(args.out).expanduser();out.mkdir(parents=True,exist_ok=True)
    try:run(args)
    except Exception:
        err=traceback.format_exc();(out/'Stage7E_Failure.txt').write_text(err)
        print(err,file=sys.stderr);sys.exit(2)

if __name__=='__main__':main()
