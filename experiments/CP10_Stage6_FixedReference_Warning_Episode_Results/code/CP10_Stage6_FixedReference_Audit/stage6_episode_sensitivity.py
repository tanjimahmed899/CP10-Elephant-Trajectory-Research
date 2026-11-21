#!/usr/bin/env python3
"""Exploratory fixed-segment-reference episode matching and depth sensitivity.

All boundaries are synthetic fixed ray-planes of FIRST camera per segment, not
surveyed geofences; all times are simulated from assumed 10 Hz sampling.
Do not use this for a claim of operational lead-time or risk control.
"""
import importlib.util, sys, io, zipfile, json
from pathlib import Path
import numpy as np, pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage6geometry',HERE/'stage6_fixed_reference_study.py')
a=importlib.util.module_from_spec(spec);sys.modules['stage6geometry']=a;spec.loader.exec_module(a)

OUT=HERE; BOUND=a.BOUND

def crossing_events(x,frames,j,b,k=3):
    """List frame-indexed crossings stable over k frames before/after."""
    x=np.asarray(x);f=np.asarray(frames,dtype=int); pos=(x[:,j]-b)
    event=[]
    for i in range(k, len(pos)-k+1):
        if f[i]!=f[i-1]+1:continue
        before=pos[i-k:i]; after=pos[i:i+k]
        if (f[i]-f[i-k]==k and f[i+k-1]-f[i]==k-1 and
            (np.all(before<0) and np.all(after>0) or np.all(before>0) and np.all(after<0))):
            direction='high' if before[0]<0 else 'low'
            event.append((int(f[i]),direction))
    return event

# Build comprehensive per-track GT reference trajectory without future use in model projection.
reftraj={}
for (video,segment),d in a.metadata.items():
    Kref,Rref,tref,W,H=d['cam'][0]
    for tid,rec in d['tracks'].items():
        fs=rec['frames'];xyz=rec['xyz'];valid=np.array([int(f) in d['idx'] for f in fs],bool)
        fs=fs[valid];xyz=xyz[valid]
        if not len(fs):continue
        worlds=[]
        for f,x in zip(fs,xyz):
           _,Rc,tc,_,_=d['cam'][d['idx'][int(f)]]
           worlds.append((x-tc)@Rc)
        xy=a.project(np.asarray(worlds),Kref,Rref,tref,W,H)
        reftraj[(video,segment,str(tid))]={'frames':fs,'xy':xy,'world':np.asarray(worlds)}

allcases=pd.read_csv(HERE/'Stage6_FixedReference_Projected_Warning_Cases.csv')
activekeys=set((str(row.video),str(row.segment),str(row.track_id)) for row in allcases[['video','segment','track_id']].drop_duplicates().itertuples(index=False))
# Episode IDs and counts for K=1 and K=3, no post-hoc choice of result.
all_events=[]
for key in sorted(activekeys):
    tr=reftraj.get(key)
    if not tr:continue
    for axis,j,b in BOUND:
        for k in [1,3]:
            events=crossing_events(tr['xy'],tr['frames'],j,b,k)
            for ft,dirn in events:
                all_events.append({'video':key[0],'segment':key[1],'track_id':key[2],
                    'axis':axis,'boundary':b,'direction':dirn,'cross_frame':ft,'persistence_k':k})
events=pd.DataFrame(all_events)
events.to_csv(OUT/'Stage6_FixedReference_TrueCrossing_Episodes_K1_K3.csv',index=False)

# Stage5 separate adaptation and retrospective evaluation
policies=['nominal','fixed_N1','fixed_N2_gapaware','fixed_N3_gapaware']
metrics=[];matched=[];alert_runs=[]
for (source,h,role),df in allcases.groupby(['source','horizon','role'],sort=False):
    e0=events.merge(df[['video','segment','track_id']].drop_duplicates().assign(track_id=lambda q:q.track_id.astype(str)),on=['video','segment','track_id'],how='inner')
    subdf=df.copy();subdf['track_id']=subdf['track_id'].astype(str)
    # only events for this dataset track-source, K levels; at least 1 opportunity within horizon
    horizon=int(h[0]) if h=='2s' else 5
    fut_len=20 if h=='2s' else 50
    keys=['video','segment','track_id','axis','boundary','direction']
    keyed={idx:g for idx,g in subdf.groupby(keys,sort=False)}
    # Unique track observation lengths (reconstructed full annotated track), track-hour denominator
    unique_keys=set((str(row.video),str(row.segment),str(row.track_id)) for row in subdf[['video','segment','track_id']].drop_duplicates().itertuples(index=False))
    track_secs=sum(len(reftraj[x]['frames'])/10 for x in unique_keys if x in reftraj)
    for k in [1,3]:
      ee=e0[e0.persistence_k==k].copy()
      eligible=[]
      for row in ee.itertuples(index=False):
        key=(row.video,row.segment,str(row.track_id),row.axis,row.boundary,row.direction)
        g=keyed.get(key)
        if g is None:continue
        available=g[(g.observation_end<row.cross_frame) & (g.observation_end+fut_len>=row.cross_frame)]
        if len(available)==0: continue
        eligible.append((row,available))
      for pol in policies:
        hit=0;lead=[]
        for ev,avail in eligible:
            hits=avail[avail[pol].astype(bool)]
            detected=not hits.empty
            # lead from FIRST policy hit among candidates forecasting this true crossing
            if detected:
                hit+=1
                first_obs=int(hits.observation_end.min())
                sec=(ev.cross_frame-first_obs)/10
                lead.append(sec)
            matched.append({'source':source,'horizon':h,'role':role,'persistence_k':k,'policy':pol,
                  'video':ev.video,'segment':ev.segment,'track_id':ev.track_id,
                  'axis':ev.axis,'boundary':ev.boundary,'direction':ev.direction,'cross_frame':ev.cross_frame,
                  'n_alert_opportunities':len(avail),'detected_before_crossing':detected,
                  'lead_seconds_to_annotated_crossing':sec if detected else None})
        # Deduplicate algorithmic consecutive alerts, distinct from event matching
        runs=[]
        for key,g in keyed.items():
            g=g.sort_values('anchor');alarm=g[pol].to_numpy(bool)
            frames=g.anchor.to_numpy(int)
            inds=np.flatnonzero(alarm)
            if len(inds)==0:continue
            start=prev=inds[0]
            for index in np.r_[inds[1:], -1]:
                if index>=0 and index==prev+1 and frames[index]-frames[prev]==10:
                    prev=index;continue
                # Interval from earliest observed decision to latest horizon evaluated
                gg=g.iloc[start:prev+1]
                onset=int(gg.observation_end.min());end=int(gg.observation_end.max()+fut_len)
                kidx=(gg.video.iloc[0],gg.segment.iloc[0],str(gg.track_id.iloc[0]),gg.axis.iloc[0],gg.boundary.iloc[0],gg.direction.iloc[0])
                ev=e0[(e0.persistence_k==k)&(e0.video==kidx[0])&(e0.segment==kidx[1])&
                     (e0.track_id.astype(str)==kidx[2])&(e0.axis==kidx[3])&
                     (e0.boundary==kidx[4])&(e0.direction==kidx[5])]
                # True event must happen AFTER first alert and BEFORE last prediction horizon expires
                ev=ev[(ev.cross_frame>onset)&(ev.cross_frame<=end)]
                istrue=len(ev)>0
                runs.append(istrue)
                alert_runs.append({'source':source,'horizon':h,'role':role,'policy':pol,'persistence_k':k,
                    'video':kidx[0],'segment':kidx[1],'track_id':kidx[2],'axis':kidx[3],
                    'boundary':kidx[4],'direction':kidx[5], 'start_observation_frame':onset,
                    'end_prediction_frame':end,'is_event_associated':istrue, 'n_alert_windows':int(prev-start+1)})
                if index<0:break
                start=prev=index
        false_runs=sum(not z for z in runs)
        metrics.append({'source':source,'horizon':h,'role':role,'persistence_k':k,'policy':pol,
           'eligible_annotated_crossing_events':len(eligible),'events_detected':hit,
           'event_recall':hit/len(eligible) if len(eligible) else np.nan,
           'median_annotated_crossing_lead_seconds':np.median(lead) if lead else np.nan,
           'n_alert_runs':len(runs),'event_unassociated_alert_runs':false_runs,
           'event_unassociated_runs_per_track_hour':false_runs/track_secs*3600 if track_secs>0 else np.nan,
           'track_hours':track_secs/3600,'n_tracks':len(unique_keys)})
met=pd.DataFrame(metrics);met.to_csv(OUT/'Stage6_FixedReference_Exploratory_Episode_Metrics.csv',index=False)
pd.DataFrame(matched).to_csv(OUT/'Stage6_FixedReference_Event_Matching_Detail.csv',index=False)
pd.DataFrame(alert_runs).to_csv(OUT/'Stage6_FixedReference_Alert_Episode_Detail.csv',index=False)

# Depth sensitivity using ground-truth future depth ONLY IN DIAGNOSTIC ARM.
# We do not use future depth to generate any deployable prediction.
flips=[]
for source,p in a.SOURCES:
 for h in ['2s','5s']:
   meta,pr,gt,hist,scale=a.load(source,p,h)
   for i,r in meta.iterrows():
     d=a.metadata[(str(r.video),str(r.segment))]
     track=d['tracks'][str(r.track_id)];pos=track['find'][int(r.anchor_frame_index)]
     fs=track['frames'][pos+np.arange(20+len(pr[i]))]; xx=track['xyz'][pos+np.arange(20+len(pr[i]))]
     Ka,Ra,ta,W,H=d['cam'][d['idx'][int(r.anchor_frame_index)]]
     Kr,Rr,tr,_,_=d['cam'][0]
     coords=[]
     for f,x in zip(fs,xx):
       _,R,t,_,_=d['cam'][d['idx'][int(f)]]
       coords.append((x-t)@R)
     coords=np.array(coords)
     z_all=(coords@Ra.T+ta)[:,2]
     z_last=z_all[19]
     truefuture=pr[i]
     const=a.pred_to_ref(truefuture,z_last,Ka,Ra,ta,Kr,Rr,tr,W,H)
     # This uses TRUE future depth -- retrospective diagnostic, not feasible at inference
     uv=truefuture*np.array([W,H]); ray=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(Ka).T
     cam=ray*z_all[20:,None];world=(cam-ta)@Ra
     oracle=a.project(world,Kr,Rr,tr,W,H)
     truthref=a.project(coords,Kr,Rr,tr,W,H)
     for ax,j,b in BOUND:
       s=1 if truthref[19,j]<b else -1
       nom1=bool(np.max((const[:,j]-b)*s)>=0)
       nom2=bool(np.max((oracle[:,j]-b)*s)>=0)
       tru=bool(np.max((truthref[20:,j]-b)*s)>=0)
       # test crossing margin: distance between GT future points and boundary
       flips.append({'source':source,'horizon':h,'role':r.get('analysis_role','EXP3_six_video'),
                     'video':r.video,'segment':r.segment,'track_id':r.track_id,'sample_index':i,
                     'axis':ax,'boundary':b,'nominal_const_depth':nom1,'nominal_oracle_future_depth':nom2,
                     'nominal_depth_flip':nom1!=nom2,'true_window_cross':tru,
                     'max_future_depth_rel_change':np.max(np.abs(z_all[20:]/z_last-1))})
flipdf=pd.DataFrame(flips);flipdf.to_csv(OUT/'Stage6_FixedReference_OracleDepth_Sensitivity_DIAGNOSTIC_ONLY.csv',index=False)
sumf=flipdf.groupby(['source','horizon','role'],sort=False).agg(n_boundary_cases=('nominal_depth_flip','size'),
            n_depth_assumption_flips=('nominal_depth_flip','sum'),fraction_depth_assumption_flips=('nominal_depth_flip','mean'),
            p90_depth_relative_change=('max_future_depth_rel_change',lambda x:x.quantile(.9))).reset_index()
sumf.to_csv(OUT/'Stage6_FixedReference_Depth_Sensitivity_Summary.csv',index=False)

# Visuals
sel=met[(met.persistence_k==3)&(met.role.ne('adaptation'))]
fig,ax=plt.subplots(figsize=(10.5,5.5),dpi=175)
rows=[]
for (src,h,role),g in sel.groupby(['source','horizon','role']):
    for _,r in g.iterrows():
        rows.append({'set':src+' '+h,'policy':r.policy,'recall':r.event_recall,'false':r.event_unassociated_alert_runs})
z=pd.DataFrame(rows)
series=['nominal','fixed_N1','fixed_N2_gapaware','fixed_N3_gapaware']
x=np.arange(len(z['set'].unique()));w=0.19
for j,pol in enumerate(series):
    v=z[z.policy==pol].set_index('set').reindex(z['set'].unique())
    ax.bar(x+(j-1.5)*w, v.recall.to_numpy()*100,w,label=pol.replace('fixed_',''))
ax.set_xticks(x,z['set'].unique());ax.set_ylim(0,106);ax.set_ylabel('Eligible crossing episodes detected (%)')
ax.set_title('Fixed-reference conditional geometric simulation — stable GT crossing episodes (K=3)')
ax.legend(fontsize=8,loc='upper center',ncol=4)
fig.text(.01,.01,'NOT a surveyed fence. Predicted 2-D paths transported using last-observed depth; 10 Hz assumed.',fontsize=8)
fig.tight_layout(rect=[0,.045,1,1]);fig.savefig(OUT/'Figure_Stage6_Conditional_FixedReference_EventRecall.png');plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5),dpi=175)
for (s,h),g in flipdf.groupby(['source','horizon']):
    ax.bar(f'{s} {h}',g.nominal_depth_flip.mean()*100)
ax.set_ylabel('Nominal warning decisions changed by oracle-depth projection (%)')
ax.set_title('Depth sensitivity of fixed-reference prediction registration')
fig.tight_layout();fig.savefig(OUT/'Figure_Stage6_Depth_Assumption_Sensitivity.png');plt.close(fig)

print('\n--- Fixed-reference exploratory event metrics, K3 ---')
print(met[(met.persistence_k==3)&(met.role.ne('adaptation'))].to_string(index=False))
print('\n--- Future-depth sensitivity (DIAGNOSTIC ONLY) ---')
print(sumf.to_string(index=False))
print('Generated',OUT)
