
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HOME=Path.home(); DOWNLOADS=HOME/"Downloads"
MAIN1=Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
REVISION=DOWNLOADS/"paper_strengthening_revision"; R13_ROOT=REVISION/"R13_end_to_end_pipeline"
R13C_DIR=R13_ROOT/"R13C_frozen_end_to_end_protocol_lock"
R13D_DIR=R13_ROOT/"R13D_frozen_detector_tracker_execution"
R13C_PROTOCOL=R13C_DIR/"R13C_FROZEN_PROTOCOL.json"
R13D_MANIFEST=R13D_DIR/"R13D_MANIFEST.json"
BYTE_BOXES=R13D_DIR/"R13D_BYTETRACK_ALL_BOXES.csv"; BYTE_FRAMES=R13D_DIR/"R13D_BYTETRACK_ALL_FRAMES.csv"
BOT_BOXES=R13D_DIR/"R13D_BOTSORT_ALL_BOXES.csv"; BOT_FRAMES=R13D_DIR/"R13D_BOTSORT_ALL_FRAMES.csv"
CLEANED=MAIN1/"trajectory_dataset"/"cleaned_track_points.csv"
OUT=R13_ROOT/"R13E_tracking_metrics"; OUT.mkdir(parents=True,exist_ok=True)
ALPHAS=np.arange(0.05,0.99,0.05); CLEAR_THRESHOLD=0.5; IDENTITY_THRESHOLD=0.5; EPS=np.finfo(float).eps

def utc_now(): return datetime.now(timezone.utc).isoformat()
def require(p):
    if not Path(p).exists(): raise FileNotFoundError(p)
def sha256_file(path,chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()
def canon_id(v):
    if pd.isna(v): return ""
    s=str(v).strip()
    try:
        x=float(s)
        if np.isfinite(x) and x.is_integer(): return str(int(x))
    except: pass
    return s

def verify_inputs():
    for p in [R13C_PROTOCOL,R13D_MANIFEST,BYTE_BOXES,BYTE_FRAMES,BOT_BOXES,BOT_FRAMES,CLEANED]: require(p)
    protocol=json.loads(R13C_PROTOCOL.read_text(encoding="utf-8"))
    manifest=json.loads(R13D_MANIFEST.read_text(encoding="utf-8"))
    expected="R13D_FROZEN_DETECTOR_BYTETRACK_BOTSORT_EXECUTION_COMPLETE_RAW_OUTPUTS_LOCKED_READY_FOR_R13E_TRACKING_METRICS"
    if manifest.get("status")!=expected: raise RuntimeError(f"R13-D not COMPLETE: {manifest.get('status')}")
    sm={x["component"]:x for x in manifest["summary"]}
    for comp,path,key in [
        ("bytetrack",BYTE_BOXES,"boxes_sha256"),("bytetrack",BYTE_FRAMES,"frames_sha256"),
        ("botsort",BOT_BOXES,"boxes_sha256"),("botsort",BOT_FRAMES,"frames_sha256")]:
        if sha256_file(path)!=sm[comp][key]: raise RuntimeError(f"Frozen hash mismatch: {path}")
    return protocol,manifest

def xywh_to_xyxy(df):
    x=pd.to_numeric(df["x"],errors="coerce").to_numpy(float); y=pd.to_numeric(df["y"],errors="coerce").to_numpy(float)
    w=pd.to_numeric(df["w"],errors="coerce").to_numpy(float); h=pd.to_numeric(df["h"],errors="coerce").to_numpy(float)
    return np.column_stack([x,y,x+w,y+h])
def pred_xyxy(df):
    return df[["x1","y1","x2","y2"]].apply(pd.to_numeric,errors="coerce").to_numpy(float)
def iou_matrix(a,b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)),float)
    a=np.asarray(a,float); b=np.asarray(b,float)
    ix1=np.maximum(a[:,None,0],b[None,:,0]); iy1=np.maximum(a[:,None,1],b[None,:,1])
    ix2=np.minimum(a[:,None,2],b[None,:,2]); iy2=np.minimum(a[:,None,3],b[None,:,3])
    iw=np.maximum(0,ix2-ix1); ih=np.maximum(0,iy2-iy1); inter=iw*ih
    aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1])
    bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1])
    union=aa[:,None]+bb[None,:]-inter
    out=np.zeros_like(inter); m=union>EPS; out[m]=inter[m]/union[m]
    return out

def load_gt(protocol):
    df=pd.read_csv(CLEANED,low_memory=False)
    s=df["split"].astype(str).str.strip().str.lower()
    m=s.isin(["test_real","testreal","test-real"])
    if not m.any(): m=s.str.contains("test",regex=False)
    df=df.loc[m].copy()
    seqs=set(protocol["data"]["sequences"]); df["sequence"]=df["sequence"].astype(str).str.strip()
    df=df[df["sequence"].isin(seqs)].copy()
    if "interpolated" in df.columns:
        interp=pd.to_numeric(df["interpolated"],errors="coerce").fillna(0)
        df=df.loc[interp==0].copy()
    df["frame"]=pd.to_numeric(df["frame"],errors="raise").astype(int)
    df["gt_identity"]=df["sequence"]+"|"+df["object_id"].map(canon_id)
    return df

def resolve_prediction_class(pred,tracker_name):
    p=pred.copy(); p["track_id"]=pd.to_numeric(p["track_id"],errors="coerce")
    p=p.loc[p["track_id"].notna() & (p["track_id"]>=0)].copy()
    if len(p)==0:
        return p,{"tracker":tracker_name,"decision":"no_valid_track_rows","classes_seen":""}
    classes=sorted(p["class_name"].astype(str).str.strip().unique().tolist())
    if len(classes)==1:
        selected=classes[0]; decision="single_class_use_all"; out=p.copy()
    else:
        elephant=[c for c in classes if "elephant" in c.lower()]
        if len(elephant)!=1: raise RuntimeError(f"{tracker_name}: ambiguous classes {classes}")
        selected=elephant[0]; decision="multi_class_filter_unique_elephant"
        out=p.loc[p["class_name"].astype(str)==selected].copy()
    return out,{"tracker":tracker_name,"decision":decision,"selected_class":selected,
                "classes_seen":"|".join(classes),"valid_track_rows_before_class_filter":len(p),
                "rows_after_class_filter":len(out)}

def build_sequence_data(sequence,gt_all,pred_all,frames_all):
    gt=gt_all.loc[gt_all["sequence"]==sequence].copy()
    pred=pred_all.loc[pred_all["sequence"]==sequence].copy()
    fr=frames_all.loc[frames_all["sequence"]==sequence].copy()
    all_frames=sorted(pd.to_numeric(fr["frame"],errors="raise").astype(int).unique().tolist())
    gu=sorted(gt["gt_identity"].astype(str).unique().tolist())
    pu=sorted(pd.to_numeric(pred["track_id"],errors="raise").astype(int).unique().tolist())
    gm={k:i for i,k in enumerate(gu)}; pm={k:i for i,k in enumerate(pu)}
    gids=[]; tids=[]; sims=[]
    pred_frame_num=pd.to_numeric(pred["frame"],errors="coerce")
    for frame in all_frames:
        g=gt.loc[gt["frame"]==frame].copy()
        p=pred.loc[pred_frame_num==frame].copy()
        gi=np.asarray([gm[x] for x in g["gt_identity"].astype(str)],int)
        ti=np.asarray([pm[int(x)] for x in pd.to_numeric(p["track_id"],errors="raise").astype(int)],int)
        gids.append(gi); tids.append(ti); sims.append(iou_matrix(xywh_to_xyxy(g),pred_xyxy(p)))
    return {"sequence":sequence,"frames":all_frames,"gt_ids":gids,"tracker_ids":tids,"similarity_scores":sims,
            "num_timesteps":len(all_frames),"num_gt_ids":len(gu),"num_tracker_ids":len(pu),
            "num_gt_dets":sum(map(len,gids)),"num_tracker_dets":sum(map(len,tids))}

def hota_finalize(r):
    r["DetRe"]=r["HOTA_TP"]/np.maximum(1,r["HOTA_TP"]+r["HOTA_FN"])
    r["DetPr"]=r["HOTA_TP"]/np.maximum(1,r["HOTA_TP"]+r["HOTA_FP"])
    r["DetA"]=r["HOTA_TP"]/np.maximum(1,r["HOTA_TP"]+r["HOTA_FN"]+r["HOTA_FP"])
    r["HOTA"]=np.sqrt(r["DetA"]*r["AssA"])
    return r

def hota_eval(data):
    n=len(ALPHAS)
    r={k:np.zeros(n,float) for k in ["HOTA_TP","HOTA_FN","HOTA_FP","AssA","AssRe","AssPr","LocA"]}
    if data["num_tracker_dets"]==0:
        r["HOTA_FN"][:]=data["num_gt_dets"]; r["LocA"][:]=1; return hota_finalize(r)
    if data["num_gt_dets"]==0:
        r["HOTA_FP"][:]=data["num_tracker_dets"]; r["LocA"][:]=1; return hota_finalize(r)
    pot=np.zeros((data["num_gt_ids"],data["num_tracker_ids"]),float)
    gc=np.zeros((data["num_gt_ids"],1),float); tc=np.zeros((1,data["num_tracker_ids"]),float)
    for gi,ti,sim in zip(data["gt_ids"],data["tracker_ids"],data["similarity_scores"]):
        if len(gi) and len(ti):
            den=sim.sum(0)[None,:]+sim.sum(1)[:,None]-sim
            si=np.zeros_like(sim); mask=den>EPS; si[mask]=sim[mask]/den[mask]
            pot[gi[:,None],ti[None,:]] += si
        if len(gi): gc[gi]+=1
        if len(ti): tc[0,ti]+=1
    den=gc+tc-pot; ga=np.divide(pot,den,out=np.zeros_like(pot),where=den>EPS)
    mcounts=[np.zeros_like(pot) for _ in ALPHAS]
    for gi,ti,sim in zip(data["gt_ids"],data["tracker_ids"],data["similarity_scores"]):
        if len(gi)==0:
            r["HOTA_FP"]+=len(ti); continue
        if len(ti)==0:
            r["HOTA_FN"]+=len(gi); continue
        score=ga[gi[:,None],ti[None,:]]*sim
        rr,cc=linear_sum_assignment(-score); ms=sim[rr,cc]
        for a,alpha in enumerate(ALPHAS):
            ok=ms>=alpha-EPS; mr=rr[ok]; mc=cc[ok]; nm=len(mr)
            r["HOTA_TP"][a]+=nm; r["HOTA_FN"][a]+=len(gi)-nm; r["HOTA_FP"][a]+=len(ti)-nm
            if nm:
                r["LocA"][a]+=sim[mr,mc].sum()
                mcounts[a][gi[mr],ti[mc]]+=1
    for a in range(n):
        mc=mcounts[a]
        aa=mc/np.maximum(1,gc+tc-mc); r["AssA"][a]=np.sum(mc*aa)/max(1,r["HOTA_TP"][a])
        ar=mc/np.maximum(1,gc); r["AssRe"][a]=np.sum(mc*ar)/max(1,r["HOTA_TP"][a])
        ap=mc/np.maximum(1,tc); r["AssPr"][a]=np.sum(mc*ap)/max(1,r["HOTA_TP"][a])
    r["LocA"]=np.maximum(1e-10,r["LocA"])/np.maximum(1e-10,r["HOTA_TP"])
    return hota_finalize(r)

def hota_combine(allr):
    c={f:sum(r[f] for r in allr) for f in ["HOTA_TP","HOTA_FN","HOTA_FP"]}
    for f in ["AssRe","AssPr","AssA"]:
        c[f]=sum(r[f]*r["HOTA_TP"] for r in allr)/np.maximum(1e-10,c["HOTA_TP"])
    c["LocA"]=np.maximum(1e-10,sum(r["LocA"]*r["HOTA_TP"] for r in allr))/np.maximum(1e-10,c["HOTA_TP"])
    return hota_finalize(c)

def identity_finalize(r):
    tp,fn,fp=map(float,[r["IDTP"],r["IDFN"],r["IDFP"]])
    r["IDR"]=tp/max(1,tp+fn); r["IDP"]=tp/max(1,tp+fp); r["IDF1"]=tp/max(1,tp+.5*fp+.5*fn)
    return r
def identity_eval(data):
    r={"IDTP":0,"IDFN":0,"IDFP":0}
    if data["num_tracker_dets"]==0: r["IDFN"]=data["num_gt_dets"]; return identity_finalize(r)
    if data["num_gt_dets"]==0: r["IDFP"]=data["num_tracker_dets"]; return identity_finalize(r)
    ng,nt=data["num_gt_ids"],data["num_tracker_ids"]
    pot=np.zeros((ng,nt)); gc=np.zeros(ng); tc=np.zeros(nt)
    for gi,ti,sim in zip(data["gt_ids"],data["tracker_ids"],data["similarity_scores"]):
        if len(gi) and len(ti):
            mi,mj=np.nonzero(sim>=IDENTITY_THRESHOLD); pot[gi[mi],ti[mj]]+=1
        if len(gi): gc[gi]+=1
        if len(ti): tc[ti]+=1
    fp=np.zeros((ng+nt,ng+nt)); fn=np.zeros_like(fp)
    fp[ng:,:nt]=1e10; fn[:ng,nt:]=1e10
    for g in range(ng): fn[g,:nt]=gc[g]; fn[g,nt+g]=gc[g]
    for t in range(nt): fp[:ng,t]=tc[t]; fp[ng+t,t]=tc[t]
    fn[:ng,:nt]-=pot; fp[:ng,:nt]-=pot
    rr,cc=linear_sum_assignment(fn+fp)
    r["IDFN"]=int(round(fn[rr,cc].sum())); r["IDFP"]=int(round(fp[rr,cc].sum()))
    r["IDTP"]=int(round(gc.sum()-r["IDFN"]))
    return identity_finalize(r)
def identity_combine(allr):
    return identity_finalize({k:sum(r[k] for r in allr) for k in ["IDTP","IDFN","IDFP"]})

def clear_finalize(r):
    gt=r["CLR_TP"]+r["CLR_FN"]
    r["CLR_Re"]=r["CLR_TP"]/max(1,gt); r["CLR_Pr"]=r["CLR_TP"]/max(1,r["CLR_TP"]+r["CLR_FP"])
    r["MOTA"]=(r["CLR_TP"]-r["CLR_FP"]-r["IDSW"])/max(1,gt)
    r["MOTP"]=r["MOTP_sum"]/max(1,r["CLR_TP"])
    r["CLR_F1"]=r["CLR_TP"]/max(1,r["CLR_TP"]+.5*r["CLR_FN"]+.5*r["CLR_FP"])
    return r
def clear_eval(data):
    ng=data["num_gt_ids"]
    r={"CLR_TP":0,"CLR_FN":0,"CLR_FP":0,"IDSW":0,"MT":0,"PT":0,"ML":0,"Frag":0,
       "CLR_Frames":data["num_timesteps"],"MOTP_sum":0.0}
    if data["num_tracker_dets"]==0: r["CLR_FN"]=data["num_gt_dets"]; r["ML"]=ng; return clear_finalize(r)
    if data["num_gt_dets"]==0: r["CLR_FP"]=data["num_tracker_dets"]; return clear_finalize(r)
    gc=np.zeros(ng); gmc=np.zeros(ng); gfrag=np.zeros(ng)
    prev=np.nan*np.zeros(ng); prev_t=np.nan*np.zeros(ng)
    for gi,ti,sim in zip(data["gt_ids"],data["tracker_ids"],data["similarity_scores"]):
        if len(gi)==0:
            r["CLR_FP"]+=len(ti); prev_t[:]=np.nan; continue
        if len(ti)==0:
            r["CLR_FN"]+=len(gi); gc[gi]+=1; prev_t[:]=np.nan; continue
        cont=(ti[None,:]==prev_t[gi[:,None]])
        score=1000.0*cont+sim; score[sim<CLEAR_THRESHOLD-EPS]=0
        rr,cc=linear_sum_assignment(-score); ok=score[rr,cc]>EPS; rr=rr[ok]; cc=cc[ok]
        mg=gi[rr]; mt=ti[cc]; pm=prev[mg]
        r["IDSW"]+=int(((~np.isnan(pm)) & (mt!=pm)).sum())
        gc[gi]+=1; gmc[mg]+=1
        notprev=np.isnan(prev_t); prev[mg]=mt; prev_t[:]=np.nan; prev_t[mg]=mt
        gfrag += notprev & (~np.isnan(prev_t))
        nm=len(mg); r["CLR_TP"]+=nm; r["CLR_FN"]+=len(gi)-nm; r["CLR_FP"]+=len(ti)-nm
        if nm: r["MOTP_sum"]+=float(sim[rr,cc].sum())
    valid=gc>0; ratio=np.zeros(ng); ratio[valid]=gmc[valid]/gc[valid]
    r["MT"]=int(np.sum(ratio[valid]>0.8)); r["PT"]=int(np.sum(ratio[valid]>=0.2)-r["MT"]); r["ML"]=int(valid.sum()-r["MT"]-r["PT"])
    gp=gfrag[gfrag>0]; r["Frag"]=int(np.sum(gp-1))
    return clear_finalize(r)
def clear_combine(allr):
    fs=["CLR_TP","CLR_FN","CLR_FP","IDSW","MT","PT","ML","Frag","CLR_Frames","MOTP_sum"]
    return clear_finalize({f:sum(r[f] for r in allr) for f in fs})

def summary_row(tracker,sequence,h,i,c,data):
    idx=int(np.argmin(np.abs(ALPHAS-0.5)))
    return {"tracker":tracker,"sequence":sequence,"num_frames":data["num_timesteps"],"num_gt_ids":data["num_gt_ids"],
            "num_tracker_ids":data["num_tracker_ids"],"num_gt_dets":data["num_gt_dets"],"num_tracker_dets":data["num_tracker_dets"],
            "HOTA":float(np.mean(h["HOTA"])),"DetA":float(np.mean(h["DetA"])),"AssA":float(np.mean(h["AssA"])),
            "DetRe":float(np.mean(h["DetRe"])),"DetPr":float(np.mean(h["DetPr"])),"LocA":float(np.mean(h["LocA"])),
            "HOTA_at_0_50":float(h["HOTA"][idx]),"DetA_at_0_50":float(h["DetA"][idx]),"AssA_at_0_50":float(h["AssA"][idx]),
            "IDF1":float(i["IDF1"]),"IDP":float(i["IDP"]),"IDR":float(i["IDR"]),"IDTP":int(i["IDTP"]),"IDFP":int(i["IDFP"]),"IDFN":int(i["IDFN"]),
            "IDSW":int(c["IDSW"]),"Frag":int(c["Frag"]),"MOTA":float(c["MOTA"]),"MOTP":float(c["MOTP"]),
            "CLR_TP":int(c["CLR_TP"]),"CLR_FP":int(c["CLR_FP"]),"CLR_FN":int(c["CLR_FN"]),
            "MT":int(c["MT"]),"PT":int(c["PT"]),"ML":int(c["ML"])}

def evaluate_tracker(name,pred_path,frames_path,gt,sequences):
    raw=pd.read_csv(pred_path,low_memory=False); frames=pd.read_csv(frames_path,low_memory=False)
    pred,audit=resolve_prediction_class(raw,name)
    pred["sequence"]=pred["sequence"].astype(str).str.strip(); pred["frame"]=pd.to_numeric(pred["frame"],errors="raise").astype(int)
    frames["sequence"]=frames["sequence"].astype(str).str.strip(); frames["frame"]=pd.to_numeric(frames["frame"],errors="raise").astype(int)
    rows=[]; hs=[]; ids=[]; cs=[]; alpha_rows=[]
    for seq in sequences:
        data=build_sequence_data(seq,gt,pred,frames); h=hota_eval(data); i=identity_eval(data); c=clear_eval(data)
        hs.append(h); ids.append(i); cs.append(c); rows.append(summary_row(name,seq,h,i,c,data))
        for a,alpha in enumerate(ALPHAS):
            alpha_rows.append({"tracker":name,"sequence":seq,"alpha":float(alpha),"HOTA":float(h["HOTA"][a]),
                               "DetA":float(h["DetA"][a]),"AssA":float(h["AssA"][a]),"DetRe":float(h["DetRe"][a]),
                               "DetPr":float(h["DetPr"][a]),"AssRe":float(h["AssRe"][a]),"AssPr":float(h["AssPr"][a]),
                               "LocA":float(h["LocA"][a]),"HOTA_TP":int(h["HOTA_TP"][a]),"HOTA_FN":int(h["HOTA_FN"][a]),"HOTA_FP":int(h["HOTA_FP"][a])})
    hc=hota_combine(hs); ic=identity_combine(ids); cc=clear_combine(cs)
    comb={"num_timesteps":sum(r["num_frames"] for r in rows),"num_gt_ids":sum(r["num_gt_ids"] for r in rows),
          "num_tracker_ids":sum(r["num_tracker_ids"] for r in rows),"num_gt_dets":sum(r["num_gt_dets"] for r in rows),
          "num_tracker_dets":sum(r["num_tracker_dets"] for r in rows)}
    agg=summary_row(name,"COMBINED_6_SEQUENCES",hc,ic,cc,comb)
    for a,alpha in enumerate(ALPHAS):
        alpha_rows.append({"tracker":name,"sequence":"COMBINED_6_SEQUENCES","alpha":float(alpha),"HOTA":float(hc["HOTA"][a]),
                           "DetA":float(hc["DetA"][a]),"AssA":float(hc["AssA"][a]),"DetRe":float(hc["DetRe"][a]),
                           "DetPr":float(hc["DetPr"][a]),"AssRe":float(hc["AssRe"][a]),"AssPr":float(hc["AssPr"][a]),
                           "LocA":float(hc["LocA"][a]),"HOTA_TP":int(hc["HOTA_TP"][a]),"HOTA_FN":int(hc["HOTA_FN"][a]),"HOTA_FP":int(hc["HOTA_FP"][a])})
    return pd.DataFrame(rows),pd.DataFrame([agg]),pd.DataFrame(alpha_rows),audit

def main():
    print("\nR13-E — TRACKING METRICS AGAINST OBSERVED TESTREAL GT")
    print("METRICS ONLY. NO inference and NO tuning. HOTA follows TrackEval alpha=0.05..0.95; IDF1 and CLEAR use IoU 0.5.\n")
    protocol,_=verify_inputs(); sequences=list(protocol["data"]["sequences"]); gt=load_gt(protocol)
    per=[]; agg=[]; alphas=[]; audits=[]
    for name,b,f in [("ByteTrack",BYTE_BOXES,BYTE_FRAMES),("BoT-SORT",BOT_BOXES,BOT_FRAMES)]:
        print("EVALUATING",name,"...")
        p,a,ac,au=evaluate_tracker(name,b,f,gt,sequences); per.append(p); agg.append(a); alphas.append(ac); audits.append(au)
    per_df=pd.concat(per,ignore_index=True); agg_df=pd.concat(agg,ignore_index=True); alpha_df=pd.concat(alphas,ignore_index=True); audit_df=pd.DataFrame(audits)
    byte=agg_df.loc[agg_df["tracker"]=="ByteTrack"].iloc[0]; bot=agg_df.loc[agg_df["tracker"]=="BoT-SORT"].iloc[0]
    comp=[]
    for m,dirn in [("HOTA","higher"),("IDF1","higher"),("DetA","higher"),("AssA","higher"),("IDSW","lower"),("Frag","lower"),("MOTA","higher")]:
        comp.append({"metric":m,"ByteTrack":byte[m],"BoT-SORT":bot[m],"ByteTrack_minus_BoTSORT":byte[m]-bot[m],"preferred_direction":dirn})
    comp_df=pd.DataFrame(comp)
    paths={
        "per_sequence":OUT/"R13E_PER_SEQUENCE_TRACKING_METRICS.csv",
        "aggregate":OUT/"R13E_AGGREGATE_TRACKING_METRICS.csv",
        "alpha":OUT/"R13E_HOTA_ALPHA_CURVES.csv",
        "comparison":OUT/"R13E_PRIMARY_VS_SENSITIVITY_COMPARISON.csv",
        "class_audit":OUT/"R13E_TRACKER_CLASS_FILTER_AUDIT.csv"}
    per_df.to_csv(paths["per_sequence"],index=False); agg_df.to_csv(paths["aggregate"],index=False); alpha_df.to_csv(paths["alpha"],index=False)
    comp_df.to_csv(paths["comparison"],index=False); audit_df.to_csv(paths["class_audit"],index=False)
    status="R13E_TRACKING_METRICS_COMPLETE_HOTA_IDF1_IDSW_FRAGMENTATION_LOCKED_READY_FOR_R13F_ERROR_PROPAGATION"
    manifest={"step":"R13E_TRACKING_METRICS","created_utc":utc_now(),"status":status,"training":False,"inference":False,"tuning":False,
              "raw_r13d_hash_verified":True,"gt_policy":"observed_only_interpolated_0","hota_alpha_grid":[float(x) for x in ALPHAS],
              "identity_iou_threshold":IDENTITY_THRESHOLD,"clear_iou_threshold":CLEAR_THRESHOLD,
              "primary_tracker":"ByteTrack","sensitivity_tracker":"BoT-SORT","aggregate_results":agg_df.to_dict(orient="records"),
              "outputs":{k:str(v) for k,v in paths.items()}}
    (OUT/"R13E_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\nTRACKER CLASS AUDIT:"); print(audit_df.to_string(index=False))
    print("\nAGGREGATE TRACKING METRICS:")
    cols=["tracker","num_frames","num_gt_ids","num_tracker_ids","num_gt_dets","num_tracker_dets","HOTA","DetA","AssA","IDF1","IDSW","Frag","MOTA","MT","PT","ML"]
    print(agg_df[cols].to_string(index=False))
    print("\nPER-SEQUENCE METRICS:")
    print(per_df[["tracker","sequence","num_gt_ids","num_tracker_ids","HOTA","DetA","AssA","IDF1","IDSW","Frag","MOTA"]].to_string(index=False))
    print("\nPRIMARY VS SENSITIVITY:"); print(comp_df.to_string(index=False))
    print("\nSTATUS:"); print(status); print("\nR13-E COMPLETE"); print("Outputs:",OUT)

if __name__=="__main__":
    main()
