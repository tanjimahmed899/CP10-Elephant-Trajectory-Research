
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HOME=Path.home(); DOWNLOADS=HOME/"Downloads"
MAIN1=Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
REV=DOWNLOADS/"paper_strengthening_revision"; R13=REV/"R13_end_to_end_pipeline"
R13D=R13/"R13D_frozen_detector_tracker_execution"; R13E=R13/"R13E_tracking_metrics"
D_MAN=R13D/"R13D_MANIFEST.json"; E_MAN=R13E/"R13E_MANIFEST.json"
DET=R13D/"R13D_DETECTIONS_ALL_BOXES.csv"; BYTE=R13D/"R13D_BYTETRACK_ALL_BOXES.csv"; BOT=R13D/"R13D_BOTSORT_ALL_BOXES.csv"
CLEAN=MAIN1/"trajectory_dataset"/"cleaned_track_points.csv"
OUT=R13/"R13F_A_detector_tracker_bottleneck_audit"; OUT.mkdir(parents=True,exist_ok=True)
PRIMARY=0.50; DIAGS=[0.30,0.50,0.70]; EPS=np.finfo(float).eps

def now(): return datetime.now(timezone.utc).isoformat()
def req(p):
    if not Path(p).exists(): raise FileNotFoundError(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        while True:
            b=f.read(1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()
def canon(v):
    if pd.isna(v): return ""
    s=str(v).strip()
    try:
        x=float(s)
        if np.isfinite(x) and x.is_integer(): return str(int(x))
    except: pass
    return s

def verify():
    for p in [D_MAN,E_MAN,DET,BYTE,BOT,CLEAN]: req(p)
    d=json.loads(D_MAN.read_text(encoding="utf-8")); e=json.loads(E_MAN.read_text(encoding="utf-8"))
    if d.get("status")!="R13D_FROZEN_DETECTOR_BYTETRACK_BOTSORT_EXECUTION_COMPLETE_RAW_OUTPUTS_LOCKED_READY_FOR_R13E_TRACKING_METRICS":
        raise RuntimeError(f"Unexpected R13-D status: {d.get('status')}")
    if e.get("status")!="R13E_TRACKING_METRICS_COMPLETE_HOTA_IDF1_IDSW_FRAGMENTATION_LOCKED_READY_FOR_R13F_ERROR_PROPAGATION":
        raise RuntimeError(f"Unexpected R13-E status: {e.get('status')}")
    sm={x["component"]:x for x in d["summary"]}
    for comp,p in [("detections",DET),("bytetrack",BYTE),("botsort",BOT)]:
        if sha(p)!=sm[comp]["boxes_sha256"]: raise RuntimeError(f"Hash mismatch: {p}")
    return d

def load_gt(seqs):
    df=pd.read_csv(CLEAN,low_memory=False)
    s=df["split"].astype(str).str.strip().str.lower()
    m=s.isin(["test_real","testreal","test-real"])
    if not m.any(): m=s.str.contains("test",regex=False)
    df=df.loc[m].copy(); df["sequence"]=df["sequence"].astype(str).str.strip()
    df=df[df["sequence"].isin(set(seqs))].copy()
    if "interpolated" in df.columns:
        it=pd.to_numeric(df["interpolated"],errors="coerce").fillna(0)
        df=df.loc[it==0].copy()
    df["frame"]=pd.to_numeric(df["frame"],errors="raise").astype(int)
    df["gt_track"]=df["sequence"]+"|"+df["object_id"].map(canon)
    for c in ["x","y","w","h"]: df[c]=pd.to_numeric(df[c],errors="raise")
    return df

def prep(path,stage):
    df=pd.read_csv(path,low_memory=False)
    df["sequence"]=df["sequence"].astype(str).str.strip()
    df["frame"]=pd.to_numeric(df["frame"],errors="raise").astype(int)
    if len(df):
        classes=sorted(df["class_name"].astype(str).str.strip().unique().tolist())
        if len(classes)>1:
            elephant=[c for c in classes if "elephant" in c.lower()]
            if len(elephant)!=1: raise RuntimeError(f"{stage}: ambiguous classes {classes}")
            df=df[df["class_name"].astype(str)==elephant[0]].copy()
    if stage.endswith("_ID_ASSIGNED"):
        tid=pd.to_numeric(df["track_id"],errors="coerce")
        df=df[tid.notna() & (tid>=0)].copy()
    for c in ["x1","y1","x2","y2","confidence"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def gxy(g):
    x=g["x"].to_numpy(float); y=g["y"].to_numpy(float); w=g["w"].to_numpy(float); h=g["h"].to_numpy(float)
    return np.column_stack([x,y,x+w,y+h])
def pxy(p): return p[["x1","y1","x2","y2"]].to_numpy(float)
def iou(a,b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)))
    ix1=np.maximum(a[:,None,0],b[None,:,0]); iy1=np.maximum(a[:,None,1],b[None,:,1])
    ix2=np.minimum(a[:,None,2],b[None,:,2]); iy2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
    aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1])
    bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1])
    uni=aa[:,None]+bb[None,:]-inter; out=np.zeros_like(inter); m=uni>EPS; out[m]=inter[m]/uni[m]
    return out

def eval_stage(stage,pred,gt,seqs,thr):
    rows=[]; matched=[]
    for seq in seqs:
        gs=gt[gt["sequence"]==seq].copy(); ps=pred[pred["sequence"]==seq].copy()
        frames=sorted(set(gs["frame"])|set(ps["frame"]))
        tp=fp=fn=0; f_any=f_match=0
        for fr in frames:
            g=gs[gs["frame"]==fr].reset_index(drop=False); p=ps[ps["frame"]==fr].reset_index(drop=False)
            if len(g) and len(p): f_any+=1
            if len(g)==0: fp+=len(p); continue
            if len(p)==0: fn+=len(g); continue
            sim=iou(gxy(g),pxy(p)); rr,cc=linear_sum_assignment(-sim); keep=sim[rr,cc]>=thr
            rr=rr[keep]; cc=cc[keep]; vals=sim[rr,cc]; n=len(rr)
            tp+=n; fp+=len(p)-n; fn+=len(g)-n
            if n:
                f_match+=1
                for k in range(n):
                    matched.append({"stage":stage,"sequence":seq,"frame":int(fr),
                                    "gt_track":str(g.iloc[rr[k]]["gt_track"]),
                                    "prediction_row_index":int(p.iloc[cc[k]]["index"]),
                                    "iou":float(vals[k])})
        prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn); f1=2*prec*rec/max(EPS,prec+rec)
        gc=len(gs); pc=len(ps)
        rows.append({"stage":stage,"sequence":seq,"iou_threshold":thr,"gt_detections":gc,"prediction_boxes":pc,
                     "TP":tp,"FP":fp,"FN":fn,"precision":prec,"recall":rec,"F1":f1,
                     "count_upper_bound_recall":min(1.0,pc/max(1,gc)),
                     "gt_frames_with_any_prediction":f_any,"gt_frames_with_match":f_match,
                     "gt_unique_frames":int(gs["frame"].nunique())})
    sdf=pd.DataFrame(rows); TP=int(sdf.TP.sum()); FP=int(sdf.FP.sum()); FN=int(sdf.FN.sum())
    prec=TP/max(1,TP+FP); rec=TP/max(1,TP+FN); f1=2*prec*rec/max(EPS,prec+rec)
    gc=int(sdf.gt_detections.sum()); pc=int(sdf.prediction_boxes.sum())
    agg=pd.DataFrame([{"stage":stage,"sequence":"COMBINED_6_SEQUENCES","iou_threshold":thr,"gt_detections":gc,"prediction_boxes":pc,
                       "TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"F1":f1,
                       "count_upper_bound_recall":min(1.0,pc/max(1,gc)),
                       "gt_frames_with_any_prediction":int(sdf.gt_frames_with_any_prediction.sum()),
                       "gt_frames_with_match":int(sdf.gt_frames_with_match.sum()),
                       "gt_unique_frames":int(sdf.gt_unique_frames.sum())}])
    return sdf,agg,pd.DataFrame(matched)

def coverage(matched,gt,stage):
    base=gt.groupby("gt_track").size().rename("gt_observed_rows").reset_index()
    if len(matched):
        mc=matched.groupby("gt_track").size().rename("matched_rows").reset_index()
    else: mc=pd.DataFrame(columns=["gt_track","matched_rows"])
    out=base.merge(mc,how="left",on="gt_track"); out["matched_rows"]=out["matched_rows"].fillna(0).astype(int)
    out["matched_fraction"]=out["matched_rows"]/out["gt_observed_rows"]; out["stage"]=stage
    return out

def main():
    print("\nR13-F-A — DETECTOR/TRACKER BOTTLENECK DECOMPOSITION")
    print("DIAGNOSTIC METRICS ONLY. NO inference, NO retraining, NO tuning. Primary IoU=0.50; 0.30/0.70 diagnostic only.\n")
    d=verify(); seqs=list(d["sequences"]); gt=load_gt(seqs)
    stages=[("DETECTOR_RAW",prep(DET,"DETECTOR_RAW")),
            ("BYTETRACK_ALL_RETURNED",prep(BYTE,"BYTETRACK_ALL_RETURNED")),
            ("BYTETRACK_ID_ASSIGNED",prep(BYTE,"BYTETRACK_ID_ASSIGNED")),
            ("BOTSORT_ALL_RETURNED",prep(BOT,"BOTSORT_ALL_RETURNED")),
            ("BOTSORT_ID_ASSIGNED",prep(BOT,"BOTSORT_ID_ASSIGNED"))]
    seqs_all=[]; aggs=[]; covs=[]
    for thr in DIAGS:
        for name,p in stages:
            s,a,m=eval_stage(name,p,gt,seqs,thr); seqs_all.append(s); aggs.append(a)
            if abs(thr-PRIMARY)<1e-9: covs.append(coverage(m,gt,name))
    seq_df=pd.concat(seqs_all,ignore_index=True); agg_df=pd.concat(aggs,ignore_index=True); cov_df=pd.concat(covs,ignore_index=True)
    primary=agg_df[np.isclose(agg_df.iou_threshold,PRIMARY)].copy()
    raw_rec=float(primary.loc[primary.stage=="DETECTOR_RAW","recall"].iloc[0]); primary["recall_loss_vs_raw_detector"]=np.nan
    for i in primary.index:
        if primary.loc[i,"stage"]!="DETECTOR_RAW": primary.loc[i,"recall_loss_vs_raw_detector"]=raw_rec-float(primary.loc[i,"recall"])
    raw=primary[primary.stage=="DETECTOR_RAW"].iloc[0]
    byt=primary[primary.stage=="BYTETRACK_ID_ASSIGNED"].iloc[0]
    bot=primary[primary.stage=="BOTSORT_ID_ASSIGNED"].iloc[0]
    conclusion=pd.DataFrame([
        {"question":"Raw detector count-only maximum recall","answer":f"{float(raw.count_upper_bound_recall):.6f}",
         "evidence":f"{int(raw.prediction_boxes)} boxes vs {int(raw.gt_detections)} GT"},
        {"question":"Actual raw detector recall @ IoU=0.50","answer":f"{float(raw.recall):.6f}",
         "evidence":f"TP={int(raw.TP)}, FN={int(raw.FN)}"},
        {"question":"ByteTrack ID-assigned recall @ IoU=0.50","answer":f"{float(byt.recall):.6f}",
         "evidence":f"{int(byt.prediction_boxes)} ID-assigned boxes"},
        {"question":"BoT-SORT ID-assigned recall @ IoU=0.50","answer":f"{float(bot.recall):.6f}",
         "evidence":f"{int(bot.prediction_boxes)} ID-assigned boxes"}])
    p1=OUT/"R13F_A_PER_SEQUENCE_STAGE_DETECTION_METRICS.csv"; p2=OUT/"R13F_A_AGGREGATE_STAGE_DETECTION_METRICS.csv"
    p3=OUT/"R13F_A_PRIMARY_IOU050_STAGE_DECOMPOSITION.csv"; p4=OUT/"R13F_A_GT_TRACK_OBSERVATION_COVERAGE_IOU050.csv"; p5=OUT/"R13F_A_CONCLUSION_TABLE.csv"
    seq_df.to_csv(p1,index=False); agg_df.to_csv(p2,index=False); primary.to_csv(p3,index=False); cov_df.to_csv(p4,index=False); conclusion.to_csv(p5,index=False)
    status="R13F_A_DETECTOR_TRACKER_BOTTLENECK_DECOMPOSITION_COMPLETE_REVIEW_BEFORE_FORECAST_PROPAGATION"
    man={"step":"R13F_A_DETECTOR_TRACKER_BOTTLENECK_DECOMPOSITION","created_utc":now(),"status":status,"training":False,"inference":False,"tuning":False,
         "primary_iou":PRIMARY,"diagnostic_ious":DIAGS,
         "raw_detector_count_upper_bound_recall":float(raw.count_upper_bound_recall),
         "raw_detector_recall_iou050":float(raw.recall),
         "bytetrack_id_recall_iou050":float(byt.recall),
         "botsort_id_recall_iou050":float(bot.recall),
         "outputs":{"per_sequence":str(p1),"aggregate":str(p2),"primary_decomposition":str(p3),"track_coverage":str(p4),"conclusion":str(p5)}}
    (OUT/"R13F_A_MANIFEST.json").write_text(json.dumps(man,indent=2),encoding="utf-8")
    print("PRIMARY IoU=0.50 STAGE DECOMPOSITION:")
    print(primary[["stage","gt_detections","prediction_boxes","TP","FP","FN","precision","recall","F1","count_upper_bound_recall","recall_loss_vs_raw_detector"]].to_string(index=False))
    print("\nCONCLUSION TABLE:"); print(conclusion.to_string(index=False))
    print("\nPER-SEQUENCE RAW DETECTOR @ IoU=0.50:")
    print(seq_df[(seq_df.stage=="DETECTOR_RAW") & np.isclose(seq_df.iou_threshold,PRIMARY)][["sequence","gt_detections","prediction_boxes","TP","FP","FN","precision","recall","F1","count_upper_bound_recall"]].to_string(index=False))
    print("\nSTATUS:"); print(status); print("\nR13-F-A COMPLETE"); print("Outputs:",OUT)

if __name__=="__main__": main()
