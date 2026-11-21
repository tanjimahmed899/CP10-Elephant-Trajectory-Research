from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")
OUT = DOWNLOADS / "step7_response_aware_warning"
OUT.mkdir(parents=True, exist_ok=True)
PROTOCOL = OUT / "STEP7A_PROTOCOL_LOCK.json"
GEOFENCE = ROOT / "step18_geofence" / "step18b_evaluation" / "step18b_all_geofence_instances.csv"
STEP18_SCRIPT = ROOT / "step18b_evaluate_locked_geofence.py"


def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def parse_bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    low = s.astype(str).str.strip().str.lower()
    true_set = {"true","1","yes","y","t"}
    false_set = {"false","0","no","n","f"}
    bad = ~low.isin(true_set | false_set)
    if bad.any():
        raise RuntimeError(f"Cannot parse boolean values: {sorted(low[bad].unique().tolist())[:20]}")
    return low.isin(true_set)


def load_protocol():
    if not PROTOCOL.exists():
        raise FileNotFoundError(f"Missing Step-7A protocol lock: {PROTOCOL}")
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for key in ["locked_step6c_q_values","response_time_sensitivity_seconds","primary_warning_logic"]:
        if key not in p:
            raise RuntimeError(f"Protocol lock missing key: {key}")
    return p


def infer_score_orientation(df):
    pred = df["predicted_crossing_bool"].to_numpy(bool)
    score = pd.to_numeric(df["risk_score"], errors="coerce").to_numpy(float)
    good = np.isfinite(score)
    pred, score = pred[good], score[good]
    rules = {
        "positive_ge_zero": score >= 0,
        "positive_gt_zero": score > 0,
        "negative_le_zero": score <= 0,
        "negative_lt_zero": score < 0,
    }
    agreements = {k: float(np.mean(v == pred)) for k,v in rules.items()}
    best = max(agreements, key=agreements.get)
    if agreements[best] < 0.995:
        raise RuntimeError(f"risk_score sign not reliable: {agreements}")
    orientation = 1.0 if best.startswith("positive") else -1.0
    return orientation, best, agreements


def source_code_excerpts():
    if not STEP18_SCRIPT.exists():
        return []
    lines = STEP18_SCRIPT.read_text(encoding="utf-8", errors="ignore").splitlines()
    terms = ["risk_score","predicted_crossing","distance_to_boundary","predicted_tte"]
    out=[]
    for i,line in enumerate(lines):
        if any(t in line.lower() for t in terms):
            out.append({"line": i+1, "excerpt": " || ".join(x.strip() for x in lines[max(0,i-2):min(len(lines),i+3)])})
        if len(out)>=30:
            break
    return out


def candidate_metadata_files():
    out=[]
    for p in ROOT.rglob("*.csv"):
        s=str(p).lower()
        if not any(t in s for t in ["metadata","fresh","test","trajectory","transformer"]):
            continue
        try:
            with open(p,"r",encoding="utf-8",errors="ignore") as f:
                header=f.readline().lower()
        except Exception:
            continue
        if "scale" in header and ("track_key" in header or "object_id" in header):
            out.append(p)
    return out


def task_tokens(task):
    t=str(task).lower()
    if "2s_to_2s" in t: return ["2s_to_2s","2to2","2s2s"]
    if "2s_to_5s" in t: return ["2s_to_5s","2to5","2s5s"]
    return [t]


def dataset_tokens(dataset):
    d=str(dataset).lower()
    if "fresh" in d or "wildbox" in d: return ["fresh","wildbox"]
    if "test" in d or "birdsai" in d: return ["test","birdsai"]
    return [d]


def score_metadata_candidate(path, unique, dataset, task):
    try:
        df=pd.read_csv(path)
    except Exception:
        return None
    if "scale" not in df.columns:
        return None
    n=int(unique["sample_index"].nunique())
    mx=int(unique["sample_index"].max())
    if len(df)<=mx:
        return None
    score=0.0
    s=str(path).lower()
    if any(t in s for t in task_tokens(task)): score += 3
    if any(t in s for t in dataset_tokens(dataset)): score += 3
    if "train" in s and ("test" in str(dataset).lower() or "fresh" in str(dataset).lower()): score -= 5
    row_match=(len(df)==n)
    if row_match: score += 4
    idx=unique.sort_values("sample_index")["sample_index"].astype(int).to_numpy()
    track_ag=np.nan; source_ag=np.nan
    u=unique.sort_values("sample_index")
    if "track_key" in df.columns:
        a=u["track_key"].astype(str).str.strip().to_numpy(); b=df.iloc[idx]["track_key"].astype(str).str.strip().to_numpy()
        track_ag=float(np.mean(a==b)); score += 10*track_ag
    if "source_group" in df.columns:
        a=u["source_group"].astype(str).str.strip().to_numpy(); b=df.iloc[idx]["source_group"].astype(str).str.strip().to_numpy()
        source_ag=float(np.mean(a==b)); score += 5*source_ag
    scales=pd.to_numeric(df["scale"],errors="coerce")
    valid=float(np.mean(np.isfinite(scales)&(scales>0)))
    score += 3*valid
    return {
        "path":str(path),"rows":len(df),"expected_samples":n,"row_match":row_match,
        "track_agreement":track_ag if np.isfinite(track_ag) else np.nan,
        "source_agreement":source_ag if np.isfinite(source_ag) else np.nan,
        "finite_positive_scale_fraction":valid,"candidate_score":score,
    }


def resolve_scale_mapping(geo):
    candidates=candidate_metadata_files(); audits=[]; maps=[]
    for (dataset,task),sub in geo.groupby(["dataset","task"]):
        unique=sub[["sample_index","track_key","source_group"]].drop_duplicates("sample_index")
        scored=[]
        for p in candidates:
            r=score_metadata_candidate(p,unique,dataset,task)
            if r is not None:
                r.update({"dataset":dataset,"task":task}); scored.append(r)
        if not scored:
            raise RuntimeError(f"No scale metadata candidates for {dataset}/{task}")
        sdf=pd.DataFrame(scored).sort_values(["candidate_score","path"],ascending=[False,True])
        audits.extend(scored); best=sdf.iloc[0]
        if not bool(best["row_match"]):
            raise RuntimeError(f"Best metadata row count mismatch for {dataset}/{task}: {best.to_dict()}")
        if pd.notna(best["track_agreement"]) and float(best["track_agreement"])<0.99:
            raise RuntimeError(f"Track agreement <0.99 for {dataset}/{task}: {best.to_dict()}")
        if pd.notna(best["source_agreement"]) and float(best["source_agreement"])<0.99:
            raise RuntimeError(f"Source agreement <0.99 for {dataset}/{task}: {best.to_dict()}")
        if float(best["finite_positive_scale_fraction"])<0.999:
            raise RuntimeError(f"Invalid scale values for {dataset}/{task}")
        mdf=pd.read_csv(best["path"]); m=unique.sort_values("sample_index").copy(); idx=m["sample_index"].astype(int).to_numpy()
        m["scale"]=pd.to_numeric(mdf.iloc[idx]["scale"].to_numpy(),errors="coerce")
        m["dataset"]=dataset; m["task"]=task; m["metadata_path"]=best["path"]
        maps.append(m)
    return pd.concat(maps,ignore_index=True), pd.DataFrame(audits)


def task_key(task):
    t=str(task).lower()
    if "2s_to_2s" in t: return "2s_to_2s"
    if "2s_to_5s" in t: return "2s_to_5s"
    raise RuntimeError(f"Unknown task: {task}")


def safe_div(a,b):
    return float(a/b) if b else np.nan


def metric_row(df,dataset,task,alpha,response_sec,level):
    true=df["true_crossing_bool"].to_numpy(bool)
    red=df["alarm_state"].eq("RED").to_numpy(); alert=df["alarm_state"].isin(["YELLOW","RED"]).to_numpy(); non=~true
    red_tp=int(np.sum(red&true)); alert_tp=int(np.sum(alert&true)); red_fp=int(np.sum(red&non)); alert_fp=int(np.sum(alert&non))
    rc=int(np.sum(red)); ac=int(np.sum(alert))
    tte=pd.to_numeric(df["true_tte_sec"],errors="coerce"); vals=tte[red&true]
    return {
        "level":level,"dataset":dataset,"task":task,"alpha":alpha,"target_coverage":1-alpha,"response_time_sec":response_sec,
        "instances":len(df),"actual_crossings":int(np.sum(true)),"actual_crossing_prevalence":safe_div(int(np.sum(true)),len(df)),
        "green_count":int(np.sum(df["alarm_state"]=="GREEN")),"yellow_count":int(np.sum(df["alarm_state"]=="YELLOW")),"red_count":rc,
        "green_rate":float(np.mean(df["alarm_state"]=="GREEN")),"yellow_rate":float(np.mean(df["alarm_state"]=="YELLOW")),"red_rate":float(np.mean(df["alarm_state"]=="RED")),
        "red_sensitivity_actual_crossing":safe_div(red_tp,int(np.sum(true))),
        "yellow_or_red_sensitivity_actual_crossing":safe_div(alert_tp,int(np.sum(true))),
        "false_red_rate_non_crossing":safe_div(red_fp,int(np.sum(non))),
        "false_yellow_or_red_rate_non_crossing":safe_div(alert_fp,int(np.sum(non))),
        "red_precision":safe_div(red_tp,rc),"yellow_or_red_precision":safe_div(alert_tp,ac),
        "median_true_tte_sec_among_red_true_crossings":float(np.nanmedian(vals)) if len(vals) and np.isfinite(vals).any() else np.nan,
    }


def classify(sub,q,alpha,response_sec,orientation):
    d=sub.copy(); d["absolute_conformal_radius"]=q*d["scale"]
    d["oriented_risk_score"]=orientation*pd.to_numeric(d["risk_score"],errors="coerce")
    d["uncertainty_risk_score"]=d["oriented_risk_score"]+d["absolute_conformal_radius"]
    d["uncertainty_intersects_geofence"]=d["uncertainty_risk_score"]>=0
    pred=d["predicted_crossing_bool"]; tte=pd.to_numeric(d["predicted_tte_sec"],errors="coerce")
    red=pred & tte.notna() & (tte<=response_sec)
    distance=pd.to_numeric(d["distance_to_boundary"],errors="coerce")
    red=red | (pred & tte.isna() & distance.notna() & (distance<=0))
    yellow=(~red) & (d["uncertainty_intersects_geofence"] | pred)
    d["alarm_state"]=np.where(red,"RED",np.where(yellow,"YELLOW","GREEN"))
    d["alpha"]=alpha; d["target_coverage"]=1-alpha; d["response_time_sec"]=response_sec
    return d


def aggregate_sample(inst):
    sev={"GREEN":0,"YELLOW":1,"RED":2}; inv={0:"GREEN",1:"YELLOW",2:"RED"}; x=inst.copy(); x["_sev"]=x["alarm_state"].map(sev)
    group_cols=["dataset","task","sample_index","source_group","track_key","alpha","target_coverage","response_time_sec"]
    rows=[]
    for keys,g in x.groupby(group_cols,dropna=False):
        r=dict(zip(group_cols,keys)); r["alarm_state"]=inv[int(g["_sev"].max())]; r["true_crossing_bool"]=bool(g["true_crossing_bool"].any())
        t=pd.to_numeric(g["true_tte_sec"],errors="coerce"); f=t[np.isfinite(t)]; r["true_tte_sec"]=float(f.min()) if len(f) else np.nan
        r["boundaries_evaluated"]=len(g); r["uncertainty_intersects_any_geofence"]=bool(g["uncertainty_intersects_geofence"].any()); rows.append(r)
    return pd.DataFrame(rows)


def track_macro(samples):
    rows=[]
    for (dataset,task,alpha,rt),g in samples.groupby(["dataset","task","alpha","response_time_sec"]):
        per=[]
        for track,tg in g.groupby("track_key"):
            m=metric_row(tg,dataset,task,alpha,rt,"track_internal"); m["track_key"]=track; per.append(m)
        tdf=pd.DataFrame(per); row={"dataset":dataset,"task":task,"alpha":alpha,"target_coverage":1-alpha,"response_time_sec":rt,"tracks":tdf["track_key"].nunique()}
        cols=[c for c in tdf.columns if c not in {"level","dataset","task","alpha","target_coverage","response_time_sec","instances","actual_crossings","green_count","yellow_count","red_count","track_key"}]
        for c in cols:
            v=pd.to_numeric(tdf[c],errors="coerce"); row["track_macro_"+c]=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("\nSTEP 7B — RESPONSE-AWARE GREEN / YELLOW / RED EVALUATION")
    protocol=load_protocol(); print("\nSTEP-7A PROTOCOL LOCK VERIFIED:"); print(PROTOCOL)
    if not GEOFENCE.exists(): raise FileNotFoundError(GEOFENCE)
    geo=pd.read_csv(GEOFENCE)
    req=["dataset","task","sample_index","source_group","track_key","distance_to_boundary","true_crossing","predicted_crossing","risk_score","true_tte_sec","predicted_tte_sec"]
    miss=[c for c in req if c not in geo.columns]
    if miss: raise RuntimeError(f"Geofence file missing columns: {miss}")
    geo["true_crossing_bool"]=parse_bool_series(geo["true_crossing"]); geo["predicted_crossing_bool"]=parse_bool_series(geo["predicted_crossing"])
    orientation,rule,agreements=infer_score_orientation(geo)
    print("\nRISK-SCORE SIGN AUDIT:"); print("selected_rule =",rule); print("orientation =",orientation); print("agreements =",agreements)
    pd.DataFrame(source_code_excerpts()).to_csv(OUT/"step7b_step18_formula_excerpts.csv",index=False)
    scale_map,scale_audit=resolve_scale_mapping(geo); scale_audit.to_csv(OUT/"step7b_scale_mapping_audit.csv",index=False); scale_map.to_csv(OUT/"step7b_sample_scale_map.csv",index=False)
    geo=geo.merge(scale_map[["dataset","task","sample_index","scale","metadata_path"]],on=["dataset","task","sample_index"],how="left",validate="many_to_one")
    if geo["scale"].isna().any() or (~np.isfinite(geo["scale"])).any() or (geo["scale"]<=0).any(): raise RuntimeError("Scale mapping invalid/missing")
    compact=(scale_audit.sort_values(["dataset","task","candidate_score"],ascending=[True,True,False]).groupby(["dataset","task"],as_index=False).first())
    print("\nSCALE MAPPING VERIFIED:"); print(compact[["dataset","task","path","rows","expected_samples","track_agreement","source_agreement","candidate_score"]].to_string(index=False))

    all_i=[]; all_s=[]; metrics=[]
    for (dataset,task),sub in geo.groupby(["dataset","task"]):
        tk=task_key(task); qinfo=protocol["locked_step6c_q_values"][tk]; grid=protocol["response_time_sensitivity_seconds"][tk]
        for alpha,qkey in [(0.10,"q90"),(0.05,"q95")]:
            q=float(qinfo[qkey])
            for rt in grid:
                inst=classify(sub,q,alpha,float(rt),orientation); all_i.append(inst); metrics.append(metric_row(inst,dataset,task,alpha,float(rt),"geofence_instance"))
                samp=aggregate_sample(inst); all_s.append(samp); metrics.append(metric_row(samp,dataset,task,alpha,float(rt),"sample_max_severity"))
    instances=pd.concat(all_i,ignore_index=True); samples=pd.concat(all_s,ignore_index=True); metrics=pd.DataFrame(metrics); tmacro=track_macro(samples)
    instances.to_csv(OUT/"step7b_all_geofence_instance_alarms.csv",index=False); samples.to_csv(OUT/"step7b_sample_level_max_alarm.csv",index=False)
    metrics.to_csv(OUT/"step7b_response_time_sensitivity_metrics.csv",index=False); tmacro.to_csv(OUT/"step7b_track_macro_metrics.csv",index=False)
    primary=metrics[(metrics["level"]=="sample_max_severity") & np.isclose(metrics["alpha"],0.10)].copy(); secondary=metrics[(metrics["level"]=="sample_max_severity") & np.isclose(metrics["alpha"],0.05)].copy()
    primary.to_csv(OUT/"TABLE_step7_primary_90pct_response_sensitivity.csv",index=False); secondary.to_csv(OUT/"TABLE_step7_secondary_95pct_response_sensitivity.csv",index=False)
    provenance={"step":"STEP_7B_RESPONSE_AWARE_WARNING_EVALUATION","step7a_protocol_sha256":sha256_file(PROTOCOL),"step18_geofence_instances_sha256":sha256_file(GEOFENCE),"step18_source_script_sha256":sha256_file(STEP18_SCRIPT) if STEP18_SCRIPT.exists() else None,"risk_score_orientation":orientation,"risk_score_rule":rule,"risk_score_rule_agreements":agreements,"scope":"Retrospective virtual-geofence decision-layer simulation using Step-6C conformal radii; not BAMBI confirmatory coverage and not field deployment."}
    (OUT/"STEP7B_PROVENANCE.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    with open(OUT/"STEP7B_SUMMARY.txt","w",encoding="utf-8") as f:
        f.write("STEP 7B — RESPONSE-AWARE GREEN / YELLOW / RED EVALUATION\n"+"="*84+"\n\n")
        f.write("SCOPE\n-----\nRetrospective virtual-geofence decision-layer simulation using selected Step-6C conformal radii. Not BAMBI confirmatory coverage and not field validation.\n\n")
        f.write("RISK-SCORE SIGN AUDIT\n---------------------\n"); f.write(f"Selected rule: {rule}\nOrientation: {orientation}\n{json.dumps(agreements,indent=2)}\n\n")
        f.write("SCALE MAPPING\n-------------\n"); f.write(compact[["dataset","task","path","rows","expected_samples","track_agreement","source_agreement","candidate_score"]].to_string(index=False))
        f.write("\n\nPRIMARY 90% SAMPLE-LEVEL RESPONSE SENSITIVITY\n---------------------------------------------\n"); f.write(primary.to_string(index=False))
        f.write("\n\nSECONDARY 95% SAMPLE-LEVEL RESPONSE SENSITIVITY\n-----------------------------------------------\n"); f.write(secondary.to_string(index=False))
        f.write("\n\nTRACK-MACRO METRICS\n-------------------\n"); f.write(tmacro.to_string(index=False))
        f.write("\n\nIMPORTANT NON-CLAIMS\n--------------------\n");
        for item in protocol.get("important_nonclaims",[]): f.write(f"- {item}\n")
    cols=["dataset","task","response_time_sec","instances","actual_crossing_prevalence","green_rate","yellow_rate","red_rate","red_sensitivity_actual_crossing","yellow_or_red_sensitivity_actual_crossing","false_red_rate_non_crossing","false_yellow_or_red_rate_non_crossing","red_precision"]
    print("\nPRIMARY 90% SAMPLE-LEVEL RESPONSE SENSITIVITY:"); print(primary[cols].to_string(index=False))
    print("\nSECONDARY 95% SAMPLE-LEVEL RESPONSE SENSITIVITY:"); print(secondary[cols].to_string(index=False))
    print("\nSTEP 7B COMPLETE"); print("Outputs:",OUT)

if __name__ == "__main__":
    main()
