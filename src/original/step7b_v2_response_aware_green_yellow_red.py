from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")
OUT = DOWNLOADS / "step7_response_aware_warning"
OUT.mkdir(parents=True, exist_ok=True)

PROTOCOL = OUT / "STEP7A_PROTOCOL_LOCK.json"

GEOFENCE = (
    ROOT / "step18_geofence" / "step18b_evaluation" /
    "step18b_all_geofence_instances.csv"
)

# Exact dataset naming inferred from the Step-18 source excerpts.
BIRDSAI_ROOT = ROOT / "final_neural_trajectory_dataset"
FRESH_ROOT_CANDIDATES = [
    DOWNLOADS / "wildbox_fresh_confirmatory_results",
    ROOT / "wildbox_fresh_confirmatory_results",
]

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
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
        raise RuntimeError(f"Cannot parse booleans: {sorted(low[bad].unique().tolist())[:20]}")
    return low.isin(true_set)

def load_protocol():
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    with open(PROTOCOL, "r", encoding="utf-8") as f:
        return json.load(f)

def task_key(task):
    t = str(task).lower()
    if "2s_to_2s" in t or "2s→2s" in t:
        return "2s_to_2s"
    if "2s_to_5s" in t or "2s→5s" in t:
        return "2s_to_5s"
    raise RuntimeError(f"Unknown task: {task}")

def locate_birdsai_files(task):
    npz = BIRDSAI_ROOT / f"{task}_birdsai_test_canonical.npz"
    meta = BIRDSAI_ROOT / f"{task}_birdsai_test_metadata.csv"
    if not npz.exists():
        raise FileNotFoundError(npz)
    if not meta.exists():
        raise FileNotFoundError(meta)
    return npz, meta

def locate_fresh_files(task):
    tried = []
    for base in FRESH_ROOT_CANDIDATES:
        npz = base / f"{task}_fresh_dataset.npz"
        meta = base / f"{task}_fresh_metadata.csv"
        tried.append((npz, meta))
        if npz.exists() and meta.exists():
            return npz, meta
    raise FileNotFoundError(
        "Fresh NPZ/metadata not found. Tried:\n" +
        "\n".join(f"{a}\n{b}" for a,b in tried)
    )

def get_dataset_files(dataset, task):
    d = str(dataset).lower()
    if "birdsai" in d or "testreal" in d:
        return locate_birdsai_files(task)
    if "wildbox" in d or "fresh" in d:
        return locate_fresh_files(task)
    raise RuntimeError(f"Unknown dataset: {dataset}")

def build_exact_scale_map(geo):
    maps = []
    audits = []

    for (dataset, task), g in geo.groupby(["dataset","task"], dropna=False):
        npz_path, meta_path = get_dataset_files(dataset, task)

        arrays = np.load(npz_path, allow_pickle=False)
        if "scale" not in arrays.files:
            raise RuntimeError(f"'scale' missing from {npz_path}. Keys={arrays.files}")

        scale = np.asarray(arrays["scale"]).reshape(-1)
        meta = pd.read_csv(meta_path, dtype={"source_group":str, "track_key":str})

        unique = (
            g[["sample_index","source_group","track_key"]]
            .drop_duplicates("sample_index")
            .sort_values("sample_index")
            .copy()
        )

        n = len(unique)
        idx = unique["sample_index"].astype(int).to_numpy()

        if len(scale) != len(meta):
            raise RuntimeError(
                f"NPZ scale length != metadata rows for {dataset}/{task}: "
                f"{len(scale)} vs {len(meta)}"
            )

        if len(scale) != n:
            raise RuntimeError(
                f"Step18 sample count mismatch for {dataset}/{task}: "
                f"NPZ={len(scale)}, metadata={len(meta)}, geofence samples={n}"
            )

        if idx.min() != 0 or idx.max() != n-1 or len(np.unique(idx)) != n:
            raise RuntimeError(
                f"sample_index is not exact 0..N-1 for {dataset}/{task}"
            )

        # Exact identity verification using Step-18's metadata.iloc[sample_index] convention.
        geo_track = unique["track_key"].astype(str).str.strip().to_numpy()
        meta_track = meta.iloc[idx]["track_key"].astype(str).str.strip().to_numpy()
        track_agreement = float(np.mean(geo_track == meta_track))

        geo_source = unique["source_group"].astype(str).str.strip().to_numpy()
        meta_source = meta.iloc[idx]["source_group"].astype(str).str.strip().to_numpy()
        source_agreement = float(np.mean(geo_source == meta_source))

        if track_agreement < 0.999999 or source_agreement < 0.999999:
            raise RuntimeError(
                f"Identity mapping failed for {dataset}/{task}: "
                f"track={track_agreement}, source={source_agreement}"
            )

        if not np.all(np.isfinite(scale)) or not np.all(scale > 0):
            raise RuntimeError(f"Invalid scale values in {npz_path}")

        m = unique.copy()
        m["scale"] = scale[idx].astype(float)
        m["dataset"] = dataset
        m["task"] = task
        m["npz_path"] = str(npz_path)
        m["metadata_path"] = str(meta_path)
        maps.append(m)

        audits.append({
            "dataset": dataset,
            "task": task,
            "samples": n,
            "npz_path": str(npz_path),
            "metadata_path": str(meta_path),
            "scale_length": int(len(scale)),
            "metadata_rows": int(len(meta)),
            "sample_index_min": int(idx.min()),
            "sample_index_max": int(idx.max()),
            "track_agreement": track_agreement,
            "source_agreement": source_agreement,
            "scale_min": float(np.min(scale)),
            "scale_median": float(np.median(scale)),
            "scale_max": float(np.max(scale)),
        })

    return pd.concat(maps, ignore_index=True), pd.DataFrame(audits)

def infer_score_orientation(df):
    pred = df["predicted_crossing_bool"].to_numpy(dtype=bool)
    score = pd.to_numeric(df["risk_score"], errors="coerce").to_numpy(dtype=float)
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
        raise RuntimeError(f"Risk-score sign audit failed: {agreements}")
    orientation = 1.0 if best.startswith("positive") else -1.0
    return orientation, best, agreements

def safe_div(a,b):
    return float(a/b) if b else np.nan

def classify_instances(sub, q, alpha, response_sec, orientation):
    d = sub.copy()
    d["absolute_conformal_radius"] = q * d["scale"]
    d["oriented_risk_score"] = orientation * pd.to_numeric(d["risk_score"], errors="coerce")
    d["uncertainty_risk_score"] = d["oriented_risk_score"] + d["absolute_conformal_radius"]
    d["uncertainty_intersects_geofence"] = d["uncertainty_risk_score"] >= 0

    pred_cross = d["predicted_crossing_bool"]
    pred_tte = pd.to_numeric(d["predicted_tte_sec"], errors="coerce")
    dist = pd.to_numeric(d["distance_to_boundary"], errors="coerce")

    red = (
        pred_cross
        & pred_tte.notna()
        & (pred_tte <= response_sec)
    )

    # Already at/inside the boundary but TTE unavailable.
    red = red | (
        pred_cross
        & pred_tte.isna()
        & dist.notna()
        & (dist <= 0)
    )

    yellow = (
        ~red
        & (
            d["uncertainty_intersects_geofence"]
            | pred_cross
        )
    )

    d["alarm_state"] = np.where(
        red, "RED",
        np.where(yellow, "YELLOW", "GREEN")
    )
    d["alpha"] = alpha
    d["target_coverage"] = 1.0 - alpha
    d["response_time_sec"] = response_sec
    return d

def aggregate_sample_level(inst):
    sev_map = {"GREEN":0,"YELLOW":1,"RED":2}
    inv = {0:"GREEN",1:"YELLOW",2:"RED"}

    x = inst.copy()
    x["_sev"] = x["alarm_state"].map(sev_map)

    group_cols = [
        "dataset","task","sample_index","source_group","track_key",
        "alpha","target_coverage","response_time_sec"
    ]

    rows = []
    for keys, g in x.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["alarm_state"] = inv[int(g["_sev"].max())]
        row["true_crossing_bool"] = bool(g["true_crossing_bool"].any())
        true_tte = pd.to_numeric(g["true_tte_sec"], errors="coerce")
        finite = true_tte[np.isfinite(true_tte)]
        row["true_tte_sec"] = float(finite.min()) if len(finite) else np.nan
        row["boundaries_evaluated"] = int(len(g))
        row["uncertainty_intersects_any_geofence"] = bool(
            g["uncertainty_intersects_geofence"].any()
        )
        rows.append(row)

    return pd.DataFrame(rows)

def metric_row(df, dataset, task, alpha, response_sec, level):
    true_cross = df["true_crossing_bool"].to_numpy(dtype=bool)
    red = df["alarm_state"].eq("RED").to_numpy()
    alert = df["alarm_state"].isin(["YELLOW","RED"]).to_numpy()
    non = ~true_cross

    red_tp = int(np.sum(red & true_cross))
    alert_tp = int(np.sum(alert & true_cross))
    red_fp = int(np.sum(red & non))
    alert_fp = int(np.sum(alert & non))
    red_n = int(np.sum(red))
    alert_n = int(np.sum(alert))

    return {
        "level": level,
        "dataset": dataset,
        "task": task,
        "alpha": alpha,
        "target_coverage": 1.0-alpha,
        "response_time_sec": response_sec,
        "instances": int(len(df)),
        "actual_crossings": int(np.sum(true_cross)),
        "actual_crossing_prevalence": safe_div(int(np.sum(true_cross)), len(df)),
        "green_rate": float(np.mean(df["alarm_state"].eq("GREEN"))),
        "yellow_rate": float(np.mean(df["alarm_state"].eq("YELLOW"))),
        "red_rate": float(np.mean(df["alarm_state"].eq("RED"))),
        "red_sensitivity_actual_crossing": safe_div(red_tp, int(np.sum(true_cross))),
        "yellow_or_red_sensitivity_actual_crossing": safe_div(alert_tp, int(np.sum(true_cross))),
        "false_red_rate_non_crossing": safe_div(red_fp, int(np.sum(non))),
        "false_yellow_or_red_rate_non_crossing": safe_div(alert_fp, int(np.sum(non))),
        "red_precision": safe_div(red_tp, red_n),
        "yellow_or_red_precision": safe_div(alert_tp, alert_n),
    }

def track_macro(sample_df):
    rows = []
    for (dataset,task,alpha,response_sec), g in sample_df.groupby(
        ["dataset","task","alpha","response_time_sec"]
    ):
        vals = []
        for track_key, tg in g.groupby("track_key"):
            r = metric_row(tg,dataset,task,alpha,response_sec,"track_internal")
            r["track_key"] = track_key
            vals.append(r)

        t = pd.DataFrame(vals)
        out = {
            "dataset": dataset,
            "task": task,
            "alpha": alpha,
            "target_coverage": 1-alpha,
            "response_time_sec": response_sec,
            "tracks": int(t["track_key"].nunique()),
        }

        for c in [
            "actual_crossing_prevalence",
            "green_rate","yellow_rate","red_rate",
            "red_sensitivity_actual_crossing",
            "yellow_or_red_sensitivity_actual_crossing",
            "false_red_rate_non_crossing",
            "false_yellow_or_red_rate_non_crossing",
            "red_precision","yellow_or_red_precision"
        ]:
            z = pd.to_numeric(t[c], errors="coerce").to_numpy(dtype=float)
            out["track_macro_"+c] = float(np.nanmean(z)) if np.isfinite(z).any() else np.nan

        rows.append(out)

    return pd.DataFrame(rows)

def main():
    print("\nSTEP 7B v2 — RESPONSE-AWARE GREEN / YELLOW / RED EVALUATION")

    protocol = load_protocol()
    print("\nSTEP-7A PROTOCOL LOCK VERIFIED:")
    print(PROTOCOL)

    if not GEOFENCE.exists():
        raise FileNotFoundError(GEOFENCE)

    geo = pd.read_csv(
        GEOFENCE,
        dtype={"source_group":str,"track_key":str}
    )

    required = [
        "dataset","task","sample_index","source_group","track_key",
        "distance_to_boundary","true_crossing","predicted_crossing",
        "risk_score","true_tte_sec","predicted_tte_sec"
    ]
    missing = [c for c in required if c not in geo.columns]
    if missing:
        raise RuntimeError(f"Missing geofence columns: {missing}")

    geo["true_crossing_bool"] = parse_bool_series(geo["true_crossing"])
    geo["predicted_crossing_bool"] = parse_bool_series(geo["predicted_crossing"])

    orientation, rule, agreements = infer_score_orientation(geo)

    print("\nRISK-SCORE SIGN AUDIT:")
    print("selected_rule =", rule)
    print("orientation =", orientation)
    print("agreements =", agreements)

    scale_map, scale_audit = build_exact_scale_map(geo)

    scale_audit.to_csv(
        OUT / "step7b_v2_exact_scale_mapping_audit.csv",
        index=False
    )

    scale_map.to_csv(
        OUT / "step7b_v2_exact_sample_scale_map.csv",
        index=False
    )

    print("\nEXACT STEP-18 SCALE MAPPING VERIFIED:")
    print(scale_audit.to_string(index=False))

    geo = geo.merge(
        scale_map[
            ["dataset","task","sample_index","scale","npz_path","metadata_path"]
        ],
        on=["dataset","task","sample_index"],
        how="left",
        validate="many_to_one"
    )

    if geo["scale"].isna().any():
        raise RuntimeError("Scale mapping left missing values.")

    all_inst = []
    all_samples = []
    metrics = []

    for (dataset,task), sub in geo.groupby(["dataset","task"]):
        tk = task_key(task)
        qinfo = protocol["locked_step6c_q_values"][tk]
        response_grid = protocol["response_time_sensitivity_seconds"][tk]

        for alpha, qname in [(0.10,"q90"),(0.05,"q95")]:
            q = float(qinfo[qname])

            for response_sec in response_grid:
                inst = classify_instances(
                    sub,
                    q=q,
                    alpha=alpha,
                    response_sec=float(response_sec),
                    orientation=orientation
                )
                all_inst.append(inst)

                metrics.append(
                    metric_row(
                        inst,dataset,task,alpha,float(response_sec),
                        "geofence_instance"
                    )
                )

                samples = aggregate_sample_level(inst)
                all_samples.append(samples)

                metrics.append(
                    metric_row(
                        samples,dataset,task,alpha,float(response_sec),
                        "sample_max_severity"
                    )
                )

    inst_all = pd.concat(all_inst, ignore_index=True)
    sample_all = pd.concat(all_samples, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)
    track_df = track_macro(sample_all)

    inst_all.to_csv(
        OUT / "step7b_v2_all_geofence_instance_alarms.csv",
        index=False
    )
    sample_all.to_csv(
        OUT / "step7b_v2_sample_level_max_alarm.csv",
        index=False
    )
    metrics_df.to_csv(
        OUT / "step7b_v2_response_time_sensitivity_metrics.csv",
        index=False
    )
    track_df.to_csv(
        OUT / "step7b_v2_track_macro_metrics.csv",
        index=False
    )

    primary = metrics_df[
        (metrics_df["level"]=="sample_max_severity")
        & np.isclose(metrics_df["alpha"],0.10)
    ].copy()

    secondary = metrics_df[
        (metrics_df["level"]=="sample_max_severity")
        & np.isclose(metrics_df["alpha"],0.05)
    ].copy()

    primary.to_csv(
        OUT / "TABLE_step7b_v2_primary_90pct_response_sensitivity.csv",
        index=False
    )
    secondary.to_csv(
        OUT / "TABLE_step7b_v2_secondary_95pct_response_sensitivity.csv",
        index=False
    )

    provenance = {
        "step":"STEP_7B_V2_RESPONSE_AWARE_WARNING",
        "step7a_protocol_sha256":sha256_file(PROTOCOL),
        "step18_geofence_instances_sha256":sha256_file(GEOFENCE),
        "risk_score_rule":rule,
        "risk_score_orientation":orientation,
        "risk_score_agreements":agreements,
        "scale_mapping_rule":"Exact Step-18 convention: sample_index directly indexes arrays['scale'] and metadata.iloc[sample_index].",
        "scope":"Retrospective virtual-geofence decision-layer simulation. Not BAMBI confirmatory coverage; not field validation."
    }

    with open(OUT/"STEP7B_V2_PROVENANCE.json","w",encoding="utf-8") as f:
        json.dump(provenance,f,indent=2)

    with open(OUT/"STEP7B_V2_SUMMARY.txt","w",encoding="utf-8") as f:
        f.write("STEP 7B v2 — RESPONSE-AWARE GREEN / YELLOW / RED EVALUATION\n")
        f.write("="*82+"\n\n")
        f.write("RISK-SCORE SIGN AUDIT\n---------------------\n")
        f.write(f"rule={rule}\norientation={orientation}\n")
        f.write(json.dumps(agreements,indent=2))
        f.write("\n\nEXACT SCALE MAPPING AUDIT\n-------------------------\n")
        f.write(scale_audit.to_string(index=False))
        f.write("\n\nPRIMARY 90% SAMPLE-LEVEL RESPONSE SENSITIVITY\n---------------------------------------------\n")
        f.write(primary.to_string(index=False))
        f.write("\n\nSECONDARY 95% SAMPLE-LEVEL RESPONSE SENSITIVITY\n-----------------------------------------------\n")
        f.write(secondary.to_string(index=False))
        f.write("\n\nTRACK-MACRO METRICS\n-------------------\n")
        f.write(track_df.to_string(index=False))

    display_cols = [
        "dataset","task","response_time_sec","instances",
        "actual_crossing_prevalence",
        "green_rate","yellow_rate","red_rate",
        "red_sensitivity_actual_crossing",
        "yellow_or_red_sensitivity_actual_crossing",
        "false_red_rate_non_crossing",
        "false_yellow_or_red_rate_non_crossing",
        "red_precision","yellow_or_red_precision"
    ]

    print("\nPRIMARY 90% SAMPLE-LEVEL RESPONSE SENSITIVITY:")
    print(primary[display_cols].to_string(index=False))

    print("\nSECONDARY 95% SAMPLE-LEVEL RESPONSE SENSITIVITY:")
    print(secondary[display_cols].to_string(index=False))

    print("\nSTEP 7B v2 COMPLETE")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
