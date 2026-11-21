from pathlib import Path
import json
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
OUT = DOWNLOADS / "step7_response_aware_warning"
INFILE = OUT / "step7b_v2_all_geofence_instance_alarms.csv"

OUT.mkdir(parents=True, exist_ok=True)

def parse_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    x = s.astype(str).str.strip().str.lower()
    true_vals = {"true","1","yes","y","t"}
    false_vals = {"false","0","no","n","f"}
    bad = ~x.isin(true_vals | false_vals)
    if bad.any():
        raise RuntimeError(f"Unparseable booleans: {sorted(x[bad].unique())[:20]}")
    return x.isin(true_vals)

def safe_div(a, b):
    return float(a / b) if b else np.nan

def metrics(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    pos = y
    neg = ~y

    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    fn = int(np.sum((~pred) & pos))
    tn = int(np.sum((~pred) & neg))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": safe_div(tp, int(np.sum(pos))),
        "false_alert_rate": safe_div(fp, int(np.sum(neg))),
        "precision": safe_div(tp, int(np.sum(pred))),
        "specificity": safe_div(tn, int(np.sum(neg))),
    }

def main():
    print("\nSTEP 7E — PER-GEOFENCE INCREMENTAL VALUE OF CONFORMAL UNCERTAINTY")

    if not INFILE.exists():
        raise FileNotFoundError(INFILE)

    df = pd.read_csv(INFILE)

    required = [
        "dataset","task","sample_index","source_group","track_key",
        "axis","boundary","direction","boundary_id",
        "alpha","target_coverage",
        "true_crossing_bool","predicted_crossing_bool",
        "uncertainty_intersects_geofence",
        "distance_to_boundary","scale",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df["true_crossing_bool"] = parse_bool(df["true_crossing_bool"])
    df["predicted_crossing_bool"] = parse_bool(df["predicted_crossing_bool"])
    df["uncertainty_intersects_geofence"] = parse_bool(
        df["uncertainty_intersects_geofence"]
    )

    # Step-7B repeats the same boundary prediction for each response-time setting.
    # Remove those duplicates before evaluating the intrinsic boundary decision layer.
    dedup_cols = [
        "dataset","task","sample_index","source_group","track_key",
        "axis","boundary","direction","boundary_id","alpha"
    ]

    d = (
        df.sort_values("response_time_sec")
        .drop_duplicates(dedup_cols, keep="first")
        .copy()
    )

    # Sanity-check deduplication.
    expected = (
        df.groupby(["dataset","task","alpha"])
        ["response_time_sec"].nunique()
        .reset_index(name="response_settings")
    )
    expected.to_csv(
        OUT / "step7e_response_setting_audit.csv",
        index=False
    )

    # Normalized current distance to boundary for diagnostics.
    dist = pd.to_numeric(d["distance_to_boundary"], errors="coerce")
    scale = pd.to_numeric(d["scale"], errors="coerce")
    if scale.isna().any() or (scale <= 0).any():
        raise RuntimeError("Invalid scale values.")
    d["normalized_current_distance"] = dist / scale

    rows = []
    per_boundary_rows = []
    source_rows = []

    group_cols = ["dataset","task","alpha"]

    for keys, g in d.groupby(group_cols):
        dataset, task, alpha = keys
        y = g["true_crossing_bool"].to_numpy(dtype=bool)
        nominal = g["predicted_crossing_bool"].to_numpy(dtype=bool)

        # Conformal alert means the calibrated uncertainty set reaches/intersects
        # the geofence. Nominal crossings are already contained within this idea,
        # but take OR explicitly for safety/clarity.
        conformal = (
            g["uncertainty_intersects_geofence"].to_numpy(dtype=bool)
            | nominal
        )

        m_nom = metrics(y, nominal)
        m_conf = metrics(y, conformal)

        nominal_miss = y & (~nominal)
        rescued = nominal_miss & conformal

        true_neg_nominal = (~y) & (~nominal)
        added_false = true_neg_nominal & conformal

        incremental = conformal & (~nominal)
        incremental_true = incremental & y
        incremental_false = incremental & (~y)

        rows.append({
            "dataset": dataset,
            "task": task,
            "alpha": float(alpha),
            "target_coverage": 1.0 - float(alpha),
            "instances": int(len(g)),
            "actual_crossings": int(np.sum(y)),
            "nominal_sensitivity": m_nom["sensitivity"],
            "nominal_false_alert_rate": m_nom["false_alert_rate"],
            "nominal_precision": m_nom["precision"],
            "conformal_sensitivity": m_conf["sensitivity"],
            "conformal_false_alert_rate": m_conf["false_alert_rate"],
            "conformal_precision": m_conf["precision"],
            "delta_sensitivity": (
                m_conf["sensitivity"] - m_nom["sensitivity"]
                if np.isfinite(m_conf["sensitivity"]) and np.isfinite(m_nom["sensitivity"])
                else np.nan
            ),
            "delta_false_alert_rate": (
                m_conf["false_alert_rate"] - m_nom["false_alert_rate"]
                if np.isfinite(m_conf["false_alert_rate"]) and np.isfinite(m_nom["false_alert_rate"])
                else np.nan
            ),
            "nominal_missed_true_crossings": int(np.sum(nominal_miss)),
            "rescued_true_crossings_by_uncertainty": int(np.sum(rescued)),
            "rescue_rate_among_nominal_misses": safe_div(
                int(np.sum(rescued)), int(np.sum(nominal_miss))
            ),
            "nominal_true_negatives": int(np.sum(true_neg_nominal)),
            "added_false_alerts_from_uncertainty": int(np.sum(added_false)),
            "added_false_alert_rate_among_nominal_true_negatives": safe_div(
                int(np.sum(added_false)), int(np.sum(true_neg_nominal))
            ),
            "incremental_alerts_total": int(np.sum(incremental)),
            "incremental_alerts_true": int(np.sum(incremental_true)),
            "incremental_alerts_false": int(np.sum(incremental_false)),
            "incremental_alert_precision": safe_div(
                int(np.sum(incremental_true)), int(np.sum(incremental))
            ),
            "false_alerts_per_rescued_crossing": safe_div(
                int(np.sum(incremental_false)), int(np.sum(rescued))
            ),
        })

        # Per-boundary breakdown.
        bcols = ["axis","boundary","direction","boundary_id"]
        for bkeys, bg in g.groupby(bcols, dropna=False):
            axis, boundary, direction, boundary_id = bkeys
            by = bg["true_crossing_bool"].to_numpy(dtype=bool)
            bn = bg["predicted_crossing_bool"].to_numpy(dtype=bool)
            bc = (
                bg["uncertainty_intersects_geofence"].to_numpy(dtype=bool)
                | bn
            )
            mn = metrics(by, bn)
            mc = metrics(by, bc)
            miss = by & (~bn)
            rescue = miss & bc
            incr = bc & (~bn)
            incr_true = incr & by
            incr_false = incr & (~by)

            per_boundary_rows.append({
                "dataset": dataset,
                "task": task,
                "alpha": float(alpha),
                "target_coverage": 1.0 - float(alpha),
                "axis": axis,
                "boundary": boundary,
                "direction": direction,
                "boundary_id": boundary_id,
                "instances": int(len(bg)),
                "actual_crossings": int(np.sum(by)),
                "nominal_sensitivity": mn["sensitivity"],
                "nominal_false_alert_rate": mn["false_alert_rate"],
                "nominal_precision": mn["precision"],
                "conformal_sensitivity": mc["sensitivity"],
                "conformal_false_alert_rate": mc["false_alert_rate"],
                "conformal_precision": mc["precision"],
                "rescued_true_crossings": int(np.sum(rescue)),
                "nominal_misses": int(np.sum(miss)),
                "rescue_rate_among_nominal_misses": safe_div(
                    int(np.sum(rescue)), int(np.sum(miss))
                ),
                "incremental_alerts": int(np.sum(incr)),
                "incremental_true": int(np.sum(incr_true)),
                "incremental_false": int(np.sum(incr_false)),
                "incremental_alert_precision": safe_div(
                    int(np.sum(incr_true)), int(np.sum(incr))
                ),
                "false_alerts_per_rescued_crossing": safe_div(
                    int(np.sum(incr_false)), int(np.sum(rescue))
                ),
            })

        # Per-source transfer/stability breakdown.
        for source, sg in g.groupby("source_group", dropna=False):
            sy = sg["true_crossing_bool"].to_numpy(dtype=bool)
            sn = sg["predicted_crossing_bool"].to_numpy(dtype=bool)
            sc = (
                sg["uncertainty_intersects_geofence"].to_numpy(dtype=bool)
                | sn
            )
            mn = metrics(sy, sn)
            mc = metrics(sy, sc)
            miss = sy & (~sn)
            rescue = miss & sc
            incr = sc & (~sn)

            source_rows.append({
                "dataset": dataset,
                "task": task,
                "alpha": float(alpha),
                "source_group": source,
                "instances": int(len(sg)),
                "actual_crossings": int(np.sum(sy)),
                "nominal_sensitivity": mn["sensitivity"],
                "nominal_false_alert_rate": mn["false_alert_rate"],
                "conformal_sensitivity": mc["sensitivity"],
                "conformal_false_alert_rate": mc["false_alert_rate"],
                "rescued_true_crossings": int(np.sum(rescue)),
                "nominal_misses": int(np.sum(miss)),
                "incremental_alert_precision": safe_div(
                    int(np.sum(incr & sy)), int(np.sum(incr))
                ),
            })

    overall = pd.DataFrame(rows)
    per_boundary = pd.DataFrame(per_boundary_rows)
    per_source = pd.DataFrame(source_rows)

    overall.to_csv(
        OUT / "TABLE_step7e_incremental_uncertainty_value_overall.csv",
        index=False
    )
    per_boundary.to_csv(
        OUT / "step7e_incremental_uncertainty_value_per_boundary.csv",
        index=False
    )
    per_source.to_csv(
        OUT / "step7e_incremental_uncertainty_value_per_source.csv",
        index=False
    )

    # Distance-stratified diagnostic for primary 90% only.
    d90 = d[np.isclose(d["alpha"], 0.10)].copy()

    # Use fixed normalized-distance bands, not outcome-tuned thresholds.
    bins = [-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf]
    labels = ["<=0.5", "0.5-1", "1-2", "2-4", ">4"]
    d90["distance_band_scale_units"] = pd.cut(
        d90["normalized_current_distance"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    distance_rows = []
    for keys, g in d90.groupby(
        ["dataset","task","distance_band_scale_units"],
        observed=True,
        dropna=False
    ):
        dataset, task, band = keys
        y = g["true_crossing_bool"].to_numpy(dtype=bool)
        n = g["predicted_crossing_bool"].to_numpy(dtype=bool)
        c = g["uncertainty_intersects_geofence"].to_numpy(dtype=bool) | n
        mn = metrics(y, n)
        mc = metrics(y, c)

        distance_rows.append({
            "dataset": dataset,
            "task": task,
            "distance_band_scale_units": str(band),
            "instances": int(len(g)),
            "actual_crossing_prevalence": safe_div(int(np.sum(y)), len(g)),
            "nominal_sensitivity": mn["sensitivity"],
            "nominal_false_alert_rate": mn["false_alert_rate"],
            "conformal_sensitivity": mc["sensitivity"],
            "conformal_false_alert_rate": mc["false_alert_rate"],
            "conformal_precision": mc["precision"],
        })

    distance_df = pd.DataFrame(distance_rows)
    distance_df.to_csv(
        OUT / "step7e_distance_stratified_diagnostic.csv",
        index=False
    )

    # Primary 90% compact table.
    primary = overall[np.isclose(overall["alpha"], 0.10)].copy()
    secondary = overall[np.isclose(overall["alpha"], 0.05)].copy()

    with open(OUT / "STEP7E_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("STEP 7E — PER-GEOFENCE INCREMENTAL VALUE OF CONFORMAL UNCERTAINTY\n")
        f.write("=" * 86 + "\n\n")
        f.write(
            "Purpose: quantify how many nominal forecast misses are rescued by "
            "the conformal uncertainty region, and how many additional false alerts "
            "are introduced. Evaluation is per geofence instance, not sample-level "
            "union across separate virtual-geofence scenarios.\n\n"
        )
        f.write("PRIMARY 90% OVERALL\n")
        f.write("-------------------\n")
        f.write(primary.to_string(index=False))
        f.write("\n\nSECONDARY 95% OVERALL\n")
        f.write("---------------------\n")
        f.write(secondary.to_string(index=False))
        f.write("\n\nPRIMARY 90% DISTANCE-STRATIFIED DIAGNOSTIC\n")
        f.write("------------------------------------------\n")
        f.write(distance_df.to_string(index=False))
        f.write("\n\nINTERPRETATION NOTE\n")
        f.write("-------------------\n")
        f.write(
            "A high rescue rate is not sufficient by itself. The operational value "
            "depends on the incremental-alert precision and the number of extra false "
            "alerts per rescued crossing. No new operational threshold is selected here.\n"
        )

    show_cols = [
        "dataset","task","target_coverage","instances","actual_crossings",
        "nominal_sensitivity","nominal_false_alert_rate","nominal_precision",
        "conformal_sensitivity","conformal_false_alert_rate","conformal_precision",
        "nominal_missed_true_crossings","rescued_true_crossings_by_uncertainty",
        "rescue_rate_among_nominal_misses",
        "incremental_alerts_total","incremental_alerts_true","incremental_alerts_false",
        "incremental_alert_precision","false_alerts_per_rescued_crossing"
    ]

    print("\nPRIMARY 90% INCREMENTAL UNCERTAINTY VALUE:")
    print(primary[show_cols].to_string(index=False))

    print("\nSECONDARY 95% INCREMENTAL UNCERTAINTY VALUE:")
    print(secondary[show_cols].to_string(index=False))

    print("\nPRIMARY 90% DISTANCE-STRATIFIED DIAGNOSTIC:")
    print(distance_df.to_string(index=False))

    print("\nSTEP 7E COMPLETE")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
