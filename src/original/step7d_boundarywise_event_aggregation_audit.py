from pathlib import Path
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
OUT = DOWNLOADS / "step7_response_aware_warning"
INFILE = OUT / "step7b_v2_all_geofence_instance_alarms.csv"

def safe_div(a, b):
    return float(a / b) if b else np.nan

def metric_block(df, alarm_col="alarm_state"):
    true_cross = df["true_crossing_bool"].astype(bool).to_numpy()
    red = df[alarm_col].eq("RED").to_numpy()
    alert = df[alarm_col].isin(["YELLOW", "RED"]).to_numpy()
    non = ~true_cross

    return {
        "instances": int(len(df)),
        "actual_crossings": int(np.sum(true_cross)),
        "actual_crossing_prevalence": safe_div(int(np.sum(true_cross)), len(df)),
        "green_rate": float(np.mean(df[alarm_col].eq("GREEN"))),
        "yellow_rate": float(np.mean(df[alarm_col].eq("YELLOW"))),
        "red_rate": float(np.mean(df[alarm_col].eq("RED"))),
        "red_sensitivity": safe_div(int(np.sum(red & true_cross)), int(np.sum(true_cross))),
        "alert_sensitivity": safe_div(int(np.sum(alert & true_cross)), int(np.sum(true_cross))),
        "false_red_rate": safe_div(int(np.sum(red & non)), int(np.sum(non))),
        "false_alert_rate": safe_div(int(np.sum(alert & non)), int(np.sum(non))),
        "red_precision": safe_div(int(np.sum(red & true_cross)), int(np.sum(red))),
        "alert_precision": safe_div(int(np.sum(alert & true_cross)), int(np.sum(alert))),
    }

def main():
    print("\nSTEP 7D — BOUNDARY-WISE / EVENT-AGGREGATION AUDIT")

    if not INFILE.exists():
        raise FileNotFoundError(INFILE)

    df = pd.read_csv(INFILE)

    required = [
        "dataset","task","sample_index","source_group","track_key",
        "alpha","response_time_sec","alarm_state","true_crossing_bool",
        "uncertainty_intersects_geofence"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # Primary Step-7A level only.
    d = df[np.isclose(pd.to_numeric(df["alpha"]), 0.10)].copy()

    # Identify exact boundary descriptors from existing Step-18 output.
    boundary_cols = [
        c for c in ["axis","boundary","direction","boundary_id"]
        if c in d.columns
    ]
    if not boundary_cols:
        raise RuntimeError("No Step-18 boundary descriptor columns found.")

    print("\nBOUNDARY DESCRIPTORS:")
    print(boundary_cols)

    defs = (
        d[boundary_cols]
        .drop_duplicates()
        .sort_values(boundary_cols)
        .reset_index(drop=True)
    )
    defs.to_csv(OUT / "step7d_boundary_definitions.csv", index=False)

    print("\nUNIQUE BOUNDARY DEFINITIONS:")
    print(defs.to_string(index=False))

    # Verify whether the same boundary set is evaluated for every sample.
    key_cols = ["dataset","task","sample_index","response_time_sec"]
    per_sample_n = (
        d.groupby(key_cols, as_index=False)
        .size()
        .rename(columns={"size":"boundary_instances"})
    )

    boundary_count_audit = (
        per_sample_n.groupby(["dataset","task","response_time_sec"], as_index=False)
        .agg(
            samples=("sample_index","size"),
            min_boundaries=("boundary_instances","min"),
            median_boundaries=("boundary_instances","median"),
            max_boundaries=("boundary_instances","max"),
        )
    )
    boundary_count_audit.to_csv(
        OUT / "step7d_boundary_count_audit.csv",
        index=False
    )

    print("\nBOUNDARY COUNT AUDIT:")
    print(boundary_count_audit.to_string(index=False))

    # Per-boundary metrics: this avoids OR-ing six separate geofence instances.
    rows = []
    group_cols = ["dataset","task","response_time_sec"] + boundary_cols

    for keys, g in d.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(metric_block(g))
        rows.append(row)

    per_boundary = pd.DataFrame(rows)
    per_boundary.to_csv(
        OUT / "step7d_per_boundary_alarm_metrics.csv",
        index=False
    )

    print("\nPER-BOUNDARY PRIMARY 90% METRICS:")
    show = [
        c for c in (
            ["dataset","task","response_time_sec"]
            + boundary_cols
            + ["instances","actual_crossing_prevalence","green_rate",
               "yellow_rate","red_rate","alert_sensitivity",
               "false_alert_rate","alert_precision"]
        )
        if c in per_boundary.columns
    ]
    print(per_boundary[show].to_string(index=False))

    # Macro summary across the six boundary scenarios.
    macro_rows = []
    for keys, g in per_boundary.groupby(["dataset","task","response_time_sec"]):
        dataset, task, response_sec = keys
        macro_rows.append({
            "dataset": dataset,
            "task": task,
            "response_time_sec": float(response_sec),
            "boundaries": int(len(g)),
            "macro_green_rate": float(np.nanmean(g["green_rate"])),
            "macro_yellow_rate": float(np.nanmean(g["yellow_rate"])),
            "macro_red_rate": float(np.nanmean(g["red_rate"])),
            "macro_alert_sensitivity": float(np.nanmean(g["alert_sensitivity"])),
            "macro_false_alert_rate": float(np.nanmean(g["false_alert_rate"])),
            "macro_alert_precision": float(np.nanmean(g["alert_precision"])),
            "min_boundary_false_alert": float(np.nanmin(g["false_alert_rate"])),
            "max_boundary_false_alert": float(np.nanmax(g["false_alert_rate"])),
            "min_boundary_sensitivity": float(np.nanmin(g["alert_sensitivity"])),
            "max_boundary_sensitivity": float(np.nanmax(g["alert_sensitivity"])),
        })

    macro = pd.DataFrame(macro_rows)
    macro.to_csv(
        OUT / "TABLE_step7d_boundary_macro_summary.csv",
        index=False
    )

    print("\nBOUNDARY-MACRO SUMMARY:")
    print(macro.to_string(index=False))

    # How many of the six boundaries alert simultaneously?
    x = d.copy()
    x["is_alert"] = x["alarm_state"].isin(["YELLOW","RED"]).astype(int)
    x["is_yellow"] = x["alarm_state"].eq("YELLOW").astype(int)
    x["is_red"] = x["alarm_state"].eq("RED").astype(int)

    sample_counts = (
        x.groupby(
            ["dataset","task","sample_index","source_group","track_key","response_time_sec"],
            as_index=False
        )
        .agg(
            boundaries=("is_alert","size"),
            alerting_boundaries=("is_alert","sum"),
            yellow_boundaries=("is_yellow","sum"),
            red_boundaries=("is_red","sum"),
            actual_crossing=("true_crossing_bool","max"),
        )
    )

    sample_counts.to_csv(
        OUT / "step7d_sample_alerting_boundary_counts.csv",
        index=False
    )

    dist = (
        sample_counts.groupby(
            ["dataset","task","response_time_sec","actual_crossing","alerting_boundaries"],
            as_index=False
        )
        .size()
        .rename(columns={"size":"samples"})
    )

    totals = (
        dist.groupby(
            ["dataset","task","response_time_sec","actual_crossing"]
        )["samples"]
        .transform("sum")
    )
    dist["fraction"] = dist["samples"] / totals

    dist.to_csv(
        OUT / "step7d_alerting_boundary_count_distribution.csv",
        index=False
    )

    print("\nALERTING-BOUNDARY COUNT DISTRIBUTION:")
    print(dist.to_string(index=False))

    # k-of-N aggregation diagnostic only.
    k_rows = []
    for (dataset, task, response_sec), g in sample_counts.groupby(
        ["dataset","task","response_time_sec"]
    ):
        n_boundaries = int(g["boundaries"].max())

        for k in range(1, n_boundaries + 1):
            true_cross = g["actual_crossing"].astype(bool).to_numpy()
            alert = (g["alerting_boundaries"].to_numpy() >= k)
            non = ~true_cross

            k_rows.append({
                "dataset": dataset,
                "task": task,
                "response_time_sec": float(response_sec),
                "k_required": k,
                "n_boundaries": n_boundaries,
                "alert_rate": float(np.mean(alert)),
                "sensitivity": safe_div(int(np.sum(alert & true_cross)), int(np.sum(true_cross))),
                "false_alert_rate": safe_div(int(np.sum(alert & non)), int(np.sum(non))),
                "precision": safe_div(int(np.sum(alert & true_cross)), int(np.sum(alert))),
            })

    kdf = pd.DataFrame(k_rows)
    kdf.to_csv(
        OUT / "step7d_k_of_n_aggregation_diagnostic.csv",
        index=False
    )

    print("\nK-OF-N AGGREGATION DIAGNOSTIC:")
    print(kdf.to_string(index=False))

    with open(
        OUT / "STEP7D_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("STEP 7D — BOUNDARY-WISE / EVENT-AGGREGATION AUDIT\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            "Purpose: determine whether Step-7B saturation is caused by "
            "collapsing multiple virtual geofence instances with an ANY/maximum-severity rule.\n"
        )
        f.write(
            "This is a secondary diagnostic. k-of-N aggregation is not an operational "
            "rule unless the six boundaries are confirmed to represent simultaneous "
            "physical boundaries rather than separate evaluation scenarios.\n\n"
        )
        f.write("BOUNDARY DEFINITIONS\n--------------------\n")
        f.write(defs.to_string(index=False))
        f.write("\n\nBOUNDARY COUNT AUDIT\n--------------------\n")
        f.write(boundary_count_audit.to_string(index=False))
        f.write("\n\nBOUNDARY-MACRO SUMMARY\n----------------------\n")
        f.write(macro.to_string(index=False))
        f.write("\n\nALERTING-BOUNDARY COUNT DISTRIBUTION\n-----------------------------------\n")
        f.write(dist.to_string(index=False))
        f.write("\n\nK-OF-N DIAGNOSTIC\n-----------------\n")
        f.write(kdf.to_string(index=False))

    print("\nSTEP 7D COMPLETE")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
