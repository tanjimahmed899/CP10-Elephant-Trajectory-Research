from pathlib import Path
import json
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
OUT = DOWNLOADS / "step7_response_aware_warning"
INFILE = OUT / "step7b_v2_all_geofence_instance_alarms.csv"

THRESHOLDS = [-1.0, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0]
FALSE_ALERT_CAPS = [0.10, 0.20, 0.30]

def safe_div(a, b):
    return float(a / b) if b else np.nan

def aggregate_samples(inst):
    sev = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    inv = {0: "GREEN", 1: "YELLOW", 2: "RED"}

    x = inst.copy()
    x["_sev"] = x["alarm_state_c"].map(sev)

    group_cols = [
        "dataset", "task", "sample_index", "source_group", "track_key",
        "alpha", "target_coverage", "response_time_sec", "penetration_threshold"
    ]

    rows = []
    for keys, g in x.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["alarm_state_c"] = inv[int(g["_sev"].max())]
        row["true_crossing_bool"] = bool(g["true_crossing_bool"].any())
        row["num_boundaries"] = int(len(g))
        row["num_uncertainty_intersections"] = int(
            np.sum(g["uncertainty_intersects_geofence"].astype(bool))
        )
        row["max_penetration_ratio"] = float(
            np.nanmax(pd.to_numeric(g["penetration_ratio"], errors="coerce"))
        )
        rows.append(row)

    return pd.DataFrame(rows)

def metrics(df):
    true_cross = df["true_crossing_bool"].astype(bool).to_numpy()
    red = df["alarm_state_c"].eq("RED").to_numpy()
    alert = df["alarm_state_c"].isin(["YELLOW", "RED"]).to_numpy()
    non = ~true_cross

    red_tp = int(np.sum(red & true_cross))
    alert_tp = int(np.sum(alert & true_cross))
    red_fp = int(np.sum(red & non))
    alert_fp = int(np.sum(alert & non))

    return {
        "samples": int(len(df)),
        "actual_crossings": int(np.sum(true_cross)),
        "actual_crossing_prevalence": safe_div(int(np.sum(true_cross)), len(df)),
        "green_rate": float(np.mean(df["alarm_state_c"].eq("GREEN"))),
        "yellow_rate": float(np.mean(df["alarm_state_c"].eq("YELLOW"))),
        "red_rate": float(np.mean(df["alarm_state_c"].eq("RED"))),
        "red_sensitivity_actual_crossing": safe_div(red_tp, int(np.sum(true_cross))),
        "yellow_or_red_sensitivity_actual_crossing": safe_div(
            alert_tp, int(np.sum(true_cross))
        ),
        "false_red_rate_non_crossing": safe_div(red_fp, int(np.sum(non))),
        "false_yellow_or_red_rate_non_crossing": safe_div(
            alert_fp, int(np.sum(non))
        ),
        "red_precision": safe_div(red_tp, int(np.sum(red))),
        "yellow_or_red_precision": safe_div(alert_tp, int(np.sum(alert))),
    }

def main():
    print("\nSTEP 7C — ALARM-SATURATION / UNCERTAINTY-PENETRATION DIAGNOSTIC")

    if not INFILE.exists():
        raise FileNotFoundError(INFILE)

    df = pd.read_csv(INFILE)

    required = [
        "dataset", "task", "sample_index", "source_group", "track_key",
        "alpha", "target_coverage", "response_time_sec",
        "absolute_conformal_radius", "oriented_risk_score",
        "uncertainty_intersects_geofence", "alarm_state",
        "true_crossing_bool"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # Primary diagnostic uses the Step-7A primary 90% conformal level.
    d90 = df[np.isclose(pd.to_numeric(df["alpha"]), 0.10)].copy()

    radius = pd.to_numeric(d90["absolute_conformal_radius"], errors="coerce")
    risk = pd.to_numeric(d90["oriented_risk_score"], errors="coerce")

    if (radius <= 0).any() or radius.isna().any():
        raise RuntimeError("Invalid conformal radius.")

    d90["penetration_ratio"] = risk / radius

    # Interpretation:
    # current Step-7B uncertainty intersection == risk + radius >= 0
    # => risk/radius >= -1.
    current_rule = (
        d90["penetration_ratio"] >= -1.0
    )
    stored_rule = d90["uncertainty_intersects_geofence"].astype(bool)

    agreement = float(np.mean(current_rule.to_numpy() == stored_rule.to_numpy()))
    if agreement < 0.999999:
        raise RuntimeError(
            f"Penetration-ratio reconstruction failed: agreement={agreement}"
        )

    print("\nCURRENT STEP-7B RULE RECONSTRUCTION:")
    print("uncertainty intersection <=> penetration_ratio >= -1")
    print("agreement =", agreement)

    all_samples = []
    metric_rows = []

    for threshold in THRESHOLDS:
        x = d90.copy()
        x["penetration_threshold"] = float(threshold)

        # Preserve RED exactly as Step-7B defined it.
        red = x["alarm_state"].eq("RED")

        # Only alter the uncertainty-driven YELLOW operating threshold.
        yellow = (
            ~red
            & (
                x["penetration_ratio"] >= threshold
            )
        )

        x["alarm_state_c"] = np.where(
            red,
            "RED",
            np.where(yellow, "YELLOW", "GREEN")
        )

        s = aggregate_samples(x)
        all_samples.append(s)

        for keys, g in s.groupby(
            ["dataset", "task", "response_time_sec", "penetration_threshold"]
        ):
            dataset, task, response_sec, th = keys
            row = {
                "dataset": dataset,
                "task": task,
                "response_time_sec": float(response_sec),
                "penetration_threshold": float(th),
            }
            row.update(metrics(g))
            metric_rows.append(row)

    sample_all = pd.concat(all_samples, ignore_index=True)
    metric_df = pd.DataFrame(metric_rows)

    sample_all.to_csv(
        OUT / "step7c_sample_level_threshold_sweep.csv",
        index=False
    )
    metric_df.to_csv(
        OUT / "step7c_threshold_sweep_metrics.csv",
        index=False
    )

    # Saturation diagnostic at the ORIGINAL Step-7B operating rule (-1).
    baseline = metric_df[
        np.isclose(metric_df["penetration_threshold"], -1.0)
    ].copy()

    baseline.to_csv(
        OUT / "TABLE_step7c_current_rule_saturation.csv",
        index=False
    )

    # Use BIRDSAI TestReal only to identify threshold candidates under false-alert caps.
    dev = metric_df[
        metric_df["dataset"].astype(str).str.contains(
            "BIRDSAI", case=False, na=False
        )
    ].copy()

    selections = []

    for (task, response_sec), g in dev.groupby(["task", "response_time_sec"]):
        for cap in FALSE_ALERT_CAPS:
            feasible = g[
                g["false_yellow_or_red_rate_non_crossing"] <= cap
            ].copy()

            if len(feasible) == 0:
                selections.append({
                    "task": task,
                    "response_time_sec": float(response_sec),
                    "false_alert_cap": cap,
                    "feasible": False,
                    "selected_threshold": np.nan,
                    "dev_sensitivity": np.nan,
                    "dev_false_alert": np.nan,
                })
                continue

            # Maximize sensitivity, then choose stricter/larger threshold if tied.
            feasible = feasible.sort_values(
                [
                    "yellow_or_red_sensitivity_actual_crossing",
                    "penetration_threshold"
                ],
                ascending=[False, False]
            )

            best = feasible.iloc[0]

            selections.append({
                "task": task,
                "response_time_sec": float(response_sec),
                "false_alert_cap": cap,
                "feasible": True,
                "selected_threshold": float(best["penetration_threshold"]),
                "dev_sensitivity": float(
                    best["yellow_or_red_sensitivity_actual_crossing"]
                ),
                "dev_false_alert": float(
                    best["false_yellow_or_red_rate_non_crossing"]
                ),
            })

    sel = pd.DataFrame(selections)

    # Apply those fixed development-selected thresholds to WildBox Fresh.
    fresh = metric_df[
        metric_df["dataset"].astype(str).str.contains(
            "WildBox", case=False, na=False
        )
    ].copy()

    ext_rows = []
    for _, r in sel.iterrows():
        if not bool(r["feasible"]):
            continue

        m = fresh[
            (fresh["task"] == r["task"])
            & np.isclose(fresh["response_time_sec"], r["response_time_sec"])
            & np.isclose(
                fresh["penetration_threshold"],
                r["selected_threshold"]
            )
        ]

        if len(m) != 1:
            continue

        q = m.iloc[0]

        ext_rows.append({
            "task": r["task"],
            "response_time_sec": r["response_time_sec"],
            "false_alert_cap": r["false_alert_cap"],
            "selected_threshold_from_BIRDSAI": r["selected_threshold"],
            "BIRDSAI_sensitivity": r["dev_sensitivity"],
            "BIRDSAI_false_alert": r["dev_false_alert"],
            "WildBox_sensitivity": float(
                q["yellow_or_red_sensitivity_actual_crossing"]
            ),
            "WildBox_false_alert": float(
                q["false_yellow_or_red_rate_non_crossing"]
            ),
            "WildBox_green_rate": float(q["green_rate"]),
            "WildBox_yellow_rate": float(q["yellow_rate"]),
            "WildBox_red_rate": float(q["red_rate"]),
            "WildBox_alert_precision": float(
                q["yellow_or_red_precision"]
            ),
        })

    ext = pd.DataFrame(ext_rows)

    sel.to_csv(
        OUT / "step7c_BIRDSAI_operating_point_candidates.csv",
        index=False
    )
    ext.to_csv(
        OUT / "TABLE_step7c_fixed_threshold_WildBox_diagnostic.csv",
        index=False
    )

    # Boundary-union saturation diagnostic.
    union_stats = (
        d90.groupby(
            ["dataset", "task", "sample_index", "response_time_sec"],
            as_index=False
        )
        .agg(
            boundary_instances=("sample_index", "size"),
            uncertainty_intersections=(
                "uncertainty_intersects_geofence", "sum"
            )
        )
    )

    union_summary = (
        union_stats.groupby(
            ["dataset", "task", "response_time_sec"],
            as_index=False
        )
        .agg(
            samples=("sample_index", "size"),
            mean_boundaries=("boundary_instances", "mean"),
            mean_uncertainty_intersections=(
                "uncertainty_intersections", "mean"
            ),
            median_uncertainty_intersections=(
                "uncertainty_intersections", "median"
            ),
            fraction_any_uncertainty_intersection=(
                "uncertainty_intersections",
                lambda s: float(np.mean(np.asarray(s) > 0))
            ),
            fraction_all_boundaries_intersect=(
                "uncertainty_intersections",
                lambda s: float(
                    np.mean(
                        np.asarray(s)
                        == union_stats.loc[s.index, "boundary_instances"].to_numpy()
                    )
                )
            )
        )
    )

    union_summary.to_csv(
        OUT / "step7c_multiboundary_union_saturation.csv",
        index=False
    )

    with open(
        OUT / "STEP7C_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 7C — ALARM-SATURATION / UNCERTAINTY-PENETRATION DIAGNOSTIC\n"
        )
        f.write("=" * 84 + "\n\n")
        f.write(
            "This is a secondary diagnostic motivated by Step-7B alarm saturation.\n"
        )
        f.write(
            "It does not replace the locked Step-7A primary analysis and does not "
            "restore a conformal coverage guarantee for the operational alarm threshold.\n\n"
        )
        f.write(
            f"Original uncertainty-rule reconstruction agreement: {agreement:.6f}\n"
        )
        f.write(
            "Original Step-7B uncertainty intersection is penetration_ratio >= -1.\n\n"
        )
        f.write("CURRENT-RULE SATURATION\n")
        f.write("-----------------------\n")
        f.write(baseline.to_string(index=False))
        f.write("\n\nMULTI-BOUNDARY UNION SATURATION\n")
        f.write("-------------------------------\n")
        f.write(union_summary.to_string(index=False))
        f.write("\n\nBIRDSAI OPERATING-POINT CANDIDATES\n")
        f.write("----------------------------------\n")
        f.write(sel.to_string(index=False))
        f.write("\n\nFIXED-THRESHOLD WILDBOX DIAGNOSTIC\n")
        f.write("-----------------------------------\n")
        f.write(ext.to_string(index=False) if len(ext) else "NONE")

    print("\nCURRENT-RULE SATURATION:")
    cols = [
        "dataset", "task", "response_time_sec",
        "green_rate", "yellow_rate", "red_rate",
        "yellow_or_red_sensitivity_actual_crossing",
        "false_yellow_or_red_rate_non_crossing",
        "yellow_or_red_precision"
    ]
    print(baseline[cols].to_string(index=False))

    print("\nMULTI-BOUNDARY UNION SATURATION:")
    print(union_summary.to_string(index=False))

    print("\nBIRDSAI OPERATING-POINT CANDIDATES:")
    print(sel.to_string(index=False))

    print("\nFIXED-THRESHOLD WILDBOX DIAGNOSTIC:")
    print(ext.to_string(index=False) if len(ext) else "NONE")

    print("\nSTEP 7C COMPLETE")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
