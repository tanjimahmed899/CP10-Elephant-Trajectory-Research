from pathlib import Path
import json
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
OUT = DOWNLOADS / "step7_response_aware_warning"
INFILE = OUT / "step7b_v2_all_geofence_instance_alarms.csv"

OUT.mkdir(parents=True, exist_ok=True)

PERSISTENCE_WINDOWS = [1, 2, 3, 5]

def parse_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    x = s.astype(str).str.strip().str.lower()
    true_vals = {"true","1","yes","y","t"}
    false_vals = {"false","0","no","n","f"}
    bad = ~x.isin(true_vals | false_vals)
    if bad.any():
        raise RuntimeError(
            f"Unparseable booleans: {sorted(x[bad].unique())[:20]}"
        )
    return x.isin(true_vals)

def safe_div(a, b):
    return float(a / b) if b else np.nan

def classify_metrics(y, alert):
    y = np.asarray(y, dtype=bool)
    alert = np.asarray(alert, dtype=bool)
    pos = y
    neg = ~y

    tp = int(np.sum(alert & pos))
    fp = int(np.sum(alert & neg))
    fn = int(np.sum((~alert) & pos))
    tn = int(np.sum((~alert) & neg))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": safe_div(tp, int(np.sum(pos))),
        "false_alert_rate": safe_div(fp, int(np.sum(neg))),
        "precision": safe_div(tp, int(np.sum(alert))),
        "specificity": safe_div(tn, int(np.sum(neg))),
    }

def longest_true_run(arr):
    arr = np.asarray(arr, dtype=bool)
    best = 0
    cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)

def apply_persistence(g, n):
    """
    Ordered-window persistence within one source/track/geofence scenario.
    The conformal set is unchanged. Only the operational Yellow trigger is
    required to persist for n consecutive available forecast windows.
    """
    g = g.sort_values("sample_index").copy()

    u = g["uncertainty_intersects_geofence"].astype(bool)

    if n == 1:
        persistent = u.copy()
    else:
        persistent = (
            u.astype(int)
            .rolling(window=n, min_periods=n)
            .sum()
            .eq(n)
        )

    nominal = g["predicted_crossing_bool"].astype(bool)

    # Red remains the nominal predicted crossing.
    # Yellow = persistent uncertainty-only warning.
    red = nominal
    yellow = (~red) & persistent

    g["persistent_uncertainty"] = persistent.to_numpy(dtype=bool)
    g["alarm_state_persistent"] = np.where(
        red,
        "RED",
        np.where(yellow, "YELLOW", "GREEN")
    )
    g["persistence_windows"] = int(n)

    return g

def main():
    print("\nSTEP 7F — TEMPORAL PERSISTENCE / HYSTERESIS DIAGNOSTIC")

    if not INFILE.exists():
        raise FileNotFoundError(INFILE)

    df = pd.read_csv(INFILE)

    required = [
        "dataset","task","sample_index","source_group","track_key",
        "axis","boundary","direction","boundary_id",
        "alpha","response_time_sec",
        "true_crossing_bool","predicted_crossing_bool",
        "uncertainty_intersects_geofence",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    for col in [
        "true_crossing_bool",
        "predicted_crossing_bool",
        "uncertainty_intersects_geofence"
    ]:
        df[col] = parse_bool(df[col])

    # Operational primary branch uses the 90% conformal level.
    d = df[np.isclose(pd.to_numeric(df["alpha"]), 0.10)].copy()

    # Step-7B repeated the intrinsic boundary decision for each response-time setting.
    # Keep one copy only. RED in Step-7F is nominal crossing itself; response-time
    # escalation is intentionally separated from persistence analysis.
    dedup_cols = [
        "dataset","task","sample_index","source_group","track_key",
        "axis","boundary","direction","boundary_id","alpha"
    ]

    d = (
        d.sort_values("response_time_sec")
        .drop_duplicates(dedup_cols, keep="first")
        .copy()
    )

    seq_cols = [
        "dataset","task","source_group","track_key",
        "axis","boundary","direction","boundary_id"
    ]

    # ------------------------------------------------------------
    # ORDERING / GAP AUDIT
    # ------------------------------------------------------------
    gap_rows = []
    run_rows = []

    for keys, g in d.groupby(seq_cols, dropna=False):
        gs = g.sort_values("sample_index")
        idx = gs["sample_index"].astype(int).to_numpy()

        diffs = np.diff(idx)

        gap_rows.append({
            **dict(zip(seq_cols, keys)),
            "windows": int(len(gs)),
            "sample_index_min": int(idx.min()),
            "sample_index_max": int(idx.max()),
            "median_index_gap": float(np.median(diffs)) if len(diffs) else np.nan,
            "p90_index_gap": float(np.percentile(diffs, 90)) if len(diffs) else np.nan,
            "max_index_gap": int(np.max(diffs)) if len(diffs) else np.nan,
        })

        run_rows.append({
            **dict(zip(seq_cols, keys)),
            "windows": int(len(gs)),
            "actual_crossing_windows": int(
                np.sum(gs["true_crossing_bool"].to_numpy(dtype=bool))
            ),
            "longest_uncertainty_run": longest_true_run(
                gs["uncertainty_intersects_geofence"].to_numpy(dtype=bool)
            ),
            "longest_nominal_crossing_run": longest_true_run(
                gs["predicted_crossing_bool"].to_numpy(dtype=bool)
            ),
        })

    gap_df = pd.DataFrame(gap_rows)
    run_df = pd.DataFrame(run_rows)

    gap_df.to_csv(
        OUT / "step7f_sequence_ordering_gap_audit.csv",
        index=False
    )
    run_df.to_csv(
        OUT / "step7f_sequence_runlength_audit.csv",
        index=False
    )

    gap_summary = (
        gap_df.groupby(["dataset","task"], as_index=False)
        .agg(
            sequences=("windows","size"),
            median_windows_per_sequence=("windows","median"),
            median_of_median_index_gaps=("median_index_gap","median"),
            median_p90_index_gap=("p90_index_gap","median"),
            max_observed_index_gap=("max_index_gap","max"),
        )
    )

    # ------------------------------------------------------------
    # PERSISTENCE SWEEP
    # ------------------------------------------------------------
    all_rows = []
    metric_rows = []
    source_rows = []

    for n in PERSISTENCE_WINDOWS:
        parts = []

        for keys, g in d.groupby(seq_cols, dropna=False):
            parts.append(
                apply_persistence(g, n)
            )

        x = pd.concat(parts, ignore_index=True)
        all_rows.append(x)

        for (dataset, task), g in x.groupby(["dataset","task"]):
            y = g["true_crossing_bool"].to_numpy(dtype=bool)
            nominal = g["predicted_crossing_bool"].to_numpy(dtype=bool)
            persistent_alert = g["alarm_state_persistent"].isin(
                ["YELLOW","RED"]
            ).to_numpy(dtype=bool)

            m_nom = classify_metrics(y, nominal)
            m_p = classify_metrics(y, persistent_alert)

            nominal_miss = y & (~nominal)
            rescued = nominal_miss & persistent_alert
            incremental = persistent_alert & (~nominal)
            incremental_true = incremental & y
            incremental_false = incremental & (~y)

            metric_rows.append({
                "dataset": dataset,
                "task": task,
                "persistence_windows": int(n),
                "instances": int(len(g)),
                "actual_crossings": int(np.sum(y)),
                "nominal_sensitivity": m_nom["sensitivity"],
                "nominal_false_alert_rate": m_nom["false_alert_rate"],
                "nominal_precision": m_nom["precision"],
                "persistent_alert_sensitivity": m_p["sensitivity"],
                "persistent_false_alert_rate": m_p["false_alert_rate"],
                "persistent_alert_precision": m_p["precision"],
                "nominal_missed_true_crossings": int(np.sum(nominal_miss)),
                "rescued_true_crossings": int(np.sum(rescued)),
                "rescue_rate_among_nominal_misses": safe_div(
                    int(np.sum(rescued)),
                    int(np.sum(nominal_miss))
                ),
                "incremental_alerts_total": int(np.sum(incremental)),
                "incremental_alerts_true": int(np.sum(incremental_true)),
                "incremental_alerts_false": int(np.sum(incremental_false)),
                "incremental_alert_precision": safe_div(
                    int(np.sum(incremental_true)),
                    int(np.sum(incremental))
                ),
                "false_alerts_per_rescued_crossing": safe_div(
                    int(np.sum(incremental_false)),
                    int(np.sum(rescued))
                ),
            })

        for (dataset, task, source), g in x.groupby(
            ["dataset","task","source_group"]
        ):
            y = g["true_crossing_bool"].to_numpy(dtype=bool)
            nominal = g["predicted_crossing_bool"].to_numpy(dtype=bool)
            alert = g["alarm_state_persistent"].isin(
                ["YELLOW","RED"]
            ).to_numpy(dtype=bool)

            m = classify_metrics(y, alert)
            miss = y & (~nominal)
            rescue = miss & alert

            source_rows.append({
                "dataset": dataset,
                "task": task,
                "source_group": source,
                "persistence_windows": int(n),
                "instances": int(len(g)),
                "actual_crossings": int(np.sum(y)),
                "sensitivity": m["sensitivity"],
                "false_alert_rate": m["false_alert_rate"],
                "precision": m["precision"],
                "nominal_misses": int(np.sum(miss)),
                "rescued_true_crossings": int(np.sum(rescue)),
                "rescue_rate_among_nominal_misses": safe_div(
                    int(np.sum(rescue)),
                    int(np.sum(miss))
                ),
            })

    all_df = pd.concat(all_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    by_source = pd.DataFrame(source_rows)

    all_df.to_csv(
        OUT / "step7f_all_persistence_instances.csv",
        index=False
    )
    metrics.to_csv(
        OUT / "TABLE_step7f_persistence_tradeoff.csv",
        index=False
    )
    by_source.to_csv(
        OUT / "step7f_persistence_tradeoff_by_source.csv",
        index=False
    )

    # ------------------------------------------------------------
    # EXTERNAL TRANSFER VIEW
    # Same persistence N is reported on BIRDSAI and WildBox.
    # No automatic "best N" is selected here.
    # ------------------------------------------------------------
    transfer = metrics.pivot_table(
        index=["task","persistence_windows"],
        columns="dataset",
        values=[
            "persistent_alert_sensitivity",
            "persistent_false_alert_rate",
            "persistent_alert_precision",
            "rescue_rate_among_nominal_misses",
            "false_alerts_per_rescued_crossing",
        ],
        aggfunc="first"
    )

    transfer.to_csv(
        OUT / "TABLE_step7f_BIRDSAI_WildBox_transfer_view.csv"
    )

    with open(
        OUT / "STEP7F_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 7F — TEMPORAL PERSISTENCE / HYSTERESIS DIAGNOSTIC\n"
        )
        f.write("=" * 84 + "\n\n")
        f.write(
            "Primary conformal level: 90%.\n"
            "Conformal q values are unchanged. Persistence is an operational "
            "post-processing rule for Yellow warnings only; nominal predicted "
            "crossings remain Red.\n\n"
        )
        f.write(
            "IMPORTANT: persistence is measured in consecutive AVAILABLE forecast "
            "windows after sorting by sample_index within source/track/geofence. "
            "Do not convert N windows to seconds unless the exact dataset window "
            "stride is separately verified.\n\n"
        )
        f.write("SEQUENCE ORDERING / GAP SUMMARY\n")
        f.write("-------------------------------\n")
        f.write(gap_summary.to_string(index=False))
        f.write("\n\nPERSISTENCE TRADE-OFF\n")
        f.write("---------------------\n")
        f.write(metrics.to_string(index=False))
        f.write("\n\nNO FINAL N IS AUTO-SELECTED\n")
        f.write("---------------------------\n")
        f.write(
            "Interpret N=1,2,3,5 as a sensitivity-versus-nuisance-alert trade-off. "
            "Any final operating point should be justified on development data and "
            "then transferred unchanged to an untouched evaluation set.\n"
        )

    print("\nSEQUENCE ORDERING / GAP SUMMARY:")
    print(gap_summary.to_string(index=False))

    show_cols = [
        "dataset","task","persistence_windows",
        "instances","actual_crossings",
        "persistent_alert_sensitivity",
        "persistent_false_alert_rate",
        "persistent_alert_precision",
        "rescued_true_crossings",
        "rescue_rate_among_nominal_misses",
        "incremental_alert_precision",
        "false_alerts_per_rescued_crossing"
    ]

    print("\nPERSISTENCE TRADE-OFF:")
    print(metrics[show_cols].to_string(index=False))

    print("\nSTEP 7F COMPLETE")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
