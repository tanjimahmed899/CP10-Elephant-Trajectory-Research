from pathlib import Path
import json
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
OUT = DOWNLOADS / "step7_response_aware_warning"
INFILE = OUT / "step7b_v2_all_geofence_instance_alarms.csv"
LOCK = OUT / "STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json"

OUT.mkdir(parents=True, exist_ok=True)

def parse_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    x = s.astype(str).str.strip().str.lower()
    tv = {"true","1","yes","y","t"}
    fv = {"false","0","no","n","f"}
    bad = ~x.isin(tv | fv)
    if bad.any():
        raise RuntimeError(f"Unparseable booleans: {sorted(x[bad].unique())[:20]}")
    return x.isin(tv)

def safe_div(a, b):
    return float(a / b) if b else np.nan

def rolling_persistence(g, n):
    g = g.sort_values("sample_index").copy()
    u = g["uncertainty_intersects_geofence"].astype(bool)
    p = (
        u.astype(int)
        .rolling(window=n, min_periods=n)
        .sum()
        .eq(n)
    )
    g["persistent_uncertainty"] = p.to_numpy(dtype=bool)
    return g

def evaluate(g):
    true_cross = g["true_crossing_bool"].to_numpy(dtype=bool)
    alarm = g["alarm_state_final"].astype(str)

    red = alarm.eq("RED").to_numpy()
    alert = alarm.isin(["YELLOW","RED"]).to_numpy()

    true_tte = pd.to_numeric(g["true_tte_sec"], errors="coerce").to_numpy(dtype=float)
    response_sec = float(g["response_time_sec"].iloc[0])

    urgent_true = (
        true_cross
        & np.isfinite(true_tte)
        & (true_tte <= response_sec)
    )

    non_cross = ~true_cross
    non_urgent = ~urgent_true

    return {
        "instances": int(len(g)),
        "actual_crossings": int(np.sum(true_cross)),
        "urgent_actual_crossings": int(np.sum(urgent_true)),
        "green_rate": float(np.mean(alarm.eq("GREEN"))),
        "yellow_rate": float(np.mean(alarm.eq("YELLOW"))),
        "red_rate": float(np.mean(alarm.eq("RED"))),
        "alert_sensitivity_any_crossing": safe_div(
            int(np.sum(alert & true_cross)),
            int(np.sum(true_cross))
        ),
        "red_sensitivity_urgent_crossing": safe_div(
            int(np.sum(red & urgent_true)),
            int(np.sum(urgent_true))
        ),
        "false_alert_rate_non_crossing": safe_div(
            int(np.sum(alert & non_cross)),
            int(np.sum(non_cross))
        ),
        "false_red_rate_non_urgent": safe_div(
            int(np.sum(red & non_urgent)),
            int(np.sum(non_urgent))
        ),
        "alert_precision_any_crossing": safe_div(
            int(np.sum(alert & true_cross)),
            int(np.sum(alert))
        ),
        "red_precision_urgent_crossing": safe_div(
            int(np.sum(red & urgent_true)),
            int(np.sum(red))
        ),
    }

def main():
    print("\nSTEP 7H — FINAL RESPONSE-AWARE EVALUATION WITH LOCKED N=2 PERSISTENCE")

    if not INFILE.exists():
        raise FileNotFoundError(INFILE)
    if not LOCK.exists():
        raise FileNotFoundError(LOCK)

    with open(LOCK, "r", encoding="utf-8") as f:
        lock = json.load(f)

    n = lock.get("selected_persistence_windows")
    if n != 2:
        raise RuntimeError(
            f"Expected locked persistence N=2, found {n}"
        )

    df = pd.read_csv(INFILE)

    required = [
        "dataset","task","sample_index","source_group","track_key",
        "axis","boundary","direction","boundary_id",
        "alpha","response_time_sec",
        "true_crossing_bool","predicted_crossing_bool",
        "uncertainty_intersects_geofence",
        "true_tte_sec","predicted_tte_sec","distance_to_boundary"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    for c in [
        "true_crossing_bool",
        "predicted_crossing_bool",
        "uncertainty_intersects_geofence"
    ]:
        df[c] = parse_bool(df[c])

    # Locked primary operational uncertainty level = 90%.
    d = df[np.isclose(pd.to_numeric(df["alpha"]), 0.10)].copy()

    seq_cols = [
        "dataset","task","source_group","track_key",
        "axis","boundary","direction","boundary_id",
        "response_time_sec"
    ]

    parts = []

    for keys, g in d.groupby(seq_cols, dropna=False):
        x = rolling_persistence(g, n)

        pred_cross = x["predicted_crossing_bool"].astype(bool)
        pred_tte = pd.to_numeric(x["predicted_tte_sec"], errors="coerce")
        distance = pd.to_numeric(x["distance_to_boundary"], errors="coerce")
        response_sec = float(x["response_time_sec"].iloc[0])

        # Final response-aware logic:
        # RED:
        #   nominal crossing predicted within response threshold,
        #   or already at/inside the boundary when TTE is unavailable.
        red = (
            pred_cross
            & pred_tte.notna()
            & (pred_tte <= response_sec)
        ) | (
            pred_cross
            & pred_tte.isna()
            & distance.notna()
            & (distance <= 0)
        )

        # YELLOW:
        #   (a) nominal crossing exists, but later than response threshold; OR
        #   (b) locked N=2 persistent 90% conformal intersection.
        yellow = (
            ~red
            & (
                pred_cross
                | x["persistent_uncertainty"].astype(bool)
            )
        )

        x["alarm_state_final"] = np.where(
            red,
            "RED",
            np.where(yellow, "YELLOW", "GREEN")
        )

        parts.append(x)

    final = pd.concat(parts, ignore_index=True)

    final.to_csv(
        OUT / "step7h_final_response_aware_instances.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Overall per-geofence-instance metrics.
    # ------------------------------------------------------------
    rows = []

    for (dataset, task, response_sec), g in final.groupby(
        ["dataset","task","response_time_sec"]
    ):
        row = {
            "dataset": dataset,
            "task": task,
            "response_time_sec": float(response_sec),
            "persistence_windows": n,
            "conformal_level": 0.90,
        }
        row.update(evaluate(g))
        rows.append(row)

    overall = pd.DataFrame(rows)

    overall.to_csv(
        OUT / "TABLE_step7h_final_response_aware_metrics.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Boundary-macro metrics, to avoid one large boundary scenario
    # dominating the overall result.
    # ------------------------------------------------------------
    bcols = ["axis","boundary","direction","boundary_id"]
    brow = []

    for keys, g in final.groupby(
        ["dataset","task","response_time_sec"] + bcols,
        dropna=False
    ):
        dataset, task, response_sec, axis, boundary, direction, boundary_id = keys

        row = {
            "dataset": dataset,
            "task": task,
            "response_time_sec": float(response_sec),
            "axis": axis,
            "boundary": boundary,
            "direction": direction,
            "boundary_id": boundary_id,
        }
        row.update(evaluate(g))
        brow.append(row)

    per_boundary = pd.DataFrame(brow)

    per_boundary.to_csv(
        OUT / "step7h_final_metrics_per_boundary.csv",
        index=False
    )

    macro_rows = []

    metric_cols = [
        "green_rate",
        "yellow_rate",
        "red_rate",
        "alert_sensitivity_any_crossing",
        "red_sensitivity_urgent_crossing",
        "false_alert_rate_non_crossing",
        "false_red_rate_non_urgent",
        "alert_precision_any_crossing",
        "red_precision_urgent_crossing",
    ]

    for keys, g in per_boundary.groupby(
        ["dataset","task","response_time_sec"]
    ):
        dataset, task, response_sec = keys

        row = {
            "dataset": dataset,
            "task": task,
            "response_time_sec": float(response_sec),
            "boundaries": int(len(g)),
            "persistence_windows": n,
            "conformal_level": 0.90,
        }

        for c in metric_cols:
            z = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            row["macro_" + c] = (
                float(np.nanmean(z))
                if np.isfinite(z).any()
                else np.nan
            )

        macro_rows.append(row)

    macro = pd.DataFrame(macro_rows)

    macro.to_csv(
        OUT / "TABLE_step7h_boundary_macro_metrics.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Final development-branch manifest.
    # ------------------------------------------------------------
    manifest = {
        "step": "STEP_7H_FINAL_RESPONSE_AWARE_DEVELOPMENT_EVALUATION",
        "development_branch_status": "FROZEN_AFTER_STEP7H",
        "primary_conformal_level": 0.90,
        "locked_persistence_windows": 2,
        "final_logic": {
            "RED": (
                "Nominal predicted crossing at or before the assumed response-time "
                "threshold; or already at/inside boundary with unavailable TTE."
            ),
            "YELLOW": (
                "Nominal crossing after the response-time threshold, or 90% conformal "
                "uncertainty intersects the same virtual geofence for 2 consecutive "
                "available forecast windows."
            ),
            "GREEN": "Neither RED nor YELLOW condition is met."
        },
        "evaluation_unit": "individual virtual-geofence scenario, not OR-union across six scenarios",
        "selection_note": (
            "N=2 was selected post hoc on BIRDSAI using the frozen Step-7G exploratory "
            "criterion. WildBox was not used to select N."
        ),
        "confirmatory_requirement": (
            "Apply this rule unchanged to a genuinely untouched confirmatory dataset "
            "before final operational claims."
        ),
        "nonclaims": [
            "N=2 is not converted to seconds because exact window stride has not been verified.",
            "Response-time thresholds are sensitivity-analysis assumptions, not measured field response times.",
            "The virtual geofence is normalized and is not a real village/farm boundary.",
            "BIRDSAI and WildBox results are retrospective development/transfer evidence.",
        ],
    }

    with open(
        OUT / "STEP7_FINAL_DEVELOPMENT_MANIFEST.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    with open(
        OUT / "STEP7_FINAL_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("STEP 7 — FINAL DEVELOPMENT-BRANCH SUMMARY\n")
        f.write("=" * 76 + "\n\n")
        f.write("LOCKED RULE\n")
        f.write("-----------\n")
        f.write(json.dumps(manifest["final_logic"], indent=2))
        f.write("\n\nOVERALL PER-GEOFENCE METRICS\n")
        f.write("----------------------------\n")
        f.write(overall.to_string(index=False))
        f.write("\n\nBOUNDARY-MACRO METRICS\n")
        f.write("----------------------\n")
        f.write(macro.to_string(index=False))
        f.write("\n\nSTATUS\n")
        f.write("------\n")
        f.write("Development branch frozen after Step 7H.\n")
        f.write(
            "Next scientific requirement: untouched confirmatory evaluation "
            "with this rule unchanged.\n"
        )

    show_cols = [
        "dataset","task","response_time_sec",
        "green_rate","yellow_rate","red_rate",
        "alert_sensitivity_any_crossing",
        "red_sensitivity_urgent_crossing",
        "false_alert_rate_non_crossing",
        "false_red_rate_non_urgent",
        "alert_precision_any_crossing",
        "red_precision_urgent_crossing"
    ]

    print("\nFINAL RESPONSE-AWARE PER-GEOFENCE METRICS:")
    print(overall[show_cols].to_string(index=False))

    macro_show = [
        "dataset","task","response_time_sec",
        "macro_green_rate","macro_yellow_rate","macro_red_rate",
        "macro_alert_sensitivity_any_crossing",
        "macro_red_sensitivity_urgent_crossing",
        "macro_false_alert_rate_non_crossing",
        "macro_false_red_rate_non_urgent",
        "macro_alert_precision_any_crossing",
        "macro_red_precision_urgent_crossing"
    ]

    print("\nBOUNDARY-MACRO METRICS:")
    print(macro[macro_show].to_string(index=False))

    print("\nSTEP 7H COMPLETE")
    print("Development branch status: FROZEN_AFTER_STEP7H")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
