from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")
OUT = DOWNLOADS / "step7_response_aware_warning"
INFILE = OUT / "TABLE_step7f_persistence_tradeoff.csv"
LOCK = OUT / "STEP7G_PROVISIONAL_OPERATING_RULE_LOCK.json"

MIN_REL_FALSE_ALERT_REDUCTION = 0.20
MIN_SENSITIVITY = 0.80
CANDIDATE_NS = [2, 3, 5]

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def pareto_frontier(df):
    """
    Maximize sensitivity, minimize false-alert rate.
    A point is Pareto-optimal if no other point has >= sensitivity
    and <= false-alert rate with at least one strict improvement.
    """
    rows = []
    for i, r in df.iterrows():
        dominated = False
        for j, q in df.iterrows():
            if i == j:
                continue
            better_or_equal = (
                q["persistent_alert_sensitivity"] >= r["persistent_alert_sensitivity"]
                and
                q["persistent_false_alert_rate"] <= r["persistent_false_alert_rate"]
            )
            strictly_better = (
                q["persistent_alert_sensitivity"] > r["persistent_alert_sensitivity"]
                or
                q["persistent_false_alert_rate"] < r["persistent_false_alert_rate"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        rows.append(not dominated)
    out = df.copy()
    out["pareto_optimal"] = rows
    return out

def main():
    print("\nSTEP 7G — PARETO ANALYSIS + PROVISIONAL OPERATING-RULE LOCK")

    if not INFILE.exists():
        raise FileNotFoundError(INFILE)

    df = pd.read_csv(INFILE)

    required = [
        "dataset","task","persistence_windows",
        "persistent_alert_sensitivity",
        "persistent_false_alert_rate",
        "persistent_alert_precision",
        "rescue_rate_among_nominal_misses",
        "false_alerts_per_rescued_crossing",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # ------------------------------------------------------------
    # Pareto table for every dataset/task.
    # ------------------------------------------------------------
    pareto_parts = []
    for (dataset, task), g in df.groupby(["dataset","task"]):
        p = pareto_frontier(
            g.sort_values("persistence_windows").reset_index(drop=True)
        )
        pareto_parts.append(p)

    pareto = pd.concat(pareto_parts, ignore_index=True)
    pareto.to_csv(
        OUT / "TABLE_step7g_persistence_pareto_frontier.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Selection uses BIRDSAI only.
    #
    # Transparent exploratory criterion:
    # Among N > 1, choose the SMALLEST N that on BOTH horizons:
    #   1) reduces false-alert rate by >=20% RELATIVE to N=1
    #   2) keeps sensitivity >=0.80
    #
    # This is explicitly POST-HOC / exploratory because Step-7F results
    # have already been inspected. It is frozen ONLY for a future
    # untouched confirmatory evaluation (e.g. BAMBI when ready).
    # ------------------------------------------------------------
    dev = df[
        df["dataset"].astype(str).str.contains(
            "BIRDSAI", case=False, na=False
        )
    ].copy()

    if dev["task"].nunique() < 2:
        raise RuntimeError("Expected both BIRDSAI horizons.")

    baseline = (
        dev[dev["persistence_windows"] == 1]
        .set_index("task")
    )

    if len(baseline) != dev["task"].nunique():
        raise RuntimeError("Missing unique N=1 baseline per BIRDSAI task.")

    candidate_rows = []

    for n in CANDIDATE_NS:
        g = dev[dev["persistence_windows"] == n].copy()

        if len(g) != dev["task"].nunique():
            continue

        all_pass = True

        for _, r in g.iterrows():
            task = r["task"]
            b = baseline.loc[task]

            base_fa = float(b["persistent_false_alert_rate"])
            cand_fa = float(r["persistent_false_alert_rate"])
            sens = float(r["persistent_alert_sensitivity"])

            rel_reduction = (
                (base_fa - cand_fa) / base_fa
                if base_fa > 0
                else np.nan
            )

            pass_reduction = (
                np.isfinite(rel_reduction)
                and rel_reduction >= MIN_REL_FALSE_ALERT_REDUCTION
            )
            pass_sensitivity = (
                np.isfinite(sens)
                and sens >= MIN_SENSITIVITY
            )

            if not (pass_reduction and pass_sensitivity):
                all_pass = False

            candidate_rows.append({
                "persistence_windows": int(n),
                "task": task,
                "baseline_false_alert_rate_N1": base_fa,
                "candidate_false_alert_rate": cand_fa,
                "relative_false_alert_reduction": rel_reduction,
                "candidate_sensitivity": sens,
                "passes_false_alert_reduction": bool(pass_reduction),
                "passes_sensitivity": bool(pass_sensitivity),
                "passes_both": bool(pass_reduction and pass_sensitivity),
            })

        if all_pass:
            selected_n = int(n)
            break
    else:
        selected_n = None

    selection_audit = pd.DataFrame(candidate_rows)
    selection_audit.to_csv(
        OUT / "step7g_selection_criterion_audit.csv",
        index=False
    )

    if selected_n is None:
        verdict = "NO_CANDIDATE_MET_LOCK_CRITERION"
    else:
        verdict = "PROVISIONAL_CANDIDATE_SELECTED"

    # ------------------------------------------------------------
    # Report the already-observed WildBox transfer for the selected N,
    # but DO NOT use WildBox to change the selected N.
    # ------------------------------------------------------------
    transfer = pd.DataFrame()

    if selected_n is not None:
        transfer = df[
            df["persistence_windows"] == selected_n
        ].copy()

        transfer.to_csv(
            OUT / "TABLE_step7g_selected_N_transfer_summary.csv",
            index=False
        )

    lock = {
        "step": "STEP_7G_PROVISIONAL_OPERATING_RULE_LOCK",
        "status": verdict,
        "selected_persistence_windows": selected_n,
        "primary_conformal_level": 0.90,
        "selection_dataset": "BIRDSAI_TestReal",
        "selection_is_post_hoc_exploratory": True,
        "selection_rule": {
            "candidate_set": CANDIDATE_NS,
            "choose": "smallest N satisfying the criterion on both horizons",
            "minimum_relative_false_alert_reduction_vs_N1": MIN_REL_FALSE_ALERT_REDUCTION,
            "minimum_sensitivity": MIN_SENSITIVITY,
        },
        "operational_interpretation": {
            "RED": "nominal predicted geofence crossing",
            "YELLOW": (
                "90% conformal uncertainty intersects the evaluated virtual geofence "
                f"for {selected_n} consecutive AVAILABLE forecast windows"
                if selected_n is not None
                else None
            ),
            "GREEN": (
                "neither RED nor persistence-qualified YELLOW"
                if selected_n is not None
                else None
            ),
        },
        "important_limits": [
            "N windows are consecutive available forecast windows, not seconds.",
            "Exact physical time duration must not be claimed until window stride is verified.",
            "The virtual geofence is a normalized experimental boundary, not a real village/farm boundary.",
            "This operating rule is provisional and selected post hoc from BIRDSAI Step-7F diagnostics.",
            "WildBox results are retrospective transfer diagnostics and were not used to choose N.",
            "The rule should be evaluated unchanged on a genuinely untouched confirmatory set before final operational claims.",
        ],
        "source_step7f_sha256": sha256_file(INFILE),
    }

    if LOCK.exists():
        with open(LOCK, "r", encoding="utf-8") as f:
            old = json.load(f)

        if old != lock:
            raise RuntimeError(
                "Existing Step-7G lock differs from current result; refusing to overwrite."
            )
        lock_status = "EXISTING_LOCK_VERIFIED"
    else:
        with open(LOCK, "w", encoding="utf-8") as f:
            json.dump(lock, f, indent=2)
        lock_status = "NEW_PROVISIONAL_LOCK_CREATED"

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    with open(
        OUT / "STEP7G_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("STEP 7G — PARETO ANALYSIS + PROVISIONAL OPERATING-RULE LOCK\n")
        f.write("=" * 84 + "\n\n")
        f.write(f"VERDICT: {verdict}\n")
        f.write(f"LOCK STATUS: {lock_status}\n")
        f.write(f"SELECTED N: {selected_n}\n\n")

        f.write("SELECTION CRITERION\n")
        f.write("-------------------\n")
        f.write(
            "Choose the smallest N>1 that, on BOTH BIRDSAI horizons, "
            "reduces false-alert rate by at least 20% relative to N=1 "
            "while retaining sensitivity >= 0.80.\n"
        )
        f.write(
            "This criterion is exploratory/post-hoc and is frozen only for "
            "future untouched confirmatory evaluation.\n\n"
        )

        f.write("SELECTION AUDIT\n")
        f.write("---------------\n")
        f.write(selection_audit.to_string(index=False))

        f.write("\n\nSELECTED-N TRANSFER SUMMARY\n")
        f.write("---------------------------\n")
        if len(transfer):
            f.write(
                transfer[
                    [
                        "dataset","task","persistence_windows",
                        "persistent_alert_sensitivity",
                        "persistent_false_alert_rate",
                        "persistent_alert_precision",
                        "rescue_rate_among_nominal_misses",
                        "false_alerts_per_rescued_crossing",
                    ]
                ].to_string(index=False)
            )
        else:
            f.write("NONE")

        f.write("\n\nIMPORTANT LIMITS\n")
        f.write("----------------\n")
        for item in lock["important_limits"]:
            f.write(f"- {item}\n")

    print("\nSELECTION AUDIT:")
    print(selection_audit.to_string(index=False))

    print("\nVERDICT:")
    print(verdict)

    print("selected_persistence_windows =", selected_n)
    print("lock_status =", lock_status)

    if len(transfer):
        print("\nSELECTED-N TRANSFER SUMMARY:")
        print(
            transfer[
                [
                    "dataset","task","persistence_windows",
                    "persistent_alert_sensitivity",
                    "persistent_false_alert_rate",
                    "persistent_alert_precision",
                    "rescue_rate_among_nominal_misses",
                    "false_alerts_per_rescued_crossing",
                ]
            ].to_string(index=False)
        )

    print("\nSTEP 7G COMPLETE")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
