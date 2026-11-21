from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

KALMAN_SCRIPT = MAIN_RESULT / "run_kalman_baseline.py"
KALMAN_DIR = MAIN_RESULT / "baseline_kalman"
TUNING_CSV = KALMAN_DIR / "kalman_tuning.csv"
SAMPLE_CSV = KALMAN_DIR / "kalman_sample_metrics.csv"
SUMMARY_TXT = KALMAN_DIR / "SUMMARY.txt"

B2B_SUMMARY = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2B_kalman_param_recovery_audit"
    / "R2D_B2B_SUMMARY.json"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2C_exact_kalman_param_lock"
)

OUT.mkdir(parents=True, exist_ok=True)


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify():
    for p in [
        KALMAN_SCRIPT,
        TUNING_CSV,
        SAMPLE_CSV,
        SUMMARY_TXT,
        B2B_SUMMARY,
    ]:
        require(p)

    obj = json.loads(
        B2B_SUMMARY.read_text(encoding="utf-8")
    )

    if (
        obj.get("status")
        != "R2D_B2B_AUDIT_COMPLETE_QR_CANDIDATE_ARTIFACTS_FOUND"
    ):
        raise RuntimeError(
            f"Unexpected B2B status: {obj.get('status')}"
        )

    return obj


def extract_best_assignment_and_sort_keys():
    text = KALMAN_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(text)

    best_assignments = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        target_names = []

        for target in node.targets:
            if isinstance(target, ast.Name):
                target_names.append(target.id)

        if "best" not in target_names:
            continue

        src = ast.get_source_segment(text, node)

        best_assignments.append(
            {
                "line": int(getattr(node, "lineno", -1)),
                "source": src,
                "node": node,
            }
        )

    if not best_assignments:
        raise RuntimeError(
            "Could not find assignment to variable 'best' in Kalman source."
        )

    # Prefer the assignment in the tuning region, before best_params is set.
    best_assignments = sorted(
        best_assignments,
        key=lambda x: x["line"],
    )

    chosen = None

    for item in best_assignments:
        if 500 <= item["line"] <= 800:
            chosen = item
            break

    if chosen is None:
        chosen = best_assignments[0]

    sort_keys = []

    # Extract dictionary-subscript keys from lambda body in source order.
    for sub in ast.walk(chosen["node"].value):
        if not isinstance(sub, ast.Lambda):
            continue

        body = sub.body

        elements = []

        if isinstance(body, ast.Tuple):
            elements = list(body.elts)
        else:
            elements = [body]

        for elt in elements:
            if not isinstance(elt, ast.Subscript):
                continue

            key = None

            if isinstance(elt.slice, ast.Constant):
                key = elt.slice.value

            if isinstance(key, str):
                sort_keys.append(key)

    # Conservative fallback from the exact source segment.
    if not sort_keys:
        src = str(chosen["source"])

        for key in [
            "track_macro_ADE",
            "track_macro_FDE",
            "window_ADE",
            "window_FDE",
        ]:
            if key in src:
                sort_keys.append(key)

    if not sort_keys:
        raise RuntimeError(
            "Could not safely recover Kalman tuning sort keys from the source."
        )

    return {
        "assignment_line": chosen["line"],
        "assignment_source": chosen["source"],
        "sort_keys": sort_keys,
    }


def validate_tuning_grid(df):
    required = {
        "task",
        "q",
        "r",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"kalman_tuning.csv missing columns: {sorted(missing)}"
        )

    counts = (
        df.groupby("task")
        .size()
        .to_dict()
    )

    unique_q = (
        df.groupby("task")["q"]
        .nunique()
        .to_dict()
    )

    unique_r = (
        df.groupby("task")["r"]
        .nunique()
        .to_dict()
    )

    rows = []

    for task in sorted(df["task"].astype(str).unique()):
        n = int(counts[task])
        nq = int(unique_q[task])
        nr = int(unique_r[task])

        full_grid = (
            n == nq * nr
        )

        rows.append(
            {
                "task": task,
                "rows": n,
                "unique_q": nq,
                "unique_r": nr,
                "q_x_r": nq * nr,
                "full_cartesian_grid": bool(full_grid),
            }
        )

    audit = pd.DataFrame(rows)

    if not audit["full_cartesian_grid"].all():
        raise RuntimeError(
            "Kalman tuning artifact is not a full q/r Cartesian grid for every task."
        )

    return audit


def select_exact_best(df, sort_keys):
    missing = [
        key
        for key in sort_keys
        if key not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Recovered source sort keys missing from tuning CSV: {missing}"
        )

    rows = []

    for task, group in df.groupby("task", sort=True):
        group = group.copy()

        # Original source selected the first element after ascending ordering
        # by the recovered criterion. Add q/r only as deterministic final
        # tie-breakers if all original criterion columns are exactly tied.
        ordered = group.sort_values(
            sort_keys + ["q", "r"],
            ascending=True,
            kind="mergesort",
        )

        first = ordered.iloc[0]

        criterion_values = {
            key: float(first[key])
            for key in sort_keys
        }

        tied = group.copy()

        for key in sort_keys:
            tied = tied[
                tied[key] == first[key]
            ]

        if len(tied) > 1:
            # If source criterion itself is tied across multiple q/r choices,
            # do not silently choose by q/r. We need prior final-evaluation evidence.
            tie_qr = (
                tied[["q", "r"]]
                .drop_duplicates()
            )

            if len(tie_qr) > 1:
                rows.append(
                    {
                        "task": task,
                        "q": None,
                        "r": None,
                        "criterion_tie": True,
                        "tie_qr_json": tie_qr.to_json(
                            orient="records"
                        ),
                        **criterion_values,
                    }
                )
                continue

        rows.append(
            {
                "task": task,
                "q": float(first["q"]),
                "r": float(first["r"]),
                "criterion_tie": False,
                "tie_qr_json": None,
                **criterion_values,
            }
        )

    return pd.DataFrame(rows)


def final_sample_qr_evidence():
    df = pd.read_csv(SAMPLE_CSV)

    required = {
        "task",
        "q",
        "r",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"kalman_sample_metrics.csv missing q/r evidence columns: {sorted(missing)}"
        )

    rows = []

    for task, group in df.groupby("task", sort=True):
        pairs = (
            group[["q", "r"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        rows.append(
            {
                "task": task,
                "sample_rows": int(len(group)),
                "unique_qr_pairs": int(len(pairs)),
                "q": float(pairs.iloc[0]["q"])
                if len(pairs) == 1
                else None,
                "r": float(pairs.iloc[0]["r"])
                if len(pairs) == 1
                else None,
                "pairs_json": pairs.to_json(
                    orient="records"
                ),
            }
        )

    return pd.DataFrame(rows)


def summary_text_evidence():
    text = SUMMARY_TXT.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    rows = []

    for line in text.splitlines():
        low = line.lower()

        if (
            "selected" in low
            and
            "q=" in low
            and
            "r=" in low
        ):
            rows.append(
                {
                    "line": line.strip(),
                }
            )

    return pd.DataFrame(rows)


def resolve_and_crosscheck(
    selected_df,
    sample_evidence_df,
):
    rows = []

    for _, row in selected_df.iterrows():
        task = str(row["task"])

        sample = sample_evidence_df[
            sample_evidence_df["task"].astype(str) == task
        ]

        if len(sample) != 1:
            raise RuntimeError(
                f"Expected one sample q/r evidence row for task {task}"
            )

        sample_row = sample.iloc[0]

        if bool(row["criterion_tie"]):
            # Resolve an exact source-criterion tie only through frozen final-evaluation q/r.
            if int(sample_row["unique_qr_pairs"]) != 1:
                raise RuntimeError(
                    f"Source criterion tied for {task}, and final sample metrics "
                    "do not contain one unique frozen q/r pair."
                )

            q = float(sample_row["q"])
            r = float(sample_row["r"])
            reason = (
                "Original tuning criterion had an exact tie; resolved by the "
                "single frozen q/r pair recorded in final kalman_sample_metrics.csv."
            )

        else:
            q = float(row["q"])
            r = float(row["r"])

            if int(sample_row["unique_qr_pairs"]) != 1:
                raise RuntimeError(
                    f"Final Kalman sample metrics contain multiple q/r pairs for {task}."
                )

            if (
                q != float(sample_row["q"])
                or
                r != float(sample_row["r"])
            ):
                raise RuntimeError(
                    f"Frozen q/r mismatch for {task}: source-selection gives "
                    f"q={q}, r={r}, but final sample metrics give "
                    f"q={sample_row['q']}, r={sample_row['r']}."
                )

            reason = (
                "Exact best q/r from original TrainReal tuning criterion; "
                "independently matched the single q/r pair recorded in final "
                "kalman_sample_metrics.csv."
            )

        rows.append(
            {
                "task": task,
                "q": q,
                "r": r,
                "crosscheck_pass": True,
                "resolution_reason": reason,
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR2D-B2C — EXACT FROZEN KALMAN q/r RECOVERY + LOCK"
    )

    print(
        "NO forecasting metric is generated. NO TestReal tuning occurs. "
        "The original TrainReal tuning criterion is recovered from source, "
        "then cross-checked against the frozen final Kalman sample table.\n"
    )

    verify()

    selection = extract_best_assignment_and_sort_keys()

    tuning = pd.read_csv(TUNING_CSV)
    grid_audit = validate_tuning_grid(tuning)

    selected = select_exact_best(
        tuning,
        selection["sort_keys"],
    )

    sample_evidence = final_sample_qr_evidence()

    resolved = resolve_and_crosscheck(
        selected,
        sample_evidence,
    )

    summary_evidence = summary_text_evidence()

    grid_path = OUT / "R2D_B2C_tuning_grid_audit.csv"
    selected_path = OUT / "R2D_B2C_source_selected_qr.csv"
    sample_path = OUT / "R2D_B2C_final_sample_qr_evidence.csv"
    resolved_path = OUT / "R2D_B2C_LOCKED_KALMAN_PARAMS.csv"
    summary_evidence_path = OUT / "R2D_B2C_summary_text_evidence.csv"

    grid_audit.to_csv(grid_path, index=False)
    selected.to_csv(selected_path, index=False)
    sample_evidence.to_csv(sample_path, index=False)
    resolved.to_csv(resolved_path, index=False)
    summary_evidence.to_csv(
        summary_evidence_path,
        index=False,
    )

    selection_json = OUT / "R2D_B2C_source_selection_criterion.json"
    selection_json.write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    status = (
        "R2D_B2C_FROZEN_KALMAN_PARAMS_EXACTLY_RECOVERED_AND_LOCKED_READY_FOR_B2"
    )

    manifest = {
        "step": "R2D_B2C_EXACT_FROZEN_KALMAN_PARAM_LOCK",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "new_metrics_generated": False,
        "testreal_tuning": False,
        "kalman_source": str(KALMAN_SCRIPT),
        "kalman_source_sha256": sha256_file(KALMAN_SCRIPT),
        "tuning_csv": str(TUNING_CSV),
        "tuning_csv_sha256": sha256_file(TUNING_CSV),
        "sample_csv": str(SAMPLE_CSV),
        "sample_csv_sha256": sha256_file(SAMPLE_CSV),
        "summary_txt": str(SUMMARY_TXT),
        "summary_txt_sha256": sha256_file(SUMMARY_TXT),
        "source_selection_criterion": selection,
        "locked_params_csv": str(resolved_path),
        "locked_params_sha256": sha256_file(resolved_path),
    }

    manifest_path = OUT / "R2D_B2C_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("SOURCE SELECTION CRITERION:")
    print(
        json.dumps(
            selection,
            indent=2,
        )
    )

    print("\nTUNING GRID AUDIT:")
    print(
        grid_audit.to_string(
            index=False
        )
    )

    print("\nLOCKED FROZEN KALMAN PARAMETERS:")
    print(
        resolved.to_string(
            index=False
        )
    )

    print("\nFINAL SAMPLE-TABLE CROSS-CHECK:")
    print(
        sample_evidence.to_string(
            index=False
        )
    )

    if len(summary_evidence) > 0:
        print("\nSUMMARY.txt PARAMETER EVIDENCE:")
        print(
            summary_evidence.to_string(
                index=False
            )
        )

    print("\nSTATUS:")
    print(status)

    print("\nR2D-B2C COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
