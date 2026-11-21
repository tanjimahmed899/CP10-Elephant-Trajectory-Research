from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

SEARCH_ROOTS = [
    MAIN_RESULT_2,
    MAIN_RESULT,
]

R2B_SUMMARY = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2B_fair_same_sample_benchmark"
    / "R2B_SUMMARY.json"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2C_missing_row_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    ("Fresh_WildBox", "2s_to_2s"): 865,
    ("Fresh_WildBox", "2s_to_5s"): 637,
    ("BIRDSAI_TestReal", "2s_to_2s"): 1053,
    ("BIRDSAI_TestReal", "2s_to_5s"): 826,
}

MODELS = [
    "Stationary",
    "ConstantVelocity",
    "KalmanCV",
    "GRU",
    "LSTM",
    "TransformerEnsemble",
]

TARGET_SCRIPTS = [
    "run_constant_velocity_baseline.py",
    "run_kalman_baseline.py",
    "train_lstm_gru.py",
    "evaluate_birdsai_testreal_frozen_ensemble.py",
    "evaluate_fresh_wildbox_holdout.py",
    "build_final_neural_dataset.py",
]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_r2b():
    if not R2B_SUMMARY.exists():
        raise FileNotFoundError(R2B_SUMMARY)

    obj = json.loads(
        R2B_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    expected_status = (
        "R2B_EXISTING_OUTPUTS_AUDITED_TARGETED_RECOMPUTATION_REQUIRED_FOR_MISSING_MODEL_SAMPLE_ROWS"
    )

    if obj.get("status") != expected_status:
        raise RuntimeError(
            f"Unexpected R2B status: {obj.get('status')}"
        )

    return obj


def dataset_hint(text):
    low = str(text).lower()

    if "fresh" in low and "wildbox" in low:
        return "Fresh_WildBox"

    if "testreal" in low or "birdsai_testreal" in low:
        return "BIRDSAI_TestReal"

    return None


def task_hint(text):
    low = str(text).lower().replace(" ", "")

    if any(x in low for x in [
        "2s_to_2s",
        "2s-to-2s",
        "2to2",
        "2s2s",
        "2→2",
    ]):
        return "2s_to_2s"

    if any(x in low for x in [
        "2s_to_5s",
        "2s-to-5s",
        "2to5",
        "2s5s",
        "2→5",
    ]):
        return "2s_to_5s"

    return None


def model_hint(text):
    low = str(text).lower().replace(" ", "").replace("_", "").replace("-", "")

    if "stationary" in low:
        return "Stationary"

    if "kalman" in low:
        return "KalmanCV"

    if (
        "constantvelocity" in low
        or
        low == "cv"
        or
        "cvbaseline" in low
    ):
        return "ConstantVelocity"

    if "lstm" in low:
        return "LSTM"

    if "gru" in low:
        return "GRU"

    if "transformer" in low:
        if (
            "ensemble" in low
            or
            "3seed" in low
            or
            "frozenensemble" in low
        ):
            return "TransformerEnsemble"
        return "Transformer"

    return None


def first_col(cols, candidates):
    lookup = {str(c).lower(): c for c in cols}
    for x in candidates:
        if x.lower() in lookup:
            return lookup[x.lower()]
    return None


def inspect_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {
            "path": str(path),
            "kind": "csv",
            "read_ok": False,
            "error": f"{type(e).__name__}: {e}",
        }, []

    info = {
        "path": str(path),
        "kind": "csv",
        "read_ok": True,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": " | ".join(map(str, df.columns)),
        "dataset_path_hint": dataset_hint(path),
        "task_path_hint": task_hint(path),
        "model_path_hint": model_hint(path),
        "sha256": sha256_file(path),
    }

    evidence = []

    cols_low = {str(c).lower(): c for c in df.columns}

    # Long-form metric evidence.
    ade_col = first_col(
        df.columns,
        ["ade", "sample_ade", "ade_mean", "mean_ade"],
    )
    fde_col = first_col(
        df.columns,
        ["fde", "sample_fde", "fde_mean", "mean_fde"],
    )
    model_col = first_col(
        df.columns,
        ["model", "method", "model_name", "baseline", "predictor"],
    )
    task_col = first_col(
        df.columns,
        ["task", "horizon", "forecast_task"],
    )
    dataset_col = first_col(
        df.columns,
        ["dataset", "split", "evaluation_set"],
    )

    if ade_col is not None and fde_col is not None:
        if model_col is not None:
            model_counts = (
                df[model_col]
                .astype(str)
                .value_counts(dropna=False)
                .to_dict()
            )
        else:
            model_counts = {}

        evidence.append({
            "path": str(path),
            "format": "long_metric",
            "rows": int(len(df)),
            "ade_col": str(ade_col),
            "fde_col": str(fde_col),
            "model_col": str(model_col) if model_col is not None else None,
            "task_col": str(task_col) if task_col is not None else None,
            "dataset_col": str(dataset_col) if dataset_col is not None else None,
            "model_counts_json": json.dumps(model_counts),
        })

    # Wide-format metric evidence, e.g. Stationary_ADE, CV_ADE, Transformer_ADE.
    column_names = [str(c) for c in df.columns]
    wide_pairs = []

    for c in column_names:
        low = c.lower()

        if "ade" not in low:
            continue

        base = re.sub(r"(?i)[_\-\s]*ade.*$", "", c).strip()

        if not base:
            continue

        candidate_fde = []

        for f in column_names:
            if "fde" in f.lower():
                fbase = re.sub(r"(?i)[_\-\s]*fde.*$", "", f).strip()
                if fbase.lower() == base.lower():
                    candidate_fde.append(f)

        if candidate_fde:
            wide_pairs.append((base, c, candidate_fde[0]))

    if wide_pairs:
        evidence.append({
            "path": str(path),
            "format": "wide_metric",
            "rows": int(len(df)),
            "wide_pairs_json": json.dumps(wide_pairs),
        })

    return info, evidence


def inspect_npz(path):
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as e:
        return {
            "path": str(path),
            "kind": "npz",
            "read_ok": False,
            "error": f"{type(e).__name__}: {e}",
        }

    arrays = {}

    for key in z.files:
        arr = z[key]
        arrays[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    return {
        "path": str(path),
        "kind": "npz",
        "read_ok": True,
        "dataset_path_hint": dataset_hint(path),
        "task_path_hint": task_hint(path),
        "model_path_hint": model_hint(path),
        "arrays_json": json.dumps(arrays),
        "sha256": sha256_file(path),
    }


def discover_result_artifacts():
    audit_rows = []
    csv_evidence = []

    tokens = [
        "fresh",
        "wildbox",
        "testreal",
        "birdsai",
        "ensemble",
        "baseline",
        "stationary",
        "constant",
        "kalman",
        "gru",
        "lstm",
        "transformer",
        "sample",
        "metric",
        "prediction",
    ]

    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            if p.suffix.lower() not in {".csv", ".npz"}:
                continue

            full = str(p).lower()

            if not any(token in full for token in tokens):
                continue

            rp = str(p.resolve()).lower()
            if rp in seen:
                continue
            seen.add(rp)

            if p.suffix.lower() == ".csv":
                info, evidence = inspect_csv(p)
                audit_rows.append(info)
                csv_evidence.extend(evidence)
            else:
                audit_rows.append(inspect_npz(p))

    return (
        pd.DataFrame(audit_rows),
        pd.DataFrame(csv_evidence),
    )


def inspect_scripts():
    rows = []
    call_rows = []

    for name in TARGET_SCRIPTS:
        hits = []

        for root in SEARCH_ROOTS:
            if not root.exists():
                continue
            hits.extend(
                [p for p in root.rglob(name) if p.is_file()]
            )

        if not hits:
            rows.append({
                "script_name": name,
                "selected_path": None,
                "sha256": None,
                "functions": None,
                "classes": None,
            })
            continue

        hits = sorted(
            hits,
            key=lambda p: (
                0 if str(MAIN_RESULT_2).lower() in str(p).lower() else 1,
                str(p).lower(),
            )
        )

        p = hits[0]
        text = p.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        try:
            tree = ast.parse(text)
        except Exception:
            rows.append({
                "script_name": name,
                "selected_path": str(p),
                "sha256": sha256_file(p),
                "functions": "<AST parse failed>",
                "classes": None,
            })
            continue

        functions = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        ]

        classes = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        ]

        rows.append({
            "script_name": name,
            "selected_path": str(p),
            "sha256": sha256_file(p),
            "functions": " | ".join(functions),
            "classes": " | ".join(classes),
        })

        interesting_calls = [
            "load",
            "evaluate",
            "predict",
            "baseline",
            "stationary",
            "constant",
            "kalman",
            "transformer",
            "lstm",
            "gru",
        ]

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            call_name = None

            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr

            if call_name is None:
                continue

            if not any(
                token in call_name.lower()
                for token in interesting_calls
            ):
                continue

            expr = ast.get_source_segment(
                text,
                node,
            )

            call_rows.append({
                "script": str(p),
                "line": int(getattr(node, "lineno", -1)),
                "call": call_name,
                "expression": expr,
            })

    return pd.DataFrame(rows), pd.DataFrame(call_rows)


def derive_missing_matrix_from_r2b():
    alignment_path = (
        DOWNLOADS
        / "paper_strengthening_revision"
        / "R2_fair_forecasting"
        / "R2B_fair_same_sample_benchmark"
        / "R2B_same_sample_alignment.csv"
    )

    if not alignment_path.exists():
        raise FileNotFoundError(alignment_path)

    df = pd.read_csv(alignment_path)

    rows = []

    for _, row in df.iterrows():
        dataset = str(row["dataset"])
        task = str(row["task"])
        expected = int(row["expected_samples"])

        for model in MODELS:
            col = f"{model}_samples"
            available = int(row[col])

            rows.append({
                "dataset": dataset,
                "task": task,
                "model": model,
                "expected_samples": expected,
                "currently_recovered_samples": available,
                "missing_samples": max(0, expected - available),
                "complete": available == expected,
            })

    return pd.DataFrame(rows)


def identify_existing_candidate_support(missing_df, artifact_df, evidence_df):
    rows = []

    for _, miss in missing_df.iterrows():
        dataset = miss["dataset"]
        task = miss["task"]
        model = miss["model"]

        if bool(miss["complete"]):
            rows.append({
                **miss.to_dict(),
                "candidate_artifact_count": 0,
                "candidate_evidence_count": 0,
                "candidate_examples": "",
                "recommended_action": "already_complete",
            })
            continue

        art = artifact_df.copy()

        mask_dataset = (
            art.get("dataset_path_hint", pd.Series(index=art.index, dtype=object))
            .fillna("")
            .astype(str)
            .eq(dataset)
        )

        mask_task = (
            art.get("task_path_hint", pd.Series(index=art.index, dtype=object))
            .fillna("")
            .astype(str)
            .eq(task)
        )

        model_low = model.lower()

        path_model = art["path"].astype(str).str.lower().str.contains(
            model_low.replace("constantvelocity", "constant"),
            regex=False,
        )

        art_subset = art[
            mask_dataset
            &
            mask_task
            &
            path_model
        ].copy()

        ev_subset = pd.DataFrame()

        if len(evidence_df) > 0:
            ev_subset = evidence_df[
                evidence_df["path"].astype(str).isin(
                    art_subset["path"].astype(str)
                )
            ].copy()

        examples = " | ".join(
            art_subset["path"].head(8).astype(str).tolist()
        )

        if len(art_subset) > 0:
            action = "inspect_existing_artifact_before_recompute"
        else:
            action = "targeted_recompute_likely_required"

        rows.append({
            **miss.to_dict(),
            "candidate_artifact_count": int(len(art_subset)),
            "candidate_evidence_count": int(len(ev_subset)),
            "candidate_examples": examples,
            "recommended_action": action,
        })

    return pd.DataFrame(rows)


def main():
    print(
        "\nR2C — MISSING SAMPLE-ROW RECOVERY AUDIT"
    )

    print(
        "NO model is retrained and NO new benchmark metric is generated. "
        "The goal is to determine whether missing R2B rows already exist in old artifacts "
        "before any targeted recomputation.\n"
    )

    r2b = verify_r2b()

    missing_df = derive_missing_matrix_from_r2b()

    artifact_df, evidence_df = discover_result_artifacts()

    script_df, call_df = inspect_scripts()

    support_df = identify_existing_candidate_support(
        missing_df,
        artifact_df,
        evidence_df,
    )

    missing_path = OUT / "R2C_missing_matrix.csv"
    artifact_path = OUT / "R2C_result_artifact_inventory.csv"
    evidence_path = OUT / "R2C_csv_metric_format_evidence.csv"
    scripts_path = OUT / "R2C_source_script_inventory.csv"
    calls_path = OUT / "R2C_source_call_inventory.csv"
    support_path = OUT / "R2C_missing_row_existing_support.csv"

    missing_df.to_csv(missing_path, index=False)
    artifact_df.to_csv(artifact_path, index=False)
    evidence_df.to_csv(evidence_path, index=False)
    script_df.to_csv(scripts_path, index=False)
    call_df.to_csv(calls_path, index=False)
    support_df.to_csv(support_path, index=False)

    unresolved = support_df[
        (~support_df["complete"])
    ].copy()

    all_have_candidates = bool(
        len(unresolved) > 0
        and
        (
            unresolved["candidate_artifact_count"]
            >
            0
        ).all()
    )

    if len(unresolved) == 0:
        status = "R2C_ALL_REQUIRED_SAMPLE_ROWS_ALREADY_COMPLETE"
    elif all_have_candidates:
        status = "R2C_EXISTING_CANDIDATES_FOUND_BUILD_RECOVERY_ADAPTER_BEFORE_RECOMPUTE"
    else:
        status = "R2C_AUDIT_COMPLETE_SOME_TARGETED_RECOMPUTATION_CONFIRMED_REQUIRED"

    summary = {
        "step": "R2C_MISSING_SAMPLE_ROW_RECOVERY_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "r2b_status": r2b.get("status"),
        "new_metrics_generated": False,
        "retraining": False,
        "missing_matrix": str(missing_path),
        "artifact_inventory": str(artifact_path),
        "metric_format_evidence": str(evidence_path),
        "source_scripts": str(scripts_path),
        "source_calls": str(calls_path),
        "existing_support": str(support_path),
    }

    summary_path = OUT / "R2C_SUMMARY.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("MISSING MATRIX:")
    print(
        missing_df.to_string(
            index=False
        )
    )

    print("\nEXISTING SUPPORT FOR MISSING ROWS:")
    print(
        support_df[
            [
                "dataset",
                "task",
                "model",
                "missing_samples",
                "candidate_artifact_count",
                "candidate_evidence_count",
                "recommended_action",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR2C COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
