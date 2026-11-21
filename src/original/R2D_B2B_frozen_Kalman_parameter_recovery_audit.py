
from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

KALMAN_SCRIPT = MAIN_RESULT / "run_kalman_baseline.py"

B2A_SUMMARY = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2_preflight"
    / "R2D_B2A_SUMMARY.json"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_B2B_kalman_param_recovery_audit"
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
    require(KALMAN_SCRIPT)
    require(B2A_SUMMARY)

    b2a = json.loads(B2A_SUMMARY.read_text(encoding="utf-8"))
    if (
        b2a.get("status")
        !=
        "R2D_B2A_PREFLIGHT_COMPLETE_READY_FOR_TESTREAL_KALMAN_GRU_LSTM_RECOMPUTATION"
    ):
        raise RuntimeError(f"Unexpected B2A status: {b2a.get('status')}")

    return b2a


def source_context():
    text = KALMAN_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )
    lines = text.splitlines()

    patterns = [
        "best_params",
        "idxmin",
        "idxmax",
        "sort_values",
        "nsmallest",
        "nlargest",
        "Q_GRID",
        "R_GRID",
        "tuning_rows",
        "validation",
        "track_macro",
        "ADE_track_macro",
        "ADE_mean",
        "to_csv",
        "json.dump",
        "json.dumps",
        "baseline_kalman",
        "kalman_summary",
    ]

    hits = []

    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(p.lower() in low for p in patterns):
            start = max(1, i - 8)
            end = min(len(lines), i + 12)

            snippet = "\n".join(
                f"{j}: {lines[j-1]}"
                for j in range(start, end + 1)
            )

            hits.append(
                {
                    "hit_line": i,
                    "hit_text": line.strip(),
                    "context_start": start,
                    "context_end": end,
                    "context": snippet,
                }
            )

    return text, pd.DataFrame(hits)


def ast_selection_evidence(text):
    tree = ast.parse(text)
    rows = []

    interesting = {
        "min",
        "max",
        "sorted",
        "sort_values",
        "idxmin",
        "idxmax",
        "nsmallest",
        "nlargest",
        "groupby",
        "to_csv",
        "dump",
        "dumps",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None

            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr

            if name is None or name not in interesting:
                continue

            src = ast.get_source_segment(text, node)

            rows.append(
                {
                    "line": int(getattr(node, "lineno", -1)),
                    "call": name,
                    "expression": src,
                }
            )

    return pd.DataFrame(rows)


def find_col(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def scan_artifacts():
    roots = [
        MAIN_RESULT,
        MAIN_RESULT_2,
        DOWNLOADS / "baseline_kalman",
    ]

    rows = []
    candidate_rows = []

    suffixes = {
        ".csv",
        ".json",
        ".txt",
    }

    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            if p.suffix.lower() not in suffixes:
                continue

            low_path = str(p).lower()

            if not any(
                token in low_path
                for token in [
                    "kalman",
                    "baseline",
                    "tuning",
                    "parameter",
                    "param",
                ]
            ):
                continue

            key = str(p.resolve()).lower()
            if key in seen:
                continue

            seen.add(key)

            base = {
                "path": str(p),
                "suffix": p.suffix.lower(),
                "sha256": sha256_file(p),
            }

            if p.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(p)
                except Exception as e:
                    rows.append(
                        {
                            **base,
                            "read_ok": False,
                            "rows": None,
                            "columns": None,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    continue

                rows.append(
                    {
                        **base,
                        "read_ok": True,
                        "rows": int(len(df)),
                        "columns": " | ".join(map(str, df.columns)),
                        "error": None,
                    }
                )

                qcol = find_col(
                    df,
                    ["q", "process_noise", "process_noise_q", "q_value"],
                )

                rcol = find_col(
                    df,
                    ["r", "measurement_noise", "measurement_noise_r", "r_value"],
                )

                taskcol = find_col(
                    df,
                    ["task", "forecast_task", "horizon"],
                )

                metric_cols = [
                    c
                    for c in df.columns
                    if any(
                        token in str(c).lower()
                        for token in [
                            "ade",
                            "fde",
                            "track_macro",
                            "validation",
                            "val_",
                        ]
                    )
                ]

                if qcol is not None and rcol is not None:
                    preview_cols = []

                    if taskcol is not None:
                        preview_cols.append(taskcol)

                    preview_cols.extend(
                        [
                            qcol,
                            rcol,
                        ]
                    )

                    preview_cols.extend(
                        [
                            c
                            for c in metric_cols
                            if c not in preview_cols
                        ][:8]
                    )

                    preview = df[preview_cols].head(50)

                    candidate_rows.append(
                        {
                            "path": str(p),
                            "sha256": sha256_file(p),
                            "rows": int(len(df)),
                            "q_col": str(qcol),
                            "r_col": str(rcol),
                            "task_col": str(taskcol) if taskcol is not None else None,
                            "metric_cols": " | ".join(map(str, metric_cols)),
                            "preview_json": preview.to_json(
                                orient="records"
                            ),
                        }
                    )

            else:
                try:
                    raw = p.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                except Exception as e:
                    rows.append(
                        {
                            **base,
                            "read_ok": False,
                            "rows": None,
                            "columns": None,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    continue

                rows.append(
                    {
                        **base,
                        "read_ok": True,
                        "rows": None,
                        "columns": None,
                        "error": None,
                    }
                )

                lines = raw.splitlines()
                matches = []

                for i, line in enumerate(lines, start=1):
                    low = line.lower()

                    if (
                        ("best" in low or "selected" in low or "tuning" in low)
                        and
                        ("q" in low or "r" in low or "kalman" in low)
                    ):
                        matches.append(
                            f"L{i}: {line.strip()[:800]}"
                        )

                if matches:
                    candidate_rows.append(
                        {
                            "path": str(p),
                            "sha256": sha256_file(p),
                            "rows": None,
                            "q_col": None,
                            "r_col": None,
                            "task_col": None,
                            "metric_cols": None,
                            "preview_json": json.dumps(matches[:100]),
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def extract_named_output_paths(text):
    rows = []

    pattern = re.compile(
        r"[\"']([^\"']+\.(?:csv|json|txt))[\"']",
        re.IGNORECASE,
    )

    for m in pattern.finditer(text):
        line = text[:m.start()].count("\n") + 1
        rows.append(
            {
                "line": line,
                "filename_literal": m.group(1),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR2D-B2B — FROZEN KALMAN PARAMETER RECOVERY AUDIT"
    )

    print(
        "NO forecasting metric is generated. NO q/r is tuned on TestReal. "
        "This step only audits the original Kalman source and prior artifacts "
        "to recover the already-selected frozen tuning parameters.\n"
    )

    verify()

    text, context_df = source_context()
    ast_df = ast_selection_evidence(text)
    artifacts_df, candidate_df = scan_artifacts()
    outputs_df = extract_named_output_paths(text)

    context_path = OUT / "R2D_B2B_kalman_source_context.csv"
    ast_path = OUT / "R2D_B2B_kalman_selection_ast_evidence.csv"
    artifact_path = OUT / "R2D_B2B_kalman_artifact_inventory.csv"
    candidate_path = OUT / "R2D_B2B_qr_candidate_artifacts.csv"
    outputs_path = OUT / "R2D_B2B_named_output_files.csv"

    context_df.to_csv(context_path, index=False)
    ast_df.to_csv(ast_path, index=False)
    artifacts_df.to_csv(artifact_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    outputs_df.to_csv(outputs_path, index=False)

    if len(candidate_df) > 0:
        status = "R2D_B2B_AUDIT_COMPLETE_QR_CANDIDATE_ARTIFACTS_FOUND"
    else:
        status = "R2D_B2B_AUDIT_COMPLETE_NO_QR_ARTIFACT_FOUND_SOURCE_RECONSTRUCTION_REQUIRED"

    summary = {
        "step": "R2D_B2B_FROZEN_KALMAN_PARAMETER_RECOVERY_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "new_metrics_generated": False,
        "testreal_tuning": False,
        "kalman_source": str(KALMAN_SCRIPT),
        "kalman_source_sha256": sha256_file(KALMAN_SCRIPT),
        "source_context_csv": str(context_path),
        "selection_ast_csv": str(ast_path),
        "artifact_inventory_csv": str(artifact_path),
        "qr_candidate_artifacts_csv": str(candidate_path),
        "named_output_files_csv": str(outputs_path),
    }

    summary_path = OUT / "R2D_B2B_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("ORIGINAL KALMAN SOURCE — RELEVANT CONTEXT:")
    if len(context_df) > 0:
        mask = context_df["hit_text"].astype(str).str.contains(
            "best_params|idxmin|sort_values|ADE_track_macro|to_csv",
            case=False,
            regex=True,
        )
        show = context_df[mask].head(20)
        if len(show) == 0:
            show = context_df.head(20)

        for _, row in show.iterrows():
            print("\n---")
            print(row["context"])
    else:
        print("No source-context hits found.")

    print("\nQ/R CANDIDATE ARTIFACTS:")
    if len(candidate_df) > 0:
        print(
            candidate_df[
                [
                    "path",
                    "rows",
                    "q_col",
                    "r_col",
                    "task_col",
                    "metric_cols",
                ]
            ].to_string(index=False)
        )
    else:
        print("No q/r candidate artifact found.")

    print("\nNAMED OUTPUT FILES IN ORIGINAL KALMAN SOURCE:")
    print(
        outputs_df.to_string(index=False)
        if len(outputs_df) > 0
        else "No output filename literals found."
    )

    print("\nSTATUS:")
    print(status)

    print("\nR2D-B2B COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
