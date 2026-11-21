from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

R3C_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R3_clustered_bootstrap"
    / "R3C_interpretation_claim_lock"
    / "R3C_MANIFEST.json"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R4_conformal_coverage"
    / "R4A_frozen_conformal_method_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    MAIN_RESULT,
    MAIN_RESULT_2,
]

TARGET_TOKENS = [
    "conformal",
    "uncertainty",
    "crossfit",
    "cross_fit",
    "q90",
    "q95",
    "coverage",
    "calibration",
]

EXPECTED_Q = {
    "2s_to_2s": {
        "q90": 2.158767888280567,
        "q95": 2.560772554862189,
    },
    "2s_to_5s": {
        "q90": 1.975129668255982,
        "q95": 2.950976253259077,
    },
}


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


def verify_r3c():
    require(R3C_MANIFEST)

    obj = json.loads(
        R3C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        obj.get("status")
        !=
        "R3C_BOOTSTRAP_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R4_CONFORMAL_COVERAGE"
    ):
        raise RuntimeError(
            f"Unexpected R3C status: {obj.get('status')}"
        )

    return obj


def likely_relevant(path):
    low = str(path).lower()
    return any(tok in low for tok in TARGET_TOKENS)


def scan_files():
    rows = []

    suffixes = {
        ".py",
        ".csv",
        ".json",
        ".txt",
    }

    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            if p.suffix.lower() not in suffixes:
                continue

            if not likely_relevant(p):
                continue

            key = str(p.resolve()).lower()

            if key in seen:
                continue

            seen.add(key)

            row = {
                "path": str(p),
                "suffix": p.suffix.lower(),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }

            if p.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(p)
                    row["rows"] = int(len(df))
                    row["columns"] = " | ".join(map(str, df.columns))
                    row["read_ok"] = True
                except Exception as e:
                    row["rows"] = None
                    row["columns"] = None
                    row["read_ok"] = False
                    row["error"] = f"{type(e).__name__}: {e}"

            else:
                row["rows"] = None
                row["columns"] = None
                row["read_ok"] = True

            rows.append(row)

    return pd.DataFrame(rows)


def text_evidence(inventory):
    patterns = [
        r"\bq90\b",
        r"\bq95\b",
        r"2\.158767888280567",
        r"2\.560772554862189",
        r"1\.975129668255982",
        r"2\.950976253259077",
        r"coverage",
        r"conformity",
        r"score",
        r"quantile",
        r"cross.?fit",
        r"calibration",
        r"ensemble",
        r"spread",
        r"radius",
        r"error",
    ]

    rx = re.compile(
        "|".join(patterns),
        flags=re.IGNORECASE,
    )

    hits = []

    for _, row in inventory.iterrows():
        p = Path(row["path"])

        if p.suffix.lower() not in {
            ".py",
            ".json",
            ".txt",
        }:
            continue

        try:
            text = p.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        lines = text.splitlines()

        for i, line in enumerate(lines, start=1):
            if not rx.search(line):
                continue

            start = max(
                1,
                i - 4,
            )

            end = min(
                len(lines),
                i + 8,
            )

            context = "\n".join(
                f"{j}: {lines[j-1]}"
                for j in range(
                    start,
                    end + 1,
                )
            )

            hits.append(
                {
                    "path": str(p),
                    "line": i,
                    "hit": line.strip(),
                    "context": context,
                }
            )

    return pd.DataFrame(hits)


def csv_q_evidence(inventory):
    rows = []

    for _, inv in inventory.iterrows():
        if inv["suffix"] != ".csv" or not inv["read_ok"]:
            continue

        p = Path(inv["path"])

        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        cols_lower = {
            str(c).lower(): c
            for c in df.columns
        }

        q_cols = [
            c
            for c in df.columns
            if any(
                tok in str(c).lower()
                for tok in [
                    "q90",
                    "q95",
                    "quantile",
                    "conformal_q",
                    "q_90",
                    "q_95",
                ]
            )
        ]

        task_col = None

        for candidate in [
            "task",
            "forecast_task",
            "horizon",
        ]:
            if candidate in cols_lower:
                task_col = cols_lower[candidate]
                break

        coverage_cols = [
            c
            for c in df.columns
            if "coverage" in str(c).lower()
        ]

        if not q_cols and not coverage_cols:
            continue

        preview_cols = []

        if task_col is not None:
            preview_cols.append(task_col)

        preview_cols.extend(
            [
                c
                for c in q_cols + coverage_cols
                if c not in preview_cols
            ]
        )

        if not preview_cols:
            continue

        preview = df[
            preview_cols
        ].head(
            100
        )

        rows.append(
            {
                "path": str(p),
                "rows": int(len(df)),
                "task_col": str(task_col) if task_col is not None else None,
                "q_cols": " | ".join(map(str, q_cols)),
                "coverage_cols": " | ".join(map(str, coverage_cols)),
                "preview_json": preview.to_json(
                    orient="records"
                ),
            }
        )

    return pd.DataFrame(rows)


def exact_q_value_search(inventory):
    target_strings = {
        str(v)
        for task in EXPECTED_Q.values()
        for v in task.values()
    }

    rows = []

    for _, inv in inventory.iterrows():
        p = Path(inv["path"])

        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
                raw = df.to_csv(index=False)
            else:
                raw = p.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
        except Exception:
            continue

        found = [
            value
            for value in target_strings
            if value in raw
        ]

        if found:
            rows.append(
                {
                    "path": str(p),
                    "exact_q_values_found": " | ".join(found),
                }
            )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR4-A — FROZEN CONFORMAL METHOD / ARTIFACT AUDIT"
    )

    print(
        "NO recalibration. NO new coverage result. "
        "This step only identifies the exact frozen conformal method, "
        "q90/q95 provenance, score definition, and reusable evaluation artifacts.\n"
    )

    verify_r3c()

    inventory = scan_files()
    evidence = text_evidence(inventory)
    csv_evidence = csv_q_evidence(inventory)
    exact_q = exact_q_value_search(inventory)

    inventory_path = OUT / "R4A_conformal_artifact_inventory.csv"
    evidence_path = OUT / "R4A_text_method_evidence.csv"
    csv_path = OUT / "R4A_csv_q_coverage_evidence.csv"
    exact_q_path = OUT / "R4A_exact_q_value_locations.csv"

    inventory.to_csv(
        inventory_path,
        index=False,
    )

    evidence.to_csv(
        evidence_path,
        index=False,
    )

    csv_evidence.to_csv(
        csv_path,
        index=False,
    )

    exact_q.to_csv(
        exact_q_path,
        index=False,
    )

    if len(exact_q) > 0:
        status = (
            "R4A_CONFORMAL_AUDIT_COMPLETE_EXACT_FROZEN_Q_EVIDENCE_FOUND_READY_FOR_METHOD_LOCK"
        )
    else:
        status = (
            "R4A_CONFORMAL_AUDIT_COMPLETE_EXACT_Q_STRING_NOT_FOUND_RECONSTRUCTION_REQUIRED"
        )

    summary = {
        "step": "R4A_FROZEN_CONFORMAL_METHOD_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "recalibration": False,
        "new_coverage_generated": False,
        "expected_frozen_q": EXPECTED_Q,
        "artifact_inventory": str(inventory_path),
        "text_method_evidence": str(evidence_path),
        "csv_q_coverage_evidence": str(csv_path),
        "exact_q_value_locations": str(exact_q_path),
    }

    manifest_path = OUT / "R4A_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("HIGH-VALUE ARTIFACT INVENTORY:")
    if len(inventory) > 0:
        show = inventory[
            inventory["path"]
            .astype(str)
            .str.contains(
                "step6|conformal|uncertainty",
                case=False,
                regex=True,
            )
        ].copy()

        print(
            show[
                [
                    "path",
                    "suffix",
                    "rows",
                    "columns",
                ]
            ].head(80).to_string(
                index=False
            )
        )
    else:
        print("No relevant artifacts found.")

    print("\nEXACT FROZEN q VALUE LOCATIONS:")
    if len(exact_q) > 0:
        print(
            exact_q.to_string(
                index=False
            )
        )
    else:
        print(
            "No exact q strings found."
        )

    print("\nCSV q/COVERAGE EVIDENCE:")
    if len(csv_evidence) > 0:
        print(
            csv_evidence[
                [
                    "path",
                    "rows",
                    "task_col",
                    "q_cols",
                    "coverage_cols",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No CSV q/coverage columns found."
        )

    print("\nMETHOD TEXT EVIDENCE — FIRST 30 HITS:")
    if len(evidence) > 0:
        for _, row in evidence.head(30).iterrows():
            print("\n---")
            print(row["path"])
            print(row["context"])
    else:
        print("No text evidence found.")

    print("\nSTATUS:")
    print(status)

    print("\nR4-A COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
