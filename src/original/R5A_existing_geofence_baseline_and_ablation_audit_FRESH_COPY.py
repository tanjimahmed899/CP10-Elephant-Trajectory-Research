from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

R4C_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R4_conformal_coverage"
    / "R4C_interpretation_claim_lock"
    / "R4C_MANIFEST.json"
)

STEP7_DIR = MAIN_RESULT_2 / "step7_response_aware_warning"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R5_geofence_baselines"
    / "R5A_existing_geofence_baseline_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

TOKENS = [
    "baseline",
    "always",
    "never",
    "trivial",
    "crossing",
    "classifier",
    "red",
    "yellow",
    "green",
    "geofence",
    "warning",
    "step7h",
    "step18",
    "persistence",
    "sensitivity",
    "specificity",
    "precision",
    "false",
    "lead",
    "tte",
    "q90",
    "nominal",
]


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


def verify_r4c():
    require(R4C_MANIFEST)

    obj = json.loads(
        R4C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        obj.get("status")
        !=
        "R4C_FIXED_CONFORMAL_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R5_GEOFENCE_BASELINES"
    ):
        raise RuntimeError(
            f"Unexpected R4C status: {obj.get('status')}"
        )

    return obj


def scan_inventory():
    if not STEP7_DIR.exists():
        raise FileNotFoundError(STEP7_DIR)

    rows = []

    for p in STEP7_DIR.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in {
            ".py",
            ".csv",
            ".json",
            ".txt",
        }:
            continue

        low = str(p).lower()

        if not any(tok in low for tok in TOKENS):
            continue

        row = {
            "path": str(p),
            "suffix": p.suffix.lower(),
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }

        if p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(p)
                row["read_ok"] = True
                row["rows"] = int(len(df))
                row["columns"] = " | ".join(map(str, df.columns))
            except Exception as e:
                row["read_ok"] = False
                row["rows"] = None
                row["columns"] = None
                row["error"] = f"{type(e).__name__}: {e}"
        else:
            row["read_ok"] = True
            row["rows"] = None
            row["columns"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def score_relevance(path, columns=""):
    text = (str(path) + " " + str(columns)).lower()

    score = 0

    weights = {
        "step7h": 8,
        "step18": 6,
        "baseline": 6,
        "always": 7,
        "never": 7,
        "trivial": 8,
        "classifier": 7,
        "crossing": 5,
        "red": 4,
        "yellow": 4,
        "green": 4,
        "sensitivity": 5,
        "precision": 5,
        "false": 5,
        "persistence": 4,
        "q90": 3,
        "tte": 3,
        "lead": 3,
    }

    for token, weight in weights.items():
        if token in text:
            score += weight

    return score


def high_value_csvs(inventory):
    rows = []

    for _, inv in inventory.iterrows():
        if inv["suffix"] != ".csv" or not inv.get("read_ok", False):
            continue

        p = Path(inv["path"])

        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        cols = [str(c) for c in df.columns]
        low_cols = [c.lower() for c in cols]

        metric_hits = [
            c
            for c in cols
            if any(
                token in c.lower()
                for token in [
                    "sensitivity",
                    "specificity",
                    "precision",
                    "recall",
                    "false",
                    "alert",
                    "warning",
                    "red",
                    "yellow",
                    "green",
                    "crossing",
                    "tte",
                    "lead",
                    "persistence",
                    "coverage",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                ]
            )
        ]

        model_hits = [
            c
            for c in cols
            if any(
                token in c.lower()
                for token in [
                    "model",
                    "method",
                    "baseline",
                    "rule",
                    "variant",
                    "policy",
                    "alarm",
                ]
            )
        ]

        rows.append(
            {
                "path": str(p),
                "rows": int(len(df)),
                "metric_cols": " | ".join(metric_hits),
                "model_or_rule_cols": " | ".join(model_hits),
                "relevance_score": score_relevance(
                    p,
                    " ".join(cols),
                ),
                "preview_json": df.head(25).to_json(
                    orient="records"
                ),
            }
        )

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    return out.sort_values(
        ["relevance_score", "rows"],
        ascending=[False, False],
    )


def text_hits(inventory):
    rx = re.compile(
        r"(always|never|trivial|baseline|direct crossing|crossing classifier|"
        r"red|yellow|green|step7h|step18|persistence|false alert|"
        r"sensitivity|precision|q90|tte|lead time)",
        flags=re.IGNORECASE,
    )

    rows = []

    for _, inv in inventory.iterrows():
        if inv["suffix"] not in {".py", ".txt", ".json"}:
            continue

        p = Path(inv["path"])

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

            start = max(1, i - 4)
            end = min(len(lines), i + 8)

            rows.append(
                {
                    "path": str(p),
                    "line": i,
                    "hit": line.strip(),
                    "context": "\n".join(
                        f"{j}: {lines[j-1]}"
                        for j in range(start, end + 1)
                    ),
                }
            )

    return pd.DataFrame(rows)


def baseline_presence(high_value, text_evidence):
    corpus = ""

    if len(high_value) > 0:
        corpus += "\n".join(
            high_value[
                [
                    "path",
                    "metric_cols",
                    "model_or_rule_cols",
                    "preview_json",
                ]
            ]
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )

    if len(text_evidence) > 0:
        corpus += "\n" + "\n".join(
            text_evidence[
                [
                    "path",
                    "hit",
                    "context",
                ]
            ]
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )

    low = corpus.lower()

    checks = [
        {
            "candidate": "always_alert_or_always_red",
            "found": (
                "always red" in low
                or
                "always_alert" in low
                or
                "always alert" in low
            ),
        },
        {
            "candidate": "never_alert_or_always_green",
            "found": (
                "always green" in low
                or
                "never_alert" in low
                or
                "never alert" in low
            ),
        },
        {
            "candidate": "current_state_only",
            "found": (
                "current state" in low
                or
                "current_position" in low
                or
                "already inside" in low
            ),
        },
        {
            "candidate": "nominal_forecast_only",
            "found": (
                "nominal" in low
                and
                "cross" in low
            ),
        },
        {
            "candidate": "uncertainty_q90_only",
            "found": (
                "q90" in low
                and
                ("yellow" in low or "uncertainty" in low)
            ),
        },
        {
            "candidate": "direct_crossing_classifier",
            "found": (
                "crossing classifier" in low
                or
                "direct crossing" in low
            ),
        },
        {
            "candidate": "persistence_ablation",
            "found": "persistence" in low,
        },
        {
            "candidate": "step7h_primary_protocol",
            "found": "step7h" in low,
        },
        {
            "candidate": "step18_secondary_protocol",
            "found": "step18" in low,
        },
    ]

    return pd.DataFrame(checks)


def main():
    print(
        "\nR5-A — EXISTING GEOFENCE BASELINE / ABLATION AUDIT"
    )

    print(
        "NO new warning experiment is run. This step audits existing Step7/Step18 "
        "artifacts to identify which trivial baselines and ablations already exist "
        "and which must be added for reviewer-facing comparison.\n"
    )

    verify_r4c()

    inventory = scan_inventory()
    high_value = high_value_csvs(inventory)
    text_evidence = text_hits(inventory)
    presence = baseline_presence(
        high_value,
        text_evidence,
    )

    inventory_path = OUT / "R5A_step7_artifact_inventory.csv"
    high_path = OUT / "R5A_high_value_geofence_csvs.csv"
    text_path = OUT / "R5A_geofence_text_evidence.csv"
    presence_path = OUT / "R5A_baseline_presence_matrix.csv"

    inventory.to_csv(inventory_path, index=False)
    high_value.to_csv(high_path, index=False)
    text_evidence.to_csv(text_path, index=False)
    presence.to_csv(presence_path, index=False)

    missing_trivial = presence[
        presence["candidate"].isin(
            [
                "always_alert_or_always_red",
                "never_alert_or_always_green",
                "current_state_only",
                "nominal_forecast_only",
            ]
        )
        &
        (~presence["found"])
    ]["candidate"].tolist()

    if len(missing_trivial) == 0:
        status = (
            "R5A_AUDIT_COMPLETE_TRIVIAL_BASELINES_ALREADY_PRESENT_READY_FOR_VALIDATION"
        )
    else:
        status = (
            "R5A_AUDIT_COMPLETE_MISSING_TRIVIAL_BASELINES_REQUIRE_TARGETED_ADDITION"
        )

    manifest = {
        "step": "R5A_EXISTING_GEOFENCE_BASELINE_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "new_warning_experiment": False,
        "step7_root": str(STEP7_DIR),
        "missing_trivial_baselines": missing_trivial,
        "inventory_csv": str(inventory_path),
        "high_value_csv": str(high_path),
        "text_evidence_csv": str(text_path),
        "presence_matrix_csv": str(presence_path),
    }

    manifest_path = OUT / "R5A_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("BASELINE PRESENCE MATRIX:")
    print(
        presence.to_string(
            index=False
        )
    )

    print("\nTOP HIGH-VALUE GEOFENCE CSVs:")
    if len(high_value) > 0:
        print(
            high_value[
                [
                    "path",
                    "rows",
                    "metric_cols",
                    "model_or_rule_cols",
                    "relevance_score",
                ]
            ].head(30).to_string(index=False)
        )
    else:
        print("No high-value CSVs found.")

    print("\nTEXT EVIDENCE — FIRST 30 HITS:")
    if len(text_evidence) > 0:
        for _, row in text_evidence.head(30).iterrows():
            print("\n---")
            print(row["path"])
            print(row["context"])
    else:
        print("No text evidence found.")

    print("\nMISSING TRIVIAL BASELINES:")
    print(
        missing_trivial
        if missing_trivial
        else "None"
    )

    print("\nSTATUS:")
    print(status)

    print("\nR5-A COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
