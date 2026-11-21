
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

CANDIDATE_ROOTS = [
    DOWNLOADS / "step2_ego_motion_validation",
    DOWNLOADS / "step2b_raw_vs_corrected_baseline",
    DOWNLOADS / "step2c_ego_motion_geofence_ablation",
    Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0") / "step2_ego_motion_validation",
    Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0") / "step2b_raw_vs_corrected_baseline",
    Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0") / "step2c_ego_motion_geofence_ablation",
]

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R8_ego_motion_ablation"
    / "R8A_existing_artifact_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

KEYWORDS = [
    "raw",
    "corrected",
    "ego",
    "motion",
    "residual",
    "stationary",
    "ade",
    "fde",
    "precision",
    "recall",
    "f1",
    "false",
    "warning",
    "auroc",
    "auprc",
    "tte",
    "geofence",
    "improvement",
    "reduction",
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


def is_interesting_column(col):
    c = str(col).lower()
    return any(k in c for k in KEYWORDS)


def read_csv_safe(path):
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def summarize_csv(path, root_label):
    df, err = read_csv_safe(path)

    base = {
        "root_label": root_label,
        "path": str(path),
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": None,
        "cols": None,
        "interesting_columns": None,
        "all_columns": None,
        "read_error": err,
    }

    if df is None:
        return base, None

    interesting = [
        str(c)
        for c in df.columns
        if is_interesting_column(c)
    ]

    base.update(
        {
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "interesting_columns": " | ".join(interesting),
            "all_columns": " | ".join(map(str, df.columns)),
            "read_error": None,
        }
    )

    preview = None

    if interesting:
        keep = []
        for c in df.columns:
            if (
                str(c) in interesting
                or str(c).lower() in {
                    "dataset",
                    "task",
                    "model",
                    "sequence",
                    "sequence_id",
                    "source_group",
                    "track_key",
                    "horizon",
                    "split",
                    "metric",
                }
            ):
                keep.append(c)

        keep = keep[:30]

        if keep:
            preview = df[keep].head(30).copy()
            preview.insert(0, "_source_file", path.name)
            preview.insert(0, "_root_label", root_label)

    return base, preview


def main():
    print("\nR8-A — EXISTING EGO-MOTION / RAW-vs-CORRECTED ABLATION AUDIT")
    print(
        "NO new model training and NO new experiment. "
        "This step inventories existing Step2/2b/2c artifacts and identifies "
        "which tables already support the ego-motion claims.\n"
    )

    existing_roots = []
    missing_roots = []

    for root in CANDIDATE_ROOTS:
        if root.exists():
            existing_roots.append(root)
        else:
            missing_roots.append(root)

    if not existing_roots:
        raise RuntimeError(
            "None of the expected Step2/2b/2c roots were found."
        )

    inventory_rows = []
    preview_tables = []

    for root in existing_roots:
        root_label = root.name

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() == ".csv":
                info, preview = summarize_csv(
                    path,
                    root_label,
                )

                inventory_rows.append(info)

                if preview is not None:
                    preview_tables.append(preview)

            elif path.suffix.lower() in {
                ".json",
                ".txt",
                ".py",
            }:
                inventory_rows.append(
                    {
                        "root_label": root_label,
                        "path": str(path),
                        "filename": path.name,
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                        "rows": None,
                        "cols": None,
                        "interesting_columns": None,
                        "all_columns": None,
                        "read_error": None,
                    }
                )

    inventory = pd.DataFrame(inventory_rows)

    if len(inventory) == 0:
        raise RuntimeError(
            "Step2/2b/2c roots exist, but no CSV/JSON/TXT/PY artifacts were found."
        )

    csv_inventory = inventory[
        inventory["filename"].astype(str).str.lower().str.endswith(".csv")
    ].copy()

    likely = csv_inventory[
        csv_inventory["interesting_columns"]
        .fillna("")
        .astype(str)
        .str.len()
        > 0
    ].copy()

    likely["priority_score"] = (
        likely["interesting_columns"]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(
            lambda s: sum(
                1
                for k in KEYWORDS
                if k in s
            )
        )
    )

    likely = likely.sort_values(
        [
            "priority_score",
            "root_label",
            "filename",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort",
    )

    preview = (
        pd.concat(
            preview_tables,
            ignore_index=True,
            sort=False,
        )
        if preview_tables
        else
        pd.DataFrame()
    )

    inventory_path = OUT / "R8A_ALL_ARTIFACT_INVENTORY.csv"
    likely_path = OUT / "R8A_LIKELY_EGO_MOTION_RESULT_TABLES.csv"
    preview_path = OUT / "R8A_INTERESTING_TABLE_PREVIEWS.csv"

    inventory.to_csv(
        inventory_path,
        index=False,
    )

    likely.to_csv(
        likely_path,
        index=False,
    )

    preview.to_csv(
        preview_path,
        index=False,
    )

    status = (
        "R8A_EXISTING_EGO_MOTION_ARTIFACT_AUDIT_COMPLETE_READY_FOR_R8B_RESULT_EXTRACTION"
    )

    manifest = {
        "step":
            "R8A_EXISTING_EGO_MOTION_ARTIFACT_AUDIT",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "status":
            status,

        "new_experiment":
            False,

        "existing_roots":
            [str(x) for x in existing_roots],

        "missing_roots":
            [str(x) for x in missing_roots],

        "artifact_count":
            int(len(inventory)),

        "csv_count":
            int(len(csv_inventory)),

        "likely_result_table_count":
            int(len(likely)),

        "inventory_csv":
            str(inventory_path),

        "inventory_sha256":
            sha256_file(inventory_path),

        "likely_tables_csv":
            str(likely_path),

        "likely_tables_sha256":
            sha256_file(likely_path),

        "preview_csv":
            str(preview_path),
    }

    (
        OUT
        / "R8A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("EXISTING ROOTS:")
    for root in existing_roots:
        print("  FOUND:", root)

    if missing_roots:
        print("\nMISSING EXPECTED ROOTS:")
        for root in missing_roots:
            print("  MISSING:", root)

    print("\nLIKELY EGO-MOTION RESULT TABLES:")
    if len(likely):
        print(
            likely[
                [
                    "root_label",
                    "filename",
                    "rows",
                    "cols",
                    "priority_score",
                    "interesting_columns",
                    "path",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print("No likely CSV result tables identified.")

    print("\nSTATUS:")
    print(status)

    print("\nR8-A COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
