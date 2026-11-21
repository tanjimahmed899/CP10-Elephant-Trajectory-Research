from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

R5C_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R5_geofence_baselines"
    / "R5C_interpretation_claim_lock"
    / "R5C_MANIFEST.json"
)

STEP7_INSTANCES = (
    MAIN_RESULT_2
    / "step7_response_aware_warning"
    / "step7h_final_response_aware_instances.csv"
)

CANDIDATE_DIRS = [
    MAIN_RESULT / "step18_geofence",
    MAIN_RESULT / "multigeofence_audit",
    MAIN_RESULT_2 / "step7_response_aware_warning",
]

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6A_validation_only_preflight"
)

OUT.mkdir(parents=True, exist_ok=True)

FEATURE_TOKENS = [
    "distance",
    "dist",
    "predicted_crossing",
    "pred_cross",
    "predicted_tte",
    "tte",
    "min_dist",
    "current",
    "velocity",
    "speed",
    "heading",
    "uncertainty",
    "q90",
    "radius",
    "boundary",
    "geofence",
    "inside",
    "intersect",
]

LABEL_TOKENS = [
    "true_crossing",
    "actual_crossing",
    "crossing_bool",
    "urgent",
]

ID_TOKENS = [
    "dataset",
    "split",
    "source",
    "track",
    "sample",
    "task",
    "boundary",
    "geofence",
    "response",
    "threshold",
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


def verify_r5c():
    require(R5C_MANIFEST)

    obj = json.loads(
        R5C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        obj.get("status")
        !=
        "R5C_TRIVIAL_BASELINE_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R6_DIRECT_CROSSING_CLASSIFIER"
    ):
        raise RuntimeError(
            f"Unexpected R5C status: {obj.get('status')}"
        )

    return obj


def find_col(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def relevant_columns(df, tokens):
    return [
        str(c)
        for c in df.columns
        if any(tok in str(c).lower() for tok in tokens)
    ]


def scan_candidates():
    rows = []

    seen = set()

    for root in CANDIDATE_DIRS:
        if not root.exists():
            continue

        for p in root.rglob("*.csv"):
            key = str(p.resolve()).lower()

            if key in seen:
                continue

            seen.add(key)

            try:
                df = pd.read_csv(p)
            except Exception as e:
                rows.append(
                    {
                        "path": str(p),
                        "sha256": sha256_file(p),
                        "read_ok": False,
                        "rows": None,
                        "columns": None,
                        "feature_cols": None,
                        "label_cols": None,
                        "id_cols": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue

            features = relevant_columns(df, FEATURE_TOKENS)
            labels = relevant_columns(df, LABEL_TOKENS)
            ids = relevant_columns(df, ID_TOKENS)

            low_path = str(p).lower()

            relevance = 0

            if "validation" in low_path:
                relevance += 10

            if "instance" in low_path:
                relevance += 8

            if "step18" in low_path:
                relevance += 5

            if "multigeofence" in low_path:
                relevance += 5

            relevance += min(
                20,
                2 * len(features)
                +
                3 * len(labels)
                +
                len(ids),
            )

            rows.append(
                {
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "read_ok": True,
                    "rows": int(len(df)),
                    "columns": " | ".join(map(str, df.columns)),
                    "feature_cols": " | ".join(features),
                    "label_cols": " | ".join(labels),
                    "id_cols": " | ".join(ids),
                    "relevance_score": relevance,
                }
            )

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    return out.sort_values(
        ["relevance_score", "rows"],
        ascending=[False, False],
    )


def evaluation_source_sets():
    require(STEP7_INSTANCES)

    df = pd.read_csv(STEP7_INSTANCES)

    dataset_col = find_col(
        df,
        ["dataset"],
    )

    source_col = find_col(
        df,
        [
            "source_group",
            "source_id",
            "source",
            "video_id",
            "video",
            "sequence_id",
            "sequence",
        ],
    )

    if dataset_col is None or source_col is None:
        raise RuntimeError(
            "Step7H instances do not expose dataset/source columns needed for overlap audit."
        )

    rows = []

    for dataset, group in df.groupby(dataset_col, sort=True):
        vals = sorted(
            group[source_col]
            .astype(str)
            .unique()
            .tolist()
        )

        rows.append(
            {
                "dataset": str(dataset),
                "source_col": str(source_col),
                "source_count": len(vals),
                "sources_json": json.dumps(vals),
            }
        )

    return pd.DataFrame(rows), set(
        df[source_col].astype(str).unique()
    )


def audit_candidate_overlap(inventory, evaluation_sources):
    rows = []

    for _, inv in inventory.iterrows():
        if not bool(inv.get("read_ok", False)):
            continue

        p = Path(inv["path"])

        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        source_col = find_col(
            df,
            [
                "source_group",
                "source_id",
                "source",
                "video_id",
                "video",
                "sequence_id",
                "sequence",
            ],
        )

        split_col = find_col(
            df,
            [
                "split",
                "role",
                "dataset_role",
            ],
        )

        source_values = set()

        if source_col is not None:
            source_values = set(
                df[source_col]
                .astype(str)
                .dropna()
                .unique()
            )

        overlap = sorted(
            source_values
            &
            evaluation_sources
        )

        split_values = []

        if split_col is not None:
            split_values = sorted(
                df[split_col]
                .astype(str)
                .dropna()
                .unique()
                .tolist()
            )

        rows.append(
            {
                "path": str(p),
                "rows": int(len(df)),
                "source_col": str(source_col) if source_col is not None else None,
                "candidate_sources": int(len(source_values)),
                "evaluation_source_overlap_count": int(len(overlap)),
                "overlap_sources_json": json.dumps(overlap),
                "split_col": str(split_col) if split_col is not None else None,
                "split_values_json": json.dumps(split_values),
                "no_evaluation_source_overlap": bool(len(overlap) == 0)
                if source_col is not None
                else None,
            }
        )

    return pd.DataFrame(rows)


def classify_candidate_readiness(inventory, overlap):
    rows = []

    overlap_lookup = {
        row["path"]: row
        for _, row in overlap.iterrows()
    }

    for _, inv in inventory.iterrows():
        if not bool(inv.get("read_ok", False)):
            continue

        p = str(inv["path"])

        feature_cols = [
            x.strip()
            for x in str(inv.get("feature_cols", "")).split("|")
            if x.strip()
        ]

        label_cols = [
            x.strip()
            for x in str(inv.get("label_cols", "")).split("|")
            if x.strip()
        ]

        id_cols = [
            x.strip()
            for x in str(inv.get("id_cols", "")).split("|")
            if x.strip()
        ]

        ov = overlap_lookup.get(p)

        no_overlap = None

        if ov is not None:
            no_overlap = ov["no_evaluation_source_overlap"]

        path_low = p.lower()

        likely_validation = (
            "validation" in path_low
            or
            "train_core" in path_low
            or
            "trainreal" in path_low
        )

        has_label = len(label_cols) > 0
        has_features = len(feature_cols) >= 2
        has_ids = len(id_cols) > 0

        leakage_safe = (
            no_overlap is True
            or
            (
                no_overlap is None
                and
                likely_validation
            )
        )

        ready = bool(
            likely_validation
            and
            has_label
            and
            has_features
            and
            has_ids
            and
            leakage_safe
        )

        rows.append(
            {
                "path": p,
                "rows": int(inv["rows"]),
                "likely_validation_or_dev": likely_validation,
                "has_crossing_label": has_label,
                "has_multiple_candidate_features": has_features,
                "has_group_ids": has_ids,
                "no_eval_source_overlap": no_overlap,
                "leakage_safe_candidate": leakage_safe,
                "direct_classifier_candidate_ready": ready,
                "feature_cols": inv.get("feature_cols"),
                "label_cols": inv.get("label_cols"),
                "id_cols": inv.get("id_cols"),
            }
        )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            [
                "direct_classifier_candidate_ready",
                "likely_validation_or_dev",
                "rows",
            ],
            ascending=[False, False, False],
        )

    return out


def main():
    print(
        "\nR6-A — DIRECT CROSSING CLASSIFIER VALIDATION-ONLY PREFLIGHT"
    )

    print(
        "NO classifier is trained. NO threshold is selected. "
        "This step identifies a leakage-safe validation/development instance table "
        "with actual-crossing labels and forecast-derived candidate features, while "
        "auditing overlap against frozen TestReal/Fresh evaluation sources.\n"
    )

    verify_r5c()

    inventory = scan_candidates()
    eval_sources, eval_source_set = evaluation_source_sets()
    overlap = audit_candidate_overlap(
        inventory,
        eval_source_set,
    )
    readiness = classify_candidate_readiness(
        inventory,
        overlap,
    )

    inventory_path = OUT / "R6A_candidate_instance_inventory.csv"
    eval_path = OUT / "R6A_frozen_evaluation_source_sets.csv"
    overlap_path = OUT / "R6A_candidate_source_overlap_audit.csv"
    readiness_path = OUT / "R6A_direct_classifier_candidate_readiness.csv"

    inventory.to_csv(
        inventory_path,
        index=False,
    )

    eval_sources.to_csv(
        eval_path,
        index=False,
    )

    overlap.to_csv(
        overlap_path,
        index=False,
    )

    readiness.to_csv(
        readiness_path,
        index=False,
    )

    ready = readiness[
        readiness[
            "direct_classifier_candidate_ready"
        ]
        ==
        True
    ].copy()

    if len(ready) > 0:
        status = (
            "R6A_PREFLIGHT_COMPLETE_LEAKAGE_SAFE_VALIDATION_CANDIDATE_FOUND_READY_FOR_R6B_FEATURE_LOCK"
        )
    else:
        status = (
            "R6A_PREFLIGHT_COMPLETE_NO_PROVEN_LEAKAGE_SAFE_VALIDATION_CANDIDATE_TARGETED_RECONSTRUCTION_REQUIRED"
        )

    manifest = {
        "step":
            "R6A_DIRECT_CROSSING_CLASSIFIER_VALIDATION_ONLY_PREFLIGHT",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "status":
            status,

        "classifier_trained":
            False,

        "threshold_selected":
            False,

        "candidate_count":
            int(len(readiness)),

        "ready_candidate_count":
            int(len(ready)),

        "inventory_csv":
            str(inventory_path),

        "evaluation_source_csv":
            str(eval_path),

        "overlap_audit_csv":
            str(overlap_path),

        "readiness_csv":
            str(readiness_path),
    }

    manifest_path = OUT / "R6A_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("FROZEN EVALUATION SOURCE SETS:")
    print(
        eval_sources.to_string(
            index=False
        )
    )

    print("\nTOP DIRECT-CLASSIFIER CANDIDATE TABLES:")
    if len(readiness) > 0:
        print(
            readiness[
                [
                    "path",
                    "rows",
                    "likely_validation_or_dev",
                    "has_crossing_label",
                    "has_multiple_candidate_features",
                    "has_group_ids",
                    "no_eval_source_overlap",
                    "leakage_safe_candidate",
                    "direct_classifier_candidate_ready",
                    "feature_cols",
                    "label_cols",
                ]
            ].head(25).to_string(
                index=False
            )
        )
    else:
        print("No candidate CSV tables found.")

    print("\nLEAKAGE-SAFE READY CANDIDATES:")
    if len(ready) > 0:
        print(
            ready[
                [
                    "path",
                    "rows",
                    "feature_cols",
                    "label_cols",
                    "id_cols",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print("None")

    print("\nSTATUS:")
    print(status)

    print("\nR6-A COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
