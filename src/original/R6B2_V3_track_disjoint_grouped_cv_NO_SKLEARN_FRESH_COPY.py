from pathlib import Path
import json
import hashlib
import math

import numpy as np
import pandas as pd


MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

VALIDATION = (
    MAIN_RESULT
    / "step18_geofence"
    / "step18a_validation_geofence_instances.csv"
)

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6B2V3_track_group_cv_no_sklearn"
)

OUT.mkdir(parents=True, exist_ok=True)

CANDIDATE_SPLITS = [5, 4, 3, 2]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        if vals.dropna().isin([0, 1]).all():
            return vals.astype(int)

    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
    }

    out = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if out.isna().any():
        bad = s[out.isna()].astype(str).unique().tolist()
        raise RuntimeError(
            f"Unparseable true_crossing values: {bad[:20]}"
        )

    return out.astype(int)


def source_distribution(df, y):
    rows = []

    for (task, src), idx in df.groupby(
        ["task", "source_group"],
        sort=True,
    ).groups.items():

        yy = y.loc[idx]

        rows.append(
            {
                "task": str(task),
                "source_group": str(src),
                "rows": int(len(idx)),
                "positives": int(yy.sum()),
                "negatives": int((1 - yy).sum()),
                "prevalence": float(yy.mean()),
                "has_both_classes": bool(yy.nunique() == 2),
            }
        )

    return pd.DataFrame(rows)


def source_loso_failure(df, y):
    rows = []

    for task, group in df.groupby("task", sort=True):
        sources = sorted(
            group["source_group"].astype(str).unique().tolist()
        )

        for heldout in sources:
            train_idx = group[
                group["source_group"].astype(str) != heldout
            ].index

            yy = y.loc[train_idx]

            rows.append(
                {
                    "task": str(task),
                    "heldout_source": heldout,
                    "train_rows": int(len(train_idx)),
                    "train_positives": int(yy.sum()),
                    "train_negatives": int((1 - yy).sum()),
                    "train_has_both_classes": bool(yy.nunique() == 2),
                }
            )

    return pd.DataFrame(rows)


def track_summary_for_task(group, y_task):
    tmp = group[
        [
            "track_key",
        ]
    ].copy()

    tmp["label"] = y_task.to_numpy(dtype=int)

    out = (
        tmp.groupby(
            "track_key",
            as_index=False,
        )
        .agg(
            rows=("label", "size"),
            positives=("label", "sum"),
        )
    )

    out["negatives"] = (
        out["rows"]
        -
        out["positives"]
    )

    out["prevalence"] = (
        out["positives"]
        /
        out["rows"]
    )

    out["has_positive"] = (
        out["positives"] > 0
    )

    out["has_negative"] = (
        out["negatives"] > 0
    )

    out["mixed_label_track"] = (
        out["has_positive"]
        &
        out["has_negative"]
    )

    return out


def deterministic_group_partition(track_df, n_splits):
    """
    Pure-pandas/numpy deterministic greedy partition.

    Objective is LABEL-DISTRIBUTION BALANCE ONLY, not model performance:
      - spread tracks with positive labels across folds,
      - spread negative-heavy tracks across folds,
      - balance fold row counts,
      - balance fold positive prevalence.

    No evaluation data and no classifier scores are used.
    """
    if len(track_df) < n_splits:
        return None

    total_rows = float(track_df["rows"].sum())
    total_pos = float(track_df["positives"].sum())
    total_neg = float(track_df["negatives"].sum())

    if total_pos <= 0 or total_neg <= 0:
        return None

    global_prev = total_pos / total_rows

    work = track_df.copy()

    # Harder / more label-informative tracks assigned first.
    work["extremeness"] = (
        work["prevalence"]
        -
        global_prev
    ).abs()

    work["minority_mass"] = np.minimum(
        work["positives"],
        work["negatives"],
    )

    work = work.sort_values(
        [
            "has_positive",
            "has_negative",
            "extremeness",
            "rows",
            "track_key",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    folds = []

    for i in range(n_splits):
        folds.append(
            {
                "fold": i,
                "tracks": [],
                "rows": 0,
                "positives": 0,
                "negatives": 0,
            }
        )

    target_rows = total_rows / n_splits
    target_pos = total_pos / n_splits
    target_neg = total_neg / n_splits

    # Seed each fold with one track first.
    for i in range(min(n_splits, len(work))):
        r = work.iloc[i]
        f = folds[i]

        f["tracks"].append(str(r["track_key"]))
        f["rows"] += int(r["rows"])
        f["positives"] += int(r["positives"])
        f["negatives"] += int(r["negatives"])

    # Greedy assignment for the rest.
    for i in range(n_splits, len(work)):
        r = work.iloc[i]

        candidates = []

        for f in folds:
            new_rows = f["rows"] + int(r["rows"])
            new_pos = f["positives"] + int(r["positives"])
            new_neg = f["negatives"] + int(r["negatives"])

            row_cost = abs(new_rows - target_rows) / max(target_rows, 1.0)
            pos_cost = abs(new_pos - target_pos) / max(target_pos, 1.0)
            neg_cost = abs(new_neg - target_neg) / max(target_neg, 1.0)

            prev = (
                new_pos / new_rows
                if new_rows > 0
                else global_prev
            )

            prev_cost = abs(prev - global_prev)

            # Deterministic balance objective.
            cost = (
                1.0 * row_cost
                +
                1.5 * pos_cost
                +
                1.0 * neg_cost
                +
                2.0 * prev_cost
            )

            candidates.append(
                (
                    cost,
                    f["rows"],
                    f["fold"],
                )
            )

        _, _, chosen_fold = min(
            candidates,
            key=lambda x: (x[0], x[1], x[2]),
        )

        f = folds[chosen_fold]

        f["tracks"].append(str(r["track_key"]))
        f["rows"] += int(r["rows"])
        f["positives"] += int(r["positives"])
        f["negatives"] += int(r["negatives"])

    return folds


def audit_partition(task_group, y_task, n_splits):
    track_df = track_summary_for_task(
        task_group,
        y_task,
    )

    folds = deterministic_group_partition(
        track_df,
        n_splits,
    )

    if folds is None:
        return None, "partition_not_constructed"

    task_group = task_group.copy()
    task_group["_label"] = y_task.to_numpy(dtype=int)

    fold_rows = []

    all_tracks = set(
        task_group["track_key"]
        .astype(str)
        .unique()
    )

    seen_val_tracks = set()

    for f in folds:
        val_tracks = set(
            f["tracks"]
        )

        train_tracks = (
            all_tracks
            -
            val_tracks
        )

        overlap = (
            val_tracks
            &
            train_tracks
        )

        val = task_group[
            task_group["track_key"]
            .astype(str)
            .isin(val_tracks)
        ]

        train = task_group[
            task_group["track_key"]
            .astype(str)
            .isin(train_tracks)
        ]

        y_val = val["_label"]
        y_train = train["_label"]

        seen_val_tracks |= val_tracks

        fold_rows.append(
            {
                "task": str(
                    task_group["task"].iloc[0]
                ),
                "n_splits": n_splits,
                "fold": int(f["fold"]) + 1,
                "train_rows": int(len(train)),
                "val_rows": int(len(val)),
                "train_positives": int(y_train.sum()),
                "train_negatives": int((1 - y_train).sum()),
                "val_positives": int(y_val.sum()),
                "val_negatives": int((1 - y_val).sum()),
                "train_has_both_classes": bool(
                    y_train.nunique() == 2
                ),
                "val_has_positive": bool(
                    y_val.sum() > 0
                ),
                "val_has_negative": bool(
                    (1 - y_val).sum() > 0
                ),
                "train_tracks": int(len(train_tracks)),
                "val_tracks": int(len(val_tracks)),
                "track_overlap_count": int(len(overlap)),
                "val_track_keys_json": json.dumps(
                    sorted(val_tracks)
                ),
            }
        )

    out = pd.DataFrame(
        fold_rows
    )

    all_rows_covered = bool(
        out["val_rows"].sum()
        ==
        len(task_group)
    )

    all_tracks_once = bool(
        seen_val_tracks
        ==
        all_tracks
    )

    feasible = bool(
        out["train_has_both_classes"].all()
        and
        (out["track_overlap_count"] == 0).all()
        and
        all_rows_covered
        and
        all_tracks_once
    )

    summary = {
        "task": str(
            task_group["task"].iloc[0]
        ),
        "n_splits": n_splits,
        "unique_tracks": len(all_tracks),
        "scheme_constructed": True,
        "all_train_folds_have_both_classes":
            bool(
                out["train_has_both_classes"].all()
            ),
        "all_val_folds_track_disjoint":
            bool(
                (out["track_overlap_count"] == 0).all()
            ),
        "all_rows_receive_oof_prediction":
            all_rows_covered,
        "all_tracks_heldout_exactly_once":
            all_tracks_once,
        "feasible":
            feasible,
        "error":
            None,
    }

    return (
        out,
        summary,
    )


def evaluate_schemes(df, y):
    fold_tables = []
    scheme_rows = []

    for task, group in df.groupby(
        "task",
        sort=True,
    ):
        group = group.reset_index(
            drop=True
        )

        y_task = y.loc[
            df["task"].astype(str)
            ==
            str(task)
        ].reset_index(
            drop=True
        )

        unique_tracks = int(
            group["track_key"]
            .astype(str)
            .nunique()
        )

        for k in CANDIDATE_SPLITS:
            if unique_tracks < k:
                scheme_rows.append(
                    {
                        "task": str(task),
                        "n_splits": k,
                        "unique_tracks": unique_tracks,
                        "scheme_constructed": False,
                        "all_train_folds_have_both_classes": False,
                        "all_val_folds_track_disjoint": False,
                        "all_rows_receive_oof_prediction": False,
                        "all_tracks_heldout_exactly_once": False,
                        "feasible": False,
                        "error": "fewer_tracks_than_splits",
                    }
                )
                continue

            result = audit_partition(
                group,
                y_task,
                k,
            )

            if len(result) == 2 and isinstance(result[1], str):
                _, err = result

                scheme_rows.append(
                    {
                        "task": str(task),
                        "n_splits": k,
                        "unique_tracks": unique_tracks,
                        "scheme_constructed": False,
                        "all_train_folds_have_both_classes": False,
                        "all_val_folds_track_disjoint": False,
                        "all_rows_receive_oof_prediction": False,
                        "all_tracks_heldout_exactly_once": False,
                        "feasible": False,
                        "error": err,
                    }
                )
                continue

            folds, summary = result

            fold_tables.append(
                folds
            )

            scheme_rows.append(
                summary
            )

    schemes = pd.DataFrame(
        scheme_rows
    )

    fold_details = (
        pd.concat(
            fold_tables,
            ignore_index=True,
        )
        if fold_tables
        else
        pd.DataFrame()
    )

    return (
        schemes,
        fold_details,
    )


def select_scheme(schemes):
    rows = []

    for task, group in schemes.groupby(
        "task",
        sort=True,
    ):
        feasible = group[
            group["feasible"]
            ==
            True
        ].copy()

        if len(feasible) == 0:
            raise RuntimeError(
                f"No feasible deterministic track-disjoint grouped CV scheme "
                f"for task {task}. Inspect feasibility CSV before proceeding."
            )

        chosen = feasible.sort_values(
            "n_splits",
            ascending=False,
        ).iloc[0]

        rows.append(
            {
                "task": str(task),
                "selected_n_splits": int(
                    chosen["n_splits"]
                ),
                "selection_rule":
                    (
                        "largest feasible deterministic track-disjoint "
                        "grouped-CV n_splits <= 5; "
                        "selection uses label distribution only, "
                        "not classifier performance"
                    ),
                "grouping_unit":
                    "track_key",
                "selection_uses_model_performance":
                    False,
                "selection_uses_evaluation_data":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    if not VALIDATION.exists():
        raise FileNotFoundError(
            VALIDATION
        )

    df = pd.read_csv(
        VALIDATION
    )

    required = {
        "task",
        "source_group",
        "track_key",
        "true_crossing",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"Validation table missing columns: {sorted(missing)}"
        )

    y = parse_bool(
        df["true_crossing"]
    )

    src_dist = source_distribution(
        df,
        y,
    )

    loso = source_loso_failure(
        df,
        y,
    )

    track_rows = []

    for task, group in df.groupby(
        "task",
        sort=True,
    ):
        mask = (
            df["task"].astype(str)
            ==
            str(task)
        )

        track_df = track_summary_for_task(
            group.reset_index(drop=True),
            y.loc[mask].reset_index(drop=True),
        )

        track_df["task"] = str(task)

        track_rows.append(
            track_df
        )

    track_dist = pd.concat(
        track_rows,
        ignore_index=True,
    )

    schemes, fold_details = evaluate_schemes(
        df,
        y,
    )

    selected = select_scheme(
        schemes
    )

    src_dist.to_csv(
        OUT / "R6B2V3_source_class_distribution.csv",
        index=False,
    )

    loso.to_csv(
        OUT / "R6B2V3_source_LOSO_failure_audit.csv",
        index=False,
    )

    track_dist.to_csv(
        OUT / "R6B2V3_track_class_distribution.csv",
        index=False,
    )

    schemes.to_csv(
        OUT / "R6B2V3_track_group_cv_feasibility.csv",
        index=False,
    )

    fold_details.to_csv(
        OUT / "R6B2V3_track_group_cv_fold_details.csv",
        index=False,
    )

    selected.to_csv(
        OUT / "R6B2V3_SELECTED_TRACK_GROUP_CV_SCHEME.csv",
        index=False,
    )

    status = (
        "R6B2V3_TRACK_DISJOINT_GROUPED_CV_FEASIBILITY_PROVEN_WITHOUT_SKLEARN_READY_FOR_PROTOCOL_REVISION"
    )

    manifest = {
        "status":
            status,

        "validation_source":
            str(VALIDATION),

        "validation_sha256":
            sha256_file(VALIDATION),

        "external_dependency_sklearn":
            False,

        "grouping_unit":
            "track_key",

        "selection_uses_model_performance":
            False,

        "selection_uses_evaluation_data":
            False,

        "selected_scheme_csv":
            str(
                OUT
                / "R6B2V3_SELECTED_TRACK_GROUP_CV_SCHEME.csv"
            ),
    }

    (
        OUT
        / "R6B2V3_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "SOURCE-LEVEL CLASS DISTRIBUTION:"
    )
    print(
        src_dist.to_string(
            index=False
        )
    )

    print(
        "\nSOURCE-LEVEL LOSO INFEASIBILITY:"
    )
    print(
        loso.to_string(
            index=False
        )
    )

    print(
        "\nTRACK-LEVEL CLASS DISTRIBUTION SUMMARY:"
    )

    track_summary = (
        track_dist.groupby(
            "task",
            as_index=False,
        )
        .agg(
            tracks=("track_key", "nunique"),
            tracks_with_positive=("has_positive", "sum"),
            tracks_with_negative=("has_negative", "sum"),
            mixed_label_tracks=("mixed_label_track", "sum"),
        )
    )

    print(
        track_summary.to_string(
            index=False
        )
    )

    print(
        "\nTRACK-GROUP CV FEASIBILITY:"
    )
    print(
        schemes.to_string(
            index=False
        )
    )

    print(
        "\nSELECTED TRACK-DISJOINT CV SCHEME:"
    )
    print(
        selected.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )
    print(
        status
    )

    print(
        "\nR6B2-V3 COMPLETE"
    )
    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
