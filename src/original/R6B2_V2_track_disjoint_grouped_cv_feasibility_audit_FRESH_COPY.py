from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


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
    / "R6B2V2_track_group_cv_feasibility_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

CANDIDATE_SPLITS = [5, 4, 3, 2]
RANDOM_STATE = 20261029


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


def track_distribution(df, y):
    rows = []

    for (task, track), idx in df.groupby(
        ["task", "track_key"],
        sort=True,
    ).groups.items():

        yy = y.loc[idx]

        rows.append(
            {
                "task": str(task),
                "track_key": str(track),
                "rows": int(len(idx)),
                "positives": int(yy.sum()),
                "negatives": int((1 - yy).sum()),
                "prevalence": float(yy.mean()),
                "has_positive": bool(yy.sum() > 0),
                "has_negative": bool((1 - yy).sum() > 0),
                "has_both_classes_within_track": bool(yy.nunique() == 2),
            }
        )

    return pd.DataFrame(rows)


def source_level_failure_summary(df, y):
    rows = []

    for task, group in df.groupby(
        "task",
        sort=True,
    ):
        sources = sorted(
            group["source_group"]
            .astype(str)
            .unique()
            .tolist()
        )

        for heldout in sources:
            train = group[
                group["source_group"].astype(str) != heldout
            ]

            yy = y.loc[
                train.index
            ]

            rows.append(
                {
                    "task": str(task),
                    "heldout_source": heldout,
                    "train_rows": int(len(train)),
                    "train_positives": int(yy.sum()),
                    "train_negatives": int((1 - yy).sum()),
                    "train_has_both_classes": bool(yy.nunique() == 2),
                }
            )

    return pd.DataFrame(rows)


def audit_sgkf_for_task(group, y_task, n_splits):
    X_dummy = np.zeros((len(group), 1), dtype=float)
    groups = group["track_key"].astype(str).to_numpy()
    y_arr = y_task.to_numpy(dtype=int)

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    try:
        splits = list(
            cv.split(
                X_dummy,
                y_arr,
                groups=groups,
            )
        )
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    for fold_id, (train_idx, val_idx) in enumerate(
        splits,
        start=1,
    ):
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])

        overlap = train_groups & val_groups

        rows.append(
            {
                "fold": fold_id,
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "train_positives": int(y_train.sum()),
                "train_negatives": int((1 - y_train).sum()),
                "val_positives": int(y_val.sum()),
                "val_negatives": int((1 - y_val).sum()),
                "train_has_both_classes": bool(
                    len(np.unique(y_train)) == 2
                ),
                "val_has_both_classes": bool(
                    len(np.unique(y_val)) == 2
                ),
                "train_tracks": int(len(train_groups)),
                "val_tracks": int(len(val_groups)),
                "track_overlap_count": int(len(overlap)),
            }
        )

    return pd.DataFrame(rows), None


def evaluate_candidate_splits(df, y):
    scheme_rows = []
    fold_rows = []

    for task, group in df.groupby(
        "task",
        sort=True,
    ):
        y_task = y.loc[group.index].reset_index(drop=True)
        group = group.reset_index(drop=True)

        unique_tracks = int(
            group["track_key"]
            .astype(str)
            .nunique()
        )

        for n_splits in CANDIDATE_SPLITS:
            if unique_tracks < n_splits:
                scheme_rows.append(
                    {
                        "task": str(task),
                        "n_splits": n_splits,
                        "unique_tracks": unique_tracks,
                        "scheme_constructed": False,
                        "all_train_folds_have_both_classes": False,
                        "all_val_folds_track_disjoint": False,
                        "all_rows_receive_oof_prediction": False,
                        "error": "fewer unique tracks than n_splits",
                    }
                )
                continue

            folds, error = audit_sgkf_for_task(
                group,
                y_task,
                n_splits,
            )

            if error is not None:
                scheme_rows.append(
                    {
                        "task": str(task),
                        "n_splits": n_splits,
                        "unique_tracks": unique_tracks,
                        "scheme_constructed": False,
                        "all_train_folds_have_both_classes": False,
                        "all_val_folds_track_disjoint": False,
                        "all_rows_receive_oof_prediction": False,
                        "error": error,
                    }
                )
                continue

            folds = folds.copy()
            folds["task"] = str(task)
            folds["n_splits"] = n_splits

            fold_rows.append(
                folds
            )

            feasible = bool(
                folds["train_has_both_classes"].all()
                and
                (folds["track_overlap_count"] == 0).all()
                and
                folds["val_rows"].sum() == len(group)
            )

            scheme_rows.append(
                {
                    "task": str(task),
                    "n_splits": n_splits,
                    "unique_tracks": unique_tracks,
                    "scheme_constructed": True,
                    "all_train_folds_have_both_classes":
                        bool(
                            folds["train_has_both_classes"].all()
                        ),
                    "all_val_folds_track_disjoint":
                        bool(
                            (folds["track_overlap_count"] == 0).all()
                        ),
                    "all_rows_receive_oof_prediction":
                        bool(
                            folds["val_rows"].sum() == len(group)
                        ),
                    "feasible": feasible,
                    "error": None,
                }
            )

    schemes = pd.DataFrame(
        scheme_rows
    )

    fold_detail = (
        pd.concat(
            fold_rows,
            ignore_index=True,
        )
        if fold_rows
        else
        pd.DataFrame()
    )

    return schemes, fold_detail


def select_scheme(schemes):
    rows = []

    for task, group in schemes.groupby(
        "task",
        sort=True,
    ):
        feasible = group[
            group.get("feasible", False) == True
        ].copy()

        if len(feasible) == 0:
            raise RuntimeError(
                f"No feasible track-group StratifiedGroupKFold scheme "
                f"for task {task}"
            )

        # Deterministic non-performance rule:
        # choose the largest feasible number of folds, up to 5.
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
                        "largest feasible track-disjoint "
                        "StratifiedGroupKFold n_splits <= 5; "
                        "no model performance used"
                    ),
                "random_state": RANDOM_STATE,
                "selection_uses_model_performance": False,
                "selection_uses_evaluation_data": False,
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

    track_dist = track_distribution(
        df,
        y,
    )

    loso = source_level_failure_summary(
        df,
        y,
    )

    schemes, fold_detail = evaluate_candidate_splits(
        df,
        y,
    )

    selected = select_scheme(
        schemes
    )

    src_dist.to_csv(
        OUT / "R6B2V2_source_class_distribution.csv",
        index=False,
    )

    track_dist.to_csv(
        OUT / "R6B2V2_track_class_distribution.csv",
        index=False,
    )

    loso.to_csv(
        OUT / "R6B2V2_source_LOSO_failure_audit.csv",
        index=False,
    )

    schemes.to_csv(
        OUT / "R6B2V2_track_group_cv_scheme_feasibility.csv",
        index=False,
    )

    fold_detail.to_csv(
        OUT / "R6B2V2_track_group_cv_fold_details.csv",
        index=False,
    )

    selected.to_csv(
        OUT / "R6B2V2_SELECTED_TRACK_GROUP_CV_SCHEME.csv",
        index=False,
    )

    status = (
        "R6B2V2_TRACK_DISJOINT_GROUPED_CV_FEASIBILITY_PROVEN_READY_FOR_PROTOCOL_REVISION"
    )

    manifest = {
        "status": status,
        "validation_source": str(VALIDATION),
        "validation_sha256": sha256_file(VALIDATION),
        "source_group_cv_feasible": False,
        "fallback_grouping_unit": "track_key",
        "cv_family": "StratifiedGroupKFold",
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "selection_uses_model_performance": False,
        "selection_uses_evaluation_data": False,
        "selected_scheme_csv":
            str(
                OUT
                / "R6B2V2_SELECTED_TRACK_GROUP_CV_SCHEME.csv"
            ),
    }

    (
        OUT
        / "R6B2V2_MANIFEST.json"
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

    summary = (
        track_dist.groupby(
            "task",
            as_index=False,
        )
        .agg(
            tracks=("track_key", "nunique"),
            tracks_with_positive=("has_positive", "sum"),
            tracks_with_negative=("has_negative", "sum"),
            mixed_label_tracks=(
                "has_both_classes_within_track",
                "sum",
            ),
        )
    )

    print(
        summary.to_string(
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
        "\nR6B2-V2 COMPLETE"
    )
    print(
        "Outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
