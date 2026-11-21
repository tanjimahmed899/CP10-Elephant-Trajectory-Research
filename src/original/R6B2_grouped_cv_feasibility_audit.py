from pathlib import Path
import json
from itertools import combinations
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
    / "R6B2_grouped_cv_feasibility_audit"
)

OUT.mkdir(parents=True, exist_ok=True)


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
        raise RuntimeError(f"Unparseable true_crossing values: {bad[:20]}")

    return out.astype(int)


def source_distribution(df):
    y = parse_bool(df["true_crossing"])
    rows = []

    for (task, src), idx in df.groupby(
        ["task", "source_group"],
        sort=True,
    ).groups.items():
        yy = y.loc[idx]

        rows.append({
            "task": str(task),
            "source_group": str(src),
            "rows": int(len(idx)),
            "positives": int(yy.sum()),
            "negatives": int((1 - yy).sum()),
            "prevalence": float(yy.mean()),
            "has_both_classes": bool(yy.nunique() == 2),
        })

    return pd.DataFrame(rows)


def loso_audit(df):
    y = parse_bool(df["true_crossing"])
    rows = []

    for task, group in df.groupby("task", sort=True):
        sources = sorted(group["source_group"].astype(str).unique())

        for heldout in sources:
            train_idx = group[
                group["source_group"].astype(str) != heldout
            ].index

            test_idx = group[
                group["source_group"].astype(str) == heldout
            ].index

            y_train = y.loc[train_idx]
            y_test = y.loc[test_idx]

            rows.append({
                "task": str(task),
                "heldout_source": heldout,
                "train_rows": int(len(train_idx)),
                "train_positives": int(y_train.sum()),
                "train_negatives": int((1-y_train).sum()),
                "train_has_both_classes": bool(y_train.nunique() == 2),
                "heldout_rows": int(len(test_idx)),
                "heldout_positives": int(y_test.sum()),
                "heldout_negatives": int((1-y_test).sum()),
                "heldout_has_both_classes": bool(y_test.nunique() == 2),
            })

    return pd.DataFrame(rows)


def candidate_two_fold_partitions(df):
    """
    Enumerate unique source-group 2-fold partitions per task.
    A partition is feasible when BOTH training complements contain both classes.
    Since each fold's training set is the opposite fold, this also means both
    folds themselves must contain both classes.
    """
    y = parse_bool(df["true_crossing"])
    rows = []

    for task, group in df.groupby("task", sort=True):
        sources = sorted(group["source_group"].astype(str).unique())
        n = len(sources)

        # Unique bipartitions: force first source into fold A to avoid duplicates.
        first = sources[0]

        for r in range(1, n):
            for comb in combinations(sources[1:], r-1):
                fold_a = tuple(sorted((first,) + comb))
                fold_b = tuple(sorted(set(sources) - set(fold_a)))

                if not fold_b:
                    continue

                idx_a = group[
                    group["source_group"].astype(str).isin(fold_a)
                ].index

                idx_b = group[
                    group["source_group"].astype(str).isin(fold_b)
                ].index

                y_a = y.loc[idx_a]
                y_b = y.loc[idx_b]

                rows.append({
                    "task": str(task),
                    "fold_A_sources": json.dumps(fold_a),
                    "fold_B_sources": json.dumps(fold_b),
                    "fold_A_rows": int(len(idx_a)),
                    "fold_A_pos": int(y_a.sum()),
                    "fold_A_neg": int((1-y_a).sum()),
                    "fold_B_rows": int(len(idx_b)),
                    "fold_B_pos": int(y_b.sum()),
                    "fold_B_neg": int((1-y_b).sum()),
                    "fold_A_has_both": bool(y_a.nunique() == 2),
                    "fold_B_has_both": bool(y_b.nunique() == 2),
                    "both_training_complements_feasible": bool(
                        y_a.nunique() == 2 and y_b.nunique() == 2
                    ),
                    "row_imbalance_abs": int(abs(len(idx_a)-len(idx_b))),
                    "prevalence_gap_abs": float(abs(y_a.mean()-y_b.mean())),
                })

    return pd.DataFrame(rows)


def choose_deterministic_partition(candidates):
    """
    Fixed non-performance selection rule:
    1) feasible only;
    2) minimize absolute prevalence gap between folds;
    3) then minimize row-count imbalance;
    4) then lexicographic source list for deterministic tie break.
    """
    chosen = []

    for task, group in candidates.groupby("task", sort=True):
        feasible = group[
            group["both_training_complements_feasible"]
        ].copy()

        if len(feasible) == 0:
            raise RuntimeError(
                f"No feasible grouped 2-fold partition for task {task}"
            )

        feasible = feasible.sort_values(
            [
                "prevalence_gap_abs",
                "row_imbalance_abs",
                "fold_A_sources",
                "fold_B_sources",
            ],
            ascending=True,
            kind="mergesort",
        )

        row = feasible.iloc[0].copy()
        row["selection_rule"] = (
            "minimize prevalence gap; then row imbalance; "
            "then lexicographic source partition"
        )

        chosen.append(row)

    return pd.DataFrame(chosen)


def main():
    if not VALIDATION.exists():
        raise FileNotFoundError(VALIDATION)

    df = pd.read_csv(VALIDATION)

    required = {"task", "source_group", "true_crossing"}

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Validation table missing columns: {sorted(missing)}"
        )

    dist = source_distribution(df)
    loso = loso_audit(df)
    parts = candidate_two_fold_partitions(df)
    chosen = choose_deterministic_partition(parts)

    dist.to_csv(
        OUT / "R6B2_source_class_distribution.csv",
        index=False,
    )

    loso.to_csv(
        OUT / "R6B2_LOSO_failure_audit.csv",
        index=False,
    )

    parts.to_csv(
        OUT / "R6B2_all_twofold_group_partitions.csv",
        index=False,
    )

    chosen.to_csv(
        OUT / "R6B2_SELECTED_GROUPED_2FOLD_PARTITION.csv",
        index=False,
    )

    status = (
        "R6B2_GROUPED_2FOLD_FEASIBILITY_PROVEN_READY_FOR_PROTOCOL_REVISION"
    )

    summary = {
        "status": status,
        "validation_source": str(VALIDATION),
        "selection_uses_model_performance": False,
        "selection_uses_evaluation_data": False,
        "selection_rule": (
            "Among leakage-safe source-group 2-fold partitions with both classes "
            "in each fold, minimize fold prevalence gap; then row imbalance; "
            "then lexicographic source ordering."
        ),
    }

    (OUT / "R6B2_MANIFEST.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("SOURCE-LEVEL CLASS DISTRIBUTION:")
    print(dist.to_string(index=False))

    print("\nLOSO FAILURE AUDIT:")
    print(loso.to_string(index=False))

    print("\nSELECTED LEAKAGE-SAFE GROUPED 2-FOLD PARTITION:")
    print(chosen.to_string(index=False))

    print("\nSTATUS:")
    print(status)

    print("\nR6B2 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
