from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

VALIDATION = (
    MAIN_RESULT
    / "step18_geofence"
    / "step18a_validation_geofence_instances.csv"
)

EVALUATION = (
    MAIN_RESULT
    / "step18_geofence"
    / "step18b_evaluation"
    / "step18b_all_geofence_instances.csv"
)

R6B3_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6B3_revised_protocol_lock"
)

R6B3_MANIFEST = R6B3_DIR / "R6B3_MANIFEST.json"
R6B3_PROTOCOL = R6B3_DIR / "R6B3_LOCKED_DIRECT_CLASSIFIER_PROTOCOL.json"
R6B3_FOLDS = R6B3_DIR / "R6B3_LOCKED_5FOLD_TRACK_PARTITIONS.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R6_direct_crossing_classifier"
    / "R6C_oof_threshold_lock_and_frozen_evaluation"
)

OUT.mkdir(parents=True, exist_ok=True)

LABEL = "true_crossing"

FEATURES = [
    "distance_to_boundary",
    "predicted_crossing_bool",
    "predicted_tte_norm",
    "predicted_tte_available",
]

TASK_HORIZON_SEC = {
    "2s_to_2s": 2.0,
    "2s_to_5s": 5.0,
}

L2_LAMBDA = 1.0
MAX_ITER = 200
TOL = 1e-9
DAMPING = 1e-8


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


def verify_protocol():
    for p in [
        VALIDATION,
        EVALUATION,
        R6B3_MANIFEST,
        R6B3_PROTOCOL,
        R6B3_FOLDS,
    ]:
        require(p)

    manifest = json.loads(
        R6B3_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("status")
        !=
        "R6B3_REVISED_TRACK_DISJOINT_FEATURE_MODEL_THRESHOLD_PROTOCOL_LOCKED_READY_FOR_R6C_OOF_FIT"
    ):
        raise RuntimeError(
            f"Unexpected R6B3 status: {manifest.get('status')}"
        )

    protocol = json.loads(
        R6B3_PROTOCOL.read_text(
            encoding="utf-8"
        )
    )

    if protocol["derived_features_in_order"] != FEATURES:
        raise RuntimeError(
            "Locked feature order does not match R6C implementation."
        )

    if float(protocol["model_protocol"]["l2_lambda"]) != L2_LAMBDA:
        raise RuntimeError(
            "Locked L2 value does not match R6C implementation."
        )

    if protocol["threshold_protocol"]["selection_metric"] != "balanced_accuracy":
        raise RuntimeError(
            "Locked threshold metric is not balanced_accuracy."
        )

    if manifest.get("source_overlap_count") != 0:
        raise RuntimeError(
            "Protocol manifest does not prove zero source overlap."
        )

    return manifest, protocol


def parse_bool_series(s, name):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(
            s,
            errors="coerce",
        )

        if vals.dropna().isin(
            [0, 1]
        ).all():
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
        bad = (
            s[out.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"{name}: unparseable binary values {bad[:20]}"
        )

    return out.astype(int)


def transform_features(df):
    x = pd.DataFrame(
        index=df.index
    )

    x["distance_to_boundary"] = pd.to_numeric(
        df["distance_to_boundary"],
        errors="raise",
    ).astype(float)

    x["predicted_crossing_bool"] = parse_bool_series(
        df["predicted_crossing"],
        "predicted_crossing",
    ).astype(float)

    raw_tte = pd.to_numeric(
        df["predicted_tte_sec"],
        errors="coerce",
    )

    horizon = (
        df["task"]
        .astype(str)
        .map(TASK_HORIZON_SEC)
        .astype(float)
    )

    if horizon.isna().any():
        raise RuntimeError(
            "Unknown task while normalizing predicted TTE."
        )

    x["predicted_tte_norm"] = (
        raw_tte
        /
        horizon
    ).clip(
        lower=0.0,
        upper=1.25,
    ).fillna(
        1.25
    ).astype(float)

    x["predicted_tte_available"] = (
        raw_tte.notna()
    ).astype(float)

    x = x[
        FEATURES
    ].copy()

    if x.isna().any().any():
        raise RuntimeError(
            "Derived feature matrix contains missing values."
        )

    return x


def fit_scaler(X):
    mean = X.mean(
        axis=0
    )

    std = X.std(
        axis=0,
        ddof=0,
    )

    std = np.where(
        std < 1e-12,
        1.0,
        std,
    )

    return mean, std


def apply_scaler(X, mean, std):
    return (
        X
        -
        mean
    ) / std


def sigmoid(z):
    z = np.clip(
        z,
        -40,
        40,
    )

    return 1.0 / (
        1.0
        +
        np.exp(-z)
    )


def balanced_sample_weights(y):
    y = np.asarray(
        y,
        dtype=int,
    )

    n = len(y)
    pos = int(
        y.sum()
    )
    neg = int(
        n - pos
    )

    if pos == 0 or neg == 0:
        raise RuntimeError(
            "Balanced logistic regression requires both classes."
        )

    w_pos = n / (
        2.0 * pos
    )

    w_neg = n / (
        2.0 * neg
    )

    return np.where(
        y == 1,
        w_pos,
        w_neg,
    )


def weighted_penalized_loss(
    X_aug,
    y,
    sample_w,
    beta,
):
    p = sigmoid(
        X_aug @ beta
    )

    eps = 1e-12

    ce = -np.sum(
        sample_w
        *
        (
            y
            *
            np.log(
                p + eps
            )
            +
            (1 - y)
            *
            np.log(
                1 - p + eps
            )
        )
    )

    penalty = (
        0.5
        *
        L2_LAMBDA
        *
        np.sum(
            beta[1:] ** 2
        )
    )

    return float(
        ce + penalty
    )


def fit_logistic_irls(X, y):
    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    n, p = X.shape

    if len(
        np.unique(y)
    ) != 2:
        raise RuntimeError(
            "IRLS training received a single-class target."
        )

    X_aug = np.column_stack(
        [
            np.ones(n),
            X,
        ]
    )

    beta = np.zeros(
        p + 1,
        dtype=float,
    )

    sample_w = balanced_sample_weights(
        y.astype(int)
    )

    penalty = np.eye(
        p + 1,
        dtype=float,
    )

    penalty[0, 0] = 0.0

    converged = False
    previous_loss = weighted_penalized_loss(
        X_aug,
        y,
        sample_w,
        beta,
    )

    for iteration in range(
        1,
        MAX_ITER + 1,
    ):
        eta = X_aug @ beta
        prob = sigmoid(
            eta
        )

        variance = (
            prob
            *
            (
                1.0 - prob
            )
        )

        variance = np.clip(
            variance,
            1e-8,
            None,
        )

        W = (
            sample_w
            *
            variance
        )

        grad = (
            X_aug.T
            @
            (
                sample_w
                *
                (
                    prob - y
                )
            )
            +
            L2_LAMBDA
            *
            penalty
            @
            beta
        )

        hessian = (
            X_aug.T
            @
            (
                W[:, None]
                *
                X_aug
            )
            +
            L2_LAMBDA
            *
            penalty
            +
            DAMPING
            *
            np.eye(
                p + 1
            )
        )

        try:
            step = np.linalg.solve(
                hessian,
                grad,
            )
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(
                hessian,
                grad,
                rcond=None,
            )[0]

        # Backtracking guarantees stable descent.
        step_scale = 1.0
        accepted = False

        for _ in range(30):
            candidate = (
                beta
                -
                step_scale
                *
                step
            )

            loss = weighted_penalized_loss(
                X_aug,
                y,
                sample_w,
                candidate,
            )

            if (
                np.isfinite(loss)
                and
                loss
                <=
                previous_loss
                +
                1e-10
            ):
                accepted = True
                beta_new = candidate
                new_loss = loss
                break

            step_scale *= 0.5

        if not accepted:
            beta_new = (
                beta
                -
                1e-4
                *
                grad
            )

            new_loss = weighted_penalized_loss(
                X_aug,
                y,
                sample_w,
                beta_new,
            )

        delta = float(
            np.max(
                np.abs(
                    beta_new
                    -
                    beta
                )
            )
        )

        beta = beta_new

        if (
            delta < TOL
            or
            abs(
                previous_loss
                -
                new_loss
            )
            <
            TOL
        ):
            converged = True
            previous_loss = new_loss
            break

        previous_loss = new_loss

    return {
        "beta": beta,
        "iterations": iteration,
        "converged": converged,
        "loss": previous_loss,
    }


def predict_proba(X, beta):
    X = np.asarray(
        X,
        dtype=float,
    )

    X_aug = np.column_stack(
        [
            np.ones(
                len(X)
            ),
            X,
        ]
    )

    return sigmoid(
        X_aug @ beta
    )


def confusion_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    tp = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )
    )

    sensitivity = (
        tp / (
            tp + fn
        )
        if (
            tp + fn
        ) > 0
        else np.nan
    )

    specificity = (
        tn / (
            tn + fp
        )
        if (
            tn + fp
        ) > 0
        else np.nan
    )

    precision = (
        tp / (
            tp + fp
        )
        if (
            tp + fp
        ) > 0
        else np.nan
    )

    f1 = (
        2.0
        *
        precision
        *
        sensitivity
        /
        (
            precision
            +
            sensitivity
        )
        if (
            pd.notna(precision)
            and
            pd.notna(sensitivity)
            and
            (
                precision
                +
                sensitivity
            )
            > 0
        )
        else np.nan
    )

    balanced_accuracy = (
        0.5
        *
        (
            sensitivity
            +
            specificity
        )
        if (
            pd.notna(sensitivity)
            and
            pd.notna(specificity)
        )
        else np.nan
    )

    accuracy = (
        (
            tp + tn
        )
        /
        len(y_true)
        if len(y_true) > 0
        else np.nan
    )

    fpr = (
        fp / (
            fp + tn
        )
        if (
            fp + tn
        ) > 0
        else np.nan
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "precision": precision,
        "F1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def auroc_score(y_true, scores):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    pos = (
        y_true == 1
    )

    neg = (
        y_true == 0
    )

    n_pos = int(
        pos.sum()
    )

    n_neg = int(
        neg.sum()
    )

    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(
        scores,
        kind="mergesort",
    )

    sorted_scores = scores[
        order
    ]

    ranks = np.empty(
        len(scores),
        dtype=float,
    )

    i = 0

    while i < len(scores):
        j = i + 1

        while (
            j < len(scores)
            and
            sorted_scores[j]
            ==
            sorted_scores[i]
        ):
            j += 1

        avg_rank = (
            i + 1
            +
            j
        ) / 2.0

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    rank_sum_pos = float(
        ranks[
            pos
        ].sum()
    )

    auc = (
        rank_sum_pos
        -
        n_pos
        *
        (
            n_pos + 1
        )
        /
        2.0
    ) / (
        n_pos
        *
        n_neg
    )

    return float(
        auc
    )


def average_precision(y_true, scores):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    n_pos = int(
        y_true.sum()
    )

    if n_pos == 0:
        return np.nan

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    y = y_true[
        order
    ]

    tp = np.cumsum(
        y
    )

    precision = (
        tp
        /
        np.arange(
            1,
            len(y) + 1
        )
    )

    return float(
        precision[
            y == 1
        ].sum()
        /
        n_pos
    )


def select_threshold(y_true, probs):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    candidates = np.unique(
        np.concatenate(
            [
                probs,
                np.array(
                    [
                        0.0,
                        0.5,
                        1.0,
                    ]
                ),
            ]
        )
    )

    rows = []

    for threshold in candidates:
        pred = (
            probs
            >=
            threshold
        ).astype(int)

        metrics = confusion_metrics(
            y_true,
            pred,
        )

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                **metrics,
            }
        )

    table = pd.DataFrame(
        rows
    )

    table["_distance_to_0_5"] = (
        table["threshold"]
        -
        0.5
    ).abs()

    selected = (
        table.sort_values(
            [
                "balanced_accuracy",
                "false_positive_rate",
                "sensitivity",
                "_distance_to_0_5",
                "threshold",
            ],
            ascending=[
                False,
                True,
                False,
                True,
                True,
            ],
            kind="mergesort",
        )
        .iloc[0]
        .copy()
    )

    table = table.drop(
        columns=[
            "_distance_to_0_5",
        ]
    )

    return selected, table


def load_fold_assignments():
    folds = pd.read_csv(
        R6B3_FOLDS
    )

    assignments = {}

    for task, group in folds.groupby(
        "task",
        sort=True,
    ):
        task_map = {}

        for _, row in group.iterrows():
            fold = int(
                row["fold"]
            )

            tracks = json.loads(
                row["val_track_keys_json"]
            )

            for track in tracks:
                if track in task_map:
                    raise RuntimeError(
                        f"Track assigned to multiple folds: {task} {track}"
                    )

                task_map[str(track)] = fold

        assignments[
            str(task)
        ] = task_map

    return assignments


def oof_fit(validation):
    assignments = load_fold_assignments()

    X_all = transform_features(
        validation
    )

    y_all = parse_bool_series(
        validation[LABEL],
        LABEL,
    )

    oof_rows = []
    fold_model_rows = []

    for task in sorted(
        TASK_HORIZON_SEC
    ):
        task_df = validation[
            validation["task"].astype(str)
            ==
            task
        ].copy()

        task_idx = task_df.index

        X_task = X_all.loc[
            task_idx
        ]

        y_task = y_all.loc[
            task_idx
        ]

        task_map = assignments[
            task
        ]

        mapped = (
            task_df["track_key"]
            .astype(str)
            .map(task_map)
        )

        if mapped.isna().any():
            missing_tracks = (
                task_df.loc[
                    mapped.isna(),
                    "track_key",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise RuntimeError(
                f"{task}: tracks missing from locked folds: {missing_tracks}"
            )

        if set(
            mapped.astype(int).unique()
        ) != {
            1,
            2,
            3,
            4,
            5,
        }:
            raise RuntimeError(
                f"{task}: locked fold IDs are incomplete."
            )

        for fold in [
            1,
            2,
            3,
            4,
            5,
        ]:
            val_mask = (
                mapped.astype(int)
                ==
                fold
            )

            train_mask = ~val_mask

            X_train = X_task.loc[
                train_mask
            ].to_numpy(
                dtype=float
            )

            X_val = X_task.loc[
                val_mask
            ].to_numpy(
                dtype=float
            )

            y_train = y_task.loc[
                train_mask
            ].to_numpy(
                dtype=int
            )

            y_val = y_task.loc[
                val_mask
            ].to_numpy(
                dtype=int
            )

            train_tracks = set(
                task_df.loc[
                    train_mask,
                    "track_key",
                ].astype(str)
            )

            val_tracks = set(
                task_df.loc[
                    val_mask,
                    "track_key",
                ].astype(str)
            )

            if (
                train_tracks
                &
                val_tracks
            ):
                raise RuntimeError(
                    f"{task} fold {fold}: track leakage detected."
                )

            if len(
                np.unique(
                    y_train
                )
            ) != 2:
                raise RuntimeError(
                    f"{task} fold {fold}: training target is single-class."
                )

            mean, std = fit_scaler(
                X_train
            )

            X_train_s = apply_scaler(
                X_train,
                mean,
                std,
            )

            X_val_s = apply_scaler(
                X_val,
                mean,
                std,
            )

            fit = fit_logistic_irls(
                X_train_s,
                y_train,
            )

            prob = predict_proba(
                X_val_s,
                fit["beta"],
            )

            heldout = task_df.loc[
                val_mask
            ].copy()

            for pos, (
                idx,
                row,
            ) in enumerate(
                heldout.iterrows()
            ):
                oof_rows.append(
                    {
                        "row_index": int(idx),
                        "task": task,
                        "fold": fold,
                        "source_group":
                            str(row["source_group"]),
                        "track_key":
                            str(row["track_key"]),
                        "sample_index":
                            row["sample_index"]
                            if "sample_index" in row.index
                            else None,
                        "true_crossing":
                            int(y_val[pos]),
                        "oof_probability":
                            float(prob[pos]),
                    }
                )

            beta = fit["beta"]

            fold_model_rows.append(
                {
                    "task": task,
                    "fold": fold,
                    "train_rows":
                        int(
                            train_mask.sum()
                        ),
                    "val_rows":
                        int(
                            val_mask.sum()
                        ),
                    "converged":
                        bool(
                            fit["converged"]
                        ),
                    "iterations":
                        int(
                            fit["iterations"]
                        ),
                    "final_loss":
                        float(
                            fit["loss"]
                        ),
                    "intercept":
                        float(
                            beta[0]
                        ),
                    **{
                        f"coef_{feature}":
                            float(
                                beta[
                                    j + 1
                                ]
                            )
                        for j, feature in enumerate(
                            FEATURES
                        )
                    },
                    **{
                        f"scaler_mean_{feature}":
                            float(
                                mean[j]
                            )
                        for j, feature in enumerate(
                            FEATURES
                        )
                    },
                    **{
                        f"scaler_std_{feature}":
                            float(
                                std[j]
                            )
                        for j, feature in enumerate(
                            FEATURES
                        )
                    },
                }
            )

    oof = pd.DataFrame(
        oof_rows
    )

    models = pd.DataFrame(
        fold_model_rows
    )

    if len(oof) != len(validation):
        raise RuntimeError(
            f"OOF rows {len(oof)} != validation rows {len(validation)}"
        )

    if oof["row_index"].nunique() != len(validation):
        raise RuntimeError(
            "Validation row did not receive exactly one OOF prediction."
        )

    return oof, models


def lock_thresholds(oof):
    threshold_rows = []
    sweep_tables = []

    for task, group in oof.groupby(
        "task",
        sort=True,
    ):
        y = group[
            "true_crossing"
        ].to_numpy(
            dtype=int
        )

        p = group[
            "oof_probability"
        ].to_numpy(
            dtype=float
        )

        selected, sweep = select_threshold(
            y,
            p,
        )

        threshold = float(
            selected[
                "threshold"
            ]
        )

        pred = (
            p
            >=
            threshold
        ).astype(int)

        metrics = confusion_metrics(
            y,
            pred,
        )

        threshold_rows.append(
            {
                "task": task,
                "locked_threshold": threshold,
                **metrics,
                "OOF_AUROC":
                    auroc_score(
                        y,
                        p,
                    ),
                "OOF_AUPRC":
                    average_precision(
                        y,
                        p,
                    ),
                "threshold_selected_on":
                    "validation OOF only",
                "evaluation_used":
                    False,
            }
        )

        sweep["task"] = task
        sweep_tables.append(
            sweep
        )

    return (
        pd.DataFrame(
            threshold_rows
        ),
        pd.concat(
            sweep_tables,
            ignore_index=True,
        ),
    )


def fit_full_validation_models(validation):
    X_all = transform_features(
        validation
    )

    y_all = parse_bool_series(
        validation[LABEL],
        LABEL,
    )

    rows = []
    model_objects = {}

    for task in sorted(
        TASK_HORIZON_SEC
    ):
        mask = (
            validation["task"]
            .astype(str)
            ==
            task
        )

        X = X_all.loc[
            mask
        ].to_numpy(
            dtype=float
        )

        y = y_all.loc[
            mask
        ].to_numpy(
            dtype=int
        )

        mean, std = fit_scaler(
            X
        )

        X_s = apply_scaler(
            X,
            mean,
            std,
        )

        fit = fit_logistic_irls(
            X_s,
            y,
        )

        beta = fit[
            "beta"
        ]

        model_objects[
            task
        ] = {
            "mean":
                mean,
            "std":
                std,
            "beta":
                beta,
        }

        rows.append(
            {
                "task":
                    task,
                "train_rows":
                    int(len(X)),
                "positives":
                    int(y.sum()),
                "negatives":
                    int(
                        len(y)
                        -
                        y.sum()
                    ),
                "converged":
                    bool(
                        fit["converged"]
                    ),
                "iterations":
                    int(
                        fit["iterations"]
                    ),
                "final_loss":
                    float(
                        fit["loss"]
                    ),
                "intercept":
                    float(
                        beta[0]
                    ),
                **{
                    f"coef_{feature}":
                        float(
                            beta[
                                j + 1
                            ]
                        )
                    for j, feature in enumerate(
                        FEATURES
                    )
                },
                **{
                    f"scaler_mean_{feature}":
                        float(
                            mean[j]
                        )
                    for j, feature in enumerate(
                        FEATURES
                    )
                },
                **{
                    f"scaler_std_{feature}":
                        float(
                            std[j]
                        )
                    for j, feature in enumerate(
                        FEATURES
                    )
                },
            }
        )

    return (
        model_objects,
        pd.DataFrame(
            rows
        ),
    )


def evaluate_once(
    evaluation,
    models,
    thresholds,
):
    X_all = transform_features(
        evaluation
    )

    y_all = parse_bool_series(
        evaluation[LABEL],
        LABEL,
    )

    threshold_map = {
        str(row["task"]):
            float(
                row["locked_threshold"]
            )
        for _, row in thresholds.iterrows()
    }

    prediction_rows = []

    for task in sorted(
        TASK_HORIZON_SEC
    ):
        mask = (
            evaluation["task"]
            .astype(str)
            ==
            task
        )

        sub = evaluation.loc[
            mask
        ].copy()

        X = X_all.loc[
            mask
        ].to_numpy(
            dtype=float
        )

        y = y_all.loc[
            mask
        ].to_numpy(
            dtype=int
        )

        obj = models[
            task
        ]

        X_s = apply_scaler(
            X,
            obj["mean"],
            obj["std"],
        )

        prob = predict_proba(
            X_s,
            obj["beta"],
        )

        threshold = threshold_map[
            task
        ]

        pred = (
            prob
            >=
            threshold
        ).astype(int)

        baseline_pred = parse_bool_series(
            sub["predicted_crossing"],
            "evaluation.predicted_crossing",
        ).to_numpy(
            dtype=int
        )

        for pos, (
            idx,
            row,
        ) in enumerate(
            sub.iterrows()
        ):
            prediction_rows.append(
                {
                    "row_index":
                        int(idx),
                    "dataset":
                        str(row["dataset"])
                        if "dataset" in row.index
                        else "UNSPECIFIED",
                    "task":
                        task,
                    "source_group":
                        str(row["source_group"]),
                    "track_key":
                        str(row["track_key"]),
                    "sample_index":
                        row["sample_index"]
                        if "sample_index" in row.index
                        else None,
                    "boundary_id":
                        row["boundary_id"]
                        if "boundary_id" in row.index
                        else None,
                    "true_crossing":
                        int(y[pos]),
                    "direct_classifier_probability":
                        float(prob[pos]),
                    "locked_threshold":
                        threshold,
                    "direct_classifier_prediction":
                        int(pred[pos]),
                    "nominal_predicted_crossing_baseline":
                        int(
                            baseline_pred[pos]
                        ),
                }
            )

    pred_df = pd.DataFrame(
        prediction_rows
    )

    metric_rows = []

    grouping = [
        "task"
    ]

    if "dataset" in pred_df.columns:
        grouping = [
            "dataset",
            "task",
        ]

    for keys, group in pred_df.groupby(
        grouping,
        sort=True,
    ):
        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        key_dict = dict(
            zip(
                grouping,
                keys,
            )
        )

        y = group[
            "true_crossing"
        ].to_numpy(
            dtype=int
        )

        p = group[
            "direct_classifier_probability"
        ].to_numpy(
            dtype=float
        )

        direct_pred = group[
            "direct_classifier_prediction"
        ].to_numpy(
            dtype=int
        )

        baseline_pred = group[
            "nominal_predicted_crossing_baseline"
        ].to_numpy(
            dtype=int
        )

        direct = confusion_metrics(
            y,
            direct_pred,
        )

        base = confusion_metrics(
            y,
            baseline_pred,
        )

        metric_rows.append(
            {
                **key_dict,
                "method":
                    "DIRECT_LOGISTIC_CLASSIFIER",
                "rows":
                    int(len(group)),
                "prevalence":
                    float(
                        y.mean()
                    ),
                **direct,
                "AUROC":
                    auroc_score(
                        y,
                        p,
                    ),
                "AUPRC":
                    average_precision(
                        y,
                        p,
                    ),
            }
        )

        metric_rows.append(
            {
                **key_dict,
                "method":
                    "NOMINAL_PREDICTED_CROSSING",
                "rows":
                    int(len(group)),
                "prevalence":
                    float(
                        y.mean()
                    ),
                **base,
                "AUROC":
                    np.nan,
                "AUPRC":
                    np.nan,
            }
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    comparison_rows = []

    for keys, group in metrics.groupby(
        grouping,
        sort=True,
    ):
        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        key_dict = dict(
            zip(
                grouping,
                keys,
            )
        )

        direct = group[
            group["method"]
            ==
            "DIRECT_LOGISTIC_CLASSIFIER"
        ]

        base = group[
            group["method"]
            ==
            "NOMINAL_PREDICTED_CROSSING"
        ]

        if (
            len(direct) != 1
            or
            len(base) != 1
        ):
            raise RuntimeError(
                f"Evaluation comparison group malformed: {key_dict}"
            )

        d = direct.iloc[0]
        b = base.iloc[0]

        comparison_rows.append(
            {
                **key_dict,
                "rows":
                    int(
                        d["rows"]
                    ),
                "direct_balanced_accuracy":
                    float(
                        d["balanced_accuracy"]
                    ),
                "baseline_balanced_accuracy":
                    float(
                        b["balanced_accuracy"]
                    ),
                "balanced_accuracy_delta":
                    float(
                        d["balanced_accuracy"]
                        -
                        b["balanced_accuracy"]
                    ),
                "direct_sensitivity":
                    float(
                        d["sensitivity"]
                    ),
                "baseline_sensitivity":
                    float(
                        b["sensitivity"]
                    ),
                "sensitivity_delta":
                    float(
                        d["sensitivity"]
                        -
                        b["sensitivity"]
                    ),
                "direct_false_positive_rate":
                    float(
                        d["false_positive_rate"]
                    ),
                "baseline_false_positive_rate":
                    float(
                        b["false_positive_rate"]
                    ),
                "false_positive_rate_delta":
                    float(
                        d["false_positive_rate"]
                        -
                        b["false_positive_rate"]
                    ),
                "direct_precision":
                    float(
                        d["precision"]
                    )
                    if pd.notna(
                        d["precision"]
                    )
                    else np.nan,
                "baseline_precision":
                    float(
                        b["precision"]
                    )
                    if pd.notna(
                        b["precision"]
                    )
                    else np.nan,
                "direct_F1":
                    float(
                        d["F1"]
                    )
                    if pd.notna(
                        d["F1"]
                    )
                    else np.nan,
                "baseline_F1":
                    float(
                        b["F1"]
                    )
                    if pd.notna(
                        b["F1"]
                    )
                    else np.nan,
                "direct_AUROC":
                    float(
                        d["AUROC"]
                    )
                    if pd.notna(
                        d["AUROC"]
                    )
                    else np.nan,
                "direct_AUPRC":
                    float(
                        d["AUPRC"]
                    )
                    if pd.notna(
                        d["AUPRC"]
                    )
                    else np.nan,
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    return (
        pred_df,
        metrics,
        comparison,
    )


def main():
    print(
        "\nR6-C — OOF DIRECT CROSSING CLASSIFIER + VALIDATION-ONLY THRESHOLD LOCK + FROZEN EVALUATION"
    )

    print(
        "This is the first classifier fit. Thresholds are selected only from "
        "track-disjoint validation OOF predictions, then frozen before the "
        "evaluation table is scored exactly once. No evaluation retuning.\n"
    )

    manifest, protocol = verify_protocol()

    validation = pd.read_csv(
        VALIDATION
    )

    evaluation = pd.read_csv(
        EVALUATION
    )

    oof, fold_models = oof_fit(
        validation
    )

    thresholds, sweep = lock_thresholds(
        oof
    )

    full_models, full_model_table = fit_full_validation_models(
        validation
    )

    eval_predictions, eval_metrics, comparison = evaluate_once(
        evaluation,
        full_models,
        thresholds,
    )

    oof_path = OUT / "R6C_validation_OOF_predictions.csv"
    fold_model_path = OUT / "R6C_fold_model_audit.csv"
    threshold_path = OUT / "R6C_LOCKED_VALIDATION_ONLY_THRESHOLDS.csv"
    sweep_path = OUT / "R6C_validation_threshold_sweep.csv"
    full_model_path = OUT / "R6C_full_validation_model_parameters.csv"
    eval_pred_path = OUT / "R6C_FROZEN_EVALUATION_PREDICTIONS.csv"
    eval_metrics_path = OUT / "R6C_FROZEN_EVALUATION_METRICS.csv"
    comparison_path = OUT / "R6C_DIRECT_CLASSIFIER_VS_NOMINAL_BASELINE.csv"

    oof.to_csv(
        oof_path,
        index=False,
    )

    fold_models.to_csv(
        fold_model_path,
        index=False,
    )

    thresholds.to_csv(
        threshold_path,
        index=False,
    )

    sweep.to_csv(
        sweep_path,
        index=False,
    )

    full_model_table.to_csv(
        full_model_path,
        index=False,
    )

    eval_predictions.to_csv(
        eval_pred_path,
        index=False,
    )

    eval_metrics.to_csv(
        eval_metrics_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    status = (
        "R6C_VALIDATION_ONLY_THRESHOLD_LOCK_AND_FROZEN_DIRECT_CLASSIFIER_EVALUATION_COMPLETE_READY_FOR_R6D_INTERPRETATION"
    )

    result_manifest = {
        "step":
            "R6C_DIRECT_CROSSING_CLASSIFIER_OOF_THRESHOLD_LOCK_FROZEN_EVALUATION",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "status":
            status,

        "threshold_selection_source":
            "validation OOF only",

        "evaluation_used_for_threshold_selection":
            False,

        "evaluation_threshold_reselection":
            False,

        "validation_source":
            str(VALIDATION),

        "validation_source_sha256":
            sha256_file(VALIDATION),

        "evaluation_source":
            str(EVALUATION),

        "evaluation_source_sha256":
            sha256_file(EVALUATION),

        "locked_protocol":
            str(R6B3_PROTOCOL),

        "locked_protocol_sha256":
            sha256_file(R6B3_PROTOCOL),

        "locked_threshold_csv":
            str(threshold_path),

        "locked_threshold_sha256":
            sha256_file(threshold_path),

        "evaluation_metrics_csv":
            str(eval_metrics_path),

        "evaluation_metrics_sha256":
            sha256_file(eval_metrics_path),

        "comparison_csv":
            str(comparison_path),

        "comparison_sha256":
            sha256_file(comparison_path),
    }

    manifest_path = OUT / "R6C_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            result_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("VALIDATION OOF THRESHOLD LOCK:")
    print(
        thresholds.to_string(
            index=False
        )
    )

    print("\nFOLD MODEL AUDIT:")
    print(
        fold_models[
            [
                "task",
                "fold",
                "train_rows",
                "val_rows",
                "converged",
                "iterations",
                "final_loss",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nFULL VALIDATION MODEL AUDIT:")
    print(
        full_model_table[
            [
                "task",
                "train_rows",
                "positives",
                "negatives",
                "converged",
                "iterations",
                "final_loss",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nFROZEN EVALUATION METRICS:")
    print(
        eval_metrics.to_string(
            index=False
        )
    )

    print("\nDIRECT CLASSIFIER vs NOMINAL CROSSING BASELINE:")
    print(
        comparison.to_string(
            index=False
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR6-C COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
