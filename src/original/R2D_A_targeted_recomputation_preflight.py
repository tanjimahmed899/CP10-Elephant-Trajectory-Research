from pathlib import Path
import ast
import hashlib
import json
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

R2C_SUMMARY = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2C_missing_row_audit"
    / "R2C_SUMMARY.json"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_targeted_recompute_preflight"
)

OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    ("BIRDSAI_TestReal", "2s_to_2s"): {
        "samples": 1053,
        "future_len": 20,
    },
    ("BIRDSAI_TestReal", "2s_to_5s"): {
        "samples": 826,
        "future_len": 50,
    },
    ("Fresh_WildBox", "2s_to_2s"): {
        "samples": 865,
        "future_len": 20,
    },
    ("Fresh_WildBox", "2s_to_5s"): {
        "samples": 637,
        "future_len": 50,
    },
}

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


def verify_r2c():
    if not R2C_SUMMARY.exists():
        raise FileNotFoundError(R2C_SUMMARY)

    obj = json.loads(
        R2C_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    if (
        obj.get("status")
        !=
        "R2C_AUDIT_COMPLETE_SOME_TARGETED_RECOMPUTATION_CONFIRMED_REQUIRED"
    ):
        raise RuntimeError(
            f"Unexpected R2C status: {obj.get('status')}"
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


def inspect_npz(path):
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    arrays = {}
    for key in z.files:
        arr = z[key]
        arrays[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    return arrays, None


def score_npz_candidate(path, dataset, task, expected_samples, future_len):
    arrays, error = inspect_npz(path)

    if arrays is None:
        return {
            "path": str(path),
            "dataset_hint": dataset_hint(path),
            "task_hint": task_hint(path),
            "score": -999,
            "error": error,
            "arrays_json": None,
        }

    score = 0

    if dataset_hint(path) == dataset:
        score += 20

    if task_hint(path) == task:
        score += 20

    shapes = [tuple(v["shape"]) for v in arrays.values()]

    # Strong evidence for exact sample count.
    if any(
        len(shape) >= 1 and shape[0] == expected_samples
        for shape in shapes
    ):
        score += 40

    # History/future typical shapes.
    if any(
        len(shape) == 3
        and shape[0] == expected_samples
        and shape[1] == 20
        and shape[2] == 2
        for shape in shapes
    ):
        score += 20

    if any(
        len(shape) == 3
        and shape[0] == expected_samples
        and shape[1] == future_len
        and shape[2] == 2
        for shape in shapes
    ):
        score += 30

    # Prefer actual dataset/window files over result-only prediction files.
    low = str(path).lower()

    if any(
        token in low
        for token in [
            "dataset",
            "window",
            "trajectory",
            "neural",
            "canonical",
        ]
    ):
        score += 10

    if any(
        token in low
        for token in [
            "prediction",
            "result",
            "ensemble",
        ]
    ):
        score -= 5

    return {
        "path": str(path),
        "dataset_hint": dataset_hint(path),
        "task_hint": task_hint(path),
        "score": score,
        "error": None,
        "arrays_json": json.dumps(arrays),
    }


def discover_npz_candidates():
    all_npz = []
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*.npz"):
            if not p.is_file():
                continue

            rp = str(p.resolve()).lower()
            if rp in seen:
                continue

            seen.add(rp)
            all_npz.append(p)

    rows = []

    for (dataset, task), cfg in EXPECTED.items():
        for p in all_npz:
            result = score_npz_candidate(
                p,
                dataset,
                task,
                cfg["samples"],
                cfg["future_len"],
            )

            if result["score"] <= 0:
                continue

            rows.append({
                "target_dataset": dataset,
                "target_task": task,
                "expected_samples": cfg["samples"],
                "expected_future_len": cfg["future_len"],
                "sha256": sha256_file(p),
                **result,
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No plausible NPZ candidates found for the four frozen evaluation sets."
        )

    return df.sort_values(
        [
            "target_dataset",
            "target_task",
            "score",
            "path",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)


def select_top_npz(npz_df):
    rows = []

    for (dataset, task), group in npz_df.groupby(
        [
            "target_dataset",
            "target_task",
        ],
        sort=False,
    ):
        top_score = group["score"].max()
        top = group[
            group["score"] == top_score
        ].copy()

        rows.append({
            "dataset": dataset,
            "task": task,
            "top_score": int(top_score),
            "top_candidate_count": int(len(top)),
            "selected_path": str(top.iloc[0]["path"]),
            "selected_sha256": str(top.iloc[0]["sha256"]),
            "selected_arrays_json": str(top.iloc[0]["arrays_json"]),
            "ambiguous_top_tie": bool(len(top) > 1),
        })

    return pd.DataFrame(rows)


def discover_metadata_candidates(selected_npz_df):
    rows = []

    for _, sel in selected_npz_df.iterrows():
        dataset = sel["dataset"]
        task = sel["task"]
        expected = EXPECTED[(dataset, task)]["samples"]

        selected_path = Path(sel["selected_path"])

        candidate_dirs = [
            selected_path.parent,
            selected_path.parent.parent,
        ]

        seen = set()

        for folder in candidate_dirs:
            if not folder.exists():
                continue

            for p in folder.rglob("*.csv"):
                if not p.is_file():
                    continue

                rp = str(p.resolve()).lower()
                if rp in seen:
                    continue

                seen.add(rp)

                try:
                    df = pd.read_csv(p)
                except Exception:
                    continue

                score = 0

                if len(df) == expected:
                    score += 50

                cols = {
                    str(c).lower()
                    for c in df.columns
                }

                if any(
                    c in cols
                    for c in [
                        "track_id",
                        "tracklet_id",
                        "track",
                    ]
                ):
                    score += 20

                if any(
                    c in cols
                    for c in [
                        "source",
                        "source_id",
                        "video",
                        "video_id",
                        "sequence",
                        "sequence_id",
                    ]
                ):
                    score += 20

                if any(
                    c in cols
                    for c in [
                        "sample_index",
                        "sample_id",
                        "window_index",
                        "window_id",
                    ]
                ):
                    score += 10

                if dataset_hint(p) == dataset:
                    score += 10

                if task_hint(p) == task:
                    score += 10

                if score <= 0:
                    continue

                rows.append({
                    "dataset": dataset,
                    "task": task,
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "rows": int(len(df)),
                    "columns": " | ".join(map(str, df.columns)),
                    "score": int(score),
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "task",
                "path",
                "sha256",
                "rows",
                "columns",
                "score",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        [
            "dataset",
            "task",
            "score",
            "path",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)


def find_script(name):
    hits = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        hits.extend(
            [
                p
                for p in root.rglob(name)
                if p.is_file()
            ]
        )

    if not hits:
        return None

    hits = sorted(
        hits,
        key=lambda p: (
            0
            if str(MAIN_RESULT_2).lower()
            in
            str(p).lower()
            else
            1,
            str(p).lower(),
        )
    )

    return hits[0]


def inspect_script(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(text)

    functions = []
    classes = []
    assignments = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            try:
                signature = ast.unparse(node.args)
            except Exception:
                signature = "<unparse_failed>"

            functions.append({
                "name": node.name,
                "line": int(node.lineno),
                "args": signature,
            })

        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "line": int(node.lineno),
            })

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                src = ast.get_source_segment(text, node)
            except Exception:
                src = None

            if src and any(
                token in src.lower()
                for token in [
                    "q=",
                    "r=",
                    "kalman",
                    "future",
                    "history",
                    "seed",
                    "model_root",
                    "device",
                ]
            ):
                assignments.append({
                    "line": int(getattr(node, "lineno", -1)),
                    "source": src,
                })

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "functions": functions,
        "classes": classes,
        "assignments": assignments,
    }


def inspect_target_scripts():
    results = []

    for name in TARGET_SCRIPTS:
        p = find_script(name)

        if p is None:
            results.append({
                "target": name,
                "found": False,
            })
            continue

        info = inspect_script(p)
        info["target"] = name
        info["found"] = True
        results.append(info)

    return results


def identify_loader_candidates(script_info):
    rows = []

    keywords = [
        "load",
        "model",
        "transformer",
        "lstm",
        "gru",
        "kalman",
        "predict",
        "evaluate",
        "canonical",
        "inverse",
    ]

    for item in script_info:
        if not item.get("found"):
            continue

        for fn in item.get("functions", []):
            name = fn["name"]

            if any(
                token in name.lower()
                for token in keywords
            ):
                rows.append({
                    "script": item["path"],
                    "script_sha256": item["sha256"],
                    "function": name,
                    "line": fn["line"],
                    "args": fn["args"],
                })

    return pd.DataFrame(rows)


def readiness(selected_npz, metadata_df, loader_df):
    checks = []

    for _, row in selected_npz.iterrows():
        checks.append({
            "check":
                f"npz::{row['dataset']}::{row['task']}",

            "pass":
                bool(
                    not row["ambiguous_top_tie"]
                ),

            "note":
                str(row["selected_path"]),
        })

        meta = metadata_df[
            (metadata_df["dataset"] == row["dataset"])
            &
            (metadata_df["task"] == row["task"])
        ]

        if len(meta) > 0:
            best = meta.iloc[0]
            meta_pass = (
                int(best["rows"])
                ==
                EXPECTED[
                    (
                        row["dataset"],
                        row["task"],
                    )
                ][
                    "samples"
                ]
            )
        else:
            best = None
            meta_pass = False

        checks.append({
            "check":
                f"metadata::{row['dataset']}::{row['task']}",

            "pass":
                bool(meta_pass),

            "note":
                (
                    str(best["path"])
                    if best is not None
                    else
                    "no exact metadata candidate"
                ),
        })

    required_loader_tokens = [
        "load_transformer",
    ]

    for token in required_loader_tokens:
        subset = loader_df[
            loader_df["function"].astype(str).str.contains(
                token,
                case=False,
                regex=False,
            )
        ]

        checks.append({
            "check":
                f"loader::{token}",

            "pass":
                bool(len(subset) > 0),

            "note":
                (
                    str(subset.iloc[0]["script"])
                    if len(subset) > 0
                    else
                    "not found"
                ),
        })

    # GRU/LSTM loader may be named generically, so require at least one function
    # in train_lstm_gru.py containing load/model/evaluate/predict.
    lstm_gru_subset = loader_df[
        loader_df["script"].astype(str).str.endswith(
            "train_lstm_gru.py"
        )
    ]

    checks.append({
        "check":
            "loader::lstm_gru_relevant_function",

        "pass":
            bool(len(lstm_gru_subset) > 0),

        "note":
            (
                " | ".join(
                    lstm_gru_subset["function"]
                    .astype(str)
                    .head(10)
                    .tolist()
                )
                if len(lstm_gru_subset) > 0
                else
                "not found"
            ),
    })

    return pd.DataFrame(checks)


def main():
    print(
        "\nR2D-A — TARGETED RECOMPUTATION PREFLIGHT"
    )

    print(
        "NO new ADE/FDE is generated. "
        "This step locks exact evaluation NPZs, metadata, and model/helper functions "
        "before targeted sample-level recomputation.\n"
    )

    r2c = verify_r2c()

    npz_df = discover_npz_candidates()
    selected_npz = select_top_npz(npz_df)

    metadata_df = discover_metadata_candidates(
        selected_npz
    )

    script_info = inspect_target_scripts()

    loader_df = identify_loader_candidates(
        script_info
    )

    readiness_df = readiness(
        selected_npz,
        metadata_df,
        loader_df,
    )

    npz_path = OUT / "R2D_A_npz_candidate_scores.csv"
    selected_path = OUT / "R2D_A_selected_npz.csv"
    metadata_path = OUT / "R2D_A_metadata_candidates.csv"
    script_path = OUT / "R2D_A_script_function_inventory.json"
    loader_path = OUT / "R2D_A_loader_function_candidates.csv"
    readiness_path = OUT / "R2D_A_readiness_checks.csv"

    npz_df.to_csv(npz_path, index=False)
    selected_npz.to_csv(selected_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    script_path.write_text(
        json.dumps(
            script_info,
            indent=2,
        ),
        encoding="utf-8",
    )

    loader_df.to_csv(loader_path, index=False)
    readiness_df.to_csv(readiness_path, index=False)

    all_pass = bool(
        readiness_df["pass"].all()
    )

    status = (
        "R2D_A_PREFLIGHT_COMPLETE_READY_FOR_TARGETED_RECOMPUTATION"
        if all_pass
        else
        "R2D_A_PREFLIGHT_COMPLETE_EXACT_ADAPTER_PATCH_REQUIRED"
    )

    summary = {
        "step": "R2D_A_TARGETED_RECOMPUTATION_PREFLIGHT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "r2c_status": r2c.get("status"),
        "new_metrics_generated": False,
        "outputs": {
            "npz_candidates": str(npz_path),
            "selected_npz": str(selected_path),
            "metadata_candidates": str(metadata_path),
            "script_inventory": str(script_path),
            "loader_candidates": str(loader_path),
            "readiness": str(readiness_path),
        },
    }

    summary_path = OUT / "R2D_A_SUMMARY.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("SELECTED EVALUATION NPZs:")
    print(
        selected_npz[
            [
                "dataset",
                "task",
                "top_score",
                "top_candidate_count",
                "selected_path",
                "ambiguous_top_tie",
            ]
        ].to_string(index=False)
    )

    print("\nTOP METADATA CANDIDATES:")
    if len(metadata_df) > 0:
        top_meta = (
            metadata_df
            .sort_values(
                [
                    "dataset",
                    "task",
                    "score",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
            .groupby(
                [
                    "dataset",
                    "task",
                ],
                as_index=False,
            )
            .first()
        )

        print(
            top_meta[
                [
                    "dataset",
                    "task",
                    "rows",
                    "score",
                    "path",
                ]
            ].to_string(index=False)
        )
    else:
        print("No metadata candidates found.")

    print("\nLOADER/FUNCTION CANDIDATES:")
    print(
        loader_df.to_string(index=False)
        if len(loader_df) > 0
        else
        "No loader candidates found."
    )

    print("\nREADINESS CHECKS:")
    print(
        readiness_df.to_string(index=False)
    )

    print("\nSTATUS:")
    print(status)

    print("\nR2D-A COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
