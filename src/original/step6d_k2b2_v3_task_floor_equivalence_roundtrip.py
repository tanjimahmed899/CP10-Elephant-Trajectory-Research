from pathlib import Path
import ast
import inspect
import json
import math
import re
import numpy as np
import pandas as pd

READ_ONLY_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

BUILD_SRC = READ_ONLY_ROOT / "build_final_neural_dataset.py"
FRESH_SRC = READ_ONLY_ROOT / "evaluate_fresh_wildbox_holdout.py"

BAMBI_DIR = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

K2A = BAMBI_DIR / "k2a_10hz"

NPZ_BY_TASK = {
    "2s_to_2s": K2A / "2s_to_2s_bambi_flight6_world_raw.npz",
    "2s_to_5s": K2A / "2s_to_5s_bambi_flight6_world_raw.npz",
}

# Exact values already present in the frozen Fresh-holdout evaluator config.
FRESH_CONFIG_SCALE_FLOORS = {
    "2s_to_2s": 0.06820264416877736,
    "2s_to_5s": 0.06732638773715147,
}

OUT = BAMBI_DIR / "k2b_representation_audit" / "equivalence_test_v3"
OUT.mkdir(parents=True, exist_ok=True)

ATOL_CANON = 1e-7
ATOL_ROUNDTRIP_FRESH_M = 1e-6

TARGETS = [
    (BUILD_SRC, "canonicalize_sample", "build_final"),
    (FRESH_SRC, "canonicalize_window", "fresh_holdout"),
]


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def top_level_maps(tree):
    funcs = {}
    imports = []
    assigns = {}

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigns[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigns[node.target.id] = node

    return funcs, imports, assigns


def called_names(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            out.add(n.func.id)
    return out


def collect_function_closure(target_name, funcs):
    needed = set()
    stack = [target_name]

    while stack:
        name = stack.pop()
        if name in needed or name not in funcs:
            continue

        needed.add(name)

        for c in called_names(funcs[name]):
            if c in funcs and c not in needed:
                stack.append(c)

    return needed


def loaded_global_names(node):
    local = set()

    for a in node.args.args + node.args.kwonlyargs:
        local.add(a.arg)

    if node.args.vararg:
        local.add(node.args.vararg.arg)

    if node.args.kwarg:
        local.add(node.args.kwarg.arg)

    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            local.add(n.id)

    out = set()

    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in local:
                out.add(n.id)

    return out


def build_isolated_namespace(path, target_name):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text)
    funcs, imports, assigns = top_level_maps(tree)

    if target_name not in funcs:
        raise RuntimeError(f"{target_name} not found in {path}")

    needed_funcs = collect_function_closure(target_name, funcs)

    needed_globals = set()
    for fname in needed_funcs:
        needed_globals |= loaded_global_names(funcs[fname])

    body = []
    body.extend(imports)

    for node in tree.body:
        include = False

        if isinstance(node, ast.Assign):
            names = [
                t.id
                for t in node.targets
                if isinstance(t, ast.Name)
            ]
            include = any(n in needed_globals for n in names)

        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            include = node.target.id in needed_globals

        if include:
            try:
                ast.literal_eval(node.value)
                body.append(node)
            except Exception:
                pass

    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in needed_funcs
        ):
            body.append(node)

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)

    ns = {}
    exec(
        compile(module, filename=str(path), mode="exec"),
        ns,
        ns
    )

    return {
        "function": ns[target_name],
        "signature": inspect.signature(ns[target_name]),
        "source_text": text,
        "tree": tree,
    }


def call_canonicalizer(fn, history, future, scale_floor):
    sig = inspect.signature(fn)
    names = list(sig.parameters.keys())

    try:
        return fn(history, future, scale_floor)
    except TypeError:
        return fn(
            **{
                names[0]: history,
                names[1]: future,
                "scale_floor": scale_floor,
            }
        )


def extract_common(result):
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected return type: {type(result)}")

    hist = result["history_canonical"]
    fut = result["future_canonical"]
    origin = result["origin"]
    scale = result["scale"]

    angle = None
    for key in ["angle", "heading_angle", "theta", "heading"]:
        if key in result:
            angle = result[key]
            break

    path = None
    for key in ["raw_path_length", "history_path", "path_length"]:
        if key in result:
            path = result[key]
            break

    if angle is None:
        raise RuntimeError(f"Angle key not found. Keys={list(result.keys())}")

    return {
        "history": np.asarray(hist, dtype=np.float64),
        "future": np.asarray(fut, dtype=np.float64),
        "origin": np.asarray(origin, dtype=np.float64),
        "angle": float(angle),
        "scale": float(scale),
        "path_length": float(path) if path is not None else np.nan,
        "origin_dtype": str(np.asarray(origin).dtype),
        "angle_type": type(angle).__name__,
        "scale_type": type(scale).__name__,
    }


def inverse_transform(canonical, origin, angle, scale):
    """
    Matches the frozen inverse-transform convention:
      local = (canonical * scale) @ R
      absolute = local + origin
    with canonicalization using (... @ R.T / scale).
    """
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([
        [c, -s],
        [s,  c],
    ], dtype=np.float64)

    local = (
        np.asarray(canonical, dtype=np.float64)
        * float(scale)
    ) @ R

    return local + np.asarray(origin, dtype=np.float64)


def sample_indices(n, max_n=250):
    if n <= max_n:
        return list(range(n))

    return sorted(
        set(
            np.linspace(
                0,
                n - 1,
                max_n,
                dtype=int
            ).tolist()
        )
    )


def source_occurrence_scan(values):
    rows = []

    files = list(READ_ONLY_ROOT.rglob("*"))

    text_ext = {
        ".py", ".json", ".txt", ".csv",
        ".md", ".yaml", ".yml"
    }

    needles = {
        task: [
            repr(value),
            f"{value:.17g}",
            f"{value:.15f}",
        ]
        for task, value in values.items()
    }

    for path in files:
        if not path.is_file():
            continue

        if path.suffix.lower() not in text_ext:
            continue

        try:
            if path.stat().st_size > 5_000_000:
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue

        lines = text.splitlines()

        for task, ns in needles.items():
            found = False

            for i, line in enumerate(lines, start=1):
                if any(n in line for n in ns):
                    rows.append({
                        "task": task,
                        "value": values[task],
                        "file": str(path),
                        "line": int(i),
                        "text": line.strip(),
                    })
                    found = True

            # Avoid duplicate broad handling; exact textual matches only.
            _ = found

    return pd.DataFrame(rows)


def metadata_scale_floor_scan():
    rows = []

    # CSVs
    for path in READ_ONLY_ROOT.rglob("*.csv"):
        try:
            if path.stat().st_size > 20_000_000:
                continue

            df = pd.read_csv(path, nrows=5000)

            for col in df.columns:
                if "scale_floor" not in col.lower():
                    continue

                vals = (
                    pd.to_numeric(
                        df[col],
                        errors="coerce"
                    )
                    .dropna()
                    .unique()
                )

                for v in vals[:20]:
                    rows.append({
                        "file": str(path),
                        "container": "csv",
                        "field": col,
                        "value": float(v),
                    })

        except Exception:
            pass

    # JSONs
    for path in READ_ONLY_ROOT.rglob("*.json"):
        try:
            if path.stat().st_size > 20_000_000:
                continue

            obj = json.loads(
                path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
            )
        except Exception:
            continue

        stack = [("", obj)]

        while stack:
            prefix, cur = stack.pop()

            if isinstance(cur, dict):
                for k, v in cur.items():
                    kp = f"{prefix}.{k}" if prefix else str(k)

                    if "scale_floor" in str(k).lower():
                        if isinstance(v, (int, float)):
                            rows.append({
                                "file": str(path),
                                "container": "json",
                                "field": kp,
                                "value": float(v),
                            })

                    if isinstance(v, (dict, list)):
                        stack.append((kp, v))

            elif isinstance(cur, list):
                for i, v in enumerate(cur[:5000]):
                    if isinstance(v, (dict, list)):
                        stack.append(
                            (
                                f"{prefix}[{i}]",
                                v
                            )
                        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nSTEP 6D-K2B2-V3 — TASK-SPECIFIC FLOOR CONFIRMATION + "
        "CANONICAL EQUIVALENCE + ROUND-TRIP AUDIT"
    )
    print(
        "No final BAMBI canonical dataset is written. "
        "No forecasting or conformal evaluation.\n"
    )

    for p in [
        BUILD_SRC,
        FRESH_SRC,
        NPZ_BY_TASK["2s_to_2s"],
        NPZ_BY_TASK["2s_to_5s"],
    ]:
        require(p)

    loaded = {}

    for path, fname, label in TARGETS:
        loaded[label] = build_isolated_namespace(
            path,
            fname
        )

        print(
            f"{label}: {path.name}::{fname} "
            f"{loaded[label]['signature']}"
        )

    rows = []

    for task, npz_path in NPZ_BY_TASK.items():
        floor = FRESH_CONFIG_SCALE_FLOORS[task]

        with np.load(npz_path, allow_pickle=False) as z:
            X = z["X_abs_xy"]
            Y = z["Y_abs_xy"]

        for i in sample_indices(len(X)):
            h = np.asarray(X[i], dtype=np.float64)
            f = np.asarray(Y[i], dtype=np.float64)

            a = extract_common(
                call_canonicalizer(
                    loaded["build_final"]["function"],
                    h,
                    f,
                    floor
                )
            )

            b = extract_common(
                call_canonicalizer(
                    loaded["fresh_holdout"]["function"],
                    h,
                    f,
                    floor
                )
            )

            # Canonical tensors are the actual frozen-model inputs/targets.
            hist_diff = float(
                np.max(
                    np.abs(
                        a["history"] - b["history"]
                    )
                )
            )

            fut_diff = float(
                np.max(
                    np.abs(
                        a["future"] - b["future"]
                    )
                )
            )

            # Metadata differences are expected because build_final explicitly
            # casts metadata to float32 while fresh evaluator retains float64.
            origin_diff = float(
                np.max(
                    np.abs(
                        a["origin"] - b["origin"]
                    )
                )
            )

            angle_diff = abs(
                (
                    a["angle"] - b["angle"] + math.pi
                )
                %
                (2.0 * math.pi)
                -
                math.pi
            )

            scale_diff = abs(
                a["scale"] - b["scale"]
            )

            # Round-trip each implementation using its own metadata.
            rec_a_hist = inverse_transform(
                a["history"],
                a["origin"],
                a["angle"],
                a["scale"]
            )

            rec_a_fut = inverse_transform(
                a["future"],
                a["origin"],
                a["angle"],
                a["scale"]
            )

            rec_b_hist = inverse_transform(
                b["history"],
                b["origin"],
                b["angle"],
                b["scale"]
            )

            rec_b_fut = inverse_transform(
                b["future"],
                b["origin"],
                b["angle"],
                b["scale"]
            )

            rows.append({
                "task": task,
                "sample_index": int(i),
                "scale_floor": float(floor),

                "history_canonical_max_abs_diff": hist_diff,
                "future_canonical_max_abs_diff": fut_diff,

                "origin_metadata_max_abs_diff_m": origin_diff,
                "angle_metadata_abs_diff_rad": float(angle_diff),
                "scale_metadata_abs_diff": float(scale_diff),

                "build_history_roundtrip_max_abs_error_m": float(
                    np.max(np.abs(rec_a_hist - h))
                ),
                "build_future_roundtrip_max_abs_error_m": float(
                    np.max(np.abs(rec_a_fut - f))
                ),

                "fresh_history_roundtrip_max_abs_error_m": float(
                    np.max(np.abs(rec_b_hist - h))
                ),
                "fresh_future_roundtrip_max_abs_error_m": float(
                    np.max(np.abs(rec_b_fut - f))
                ),

                "build_origin_dtype": a["origin_dtype"],
                "fresh_origin_dtype": b["origin_dtype"],
                "build_angle_type": a["angle_type"],
                "fresh_angle_type": b["angle_type"],
                "build_scale_type": a["scale_type"],
                "fresh_scale_type": b["scale_type"],
            })

    eq = pd.DataFrame(rows)

    eq.to_csv(
        OUT / "TABLE_step6dk2b2v3_task_specific_equivalence.csv",
        index=False
    )

    exact_occurrences = source_occurrence_scan(
        FRESH_CONFIG_SCALE_FLOORS
    )

    exact_occurrences.to_csv(
        OUT / "step6dk2b2v3_exact_scale_floor_occurrences.csv",
        index=False
    )

    metadata_floors = metadata_scale_floor_scan()

    metadata_floors.to_csv(
        OUT / "step6dk2b2v3_metadata_scale_floor_inventory.csv",
        index=False
    )

    # Match inventory values to task floors.
    floor_matches = []

    for task, expected in FRESH_CONFIG_SCALE_FLOORS.items():
        if len(metadata_floors):
            sub = metadata_floors[
                np.isclose(
                    metadata_floors["value"].astype(float),
                    expected,
                    rtol=0,
                    atol=1e-12
                )
            ].copy()
        else:
            sub = pd.DataFrame()

        floor_matches.append({
            "task": task,
            "expected_scale_floor": float(expected),
            "exact_text_occurrences": int(
                (exact_occurrences["task"] == task).sum()
                if len(exact_occurrences)
                else 0
            ),
            "metadata_matches": int(len(sub)),
            "metadata_match_files": (
                sorted(sub["file"].unique().tolist())
                if len(sub)
                else []
            ),
        })

    # Per-task summary.
    task_summaries = {}

    for task in NPZ_BY_TASK:
        q = eq[eq["task"] == task]

        task_summaries[task] = {
            "scale_floor":
                float(FRESH_CONFIG_SCALE_FLOORS[task]),

            "samples_tested":
                int(len(q)),

            "max_history_canonical_diff":
                float(q["history_canonical_max_abs_diff"].max()),

            "max_future_canonical_diff":
                float(q["future_canonical_max_abs_diff"].max()),

            "max_origin_metadata_diff_m":
                float(q["origin_metadata_max_abs_diff_m"].max()),

            "max_angle_metadata_diff_rad":
                float(q["angle_metadata_abs_diff_rad"].max()),

            "max_scale_metadata_diff":
                float(q["scale_metadata_abs_diff"].max()),

            "max_build_roundtrip_error_m":
                float(
                    max(
                        q["build_history_roundtrip_max_abs_error_m"].max(),
                        q["build_future_roundtrip_max_abs_error_m"].max(),
                    )
                ),

            "max_fresh_roundtrip_error_m":
                float(
                    max(
                        q["fresh_history_roundtrip_max_abs_error_m"].max(),
                        q["fresh_future_roundtrip_max_abs_error_m"].max(),
                    )
                ),

            "build_origin_dtypes":
                sorted(q["build_origin_dtype"].unique().tolist()),

            "fresh_origin_dtypes":
                sorted(q["fresh_origin_dtype"].unique().tolist()),
        }

    max_canonical_diff = float(
        max(
            eq["history_canonical_max_abs_diff"].max(),
            eq["future_canonical_max_abs_diff"].max(),
        )
    )

    max_fresh_roundtrip = float(
        max(
            eq["fresh_history_roundtrip_max_abs_error_m"].max(),
            eq["fresh_future_roundtrip_max_abs_error_m"].max(),
        )
    )

    canonical_equivalent = (
        max_canonical_diff <= ATOL_CANON
    )

    fresh_roundtrip_pass = (
        max_fresh_roundtrip
        <= ATOL_ROUNDTRIP_FRESH_M
    )

    all_tasks_have_source_occurrence = all(
        x["exact_text_occurrences"] >= 1
        for x in floor_matches
    )

    status_parts = []

    status_parts.append(
        "CANONICAL_MODEL_TENSORS_EQUIVALENT"
        if canonical_equivalent
        else "CANONICAL_MODEL_TENSOR_MISMATCH"
    )

    status_parts.append(
        "FRESH_FLOAT64_METADATA_ROUNDTRIP_PASS"
        if fresh_roundtrip_pass
        else "FRESH_ROUNDTRIP_FAIL"
    )

    status_parts.append(
        "TASK_FLOORS_CONFIRMED_IN_SOURCE"
        if all_tasks_have_source_occurrence
        else "TASK_FLOOR_SOURCE_CONFIRMATION_INCOMPLETE"
    )

    status = "__".join(status_parts)

    recommendation = (
        "If all PASS/CONFIRMED: use the fresh-holdout canonicalization "
        "implementation for BAMBI because its canonical tensors are identical "
        "to the training canonicalizer while its float64 origin/angle/scale "
        "metadata avoids UTM precision loss during inverse transformation."
    )

    summary = {
        "step":
            "STEP_6D_K2B2_V3_TASK_SPECIFIC_FLOOR_AND_EQUIVALENCE_AUDIT",

        "task_scale_floors":
            FRESH_CONFIG_SCALE_FLOORS,

        "task_summaries":
            task_summaries,

        "floor_source_matches":
            floor_matches,

        "max_canonical_tensor_diff":
            max_canonical_diff,

        "max_fresh_roundtrip_error_m":
            max_fresh_roundtrip,

        "status":
            status,

        "recommendation_if_pass":
            recommendation,

        "important_interpretation": [
            "SCALE_FLOOR_QUANTILE=0.05 is the 5th-percentile rule, not the numeric scale floor itself.",
            "The frozen model consumes canonical tensors, for which the two implementations are tested directly.",
            "Metadata precision is separately audited because BAMBI UTM coordinates are large.",
            "No forecasting, conformal coverage, or Step-7 evaluation is performed here."
        ],
    }

    with open(
        OUT / "STEP6D_K2B2_V3_SUMMARY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    with open(
        OUT / "STEP6D_K2B2_V3_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K2B2-V3 — TASK-SPECIFIC FLOOR + CANONICAL EQUIVALENCE AUDIT\n"
        )
        f.write("=" * 88 + "\n\n")

        f.write("TASK SUMMARIES\n--------------\n")
        f.write(json.dumps(task_summaries, indent=2))

        f.write("\n\nSCALE-FLOOR SOURCE MATCHES\n--------------------------\n")
        f.write(json.dumps(floor_matches, indent=2))

        f.write("\n\nSTATUS\n------\n")
        f.write(status + "\n")

        f.write("\nRECOMMENDATION IF PASS\n----------------------\n")
        f.write(recommendation + "\n")

    print("\nTASK SUMMARIES:")
    print(json.dumps(task_summaries, indent=2))

    print("\nSCALE-FLOOR SOURCE MATCHES:")
    print(json.dumps(floor_matches, indent=2))

    print("\nSTATUS:")
    print(status)

    print("\nRECOMMENDATION IF PASS:")
    print(recommendation)

    print("\nSTEP 6D-K2B2-V3 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
