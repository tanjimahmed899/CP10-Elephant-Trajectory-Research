from pathlib import Path
import ast
import inspect
import json
import math
import numpy as np
import pandas as pd

READ_ONLY_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

BUILD_SRC = READ_ONLY_ROOT / "build_final_neural_dataset.py"
FRESH_SRC = READ_ONLY_ROOT / "evaluate_fresh_wildbox_holdout.py"
TRAIN_SRC = READ_ONLY_ROOT / "train_transformer.py"
EVAL_SRC = READ_ONLY_ROOT / "evaluate_transformer_ensemble.py"

BAMBI_DIR = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

K2A = BAMBI_DIR / "k2a_10hz"
RESAMPLED = K2A / "step6dk2a_resampled_10hz_world_trajectory.csv"

OLD_NPZ = {
    "2s_to_2s": K2A / "2s_to_2s_bambi_flight6_world_raw.npz",
    "2s_to_5s": K2A / "2s_to_5s_bambi_flight6_world_raw.npz",
}

OUT = BAMBI_DIR / "k2b_representation_audit" / "precision_and_exact_inverse"
OUT.mkdir(parents=True, exist_ok=True)

FLOAT64_DIR = OUT / "float64_raw_windows"
FLOAT64_DIR.mkdir(parents=True, exist_ok=True)

TARGET_HZ = 10.0
DT = 0.1
HISTORY_STEPS = 20

TASKS = {
    "2s_to_2s": {
        "future_steps": 20,
        "scale_floor": 0.06820264416877736,
    },
    "2s_to_5s": {
        "future_steps": 50,
        "scale_floor": 0.06732638773715147,
    },
}

ATOL_CANON = 1e-7
ATOL_ROUNDTRIP_M = 1e-6


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def strip_annotations(node):
    """
    Remove annotations from isolated copied functions so source extraction
    does not depend on unrelated typing imports.
    """
    node = ast.fix_missing_locations(node)

    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.returns = None

            for arg in (
                list(n.args.args)
                + list(n.args.kwonlyargs)
                + list(n.args.posonlyargs)
            ):
                arg.annotation = None

            if n.args.vararg:
                n.args.vararg.annotation = None

            if n.args.kwarg:
                n.args.kwarg.annotation = None

        elif isinstance(n, ast.AnnAssign):
            n.annotation = None

    return node


def top_level_maps(tree):
    funcs = {}
    literal_assigns = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node

        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        ast.literal_eval(node.value)
                        literal_assigns[t.id] = node
                    except Exception:
                        pass

        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                ast.literal_eval(node.value)
                literal_assigns[node.target.id] = node
            except Exception:
                pass

    return funcs, literal_assigns


def called_names(node):
    out = set()

    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                out.add(n.func.id)

    return out


def loaded_names(node):
    local = set()

    for a in (
        list(node.args.args)
        + list(node.args.kwonlyargs)
        + list(node.args.posonlyargs)
    ):
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


def function_closure(target, funcs):
    needed = set()
    stack = [target]

    while stack:
        name = stack.pop()

        if name in needed or name not in funcs:
            continue

        needed.add(name)

        for c in called_names(funcs[name]):
            if c in funcs and c not in needed:
                stack.append(c)

    return needed


def load_isolated_function(path, target):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text)

    funcs, literal_assigns = top_level_maps(tree)

    if target not in funcs:
        raise RuntimeError(f"{target} not found in {path}")

    needed_funcs = function_closure(target, funcs)

    needed_globals = set()

    for fname in needed_funcs:
        needed_globals |= loaded_names(funcs[fname])

    body = []

    for name, node in literal_assigns.items():
        if name in needed_globals:
            body.append(node)

    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in needed_funcs
        ):
            body.append(node)

    mod = ast.Module(body=body, type_ignores=[])
    mod = strip_annotations(mod)
    ast.fix_missing_locations(mod)

    ns = {
        "np": np,
        "math": math,
    }

    exec(
        compile(mod, filename=str(path), mode="exec"),
        ns,
        ns
    )

    fn = ns[target]

    return {
        "function": fn,
        "signature": inspect.signature(fn),
        "source_text": text,
        "tree": tree,
        "needed_functions": sorted(needed_funcs),
    }


def source_function_text(path, target):
    text = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text)

    matches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == target
    ]

    if len(matches) != 1:
        return ""

    return ast.get_source_segment(text, matches[0]) or ""


def rebuild_float64_windows(resampled, future_steps):
    X = []
    Y = []
    meta = []

    total = HISTORY_STEPS + future_steps
    sample_index = 0

    for block_key, g in resampled.groupby("block_key", sort=True):
        g = g.sort_values("time_sec").reset_index(drop=True)

        t = g["time_sec"].to_numpy(dtype=np.float64)
        xy = g[["utm_x", "utm_y"]].to_numpy(dtype=np.float64)

        if len(g) < total:
            continue

        if len(t) > 1 and not np.allclose(
            np.diff(t),
            DT,
            atol=1e-9,
            rtol=0,
        ):
            raise RuntimeError(f"Non-10Hz block: {block_key}")

        for start in range(0, len(g) - total + 1):
            h1 = start + HISTORY_STEPS
            f1 = h1 + future_steps

            X.append(xy[start:h1].copy())
            Y.append(xy[h1:f1].copy())

            first = g.iloc[start]
            last = g.iloc[f1 - 1]

            meta.append({
                "sample_index": int(sample_index),
                "track_id": int(first["track_id"]),
                "track_key": str(block_key),
                "window_start_time_sec": float(first["time_sec"]),
                "window_end_time_sec": float(last["time_sec"]),
            })

            sample_index += 1

    if not X:
        return (
            np.empty((0, HISTORY_STEPS, 2), dtype=np.float64),
            np.empty((0, future_steps, 2), dtype=np.float64),
            pd.DataFrame(meta),
        )

    return (
        np.stack(X).astype(np.float64, copy=False),
        np.stack(Y).astype(np.float64, copy=False),
        pd.DataFrame(meta),
    )


def extract_canonical(result):
    angle = None

    for key in ["angle", "heading_angle", "theta", "heading"]:
        if key in result:
            angle = result[key]
            break

    if angle is None:
        raise RuntimeError(f"No angle key in {list(result.keys())}")

    return {
        "history": np.asarray(result["history_canonical"], dtype=np.float64),
        "future": np.asarray(result["future_canonical"], dtype=np.float64),
        "origin": np.asarray(result["origin"]),
        "angle": float(angle),
        "scale": float(result["scale"]),
    }


def call_canonicalizer(fn, h, f, floor):
    try:
        return fn(h, f, floor)
    except TypeError:
        sig = inspect.signature(fn)
        names = list(sig.parameters.keys())

        return fn(
            **{
                names[0]: h,
                names[1]: f,
                "scale_floor": floor,
            }
        )


def call_inverse(fn, canonical, origin, angle, scale):
    canonical_b = np.asarray(canonical, dtype=np.float64)[None, ...]
    origin_b = np.asarray(origin, dtype=np.float64)[None, ...]
    angle_b = np.asarray([angle], dtype=np.float64)
    scale_b = np.asarray([scale], dtype=np.float64)

    # First use the natural four-positional convention.
    try:
        out = fn(
            canonical_b,
            origin_b,
            angle_b,
            scale_b
        )
    except TypeError:
        sig = inspect.signature(fn)
        kwargs = {}

        for name in sig.parameters:
            low = name.lower()

            if "canon" in low or "pred" in low or "traj" in low:
                kwargs[name] = canonical_b
            elif "origin" in low:
                kwargs[name] = origin_b
            elif "angle" in low or "theta" in low or "heading" in low:
                kwargs[name] = angle_b
            elif "scale" in low:
                kwargs[name] = scale_b

        out = fn(**kwargs)

    out = np.asarray(out, dtype=np.float64)

    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    return out


def sample_indices(n, max_n=250):
    if n <= max_n:
        return list(range(n))

    return sorted(
        set(
            np.linspace(0, n - 1, max_n, dtype=int).tolist()
        )
    )


def quantization_stats(old_arr, new_arr):
    if old_arr.shape != new_arr.shape:
        raise RuntimeError(
            f"Shape mismatch old={old_arr.shape}, new={new_arr.shape}"
        )

    d = np.abs(
        old_arr.astype(np.float64)
        -
        new_arr.astype(np.float64)
    )

    radial = np.linalg.norm(
        old_arr.astype(np.float64)
        -
        new_arr.astype(np.float64),
        axis=-1
    )

    return {
        "max_abs_coordinate_error_m": float(d.max()),
        "median_abs_coordinate_error_m": float(np.median(d)),
        "p95_abs_coordinate_error_m": float(np.quantile(d, 0.95)),
        "max_radial_xy_error_m": float(radial.max()),
        "median_radial_xy_error_m": float(np.median(radial)),
        "p95_radial_xy_error_m": float(np.quantile(radial, 0.95)),
    }


def main():
    print(
        "\nSTEP 6D-K2B2-V4 — FLOAT64 UTM REPAIR + ORIGINAL-SOURCE INVERSE ROUNDTRIP"
    )
    print(
        "No forecasting. No conformal evaluation. No Step-7 evaluation.\n"
    )

    for p in [
        BUILD_SRC,
        FRESH_SRC,
        TRAIN_SRC,
        EVAL_SRC,
        RESAMPLED,
        OLD_NPZ["2s_to_2s"],
        OLD_NPZ["2s_to_5s"],
    ]:
        require(p)

    resampled = pd.read_csv(RESAMPLED)

    # Rebuild precision-safe absolute UTM windows from the float64 CSV.
    rebuilt = {}
    quant_audit = {}

    for task, cfg in TASKS.items():
        X64, Y64, meta = rebuild_float64_windows(
            resampled,
            cfg["future_steps"]
        )

        old = np.load(
            OLD_NPZ[task],
            allow_pickle=False
        )

        Xold = old["X_abs_xy"]
        Yold = old["Y_abs_xy"]

        if X64.shape != Xold.shape or Y64.shape != Yold.shape:
            raise RuntimeError(
                f"{task}: window count/shape changed during float64 repair. "
                f"old X{Xold.shape} Y{Yold.shape}; "
                f"new X{X64.shape} Y{Y64.shape}"
            )

        qx = quantization_stats(Xold, X64)
        qy = quantization_stats(Yold, Y64)

        quant_audit[task] = {
            "windows": int(len(X64)),
            "X_dtype_old": str(Xold.dtype),
            "Y_dtype_old": str(Yold.dtype),
            "X_dtype_new": str(X64.dtype),
            "Y_dtype_new": str(Y64.dtype),
            "history_quantization": qx,
            "future_quantization": qy,
        }

        np.savez_compressed(
            FLOAT64_DIR / f"{task}_bambi_flight6_world_raw_float64.npz",
            X_abs_xy=X64,
            Y_abs_xy=Y64,
        )

        meta.to_csv(
            FLOAT64_DIR / f"{task}_bambi_flight6_metadata.csv",
            index=False
        )

        rebuilt[task] = (X64, Y64)

    # Load original source functions, not hand-written transforms.
    build_can = load_isolated_function(
        BUILD_SRC,
        "canonicalize_sample"
    )

    fresh_can = load_isolated_function(
        FRESH_SRC,
        "canonicalize_window"
    )

    train_inv = load_isolated_function(
        TRAIN_SRC,
        "inverse_transform_batch"
    )

    eval_inv = load_isolated_function(
        EVAL_SRC,
        "inverse_transform_batch"
    )

    print("SOURCE FUNCTION SIGNATURES:")
    print(
        " build canonicalizer:",
        build_can["signature"]
    )
    print(
        " fresh canonicalizer:",
        fresh_can["signature"]
    )
    print(
        " train inverse:",
        train_inv["signature"]
    )
    print(
        " eval inverse:",
        eval_inv["signature"]
    )

    # Save exact source functions for audit.
    source_bundle = {
        "build_canonicalize_sample": source_function_text(
            BUILD_SRC,
            "canonicalize_sample"
        ),
        "fresh_canonicalize_window": source_function_text(
            FRESH_SRC,
            "canonicalize_window"
        ),
        "train_inverse_transform_batch": source_function_text(
            TRAIN_SRC,
            "inverse_transform_batch"
        ),
        "eval_inverse_transform_batch": source_function_text(
            EVAL_SRC,
            "inverse_transform_batch"
        ),
    }

    with open(
        OUT / "STEP6D_K2B2_V4_EXACT_SOURCE_FUNCTIONS.txt",
        "w",
        encoding="utf-8"
    ) as f:
        for name, src in source_bundle.items():
            f.write("\n" + "=" * 100 + "\n")
            f.write(name + "\n")
            f.write("=" * 100 + "\n")
            f.write(src + "\n")

    rows = []

    for task, cfg in TASKS.items():
        floor = cfg["scale_floor"]
        X64, Y64 = rebuilt[task]

        for i in sample_indices(len(X64)):
            h = X64[i]
            fut = Y64[i]

            a = extract_canonical(
                call_canonicalizer(
                    build_can["function"],
                    h,
                    fut,
                    floor
                )
            )

            b = extract_canonical(
                call_canonicalizer(
                    fresh_can["function"],
                    h,
                    fut,
                    floor
                )
            )

            hist_can_diff = float(
                np.max(
                    np.abs(
                        a["history"] - b["history"]
                    )
                )
            )

            fut_can_diff = float(
                np.max(
                    np.abs(
                        a["future"] - b["future"]
                    )
                )
            )

            # Use fresh float64 metadata for BAMBI world-coordinate inversion.
            rec_train_h = call_inverse(
                train_inv["function"],
                b["history"],
                b["origin"],
                b["angle"],
                b["scale"]
            )

            rec_train_f = call_inverse(
                train_inv["function"],
                b["future"],
                b["origin"],
                b["angle"],
                b["scale"]
            )

            rec_eval_h = call_inverse(
                eval_inv["function"],
                b["history"],
                b["origin"],
                b["angle"],
                b["scale"]
            )

            rec_eval_f = call_inverse(
                eval_inv["function"],
                b["future"],
                b["origin"],
                b["angle"],
                b["scale"]
            )

            rows.append({
                "task": task,
                "sample_index": int(i),
                "scale_floor": float(floor),

                "history_canonical_diff": hist_can_diff,
                "future_canonical_diff": fut_can_diff,

                "train_inverse_history_max_error_m": float(
                    np.max(np.abs(rec_train_h - h))
                ),
                "train_inverse_future_max_error_m": float(
                    np.max(np.abs(rec_train_f - fut))
                ),

                "eval_inverse_history_max_error_m": float(
                    np.max(np.abs(rec_eval_h - h))
                ),
                "eval_inverse_future_max_error_m": float(
                    np.max(np.abs(rec_eval_f - fut))
                ),

                "train_vs_eval_inverse_history_max_diff_m": float(
                    np.max(np.abs(rec_train_h - rec_eval_h))
                ),
                "train_vs_eval_inverse_future_max_diff_m": float(
                    np.max(np.abs(rec_train_f - rec_eval_f))
                ),

                "fresh_origin_dtype": str(np.asarray(b["origin"]).dtype),
            })

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT / "TABLE_step6dk2b2v4_exact_inverse_roundtrip.csv",
        index=False
    )

    task_summary = {}

    for task in TASKS:
        q = df[df["task"] == task]

        task_summary[task] = {
            "scale_floor": TASKS[task]["scale_floor"],
            "samples_tested": int(len(q)),

            "max_history_canonical_diff": float(
                q["history_canonical_diff"].max()
            ),
            "max_future_canonical_diff": float(
                q["future_canonical_diff"].max()
            ),

            "max_train_inverse_roundtrip_error_m": float(
                max(
                    q["train_inverse_history_max_error_m"].max(),
                    q["train_inverse_future_max_error_m"].max(),
                )
            ),

            "max_eval_inverse_roundtrip_error_m": float(
                max(
                    q["eval_inverse_history_max_error_m"].max(),
                    q["eval_inverse_future_max_error_m"].max(),
                )
            ),

            "max_train_vs_eval_inverse_diff_m": float(
                max(
                    q["train_vs_eval_inverse_history_max_diff_m"].max(),
                    q["train_vs_eval_inverse_future_max_diff_m"].max(),
                )
            ),

            "float64_raw_quantization_audit": quant_audit[task],
        }

    max_can = float(
        max(
            df["history_canonical_diff"].max(),
            df["future_canonical_diff"].max(),
        )
    )

    max_roundtrip = float(
        max(
            df["train_inverse_history_max_error_m"].max(),
            df["train_inverse_future_max_error_m"].max(),
            df["eval_inverse_history_max_error_m"].max(),
            df["eval_inverse_future_max_error_m"].max(),
        )
    )

    max_inverse_disagreement = float(
        max(
            df["train_vs_eval_inverse_history_max_diff_m"].max(),
            df["train_vs_eval_inverse_future_max_diff_m"].max(),
        )
    )

    statuses = []

    statuses.append(
        "FLOAT64_RAW_WINDOWS_REBUILT_WITH_IDENTICAL_COUNTS"
    )

    statuses.append(
        "CANONICAL_MODEL_TENSORS_EQUIVALENT"
        if max_can <= ATOL_CANON
        else "CANONICAL_MODEL_TENSOR_MISMATCH"
    )

    statuses.append(
        "ORIGINAL_SOURCE_INVERSE_ROUNDTRIP_PASS"
        if max_roundtrip <= ATOL_ROUNDTRIP_M
        else "ORIGINAL_SOURCE_INVERSE_ROUNDTRIP_FAIL"
    )

    statuses.append(
        "TRAIN_EVAL_INVERSE_AGREE"
        if max_inverse_disagreement <= ATOL_ROUNDTRIP_M
        else "TRAIN_EVAL_INVERSE_DISAGREE"
    )

    status = "__".join(statuses)

    summary = {
        "step":
            "STEP_6D_K2B2_V4_FLOAT64_UTM_AND_EXACT_SOURCE_INVERSE_AUDIT",

        "task_summaries":
            task_summary,

        "max_canonical_tensor_diff":
            max_can,

        "max_original_source_roundtrip_error_m":
            max_roundtrip,

        "max_train_vs_eval_inverse_diff_m":
            max_inverse_disagreement,

        "status":
            status,

        "important_interpretation": [
            "Absolute UTM windows are rebuilt as float64 before canonicalization.",
            "Canonical model tensors may still be stored as float32 after canonicalization.",
            "The round-trip uses the original project inverse_transform_batch implementation, not a hand-written reconstruction.",
            "The fixed task-specific training-derived scale floors are preserved.",
            "No Transformer prediction, conformal coverage, or warning evaluation is performed."
        ],
    }

    with open(
        OUT / "STEP6D_K2B2_V4_SUMMARY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    with open(
        OUT / "STEP6D_K2B2_V4_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K2B2-V4 — FLOAT64 UTM REPAIR + ORIGINAL-SOURCE INVERSE AUDIT\n"
        )
        f.write("=" * 92 + "\n\n")

        f.write("FLOAT32 -> FLOAT64 RAW-WINDOW PRECISION AUDIT\n")
        f.write("---------------------------------------------\n")
        f.write(json.dumps(quant_audit, indent=2))

        f.write("\n\nTASK ROUNDTRIP SUMMARIES\n")
        f.write("------------------------\n")
        f.write(json.dumps(task_summary, indent=2))

        f.write("\n\nSTATUS\n")
        f.write("------\n")
        f.write(status + "\n")

    print("\nFLOAT32 -> FLOAT64 RAW-WINDOW PRECISION AUDIT:")
    print(json.dumps(quant_audit, indent=2))

    print("\nTASK ROUNDTRIP SUMMARIES:")
    print(json.dumps(task_summary, indent=2))

    print("\nSTATUS:")
    print(status)

    print("\nSTEP 6D-K2B2-V4 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
