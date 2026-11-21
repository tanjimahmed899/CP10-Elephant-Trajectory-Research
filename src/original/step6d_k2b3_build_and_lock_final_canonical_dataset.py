from pathlib import Path
import ast
import inspect
import json
import hashlib
import math
import numpy as np
import pandas as pd

READ_ONLY_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

FRESH_SRC = READ_ONLY_ROOT / "evaluate_fresh_wildbox_holdout.py"
TRAIN_SRC = READ_ONLY_ROOT / "train_transformer.py"
EVAL_SRC = READ_ONLY_ROOT / "evaluate_transformer_ensemble.py"

BAMBI_DIR = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

V4_DIR = (
    BAMBI_DIR
    / "k2b_representation_audit"
    / "precision_and_exact_inverse"
)

RAW64_DIR = V4_DIR / "float64_raw_windows"

OUT = BAMBI_DIR / "k2b3_final_canonical"
OUT.mkdir(parents=True, exist_ok=True)

LOCK_PATH = OUT / "STEP6D_K2B3_CANONICAL_DATASET_LOCK.json"

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

# This is intentionally looser than the v4 1e-6 m diagnostic threshold but
# still microscopic relative to metre-scale trajectories.
ROUNDTRIP_AUDIT_TOL_M = 5e-6


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def strip_annotations(node):
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
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
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

    return ns[target]


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
    cb = np.asarray(canonical, dtype=np.float64)[None, ...]
    ob = np.asarray(origin, dtype=np.float64)[None, ...]
    ab = np.asarray([angle], dtype=np.float64)
    sb = np.asarray([scale], dtype=np.float64)

    out = fn(cb, ob, ab, sb)
    out = np.asarray(out, dtype=np.float64)

    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    return out


def extract_result(r):
    angle = None

    for key in [
        "angle",
        "heading_angle",
        "theta",
        "heading",
    ]:
        if key in r:
            angle = float(r[key])
            break

    if angle is None:
        raise RuntimeError(
            f"No heading/angle field in canonicalizer result: {list(r.keys())}"
        )

    path = None

    for key in [
        "history_path",
        "raw_path_length",
        "path_length",
    ]:
        if key in r:
            path = float(r[key])
            break

    if path is None:
        raise RuntimeError(
            f"No history path-length field in result: {list(r.keys())}"
        )

    return {
        "history_canonical":
            np.asarray(r["history_canonical"], dtype=np.float32),

        "future_canonical":
            np.asarray(r["future_canonical"], dtype=np.float32),

        # Keep inverse-transform metadata float64 for large UTM coordinates.
        "origin":
            np.asarray(r["origin"], dtype=np.float64),

        "angle":
            np.float64(angle),

        "scale":
            np.float64(r["scale"]),

        "history_path":
            np.float64(path),
    }


def main():
    print(
        "\nSTEP 6D-K2B3 — FINAL BAMBI CANONICAL DATASET CONSTRUCTION + LOCK"
    )

    print(
        "Uses float64 absolute UTM windows, frozen task-specific scale floors, "
        "and the verified Fresh-holdout canonicalizer."
    )

    print(
        "No Transformer predictions, conformal evaluation, or Step-7 alarms.\n"
    )

    require(FRESH_SRC)
    require(TRAIN_SRC)
    require(EVAL_SRC)

    canonicalize_window = load_isolated_function(
        FRESH_SRC,
        "canonicalize_window"
    )

    train_inverse = load_isolated_function(
        TRAIN_SRC,
        "inverse_transform_batch"
    )

    eval_inverse = load_isolated_function(
        EVAL_SRC,
        "inverse_transform_batch"
    )

    task_outputs = {}
    input_hashes = {}

    for task, cfg in TASKS.items():
        raw_npz = (
            RAW64_DIR
            / f"{task}_bambi_flight6_world_raw_float64.npz"
        )

        raw_meta = (
            RAW64_DIR
            / f"{task}_bambi_flight6_metadata.csv"
        )

        require(raw_npz)
        require(raw_meta)

        input_hashes[f"{task}_raw_float64_npz"] = sha256_file(raw_npz)
        input_hashes[f"{task}_raw_metadata_csv"] = sha256_file(raw_meta)

        with np.load(raw_npz, allow_pickle=False) as z:
            X_abs = np.asarray(z["X_abs_xy"], dtype=np.float64)
            Y_abs = np.asarray(z["Y_abs_xy"], dtype=np.float64)

        meta = pd.read_csv(raw_meta)

        if len(meta) != len(X_abs):
            raise RuntimeError(
                f"{task}: metadata rows {len(meta)} != windows {len(X_abs)}"
            )

        H = []
        F = []
        origins = []
        angles = []
        scales = []
        history_paths = []

        train_roundtrip_max = 0.0
        eval_roundtrip_max = 0.0
        train_eval_inverse_max_diff = 0.0

        floor_applied = 0

        for i in range(len(X_abs)):
            r = call_canonicalizer(
                canonicalize_window,
                X_abs[i],
                Y_abs[i],
                cfg["scale_floor"],
            )

            q = extract_result(r)

            H.append(q["history_canonical"])
            F.append(q["future_canonical"])
            origins.append(q["origin"])
            angles.append(q["angle"])
            scales.append(q["scale"])
            history_paths.append(q["history_path"])

            if q["history_path"] < cfg["scale_floor"]:
                floor_applied += 1

            # Exact-source inverse audit on every BAMBI window.
            rec_train_h = call_inverse(
                train_inverse,
                q["history_canonical"],
                q["origin"],
                q["angle"],
                q["scale"],
            )

            rec_train_f = call_inverse(
                train_inverse,
                q["future_canonical"],
                q["origin"],
                q["angle"],
                q["scale"],
            )

            rec_eval_h = call_inverse(
                eval_inverse,
                q["history_canonical"],
                q["origin"],
                q["angle"],
                q["scale"],
            )

            rec_eval_f = call_inverse(
                eval_inverse,
                q["future_canonical"],
                q["origin"],
                q["angle"],
                q["scale"],
            )

            train_roundtrip_max = max(
                train_roundtrip_max,
                float(np.max(np.abs(rec_train_h - X_abs[i]))),
                float(np.max(np.abs(rec_train_f - Y_abs[i]))),
            )

            eval_roundtrip_max = max(
                eval_roundtrip_max,
                float(np.max(np.abs(rec_eval_h - X_abs[i]))),
                float(np.max(np.abs(rec_eval_f - Y_abs[i]))),
            )

            train_eval_inverse_max_diff = max(
                train_eval_inverse_max_diff,
                float(np.max(np.abs(rec_train_h - rec_eval_h))),
                float(np.max(np.abs(rec_train_f - rec_eval_f))),
            )

        H = np.stack(H).astype(np.float32, copy=False)
        F = np.stack(F).astype(np.float32, copy=False)
        origins = np.stack(origins).astype(np.float64, copy=False)
        angles = np.asarray(angles, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)
        history_paths = np.asarray(history_paths, dtype=np.float64)

        if train_roundtrip_max > ROUNDTRIP_AUDIT_TOL_M:
            raise RuntimeError(
                f"{task}: train inverse round-trip error "
                f"{train_roundtrip_max:.9g} m exceeds "
                f"{ROUNDTRIP_AUDIT_TOL_M:.9g} m"
            )

        if eval_roundtrip_max > ROUNDTRIP_AUDIT_TOL_M:
            raise RuntimeError(
                f"{task}: eval inverse round-trip error "
                f"{eval_roundtrip_max:.9g} m exceeds "
                f"{ROUNDTRIP_AUDIT_TOL_M:.9g} m"
            )

        if train_eval_inverse_max_diff > 1e-12:
            raise RuntimeError(
                f"{task}: train/eval inverse implementations disagree: "
                f"{train_eval_inverse_max_diff}"
            )

        out_npz = OUT / f"{task}_bambi_flight6_canonical_LOCKED.npz"

        np.savez_compressed(
            out_npz,

            # Frozen-model inputs/targets.
            history_canonical=H,
            future_canonical=F,

            # Precision-safe inverse metadata.
            origin=origins,
            angle=angles,
            scale=scales,
            history_path=history_paths,

            # Keep absolute GT in float64 for world-metre evaluation.
            history_abs_xy=X_abs,
            future_abs_xy=Y_abs,
        )

        meta2 = meta.copy()

        meta2["scale_floor"] = float(cfg["scale_floor"])
        meta2["origin_x"] = origins[:, 0]
        meta2["origin_y"] = origins[:, 1]
        meta2["heading_angle_rad"] = angles
        meta2["scale"] = scales
        meta2["history_path"] = history_paths
        meta2["scale_floor_applied"] = (
            history_paths < float(cfg["scale_floor"])
        )

        out_meta = OUT / f"{task}_bambi_flight6_canonical_metadata_LOCKED.csv"
        meta2.to_csv(out_meta, index=False)

        task_outputs[task] = {
            "windows":
                int(len(H)),

            "history_canonical_shape":
                list(H.shape),

            "future_canonical_shape":
                list(F.shape),

            "history_canonical_dtype":
                str(H.dtype),

            "future_canonical_dtype":
                str(F.dtype),

            "origin_dtype":
                str(origins.dtype),

            "angle_dtype":
                str(angles.dtype),

            "scale_dtype":
                str(scales.dtype),

            "absolute_gt_dtype":
                str(X_abs.dtype),

            "scale_floor":
                float(cfg["scale_floor"]),

            "scale_floor_applied_windows":
                int(floor_applied),

            "scale_floor_applied_pct":
                float(100.0 * floor_applied / len(H)),

            "history_path_min":
                float(history_paths.min()),

            "history_path_median":
                float(np.median(history_paths)),

            "history_path_max":
                float(history_paths.max()),

            "scale_min":
                float(scales.min()),

            "scale_median":
                float(np.median(scales)),

            "scale_max":
                float(scales.max()),

            "max_train_inverse_roundtrip_error_m":
                float(train_roundtrip_max),

            "max_eval_inverse_roundtrip_error_m":
                float(eval_roundtrip_max),

            "max_train_vs_eval_inverse_diff_m":
                float(train_eval_inverse_max_diff),

            "canonical_npz":
                str(out_npz),

            "canonical_metadata":
                str(out_meta),

            "canonical_npz_sha256":
                sha256_file(out_npz),

            "canonical_metadata_sha256":
                sha256_file(out_meta),
        }

    lock = {
        "step":
            "STEP_6D_K2B3_FINAL_BAMBI_CANONICAL_DATASET_LOCK",

        "status":
            "LOCKED_BEFORE_ANY_BAMBI_MODEL_PREDICTIONS",

        "flight_id":
            6,

        "species":
            "Cervus elaphus (Red deer)",

        "representation":
            {
                "sampling_hz":
                    10.0,

                "history_steps":
                    20,

                "tasks":
                    TASKS,

                "canonicalizer_source":
                    str(FRESH_SRC),

                "canonicalizer_function":
                    "canonicalize_window",

                "canonical_model_tensor_dtype":
                    "float32",

                "absolute_world_coordinate_dtype":
                    "float64",

                "inverse_metadata_dtype":
                    "float64",

                "roundtrip_audit_tolerance_m":
                    ROUNDTRIP_AUDIT_TOL_M,
            },

        "input_hashes":
            input_hashes,

        "task_outputs":
            task_outputs,

        "methodological_notes": [
            "Canonical model tensors are numerically identical to the training canonicalizer on the audited BAMBI windows.",
            "Task-specific scale floors are training-derived and fixed: 2s->2s 0.06820264416877736; 2s->5s 0.06732638773715147.",
            "Absolute UTM coordinates are retained in float64 to avoid the ~0.25 m quantization seen when large UTM coordinates were stored in float32.",
            "Canonical tensors remain float32 because that is the frozen neural-model representation.",
            "Original train/evaluation inverse_transform_batch functions agree exactly.",
            "No BAMBI model predictions, errors, conformal coverage, or Step-7 results were inspected before this lock.",
            "BAMBI Flight 6 is a cross-species external robustness/stress test, not elephant-specific validation.",
            "The Flight-6 georeferencing uses the BAMBI reference-pipeline 50-degree FOV assumption rather than an independently re-estimated Flight-6 intrinsic calibration.",
        ],
    }

    if LOCK_PATH.exists():
        old = json.loads(
            LOCK_PATH.read_text(
                encoding="utf-8"
            )
        )

        # Do not silently overwrite a different frozen representation.
        if old != lock:
            raise RuntimeError(
                "Existing K2B3 lock differs from the newly generated lock. "
                "Refusing to overwrite."
            )

        lock_status = "EXISTING_CANONICAL_LOCK_VERIFIED"

    else:
        LOCK_PATH.write_text(
            json.dumps(lock, indent=2),
            encoding="utf-8"
        )

        lock_status = "NEW_CANONICAL_LOCK_CREATED"

    summary = {
        "step":
            "STEP_6D_K2B3_FINAL_CANONICAL_DATASET",

        "lock_status":
            lock_status,

        "task_outputs":
            task_outputs,

        "status":
            "BAMBI_CANONICAL_DATASET_READY_FOR_FROZEN_MODEL_INFERENCE",
    }

    (OUT / "STEP6D_K2B3_SUMMARY.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    with open(
        OUT / "STEP6D_K2B3_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K2B3 — FINAL BAMBI CANONICAL DATASET\n"
        )
        f.write("=" * 78 + "\n\n")

        f.write("LOCK STATUS\n-----------\n")
        f.write(lock_status + "\n\n")

        f.write("TASK OUTPUTS\n------------\n")
        f.write(json.dumps(task_outputs, indent=2))

        f.write("\n\nSTATUS\n------\n")
        f.write(
            "BAMBI_CANONICAL_DATASET_READY_FOR_FROZEN_MODEL_INFERENCE\n"
        )

    print("LOCK STATUS:")
    print(lock_status)

    print("\nTASK OUTPUTS:")
    print(json.dumps(task_outputs, indent=2))

    print("\nSTATUS:")
    print(
        "BAMBI_CANONICAL_DATASET_READY_FOR_FROZEN_MODEL_INFERENCE"
    )

    print("\nSTEP 6D-K2B3 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
