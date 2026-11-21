from pathlib import Path
import importlib.util
import inspect
import json
import hashlib
import sys

import numpy as np
import pandas as pd
import torch


READ_ONLY_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

PRIMARY_EVAL = (
    READ_ONLY_ROOT
    / "evaluate_birdsai_testreal_frozen_ensemble.py"
)

MODEL_LOCK = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_frozen_ensemble"
    / "model_lock"
    / "STEP6D_K3A1_FROZEN_TRANSFORMER_MODEL_LOCK.json"
)

CANONICAL_DIR = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k2b3_final_canonical"
)

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_frozen_ensemble"
    / "first_inference"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "2s_to_2s": {
        "future_steps": 20,
        "npz": CANONICAL_DIR / "2s_to_2s_bambi_flight6_canonical_LOCKED.npz",
        "meta": CANONICAL_DIR / "2s_to_2s_bambi_flight6_canonical_metadata_LOCKED.csv",
    },
    "2s_to_5s": {
        "future_steps": 50,
        "npz": CANONICAL_DIR / "2s_to_5s_bambi_flight6_canonical_LOCKED.npz",
        "meta": CANONICAL_DIR / "2s_to_5s_bambi_flight6_canonical_metadata_LOCKED.csv",
    },
}

BATCH_SIZE = 256
DEVICE = "cpu"


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


def load_isolated_evaluator_namespace(path):
    """
    Load only the dependency closure required by the original evaluator's
    load_transformer() and inverse_transform_batch(), without executing
    the evaluator's top-level benchmark workflow.

    This version resolves derived globals such as:
        ROOT = Path(...)
        MODEL_ROOT = ROOT / "transformer_results"
    plus any recursively required constants, helper functions, and classes.
    """
    import ast
    import copy

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text
    )

    # ------------------------------------------------------------
    # Index top-level definitions and assignments.
    # ------------------------------------------------------------
    funcs = {}
    classes = {}
    assignments = {}
    imports = []

    def assignment_target_names(node):
        names = []

        if isinstance(node, ast.Assign):
            targets = node.targets

        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]

        else:
            return names

        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)

            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)

        return names

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node

        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in assignment_target_names(node):
                assignments[name] = node

    # ------------------------------------------------------------
    # Utility: collect names loaded by a node, excluding obvious locals.
    # ------------------------------------------------------------
    def loaded_names(node):
        loaded = set()
        stored = set()

        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Load):
                    loaded.add(n.id)

                elif isinstance(n.ctx, ast.Store):
                    stored.add(n.id)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in (
                list(node.args.args)
                + list(node.args.kwonlyargs)
                + list(node.args.posonlyargs)
            ):
                stored.add(a.arg)

            if node.args.vararg:
                stored.add(node.args.vararg.arg)

            if node.args.kwarg:
                stored.add(node.args.kwarg.arg)

        return loaded - stored

    # ------------------------------------------------------------
    # Build recursive dependency closure.
    # ------------------------------------------------------------
    required_funcs = {
        "load_transformer",
        "inverse_transform_batch",
    }

    required_classes = set()
    required_globals = set()

    queue = [
        ("func", "load_transformer"),
        ("func", "inverse_transform_batch"),
    ]

    seen = set()

    while queue:
        kind, name = queue.pop()

        key = (kind, name)

        if key in seen:
            continue

        seen.add(key)

        if kind == "func":
            if name not in funcs:
                continue

            node = funcs[name]

        elif kind == "class":
            if name not in classes:
                continue

            node = classes[name]

        elif kind == "global":
            if name not in assignments:
                continue

            node = assignments[name]

        else:
            continue

        deps = loaded_names(node)

        for dep in deps:
            if dep in funcs:
                if dep not in required_funcs:
                    required_funcs.add(dep)
                    queue.append(("func", dep))

            elif dep in classes:
                if dep not in required_classes:
                    required_classes.add(dep)
                    queue.append(("class", dep))

            elif dep in assignments:
                if dep not in required_globals:
                    required_globals.add(dep)
                    queue.append(("global", dep))

    # A class may be instantiated by name without the loader AST seeing
    # internal helper-class dependencies until the class body is inspected.
    # The queue above recursively inspects class bodies once discovered.

    # ------------------------------------------------------------
    # Preserve original top-level order for imports, required assignments,
    # required classes and required functions.
    # ------------------------------------------------------------
    body = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(
                copy.deepcopy(node)
            )

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_names = assignment_target_names(node)

            if any(
                name in required_globals
                for name in target_names
            ):
                body.append(
                    copy.deepcopy(node)
                )

        elif isinstance(node, ast.ClassDef):
            if node.name in required_classes:
                body.append(
                    copy.deepcopy(node)
                )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in required_funcs:
                body.append(
                    copy.deepcopy(node)
                )

    isolated = ast.Module(
        body=body,
        type_ignores=[]
    )

    ast.fix_missing_locations(
        isolated
    )

    ns = {
        "__name__":
            "bambi_locked_primary_frozen_evaluator_isolated",

        "__file__":
            str(path),
    }

    exec(
        compile(
            isolated,
            filename=str(path),
            mode="exec"
        ),
        ns,
        ns
    )

    required_exports = [
        "load_transformer",
        "inverse_transform_batch",
    ]

    missing = [
        name
        for name in required_exports
        if name not in ns
    ]

    if missing:
        raise RuntimeError(
            "Isolated evaluator namespace missing required definitions: "
            f"{missing}"
        )

    print(
        "ISOLATED DEPENDENCY CLOSURE:"
    )

    print(
        "  functions =",
        sorted(required_funcs)
    )

    print(
        "  classes   =",
        sorted(required_classes)
    )

    print(
        "  globals   =",
        sorted(required_globals)
    )

    return ns


def call_load_transformer(
    fn,
    checkpoint_path,
    task,
    future_steps,
    seed,
    device
):
    sig = inspect.signature(fn)

    kwargs = {}

    for name, p in sig.parameters.items():
        low = name.lower()

        if any(
            key in low
            for key in [
                "path",
                "checkpoint",
                "ckpt",
                "weight",
                "model_file",
            ]
        ):
            kwargs[name] = checkpoint_path

        elif low in {
            "obs_frames",
            "obs_len",
            "history_frames",
            "history_len",
            "input_len",
            "input_length",
            "seq_len",
        } or (
            "obs" in low
            and
            (
                "frame" in low
                or
                "len" in low
                or
                "step" in low
            )
        ):
            kwargs[name] = 20

        elif low in {
            "future_frames",
            "future_len",
            "pred_frames",
            "pred_len",
            "prediction_len",
            "output_len",
            "output_length",
            "horizon_frames",
        } or (
            (
                "future" in low
                or
                "pred" in low
                or
                "output" in low
            )
            and
            (
                "frame" in low
                or
                "len" in low
                or
                "step" in low
            )
        ):
            kwargs[name] = future_steps

        elif "device" in low:
            kwargs[name] = device

        elif low == "task" or "task_name" in low:
            kwargs[name] = task

        elif "seed" in low:
            kwargs[name] = seed

        elif p.default is not inspect._empty:
            # Let source default apply.
            continue

        else:
            raise RuntimeError(
                "Unresolved required load_transformer argument "
                f"{name!r}; signature={sig}"
            )

    model = fn(
        **kwargs
    )

    # Some loaders may return (model, metadata/config).
    if isinstance(model, (tuple, list)):
        candidates = [
            x
            for x in model
            if isinstance(x, torch.nn.Module)
        ]

        if len(candidates) == 1:
            model = candidates[0]

    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            "load_transformer did not return a torch.nn.Module. "
            f"Returned type={type(model)}"
        )

    model = model.to(
        device
    )

    model.eval()

    return model, kwargs, str(sig)


def normalize_model_output(output, expected_future_steps):
    if isinstance(output, dict):
        for key in [
            "prediction",
            "predictions",
            "pred",
            "output",
            "trajectory",
            "future",
        ]:
            if key in output:
                output = output[key]
                break

    if isinstance(output, (tuple, list)):
        tensor_items = [
            x
            for x in output
            if torch.is_tensor(x)
        ]

        if not tensor_items:
            raise RuntimeError(
                "Model output tuple/list has no tensor."
            )

        # Prefer tensor matching [B, future, 2].
        matches = [
            x
            for x in tensor_items
            if x.ndim == 3
            and
            x.shape[-1] == 2
            and
            x.shape[1] == expected_future_steps
        ]

        output = (
            matches[0]
            if matches
            else
            tensor_items[0]
        )

    if not torch.is_tensor(output):
        raise RuntimeError(
            f"Unsupported model output type: {type(output)}"
        )

    if output.ndim != 3:
        raise RuntimeError(
            f"Expected 3D prediction tensor, got shape {tuple(output.shape)}"
        )

    # Handle [B,F,2].
    if (
        output.shape[1] == expected_future_steps
        and
        output.shape[2] == 2
    ):
        return output

    # Handle [B,2,F].
    if (
        output.shape[1] == 2
        and
        output.shape[2] == expected_future_steps
    ):
        return output.transpose(
            1,
            2
        )

    raise RuntimeError(
        "Prediction tensor shape does not match expected future horizon: "
        f"shape={tuple(output.shape)}, expected future={expected_future_steps}"
    )


def predict_model(
    model,
    X,
    future_steps,
    device
):
    preds = []

    with torch.no_grad():
        for start in range(
            0,
            len(X),
            BATCH_SIZE
        ):
            xb_np = np.asarray(
                X[
                    start:
                    start
                    +
                    BATCH_SIZE
                ],
                dtype=np.float32
            )

            xb = torch.from_numpy(
                xb_np
            ).to(
                device
            )

            out = model(
                xb
            )

            out = normalize_model_output(
                out,
                future_steps
            )

            preds.append(
                out.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False
                )
            )

    return np.concatenate(
        preds,
        axis=0
    )


def call_inverse_batch(
    fn,
    canonical,
    origin,
    angle,
    scale
):
    # The original project function uses:
    # inverse_transform_batch(canonical, origin, angle, scale)
    out = fn(
        canonical,
        origin,
        angle,
        scale
    )

    return np.asarray(
        out,
        dtype=np.float64
    )


def window_metrics(
    pred_abs,
    gt_abs
):
    err = np.linalg.norm(
        pred_abs
        -
        gt_abs,
        axis=2
    )

    ade_per_window = err.mean(
        axis=1
    )

    fde_per_window = err[
        :,
        -1
    ]

    return (
        err,
        ade_per_window,
        fde_per_window
    )


def track_macro_metric(
    meta,
    values
):
    q = meta[
        [
            "track_id"
        ]
    ].copy()

    q["value"] = np.asarray(
        values,
        dtype=float
    )

    per_track = (
        q.groupby(
            "track_id"
        )[
            "value"
        ]
        .mean()
    )

    return (
        float(
            per_track.mean()
        ),
        per_track
    )


def main():
    print(
        "\nSTEP 6D-K3A2-V3 — FIRST LOCKED BAMBI FROZEN-ENSEMBLE INFERENCE"
    )

    print(
        "Forecasting only. No conformal q90/q95. No Step-7 alarms.\n"
    )

    require(
        PRIMARY_EVAL
    )

    require(
        MODEL_LOCK
    )

    model_lock = json.loads(
        MODEL_LOCK.read_text(
            encoding="utf-8"
        )
    )

    if (
        model_lock.get(
            "status"
        )
        !=
        "LOCKED_BEFORE_ANY_BAMBI_MODEL_PREDICTIONS"
    ):
        raise RuntimeError(
            "Unexpected model-lock status."
        )

    # ------------------------------------------------------------
    # Verify every locked checkpoint hash before first inference.
    # ------------------------------------------------------------
    checkpoint_rows = []

    for rec in model_lock[
        "checkpoint_set"
    ]:
        p = Path(
            rec[
                "checkpoint"
            ]
        )

        require(
            p
        )

        actual = sha256_file(
            p
        )

        match = (
            actual
            ==
            rec[
                "sha256"
            ]
        )

        checkpoint_rows.append({
            "task":
                rec[
                    "task"
                ],

            "seed":
                int(
                    rec[
                        "seed"
                    ]
                ),

            "checkpoint":
                str(
                    p
                ),

            "expected_sha256":
                rec[
                    "sha256"
                ],

            "actual_sha256":
                actual,

            "match":
                bool(
                    match
                ),
        })

    checkpoint_df = pd.DataFrame(
        checkpoint_rows
    )

    if not checkpoint_df[
        "match"
    ].all():
        raise RuntimeError(
            "Frozen checkpoint hash mismatch. "
            "Inference aborted."
        )

    # ------------------------------------------------------------
    # Import original frozen evaluator.
    # ------------------------------------------------------------
    evaluator_ns = load_isolated_evaluator_namespace(
        PRIMARY_EVAL
    )

    load_transformer = evaluator_ns[
        "load_transformer"
    ]

    inverse_transform_batch = evaluator_ns[
        "inverse_transform_batch"
    ]

    print(
        "ISOLATED SOURCE LOAD: PASS"
    )

    print(
        "No top-level BIRDSAI benchmark code executed."
    )

    summary_by_task = {}
    loader_audit_rows = []

    for task, cfg in TASKS.items():
        print(
            f"\n--- {task} ---"
        )

        require(
            cfg[
                "npz"
            ]
        )

        require(
            cfg[
                "meta"
            ]
        )

        with np.load(
            cfg[
                "npz"
            ],
            allow_pickle=False
        ) as z:
            X = np.asarray(
                z[
                    "history_canonical"
                ],
                dtype=np.float32
            )

            Y_can = np.asarray(
                z[
                    "future_canonical"
                ],
                dtype=np.float32
            )

            origin = np.asarray(
                z[
                    "origin"
                ],
                dtype=np.float64
            )

            angle = np.asarray(
                z[
                    "angle"
                ],
                dtype=np.float64
            )

            scale = np.asarray(
                z[
                    "scale"
                ],
                dtype=np.float64
            )

            Y_abs = np.asarray(
                z[
                    "future_abs_xy"
                ],
                dtype=np.float64
            )

        meta = pd.read_csv(
            cfg[
                "meta"
            ]
        )

        if len(
            meta
        ) != len(
            X
        ):
            raise RuntimeError(
                f"{task}: metadata/window count mismatch."
            )

        seed_preds_can = []
        seed_metrics = {}

        task_checkpoint_records = [
            rec
            for rec in model_lock[
                "checkpoint_set"
            ]
            if rec[
                "task"
            ]
            ==
            task
        ]

        task_checkpoint_records = sorted(
            task_checkpoint_records,
            key=lambda r: int(
                r[
                    "seed"
                ]
            )
        )

        if [
            int(
                r[
                    "seed"
                ]
            )
            for r in task_checkpoint_records
        ] != [
            42,
            52,
            62
        ]:
            raise RuntimeError(
                f"{task}: locked seed set is not [42,52,62]."
            )

        for rec in task_checkpoint_records:
            seed = int(
                rec[
                    "seed"
                ]
            )

            ckpt = Path(
                rec[
                    "checkpoint"
                ]
            )

            print(
                f"Loading seed {seed}: {ckpt.name}"
            )

            model, loader_kwargs, loader_signature = call_load_transformer(
                load_transformer,
                ckpt,
                task,
                cfg[
                    "future_steps"
                ],
                seed,
                DEVICE
            )

            loader_audit_rows.append({
                "task":
                    task,

                "seed":
                    seed,

                "loader_signature":
                    loader_signature,

                "loader_kwargs":
                    json.dumps(
                        {
                            k: str(v)
                            for k, v in loader_kwargs.items()
                        },
                        sort_keys=True
                    ),
            })

            pred_can = predict_model(
                model,
                X,
                cfg[
                    "future_steps"
                ],
                DEVICE
            )

            if pred_can.shape != Y_can.shape:
                raise RuntimeError(
                    f"{task} seed{seed}: prediction shape "
                    f"{pred_can.shape} != target shape {Y_can.shape}"
                )

            pred_abs = call_inverse_batch(
                inverse_transform_batch,
                pred_can,
                origin,
                angle,
                scale
            )

            if pred_abs.shape != Y_abs.shape:
                raise RuntimeError(
                    f"{task} seed{seed}: inverse shape "
                    f"{pred_abs.shape} != GT shape {Y_abs.shape}"
                )

            _, ade_w, fde_w = window_metrics(
                pred_abs,
                Y_abs
            )

            ade_track_macro, _ = track_macro_metric(
                meta,
                ade_w
            )

            fde_track_macro, _ = track_macro_metric(
                meta,
                fde_w
            )

            seed_metrics[
                str(
                    seed
                )
            ] = {
                "window_ADE_m":
                    float(
                        ade_w.mean()
                    ),

                "window_FDE_m":
                    float(
                        fde_w.mean()
                    ),

                "track_macro_ADE_m":
                    ade_track_macro,

                "track_macro_FDE_m":
                    fde_track_macro,
            }

            np.save(
                OUT
                /
                f"{task}_seed{seed}_prediction_canonical.npy",
                pred_can
            )

            seed_preds_can.append(
                pred_can
            )

            # Free one model before loading next seed.
            del model

        # --------------------------------------------------------
        # Locked ensemble = arithmetic mean across seeds.
        # --------------------------------------------------------
        stack = np.stack(
            seed_preds_can,
            axis=0
        ).astype(
            np.float64
        )

        ensemble_can = stack.mean(
            axis=0
        ).astype(
            np.float32
        )

        ensemble_abs = call_inverse_batch(
            inverse_transform_batch,
            ensemble_can,
            origin,
            angle,
            scale
        )

        err_step, ade_w, fde_w = window_metrics(
            ensemble_abs,
            Y_abs
        )

        ade_track_macro, ade_per_track = track_macro_metric(
            meta,
            ade_w
        )

        fde_track_macro, fde_per_track = track_macro_metric(
            meta,
            fde_w
        )

        per_window = meta.copy()

        per_window[
            "ADE_m"
        ] = ade_w

        per_window[
            "FDE_m"
        ] = fde_w

        per_window[
            "max_radial_error_m"
        ] = err_step.max(
            axis=1
        )

        per_window[
            "endpoint_radial_error_m"
        ] = err_step[
            :,
            -1
        ]

        per_window.to_csv(
            OUT
            /
            f"{task}_ensemble_per_window_errors.csv",
            index=False
        )

        per_track = pd.DataFrame({
            "track_id":
                sorted(
                    set(
                        meta[
                            "track_id"
                        ]
                        .astype(
                            int
                        )
                        .tolist()
                    )
                )
        })

        per_track[
            "ADE_m"
        ] = per_track[
            "track_id"
        ].map(
            ade_per_track
        )

        per_track[
            "FDE_m"
        ] = per_track[
            "track_id"
        ].map(
            fde_per_track
        )

        # Explicit max-error track aggregation for the next conformal step.
        temp = per_window[
            [
                "track_id",
                "max_radial_error_m",
                "endpoint_radial_error_m",
            ]
        ].copy()

        max_by_track = (
            temp.groupby(
                "track_id"
            )[
                [
                    "max_radial_error_m",
                    "endpoint_radial_error_m",
                ]
            ]
            .max()
        )

        per_track[
            "track_max_of_window_max_radial_error_m"
        ] = per_track[
            "track_id"
        ].map(
            max_by_track[
                "max_radial_error_m"
            ]
        )

        per_track[
            "track_max_endpoint_radial_error_m"
        ] = per_track[
            "track_id"
        ].map(
            max_by_track[
                "endpoint_radial_error_m"
            ]
        )

        per_track.to_csv(
            OUT
            /
            f"{task}_ensemble_per_track_metrics.csv",
            index=False
        )

        np.savez_compressed(
            OUT
            /
            f"{task}_BAMBI_Flight6_first_locked_inference.npz",

            prediction_canonical=
                ensemble_can,

            prediction_abs_xy=
                ensemble_abs.astype(
                    np.float64
                ),

            ground_truth_abs_xy=
                Y_abs.astype(
                    np.float64
                ),

            radial_error_m=
                err_step.astype(
                    np.float64
                ),

            seed_prediction_canonical=
                stack.astype(
                    np.float32
                ),
        )

        summary_by_task[
            task
        ] = {
            "windows":
                int(
                    len(
                        X
                    )
                ),

            "tracks":
                int(
                    meta[
                        "track_id"
                    ].nunique()
                ),

            "seeds":
                [
                    42,
                    52,
                    62
                ],

            "ensemble_rule":
                "arithmetic mean of the three frozen canonical predictions",

            "window_ADE_m":
                float(
                    ade_w.mean()
                ),

            "window_FDE_m":
                float(
                    fde_w.mean()
                ),

            "track_macro_ADE_m":
                float(
                    ade_track_macro
                ),

            "track_macro_FDE_m":
                float(
                    fde_track_macro
                ),

            "median_window_ADE_m":
                float(
                    np.median(
                        ade_w
                    )
                ),

            "median_window_FDE_m":
                float(
                    np.median(
                        fde_w
                    )
                ),

            "p95_window_ADE_m":
                float(
                    np.quantile(
                        ade_w,
                        0.95
                    )
                ),

            "p95_window_FDE_m":
                float(
                    np.quantile(
                        fde_w,
                        0.95
                    )
                ),

            "max_window_max_radial_error_m":
                float(
                    err_step.max()
                ),

            "seed_metrics":
                seed_metrics,
        }

        print(
            json.dumps(
                summary_by_task[
                    task
                ],
                indent=2
            )
        )

    loader_audit = pd.DataFrame(
        loader_audit_rows
    )

    loader_audit.to_csv(
        OUT
        /
        "step6dk3a2_loader_call_audit.csv",
        index=False
    )

    output_hashes = []

    for p in sorted(
        OUT.glob(
            "*"
        )
    ):
        if (
            p.is_file()
            and
            p.name
            not in {
                "STEP6D_K3A2_OUTPUT_HASHES.csv",
                "STEP6D_K3A2_SUMMARY.json",
                "STEP6D_K3A2_SUMMARY.txt",
            }
        ):
            output_hashes.append({
                "file":
                    str(
                        p
                    ),

                "sha256":
                    sha256_file(
                        p
                    ),
            })

    output_hashes_df = pd.DataFrame(
        output_hashes
    )

    output_hashes_df.to_csv(
        OUT
        /
        "STEP6D_K3A2_OUTPUT_HASHES.csv",
        index=False
    )

    summary = {
        "step":
            "STEP_6D_K3A2_V3_FIRST_LOCKED_BAMBI_FROZEN_ENSEMBLE_INFERENCE",

        "device":
            DEVICE,

        "model_lock":
            str(
                MODEL_LOCK
            ),

        "model_lock_sha256":
            sha256_file(
                MODEL_LOCK
            ),

        "checkpoint_hashes_verified":
            True,

        "tasks":
            summary_by_task,

        "conformal_applied":
            False,

        "step7_applied":
            False,

        "status":
            "FIRST_LOCKED_BAMBI_FORECAST_INFERENCE_COMPLETE_READY_FOR_FIXED_CONFORMAL_EVALUATION",

        "important_interpretation": [
            "This is the first BAMBI forecast inference after canonical and model locks.",
            "No model retraining, checkpoint selection, hyperparameter tuning, or recalibration used BAMBI outcomes.",
            "ADE/FDE are reported in georeferenced world metres.",
            "BAMBI Flight 6 is a cross-species external robustness/stress test, not elephant-specific validation.",
            "Ensemble spread, if saved, is not a calibrated probability.",
            "No Step6C q90/q95 conformal coverage or Step-7 warning results are computed in K3A2.",
        ],
    }

    (
        OUT
        /
        "STEP6D_K3A2_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    with open(
        OUT
        /
        "STEP6D_K3A2_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K3A2-V3 — FIRST LOCKED BAMBI FROZEN-ENSEMBLE INFERENCE\n"
        )

        f.write(
            "=" * 88
            +
            "\n\n"
        )

        f.write(
            "TASK RESULTS\n"
        )

        f.write(
            "------------\n"
        )

        f.write(
            json.dumps(
                summary_by_task,
                indent=2
            )
        )

        f.write(
            "\n\nSTATUS\n"
        )

        f.write(
            "------\n"
        )

        f.write(
            summary[
                "status"
            ]
            +
            "\n"
        )

    print(
        "\nFINAL TASK RESULTS:"
    )

    print(
        json.dumps(
            summary_by_task,
            indent=2
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        summary[
            "status"
        ]
    )

    print(
        "\nSTEP 6D-K3A2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
