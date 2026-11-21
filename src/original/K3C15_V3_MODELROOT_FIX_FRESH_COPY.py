from pathlib import Path
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

MAIN_RESULT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

EVALUATOR_SOURCE = (
    MAIN_RESULT
    / "evaluate_birdsai_testreal_frozen_ensemble.py"
)

EXPECTED_EVALUATOR_SHA256 = (
    "b3002766925305adb2fd54339cda44188223aa85351c3c8299e2bbf1bab7ee66"
)

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

K3C13_DIR = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_canonical_lock_v1"
)

K3C13_MANIFEST = (
    K3C13_DIR
    / "STEP6D_K3C13_CANONICAL_DATASET_LOCK.json"
)

EXPECTED_K3C13_SHA256 = (
    "71d83c774233538342f4b410a789750943e61a8513fe983249c8f8f5a2aee605"
)

K3C11_DIR = (
    BASE
    / "k3_operational_warning"
    / "anchor_normalized_reconstruction_v1"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "secondary_frozen_inference_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [
    42,
    52,
    62,
]

TASKS = {
    "2s_to_2s": {
        "future_length":
            20,

        "expected_checkpoint_hashes": {
            42:
                "8244d6dc60be34d5c7f3dcbb1a50036152f8c5e96d69f3b3f5a93ac6e4a7f14f",

            52:
                "9264f2d81945601149cc2f6a970a2765942cb9a45e711c4d552434e9feb75eb5",

            62:
                "0257eb576a78a6a40da51d8bbf728df30d5d8b77cd6ca7e21be81fa0fd3cded2",
        },

        "checkpoint_paths": {
            42:
                MAIN_RESULT
                / "transformer_results"
                / "2s_to_2s_Transformer_seed42.pt",

            52:
                MAIN_RESULT
                / "transformer_results"
                / "2s_to_2s_Transformer_seed52.pt",

            62:
                MAIN_RESULT
                / "transformer_results"
                / "2s_to_2s_Transformer_seed62.pt",
        },
    },

    "2s_to_5s": {
        "future_length":
            50,

        "expected_checkpoint_hashes": {
            42:
                "846e1398eb8354e871ea80b7a90888f99d02ce10dac56f0787bf3205588bbb0a",

            52:
                "e8316f3db5a60e72d1fee28384726b5353ffaf8c68ffeec64abb2326b4098958",

            62:
                "44d1d6d0e15af6844667b44e2bf6a4cb574b90884df8250af138e7841ec82b3a",
        },

        "checkpoint_paths": {
            42:
                MAIN_RESULT
                / "transformer_results"
                / "2s_to_5s_Transformer_seed42.pt",

            52:
                MAIN_RESULT
                / "transformer_results"
                / "2s_to_5s_Transformer_seed52.pt",

            62:
                MAIN_RESULT
                / "transformer_results"
                / "2s_to_5s_Transformer_seed62.pt",
        },
    },
}

BATCH_SIZE = 128


def require(path):
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )


def sha256_file(
    path,
    chunk=1024 * 1024,
):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            block = f.read(
                chunk
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def verify_frozen_inputs():
    require(
        EVALUATOR_SOURCE
    )

    evaluator_hash = sha256_file(
        EVALUATOR_SOURCE
    )

    if evaluator_hash != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            "Frozen evaluator source hash mismatch. "
            f"Expected {EXPECTED_EVALUATOR_SHA256}, got {evaluator_hash}"
        )

    require(
        K3C13_MANIFEST
    )

    k3c13_hash = sha256_file(
        K3C13_MANIFEST
    )

    if k3c13_hash != EXPECTED_K3C13_SHA256:
        raise RuntimeError(
            "K3C13 dataset lock hash mismatch. "
            f"Expected {EXPECTED_K3C13_SHA256}, got {k3c13_hash}"
        )

    manifest = json.loads(
        K3C13_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get(
            "status"
        )
        !=
        "DERIVED_ANCHOR_NORMALIZED_CANONICAL_DATASET_LOCKED_BEFORE_SECONDARY_FROZEN_INFERENCE"
    ):
        raise RuntimeError(
            "K3C13 manifest is not in the expected pre-inference locked state."
        )

    checkpoint_rows = []

    for task, config in TASKS.items():
        for seed in SEEDS:
            path = config[
                "checkpoint_paths"
            ][
                seed
            ]

            expected = config[
                "expected_checkpoint_hashes"
            ][
                seed
            ]

            require(
                path
            )

            actual = sha256_file(
                path
            )

            if actual != expected:
                raise RuntimeError(
                    f"{task} seed {seed} checkpoint hash mismatch.\n"
                    f"Expected: {expected}\n"
                    f"Actual:   {actual}\n"
                    f"File: {path}"
                )

            checkpoint_rows.append({
                "task":
                    task,

                "seed":
                    seed,

                "checkpoint":
                    str(
                        path
                    ),

                "sha256":
                    actual,
            })

    return (
        evaluator_hash,
        k3c13_hash,
        pd.DataFrame(
            checkpoint_rows
        ),
    )



def _bound_names_from_node(node):
    """
    Return top-level names bound by a statement.
    """
    names = set()

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
        return names

    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        targets = []

    def collect_target(target):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                collect_target(elt)

    for target in targets:
        collect_target(target)

    return names


def _referenced_load_names(node):
    """
    Names loaded/referenced by a statement/function/class.
    """
    refs = set()

    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            refs.add(sub.id)

    return refs


def _imported_aliases(node):
    aliases = set()

    if isinstance(node, ast.Import):
        for alias in node.names:
            aliases.add(alias.asname or alias.name.split(".")[0])

    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*":
                continue
            aliases.add(alias.asname or alias.name)

    return aliases


def build_safe_frozen_loader_namespace():
    """
    Build a minimal executable namespace from the exact frozen evaluator source
    WITHOUT importing/executing its top-level benchmark.

    Dependency closure starts from load_transformer and recursively includes
    only the top-level functions/classes/assignments it references, plus imports.
    """
    source_text = EVALUATOR_SOURCE.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(
        source_text,
        filename=str(EVALUATOR_SOURCE),
    )

    # Index top-level statements by bound name.
    definitions = {}

    import_nodes = []
    import_alias_to_node = {}

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)

            for alias_name in _imported_aliases(node):
                import_alias_to_node[alias_name] = node

            continue

        for name in _bound_names_from_node(node):
            definitions[name] = node

    if "load_transformer" not in definitions:
        raise RuntimeError(
            "Exact evaluator source does not define load_transformer()."
        )

    # Recursive dependency closure.
    required_names = {"load_transformer"}
    queue = ["load_transformer"]
    selected_nodes = set()
    selected_import_nodes = set()

    builtin_names = set(dir(__builtins__))

    while queue:
        name = queue.pop()

        if name in definitions:
            node = definitions[name]

            if id(node) in selected_nodes:
                continue

            selected_nodes.add(id(node))

            refs = _referenced_load_names(node)

            # Function/class own bound argument/local names should not be treated
            # as top-level dependencies.
            local_names = set()

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in (
                    list(node.args.posonlyargs)
                    + list(node.args.args)
                    + list(node.args.kwonlyargs)
                ):
                    local_names.add(arg.arg)

                if node.args.vararg is not None:
                    local_names.add(node.args.vararg.arg)

                if node.args.kwarg is not None:
                    local_names.add(node.args.kwarg.arg)

                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and isinstance(
                        sub.ctx,
                        (ast.Store, ast.Param),
                    ):
                        local_names.add(sub.id)

            elif isinstance(node, ast.ClassDef):
                # Methods' local variables do not need top-level resolution.
                for sub in ast.walk(node):
                    if isinstance(sub, ast.FunctionDef):
                        for arg in (
                            list(sub.args.posonlyargs)
                            + list(sub.args.args)
                            + list(sub.args.kwonlyargs)
                        ):
                            local_names.add(arg.arg)

                        if sub.args.vararg is not None:
                            local_names.add(sub.args.vararg.arg)

                        if sub.args.kwarg is not None:
                            local_names.add(sub.args.kwarg.arg)

            refs -= local_names
            refs -= builtin_names

            for ref in refs:
                if ref in definitions and ref not in required_names:
                    required_names.add(ref)
                    queue.append(ref)

                elif ref in import_alias_to_node:
                    selected_import_nodes.add(
                        id(import_alias_to_node[ref])
                    )

        elif name in import_alias_to_node:
            selected_import_nodes.add(
                id(import_alias_to_node[name])
            )

    # Preserve original source order. Include:
    # 1) imports referenced by the closure
    # 2) selected definition/assignment nodes
    body = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if id(node) in selected_import_nodes:
                body.append(node)
            continue

        if id(node) in selected_nodes:
            body.append(node)

    module_ast = ast.Module(
        body=body,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module_ast
    )

    namespace = {
        "__name__":
            "_k3c15_safe_frozen_loader_",

        "__file__":
            str(EVALUATOR_SOURCE),

        "__package__":
            None,
    }

    # Make relative/local imports resolvable exactly from Main Result.
    if str(MAIN_RESULT) not in sys.path:
        sys.path.insert(
            0,
            str(MAIN_RESULT),
        )

    exec(
        compile(
            module_ast,
            str(EVALUATOR_SOURCE),
            "exec",
        ),
        namespace,
        namespace,
    )

    if "load_transformer" not in namespace:
        raise RuntimeError(
            "Safe AST extraction did not produce load_transformer()."
        )

    # ------------------------------------------------------------------
    # Bind the exact loader's MODEL_ROOT to the VERIFIED frozen checkpoint
    # directory. The original evaluator defines MODEL_ROOT relative to the
    # user's Downloads directory; that path is not where the six locked
    # checkpoints actually live in this project.
    #
    # This changes no loader code and no checkpoint bytes. It only supplies
    # the already-proven K3C14 checkpoint directory to the extracted loader
    # namespace before any model is loaded.
    # ------------------------------------------------------------------
    verified_model_root = MAIN_RESULT / "transformer_results"

    if not verified_model_root.exists():
        raise FileNotFoundError(
            f"Verified frozen model root does not exist: {verified_model_root}"
        )

    namespace["MODEL_ROOT"] = verified_model_root

    # The function resolves globals through its __globals__ dictionary, which
    # is the same namespace object. Assert that the binding took effect.
    loader_globals = namespace["load_transformer"].__globals__

    if Path(loader_globals["MODEL_ROOT"]) != verified_model_root:
        raise RuntimeError(
            "Failed to bind load_transformer MODEL_ROOT to the verified "
            f"checkpoint directory: {verified_model_root}"
        )

    # Provenance audit of what was actually selected.
    selected_bound_names = sorted(
        name
        for name, node in definitions.items()
        if id(node) in selected_nodes
    )

    selected_imports = []

    for node in tree.body:
        if (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and
            id(node) in selected_import_nodes
        ):
            try:
                selected_imports.append(
                    ast.unparse(node)
                )
            except Exception:
                selected_imports.append(
                    f"<import line {getattr(node, 'lineno', '?')}>"
                )

    return namespace, {
        "selected_bound_names":
            selected_bound_names,

        "selected_imports":
            selected_imports,

        "top_level_benchmark_executed":
            False,

        "model_root_binding":
            str(
                verified_model_root
            ),

        "model_root_binding_mode":
            "runtime namespace binding to K3C14-verified frozen checkpoint directory; no source/checkpoint modification",
    }


def import_frozen_evaluator():
    """
    Return a minimal namespace object exposing the exact original
    load_transformer() without importing/executing the evaluator's top-level
    BIRDSAI benchmark.
    """
    namespace, extraction_audit = build_safe_frozen_loader_namespace()

    loader = namespace[
        "load_transformer"
    ]

    sig = inspect.signature(
        loader
    )

    if list(
        sig.parameters.keys()
    ) != [
        "task",
        "seed",
        "future_length",
    ]:
        raise RuntimeError(
            f"Unexpected load_transformer signature: {sig}"
        )

    class LoaderNamespace:
        pass

    evaluator = LoaderNamespace()

    evaluator.load_transformer = loader

    evaluator._safe_extraction_audit = extraction_audit

    return (
        evaluator,
        sig,
    )

def extract_torch_module(
    loaded,
    torch,
):
    if isinstance(
        loaded,
        torch.nn.Module,
    ):
        return loaded

    if isinstance(
        loaded,
        (
            tuple,
            list,
        ),
    ):
        modules = [
            item
            for item in loaded
            if isinstance(
                item,
                torch.nn.Module,
            )
        ]

        if len(
            modules
        ) == 1:
            return modules[
                0
            ]

    if isinstance(
        loaded,
        dict,
    ):
        modules = [
            value
            for value in loaded.values()
            if isinstance(
                value,
                torch.nn.Module,
            )
        ]

        if len(
            modules
        ) == 1:
            return modules[
                0
            ]

    raise RuntimeError(
        "load_transformer() return value does not contain exactly one torch.nn.Module. "
        f"Type={type(loaded)}"
    )


def extract_prediction_tensor(
    output,
    torch,
):
    if torch.is_tensor(
        output
    ):
        return output

    if isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):
        tensors = [
            x
            for x in output
            if torch.is_tensor(
                x
            )
        ]

        if len(
            tensors
        ) == 1:
            return tensors[
                0
            ]

    if isinstance(
        output,
        dict,
    ):
        preferred_keys = [
            "prediction",
            "predictions",
            "pred",
            "output",
            "trajectory",
            "future",
        ]

        for key in preferred_keys:
            value = output.get(
                key
            )

            if torch.is_tensor(
                value
            ):
                return value

        tensors = [
            value
            for value in output.values()
            if torch.is_tensor(
                value
            )
        ]

        if len(
            tensors
        ) == 1:
            return tensors[
                0
            ]

    raise RuntimeError(
        "Could not uniquely extract prediction tensor from model output. "
        f"Output type={type(output)}"
    )


def load_locked_task_data(
    task,
    future_length,
):
    canonical_npz = (
        K3C13_DIR
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_canonical_LOCKED.npz"
    )

    canonical_metadata = (
        K3C13_DIR
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_canonical_metadata_LOCKED.csv"
    )

    source_anchor_npz = (
        K3C11_DIR
        /
        f"{task}_BAMBI_Flight6_anchor_normalized_windows.npz"
    )

    for p in [
        canonical_npz,
        canonical_metadata,
        source_anchor_npz,
    ]:
        require(
            p
        )

    arrays = np.load(
        canonical_npz,
        allow_pickle=False,
    )

    required = {
        "history_canonical",
        "future_canonical",
        "origin",
        "angle",
        "scale",
        "raw_path_length",
    }

    missing = (
        required
        -
        set(
            arrays.files
        )
    )

    if missing:
        raise RuntimeError(
            f"{task}: locked canonical NPZ missing arrays {sorted(missing)}"
        )

    history = np.asarray(
        arrays[
            "history_canonical"
        ],
        dtype=np.float32,
    )

    future = np.asarray(
        arrays[
            "future_canonical"
        ],
        dtype=np.float32,
    )

    origin = np.asarray(
        arrays[
            "origin"
        ],
        dtype=np.float32,
    )

    angle = np.asarray(
        arrays[
            "angle"
        ],
        dtype=np.float32,
    )

    scale = np.asarray(
        arrays[
            "scale"
        ],
        dtype=np.float32,
    )

    metadata = pd.read_csv(
        canonical_metadata
    )

    anchor_arrays = np.load(
        source_anchor_npz,
        allow_pickle=False,
    )

    history_anchor = np.asarray(
        anchor_arrays[
            "history_anchor_norm"
        ],
        dtype=np.float64,
    )

    future_anchor = np.asarray(
        anchor_arrays[
            "future_anchor_norm"
        ],
        dtype=np.float64,
    )

    n = history.shape[
        0
    ]

    if history.shape != (
        n,
        20,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected history shape {history.shape}"
        )

    if future.shape != (
        n,
        future_length,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected future shape {future.shape}"
        )

    if origin.shape != (
        n,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected origin shape {origin.shape}"
        )

    if angle.shape != (
        n,
    ):
        raise RuntimeError(
            f"{task}: unexpected angle shape {angle.shape}"
        )

    if scale.shape != (
        n,
    ):
        raise RuntimeError(
            f"{task}: unexpected scale shape {scale.shape}"
        )

    if history_anchor.shape != (
        n,
        20,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected anchor history shape {history_anchor.shape}"
        )

    if future_anchor.shape != (
        n,
        future_length,
        2,
    ):
        raise RuntimeError(
            f"{task}: unexpected anchor future shape {future_anchor.shape}"
        )

    if len(
        metadata
    ) != n:
        raise RuntimeError(
            f"{task}: metadata count {len(metadata)} != sample count {n}"
        )

    if not all(
        np.isfinite(
            arr
        ).all()
        for arr in [
            history,
            future,
            origin,
            angle,
            scale,
            history_anchor,
            future_anchor,
        ]
    ):
        raise RuntimeError(
            f"{task}: nonfinite values found in locked input arrays."
        )

    return {
        "canonical_npz":
            canonical_npz,

        "canonical_metadata":
            canonical_metadata,

        "anchor_npz":
            source_anchor_npz,

        "history":
            history,

        "future":
            future,

        "origin":
            origin,

        "angle":
            angle,

        "scale":
            scale,

        "metadata":
            metadata,

        "history_anchor":
            history_anchor,

        "future_anchor":
            future_anchor,
    }


def inverse_candidates(
    canonical,
    origin,
    angle,
    scale,
):
    canonical = np.asarray(
        canonical,
        dtype=np.float64,
    )

    origin = np.asarray(
        origin,
        dtype=np.float64,
    )

    angle = np.asarray(
        angle,
        dtype=np.float64,
    )

    scale = np.asarray(
        scale,
        dtype=np.float64,
    )

    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )

    # Candidate A:
    # R = [[cos, -sin], [sin, cos]]
    RA = np.stack(
        [
            np.stack(
                [
                    c,
                    -s,
                ],
                axis=1,
            ),
            np.stack(
                [
                    s,
                    c,
                ],
                axis=1,
            ),
        ],
        axis=1,
    )

    # Candidate B:
    # R = [[cos, sin], [-sin, cos]]
    RB = np.stack(
        [
            np.stack(
                [
                    c,
                    s,
                ],
                axis=1,
            ),
            np.stack(
                [
                    -s,
                    c,
                ],
                axis=1,
            ),
        ],
        axis=1,
    )

    scaled = (
        canonical
        *
        scale[
            :,
            None,
            None,
        ]
    )

    outA = (
        np.einsum(
            "nhi,nij->nhj",
            scaled,
            RA,
        )
        +
        origin[
            :,
            None,
            :,
        ]
    )

    outB = (
        np.einsum(
            "nhi,nij->nhj",
            scaled,
            RB,
        )
        +
        origin[
            :,
            None,
            :,
        ]
    )

    return {
        "R_standard":
            outA,

        "R_transposed_sign":
            outB,
    }


def prove_inverse_convention(
    data,
):
    candidates = inverse_candidates(
        data[
            "future"
        ],
        data[
            "origin"
        ],
        data[
            "angle"
        ],
        data[
            "scale"
        ],
    )

    rows = []

    for name, reconstructed in candidates.items():
        diff = np.linalg.norm(
            reconstructed
            -
            data[
                "future_anchor"
            ],
            axis=2,
        )

        rows.append({
            "convention":
                name,

            "median_error":
                float(
                    np.median(
                        diff
                    )
                ),

            "p95_error":
                float(
                    np.quantile(
                        diff,
                        0.95
                    )
                ),

            "max_error":
                float(
                    np.max(
                        diff
                    )
                ),
        })

    df = pd.DataFrame(
        rows
    ).sort_values(
        [
            "max_error",
            "p95_error",
            "median_error",
        ]
    ).reset_index(
        drop=True
    )

    best = df.iloc[
        0
    ]

    if float(
        best[
            "max_error"
        ]
    ) > 5e-6:
        raise RuntimeError(
            "Could not prove exact canonical->anchor inverse convention. "
            f"Best max error={best['max_error']}"
        )

    if len(
        df
    ) > 1:
        second = df.iloc[
            1
        ]

        # Require the correct convention to be decisively exact.
        if (
            float(
                second[
                    "max_error"
                ]
            )
            <=
            5e-6
            and
            str(
                second[
                    "convention"
                ]
            )
            !=
            str(
                best[
                    "convention"
                ]
            )
        ):
            raise RuntimeError(
                "Inverse convention is numerically ambiguous."
            )

    return (
        str(
            best[
                "convention"
            ]
        ),
        df,
    )


def apply_inverse(
    canonical,
    origin,
    angle,
    scale,
    convention,
):
    candidates = inverse_candidates(
        canonical,
        origin,
        angle,
        scale,
    )

    if convention not in candidates:
        raise RuntimeError(
            f"Unknown inverse convention {convention}"
        )

    return candidates[
        convention
    ]


def forward_in_batches(
    model,
    history,
    future_length,
    torch,
):
    model.eval()

    try:
        device = next(
            model.parameters()
        ).device
    except StopIteration:
        device = torch.device(
            "cpu"
        )

    outputs = []

    with torch.no_grad():
        for start in range(
            0,
            len(
                history
            ),
            BATCH_SIZE,
        ):
            end = min(
                start
                +
                BATCH_SIZE,
                len(
                    history
                ),
            )

            x = torch.as_tensor(
                history[
                    start:end
                ],
                dtype=torch.float32,
                device=device,
            )

            raw_output = model(
                x
            )

            pred = extract_prediction_tensor(
                raw_output,
                torch,
            )

            pred = pred.detach().to(
                "cpu"
            )

            arr = pred.numpy()

            # Accept exact [B,H,2] or [H,B,2] only.
            if arr.shape == (
                end
                -
                start,
                future_length,
                2,
            ):
                pass

            elif arr.shape == (
                future_length,
                end
                -
                start,
                2,
            ):
                arr = np.transpose(
                    arr,
                    (
                        1,
                        0,
                        2,
                    ),
                )

            else:
                raise RuntimeError(
                    "Unexpected Transformer output shape "
                    f"{arr.shape}; expected {(end-start, future_length, 2)} "
                    f"or {(future_length, end-start, 2)}"
                )

            outputs.append(
                arr.astype(
                    np.float32,
                    copy=False,
                )
            )

    prediction = np.concatenate(
        outputs,
        axis=0,
    )

    if prediction.shape != (
        len(
            history
        ),
        future_length,
        2,
    ):
        raise RuntimeError(
            f"Final prediction shape mismatch: {prediction.shape}"
        )

    if not np.isfinite(
        prediction
    ).all():
        raise RuntimeError(
            "Nonfinite values found in Transformer predictions."
        )

    return prediction


def task_metrics(
    prediction_anchor,
    gt_anchor,
    metadata,
):
    radial = np.linalg.norm(
        prediction_anchor
        -
        gt_anchor,
        axis=2,
    )

    ade_per_window = radial.mean(
        axis=1
    )

    fde_per_window = radial[
        :,
        -1
    ]

    metrics = {
        "ade_instance_mean":
            float(
                ade_per_window.mean()
            ),

        "fde_instance_mean":
            float(
                fde_per_window.mean()
            ),

        "ade_instance_median":
            float(
                np.median(
                    ade_per_window
                )
            ),

        "fde_instance_median":
            float(
                np.median(
                    fde_per_window
                )
            ),

        "ade_p95":
            float(
                np.quantile(
                    ade_per_window,
                    0.95,
                )
            ),

        "fde_p95":
            float(
                np.quantile(
                    fde_per_window,
                    0.95,
                )
            ),

        "max_radial_error":
            float(
                radial.max()
            ),
    }

    if "track_id" in metadata.columns:
        temp = pd.DataFrame({
            "track_id":
                metadata[
                    "track_id"
                ].to_numpy(),

            "ade":
                ade_per_window,

            "fde":
                fde_per_window,
        })

        track = (
            temp
            .groupby(
                "track_id",
                as_index=False,
            )
            .agg(
                ade=(
                    "ade",
                    "mean"
                ),

                fde=(
                    "fde",
                    "mean"
                ),
            )
        )

        metrics[
            "ade_track_macro"
        ] = float(
            track[
                "ade"
            ].mean()
        )

        metrics[
            "fde_track_macro"
        ] = float(
            track[
                "fde"
            ].mean()
        )

    return (
        metrics,
        radial,
    )


def run_task(
    task,
    config,
    evaluator,
    torch,
):
    future_length = config[
        "future_length"
    ]

    data = load_locked_task_data(
        task,
        future_length,
    )

    inverse_convention, inverse_audit = prove_inverse_convention(
        data
    )

    inverse_audit_path = (
        OUT
        /
        f"TABLE_{task}_inverse_convention_proof.csv"
    )

    inverse_audit.to_csv(
        inverse_audit_path,
        index=False,
    )

    seed_predictions = []

    seed_records = []

    model_forward_signatures = []

    for seed in SEEDS:
        loaded = evaluator.load_transformer(
            task,
            seed,
            future_length,
        )

        model = extract_torch_module(
            loaded,
            torch,
        )

        model_forward_signatures.append({
            "seed":
                seed,

            "model_type":
                f"{type(model).__module__}.{type(model).__name__}",

            "forward_signature":
                str(
                    inspect.signature(
                        model.forward
                    )
                ),
        })

        prediction = forward_in_batches(
            model,
            data[
                "history"
            ],
            future_length,
            torch,
        )

        seed_predictions.append(
            prediction
        )

        seed_records.append({
            "seed":
                seed,

            "checkpoint":
                str(
                    config[
                        "checkpoint_paths"
                    ][
                        seed
                    ]
                ),

            "checkpoint_sha256":
                config[
                    "expected_checkpoint_hashes"
                ][
                    seed
                ],

            "prediction_canonical_abs_max":
                float(
                    np.abs(
                        prediction
                    ).max()
                ),
        })

    seed_prediction_canonical = np.stack(
        seed_predictions,
        axis=0,
    )

    ensemble_prediction_canonical = seed_prediction_canonical.mean(
        axis=0
    ).astype(
        np.float32
    )

    prediction_anchor = apply_inverse(
        ensemble_prediction_canonical,
        data[
            "origin"
        ],
        data[
            "angle"
        ],
        data[
            "scale"
        ],
        inverse_convention,
    ).astype(
        np.float64
    )

    gt_anchor = data[
        "future_anchor"
    ].astype(
        np.float64
    )

    metrics, radial_error = task_metrics(
        prediction_anchor,
        gt_anchor,
        data[
            "metadata"
        ],
    )

    # Ensemble-spread diagnostic only; not calibrated probability.
    seed_anchor = np.stack(
        [
            apply_inverse(
                seed_prediction_canonical[
                    idx
                ],
                data[
                    "origin"
                ],
                data[
                    "angle"
                ],
                data[
                    "scale"
                ],
                inverse_convention,
            )
            for idx in range(
                len(
                    SEEDS
                )
            )
        ],
        axis=0,
    ).astype(
        np.float64
    )

    ensemble_spread = np.linalg.norm(
        seed_anchor
        -
        prediction_anchor[
            None,
            ...,
        ],
        axis=3,
    ).std(
        axis=0
    )

    out_npz = (
        OUT
        /
        f"{task}_BAMBI_Flight6_secondary_frozen_ensemble_predictions_LOCKED.npz"
    )

    np.savez_compressed(
        out_npz,
        prediction_canonical=ensemble_prediction_canonical,
        prediction_anchor_norm=prediction_anchor,
        ground_truth_anchor_norm=gt_anchor,
        radial_error_anchor_norm=radial_error,
        seed_prediction_canonical=seed_prediction_canonical,
        seed_prediction_anchor_norm=seed_anchor,
        ensemble_spread_anchor_norm=ensemble_spread,
    )

    metadata_out = (
        OUT
        /
        f"{task}_BAMBI_Flight6_secondary_frozen_inference_metadata_LOCKED.csv"
    )

    data[
        "metadata"
    ].to_csv(
        metadata_out,
        index=False,
    )

    summary = {
        "task":
            task,

        "samples":
            int(
                len(
                    data[
                        "history"
                    ]
                )
            ),

        "tracks":
            int(
                data[
                    "metadata"
                ][
                    "track_id"
                ].nunique()
            )
            if
            "track_id"
            in
            data[
                "metadata"
            ].columns
            else
            None,

        "future_length":
            int(
                future_length
            ),

        "seeds":
            SEEDS,

        "inverse_convention":
            inverse_convention,

        "inverse_proof_csv":
            str(
                inverse_audit_path
            ),

        "model_forward_signatures":
            model_forward_signatures,

        "seed_records":
            seed_records,

        "metrics_anchor_normalized":
            metrics,

        "ensemble_spread_note":
            "Diagnostic spread across the three frozen seeds; not a calibrated probability.",

        "prediction_npz":
            str(
                out_npz
            ),

        "prediction_npz_sha256":
            sha256_file(
                out_npz
            ),

        "metadata_csv":
            str(
                metadata_out
            ),

        "metadata_csv_sha256":
            sha256_file(
                metadata_out
            ),

        "locked_input_npz":
            str(
                data[
                    "canonical_npz"
                ]
            ),

        "locked_input_npz_sha256":
            sha256_file(
                data[
                    "canonical_npz"
                ]
            ),
    }

    return summary


def main():
    print(
        "\nSTEP 6D-K3C15-V3 — SECONDARY FROZEN 3-SEED TRANSFORMER INFERENCE"
    )

    print(
        "Uses the K3C13-locked Step18-compatible BAMBI canonical representation."
    )

    print(
        "NO model retraining, NO model substitution, NO q recalibration, "
        "NO geofence metric, and NO Step7 alarm is performed.\n"
    )

    (
        evaluator_hash,
        k3c13_hash,
        checkpoint_df,
    ) = verify_frozen_inputs()

    print(
        "VERIFIED EVALUATOR SHA256:"
    )
    print(
        evaluator_hash
    )

    print(
        "\nVERIFIED K3C13 MANIFEST SHA256:"
    )
    print(
        k3c13_hash
    )

    print(
        "\nVERIFIED CHECKPOINTS:"
    )
    print(
        checkpoint_df.to_string(
            index=False
        )
    )

    evaluator, loader_signature = import_frozen_evaluator()

    import torch

    print(
        "\nTORCH VERSION:"
    )
    print(
        torch.__version__
    )

    print(
        "\nload_transformer SIGNATURE:"
    )
    print(
        loader_signature
    )

    print(
        "\nSAFE EVALUATOR EXTRACTION AUDIT:"
    )
    print(
        json.dumps(
            evaluator._safe_extraction_audit,
            indent=2,
        )
    )

    task_summaries = []

    for task, config in TASKS.items():
        print(
            f"\nRUNNING {task} — frozen seeds {SEEDS}"
        )

        summary = run_task(
            task,
            config,
            evaluator,
            torch,
        )

        task_summaries.append(
            summary
        )

        print(
            f"\n{task} SUMMARY:"
        )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

    manifest = {
        "step":
            "STEP_6D_K3C15_V2_SECONDARY_FROZEN_3SEED_TRANSFORMER_INFERENCE",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "SECONDARY_FROZEN_INFERENCE_COMPLETE_PREDICTIONS_LOCKED_NO_STEP7_YET",

        "analysis_role":
            (
                "Secondary/exploratory frozen operational-transfer analysis on the "
                "K3C13 Step18-compatible BAMBI representation. "
                "Not a new pristine confirmatory uncertainty holdout."
            ),

        "evaluator_source":
            str(
                EVALUATOR_SOURCE
            ),

        "evaluator_sha256":
            evaluator_hash,

        "load_transformer_signature":
            str(
                loader_signature
            ),

        "safe_evaluator_extraction_audit":
            evaluator._safe_extraction_audit,

        "k3c13_manifest":
            str(
                K3C13_MANIFEST
            ),

        "k3c13_manifest_sha256":
            k3c13_hash,

        "checkpoint_mapping":
            checkpoint_df.to_dict(
                orient="records"
            ),

        "tasks":
            task_summaries,

        "immutable_constraints_respected": {
            "model_retraining":
                False,

            "model_substitution":
                False,

            "bambi_q_recalibration":
                False,

            "bambi_threshold_tuning":
                False,

            "step7_metrics_generated":
                False,
        },

        "nonclaims": [
            "This is not a new untouched confirmatory uncertainty holdout.",
            "No conformal coverage result is produced in K3C15.",
            "No Step7 geofence or alarm result is produced in K3C15.",
            "Ensemble spread is not interpreted as calibrated probability.",
        ],
    }

    manifest_path = (
        OUT
        /
        "STEP6D_K3C15_SECONDARY_FROZEN_INFERENCE_LOCK.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nMANIFEST:"
    )
    print(
        manifest_path
    )

    print(
        "MANIFEST SHA256:"
    )
    print(
        sha256_file(
            manifest_path
        )
    )

    print(
        "\nSTATUS:"
    )
    print(
        "SECONDARY_FROZEN_INFERENCE_COMPLETE_PREDICTIONS_LOCKED_NO_STEP7_YET"
    )

    print(
        "\nSTEP 6D-K3C15 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
