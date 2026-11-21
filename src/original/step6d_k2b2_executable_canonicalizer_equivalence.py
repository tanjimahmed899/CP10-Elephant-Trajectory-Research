from pathlib import Path
import ast
import inspect
import json
import math
import re
import numpy as np

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

NPZ_22 = K2A / "2s_to_2s_bambi_flight6_world_raw.npz"
NPZ_25 = K2A / "2s_to_5s_bambi_flight6_world_raw.npz"

OUT = BAMBI_DIR / "k2b_representation_audit" / "equivalence_test"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    (BUILD_SRC, "canonicalize_sample", "build_final"),
    (FRESH_SRC, "canonicalize_window", "fresh_holdout"),
]

ATOL_CANON = 2e-6
ATOL_META = 2e-6


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def top_level_maps(tree):
    funcs = {}
    imports = []
    simple_assigns = {}

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node

        elif isinstance(node, ast.Assign):
            # Only retain literal/simple numeric constants when needed.
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    value = ast.literal_eval(node.value)
                    simple_assigns[node.targets[0].id] = (node, value)
                except Exception:
                    pass

        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                value = ast.literal_eval(node.value)
                simple_assigns[node.target.id] = (node, value)
            except Exception:
                pass

    return funcs, imports, simple_assigns


def called_names(node):
    out = set()

    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                out.add(n.func.id)

    return out


def loaded_global_names(node):
    local = set()
    globals_used = set()

    # Function args are local.
    for a in node.args.args + node.args.kwonlyargs:
        local.add(a.arg)

    if node.args.vararg:
        local.add(node.args.vararg.arg)

    if node.args.kwarg:
        local.add(node.args.kwarg.arg)

    # Assigned names are local.
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            local.add(n.id)

    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in local:
                globals_used.add(n.id)

    return globals_used


def collect_function_closure(target_name, funcs):
    needed = set()
    stack = [target_name]

    while stack:
        name = stack.pop()

        if name in needed:
            continue

        if name not in funcs:
            continue

        needed.add(name)

        for c in called_names(funcs[name]):
            if c in funcs and c not in needed:
                stack.append(c)

    return needed


def build_isolated_namespace(path, target_name):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(text)

    funcs, imports, simple_assigns = top_level_maps(tree)

    if target_name not in funcs:
        raise RuntimeError(
            f"{target_name} not found in {path}"
        )

    needed_funcs = collect_function_closure(
        target_name,
        funcs
    )

    # Collect referenced simple constants recursively.
    needed_constants = set()

    for fname in needed_funcs:
        g = loaded_global_names(funcs[fname])

        for name in g:
            if name in simple_assigns:
                needed_constants.add(name)

    body = []

    # Imports are safe to execute and needed for np/math/etc.
    body.extend(imports)

    for name in sorted(needed_constants):
        body.append(
            simple_assigns[name][0]
        )

    # Preserve source order for helper defs.
    ordered_funcs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in needed_funcs
    ]

    body.extend(ordered_funcs)

    isolated = ast.Module(
        body=body,
        type_ignores=[]
    )

    ast.fix_missing_locations(isolated)

    ns = {}

    exec(
        compile(
            isolated,
            filename=str(path),
            mode="exec"
        ),
        ns,
        ns
    )

    fn = ns[target_name]

    return {
        "function": fn,
        "namespace": ns,
        "signature": inspect.signature(fn),
        "needed_functions": sorted(needed_funcs),
        "needed_constants": sorted(needed_constants),
        "source_text": text,
        "tree": tree,
    }


def resolve_scale_floor(info, target_name):
    sig = info["signature"]

    if "scale_floor" in sig.parameters:
        p = sig.parameters["scale_floor"]

        if p.default is not inspect._empty:
            try:
                return float(p.default), "function_default"
            except Exception:
                pass

    # Search call sites for a literal scale_floor keyword.
    text = info["source_text"]
    tree = info["tree"]

    literal_values = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        called = None

        if isinstance(node.func, ast.Name):
            called = node.func.id

        elif isinstance(node.func, ast.Attribute):
            called = node.func.attr

        if called != target_name:
            continue

        for kw in node.keywords:
            if kw.arg != "scale_floor":
                continue

            try:
                literal_values.append(
                    float(ast.literal_eval(kw.value))
                )
            except Exception:
                pass

    if literal_values:
        uniq = sorted(
            set(
                round(v, 15)
                for v in literal_values
            )
        )

        if len(uniq) == 1:
            return float(uniq[0]), "literal_callsite"

    return None, "unresolved"


def call_canonicalizer(fn, history, future, scale_floor):
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    kwargs = {}

    if "scale_floor" in sig.parameters:
        kwargs["scale_floor"] = scale_floor

    try:
        return fn(
            history,
            future,
            **kwargs
        )
    except TypeError:
        # Some implementations may use keyword-only names.
        names = list(sig.parameters.keys())

        if len(names) < 2:
            raise

        kw = {
            names[0]: history,
            names[1]: future,
        }

        if "scale_floor" in sig.parameters:
            kw["scale_floor"] = scale_floor

        return fn(**kw)


def extract_common(result):
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Canonicalizer returned {type(result)}, expected dict."
        )

    hist = result.get("history_canonical")
    fut = result.get("future_canonical")
    origin = result.get("origin")

    angle = None

    for key in [
        "angle",
        "heading_angle",
        "theta",
        "heading",
    ]:
        if key in result:
            angle = result[key]
            break

    scale = result.get("scale")

    path_length = None

    for key in [
        "raw_path_length",
        "history_path",
        "path_length",
    ]:
        if key in result:
            path_length = result[key]
            break

    if hist is None or fut is None or origin is None or angle is None or scale is None:
        raise RuntimeError(
            "Canonicalizer result missing required fields. "
            f"Keys={list(result.keys())}"
        )

    return {
        "history": np.asarray(hist, dtype=np.float64),
        "future": np.asarray(fut, dtype=np.float64),
        "origin": np.asarray(origin, dtype=np.float64),
        "angle": float(angle),
        "scale": float(scale),
        "path_length": (
            float(path_length)
            if path_length is not None
            else np.nan
        ),
        "keys": list(result.keys()),
    }


def angular_diff(a, b):
    return abs(
        (a - b + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )


def compare_pair(a, b):
    return {
        "history_max_abs_diff":
            float(
                np.max(
                    np.abs(
                        a["history"]
                        -
                        b["history"]
                    )
                )
            ),

        "future_max_abs_diff":
            float(
                np.max(
                    np.abs(
                        a["future"]
                        -
                        b["future"]
                    )
                )
            ),

        "origin_max_abs_diff":
            float(
                np.max(
                    np.abs(
                        a["origin"]
                        -
                        b["origin"]
                    )
                )
            ),

        "angle_periodic_abs_diff":
            float(
                angular_diff(
                    a["angle"],
                    b["angle"]
                )
            ),

        "scale_abs_diff":
            float(
                abs(
                    a["scale"]
                    -
                    b["scale"]
                )
            ),

        "path_length_abs_diff":
            (
                float(
                    abs(
                        a["path_length"]
                        -
                        b["path_length"]
                    )
                )
                if (
                    np.isfinite(
                        a["path_length"]
                    )
                    and
                    np.isfinite(
                        b["path_length"]
                    )
                )
                else np.nan
            ),
    }


def synthetic_cases():
    cases = []

    # Straight horizontal.
    h = np.column_stack([
        np.arange(20, dtype=float),
        np.zeros(20, dtype=float),
    ])
    f = np.column_stack([
        np.arange(20, 40, dtype=float),
        np.zeros(20, dtype=float),
    ])
    cases.append(("straight_x", h, f))

    # Straight diagonal.
    h = np.column_stack([
        np.arange(20, dtype=float),
        np.arange(20, dtype=float) * 0.5,
    ])
    f = np.column_stack([
        np.arange(20, 40, dtype=float),
        np.arange(20, 40, dtype=float) * 0.5,
    ])
    cases.append(("diagonal", h, f))

    # Curved.
    t = np.linspace(0, 1.9, 20)
    h = np.column_stack([
        3.0 * t,
        np.sin(t),
    ])
    tf = np.linspace(2.0, 3.9, 20)
    f = np.column_stack([
        3.0 * tf,
        np.sin(tf),
    ])
    cases.append(("curved", h, f))

    # Nearly stationary — exercises scale floor.
    h = np.column_stack([
        np.linspace(0, 1e-10, 20),
        np.linspace(0, -1e-10, 20),
    ])
    f = np.column_stack([
        np.linspace(1e-10, 2e-10, 20),
        np.linspace(-1e-10, -2e-10, 20),
    ])
    cases.append(("near_stationary", h, f))

    # Translation invariance case.
    base_h = np.column_stack([
        np.arange(20, dtype=float) * 0.3,
        np.arange(20, dtype=float) * -0.15,
    ]) + np.array([586950.0, 5338720.0])

    base_f = np.column_stack([
        np.arange(20, 40, dtype=float) * 0.3,
        np.arange(20, 40, dtype=float) * -0.15,
    ]) + np.array([586950.0, 5338720.0])

    cases.append(
        ("utm_scale_translation", base_h, base_f)
    )

    return cases


def evaluate_cases(
    info_a,
    info_b,
    scale_floor
):
    rows = []

    for name, h, f in synthetic_cases():
        ra = extract_common(
            call_canonicalizer(
                info_a["function"],
                h,
                f,
                scale_floor
            )
        )

        rb = extract_common(
            call_canonicalizer(
                info_b["function"],
                h,
                f,
                scale_floor
            )
        )

        d = compare_pair(
            ra,
            rb
        )

        rows.append({
            "source":
                "synthetic",

            "case":
                name,

            **d,
        })

    return rows


def evaluate_real_windows(
    info_a,
    info_b,
    scale_floor,
    npz_path,
    task
):
    z = np.load(
        npz_path,
        allow_pickle=False
    )

    X = z["X_abs_xy"]
    Y = z["Y_abs_xy"]

    n = len(X)

    # Deterministic coverage of beginning, middle, end + evenly spaced windows.
    idx = sorted(
        set(
            np.linspace(
                0,
                n - 1,
                min(n, 200),
                dtype=int
            ).tolist()
        )
    )

    rows = []

    for i in idx:
        h = np.asarray(
            X[i],
            dtype=np.float64
        )

        f = np.asarray(
            Y[i],
            dtype=np.float64
        )

        ra = extract_common(
            call_canonicalizer(
                info_a["function"],
                h,
                f,
                scale_floor
            )
        )

        rb = extract_common(
            call_canonicalizer(
                info_b["function"],
                h,
                f,
                scale_floor
            )
        )

        d = compare_pair(
            ra,
            rb
        )

        rows.append({
            "source":
                task,

            "case":
                f"sample_{i}",

            **d,
        })

    return rows


def main():
    print(
        "\nSTEP 6D-K2B2 — EXECUTABLE CANONICALIZER EQUIVALENCE TEST"
    )

    print(
        "Original sources remain read-only. No model evaluation.\n"
    )

    for p in [
        BUILD_SRC,
        FRESH_SRC,
        NPZ_22,
        NPZ_25,
    ]:
        require(p)

    loaded = {}

    for path, fname, label in TARGETS:
        info = build_isolated_namespace(
            path,
            fname
        )

        loaded[label] = info

        print(
            f"{label}: {path} :: {fname}"
        )

        print(
            f"  signature = {info['signature']}"
        )

        print(
            f"  isolated helper closure = "
            f"{info['needed_functions']}"
        )

        print(
            f"  simple constants = "
            f"{info['needed_constants']}"
        )

    floor_a, src_a = resolve_scale_floor(
        loaded[
            "build_final"
        ],
        "canonicalize_sample"
    )

    floor_b, src_b = resolve_scale_floor(
        loaded[
            "fresh_holdout"
        ],
        "canonicalize_window"
    )

    print(
        "\nSCALE FLOOR AUDIT:"
    )

    print(
        "  build_final:",
        floor_a,
        src_a
    )

    print(
        "  fresh_holdout:",
        floor_b,
        src_b
    )

    resolved = [
        v
        for v in [
            floor_a,
            floor_b,
        ]
        if v is not None
    ]

    if len(resolved) == 0:
        raise RuntimeError(
            "Could not resolve scale_floor from either source. "
            "Do not canonicalize BAMBI yet."
        )

    # If only one source resolves it, use it for both exact functions.
    scale_floor = float(
        resolved[0]
    )

    if len(resolved) == 2:
        if not np.isclose(
            resolved[0],
            resolved[1],
            rtol=0,
            atol=1e-15
        ):
            raise RuntimeError(
                f"Scale-floor mismatch: {resolved}"
            )

    rows = []

    rows.extend(
        evaluate_cases(
            loaded[
                "build_final"
            ],
            loaded[
                "fresh_holdout"
            ],
            scale_floor
        )
    )

    rows.extend(
        evaluate_real_windows(
            loaded[
                "build_final"
            ],
            loaded[
                "fresh_holdout"
            ],
            scale_floor,
            NPZ_22,
            "BAMBI_2s_to_2s"
        )
    )

    rows.extend(
        evaluate_real_windows(
            loaded[
                "build_final"
            ],
            loaded[
                "fresh_holdout"
            ],
            scale_floor,
            NPZ_25,
            "BAMBI_2s_to_5s"
        )
    )

    import pandas as pd

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUT
        /
        "TABLE_step6dk2b2_equivalence_differences.csv",
        index=False
    )

    maxes = {
        col:
            float(
                pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).max()
            )
        for col in [
            "history_max_abs_diff",
            "future_max_abs_diff",
            "origin_max_abs_diff",
            "angle_periodic_abs_diff",
            "scale_abs_diff",
            "path_length_abs_diff",
        ]
    }

    pass_numeric = (
        maxes[
            "history_max_abs_diff"
        ]
        <= ATOL_CANON
        and
        maxes[
            "future_max_abs_diff"
        ]
        <= ATOL_CANON
        and
        maxes[
            "origin_max_abs_diff"
        ]
        <= ATOL_META
        and
        maxes[
            "angle_periodic_abs_diff"
        ]
        <= ATOL_META
        and
        maxes[
            "scale_abs_diff"
        ]
        <= ATOL_META
    )

    status = (
        "CANONICALIZERS_NUMERICALLY_EQUIVALENT"
        if pass_numeric
        else
        "CANONICALIZER_MISMATCH_DO_NOT_PROCEED"
    )

    report = {
        "step":
            "STEP_6D_K2B2_EXECUTABLE_CANONICALIZER_EQUIVALENCE",

        "scale_floor_used":
            scale_floor,

        "scale_floor_build_source":
            src_a,

        "scale_floor_fresh_source":
            src_b,

        "tests":
            {
                "synthetic_cases":
                    len(
                        df[
                            df[
                                "source"
                            ]
                            ==
                            "synthetic"
                        ]
                    ),

                "bambi_2s_to_2s_samples":
                    len(
                        df[
                            df[
                                "source"
                            ]
                            ==
                            "BAMBI_2s_to_2s"
                        ]
                    ),

                "bambi_2s_to_5s_samples":
                    len(
                        df[
                            df[
                                "source"
                            ]
                            ==
                            "BAMBI_2s_to_5s"
                        ]
                    ),
            },

        "max_differences":
            maxes,

        "tolerances":
            {
                "canonical":
                    ATOL_CANON,

                "metadata":
                    ATOL_META,
            },

        "status":
            status,

        "next_if_pass":
            (
                "Use build_final_neural_dataset.py::canonicalize_sample "
                "as the frozen canonicalizer for BAMBI K2B3."
            ),
    }

    with open(
        OUT
        /
        "STEP6D_K2B2_SUMMARY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    with open(
        OUT
        /
        "STEP6D_K2B2_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K2B2 — EXECUTABLE CANONICALIZER EQUIVALENCE TEST\n"
        )

        f.write(
            "="
            *
            84
            +
            "\n\n"
        )

        f.write(
            "SCALE FLOOR\n"
        )

        f.write(
            "-----------\n"
        )

        f.write(
            f"scale_floor_used = {scale_floor}\n"
            f"build source = {src_a}\n"
            f"fresh source = {src_b}\n"
        )

        f.write(
            "\nMAX DIFFERENCES\n"
        )

        f.write(
            "---------------\n"
        )

        f.write(
            json.dumps(
                maxes,
                indent=2
            )
        )

        f.write(
            "\n\nTEST COUNTS\n"
        )

        f.write(
            "-----------\n"
        )

        f.write(
            json.dumps(
                report[
                    "tests"
                ],
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
            status
            +
            "\n"
        )

    print(
        "\nMAX DIFFERENCES:"
    )

    print(
        json.dumps(
            maxes,
            indent=2
        )
    )

    print(
        "\nTEST COUNTS:"
    )

    print(
        json.dumps(
            report[
                "tests"
            ],
            indent=2
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nSTEP 6D-K2B2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
