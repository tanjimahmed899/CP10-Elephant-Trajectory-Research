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

NPZ_22 = K2A / "2s_to_2s_bambi_flight6_world_raw.npz"
NPZ_25 = K2A / "2s_to_5s_bambi_flight6_world_raw.npz"

OUT = BAMBI_DIR / "k2b_representation_audit" / "equivalence_test_v2"
OUT.mkdir(parents=True, exist_ok=True)

ATOL_CANON = 2e-6
ATOL_META = 2e-6

# These are ONLY equivalence probes, not the final pipeline value.
PROBE_SCALE_FLOORS = [
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
]

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

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                assigns[node.target.id] = node

    return funcs, imports, assigns


def called_names(node):
    out = set()

    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
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
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(text)

    funcs, imports, assigns = top_level_maps(tree)

    if target_name not in funcs:
        raise RuntimeError(
            f"{target_name} not found in {path}"
        )

    needed_funcs = collect_function_closure(
        target_name,
        funcs
    )

    needed_globals = set()

    for fname in needed_funcs:
        needed_globals |= loaded_global_names(
            funcs[fname]
        )

    body = []

    body.extend(imports)

    # Add simple/literal top-level assignments referenced by the target closure.
    for node in tree.body:
        include = False

        if isinstance(node, ast.Assign):
            names = [
                t.id
                for t in node.targets
                if isinstance(t, ast.Name)
            ]
            include = any(
                n in needed_globals
                for n in names
            )

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                include = (
                    node.target.id
                    in needed_globals
                )

        if include:
            try:
                # Include only assignments that are safe to execute in isolation.
                ast.literal_eval(
                    node.value
                )
                body.append(node)
            except Exception:
                pass

    for node in tree.body:
        if (
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and
            node.name in needed_funcs
        ):
            body.append(node)

    module = ast.Module(
        body=body,
        type_ignores=[]
    )

    ast.fix_missing_locations(module)

    ns = {}

    exec(
        compile(
            module,
            filename=str(path),
            mode="exec"
        ),
        ns,
        ns
    )

    return {
        "function": ns[target_name],
        "signature": inspect.signature(
            ns[target_name]
        ),
        "source_text": text,
        "tree": tree,
        "funcs": funcs,
    }


def call_canonicalizer(fn, history, future, scale_floor):
    sig = inspect.signature(fn)

    kwargs = {}

    if "scale_floor" in sig.parameters:
        kwargs["scale_floor"] = float(scale_floor)

    try:
        return fn(
            history,
            future,
            **kwargs
        )

    except TypeError:
        names = list(
            sig.parameters.keys()
        )

        kw = {
            names[0]: history,
            names[1]: future,
        }

        if "scale_floor" in sig.parameters:
            kw["scale_floor"] = float(
                scale_floor
            )

        return fn(
            **kw
        )


def extract_common(result):
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected return type: {type(result)}"
        )

    hist = result.get(
        "history_canonical"
    )

    fut = result.get(
        "future_canonical"
    )

    origin = result.get(
        "origin"
    )

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

    scale = result.get(
        "scale"
    )

    path_length = None

    for key in [
        "raw_path_length",
        "history_path",
        "path_length",
    ]:
        if key in result:
            path_length = result[key]
            break

    required = [
        hist,
        fut,
        origin,
        angle,
        scale,
    ]

    if any(
        x is None
        for x in required
    ):
        raise RuntimeError(
            f"Missing required return fields. Keys={list(result.keys())}"
        )

    return {
        "history":
            np.asarray(
                hist,
                dtype=np.float64
            ),

        "future":
            np.asarray(
                fut,
                dtype=np.float64
            ),

        "origin":
            np.asarray(
                origin,
                dtype=np.float64
            ),

        "angle":
            float(
                angle
            ),

        "scale":
            float(
                scale
            ),

        "path_length":
            (
                float(
                    path_length
                )
                if path_length is not None
                else np.nan
            ),
    }


def angle_diff(a, b):
    return abs(
        (
            a
            -
            b
            +
            math.pi
        )
        %
        (
            2.0
            *
            math.pi
        )
        -
        math.pi
    )


def compare(a, b):
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
                angle_diff(
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

    h = np.column_stack([
        np.arange(
            20,
            dtype=float
        ),
        np.zeros(
            20,
            dtype=float
        ),
    ])

    f = np.column_stack([
        np.arange(
            20,
            40,
            dtype=float
        ),
        np.zeros(
            20,
            dtype=float
        ),
    ])

    cases.append(
        (
            "straight_x",
            h,
            f
        )
    )

    h = np.column_stack([
        np.arange(
            20,
            dtype=float
        ),
        np.arange(
            20,
            dtype=float
        )
        *
        0.5,
    ])

    f = np.column_stack([
        np.arange(
            20,
            40,
            dtype=float
        ),
        np.arange(
            20,
            40,
            dtype=float
        )
        *
        0.5,
    ])

    cases.append(
        (
            "diagonal",
            h,
            f
        )
    )

    t = np.linspace(
        0,
        1.9,
        20
    )

    h = np.column_stack([
        3.0
        *
        t,
        np.sin(
            t
        ),
    ])

    tf = np.linspace(
        2.0,
        3.9,
        20
    )

    f = np.column_stack([
        3.0
        *
        tf,
        np.sin(
            tf
        ),
    ])

    cases.append(
        (
            "curved",
            h,
            f
        )
    )

    # Scale-floor sensitive.
    h = np.column_stack([
        np.linspace(
            0,
            1e-10,
            20
        ),
        np.linspace(
            0,
            -1e-10,
            20
        ),
    ])

    f = np.column_stack([
        np.linspace(
            1e-10,
            2e-10,
            20
        ),
        np.linspace(
            -1e-10,
            -2e-10,
            20
        ),
    ])

    cases.append(
        (
            "near_stationary",
            h,
            f
        )
    )

    offset = np.array([
        586950.0,
        5338720.0
    ])

    h = np.column_stack([
        np.arange(
            20,
            dtype=float
        )
        *
        0.3,
        np.arange(
            20,
            dtype=float
        )
        *
        -0.15,
    ]) + offset

    f = np.column_stack([
        np.arange(
            20,
            40,
            dtype=float
        )
        *
        0.3,
        np.arange(
            20,
            40,
            dtype=float
        )
        *
        -0.15,
    ]) + offset

    cases.append(
        (
            "utm_translation",
            h,
            f
        )
    )

    return cases


def ast_to_source(text, node):
    seg = ast.get_source_segment(
        text,
        node
    )

    if seg is not None:
        return " ".join(
            seg.strip().split()
        )

    return ""


def find_scale_floor_evidence(path, canonical_func_name):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text
    )

    rows = []

    # 1. Every lexical occurrence with source context.
    lines = text.splitlines()

    for i, line in enumerate(
        lines,
        start=1
    ):
        if "scale_floor" in line:
            lo = max(
                1,
                i - 3
            )

            hi = min(
                len(lines),
                i + 3
            )

            context = "\n".join(
                f"{j:05d}: {lines[j-1]}"
                for j in range(
                    lo,
                    hi + 1
                )
            )

            rows.append({
                "kind":
                    "lexical_context",

                "line":
                    i,

                "expression":
                    line.strip(),

                "context":
                    context,

                "resolved_numeric":
                    np.nan,
            })

    # 2. Top-level assignments with scale/floor in variable name.
    for node in tree.body:
        target_names = []

        if isinstance(
            node,
            ast.Assign
        ):
            target_names = [
                t.id
                for t in node.targets
                if isinstance(
                    t,
                    ast.Name
                )
            ]

        elif (
            isinstance(
                node,
                ast.AnnAssign
            )
            and
            isinstance(
                node.target,
                ast.Name
            )
        ):
            target_names = [
                node.target.id
            ]

        for name in target_names:
            low = name.lower()

            if (
                "scale" not in low
                and
                "floor" not in low
            ):
                continue

            try:
                value = ast.literal_eval(
                    node.value
                )

                numeric = (
                    float(
                        value
                    )
                    if isinstance(
                        value,
                        (int, float)
                    )
                    else np.nan
                )

            except Exception:
                numeric = np.nan

            rows.append({
                "kind":
                    "top_level_assignment",

                "line":
                    int(
                        node.lineno
                    ),

                "expression":
                    ast_to_source(
                        text,
                        node
                    ),

                "context":
                    "",

                "resolved_numeric":
                    numeric,
            })

    # 3. argparse add_argument defaults that mention scale/floor.
    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call
        ):
            continue

        func_name = ""

        if isinstance(
            node.func,
            ast.Attribute
        ):
            func_name = node.func.attr

        elif isinstance(
            node.func,
            ast.Name
        ):
            func_name = node.func.id

        if func_name != "add_argument":
            continue

        arg_text = ast_to_source(
            text,
            node
        )

        if (
            "scale" not in arg_text.lower()
            and
            "floor" not in arg_text.lower()
        ):
            continue

        default_val = np.nan

        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    v = ast.literal_eval(
                        kw.value
                    )

                    if isinstance(
                        v,
                        (int, float)
                    ):
                        default_val = float(
                            v
                        )
                except Exception:
                    pass

        rows.append({
            "kind":
                "argparse_default",

            "line":
                int(
                    node.lineno
                ),

            "expression":
                arg_text,

            "context":
                "",

            "resolved_numeric":
                default_val,
        })

    # 4. Canonicalizer call sites and scale_floor argument expression.
    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call
        ):
            continue

        called = None

        if isinstance(
            node.func,
            ast.Name
        ):
            called = node.func.id

        elif isinstance(
            node.func,
            ast.Attribute
        ):
            called = node.func.attr

        if called != canonical_func_name:
            continue

        scale_expr = None

        for kw in node.keywords:
            if kw.arg == "scale_floor":
                scale_expr = ast_to_source(
                    text,
                    kw.value
                )

        if scale_expr is None and len(
            node.args
        ) >= 3:
            scale_expr = ast_to_source(
                text,
                node.args[2]
            )

        rows.append({
            "kind":
                "canonicalizer_callsite",

            "line":
                int(
                    node.lineno
                ),

            "expression":
                ast_to_source(
                    text,
                    node
                ),

            "context":
                (
                    f"scale_floor_expression={scale_expr}"
                ),

            "resolved_numeric":
                np.nan,
        })

    df = pd.DataFrame(
        rows
    )

    if len(
        df
    ):
        df = df.drop_duplicates(
            subset=[
                "kind",
                "line",
                "expression",
            ]
        ).sort_values(
            [
                "line",
                "kind"
            ]
        )

    return df


def run_equivalence(
    build_info,
    fresh_info
):
    rows = []

    # Synthetic cases across multiple floor probes.
    for floor in PROBE_SCALE_FLOORS:
        for case_name, h, f in synthetic_cases():
            a = extract_common(
                call_canonicalizer(
                    build_info[
                        "function"
                    ],
                    h,
                    f,
                    floor
                )
            )

            b = extract_common(
                call_canonicalizer(
                    fresh_info[
                        "function"
                    ],
                    h,
                    f,
                    floor
                )
            )

            d = compare(
                a,
                b
            )

            rows.append({
                "dataset":
                    "synthetic",

                "task":
                    "synthetic",

                "sample":
                    case_name,

                "scale_floor_probe":
                    floor,

                **d,
            })

    # Real BAMBI windows. Path length is expected to dominate plausible
    # scale floors, but we still probe all values to test implementation equality.
    for npz_path, task in [
        (
            NPZ_22,
            "2s_to_2s"
        ),
        (
            NPZ_25,
            "2s_to_5s"
        ),
    ]:
        z = np.load(
            npz_path,
            allow_pickle=False
        )

        X = z[
            "X_abs_xy"
        ]

        Y = z[
            "Y_abs_xy"
        ]

        n = len(
            X
        )

        indices = sorted(
            set(
                np.linspace(
                    0,
                    n - 1,
                    min(
                        n,
                        200
                    ),
                    dtype=int
                ).tolist()
            )
        )

        for floor in PROBE_SCALE_FLOORS:
            for i in indices:
                h = np.asarray(
                    X[i],
                    dtype=np.float64
                )

                f = np.asarray(
                    Y[i],
                    dtype=np.float64
                )

                a = extract_common(
                    call_canonicalizer(
                        build_info[
                            "function"
                        ],
                        h,
                        f,
                        floor
                    )
                )

                b = extract_common(
                    call_canonicalizer(
                        fresh_info[
                            "function"
                        ],
                        h,
                        f,
                        floor
                    )
                )

                d = compare(
                    a,
                    b
                )

                rows.append({
                    "dataset":
                        "BAMBI_Flight6",

                    "task":
                        task,

                    "sample":
                        f"sample_{i}",

                    "scale_floor_probe":
                        floor,

                    **d,
                })

    return pd.DataFrame(
        rows
    )


def summarize_equivalence(df):
    metric_cols = [
        "history_max_abs_diff",
        "future_max_abs_diff",
        "origin_max_abs_diff",
        "angle_periodic_abs_diff",
        "scale_abs_diff",
        "path_length_abs_diff",
    ]

    maxes = {}

    for c in metric_cols:
        s = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        maxes[c] = (
            float(
                s.max()
            )
            if s.notna().any()
            else np.nan
        )

    pass_numeric = (
        maxes[
            "history_max_abs_diff"
        ]
        <=
        ATOL_CANON
        and
        maxes[
            "future_max_abs_diff"
        ]
        <=
        ATOL_CANON
        and
        maxes[
            "origin_max_abs_diff"
        ]
        <=
        ATOL_META
        and
        maxes[
            "angle_periodic_abs_diff"
        ]
        <=
        ATOL_META
        and
        maxes[
            "scale_abs_diff"
        ]
        <=
        ATOL_META
    )

    return maxes, pass_numeric


def unique_numeric_candidates(*dfs):
    vals = []

    for df in dfs:
        if not len(
            df
        ):
            continue

        s = pd.to_numeric(
            df[
                "resolved_numeric"
            ],
            errors="coerce"
        ).dropna()

        vals.extend(
            s.tolist()
        )

    return sorted(
        set(
            float(v)
            for v in vals
        )
    )


def main():
    print(
        "\nSTEP 6D-K2B2-V2 — CANONICALIZER EQUIVALENCE + SCALE-FLOOR SOURCE AUDIT"
    )

    print(
        "No BAMBI final canonicalization. No forecasting. No conformal evaluation.\n"
    )

    for p in [
        BUILD_SRC,
        FRESH_SRC,
        NPZ_22,
        NPZ_25,
    ]:
        require(
            p
        )

    loaded = {}

    for path, fname, label in TARGETS:
        loaded[
            label
        ] = build_isolated_namespace(
            path,
            fname
        )

        print(
            f"{label}: {path.name}::{fname} "
            f"{loaded[label]['signature']}"
        )

    print(
        "\nRUNNING NUMERICAL EQUIVALENCE PROBES..."
    )

    eq = run_equivalence(
        loaded[
            "build_final"
        ],
        loaded[
            "fresh_holdout"
        ]
    )

    eq.to_csv(
        OUT
        /
        "TABLE_step6dk2b2v2_equivalence_probes.csv",
        index=False
    )

    maxes, pass_numeric = summarize_equivalence(
        eq
    )

    print(
        "\nAUDITING ACTUAL SCALE_FLOOR SOURCE..."
    )

    build_evidence = find_scale_floor_evidence(
        BUILD_SRC,
        "canonicalize_sample"
    )

    fresh_evidence = find_scale_floor_evidence(
        FRESH_SRC,
        "canonicalize_window"
    )

    build_evidence.to_csv(
        OUT
        /
        "step6dk2b2v2_build_scale_floor_evidence.csv",
        index=False
    )

    fresh_evidence.to_csv(
        OUT
        /
        "step6dk2b2v2_fresh_scale_floor_evidence.csv",
        index=False
    )

    numeric_candidates = unique_numeric_candidates(
        build_evidence,
        fresh_evidence
    )

    print(
        "\nMAX NUMERICAL DIFFERENCES:"
    )

    print(
        json.dumps(
            maxes,
            indent=2
        )
    )

    print(
        "\nBUILD PIPELINE SCALE_FLOOR EVIDENCE:"
    )

    if len(
        build_evidence
    ):
        print(
            build_evidence[
                [
                    "kind",
                    "line",
                    "expression",
                    "context",
                    "resolved_numeric",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nFRESH HOLDOUT SCALE_FLOOR EVIDENCE:"
    )

    if len(
        fresh_evidence
    ):
        print(
            fresh_evidence[
                [
                    "kind",
                    "line",
                    "expression",
                    "context",
                    "resolved_numeric",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nDIRECT NUMERIC CANDIDATES FOUND:"
    )

    print(
        numeric_candidates
    )

    equivalence_status = (
        "CANONICALIZERS_NUMERICALLY_EQUIVALENT_ACROSS_PROBES"
        if pass_numeric
        else
        "CANONICALIZER_MISMATCH_DO_NOT_PROCEED"
    )

    # Actual scale floor is considered resolved only when source evidence
    # yields one unambiguous positive numeric candidate that is clearly
    # connected to scale/floor logic. We do not guess from the probe list.
    positive_candidates = [
        v
        for v in numeric_candidates
        if np.isfinite(
            v
        )
        and
        v
        >
        0
    ]

    if len(
        positive_candidates
    ) == 1:
        scale_status = (
            "SINGLE_DIRECT_NUMERIC_CANDIDATE_FOUND_REVIEW_CONTEXT_BEFORE_LOCK"
        )
    else:
        scale_status = (
            "ACTUAL_SCALE_FLOOR_STILL_REQUIRES_CONTEXT_RESOLUTION"
        )

    overall_status = (
        equivalence_status
        +
        "__"
        +
        scale_status
    )

    report = {
        "step":
            "STEP_6D_K2B2_V2_EQUIVALENCE_AND_SCALE_FLOOR_AUDIT",

        "probe_scale_floors":
            PROBE_SCALE_FLOORS,

        "equivalence_rows":
            int(
                len(
                    eq
                )
            ),

        "max_differences":
            maxes,

        "equivalence_status":
            equivalence_status,

        "direct_numeric_candidates":
            positive_candidates,

        "scale_floor_status":
            scale_status,

        "overall_status":
            overall_status,

        "important_note":
            (
                "Probe scale floors are used only to test equivalence. "
                "They are NOT candidates for final BAMBI canonicalization "
                "unless independently resolved from the original pipeline source."
            ),
    }

    with open(
        OUT
        /
        "STEP6D_K2B2_V2_SUMMARY.json",
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
        "STEP6D_K2B2_V2_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K2B2-V2 — EQUIVALENCE + SCALE-FLOOR SOURCE AUDIT\n"
        )

        f.write(
            "="
            *
            84
            +
            "\n\n"
        )

        f.write(
            "MAX NUMERICAL DIFFERENCES\n"
        )

        f.write(
            "-------------------------\n"
        )

        f.write(
            json.dumps(
                maxes,
                indent=2
            )
        )

        f.write(
            "\n\nEQUIVALENCE STATUS\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            equivalence_status
            +
            "\n"
        )

        f.write(
            "\nBUILD PIPELINE SCALE_FLOOR EVIDENCE\n"
        )

        f.write(
            "-----------------------------------\n"
        )

        f.write(
            build_evidence.to_string(
                index=False
            )
            if len(
                build_evidence
            )
            else "NONE"
        )

        f.write(
            "\n\nFRESH HOLDOUT SCALE_FLOOR EVIDENCE\n"
        )

        f.write(
            "----------------------------------\n"
        )

        f.write(
            fresh_evidence.to_string(
                index=False
            )
            if len(
                fresh_evidence
            )
            else "NONE"
        )

        f.write(
            "\n\nDIRECT NUMERIC CANDIDATES FOUND\n"
        )

        f.write(
            "-------------------------------\n"
        )

        f.write(
            json.dumps(
                positive_candidates
            )
        )

        f.write(
            "\n\nOVERALL STATUS\n"
        )

        f.write(
            "--------------\n"
        )

        f.write(
            overall_status
            +
            "\n"
        )

        f.write(
            "\nProbe values are NOT used as the final pipeline scale floor.\n"
        )

    print(
        "\nEQUIVALENCE STATUS:"
    )

    print(
        equivalence_status
    )

    print(
        "\nSCALE_FLOOR STATUS:"
    )

    print(
        scale_status
    )

    print(
        "\nOVERALL STATUS:"
    )

    print(
        overall_status
    )

    print(
        "\nSTEP 6D-K2B2-V2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
