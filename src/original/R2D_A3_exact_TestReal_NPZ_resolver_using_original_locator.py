from pathlib import Path
import ast
import hashlib
import json
import os
from contextlib import contextmanager

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

EVALUATOR = (
    MAIN_RESULT
    / "evaluate_birdsai_testreal_frozen_ensemble.py"
)

R2D_A_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_targeted_recompute_preflight"
)

CANDIDATE_CSV = (
    R2D_A_DIR
    / "R2D_A_npz_candidate_scores.csv"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_targeted_recompute_preflight_v3"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def require(path):
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(p)


@contextmanager
def pushd(path):
    old = Path.cwd()

    os.chdir(path)

    try:
        yield

    finally:
        os.chdir(old)


def bound_names(node):
    names = set()

    if isinstance(node, ast.Assign):
        targets = node.targets

    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    else:
        return names

    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.add(sub.id)

    return names


def loaded_names(node):
    names = set()

    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Name)
            and
            isinstance(sub.ctx, ast.Load)
        ):
            names.add(sub.id)

    return names


def safe_extract_function_namespace(
    source_path,
    target_function,
):
    text = source_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(text)

    imports = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
            )
        )
    ]

    funcs = {
        node.name:
            node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        )
    }

    classes = {
        node.name:
            node
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef
        )
    }

    assignments = {}

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            for name in bound_names(node):
                assignments[
                    name
                ] = node

    if target_function not in funcs:
        raise RuntimeError(
            f"{target_function} not found in {source_path}"
        )

    selected_funcs = {
        target_function
    }

    selected_classes = set()

    selected_assignments = set()

    pending_names = set(
        loaded_names(
            funcs[
                target_function
            ]
        )
    )

    processed_names = set()

    while True:
        new_names = (
            pending_names
            -
            processed_names
        )

        if not new_names:
            break

        for name in list(
            new_names
        ):
            processed_names.add(
                name
            )

            if (
                name in funcs
                and
                name not in selected_funcs
            ):
                selected_funcs.add(
                    name
                )

                pending_names.update(
                    loaded_names(
                        funcs[
                            name
                        ]
                    )
                )

            elif (
                name in classes
                and
                name not in selected_classes
            ):
                selected_classes.add(
                    name
                )

                pending_names.update(
                    loaded_names(
                        classes[
                            name
                        ]
                    )
                )

            elif (
                name in assignments
                and
                name not in selected_assignments
            ):
                selected_assignments.add(
                    name
                )

                pending_names.update(
                    loaded_names(
                        assignments[
                            name
                        ]
                    )
                )

    selected_nodes = []

    selected_nodes.extend(
        imports
    )

    # Preserve original source order for dependent assignments/classes/functions.
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            names = bound_names(
                node
            )

            if names & selected_assignments:
                selected_nodes.append(
                    node
                )

        elif isinstance(
            node,
            ast.ClassDef
        ):
            if node.name in selected_classes:
                selected_nodes.append(
                    node
                )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            if node.name in selected_funcs:
                selected_nodes.append(
                    node
                )

    module_ast = ast.Module(
        body=selected_nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module_ast
    )

    namespace = {
        "__file__":
            str(
                source_path
            ),

        "__name__":
            "_r2d_a3_safe_evaluator_",
    }

    exec(
        compile(
            module_ast,
            str(
                source_path
            ),
            "exec",
        ),
        namespace,
    )

    fn = namespace.get(
        target_function
    )

    if fn is None:
        raise RuntimeError(
            f"Safe extraction did not create {target_function}"
        )

    return {
        "function":
            fn,

        "namespace":
            namespace,

        "selected_functions":
            sorted(
                selected_funcs
            ),

        "selected_classes":
            sorted(
                selected_classes
            ),

        "selected_assignments":
            sorted(
                selected_assignments
            ),

        "source_text":
            text,

        "target_ast":
            funcs[
                target_function
            ],
    }


def function_signature_from_ast(fn_node):
    args = []

    total_args = (
        list(
            fn_node.args.posonlyargs
        )
        +
        list(
            fn_node.args.args
        )
    )

    defaults = [
        None
    ] * (
        len(
            total_args
        )
        -
        len(
            fn_node.args.defaults
        )
    ) + list(
        fn_node.args.defaults
    )

    for arg, default in zip(
        total_args,
        defaults,
    ):
        args.append({
            "name":
                arg.arg,

            "has_default":
                default is not None,

            "default":
                (
                    ast.unparse(
                        default
                    )
                    if default is not None
                    else None
                ),
        })

    return args


def call_locate_function(
    fn,
    signature,
    task,
):
    names = [
        x[
            "name"
        ]
        for x in signature
    ]

    kwargs = {}

    # Known task aliases from the original pipeline.
    if "task" in names:
        kwargs[
            "task"
        ] = task

    if "future_length" in names:
        kwargs[
            "future_length"
        ] = (
            20
            if task == "2s_to_2s"
            else
            50
        )

    if "future_frames" in names:
        kwargs[
            "future_frames"
        ] = (
            20
            if task == "2s_to_2s"
            else
            50
        )

    required_without_values = [
        x[
            "name"
        ]
        for x in signature
        if (
            not x[
                "has_default"
            ]
            and
            x[
                "name"
            ]
            not in kwargs
        )
    ]

    if required_without_values:
        raise RuntimeError(
            "locate_test_npz has required parameters this resolver cannot "
            f"safely infer: {required_without_values}"
        )

    result = fn(
        **kwargs
    )

    return result


def normalize_returned_path(result):
    if isinstance(
        result,
        Path
    ):
        return result

    if isinstance(
        result,
        str
    ):
        return Path(
            result
        )

    if isinstance(
        result,
        (
            tuple,
            list,
        )
    ):
        path_like = [
            item
            for item in result
            if isinstance(
                item,
                (
                    str,
                    Path,
                )
            )
        ]

        if len(
            path_like
        ) == 1:
            return Path(
                path_like[
                    0
                ]
            )

    raise RuntimeError(
        f"Unexpected locate_test_npz return type: {type(result)} value={result!r}"
    )


def top_testreal_candidates():
    require(
        CANDIDATE_CSV
    )

    df = pd.read_csv(
        CANDIDATE_CSV
    )

    rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        subset = df[
            (
                df[
                    "target_dataset"
                ]
                ==
                "BIRDSAI_TestReal"
            )
            &
            (
                df[
                    "target_task"
                ]
                ==
                task
            )
        ].copy()

        if len(
            subset
        ) == 0:
            raise RuntimeError(
                f"No TestReal candidates in R2D-A for {task}"
            )

        max_score = subset[
            "score"
        ].max()

        top = subset[
            subset[
                "score"
            ]
            ==
            max_score
        ].copy()

        for _, row in top.iterrows():
            rows.append({
                "task":
                    task,

                "path":
                    str(
                        Path(
                            row[
                                "path"
                            ]
                        ).resolve()
                    ),

                "sha256":
                    sha256_file(
                        Path(
                            row[
                                "path"
                            ]
                        )
                    ),

                "score":
                    int(
                        row[
                            "score"
                        ]
                    ),
            })

    return pd.DataFrame(
        rows
    )


def search_prior_provenance_for_path(
    candidate_path,
):
    basename = candidate_path.name
    evidence = []

    suffixes = {
        ".json",
        ".csv",
        ".txt",
        ".md",
        ".py",
    }

    for root in [
        MAIN_RESULT_2,
        MAIN_RESULT,
    ]:
        if not root.exists():
            continue

        for p in root.rglob(
            "*"
        ):
            if (
                not p.is_file()
                or
                p.suffix.lower()
                not in suffixes
            ):
                continue

            full = str(
                p
            ).lower()

            if not any(
                token in full
                for token in [
                    "testreal",
                    "birdsai",
                    "frozen",
                    "ensemble",
                    "audit",
                    "result",
                    "summary",
                    "manifest",
                ]
            ):
                continue

            try:
                raw = p.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            if basename.lower() not in raw.lower():
                continue

            matched_lines = []

            for line_no, line in enumerate(
                raw.splitlines(),
                start=1,
            ):
                if basename.lower() in line.lower():
                    matched_lines.append(
                        f"L{line_no}: {line.strip()[:600]}"
                    )

            evidence.append({
                "candidate_path":
                    str(
                        candidate_path
                    ),

                "evidence_file":
                    str(
                        p
                    ),

                "evidence_sha256":
                    sha256_file(
                        p
                    ),

                "evidence_lines":
                    " || ".join(
                        matched_lines[
                            :10
                        ]
                    ),
            })

    return evidence


def main():
    print(
        "\nR2D-A3 — EXACT TESTREAL NPZ RESOLUTION USING ORIGINAL locate_test_npz()"
    )

    print(
        "NO ADE/FDE is generated. The original evaluator's locate_test_npz() "
        "is safely AST-extracted; its top-level benchmark code is NOT executed.\n"
    )

    require(
        EVALUATOR
    )

    extracted = safe_extract_function_namespace(
        EVALUATOR,
        "locate_test_npz",
    )

    signature = function_signature_from_ast(
        extracted[
            "target_ast"
        ]
    )

    print(
        "SAFE EXTRACTION:"
    )

    print(
        json.dumps(
            {
                "evaluator":
                    str(
                        EVALUATOR
                    ),

                "evaluator_sha256":
                    sha256_file(
                        EVALUATOR
                    ),

                "locate_test_npz_signature":
                    signature,

                "selected_functions":
                    extracted[
                        "selected_functions"
                    ],

                "selected_classes":
                    extracted[
                        "selected_classes"
                    ],

                "selected_assignments":
                    extracted[
                        "selected_assignments"
                    ],

                "top_level_benchmark_executed":
                    False,
            },
            indent=2,
        )
    )

    candidates = top_testreal_candidates()

    candidate_paths_by_task = {
        task:
            {
                os.path.normcase(
                    str(
                        Path(
                            p
                        ).resolve()
                    )
                )
                for p in candidates[
                    candidates[
                        "task"
                    ]
                    ==
                    task
                ][
                    "path"
                ].tolist()
            }
        for task in [
            "2s_to_2s",
            "2s_to_5s",
        ]
    }

    resolution_rows = []
    call_rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        successes = []

        for cwd in [
            MAIN_RESULT,
            MAIN_RESULT_2,
            DOWNLOADS,
        ]:
            if not cwd.exists():
                continue

            try:
                with pushd(
                    cwd
                ):
                    result = call_locate_function(
                        extracted[
                            "function"
                        ],
                        signature,
                        task,
                    )

                returned = normalize_returned_path(
                    result
                )

                if not returned.is_absolute():
                    returned = (
                        cwd
                        /
                        returned
                    )

                returned = returned.resolve()

                in_top_candidates = (
                    os.path.normcase(
                        str(
                            returned
                        )
                    )
                    in
                    candidate_paths_by_task[
                        task
                    ]
                )

                row = {
                    "task":
                        task,

                    "cwd":
                        str(
                            cwd
                        ),

                    "success":
                        True,

                    "returned_path":
                        str(
                            returned
                        ),

                    "returned_exists":
                        returned.exists(),

                    "returned_sha256":
                        (
                            sha256_file(
                                returned
                            )
                            if returned.exists()
                            else None
                        ),

                    "is_r2d_a_top_candidate":
                        bool(
                            in_top_candidates
                        ),

                    "error":
                        None,
                }

                call_rows.append(
                    row
                )

                if (
                    returned.exists()
                    and
                    in_top_candidates
                ):
                    successes.append(
                        row
                    )

            except Exception as e:
                call_rows.append({
                    "task":
                        task,

                    "cwd":
                        str(
                            cwd
                        ),

                    "success":
                        False,

                    "returned_path":
                        None,

                    "returned_exists":
                        False,

                    "returned_sha256":
                        None,

                    "is_r2d_a_top_candidate":
                        False,

                    "error":
                        f"{type(e).__name__}: {e}",
                })

        unique_success_paths = sorted(
            {
                row[
                    "returned_path"
                ]
                for row in successes
            }
        )

        if len(
            unique_success_paths
        ) == 1:
            selected = Path(
                unique_success_paths[
                    0
                ]
            )

            reason = (
                "Original evaluator locate_test_npz() resolved uniquely to this "
                "R2D-A top candidate across successful safe calls."
            )

        else:
            # Secondary evidence: prior frozen result provenance files.
            evidence = []

            task_candidates = candidates[
                candidates[
                    "task"
                ]
                ==
                task
            ]

            for _, cand in task_candidates.iterrows():
                evidence.extend(
                    search_prior_provenance_for_path(
                        Path(
                            cand[
                                "path"
                            ]
                        )
                    )
                )

            evidence_df = pd.DataFrame(
                evidence
            )

            if len(
                evidence_df
            ) > 0:
                counts = (
                    evidence_df[
                        "candidate_path"
                    ]
                    .value_counts()
                )

                top_count = counts.max()

                best_paths = counts[
                    counts
                    ==
                    top_count
                ].index.tolist()

                if len(
                    best_paths
                ) == 1:
                    selected = Path(
                        best_paths[
                            0
                        ]
                    )

                    reason = (
                        "locate_test_npz() did not resolve uniquely under the audited "
                        "working directories, but prior frozen provenance references "
                        "uniquely favored this top candidate."
                    )

                else:
                    raise RuntimeError(
                        f"Still ambiguous for {task}: locate_test_npz successful "
                        f"paths={unique_success_paths}; provenance best paths={best_paths}"
                    )

            else:
                raise RuntimeError(
                    f"Still ambiguous for {task}: locate_test_npz successful "
                    f"paths={unique_success_paths}; no prior provenance evidence found."
                )

        resolution_rows.append({
            "task":
                task,

            "selected_path":
                str(
                    selected
                ),

            "selected_sha256":
                sha256_file(
                    selected
                ),

            "reason":
                reason,
        })

    resolution_df = pd.DataFrame(
        resolution_rows
    )

    calls_df = pd.DataFrame(
        call_rows
    )

    resolution_path = (
        OUT
        /
        "R2D_A3_resolved_testreal_npz.csv"
    )

    calls_path = (
        OUT
        /
        "R2D_A3_locate_test_npz_calls.csv"
    )

    resolution_df.to_csv(
        resolution_path,
        index=False,
    )

    calls_df.to_csv(
        calls_path,
        index=False,
    )

    summary = {
        "step":
            "R2D_A3_EXACT_TESTREAL_NPZ_RESOLUTION",

        "status":
            "R2D_A3_TESTREAL_NPZ_EXACTLY_RESOLVED_READY_FOR_TARGETED_RECOMPUTATION",

        "new_metrics_generated":
            False,

        "evaluator_source":
            str(
                EVALUATOR
            ),

        "evaluator_sha256":
            sha256_file(
                EVALUATOR
            ),

        "resolved_csv":
            str(
                resolution_path
            ),

        "safe_call_audit_csv":
            str(
                calls_path
            ),
    }

    summary_path = (
        OUT
        /
        "R2D_A3_SUMMARY.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nRESOLVED TESTREAL NPZs:"
    )

    print(
        resolution_df.to_string(
            index=False
        )
    )

    print(
        "\nSAFE locate_test_npz() CALL AUDIT:"
    )

    print(
        calls_df.to_string(
            index=False
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
        "\nR2D-A3 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
