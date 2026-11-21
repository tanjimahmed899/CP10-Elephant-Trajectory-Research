
from pathlib import Path
import ast
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R12_ROOT = REVISION_ROOT / "R12_track_accounting"

V8_DIR = (
    R12_ROOT
    / "R12E_v8_trace_core_10hz_dataset_generator"
)
V8_MANIFEST = V8_DIR / "R12E_V8_MANIFEST.json"

OUT = (
    R12_ROOT
    / "R12E_v9_exact_core10hz_read_write_dataflow_provenance"
)
OUT.mkdir(parents=True, exist_ok=True)

TARGET_FILES = [
    "birdsai_test_real_core_2s_to_2s_10hz.npz",
    "birdsai_test_real_core_2s_to_2s_10hz_metadata.csv",
    "birdsai_test_real_core_2s_to_5s_10hz.npz",
    "birdsai_test_real_core_2s_to_5s_10hz_metadata.csv",
    "birdsai_train_real_core_2s_to_2s_10hz.npz",
    "birdsai_train_real_core_2s_to_2s_10hz_metadata.csv",
    "birdsai_train_real_core_2s_to_5s_10hz.npz",
    "birdsai_train_real_core_2s_to_5s_10hz_metadata.csv",
]

SEARCH_ROOTS = [
    MAIN1,
    MAIN2,
    DOWNLOADS,
]

SKIP_DIR_TOKENS = [
    "site-packages",
    "__pycache__",
    ".git",
    "node_modules",
    "paper_strengthening_revision",
    "anaconda",
    "miniconda",
    "venv",
    ".venv",
    "torch",
    "ultralytics",
]

MAX_BYTES = 20_000_000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def verify_v8():
    require(V8_MANIFEST)

    m = json.loads(
        V8_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    allowed = {
        "R12E_V8_CORE_10HZ_WRITER_LOCATED_READY_FOR_R12E_V9_EXACT_FILTER_EXPRESSION_LOCK",
        "R12E_V8_CORE_10HZ_WRITER_CANDIDATES_FOUND_BUT_NOT_UNIQUELY_LOCKED_REVIEW_TOP_WRITER",
    }

    if m.get("status") not in allowed:
        raise RuntimeError(
            f"Unexpected R12-E-V8 status: {m.get('status')}"
        )

    return m


def should_skip_dir(path):
    low = str(path).lower()

    return any(
        token.lower() in low
        for token in SKIP_DIR_TOKENS
    )


def iter_python_files():
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for current_root, dirs, files in os.walk(root):
            current = Path(current_root)

            # Prune skipped directories in-place.
            dirs[:] = [
                d
                for d in dirs
                if not should_skip_dir(
                    current / d
                )
            ]

            if should_skip_dir(current):
                continue

            for name in files:
                if not name.lower().endswith(".py"):
                    continue

                p = current / name

                try:
                    if p.stat().st_size > MAX_BYTES:
                        continue
                except Exception:
                    continue

                key = str(
                    p.resolve()
                ).lower()

                if key in seen:
                    continue

                seen.add(key)
                yield p


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        base = call_name(node.value)

        if base:
            return f"{base}.{node.attr}"

        return node.attr

    return ""


def target_names_in_text(text):
    low = text.lower()

    return [
        name
        for name in TARGET_FILES
        if name.lower() in low
    ]


def collect_assignments(tree, text):
    """
    Map variable names to assignment source expressions that contain
    target filenames or unified_trajectory_10hz path references.
    """
    rows = []
    var_map = {}

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            continue

        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value

        else:
            targets = [
                node.target
            ]
            value_node = node.value

        names = []

        for target in targets:
            if isinstance(target, ast.Name):
                names.append(
                    target.id
                )

        if not names:
            continue

        src = ast.get_source_segment(
            text,
            value_node,
        ) or ""

        hits = target_names_in_text(src)

        unified = (
            "unified_trajectory_10hz"
            in
            src.lower()
        )

        if not hits and not unified:
            continue

        for name in names:
            rec = {
                "variable":
                    name,

                "line":
                    int(
                        getattr(
                            node,
                            "lineno",
                            0,
                        )
                    ),

                "assignment_expression":
                    src,

                "target_filename_hits":
                    "|".join(
                        hits
                    ),

                "contains_unified_trajectory_10hz":
                    unified,
            }

            rows.append(rec)

            var_map.setdefault(
                name,
                []
            ).append(rec)

    return rows, var_map


def classify_call_operation(name):
    low = name.lower()

    read_names = [
        "np.load",
        "numpy.load",
        "pd.read_csv",
        "pandas.read_csv",
        "read_csv",
    ]

    write_names = [
        "np.savez",
        "numpy.savez",
        "np.savez_compressed",
        "numpy.savez_compressed",
        "to_csv",
    ]

    if any(
        low.endswith(x)
        or
        low == x
        for x in read_names
    ):
        return "READ"

    if any(
        low.endswith(x)
        or
        low == x
        for x in write_names
    ):
        return "WRITE"

    return "OTHER"


def names_referenced(node):
    return sorted(
        {
            n.id
            for n in ast.walk(node)
            if isinstance(
                n,
                ast.Name
            )
        }
    )


def inspect_file(path):
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return [], [], []

    if (
        not target_names_in_text(text)
        and
        "unified_trajectory_10hz"
        not in
        text.lower()
    ):
        return [], [], []

    try:
        tree = ast.parse(
            text,
            filename=str(path)
        )
    except Exception:
        return [], [], []

    assignment_rows, var_map = collect_assignments(
        tree,
        text,
    )

    call_rows = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        cname = call_name(
            node.func
        )

        op = classify_call_operation(
            cname
        )

        if op == "OTHER":
            continue

        src = ast.get_source_segment(
            text,
            node,
        ) or ""

        direct_hits = target_names_in_text(
            src
        )

        referenced = names_referenced(
            node
        )

        indirect_hits = []

        linked_vars = []

        for var in referenced:
            if var not in var_map:
                continue

            linked_vars.append(
                var
            )

            for rec in var_map[
                var
            ]:
                hits = str(
                    rec[
                        "target_filename_hits"
                    ]
                )

                if hits:
                    indirect_hits.extend(
                        [
                            h
                            for h in hits.split("|")
                            if h
                        ]
                    )

        all_hits = sorted(
            set(
                direct_hits
                +
                indirect_hits
            )
        )

        if not all_hits:
            continue

        call_rows.append(
            {
                "operation":
                    op,

                "call_name":
                    cname,

                "line":
                    int(
                        getattr(
                            node,
                            "lineno",
                            0,
                        )
                    ),

                "target_filename_hits":
                    "|".join(
                        all_hits
                    ),

                "linked_path_variables":
                    "|".join(
                        sorted(
                            set(
                                linked_vars
                            )
                        )
                    ),

                "call_expression":
                    src,
            }
        )

    lexical_rows = []

    lines = text.splitlines()

    for i, line in enumerate(
        lines,
        start=1,
    ):
        hits = target_names_in_text(
            line
        )

        if not hits:
            continue

        start = max(
            1,
            i - 8,
        )

        end = min(
            len(lines),
            i + 8,
        )

        lexical_rows.append(
            {
                "line":
                    i,

                "target_filename_hits":
                    "|".join(
                        hits
                    ),

                "anchor_text":
                    line,

                "context":
                    "\n".join(
                        f"{j}: {lines[j-1]}"
                        for j in range(
                            start,
                            end + 1,
                        )
                    ),
            }
        )

    return assignment_rows, call_rows, lexical_rows


def main():
    print(
        "\nR12-E-V9 — EXACT CORE-10HZ READ/WRITE DATAFLOW PROVENANCE"
    )

    print(
        "DIAGNOSTIC ONLY. NO training and NO inference. "
        "R12-E-V8 could falsely label a consumer as a writer because target filenames "
        "and unrelated write calls co-occurred in the same script. This step performs "
        "AST-level dataflow classification: target path variables are traced into "
        "np.load/read_csv (READ) versus np.savez*/to_csv (WRITE). It scans the original "
        "Main Result trees and Python scripts under Downloads to recover the actual "
        "producer if it still exists. No original file is modified.\n"
    )

    verify_v8()

    file_rows = []
    assignment_rows = []
    call_rows = []
    lexical_rows = []

    for p in iter_python_files():
        assigns, calls, lexical = inspect_file(
            p
        )

        if (
            not assigns
            and
            not calls
            and
            not lexical
        ):
            continue

        file_hash = sha256_file(
            p
        )

        read_hits = sorted(
            {
                h
                for rec in calls
                if rec[
                    "operation"
                ]
                ==
                "READ"
                for h in str(
                    rec[
                        "target_filename_hits"
                    ]
                ).split("|")
                if h
            }
        )

        write_hits = sorted(
            {
                h
                for rec in calls
                if rec[
                    "operation"
                ]
                ==
                "WRITE"
                for h in str(
                    rec[
                        "target_filename_hits"
                    ]
                ).split("|")
                if h
            }
        )

        file_rows.append(
            {
                "path":
                    str(p),

                "sha256":
                    file_hash,

                "target_read_count":
                    int(
                        len(
                            read_hits
                        )
                    ),

                "target_write_count":
                    int(
                        len(
                            write_hits
                        )
                    ),

                "target_read_files":
                    "|".join(
                        read_hits
                    ),

                "target_write_files":
                    "|".join(
                        write_hits
                    ),

                "classification":
                    (
                        "PRODUCER"
                        if write_hits
                        and not read_hits
                        else
                        "PRODUCER_AND_CONSUMER"
                        if write_hits
                        and read_hits
                        else
                        "CONSUMER"
                        if read_hits
                        else
                        "REFERENCE_ONLY"
                    ),
            }
        )

        for rec in assigns:
            rec = dict(rec)
            rec[
                "source_file"
            ] = str(p)
            rec[
                "source_sha256"
            ] = file_hash
            assignment_rows.append(
                rec
            )

        for rec in calls:
            rec = dict(rec)
            rec[
                "source_file"
            ] = str(p)
            rec[
                "source_sha256"
            ] = file_hash
            call_rows.append(
                rec
            )

        for rec in lexical:
            rec = dict(rec)
            rec[
                "source_file"
            ] = str(p)
            rec[
                "source_sha256"
            ] = file_hash
            lexical_rows.append(
                rec
            )

    files_df = pd.DataFrame(
        file_rows
    )

    if len(
        files_df
    ):
        files_df[
            "_rank"
        ] = files_df[
            "classification"
        ].map(
            {
                "PRODUCER":
                    0,

                "PRODUCER_AND_CONSUMER":
                    1,

                "CONSUMER":
                    2,

                "REFERENCE_ONLY":
                    3,
            }
        ).fillna(
            9
        )

        files_df = (
            files_df
            .sort_values(
                [
                    "_rank",
                    "target_write_count",
                    "target_read_count",
                    "path",
                ],
                ascending=[
                    True,
                    False,
                    False,
                    True,
                ]
            )
            .drop(
                columns=[
                    "_rank"
                ]
            )
            .reset_index(
                drop=True
            )
        )

    assignments_df = pd.DataFrame(
        assignment_rows
    )

    calls_df = pd.DataFrame(
        call_rows
    )

    lexical_df = pd.DataFrame(
        lexical_rows
    )

    producers = (
        files_df[
            files_df[
                "classification"
            ].isin(
                [
                    "PRODUCER",
                    "PRODUCER_AND_CONSUMER",
                ]
            )
        ].copy()
        if len(
            files_df
        )
        else pd.DataFrame()
    )

    # Require target WRITE dataflow for at least the four TestReal target files.
    test_targets = {
        name
        for name in TARGET_FILES
        if "test_real" in name
    }

    producer_lock = ""

    complete_producers = []

    for _, row in producers.iterrows():
        writes = {
            x
            for x in str(
                row[
                    "target_write_files"
                ]
            ).split("|")
            if x
        }

        if test_targets.issubset(
            writes
        ):
            complete_producers.append(
                row.to_dict()
            )

    if len(
        complete_producers
    ) == 1:
        producer_lock = str(
            complete_producers[
                0
            ][
                "path"
            ]
        )

        status = (
            "R12E_V9_ACTUAL_CORE10HZ_PRODUCER_EXACTLY_LOCKED_READY_FOR_R12E_V10_FILTER_RULE_EXTRACTION"
        )

    elif len(
        complete_producers
    ) > 1:
        status = (
            "R12E_V9_MULTIPLE_COMPLETE_CORE10HZ_PRODUCERS_FOUND_REVIEW_EXECUTION_PROVENANCE"
        )

    elif len(
        producers
    ):
        status = (
            "R12E_V9_PARTIAL_CORE10HZ_PRODUCER_FOUND_REVIEW_WRITE_COVERAGE"
        )

    else:
        status = (
            "R12E_V9_NO_TARGET_CORE10HZ_WRITER_FOUND_SOURCE_MAY_BE_MISSING"
        )

    files_path = (
        OUT
        /
        "R12E_V9_SOURCE_READ_WRITE_CLASSIFICATION.csv"
    )

    assignments_path = (
        OUT
        /
        "R12E_V9_TARGET_PATH_ASSIGNMENTS.csv"
    )

    calls_path = (
        OUT
        /
        "R12E_V9_TARGET_READ_WRITE_CALLS.csv"
    )

    lexical_path = (
        OUT
        /
        "R12E_V9_TARGET_FILENAME_CONTEXT.csv"
    )

    producers_path = (
        OUT
        /
        "R12E_V9_PRODUCER_CANDIDATES.csv"
    )

    files_df.to_csv(
        files_path,
        index=False
    )

    assignments_df.to_csv(
        assignments_path,
        index=False
    )

    calls_df.to_csv(
        calls_path,
        index=False
    )

    lexical_df.to_csv(
        lexical_path,
        index=False
    )

    producers.to_csv(
        producers_path,
        index=False
    )

    manifest = {
        "step":
            "R12E_V9_EXACT_CORE10HZ_READ_WRITE_DATAFLOW_PROVENANCE",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "v8_writer_attribution_treated_as_unconfirmed":
            True,

        "actual_producer_lock":
            producer_lock,

        "complete_producer_count":
            int(
                len(
                    complete_producers
                )
            ),

        "outputs": {
            "source_classification":
                str(
                    files_path
                ),

            "path_assignments":
                str(
                    assignments_path
                ),

            "read_write_calls":
                str(
                    calls_path
                ),

            "producer_candidates":
                str(
                    producers_path
                ),

            "filename_context":
                str(
                    lexical_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_V9_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8"
    )

    print(
        "SOURCE READ/WRITE CLASSIFICATION:"
    )

    if len(
        files_df
    ):
        print(
            files_df.to_string(
                index=False
            )
        )
    else:
        print(
            "No Python source referencing the target core-10Hz files was found."
        )

    print(
        "\nTARGET READ/WRITE CALLS:"
    )

    if len(
        calls_df
    ):
        print(
            calls_df[
                [
                    "operation",
                    "source_file",
                    "line",
                    "call_name",
                    "target_filename_hits",
                    "linked_path_variables",
                    "call_expression",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No AST-classifiable target read/write calls found."
        )

    print(
        "\nPRODUCER CANDIDATES:"
    )

    if len(
        producers
    ):
        print(
            producers.to_string(
                index=False
            )
        )
    else:
        print(
            "No producer candidate found."
        )

    print(
        "\nACTUAL PRODUCER LOCK:"
    )

    print(
        producer_lock
        if producer_lock
        else
        "Not uniquely locked."
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-E-V9 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
