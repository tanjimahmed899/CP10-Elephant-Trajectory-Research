
from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")

FINAL_DIR = MAIN1 / "final_neural_trajectory_dataset"

FRESH_EVAL = MAIN1 / "evaluate_fresh_wildbox_holdout.py"

K2B2_V3 = (
    MAIN2
    / "step6d_k2b2_v3_task_floor_equivalence_roundtrip.py"
)

K2B3_BUILD = (
    MAIN2
    / "step6d_k2b3_build_and_lock_final_canonical_dataset.py"
)

K2B3_LOCK = (
    MAIN2
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k2b3_final_canonical"
    / "STEP6D_K2B3_CANONICAL_DATASET_LOCK.json"
)

R9H_B_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_B_exact_geometry_formula_lock"
    / "R9H_B_MANIFEST.json"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C0_V3_authoritative_scale_floor_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]

ATOL_EXACT = 1e-15
ATOL_METADATA = 1e-8


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


def verify_r9h_b():
    require(R9H_B_MANIFEST)

    m = json.loads(
        R9H_B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R9H_B_EXACT_FROZEN_CANONICALIZATION_FORMULA_LOCKED_READY_FOR_R9H_C_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R9H-B status: {m.get('status')}"
        )

    return m


def literal_eval_assignment(path, variable_names):
    """
    Extract a top-level literal dict assignment by AST.
    Supports:
      VAR = {"2s_to_2s": 0.068..., "2s_to_5s": 0.067...}
    and nested dictionaries.
    """
    require(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    tree = ast.parse(
        text,
        filename=str(path)
    )

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        names = [
            t.id
            for t in node.targets
            if isinstance(t, ast.Name)
        ]

        if not any(
            n in variable_names
            for n in names
        ):
            continue

        try:
            value = ast.literal_eval(
                node.value
            )
        except Exception:
            continue

        if isinstance(value, dict):
            return value

    return None


def parse_direct_task_value_dict(path, variable_names):
    d = literal_eval_assignment(
        path,
        variable_names
    )

    if not isinstance(d, dict):
        return {}

    out = {}

    for task in TASKS:
        if task not in d:
            continue

        value = d[task]

        if isinstance(value, (int, float)):
            out[task] = float(value)

    return out


def parse_nested_task_scale_floor_dict(path, variable_names):
    d = literal_eval_assignment(
        path,
        variable_names
    )

    if not isinstance(d, dict):
        return {}

    out = {}

    for task in TASKS:
        if task not in d:
            continue

        value = d[task]

        if (
            isinstance(value, dict)
            and "scale_floor" in value
        ):
            out[task] = float(
                value["scale_floor"]
            )

    return out


def parse_explicit_task_scale_floor_regex(path):
    """
    Fallback for a source that contains explicit task blocks but whose
    top-level dict cannot be literal-evaluated due pathlib/other objects.
    """
    require(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    out = {}

    for task in TASKS:
        pattern = re.compile(
            rf'["\']{re.escape(task)}["\']\s*:\s*\{{'
            rf'(?P<body>.{{0,1800}}?)'
            rf'["\']scale_floor["\']\s*:\s*'
            rf'(?P<value>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)',
            flags=re.DOTALL
        )

        m = pattern.search(
            text
        )

        if m:
            out[task] = float(
                m.group("value")
            )

    return out


def parse_fresh_eval():
    # Most likely TASKS/CONFIG style nested dict.
    for names in [
        ["TASKS"],
        ["CONFIG"],
        ["TASK_CONFIG"],
        ["TASK_CONFIGS"],
    ]:
        out = parse_nested_task_scale_floor_dict(
            FRESH_EVAL,
            names
        )

        if len(out) == len(TASKS):
            return out

    out = parse_explicit_task_scale_floor_regex(
        FRESH_EVAL
    )

    return out


def parse_k2b2_v3():
    # Known authoritative form from prior audit:
    # FRESH_CONFIG_SCALE_FLOORS = {
    #   "2s_to_2s": 0.06820264416877736,
    #   "2s_to_5s": 0.06732638773715147
    # }
    out = parse_direct_task_value_dict(
        K2B2_V3,
        [
            "FRESH_CONFIG_SCALE_FLOORS",
            "TASK_SCALE_FLOORS",
            "SCALE_FLOORS",
        ]
    )

    if len(out) == len(TASKS):
        return out

    # fallback direct-task regex
    text = K2B2_V3.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    result = {}

    for task in TASKS:
        m = re.search(
            rf'["\']{re.escape(task)}["\']\s*:\s*'
            rf'(?P<value>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)',
            text
        )

        if m:
            result[task] = float(
                m.group("value")
            )

    return result


def parse_k2b3_build():
    for names in [
        ["TASKS"],
        ["CONFIG"],
        ["TASK_CONFIG"],
        ["TASK_CONFIGS"],
    ]:
        out = parse_nested_task_scale_floor_dict(
            K2B3_BUILD,
            names
        )

        if len(out) == len(TASKS):
            return out

    return parse_explicit_task_scale_floor_regex(
        K2B3_BUILD
    )


def parse_k2b3_lock_json():
    require(K2B3_LOCK)

    data = json.loads(
        K2B3_LOCK.read_text(
            encoding="utf-8"
        )
    )

    out = {}

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if (
                    k in TASKS
                    and isinstance(v, dict)
                    and "scale_floor" in v
                ):
                    try:
                        out[k] = float(
                            v["scale_floor"]
                        )
                    except Exception:
                        pass

                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    return out


def load_final_metadata_values(task):
    p = (
        FINAL_DIR
        / f"{task}_birdsai_trainval_metadata.csv"
    )

    require(p)

    df = pd.read_csv(
        p
    )

    if "scale_floor" not in df.columns:
        raise RuntimeError(
            f"{p} missing scale_floor column."
        )

    vals = pd.to_numeric(
        df["scale_floor"],
        errors="coerce"
    ).dropna().to_numpy(
        dtype=np.float64
    )

    if len(vals) == 0:
        raise RuntimeError(
            f"{p} has no numeric scale_floor values."
        )

    return (
        p,
        np.unique(vals)
    )


def close(a, b, atol):
    return bool(
        np.isclose(
            float(a),
            float(b),
            rtol=0.0,
            atol=atol
        )
    )


def ensure_complete(label, mapping):
    missing = [
        task
        for task in TASKS
        if task not in mapping
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing task scale_floor(s): {missing}. "
            f"Recovered mapping={mapping}"
        )


def main():
    print(
        "\nR9-H-C0-V3 — AUTHORITATIVE TASK SCALE-FLOOR LOCK (PARSER FIX)"
    )

    print(
        "Fixes the V2 parser failure by supporting both nested "
        "task->{scale_floor: value} dictionaries and direct "
        "task->numeric floor dictionaries such as FRESH_CONFIG_SCALE_FLOORS. "
        "Only authoritative frozen sources are used. NO training, NO inference, "
        "NO geometry metrics, NO frozen overwrite.\n"
    )

    verify_r9h_b()

    for p in [
        FRESH_EVAL,
        K2B2_V3,
        K2B3_BUILD,
        K2B3_LOCK,
    ]:
        require(p)

    sources = {
        "fresh_holdout_evaluator":
            parse_fresh_eval(),

        "k2b2_v3_equivalence_audit":
            parse_k2b2_v3(),

        "k2b3_final_builder":
            parse_k2b3_build(),

        "k2b3_lock_json":
            parse_k2b3_lock_json(),
    }

    for label, mapping in sources.items():
        ensure_complete(
            label,
            mapping
        )

    print(
        "RECOVERED AUTHORITATIVE SOURCE MAPPINGS:"
    )

    print(
        json.dumps(
            sources,
            indent=2
        )
    )

    rows = []
    locked = {}

    source_paths = {
        "fresh_holdout_evaluator":
            FRESH_EVAL,

        "k2b2_v3_equivalence_audit":
            K2B2_V3,

        "k2b3_final_builder":
            K2B3_BUILD,

        "k2b3_lock_json":
            K2B3_LOCK,
    }

    for task in TASKS:
        explicit_values = [
            (
                source_name,
                float(
                    mapping[task]
                )
            )
            for source_name, mapping
            in sources.items()
        ]

        reference = explicit_values[0][1]

        for source_name, value in explicit_values[1:]:
            if not close(
                value,
                reference,
                ATOL_EXACT
            ):
                raise RuntimeError(
                    f"{task}: authoritative mismatch: {explicit_values}"
                )

        meta_path, meta_unique = (
            load_final_metadata_values(
                task
            )
        )

        if not all(
            close(
                value,
                reference,
                ATOL_METADATA
            )
            for value in meta_unique
        ):
            raise RuntimeError(
                f"{task}: final metadata values {meta_unique.tolist()} "
                f"do not match locked reference {reference}."
            )

        locked[task] = reference

        for source_name, value in explicit_values:
            rows.append(
                {
                    "task":
                        task,

                    "source_role":
                        source_name,

                    "source_path":
                        str(
                            source_paths[
                                source_name
                            ]
                        ),

                    "value":
                        value,

                    "match_type":
                        "explicit_authoritative_task_floor",

                    "matches_locked_value":
                        True,
                }
            )

        for value in meta_unique:
            rows.append(
                {
                    "task":
                        task,

                    "source_role":
                        "final_neural_trainval_metadata",

                    "source_path":
                        str(
                            meta_path
                        ),

                    "value":
                        float(
                            value
                        ),

                    "match_type":
                        "metadata_unique_value",

                    "matches_locked_value":
                        close(
                            value,
                            reference,
                            ATOL_METADATA
                        ),
                }
            )

    evidence = pd.DataFrame(
        rows
    )

    lock_table = pd.DataFrame(
        [
            {
                "task":
                    task,

                "locked_scale_floor":
                    locked[task],

                "explicit_authoritative_sources":
                    4,

                "all_explicit_sources_match":
                    True,

                "final_metadata_matches":
                    True,

                "exact_source_tolerance":
                    ATOL_EXACT,

                "metadata_tolerance":
                    ATOL_METADATA,
            }
            for task in TASKS
        ]
    )

    evidence_path = (
        OUT
        / "R9H_C0_V3_SCALE_FLOOR_EVIDENCE.csv"
    )

    lock_path = (
        OUT
        / "R9H_C0_V3_LOCKED_TASK_SCALE_FLOORS.csv"
    )

    evidence.to_csv(
        evidence_path,
        index=False
    )

    lock_table.to_csv(
        lock_path,
        index=False
    )

    status = (
        "R9H_C0_V3_AUTHORITATIVE_TASK_SCALE_FLOORS_LOCKED_READY_FOR_R9H_C1_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
    )

    manifest = {
        "step":
            "R9H_C0_V3_AUTHORITATIVE_TASK_SCALE_FLOOR_LOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "locked_scale_floors":
            locked,

        "parser_fix":
            (
                "Supports direct task->numeric dictionaries in addition "
                "to nested task->{scale_floor:value} dictionaries."
            ),

        "new_training":
            False,

        "model_inference":
            False,

        "geometry_distortion_metrics_run":
            False,

        "bootstrap_performed":
            False,

        "frozen_results_modified":
            False,

        "evidence_csv":
            str(
                evidence_path
            ),

        "lock_csv":
            str(
                lock_path
            ),

        "source_hashes": {
            label:
                sha256_file(
                    path
                )
            for label, path
            in source_paths.items()
        },
    }

    (
        OUT
        / "R9H_C0_V3_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nAUTHORITATIVE SCALE-FLOOR LOCK:"
    )

    print(
        lock_table.to_string(
            index=False
        )
    )

    print(
        "\nSOURCE-BY-SOURCE EVIDENCE:"
    )

    print(
        evidence.to_string(
            index=False
        )
    )

    print(
        "\nLOCKED VALUES:"
    )

    print(
        json.dumps(
            locked,
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
        "\nR9-H-C0-V3 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
