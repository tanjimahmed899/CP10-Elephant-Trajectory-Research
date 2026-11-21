
from pathlib import Path
import json
import hashlib
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
    / "R9H_C0_V2_authoritative_scale_floor_lock"
)

OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "2s_to_2s",
    "2s_to_5s",
]

ATOL_EXACT_TEXT = 1e-15
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


def parse_task_scale_floors_from_text(path):
    """
    Parse only explicit task blocks containing scale_floor.
    Avoids unrelated nearby numbers such as quantiles/path statistics.
    """
    require(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    results = {}

    for task in TASKS:
        # Match task label, then a bounded block, then scale_floor numeric literal.
        pattern = re.compile(
            rf'["\']{re.escape(task)}["\']\s*:\s*\{{'
            rf'(?P<body>.{{0,1200}}?)'
            rf'["\']scale_floor["\']\s*:\s*'
            rf'(?P<value>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)',
            flags=re.DOTALL
        )

        m = pattern.search(
            text
        )

        if not m:
            continue

        results[task] = float(
            m.group("value")
        )

    return results


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


def load_k2b3_lock_values():
    require(K2B3_LOCK)

    data = json.loads(
        K2B3_LOCK.read_text(
            encoding="utf-8"
        )
    )

    out = {}

    # Search recursively for a task dictionary with scale_floor.
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


def close(a, b, atol):
    return bool(
        np.isclose(
            float(a),
            float(b),
            rtol=0.0,
            atol=atol
        )
    )


def main():
    print(
        "\nR9-H-C0-V2 — AUTHORITATIVE TASK SCALE-FLOOR LOCK"
    )

    print(
        "This replaces the noisy broad candidate scanner with a source-specific lock. "
        "Only explicit task->scale_floor mappings and frozen final metadata are used. "
        "Quantile rules, path minima/medians, probe values and unrelated nearby numbers "
        "are excluded. NO training, NO model inference, NO geometry metrics, NO frozen overwrite.\n"
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
            parse_task_scale_floors_from_text(
                FRESH_EVAL
            ),

        "k2b2_v3_equivalence_audit":
            parse_task_scale_floors_from_text(
                K2B2_V3
            ),

        "k2b3_final_builder":
            parse_task_scale_floors_from_text(
                K2B3_BUILD
            ),

        "k2b3_lock_json":
            load_k2b3_lock_values(),
    }

    rows = []
    locked = {}

    for task in TASKS:
        explicit_values = []

        for source_name, mapping in sources.items():
            if task not in mapping:
                raise RuntimeError(
                    f"{source_name} did not provide {task} scale_floor."
                )

            explicit_values.append(
                (
                    source_name,
                    float(
                        mapping[task]
                    )
                )
            )

        reference = explicit_values[0][1]

        for source_name, value in explicit_values[1:]:
            if not close(
                value,
                reference,
                ATOL_EXACT_TEXT
            ):
                raise RuntimeError(
                    f"{task}: authoritative explicit-source mismatch: "
                    f"{explicit_values}"
                )

        meta_path, meta_unique = (
            load_final_metadata_values(
                task
            )
        )

        metadata_match = bool(
            all(
                close(
                    value,
                    reference,
                    ATOL_METADATA
                )
                for value in meta_unique
            )
        )

        if not metadata_match:
            raise RuntimeError(
                f"{task}: final trainval metadata scale_floor values "
                f"{meta_unique.tolist()} do not match authoritative "
                f"reference {reference} within {ATOL_METADATA}."
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
                        {
                            "fresh_holdout_evaluator":
                                str(FRESH_EVAL),

                            "k2b2_v3_equivalence_audit":
                                str(K2B2_V3),

                            "k2b3_final_builder":
                                str(K2B3_BUILD),

                            "k2b3_lock_json":
                                str(K2B3_LOCK),
                        }[
                            source_name
                        ],

                    "value":
                        value,

                    "match_type":
                        "explicit_task_scale_floor",

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

                "final_metadata_matches_within_tolerance":
                    True,

                "metadata_tolerance":
                    ATOL_METADATA,
            }
            for task in TASKS
        ]
    )

    evidence_path = (
        OUT
        / "R9H_C0_V2_SCALE_FLOOR_EVIDENCE.csv"
    )

    lock_path = (
        OUT
        / "R9H_C0_V2_LOCKED_TASK_SCALE_FLOORS.csv"
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
        "R9H_C0_V2_AUTHORITATIVE_TASK_SCALE_FLOORS_LOCKED_READY_FOR_R9H_C1_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
    )

    manifest = {
        "step":
            "R9H_C0_V2_AUTHORITATIVE_TASK_SCALE_FLOOR_LOCK",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "locked_scale_floors":
            locked,

        "source_policy":
            (
                "Only explicit task->scale_floor mappings from frozen pipeline "
                "sources plus final neural trainval metadata are accepted. "
                "SCALE_FLOOR_QUANTILE, path-length statistics and probe values are excluded."
            ),

        "authoritative_sources": {
            "fresh_holdout_evaluator":
                {
                    "path":
                        str(FRESH_EVAL),

                    "sha256":
                        sha256_file(
                            FRESH_EVAL
                        ),
                },

            "k2b2_v3_equivalence_audit":
                {
                    "path":
                        str(K2B2_V3),

                    "sha256":
                        sha256_file(
                            K2B2_V3
                        ),
                },

            "k2b3_final_builder":
                {
                    "path":
                        str(K2B3_BUILD),

                    "sha256":
                        sha256_file(
                            K2B3_BUILD
                        ),
                },

            "k2b3_lock_json":
                {
                    "path":
                        str(K2B3_LOCK),

                    "sha256":
                        sha256_file(
                            K2B3_LOCK
                        ),
                },
        },

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
    }

    (
        OUT
        / "R9H_C0_V2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "AUTHORITATIVE SCALE-FLOOR LOCK:"
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
        "\nR9-H-C0-V2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
