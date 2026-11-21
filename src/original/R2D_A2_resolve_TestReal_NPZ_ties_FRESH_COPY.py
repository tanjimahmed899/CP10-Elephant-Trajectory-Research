from pathlib import Path
import ast
import hashlib
import json
import os

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

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

SELECTED_CSV = (
    R2D_A_DIR
    / "R2D_A_selected_npz.csv"
)

EVALUATOR = (
    MAIN_RESULT
    / "evaluate_birdsai_testreal_frozen_ensemble.py"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R2_fair_forecasting"
    / "R2D_targeted_recompute_preflight_v2"
)

OUT.mkdir(parents=True, exist_ok=True)


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


def npz_signature(path):
    z = np.load(path, allow_pickle=False)

    sig = {}

    for key in sorted(z.files):
        arr = np.asarray(z[key])

        h = hashlib.sha256(
            arr.tobytes(order="C")
        ).hexdigest()

        sig[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "array_sha256": h,
        }

    return sig


def signatures_equal(sig_a, sig_b):
    return sig_a == sig_b


def root_preference(path):
    s = os.path.normcase(str(path))

    mr2 = os.path.normcase(str(MAIN_RESULT_2))
    mr = os.path.normcase(str(MAIN_RESULT))

    if s.startswith(mr2):
        return 0

    if s.startswith(mr):
        return 1

    return 9


def extract_evaluator_literals():
    require(EVALUATOR)

    text = EVALUATOR.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(text)

    literals = []

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and
            isinstance(node.value, str)
        ):
            value = node.value

            if (
                ".npz" in value.lower()
                or
                "testreal" in value.lower()
                or
                "canonical" in value.lower()
            ):
                literals.append({
                    "line": int(
                        getattr(
                            node,
                            "lineno",
                            -1
                        )
                    ),
                    "value": value,
                })

    return literals


def evaluator_path_evidence(path, literals):
    path_str = str(path).lower()
    name = path.name.lower()

    score = 0
    matched = []

    for item in literals:
        value = str(item["value"]).lower()

        if name in value:
            score += 100
            matched.append(item)

        elif (
            "birdsai"
            in value
            and
            "test"
            in value
            and
            "canonical"
            in value
            and
            "birdsai"
            in path_str
            and
            "test"
            in path_str
            and
            "canonical"
            in path_str
        ):
            score += 10
            matched.append(item)

    return score, matched


def load_top_ties():
    require(CANDIDATE_CSV)

    df = pd.read_csv(CANDIDATE_CSV)

    out = []

    for (dataset, task), group in df.groupby(
        [
            "target_dataset",
            "target_task",
        ],
        sort=False,
    ):
        if dataset != "BIRDSAI_TestReal":
            continue

        top_score = group["score"].max()

        top = group[
            group["score"]
            ==
            top_score
        ].copy()

        out.append(
            (
                dataset,
                task,
                top_score,
                top,
            )
        )

    return out


def resolve_tie(
    dataset,
    task,
    top_score,
    top,
    evaluator_literals,
):
    records = []

    for _, row in top.iterrows():
        path = Path(
            row["path"]
        )

        require(path)

        file_hash = sha256_file(path)
        sig = npz_signature(path)
        ev_score, ev_matches = evaluator_path_evidence(
            path,
            evaluator_literals,
        )

        records.append({
            "dataset": dataset,
            "task": task,
            "path": str(path),
            "score": int(top_score),
            "file_sha256": file_hash,
            "npz_signature": sig,
            "evaluator_evidence_score": int(ev_score),
            "evaluator_matches": ev_matches,
            "root_preference": root_preference(path),
        })

    # Case 1: byte-identical copies.
    hashes = {
        r["file_sha256"]
        for r in records
    }

    if len(hashes) == 1:
        selected = sorted(
            records,
            key=lambda r: (
                r["root_preference"],
                r["path"].lower(),
            )
        )[0]

        reason = (
            "All top-scoring TestReal NPZ candidates are byte-identical "
            "by SHA256; one canonical path selected deterministically."
        )

        return selected, records, reason

    # Case 2: file-level hashes differ, but all arrays are identical.
    first_sig = records[0]["npz_signature"]

    if all(
        signatures_equal(
            first_sig,
            r["npz_signature"],
        )
        for r in records[1:]
    ):
        selected = sorted(
            records,
            key=lambda r: (
                r["root_preference"],
                r["path"].lower(),
            )
        )[0]

        reason = (
            "Top-scoring files differ at container-byte level but contain "
            "identical NPZ array names/shapes/dtypes/values; canonical path "
            "selected deterministically."
        )

        return selected, records, reason

    # Case 3: contents differ. Require original evaluator provenance.
    max_ev = max(
        r["evaluator_evidence_score"]
        for r in records
    )

    best = [
        r
        for r in records
        if r["evaluator_evidence_score"]
        ==
        max_ev
    ]

    if max_ev > 0 and len(best) == 1:
        selected = best[0]

        reason = (
            "Top-scoring candidates contain different arrays; selected the "
            "unique candidate supported by the original frozen evaluator's "
            "source literals/provenance."
        )

        return selected, records, reason

    raise RuntimeError(
        f"Could not safely resolve TestReal NPZ tie for {task}. "
        "Candidates are not identical and evaluator provenance is not unique."
    )


def main():
    print(
        "\nR2D-A2 — TESTREAL NPZ TIE RESOLUTION / EXACT ADAPTER PATCH"
    )

    print(
        "NO new forecasting metric is generated. "
        "This step resolves only the two ambiguous TestReal NPZ selections.\n"
    )

    require(SELECTED_CSV)
    require(EVALUATOR)

    evaluator_literals = extract_evaluator_literals()

    ties = load_top_ties()

    if len(ties) != 2:
        raise RuntimeError(
            f"Expected exactly two TestReal task tie groups; found {len(ties)}."
        )

    selected_rows = []
    evidence_rows = []

    for dataset, task, top_score, top in ties:
        selected, records, reason = resolve_tie(
            dataset,
            task,
            top_score,
            top,
            evaluator_literals,
        )

        selected_rows.append({
            "dataset": dataset,
            "task": task,
            "selected_path": selected["path"],
            "selected_sha256": selected["file_sha256"],
            "resolution_reason": reason,
            "candidate_count": len(records),
            "evaluator_evidence_score": selected[
                "evaluator_evidence_score"
            ],
        })

        for r in records:
            evidence_rows.append({
                "dataset": dataset,
                "task": task,
                "path": r["path"],
                "file_sha256": r["file_sha256"],
                "evaluator_evidence_score": r[
                    "evaluator_evidence_score"
                ],
                "root_preference": r["root_preference"],
                "npz_signature_json": json.dumps(
                    r["npz_signature"],
                    sort_keys=True,
                ),
                "evaluator_matches_json": json.dumps(
                    r["evaluator_matches"],
                ),
            })

    selected_df = pd.DataFrame(selected_rows)
    evidence_df = pd.DataFrame(evidence_rows)

    selected_path = (
        OUT
        / "R2D_A2_resolved_testreal_npz.csv"
    )

    evidence_path = (
        OUT
        / "R2D_A2_testreal_npz_tie_evidence.csv"
    )

    selected_df.to_csv(
        selected_path,
        index=False,
    )

    evidence_df.to_csv(
        evidence_path,
        index=False,
    )

    summary = {
        "step":
            "R2D_A2_TESTREAL_NPZ_TIE_RESOLUTION",

        "status":
            "R2D_A2_TESTREAL_NPZ_TIES_RESOLVED_READY_FOR_TARGETED_RECOMPUTATION",

        "new_metrics_generated":
            False,

        "evaluator_source":
            str(EVALUATOR),

        "evaluator_sha256":
            sha256_file(EVALUATOR),

        "resolved_testreal_npz_csv":
            str(selected_path),

        "tie_evidence_csv":
            str(evidence_path),
    }

    summary_path = (
        OUT
        / "R2D_A2_SUMMARY.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "RESOLVED TESTREAL NPZs:"
    )

    print(
        selected_df.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )
    print(
        summary["status"]
    )

    print(
        "\nR2D-A2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
