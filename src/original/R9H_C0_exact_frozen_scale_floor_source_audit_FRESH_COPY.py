
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

BUILD_SRC = MAIN1 / "build_final_neural_dataset.py"

R9H_B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_B_exact_geometry_formula_lock"
)

R9H_B_MANIFEST = R9H_B_DIR / "R9H_B_MANIFEST.json"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
    / "R9H_C0_exact_scale_floor_source_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

TASK_ALIASES = {
    "2s_to_2s": [
        "2s_to_2s",
        "2s->2s",
        "2s→2s",
        "2s to 2s",
    ],
    "2s_to_5s": [
        "2s_to_5s",
        "2s->5s",
        "2s→5s",
        "2s to 5s",
    ],
}

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".txt",
    ".md",
    ".csv",
}

KEYWORDS = [
    "scale_floor",
    "scale floor",
    "SCALE_FLOOR",
    "SCALE_FLOOR_QUANTILE",
]


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

    require(BUILD_SRC)

    if (
        Path(m["selected_build_src"]).resolve()
        !=
        BUILD_SRC.resolve()
    ):
        raise RuntimeError(
            "R9H-B selected build source does not match expected BUILD_SRC."
        )

    actual_hash = sha256_file(
        BUILD_SRC
    )

    if (
        m.get("selected_build_src_sha256")
        !=
        actual_hash
    ):
        raise RuntimeError(
            "build_final_neural_dataset.py hash changed after R9H-B lock."
        )

    return m


def extract_build_scale_context():
    text = BUILD_SRC.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()
    selected = set()

    for i, line in enumerate(
        lines,
        start=1
    ):
        low = line.lower()

        if (
            "scale_floor" in low
            or "scale floor" in low
            or "scale_floor_quantile" in low
        ):
            for j in range(
                max(1, i - 10),
                min(len(lines), i + 14) + 1
            ):
                selected.add(j)

    rows = [
        {
            "line_number": j,
            "line": lines[j - 1],
        }
        for j in sorted(selected)
    ]

    return pd.DataFrame(
        rows
    )


def detect_task_from_text(text):
    low = text.lower()

    found = []

    for task, aliases in TASK_ALIASES.items():
        if any(
            alias.lower() in low
            for alias in aliases
        ):
            found.append(task)

    return found


def parse_numeric_near_scale_floor(text):
    """
    Extract numeric values that appear close to 'scale_floor' / 'scale floor'.
    Returns evidence snippets, not assumptions.
    """
    rows = []

    lines = text.splitlines()

    num_pat = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"([-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    )

    for i, line in enumerate(
        lines,
        start=1
    ):
        low = line.lower()

        if (
            "scale_floor" not in low
            and "scale floor" not in low
        ):
            continue

        a = max(
            1,
            i - 2
        )

        b = min(
            len(lines),
            i + 2
        )

        snippet = "\n".join(
            f"{j}: {lines[j - 1]}"
            for j in range(
                a,
                b + 1
            )
        )

        tasks = detect_task_from_text(
            snippet
        )

        nums = []

        for j in range(
            a,
            b + 1
        ):
            for m in num_pat.finditer(
                lines[j - 1]
            ):
                try:
                    value = float(
                        m.group(1)
                    )
                except Exception:
                    continue

                if np.isfinite(value):
                    nums.append(
                        value
                    )

        for value in nums:
            # Keep plausible positive trajectory-scale values.
            if value <= 0:
                continue

            rows.append(
                {
                    "line_number":
                        i,

                    "tasks_in_context":
                        "|".join(
                            tasks
                        ),

                    "candidate_value":
                        float(
                            value
                        ),

                    "snippet":
                        snippet,
                }
            )

    return rows


def scan_text_files():
    rows = []

    for root in [
        MAIN1,
        MAIN2,
    ]:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if (
                not p.is_file()
                or p.suffix.lower() not in TEXT_SUFFIXES
            ):
                continue

            # Avoid huge CSV reads as full text.
            try:
                size = p.stat().st_size
            except Exception:
                continue

            if p.suffix.lower() == ".csv":
                if size > 50_000_000:
                    continue

                try:
                    df = pd.read_csv(
                        p,
                        nrows=5
                    )
                except Exception:
                    continue

                scale_cols = [
                    c
                    for c in df.columns
                    if "scale_floor" in str(c).lower()
                    or "scale floor" in str(c).lower()
                ]

                if scale_cols:
                    task_cols = [
                        c
                        for c in df.columns
                        if str(c).lower() in {
                            "task",
                            "horizon",
                            "prediction_task",
                        }
                    ]

                    for c in scale_cols:
                        vals = pd.to_numeric(
                            df[c],
                            errors="coerce"
                        ).dropna()

                        for value in vals.unique():
                            task_text = ""

                            if task_cols:
                                task_text = " | ".join(
                                    str(x)
                                    for x in df[
                                        task_cols[0]
                                    ].dropna().unique()
                                )

                            rows.append(
                                {
                                    "path":
                                        str(p),

                                    "source_type":
                                        "csv_column",

                                    "task_text":
                                        task_text,

                                    "candidate_value":
                                        float(value),

                                    "evidence":
                                        f"column={c}",
                                }
                            )

                continue

            if size > 20_000_000:
                continue

            try:
                text = p.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
            except Exception:
                continue

            low = text.lower()

            if (
                "scale_floor" not in low
                and "scale floor" not in low
            ):
                continue

            for r in parse_numeric_near_scale_floor(
                text
            ):
                rows.append(
                    {
                        "path":
                            str(p),

                        "source_type":
                            p.suffix.lower().lstrip("."),

                        "task_text":
                            r[
                                "tasks_in_context"
                            ],

                        "candidate_value":
                            r[
                                "candidate_value"
                            ],

                        "evidence":
                            r[
                                "snippet"
                            ],
                    }
                )

    return pd.DataFrame(
        rows
    )


def inspect_npz_for_floor_keys():
    rows = []

    for root in [
        MAIN1,
        MAIN2,
    ]:
        if not root.exists():
            continue

        for p in root.rglob(
            "*.npz"
        ):
            name_low = p.name.lower()

            if not any(
                token in name_low
                for token in [
                    "canonical",
                    "neural",
                    "train",
                    "test",
                    "fresh",
                    "dataset",
                ]
            ):
                continue

            try:
                arr = np.load(
                    p,
                    mmap_mode="r"
                )
            except Exception:
                continue

            keys = list(
                arr.files
            )

            floor_keys = [
                k
                for k in keys
                if "scale_floor" in k.lower()
                or "scale floor" in k.lower()
            ]

            for k in floor_keys:
                a = np.asarray(
                    arr[k]
                )

                if a.size == 0:
                    continue

                vals = np.unique(
                    a.astype(
                        np.float64
                    ).ravel()
                )

                for value in vals[:20]:
                    if (
                        np.isfinite(
                            value
                        )
                        and value > 0
                    ):
                        rows.append(
                            {
                                "path":
                                    str(p),

                                "source_type":
                                    "npz_key",

                                "task_text":
                                    "|".join(
                                        detect_task_from_text(
                                            str(p)
                                        )
                                    ),

                                "candidate_value":
                                    float(
                                        value
                                    ),

                                "evidence":
                                    f"key={k}; shape={a.shape}",
                            }
                        )

    return pd.DataFrame(
        rows
    )


def classify_task(row):
    text = (
        str(
            row.get(
                "task_text",
                ""
            )
        )
        +
        "\n"
        +
        str(
            row.get(
                "path",
                ""
            )
        )
        +
        "\n"
        +
        str(
            row.get(
                "evidence",
                ""
            )
        )
    )

    tasks = detect_task_from_text(
        text
    )

    if len(tasks) == 1:
        return tasks[0]

    return ""


def build_candidate_summary(candidates):
    if len(candidates) == 0:
        return pd.DataFrame()

    c = candidates.copy()

    c["task"] = c.apply(
        classify_task,
        axis=1
    )

    # Exclude obvious non-floor constants from summary:
    # quantile fractions, dimensions, years, line numbers etc.
    plausible = c[
        (
            c["candidate_value"]
            >
            1e-8
        )
        &
        (
            c["candidate_value"]
            <
            1.0
        )
    ].copy()

    rows = []

    for task in TASK_ALIASES:
        g = plausible[
            plausible["task"]
            ==
            task
        ].copy()

        if len(g) == 0:
            rows.append(
                {
                    "task":
                        task,

                    "candidate_count":
                        0,

                    "unique_candidate_values":
                        "",

                    "strong_consensus_value":
                        np.nan,

                    "consensus":
                        False,
                }
            )

            continue

        vals = sorted(
            set(
                round(
                    float(v),
                    15
                )
                for v in g[
                    "candidate_value"
                ].tolist()
            )
        )

        # Strong consensus only if exactly one plausible value is found
        # for this task across evidence sources.
        consensus = (
            len(vals)
            ==
            1
        )

        rows.append(
            {
                "task":
                    task,

                "candidate_count":
                    int(
                        len(g)
                    ),

                "unique_candidate_values":
                    " | ".join(
                        f"{v:.15g}"
                        for v in vals
                    ),

                "strong_consensus_value":
                    vals[0]
                    if consensus
                    else np.nan,

                "consensus":
                    bool(
                        consensus
                    ),
            }
        )

    return pd.DataFrame(
        rows
    ), c, plausible


def main():
    print(
        "\nR9-H-C0 — EXACT FROZEN SCALE-FLOOR SOURCE AUDIT"
    )

    print(
        "DIAGNOSTIC ONLY. R9-H-B already locked the exact canonicalizer formula. "
        "This step recovers task-specific scale_floor values from existing frozen "
        "project sources/results before any anisotropy-vs-isotropy canonical geometry "
        "comparison. NO training, NO model inference, NO bootstrap, NO frozen overwrite.\n"
    )

    r9h_b = verify_r9h_b()

    build_context = (
        extract_build_scale_context()
    )

    text_candidates = (
        scan_text_files()
    )

    npz_candidates = (
        inspect_npz_for_floor_keys()
    )

    frames = [
        df
        for df in [
            text_candidates,
            npz_candidates,
        ]
        if len(df)
    ]

    if frames:
        candidates = pd.concat(
            frames,
            ignore_index=True
        )

        candidates = candidates.drop_duplicates(
            subset=[
                "path",
                "source_type",
                "task_text",
                "candidate_value",
                "evidence",
            ]
        ).reset_index(
            drop=True
        )
    else:
        candidates = pd.DataFrame(
            columns=[
                "path",
                "source_type",
                "task_text",
                "candidate_value",
                "evidence",
            ]
        )

    summary, classified, plausible = (
        build_candidate_summary(
            candidates
        )
    )

    build_context_path = (
        OUT
        / "R9H_C0_BUILD_FINAL_SCALE_FLOOR_CONTEXT.csv"
    )

    candidates_path = (
        OUT
        / "R9H_C0_ALL_SCALE_FLOOR_CANDIDATES.csv"
    )

    plausible_path = (
        OUT
        / "R9H_C0_PLAUSIBLE_TASK_SCALE_FLOOR_CANDIDATES.csv"
    )

    summary_path = (
        OUT
        / "R9H_C0_TASK_SCALE_FLOOR_CONSENSUS.csv"
    )

    build_context.to_csv(
        build_context_path,
        index=False
    )

    classified.to_csv(
        candidates_path,
        index=False
    )

    plausible.to_csv(
        plausible_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    both_consensus = bool(
        len(summary) == 2
        and summary["consensus"].all()
        and summary[
            "strong_consensus_value"
        ].notna().all()
    )

    if both_consensus:
        status = (
            "R9H_C0_EXACT_TASK_SCALE_FLOORS_RECOVERED_READY_FOR_R9H_C1_ANISOTROPY_GEOMETRY_DISTORTION_AUDIT"
        )
    else:
        status = (
            "R9H_C0_SCALE_FLOOR_CANDIDATES_AUDITED_NEEDS_EXACT_TASK_FLOOR_RESOLUTION_BEFORE_R9H_C1"
        )

    manifest = {
        "step":
            "R9H_C0_EXACT_FROZEN_SCALE_FLOOR_SOURCE_AUDIT",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "r9h_b_manifest":
            str(
                R9H_B_MANIFEST
            ),

        "r9h_b_manifest_sha256":
            sha256_file(
                R9H_B_MANIFEST
            ),

        "build_source":
            str(
                BUILD_SRC
            ),

        "build_source_sha256":
            sha256_file(
                BUILD_SRC
            ),

        "task_consensus":
            summary.to_dict(
                orient="records"
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

        "outputs": {
            "build_scale_floor_context":
                str(
                    build_context_path
                ),

            "all_candidates":
                str(
                    candidates_path
                ),

            "plausible_candidates":
                str(
                    plausible_path
                ),

            "task_consensus":
                str(
                    summary_path
                ),
        },
    }

    (
        OUT
        / "R9H_C0_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "BUILD_FINAL SCALE-FLOOR CONTEXT:"
    )

    print(
        build_context.to_string(
            index=False
        )
    )

    print(
        "\nTASK SCALE-FLOOR CONSENSUS:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nPLAUSIBLE TASK-LABELED SCALE-FLOOR EVIDENCE:"
    )

    if len(plausible):
        cols = [
            "task",
            "candidate_value",
            "source_type",
            "path",
            "evidence",
        ]

        print(
            plausible[
                cols
            ].head(
                200
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No plausible task-labeled floor evidence found."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR9-H-C0 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
