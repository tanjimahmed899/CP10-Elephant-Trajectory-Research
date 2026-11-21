from pathlib import Path
import ast
import json
import re
import numpy as np
import pandas as pd

DOWNLOADS = Path(r"C:\Users\HP\Downloads")

READ_ONLY_ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

BAMBI_ROOT = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

K2A_DIR = BAMBI_ROOT / "k2a_10hz"

OUT = BAMBI_ROOT / "k2b_representation_audit"
OUT.mkdir(parents=True, exist_ok=True)

K2A_RESAMPLED = (
    K2A_DIR
    / "step6dk2a_resampled_10hz_world_trajectory.csv"
)

K2A_META_22 = (
    K2A_DIR
    / "2s_to_2s_bambi_flight6_metadata.csv"
)

K2A_META_25 = (
    K2A_DIR
    / "2s_to_5s_bambi_flight6_metadata.csv"
)

SOURCE_TRAJ = (
    BAMBI_ROOT
    / "preprocessing_lock"
    / "step6dk1s_world_trajectory_with_locked_global_segments.csv"
)

KEYWORDS = [
    "canonical",
    "canonicalize",
    "canonicalise",
    "origin",
    "scale",
    "rotation",
    "rotate",
    "theta",
    "atan2",
    "heading",
    "normaliz",
    "standardiz",
    "history[-1]",
    "hist[-1]",
    "last observed",
    "last_observed",
    "relative",
]

HIGH_VALUE_PATTERNS = [
    r"canonical",
    r"atan2",
    r"rotation",
    r"origin",
    r"scale",
    r"history\s*\[\s*-1\s*\]",
    r"hist\s*\[\s*-1\s*\]",
    r"np\.linalg\.norm",
    r"cos\s*\(",
    r"sin\s*\(",
]

MAX_FILE_BYTES = 2_000_000
MAX_SNIPPETS_PER_FILE = 30


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def safe_read_text(path):
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        return None


def line_snippet(lines, lineno, radius=5):
    start = max(0, lineno - 1 - radius)
    end = min(len(lines), lineno + radius)
    out = []
    for i in range(start, end):
        out.append(
            f"{i+1:05d}: {lines[i]}"
        )
    return "\n".join(out)


def score_text(text):
    low = text.lower()
    score = 0

    for k in KEYWORDS:
        if k in low:
            score += 1

    # Stronger weight for likely exact transform ingredients.
    for pat in HIGH_VALUE_PATTERNS:
        if re.search(pat, text, flags=re.I):
            score += 3

    return score


def ast_function_candidates(path, text):
    rows = []

    try:
        tree = ast.parse(text)
    except Exception:
        return rows

    lines = text.splitlines()

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue

        try:
            seg = ast.get_source_segment(text, node)
        except Exception:
            seg = None

        if not seg:
            continue

        score = score_text(seg)

        if score <= 0:
            continue

        rows.append({
            "file": str(path),
            "function": node.name,
            "line_start": int(node.lineno),
            "line_end": int(
                getattr(
                    node,
                    "end_lineno",
                    node.lineno
                )
            ),
            "score": int(score),
            "snippet": line_snippet(
                lines,
                node.lineno,
                radius=8
            ),
        })

    return rows


def scan_python_sources():
    if not READ_ONLY_ROOT.exists():
        raise FileNotFoundError(
            f"Read-only project root not found:\n{READ_ONLY_ROOT}"
        )

    py_files = list(
        READ_ONLY_ROOT.rglob("*.py")
    )

    file_rows = []
    function_rows = []

    for path in py_files:
        text = safe_read_text(path)

        if text is None:
            continue

        s = score_text(text)

        if s > 0:
            file_rows.append({
                "file": str(path),
                "score": int(s),
                "size_bytes": int(path.stat().st_size),
            })

        function_rows.extend(
            ast_function_candidates(
                path,
                text
            )
        )

    files_df = pd.DataFrame(file_rows)

    if len(files_df):
        files_df = files_df.sort_values(
            ["score", "file"],
            ascending=[False, True]
        )

    funcs_df = pd.DataFrame(function_rows)

    if len(funcs_df):
        funcs_df = funcs_df.sort_values(
            ["score", "file", "line_start"],
            ascending=[False, True, True]
        )

    return py_files, files_df, funcs_df


def extract_keyword_snippets(files_df):
    rows = []

    for _, r in files_df.head(30).iterrows():
        path = Path(r["file"])
        text = safe_read_text(path)

        if text is None:
            continue

        lines = text.splitlines()
        seen = set()
        count = 0

        for i, line in enumerate(lines, start=1):
            low = line.lower()

            if not any(k in low for k in KEYWORDS):
                continue

            key = (str(path), i)

            if key in seen:
                continue

            seen.add(key)

            rows.append({
                "file": str(path),
                "line": int(i),
                "text": line.strip(),
                "context": line_snippet(
                    lines,
                    i,
                    radius=4
                ),
            })

            count += 1

            if count >= MAX_SNIPPETS_PER_FILE:
                break

    return pd.DataFrame(rows)


def inspect_npz_and_metadata():
    npz_rows = []
    meta_rows = []

    # Search only likely canonical/trajectory NPZ files.
    candidates = []

    for p in READ_ONLY_ROOT.rglob("*.npz"):
        name = p.name.lower()

        if (
            "canonical" in name
            or "2s_to_2s" in name
            or "2s_to_5s" in name
            or "trajectory" in name
        ):
            candidates.append(p)

    for path in candidates[:300]:
        try:
            with np.load(
                path,
                allow_pickle=False
            ) as z:
                keys = list(z.files)
                shapes = {}

                for k in keys:
                    try:
                        arr = z[k]
                        shapes[k] = list(arr.shape)
                    except Exception:
                        shapes[k] = "UNREADABLE"

                npz_rows.append({
                    "file": str(path),
                    "keys": json.dumps(keys),
                    "shapes": json.dumps(shapes),
                })
        except Exception as e:
            npz_rows.append({
                "file": str(path),
                "keys": "ERROR",
                "shapes": str(e),
            })

    # Search matching metadata CSVs.
    for path in READ_ONLY_ROOT.rglob("*.csv"):
        name = path.name.lower()

        if not (
            "metadata" in name
            or "canonical" in name
            or "scale" in name
        ):
            continue

        try:
            df = pd.read_csv(
                path,
                nrows=5
            )

            cols = list(df.columns)

            interesting = [
                c
                for c in cols
                if any(
                    k in c.lower()
                    for k in [
                        "scale",
                        "origin",
                        "theta",
                        "rotation",
                        "heading",
                        "track",
                        "source",
                    ]
                )
            ]

            meta_rows.append({
                "file": str(path),
                "columns": json.dumps(cols),
                "interesting_columns": json.dumps(interesting),
            })
        except Exception:
            pass

    return (
        pd.DataFrame(npz_rows),
        pd.DataFrame(meta_rows)
    )


def inspect_bambi_track_loss():
    src = pd.read_csv(SOURCE_TRAJ)
    res = pd.read_csv(K2A_RESAMPLED)

    src_tracks = set(
        src["track_id"]
        .astype(int)
        .unique()
        .tolist()
    )

    res_tracks = set(
        res["track_id"]
        .astype(int)
        .unique()
        .tolist()
    )

    missing = sorted(
        src_tracks
        -
        res_tracks
    )

    rows = []

    for tid in missing:
        g = src[
            src["track_id"].astype(int)
            ==
            int(tid)
        ].sort_values(
            "frame_id"
        )

        rows.append({
            "track_id":
                int(tid),

            "rows":
                int(
                    len(g)
                ),

            "frame_min":
                int(
                    g["frame_id"].min()
                ),

            "frame_max":
                int(
                    g["frame_id"].max()
                ),

            "global_segments":
                json.dumps(
                    sorted(
                        g[
                            "global_segment_id"
                        ]
                        .astype(int)
                        .unique()
                        .tolist()
                    )
                ),
        })

    return pd.DataFrame(rows)


def main():
    print(
        "\nSTEP 6D-K2B0 — EXACT CANONICAL REPRESENTATION SOURCE AUDIT"
    )

    print(
        "Read-only audit only. No BAMBI canonicalization or model evaluation.\n"
    )

    for p in [
        K2A_RESAMPLED,
        K2A_META_22,
        K2A_META_25,
        SOURCE_TRAJ,
    ]:
        require(p)

    py_files, files_df, funcs_df = scan_python_sources()

    snippets_df = extract_keyword_snippets(
        files_df
    )

    npz_df, meta_df = inspect_npz_and_metadata()

    missing_tracks_df = inspect_bambi_track_loss()

    files_df.to_csv(
        OUT
        /
        "step6dk2b0_ranked_python_files.csv",
        index=False
    )

    funcs_df.to_csv(
        OUT
        /
        "TABLE_step6dk2b0_ranked_transform_functions.csv",
        index=False
    )

    snippets_df.to_csv(
        OUT
        /
        "step6dk2b0_keyword_source_snippets.csv",
        index=False
    )

    npz_df.to_csv(
        OUT
        /
        "step6dk2b0_candidate_npz_inventory.csv",
        index=False
    )

    meta_df.to_csv(
        OUT
        /
        "step6dk2b0_candidate_metadata_inventory.csv",
        index=False
    )

    missing_tracks_df.to_csv(
        OUT
        /
        "step6dk2b0_bambi_track_loss_audit.csv",
        index=False
    )

    top_files = (
        files_df.head(20)
        if len(files_df)
        else pd.DataFrame()
    )

    top_funcs = (
        funcs_df.head(30)
        if len(funcs_df)
        else pd.DataFrame()
    )

    # Look for especially promising exact transform functions.
    exact_candidates = pd.DataFrame()

    if len(funcs_df):
        mask = (
            funcs_df["function"]
            .str.lower()
            .str.contains(
                "canon|normal|transform|window|dataset|represent|rotate|scale",
                regex=True
            )
        )

        exact_candidates = (
            funcs_df[mask]
            .head(20)
            .copy()
        )

    summary = {
        "step":
            "STEP_6D_K2B0_EXACT_CANONICAL_REPRESENTATION_SOURCE_AUDIT",

        "python_files_scanned":
            int(
                len(py_files)
            ),

        "keyword_positive_python_files":
            int(
                len(files_df)
            ),

        "candidate_functions":
            int(
                len(funcs_df)
            ),

        "candidate_npz_files":
            int(
                len(npz_df)
            ),

        "candidate_metadata_files":
            int(
                len(meta_df)
            ),

        "bambi_source_tracks":
            int(
                pd.read_csv(
                    SOURCE_TRAJ
                )[
                    "track_id"
                ].nunique()
            ),

        "bambi_resampled_tracks":
            int(
                pd.read_csv(
                    K2A_RESAMPLED
                )[
                    "track_id"
                ].nunique()
            ),

        "bambi_missing_resampled_track_ids":
            (
                missing_tracks_df[
                    "track_id"
                ].astype(int).tolist()
                if len(missing_tracks_df)
                else []
            ),

        "status":
            (
                "SOURCE_CANDIDATES_FOUND_PENDING_EXACT_FORMULA_CONFIRMATION"
                if len(funcs_df)
                else
                "NO_TRANSFORM_FUNCTION_FOUND_MANUAL_SOURCE_AUDIT_REQUIRED"
            ),
    }

    with open(
        OUT
        /
        "STEP6D_K2B0_SUMMARY.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    with open(
        OUT
        /
        "STEP6D_K2B0_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-K2B0 — EXACT CANONICAL REPRESENTATION SOURCE AUDIT\n"
        )

        f.write(
            "="
            *
            84
            +
            "\n\n"
        )

        f.write(
            "SUMMARY\n-------\n"
        )

        f.write(
            json.dumps(
                summary,
                indent=2
            )
        )

        f.write(
            "\n\nTOP RANKED PYTHON FILES\n"
        )

        f.write(
            "-----------------------\n"
        )

        f.write(
            top_files.to_string(
                index=False
            )
            if len(top_files)
            else "NONE"
        )

        f.write(
            "\n\nTOP RANKED TRANSFORM FUNCTIONS\n"
        )

        f.write(
            "------------------------------\n"
        )

        if len(top_funcs):
            for _, r in top_funcs.iterrows():
                f.write(
                    f"\nFILE: {r['file']}\n"
                    f"FUNCTION: {r['function']}\n"
                    f"LINES: {r['line_start']}-{r['line_end']}\n"
                    f"SCORE: {r['score']}\n"
                    f"{r['snippet']}\n"
                    + "-" * 72
                    + "\n"
                )
        else:
            f.write(
                "NONE\n"
            )

        f.write(
            "\n\nHIGH-PROBABILITY EXACT CANDIDATES\n"
        )

        f.write(
            "---------------------------------\n"
        )

        if len(exact_candidates):
            for _, r in exact_candidates.iterrows():
                f.write(
                    f"\nFILE: {r['file']}\n"
                    f"FUNCTION: {r['function']}\n"
                    f"LINES: {r['line_start']}-{r['line_end']}\n"
                    f"SCORE: {r['score']}\n"
                    f"{r['snippet']}\n"
                    + "-" * 72
                    + "\n"
                )
        else:
            f.write(
                "NONE\n"
            )

        f.write(
            "\n\nBAMBI TRACK-LOSS AUDIT\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            missing_tracks_df.to_string(
                index=False
            )
            if len(missing_tracks_df)
            else "NO TRACK LOSS"
        )

        f.write(
            "\n\nSTATUS\n------\n"
        )

        f.write(
            summary[
                "status"
            ]
            +
            "\n"
        )

        f.write(
            "\nIMPORTANT: No canonical formula is inferred or applied in K2B0. "
            "The exact source implementation must be confirmed before BAMBI is "
            "transformed for the frozen models.\n"
        )

    print(
        "SUMMARY:"
    )

    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    print(
        "\nTOP RANKED PYTHON FILES:"
    )

    if len(top_files):
        print(
            top_files.to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nHIGH-PROBABILITY EXACT CANDIDATES:"
    )

    if len(exact_candidates):
        print(
            exact_candidates[
                [
                    "file",
                    "function",
                    "line_start",
                    "line_end",
                    "score",
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
        "\nBAMBI TRACK-LOSS AUDIT:"
    )

    print(
        missing_tracks_df.to_string(
            index=False
        )
        if len(missing_tracks_df)
        else "NO TRACK LOSS"
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
        "\nSTEP 6D-K2B0 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
