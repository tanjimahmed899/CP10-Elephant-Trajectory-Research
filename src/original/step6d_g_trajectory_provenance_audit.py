from pathlib import Path
import hashlib
import json
import re
import pandas as pd
import numpy as np


ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

DL = Path(
    r"C:\Users\HP\Downloads"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step6d_uncertainty_holdout"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CANDIDATE_DIRS = [
    ROOT / "trajectory_dataset",
    ROOT / "corrected_trajectory_dataset",
]

# Places that prove a source/track has already been used in model development/evaluation.
USED_DIRS = [
    ROOT / "final_neural_trajectory_dataset",
    ROOT / "unified_trajectory_10hz",
    ROOT / "wildbox_trajectory_dataset",
    ROOT / "wildbox_fresh_confirmatory_results",
    ROOT / "baseline_constant_velocity",
    ROOT / "baseline_kalman",
    ROOT / "lstm_gru_results",
    ROOT / "transformer_results",
    ROOT / "transformer_ensemble_validation",
]

TEXT_EXTS = {
    ".csv", ".json", ".txt", ".md", ".yaml", ".yml"
}

SOURCE_COL_CANDIDATES = [
    "source",
    "source_id",
    "source_group",
    "sequence",
    "sequence_id",
    "video",
    "video_id",
    "clip",
    "clip_id",
    "dataset",
]

TRACK_COL_CANDIDATES = [
    "track_key",
    "track_id",
    "object_id",
    "trajectory_id",
    "id",
]

FRAME_COL_CANDIDATES = [
    "frame",
    "frame_id",
    "frame_idx",
    "frame_index",
    "time",
    "timestamp",
]

XY_PAIRS = [
    ("x", "y"),
    ("cx", "cy"),
    ("center_x", "center_y"),
    ("world_x", "world_y"),
    ("x_corr", "y_corr"),
    ("corrected_x", "corrected_y"),
]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def semicolon(values, max_n=50):
    vals = [str(x) for x in values][:max_n]
    return ";".join(vals)


def inspect_csv(path):
    row = {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 6),
        "sha256": sha256_file(path),
        "rows": None,
        "columns": "",
        "source_columns": "",
        "track_columns": "",
        "frame_columns": "",
        "xy_pairs": "",
        "unique_sources_preview": "",
        "unique_tracks": None,
        "parse_error": "",
    }

    try:
        df = pd.read_csv(path)

        row["rows"] = len(df)
        row["columns"] = semicolon(df.columns)

        source_cols = [
            c for c in SOURCE_COL_CANDIDATES
            if c in df.columns
        ]

        track_cols = [
            c for c in TRACK_COL_CANDIDATES
            if c in df.columns
        ]

        frame_cols = [
            c for c in FRAME_COL_CANDIDATES
            if c in df.columns
        ]

        xy_pairs = [
            f"{x}+{y}"
            for x, y in XY_PAIRS
            if x in df.columns and y in df.columns
        ]

        row["source_columns"] = semicolon(source_cols)
        row["track_columns"] = semicolon(track_cols)
        row["frame_columns"] = semicolon(frame_cols)
        row["xy_pairs"] = semicolon(xy_pairs)

        if source_cols:
            s = (
                df[source_cols[0]]
                .dropna()
                .astype(str)
                .drop_duplicates()
            )
            row["unique_sources_preview"] = semicolon(
                s.tolist(),
                max_n=30
            )

        if track_cols:
            row["unique_tracks"] = int(
                df[track_cols[0]]
                .astype(str)
                .nunique()
            )

    except Exception as e:
        row["parse_error"] = (
            f"{type(e).__name__}: {e}"
        )

    return row


def inspect_npz(path):
    row = {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 6),
        "sha256": sha256_file(path),
        "rows": None,
        "columns": "",
        "source_columns": "",
        "track_columns": "",
        "frame_columns": "",
        "xy_pairs": "",
        "unique_sources_preview": "",
        "unique_tracks": None,
        "parse_error": "",
    }

    try:
        z = np.load(path, allow_pickle=False)
        keys = list(z.files)

        row["columns"] = semicolon(keys)

        # infer sample count from first array
        for k in keys:
            arr = z[k]
            if hasattr(arr, "shape") and len(arr.shape) >= 1:
                row["rows"] = int(arr.shape[0])
                break

        source_cols = [
            c for c in SOURCE_COL_CANDIDATES
            if c in keys
        ]

        track_cols = [
            c for c in TRACK_COL_CANDIDATES
            if c in keys
        ]

        frame_cols = [
            c for c in FRAME_COL_CANDIDATES
            if c in keys
        ]

        row["source_columns"] = semicolon(source_cols)
        row["track_columns"] = semicolon(track_cols)
        row["frame_columns"] = semicolon(frame_cols)

    except Exception as e:
        row["parse_error"] = (
            f"{type(e).__name__}: {e}"
        )

    return row


def inspect_json(path):
    row = {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 6),
        "sha256": sha256_file(path),
        "rows": None,
        "columns": "",
        "source_columns": "",
        "track_columns": "",
        "frame_columns": "",
        "xy_pairs": "",
        "unique_sources_preview": "",
        "unique_tracks": None,
        "parse_error": "",
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            row["columns"] = semicolon(
                sorted(obj.keys())
            )

            if isinstance(obj.get("tracks"), list):
                row["unique_tracks"] = len(
                    obj["tracks"]
                )

        elif isinstance(obj, list):
            row["rows"] = len(obj)
            if obj and isinstance(obj[0], dict):
                row["columns"] = semicolon(
                    sorted(obj[0].keys())
                )

    except Exception as e:
        row["parse_error"] = (
            f"{type(e).__name__}: {e}"
        )

    return row


def inspect_file(path):
    if path.suffix.lower() == ".csv":
        return inspect_csv(path)

    if path.suffix.lower() == ".npz":
        return inspect_npz(path)

    if path.suffix.lower() == ".json":
        return inspect_json(path)

    return {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 6),
        "sha256": sha256_file(path),
        "rows": None,
        "columns": "",
        "source_columns": "",
        "track_columns": "",
        "frame_columns": "",
        "xy_pairs": "",
        "unique_sources_preview": "",
        "unique_tracks": None,
        "parse_error": "",
    }


def gather_source_tokens(inspected_df):
    tokens = set()

    for value in inspected_df[
        "unique_sources_preview"
    ].fillna(""):
        for token in str(value).split(";"):
            token = token.strip()
            if token:
                tokens.add(token)

    return sorted(tokens)


def scan_token_in_used_dirs(token):
    hits = []

    token_low = token.lower()

    for d in USED_DIRS:
        if not d.exists():
            continue

        for p in d.rglob("*"):
            if not p.is_file():
                continue

            path_hit = (
                token_low in str(p).lower()
            )

            content_hit = False
            count = 0

            if p.suffix.lower() in TEXT_EXTS:
                try:
                    if p.stat().st_size <= 50 * 1024 * 1024:
                        txt = p.read_text(
                            encoding="utf-8",
                            errors="ignore"
                        )
                        count = txt.lower().count(
                            token_low
                        )
                        content_hit = count > 0
                except Exception:
                    pass

            if path_hit or content_hit:
                hits.append({
                    "token": token,
                    "path": str(p),
                    "path_hit": path_hit,
                    "content_hit": content_hit,
                    "content_hit_count": count,
                })

    return hits


def main():
    inspected_rows = []

    for d in CANDIDATE_DIRS:
        if not d.exists():
            continue

        for p in d.rglob("*"):
            if not p.is_file():
                continue

            if p.suffix.lower() not in {
                ".csv", ".npz", ".json", ".txt"
            }:
                continue

            row = inspect_file(p)
            row["candidate_dir"] = d.name
            inspected_rows.append(row)

    inspected = pd.DataFrame(
        inspected_rows
    )

    inspected.to_csv(
        OUT
        /
        "step6d_trajectory_candidate_file_audit.csv",
        index=False
    )

    tokens = gather_source_tokens(
        inspected
    )

    token_hits = []

    for token in tokens:
        token_hits.extend(
            scan_token_in_used_dirs(
                token
            )
        )

    token_hits_df = pd.DataFrame(
        token_hits
    )

    if len(token_hits_df) == 0:
        token_hits_df = pd.DataFrame(
            columns=[
                "token",
                "path",
                "path_hit",
                "content_hit",
                "content_hit_count",
            ]
        )

    token_hits_df.to_csv(
        OUT
        /
        "step6d_trajectory_candidate_source_overlap.csv",
        index=False
    )

    source_summary_rows = []

    for token in tokens:
        hits = token_hits_df[
            token_hits_df["token"] == token
        ]

        source_summary_rows.append({
            "source_token": token,
            "used_overlap_files": len(hits),
            "eligible_as_untouched_by_exact_token": (
                len(hits) == 0
            ),
        })

    source_summary = pd.DataFrame(
        source_summary_rows
    )

    source_summary.to_csv(
        OUT
        /
        "step6d_trajectory_candidate_source_summary.csv",
        index=False
    )

    # Dataset-level verdict is conservative.
    candidate_has_structure = False

    if len(inspected):
        candidate_has_structure = bool(
            (
                inspected["frame_columns"].fillna("").str.len() > 0
            ).any()
            and
            (
                (
                    inspected["xy_pairs"].fillna("").str.len() > 0
                )
                |
                (
                    inspected["columns"].fillna("").str.contains(
                        "history|future|trajectory|point|position",
                        case=False,
                        regex=True
                    )
                )
            ).any()
        )

    all_source_tokens_unused = (
        len(source_summary) > 0
        and
        bool(
            source_summary[
                "eligible_as_untouched_by_exact_token"
            ].all()
        )
    )

    if (
        candidate_has_structure
        and
        all_source_tokens_unused
    ):
        verdict = (
            "POTENTIALLY_ELIGIBLE_REQUIRES_PROVENANCE_AUDIT"
        )
        reason = (
            "Trajectory-like structure exists and extracted source tokens "
            "show no exact overlap in known-used model/evaluation directories. "
            "Need provenance audit before locking."
        )

    elif not candidate_has_structure:
        verdict = (
            "NOT_READY_STRUCTURE_INSUFFICIENT_FOR_CONFIRMATORY_HOLDOUT"
        )
        reason = (
            "Files do not clearly expose frame/time plus trajectory coordinates "
            "needed for a confirmatory trajectory holdout."
        )

    else:
        verdict = (
            "NOT_UNTOUCHED_SOURCE_OVERLAP_DETECTED"
        )
        reason = (
            "One or more extracted source tokens already occur in known-used "
            "training/evaluation directories."
        )

    with open(
        OUT
        /
        "STEP6D_G_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-G: TRAJECTORY DATASET PROVENANCE / CONTAMINATION AUDIT\n"
        )
        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "No model predictions or forecast errors were computed.\n\n"
        )

        f.write(
            f"Candidate files inspected: {len(inspected)}\n"
        )
        f.write(
            f"Extracted source tokens: {len(tokens)}\n\n"
        )

        f.write(
            "FILE STRUCTURE AUDIT\n"
        )
        f.write(
            "--------------------\n"
        )

        if len(inspected):
            cols = [
                "candidate_dir",
                "path",
                "rows",
                "columns",
                "source_columns",
                "track_columns",
                "frame_columns",
                "xy_pairs",
                "unique_sources_preview",
                "unique_tracks",
                "parse_error",
            ]

            f.write(
                inspected[cols].to_string(
                    index=False
                )
            )
        else:
            f.write("NONE")

        f.write(
            "\n\nSOURCE OVERLAP SUMMARY\n"
        )
        f.write(
            "----------------------\n"
        )

        if len(source_summary):
            f.write(
                source_summary.to_string(
                    index=False
                )
            )
        else:
            f.write("NO SOURCE TOKENS EXTRACTED")

        f.write(
            "\n\nVERDICT\n"
        )
        f.write(
            "-------\n"
        )

        f.write(
            verdict + "\n"
        )
        f.write(
            reason + "\n"
        )

    print(
        "\nSTEP 6D-G TRAJECTORY DATASET AUDIT COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )

    print(
        "\nFILE STRUCTURE AUDIT:"
    )

    if len(inspected):
        print(
            inspected[
                [
                    "candidate_dir",
                    "path",
                    "rows",
                    "source_columns",
                    "track_columns",
                    "frame_columns",
                    "xy_pairs",
                    "unique_sources_preview",
                    "unique_tracks",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print("NONE")

    print(
        "\nSOURCE OVERLAP SUMMARY:"
    )

    if len(source_summary):
        print(
            source_summary.to_string(
                index=False
            )
        )
    else:
        print(
            "NO SOURCE TOKENS EXTRACTED"
        )

    print(
        "\nVERDICT:"
    )
    print(
        verdict
    )
    print(
        reason
    )


if __name__ == "__main__":
    main()
