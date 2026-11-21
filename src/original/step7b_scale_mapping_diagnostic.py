from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

ROOT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")
OUT = DOWNLOADS / "step7_response_aware_warning"
OUT.mkdir(parents=True, exist_ok=True)

GEOFENCE = (
    ROOT / "step18_geofence" / "step18b_evaluation" /
    "step18b_all_geofence_instances.csv"
)

STEP18 = ROOT / "step18b_evaluate_locked_geofence.py"


def norm(x):
    return str(x).strip()


def read_header(path):
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def extract_step18_excerpts():
    if not STEP18.exists():
        return pd.DataFrame()

    txt = STEP18.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    terms = [
        "sample_index",
        "metadata",
        "scale",
        "track_key",
        "source_group",
        "testreal",
        "fresh",
        "npz",
        "csv",
    ]

    rows = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(t in low for t in terms):
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            rows.append({
                "line": i + 1,
                "excerpt": " || ".join(x.strip() for x in lines[start:end]),
            })

    return pd.DataFrame(rows)


def discover_scale_csvs():
    rows = []

    search_roots = [
        ROOT / "final_neural_trajectory_dataset",
        ROOT / "wildbox_fresh_confirmatory_results",
        ROOT / "transformer_results",
        ROOT / "transformer_ensemble_validation",
        ROOT / "step18_geofence",
        ROOT,
    ]

    seen = set()

    for base in search_roots:
        if not base.exists():
            continue

        for p in base.rglob("*.csv"):
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)

            cols = read_header(p)
            lowcols = [str(c).lower() for c in cols]

            if "scale" not in lowcols:
                continue

            try:
                nrows = sum(1 for _ in open(p, "rb")) - 1
            except Exception:
                nrows = None

            rows.append({
                "path": sp,
                "rows": nrows,
                "columns": ";".join(map(str, cols)),
                "has_track_key": "track_key" in lowcols,
                "has_source_group": "source_group" in lowcols,
                "has_sample_id": "sample_id" in lowcols,
                "has_sample_index": "sample_index" in lowcols,
                "has_task": "task" in lowcols,
                "has_dataset": "dataset" in lowcols,
            })

    return pd.DataFrame(rows)


def score_candidate(path, g, dataset, task):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "scale" not in df.columns:
        return None

    gu = (
        g[["sample_index", "track_key", "source_group"]]
        .drop_duplicates("sample_index")
        .sort_values("sample_index")
        .copy()
    )

    n = len(gu)
    idx = gu["sample_index"].astype(int).to_numpy()
    max_idx = int(idx.max()) if len(idx) else -1

    result = {
        "dataset": dataset,
        "task": task,
        "path": str(path),
        "rows": int(len(df)),
        "expected_samples": int(n),
        "sample_index_min": int(idx.min()) if len(idx) else None,
        "sample_index_max": int(max_idx),
        "row_count_exact": bool(len(df) == n),
        "row_index_compatible": bool(len(df) > max_idx),
        "track_set_overlap": None,
        "source_set_overlap": None,
        "direct_index_track_agreement": None,
        "direct_index_source_agreement": None,
        "sequential_track_agreement": None,
        "sequential_source_agreement": None,
        "finite_positive_scale_fraction": None,
        "filename_score": 0.0,
    }

    s = str(path).lower()
    t = str(task).lower()
    d = str(dataset).lower()

    if "2s_to_2s" in t and "2s_to_2s" in s:
        result["filename_score"] += 3
    if "2s_to_5s" in t and "2s_to_5s" in s:
        result["filename_score"] += 3
    if "fresh" in d and ("fresh" in s or "wildbox" in s):
        result["filename_score"] += 3
    if ("testreal" in d or "birdsai" in d) and ("test" in s or "birdsai" in s):
        result["filename_score"] += 3
    if "train" in s and ("testreal" in d or "fresh" in d):
        result["filename_score"] -= 3
    if len(df) == n:
        result["filename_score"] += 4

    scale = pd.to_numeric(df["scale"], errors="coerce")
    result["finite_positive_scale_fraction"] = float(
        np.mean(np.isfinite(scale) & (scale > 0))
    )

    if "track_key" in df.columns:
        gtracks = set(gu["track_key"].map(norm))
        dtracks = set(df["track_key"].map(norm))
        result["track_set_overlap"] = (
            len(gtracks & dtracks) / len(gtracks)
            if gtracks else np.nan
        )

        if len(df) > max_idx:
            a = gu["track_key"].map(norm).to_numpy()
            b = df.iloc[idx]["track_key"].map(norm).to_numpy()
            result["direct_index_track_agreement"] = float(np.mean(a == b))

        if len(df) == n:
            a = gu["track_key"].map(norm).to_numpy()
            b = df["track_key"].map(norm).to_numpy()
            result["sequential_track_agreement"] = float(np.mean(a == b))

    if "source_group" in df.columns:
        gs = set(gu["source_group"].map(norm))
        ds = set(df["source_group"].map(norm))
        result["source_set_overlap"] = (
            len(gs & ds) / len(gs)
            if gs else np.nan
        )

        if len(df) > max_idx:
            a = gu["source_group"].map(norm).to_numpy()
            b = df.iloc[idx]["source_group"].map(norm).to_numpy()
            result["direct_index_source_agreement"] = float(np.mean(a == b))

        if len(df) == n:
            a = gu["source_group"].map(norm).to_numpy()
            b = df["source_group"].map(norm).to_numpy()
            result["sequential_source_agreement"] = float(np.mean(a == b))

    # Composite score for ranking only; no automatic scientific decision.
    for key, weight in [
        ("track_set_overlap", 5),
        ("source_set_overlap", 3),
        ("direct_index_track_agreement", 10),
        ("direct_index_source_agreement", 6),
        ("sequential_track_agreement", 10),
        ("sequential_source_agreement", 6),
        ("finite_positive_scale_fraction", 2),
    ]:
        v = result.get(key)
        if v is not None and np.isfinite(v):
            result["filename_score"] += weight * float(v)

    return result


def main():
    print("\nSTEP 7B SCALE-MAPPING DIAGNOSTIC")

    if not GEOFENCE.exists():
        raise FileNotFoundError(GEOFENCE)

    geo = pd.read_csv(GEOFENCE)

    group_summary = (
        geo.groupby(["dataset", "task"], dropna=False)
        .agg(
            instances=("sample_index", "size"),
            samples=("sample_index", "nunique"),
            tracks=("track_key", "nunique"),
            sources=("source_group", "nunique"),
            sample_index_min=("sample_index", "min"),
            sample_index_max=("sample_index", "max"),
        )
        .reset_index()
    )

    group_summary.to_csv(
        OUT / "step7b_diag_geofence_group_index_ranges.csv",
        index=False
    )

    print("\nGEOFENCE GROUP / SAMPLE INDEX RANGES:")
    print(group_summary.to_string(index=False))

    excerpts = extract_step18_excerpts()
    excerpts.to_csv(
        OUT / "step7b_diag_step18_mapping_excerpts.csv",
        index=False
    )

    print("\nSTEP18 MAPPING EXCERPTS:")
    if len(excerpts):
        print(excerpts.head(80).to_string(index=False))
    else:
        print("NONE")

    scale_files = discover_scale_csvs()
    scale_files.to_csv(
        OUT / "step7b_diag_all_scale_csvs.csv",
        index=False
    )

    print("\nSCALE CSV FILES FOUND:")
    print(scale_files.to_string(index=False) if len(scale_files) else "NONE")

    scored = []

    for (dataset, task), g in geo.groupby(["dataset", "task"]):
        for p in scale_files["path"].tolist() if len(scale_files) else []:
            r = score_candidate(Path(p), g, dataset, task)
            if r is not None:
                scored.append(r)

    scored_df = pd.DataFrame(scored)

    if len(scored_df):
        scored_df = scored_df.sort_values(
            ["dataset", "task", "filename_score", "path"],
            ascending=[True, True, False, True]
        )

    scored_df.to_csv(
        OUT / "step7b_diag_ranked_scale_candidates.csv",
        index=False
    )

    print("\nTOP SCALE CANDIDATES BY DATASET/TASK:")

    if len(scored_df):
        show_cols = [
            "dataset",
            "task",
            "path",
            "rows",
            "expected_samples",
            "sample_index_min",
            "sample_index_max",
            "row_count_exact",
            "row_index_compatible",
            "track_set_overlap",
            "source_set_overlap",
            "direct_index_track_agreement",
            "direct_index_source_agreement",
            "sequential_track_agreement",
            "sequential_source_agreement",
            "finite_positive_scale_fraction",
            "filename_score",
        ]

        top = (
            scored_df
            .groupby(["dataset", "task"], group_keys=False)
            .head(12)
        )

        print(top[show_cols].to_string(index=False))
    else:
        print("NONE")

    with open(
        OUT / "STEP7B_SCALE_DIAGNOSTIC_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("STEP 7B SCALE-MAPPING DIAGNOSTIC\n")
        f.write("=" * 72 + "\n\n")
        f.write("GEOFENCE GROUP / SAMPLE INDEX RANGES\n")
        f.write(group_summary.to_string(index=False))
        f.write("\n\nSTEP18 MAPPING EXCERPTS\n")
        f.write(excerpts.head(120).to_string(index=False) if len(excerpts) else "NONE")
        f.write("\n\nTOP SCALE CANDIDATES\n")
        if len(scored_df):
            f.write(
                scored_df.groupby(["dataset", "task"], group_keys=False)
                .head(12)
                .to_string(index=False)
            )
        else:
            f.write("NONE")

    print("\nDIAGNOSTIC COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
