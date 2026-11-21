
from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R12_ROOT = REVISION_ROOT / "R12_track_accounting"

V10_DIR = R12_ROOT / "R12E_v10_direct_unified10hz_builder_audit"
V10_MANIFEST = V10_DIR / "R12E_V10_MANIFEST.json"

R12C_V2_DIR = R12_ROOT / "R12C_v2_resolve_45_total_tracks_with_canonical_id_mapping"
R12C_V2_MANIFEST = R12C_V2_DIR / "R12C_V2_MANIFEST.json"

V3_DIR = R12_ROOT / "R12E_v3_robust_segment_rule_reconstruction"
V3_MANIFEST = V3_DIR / "R12E_V3_MANIFEST.json"

UNIFIED_BUILDER = MAIN1 / "build_unified_10hz_dataset.py"
CLEANED = MAIN1 / "trajectory_dataset" / "cleaned_track_points.csv"
CORR_2 = MAIN1 / "corrected_trajectory_dataset" / "corrected_windows_2s_to_2s.csv"
CORR_5 = MAIN1 / "corrected_trajectory_dataset" / "corrected_windows_2s_to_5s.csv"
FINAL_2 = MAIN1 / "final_neural_trajectory_dataset" / "2s_to_2s_birdsai_test_metadata.csv"
FINAL_5 = MAIN1 / "final_neural_trajectory_dataset" / "2s_to_5s_birdsai_test_metadata.csv"

OUT = R12_ROOT / "R12E_v11_full_exact_track_accounting_reconstruction"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    "total_original_tracks": 45,
    "2s": {"prefilter_rows": 1070, "final_rows": 1053, "final_tracks": 37, "final_segments": 89},
    "5s": {"prefilter_rows": 827, "final_rows": 826, "final_tracks": 31, "final_segments": 63},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_upstream():
    for p in [
        V10_MANIFEST, R12C_V2_MANIFEST, V3_MANIFEST,
        UNIFIED_BUILDER, CLEANED, CORR_2, CORR_5, FINAL_2, FINAL_5
    ]:
        require(p)

    v10 = json.loads(V10_MANIFEST.read_text(encoding="utf-8"))
    r12c = json.loads(R12C_V2_MANIFEST.read_text(encoding="utf-8"))
    v3 = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))

    if v10.get("status") != "R12E_V10_UNIFIED10HZ_PRODUCER_AND_SINGLE_EXACT_FILTER_RULE_LOCKED_READY_FOR_R12E_V11_FULL_RECONSTRUCTION":
        raise RuntimeError(f"Unexpected V10 status: {v10.get('status')}")

    if r12c.get("status") != "R12C_V2_45_TOTAL_TRACKS_37_2S_ELIGIBLE_31_5S_ELIGIBLE_EXACTLY_RESOLVED_READY_FOR_R12D_SEGMENT_BUILD_RULE_LOCK":
        raise RuntimeError(f"Unexpected R12C-V2 status: {r12c.get('status')}")

    if v3.get("status") != "R12E_V3_RULE_EXTRACTED_BUT_RECONSTRUCTION_NOT_EXACT_REVIEW_BEFORE_R12F":
        raise RuntimeError(f"Unexpected V3 status: {v3.get('status')}")


def canonical_id(value):
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        x = float(s)
        if np.isfinite(x) and x.is_integer():
            return str(int(x))
    except Exception:
        pass
    return s


def normalize(df):
    out = df.copy()

    if "sequence" in out.columns:
        out["sequence_canon"] = out["sequence"].astype(str).str.strip()

    if "object_id" in out.columns:
        out["object_id_canon"] = out["object_id"].map(canonical_id)

    if "segment_id" in out.columns:
        out["segment_id_canon"] = out["segment_id"].astype(str).str.strip()

    for c in ["obs_start_frame", "obs_end_frame", "future_end_frame"]:
        if c in out.columns:
            vals = pd.to_numeric(out[c], errors="coerce")
            out[c + "_canon"] = vals.map(
                lambda x: "" if pd.isna(x) else str(int(x))
            )

    if {"sequence_canon", "object_id_canon"}.issubset(out.columns):
        out["original_track_key_canon"] = out["sequence_canon"] + "|" + out["object_id_canon"]

    if {"sequence_canon", "object_id_canon", "segment_id_canon"}.issubset(out.columns):
        out["segment_key_canon"] = (
            out["sequence_canon"] + "|" + out["object_id_canon"] + "|" + out["segment_id_canon"]
        )

    return out


def testreal_mask(df):
    if "split" not in df.columns:
        raise RuntimeError("Missing split column.")

    s = df["split"].astype(str).str.strip().str.lower()
    exact = s.isin(["test_real", "testreal", "test-real"])
    if exact.any():
        return exact
    return s.str.contains("test", regex=False)


def physical_key(df):
    required = [
        "sequence_canon", "object_id_canon", "segment_id_canon",
        "obs_start_frame_canon", "future_end_frame_canon",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Cannot build physical key; missing {missing}")

    key = df[required[0]].astype(str)
    for c in required[1:]:
        key = key + "|" + df[c].astype(str)
    return key


def extract_exact_filter_from_source():
    text = UNIFIED_BUILDER.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text, filename=str(UNIFIED_BUILDER))

    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "core" not in target_names:
            continue

        src = ast.get_source_segment(text, node) or ""
        low = src.lower()

        matches.append(
            {
                "line": int(node.lineno),
                "end_line": int(getattr(node, "end_lineno", node.lineno)),
                "expression": src,
                "accepted_eq_1": ("accepted" in low and "==" in src and "1" in src),
                "quality_tier_eq_core": ("quality_tier" in low and "core" in low and "==" in src),
                "sample_id_ge_0": ("sample_id" in low and ">=" in src and "0" in src),
            }
        )

    df = pd.DataFrame(matches)

    if len(df) != 1:
        raise RuntimeError(f"Expected exactly one assignment to core, found {len(df)}")

    row = df.iloc[0]
    for c in ["accepted_eq_1", "quality_tier_eq_core", "sample_id_ge_0"]:
        if not bool(row[c]):
            raise RuntimeError(f"Source core assignment failed condition check: {c}")

    return df


def apply_exact_filter(test_corr):
    required = {"accepted", "quality_tier", "sample_id"}
    missing = required - set(test_corr.columns)
    if missing:
        raise RuntimeError(f"Corrected metadata missing filter columns: {sorted(missing)}")

    accepted = pd.to_numeric(test_corr["accepted"], errors="coerce") == 1
    core = test_corr["quality_tier"].astype(str).str.strip().str.lower() == "core"
    nonnegative_id = pd.to_numeric(test_corr["sample_id"], errors="coerce") >= 0

    mask = accepted & core & nonnegative_id
    return test_corr.loc[mask].copy(), mask


def stats(df):
    return {
        "rows": int(len(df)),
        "tracks": int(df["original_track_key_canon"].nunique()),
        "segments": int(df["segment_key_canon"].nunique()),
    }


def compare_filtered_to_final(task, filtered, final):
    f = filtered.copy()
    z = final.copy()

    f["_physical_key"] = physical_key(f)
    z["_physical_key"] = physical_key(z)

    fset = set(f["_physical_key"].astype(str))
    zset = set(z["_physical_key"].astype(str))

    fs = stats(f)
    zs = stats(z)

    only_filtered = fset - zset
    only_final = zset - fset

    summary = {
        "task": task,
        "filtered_rows": fs["rows"],
        "final_rows": zs["rows"],
        "filtered_tracks": fs["tracks"],
        "final_tracks": zs["tracks"],
        "filtered_segments": fs["segments"],
        "final_segments": zs["segments"],
        "filtered_unique_windows": len(fset),
        "final_unique_windows": len(zset),
        "only_filtered": len(only_filtered),
        "only_final": len(only_final),
        "exact_physical_window_set_match": fset == zset,
    }

    return summary


def removed_reason_summary(task, removed):
    rows = []

    for col in ["accepted", "quality_tier"]:
        if col not in removed.columns:
            continue

        vc = removed[col].astype(str).value_counts(dropna=False)
        for value, count in vc.items():
            rows.append(
                {
                    "task": task,
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "min": np.nan,
                    "median": np.nan,
                    "max": np.nan,
                }
            )

    if "repaired_motion_pct" in removed.columns:
        vals = pd.to_numeric(removed["repaired_motion_pct"], errors="coerce")
        rows.append(
            {
                "task": task,
                "column": "repaired_motion_pct",
                "value": "numeric_summary",
                "count": int(vals.notna().sum()),
                "min": float(vals.min()) if vals.notna().any() else np.nan,
                "median": float(vals.median()) if vals.notna().any() else np.nan,
                "max": float(vals.max()) if vals.notna().any() else np.nan,
            }
        )

    return rows


def count_total_testreal_original_tracks():
    df = pd.read_csv(CLEANED, low_memory=False)

    labels = df["split"].astype(str).str.strip().str.lower()
    mask = labels.isin(["test_real", "testreal", "test-real"])
    if not mask.any():
        mask = labels.str.contains("test", regex=False)

    t = df.loc[mask].copy()
    t["sequence_canon"] = t["sequence"].astype(str).str.strip()
    t["object_id_canon"] = t["object_id"].map(canonical_id)
    t["track"] = t["sequence_canon"] + "|" + t["object_id_canon"]

    return int(t["track"].nunique())


def main():
    print("\nR12-E-V11 — FULL EXACT TRACK-ACCOUNTING RECONSTRUCTION")
    print(
        "NO training and NO inference. Applies the exact source-implemented core filter "
        "(accepted == 1 AND quality_tier == 'core' AND sample_id >= 0) to TestReal corrected "
        "windows and verifies zero-mismatch physical window identity against frozen final metadata. "
        "No original file is modified.\n"
    )

    verify_upstream()
    source_filter_df = extract_exact_filter_from_source()

    corr2 = normalize(pd.read_csv(CORR_2, low_memory=False))
    corr5 = normalize(pd.read_csv(CORR_5, low_memory=False))
    final2 = normalize(pd.read_csv(FINAL_2, low_memory=False))
    final5 = normalize(pd.read_csv(FINAL_5, low_memory=False))

    test2 = corr2.loc[testreal_mask(corr2)].copy()
    test5 = corr5.loc[testreal_mask(corr5)].copy()

    filt2, mask2 = apply_exact_filter(test2)
    filt5, mask5 = apply_exact_filter(test5)

    removed2 = test2.loc[~mask2].copy()
    removed5 = test5.loc[~mask5].copy()

    cmp2 = compare_filtered_to_final("2s_to_2s", filt2, final2)
    cmp5 = compare_filtered_to_final("2s_to_5s", filt5, final5)

    comparison_df = pd.DataFrame([cmp2, cmp5])

    removed_reason_df = pd.DataFrame(
        removed_reason_summary("2s_to_2s", removed2)
        + removed_reason_summary("2s_to_5s", removed5)
    )

    total_original_tracks = count_total_testreal_original_tracks()

    chain_df = pd.DataFrame(
        [
            {
                "stage": "BIRDSAI TestReal original dataset-level tracks",
                "2s_to_2s": total_original_tracks,
                "2s_to_5s": total_original_tracks,
                "unit": "unique (sequence, object_id)",
            },
            {
                "stage": "Corrected TestReal pre-filter windows",
                "2s_to_2s": len(test2),
                "2s_to_5s": len(test5),
                "unit": "windows",
            },
            {
                "stage": "Exact source-rule retained windows",
                "2s_to_2s": len(filt2),
                "2s_to_5s": len(filt5),
                "unit": "windows",
            },
            {
                "stage": "Final represented original tracks",
                "2s_to_2s": filt2["original_track_key_canon"].nunique(),
                "2s_to_5s": filt5["original_track_key_canon"].nunique(),
                "unit": "unique (sequence, object_id)",
            },
            {
                "stage": "Final forecastable segments / tracklets",
                "2s_to_2s": filt2["segment_key_canon"].nunique(),
                "2s_to_5s": filt5["segment_key_canon"].nunique(),
                "unit": "unique (sequence, object_id, segment_id)",
            },
        ]
    )

    exact_pass = (
        total_original_tracks == EXPECTED["total_original_tracks"]
        and len(test2) == EXPECTED["2s"]["prefilter_rows"]
        and len(test5) == EXPECTED["5s"]["prefilter_rows"]
        and len(filt2) == EXPECTED["2s"]["final_rows"]
        and len(filt5) == EXPECTED["5s"]["final_rows"]
        and filt2["original_track_key_canon"].nunique() == EXPECTED["2s"]["final_tracks"]
        and filt5["original_track_key_canon"].nunique() == EXPECTED["5s"]["final_tracks"]
        and filt2["segment_key_canon"].nunique() == EXPECTED["2s"]["final_segments"]
        and filt5["segment_key_canon"].nunique() == EXPECTED["5s"]["final_segments"]
        and cmp2["exact_physical_window_set_match"]
        and cmp5["exact_physical_window_set_match"]
        and cmp2["only_filtered"] == 0
        and cmp2["only_final"] == 0
        and cmp5["only_filtered"] == 0
        and cmp5["only_final"] == 0
    )

    source_filter_path = OUT / "R12E_V11_EXACT_SOURCE_FILTER_EXPRESSION.csv"
    comparison_path = OUT / "R12E_V11_FILTERED_VS_FROZEN_PHYSICAL_WINDOW_MATCH.csv"
    chain_path = OUT / "R12E_V11_FULL_TRACK_ACCOUNTING_CHAIN.csv"
    removed2_path = OUT / "R12E_V11_REMOVED_TESTREAL_2S_WINDOWS.csv"
    removed5_path = OUT / "R12E_V11_REMOVED_TESTREAL_5S_WINDOWS.csv"
    reason_path = OUT / "R12E_V11_REMOVED_WINDOW_REASON_SUMMARY.csv"

    source_filter_df.to_csv(source_filter_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    chain_df.to_csv(chain_path, index=False)
    removed2.to_csv(removed2_path, index=False)
    removed5.to_csv(removed5_path, index=False)
    removed_reason_df.to_csv(reason_path, index=False)

    if exact_pass:
        status = (
            "R12E_V11_FULL_EXACT_RECONSTRUCTION_PASS_45_ORIGINAL_37_31_TRACKS_89_63_SEGMENTS_1053_826_WINDOWS_ZERO_PHYSICAL_MISMATCH_READY_FOR_R12F_FINAL_LOCK"
        )
    else:
        status = "R12E_V11_RECONSTRUCTION_NOT_EXACT_REVIEW_BEFORE_R12F"

    manuscript = f"""R12 track-accounting provenance — source-locked final reconstruction

The BIRDSAI TestReal split contains {total_original_tracks} original dataset-level tracks, defined as unique (sequence, object_id) pairs.

The source-implemented unified 10-Hz core filter is:
accepted == 1 AND quality_tier == "core" AND sample_id >= 0

After applying that rule:
- 2 s: {len(filt2)} windows, {filt2["original_track_key_canon"].nunique()} original tracks, {filt2["segment_key_canon"].nunique()} forecastable segments/tracklets.
- 5 s: {len(filt5)} windows, {filt5["original_track_key_canon"].nunique()} original tracks, {filt5["segment_key_canon"].nunique()} forecastable segments/tracklets.

Physical window identity match to frozen final metadata:
- 2 s: only_filtered={cmp2["only_filtered"]}, only_final={cmp2["only_final"]}
- 5 s: only_filtered={cmp5["only_filtered"]}, only_final={cmp5["only_final"]}

Reviewer-safe wording:
"The BIRDSAI TestReal split contains 45 original tracks. Original trajectories were segmented when more than three consecutive frames were missing; smaller gaps were retained and interpolated. Candidate windows were then restricted by the source-implemented core-quality rule (accepted=1, quality_tier='core', sample_id>=0). The final 2-s evaluation set contains 1,053 windows from 37 original tracks and 89 forecastable segments, while the final 5-s evaluation set contains 826 windows from 31 original tracks and 63 forecastable segments."

Terminology lock:
- 45 = original dataset-level tracks.
- 37 / 31 = original tracks represented in final 2-s / 5-s evaluation sets.
- 89 / 63 = forecastable segments/tracklets.
- Never describe 89 or 63 as the number of original TestReal tracks.
"""

    manuscript_path = OUT / "R12E_V11_MANUSCRIPT_READY_TRACK_ACCOUNTING.txt"
    manuscript_path.write_text(manuscript, encoding="utf-8")

    manifest = {
        "step": "R12E_V11_FULL_EXACT_TRACK_ACCOUNTING_RECONSTRUCTION",
        "created_utc": utc_now(),
        "status": status,
        "training": False,
        "inference": False,
        "original_files_modified": False,
        "unified_builder": str(UNIFIED_BUILDER),
        "unified_builder_sha256": sha256_file(UNIFIED_BUILDER),
        "exact_filter": "accepted == 1 AND quality_tier == 'core' AND sample_id >= 0",
        "total_original_tracks": total_original_tracks,
        "final_2s": {
            "windows": len(filt2),
            "original_tracks": int(filt2["original_track_key_canon"].nunique()),
            "segments": int(filt2["segment_key_canon"].nunique()),
            "physical_window_mismatch": int(cmp2["only_filtered"] + cmp2["only_final"]),
        },
        "final_5s": {
            "windows": len(filt5),
            "original_tracks": int(filt5["original_track_key_canon"].nunique()),
            "segments": int(filt5["segment_key_canon"].nunique()),
            "physical_window_mismatch": int(cmp5["only_filtered"] + cmp5["only_final"]),
        },
        "outputs": {
            "source_filter_expression": str(source_filter_path),
            "physical_window_match": str(comparison_path),
            "full_track_accounting_chain": str(chain_path),
            "removed_2s_windows": str(removed2_path),
            "removed_5s_windows": str(removed5_path),
            "removed_reason_summary": str(reason_path),
            "manuscript_ready": str(manuscript_path),
        },
    }

    (OUT / "R12E_V11_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("EXACT SOURCE FILTER:")
    print(source_filter_df.to_string(index=False))

    print("\nFILTERED VS FROZEN PHYSICAL WINDOW MATCH:")
    print(comparison_df.to_string(index=False))

    print("\nFULL TRACK ACCOUNTING CHAIN:")
    print(chain_df.to_string(index=False))

    print("\nREMOVED WINDOW REASON SUMMARY:")
    if len(removed_reason_df):
        print(removed_reason_df.to_string(index=False))
    else:
        print("No removed windows.")

    print("\nSTATUS:")
    print(status)

    print("\nR12-E-V11 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
