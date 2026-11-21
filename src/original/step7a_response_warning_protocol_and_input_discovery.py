from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")
DOWNLOADS = Path(r"C:\Users\HP\Downloads")
STEP6C = DOWNLOADS / "step6c_crossfit_conformal"
OUT = DOWNLOADS / "step7_response_aware_warning"
OUT.mkdir(parents=True, exist_ok=True)

TEXT_EXTS = {".py", ".txt", ".md", ".json", ".yaml", ".yml"}

Q_LOCK = {
    "2s_to_2s": {
        "q90": 2.158768,
        "q95": 2.560773,
        "calibration_tracks": 62,
        "calibration_coverage_90": 0.919355,
        "calibration_coverage_95": 0.967742,
    },
    "2s_to_5s": {
        "q90": 1.975130,
        "q95": 2.950976,
        "calibration_tracks": 34,
        "calibration_coverage_90": 0.941176,
        "calibration_coverage_95": 1.0,
    },
}

def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_read_text(path, max_bytes=8 * 1024 * 1024):
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def discover_files():
    roots = []
    if STEP6C.exists():
        roots.append(("step6c_downloads", STEP6C))
    if ROOT.exists():
        roots.append(("main_result", ROOT))

    rows = []
    terms = [
        "step6c", "conformal", "geofence", "warning",
        "trajectory", "prediction", "forecast", "ensemble",
        "fresh", "testreal", "corrected"
    ]
    for root_label, base in roots:
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            s = str(p).lower()
            if not any(term in s for term in terms):
                continue
            try:
                size = int(p.stat().st_size)
            except Exception:
                size = None
            rows.append({
                "root_label": root_label,
                "path": str(p),
                "suffix": p.suffix.lower(),
                "size_bytes": size,
            })
    return pd.DataFrame(rows)

def scan_text_candidates(files_df):
    rows = []
    patterns = [
        "geofence", "radius", "threshold", "signed distance",
        "boundary", "q90", "q95", "2.158768", "2.560773",
        "1.975130", "2.950976", "scale", "canonical",
        "prediction", "future", "track"
    ]
    if len(files_df) == 0:
        return pd.DataFrame()

    for _, row in files_df.iterrows():
        p = Path(row["path"])
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        txt = safe_read_text(p)
        if not txt:
            continue
        low = txt.lower()
        matched = [pat for pat in patterns if pat.lower() in low]
        if not matched:
            continue

        excerpts = []
        lines = txt.splitlines()
        for i, line in enumerate(lines):
            ll = line.lower()
            if any(pat.lower() in ll for pat in matched):
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                excerpts.append(" || ".join(x.strip() for x in lines[start:end]))
            if len(excerpts) >= 12:
                break

        rows.append({
            "path": str(p),
            "matched_terms": ";".join(matched),
            "excerpt_preview": " /// ".join(excerpts),
        })
    return pd.DataFrame(rows)

def inspect_csv(path):
    try:
        df = pd.read_csv(path, nrows=5)
        cols = list(map(str, df.columns))
        joined = " ".join(cols).lower()
        return {
            "path": str(path),
            "type": "csv",
            "columns": ";".join(cols),
            "sample_rows": int(len(df)),
            "prediction_like": any(term in joined for term in ["pred", "forecast", "mean", "ensemble"]),
            "future_like": any(term in joined for term in ["future", "target", "gt", "actual", "truth"]),
            "track_like": any(term in joined for term in ["track", "object_id"]),
            "scale_like": any(term in joined for term in ["scale", "norm"]),
            "geofence_like": any(term in joined for term in ["geofence", "boundary", "distance", "cross"]),
        }
    except Exception as e:
        return {"path": str(path), "type": "csv", "error": str(e)}

def inspect_npz(path):
    try:
        z = np.load(path, allow_pickle=False)
        keys = list(z.files)
        shapes = []
        for k in keys:
            try:
                shapes.append(f"{k}:{z[k].shape}")
            except Exception:
                shapes.append(f"{k}:?")
        joined = " ".join(keys).lower()
        return {
            "path": str(path),
            "type": "npz",
            "keys": ";".join(keys),
            "shapes": ";".join(shapes),
            "prediction_like": any(term in joined for term in ["pred", "forecast", "mean", "ensemble"]),
            "future_like": any(term in joined for term in ["future", "target", "gt", "actual", "truth"]),
            "track_like": any(term in joined for term in ["track", "object"]),
            "scale_like": any(term in joined for term in ["scale", "norm"]),
            "geofence_like": any(term in joined for term in ["geofence", "boundary", "distance", "cross"]),
        }
    except Exception as e:
        return {"path": str(path), "type": "npz", "error": str(e)}

def inspect_array_and_table_candidates(files_df):
    rows = []
    if len(files_df) == 0:
        return pd.DataFrame()
    seen = set()
    for _, row in files_df.iterrows():
        p = Path(row["path"])
        if str(p) in seen:
            continue
        seen.add(str(p))
        if p.suffix.lower() == ".csv":
            rows.append(inspect_csv(p))
        elif p.suffix.lower() == ".npz":
            rows.append(inspect_npz(p))
    return pd.DataFrame(rows)

def find_step6c_source_script():
    candidates = []
    for p in [
        DOWNLOADS / "step6c_crossfit_conformal.py",
        STEP6C / "step6c_crossfit_conformal.py",
    ]:
        if p.exists():
            candidates.append(p)

    for base in [DOWNLOADS, ROOT]:
        if not base.exists():
            continue
        try:
            for p in base.rglob("*step6c*conformal*.py"):
                candidates.append(p)
        except Exception:
            pass

    rows, seen = [], set()
    for p in candidates:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        rows.append({
            "path": s,
            "sha256": sha256_file(p),
            "size_bytes": int(p.stat().st_size),
        })
    return pd.DataFrame(rows)

def find_geofence_sources():
    rows = []
    if not ROOT.exists():
        return pd.DataFrame()
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        s = str(p).lower()
        if "geofence" not in s and "step2c" not in s and "warning" not in s:
            continue
        try:
            size = int(p.stat().st_size)
            rows.append({
                "path": str(p),
                "suffix": p.suffix.lower(),
                "size_bytes": size,
                "sha256": sha256_file(p) if size <= 100 * 1024 * 1024 else "",
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

def write_protocol_lock(step6c_scripts, geofence_sources):
    protocol = {
        "step": "STEP_7A_RESPONSE_AWARE_WARNING_PROTOCOL_LOCK",
        "performance_evaluated_in_this_step": False,
        "bambi_confirmatory_coverage_available": False,
        "scope": (
            "Virtual-geofence response-aware simulation using the selected "
            "Step-6C track-level cross-fitted conformal method. "
            "This is not field validation and does not establish actual team response times."
        ),
        "primary_warning_logic": {
            "GREEN": (
                "Across the evaluated forecast interval, the calibrated uncertainty "
                "region remains outside the virtual geofence."
            ),
            "YELLOW": (
                "The nominal predicted path remains outside the geofence but the "
                "calibrated uncertainty region intersects/touches the virtual geofence; "
                "or a nominal crossing is predicted after the assumed response-time "
                "threshold but within the forecast horizon."
            ),
            "RED": (
                "The nominal predicted path is already inside the virtual geofence "
                "or is predicted to cross it within the assumed response-time threshold."
            ),
        },
        "secondary_conservative_logic": {
            "GREEN": (
                "The calibrated uncertainty region remains outside the virtual geofence "
                "over the horizon."
            ),
            "YELLOW": (
                "The calibrated uncertainty region first intersects the geofence after "
                "the assumed response-time threshold."
            ),
            "RED": (
                "The calibrated uncertainty region is already intersecting the geofence "
                "or first intersects within the assumed response-time threshold."
            ),
        },
        "conformal_levels": {
            "primary": 0.90,
            "secondary_sensitivity": 0.95,
        },
        "locked_step6c_q_values": Q_LOCK,
        "response_time_sensitivity_seconds": {
            "2s_to_2s": [0.5, 1.0, 1.5, 2.0],
            "2s_to_5s": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
        },
        "sampling_rate_hz": 10,
        "evaluation_units": [
            "window-level alarm state",
            "track-macro alarm rates",
            "actual-future crossing classification",
        ],
        "primary_metrics_planned": [
            "Green/Yellow/Red rate",
            "actual future crossing prevalence",
            "Red sensitivity for actual crossings",
            "Yellow-or-Red sensitivity for actual crossings",
            "false Red rate among non-crossing windows",
            "false Yellow-or-Red rate among non-crossing windows",
            "track-macro versions",
            "response-time sensitivity curves",
        ],
        "important_nonclaims": [
            "2 s and 5 s forecast horizons are not measured team response times.",
            "The virtual geofence is not yet a real village/farm boundary.",
            "Step 7 is a simulation/decision-layer experiment, not field deployment.",
            "Until BAMBI confirmatory uncertainty coverage is completed, Step-6C external coverage remains retrospective diagnostic evidence.",
        ],
        "step6c_source_scripts": step6c_scripts.to_dict(orient="records") if len(step6c_scripts) else [],
        "geofence_source_files": geofence_sources.to_dict(orient="records") if len(geofence_sources) else [],
    }

    path = OUT / "STEP7A_PROTOCOL_LOCK.json"

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)

        if (
            existing.get("primary_warning_logic") != protocol["primary_warning_logic"]
            or existing.get("response_time_sensitivity_seconds")
            != protocol["response_time_sensitivity_seconds"]
            or existing.get("locked_step6c_q_values") != protocol["locked_step6c_q_values"]
        ):
            raise RuntimeError(
                "Existing Step-7 protocol lock differs from current protocol; refusing to overwrite."
            )
        lock_status = "EXISTING_PROTOCOL_LOCK_VERIFIED"
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(protocol, f, indent=2)
        lock_status = "NEW_PROTOCOL_LOCK_CREATED"

    return path, lock_status, protocol

def readiness_verdict(array_table_audit, geofence_sources):
    if len(array_table_audit) == 0:
        return "NEEDS_INPUT_MAPPING", "No inspectable CSV/NPZ candidates were found."

    pred = fut = track = scale = False
    geo = len(geofence_sources) > 0

    for _, r in array_table_audit.iterrows():
        pred = pred or bool(r.get("prediction_like", False))
        fut = fut or bool(r.get("future_like", False))
        track = track or bool(r.get("track_like", False))
        scale = scale or bool(r.get("scale_like", False))
        geo = geo or bool(r.get("geofence_like", False))

    if pred and fut and track and geo:
        return (
            "LIKELY_READY_FOR_STEP7B",
            "Prediction/future/track information and geofence-related sources were detected."
        )

    missing = []
    if not pred:
        missing.append("prediction-like data")
    if not fut:
        missing.append("future/ground-truth data")
    if not track:
        missing.append("track identifiers")
    if not geo:
        missing.append("geofence definition/source")

    return "NEEDS_INPUT_MAPPING", "Missing/not automatically detected: " + ", ".join(missing)

def main():
    print("\nSTEP 7A — RESPONSE-AWARE WARNING PROTOCOL LOCK + INPUT DISCOVERY")

    files = discover_files()
    files.to_csv(OUT / "step7a_candidate_files.csv", index=False)

    text_audit = scan_text_candidates(files)
    text_audit.to_csv(OUT / "step7a_text_definition_audit.csv", index=False)

    arrays = inspect_array_and_table_candidates(files)
    arrays.to_csv(OUT / "step7a_array_table_audit.csv", index=False)

    step6c_scripts = find_step6c_source_script()
    step6c_scripts.to_csv(OUT / "step7a_step6c_source_scripts.csv", index=False)

    geofence_sources = find_geofence_sources()
    geofence_sources.to_csv(OUT / "step7a_geofence_source_files.csv", index=False)

    lock_path, lock_status, protocol = write_protocol_lock(step6c_scripts, geofence_sources)

    verdict, detail = readiness_verdict(arrays, geofence_sources)

    ranked = arrays.copy()
    if len(ranked):
        for col in [
            "prediction_like", "future_like", "track_like",
            "scale_like", "geofence_like"
        ]:
            if col not in ranked.columns:
                ranked[col] = False
            ranked[col] = ranked[col].fillna(False)

        ranked["readiness_score"] = (
            ranked[
                [
                    "prediction_like", "future_like", "track_like",
                    "scale_like", "geofence_like"
                ]
            ]
            .astype(int)
            .sum(axis=1)
        )
        ranked = ranked.sort_values(
            ["readiness_score", "path"],
            ascending=[False, True]
        )
        ranked.head(40).to_csv(
            OUT / "step7a_ranked_data_candidates.csv",
            index=False
        )

    with open(OUT / "STEP7A_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("STEP 7A — RESPONSE-AWARE WARNING PROTOCOL LOCK + INPUT DISCOVERY\n")
        f.write("=" * 84 + "\n\n")
        f.write(f"PROTOCOL LOCK STATUS: {lock_status}\n")
        f.write(f"LOCK FILE: {lock_path}\n\n")
        f.write("PRIMARY WARNING LOGIC\n---------------------\n")
        for state in ["GREEN", "YELLOW", "RED"]:
            f.write(f"{state}: {protocol['primary_warning_logic'][state]}\n")

        f.write("\nRESPONSE-TIME SENSITIVITY GRID (seconds)\n")
        f.write("----------------------------------------\n")
        f.write(json.dumps(protocol["response_time_sensitivity_seconds"], indent=2))

        f.write("\n\nLOCKED STEP-6C q VALUES\n------------------------\n")
        f.write(json.dumps(Q_LOCK, indent=2))

        f.write("\n\nINPUT DISCOVERY\n---------------\n")
        f.write(f"Candidate files: {len(files)}\n")
        f.write(f"Inspectable CSV/NPZ files: {len(arrays)}\n")
        f.write(f"Step-6C source scripts found: {len(step6c_scripts)}\n")
        f.write(f"Geofence-related source files found: {len(geofence_sources)}\n")
        f.write(f"\nREADINESS VERDICT: {verdict}\n{detail}\n")

        if len(ranked):
            f.write("\nTOP DATA CANDIDATES\n-------------------\n")
            cols = [
                c for c in ["path", "type", "keys", "columns", "shapes", "readiness_score"]
                if c in ranked.columns
            ]
            f.write(ranked.head(20)[cols].to_string(index=False))

        if len(geofence_sources):
            f.write("\n\nGEOFENCE SOURCE FILES\n---------------------\n")
            f.write(geofence_sources.head(30).to_string(index=False))

    print("\nPROTOCOL LOCK:")
    print(lock_status)
    print(lock_path)

    print("\nPRIMARY GREEN / YELLOW / RED LOGIC:")
    for state in ["GREEN", "YELLOW", "RED"]:
        print(f"{state}: {protocol['primary_warning_logic'][state]}")

    print("\nINPUT DISCOVERY:")
    print("candidate_files =", len(files))
    print("inspectable_csv_npz =", len(arrays))
    print("step6c_source_scripts =", len(step6c_scripts))
    print("geofence_source_files =", len(geofence_sources))

    print("\nREADINESS VERDICT:")
    print(verdict)
    print(detail)

    if len(ranked):
        print("\nTOP DATA CANDIDATES:")
        cols = [
            c for c in ["path", "type", "keys", "columns", "shapes", "readiness_score"]
            if c in ranked.columns
        ]
        print(ranked.head(20)[cols].to_string(index=False))

    if len(geofence_sources):
        print("\nGEOFENCE SOURCE FILES:")
        print(geofence_sources.head(30).to_string(index=False))

    print("\nSTEP 7A COMPLETE")
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
