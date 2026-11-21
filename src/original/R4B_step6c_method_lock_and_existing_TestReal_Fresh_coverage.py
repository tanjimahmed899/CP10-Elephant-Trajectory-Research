from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN_RESULT_2 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result 2.0")

STEP6C = MAIN_RESULT_2 / "step6c_crossfit_conformal"

CAL_SUMMARY = STEP6C / "track_conformal_calibration_summary.csv"
RETRO = STEP6C / "track_conformal_retrospective_evaluation.csv"
PROTOCOL = STEP6C / "crossfit_protocol_audit.csv"
TRACK_SCORES = STEP6C / "crossfit_track_calibration_scores.csv"
OOF_SCORES = STEP6C / "crossfit_oof_sample_scores.csv"

R4A_MANIFEST = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R4_conformal_coverage"
    / "R4A_frozen_conformal_method_audit"
    / "R4A_MANIFEST.json"
)

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R4_conformal_coverage"
    / "R4B_method_lock_and_existing_coverage"
)

OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_Q = {
    ("2s_to_2s", 0.10): 2.158767888280567,
    ("2s_to_2s", 0.05): 2.560772554862189,
    ("2s_to_5s", 0.10): 1.975129668255982,
    ("2s_to_5s", 0.05): 2.950976253259077,
}

EXPECTED_CAL_TRACKS = {
    "2s_to_2s": 62,
    "2s_to_5s": 34,
}


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


def verify_r4a():
    require(R4A_MANIFEST)

    obj = json.loads(
        R4A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        obj.get("status")
        !=
        "R4A_CONFORMAL_AUDIT_COMPLETE_EXACT_FROZEN_Q_EVIDENCE_FOUND_READY_FOR_METHOD_LOCK"
    ):
        raise RuntimeError(
            f"Unexpected R4A status: {obj.get('status')}"
        )

    return obj


def normalize_alpha(row):
    if "alpha" in row and pd.notna(row["alpha"]):
        a = float(row["alpha"])
        if a > 1:
            a = a / 100.0
        return a

    if "target_coverage" in row and pd.notna(row["target_coverage"]):
        target = float(row["target_coverage"])
        if target > 1:
            target = target / 100.0
        return 1.0 - target

    raise RuntimeError("Cannot infer alpha.")


def lock_calibration():
    df = pd.read_csv(CAL_SUMMARY)

    required = {
        "task",
        "alpha",
        "target_coverage",
        "calibration_tracks",
        "normalized_track_simultaneous_radius",
        "order_statistic_k",
        "calibration_track_simultaneous_coverage",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Calibration summary missing columns: {sorted(missing)}"
        )

    if len(df) != 4:
        raise RuntimeError(
            f"Expected 4 calibration rows, found {len(df)}"
        )

    rows = []

    for _, row in df.iterrows():
        task = str(row["task"])
        alpha = normalize_alpha(row)
        q = float(row["normalized_track_simultaneous_radius"])

        key = (task, round(alpha, 2))

        if key not in EXPECTED_Q:
            raise RuntimeError(
                f"Unexpected task/alpha pair: {key}"
            )

        expected_q = EXPECTED_Q[key]

        if abs(q - expected_q) > 1e-12:
            raise RuntimeError(
                f"Frozen q mismatch for {task}, alpha={alpha}: "
                f"{q} != {expected_q}"
            )

        tracks = int(row["calibration_tracks"])

        if tracks != EXPECTED_CAL_TRACKS[task]:
            raise RuntimeError(
                f"Calibration track count mismatch for {task}: "
                f"{tracks} != {EXPECTED_CAL_TRACKS[task]}"
            )

        rows.append(
            {
                "task": task,
                "alpha": alpha,
                "target_coverage": float(row["target_coverage"]),
                "calibration_tracks": tracks,
                "q_fixed_step6c": q,
                "order_statistic_k": int(row["order_statistic_k"]),
                "calibration_track_simultaneous_coverage":
                    float(row["calibration_track_simultaneous_coverage"]),
                "q_match_expected": True,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["task", "alpha"],
        ascending=[True, False],
    )


def audit_protocol():
    df = pd.read_csv(PROTOCOL)

    required = {
        "task",
        "primary_calibration_unit",
        "primary_score",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Protocol audit missing columns: {sorted(missing)}"
        )

    if len(df) != 2:
        raise RuntimeError(
            f"Expected 2 protocol rows, found {len(df)}"
        )

    out = df.copy()

    out["track_unit_ok"] = (
        out["primary_calibration_unit"]
        .astype(str)
        .str.lower()
        .str.contains("track")
    )

    out["score_looks_simultaneous_max_error"] = (
        out["primary_score"]
        .astype(str)
        .str.lower()
        .apply(
            lambda s:
                ("max" in s)
                and
                ("error" in s)
        )
    )

    if not out["track_unit_ok"].all():
        raise RuntimeError(
            "Step6C primary calibration unit is not track-based for every task."
        )

    if not out["score_looks_simultaneous_max_error"].all():
        raise RuntimeError(
            "Step6C primary score is not recognizable as a max-error simultaneous score."
        )

    return out


def audit_track_score_counts():
    df = pd.read_csv(TRACK_SCORES)

    required = {
        "task",
        "track_key",
        "track_max_canonical_error",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Track calibration scores missing columns: {sorted(missing)}"
        )

    rows = []

    for task, group in df.groupby("task", sort=True):
        tracks = int(group["track_key"].astype(str).nunique())
        expected = EXPECTED_CAL_TRACKS[str(task)]

        rows.append(
            {
                "task": str(task),
                "rows": int(len(group)),
                "unique_tracks": tracks,
                "expected_tracks": expected,
                "track_count_pass": bool(tracks == expected),
                "min_track_max_canonical_error":
                    float(group["track_max_canonical_error"].min()),
                "median_track_max_canonical_error":
                    float(group["track_max_canonical_error"].median()),
                "max_track_max_canonical_error":
                    float(group["track_max_canonical_error"].max()),
            }
        )

    out = pd.DataFrame(rows)

    if not out["track_count_pass"].all():
        raise RuntimeError(
            "Cross-fit track score count mismatch."
        )

    return out


def extract_existing_coverage():
    df = pd.read_csv(RETRO)

    required = {
        "task",
        "dataset",
        "dataset_role_for_uncertainty",
        "alpha",
        "target_coverage",
        "samples",
        "tracks",
        "sources",
        "normalized_track_simultaneous_radius",
        "window_simultaneous_coverage",
        "window_coverage_gap_pp",
        "track_macro_window_coverage",
        "fully_covered_track_fraction",
        "stepwise_window_coverage",
        "mean_ADE",
        "mean_FDE",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Retrospective evaluation missing columns: {sorted(missing)}"
        )

    if len(df) != 8:
        raise RuntimeError(
            f"Expected 8 retrospective evaluation rows, found {len(df)}"
        )

    rows = []

    for _, row in df.iterrows():
        task = str(row["task"])
        alpha = normalize_alpha(row)
        q = float(row["normalized_track_simultaneous_radius"])

        key = (task, round(alpha, 2))

        if key not in EXPECTED_Q:
            raise RuntimeError(
                f"Unexpected retrospective task/alpha pair: {key}"
            )

        if abs(q - EXPECTED_Q[key]) > 1e-12:
            raise RuntimeError(
                f"Retrospective row uses wrong frozen q for {task}, alpha={alpha}: "
                f"{q} != {EXPECTED_Q[key]}"
            )

        out = {
            "task": task,
            "dataset": str(row["dataset"]),
            "dataset_role_for_uncertainty":
                str(row["dataset_role_for_uncertainty"]),
            "alpha": alpha,
            "target_coverage": float(row["target_coverage"]),
            "q_fixed_step6c": q,
            "samples": int(row["samples"]),
            "tracks": int(row["tracks"]),
            "sources": int(row["sources"]),
            "window_simultaneous_coverage":
                float(row["window_simultaneous_coverage"]),
            "window_coverage_gap_pp":
                float(row["window_coverage_gap_pp"]),
            "track_macro_window_coverage":
                float(row["track_macro_window_coverage"]),
            "fully_covered_track_fraction":
                float(row["fully_covered_track_fraction"]),
            "stepwise_window_coverage":
                float(row["stepwise_window_coverage"]),
            "mean_ADE":
                float(row["mean_ADE"]),
            "mean_FDE":
                float(row["mean_FDE"]),
        }

        for col in [
            "window_CI_low",
            "window_CI_high",
            "track_macro_CI_low",
            "track_macro_CI_high",
            "fully_covered_track_CI_low",
            "fully_covered_track_CI_high",
            "mean_absolute_radius",
            "median_absolute_radius",
            "p95_absolute_radius",
            "mean_ensemble_disagreement",
            "max_abs_vs_scaled_canonical_error_diff",
        ]:
            if col in row.index and pd.notna(row[col]):
                out[col] = float(row[col])

        rows.append(out)

    out = pd.DataFrame(rows)

    # Require exactly 2 datasets x 2 tasks x 2 alphas.
    combos = (
        out.groupby(
            ["dataset", "task", "alpha"]
        )
        .size()
    )

    if not (combos == 1).all():
        raise RuntimeError(
            "Retrospective evaluation has duplicate/missing dataset-task-alpha rows."
        )

    return out.sort_values(
        ["dataset", "task", "alpha"],
        ascending=[True, True, False],
    )


def role_audit(coverage):
    rows = []

    for dataset, group in coverage.groupby("dataset", sort=True):
        roles = sorted(
            group["dataset_role_for_uncertainty"]
            .astype(str)
            .unique()
            .tolist()
        )

        rows.append(
            {
                "dataset": dataset,
                "rows": int(len(group)),
                "roles": " | ".join(roles),
                "tasks": " | ".join(
                    sorted(group["task"].unique())
                ),
                "alphas": " | ".join(
                    map(
                        str,
                        sorted(group["alpha"].unique())
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR4-B — STEP6C METHOD LOCK + EXISTING TESTREAL/FRESH COVERAGE EXTRACTION"
    )

    print(
        "NO recalibration. NO new forecasting inference. NO q selection. "
        "This step locks the exact Step6C track-level cross-fit conformal method "
        "and reuses its existing retrospective TestReal/Fresh evaluation.\n"
    )

    verify_r4a()

    for p in [
        CAL_SUMMARY,
        RETRO,
        PROTOCOL,
        TRACK_SCORES,
        OOF_SCORES,
    ]:
        require(p)

    protocol = audit_protocol()
    locked_q = lock_calibration()
    track_audit = audit_track_score_counts()
    coverage = extract_existing_coverage()
    roles = role_audit(coverage)

    protocol_path = OUT / "R4B_step6c_protocol_lock.csv"
    q_path = OUT / "R4B_LOCKED_FIXED_Q_VALUES.csv"
    track_path = OUT / "R4B_calibration_track_audit.csv"
    coverage_path = OUT / "R4B_EXISTING_TESTREAL_FRESH_FIXED_COVERAGE.csv"
    role_path = OUT / "R4B_dataset_role_audit.csv"

    protocol.to_csv(protocol_path, index=False)
    locked_q.to_csv(q_path, index=False)
    track_audit.to_csv(track_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    roles.to_csv(role_path, index=False)

    status = (
        "R4B_STEP6C_METHOD_AND_EXISTING_TESTREAL_FRESH_COVERAGE_LOCKED_READY_FOR_R4C_INTERPRETATION"
    )

    manifest = {
        "step": "R4B_STEP6C_METHOD_LOCK_EXISTING_COVERAGE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "recalibration": False,
        "new_forecasting_inference": False,
        "new_q_selection": False,
        "coverage_recomputed": False,
        "coverage_source": str(RETRO),
        "coverage_source_sha256": sha256_file(RETRO),
        "calibration_source": str(CAL_SUMMARY),
        "calibration_source_sha256": sha256_file(CAL_SUMMARY),
        "protocol_source": str(PROTOCOL),
        "protocol_source_sha256": sha256_file(PROTOCOL),
        "track_scores_source": str(TRACK_SCORES),
        "track_scores_source_sha256": sha256_file(TRACK_SCORES),
        "locked_q_csv": str(q_path),
        "locked_q_sha256": sha256_file(q_path),
        "coverage_csv": str(coverage_path),
        "coverage_sha256": sha256_file(coverage_path),
    }

    manifest_path = OUT / "R4B_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("STEP6C PROTOCOL LOCK:")
    print(protocol.to_string(index=False))

    print("\nLOCKED FIXED q90/q95:")
    print(locked_q.to_string(index=False))

    print("\nCALIBRATION TRACK AUDIT:")
    print(track_audit.to_string(index=False))

    print("\nEXISTING TESTREAL/FRESH FIXED-CONFORMAL COVERAGE:")
    print(coverage.to_string(index=False))

    print("\nDATASET ROLE AUDIT:")
    print(roles.to_string(index=False))

    print("\nSTATUS:")
    print(status)

    print("\nR4-B COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
