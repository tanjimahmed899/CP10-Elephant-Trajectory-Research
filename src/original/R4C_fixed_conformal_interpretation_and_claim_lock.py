from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

R4B_DIR = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R4_conformal_coverage"
    / "R4B_method_lock_and_existing_coverage"
)

R4B_MANIFEST = R4B_DIR / "R4B_MANIFEST.json"
COVERAGE_CSV = R4B_DIR / "R4B_EXISTING_TESTREAL_FRESH_FIXED_COVERAGE.csv"
Q_CSV = R4B_DIR / "R4B_LOCKED_FIXED_Q_VALUES.csv"

OUT = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R4_conformal_coverage"
    / "R4C_interpretation_claim_lock"
)

OUT.mkdir(parents=True, exist_ok=True)


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


def verify():
    for p in [R4B_MANIFEST, COVERAGE_CSV, Q_CSV]:
        require(p)

    manifest = json.loads(
        R4B_MANIFEST.read_text(encoding="utf-8")
    )

    if (
        manifest.get("status")
        !=
        "R4B_STEP6C_METHOD_AND_EXISTING_TESTREAL_FRESH_COVERAGE_LOCKED_READY_FOR_R4C_INTERPRETATION"
    ):
        raise RuntimeError(
            f"Unexpected R4B status: {manifest.get('status')}"
        )

    expected = manifest.get("coverage_sha256")
    if expected and sha256_file(COVERAGE_CSV) != expected:
        raise RuntimeError("R4B coverage CSV hash mismatch.")

    return manifest


def classify_window(row):
    target = float(row["target_coverage"])
    cov = float(row["window_simultaneous_coverage"])
    low = float(row["window_CI_low"])
    high = float(row["window_CI_high"])

    if low > target:
        return "clear_overcoverage"
    if high < target:
        return "clear_undercoverage"
    if cov > target:
        return "point_overcoverage_CI_includes_target"
    if cov < target:
        return "point_undercoverage_CI_includes_target"
    return "on_target"


def classify_track_macro(row):
    target = float(row["target_coverage"])
    cov = float(row["track_macro_window_coverage"])
    low = float(row["track_macro_CI_low"])
    high = float(row["track_macro_CI_high"])

    if low > target:
        return "clear_overcoverage"
    if high < target:
        return "clear_undercoverage"
    if cov > target:
        return "point_overcoverage_CI_includes_target"
    if cov < target:
        return "point_undercoverage_CI_includes_target"
    return "on_target"


def make_interpretation(df):
    out = df.copy()

    out["window_gap_pp_recomputed"] = (
        100.0
        *
        (
            out["window_simultaneous_coverage"]
            -
            out["target_coverage"]
        )
    )

    out["track_macro_gap_pp"] = (
        100.0
        *
        (
            out["track_macro_window_coverage"]
            -
            out["target_coverage"]
        )
    )

    out["window_calibration_class"] = out.apply(
        classify_window,
        axis=1,
    )

    out["track_macro_calibration_class"] = out.apply(
        classify_track_macro,
        axis=1,
    )

    out["manuscript_role"] = out[
        "dataset_role_for_uncertainty"
    ].map(
        {
            "RETROSPECTIVE_AFTER_STEP6A_6B":
                "retrospective_supporting_evidence",
            "DIAGNOSTIC_ONLY_ALREADY_INSPECTED":
                "diagnostic_supporting_evidence",
        }
    )

    return out


def make_claim_lock(interp):
    rows = [
        {
            "claim_id": "R4C_CLAIM_01",
            "status": "SUPPORTED",
            "claim": (
                "The Step6C uncertainty method is a fixed, track-level cross-fit "
                "conformal procedure using a trackwise maximum canonical Euclidean "
                "error score; TestReal and Fresh evaluations reuse the locked q values "
                "without post-test recalibration."
            ),
            "basis": (
                "R4B protocol lock and exact q provenance."
            ),
        },
        {
            "claim_id": "R4C_CLAIM_02",
            "status": "SUPPORTED",
            "claim": (
                "At the 2-second horizon, fixed conformal prediction regions are "
                "conservative on both TestReal and Fresh WildBox."
            ),
            "basis": (
                "Both 90% and 95% window-level coverages exceed targets on both datasets."
            ),
        },
        {
            "claim_id": "R4C_CLAIM_03",
            "status": "SUPPORTED_WITH_LIMITATION",
            "claim": (
                "At the 5-second horizon on TestReal, fixed conformal coverage remains "
                "above nominal targets, but this evidence is retrospective rather than "
                "a pristine confirmatory uncertainty test."
            ),
            "basis": (
                "TestReal 5s window coverage is 94.07% for 90% target and 98.06% "
                "for 95% target; dataset role is RETROSPECTIVE_AFTER_STEP6A_6B."
            ),
        },
        {
            "claim_id": "R4C_CLAIM_04",
            "status": "SUPPORTED",
            "claim": (
                "Fresh WildBox 5-second q90 transfer under-covers, especially at the "
                "track-macro level, so long-horizon 90% uncertainty transfer is not "
                "reliably calibrated across domain."
            ),
            "basis": (
                "Fresh 5s q90 window coverage is 86.03% and track-macro coverage "
                "is 79.17% versus a 90% target."
            ),
        },
        {
            "claim_id": "R4C_CLAIM_05",
            "status": "SUPPORTED_WITH_LIMITATION",
            "claim": (
                "Fresh WildBox 5-second q95 transfer is closer to nominal coverage "
                "than q90, but does not justify a claim of perfect calibration."
            ),
            "basis": (
                "Window coverage is 93.56% versus 95% target while track-macro "
                "coverage is 95.50%; Fresh is diagnostic-only/already inspected."
            ),
        },
        {
            "claim_id": "R4C_CLAIM_06",
            "status": "NOT_SUPPORTED",
            "claim": (
                "The fixed conformal uncertainty method is calibrated at 90% and 95% "
                "across all horizons and datasets."
            ),
            "basis": (
                "Fresh WildBox 5s q90 materially under-covers and q95 is mixed."
            ),
        },
        {
            "claim_id": "R4C_CLAIM_07",
            "status": "NOT_SUPPORTED",
            "claim": (
                "Fresh WildBox provides pristine confirmatory uncertainty validation."
            ),
            "basis": (
                "Its recorded role is DIAGNOSTIC_ONLY_ALREADY_INSPECTED."
            ),
        },
        {
            "claim_id": "R4C_CLAIM_08",
            "status": "NOT_SUPPORTED",
            "claim": (
                "BIRDSAI TestReal provides a pristine untouched uncertainty holdout."
            ),
            "basis": (
                "Its recorded role is RETROSPECTIVE_AFTER_STEP6A_6B."
            ),
        },
        {
            "claim_id": "R4C_CLAIM_09",
            "status": "SUPPORTED",
            "claim": (
                "The uncertainty results support a horizon-dependent transfer story: "
                "short-horizon uncertainty is conservative, whereas long-horizon "
                "cross-domain calibration degrades."
            ),
            "basis": (
                "2s coverage is consistently above nominal; Fresh 5s q90 under-covers."
            ),
        },
    ]

    return pd.DataFrame(rows)


def make_manuscript_table(interp):
    keep = interp[
        [
            "dataset",
            "task",
            "alpha",
            "target_coverage",
            "q_fixed_step6c",
            "samples",
            "tracks",
            "sources",
            "window_simultaneous_coverage",
            "window_CI_low",
            "window_CI_high",
            "window_gap_pp_recomputed",
            "track_macro_window_coverage",
            "track_macro_CI_low",
            "track_macro_CI_high",
            "track_macro_gap_pp",
            "fully_covered_track_fraction",
            "stepwise_window_coverage",
            "window_calibration_class",
            "track_macro_calibration_class",
            "manuscript_role",
        ]
    ].copy()

    return keep


def main():
    print(
        "\nR4-C — FIXED-CONFORMAL INTERPRETATION + CLAIM LOCK"
    )

    print(
        "NO recalibration and NO new coverage computation. "
        "This step interprets the locked R4B results and defines "
        "manuscript-safe uncertainty claims.\n"
    )

    verify()

    df = pd.read_csv(COVERAGE_CSV)

    interp = make_interpretation(df)
    claims = make_claim_lock(interp)
    manuscript = make_manuscript_table(interp)

    interp_path = OUT / "R4C_coverage_interpretation.csv"
    claim_path = OUT / "R4C_CLAIM_LOCK.csv"
    manuscript_path = OUT / "R4C_manuscript_uncertainty_table.csv"

    interp.to_csv(interp_path, index=False)
    claims.to_csv(claim_path, index=False)
    manuscript.to_csv(manuscript_path, index=False)

    status = (
        "R4C_FIXED_CONFORMAL_INTERPRETATION_AND_CLAIM_LOCK_COMPLETE_READY_FOR_R5_GEOFENCE_BASELINES"
    )

    manifest = {
        "step": "R4C_FIXED_CONFORMAL_INTERPRETATION_CLAIM_LOCK",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "recalibration": False,
        "new_coverage_computation": False,
        "source_r4b_coverage": str(COVERAGE_CSV),
        "source_r4b_coverage_sha256": sha256_file(COVERAGE_CSV),
        "claim_lock_csv": str(claim_path),
        "claim_lock_sha256": sha256_file(claim_path),
        "manuscript_table_csv": str(manuscript_path),
    }

    manifest_path = OUT / "R4C_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("COVERAGE INTERPRETATION:")
    print(
        manuscript.to_string(
            index=False
        )
    )

    print("\nCLAIM LOCK:")
    print(
        claims.to_string(
            index=False
        )
    )

    print("\nSTATUS:")
    print(status)

    print("\nR4-C COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
