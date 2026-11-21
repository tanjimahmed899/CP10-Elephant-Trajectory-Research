from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step6d_uncertainty_holdout"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CANDIDATES = [
    "0000000054_0000000000",
    "0000000056_0000000000",
]

RAW_POINTS = (
    ROOT
    /
    "trajectory_dataset"
    /
    "cleaned_track_points.csv"
)

EGO = (
    ROOT
    /
    "corrected_trajectory_dataset"
    /
    "repaired_ego_motion.csv"
)

CORR_DIR = (
    ROOT
    /
    "corrected_trajectory_dataset"
)

USED_DIR_TERMS = [
    "final_neural_trajectory_dataset",
    "unified_trajectory_10hz",
    "wildbox_trajectory_dataset",
    "wildbox_fresh_confirmatory_results",
    "wildbox_fresh_holdout_zips",
    "baseline_constant_velocity",
    "baseline_kalman",
    "lstm_gru_results",
    "transformer_results",
    "transformer_ensemble_validation",
    "step2",
    "step3_fresh_wildbox_full_comparison",
    "step4",
    "step5",
    "step6_conformal_uncertainty",
    "step6b_scale_adaptive_conformal",
    "step6c_crossfit_conformal",
]

TEXT_EXTS = {
    ".csv", ".json", ".txt", ".md", ".py", ".yaml", ".yml"
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


def is_model_or_eval_path(path):
    s = str(path).lower()
    return any(
        term.lower() in s
        for term in USED_DIR_TERMS
    )


def scan_occurrences(token):
    rows = []

    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue

        if OUT in p.parents:
            continue

        path_hit = token.lower() in str(p).lower()

        content_hit = False
        count = 0

        if p.suffix.lower() in TEXT_EXTS:
            try:
                if p.stat().st_size <= 100 * 1024 * 1024:
                    txt = p.read_text(
                        encoding="utf-8",
                        errors="ignore"
                    )
                    count = txt.lower().count(
                        token.lower()
                    )
                    content_hit = count > 0
            except Exception:
                pass

        if path_hit or content_hit:
            rows.append({
                "source": token,
                "path": str(p),
                "path_hit": path_hit,
                "content_hit": content_hit,
                "content_hit_count": count,
                "model_or_eval_path": is_model_or_eval_path(p),
            })

    return rows


def longest_contiguous_run(frames):
    frames = sorted(
        set(
            int(x)
            for x in frames
        )
    )

    if not frames:
        return 0

    best = 1
    cur = 1

    for a, b in zip(
        frames[:-1],
        frames[1:]
    ):
        if b == a + 1:
            cur += 1
        else:
            best = max(
                best,
                cur
            )
            cur = 1

    best = max(
        best,
        cur
    )

    return best


def structural_raw_summary():
    if not RAW_POINTS.exists():
        raise FileNotFoundError(
            RAW_POINTS
        )

    df = pd.read_csv(
        RAW_POINTS,
        dtype={
            "sequence": str,
            "object_id": str
        }
    )

    required = {
        "sequence",
        "object_id",
        "frame",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing raw columns: {missing}"
        )

    rows = []

    for source in CANDIDATES:
        g = df[
            df[
                "sequence"
            ].astype(
                str
            )
            ==
            source
        ].copy()

        if len(g) == 0:
            rows.append({
                "source": source,
                "raw_rows": 0,
                "raw_tracks": 0,
                "min_frame": None,
                "max_frame": None,
                "unique_frames": 0,
                "longest_contiguous_track_frames": 0,
                "tracks_with_at_least_120_contiguous_frames": 0,
                "tracks_with_at_least_210_contiguous_frames": 0,
                "structurally_supports_2s_to_2s_at_30hz": False,
                "structurally_supports_2s_to_5s_at_30hz": False,
            })
            continue

        per_track = []

        for track, tg in g.groupby(
            "object_id"
        ):
            run = longest_contiguous_run(
                tg[
                    "frame"
                ].tolist()
            )

            per_track.append({
                "track": str(track),
                "rows": len(tg),
                "unique_frames": tg["frame"].nunique(),
                "longest_contiguous_run": run,
            })

        tdf = pd.DataFrame(
            per_track
        )

        # Structural thresholds only:
        # 2 s history + 2 s future at ~30 Hz -> about 120 frames total.
        # 2 s history + 5 s future at ~30 Hz -> about 210 frames total.
        n_120 = int(
            (
                tdf[
                    "longest_contiguous_run"
                ]
                >=
                120
            ).sum()
        )

        n_210 = int(
            (
                tdf[
                    "longest_contiguous_run"
                ]
                >=
                210
            ).sum()
        )

        rows.append({
            "source": source,
            "raw_rows": len(g),
            "raw_tracks": g["object_id"].nunique(),
            "min_frame": int(
                g["frame"].min()
            ),
            "max_frame": int(
                g["frame"].max()
            ),
            "unique_frames": g["frame"].nunique(),
            "longest_contiguous_track_frames": int(
                tdf[
                    "longest_contiguous_run"
                ].max()
            ),
            "tracks_with_at_least_120_contiguous_frames": n_120,
            "tracks_with_at_least_210_contiguous_frames": n_210,
            "structurally_supports_2s_to_2s_at_30hz": n_120 > 0,
            "structurally_supports_2s_to_5s_at_30hz": n_210 > 0,
        })

    return pd.DataFrame(
        rows
    )


def ego_summary():
    if not EGO.exists():
        return pd.DataFrame(
            [
                {
                    "source": s,
                    "ego_rows": 0,
                    "ego_columns": "",
                    "ego_available": False
                }
                for s in CANDIDATES
            ]
        )

    df = pd.read_csv(
        EGO,
        dtype={
            "sequence": str
        }
    )

    rows = []

    for source in CANDIDATES:
        if "sequence" in df.columns:
            g = df[
                df[
                    "sequence"
                ].astype(
                    str
                )
                ==
                source
            ]
        else:
            g = df.iloc[0:0]

        rows.append({
            "source": source,
            "ego_rows": len(g),
            "ego_columns": ";".join(
                map(
                    str,
                    df.columns
                )
            ),
            "ego_available": len(g) > 0,
        })

    return pd.DataFrame(
        rows
    )


def corrected_existing_presence():
    rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s"
    ]:
        csv_path = (
            CORR_DIR
            /
            f"corrected_windows_{task}.csv"
        )

        if csv_path.exists():
            df = pd.read_csv(
                csv_path,
                dtype={
                    "sequence": str,
                    "object_id": str
                }
            )
        else:
            df = pd.DataFrame()

        for source in CANDIDATES:
            if (
                len(df)
                and
                "sequence" in df.columns
            ):
                g = df[
                    df[
                        "sequence"
                    ].astype(
                        str
                    )
                    ==
                    source
                ]
            else:
                g = df.iloc[0:0]

            rows.append({
                "task": task,
                "source": source,
                "existing_corrected_window_rows": len(g),
                "existing_corrected_window_tracks": (
                    g[
                        "object_id"
                    ].astype(
                        str
                    ).nunique()
                    if (
                        len(g)
                        and
                        "object_id" in g.columns
                    )
                    else 0
                ),
            })

    return pd.DataFrame(
        rows
    )


def inspect_corrected_npz():
    rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s"
    ]:
        p = (
            CORR_DIR
            /
            f"corrected_{task}.npz"
        )

        if not p.exists():
            rows.append({
                "task": task,
                "npz_exists": False,
                "keys": "",
                "first_dim_preview": "",
                "sha256": "",
            })
            continue

        z = np.load(
            p,
            allow_pickle=False
        )

        dims = []

        for k in z.files:
            arr = z[k]
            dims.append(
                f"{k}:{getattr(arr, 'shape', None)}"
            )

        rows.append({
            "task": task,
            "npz_exists": True,
            "keys": ";".join(
                z.files
            ),
            "first_dim_preview": ";".join(
                dims
            ),
            "sha256": sha256_file(
                p
            ),
        })

    return pd.DataFrame(
        rows
    )


def main():
    raw = structural_raw_summary()
    ego = ego_summary()
    existing = corrected_existing_presence()
    npz_audit = inspect_corrected_npz()

    occurrence_rows = []

    for source in CANDIDATES:
        occurrence_rows.extend(
            scan_occurrences(
                source
            )
        )

    occ = pd.DataFrame(
        occurrence_rows
    )

    if len(occ) == 0:
        occ = pd.DataFrame(
            columns=[
                "source",
                "path",
                "path_hit",
                "content_hit",
                "content_hit_count",
                "model_or_eval_path",
            ]
        )

    contam_summary = []

    for source in CANDIDATES:
        g = occ[
            occ[
                "source"
            ]
            ==
            source
        ]

        bad = g[
            g[
                "model_or_eval_path"
            ]
            ==
            True
        ]

        contam_summary.append({
            "source": source,
            "all_project_occurrence_files": len(
                g
            ),
            "model_or_eval_occurrence_files": len(
                bad
            ),
            "untouched_by_model_eval_paths": len(
                bad
            ) == 0,
        })

    contam = pd.DataFrame(
        contam_summary
    )

    joint = (
        raw
        .merge(
            ego,
            on="source",
            how="left"
        )
        .merge(
            contam,
            on="source",
            how="left"
        )
    )

    joint["eligible_raw_source_for_2s_to_2s_build"] = (
        joint[
            "structurally_supports_2s_to_2s_at_30hz"
        ]
        &
        joint[
            "ego_available"
        ]
        &
        joint[
            "untouched_by_model_eval_paths"
        ]
    )

    joint["eligible_raw_source_for_2s_to_5s_build"] = (
        joint[
            "structurally_supports_2s_to_5s_at_30hz"
        ]
        &
        joint[
            "ego_available"
        ]
        &
        joint[
            "untouched_by_model_eval_paths"
        ]
    )

    joint.to_csv(
        OUT
        /
        "step6d_0054_0056_structural_viability.csv",
        index=False
    )

    occ.to_csv(
        OUT
        /
        "step6d_0054_0056_full_occurrence_audit.csv",
        index=False
    )

    existing.to_csv(
        OUT
        /
        "step6d_0054_0056_existing_corrected_window_presence.csv",
        index=False
    )

    npz_audit.to_csv(
        OUT
        /
        "step6d_corrected_npz_structure_audit.csv",
        index=False
    )

    # Pre-lock raw sources BEFORE any model prediction/error.
    eligible_any = joint[
        joint[
            "eligible_raw_source_for_2s_to_2s_build"
        ]
        |
        joint[
            "eligible_raw_source_for_2s_to_5s_build"
        ]
    ].copy()

    lock_created = len(
        eligible_any
    ) > 0

    lock = {
        "lock_stage":
            "STEP_6D_BIRDSAI_UNUSED_SOURCE_PRE_RESULT_LOCK",

        "selection_rule":
            (
                "Lock ALL exact source tokens among "
                "0000000054_0000000000 and 0000000056_0000000000 "
                "that have zero occurrence in known model/evaluation paths, "
                "have ego-motion correction data, and structurally support "
                "the requested horizon. No source is selected using model "
                "prediction or forecast-error performance."
            ),

        "model_predictions_or_forecast_errors_seen_before_lock":
            False,

        "candidate_sources":
            CANDIDATES,

        "source_structural_audit":
            joint.to_dict(
                orient="records"
            ),

        "raw_points_path":
            str(
                RAW_POINTS
            ),

        "raw_points_sha256":
            sha256_file(
                RAW_POINTS
            ),

        "ego_motion_path":
            str(
                EGO
            ),

        "ego_motion_sha256":
            (
                sha256_file(
                    EGO
                )
                if EGO.exists()
                else None
            ),

        "scope_note":
            (
                "These are previously unused BIRDSAI sequences from the same "
                "overall dataset family, so this is an untouched source-level "
                "confirmation, not an independent external-dataset confirmation."
            )
    }

    if lock_created:
        with open(
            OUT
            /
            "STEP6D_BIRDSAI_UNUSED_SOURCE_PRELOCK.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                lock,
                f,
                indent=2
            )

    with open(
        OUT
        /
        "STEP6D_H_SUMMARY.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "STEP 6D-H: UNUSED BIRDSAI SOURCE STRUCTURAL VIABILITY AUDIT\n"
        )
        f.write(
            "=" * 82 + "\n\n"
        )

        f.write(
            "No model predictions or forecast errors were computed.\n\n"
        )

        f.write(
            "SOURCE STRUCTURAL VIABILITY\n"
        )
        f.write(
            "---------------------------\n"
        )
        f.write(
            joint.to_string(
                index=False
            )
        )

        f.write(
            "\n\nEXISTING CORRECTED WINDOW PRESENCE\n"
        )
        f.write(
            "----------------------------------\n"
        )
        f.write(
            existing.to_string(
                index=False
            )
        )

        f.write(
            "\n\nCORRECTED NPZ STRUCTURE\n"
        )
        f.write(
            "-----------------------\n"
        )
        f.write(
            npz_audit.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        if lock_created:
            f.write(
                "PRE-RESULT RAW-SOURCE LOCK CREATED.\n"
            )
        else:
            f.write(
                "NO PRE-LOCK CREATED: neither source is structurally eligible.\n"
            )

    print(
        "\nSTEP 6D-H UNUSED BIRDSAI SOURCE AUDIT COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )

    print(
        "\nSOURCE STRUCTURAL VIABILITY:"
    )

    print(
        joint.to_string(
            index=False
        )
    )

    print(
        "\nEXISTING CORRECTED WINDOW PRESENCE:"
    )

    print(
        existing.to_string(
            index=False
        )
    )

    print(
        "\nCORRECTED NPZ STRUCTURE:"
    )

    print(
        npz_audit.to_string(
            index=False
        )
    )

    if lock_created:
        print(
            "\nPRE-RESULT RAW-SOURCE LOCK CREATED:"
        )

        print(
            OUT
            /
            "STEP6D_BIRDSAI_UNUSED_SOURCE_PRELOCK.json"
        )
    else:
        print(
            "\nNO PRE-RESULT LOCK CREATED."
        )


if __name__ == "__main__":
    main()
