
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R12_ROOT = REVISION_ROOT / "R12_track_accounting"

OLD_R12C_DIR = (
    R12_ROOT
    / "R12C_resolve_45_total_tracks_and_forecast_eligibility"
)
OLD_R12C_MANIFEST = OLD_R12C_DIR / "R12C_MANIFEST.json"

FINAL_NEURAL = MAIN1 / "final_neural_trajectory_dataset"
UPSTREAM_45 = (
    MAIN1
    / "trajectory_dataset"
    / "cleaned_track_points.csv"
)

OUT = (
    R12_ROOT
    / "R12C_v2_resolve_45_total_tracks_with_canonical_id_mapping"
)
OUT.mkdir(parents=True, exist_ok=True)

TASK_META = {
    "2s_to_2s":
        FINAL_NEURAL
        / "2s_to_2s_birdsai_test_metadata.csv",

    "2s_to_5s":
        FINAL_NEURAL
        / "2s_to_5s_birdsai_test_metadata.csv",
}

EXPECTED = {
    "total_tracks": 45,
    "2s_tracks": 37,
    "5s_tracks": 31,
    "2s_segments": 89,
    "5s_segments": 63,
}

EXPECTED_UPSTREAM_SHA256 = (
    "bd97998b80af087deff10bac8e342f32d5d7b655c5088da03efc79a0ff74477b"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def canonical_object_id(value):
    """
    Canonicalize numeric-looking IDs only for identity matching.
    Examples:
      1      -> "1"
      "1"    -> "1"
      1.0    -> "1"
      "1.0"  -> "1"
      "001"  -> "1"
    Non-numeric strings are preserved trimmed.
    """
    if pd.isna(value):
        return ""

    s = str(value).strip()

    if s == "":
        return ""

    try:
        x = float(s)

        if np.isfinite(x) and x.is_integer():
            return str(int(x))

    except Exception:
        pass

    return s


def canonical_sequence(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def verify_old_r12c():
    require(OLD_R12C_MANIFEST)

    m = json.loads(
        OLD_R12C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_status = (
        "R12C_45_SOURCE_FOUND_BUT_ELIGIBILITY_COUNTS_DO_NOT_MATCH_EXPECTATION_REVIEW"
    )

    if m.get("status") != expected_status:
        raise RuntimeError(
            f"Unexpected old R12-C status: {m.get('status')}"
        )

    return m


def load_upstream_45():
    require(UPSTREAM_45)

    current_hash = sha256_file(
        UPSTREAM_45
    )

    if current_hash != EXPECTED_UPSTREAM_SHA256:
        raise RuntimeError(
            "Authoritative cleaned_track_points.csv hash changed."
        )

    df = pd.read_csv(
        UPSTREAM_45,
        low_memory=False,
    )

    required = {
        "split",
        "sequence",
        "object_id",
        "frame",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Upstream source missing columns: {sorted(missing)}"
        )

    test_mask = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "test_real",
                "testreal",
                "test-real",
            ]
        )
    )

    if not test_mask.any():
        test_mask = (
            df["split"]
            .astype(str)
            .str.lower()
            .str.contains(
                "test",
                regex=False
            )
        )

    test = df.loc[
        test_mask
    ].copy()

    test["sequence_canon"] = (
        test["sequence"]
        .map(
            canonical_sequence
        )
    )

    test["object_id_canon"] = (
        test["object_id"]
        .map(
            canonical_object_id
        )
    )

    test["original_track_key"] = (
        test["sequence_canon"]
        +
        "|"
        +
        test["object_id_canon"]
    )

    tracks = (
        test[
            [
                "sequence_canon",
                "object_id_canon",
                "original_track_key",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "sequence_canon": "sequence",
                "object_id_canon": "object_id",
            }
        )
        .sort_values(
            [
                "sequence",
                "object_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        test,
        tracks,
        current_hash,
    )


def load_forecast_metadata(task, path):
    require(path)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "sequence",
        "object_id",
        "segment_id",
        "track_key",
        "sample_id",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"{task}: missing columns {sorted(missing)}"
        )

    df["sequence_canon"] = (
        df["sequence"]
        .map(
            canonical_sequence
        )
    )

    df["object_id_canon"] = (
        df["object_id"]
        .map(
            canonical_object_id
        )
    )

    df["original_track_key"] = (
        df["sequence_canon"]
        +
        "|"
        +
        df["object_id_canon"]
    )

    return df


def track_length_table(upstream_test):
    frame = pd.to_numeric(
        upstream_test["frame"],
        errors="coerce",
    )

    tmp = upstream_test.copy()
    tmp["frame_numeric"] = frame

    rows = []

    for key, g in tmp.groupby(
        "original_track_key",
        sort=True,
    ):
        vals = (
            g["frame_numeric"]
            .dropna()
            .sort_values()
            .unique()
        )

        seq = str(
            g["sequence_canon"]
            .iloc[0]
        )

        obj = str(
            g["object_id_canon"]
            .iloc[0]
        )

        if len(vals):
            diffs = np.diff(
                vals
            )

            rows.append(
                {
                    "sequence":
                        seq,

                    "object_id":
                        obj,

                    "original_track_key":
                        key,

                    "rows":
                        int(
                            len(g)
                        ),

                    "unique_frames":
                        int(
                            len(vals)
                        ),

                    "min_frame":
                        float(
                            vals.min()
                        ),

                    "max_frame":
                        float(
                            vals.max()
                        ),

                    "frame_span":
                        float(
                            vals.max()
                            -
                            vals.min()
                            +
                            1
                        ),

                    "max_frame_gap":
                        (
                            float(
                                diffs.max()
                            )
                            if len(diffs)
                            else 0.0
                        ),

                    "median_frame_gap":
                        (
                            float(
                                np.median(
                                    diffs
                                )
                            )
                            if len(diffs)
                            else 0.0
                        ),
                }
            )

        else:
            rows.append(
                {
                    "sequence":
                        seq,

                    "object_id":
                        obj,

                    "original_track_key":
                        key,

                    "rows":
                        int(
                            len(g)
                        ),

                    "unique_frames":
                        0,

                    "min_frame":
                        np.nan,

                    "max_frame":
                        np.nan,

                    "frame_span":
                        np.nan,

                    "max_frame_gap":
                        np.nan,

                    "median_frame_gap":
                        np.nan,
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR12-C-V2 — 45 TOTAL TESTREAL TRACKS WITH CANONICAL ID MAPPING"
    )

    print(
        "PATCHED DIAGNOSTIC ONLY. NO training and NO inference. "
        "The previous R12-C correctly identified the authoritative 45-track source, "
        "but compared object_id values as raw strings. In cleaned_track_points.csv, "
        "IDs were parsed as values such as 1.0/10.0, while frozen forecasting metadata "
        "uses integer-style IDs such as 1/10. This step canonicalizes only identity "
        "formatting before set comparison; no trajectory coordinates, windows, labels, "
        "or model results are changed.\n"
    )

    verify_old_r12c()

    (
        upstream_test,
        total_tracks,
        upstream_hash,
    ) = load_upstream_45()

    meta_2s = load_forecast_metadata(
        "2s_to_2s",
        TASK_META[
            "2s_to_2s"
        ],
    )

    meta_5s = load_forecast_metadata(
        "2s_to_5s",
        TASK_META[
            "2s_to_5s"
        ],
    )

    set_total = set(
        total_tracks[
            "original_track_key"
        ].astype(str)
    )

    set_2s = set(
        meta_2s[
            "original_track_key"
        ].astype(str)
        .unique()
    )

    set_5s = set(
        meta_5s[
            "original_track_key"
        ].astype(str)
        .unique()
    )

    # Integrity: every forecast-represented track must come from upstream TestReal.
    extra_2s = sorted(
        set_2s
        -
        set_total
    )

    extra_5s = sorted(
        set_5s
        -
        set_total
    )

    if extra_2s:
        raise RuntimeError(
            f"2s forecast metadata contains tracks absent from authoritative TestReal source: {extra_2s}"
        )

    if extra_5s:
        raise RuntimeError(
            f"5s forecast metadata contains tracks absent from authoritative TestReal source: {extra_5s}"
        )

    if not set_5s.issubset(
        set_2s
    ):
        raise RuntimeError(
            "5s represented-track set is not a subset of 2s represented-track set."
        )

    eligibility_rows = []

    for _, row in total_tracks.iterrows():
        key = str(
            row[
                "original_track_key"
            ]
        )

        in_2s = key in set_2s
        in_5s = key in set_5s

        if in_5s:
            cls = (
                "ELIGIBLE_2S_AND_5S"
            )

        elif in_2s:
            cls = (
                "ELIGIBLE_2S_ONLY"
            )

        else:
            cls = (
                "NO_VALID_2S_FORECAST_WINDOWS"
            )

        eligibility_rows.append(
            {
                "sequence":
                    row[
                        "sequence"
                    ],

                "object_id":
                    row[
                        "object_id"
                    ],

                "original_track_key":
                    key,

                "represented_in_2s":
                    in_2s,

                "represented_in_5s":
                    in_5s,

                "eligibility_class":
                    cls,
            }
        )

    eligibility = pd.DataFrame(
        eligibility_rows
    )

    lengths = track_length_table(
        upstream_test
    )

    detailed = eligibility.merge(
        lengths,
        on=[
            "sequence",
            "object_id",
            "original_track_key",
        ],
        how="left",
        validate="one_to_one",
    )

    class_summary = (
        eligibility.groupby(
            "eligibility_class",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "tracks"
            }
        )
    )

    total_n = len(
        set_total
    )

    n_2s = len(
        set_2s
    )

    n_5s = len(
        set_5s
    )

    n_both = int(
        (
            eligibility[
                "eligibility_class"
            ]
            ==
            "ELIGIBLE_2S_AND_5S"
        ).sum()
    )

    n_2s_only = int(
        (
            eligibility[
                "eligibility_class"
            ]
            ==
            "ELIGIBLE_2S_ONLY"
        ).sum()
    )

    n_no_2s = int(
        (
            eligibility[
                "eligibility_class"
            ]
            ==
            "NO_VALID_2S_FORECAST_WINDOWS"
        ).sum()
    )

    segments_2s = int(
        meta_2s[
            "track_key"
        ].nunique()
    )

    segments_5s = int(
        meta_5s[
            "track_key"
        ].nunique()
    )

    expected_counts_pass = bool(
        total_n
        ==
        EXPECTED[
            "total_tracks"
        ]
        and
        n_2s
        ==
        EXPECTED[
            "2s_tracks"
        ]
        and
        n_5s
        ==
        EXPECTED[
            "5s_tracks"
        ]
        and
        segments_2s
        ==
        EXPECTED[
            "2s_segments"
        ]
        and
        segments_5s
        ==
        EXPECTED[
            "5s_segments"
        ]
        and
        n_no_2s
        ==
        8
        and
        n_2s_only
        ==
        6
        and
        n_both
        ==
        31
    )

    # Save outputs.
    source_lock = pd.DataFrame(
        [
            {
                "authoritative_source":
                    str(
                        UPSTREAM_45
                    ),

                "sha256":
                    upstream_hash,

                "testreal_rows":
                    int(
                        len(
                            upstream_test
                        )
                    ),

                "canonical_id_rule":
                    (
                        "numeric object_id values are normalized to integer strings "
                        "(e.g., 1.0 -> 1) before identity matching"
                    ),

                "total_original_tracks":
                    total_n,
            }
        ]
    )

    forecast_lock = pd.DataFrame(
        [
            {
                "task":
                    "2s_to_2s",

                "metadata_path":
                    str(
                        TASK_META[
                            "2s_to_2s"
                        ]
                    ),

                "metadata_sha256":
                    sha256_file(
                        TASK_META[
                            "2s_to_2s"
                        ]
                    ),

                "represented_original_tracks":
                    n_2s,

                "forecast_segments":
                    segments_2s,

                "samples":
                    int(
                        len(
                            meta_2s
                        )
                    ),
            },
            {
                "task":
                    "2s_to_5s",

                "metadata_path":
                    str(
                        TASK_META[
                            "2s_to_5s"
                        ]
                    ),

                "metadata_sha256":
                    sha256_file(
                        TASK_META[
                            "2s_to_5s"
                        ]
                    ),

                "represented_original_tracks":
                    n_5s,

                "forecast_segments":
                    segments_5s,

                "samples":
                    int(
                        len(
                            meta_5s
                        )
                    ),
            },
        ]
    )

    source_lock_path = (
        OUT
        /
        "R12C_V2_AUTHORITATIVE_45_TRACK_SOURCE_LOCK.csv"
    )

    forecast_lock_path = (
        OUT
        /
        "R12C_V2_FORECAST_REPRESENTATION_LOCK.csv"
    )

    eligibility_path = (
        OUT
        /
        "R12C_V2_TRACK_FORECAST_ELIGIBILITY.csv"
    )

    detailed_path = (
        OUT
        /
        "R12C_V2_TRACK_ELIGIBILITY_WITH_LENGTH_EVIDENCE.csv"
    )

    class_summary_path = (
        OUT
        /
        "R12C_V2_ELIGIBILITY_CLASS_SUMMARY.csv"
    )

    source_lock.to_csv(
        source_lock_path,
        index=False
    )

    forecast_lock.to_csv(
        forecast_lock_path,
        index=False
    )

    eligibility.to_csv(
        eligibility_path,
        index=False
    )

    detailed.to_csv(
        detailed_path,
        index=False
    )

    class_summary.to_csv(
        class_summary_path,
        index=False
    )

    if expected_counts_pass:
        status = (
            "R12C_V2_45_TOTAL_TRACKS_37_2S_ELIGIBLE_31_5S_ELIGIBLE_EXACTLY_RESOLVED_READY_FOR_R12D_SEGMENT_BUILD_RULE_LOCK"
        )
    else:
        status = (
            "R12C_V2_CANONICAL_ID_MAPPING_COMPLETE_COUNTS_STILL_REQUIRE_REVIEW"
        )

    manuscript = f"""R12 track-accounting population resolution — canonical ID corrected

Authoritative TestReal population
---------------------------------
The BIRDSAI TestReal portion of cleaned_track_points.csv contains {total_n} unique original tracks after canonicalizing object_id representation. Original track identity is (sequence, object_id).

The previous R12-C mismatch was an identity-formatting artifact: upstream object IDs were parsed as numeric-looking strings such as "1.0", whereas the frozen forecasting metadata uses integer-style IDs such as "1". Canonicalizing only the ID representation resolves the set mapping without altering any trajectory data.

Forecast eligibility
--------------------
Total TestReal original tracks: {total_n}

Eligible for both 2 s and 5 s forecasting: {n_both}
Eligible for 2 s but not 5 s forecasting: {n_2s_only}
No valid 2 s forecasting window: {n_no_2s}

Therefore:

45 total original TestReal tracks
→ {n_2s} tracks represented in the 2 s task
→ {segments_2s} forecastable 2 s segments/tracklets
→ {n_5s} tracks represented in the 5 s task
→ {segments_5s} forecastable 5 s segments/tracklets

Reviewer-safe wording
---------------------
"The BIRDSAI TestReal split contains 45 original tracks. After forecasting-window eligibility filtering, 37 original tracks contribute to the 2-s task and 31 contribute to the 5-s task. Continuity-based segmentation of these eligible trajectories yields 89 forecastable 2-s segments and 63 forecastable 5-s segments."

Do not write:
"TestReal contains 89 tracks."

Remaining R12 task
------------------
R12-D must lock the exact continuity/gap rule used to construct segment_id and explain why eligible original tracks split into multiple forecastable segments.
"""

    manuscript_path = (
        OUT
        /
        "R12C_V2_MANUSCRIPT_READY_45_37_31_RESOLUTION.txt"
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8"
    )

    manifest = {
        "step":
            "R12C_V2_CANONICAL_ID_TRACK_ELIGIBILITY_RESOLUTION",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "data_modified":
            False,

        "identity_normalization_only":
            True,

        "canonical_object_id_rule":
            "numeric-looking IDs normalize to integer string form; e.g. 1.0 -> 1",

        "total_testreal_tracks":
            total_n,

        "represented_2s_tracks":
            n_2s,

        "represented_5s_tracks":
            n_5s,

        "2s_and_5s_eligible_tracks":
            n_both,

        "2s_only_tracks":
            n_2s_only,

        "no_valid_2s_tracks":
            n_no_2s,

        "2s_forecast_segments":
            segments_2s,

        "5s_forecast_segments":
            segments_5s,

        "counts_exactly_match_expected_resolution":
            expected_counts_pass,

        "remaining_requirement":
            (
                "Lock exact continuity/gap segmentation rule producing segment_id."
            ),

        "outputs": {
            "authoritative_source_lock":
                str(
                    source_lock_path
                ),

            "forecast_representation_lock":
                str(
                    forecast_lock_path
                ),

            "eligibility":
                str(
                    eligibility_path
                ),

            "eligibility_with_length_evidence":
                str(
                    detailed_path
                ),

            "eligibility_summary":
                str(
                    class_summary_path
                ),

            "manuscript_ready_resolution":
                str(
                    manuscript_path
                ),
        },
    }

    (
        OUT
        /
        "R12C_V2_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "AUTHORITATIVE 45-TRACK SOURCE LOCK:"
    )

    print(
        source_lock.to_string(
            index=False
        )
    )

    print(
        "\nFORECAST REPRESENTATION LOCK:"
    )

    print(
        forecast_lock.to_string(
            index=False
        )
    )

    print(
        "\nELIGIBILITY CLASS SUMMARY:"
    )

    print(
        class_summary.to_string(
            index=False
        )
    )

    print(
        "\nTRACK ACCOUNTING RESOLUTION:"
    )

    print(
        f"  Total TestReal original tracks: {total_n}"
    )

    print(
        f"  2s + 5s eligible: {n_both}"
    )

    print(
        f"  2s-only eligible: {n_2s_only}"
    )

    print(
        f"  No valid 2s window: {n_no_2s}"
    )

    print(
        f"  2s represented tracks -> segments: {n_2s} -> {segments_2s}"
    )

    print(
        f"  5s represented tracks -> segments: {n_5s} -> {segments_5s}"
    )

    print(
        "\nTRACKS NOT REPRESENTED IN 2s:"
    )

    print(
        detailed[
            detailed[
                "represented_in_2s"
            ]
            ==
            False
        ].to_string(
            index=False
        )
    )

    print(
        "\nTRACKS REPRESENTED IN 2s BUT NOT 5s:"
    )

    print(
        detailed[
            (
                detailed[
                    "represented_in_2s"
                ]
                ==
                True
            )
            &
            (
                detailed[
                    "represented_in_5s"
                ]
                ==
                False
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR12-C-V2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
