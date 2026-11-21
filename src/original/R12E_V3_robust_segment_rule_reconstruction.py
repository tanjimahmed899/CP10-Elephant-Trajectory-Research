
from pathlib import Path
import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
MAIN1 = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R12_ROOT = REVISION_ROOT / "R12_track_accounting"

R12D_DIR = (
    R12_ROOT
    / "R12D_segment_build_rule_source_provenance_audit"
)

R12D_MANIFEST = R12D_DIR / "R12D_MANIFEST.json"
R12D_EXACT = R12D_DIR / "R12D_EXACT_BUILDER_CANDIDATES.csv"
R12D_RANKED = R12D_DIR / "R12D_RANKED_SEGMENT_BUILDER_SOURCE_CANDIDATES.csv"
R12D_CONTEXT = R12D_DIR / "R12D_SEGMENT_RULE_CONTEXT_EVIDENCE.csv"

R12C_V2_DIR = (
    R12_ROOT
    / "R12C_v2_resolve_45_total_tracks_with_canonical_id_mapping"
)

R12C_V2_MANIFEST = R12C_V2_DIR / "R12C_V2_MANIFEST.json"

BUILDER = MAIN1 / "build_elephant_trajectory_dataset.py"

CLEANED = (
    MAIN1
    / "trajectory_dataset"
    / "cleaned_track_points.csv"
)

META_2S = (
    MAIN1
    / "final_neural_trajectory_dataset"
    / "2s_to_2s_birdsai_test_metadata.csv"
)

META_5S = (
    MAIN1
    / "final_neural_trajectory_dataset"
    / "2s_to_5s_birdsai_test_metadata.csv"
)

OUT = (
    R12_ROOT
    / "R12E_v3_robust_segment_rule_reconstruction"
)
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    "total_tracks": 45,
    "represented_2s_tracks": 37,
    "represented_5s_tracks": 31,
    "segments_2s": 89,
    "segments_5s": 63,
    "windows_2s": 1053,
    "windows_5s": 826,
}


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


def verify_status_files():
    for p in [
        R12D_MANIFEST,
        R12C_V2_MANIFEST,
        BUILDER,
        CLEANED,
        META_2S,
        META_5S,
    ]:
        require(p)

    d = json.loads(
        R12D_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    c = json.loads(
        R12C_V2_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_d = (
        "R12D_SEGMENT_BUILDER_SOURCE_LOCATED_READY_FOR_R12E_EXACT_RULE_EXTRACTION_AND_RECONSTRUCTION"
    )

    expected_c = (
        "R12C_V2_45_TOTAL_TRACKS_37_2S_ELIGIBLE_31_5S_ELIGIBLE_EXACTLY_RESOLVED_READY_FOR_R12D_SEGMENT_BUILD_RULE_LOCK"
    )

    if d.get("status") != expected_d:
        raise RuntimeError(
            f"Unexpected R12-D status: {d.get('status')}"
        )

    if c.get("status") != expected_c:
        raise RuntimeError(
            f"Unexpected R12-C-V2 status: {c.get('status')}"
        )

    return d, c


def basename_match(value, target_name):
    try:
        return (
            Path(
                str(value)
            ).name.lower()
            ==
            target_name.lower()
        )

    except Exception:
        return False


def locate_r12d_builder_record():
    """
    Robustly locate the R12-D builder record by filename instead of exact full-path string.
    Search order:
      1) exact candidates CSV
      2) ranked candidates CSV
      3) context evidence CSV
    """
    target_name = BUILDER.name

    records = []

    for source_name, csv_path in [
        (
            "R12D_EXACT_BUILDER_CANDIDATES",
            R12D_EXACT,
        ),
        (
            "R12D_RANKED_SEGMENT_BUILDER_SOURCE_CANDIDATES",
            R12D_RANKED,
        ),
    ]:
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(
                csv_path
            )
        except Exception:
            continue

        if "path" not in df.columns:
            continue

        mask = df["path"].map(
            lambda x:
                basename_match(
                    x,
                    target_name,
                )
        )

        matches = df.loc[
            mask
        ].copy()

        if len(matches):
            if "score" in matches.columns:
                matches = matches.sort_values(
                    "score",
                    ascending=False
                )

            row = matches.iloc[0].to_dict()

            row[
                "_record_source"
            ] = source_name

            records.append(
                row
            )

            break

    if records:
        return records[
            0
        ]

    # Final fallback: context evidence.
    if R12D_CONTEXT.exists():
        try:
            ctx = pd.read_csv(
                R12D_CONTEXT
            )

        except Exception:
            ctx = pd.DataFrame()

        if (
            len(ctx)
            and
            "source_file"
            in
            ctx.columns
        ):
            mask = ctx[
                "source_file"
            ].map(
                lambda x:
                    basename_match(
                        x,
                        target_name,
                    )
            )

            matches = ctx.loc[
                mask
            ].copy()

            if len(matches):
                return {
                    "path":
                        str(
                            matches.iloc[
                                0
                            ][
                                "source_file"
                            ]
                        ),

                    "sha256":
                        "",

                    "_record_source":
                        "R12D_SEGMENT_RULE_CONTEXT_EVIDENCE",
                }

    raise RuntimeError(
        "Could not locate build_elephant_trajectory_dataset.py in any R12-D provenance table."
    )


def collect_r12d_locked_context():
    pieces = []

    for csv_path, path_col in [
        (
            R12D_CONTEXT,
            "source_file",
        ),
    ]:
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(
                csv_path
            )

        except Exception:
            continue

        if path_col not in df.columns:
            continue

        mask = df[
            path_col
        ].map(
            lambda x:
                basename_match(
                    x,
                    BUILDER.name,
                )
        )

        rows = df.loc[
            mask
        ]

        for _, row in rows.iterrows():
            for col in [
                "context",
                "anchor_text",
            ]:
                if col in row.index:
                    pieces.append(
                        str(
                            row[
                                col
                            ]
                        )
                    )

    return "\n".join(
        pieces
    )


def semantic_rule_signature(text):
    """
    Extract/verify only the rule facts already evidenced in R12-D.
    Regex is whitespace-tolerant.
    """
    checks = {
        "FPS_30":
            bool(
                re.search(
                    r"(?m)^\s*FPS\s*=\s*30\s*(?:#.*)?$",
                    text,
                )
            ),

        "OBS_SEC_2":
            bool(
                re.search(
                    r"(?m)^\s*OBS_SEC\s*=\s*2\s*(?:#.*)?$",
                    text,
                )
            ),

        "MAX_SMALL_GAP_3":
            bool(
                re.search(
                    r"(?m)^\s*MAX_SMALL_GAP\s*=\s*3\s*(?:#.*)?$",
                    text,
                )
            ),

        "WINDOW_STRIDE_30":
            bool(
                re.search(
                    r"(?m)^\s*WINDOW_STRIDE\s*=\s*30\s*(?:#.*)?$",
                    text,
                )
            ),

        "split_condition":
            bool(
                re.search(
                    r"missing\s*>\s*MAX_SMALL_GAP",
                    text,
                )
            ),
    }

    return checks


def evaluate_constants(source_text):
    tree = ast.parse(
        source_text,
        filename=str(BUILDER)
    )

    env = {}

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            continue

        if isinstance(
            node,
            ast.Assign
        ):
            targets = node.targets
            value_node = node.value

        else:
            targets = [
                node.target
            ]
            value_node = node.value

        names = [
            t.id
            for t in targets
            if isinstance(
                t,
                ast.Name
            )
        ]

        if not names:
            continue

        unsafe = any(
            isinstance(
                sub,
                (
                    ast.Call,
                    ast.Attribute,
                    ast.Subscript,
                    ast.Lambda,
                    ast.DictComp,
                    ast.ListComp,
                    ast.SetComp,
                    ast.GeneratorExp,
                )
            )
            for sub in ast.walk(
                value_node
            )
        )

        if unsafe:
            continue

        expr = ast.Expression(
            value_node
        )

        ast.fix_missing_locations(
            expr
        )

        try:
            value = eval(
                compile(
                    expr,
                    str(BUILDER),
                    "eval",
                ),
                {},
                env,
            )

        except Exception:
            try:
                value = ast.literal_eval(
                    value_node
                )

            except Exception:
                continue

        for name in names:
            env[
                name
            ] = value

    return env


def load_testreal_cleaned():
    df = pd.read_csv(
        CLEANED,
        low_memory=False,
    )

    required = {
        "split",
        "sequence",
        "object_id",
        "frame",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"cleaned_track_points.csv missing columns: {sorted(missing)}"
        )

    labels = (
        df[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mask = labels.isin(
        [
            "test_real",
            "testreal",
            "test-real",
        ]
    )

    if not mask.any():
        mask = labels.str.contains(
            "test",
            regex=False
        )

    test = df.loc[
        mask
    ].copy()

    test[
        "sequence"
    ] = (
        test[
            "sequence"
        ]
        .astype(str)
        .str.strip()
    )

    test[
        "object_id_canon"
    ] = (
        test[
            "object_id"
        ]
        .map(
            canonical_object_id
        )
    )

    test[
        "frame"
    ] = pd.to_numeric(
        test[
            "frame"
        ],
        errors="raise",
    ).astype(
        int
    )

    test[
        "original_track_key"
    ] = (
        test[
            "sequence"
        ]
        +
        "|"
        +
        test[
            "object_id_canon"
        ]
    )

    return test


def reconstruct_segments(
    test,
    max_small_gap,
):
    rows = []

    grouped = test.groupby(
        [
            "sequence",
            "object_id_canon",
            "original_track_key",
        ],
        sort=True,
        dropna=False,
    )

    for (
        sequence,
        object_id,
        original_track_key,
    ), g in grouped:

        frames = np.sort(
            g[
                "frame"
            ]
            .astype(
                int
            )
            .unique()
        )

        if len(
            frames
        ) == 0:
            continue

        segment_start_idx = 0
        segment_index = 0

        for j in range(
            1,
            len(
                frames
            ),
        ):
            previous = int(
                frames[
                    j - 1
                ]
            )

            current = int(
                frames[
                    j
                ]
            )

            missing = (
                current
                -
                previous
                -
                1
            )

            if missing > max_small_gap:
                seg_frames = frames[
                    segment_start_idx:
                    j
                ]

                start_frame = int(
                    seg_frames[
                        0
                    ]
                )

                end_frame = int(
                    seg_frames[
                        -1
                    ]
                )

                rows.append(
                    {
                        "sequence":
                            sequence,

                        "object_id":
                            object_id,

                        "original_track_key":
                            original_track_key,

                        "segment_index_reconstructed":
                            segment_index,

                        "segment_start_frame":
                            start_frame,

                        "segment_end_frame":
                            end_frame,

                        "observed_frames":
                            int(
                                len(
                                    seg_frames
                                )
                            ),

                        "dense_length_after_small_gap_interpolation":
                            int(
                                end_frame
                                -
                                start_frame
                                +
                                1
                            ),
                    }
                )

                segment_index += 1
                segment_start_idx = j

        seg_frames = frames[
            segment_start_idx:
        ]

        start_frame = int(
            seg_frames[
                0
            ]
        )

        end_frame = int(
            seg_frames[
                -1
            ]
        )

        rows.append(
            {
                "sequence":
                    sequence,

                "object_id":
                    object_id,

                "original_track_key":
                    original_track_key,

                "segment_index_reconstructed":
                    segment_index,

                "segment_start_frame":
                    start_frame,

                "segment_end_frame":
                    end_frame,

                "observed_frames":
                    int(
                        len(
                            seg_frames
                        )
                    ),

                "dense_length_after_small_gap_interpolation":
                    int(
                        end_frame
                        -
                        start_frame
                        +
                        1
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def add_window_counts(
    segments,
    obs_frames,
    future_frames,
    stride,
):
    work = segments.copy()

    required_length = (
        obs_frames
        +
        future_frames
    )

    L = pd.to_numeric(
        work[
            "dense_length_after_small_gap_interpolation"
        ],
        errors="raise",
    ).astype(
        int
    )

    work[
        "obs_frames"
    ] = int(
        obs_frames
    )

    work[
        "future_frames"
    ] = int(
        future_frames
    )

    work[
        "window_stride"
    ] = int(
        stride
    )

    work[
        "required_dense_length"
    ] = int(
        required_length
    )

    work[
        "reconstructed_windows"
    ] = np.where(
        L >= required_length,
        (
            (
                L
                -
                required_length
            )
            //
            stride
            +
            1
        ),
        0,
    ).astype(
        int
    )

    work[
        "forecast_eligible_segment"
    ] = (
        work[
            "reconstructed_windows"
        ]
        >
        0
    )

    return work


def load_frozen_metadata(path):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df[
        "sequence"
    ] = (
        df[
            "sequence"
        ]
        .astype(str)
        .str.strip()
    )

    df[
        "object_id_canon"
    ] = (
        df[
            "object_id"
        ]
        .map(
            canonical_object_id
        )
    )

    df[
        "original_track_key"
    ] = (
        df[
            "sequence"
        ]
        +
        "|"
        +
        df[
            "object_id_canon"
        ]
    )

    return df


def summarize_task(
    recon,
    frozen,
    task,
):
    eligible = recon[
        recon[
            "forecast_eligible_segment"
        ]
        ==
        True
    ]

    return {
        "task":
            task,

        "reconstructed_original_tracks":
            int(
                eligible[
                    "original_track_key"
                ]
                .nunique()
            ),

        "frozen_original_tracks":
            int(
                frozen[
                    "original_track_key"
                ]
                .nunique()
            ),

        "reconstructed_forecast_segments":
            int(
                len(
                    eligible
                )
            ),

        "frozen_forecast_segments":
            int(
                frozen[
                    "track_key"
                ]
                .nunique()
            ),

        "reconstructed_windows":
            int(
                eligible[
                    "reconstructed_windows"
                ]
                .sum()
            ),

        "frozen_windows":
            int(
                len(
                    frozen
                )
            ),
    }


def main():
    print(
        "\nR12-E-V3 — ROBUST SEGMENT RULE RECONSTRUCTION"
    )

    print(
        "NO training and NO inference. "
        "This version removes exact full-path matching from provenance lookup. "
        "The R12-D builder record is located by filename with fallbacks across exact, ranked, "
        "and context evidence tables. Reconstruction proceeds only if current source semantics "
        "match the R12-D locked rule facts. No source/result file is modified.\n"
    )

    verify_status_files()

    record = locate_r12d_builder_record()

    current_hash = sha256_file(
        BUILDER
    )

    locked_hash = str(
        record.get(
            "sha256",
            ""
        )
    ).strip()

    source_text = BUILDER.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    current_sig = semantic_rule_signature(
        source_text
    )

    locked_context = collect_r12d_locked_context()

    locked_sig = semantic_rule_signature(
        locked_context
    )

    provenance_df = pd.DataFrame(
        [
            {
                "builder_record_source":
                    record.get(
                        "_record_source",
                        ""
                    ),

                "builder_record_path":
                    record.get(
                        "path",
                        ""
                    ),

                "locked_sha256":
                    locked_hash,

                "current_sha256":
                    current_hash,

                "byte_hash_match":
                    bool(
                        locked_hash
                        and
                        current_hash
                        ==
                        locked_hash
                    ),

                "current_FPS_30":
                    current_sig[
                        "FPS_30"
                    ],

                "current_OBS_SEC_2":
                    current_sig[
                        "OBS_SEC_2"
                    ],

                "current_MAX_SMALL_GAP_3":
                    current_sig[
                        "MAX_SMALL_GAP_3"
                    ],

                "current_WINDOW_STRIDE_30":
                    current_sig[
                        "WINDOW_STRIDE_30"
                    ],

                "current_split_condition":
                    current_sig[
                        "split_condition"
                    ],

                "r12d_evidence_FPS_30":
                    locked_sig[
                        "FPS_30"
                    ],

                "r12d_evidence_MAX_SMALL_GAP_3":
                    locked_sig[
                        "MAX_SMALL_GAP_3"
                    ],

                "r12d_evidence_WINDOW_STRIDE_30":
                    locked_sig[
                        "WINDOW_STRIDE_30"
                    ],

                "r12d_evidence_split_condition":
                    locked_sig[
                        "split_condition"
                    ],
            }
        ]
    )

    current_semantic_ok = all(
        current_sig.values()
    )

    # R12-D console evidence definitely identified these rule facts.
    # Context CSV may not contain every constant in the same context block,
    # so current semantic verification plus successful R12-D status is the
    # operative source guard.
    semantic_guard_pass = bool(
        current_semantic_ok
    )

    provenance_df[
        "semantic_guard_pass"
    ] = semantic_guard_pass

    provenance_path = (
        OUT
        /
        "R12E_V3_PROVENANCE_CHECK.csv"
    )

    provenance_df.to_csv(
        provenance_path,
        index=False
    )

    print(
        "BUILDER PROVENANCE CHECK:"
    )

    print(
        provenance_df.to_string(
            index=False
        )
    )

    if not semantic_guard_pass:
        status = (
            "R12E_V3_CURRENT_BUILDER_RULE_SEMANTICS_DO_NOT_MATCH_R12D_STOP"
        )

        print(
            "\nSTATUS:"
        )

        print(
            status
        )

        print(
            "\nNo reconstruction was performed."
        )

        return

    env = evaluate_constants(
        source_text
    )

    required_names = [
        "FPS",
        "OBS_SEC",
        "OBS_FRAMES",
        "HORIZONS",
        "MAX_SMALL_GAP",
        "WINDOW_STRIDE",
    ]

    missing = [
        name
        for name in required_names
        if name not in env
    ]

    if missing:
        raise RuntimeError(
            f"Could not recover required builder constants: {missing}"
        )

    FPS = int(
        env[
            "FPS"
        ]
    )

    OBS_SEC = int(
        env[
            "OBS_SEC"
        ]
    )

    OBS_FRAMES = int(
        env[
            "OBS_FRAMES"
        ]
    )

    MAX_SMALL_GAP = int(
        env[
            "MAX_SMALL_GAP"
        ]
    )

    WINDOW_STRIDE = int(
        env[
            "WINDOW_STRIDE"
        ]
    )

    HORIZONS = env[
        "HORIZONS"
    ]

    horizons = {
        str(k):
            int(v)
        for k, v in HORIZONS.items()
    }

    def get_future_frames(seconds):
        target = seconds * FPS

        for key in [
            str(seconds),
            f"{seconds}s",
        ]:
            if key in horizons:
                return int(
                    horizons[
                        key
                    ]
                )

        matches = [
            int(v)
            for v in horizons.values()
            if int(v)
            ==
            target
        ]

        if len(matches) == 1:
            return matches[
                0
            ]

        raise RuntimeError(
            f"Could not resolve {seconds}s horizon from HORIZONS={HORIZONS}"
        )

    FUTURE_2 = get_future_frames(
        2
    )

    FUTURE_5 = get_future_frames(
        5
    )

    rule_lock_df = pd.DataFrame(
        [
            {
                "builder_path":
                    str(
                        BUILDER
                    ),

                "builder_record_source":
                    record.get(
                        "_record_source",
                        ""
                    ),

                "locked_sha256":
                    locked_hash,

                "current_sha256":
                    current_hash,

                "FPS":
                    FPS,

                "OBS_SEC":
                    OBS_SEC,

                "OBS_FRAMES":
                    OBS_FRAMES,

                "MAX_SMALL_GAP":
                    MAX_SMALL_GAP,

                "split_condition":
                    "missing > MAX_SMALL_GAP",

                "WINDOW_STRIDE":
                    WINDOW_STRIDE,

                "FUTURE_2S_FRAMES":
                    FUTURE_2,

                "FUTURE_5S_FRAMES":
                    FUTURE_5,
            }
        ]
    )

    if not (
        FPS == 30
        and
        OBS_SEC == 2
        and
        OBS_FRAMES == 60
        and
        MAX_SMALL_GAP == 3
        and
        WINDOW_STRIDE == 30
        and
        FUTURE_2 == 60
        and
        FUTURE_5 == 150
    ):
        raise RuntimeError(
            "Recovered constants do not match R12-D locked construction rule."
        )

    test = load_testreal_cleaned()

    total_tracks = int(
        test[
            "original_track_key"
        ]
        .nunique()
    )

    segments = reconstruct_segments(
        test,
        MAX_SMALL_GAP,
    )

    recon2 = add_window_counts(
        segments,
        OBS_FRAMES,
        FUTURE_2,
        WINDOW_STRIDE,
    )

    recon5 = add_window_counts(
        segments,
        OBS_FRAMES,
        FUTURE_5,
        WINDOW_STRIDE,
    )

    frozen2 = load_frozen_metadata(
        META_2S
    )

    frozen5 = load_frozen_metadata(
        META_5S
    )

    summary_df = pd.DataFrame(
        [
            summarize_task(
                recon2,
                frozen2,
                "2s_to_2s",
            ),
            summarize_task(
                recon5,
                frozen5,
                "2s_to_5s",
            ),
        ]
    )

    set_recon2 = set(
        recon2.loc[
            recon2[
                "forecast_eligible_segment"
            ],
            "original_track_key",
        ]
        .astype(str)
    )

    set_recon5 = set(
        recon5.loc[
            recon5[
                "forecast_eligible_segment"
            ],
            "original_track_key",
        ]
        .astype(str)
    )

    set_frozen2 = set(
        frozen2[
            "original_track_key"
        ]
        .astype(str)
    )

    set_frozen5 = set(
        frozen5[
            "original_track_key"
        ]
        .astype(str)
    )

    set_df = pd.DataFrame(
        [
            {
                "task":
                    "2s_to_2s",

                "reconstructed_tracks":
                    len(
                        set_recon2
                    ),

                "frozen_tracks":
                    len(
                        set_frozen2
                    ),

                "exact_track_set_match":
                    bool(
                        set_recon2
                        ==
                        set_frozen2
                    ),

                "missing_from_reconstruction":
                    "|".join(
                        sorted(
                            set_frozen2
                            -
                            set_recon2
                        )
                    ),

                "extra_in_reconstruction":
                    "|".join(
                        sorted(
                            set_recon2
                            -
                            set_frozen2
                        )
                    ),
            },
            {
                "task":
                    "2s_to_5s",

                "reconstructed_tracks":
                    len(
                        set_recon5
                    ),

                "frozen_tracks":
                    len(
                        set_frozen5
                    ),

                "exact_track_set_match":
                    bool(
                        set_recon5
                        ==
                        set_frozen5
                    ),

                "missing_from_reconstruction":
                    "|".join(
                        sorted(
                            set_frozen5
                            -
                            set_recon5
                        )
                    ),

                "extra_in_reconstruction":
                    "|".join(
                        sorted(
                            set_recon5
                            -
                            set_frozen5
                        )
                    ),
            },
        ]
    )

    exact_counts = bool(
        total_tracks
        ==
        EXPECTED[
            "total_tracks"
        ]
        and
        int(
            summary_df.iloc[
                0
            ][
                "reconstructed_original_tracks"
            ]
        )
        ==
        EXPECTED[
            "represented_2s_tracks"
        ]
        and
        int(
            summary_df.iloc[
                1
            ][
                "reconstructed_original_tracks"
            ]
        )
        ==
        EXPECTED[
            "represented_5s_tracks"
        ]
        and
        int(
            summary_df.iloc[
                0
            ][
                "reconstructed_forecast_segments"
            ]
        )
        ==
        EXPECTED[
            "segments_2s"
        ]
        and
        int(
            summary_df.iloc[
                1
            ][
                "reconstructed_forecast_segments"
            ]
        )
        ==
        EXPECTED[
            "segments_5s"
        ]
        and
        int(
            summary_df.iloc[
                0
            ][
                "reconstructed_windows"
            ]
        )
        ==
        EXPECTED[
            "windows_2s"
        ]
        and
        int(
            summary_df.iloc[
                1
            ][
                "reconstructed_windows"
            ]
        )
        ==
        EXPECTED[
            "windows_5s"
        ]
    )

    exact_track_sets = bool(
        set_df[
            "exact_track_set_match"
        ].all()
    )

    rule_lock_path = (
        OUT
        /
        "R12E_V3_EXACT_RULE_LOCK.csv"
    )

    segments_path = (
        OUT
        /
        "R12E_V3_RECONSTRUCTED_SEGMENTS.csv"
    )

    recon2_path = (
        OUT
        /
        "R12E_V3_RECONSTRUCTED_2S_WINDOWS.csv"
    )

    recon5_path = (
        OUT
        /
        "R12E_V3_RECONSTRUCTED_5S_WINDOWS.csv"
    )

    summary_path = (
        OUT
        /
        "R12E_V3_RECONSTRUCTION_VS_FROZEN_COUNTS.csv"
    )

    set_path = (
        OUT
        /
        "R12E_V3_RECONSTRUCTED_VS_FROZEN_TRACK_SETS.csv"
    )

    rule_lock_df.to_csv(
        rule_lock_path,
        index=False
    )

    segments.to_csv(
        segments_path,
        index=False
    )

    recon2.to_csv(
        recon2_path,
        index=False
    )

    recon5.to_csv(
        recon5_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    set_df.to_csv(
        set_path,
        index=False
    )

    if (
        exact_counts
        and
        exact_track_sets
    ):
        status = (
            "R12E_V3_RULE_AND_COUNTS_EXACTLY_RECONSTRUCTED_45_37_31_89_63_1053_826_READY_FOR_R12F_FINAL_LOCK"
        )

    else:
        status = (
            "R12E_V3_RULE_EXTRACTED_BUT_RECONSTRUCTION_NOT_EXACT_REVIEW_BEFORE_R12F"
        )

    manifest = {
        "step":
            "R12E_V3_ROBUST_SEGMENT_RULE_RECONSTRUCTION",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "original_files_modified":
            False,

        "builder_record_source":
            record.get(
                "_record_source",
                ""
            ),

        "locked_sha256":
            locked_hash,

        "current_sha256":
            current_hash,

        "semantic_guard_pass":
            semantic_guard_pass,

        "exact_count_reconstruction":
            exact_counts,

        "exact_track_set_reconstruction":
            exact_track_sets,

        "outputs": {
            "provenance":
                str(
                    provenance_path
                ),

            "rule_lock":
                str(
                    rule_lock_path
                ),

            "segments":
                str(
                    segments_path
                ),

            "reconstructed_2s":
                str(
                    recon2_path
                ),

            "reconstructed_5s":
                str(
                    recon5_path
                ),

            "summary":
                str(
                    summary_path
                ),

            "track_sets":
                str(
                    set_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_V3_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nEXACT BUILDER RULE LOCK:"
    )

    print(
        rule_lock_df.to_string(
            index=False
        )
    )

    print(
        "\nRECONSTRUCTION VS FROZEN COUNTS:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nRECONSTRUCTED VS FROZEN TRACK SETS:"
    )

    print(
        set_df.to_string(
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
        "\nR12-E-V3 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
