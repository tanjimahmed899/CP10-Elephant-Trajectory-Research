
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
R12D_CONTEXT = R12D_DIR / "R12D_SEGMENT_RULE_CONTEXT_EVIDENCE.csv"

R12C_V2_DIR = (
    R12_ROOT
    / "R12C_v2_resolve_45_total_tracks_with_canonical_id_mapping"
)
R12C_V2_MANIFEST = R12C_V2_DIR / "R12C_V2_MANIFEST.json"

BUILDER = MAIN1 / "build_elephant_trajectory_dataset.py"
CLEANED = MAIN1 / "trajectory_dataset" / "cleaned_track_points.csv"

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
    / "R12E_v2_exact_segment_rule_reconstruction"
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


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def normalized_text(text):
    # Remove only representation-level differences:
    # line endings + trailing whitespace + final blank lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


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


def verify_upstream():
    for p in [
        R12D_MANIFEST,
        R12D_EXACT,
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

    exact = pd.read_csv(
        R12D_EXACT
    )

    match = exact[
        exact["path"]
        .astype(str)
        .str.replace("/", "\\", regex=False)
        .str.lower()
        ==
        str(BUILDER).replace("/", "\\").lower()
    ]

    if len(match) != 1:
        raise RuntimeError(
            f"Expected exactly one R12-D locked row for builder, found {len(match)}"
        )

    locked_hash = str(
        match.iloc[0]["sha256"]
    )

    current_hash = sha256_file(
        BUILDER
    )

    return d, c, locked_hash, current_hash


def extract_rule_signature(source_text):
    """
    Extract only the specific source facts required for R12 reconstruction.
    This is used as a semantic provenance check if the byte hash changed.
    """
    lines = source_text.splitlines()

    signature_lines = []

    patterns = [
        r"^\s*FPS\s*=",
        r"^\s*OBS_SEC\s*=",
        r"^\s*OBS_FRAMES\s*=",
        r"^\s*HORIZONS\s*=",
        r"^\s*MAX_SMALL_GAP\s*=",
        r"^\s*WINDOW_STRIDE\s*=",
        r"missing\s*>\s*MAX_SMALL_GAP",
    ]

    for i, line in enumerate(
        lines,
        start=1
    ):
        if any(
            re.search(
                pat,
                line,
            )
            for pat in patterns
        ):
            signature_lines.append(
                f"{i}:{line.strip()}"
            )

    return "\n".join(
        signature_lines
    )


def r12d_locked_rule_evidence():
    if not R12D_CONTEXT.exists():
        return ""

    df = pd.read_csv(
        R12D_CONTEXT
    )

    if "source_file" not in df.columns:
        return ""

    rows = df[
        df["source_file"]
        .astype(str)
        .str.replace("/", "\\", regex=False)
        .str.lower()
        ==
        str(BUILDER).replace("/", "\\").lower()
    ].copy()

    if len(rows) == 0:
        return ""

    cols = [
        c
        for c in [
            "context",
            "anchor_text",
        ]
        if c in rows.columns
    ]

    if not cols:
        return ""

    pieces = []

    for _, row in rows.iterrows():
        for c in cols:
            pieces.append(
                str(
                    row[c]
                )
            )

    return "\n".join(
        pieces
    )


def provenance_check(
    locked_hash,
    current_hash,
    source_text,
):
    byte_hash_match = (
        current_hash
        ==
        locked_hash
    )

    current_norm_hash = sha256_bytes(
        normalized_text(
            source_text
        ).encode(
            "utf-8"
        )
    )

    current_signature = extract_rule_signature(
        source_text
    )

    locked_evidence = r12d_locked_rule_evidence()

    required_fragments = [
        "FPS = 30",
        "MAX_SMALL_GAP = 3",
        "WINDOW_STRIDE = 30",
        "missing > MAX_SMALL_GAP",
    ]

    current_has_required = all(
        frag in source_text
        for frag in required_fragments
    )

    locked_evidence_supports_required = all(
        frag in locked_evidence
        for frag in required_fragments
    )

    semantic_rule_match = bool(
        current_has_required
        and
        locked_evidence_supports_required
    )

    result = {
        "locked_byte_sha256":
            locked_hash,

        "current_byte_sha256":
            current_hash,

        "byte_hash_match":
            byte_hash_match,

        "current_normalized_text_sha256":
            current_norm_hash,

        "current_rule_signature":
            current_signature,

        "r12d_locked_evidence_contains_required_rule":
            locked_evidence_supports_required,

        "current_source_contains_required_rule":
            current_has_required,

        "semantic_rule_match":
            semantic_rule_match,
    }

    return result


def evaluate_constants(source_text):
    tree = ast.parse(
        source_text,
        filename=str(BUILDER)
    )

    env = {}

    # Evaluate safe top-level numeric/dict assignments in source order.
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

        # Reject calls, attrs, subscripts, comprehensions.
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

    return tree, env


def source_rule_lines(source_text):
    rows = []

    keywords = [
        "FPS",
        "OBS_SEC",
        "OBS_FRAMES",
        "HORIZONS",
        "MAX_SMALL_GAP",
        "WINDOW_STRIDE",
        "missing > MAX_SMALL_GAP",
        "segment_id",
        "obs_start_frame",
        "obs_end_frame",
        "future_end_frame",
        "interpolate_segment",
    ]

    for i, line in enumerate(
        source_text.splitlines(),
        start=1,
    ):
        if any(
            key in line
            for key in keywords
        ):
            rows.append(
                {
                    "line_number":
                        i,

                    "line":
                        line,
                }
            )

    return pd.DataFrame(
        rows
    )


def load_testreal_cleaned():
    df = pd.read_csv(
        CLEANED,
        low_memory=False,
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
            .astype(int)
            .unique()
        )

        if len(frames) == 0:
            continue

        starts = [
            0
        ]

        for j in range(
            1,
            len(frames)
        ):
            missing = (
                int(
                    frames[j]
                )
                -
                int(
                    frames[j - 1]
                )
                -
                1
            )

            if missing > max_small_gap:
                starts.append(
                    j
                )

        starts.append(
            len(frames)
        )

        for seg_idx in range(
            len(starts)
            -
            1
        ):
            seg_frames = frames[
                starts[
                    seg_idx
                ]:
                starts[
                    seg_idx
                    +
                    1
                ]
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
                        int(
                            seg_idx
                        ),

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

    needed = (
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
        "required_dense_length"
    ] = int(
        needed
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
        "reconstructed_windows"
    ] = np.where(
        L >= needed,
        (
            (
                L
                -
                needed
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
        "\nR12-E-V2 — HASH-SAFE EXACT SEGMENT RULE RECONSTRUCTION"
    )

    print(
        "NO training and NO inference. "
        "The previous R12-E stopped before reconstruction because the current builder "
        "byte hash differed from the R12-D locked hash. This patched step reads the locked "
        "hash directly from R12-D, reports the mismatch explicitly, and permits reconstruction "
        "only if the current source still contains the exact R12-D-locked rule facts "
        "(FPS=30, MAX_SMALL_GAP=3, WINDOW_STRIDE=30, and missing > MAX_SMALL_GAP). "
        "No source or result file is modified.\n"
    )

    (
        _,
        _,
        locked_hash,
        current_hash,
    ) = verify_upstream()

    source_text = BUILDER.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    provenance = provenance_check(
        locked_hash,
        current_hash,
        source_text,
    )

    provenance_df = pd.DataFrame(
        [
            provenance
        ]
    )

    provenance_path = (
        OUT
        /
        "R12E_V2_BUILDER_HASH_AND_SEMANTIC_PROVENANCE_CHECK.csv"
    )

    provenance_df.to_csv(
        provenance_path,
        index=False
    )

    print(
        "BUILDER PROVENANCE CHECK:"
    )

    print(
        provenance_df[
            [
                "locked_byte_sha256",
                "current_byte_sha256",
                "byte_hash_match",
                "r12d_locked_evidence_contains_required_rule",
                "current_source_contains_required_rule",
                "semantic_rule_match",
            ]
        ].to_string(
            index=False
        )
    )

    if not provenance[
        "semantic_rule_match"
    ]:
        status = (
            "R12E_V2_BUILDER_SOURCE_SEMANTIC_RULE_MISMATCH_STOP_REVIEW_SOURCE_CHANGE"
        )

        manifest = {
            "step":
                "R12E_V2_HASH_SAFE_SEGMENT_RULE_RECONSTRUCTION",

            "created_utc":
                utc_now(),

            "status":
                status,

            "training":
                False,

            "inference":
                False,

            "reconstruction_started":
                False,

            "locked_hash":
                locked_hash,

            "current_hash":
                current_hash,

            "semantic_rule_match":
                False,

            "output":
                str(
                    provenance_path
                ),
        }

        (
            OUT
            /
            "R12E_V2_MANIFEST.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2
            ),
            encoding="utf-8"
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

    tree, env = evaluate_constants(
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

        direct_keys = [
            str(seconds),
            f"{seconds}s",
        ]

        for key in direct_keys:
            if key in horizons:
                return int(
                    horizons[
                        key
                    ]
                )

        vals = [
            int(v)
            for v in horizons.values()
            if int(v) == target
        ]

        if len(vals) == 1:
            return vals[0]

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

                "locked_R12D_sha256":
                    locked_hash,

                "current_sha256":
                    current_hash,

                "byte_hash_match":
                    provenance[
                        "byte_hash_match"
                    ],

                "semantic_rule_match":
                    provenance[
                        "semantic_rule_match"
                    ],

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

    if (
        FPS != 30
        or
        OBS_SEC != 2
        or
        OBS_FRAMES != 60
        or
        MAX_SMALL_GAP != 3
        or
        WINDOW_STRIDE != 30
        or
        FUTURE_2 != 60
        or
        FUTURE_5 != 150
    ):
        raise RuntimeError(
            "Recovered builder constants do not match R12-D locked rule."
        )

    test = load_testreal_cleaned()

    total_tracks = int(
        test[
            "original_track_key"
        ].nunique()
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

    rule_lines_df = source_rule_lines(
        source_text
    )

    rule_lock_path = (
        OUT
        /
        "R12E_V2_EXACT_RULE_LOCK.csv"
    )

    segments_path = (
        OUT
        /
        "R12E_V2_RECONSTRUCTED_SEGMENTS.csv"
    )

    recon2_path = (
        OUT
        /
        "R12E_V2_RECONSTRUCTED_2S_WINDOWS.csv"
    )

    recon5_path = (
        OUT
        /
        "R12E_V2_RECONSTRUCTED_5S_WINDOWS.csv"
    )

    summary_path = (
        OUT
        /
        "R12E_V2_RECONSTRUCTION_VS_FROZEN_COUNTS.csv"
    )

    set_path = (
        OUT
        /
        "R12E_V2_RECONSTRUCTED_VS_FROZEN_TRACK_SETS.csv"
    )

    source_lines_path = (
        OUT
        /
        "R12E_V2_KEY_SOURCE_LINES.csv"
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

    rule_lines_df.to_csv(
        source_lines_path,
        index=False
    )

    if (
        exact_counts
        and
        exact_track_sets
    ):
        status = (
            "R12E_V2_RULE_AND_COUNTS_EXACTLY_RECONSTRUCTED_45_37_31_89_63_1053_826_READY_FOR_R12F_FINAL_LOCK"
        )

    else:
        status = (
            "R12E_V2_RULE_EXTRACTED_BUT_COUNTS_NOT_EXACT_REVIEW_RECONSTRUCTION"
        )

    manifest = {
        "step":
            "R12E_V2_HASH_SAFE_SEGMENT_RULE_RECONSTRUCTION",

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

        "locked_R12D_hash":
            locked_hash,

        "current_builder_hash":
            current_hash,

        "byte_hash_match":
            provenance[
                "byte_hash_match"
            ],

        "semantic_rule_match":
            provenance[
                "semantic_rule_match"
            ],

        "exact_count_reconstruction":
            exact_counts,

        "exact_track_set_reconstruction":
            exact_track_sets,

        "outputs": {
            "provenance_check":
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

            "source_lines":
                str(
                    source_lines_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_V2_MANIFEST.json"
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
        "\nR12-E-V2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
