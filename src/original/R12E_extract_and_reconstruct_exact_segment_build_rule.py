
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

R12D_DIR = (
    R12_ROOT
    / "R12D_segment_build_rule_source_provenance_audit"
)
R12D_MANIFEST = R12D_DIR / "R12D_MANIFEST.json"

R12C_V2_DIR = (
    R12_ROOT
    / "R12C_v2_resolve_45_total_tracks_with_canonical_id_mapping"
)
R12C_V2_MANIFEST = R12C_V2_DIR / "R12C_V2_MANIFEST.json"

BUILDER = MAIN1 / "build_elephant_trajectory_dataset.py"
EXPECTED_BUILDER_SHA256 = (
    "3395a100b1199383a965216bfe8167e31b6d0382bdd82ced62edba3576c94b51"
)

# R12-D console reported this hash:
R12D_REPORTED_BUILDER_SHA256 = (
    "3395a100b1199383a965216bfe8167e31b6d0382bdd82ced62edba3576c94b51"
)

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
    / "R12E_exact_segment_rule_extraction_and_reconstruction"
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


def verify_upstream():
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

    current_builder_hash = sha256_file(
        BUILDER
    )

    if current_builder_hash != R12D_REPORTED_BUILDER_SHA256:
        raise RuntimeError(
            "build_elephant_trajectory_dataset.py hash differs from R12-D locked source."
        )

    return d, c, current_builder_hash


def ast_constants(source_text):
    tree = ast.parse(
        source_text,
        filename=str(BUILDER)
    )

    constants = {}

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            continue

        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value

        else:
            targets = [
                node.target
            ]
            value_node = node.value

        names = []

        for t in targets:
            if isinstance(
                t,
                ast.Name
            ):
                names.append(
                    t.id
                )

        if not names:
            continue

        try:
            value = ast.literal_eval(
                value_node
            )

        except Exception:
            continue

        for name in names:
            constants[
                name
            ] = value

    return tree, constants


def evaluate_known_constants(source_text):
    """
    AST literal_eval cannot evaluate OBS_FRAMES = OBS_SEC * FPS.
    Execute only simple top-level constant assignments that do not involve imports,
    function calls, attribute access, or subscripting.
    """
    tree = ast.parse(
        source_text,
        filename=str(BUILDER)
    )

    safe_nodes = []

    allowed_names = set()

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            )
        ):
            continue

        value = node.value

        unsafe = False

        for sub in ast.walk(
            value
        ):
            if isinstance(
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
            ):
                unsafe = True
                break

        if unsafe:
            continue

        safe_nodes.append(
            node
        )

    mod = ast.Module(
        body=safe_nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        mod
    )

    ns = {}

    try:
        exec(
            compile(
                mod,
                str(BUILDER),
                "exec",
            ),
            {},
            ns,
        )

    except Exception:
        # Fall back to partial manual extraction.
        pass

    return {
        k: v
        for k, v in ns.items()
        if not k.startswith("__")
    }


def extract_rule_functions(
    source_text,
    tree,
):
    lines = source_text.splitlines()

    function_rows = []
    source_blocks = []

    trigger_terms = {
        "segment",
        "gap",
        "interpolate",
        "window",
        "horizon",
        "history",
        "future",
        "stride",
    }

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            continue

        start = int(
            node.lineno
        )

        end = int(
            getattr(
                node,
                "end_lineno",
                start,
            )
        )

        block = "\n".join(
            lines[
                start - 1:
                end
            ]
        )

        low = block.lower()

        matched = sorted(
            term
            for term in trigger_terms
            if term in low
        )

        if not matched:
            continue

        function_rows.append(
            {
                "function_name":
                    node.name,

                "start_line":
                    start,

                "end_line":
                    end,

                "matched_terms":
                    "|".join(
                        matched
                    ),

                "contains_MAX_SMALL_GAP":
                    "MAX_SMALL_GAP"
                    in
                    block,

                "contains_WINDOW_STRIDE":
                    "WINDOW_STRIDE"
                    in
                    block,

                "contains_segment_id":
                    "segment_id"
                    in
                    block,
            }
        )

        source_blocks.append(
            {
                "function_name":
                    node.name,

                "start_line":
                    start,

                "end_line":
                    end,

                "source":
                    block,
            }
        )

    return (
        pd.DataFrame(
            function_rows
        ),
        pd.DataFrame(
            source_blocks
        ),
    )


def extract_key_source_lines(
    source_text,
):
    lines = source_text.splitlines()

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

    rows = []

    for i, line in enumerate(
        lines,
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


def split_segments_from_source_rule(
    test,
    max_small_gap,
):
    """
    Reconstruct the source rule visible in build_elephant_trajectory_dataset.py:
    split an original track when the number of missing frames between consecutive
    observed frames exceeds MAX_SMALL_GAP.

    missing = current_frame - previous_frame - 1
    new segment iff missing > MAX_SMALL_GAP

    Small gaps up to MAX_SMALL_GAP are therefore retained within one segment and
    subsequently eligible for interpolation by the builder.
    """
    rows = []

    segment_table = []

    for (
        sequence,
        object_id,
        original_track_key,
    ), g in test.groupby(
        [
            "sequence",
            "object_id_canon",
            "original_track_key",
        ],
        sort=True,
        dropna=False,
    ):
        frames = np.sort(
            g[
                "frame"
            ].astype(
                int
            ).unique()
        )

        if len(frames) == 0:
            continue

        current = [
            int(
                frames[0]
            )
        ]

        segment_index = 0

        for frame in frames[
            1:
        ]:
            frame = int(
                frame
            )

            previous = int(
                current[
                    -1
                ]
            )

            missing = (
                frame
                -
                previous
                -
                1
            )

            if missing > max_small_gap:
                segment_table.append(
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
                            int(
                                current[
                                    0
                                ]
                            ),

                        "segment_end_frame":
                            int(
                                current[
                                    -1
                                ]
                            ),

                        "observed_frames":
                            int(
                                len(
                                    current
                                )
                            ),

                        "dense_length_after_small_gap_interpolation":
                            int(
                                current[
                                    -1
                                ]
                                -
                                current[
                                    0
                                ]
                                +
                                1
                            ),
                    }
                )

                segment_index += 1

                current = [
                    frame
                ]

            else:
                current.append(
                    frame
                )

        segment_table.append(
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
                    int(
                        current[
                            0
                        ]
                    ),

                "segment_end_frame":
                    int(
                        current[
                            -1
                        ]
                    ),

                "observed_frames":
                    int(
                        len(
                            current
                        )
                    ),

                "dense_length_after_small_gap_interpolation":
                    int(
                        current[
                            -1
                        ]
                        -
                        current[
                            0
                        ]
                        +
                        1
                    ),
            }
        )

    return pd.DataFrame(
        segment_table
    )


def reconstruct_window_counts(
    segments,
    obs_frames,
    future_frames,
    stride,
):
    """
    Reconstruct window count for each dense segment after small-gap interpolation.

    For a dense segment of length L:
      a valid window needs OBS_FRAMES + FUTURE_FRAMES frames total.
      starts advance by WINDOW_STRIDE.
    """
    rows = []

    needed = (
        obs_frames
        +
        future_frames
    )

    for _, row in segments.iterrows():
        L = int(
            row[
                "dense_length_after_small_gap_interpolation"
            ]
        )

        if L < needed:
            windows = 0

        else:
            windows = (
                (
                    L
                    -
                    needed
                )
                //
                stride
                +
                1
            )

        rec = row.to_dict()

        rec[
            "obs_frames"
        ] = int(
            obs_frames
        )

        rec[
            "future_frames"
        ] = int(
            future_frames
        )

        rec[
            "window_stride"
        ] = int(
            stride
        )

        rec[
            "required_dense_length"
        ] = int(
            needed
        )

        rec[
            "reconstructed_windows"
        ] = int(
            windows
        )

        rec[
            "forecast_eligible_segment"
        ] = bool(
            windows > 0
        )

        rows.append(
            rec
        )

    return pd.DataFrame(
        rows
    )


def load_frozen_metadata(
    path,
):
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


def task_summary(
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
                ].nunique()
            ),

        "frozen_original_tracks":
            int(
                frozen[
                    "original_track_key"
                ].nunique()
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
                ].nunique()
            ),

        "reconstructed_windows":
            int(
                eligible[
                    "reconstructed_windows"
                ].sum()
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
        "\nR12-E — EXACT SEGMENT RULE EXTRACTION AND COUNT RECONSTRUCTION"
    )

    print(
        "NO training and NO inference. "
        "This step uses the R12-D locked source build_elephant_trajectory_dataset.py. "
        "It extracts the builder constants/rule lines, reconstructs original-track "
        "segmentation from cleaned_track_points.csv using the source gap rule, then "
        "reconstructs 2s/5s forecast-window eligibility and compares the resulting "
        "track/segment/window counts against the frozen metadata. "
        "No original files are modified.\n"
    )

    _, _, builder_hash = verify_upstream()

    source_text = BUILDER.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree, literal_constants = ast_constants(
        source_text
    )

    evaluated_constants = evaluate_known_constants(
        source_text
    )

    merged_constants = dict(
        literal_constants
    )

    merged_constants.update(
        evaluated_constants
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
        if name not in merged_constants
    ]

    if missing:
        raise RuntimeError(
            f"Could not recover required builder constants: {missing}"
        )

    FPS = int(
        merged_constants[
            "FPS"
        ]
    )

    OBS_SEC = int(
        merged_constants[
            "OBS_SEC"
        ]
    )

    OBS_FRAMES = int(
        merged_constants[
            "OBS_FRAMES"
        ]
    )

    HORIZONS = merged_constants[
        "HORIZONS"
    ]

    MAX_SMALL_GAP = int(
        merged_constants[
            "MAX_SMALL_GAP"
        ]
    )

    WINDOW_STRIDE = int(
        merged_constants[
            "WINDOW_STRIDE"
        ]
    )

    # Normalize HORIZONS dict keys/values.
    horizons_norm = {
        str(k): int(v)
        for k, v in HORIZONS.items()
    }

    # Expected common source forms:
    # {2: 60, 5: 150} or {"2s": 60, "5s": 150}, etc.
    def find_horizon_frames(seconds):
        candidates = [
            str(seconds),
            f"{seconds}s",
            f"{seconds}s_to_{seconds}s",
        ]

        for key in candidates:
            if key in horizons_norm:
                return int(
                    horizons_norm[
                        key
                    ]
                )

        # Fall back to unique value equal to seconds*FPS.
        target = seconds * FPS

        matches = [
            int(v)
            for v in horizons_norm.values()
            if int(v) == target
        ]

        if len(matches) == 1:
            return matches[0]

        raise RuntimeError(
            f"Could not identify {seconds}s horizon in HORIZONS={HORIZONS}"
        )

    FUTURE_2 = find_horizon_frames(
        2
    )

    FUTURE_5 = find_horizon_frames(
        5
    )

    if OBS_FRAMES != OBS_SEC * FPS:
        raise RuntimeError(
            f"OBS_FRAMES mismatch: {OBS_FRAMES} vs OBS_SEC*FPS={OBS_SEC*FPS}"
        )

    if FUTURE_2 != 2 * FPS:
        raise RuntimeError(
            f"2s future frames mismatch: {FUTURE_2} vs {2*FPS}"
        )

    if FUTURE_5 != 5 * FPS:
        raise RuntimeError(
            f"5s future frames mismatch: {FUTURE_5} vs {5*FPS}"
        )

    function_df, block_df = extract_rule_functions(
        source_text,
        tree,
    )

    key_lines_df = extract_key_source_lines(
        source_text
    )

    constants_df = pd.DataFrame(
        [
            {
                "name":
                    "FPS",

                "value":
                    FPS,
            },
            {
                "name":
                    "OBS_SEC",

                "value":
                    OBS_SEC,
            },
            {
                "name":
                    "OBS_FRAMES",

                "value":
                    OBS_FRAMES,
            },
            {
                "name":
                    "MAX_SMALL_GAP",

                "value":
                    MAX_SMALL_GAP,
            },
            {
                "name":
                    "WINDOW_STRIDE",

                "value":
                    WINDOW_STRIDE,
            },
            {
                "name":
                    "FUTURE_2S_FRAMES",

                "value":
                    FUTURE_2,
            },
            {
                "name":
                    "FUTURE_5S_FRAMES",

                "value":
                    FUTURE_5,
            },
        ]
    )

    test = load_testreal_cleaned()

    total_tracks = int(
        test[
            "original_track_key"
        ].nunique()
    )

    segments = split_segments_from_source_rule(
        test,
        MAX_SMALL_GAP,
    )

    recon2 = reconstruct_window_counts(
        segments,
        OBS_FRAMES,
        FUTURE_2,
        WINDOW_STRIDE,
    )

    recon5 = reconstruct_window_counts(
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
            task_summary(
                recon2,
                frozen2,
                "2s_to_2s",
            ),
            task_summary(
                recon5,
                frozen5,
                "2s_to_5s",
            ),
        ]
    )

    # Compare reconstructed represented-track sets.
    set_recon2 = set(
        recon2.loc[
            recon2[
                "forecast_eligible_segment"
            ],
            "original_track_key",
        ].astype(str)
    )

    set_recon5 = set(
        recon5.loc[
            recon5[
                "forecast_eligible_segment"
            ],
            "original_track_key",
        ].astype(str)
    )

    set_frozen2 = set(
        frozen2[
            "original_track_key"
        ].astype(str)
    )

    set_frozen5 = set(
        frozen5[
            "original_track_key"
        ].astype(str)
    )

    set_compare_df = pd.DataFrame(
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

                "exact_track_set_match":
                    bool(
                        set_recon2
                        ==
                        set_frozen2
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

                "exact_track_set_match":
                    bool(
                        set_recon5
                        ==
                        set_frozen5
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
            summary_df.loc[
                summary_df[
                    "task"
                ]
                ==
                "2s_to_2s",
                "reconstructed_original_tracks",
            ].iloc[
                0
            ]
        )
        ==
        EXPECTED[
            "represented_2s_tracks"
        ]
        and
        int(
            summary_df.loc[
                summary_df[
                    "task"
                ]
                ==
                "2s_to_5s",
                "reconstructed_original_tracks",
            ].iloc[
                0
            ]
        )
        ==
        EXPECTED[
            "represented_5s_tracks"
        ]
        and
        int(
            summary_df.loc[
                summary_df[
                    "task"
                ]
                ==
                "2s_to_2s",
                "reconstructed_forecast_segments",
            ].iloc[
                0
            ]
        )
        ==
        EXPECTED[
            "segments_2s"
        ]
        and
        int(
            summary_df.loc[
                summary_df[
                    "task"
                ]
                ==
                "2s_to_5s",
                "reconstructed_forecast_segments",
            ].iloc[
                0
            ]
        )
        ==
        EXPECTED[
            "segments_5s"
        ]
        and
        int(
            summary_df.loc[
                summary_df[
                    "task"
                ]
                ==
                "2s_to_2s",
                "reconstructed_windows",
            ].iloc[
                0
            ]
        )
        ==
        EXPECTED[
            "windows_2s"
        ]
        and
        int(
            summary_df.loc[
                summary_df[
                    "task"
                ]
                ==
                "2s_to_5s",
                "reconstructed_windows",
            ].iloc[
                0
            ]
        )
        ==
        EXPECTED[
            "windows_5s"
        ]
    )

    exact_track_sets = bool(
        set_compare_df[
            "exact_track_set_match"
        ].all()
    )

    # Source-rule lock.
    rule_lock_df = pd.DataFrame(
        [
            {
                "builder_path":
                    str(
                        BUILDER
                    ),

                "builder_sha256":
                    builder_hash,

                "fps":
                    FPS,

                "observation_seconds":
                    OBS_SEC,

                "observation_frames":
                    OBS_FRAMES,

                "max_small_gap_missing_frames":
                    MAX_SMALL_GAP,

                "segment_split_condition":
                    "missing > MAX_SMALL_GAP",

                "small_gap_interpolation":
                    "gaps with missing <= MAX_SMALL_GAP remain within the segment and are interpolated by builder",

                "window_stride_raw_frames":
                    WINDOW_STRIDE,

                "window_stride_seconds_at_30fps":
                    WINDOW_STRIDE
                    /
                    FPS,

                "future_2s_frames":
                    FUTURE_2,

                "future_5s_frames":
                    FUTURE_5,

                "required_dense_frames_2s":
                    OBS_FRAMES
                    +
                    FUTURE_2,

                "required_dense_frames_5s":
                    OBS_FRAMES
                    +
                    FUTURE_5,
            }
        ]
    )

    constants_path = (
        OUT
        /
        "R12E_BUILDER_CONSTANTS.csv"
    )

    key_lines_path = (
        OUT
        /
        "R12E_KEY_SOURCE_LINES.csv"
    )

    function_path = (
        OUT
        /
        "R12E_RULE_FUNCTION_INDEX.csv"
    )

    blocks_path = (
        OUT
        /
        "R12E_RULE_FUNCTION_SOURCE_BLOCKS.csv"
    )

    rule_lock_path = (
        OUT
        /
        "R12E_EXACT_SEGMENT_AND_WINDOW_RULE_LOCK.csv"
    )

    segment_path = (
        OUT
        /
        "R12E_RECONSTRUCTED_CONTIGUITY_SEGMENTS.csv"
    )

    recon2_path = (
        OUT
        /
        "R12E_RECONSTRUCTED_2S_SEGMENT_WINDOWS.csv"
    )

    recon5_path = (
        OUT
        /
        "R12E_RECONSTRUCTED_5S_SEGMENT_WINDOWS.csv"
    )

    summary_path = (
        OUT
        /
        "R12E_RECONSTRUCTION_VS_FROZEN_COUNTS.csv"
    )

    track_set_path = (
        OUT
        /
        "R12E_RECONSTRUCTED_VS_FROZEN_TRACK_SETS.csv"
    )

    constants_df.to_csv(
        constants_path,
        index=False
    )

    key_lines_df.to_csv(
        key_lines_path,
        index=False
    )

    function_df.to_csv(
        function_path,
        index=False
    )

    block_df.to_csv(
        blocks_path,
        index=False
    )

    rule_lock_df.to_csv(
        rule_lock_path,
        index=False
    )

    segments.to_csv(
        segment_path,
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

    set_compare_df.to_csv(
        track_set_path,
        index=False
    )

    if (
        exact_counts
        and
        exact_track_sets
    ):
        status = (
            "R12E_SEGMENT_GAP_WINDOW_RULE_EXACTLY_RECONSTRUCTED_45_37_31_89_63_1053_826_READY_FOR_R12F_FINAL_TRACK_ACCOUNTING_LOCK"
        )

    else:
        status = (
            "R12E_SOURCE_RULE_EXTRACTED_BUT_RECONSTRUCTION_NOT_EXACT_REVIEW_BEFORE_R12F"
        )

    manuscript = f"""R12 exact track-accounting and segmentation provenance

Source-locked construction rule
-------------------------------
The trajectory-window builder is:
{BUILDER}

SHA256:
{builder_hash}

The source defines:
- raw frame rate: {FPS} Hz
- observation history: {OBS_SEC} s = {OBS_FRAMES} frames
- 2 s future horizon: {FUTURE_2} frames
- 5 s future horizon: {FUTURE_5} frames
- maximum interpolated small gap: {MAX_SMALL_GAP} missing frames
- split condition: missing > MAX_SMALL_GAP
- sliding-window stride: {WINDOW_STRIDE} raw frames ({WINDOW_STRIDE/FPS:.3f} s)

For consecutive observed frames, missing = current_frame - previous_frame - 1. If missing exceeds {MAX_SMALL_GAP}, the original (sequence, object_id) trajectory is split into a new segment. Gaps of at most {MAX_SMALL_GAP} missing frames remain in the same segment and are interpolated by the builder. Forecast windows are then generated from the dense segment using {OBS_FRAMES} observation frames plus the horizon-specific future length, advancing by {WINDOW_STRIDE} frames.

Population and forecast accounting
----------------------------------
The authoritative TestReal source contains 45 original (sequence, object_id) tracks.

Reconstruction versus frozen metadata:
{summary_df.to_string(index=False)}

The 5 s represented-track set is a subset of the 2 s represented-track set. The term track_key in downstream forecasting metadata is a segment/tracklet key, not the original dataset-level track count.

Reviewer-safe wording
---------------------
"The BIRDSAI TestReal split contains 45 original tracks. Trajectories were split when more than {MAX_SMALL_GAP} consecutive frames were missing; smaller gaps were retained within a segment and interpolated. With a {OBS_SEC}-s observation history and a {WINDOW_STRIDE/FPS:.1f}-s sliding stride, 37 original tracks yielded 89 forecastable 2-s segments and 31 original tracks yielded 63 forecastable 5-s segments. These produced 1,053 and 826 evaluation windows, respectively."

Do not write:
"TestReal contains 89 tracks."
"""

    manuscript_path = (
        OUT
        /
        "R12E_MANUSCRIPT_READY_EXACT_SEGMENTATION_PROVENANCE.txt"
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8"
    )

    manifest = {
        "step":
            "R12E_EXACT_SEGMENT_RULE_EXTRACTION_AND_RECONSTRUCTION",

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

        "builder_path":
            str(
                BUILDER
            ),

        "builder_sha256":
            builder_hash,

        "rule": {
            "FPS":
                FPS,

            "OBS_SEC":
                OBS_SEC,

            "OBS_FRAMES":
                OBS_FRAMES,

            "MAX_SMALL_GAP":
                MAX_SMALL_GAP,

            "segment_split_condition":
                "missing > MAX_SMALL_GAP",

            "WINDOW_STRIDE":
                WINDOW_STRIDE,

            "FUTURE_2S_FRAMES":
                FUTURE_2,

            "FUTURE_5S_FRAMES":
                FUTURE_5,
        },

        "exact_count_reconstruction":
            exact_counts,

        "exact_represented_track_set_reconstruction":
            exact_track_sets,

        "outputs": {
            "builder_constants":
                str(
                    constants_path
                ),

            "key_source_lines":
                str(
                    key_lines_path
                ),

            "rule_function_index":
                str(
                    function_path
                ),

            "rule_function_source_blocks":
                str(
                    blocks_path
                ),

            "rule_lock":
                str(
                    rule_lock_path
                ),

            "reconstructed_segments":
                str(
                    segment_path
                ),

            "reconstructed_2s_windows":
                str(
                    recon2_path
                ),

            "reconstructed_5s_windows":
                str(
                    recon5_path
                ),

            "reconstruction_vs_frozen_counts":
                str(
                    summary_path
                ),

            "track_set_match":
                str(
                    track_set_path
                ),

            "manuscript_ready_provenance":
                str(
                    manuscript_path
                ),
        },
    }

    (
        OUT
        /
        "R12E_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "EXACT BUILDER RULE LOCK:"
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
        "\nRECONSTRUCTED VS FROZEN REPRESENTED-TRACK SETS:"
    )

    print(
        set_compare_df.to_string(
            index=False
        )
    )

    print(
        "\nKEY SOURCE LINES:"
    )

    print(
        key_lines_df.to_string(
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
        "\nR12-E COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
