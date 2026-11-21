from pathlib import Path
import ast
import hashlib
import json
import re
import pandas as pd

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

CANDIDATES = [
    ROOT / "build_final_neural_dataset.py",
    ROOT / "build_unified_10hz_dataset.py",
    ROOT / "step18a_lock_geofence_protocol.py",
    ROOT / "step18b_evaluate_locked_geofence.py",
    ROOT / "audit_multigeofence.py",
    ROOT / "audit_virtual_geofences.py",
]

STEP18_LOCK = (
    ROOT
    / "step18_geofence"
    / "STEP18_PROTOCOL_LOCKED.json"
)

BAMBI_META_DIR = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k2b3_final_canonical"
)

OUT = (
    Path(r"C:\Users\HP\Downloads")
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
    / "k3_operational_warning"
    / "coordinate_compatibility_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

KEYWORDS = [
    "normalize",
    "normalized",
    "image",
    "image_width",
    "image_height",
    "width",
    "height",
    "coordinate",
    "x_norm",
    "y_norm",
    "bbox",
    "center",
    "corrected",
    "trajectory",
    "origin",
    "scale",
    "canonical",
    "pixel",
    "frame_width",
    "frame_height",
    "divide",
]


def require(path):
    if not path.exists():
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


def safe_read(path):
    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        return None


def function_rows(path, text):
    rows = []

    try:
        tree = ast.parse(text)
    except Exception:
        return rows

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        ):
            continue

        src = ast.get_source_segment(
            text,
            node
        ) or ""

        low = src.lower()

        hits = [
            k
            for k in KEYWORDS
            if k.lower() in low
        ]

        if not hits:
            continue

        score = len(
            set(
                hits
            )
        )

        if (
            "normalize"
            in low
            or
            "normalized"
            in low
        ):
            score += 5

        if (
            "image_width"
            in low
            or
            "frame_width"
            in low
        ):
            score += 4

        if (
            "image_height"
            in low
            or
            "frame_height"
            in low
        ):
            score += 4

        if "canonical" in low:
            score += 2

        rows.append({
            "file":
                str(
                    path
                ),

            "function":
                node.name,

            "line_start":
                int(
                    node.lineno
                ),

            "line_end":
                int(
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno
                    )
                ),

            "score":
                int(
                    score
                ),

            "hits":
                ";".join(
                    sorted(
                        set(
                            hits
                        )
                    )
                ),

            "source":
                src,
        })

    return rows


def lexical_contexts(path, text):
    rows = []
    lines = text.splitlines()

    for i, line in enumerate(
        lines,
        start=1
    ):
        low = line.lower()

        hits = [
            k
            for k in KEYWORDS
            if k.lower() in low
        ]

        if not hits:
            continue

        # Keep contexts likely to encode actual coordinate transforms.
        nearby = "\n".join(
            lines[
                max(
                    0,
                    i - 5
                ):
                min(
                    len(
                        lines
                    ),
                    i + 4
                )
            ]
        ).lower()

        strong = any(
            token in nearby
            for token in [
                " / ",
                "/ width",
                "/ height",
                "image_width",
                "image_height",
                "frame_width",
                "frame_height",
                "normalize",
                "normalized",
                "corrected_x",
                "corrected_y",
                "x_norm",
                "y_norm",
            ]
        )

        if not strong:
            continue

        lo = max(
            1,
            i - 4
        )

        hi = min(
            len(
                lines
            ),
            i + 4
        )

        context = "\n".join(
            f"{j:05d}: {lines[j-1]}"
            for j in range(
                lo,
                hi + 1
            )
        )

        rows.append({
            "file":
                str(
                    path
                ),

            "line":
                int(
                    i
                ),

            "hits":
                ";".join(
                    sorted(
                        set(
                            hits
                        )
                    )
                ),

            "text":
                line.strip(),

            "context":
                context,
        })

    return rows


def find_coordinate_csvs():
    roots = [
        ROOT,
    ]

    rows = []

    terms = [
        "corrected",
        "trajectory",
        "canonical",
        "neural",
        "geofence",
    ]

    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob(
            "*.csv"
        ):
            low = str(
                p
            ).lower()

            if not any(
                t in low
                for t in terms
            ):
                continue

            try:
                df = pd.read_csv(
                    p,
                    nrows=5
                )
            except Exception:
                continue

            coord_cols = [
                c
                for c in df.columns
                if any(
                    t in c.lower()
                    for t in [
                        "x",
                        "y",
                        "width",
                        "height",
                        "coord",
                        "correct",
                        "norm",
                    ]
                )
            ]

            if not coord_cols:
                continue

            rows.append({
                "file":
                    str(
                        p
                    ),

                "columns":
                    ";".join(
                        df.columns.astype(
                            str
                        ).tolist()
                    ),

                "coordinate_like_columns":
                    ";".join(
                        coord_cols
                    ),
            })

    return pd.DataFrame(
        rows
    )


def bambi_metadata_inventory():
    rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        path = (
            BAMBI_META_DIR
            / f"{task}_bambi_flight6_canonical_metadata_LOCKED.csv"
        )

        require(
            path
        )

        df = pd.read_csv(
            path
        )

        rows.append({
            "task":
                task,

            "rows":
                int(
                    len(
                        df
                    )
                ),

            "columns":
                ";".join(
                    df.columns.astype(
                        str
                    ).tolist()
                ),

            "origin_x_min":
                float(
                    pd.to_numeric(
                        df[
                            "origin_x"
                        ],
                        errors="coerce"
                    ).min()
                )
                if
                "origin_x"
                in
                df.columns
                else None,

            "origin_x_max":
                float(
                    pd.to_numeric(
                        df[
                            "origin_x"
                        ],
                        errors="coerce"
                    ).max()
                )
                if
                "origin_x"
                in
                df.columns
                else None,

            "origin_y_min":
                float(
                    pd.to_numeric(
                        df[
                            "origin_y"
                        ],
                        errors="coerce"
                    ).min()
                )
                if
                "origin_y"
                in
                df.columns
                else None,

            "origin_y_max":
                float(
                    pd.to_numeric(
                        df[
                            "origin_y"
                        ],
                        errors="coerce"
                    ).max()
                )
                if
                "origin_y"
                in
                df.columns
                else None,
        })

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nSTEP 6D-K3C4 — STEP18/BAMBI COORDINATE-SYSTEM COMPATIBILITY AUDIT"
    )

    print(
        "Read-only audit. NO BAMBI geofence reconstruction or alarms are generated.\n"
    )

    require(
        STEP18_LOCK
    )

    lock = json.loads(
        STEP18_LOCK.read_text(
            encoding="utf-8"
        )
    )

    fn_rows = []
    lex_rows = []

    for path in CANDIDATES:
        if not path.exists():
            continue

        text = safe_read(
            path
        )

        if text is None:
            continue

        fn_rows.extend(
            function_rows(
                path,
                text
            )
        )

        lex_rows.extend(
            lexical_contexts(
                path,
                text
            )
        )

    functions = pd.DataFrame(
        fn_rows
    )

    lexical = pd.DataFrame(
        lex_rows
    )

    if len(
        functions
    ):
        functions = functions.sort_values(
            [
                "score",
                "file",
                "line_start",
            ],
            ascending=[
                False,
                True,
                True,
            ]
        )

    functions.to_csv(
        OUT
        / "TABLE_step6dk3c4_ranked_coordinate_functions.csv",
        index=False
    )

    lexical.to_csv(
        OUT
        / "TABLE_step6dk3c4_coordinate_transform_contexts.csv",
        index=False
    )

    csv_inventory = find_coordinate_csvs()

    csv_inventory.to_csv(
        OUT
        / "TABLE_step6dk3c4_coordinate_csv_inventory.csv",
        index=False
    )

    bambi = bambi_metadata_inventory()

    bambi.to_csv(
        OUT
        / "TABLE_step6dk3c4_bambi_coordinate_inventory.csv",
        index=False
    )

    with open(
        OUT
        / "STEP6D_K3C4_STEP18_LOCK_COPY.json",
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
        / "STEP6D_K3C4_SOURCE_BUNDLE.txt",
        "w",
        encoding="utf-8"
    ) as f:
        for _, row in functions.head(
            50
        ).iterrows():
            f.write(
                "=" * 100
                +
                "\n"
            )

            f.write(
                f"FILE: {row['file']}\n"
                f"FUNCTION: {row['function']}\n"
                f"LINES: {row['line_start']}-{row['line_end']}\n"
                f"SCORE: {row['score']}\n"
                f"HITS: {row['hits']}\n"
            )

            f.write(
                "=" * 100
                +
                "\n"
            )

            f.write(
                row[
                    "source"
                ]
            )

            f.write(
                "\n\n"
            )

    print(
        "STEP18 COORDINATE SYSTEM:"
    )

    print(
        lock.get(
            "coordinate_system"
        )
    )

    print(
        "\nSTEP18 BOUNDARIES:"
    )

    print(
        lock.get(
            "boundary_thresholds"
        )
    )

    print(
        "\nTOP COORDINATE/NORMALIZATION FUNCTIONS:"
    )

    if len(
        functions
    ):
        print(
            functions.head(
                30
            )[
                [
                    "file",
                    "function",
                    "line_start",
                    "line_end",
                    "score",
                    "hits",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nCOORDINATE TRANSFORM CONTEXTS:"
    )

    if len(
        lexical
    ):
        print(
            lexical.head(
                80
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "NONE"
        )

    print(
        "\nBAMBI COORDINATE INVENTORY:"
    )

    print(
        bambi.to_string(
            index=False
        )
    )

    # Conservative automatic decision:
    coord_text = str(
        lock.get(
            "coordinate_system",
            ""
        )
    ).lower()

    step18_is_normalized_image = (
        "normalized"
        in
        coord_text
        and
        "image"
        in
        coord_text
    )

    bambi_is_metric_world = bool(
        (
            bambi[
                "origin_x_min"
            ]
            >
            1000
        ).all()
        and
        (
            bambi[
                "origin_y_min"
            ]
            >
            1000
        ).all()
    )

    exact_mapping_evidence = False

    if len(
        lexical
    ):
        joined = " ".join(
            lexical[
                "context"
            ].astype(
                str
            ).tolist()
        ).lower()

        exact_mapping_evidence = any(
            token in joined
            for token in [
                "/ image_width",
                "/ image_height",
                "/ frame_width",
                "/ frame_height",
                "x_norm",
                "y_norm",
            ]
        )

    if (
        step18_is_normalized_image
        and
        bambi_is_metric_world
        and
        not exact_mapping_evidence
    ):
        status = (
            "DIRECT_STEP18_TO_BAMBI_INCOMPATIBLE_"
            "EXACT_NORMALIZED_IMAGE_MAPPING_NOT_YET_RECOVERED"
        )

    elif (
        step18_is_normalized_image
        and
        bambi_is_metric_world
        and
        exact_mapping_evidence
    ):
        status = (
            "NORMALIZED_IMAGE_MAPPING_CANDIDATE_FOUND_"
            "PENDING_BAMBI_REPRODUCIBILITY_CHECK"
        )

    else:
        status = (
            "COORDINATE_COMPATIBILITY_REQUIRES_MANUAL_SOURCE_REVIEW"
        )

    summary = {
        "step":
            "STEP_6D_K3C4_STEP18_BAMBI_COORDINATE_SYSTEM_COMPATIBILITY_AUDIT",

        "step18_coordinate_system":
            lock.get(
                "coordinate_system"
            ),

        "step18_boundaries":
            lock.get(
                "boundary_thresholds"
            ),

        "step18_is_normalized_image_plane":
            bool(
                step18_is_normalized_image
            ),

        "bambi_origins_are_metric_world_like":
            bool(
                bambi_is_metric_world
            ),

        "exact_mapping_evidence_found":
            bool(
                exact_mapping_evidence
            ),

        "status":
            status,

        "rule":
            (
                "Do not apply Step18 thresholds 0.25/0.50/0.75 directly "
                "to BAMBI UTM/world-metre coordinates."
            ),
    }

    (
        OUT
        / "STEP6D_K3C4_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
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
        "\nSTEP 6D-K3C4 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
