
from pathlib import Path
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R18_ROOT = REVISION / "R18_meter_space_wildbox"

R17E = (
    REVISION
    / "R17_mixed_species_robustness"
    / "R17E_claim_interpretation_lock"
    / "R17E_CLAIM_INTERPRETATION_LOCK.json"
)

OUT = R18_ROOT / "R18A2_wildbox_only_metric_scale_audit"
OUT.mkdir(parents=True, exist_ok=True)

CAPSTONE_ROOT = Path(r"E:\AIUB\Capstone Project")

KNOWN_WILDBOX_ZIPS = [
    "DJI_20240119152335_0005_V.zip",
    "DJI_20240626142859_0001_V.zip",
    "DJI_20260227084135_0002_V.zip",
]

SEARCH_ROOTS = [
    CAPSTONE_ROOT / "Journal Based Results" / "Main Result" / "wildbox_fresh_holdout_zips",
    DOWNLOADS,
]

TEXT_EXTS = {
    ".json", ".csv", ".txt", ".yaml", ".yml",
    ".xml", ".xmp", ".srt", ".kml", ".gpx",
}

# IMPORTANT:
# Generic "height" is intentionally NOT a metric-scale keyword because it often
# means image height or camera orthographic height. We require explicit physical
# scale wording.
METRIC_SCALE_KEY_PATTERNS = [
    r"(^|[_\-.])altitude($|[_\-.])",
    r"relative[_\-.]?altitude",
    r"absolute[_\-.]?altitude",
    r"flight[_\-.]?height",
    r"height[_\-.]?(m|meter|meters)$",
    r"\bagl\b",
    r"above[_\-.]?ground",
    r"ground[_\-.]?sample[_\-.]?distance",
    r"\bgsd\b",
    r"meters?[_\-.]?per[_\-.]?(pixel|px|unit)",
    r"(pixel|px|unit)[_\-.]?per[_\-.]?meters?",
    r"metric[_\-.]?scale",
    r"scale[_\-.]?(m|meter|meters)",
    r"known[_\-.]?distance",
    r"camera[_\-.]?height[_\-.]?(m|meter|meters)",
]

INTRINSIC_KEY_PATTERNS = [
    r"intrinsic",
    r"camera[_\-.]?matrix",
    r"(^|[_\-.])fx($|[_\-.])",
    r"(^|[_\-.])fy($|[_\-.])",
    r"(^|[_\-.])cx($|[_\-.])",
    r"(^|[_\-.])cy($|[_\-.])",
    r"focal[_\-.]?length",
]

EXTRINSIC_KEY_PATTERNS = [
    r"extrinsic",
    r"rotation",
    r"translation",
    r"camera[_\-.]?pose",
    r"world[_\-.]?to[_\-.]?camera",
    r"camera[_\-.]?to[_\-.]?world",
]

FRAME_GEOMETRY_KEY_PATTERNS = [
    r"(^|[_\-.])width($|[_\-.])",
    r"(^|[_\-.])height($|[_\-.])",
    r"resolution",
    r"image[_\-.]?size",
    r"frame[_\-.]?size",
]

GPS_KEY_PATTERNS = [
    r"latitude",
    r"longitude",
    r"(^|[_\-.])lat($|[_\-.])",
    r"(^|[_\-.])lon(g)?($|[_\-.])",
    r"\bgps\b",
]

FOV_KEY_PATTERNS = [
    r"(^|[_\-.])fov($|[_\-.])",
    r"fovx",
    r"fovy",
    r"field[_\-.]?of[_\-.]?view",
]

MAX_MEMBER_TEXT_BYTES = 20_000_000


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
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def verify_r17():
    require(R17E)

    m = read_json(R17E)

    expected = (
        "R17_MIXED_SPECIES_ROBUSTNESS_FULLY_ASSESSED_FALSE_ALARM_REDUCTION_SUPPORTED_"
        "BUT_MAJOR_SENSITIVITY_LOSS_LOCKED_CURRENT_4CLASS_NOT_SUPPORTED_AS_REPLACEMENT_"
        "READY_FOR_R18_METER_SPACE_WILDBOX"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R17-E status: {m.get('status')}"
        )

    return m


def regex_any(text, patterns):
    s = str(text).lower()

    return any(
        re.search(
            pat,
            s,
            flags=re.IGNORECASE,
        )
        is not None
        for pat in patterns
    )


def locate_zip(name):
    hits = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        try:
            for p in root.rglob(name):
                if p.is_file():
                    hits.append(p)
        except Exception:
            pass

    # De-duplicate exact path strings.
    unique = {}

    for p in hits:
        unique[str(p)] = p

    hits = list(
        unique.values()
    )

    # Prefer the authoritative Main Result fresh-holdout location.
    hits.sort(
        key=lambda p: (
            0
            if "wildbox_fresh_holdout_zips" in str(p).lower()
            else 1,
            len(str(p)),
            str(p),
        )
    )

    return (
        hits[0]
        if hits
        else None
    )


def flatten_json(obj, prefix=""):
    rows = []

    if isinstance(
        obj,
        dict,
    ):
        for key, value in obj.items():
            p = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten_json(
                    value,
                    p,
                )
            )

    elif isinstance(
        obj,
        list,
    ):
        # Keep the list as one value when it is scalar/numeric; recurse into
        # nested dict/list structures otherwise.
        if all(
            not isinstance(
                x,
                (dict, list),
            )
            for x in obj
        ):
            rows.append(
                (
                    prefix,
                    obj,
                )
            )
        else:
            for i, value in enumerate(
                obj
            ):
                rows.extend(
                    flatten_json(
                        value,
                        f"{prefix}[{i}]",
                    )
                )

    else:
        rows.append(
            (
                prefix,
                obj,
            )
        )

    return rows


def numeric_value_present(value):
    if isinstance(
        value,
        bool,
    ):
        return False

    if isinstance(
        value,
        (int, float),
    ):
        try:
            return math.isfinite(
                float(value)
            )
        except Exception:
            return False

    if isinstance(
        value,
        list,
    ):
        numeric = []

        for x in value:
            if isinstance(
                x,
                bool,
            ):
                continue

            if isinstance(
                x,
                (int, float),
            ):
                try:
                    if math.isfinite(
                        float(x)
                    ):
                        numeric.append(
                            float(x)
                        )
                except Exception:
                    pass

        return bool(
            numeric
        )

    s = str(
        value
    ).strip()

    # Explicit number somewhere in scalar text.
    return (
        re.search(
            r"[-+]?\d+(?:\.\d+)?",
            s,
        )
        is not None
    )


def value_preview(value, max_chars=300):
    try:
        s = json.dumps(
            value,
            ensure_ascii=False,
        )
    except Exception:
        s = str(
            value
        )

    return s[:max_chars]


def inspect_json_member(zip_path, member, raw_bytes):
    rows = []

    try:
        data = json.loads(
            raw_bytes.decode(
                "utf-8",
                errors="strict",
            )
        )
    except Exception:
        return rows

    for key_path, value in flatten_json(
        data
    ):
        is_metric = regex_any(
            key_path,
            METRIC_SCALE_KEY_PATTERNS,
        )

        is_intrinsic = regex_any(
            key_path,
            INTRINSIC_KEY_PATTERNS,
        )

        is_extrinsic = regex_any(
            key_path,
            EXTRINSIC_KEY_PATTERNS,
        )

        is_frame = regex_any(
            key_path,
            FRAME_GEOMETRY_KEY_PATTERNS,
        )

        is_gps = regex_any(
            key_path,
            GPS_KEY_PATTERNS,
        )

        is_fov = regex_any(
            key_path,
            FOV_KEY_PATTERNS,
        )

        if not any(
            [
                is_metric,
                is_intrinsic,
                is_extrinsic,
                is_frame,
                is_gps,
                is_fov,
            ]
        ):
            continue

        rows.append(
            {
                "zip_path":
                    str(
                        zip_path
                    ),

                "member":
                    member,

                "key_path":
                    key_path,

                "value_preview":
                    value_preview(
                        value
                    ),

                "numeric_value_present":
                    numeric_value_present(
                        value
                    ),

                "metric_scale_key":
                    is_metric,

                "intrinsic_key":
                    is_intrinsic,

                "extrinsic_key":
                    is_extrinsic,

                "frame_geometry_key":
                    is_frame,

                "gps_key":
                    is_gps,

                "fov_key":
                    is_fov,
            }
        )

    return rows


def inspect_text_member(zip_path, member, text):
    rows = []

    # Only source-specific metadata lines count. Library/source-code files are
    # not used as physical evidence.
    for line_no, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        s = line.strip()

        if not s:
            continue

        low_member = member.lower()

        # Exclude obvious source code / package docs from evidence.
        if any(
            token in low_member
            for token in [
                ".py",
                "site-packages",
                "source",
                "readme",
                "license",
            ]
        ):
            continue

        categories = {
            "metric_scale_key":
                regex_any(
                    s,
                    METRIC_SCALE_KEY_PATTERNS,
                ),

            "intrinsic_key":
                regex_any(
                    s,
                    INTRINSIC_KEY_PATTERNS,
                ),

            "extrinsic_key":
                regex_any(
                    s,
                    EXTRINSIC_KEY_PATTERNS,
                ),

            "frame_geometry_key":
                regex_any(
                    s,
                    FRAME_GEOMETRY_KEY_PATTERNS,
                ),

            "gps_key":
                regex_any(
                    s,
                    GPS_KEY_PATTERNS,
                ),

            "fov_key":
                regex_any(
                    s,
                    FOV_KEY_PATTERNS,
                ),
        }

        if not any(
            categories.values()
        ):
            continue

        rows.append(
            {
                "zip_path":
                    str(
                        zip_path
                    ),

                "member":
                    member,

                "line_no":
                    line_no,

                "key_path":
                    "TEXT_LINE",

                "value_preview":
                    s[:300],

                "numeric_value_present":
                    numeric_value_present(
                        s
                    ),

                **categories,
            }
        )

    return rows


def inspect_zip(zip_path):
    member_rows = []
    evidence_rows = []

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            suffix = Path(
                info.filename
            ).suffix.lower()

            member_rows.append(
                {
                    "zip_path":
                        str(
                            zip_path
                        ),

                    "member":
                        info.filename,

                    "suffix":
                        suffix,

                    "size_bytes":
                        int(
                            info.file_size
                        ),
                }
            )

            if (
                suffix
                not in TEXT_EXTS
                or
                info.file_size
                >
                MAX_MEMBER_TEXT_BYTES
            ):
                continue

            try:
                raw = zf.read(
                    info
                )
            except Exception:
                continue

            if suffix == ".json":
                evidence_rows.extend(
                    inspect_json_member(
                        zip_path,
                        info.filename,
                        raw,
                    )
                )

            else:
                try:
                    text = raw.decode(
                        "utf-8",
                        errors="ignore",
                    )
                except Exception:
                    text = ""

                evidence_rows.extend(
                    inspect_text_member(
                        zip_path,
                        info.filename,
                        text,
                    )
                )

    return (
        pd.DataFrame(
            member_rows
        ),
        pd.DataFrame(
            evidence_rows
        ),
    )


def source_summary(zip_name, zip_path, evidence):
    if len(
        evidence
    ) == 0:
        e = pd.DataFrame()
    else:
        e = evidence.loc[
            evidence[
                "zip_path"
            ]
            ==
            str(
                zip_path
            )
        ].copy()

    def present(
        col,
        numeric_required=False,
    ):
        if len(
            e
        ) == 0 or col not in e.columns:
            return False

        mask = e[
            col
        ].fillna(
            False
        ).astype(
            bool
        )

        if numeric_required:
            mask = (
                mask
                &
                e[
                    "numeric_value_present"
                ].fillna(
                    False
                ).astype(
                    bool
                )
            )

        return bool(
            mask.any()
        )

    metric_scale_anchor = present(
        "metric_scale_key",
        numeric_required=True,
    )

    intrinsics = present(
        "intrinsic_key",
        numeric_required=True,
    )

    extrinsics = present(
        "extrinsic_key",
        numeric_required=True,
    )

    frame_geometry = present(
        "frame_geometry_key",
        numeric_required=True,
    )

    gps = present(
        "gps_key",
        numeric_required=True,
    )

    fov = present(
        "fov_key",
        numeric_required=True,
    )

    # A metric conversion needs an explicit metric anchor. Intrinsics/extrinsics
    # alone do not establish meters because monocular/SfM-style reconstruction
    # may be only up-to-scale.
    source_ready = bool(
        metric_scale_anchor
        and
        (
            intrinsics
            or
            fov
        )
        and
        frame_geometry
    )

    return {
        "zip_name":
            zip_name,

        "zip_path":
            str(
                zip_path
            ),

        "metric_scale_anchor_numeric":
            metric_scale_anchor,

        "intrinsics_numeric":
            intrinsics,

        "extrinsics_numeric":
            extrinsics,

        "fov_numeric":
            fov,

        "gps_numeric":
            gps,

        "frame_geometry_numeric":
            frame_geometry,

        "direct_meter_conversion_ready":
            source_ready,

        "scientific_note":
            (
                "Explicit metric scale + camera geometry + frame geometry found."
                if source_ready
                else
                (
                    "Camera geometry may exist, but no explicit metric-scale anchor was verified; "
                    "intrinsics/extrinsics alone are insufficient to claim meters."
                )
            ),
    }


def main():
    print(
        "\nR18-A2 — STRICT WILDBOX-ONLY METRIC-SCALE AUDIT"
    )

    print(
        "CORRECTIVE AUDIT. NO coordinate conversion and NO forecast rerun. "
        "This version fixes two issues in R18-A: (1) non-WildBox/BAMBI metadata is excluded, "
        "and (2) generic 'height' no longer counts as altitude. A direct meter conversion is "
        "allowed only when an explicit source-specific metric-scale anchor is verified.\n"
    )

    verify_r17()

    provenance_rows = []
    all_members = []
    all_evidence = []

    located = {}

    for name in KNOWN_WILDBOX_ZIPS:
        p = locate_zip(
            name
        )

        if p is None:
            provenance_rows.append(
                {
                    "zip_name":
                        name,

                    "found":
                        False,

                    "path":
                        "",

                    "size_bytes":
                        0,

                    "sha256":
                        "",
                }
            )

            continue

        located[
            name
        ] = p

        provenance_rows.append(
            {
                "zip_name":
                    name,

                "found":
                    True,

                "path":
                    str(
                        p
                    ),

                "size_bytes":
                    p.stat().st_size,

                "sha256":
                    sha256_file(
                        p
                    ),
            }
        )

        print(
            f"INSPECTING {name}"
        )

        members, evidence = inspect_zip(
            p
        )

        if len(
            members
        ):
            all_members.append(
                members
            )

        if len(
            evidence
        ):
            all_evidence.append(
                evidence
            )

    provenance = pd.DataFrame(
        provenance_rows
    )

    members_df = (
        pd.concat(
            all_members,
            ignore_index=True,
            sort=False,
        )
        if all_members
        else
        pd.DataFrame()
    )

    evidence_df = (
        pd.concat(
            all_evidence,
            ignore_index=True,
            sort=False,
        )
        if all_evidence
        else
        pd.DataFrame()
    )

    source_rows = []

    for name in KNOWN_WILDBOX_ZIPS:
        if name not in located:
            source_rows.append(
                {
                    "zip_name":
                        name,

                    "zip_path":
                        "",

                    "metric_scale_anchor_numeric":
                        False,

                    "intrinsics_numeric":
                        False,

                    "extrinsics_numeric":
                        False,

                    "fov_numeric":
                        False,

                    "gps_numeric":
                        False,

                    "frame_geometry_numeric":
                        False,

                    "direct_meter_conversion_ready":
                        False,

                    "scientific_note":
                        "Source ZIP not found.",
                }
            )

            continue

        source_rows.append(
            source_summary(
                name,
                located[
                    name
                ],
                evidence_df,
            )
        )

    source_summary_df = pd.DataFrame(
        source_rows
    )

    all_sources_found = bool(
        len(
            located
        )
        ==
        len(
            KNOWN_WILDBOX_ZIPS
        )
    )

    all_sources_meter_ready = bool(
        all_sources_found
        and
        source_summary_df[
            "direct_meter_conversion_ready"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .all()
    )

    any_source_meter_ready = bool(
        source_summary_df[
            "direct_meter_conversion_ready"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .any()
    )

    geometry_without_scale = bool(
        (
            source_summary_df[
                "intrinsics_numeric"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
            |
            source_summary_df[
                "extrinsics_numeric"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )
        .any()
        and
        not
        source_summary_df[
            "metric_scale_anchor_numeric"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .any()
    )

    if all_sources_meter_ready:
        status = (
            "R18A2_WILDBOX_ALL_SOURCES_HAVE_VERIFIED_METRIC_SCALE_AND_CAMERA_GEOMETRY_"
            "READY_FOR_R18B_FROZEN_METER_SPACE_PROTOCOL_LOCK"
        )

        next_step = (
            "R18B_FROZEN_METER_SPACE_PROTOCOL_LOCK"
        )

    elif any_source_meter_ready:
        status = (
            "R18A2_ONLY_PARTIAL_WILDBOX_SOURCES_HAVE_VERIFIED_METRIC_SCALE_"
            "NO_ALL_SOURCE_METER_CLAIM_READY_FOR_SOURCE_RESTRICTED_REVIEW"
        )

        next_step = (
            "R18B_SOURCE_RESTRICTED_METRIC_SCALE_REVIEW"
        )

    else:
        status = (
            "R18A2_WILDBOX_CAMERA_GEOMETRY_PRESENT_BUT_NO_VERIFIED_METRIC_SCALE_ANCHOR_"
            "NO_DEFENSIBLE_METER_CONVERSION_READY_FOR_R18B_LIMITATION_LOCK"
        )

        next_step = (
            "R18B_METER_SPACE_LIMITATION_LOCK"
        )

    summary = pd.DataFrame(
        [
            {
                "sources_expected":
                    len(
                        KNOWN_WILDBOX_ZIPS
                    ),

                "sources_found":
                    len(
                        located
                    ),

                "sources_with_metric_scale_anchor":
                    int(
                        source_summary_df[
                            "metric_scale_anchor_numeric"
                        ]
                        .fillna(
                            False
                        )
                        .astype(
                            bool
                        )
                        .sum()
                    ),

                "sources_with_intrinsics":
                    int(
                        source_summary_df[
                            "intrinsics_numeric"
                        ]
                        .fillna(
                            False
                        )
                        .astype(
                            bool
                        )
                        .sum()
                    ),

                "sources_with_extrinsics":
                    int(
                        source_summary_df[
                            "extrinsics_numeric"
                        ]
                        .fillna(
                            False
                        )
                        .astype(
                            bool
                        )
                        .sum()
                    ),

                "sources_direct_meter_ready":
                    int(
                        source_summary_df[
                            "direct_meter_conversion_ready"
                        ]
                        .fillna(
                            False
                        )
                        .astype(
                            bool
                        )
                        .sum()
                    ),

                "geometry_without_verified_metric_scale":
                    geometry_without_scale,

                "all_sources_meter_ready":
                    all_sources_meter_ready,

                "next_step":
                    next_step,

                "status":
                    status,
            }
        ]
    )

    provenance_path = (
        OUT
        / "R18A2_WILDBOX_ZIP_PROVENANCE.csv"
    )

    member_path = (
        OUT
        / "R18A2_WILDBOX_MEMBER_INVENTORY.csv"
    )

    evidence_path = (
        OUT
        / "R18A2_WILDBOX_ONLY_PHYSICAL_EVIDENCE.csv"
    )

    source_summary_path = (
        OUT
        / "R18A2_PER_SOURCE_SCALE_READINESS.csv"
    )

    summary_path = (
        OUT
        / "R18A2_SCALE_READINESS_SUMMARY.csv"
    )

    provenance.to_csv(
        provenance_path,
        index=False,
    )

    members_df.to_csv(
        member_path,
        index=False,
    )

    evidence_df.to_csv(
        evidence_path,
        index=False,
    )

    source_summary_df.to_csv(
        source_summary_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "step":
            "R18A2_STRICT_WILDBOX_ONLY_METRIC_SCALE_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "coordinate_conversion_performed":
            False,

        "forecast_rerun":
            False,

        "corrective_reason": [
            "R18-A aggregated non-WildBox/BAMBI metadata into physical-scale evidence.",
            "R18-A generic 'height' keyword could confuse image/frame height with physical altitude.",
            "R18-A2 restricts evidence to the exact three WildBox ZIPs and requires an explicit numeric metric-scale anchor.",
        ],

        "scientific_rule":
            (
                "Intrinsics, extrinsics, or monocular/SfM camera poses do not by themselves establish metric scale. "
                "A meter-space conversion requires an explicit source-specific metric anchor such as altitude in meters, "
                "known distance, GSD/meters-per-pixel, or independently calibrated metric scale."
            ),

        "sources_expected":
            KNOWN_WILDBOX_ZIPS,

        "sources_found":
            int(
                len(
                    located
                )
            ),

        "sources_direct_meter_ready":
            int(
                summary.iloc[
                    0
                ][
                    "sources_direct_meter_ready"
                ]
            ),

        "outputs": {
            "provenance":
                str(
                    provenance_path
                ),

            "member_inventory":
                str(
                    member_path
                ),

            "physical_evidence":
                str(
                    evidence_path
                ),

            "per_source_readiness":
                str(
                    source_summary_path
                ),

            "summary":
                str(
                    summary_path
                ),
        },
    }

    manifest_path = (
        OUT
        / "R18A2_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nPER-SOURCE SCALE READINESS:"
    )

    print(
        source_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSTRICT WILDBOX-ONLY SCALE SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    if len(
        evidence_df
    ):
        print(
            "\nRELEVANT WILDBOX EVIDENCE ROWS:"
        )

        cols = [
            "zip_path",
            "member",
            "key_path",
            "value_preview",
            "numeric_value_present",
            "metric_scale_key",
            "intrinsic_key",
            "extrinsic_key",
            "frame_geometry_key",
            "gps_key",
            "fov_key",
        ]

        cols = [
            c
            for c in cols
            if c in evidence_df.columns
        ]

        relevant = evidence_df.loc[
            evidence_df[
                [
                    c
                    for c in [
                        "metric_scale_key",
                        "intrinsic_key",
                        "extrinsic_key",
                        "frame_geometry_key",
                        "gps_key",
                        "fov_key",
                    ]
                    if c in evidence_df.columns
                ]
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
            .any(
                axis=1
            )
        ]

        print(
            relevant[
                cols
            ]
            .head(
                120
            )
            .to_string(
                index=False
            )
        )

        if len(
            relevant
        ) > 120:
            print(
                f"... {len(relevant) - 120} additional evidence rows saved to CSV."
            )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR18-A2 COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
