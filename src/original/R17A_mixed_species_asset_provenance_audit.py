
from pathlib import Path
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
CAPSTONE = Path(r"E:\AIUB\Capstone Project")
JOURNAL = CAPSTONE / "Journal Based Results"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"
OUT = R17_ROOT / "R17A_mixed_species_asset_provenance_audit"
OUT.mkdir(parents=True, exist_ok=True)

R16B_MANIFEST = (
    REVISION
    / "R16_latency"
    / "R16B_operational_response_latency_context"
    / "R16B_MANIFEST.json"
)

KNOWN_MULTI_ANIMAL_CKPTS = [
    CAPSTONE
    / "capstone"
    / "wildlife_project"
    / "wildlife_training"
    / "four_animals_v5_final_50"
    / "weights"
    / "best.pt",

    CAPSTONE
    / "capstone"
    / "wildlife_project"
    / "wildlife_training"
    / "four_animals_v4_improved"
    / "weights"
    / "best.pt",

    CAPSTONE
    / "capstone"
    / "wildlife_project"
    / "wildlife_training"
    / "four_animals_v3"
    / "weights"
    / "best.pt",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
TEXT_EXTS = {".yaml", ".yml", ".json", ".csv", ".txt"}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "site-packages",
    "node_modules",
    ".venv",
    "venv",
    "paper_strengthening_revision",
}

SPECIES_HINTS = [
    "elephant",
    "deer",
    "red deer",
    "cervus",
    "human",
    "person",
    "cow",
    "cattle",
    "buffalo",
    "boar",
    "pig",
    "horse",
    "dog",
    "goat",
    "sheep",
    "bambi",
    "wildlife",
]

THERMAL_HINTS = [
    "thermal",
    "infrared",
    "ir",
    "night",
    "nightvision",
    "night_vision",
]

RGB_HINTS = [
    "rgb",
    "daylight",
    "visible",
    "color",
]


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


def skipped(path):
    parts = {x.lower() for x in Path(path).parts}
    return any(x.lower() in parts for x in SKIP_DIRS)


def iter_files(root):
    if not root.exists():
        return

    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)

        dirs[:] = [
            d
            for d in dirs
            if not skipped(current / d)
        ]

        if skipped(current):
            continue

        for name in files:
            yield current / name


def verify_r16_closed():
    require(R16B_MANIFEST)

    m = json.loads(
        R16B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R16_OPERATIONAL_LATENCY_CONTEXT_FULLY_LOCKED_COMPUTE_MEASURED_NONCOMPUTE_UNMEASURED_NO_2S_5S_RESPONSE_SUFFICIENCY_CLAIM_READY_FOR_R17_MIXED_SPECIES_ROBUSTNESS"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"R16-B not in required locked state: {m.get('status')}"
        )

    return m


def inspect_yolo_checkpoint(path):
    result = {
        "checkpoint": str(path),
        "exists": path.exists(),
        "sha256": "",
        "model_names": "",
        "num_classes": None,
        "contains_elephant_class": False,
        "contains_non_elephant_classes": False,
        "load_error": "",
    }

    if not path.exists():
        return result

    result["sha256"] = sha256_file(path)

    try:
        from ultralytics import YOLO

        model = YOLO(str(path))

        names = getattr(model, "names", None)

        if isinstance(names, dict):
            ordered = [
                str(names[k])
                for k in sorted(names)
            ]
        elif isinstance(names, (list, tuple)):
            ordered = [str(x) for x in names]
        else:
            ordered = []

        result["model_names"] = "|".join(ordered)
        result["num_classes"] = len(ordered)

        low = [x.lower() for x in ordered]

        result["contains_elephant_class"] = any(
            "elephant" in x
            for x in low
        )

        result["contains_non_elephant_classes"] = any(
            "elephant" not in x
            for x in low
        )

    except Exception as exc:
        result["load_error"] = repr(exc)

    return result


def checkpoint_audit():
    rows = []
    seen = set()

    for p in KNOWN_MULTI_ANIMAL_CKPTS:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(inspect_yolo_checkpoint(p))

    training_root = (
        CAPSTONE
        / "capstone"
        / "wildlife_project"
        / "wildlife_training"
    )

    if training_root.exists():
        for p in iter_files(training_root):
            if p.name.lower() != "best.pt":
                continue

            key = str(p).lower()

            if key in seen:
                continue

            seen.add(key)
            rows.append(inspect_yolo_checkpoint(p))

    return pd.DataFrame(rows)


def read_small_text(path, max_bytes=4_000_000):
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def config_audit():
    rows = []

    roots = [
        CAPSTONE / "capstone" / "wildlife_project",
        CAPSTONE / "Final 1.0",
        JOURNAL,
    ]

    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for p in iter_files(root):
            if p.suffix.lower() not in {".yaml", ".yml", ".json"}:
                continue

            key = str(p).lower()

            if key in seen:
                continue

            seen.add(key)

            txt = read_small_text(p)
            low = txt.lower()

            species_hits = sorted(
                {
                    h
                    for h in SPECIES_HINTS
                    if h in low
                }
            )

            path_low = key

            likely_detector_config = (
                "data.yaml" in p.name.lower()
                or "dataset" in p.name.lower()
                or "wildlife" in path_low
                or "four_animals" in path_low
            )

            if not species_hits and not likely_detector_config:
                continue

            rows.append(
                {
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "species_hits": "|".join(species_hits),
                    "likely_detector_config": likely_detector_config,
                }
            )

    return pd.DataFrame(rows)


def media_asset_audit():
    rows = []
    dir_counts = {}

    roots = [
        CAPSTONE,
        JOURNAL,
    ]

    for root in roots:
        if not root.exists():
            continue

        for current_root, dirs, files in os.walk(root):
            current = Path(current_root)

            dirs[:] = [
                d
                for d in dirs
                if not skipped(current / d)
            ]

            if skipped(current):
                continue

            low_path = str(current).lower()

            species_hits = sorted(
                {
                    h
                    for h in SPECIES_HINTS
                    if h in low_path
                }
            )

            thermal_hits = sorted(
                {
                    h
                    for h in THERMAL_HINTS
                    if h in low_path
                }
            )

            rgb_hits = sorted(
                {
                    h
                    for h in RGB_HINTS
                    if h in low_path
                }
            )

            image_count = 0
            video_count = 0
            text_count = 0

            for name in files:
                p = current / name
                ext = p.suffix.lower()

                if ext in IMAGE_EXTS:
                    image_count += 1

                elif ext in VIDEO_EXTS:
                    video_count += 1

                elif ext in TEXT_EXTS:
                    text_count += 1

            relevant = (
                bool(species_hits)
                or bool(thermal_hits)
                or "bambi" in low_path
                or "wildlife" in low_path
            )

            if not relevant:
                continue

            if (
                image_count == 0
                and video_count == 0
                and text_count == 0
            ):
                continue

            rows.append(
                {
                    "directory": str(current),
                    "image_count": image_count,
                    "video_count": video_count,
                    "text_or_label_count": text_count,
                    "species_hints": "|".join(species_hits),
                    "thermal_or_night_hints": "|".join(thermal_hits),
                    "rgb_hints": "|".join(rgb_hits),
                    "bambi_hint": "bambi" in low_path,
                    "wildlife_hint": "wildlife" in low_path,
                }
            )

    return pd.DataFrame(rows)


def bambi_specific_audit():
    rows = []

    if not CAPSTONE.exists():
        return pd.DataFrame()

    for p in iter_files(CAPSTONE):
        low = str(p).lower()

        if "bambi" not in low:
            continue

        if p.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS | TEXT_EXTS:
            continue

        rows.append(
            {
                "path": str(p),
                "suffix": p.suffix.lower(),
                "size_bytes": p.stat().st_size,
                "flight6_hint": (
                    "flight 6" in low
                    or "flight_6" in low
                    or "flight6" in low
                    or "flight-6" in low
                ),
                "red_deer_hint": (
                    "red deer" in low
                    or "red_deer" in low
                    or "cervus" in low
                    or "deer" in low
                ),
                "human_hint": (
                    "human" in low
                    or "person" in low
                ),
                "thermal_hint": any(
                    h in low
                    for h in THERMAL_HINTS
                ),
            }
        )

    return pd.DataFrame(rows)


def choose_multiclass_checkpoint(ckpt_df):
    if len(ckpt_df) == 0:
        return "", False, "No checkpoints found."

    valid = ckpt_df.loc[
        ckpt_df["exists"].astype(bool)
        & ckpt_df["contains_elephant_class"].astype(bool)
        & ckpt_df["contains_non_elephant_classes"].astype(bool)
    ].copy()

    if len(valid) == 0:
        return "", False, "No checkpoint exposes elephant + non-elephant classes."

    # Prefer known final v5, then improved v4, then any other valid multiclass checkpoint.
    priorities = []

    for idx, row in valid.iterrows():
        p = str(row["checkpoint"]).lower()

        if "four_animals_v5_final_50" in p:
            score = 300
        elif "four_animals_v4_improved" in p:
            score = 200
        elif "four_animals_v3" in p:
            score = 100
        else:
            score = 0

        priorities.append((score, idx))

    priorities.sort(
        reverse=True
    )

    best_idx = priorities[0][1]
    best = valid.loc[best_idx]

    return (
        str(best["checkpoint"]),
        True,
        (
            "Preferred by frozen provenance rule: v5_final_50 > v4_improved > v3 > other "
            "among checkpoints exposing elephant and at least one non-elephant class."
        ),
    )


def main():
    print(
        "\nR17-A — MIXED-SPECIES ASSET / PROVENANCE AUDIT"
    )

    print(
        "AUDIT ONLY. NO training and NO inference. "
        "This step inventories the mixed-species detector checkpoint(s), model class names, "
        "dataset/config provenance, wildlife/BAMBI assets, RGB/thermal/night hints, and negative-species material. "
        "It does not yet claim mixed-species robustness and does not modify original project files.\n"
    )

    verify_r16_closed()

    ckpt_df = checkpoint_audit()
    config_df = config_audit()
    media_df = media_asset_audit()
    bambi_df = bambi_specific_audit()

    selected_ckpt, ckpt_ok, ckpt_rule = choose_multiclass_checkpoint(
        ckpt_df
    )

    if ckpt_ok:
        selected_row = ckpt_df.loc[
            ckpt_df["checkpoint"] == selected_ckpt
        ].iloc[0]

        selected_classes = str(
            selected_row[
                "model_names"
            ]
        )

        selected_sha = str(
            selected_row[
                "sha256"
            ]
        )
    else:
        selected_classes = ""
        selected_sha = ""

    mixed_media_ok = bool(
        len(media_df) > 0
    )

    bambi_ok = bool(
        len(bambi_df) > 0
    )

    thermal_hint_ok = bool(
        len(media_df) > 0
        and media_df[
            "thermal_or_night_hints"
        ].astype(str).str.len().gt(0).any()
    ) or bool(
        len(bambi_df) > 0
        and bambi_df[
            "thermal_hint"
        ].astype(bool).any()
    )

    summary = pd.DataFrame(
        [
            {
                "component": "Mixed-species detector checkpoint with elephant + confuser classes",
                "found": ckpt_ok,
                "count": int(
                    ckpt_df[
                        "contains_elephant_class"
                    ].astype(bool).mul(
                        ckpt_df[
                            "contains_non_elephant_classes"
                        ].astype(bool)
                    ).sum()
                ) if len(ckpt_df) else 0,
                "detail": selected_ckpt,
            },
            {
                "component": "Dataset/config provenance candidates",
                "found": len(config_df) > 0,
                "count": len(config_df),
                "detail": "",
            },
            {
                "component": "Mixed-species / wildlife media directories",
                "found": mixed_media_ok,
                "count": len(media_df),
                "detail": "",
            },
            {
                "component": "BAMBI-specific assets",
                "found": bambi_ok,
                "count": len(bambi_df),
                "detail": "",
            },
            {
                "component": "Thermal/night asset hint",
                "found": thermal_hint_ok,
                "count": int(thermal_hint_ok),
                "detail": "",
            },
        ]
    )

    core_ready = bool(
        ckpt_ok
        and mixed_media_ok
    )

    if core_ready:
        status = (
            "R17A_MIXED_SPECIES_CORE_ASSETS_FOUND_CHECKPOINT_CLASSES_AND_MEDIA_AUDITED_READY_FOR_R17B_FROZEN_ROBUSTNESS_PROTOCOL_LOCK"
        )
    else:
        missing = []

        if not ckpt_ok:
            missing.append(
                "MULTICLASS_ELEPHANT_CHECKPOINT"
            )

        if not mixed_media_ok:
            missing.append(
                "MIXED_SPECIES_MEDIA"
            )

        status = (
            "R17A_MIXED_SPECIES_ASSETS_INCOMPLETE_MISSING_"
            +
            "_".join(
                missing
            )
        )

    ckpt_path = OUT / "R17A_MULTICLASS_CHECKPOINT_AUDIT.csv"
    config_path = OUT / "R17A_DATASET_CONFIG_PROVENANCE.csv"
    media_path = OUT / "R17A_MIXED_SPECIES_MEDIA_DIRECTORIES.csv"
    bambi_path = OUT / "R17A_BAMBI_ASSET_AUDIT.csv"
    summary_path = OUT / "R17A_SUMMARY.csv"

    ckpt_df.to_csv(
        ckpt_path,
        index=False,
    )

    config_df.to_csv(
        config_path,
        index=False,
    )

    media_df.to_csv(
        media_path,
        index=False,
    )

    bambi_df.to_csv(
        bambi_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "step":
            "R17A_MIXED_SPECIES_ASSET_PROVENANCE_AUDIT",

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

        "selected_multiclass_checkpoint":
            selected_ckpt,

        "selected_multiclass_checkpoint_sha256":
            selected_sha,

        "selected_model_classes":
            selected_classes,

        "checkpoint_selection_rule":
            ckpt_rule,

        "bambi_assets_found":
            bambi_ok,

        "thermal_or_night_asset_hint_found":
            thermal_hint_ok,

        "important_claim_limits": [
            "BAMBI deer/human is an elephant-negative cross-species/cross-domain stress test, not elephant-specific validation.",
            "RGB and thermal/night evidence must be reported separately.",
            "No mixed-species robustness claim is allowed until inference and class-filtering metrics are executed under a frozen protocol.",
            "The historical single-class detector cannot by itself validate mixed-species elephant filtering.",
        ],

        "outputs": {
            "checkpoint_audit":
                str(
                    ckpt_path
                ),

            "config_provenance":
                str(
                    config_path
                ),

            "media_directories":
                str(
                    media_path
                ),

            "bambi_asset_audit":
                str(
                    bambi_path
                ),

            "summary":
                str(
                    summary_path
                ),
        },
    }

    (
        OUT
        /
        "R17A_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "ASSET SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nMULTI-CLASS CHECKPOINT AUDIT:"
    )

    if len(
        ckpt_df
    ):
        print(
            ckpt_df[
                [
                    "checkpoint",
                    "num_classes",
                    "model_names",
                    "contains_elephant_class",
                    "contains_non_elephant_classes",
                    "load_error",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No checkpoint candidates."
        )

    print(
        "\nSELECTED MULTI-CLASS CHECKPOINT:"
    )

    print(
        selected_ckpt
        if selected_ckpt
        else
        "NONE"
    )

    print(
        "CLASSES:",
        selected_classes
        if selected_classes
        else
        "UNKNOWN",
    )

    print(
        "\nBAMBI ASSET AUDIT:"
    )

    if len(
        bambi_df
    ):
        print(
            bambi_df.head(
                80
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No BAMBI-named raw assets found in the scanned Capstone tree."
        )

    print(
        "\nMIXED-SPECIES MEDIA DIRECTORIES:"
    )

    if len(
        media_df
    ):
        print(
            media_df.head(
                80
            ).to_string(
                index=False
            )
        )
    else:
        print(
            "No mixed-species/wildlife media directories found."
        )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR17-A COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
