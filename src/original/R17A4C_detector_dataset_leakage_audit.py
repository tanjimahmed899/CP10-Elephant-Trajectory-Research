
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
from PIL import Image, ImageOps


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

R17A4B_DIR = R17_ROOT / "R17A4B_frozen_pixel_download"
R17A4B_MANIFEST = R17A4B_DIR / "R17A4B_MANIFEST.json"
R17A4B_AUDIT = R17A4B_DIR / "R17A4B_PIXEL_DOWNLOAD_AUDIT.csv"

V5_ROOT = Path(
    r"E:\AIUB\Capstone Project\capstone\wildlife_project\V5"
)

EXPORTED_ROOT = Path(
    r"E:\AIUB\Capstone Project\capstone\wildlife_project\exported_dataset\project-2-dataset-instance1786346684.783013-ultralytics-_yolo-_detection-10"
)

OUT = R17_ROOT / "R17A4C_detector_dataset_leakage_audit"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# Conservative perceptual candidate threshold.
# IMPORTANT: a perceptual candidate is NOT automatically called leakage.
DHASH_MAX = 4
AHASH_MAX = 4


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


def popcount64(x):
    return int(x).bit_count()


def dhash64(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        im = im.resize((9, 8), Image.Resampling.LANCZOS)
        px = list(im.getdata())

    value = 0
    bit = 0

    for y in range(8):
        row = y * 9
        for x in range(8):
            left = px[row + x]
            right = px[row + x + 1]

            if left > right:
                value |= (1 << bit)

            bit += 1

    return value


def ahash64(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        im = im.resize((8, 8), Image.Resampling.LANCZOS)
        px = list(im.getdata())

    mean = sum(px) / len(px)

    value = 0

    for bit, p in enumerate(px):
        if p >= mean:
            value |= (1 << bit)

    return value


def image_meta(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size

    return int(w), int(h)


def verify_upstream():
    require(R17A4B_MANIFEST)
    require(R17A4B_AUDIT)

    m = json.loads(
        R17A4B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R17A4B_FROZEN_453_COCO_PIXELS_DOWNLOADED_VERIFIED_AND_SHA256_LOCKED_READY_FOR_R17B_ROBUSTNESS_PROTOCOL_LOCK"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R17-A4B status: {m.get('status')}"
        )

    if int(m.get("valid_pixel_files", -1)) != 453:
        raise RuntimeError(
            "R17-A4B does not contain 453 valid frozen images."
        )

    if int(m.get("failed_pixel_files", -1)) != 0:
        raise RuntimeError(
            "R17-A4B contains failed pixel downloads."
        )

    return m


def infer_split(path, root):
    try:
        rel = path.relative_to(root)
    except Exception:
        return "UNKNOWN"

    parts = [p.lower() for p in rel.parts]

    if "train" in parts:
        return "Train"

    if "validation" in parts:
        return "Validation"

    if "valid" in parts:
        return "Validation"

    if "val" in parts:
        return "Validation"

    if "test" in parts:
        return "Test"

    return "UNKNOWN"


def scan_reference_root(root, source_name):
    rows = []

    if not root.exists():
        return pd.DataFrame()

    image_root = root / "images"

    if not image_root.exists():
        image_root = root

    files = sorted(
        p
        for p in image_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    )

    for i, p in enumerate(files, start=1):
        try:
            w, h = image_meta(p)

            rows.append(
                {
                    "source_dataset": source_name,
                    "split": infer_split(p, root),
                    "path": str(p),
                    "file_name": p.name,
                    "width": w,
                    "height": h,
                    "sha256": sha256_file(p),
                    "dhash64": f"{dhash64(p):016x}",
                    "ahash64": f"{ahash64(p):016x}",
                    "read_error": "",
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "source_dataset": source_name,
                    "split": infer_split(p, root),
                    "path": str(p),
                    "file_name": p.name,
                    "width": "",
                    "height": "",
                    "sha256": "",
                    "dhash64": "",
                    "ahash64": "",
                    "read_error": f"{type(exc).__name__}: {exc}",
                }
            )

        if i % 250 == 0 or i == len(files):
            print(
                f"  {source_name}: {i}/{len(files)} reference images hashed"
            )

    return pd.DataFrame(rows)


def load_frozen_coco():
    df = pd.read_csv(
        R17A4B_AUDIT,
        low_memory=False,
    )

    good = df["status"].isin(
        [
            "EXISTING_VALID",
            "DOWNLOADED_VALID",
        ]
    )

    df = df.loc[good].copy()

    if len(df) != 453:
        raise RuntimeError(
            f"Expected 453 valid frozen COCO images, found {len(df)}."
        )

    rows = []

    for i, r in df.iterrows():
        p = Path(
            str(r["local_path"])
        )

        require(p)

        w, h = image_meta(p)

        digest = sha256_file(p)

        locked_digest = str(
            r["sha256"]
        ).strip()

        if digest != locked_digest:
            raise RuntimeError(
                f"Frozen COCO image hash changed: {p}"
            )

        rows.append(
            {
                "evaluation_group": str(
                    r["evaluation_group"]
                ),
                "split": str(
                    r["split"]
                ),
                "image_id": int(
                    r["image_id"]
                ),
                "path": str(p),
                "file_name": p.name,
                "width": w,
                "height": h,
                "sha256": digest,
                "dhash64": f"{dhash64(p):016x}",
                "ahash64": f"{ahash64(p):016x}",
            }
        )

    return pd.DataFrame(rows)


def exact_overlap(coco, ref):
    ref_by_sha = {}

    for _, r in ref.iterrows():
        sha = str(r["sha256"]).strip()

        if not sha:
            continue

        ref_by_sha.setdefault(
            sha,
            [],
        ).append(r)

    rows = []

    for _, c in coco.iterrows():
        matches = ref_by_sha.get(
            c["sha256"],
            [],
        )

        for r in matches:
            rows.append(
                {
                    "coco_group": c["evaluation_group"],
                    "coco_split": c["split"],
                    "coco_image_id": c["image_id"],
                    "coco_path": c["path"],
                    "reference_dataset": r["source_dataset"],
                    "reference_split": r["split"],
                    "reference_path": r["path"],
                    "sha256": c["sha256"],
                }
            )

    return pd.DataFrame(rows)


def perceptual_candidates(coco, ref):
    ref_clean = ref.loc[
        (ref["read_error"] == "")
        & (ref["dhash64"] != "")
        & (ref["ahash64"] != "")
    ].copy()

    ref_records = []

    for _, r in ref_clean.iterrows():
        ref_records.append(
            {
                "source_dataset": r["source_dataset"],
                "split": r["split"],
                "path": r["path"],
                "file_name": r["file_name"],
                "width": int(r["width"]),
                "height": int(r["height"]),
                "sha256": r["sha256"],
                "dhash": int(r["dhash64"], 16),
                "ahash": int(r["ahash64"], 16),
            }
        )

    rows = []

    for idx, c in coco.iterrows():
        cd = int(c["dhash64"], 16)
        ca = int(c["ahash64"], 16)

        cw = int(c["width"])
        ch = int(c["height"])

        car = cw / ch if ch else 0.0

        for r in ref_records:
            # Exact matches are handled separately.
            if c["sha256"] == r["sha256"]:
                continue

            rar = (
                r["width"] / r["height"]
                if r["height"]
                else 0.0
            )

            # Loose aspect-ratio gate to avoid obviously unrelated
            # images generating hash collisions.
            if car and rar:
                rel_ar_diff = abs(car - rar) / max(
                    abs(car),
                    abs(rar),
                )

                if rel_ar_diff > 0.10:
                    continue

            dd = popcount64(
                cd ^ r["dhash"]
            )

            if dd > DHASH_MAX:
                continue

            ad = popcount64(
                ca ^ r["ahash"]
            )

            if ad > AHASH_MAX:
                continue

            rows.append(
                {
                    "coco_group": c["evaluation_group"],
                    "coco_split": c["split"],
                    "coco_image_id": c["image_id"],
                    "coco_path": c["path"],
                    "reference_dataset": r["source_dataset"],
                    "reference_split": r["split"],
                    "reference_path": r["path"],
                    "dhash_hamming": dd,
                    "ahash_hamming": ad,
                    "coco_width": cw,
                    "coco_height": ch,
                    "reference_width": r["width"],
                    "reference_height": r["height"],
                    "interpretation": (
                        "PERCEPTUAL_CANDIDATE_ONLY_MANUAL_REVIEW_REQUIRED_NOT_AUTOMATIC_LEAKAGE"
                    ),
                }
            )

        if (
            (idx + 1) % 75 == 0
            or (idx + 1) == len(coco)
        ):
            print(
                f"  perceptual scan: {idx + 1}/{len(coco)} frozen COCO images checked"
            )

    return pd.DataFrame(rows)


def main():
    print(
        "\nR17-A4C — DETECTOR-DATASET LEAKAGE / DUPLICATE AUDIT"
    )

    print(
        "AUDIT ONLY. NO training and NO inference. "
        "This checks whether the frozen 453-image COCO evaluation set overlaps the existing detector dataset. "
        "Exact byte duplicates are identified by SHA256. Potential near-duplicates are conservatively flagged "
        "using two 64-bit perceptual hashes plus an aspect-ratio gate; perceptual candidates are NOT automatically "
        "called leakage and require review.\n"
    )

    verify_upstream()

    print(
        "HASHING FROZEN COCO EVALUATION IMAGES ..."
    )

    coco = load_frozen_coco()

    print(
        f"  frozen COCO images: {len(coco)}"
    )

    print(
        "\nHASHING DETECTOR DATASET REFERENCE IMAGES ..."
    )

    v5 = scan_reference_root(
        V5_ROOT,
        "V5_SELECTED_CHECKPOINT_DATASET",
    )

    exported = scan_reference_root(
        EXPORTED_ROOT,
        "EXPORTED_EARLIER_DATASET",
    )

    reference = pd.concat(
        [
            v5,
            exported,
        ],
        ignore_index=True,
        sort=False,
    )

    reference_path = (
        OUT
        / "R17A4C_REFERENCE_IMAGE_HASHES.csv"
    )

    reference.to_csv(
        reference_path,
        index=False,
    )

    coco_path = (
        OUT
        / "R17A4C_FROZEN_COCO_IMAGE_HASHES.csv"
    )

    coco.to_csv(
        coco_path,
        index=False,
    )

    print(
        "\nCHECKING EXACT SHA256 OVERLAP ..."
    )

    exact = exact_overlap(
        coco,
        reference,
    )

    exact_path = (
        OUT
        / "R17A4C_EXACT_SHA256_OVERLAPS.csv"
    )

    exact.to_csv(
        exact_path,
        index=False,
    )

    print(
        "\nCHECKING PERCEPTUAL NEAR-DUPLICATE CANDIDATES ..."
    )

    near = perceptual_candidates(
        coco,
        reference,
    )

    near_path = (
        OUT
        / "R17A4C_PERCEPTUAL_NEAR_DUPLICATE_CANDIDATES.csv"
    )

    near.to_csv(
        near_path,
        index=False,
    )

    exact_n = int(
        len(exact)
    )

    near_n = int(
        len(near)
    )

    v5_valid = int(
        (
            (v5["read_error"] == "")
            if len(v5)
            else pd.Series(
                dtype=bool
            )
        ).sum()
    )

    exported_valid = int(
        (
            (exported["read_error"] == "")
            if len(exported)
            else pd.Series(
                dtype=bool
            )
        ).sum()
    )

    if exact_n > 0:
        status = (
            "R17A4C_EXACT_DETECTOR_DATASET_OVERLAP_FOUND_REFREEZE_EXTERNAL_SUBSET_BEFORE_R17B"
        )
        ready_for_r17b = False

    elif near_n > 0:
        status = (
            "R17A4C_NO_EXACT_OVERLAP_BUT_PERCEPTUAL_CANDIDATES_REQUIRE_REVIEW_BEFORE_R17B"
        )
        ready_for_r17b = False

    else:
        status = (
            "R17A4C_NO_EXACT_OR_FLAGGED_NEAR_DUPLICATE_OVERLAP_READY_FOR_R17B_FROZEN_ROBUSTNESS_PROTOCOL_LOCK"
        )
        ready_for_r17b = True

    summary = pd.DataFrame(
        [
            {
                "frozen_coco_images":
                    len(coco),

                "v5_reference_images_valid":
                    v5_valid,

                "exported_reference_images_valid":
                    exported_valid,

                "exact_sha256_overlap_pairs":
                    exact_n,

                "perceptual_candidate_pairs":
                    near_n,

                "dhash_max":
                    DHASH_MAX,

                "ahash_max":
                    AHASH_MAX,

                "ready_for_r17b":
                    ready_for_r17b,

                "status":
                    status,
            }
        ]
    )

    summary_path = (
        OUT
        / "R17A4C_LEAKAGE_AUDIT_SUMMARY.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "step":
            "R17A4C_DETECTOR_DATASET_LEAKAGE_AUDIT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "model_inference":
            False,

        "frozen_coco_images":
            int(
                len(coco)
            ),

        "reference_datasets": [
            str(V5_ROOT),
            str(EXPORTED_ROOT),
        ],

        "exact_overlap_method":
            "SHA256",

        "near_duplicate_method":
            (
                f"64-bit dHash <= {DHASH_MAX} AND 64-bit aHash <= {AHASH_MAX} "
                "AND relative aspect-ratio difference <= 10%; candidates require manual review"
            ),

        "exact_sha256_overlap_pairs":
            exact_n,

        "perceptual_candidate_pairs":
            near_n,

        "ready_for_r17b":
            ready_for_r17b,

        "important_interpretation": [
            "Exact SHA256 overlap is treated as confirmed duplicate leakage.",
            "Perceptual-hash matches are screening candidates only and are not automatically labeled leakage.",
            "This audit compares against the selected V5 detector dataset and the earlier exported detector dataset found in R17-A.",
            "If exact overlap is found, affected external evaluation images must be excluded and the frozen subset regenerated before inference.",
        ],

        "outputs": {
            "summary":
                str(summary_path),

            "reference_hashes":
                str(reference_path),

            "frozen_coco_hashes":
                str(coco_path),

            "exact_overlaps":
                str(exact_path),

            "perceptual_candidates":
                str(near_path),
        },
    }

    (
        OUT
        / "R17A4C_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nLEAKAGE AUDIT SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    if exact_n:
        print(
            "\nEXACT SHA256 OVERLAPS:"
        )

        print(
            exact.to_string(
                index=False
            )
        )

    if near_n:
        print(
            "\nPERCEPTUAL NEAR-DUPLICATE CANDIDATES:"
        )

        print(
            near.to_string(
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
        "\nR17-A4C COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
