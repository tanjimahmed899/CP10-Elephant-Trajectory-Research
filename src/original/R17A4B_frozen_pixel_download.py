
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import urllib.error

import pandas as pd
from PIL import Image


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"
R17_ROOT = REVISION / "R17_mixed_species_robustness"

R17A4A_DIR = R17_ROOT / "R17A4A_frozen_subset_selection"
R17A4A_MANIFEST = R17A4A_DIR / "R17A4A_MANIFEST.json"
PRIMARY_CSV = R17A4A_DIR / "R17A4A_PRIMARY_BALANCED_453.csv"

R17A3B_DIR = R17_ROOT / "R17A3B_coco_animal_cooccurrence_audit"
R17A3B_MANIFEST = R17A3B_DIR / "R17A3B_MANIFEST.json"
TRAIN_JSON = R17A3B_DIR / "coco_metadata" / "annotations" / "instances_train2017.json"
VAL_JSON = R17A3B_DIR / "coco_metadata" / "annotations" / "instances_val2017.json"

OUT = R17_ROOT / "R17A4B_frozen_pixel_download"
PIXELS = OUT / "images"
PIXELS.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0 R17-COCO-Evaluation/1.0"
MAX_RETRIES_PER_SOURCE = 2


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def verify_upstream():
    for p in [
        R17A4A_MANIFEST,
        PRIMARY_CSV,
        R17A3B_MANIFEST,
        TRAIN_JSON,
        VAL_JSON,
    ]:
        require(p)

    a4a = json.loads(
        R17A4A_MANIFEST.read_text(encoding="utf-8")
    )

    a3b = json.loads(
        R17A3B_MANIFEST.read_text(encoding="utf-8")
    )

    expected_a4a = (
        "R17A4A_FROZEN_EVALUATION_SUBSET_LOCKED_453_COCO_PRIMARY_PLUS_DEDUPLICATED_IN_VOCAB_SUPPLEMENT_READY_FOR_R17A4B_PIXEL_DOWNLOAD"
    )

    expected_a3b = (
        "R17A3B_COCO_ANIMAL_AMONG_ANIMALS_POOL_VERIFIED_READY_FOR_R17A4_FROZEN_SUBSET_SELECTION_AND_PIXEL_DOWNLOAD"
    )

    if a4a.get("status") != expected_a4a:
        raise RuntimeError(
            f"Unexpected R17-A4A status: {a4a.get('status')}"
        )

    if a3b.get("status") != expected_a3b:
        raise RuntimeError(
            f"Unexpected R17-A3B status: {a3b.get('status')}"
        )

    if sha256_file(PRIMARY_CSV) != a4a["output_sha256"]["primary"]:
        raise RuntimeError(
            "Frozen R17-A4A primary subset CSV hash mismatch."
        )

    if sha256_file(TRAIN_JSON) != a3b["train_json_sha256"]:
        raise RuntimeError(
            "Frozen train2017 annotation JSON hash mismatch."
        )

    if sha256_file(VAL_JSON) != a3b["val_json_sha256"]:
        raise RuntimeError(
            "Frozen val2017 annotation JSON hash mismatch."
        )

    return a4a, a3b


def load_image_metadata():
    mapping = {}

    for split, path in [
        ("train2017", TRAIN_JSON),
        ("val2017", VAL_JSON),
    ]:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        for item in data["images"]:
            image_id = int(item["id"])

            mapping[(split, image_id)] = {
                "file_name": str(item["file_name"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "coco_url": str(item.get("coco_url", "")).strip(),
                "flickr_url": str(item.get("flickr_url", "")).strip(),
            }

    return mapping


def normalize_split(split):
    s = str(split).strip().lower()

    if s == "train2017":
        return "train2017"

    if s == "val2017":
        return "val2017"

    raise RuntimeError(
        f"Unexpected COCO split: {split}"
    )


def verify_image_file(path, expected_w, expected_h):
    try:
        with Image.open(path) as im:
            im.verify()

        with Image.open(path) as im:
            w, h = im.size

        return (
            int(w) == int(expected_w)
            and int(h) == int(expected_h)
        )
    except Exception:
        return False


def download_url(url, destination):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/*,*/*;q=0.8",
        },
    )

    with urllib.request.urlopen(
        req,
        timeout=60,
    ) as response:
        with open(destination, "wb") as f:
            while True:
                chunk = response.read(
                    1024 * 1024
                )
                if not chunk:
                    break
                f.write(chunk)


def source_urls(meta):
    urls = []

    coco_url = meta["coco_url"]

    if coco_url:
        # COCO metadata commonly carries plain HTTP image URLs.
        # Preserve that canonical metadata URL first.
        urls.append(
            ("COCO_METADATA_URL", coco_url)
        )

    flickr_url = meta["flickr_url"]

    if flickr_url:
        urls.append(
            ("FLICKR_METADATA_URL", flickr_url)
        )

    # De-duplicate while preserving order.
    seen = set()
    out = []

    for label, url in urls:
        if url in seen:
            continue
        seen.add(url)
        out.append((label, url))

    return out


def main():
    print(
        "\nR17-A4B — FROZEN COCO PIXEL DOWNLOAD"
    )

    print(
        "PIXEL ACQUISITION ONLY. NO model inference and NO tuning. "
        "The already frozen 453-image R17-A4A subset is downloaded without changing image IDs. "
        "Each downloaded file must match the width and height stored in the checksum-verified COCO annotation JSON. "
        "Successful local files are SHA256-locked. Existing valid files are resumed/skipped.\n"
    )

    a4a, a3b = verify_upstream()

    selected = pd.read_csv(
        PRIMARY_CSV,
        low_memory=False,
    )

    if len(selected) != 453:
        raise RuntimeError(
            f"Frozen primary subset should contain 453 rows, found {len(selected)}."
        )

    selected["image_id"] = pd.to_numeric(
        selected["image_id"],
        errors="raise",
    ).astype(int)

    selected["split"] = (
        selected["split"]
        .map(normalize_split)
    )

    metadata = load_image_metadata()

    audit_rows = []

    total = len(selected)

    for index, row in selected.iterrows():
        split = row["split"]
        image_id = int(row["image_id"])
        group = str(row["evaluation_group"])

        key = (split, image_id)

        if key not in metadata:
            raise RuntimeError(
                f"Selected image missing from verified COCO annotation metadata: {key}"
            )

        meta = metadata[key]

        if (
            str(row["file_name"]).strip()
            != meta["file_name"]
        ):
            raise RuntimeError(
                f"Frozen file-name mismatch for {key}: "
                f"{row['file_name']} vs {meta['file_name']}"
            )

        group_dir = PIXELS / group / split
        group_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dest = group_dir / meta["file_name"]

        status = ""
        source_label = ""
        source_url = ""
        errors = []

        if (
            dest.exists()
            and verify_image_file(
                dest,
                meta["width"],
                meta["height"],
            )
        ):
            status = "EXISTING_VALID"
        else:
            if dest.exists():
                dest.unlink()

            downloaded = False

            for source_label_try, url in source_urls(meta):
                for attempt in range(
                    1,
                    MAX_RETRIES_PER_SOURCE + 1,
                ):
                    tmp = dest.with_suffix(
                        dest.suffix + ".part"
                    )

                    if tmp.exists():
                        tmp.unlink()

                    try:
                        download_url(
                            url,
                            tmp,
                        )

                        if not verify_image_file(
                            tmp,
                            meta["width"],
                            meta["height"],
                        ):
                            raise RuntimeError(
                                "Downloaded image dimension/integrity verification failed."
                            )

                        tmp.replace(dest)

                        downloaded = True
                        status = "DOWNLOADED_VALID"
                        source_label = source_label_try
                        source_url = url
                        break

                    except Exception as exc:
                        if tmp.exists():
                            tmp.unlink()

                        errors.append(
                            f"{source_label_try} attempt {attempt}: {type(exc).__name__}: {exc}"
                        )

                        time.sleep(
                            min(
                                2 * attempt,
                                4,
                            )
                        )

                if downloaded:
                    break

            if not downloaded:
                status = "FAILED_ALL_SOURCES"

        if status in {
            "EXISTING_VALID",
            "DOWNLOADED_VALID",
        }:
            digest = sha256_file(dest)
            size_bytes = dest.stat().st_size
        else:
            digest = ""
            size_bytes = 0

        audit_rows.append(
            {
                "evaluation_group": group,
                "split": split,
                "image_id": image_id,
                "file_name": meta["file_name"],
                "expected_width": meta["width"],
                "expected_height": meta["height"],
                "local_path": str(dest),
                "status": status,
                "source_label": source_label,
                "source_url": source_url,
                "size_bytes": size_bytes,
                "sha256": digest,
                "errors": " || ".join(errors),
            }
        )

        n = index + 1

        if (
            n == 1
            or n % 25 == 0
            or n == total
        ):
            print(
                f"  {n}/{total} processed"
            )

    audit = pd.DataFrame(
        audit_rows
    )

    audit_path = OUT / "R17A4B_PIXEL_DOWNLOAD_AUDIT.csv"

    audit.to_csv(
        audit_path,
        index=False,
    )

    summary = (
        audit.groupby(
            [
                "evaluation_group",
                "status",
            ],
            dropna=False,
        )
        .size()
        .rename("image_count")
        .reset_index()
    )

    summary_path = OUT / "R17A4B_DOWNLOAD_SUMMARY.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    valid_mask = audit["status"].isin(
        [
            "EXISTING_VALID",
            "DOWNLOADED_VALID",
        ]
    )

    valid_count = int(
        valid_mask.sum()
    )

    failed_count = int(
        len(audit) - valid_count
    )

    duplicate_hashes = int(
        audit.loc[
            valid_mask,
            "sha256",
        ]
        .value_counts()
        .gt(1)
        .sum()
    )

    if valid_count == 453 and duplicate_hashes == 0:
        status = (
            "R17A4B_FROZEN_453_COCO_PIXELS_DOWNLOADED_VERIFIED_AND_SHA256_LOCKED_READY_FOR_R17B_ROBUSTNESS_PROTOCOL_LOCK"
        )
    else:
        status = (
            "R17A4B_PIXEL_DOWNLOAD_INCOMPLETE_OR_DUPLICATE_REVIEW_REQUIRED"
        )

    manifest = {
        "step":
            "R17A4B_FROZEN_COCO_PIXEL_DOWNLOAD",

        "status":
            status,

        "training":
            False,

        "model_inference":
            False,

        "tuning":
            False,

        "frozen_subset_rows":
            int(len(selected)),

        "valid_pixel_files":
            valid_count,

        "failed_pixel_files":
            failed_count,

        "duplicate_sha256_groups":
            duplicate_hashes,

        "primary_subset_sha256":
            sha256_file(PRIMARY_CSV),

        "annotation_integrity_source":
            "R17A3B checksum-verified COCO 2017 annotation JSON",

        "image_validation":
            "Pillow decode + exact width/height match against verified COCO metadata + local SHA256 lock",

        "important_limit":
            (
                "Image transport is not used as annotation truth; annotation identity and dimensions "
                "come from the checksum-verified COCO JSON. Local SHA256 values are locked after acquisition."
            ),

        "outputs": {
            "pixel_download_audit":
                str(audit_path),
            "download_summary":
                str(summary_path),
            "pixel_root":
                str(PIXELS),
        },
    }

    manifest_path = OUT / "R17A4B_MANIFEST.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nDOWNLOAD SUMMARY:"
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nINTEGRITY SUMMARY:"
    )
    print(
        f"valid pixel files = {valid_count}/453"
    )
    print(
        f"failed pixel files = {failed_count}"
    )
    print(
        f"duplicate SHA256 groups = {duplicate_hashes}"
    )

    if failed_count:
        print(
            "\nFAILED FILES:"
        )
        print(
            audit.loc[
                ~valid_mask,
                [
                    "evaluation_group",
                    "split",
                    "image_id",
                    "file_name",
                    "errors",
                ],
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
        "\nR17-A4B COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
