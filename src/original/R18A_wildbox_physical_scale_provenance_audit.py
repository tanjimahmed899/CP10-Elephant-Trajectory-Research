from pathlib import Path
import json
import hashlib
import zipfile
import re
import subprocess
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
REVISION = DOWNLOADS / "paper_strengthening_revision"

R17E = (
    REVISION
    / "R17_mixed_species_robustness"
    / "R17E_claim_interpretation_lock"
    / "R17E_CLAIM_INTERPRETATION_LOCK.json"
)

OUT = (
    REVISION
    / "R18_meter_space_wildbox"
    / "R18A_physical_scale_provenance_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

CAPSTONE_ROOT = Path(r"E:\AIUB\Capstone Project")

KNOWN_WILDBOX_ZIPS = [
    "DJI_20240119152335_0005_V.zip",
    "DJI_20240626142859_0001_V.zip",
    "DJI_20260227084135_0002_V.zip",
]

KNOWN_RESULT_ROOTS = [
    DOWNLOADS / "step3_fresh_wildbox_full_comparison",
    CAPSTONE_ROOT / "Journal Based Results" / "Main Result",
    CAPSTONE_ROOT / "Journal Based Results" / "Main Result 2.0",
]

TEXT_EXTS = {
    ".txt", ".csv", ".json", ".xml", ".srt", ".kml",
    ".gpx", ".xmp", ".yaml", ".yml", ".ini", ".log"
}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

KEYWORD_GROUPS = {
    "altitude_or_height": [
        r"\baltitude\b", r"\bheight\b", r"\brelativealtitude\b",
        r"\babsolutealtitude\b", r"\bflightheight\b", r"\balt\b"
    ],
    "focal_or_fov": [
        r"\bfocal\b", r"\bfov\b", r"field.?of.?view",
        r"focal.?length", r"sensor.?width", r"sensor.?height"
    ],
    "intrinsics": [
        r"intrinsic", r"camera.?matrix", r"\bfx\b", r"\bfy\b",
        r"\bcx\b", r"\bcy\b", r"principal.?point"
    ],
    "gps": [r"\bgps\b", r"latitude", r"longitude", r"\blat\b", r"\blon\b"],
    "gimbal_or_pose": [
        r"gimbal", r"\byaw\b", r"\bpitch\b", r"\broll\b", r"attitude", r"pose"
    ],
    "frame_geometry": [
        r"\bwidth\b", r"\bheight\b", r"resolution",
        r"frame.?size", r"image.?size"
    ],
}

MAX_TEXT_BYTES = 2_000_000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_r17_complete():
    if not R17E.exists():
        raise FileNotFoundError(f"Missing R17-E file: {R17E}")

    m = read_json(R17E)
    print("R17-E manifest found.")

    status = m.get("status", "")
    print("R17-E status:", status)
    return m


def locate_exact_filename(root, filename, max_hits=10):
    hits = []
    if not root.exists():
        return hits

    try:
        for p in root.rglob(filename):
            if p.is_file():
                hits.append(p)
                if len(hits) >= max_hits:
                    break
    except Exception:
        pass

    return hits


def locate_wildbox_zips():
    rows = []
    search_roots = [DOWNLOADS, CAPSTONE_ROOT]

    for name in KNOWN_WILDBOX_ZIPS:
        all_hits = []
        for root in search_roots:
            all_hits.extend(locate_exact_filename(root, name, max_hits=10))

        unique = {}
        for p in all_hits:
            unique[str(p)] = p

        if not unique:
            rows.append({
                "zip_name": name,
                "found": False,
                "path": "",
                "size_bytes": 0,
                "sha256": "",
            })
        else:
            for p in unique.values():
                rows.append({
                    "zip_name": name,
                    "found": True,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })

    return pd.DataFrame(rows)


def keyword_hits(text):
    lower = text.lower()
    result = {}
    for group, pats in KEYWORD_GROUPS.items():
        matched = []
        for pat in pats:
            if re.search(pat, lower, flags=re.IGNORECASE):
                matched.append(pat)
        result[group] = bool(matched)
    return result


def preview_lines(text, max_lines=12):
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        lower = s.lower()

        found = False
        for pats in KEYWORD_GROUPS.values():
            for pat in pats:
                if re.search(pat, lower, flags=re.IGNORECASE):
                    found = True
                    break
            if found:
                break

        if found:
            lines.append(s[:400])

        if len(lines) >= max_lines:
            break

    return " || ".join(lines)


def inspect_zip(path):
    member_rows = []
    text_rows = []

    try:
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                suffix = Path(info.filename).suffix.lower()

                member_rows.append({
                    "zip_path": str(path),
                    "member": info.filename,
                    "suffix": suffix,
                    "size_bytes": int(info.file_size),
                    "compressed_bytes": int(info.compress_size),
                    "is_text_metadata": suffix in TEXT_EXTS,
                    "is_video": suffix in VIDEO_EXTS,
                    "is_image": suffix in IMAGE_EXTS,
                })

                if suffix in TEXT_EXTS and info.file_size <= MAX_TEXT_BYTES:
                    try:
                        data = zf.read(info)
                        text = data.decode("utf-8", errors="ignore")
                        hits = keyword_hits(text)

                        text_rows.append({
                            "zip_path": str(path),
                            "member": info.filename,
                            "suffix": suffix,
                            "size_bytes": int(info.file_size),
                            **hits,
                            "preview": preview_lines(text),
                        })
                    except Exception as exc:
                        text_rows.append({
                            "zip_path": str(path),
                            "member": info.filename,
                            "suffix": suffix,
                            "size_bytes": int(info.file_size),
                            "altitude_or_height": False,
                            "focal_or_fov": False,
                            "intrinsics": False,
                            "gps": False,
                            "gimbal_or_pose": False,
                            "frame_geometry": False,
                            "preview": f"READ_ERROR: {type(exc).__name__}: {exc}",
                        })
    except zipfile.BadZipFile as exc:
        member_rows.append({
            "zip_path": str(path),
            "member": "",
            "suffix": "",
            "size_bytes": 0,
            "compressed_bytes": 0,
            "is_text_metadata": False,
            "is_video": False,
            "is_image": False,
            "error": f"BadZipFile: {exc}",
        })

    return pd.DataFrame(member_rows), pd.DataFrame(text_rows)


def scan_result_files():
    rows = []

    keywords = (
        "wildbox", "fresh", "metadata", "prediction", "comparison",
        "trajectory", "camera", "intrinsic", "extrinsic", "ego",
        "fov", "altitude"
    )

    allowed = {".csv", ".json", ".txt", ".npz", ".npy", ".pt", ".yaml", ".yml"}

    for root in KNOWN_RESULT_ROOTS:
        if not root.exists():
            continue

        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in allowed:
                    continue

                low = str(p).lower()
                if not any(k in low for k in keywords):
                    continue

                rows.append({
                    "root": str(root),
                    "path": str(p),
                    "suffix": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p) if p.stat().st_size <= 200_000_000 else "",
                })
        except Exception:
            pass

    return pd.DataFrame(rows)


def inspect_local_text_candidates(files_df):
    rows = []

    if len(files_df) == 0:
        return pd.DataFrame()

    for _, r in files_df.iterrows():
        p = Path(str(r["path"]))
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        if p.stat().st_size > MAX_TEXT_BYTES:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        hits = keyword_hits(text)
        if not any(hits.values()):
            continue

        rows.append({
            "path": str(p),
            "size_bytes": p.stat().st_size,
            **hits,
            "preview": preview_lines(text),
        })

    return pd.DataFrame(rows)


def ffprobe_available():
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False
        )
        return result.returncode == 0 or "ffprobe" in (result.stdout + result.stderr).lower()
    except Exception:
        return False


def ffprobe_local_videos():
    rows = []
    candidate_roots = [DOWNLOADS, CAPSTONE_ROOT]
    local_videos = []

    for root in candidate_roots:
        if not root.exists():
            continue

        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in VIDEO_EXTS:
                    continue

                low = str(p).lower()
                if "dji_" in low or "wildbox" in low:
                    local_videos.append(p)
        except Exception:
            pass

    unique = {}
    for p in local_videos:
        unique[str(p)] = p

    if not ffprobe_available():
        return pd.DataFrame([{
            "ffprobe_available": False,
            "video_path": "",
            "width": "",
            "height": "",
            "fps": "",
            "duration_s": "",
            "codec": "",
            "tags_json": "",
        }])

    for p in unique.values():
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name:format=duration:format_tags",
            "-of", "json",
            str(p),
        ]

        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)

            if cp.returncode != 0:
                rows.append({
                    "ffprobe_available": True,
                    "video_path": str(p),
                    "width": "",
                    "height": "",
                    "fps": "",
                    "duration_s": "",
                    "codec": "",
                    "tags_json": "",
                    "error": cp.stderr.strip()[:500],
                })
                continue

            data = json.loads(cp.stdout)
            streams = data.get("streams", [])
            stream = streams[0] if streams else {}
            fmt = data.get("format", {})

            rows.append({
                "ffprobe_available": True,
                "video_path": str(p),
                "width": stream.get("width", ""),
                "height": stream.get("height", ""),
                "fps": stream.get("r_frame_rate", ""),
                "duration_s": fmt.get("duration", ""),
                "codec": stream.get("codec_name", ""),
                "tags_json": json.dumps(fmt.get("tags", {}), ensure_ascii=False),
            })
        except Exception as exc:
            rows.append({
                "ffprobe_available": True,
                "video_path": str(p),
                "width": "",
                "height": "",
                "fps": "",
                "duration_s": "",
                "codec": "",
                "tags_json": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

    return pd.DataFrame(rows)


def main():
    print("\nR18-A — WILDBOX PHYSICAL-SCALE / METER-SPACE PROVENANCE AUDIT")
    print(
        "AUDIT ONLY. NO coordinate conversion and NO new forecasting inference. "
        "This step checks whether the frozen WildBox sources contain enough physical-scale evidence "
        "(altitude/height plus camera geometry/intrinsics/FOV plus frame geometry) to justify a meter-space conversion.\n"
    )

    verify_r17_complete()

    print("LOCATING KNOWN FRESH WILDBOX ZIP SOURCES ...")
    zip_df = locate_wildbox_zips()
    zip_df.to_csv(OUT / "R18A_WILDBOX_ZIP_PROVENANCE.csv", index=False)
    print(zip_df.to_string(index=False))

    found_paths = [
        Path(str(r["path"]))
        for _, r in zip_df.iterrows()
        if bool(r["found"])
    ]

    all_members = []
    all_text = []

    for p in found_paths:
        print(f"\nINSPECTING ZIP: {p}")
        members, text = inspect_zip(p)
        if len(members):
            all_members.append(members)
        if len(text):
            all_text.append(text)

    members_df = pd.concat(all_members, ignore_index=True, sort=False) if all_members else pd.DataFrame()
    text_df = pd.concat(all_text, ignore_index=True, sort=False) if all_text else pd.DataFrame()

    members_df.to_csv(OUT / "R18A_ZIP_MEMBER_INVENTORY.csv", index=False)
    text_df.to_csv(OUT / "R18A_ZIP_TEXT_METADATA_EVIDENCE.csv", index=False)

    print("\nSCANNING EXISTING WILDBOX RESULT / METADATA FILES ...")
    result_files = scan_result_files()
    result_files.to_csv(OUT / "R18A_EXISTING_WILDBOX_RESULT_FILE_INVENTORY.csv", index=False)

    local_text = inspect_local_text_candidates(result_files)
    local_text.to_csv(OUT / "R18A_LOCAL_TEXT_METADATA_EVIDENCE.csv", index=False)

    print("\nCHECKING ALREADY-LOCAL VIDEO CONTAINER METADATA WITH FFPROBE ...")
    ffprobe_df = ffprobe_local_videos()
    ffprobe_df.to_csv(OUT / "R18A_LOCAL_VIDEO_FFPROBE_METADATA.csv", index=False)

    evidence_frames = []
    if len(text_df):
        evidence_frames.append(text_df)
    if len(local_text):
        evidence_frames.append(local_text)

    evidence = pd.concat(evidence_frames, ignore_index=True, sort=False) if evidence_frames else pd.DataFrame()

    def any_flag(col):
        if len(evidence) == 0 or col not in evidence.columns:
            return False
        return bool(evidence[col].fillna(False).astype(bool).any())

    has_altitude = any_flag("altitude_or_height")
    has_camera_geometry = any_flag("focal_or_fov") or any_flag("intrinsics")
    has_gps = any_flag("gps")
    has_pose = any_flag("gimbal_or_pose")
    has_frame_geometry_text = any_flag("frame_geometry")

    ffprobe_width_height = False
    if len(ffprobe_df) and "width" in ffprobe_df.columns and "height" in ffprobe_df.columns:
        try:
            ffprobe_width_height = bool(
                (pd.to_numeric(ffprobe_df["width"], errors="coerce") > 0).fillna(False).any()
                and
                (pd.to_numeric(ffprobe_df["height"], errors="coerce") > 0).fillna(False).any()
            )
        except Exception:
            ffprobe_width_height = False

    has_frame_geometry = has_frame_geometry_text or ffprobe_width_height

    video_members = 0
    if len(members_df) and "is_video" in members_df.columns:
        video_members = int(members_df["is_video"].fillna(False).astype(bool).sum())

    direct_scale_ready = bool(has_altitude and has_camera_geometry and has_frame_geometry)

    if direct_scale_ready:
        status = "R18A_PHYSICAL_SCALE_EVIDENCE_FOUND_ALTITUDE_CAMERA_GEOMETRY_AND_FRAME_GEOMETRY_READY_FOR_R18B_FROZEN_METER_SPACE_PROTOCOL_LOCK"
        next_step = "R18B_FROZEN_METER_SPACE_PROTOCOL_LOCK"
    elif video_members > 0:
        status = "R18A_DIRECT_METER_SCALE_NOT_YET_JUSTIFIED_RAW_VIDEO_PRESENT_READY_FOR_R18B_MEDIA_METADATA_EXTRACTION_BEFORE_ANY_METER_CONVERSION"
        next_step = "R18B_MEDIA_METADATA_EXTRACTION"
    else:
        status = "R18A_PHYSICAL_SCALE_EVIDENCE_INSUFFICIENT_NO_JUSTIFIED_METER_CONVERSION_LOCK_LIMITATION_OR_ACQUIRE_CAMERA_METADATA"
        next_step = "R18_LIMITATION_LOCK_OR_CAMERA_METADATA_ACQUISITION"

    summary = pd.DataFrame([{
        "known_zip_sources_expected": len(KNOWN_WILDBOX_ZIPS),
        "zip_files_found": int(zip_df["found"].fillna(False).astype(bool).sum()) if len(zip_df) else 0,
        "zip_video_members": video_members,
        "altitude_or_height_evidence": has_altitude,
        "focal_fov_or_intrinsics_evidence": has_camera_geometry,
        "gps_evidence": has_gps,
        "gimbal_or_pose_evidence": has_pose,
        "frame_geometry_evidence": has_frame_geometry,
        "direct_meter_scale_ready": direct_scale_ready,
        "next_step": next_step,
        "status": status,
    }])

    summary.to_csv(OUT / "R18A_PHYSICAL_SCALE_READINESS_SUMMARY.csv", index=False)

    manifest = {
        "step": "R18A_WILDBOX_PHYSICAL_SCALE_PROVENANCE_AUDIT",
        "created_utc": utc_now(),
        "status": status,
        "coordinate_conversion_performed": False,
        "forecast_inference_performed": False,
        "known_wildbox_zip_names": KNOWN_WILDBOX_ZIPS,
        "zip_files_found": int(summary.iloc[0]["zip_files_found"]),
        "zip_video_members": video_members,
        "evidence": {
            "altitude_or_height": has_altitude,
            "focal_fov_or_intrinsics": has_camera_geometry,
            "gps": has_gps,
            "gimbal_or_pose": has_pose,
            "frame_geometry": has_frame_geometry,
            "direct_meter_scale_ready": direct_scale_ready,
        },
        "scientific_rule": (
            "Normalized/image-plane WildBox errors must not be converted to meters unless physical scale "
            "is justified by source-specific camera geometry and distance/altitude evidence."
        ),
    }

    (OUT / "R18A_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\nPHYSICAL-SCALE READINESS SUMMARY:")
    print(summary.to_string(index=False))

    if len(text_df):
        print("\nZIP TEXT METADATA EVIDENCE:")
        cols = [
            c for c in [
                "zip_path", "member", "altitude_or_height", "focal_or_fov",
                "intrinsics", "gps", "gimbal_or_pose", "frame_geometry", "preview"
            ] if c in text_df.columns
        ]
        print(text_df[cols].to_string(index=False))

    if len(local_text):
        print("\nLOCAL TEXT METADATA EVIDENCE:")
        print(local_text.to_string(index=False))

    if len(ffprobe_df):
        print("\nLOCAL VIDEO FFPROBE METADATA:")
        print(ffprobe_df.to_string(index=False))

    print("\nSTATUS:")
    print(status)

    print("\nR18-A COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()