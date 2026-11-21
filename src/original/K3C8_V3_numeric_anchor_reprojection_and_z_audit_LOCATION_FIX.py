from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import math
import re

import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
except Exception:
    Transformer = None


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

WORLD_CSV = BASE / "step6dk_valid_world_trajectory_rows.csv"

INFERENCE_DIR = (
    BASE
    / "k3_frozen_ensemble"
    / "first_inference"
)

OUT = (
    BASE
    / "k3_operational_warning"
    / "numeric_anchor_reprojection_equivalence_test_v3"
)

OUT.mkdir(parents=True, exist_ok=True)

FOVY_DEG = 50.0  # locked BAMBI reference-pipeline assumption, not measured calibration


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


def discover_one(filename, prefer_tokens=()):
    matches = sorted(
        p for p in DOWNLOADS.rglob(filename)
        if p.is_file()
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {DOWNLOADS}"
        )

    preferred = [
        p for p in matches
        if all(
            token.lower() in str(p).lower()
            for token in prefer_tokens
        )
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(matches) == 1:
        return matches[0]

    # Prefer known BAMBI folders.
    preferred = [
        p for p in matches
        if any(
            token in str(p).lower()
            for token in [
                "bambi_downloads",
                "step6d_uncertainty_holdout",
                "dem",
            ]
        )
    ]

    if len(preferred) == 1:
        return preferred[0]

    raise RuntimeError(
        f"Multiple unresolved candidates for {filename}:\n"
        + "\n".join(str(p) for p in matches)
    )


def load_official_module():
    spec = importlib.util.find_spec(
        "bambi.georeference_mot17_tracks"
    )

    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Installed bambi.georeference_mot17_tracks not found."
        )

    module = __import__(
        "bambi.georeference_mot17_tracks",
        fromlist=["*"]
    )

    return module, Path(spec.origin)


def normalize_correction_dict(raw, frame_idx):
    """
    Reproduce the correction selection semantics without changing source files.
    Supports underscore/spaced key variants found across downloaded/audited files.
    """
    rotation = raw.get(
        "rotation",
        {"x": 0.0, "y": 0.0, "z": 0.0}
    )

    translation = raw.get(
        "translation",
        {"x": 0.0, "y": 0.0, "z": 0.0}
    )

    additions = (
        raw.get("fine_corrections")
        or raw.get("additional_corrections")
        or raw.get("additional corrections")
        or []
    )

    chosen_rot = dict(rotation)
    chosen_trans = dict(translation)

    for item in additions:
        start = (
            item.get("start frame")
            if "start frame" in item
            else item.get("start_frame")
        )

        end = (
            item.get("end frame")
            if "end frame" in item
            else item.get("end_frame")
        )

        if start is None or end is None:
            continue

        # Match official source semantics: start < frame_idx < end
        if int(start) < int(frame_idx) < int(end):
            chosen_rot = dict(
                item.get("rotation", chosen_rot)
            )
            chosen_trans = dict(
                item.get("translation", chosen_trans)
            )
            break

    return chosen_rot, chosen_trans



def find_dem_json():
    candidates = sorted(
        p
        for p in DOWNLOADS.rglob("6_matched_dem.json")
        if p.is_file()
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not find 6_matched_dem.json under Downloads."
        )

    preferred = [
        p
        for p in candidates
        if "dem" in str(p.parent).lower()
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Multiple unresolved 6_matched_dem.json candidates:\n"
        + "\n".join(str(p) for p in candidates)
    )


def pose_has_required_geometry(poses):
    if not isinstance(poses, dict):
        return False

    images = poses.get("images")

    if not isinstance(images, list) or not images:
        return False

    # Require a representative subset to contain location.
    probe_indices = sorted(
        set(
            [
                0,
                min(100, len(images) - 1),
                min(1000, len(images) - 1),
                len(images) - 1,
            ]
        )
    )

    return all(
        isinstance(images[i], dict)
        and
        "location" in images[i]
        for i in probe_indices
    )


def discover_enriched_pose_json():
    """
    Prefer an existing BAMBI working/enriched pose JSON that already contains
    per-frame 'location'. Never modify it.
    """
    candidates = sorted(
        p
        for p in DOWNLOADS.rglob("*.json")
        if p.is_file()
        and "6" in p.name
        and "pose" in p.name.lower()
    )

    hits = []

    for p in candidates:
        try:
            raw = json.loads(
                p.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if pose_has_required_geometry(raw):
            hits.append(
                (
                    p,
                    raw
                )
            )

    if not hits:
        return None, None

    # Prefer files under step6d / georef / working folders.
    preferred = [
        x
        for x in hits
        if any(
            token in str(x[0]).lower()
            for token in [
                "step6d",
                "georef",
                "working",
                "dem",
            ]
        )
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(hits) == 1:
        return hits[0]

    # If multiple enriched files exist, prefer the one with fovy already present.
    fovy_hits = []

    for p, raw in hits:
        images = raw.get(
            "images",
            []
        )

        if (
            images
            and
            isinstance(images[0], dict)
            and
            "fovy" in images[0]
        ):
            fovy_hits.append(
                (
                    p,
                    raw
                )
            )

    if len(fovy_hits) == 1:
        return fovy_hits[0]

    # Ambiguous enriched copies: do not guess.
    raise RuntimeError(
        "Multiple enriched BAMBI pose JSON files with 'location' were found:\n"
        + "\n".join(
            str(p)
            for p, _ in hits
        )
    )


def enrich_raw_pose_in_memory(raw_poses, dem_json_path):
    """
    Reproduce add_relative_dem_position_to_poses.py semantics in-memory:
      location = [easting-origin_x, northing-origin_y, alt-origin_z]
      rotation = [pitch, roll, yaw]

    Original pose JSON is NOT modified on disk.
    """
    if Transformer is None:
        raise RuntimeError(
            "pyproj is required for in-memory pose enrichment but is unavailable."
        )

    dem_meta = json.loads(
        dem_json_path.read_text(
            encoding="utf-8"
        )
    )

    if "crs" not in dem_meta or "origin" not in dem_meta:
        raise RuntimeError(
            "DEM metadata lacks required 'crs' and/or 'origin'."
        )

    origin = np.asarray(
        dem_meta[
            "origin"
        ],
        dtype=float
    )

    if origin.size < 3:
        raise RuntimeError(
            "DEM origin must have at least three values."
        )

    target_crs = dem_meta[
        "crs"
    ]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True,
    )

    poses = dict(
        raw_poses
    )

    images_in = raw_poses.get(
        "images",
        []
    )

    if not isinstance(images_in, list) or not images_in:
        raise RuntimeError(
            "Raw pose JSON has no non-empty images list."
        )

    images_out = []

    for idx, image in enumerate(
        images_in
    ):
        if not isinstance(image, dict):
            raise RuntimeError(
                f"Pose image entry {idx} is not a dict."
            )

        item = dict(
            image
        )

        if "location" not in item:
            for key in [
                "lat",
                "lng",
                "alt",
            ]:
                if key not in item:
                    raise RuntimeError(
                        f"Pose image {idx} missing '{key}' required for location reconstruction."
                    )

            easting, northing = transformer.transform(
                float(
                    item[
                        "lng"
                    ]
                ),
                float(
                    item[
                        "lat"
                    ]
                ),
            )

            item[
                "location"
            ] = [
                round(
                    float(
                        easting
                        -
                        origin[
                            0
                        ]
                    ),
                    6,
                ),
                round(
                    float(
                        northing
                        -
                        origin[
                            1
                        ]
                    ),
                    6,
                ),
                round(
                    float(
                        item[
                            "alt"
                        ]
                        -
                        origin[
                            2
                        ]
                    ),
                    6,
                ),
            ]

        if "rotation" not in item:
            item[
                "rotation"
            ] = [
                float(
                    item.get(
                        "pitch",
                        0.0
                    )
                ),
                float(
                    item.get(
                        "roll",
                        0.0
                    )
                ),
                float(
                    item.get(
                        "yaw",
                        0.0
                    )
                ),
            ]

        images_out.append(
            item
        )

    poses[
        "images"
    ] = images_out

    return poses


def prepare_pose_geometry(raw_pose_path):
    """
    Return a read-only in-memory pose structure containing 'location'.
    First prefer an existing enriched working file; otherwise reconstruct
    exactly from raw lat/lng/alt + DEM origin.
    """
    enriched_path, enriched = discover_enriched_pose_json()

    if enriched is not None:
        return (
            enriched,
            {
                "mode":
                    "existing_enriched_pose_json",

                "source":
                    str(
                        enriched_path
                    ),
            }
        )

    raw = json.loads(
        raw_pose_path.read_text(
            encoding="utf-8"
        )
    )

    dem_json = find_dem_json()

    enriched = enrich_raw_pose_in_memory(
        raw,
        dem_json
    )

    return (
        enriched,
        {
            "mode":
                "in_memory_reconstruction_from_raw_pose_and_dem_origin",

            "raw_pose_source":
                str(
                    raw_pose_path
                ),

            "dem_json":
                str(
                    dem_json
                ),

            "note":
                "No pose JSON was modified on disk.",
        }
    )


def make_camera(module, poses, correction, frame_idx):
    rot, trans = normalize_correction_dict(
        correction,
        frame_idx
    )

    Vector3 = module.Vector3

    cor_rotation_eulers = Vector3(
        [
            float(rot.get("x", 0.0)),
            float(rot.get("y", 0.0)),
            float(rot.get("z", 0.0)),
        ],
        dtype="f4",
    )

    cor_translation = Vector3(
        [
            float(trans.get("x", 0.0)),
            float(trans.get("y", 0.0)),
            float(trans.get("z", 0.0)),
        ],
        dtype="f4",
    )

    # -----------------------------------------------------------------
    # IMPORTANT BAMBI compatibility patch:
    #
    # Official get_camera_for_frame() reads cur_frame_data['fovy'][0]
    # BEFORE it applies overrule_fov. Flight-6 original matched poses do
    # not contain a 'fovy' key. To preserve the original file as READ-ONLY,
    # create a temporary in-memory view for this one frame and inject only
    # the locked BAMBI reference-pipeline assumption FOVY=50 degrees.
    #
    # No source JSON is modified on disk.
    # -----------------------------------------------------------------
    idx = int(frame_idx)

    if "images" not in poses:
        raise RuntimeError("Matched poses JSON has no 'images' list.")

    if idx < 0 or idx >= len(poses["images"]):
        raise IndexError(
            f"frame_idx {idx} outside pose list length {len(poses['images'])}"
        )

    poses_for_camera = dict(poses)
    images_for_camera = list(poses["images"])
    frame_data = dict(images_for_camera[idx])

    original_fovy_present = "fovy" in frame_data

    if not original_fovy_present:
        frame_data["fovy"] = [float(FOVY_DEG)]
    else:
        # Normalize unusual scalar/empty forms only in-memory so the
        # official function can safely execute its initial ['fovy'][0].
        fovy_value = frame_data["fovy"]

        if isinstance(fovy_value, (int, float)):
            frame_data["fovy"] = [float(fovy_value)]

        elif isinstance(fovy_value, (list, tuple)):
            if len(fovy_value) == 0:
                frame_data["fovy"] = [float(FOVY_DEG)]

        else:
            frame_data["fovy"] = [float(FOVY_DEG)]

    images_for_camera[idx] = frame_data
    poses_for_camera["images"] = images_for_camera

    camera = module.get_camera_for_frame(
        poses_for_camera,
        idx,
        cor_rotation_eulers,
        cor_translation,
        overrule_fov=FOVY_DEG,
    )

    return camera


def camera_inventory(camera):
    rows = []

    cls = type(camera)

    for name in sorted(set(dir(camera))):
        if name.startswith("__"):
            continue

        try:
            value = getattr(camera, name)
        except Exception as e:
            rows.append({
                "name": name,
                "kind": "attribute_error",
                "signature": None,
                "preview": f"{type(e).__name__}: {e}",
            })
            continue

        if callable(value):
            try:
                sig = str(inspect.signature(value))
            except Exception:
                sig = None

            rows.append({
                "name": name,
                "kind": "callable",
                "signature": sig,
                "preview": None,
            })
        else:
            preview = repr(value)
            rows.append({
                "name": name,
                "kind": "attribute",
                "signature": None,
                "preview": preview[:500],
            })

    return pd.DataFrame(rows)


def camera_class_source(camera):
    cls = type(camera)

    try:
        source = inspect.getsource(cls)
    except Exception as e:
        source = (
            f"# Could not inspect source for {cls}: "
            f"{type(e).__name__}: {e}\n"
        )

    try:
        source_file = inspect.getsourcefile(cls)
    except Exception:
        source_file = None

    return source_file, source


def parse_mot_gt(path):
    df = pd.read_csv(
        path,
        header=None
    )

    if df.shape[1] < 6:
        raise RuntimeError(
            "MOT GT has fewer than six columns."
        )

    out = pd.DataFrame({
        "frame_id":
            pd.to_numeric(
                df.iloc[:, 0],
                errors="coerce"
            ),

        "track_id":
            pd.to_numeric(
                df.iloc[:, 1],
                errors="coerce"
            ),

        "bb_left":
            pd.to_numeric(
                df.iloc[:, 2],
                errors="coerce"
            ),

        "bb_top":
            pd.to_numeric(
                df.iloc[:, 3],
                errors="coerce"
            ),

        "bb_width":
            pd.to_numeric(
                df.iloc[:, 4],
                errors="coerce"
            ),

        "bb_height":
            pd.to_numeric(
                df.iloc[:, 5],
                errors="coerce"
            ),
    })

    out = out.dropna().copy()

    out["frame_id"] = out["frame_id"].astype(int)
    out["track_id"] = out["track_id"].astype(int)
    out["bbox_center_x_px"] = (
        out["bb_left"]
        +
        0.5 * out["bb_width"]
    )
    out["bbox_center_y_px"] = (
        out["bb_top"]
        +
        0.5 * out["bb_height"]
    )

    return out


def dem_audit(dem_tif, world_df, inference_paths):
    try:
        import rasterio
    except Exception as e:
        return {
            "status":
                f"RASTERIO_UNAVAILABLE:{type(e).__name__}:{e}"
        }, pd.DataFrame(), pd.DataFrame()

    rows = []
    pred_rows = []

    with rasterio.open(dem_tif) as src:
        dem_info = {
            "status":
                "OPENED",

            "crs":
                str(src.crs),

            "bounds":
                {
                    "left": float(src.bounds.left),
                    "bottom": float(src.bounds.bottom),
                    "right": float(src.bounds.right),
                    "top": float(src.bounds.top),
                },

            "width":
                int(src.width),

            "height":
                int(src.height),

            "nodata":
                None if src.nodata is None else float(src.nodata),
        }

        # Existing georeferenced world points vs DEM height.
        coords = list(
            zip(
                world_df["utm_x_center"].astype(float),
                world_df["utm_y_center"].astype(float),
            )
        )

        samples = np.asarray(
            [
                float(v[0])
                for v in src.sample(coords)
            ],
            dtype=float
        )

        z_world = world_df[
            "utm_z_center"
        ].astype(float).to_numpy()

        valid = np.isfinite(samples)

        if src.nodata is not None:
            valid &= (
                samples
                !=
                float(src.nodata)
            )

        diff = (
            z_world[valid]
            -
            samples[valid]
        )

        rows.append({
            "kind":
                "world_center_vs_dem",

            "n_total":
                int(len(samples)),

            "n_valid_dem":
                int(valid.sum()),

            "valid_fraction":
                float(valid.mean())
                if len(valid)
                else None,

            "median_abs_z_diff_m":
                float(
                    np.median(
                        np.abs(diff)
                    )
                )
                if valid.any()
                else None,

            "p95_abs_z_diff_m":
                float(
                    np.quantile(
                        np.abs(diff),
                        0.95
                    )
                )
                if valid.any()
                else None,

            "max_abs_z_diff_m":
                float(
                    np.max(
                        np.abs(diff)
                    )
                )
                if valid.any()
                else None,
        })

        # Frozen predictions: prove a deterministic DEM z can be sampled.
        for task, npz_path in inference_paths.items():
            if not npz_path.exists():
                continue

            arrays = np.load(
                npz_path,
                allow_pickle=False
            )

            if "prediction_abs_xy" not in arrays.files:
                continue

            pred_xy = np.asarray(
                arrays["prediction_abs_xy"],
                dtype=float
            )

            flat = pred_xy.reshape(
                -1,
                2
            )

            inside = (
                (flat[:, 0] >= src.bounds.left)
                &
                (flat[:, 0] <= src.bounds.right)
                &
                (flat[:, 1] >= src.bounds.bottom)
                &
                (flat[:, 1] <= src.bounds.top)
            )

            z = np.full(
                len(flat),
                np.nan,
                dtype=float
            )

            if inside.any():
                sampled = np.asarray(
                    [
                        float(v[0])
                        for v in src.sample(
                            [
                                tuple(x)
                                for x in flat[inside]
                            ]
                        )
                    ],
                    dtype=float
                )

                z[inside] = sampled

            finite = np.isfinite(z)

            if src.nodata is not None:
                finite &= (
                    z
                    !=
                    float(src.nodata)
                )

            pred_rows.append({
                "task":
                    task,

                "prediction_points":
                    int(len(flat)),

                "inside_dem_bounds":
                    int(inside.sum()),

                "inside_fraction":
                    float(inside.mean()),

                "finite_dem_z":
                    int(finite.sum()),

                "finite_dem_z_fraction":
                    float(finite.mean()),
            })

    return (
        dem_info,
        pd.DataFrame(rows),
        pd.DataFrame(pred_rows)
    )


def try_camera_projection_methods(
    camera,
    point_local_xyz
):
    """
    Best-effort, read-only API discovery. We only call obvious projection-like
    methods with one point argument. Failures are recorded, never hidden.
    """
    rows = []

    candidate_names = [
        name
        for name in dir(camera)
        if any(
            token in name.lower()
            for token in [
                "project",
                "pixel",
                "screen",
                "image",
            ]
        )
        and not name.startswith("__")
    ]

    point = np.asarray(
        point_local_xyz,
        dtype=float
    )

    for name in sorted(set(candidate_names)):
        try:
            fn = getattr(camera, name)
        except Exception:
            continue

        if not callable(fn):
            continue

        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None

        # Only try methods that plausibly accept one user argument.
        can_try = True

        if sig is not None:
            params = [
                p
                for p in sig.parameters.values()
                if p.kind
                in
                (
                    p.POSITIONAL_ONLY,
                    p.POSITIONAL_OR_KEYWORD,
                )
                and p.default
                is
                p.empty
            ]

            # Bound method should expose only required user args.
            if len(params) > 1:
                can_try = False

        if not can_try:
            rows.append({
                "method": name,
                "signature": str(sig),
                "attempted": False,
                "status": "SKIPPED_SIGNATURE",
                "result": None,
            })
            continue

        success = False
        result_repr = None
        error = None

        for candidate in [
            point,
            point.tolist(),
            point.reshape(1, 3),
        ]:
            try:
                result = fn(candidate)
                result_repr = repr(result)[:1000]
                success = True
                break
            except Exception as e:
                error = (
                    f"{type(e).__name__}: {e}"
                )

        rows.append({
            "method": name,
            "signature": str(sig),
            "attempted": True,
            "status":
                "SUCCESS"
                if success
                else f"FAILED:{error}",

            "result":
                result_repr,
        })

    return pd.DataFrame(rows)


def main():
    print(
        "\nSTEP 6D-K3C8-V3 — NUMERIC ANCHOR-REPROJECTION + PREDICTION-Z PROVENANCE AUDIT"
    )

    print(
        "Read-only. NO model retraining, BAMBI recalibration, geofence evaluation, or Step7 alarm is performed.\n"
    )

    require(WORLD_CSV)

    gt_path = discover_one(
        "6_gt.txt",
        prefer_tokens=("bambi_downloads",)
    )

    poses_path = discover_one(
        "6_matched_poses.json",
        prefer_tokens=("bambi_downloads",)
    )

    correction_path = discover_one(
        "6_correction.json",
        prefer_tokens=("bambi_downloads",)
    )

    dem_tif = discover_one(
        "6_matched_dem.tif"
    )

    module, module_path = load_official_module()

    poses, pose_geometry_provenance = prepare_pose_geometry(
        poses_path
    )

    correction = json.loads(
        correction_path.read_text(
            encoding="utf-8"
        )
    )

    world = pd.read_csv(
        WORLD_CSV
    )

    mot = parse_mot_gt(
        gt_path
    )

    needed_world = {
        "frame_id",
        "track_id",
        "utm_x_center",
        "utm_y_center",
        "utm_z_center",
    }

    missing = (
        needed_world
        -
        set(world.columns)
    )

    if missing:
        raise RuntimeError(
            f"World CSV missing columns: {sorted(missing)}"
        )

    merged = world.merge(
        mot[
            [
                "frame_id",
                "track_id",
                "bbox_center_x_px",
                "bbox_center_y_px",
            ]
        ],
        on=[
            "frame_id",
            "track_id",
        ],
        how="inner",
    )

    if len(merged) == 0:
        raise RuntimeError(
            "No world/MOT frame+track rows merged."
        )

    representative = (
        merged
        .sort_values(
            [
                "frame_id",
                "track_id",
            ]
        )
        .iloc[
            len(merged) // 2
        ]
    )

    rep_frame = int(
        representative[
            "frame_id"
        ]
    )

    camera = make_camera(
        module,
        poses,
        correction,
        rep_frame
    )

    cam_inventory = camera_inventory(
        camera
    )

    cam_inventory.to_csv(
        OUT
        / "TABLE_step6dk3c8_camera_api_inventory.csv",
        index=False
    )

    camera_source_file, camera_source = (
        camera_class_source(
            camera
        )
    )

    (
        OUT
        / "STEP6D_K3C8_CAMERA_CLASS_SOURCE.txt"
    ).write_text(
        (
            f"CAMERA TYPE: {type(camera)}\n"
            f"SOURCE FILE: {camera_source_file}\n\n"
            f"{camera_source}"
        ),
        encoding="utf-8"
    )

    # ------------------------------------------------------------
    # Determine DEM/local origin from sidecar JSON if available.
    # ------------------------------------------------------------
    dem_json = None

    candidates = sorted(
        p
        for p in DOWNLOADS.rglob(
            "6_matched_dem.json"
        )
        if p.is_file()
    )

    if len(candidates) == 1:
        dem_json = candidates[0]
    elif len(candidates) > 1:
        preferred = [
            p
            for p in candidates
            if "dem" in str(p.parent).lower()
        ]

        if len(preferred) == 1:
            dem_json = preferred[0]

    dem_meta = None
    dem_origin = None

    if dem_json is not None:
        try:
            dem_meta = json.loads(
                dem_json.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(
                dem_meta,
                dict
            ) and "origin" in dem_meta:
                origin = np.asarray(
                    dem_meta[
                        "origin"
                    ],
                    dtype=float
                )

                if origin.size >= 3:
                    dem_origin = origin[:3]
        except Exception:
            dem_meta = None
            dem_origin = None

    rep_world_abs = np.array(
        [
            float(
                representative[
                    "utm_x_center"
                ]
            ),
            float(
                representative[
                    "utm_y_center"
                ]
            ),
            float(
                representative[
                    "utm_z_center"
                ]
            ),
        ],
        dtype=float,
    )

    if dem_origin is not None:
        rep_world_local = (
            rep_world_abs
            -
            dem_origin
        )
    else:
        rep_world_local = (
            rep_world_abs.copy()
        )

    projection_attempts = (
        try_camera_projection_methods(
            camera,
            rep_world_local
        )
    )

    projection_attempts.to_csv(
        OUT
        / "TABLE_step6dk3c8_camera_projection_attempts.csv",
        index=False
    )

    # ------------------------------------------------------------
    # DEM z provenance audit.
    # ------------------------------------------------------------
    inference_paths = {
        "2s_to_2s":
            INFERENCE_DIR
            / "2s_to_2s_BAMBI_Flight6_first_locked_inference.npz",

        "2s_to_5s":
            INFERENCE_DIR
            / "2s_to_5s_BAMBI_Flight6_first_locked_inference.npz",
    }

    dem_info, dem_world_check, pred_dem_check = (
        dem_audit(
            dem_tif,
            world,
            inference_paths
        )
    )

    dem_world_check.to_csv(
        OUT
        / "TABLE_step6dk3c8_world_z_vs_dem.csv",
        index=False
    )

    pred_dem_check.to_csv(
        OUT
        / "TABLE_step6dk3c8_prediction_dem_z_availability.csv",
        index=False
    )

    (
        OUT
        / "STEP6D_K3C8_DEM_INFO.json"
    ).write_text(
        json.dumps(
            dem_info,
            indent=2
        ),
        encoding="utf-8"
    )

    # ------------------------------------------------------------
    # Print concise evidence.
    # ------------------------------------------------------------
    print("OFFICIAL GEOREFERENCE SOURCE:")
    print(module_path)

    print("\nPOSE GEOMETRY PROVENANCE:")
    print(
        json.dumps(
            pose_geometry_provenance,
            indent=2
        )
    )

    print("\nCAMERA TYPE:")
    print(type(camera))

    print("\nREPRESENTATIVE FRAME/TRACK:")
    print(
        json.dumps(
            {
                "frame_id":
                    rep_frame,

                "track_id":
                    int(
                        representative[
                            "track_id"
                        ]
                    ),

                "bbox_center_px":
                    [
                        float(
                            representative[
                                "bbox_center_x_px"
                            ]
                        ),
                        float(
                            representative[
                                "bbox_center_y_px"
                            ]
                        ),
                    ],

                "world_abs_xyz":
                    rep_world_abs.tolist(),

                "dem_origin":
                    (
                        dem_origin.tolist()
                        if dem_origin
                        is not None
                        else None
                    ),

                "world_local_xyz":
                    rep_world_local.tolist(),
            },
            indent=2
        )
    )

    print("\nCAMERA API INVENTORY:")
    print(
        cam_inventory.to_string(
            index=False
        )
    )

    print("\nPROJECTION METHOD ATTEMPTS:")
    print(
        projection_attempts.to_string(
            index=False
        )
        if len(
            projection_attempts
        )
        else "NONE"
    )

    print("\nDEM INFO:")
    print(
        json.dumps(
            dem_info,
            indent=2
        )
    )

    print("\nWORLD-Z vs DEM:")
    print(
        dem_world_check.to_string(
            index=False
        )
        if len(
            dem_world_check
        )
        else "NONE"
    )

    print("\nFROZEN PREDICTION DEM-Z AVAILABILITY:")
    print(
        pred_dem_check.to_string(
            index=False
        )
        if len(
            pred_dem_check
        )
        else "NONE"
    )

    # ------------------------------------------------------------
    # Conservative decision.
    # ------------------------------------------------------------
    successful_projection_api = bool(
        len(
            projection_attempts
        )
        and
        (
            projection_attempts[
                "status"
            ]
            ==
            "SUCCESS"
        ).any()
    )

    dem_z_available = bool(
        len(
            pred_dem_check
        )
        and
        (
            pd.to_numeric(
                pred_dem_check[
                    "finite_dem_z_fraction"
                ],
                errors="coerce"
            )
            >=
            0.999
        ).all()
    )

    world_dem_consistency = False

    if len(
        dem_world_check
    ):
        row = dem_world_check.iloc[0]

        # This is deliberately a loose geometry sanity threshold, not model tuning.
        # Stored bbox-world center and DEM-at-center need not be identical.
        val = row.get(
            "p95_abs_z_diff_m"
        )

        if pd.notna(
            val
        ):
            world_dem_consistency = bool(
                float(val)
                <=
                10.0
            )

    if (
        successful_projection_api
        and
        dem_z_available
        and
        world_dem_consistency
    ):
        status = (
            "BAMBI_NUMERIC_REPROJECTION_PRIMITIVES_AND_PREDICTION_Z_SOURCE_AVAILABLE_"
            "READY_FOR_EXACT_PIXEL_ROUNDTRIP_TEST"
        )

    elif (
        dem_z_available
        and
        world_dem_consistency
    ):
        status = (
            "PREDICTION_Z_SOURCE_AVAILABLE_BUT_EXACT_WORLD_TO_IMAGE_API_NOT_YET_RESOLVED_"
            "DO_NOT_RUN_STEP7"
        )

    else:
        status = (
            "BAMBI_ANCHOR_REPROJECTION_OR_PREDICTION_Z_PROVENANCE_INCOMPLETE_"
            "DO_NOT_RUN_STEP7"
        )

    summary = {
        "step":
            "STEP_6D_K3C8_V3_NUMERIC_ANCHOR_REPROJECTION_AND_PREDICTION_Z_PROVENANCE_AUDIT",

        "official_zero_based_mapping_already_proven":
            True,

        "pose_geometry_provenance":
            pose_geometry_provenance,

        "fovy_degrees_assumption":
            FOVY_DEG,

        "fovy_status":
            (
                "BAMBI reference-pipeline assumption/default; "
                "not independently measured calibration"
            ),

        "camera_type":
            str(
                type(
                    camera
                )
            ),

        "camera_source_file":
            camera_source_file,

        "dem_tif":
            str(
                dem_tif
            ),

        "dem_json":
            (
                str(
                    dem_json
                )
                if dem_json
                is not None
                else None
            ),

        "dem_origin":
            (
                dem_origin.tolist()
                if dem_origin
                is not None
                else None
            ),

        "successful_projection_api":
            successful_projection_api,

        "prediction_dem_z_available":
            dem_z_available,

        "world_dem_consistency":
            world_dem_consistency,

        "status":
            status,

        "important_nonclaim":
            (
                "This step does not run Step18/Step7 and does not validate "
                "a real village/farm boundary."
            ),
    }

    (
        OUT
        / "STEP6D_K3C8_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSTATUS:")
    print(status)

    print("\nSTEP 6D-K3C8 COMPLETE")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
