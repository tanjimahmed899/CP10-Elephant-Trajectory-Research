from pathlib import Path
import hashlib
import importlib
import importlib.util
import inspect
import json
import math

import numpy as np
import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "step6d_uncertainty_holdout"
    / "bambi_flight6"
    / "step6dk_georef"
)

WORLD_CSV = BASE / "step6dk_valid_world_trajectory_rows.csv"

OUT = (
    BASE
    / "k3_operational_warning"
    / "camera_matrix_pixel_roundtrip"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

FOVY_DEG = 50.0
IMAGE_W = 1024.0
IMAGE_H = 1024.0
SAMPLE_ROWS = 8


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
        p
        for p in DOWNLOADS.rglob(filename)
        if p.is_file()
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {DOWNLOADS}"
        )

    if prefer_tokens:
        preferred = [
            p
            for p in matches
            if all(
                token.lower()
                in
                str(p).lower()
                for token in prefer_tokens
            )
        ]

        if len(preferred) == 1:
            return preferred[0]

    if len(matches) == 1:
        return matches[0]

    preferred = [
        p
        for p in matches
        if any(
            token in str(p).lower()
            for token in [
                "bambi_downloads",
                "step6dk_georef",
                "dem",
            ]
        )
    ]

    if len(preferred) == 1:
        return preferred[0]

    raise RuntimeError(
        f"Multiple unresolved candidates for {filename}:\n"
        + "\n".join(
            str(p)
            for p in matches
        )
    )


def discover_enriched_pose():
    candidates = sorted(
        p
        for p in DOWNLOADS.rglob("6_matched_poses.json")
        if p.is_file()
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

        images = raw.get(
            "images",
            []
        )

        if (
            isinstance(images, list)
            and
            images
            and
            isinstance(images[0], dict)
            and
            "location" in images[0]
        ):
            hits.append(
                (
                    p,
                    raw
                )
            )

    if not hits:
        raise RuntimeError(
            "No enriched 6_matched_poses.json with 'location' found."
        )

    preferred = [
        x
        for x in hits
        if (
            "step6dk_georef"
            in
            str(
                x[0]
            ).lower()
            and
            "correction"
            in
            str(
                x[0]
            ).lower()
        )
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(hits) == 1:
        return hits[0]

    raise RuntimeError(
        "Multiple enriched pose files found:\n"
        + "\n".join(
            str(p)
            for p, _
            in hits
        )
    )


def load_official_module():
    spec = importlib.util.find_spec(
        "bambi.georeference_mot17_tracks"
    )

    if spec is None or spec.origin is None:
        raise RuntimeError(
            "bambi.georeference_mot17_tracks not installed."
        )

    module = importlib.import_module(
        "bambi.georeference_mot17_tracks"
    )

    return (
        module,
        Path(
            spec.origin
        )
    )


def normalize_correction_dict(
    raw,
    frame_idx
):
    rotation = raw.get(
        "rotation",
        {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
    )

    translation = raw.get(
        "translation",
        {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
    )

    additions = (
        raw.get(
            "fine_corrections"
        )
        or
        raw.get(
            "additional_corrections"
        )
        or
        raw.get(
            "additional corrections"
        )
        or
        []
    )

    chosen_rot = dict(
        rotation
    )

    chosen_trans = dict(
        translation
    )

    for item in additions:
        start = (
            item.get(
                "start frame"
            )
            if
            "start frame"
            in
            item
            else
            item.get(
                "start_frame"
            )
        )

        end = (
            item.get(
                "end frame"
            )
            if
            "end frame"
            in
            item
            else
            item.get(
                "end_frame"
            )
        )

        if (
            start is not None
            and
            end is not None
            and
            int(
                start
            )
            <
            int(
                frame_idx
            )
            <
            int(
                end
            )
        ):
            chosen_rot = dict(
                item.get(
                    "rotation",
                    chosen_rot
                )
            )

            chosen_trans = dict(
                item.get(
                    "translation",
                    chosen_trans
                )
            )

            break

    return (
        chosen_rot,
        chosen_trans
    )


def ensure_fovy_in_memory(
    poses,
    frame_idx
):
    poses_temp = dict(
        poses
    )

    images = list(
        poses[
            "images"
        ]
    )

    item = dict(
        images[
            int(
                frame_idx
            )
        ]
    )

    fovy = item.get(
        "fovy"
    )

    if isinstance(
        fovy,
        (
            list,
            tuple,
        )
    ) and len(
        fovy
    ):
        pass

    elif isinstance(
        fovy,
        (
            int,
            float,
        )
    ):
        item[
            "fovy"
        ] = [
            float(
                fovy
            )
        ]

    else:
        item[
            "fovy"
        ] = [
            FOVY_DEG
        ]

    images[
        int(
            frame_idx
        )
    ] = item

    poses_temp[
        "images"
    ] = images

    return poses_temp


def make_camera(
    module,
    poses,
    correction,
    frame_idx
):
    rot, trans = (
        normalize_correction_dict(
            correction,
            frame_idx
        )
    )

    Vector3 = module.Vector3

    cor_rotation_eulers = Vector3(
        [
            float(
                rot.get(
                    "x",
                    0.0
                )
            ),
            float(
                rot.get(
                    "y",
                    0.0
                )
            ),
            float(
                rot.get(
                    "z",
                    0.0
                )
            ),
        ],
        dtype="f4",
    )

    cor_translation = Vector3(
        [
            float(
                trans.get(
                    "x",
                    0.0
                )
            ),
            float(
                trans.get(
                    "y",
                    0.0
                )
            ),
            float(
                trans.get(
                    "z",
                    0.0
                )
            ),
        ],
        dtype="f4",
    )

    poses_temp = ensure_fovy_in_memory(
        poses,
        frame_idx
    )

    return module.get_camera_for_frame(
        poses_temp,
        int(
            frame_idx
        ),
        cor_rotation_eulers,
        cor_translation,
        overrule_fov=FOVY_DEG,
    )


def parse_mot_gt(path):
    df = pd.read_csv(
        path,
        header=None
    )

    out = pd.DataFrame({
        "frame_id":
            pd.to_numeric(
                df.iloc[
                    :,
                    0
                ],
                errors="coerce"
            ),

        "track_id":
            pd.to_numeric(
                df.iloc[
                    :,
                    1
                ],
                errors="coerce"
            ),

        "left":
            pd.to_numeric(
                df.iloc[
                    :,
                    2
                ],
                errors="coerce"
            ),

        "top":
            pd.to_numeric(
                df.iloc[
                    :,
                    3
                ],
                errors="coerce"
            ),

        "width":
            pd.to_numeric(
                df.iloc[
                    :,
                    4
                ],
                errors="coerce"
            ),

        "height":
            pd.to_numeric(
                df.iloc[
                    :,
                    5
                ],
                errors="coerce"
            ),
    })

    out = out.dropna().copy()

    out[
        "frame_id"
    ] = out[
        "frame_id"
    ].astype(
        int
    )

    out[
        "track_id"
    ] = out[
        "track_id"
    ].astype(
        int
    )

    return out


def get_render_dependencies(
    module,
    dem_glb
):
    required_names = [
        "read_gltf",
        "process_render_data",
        "get_aabb",
        "label_to_world_coordinates",
    ]

    missing = [
        name
        for name in required_names
        if not hasattr(
            module,
            name
        )
    ]

    if missing:
        raise RuntimeError(
            f"Official BAMBI module lacks required functions: {missing}"
        )

    read_gltf = getattr(
        module,
        "read_gltf"
    )

    process_render_data = getattr(
        module,
        "process_render_data"
    )

    get_aabb = getattr(
        module,
        "get_aabb"
    )

    label_to_world_coordinates = getattr(
        module,
        "label_to_world_coordinates"
    )

    mesh_data, texture_data = read_gltf(
        str(
            dem_glb
        )
    )

    # Match official pipeline.
    try:
        Trimesh = getattr(
            module,
            "Trimesh"
        )
    except Exception:
        from trimesh import Trimesh

    tri_mesh = Trimesh(
        vertices=mesh_data.vertices,
        faces=mesh_data.indices
    )

    mesh_data, texture_data = process_render_data(
        mesh_data,
        texture_data
    )

    mesh_aabb = get_aabb(
        mesh_data.vertices
    )

    return {
        "mesh_data":
            mesh_data,

        "texture_data":
            texture_data,

        "tri_mesh":
            tri_mesh,

        "mesh_aabb":
            mesh_aabb,

        "label_to_world_coordinates":
            label_to_world_coordinates,
    }


def call_label_to_world(
    fn,
    corners,
    deps,
    camera
):
    sig = inspect.signature(
        fn
    )

    params = list(
        sig.parameters.keys()
    )

    # Official function signature used in installed BAMBI package is expected
    # to match the source already audited. We still dispatch by parameter names.
    kwargs = {}

    mapping = {
        "labels":
            corners,

        "label":
            corners,

        "bbox_corners":
            corners,

        "coordinates":
            corners,

        "mesh_data":
            deps[
                "mesh_data"
            ],

        "tri_mesh":
            deps[
                "tri_mesh"
            ],

        "mesh":
            deps[
                "tri_mesh"
            ],

        "mesh_aabb":
            deps[
                "mesh_aabb"
            ],

        "aabb":
            deps[
                "mesh_aabb"
            ],

        "camera":
            camera,
    }

    can_keyword = all(
        p in mapping
        for p in params
    )

    if can_keyword:
        for p in params:
            kwargs[
                p
            ] = mapping[
                p
            ]

        return fn(
            **kwargs
        )

    # Known installed-source positional form from the audited georeference script.
    return fn(
        corners,
        deps[
            "mesh_data"
        ],
        deps[
            "tri_mesh"
        ],
        deps[
            "mesh_aabb"
        ],
        camera,
    )


def matrix_candidates(
    view,
    proj,
    world_xyz
):
    p_col = np.array(
        [
            world_xyz[
                0
            ],
            world_xyz[
                1
            ],
            world_xyz[
                2
            ],
            1.0,
        ],
        dtype=float,
    )

    p_row = p_col.copy()

    candidates = []

    # Column-vector conventions.
    candidates.append(
        (
            "col_PV",
            proj
            @
            view
            @
            p_col,
        )
    )

    candidates.append(
        (
            "col_VP",
            view
            @
            proj
            @
            p_col,
        )
    )

    candidates.append(
        (
            "col_PT_VT",
            proj.T
            @
            view.T
            @
            p_col,
        )
    )

    candidates.append(
        (
            "col_VT_PT",
            view.T
            @
            proj.T
            @
            p_col,
        )
    )

    # Row-vector conventions.
    candidates.append(
        (
            "row_VP",
            p_row
            @
            view
            @
            proj,
        )
    )

    candidates.append(
        (
            "row_PV",
            p_row
            @
            proj
            @
            view,
        )
    )

    candidates.append(
        (
            "row_VT_PT",
            p_row
            @
            view.T
            @
            proj.T,
        )
    )

    candidates.append(
        (
            "row_PT_VT",
            p_row
            @
            proj.T
            @
            view.T,
        )
    )

    return candidates


def clip_to_pixel_variants(
    clip
):
    clip = np.asarray(
        clip,
        dtype=float
    ).reshape(
        4
    )

    if (
        not np.isfinite(
            clip
        ).all()
        or
        abs(
            clip[
                3
            ]
        )
        <
        1e-12
    ):
        return []

    ndc = (
        clip[
            :3
        ]
        /
        clip[
            3
        ]
    )

    x = float(
        ndc[
            0
        ]
    )

    y = float(
        ndc[
            1
        ]
    )

    variants = [
        (
            "top_left",
            (
                x
                +
                1.0
            )
            *
            0.5
            *
            IMAGE_W,
            (
                1.0
                -
                y
            )
            *
            0.5
            *
            IMAGE_H,
        ),

        (
            "bottom_left",
            (
                x
                +
                1.0
            )
            *
            0.5
            *
            IMAGE_W,
            (
                y
                +
                1.0
            )
            *
            0.5
            *
            IMAGE_H,
        ),

        (
            "top_left_xflip",
            (
                1.0
                -
                x
            )
            *
            0.5
            *
            IMAGE_W,
            (
                1.0
                -
                y
            )
            *
            0.5
            *
            IMAGE_H,
        ),

        (
            "bottom_left_xflip",
            (
                1.0
                -
                x
            )
            *
            0.5
            *
            IMAGE_W,
            (
                y
                +
                1.0
            )
            *
            0.5
            *
            IMAGE_H,
        ),
    ]

    out = []

    for name, u, v in variants:
        out.append(
            (
                name,
                float(
                    u
                ),
                float(
                    v
                ),
                x,
                y,
                float(
                    ndc[
                        2
                    ]
                ),
            )
        )

    return out


def main():
    print(
        "\nSTEP 6D-K3C9 — EXACT CAMERA MATRIX PIXEL ROUND-TRIP AUDIT"
    )

    print(
        "Read-only geometry/API audit. NO model tuning, recalibration, geofence evaluation, or Step7 alarm is performed.\n"
    )

    require(
        WORLD_CSV
    )

    gt_path = discover_one(
        "6_gt.txt",
        prefer_tokens=(
            "bambi_downloads",
        )
    )

    correction_path = discover_one(
        "6_correction.json",
        prefer_tokens=(
            "bambi_downloads",
        )
    )

    dem_glb = discover_one(
        "6_matched_dem.glb"
    )

    pose_path, poses = discover_enriched_pose()

    correction = json.loads(
        correction_path.read_text(
            encoding="utf-8"
        )
    )

    module, module_path = (
        load_official_module()
    )

    world = pd.read_csv(
        WORLD_CSV
    )

    mot = parse_mot_gt(
        gt_path
    )

    valid_keys = world[
        [
            "frame_id",
            "track_id",
        ]
    ].drop_duplicates()

    merged = valid_keys.merge(
        mot,
        on=[
            "frame_id",
            "track_id",
        ],
        how="inner",
    )

    if len(
        merged
    ) == 0:
        raise RuntimeError(
            "No valid frame/track MOT rows overlap world trajectory."
        )

    merged = merged.sort_values(
        [
            "frame_id",
            "track_id",
        ]
    ).reset_index(
        drop=True
    )

    if (
        len(
            merged
        )
        <=
        SAMPLE_ROWS
    ):
        samples = merged.copy()

    else:
        indices = np.linspace(
            0,
            len(
                merged
            )
            -
            1,
            SAMPLE_ROWS,
            dtype=int,
        )

        samples = merged.iloc[
            indices
        ].copy()

    deps = get_render_dependencies(
        module,
        dem_glb
    )

    roundtrip_rows = []
    worldpoint_rows = []

    first_camera = None

    for sample_no, row in samples.iterrows():
        frame_id = int(
            row[
                "frame_id"
            ]
        )

        track_id = int(
            row[
                "track_id"
            ]
        )

        camera = make_camera(
            module,
            poses,
            correction,
            frame_id
        )

        if first_camera is None:
            first_camera = camera

        left = float(
            row[
                "left"
            ]
        )

        top = float(
            row[
                "top"
            ]
        )

        width = float(
            row[
                "width"
            ]
        )

        height = float(
            row[
                "height"
            ]
        )

        x1 = left
        y1 = top
        x2 = (
            left
            +
            width
        )
        y2 = (
            top
            +
            height
        )

        target_pixels = np.asarray(
            [
                [
                    x1,
                    y1,
                ],
                [
                    x2,
                    y1,
                ],
                [
                    x2,
                    y2,
                ],
                [
                    x1,
                    y2,
                ],
            ],
            dtype=float,
        )

        corners_flat = [
            x1,
            y1,
            x2,
            y1,
            x2,
            y2,
            x1,
            y2,
        ]

        world_points = call_label_to_world(
            deps[
                "label_to_world_coordinates"
            ],
            corners_flat,
            deps,
            camera
        )

        world_points = np.asarray(
            world_points,
            dtype=float
        )

        if (
            world_points.ndim
            !=
            2
            or
            world_points.shape[
                1
            ]
            <
            3
        ):
            raise RuntimeError(
                f"Unexpected label_to_world_coordinates shape: {world_points.shape}"
            )

        if world_points.shape[
            0
        ] != 4:
            raise RuntimeError(
                "Expected exactly four returned world points for four bbox corners, "
                f"got {world_points.shape[0]}"
            )

        world_points = world_points[
            :,
            :3
        ]

        view = np.asarray(
            camera.get_view(
                dtype="f8"
            ),
            dtype=float,
        ).reshape(
            4,
            4
        )

        proj = np.asarray(
            camera.get_proj(
                dtype="f8"
            ),
            dtype=float,
        ).reshape(
            4,
            4
        )

        for corner_index in range(
            4
        ):
            point = world_points[
                corner_index
            ]

            target = target_pixels[
                corner_index
            ]

            worldpoint_rows.append({
                "frame_id":
                    frame_id,

                "track_id":
                    track_id,

                "corner_index":
                    corner_index,

                "target_u_px":
                    float(
                        target[
                            0
                        ]
                    ),

                "target_v_px":
                    float(
                        target[
                            1
                        ]
                    ),

                "world_x":
                    float(
                        point[
                            0
                        ]
                    ),

                "world_y":
                    float(
                        point[
                            1
                        ]
                    ),

                "world_z":
                    float(
                        point[
                            2
                        ]
                    ),
            })

            for matrix_name, clip in matrix_candidates(
                view,
                proj,
                point
            ):
                variants = (
                    clip_to_pixel_variants(
                        clip
                    )
                )

                for (
                    pixel_variant,
                    u,
                    v,
                    ndc_x,
                    ndc_y,
                    ndc_z,
                ) in variants:
                    err = math.hypot(
                        u
                        -
                        target[
                            0
                        ],
                        v
                        -
                        target[
                            1
                        ],
                    )

                    roundtrip_rows.append({
                        "frame_id":
                            frame_id,

                        "track_id":
                            track_id,

                        "corner_index":
                            corner_index,

                        "matrix_convention":
                            matrix_name,

                        "pixel_mapping":
                            pixel_variant,

                        "target_u_px":
                            float(
                                target[
                                    0
                                ]
                            ),

                        "target_v_px":
                            float(
                                target[
                                    1
                                ]
                            ),

                        "projected_u_px":
                            u,

                        "projected_v_px":
                            v,

                        "reprojection_error_px":
                            float(
                                err
                            ),

                        "ndc_x":
                            ndc_x,

                        "ndc_y":
                            ndc_y,

                        "ndc_z":
                            ndc_z,
                    })

    roundtrip = pd.DataFrame(
        roundtrip_rows
    )

    worldpoints = pd.DataFrame(
        worldpoint_rows
    )

    roundtrip.to_csv(
        OUT
        / "TABLE_step6dk3c9_all_projection_conventions.csv",
        index=False
    )

    worldpoints.to_csv(
        OUT
        / "TABLE_step6dk3c9_official_corner_worldpoints.csv",
        index=False
    )

    summary = (
        roundtrip
        .groupby(
            [
                "matrix_convention",
                "pixel_mapping",
            ],
            as_index=False
        )
        .agg(
            n=(
                "reprojection_error_px",
                "size"
            ),

            median_error_px=(
                "reprojection_error_px",
                "median"
            ),

            mean_error_px=(
                "reprojection_error_px",
                "mean"
            ),

            p95_error_px=(
                "reprojection_error_px",
                lambda x:
                    float(
                        np.quantile(
                            np.asarray(
                                x,
                                dtype=float
                            ),
                            0.95
                        )
                    )
            ),

            max_error_px=(
                "reprojection_error_px",
                "max"
            ),
        )
        .sort_values(
            [
                "median_error_px",
                "p95_error_px",
                "max_error_px",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        OUT
        / "TABLE_step6dk3c9_projection_convention_summary.csv",
        index=False
    )

    best = summary.iloc[
        0
    ]

    # Source provenance.
    camera_cls = type(
        first_camera
    )

    try:
        camera_source = inspect.getsource(
            camera_cls
        )
    except Exception as e:
        camera_source = (
            f"# Camera source unavailable: {type(e).__name__}: {e}\n"
        )

    try:
        camera_source_file = inspect.getsourcefile(
            camera_cls
        )
    except Exception:
        camera_source_file = None

    try:
        label_source = inspect.getsource(
            deps[
                "label_to_world_coordinates"
            ]
        )
    except Exception as e:
        label_source = (
            f"# label_to_world_coordinates source unavailable: "
            f"{type(e).__name__}: {e}\n"
        )

    (
        OUT
        / "STEP6D_K3C9_GEOMETRY_SOURCE_BUNDLE.txt"
    ).write_text(
        (
            f"OFFICIAL BAMBI GEOREFERENCE SOURCE: {module_path}\n"
            f"SHA256: {sha256_file(module_path)}\n\n"
            f"CAMERA SOURCE FILE: {camera_source_file}\n\n"
            "===== CAMERA CLASS =====\n"
            f"{camera_source}\n\n"
            "===== label_to_world_coordinates =====\n"
            f"{label_source}\n"
        ),
        encoding="utf-8"
    )

    print(
        "OFFICIAL BAMBI GEOREFERENCE SOURCE:"
    )
    print(
        module_path
    )

    print(
        "\nENRICHED POSE SOURCE:"
    )
    print(
        pose_path
    )

    print(
        "\nDEM GLB:"
    )
    print(
        dem_glb
    )

    print(
        "\nSAMPLED VALID BBOX INSTANCES:"
    )
    print(
        samples[
            [
                "frame_id",
                "track_id",
                "left",
                "top",
                "width",
                "height",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nTOP PROJECTION CONVENTIONS:"
    )
    print(
        summary.head(
            12
        ).to_string(
            index=False
        )
    )

    exact_roundtrip = bool(
        float(
            best[
                "median_error_px"
            ]
        )
        <=
        1.0
        and
        float(
            best[
                "p95_error_px"
            ]
        )
        <=
        2.0
    )

    strong_roundtrip = bool(
        float(
            best[
                "median_error_px"
            ]
        )
        <=
        2.0
        and
        float(
            best[
                "p95_error_px"
            ]
        )
        <=
        5.0
    )

    second_gap = None

    if len(
        summary
    ) >= 2:
        second_gap = float(
            summary.iloc[
                1
            ][
                "median_error_px"
            ]
            -
            best[
                "median_error_px"
            ]
        )

    if exact_roundtrip:
        status = (
            "EXACT_CAMERA_MATRIX_PIXEL_ROUNDTRIP_CONFIRMED_"
            "READY_FOR_BAMBI_ANCHOR_NORMALIZED_TRAJECTORY_CONSTRUCTION"
        )

    elif strong_roundtrip:
        status = (
            "CAMERA_MATRIX_PIXEL_ROUNDTRIP_STRONGLY_SUPPORTED_"
            "PENDING_TOLERANCE_REVIEW_BEFORE_STEP7"
        )

    else:
        status = (
            "CAMERA_MATRIX_PIXEL_ROUNDTRIP_NOT_CONFIRMED_"
            "DO_NOT_RUN_STEP7"
        )

    result = {
        "step":
            "STEP_6D_K3C9_EXACT_CAMERA_MATRIX_PIXEL_ROUNDTRIP_AUDIT",

        "fovy_degrees":
            FOVY_DEG,

        "fovy_status":
            "assumed/default BAMBI reference-pipeline value; not independently measured calibration",

        "sample_bbox_instances":
            int(
                len(
                    samples
                )
            ),

        "roundtrip_corner_points":
            int(
                len(
                    worldpoints
                )
            ),

        "best_matrix_convention":
            str(
                best[
                    "matrix_convention"
                ]
            ),

        "best_pixel_mapping":
            str(
                best[
                    "pixel_mapping"
                ]
            ),

        "best_median_error_px":
            float(
                best[
                    "median_error_px"
                ]
            ),

        "best_p95_error_px":
            float(
                best[
                    "p95_error_px"
                ]
            ),

        "best_max_error_px":
            float(
                best[
                    "max_error_px"
                ]
            ),

        "second_best_median_gap_px":
            second_gap,

        "status":
            status,

        "nonclaims": [
            "No Step18 or Step7 BAMBI evaluation was run.",
            "No model or q value was tuned.",
            "No field boundary was validated.",
            "The Flight-6 50 degree FOVY remains an assumption/default from the BAMBI reference pipeline.",
        ],
    }

    (
        OUT
        / "STEP6D_K3C9_SUMMARY.json"
    ).write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nBEST CONVENTION:"
    )
    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print(
        "\nSTATUS:"
    )
    print(
        status
    )

    print(
        "\nSTEP 6D-K3C9 COMPLETE"
    )
    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
