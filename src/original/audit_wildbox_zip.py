from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
import json
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ZIP_FILE = Path(
    r"C:\Users\HP\Downloads\DJI_20240119152552_0006_V.zip"
)

OUT = Path(
    r"C:\Users\HP\Downloads\wildbox_zip_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

IMAGE_EXT = {
    ".jpg",
    ".jpeg",
    ".png"
}


def frame_number(filename):

    stem = Path(
        filename
    ).stem

    numbers = re.findall(
        r"\d+",
        stem
    )

    if not numbers:
        return None

    return int(
        numbers[-1]
    )


def recursive_key_values(
    obj,
    wanted_keys
):

    found = []

    if isinstance(
        obj,
        dict
    ):

        for key, value in (
            obj.items()
        ):

            if str(
                key
            ).lower() in wanted_keys:

                found.append(
                    value
                )

            found.extend(
                recursive_key_values(
                    value,
                    wanted_keys
                )
            )


    elif isinstance(
        obj,
        list
    ):

        for item in obj:

            found.extend(
                recursive_key_values(
                    item,
                    wanted_keys
                )
            )

    return found


# ============================================================
# OPEN ZIP
# ============================================================

if not ZIP_FILE.exists():

    raise FileNotFoundError(
        ZIP_FILE
    )


with ZipFile(
    ZIP_FILE,
    "r"
) as z:


    infos = [

        info

        for info
        in z.infolist()

        if not info.is_dir()

    ]


    names = [

        info.filename

        for info
        in infos

    ]


    # ========================================================
    # BASIC INVENTORY
    # ========================================================

    extension_counts = Counter()


    for name in names:

        suffix = (
            Path(name)
            .suffix
            .lower()
        )

        extension_counts[
            suffix
        ] += 1


    image_files = [

        name

        for name in names

        if (
            Path(name)
            .suffix
            .lower()
            in IMAGE_EXT
        )

    ]


    json_files = [

        name

        for name in names

        if (
            Path(name)
            .suffix
            .lower()
            ==
            ".json"
        )

    ]


    txt_files = [

        name

        for name in names

        if (
            Path(name)
            .suffix
            .lower()
            ==
            ".txt"
        )

    ]


    print(
        "Archive files:",
        len(names)
    )

    print(
        "Images:",
        len(image_files)
    )

    print(
        "JSON:",
        len(json_files)
    )

    print(
        "TXT:",
        len(txt_files)
    )


    # ========================================================
    # FRAME FOLDER AUDIT
    # ========================================================

    frame_groups = defaultdict(
        list
    )


    for name in image_files:

        parent = str(
            Path(name).parent
        )

        number = frame_number(
            name
        )


        if number is not None:

            frame_groups[
                parent
            ].append(
                number
            )


    frame_rows = []


    for folder, frames in (
        frame_groups.items()
    ):


        frames = sorted(
            set(frames)
        )


        if not frames:
            continue


        first = min(
            frames
        )

        last = max(
            frames
        )

        span = (
            last
            -
            first
            +
            1
        )


        frame_rows.append({

            "folder":
                folder,

            "observed_frames":
                len(frames),

            "first_frame":
                first,

            "last_frame":
                last,

            "frame_span":
                span,

            "missing_inside_span":
                span
                -
                len(frames),

            "coverage_pct":

                100
                *
                len(frames)
                /
                span

        })


    frame_df = pd.DataFrame(
        frame_rows
    )


    frame_df.to_csv(

        OUT
        /
        "frame_folder_summary.csv",

        index=False

    )


    # ========================================================
    # FIND METADATA FILES
    # ========================================================

    tracking_files = [

        name

        for name in json_files

        if (
            "tracking_summary"
            in name.lower()
        )

    ]


    camera_files = [

        name

        for name in json_files

        if (
            "camera"
            in Path(name)
            .name
            .lower()
        )

    ]


    print(
        "Tracking summaries:",
        len(tracking_files)
    )

    print(
        "Camera JSON files:",
        len(camera_files)
    )


    # ========================================================
    # EXTRACT TRACKING / CAMERA METADATA
    # ========================================================

    metadata_manifest = []


    for name in (

        tracking_files
        +
        camera_files

    ):


        safe_name = (
            name
            .replace(
                "/",
                "__"
            )
            .replace(
                "\\",
                "__"
            )
        )


        output_path = (
            OUT
            /
            safe_name
        )


        with z.open(
            name
        ) as src:

            raw = src.read()


        with open(
            output_path,
            "wb"
        ) as dst:

            dst.write(
                raw
            )


        metadata_manifest.append({

            "archive_path":
                name,

            "extracted_name":
                safe_name,

            "size_bytes":
                len(raw)

        })


    pd.DataFrame(
        metadata_manifest
    ).to_csv(

        OUT
        /
        "metadata_manifest.csv",

        index=False

    )


    # ========================================================
    # TRACKING SUMMARY AUDIT
    # ========================================================

    tracking_rows = []


    possible_track_ids = set()

    fps_candidates = []


    for name in tracking_files:


        try:

            with z.open(
                name
            ) as f:

                data = json.loads(
                    f.read()
                    .decode(
                        "utf-8"
                    )
                )


            ids = recursive_key_values(

                data,

                {
                    "track_id",
                    "tracking_id",
                    "object_id",
                    "group_id"
                }

            )


            for value in ids:

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float
                    )
                ):

                    possible_track_ids.add(
                        str(value)
                    )


            fps_values = (
                recursive_key_values(

                    data,

                    {
                        "fps",
                        "frame_rate",
                        "framerate"
                    }

                )
            )


            for value in fps_values:

                if isinstance(
                    value,
                    (
                        int,
                        float
                    )
                ):

                    fps_candidates.append(
                        value
                    )


            tracking_rows.append({

                "file":
                    name,

                "top_level_type":
                    type(
                        data
                    ).__name__,

                "possible_track_ids":
                    len(
                        set(
                            str(x)
                            for x
                            in ids
                            if isinstance(
                                x,
                                (
                                    str,
                                    int,
                                    float
                                )
                            )
                        )
                    )

            })


        except Exception as e:

            tracking_rows.append({

                "file":
                    name,

                "error":
                    str(e)

            })


    pd.DataFrame(
        tracking_rows
    ).to_csv(

        OUT
        /
        "tracking_summary_audit.csv",

        index=False

    )


    # ========================================================
    # KITTI / LABEL TXT AUDIT
    # ========================================================

    label_files = [

        name

        for name in txt_files

        if (

            "label"
            in name.lower()

            or

            "kitti"
            in name.lower()

        )

    ]


    class_counter = Counter()

    nonempty_label_files = 0

    total_label_rows = 0

    sample_lines = []


    for label_file in (
        label_files
    ):


        try:

            with z.open(
                label_file
            ) as f:

                text = (
                    f.read()
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                )


            lines = [

                line.strip()

                for line
                in text.splitlines()

                if line.strip()

            ]


            if lines:

                nonempty_label_files += 1


            total_label_rows += len(
                lines
            )


            for line in lines:

                parts = line.split()

                if parts:

                    class_counter[
                        parts[0]
                    ] += 1


            if (
                len(sample_lines)
                <
                50
            ):

                for line in (
                    lines[:5]
                ):

                    sample_lines.append(

                        f"{label_file} :: "
                        f"{line}"

                    )


        except Exception:

            pass


    label_class_df = pd.DataFrame([

        {

            "class_token":
                cls,

            "rows":
                count

        }

        for cls, count
        in class_counter.items()

    ])


    if len(
        label_class_df
    ):

        label_class_df = (
            label_class_df
            .sort_values(
                "rows",
                ascending=False
            )
        )


    label_class_df.to_csv(

        OUT
        /
        "label_class_summary.csv",

        index=False

    )


    with open(

        OUT
        /
        "sample_label_lines.txt",

        "w",
        encoding="utf-8"

    ) as f:


        for line in (
            sample_lines
        ):

            f.write(
                line
                +
                "\n"
            )


    # ========================================================
    # ZIP STRUCTURE SAMPLE
    # ========================================================

    with open(

        OUT
        /
        "archive_structure_sample.txt",

        "w",
        encoding="utf-8"

    ) as f:


        for name in (
            names[:300]
        ):

            f.write(
                name
                +
                "\n"
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    with open(

        OUT
        /
        "SUMMARY.txt",

        "w",
        encoding="utf-8"

    ) as f:


        f.write(
            "WILDBOX ELEPHANT VIDEO ZIP AUDIT\n"
        )

        f.write(
            "=" * 70
            +
            "\n\n"
        )


        f.write(
            f"Archive: "
            f"{ZIP_FILE.name}\n"
        )


        f.write(
            f"Archive files: "
            f"{len(names)}\n"
        )


        f.write(
            f"Image files: "
            f"{len(image_files)}\n"
        )


        f.write(
            f"JSON files: "
            f"{len(json_files)}\n"
        )


        f.write(
            f"TXT files: "
            f"{len(txt_files)}\n\n"
        )


        f.write(
            f"Frame folders: "
            f"{len(frame_df)}\n"
        )


        if len(
            frame_df
        ):

            f.write(
                f"Total observed image frames: "
                f"{int(frame_df['observed_frames'].sum())}\n"
            )


        f.write(
            "\n"
        )


        f.write(
            f"Tracking-summary files: "
            f"{len(tracking_files)}\n"
        )


        f.write(
            f"Possible track IDs discovered "
            f"from tracking summaries: "
            f"{len(possible_track_ids)}\n"
        )


        f.write(
            f"Camera metadata JSON files: "
            f"{len(camera_files)}\n"
        )


        f.write(
            f"FPS candidates found: "
            f"{sorted(set(fps_candidates))}\n\n"
        )


        f.write(
            f"Label files detected: "
            f"{len(label_files)}\n"
        )


        f.write(
            f"Non-empty label files: "
            f"{nonempty_label_files}\n"
        )


        f.write(
            f"Total label rows: "
            f"{total_label_rows}\n"
        )


        f.write(
            "\nLabel class/token counts:\n"
        )


        for cls, count in (
            class_counter.most_common()
        ):

            f.write(

                f"{cls}: "
                f"{count}\n"

            )


print(
    "\n===================================="
)

print(
    "WILDBOX ZIP AUDIT COMPLETE"
)

print(
    "===================================="
)

print(
    "\nResults saved to:"
)

print(
    OUT
)