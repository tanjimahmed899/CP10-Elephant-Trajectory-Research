from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json
import re
import statistics
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DOWNLOADS = Path(
    r"C:\Users\HP\Downloads"
)

ZIP_FILES = [

    DOWNLOADS /
    "DJI_20240119152552_0006_V.zip",

    DOWNLOADS /
    "DJI_20240626143832_0004_V.zip",

    DOWNLOADS /
    "DJI_20260227084907_0004_V.zip"
]

OUT = DOWNLOADS / "wildbox_batch_audit"

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


def frame_number(name):

    stem = Path(name).stem

    nums = re.findall(
        r"\d+",
        stem
    )

    if not nums:
        return None

    return int(
        nums[-1]
    )


def find_segment(name):

    parts = Path(name).parts

    for p in parts:

        if re.fullmatch(
            r"seg\d+",
            p,
            flags=re.I
        ):

            return p

    return "unknown"


def compact_structure(obj, depth=0):

    if depth > 3:

        return type(obj).__name__


    if isinstance(obj, dict):

        result = {}

        for k, v in list(
            obj.items()
        )[:10]:

            result[str(k)] = (
                compact_structure(
                    v,
                    depth + 1
                )
            )

        return result


    if isinstance(obj, list):

        if len(obj) == 0:

            return {
                "type": "list",
                "len": 0
            }

        return {

            "type":
                "list",

            "len":
                len(obj),

            "sample":
                compact_structure(
                    obj[0],
                    depth + 1
                )
        }


    return type(obj).__name__


def count_tracks(data):

    # Most likely WildBox structure
    if isinstance(
        data,
        dict
    ):

        if "tracks" in data:

            tracks = data[
                "tracks"
            ]

            if isinstance(
                tracks,
                dict
            ):

                return len(
                    tracks
                )

            if isinstance(
                tracks,
                list
            ):

                return len(
                    tracks
                )


    return None


def inspect_track_lengths(data):

    rows = []

    if not isinstance(
        data,
        dict
    ):

        return rows


    tracks = data.get(
        "tracks"
    )


    if isinstance(
        tracks,
        dict
    ):

        for track_id, track in (
            tracks.items()
        ):

            length = None


            if isinstance(
                track,
                list
            ):

                length = len(
                    track
                )


            elif isinstance(
                track,
                dict
            ):

                # common possibilities

                for key in [

                    "frames",
                    "trajectory",
                    "positions",
                    "detections",
                    "observations"

                ]:

                    if (

                        key in track

                        and

                        isinstance(
                            track[key],
                            list
                        )

                    ):

                        length = len(
                            track[key]
                        )

                        break


                if length is None:

                    # Sometimes frame IDs themselves
                    # are dictionary keys.

                    numeric_keys = [

                        k

                        for k
                        in track.keys()

                        if str(k).isdigit()

                    ]


                    if numeric_keys:

                        length = len(
                            numeric_keys
                        )


            rows.append({

                "track_id":
                    str(track_id),

                "observations":
                    length
            })


    elif isinstance(
        tracks,
        list
    ):

        for i, track in enumerate(
            tracks
        ):

            rows.append({

                "track_id":
                    str(i),

                "observations":
                    (
                        len(track)
                        if isinstance(
                            track,
                            list
                        )
                        else None
                    )
            })


    return rows


# ============================================================
# PROCESS
# ============================================================

segment_rows = []

tracking_rows = []

camera_rows = []

track_length_rows = []

label_rows = []

schema_samples = {}


for zip_path in ZIP_FILES:


    print(
        "\nProcessing:",
        zip_path.name
    )


    if not zip_path.exists():

        print(
            "MISSING:",
            zip_path
        )

        continue


    video_name = (
        zip_path.stem
    )


    with ZipFile(
        zip_path,
        "r"
    ) as z:


        names = [

            x.filename

            for x
            in z.infolist()

            if not x.is_dir()

        ]


        segments = sorted(
            set(
                find_segment(
                    name
                )

                for name
                in names

                if find_segment(
                    name
                )
                !=
                "unknown"
            )
        )


        schema_samples[
            video_name
        ] = {}


        for seg in segments:


            seg_names = [

                name

                for name
                in names

                if (
                    f"/{seg}/"
                    in name.replace(
                        "\\",
                        "/"
                    )
                )
            ]


            # ====================================
            # IMAGE / FRAME ANALYSIS
            # ====================================

            images = [

                name

                for name
                in seg_names

                if (
                    Path(name)
                    .suffix
                    .lower()
                    in IMAGE_EXT
                )
            ]


            frame_nums = sorted(

                f

                for f in [

                    frame_number(
                        name
                    )

                    for name
                    in images

                ]

                if f is not None
            )


            frame_steps = [

                frame_nums[i]
                -
                frame_nums[i - 1]

                for i in range(
                    1,
                    len(frame_nums)
                )

            ]


            median_step = (

                statistics.median(
                    frame_steps
                )

                if frame_steps

                else None
            )


            # If source footage was 30fps,
            # step=3 would imply 10Hz.
            #
            # We record the observed step only.
            # No FPS assumption is locked here.


            # ====================================
            # LABEL FILES
            # ====================================

            txt_files = [

                name

                for name
                in seg_names

                if (
                    Path(name)
                    .suffix
                    .lower()
                    ==
                    ".txt"
                )
            ]


            classes = Counter()

            label_count = 0


            for txt in txt_files:

                try:

                    text = (

                        z.read(
                            txt
                        )

                        .decode(
                            "utf-8",
                            errors="ignore"
                        )

                    )

                except Exception:

                    continue


                for line in (
                    text.splitlines()
                ):

                    line = (
                        line.strip()
                    )

                    if not line:
                        continue


                    parts = (
                        line.split()
                    )


                    if not parts:
                        continue


                    classes[
                        parts[0]
                    ] += 1

                    label_count += 1


                    # KITTI-like row
                    if len(parts) >= 16:

                        try:

                            label_rows.append({

                                "video":
                                    video_name,

                                "segment":
                                    seg,

                                "class":
                                    parts[0],

                                "bbox_x1":
                                    float(
                                        parts[4]
                                    ),

                                "bbox_y1":
                                    float(
                                        parts[5]
                                    ),

                                "bbox_x2":
                                    float(
                                        parts[6]
                                    ),

                                "bbox_y2":
                                    float(
                                        parts[7]
                                    ),

                                "loc_x":
                                    float(
                                        parts[11]
                                    ),

                                "loc_y":
                                    float(
                                        parts[12]
                                    ),

                                "loc_z":
                                    float(
                                        parts[13]
                                    )
                            })

                        except Exception:

                            pass


            segment_rows.append({

                "video":
                    video_name,

                "segment":
                    seg,

                "image_frames":
                    len(
                        images
                    ),

                "first_frame_number":
                    (
                        min(
                            frame_nums
                        )
                        if frame_nums
                        else None
                    ),

                "last_frame_number":
                    (
                        max(
                            frame_nums
                        )
                        if frame_nums
                        else None
                    ),

                "median_frame_step":
                    median_step,

                "label_files":
                    len(
                        txt_files
                    ),

                "label_rows":
                    label_count,

                "classes":
                    " | ".join(
                        sorted(
                            classes.keys()
                        )
                    )
            })


            # ====================================
            # TRACKING SUMMARY
            # ====================================

            tracking_files = [

                name

                for name
                in seg_names

                if (
                    Path(name)
                    .name
                    .lower()
                    ==
                    "tracking_summary.json"
                )
            ]


            for tf in tracking_files:

                try:

                    data = json.loads(

                        z.read(
                            tf
                        ).decode(
                            "utf-8"
                        )
                    )


                    top_keys = (

                        list(
                            data.keys()
                        )

                        if isinstance(
                            data,
                            dict
                        )

                        else []
                    )


                    n_tracks = (
                        count_tracks(
                            data
                        )
                    )


                    tracking_rows.append({

                        "video":
                            video_name,

                        "segment":
                            seg,

                        "archive_path":
                            tf,

                        "top_level_type":
                            type(
                                data
                            ).__name__,

                        "top_level_keys":
                            " | ".join(
                                map(
                                    str,
                                    top_keys
                                )
                            ),

                        "tracks_detected":
                            n_tracks
                    })


                    for tr in (
                        inspect_track_lengths(
                            data
                        )
                    ):

                        tr.update({

                            "video":
                                video_name,

                            "segment":
                                seg
                        })

                        track_length_rows.append(
                            tr
                        )


                    schema_samples[
                        video_name
                    ][
                        f"{seg}_tracking"
                    ] = (
                        compact_structure(
                            data
                        )
                    )


                except Exception as e:

                    tracking_rows.append({

                        "video":
                            video_name,

                        "segment":
                            seg,

                        "archive_path":
                            tf,

                        "error":
                            str(e)
                    })


            # ====================================
            # CAMERA JSON
            # ====================================

            camera_files = [

                name

                for name
                in seg_names

                if (
                    Path(name)
                    .name
                    .lower()
                    ==
                    "cameras.json"
                )
            ]


            for cf in camera_files:

                try:

                    data = json.loads(

                        z.read(
                            cf
                        ).decode(
                            "utf-8"
                        )
                    )


                    top_keys = (

                        list(
                            data.keys()
                        )

                        if isinstance(
                            data,
                            dict
                        )

                        else []
                    )


                    camera_rows.append({

                        "video":
                            video_name,

                        "segment":
                            seg,

                        "archive_path":
                            cf,

                        "top_level_type":
                            type(
                                data
                            ).__name__,

                        "top_level_keys":
                            " | ".join(
                                map(
                                    str,
                                    top_keys
                                )
                            ),

                        "top_level_length":
                            (
                                len(data)
                                if hasattr(
                                    data,
                                    "__len__"
                                )
                                else None
                            )
                    })


                    schema_samples[
                        video_name
                    ][
                        f"{seg}_camera"
                    ] = (
                        compact_structure(
                            data
                        )
                    )


                except Exception as e:

                    camera_rows.append({

                        "video":
                            video_name,

                        "segment":
                            seg,

                        "archive_path":
                            cf,

                        "error":
                            str(e)
                    })


# ============================================================
# SAVE
# ============================================================

segment_df = pd.DataFrame(
    segment_rows
)

tracking_df = pd.DataFrame(
    tracking_rows
)

camera_df = pd.DataFrame(
    camera_rows
)

tracks_df = pd.DataFrame(
    track_length_rows
)

labels_df = pd.DataFrame(
    label_rows
)


segment_df.to_csv(

    OUT /
    "segment_summary.csv",

    index=False
)


tracking_df.to_csv(

    OUT /
    "tracking_schema.csv",

    index=False
)


camera_df.to_csv(

    OUT /
    "camera_schema.csv",

    index=False
)


tracks_df.to_csv(

    OUT /
    "track_lengths.csv",

    index=False
)


labels_df.to_csv(

    OUT /
    "label_3d_summary.csv",

    index=False
)


with open(

    OUT /
    "schema_samples.json",

    "w",
    encoding="utf-8"

) as f:

    json.dump(

        schema_samples,

        f,

        indent=2,

        ensure_ascii=False
    )


# ============================================================
# TEXT SUMMARY
# ============================================================

with open(

    OUT /
    "SUMMARY.txt",

    "w",
    encoding="utf-8"

) as f:


    f.write(
        "WILDBOX 3-VIDEO BATCH AUDIT\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )


    for video in (
        segment_df[
            "video"
        ].unique()
    ):


        sub = segment_df[

            segment_df[
                "video"
            ]
            ==
            video

        ]


        f.write(
            video
            +
            "\n"
        )


        f.write(

            f"Segments: "
            f"{len(sub)}\n"

        )


        f.write(

            f"Image frames: "
            f"{int(sub['image_frames'].sum())}\n"

        )


        f.write(

            f"Label rows: "
            f"{int(sub['label_rows'].sum())}\n"

        )


        f.write(

            "Median frame steps: "
            +
            ", ".join(

                str(x)

                for x in
                sub[
                    "median_frame_step"
                ].tolist()

            )
            +
            "\n"
        )


        tsub = tracking_df[

            tracking_df[
                "video"
            ]
            ==
            video

        ]


        if len(tsub):

            f.write(

                "Tracks detected per segment: "
                +
                ", ".join(

                    str(x)

                    for x in
                    tsub[
                        "tracks_detected"
                    ].tolist()

                )
                +
                "\n"
            )


        f.write(
            "\n"
        )


print(
    "\n======================================"
)

print(
    "WILDBOX BATCH AUDIT COMPLETE"
)

print(
    "======================================"
)

print(
    "\nOutput:"
)

print(
    OUT
)