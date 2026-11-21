from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import math


# ============================================================
# PATHS
# ============================================================

TRAIN_ROOT = Path(
    r"C:\Users\HP\Downloads\conservation_drones_train_real\TrainReal"
)

TEST_ROOT = Path(
    r"C:\Users\HP\Downloads\conservation_drones_test_real\TestReal"
)

OUT = Path(
    r"C:\Users\HP\Downloads\ego_motion_full"
)

OUT.mkdir(exist_ok=True)


# ============================================================
# BIRDSAI ANNOTATION FORMAT
# ============================================================

COLS = [
    "frame",
    "object_id",
    "x",
    "y",
    "w",
    "h",
    "class",
    "species",
    "occlusion",
    "noise"
]


# ============================================================
# ELEPHANT-CONTAINING SEQUENCES
# ============================================================

TRAIN_SEQUENCES = [

    "0000000010_0000000000",

    "0000000054_0000000000",

    "0000000056_0000000000",

    "0000000351_0000000000",

    "0000000367_0000000000"
]


TEST_SEQUENCES = [

    "0000000055_0000000000",

    "0000000058_0000000000",

    "0000000349_0000000000",

    "0000000352_0000000000",

    "0000000353_0000000000",

    "0000000357_0000000000"
]


# ============================================================
# PROCESSING SETTINGS
# ============================================================

SCALE = 0.5

MASK_MARGIN = 10


# Reliability threshold

MIN_FEATURES = 40

MIN_INLIER_RATIO = 0.55

MIN_SCALE = 0.95

MAX_SCALE = 1.05

MAX_ROTATION = 5.0


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(path):

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        return None


    img = cv2.resize(

        img,

        None,

        fx=SCALE,

        fy=SCALE
    )


    # Thermal contrast enhancement

    img = cv2.equalizeHist(img)


    return img


# ============================================================
# BACKGROUND MASK
# ============================================================

def make_background_mask(
    image_shape,
    annotations,
    frame_number
):

    height, width = image_shape

    mask = np.ones(

        (height, width),

        dtype=np.uint8

    ) * 255


    frame_annotations = annotations[

        annotations["frame"]
        ==
        frame_number

    ]


    for _, row in (
        frame_annotations.iterrows()
    ):

        x = int(
            row["x"] * SCALE
        )

        y = int(
            row["y"] * SCALE
        )

        w = int(
            row["w"] * SCALE
        )

        h = int(
            row["h"] * SCALE
        )


        margin = int(
            MASK_MARGIN * SCALE
        )


        x1 = max(
            0,
            x - margin
        )

        y1 = max(
            0,
            y - margin
        )

        x2 = min(
            width - 1,
            x + w + margin
        )

        y2 = min(
            height - 1,
            y + h + margin
        )


        cv2.rectangle(

            mask,

            (x1, y1),

            (x2, y2),

            0,

            -1
        )


    return mask


# ============================================================
# MOTION ESTIMATION
# ============================================================

def estimate_motion(
    prev_img,
    curr_img,
    background_mask
):

    points_old = (
        cv2.goodFeaturesToTrack(

            prev_img,

            maxCorners=700,

            qualityLevel=0.01,

            minDistance=8,

            blockSize=7,

            mask=background_mask
        )
    )


    if (
        points_old is None
        or
        len(points_old)
        < MIN_FEATURES
    ):

        return None


    points_new, status, error = (

        cv2.calcOpticalFlowPyrLK(

            prev_img,

            curr_img,

            points_old,

            None,

            winSize=(21, 21),

            maxLevel=3,

            criteria=(

                cv2.TERM_CRITERIA_EPS
                |
                cv2.TERM_CRITERIA_COUNT,

                30,

                0.01
            )
        )
    )


    if points_new is None:

        return None


    status = status.flatten()


    good_old = points_old[
        status == 1
    ]


    good_new = points_new[
        status == 1
    ]


    if len(good_old) < MIN_FEATURES:

        return None


    # current frame -> previous frame

    M, inliers = (

        cv2.estimateAffinePartial2D(

            good_new,

            good_old,

            method=cv2.RANSAC,

            ransacReprojThreshold=3.0,

            maxIters=3000,

            confidence=0.99,

            refineIters=10
        )
    )


    if M is None:

        return None


    if inliers is None:

        inlier_ratio = 0

    else:

        inlier_ratio = float(

            inliers.sum()
            /
            len(inliers)

        )


    dx = (
        M[0, 2]
        /
        SCALE
    )


    dy = (
        M[1, 2]
        /
        SCALE
    )


    rotation = math.degrees(

        math.atan2(

            M[1, 0],

            M[0, 0]

        )
    )


    scale_value = math.sqrt(

        M[0, 0] ** 2

        +

        M[1, 0] ** 2
    )


    # ===================================
    # QUALITY CHECK
    # ===================================

    reliable = (

        len(good_old)
        >= MIN_FEATURES

        and

        inlier_ratio
        >= MIN_INLIER_RATIO

        and

        MIN_SCALE
        <= scale_value
        <= MAX_SCALE

        and

        abs(rotation)
        <= MAX_ROTATION
    )


    return {

        "features":
            len(good_old),

        "inlier_ratio":
            inlier_ratio,

        "dx_pixels":
            dx,

        "dy_pixels":
            dy,

        "rotation_deg":
            rotation,

        "scale":
            scale_value,

        "reliable":
            int(reliable)
    }


# ============================================================
# PROCESS ONE SEQUENCE
# ============================================================

def process_sequence(
    root,
    split,
    sequence
):

    image_folder = (

        root
        /
        "images"
        /
        sequence
    )


    annotation_file = (

        root
        /
        "annotations"
        /
        f"{sequence}.csv"
    )


    if not image_folder.exists():

        print(
            "Image folder missing:",
            image_folder
        )

        return None


    if not annotation_file.exists():

        print(
            "Annotation missing:",
            annotation_file
        )

        return None


    annotations = pd.read_csv(

        annotation_file,

        header=None,

        names=COLS
    )


    images = sorted([

        p

        for p
        in image_folder.iterdir()

        if p.suffix.lower()
        in [
            ".jpg",
            ".jpeg",
            ".png"
        ]

    ])


    print(
        "\nProcessing",
        split,
        sequence,
        "frames:",
        len(images)
    )


    results = []


    for i in range(
        len(images) - 1
    ):


        prev_img = prepare_image(
            images[i]
        )


        curr_img = prepare_image(
            images[i + 1]
        )


        if (
            prev_img is None

            or

            curr_img is None
        ):

            continue


        background_mask = (

            make_background_mask(

                prev_img.shape,

                annotations,

                i
            )

        )


        result = estimate_motion(

            prev_img,

            curr_img,

            background_mask
        )


        if result is None:

            results.append({

                "split":
                    split,

                "sequence":
                    sequence,

                "pair":
                    i,

                "frame_a":
                    images[i].name,

                "frame_b":
                    images[i+1].name,

                "estimated":
                    0,

                "reliable":
                    0
            })


        else:

            row = {

                "split":
                    split,

                "sequence":
                    sequence,

                "pair":
                    i,

                "frame_a":
                    images[i].name,

                "frame_b":
                    images[i+1].name,

                "estimated":
                    1
            }


            row.update(
                result
            )


            results.append(
                row
            )


        if (
            (i + 1)
            % 500
            == 0
        ):

            print(

                i + 1,

                "/",

                len(images) - 1
            )


    df = pd.DataFrame(
        results
    )


    df.to_csv(

        OUT
        /
        f"{split}_{sequence}_motion.csv",

        index=False
    )


    return df


# ============================================================
# RUN DATASET
# ============================================================

all_results = []


for sequence in TRAIN_SEQUENCES:

    result = process_sequence(

        TRAIN_ROOT,

        "train",

        sequence
    )

    if result is not None:

        all_results.append(
            result
        )


for sequence in TEST_SEQUENCES:

    result = process_sequence(

        TEST_ROOT,

        "test",

        sequence
    )

    if result is not None:

        all_results.append(
            result
        )


# ============================================================
# COMBINE
# ============================================================

full = pd.concat(

    all_results,

    ignore_index=True
)


full.to_csv(

    OUT
    /
    "all_ego_motion.csv",

    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = []


for (
    split,
    sequence
), group in full.groupby(
    [
        "split",
        "sequence"
    ]
):


    total_pairs = len(group)


    estimated = int(

        (
            group["estimated"]
            ==
            1
        ).sum()

    )


    reliable = int(

        (
            group["reliable"]
            ==
            1
        ).sum()

    )


    valid = group[

        group["estimated"]
        ==
        1

    ]


    reliable_group = group[

        group["reliable"]
        ==
        1

    ]


    row = {

        "split":
            split,

        "sequence":
            sequence,

        "total_pairs":
            total_pairs,

        "estimated_pairs":
            estimated,

        "reliable_pairs":
            reliable,

        "estimate_rate_pct":

            (
                100
                *
                estimated
                /
                total_pairs
            )
            if total_pairs
            else 0,

        "reliable_rate_pct":

            (
                100
                *
                reliable
                /
                total_pairs
            )
            if total_pairs
            else 0
    }


    if len(valid) > 0:

        row[

            "median_inlier_all"

        ] = valid[

            "inlier_ratio"

        ].median()


    if len(reliable_group) > 0:

        row[

            "median_inlier_reliable"

        ] = reliable_group[

            "inlier_ratio"

        ].median()


        row[

            "median_dx"

        ] = reliable_group[

            "dx_pixels"

        ].median()


        row[

            "median_dy"

        ] = reliable_group[

            "dy_pixels"

        ].median()


        row[

            "median_rotation"

        ] = reliable_group[

            "rotation_deg"

        ].median()


    summary.append(
        row
    )


summary_df = pd.DataFrame(
    summary
)


summary_df.to_csv(

    OUT
    /
    "ego_motion_full_summary.csv",

    index=False
)


# ============================================================
# TEXT SUMMARY
# ============================================================

with open(

    OUT
    /
    "SUMMARY.txt",

    "w"

) as f:


    f.write(
        "ROBUST EGO MOTION SUMMARY\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )


    for _, row in (
        summary_df.iterrows()
    ):


        f.write(

            f"{row['split']} "
            f"{row['sequence']}\n"
        )


        f.write(

            f"Pairs: "
            f"{int(row['total_pairs'])}\n"
        )


        f.write(

            f"Estimated: "
            f"{int(row['estimated_pairs'])}"
            f" "
            f"({row['estimate_rate_pct']:.2f}%)\n"
        )


        f.write(

            f"Reliable: "
            f"{int(row['reliable_pairs'])}"
            f" "
            f"({row['reliable_rate_pct']:.2f}%)\n"
        )


        f.write(
            "\n"
        )


print(
    "\nFULL EGO MOTION COMPLETE"
)

print(
    "\nResults:"
)

print(
    OUT
)

print(
    "\nSummary:"
)

print(
    summary_df
)