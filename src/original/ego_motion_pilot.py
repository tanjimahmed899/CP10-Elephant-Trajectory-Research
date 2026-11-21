from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import math

# =========================================
# DATASET PATHS
# =========================================

TRAIN_IMAGES = Path(
    r"C:\Users\HP\Downloads\conservation_drones_train_real\TrainReal\images"
)

TEST_IMAGES = Path(
    r"C:\Users\HP\Downloads\conservation_drones_test_real\TestReal\images"
)

OUT = Path(
    r"C:\Users\HP\Downloads\ego_motion_pilot"
)

OUT.mkdir(exist_ok=True)

# Important sequences
SEQUENCES = [

    (
        "train_0367",
        TRAIN_IMAGES / "0000000367_0000000000"
    ),

    (
        "test_0352",
        TEST_IMAGES / "0000000352_0000000000"
    ),

    (
        "test_0353",
        TEST_IMAGES / "0000000353_0000000000"
    )
]

# Keep pilot reasonably fast
MAX_PAIRS = 300

# Downscale for CPU speed
SCALE = 0.5


# =========================================
# PREPROCESS
# =========================================

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

    # Improve thermal contrast
    img = cv2.equalizeHist(img)

    return img


# =========================================
# MOTION ESTIMATION
# =========================================

def estimate_motion(prev_img, curr_img):

    # Find background-like feature points
    p0 = cv2.goodFeaturesToTrack(

        prev_img,

        maxCorners=500,

        qualityLevel=0.01,

        minDistance=8,

        blockSize=7
    )

    if p0 is None or len(p0) < 20:
        return None

    # Track points to current frame
    p1, status, error = (
        cv2.calcOpticalFlowPyrLK(

            prev_img,
            curr_img,
            p0,
            None,

            winSize=(21, 21),

            maxLevel=3
        )
    )

    if p1 is None:
        return None

    good_old = p0[
        status.flatten() == 1
    ]

    good_new = p1[
        status.flatten() == 1
    ]

    if len(good_old) < 15:
        return None

    # Find transform:
    # current frame -> previous frame
    M, inliers = (
        cv2.estimateAffinePartial2D(

            good_new,
            good_old,

            method=cv2.RANSAC,

            ransacReprojThreshold=3.0,

            maxIters=2000,

            confidence=0.99
        )
    )

    if M is None:
        return None

    if inliers is None:

        inlier_ratio = 0

    else:

        inlier_ratio = (
            inliers.sum()
            / len(inliers)
        )

    # Translation back to original-resolution pixels
    dx = M[0, 2] / SCALE
    dy = M[1, 2] / SCALE

    # Rotation
    rotation = math.degrees(

        math.atan2(
            M[1, 0],
            M[0, 0]
        )
    )

    # Approximate scale
    scale = math.sqrt(

        M[0, 0] ** 2
        +
        M[1, 0] ** 2
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
            scale
    }


# =========================================
# PROCESS SEQUENCE
# =========================================

all_results = []

summary = []


for sequence_name, folder in SEQUENCES:

    print(
        "\nProcessing:",
        sequence_name
    )

    images = sorted([

        p for p in folder.iterdir()

        if p.suffix.lower()
        in [".jpg", ".jpeg", ".png"]

    ])

    if len(images) < 2:

        print(
            "Not enough images:",
            folder
        )

        continue

    total_pairs = min(

        len(images) - 1,

        MAX_PAIRS
    )

    success = 0

    sequence_results = []


    for i in range(total_pairs):

        prev_img = prepare_image(
            images[i]
        )

        curr_img = prepare_image(
            images[i + 1]
        )

        if (
            prev_img is None
            or curr_img is None
        ):

            continue

        motion = estimate_motion(

            prev_img,
            curr_img
        )

        if motion is None:

            sequence_results.append({

                "sequence":
                    sequence_name,

                "pair":
                    i,

                "success":
                    0
            })

            continue


        success += 1


        row = {

            "sequence":
                sequence_name,

            "pair":
                i,

            "frame_a":
                images[i].name,

            "frame_b":
                images[i + 1].name,

            "success":
                1
        }

        row.update(motion)

        sequence_results.append(row)


        if (i + 1) % 50 == 0:

            print(
                f"{i+1}/{total_pairs}"
            )


    seq_df = pd.DataFrame(
        sequence_results
    )


    seq_df.to_csv(

        OUT /
        f"{sequence_name}_motion.csv",

        index=False
    )


    valid = seq_df[
        seq_df["success"] == 1
    ]


    success_rate = (

        100 * success
        / total_pairs
    )


    if len(valid) > 0:

        summary.append({

            "sequence":
                sequence_name,

            "pairs_checked":
                total_pairs,

            "successful_pairs":
                success,

            "success_rate_pct":
                success_rate,

            "median_features":
                valid[
                    "features"
                ].median(),

            "median_inlier_ratio":
                valid[
                    "inlier_ratio"
                ].median(),

            "median_translation_x":
                valid[
                    "dx_pixels"
                ].median(),

            "median_translation_y":
                valid[
                    "dy_pixels"
                ].median(),

            "median_rotation_deg":
                valid[
                    "rotation_deg"
                ].median(),

            "median_scale":
                valid[
                    "scale"
                ].median()
        })

    else:

        summary.append({

            "sequence":
                sequence_name,

            "pairs_checked":
                total_pairs,

            "successful_pairs":
                0,

            "success_rate_pct":
                0
        })


summary_df = pd.DataFrame(
    summary
)


summary_df.to_csv(

    OUT /
    "ego_motion_summary.csv",

    index=False
)


print("\nDONE")

print(
    "\nSummary:"
)

print(summary_df)

print(
    "\nResults saved:"
)

print(OUT)