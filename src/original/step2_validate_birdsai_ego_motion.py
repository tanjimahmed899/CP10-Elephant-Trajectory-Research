from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import math
import zlib

# ============================================================
# PATHS
# ============================================================
TRAIN_ROOT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result\conservation_drones_train_real\TrainReal")
TEST_ROOT = Path(r"E:\AIUB\Capstone Project\Journal Based Results\Main Result\conservation_drones_test_real\TestReal")
OUT = Path(r"C:\Users\HP\Downloads\step2_ego_motion_validation")
OUT.mkdir(parents=True, exist_ok=True)

COLS = [
    "frame", "object_id", "x", "y", "w", "h", "class", "species",
    "occlusion", "noise"
]

TRAIN_SEQUENCES = [
    "0000000010_0000000000",
    "0000000054_0000000000",
    "0000000056_0000000000",
    "0000000351_0000000000",
    "0000000367_0000000000",
]

TEST_SEQUENCES = [
    "0000000055_0000000000",
    "0000000058_0000000000",
    "0000000349_0000000000",
    "0000000352_0000000000",
    "0000000353_0000000000",
    "0000000357_0000000000",
]

# ============================================================
# SETTINGS
# ============================================================
SCALE = 0.5
MASK_MARGIN = 10

MIN_FEATURES = 40
MIN_INLIER_RATIO = 0.55

MIN_SCALE = 0.95
MAX_SCALE = 1.05
MAX_ROTATION = 5.0

FIT_FRACTION = 0.70
MIN_VALIDATION_POINTS = 10


# ============================================================
# IMAGE PREPARATION
# ============================================================
def prepare_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(
        img,
        None,
        fx=SCALE,
        fy=SCALE
    )

    img = cv2.equalizeHist(img)

    return img


# ============================================================
# REMOVE ELEPHANT / OBJECT REGIONS
# ============================================================
def make_background_mask(image_shape, annotations, frame_number):

    h, w = image_shape

    mask = np.full(
        (h, w),
        255,
        dtype=np.uint8
    )

    rows = annotations[
        annotations["frame"] == frame_number
    ]

    margin = int(MASK_MARGIN * SCALE)

    for _, r in rows.iterrows():

        x = int(r["x"] * SCALE)
        y = int(r["y"] * SCALE)

        bw = int(r["w"] * SCALE)
        bh = int(r["h"] * SCALE)

        x1 = max(
            0,
            x - margin
        )

        y1 = max(
            0,
            y - margin
        )

        x2 = min(
            w - 1,
            x + bw + margin
        )

        y2 = min(
            h - 1,
            y + bh + margin
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
# CHECK WHETHER POINTS ARE INSIDE BACKGROUND MASK
# ============================================================
def points_in_mask(points, mask):

    pts = points.reshape(-1, 2)

    xs = np.clip(
        np.round(pts[:, 0]).astype(int),
        0,
        mask.shape[1] - 1
    )

    ys = np.clip(
        np.round(pts[:, 1]).astype(int),
        0,
        mask.shape[0] - 1
    )

    return mask[ys, xs] > 0


# ============================================================
# TRANSFORM POINTS
# ============================================================
def transform_points(M, pts):

    pts = np.asarray(
        pts,
        dtype=np.float64
    ).reshape(-1, 2)

    homog = np.c_[
        pts,
        np.ones(len(pts))
    ]

    return (M @ homog.T).T


# ============================================================
# VALIDATE ONE FRAME PAIR
# ============================================================
def validate_pair(
    prev_img,
    curr_img,
    prev_mask,
    curr_mask,
    seed
):

    # --------------------------------------------------------
    # Detect background features
    # --------------------------------------------------------
    p0 = cv2.goodFeaturesToTrack(
        prev_img,
        maxCorners=700,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        mask=prev_mask
    )

    if p0 is None:
        return None

    if len(p0) < MIN_FEATURES:
        return None

    # --------------------------------------------------------
    # Track features using KLT optical flow
    # --------------------------------------------------------
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_img,
        curr_img,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if p1 is None:
        return None

    ok = status.flatten() == 1

    old = p0[ok].reshape(-1, 2)
    new = p1[ok].reshape(-1, 2)

    if len(old) == 0:
        return None

    # --------------------------------------------------------
    # Only keep points that remain background in both frames
    # --------------------------------------------------------
    keep = (
        points_in_mask(old, prev_mask) &
        points_in_mask(new, curr_mask)
    )

    old = old[keep]
    new = new[keep]

    if len(old) < MIN_FEATURES:
        return None

    # --------------------------------------------------------
    # Random split:
    #
    # 70% points = transform fitting
    # 30% points = completely held-out validation
    # --------------------------------------------------------
    rng = np.random.default_rng(seed)

    idx = np.arange(len(old))

    rng.shuffle(idx)

    n_fit = int(
        round(
            FIT_FRACTION * len(idx)
        )
    )

    n_fit = min(
        max(
            n_fit,
            MIN_FEATURES
        ),
        len(idx) - MIN_VALIDATION_POINTS
    )

    if n_fit < MIN_FEATURES:
        return None

    if len(idx) - n_fit < MIN_VALIDATION_POINTS:
        return None

    fit_idx = idx[:n_fit]
    val_idx = idx[n_fit:]

    old_fit = old[fit_idx]
    new_fit = new[fit_idx]

    old_val = old[val_idx]
    new_val = new[val_idx]

    # --------------------------------------------------------
    # Estimate camera motion
    #
    # Current frame -> Previous frame
    # --------------------------------------------------------
    M, inliers = cv2.estimateAffinePartial2D(
        new_fit,
        old_fit,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.99
    )

    if M is None:
        return None

    # --------------------------------------------------------
    # Motion-quality statistics
    # --------------------------------------------------------
    if inliers is not None:
        inlier_ratio = float(
            inliers.sum() /
            len(inliers)
        )
    else:
        inlier_ratio = 0.0

    rotation_deg = math.degrees(
        math.atan2(
            M[1, 0],
            M[0, 0]
        )
    )

    scale = math.sqrt(
        M[0, 0] ** 2 +
        M[1, 0] ** 2
    )

    reliable = (
        len(old_fit) >= MIN_FEATURES
        and
        inlier_ratio >= MIN_INLIER_RATIO
        and
        MIN_SCALE <= scale <= MAX_SCALE
        and
        abs(rotation_deg) <= MAX_ROTATION
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Residual error is measured ONLY on held-out points,
    # NOT on the points used to fit the transform.
    # --------------------------------------------------------
    pred_prev = transform_points(
        M,
        new_val
    )

    # Raw apparent background motion
    raw_err = np.linalg.norm(
        new_val - old_val,
        axis=1
    ) / SCALE

    # Motion after compensation
    corrected_err = np.linalg.norm(
        pred_prev - old_val,
        axis=1
    ) / SCALE

    raw_median = float(
        np.median(raw_err)
    )

    corr_median = float(
        np.median(corrected_err)
    )

    raw_p95 = float(
        np.percentile(
            raw_err,
            95
        )
    )

    corr_p95 = float(
        np.percentile(
            corrected_err,
            95
        )
    )

    reduction = np.nan

    if raw_median > 1e-12:

        reduction = (
            100.0 *
            (
                raw_median -
                corr_median
            ) /
            raw_median
        )

    return {

        "total_matches":
            len(old),

        "fit_points":
            len(old_fit),

        "validation_points":
            len(old_val),

        "fit_inlier_ratio":
            inlier_ratio,

        "rotation_deg":
            rotation_deg,

        "scale":
            scale,

        "reliable":
            int(reliable),

        "raw_median_px":
            raw_median,

        "corrected_median_px":
            corr_median,

        "raw_p95_px":
            raw_p95,

        "corrected_p95_px":
            corr_p95,

        "median_residual_reduction_pct":
            reduction
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
        root /
        "images" /
        sequence
    )

    annotation_file = (
        root /
        "annotations" /
        f"{sequence}.csv"
    )

    if (
        not image_folder.exists()
        or
        not annotation_file.exists()
    ):

        print(
            "Missing:",
            split,
            sequence
        )

        return []

    ann = pd.read_csv(
        annotation_file,
        header=None,
        names=COLS
    )

    images = sorted([
        p
        for p in image_folder.iterdir()
        if p.suffix.lower()
        in [
            ".jpg",
            ".jpeg",
            ".png"
        ]
    ])

    rows = []

    print(
        f"Processing {split} "
        f"{sequence}: "
        f"{len(images)} frames"
    )

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

        prev_mask = make_background_mask(
            prev_img.shape,
            ann,
            i
        )

        curr_mask = make_background_mask(
            curr_img.shape,
            ann,
            i + 1
        )

        seed = (
            zlib.crc32(
                f"{split}:{sequence}:{i}".encode()
            )
            &
            0xffffffff
        )

        result = validate_pair(
            prev_img,
            curr_img,
            prev_mask,
            curr_mask,
            seed
        )

        if result is None:

            rows.append({

                "split":
                    split,

                "sequence":
                    sequence,

                "pair":
                    i,

                "frame_a":
                    images[i].name,

                "frame_b":
                    images[i + 1].name,

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
                    images[i + 1].name,

                "estimated":
                    1
            }

            row.update(
                result
            )

            rows.append(
                row
            )

    return rows


# ============================================================
# RUN TRAINREAL
# ============================================================
all_rows = []

for seq in TRAIN_SEQUENCES:

    all_rows.extend(

        process_sequence(
            TRAIN_ROOT,
            "train_real",
            seq
        )

    )


# ============================================================
# RUN TESTREAL
# ============================================================
for seq in TEST_SEQUENCES:

    all_rows.extend(

        process_sequence(
            TEST_ROOT,
            "test_real",
            seq
        )

    )


# ============================================================
# SAVE ALL FRAME-PAIR RESULTS
# ============================================================
pairs = pd.DataFrame(
    all_rows
)

pairs.to_csv(
    OUT /
    "heldout_background_residual_pairs.csv",
    index=False
)


# ============================================================
# KEEP ONLY RELIABLE TRANSFORMS
# ============================================================
valid = pairs[
    (
        pairs.get(
            "estimated",
            0
        ) == 1
    )
    &
    (
        pairs.get(
            "reliable",
            0
        ) == 1
    )
].copy()


# ============================================================
# SEQUENCE-LEVEL SUMMARY
# ============================================================
summary_rows = []

for (
    split,
    sequence
), g in valid.groupby(
    [
        "split",
        "sequence"
    ]
):

    summary_rows.append({

        "split":
            split,

        "sequence":
            sequence,

        "reliable_pairs":
            len(g),

        "median_raw_background_motion_px":
            float(
                g[
                    "raw_median_px"
                ].median()
            ),

        "median_corrected_residual_px":
            float(
                g[
                    "corrected_median_px"
                ].median()
            ),

        "p95_raw_background_motion_px":
            float(
                g[
                    "raw_p95_px"
                ].median()
            ),

        "p95_corrected_residual_px":
            float(
                g[
                    "corrected_p95_px"
                ].median()
            ),

        "median_residual_reduction_pct":
            float(
                g[
                    "median_residual_reduction_pct"
                ].median()
            ),

        "median_fit_inlier_ratio":
            float(
                g[
                    "fit_inlier_ratio"
                ].median()
            ),

        "median_validation_points":
            float(
                g[
                    "validation_points"
                ].median()
            )
    })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT /
    "heldout_background_residual_summary.csv",
    index=False
)


# ============================================================
# SPLIT-LEVEL SUMMARY
# ============================================================
split_rows = []

for split, g in valid.groupby(
    "split"
):

    split_rows.append({

        "split":
            split,

        "reliable_pairs":
            len(g),

        "sequences":
            g[
                "sequence"
            ].nunique(),

        "median_raw_background_motion_px":
            float(
                g[
                    "raw_median_px"
                ].median()
            ),

        "median_corrected_residual_px":
            float(
                g[
                    "corrected_median_px"
                ].median()
            ),

        "median_residual_reduction_pct":
            float(
                g[
                    "median_residual_reduction_pct"
                ].median()
            ),

        "p95_corrected_residual_px":
            float(
                np.percentile(
                    g[
                        "corrected_median_px"
                    ],
                    95
                )
            )
    })


split_summary = pd.DataFrame(
    split_rows
)

split_summary.to_csv(
    OUT /
    "heldout_background_residual_split_summary.csv",
    index=False
)


# ============================================================
# SAVE HUMAN-READABLE SUMMARY
# ============================================================
with open(
    OUT / "SUMMARY.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 2A: BIRDSAI EGO-MOTION "
        "GEOMETRIC CONSISTENCY VALIDATION\n"
    )

    f.write(
        "=" * 72 +
        "\n"
    )

    f.write(
        "Background features inside annotated "
        "object masks are excluded.\n"
    )

    f.write(
        "A 70% subset of background feature "
        "correspondences fits the affine transform.\n"
    )

    f.write(
        "Residual motion is evaluated on the "
        "held-out 30% feature correspondences.\n"
    )

    f.write(
        "Therefore the reported residual is NOT "
        "measured on the same points used to "
        "fit the transform.\n\n"
    )

    f.write(
        split_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\nIMPORTANT: "
        "This is a geometric-consistency test, "
        "not independent metric ground truth.\n"
    )


# ============================================================
# FINISHED
# ============================================================
print(
    "\nDONE. Outputs written to:",
    OUT
)

print(
    split_summary.to_string(
        index=False
    )
)