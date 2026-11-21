from pathlib import Path
import pandas as pd
import cv2

# ==============================
# YOUR DATASET PATHS
# ==============================

NESTLER = Path(
    r"C:\Users\HP\Downloads\Elephant\Elephant"
)

TRAIN = Path(
    r"C:\Users\HP\Downloads\conservation_drones_train_real\TrainReal"
)

TEST = Path(
    r"C:\Users\HP\Downloads\conservation_drones_test_real\TestReal"
)

OUT = Path(
    r"C:\Users\HP\Downloads\dataset_audit_results"
)

OUT.mkdir(exist_ok=True)

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


# ==============================
# NESTLER AUDIT
# ==============================

def audit_nestler():

    results = []

    for video in NESTLER.glob("*.mp4"):

        cap = cv2.VideoCapture(str(video))

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        duration = frames / fps if fps > 0 else 0

        results.append({
            "video": video.name,
            "fps": fps,
            "frames": frames,
            "duration_sec": duration,
            "width": width,
            "height": height
        })

    df = pd.DataFrame(results)

    df.to_csv(
        OUT / "nestler_video_audit.csv",
        index=False
    )

    return df


# ==============================
# BIRDSAI AUDIT
# ==============================

def audit_birdsai(root, split_name):

    annotation_folder = root / "annotations"

    sequence_results = []
    track_results = []

    for csv_file in annotation_folder.glob("*.csv"):

        try:

            df = pd.read_csv(
                csv_file,
                header=None,
                names=COLS
            )

        except Exception as e:

            print("Could not read:", csv_file)
            print(e)

            continue

        # -----------------------
        # ELEPHANT ONLY
        # species = 1
        # -----------------------

        elephants = df[df["species"] == 1].copy()

        sequence_results.append({

            "sequence":
                csv_file.stem,

            "all_annotations":
                len(df),

            "elephant_boxes":
                len(elephants),

            "elephant_frames":
                elephants["frame"].nunique(),

            "elephant_ids":
                elephants["object_id"].nunique(),

            "occluded_boxes":
                int(
                    (elephants["occlusion"] == 1).sum()
                ),

            "noisy_boxes":
                int(
                    (elephants["noise"] == 1).sum()
                )
        })


        # ======================
        # TRACK LEVEL ANALYSIS
        # ======================

        for object_id, track in elephants.groupby(
            "object_id"
        ):

            frames = sorted(
                track["frame"]
                .astype(int)
                .unique()
            )

            if len(frames) == 0:
                continue

            first_frame = frames[0]
            last_frame = frames[-1]

            frame_span = (
                last_frame
                - first_frame
                + 1
            )

            observed_frames = len(frames)

            duration_30hz = (
                frame_span / 30
            )

            occlusion_percent = (
                (track["occlusion"] == 1)
                .mean()
                * 100
            )

            noise_percent = (
                (track["noise"] == 1)
                .mean()
                * 100
            )

            track_results.append({

                "split":
                    split_name,

                "sequence":
                    csv_file.stem,

                "object_id":
                    object_id,

                "first_frame":
                    first_frame,

                "last_frame":
                    last_frame,

                "frame_span":
                    frame_span,

                "observed_frames":
                    observed_frames,

                "duration_sec_30hz":
                    duration_30hz,

                "occlusion_percent":
                    occlusion_percent,

                "noise_percent":
                    noise_percent
            })


    sequence_df = pd.DataFrame(
        sequence_results
    )

    track_df = pd.DataFrame(
        track_results
    )


    sequence_df.to_csv(

        OUT /
        f"{split_name}_sequence_audit.csv",

        index=False
    )


    track_df.to_csv(

        OUT /
        f"{split_name}_elephant_tracks.csv",

        index=False
    )


    return sequence_df, track_df


# ==============================
# RUN EVERYTHING
# ==============================

nestler = audit_nestler()

train_seq, train_tracks = audit_birdsai(
    TRAIN,
    "train_real"
)

test_seq, test_tracks = audit_birdsai(
    TEST,
    "test_real"
)


# ==============================
# SUMMARY
# ==============================

with open(
    OUT / "SUMMARY.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ELEPHANT DATASET AUDIT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )


    # --------------------------
    # NESTLER
    # --------------------------

    f.write(
        "NESTLER\n"
    )

    f.write(
        f"Videos: {len(nestler)}\n"
    )

    if len(nestler) > 0:

        f.write(
            f"Total frames: "
            f"{nestler['frames'].sum()}\n"
        )

        f.write(
            f"Total duration: "
            f"{nestler['duration_sec'].sum():.2f} sec\n"
        )


    # --------------------------
    # BIRDSAI FUNCTION
    # --------------------------

    def summarize(name, seq, tracks):

        f.write(
            "\n" + name + "\n"
        )

        f.write(
            f"Sequences: {len(seq)}\n"
        )

        elephant_sequences = (
            seq["elephant_boxes"] > 0
        ).sum()

        f.write(
            f"Elephant sequences: "
            f"{elephant_sequences}\n"
        )

        f.write(
            f"Elephant bounding boxes: "
            f"{seq['elephant_boxes'].sum()}\n"
        )

        f.write(
            f"Elephant tracks: "
            f"{len(tracks)}\n"
        )

        if len(tracks) > 0:

            f.write(
                "Track length frames "
                "(min / median / max): "
            )

            f.write(
                f"{tracks['observed_frames'].min()} / "
                f"{tracks['observed_frames'].median()} / "
                f"{tracks['observed_frames'].max()}\n"
            )

            f.write(
                "Track duration seconds "
                "@30Hz (min / median / max): "
            )

            f.write(
                f"{tracks['duration_sec_30hz'].min():.2f} / "
                f"{tracks['duration_sec_30hz'].median():.2f} / "
                f"{tracks['duration_sec_30hz'].max():.2f}\n"
            )

            f.write(
                f"Mean occlusion: "
                f"{tracks['occlusion_percent'].mean():.2f}%\n"
            )

            f.write(
                f"Mean noise: "
                f"{tracks['noise_percent'].mean():.2f}%\n"
            )


    summarize(
        "BIRDSAI TRAIN REAL",
        train_seq,
        train_tracks
    )

    summarize(
        "BIRDSAI TEST REAL",
        test_seq,
        test_tracks
    )


print("\nAUDIT COMPLETE")

print(
    "\nResults saved here:"
)

print(OUT)