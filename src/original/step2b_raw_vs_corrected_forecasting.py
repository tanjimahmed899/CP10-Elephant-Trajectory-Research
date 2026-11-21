from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

TRAJ_DIR = (
    ROOT /
    "trajectory_dataset"
)

CORR_DIR = (
    ROOT /
    "corrected_trajectory_dataset"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step2b_raw_vs_corrected_baseline"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RAW_FPS = 30

TARGET_FPS = 10

DOWNSAMPLE = (
    RAW_FPS //
    TARGET_FPS
)

# 2 seconds at 30 Hz
OBS_RAW = 60


TASKS = {

    "2s_to_2s": {

        "future_raw":
            60,

        "future_10hz":
            20
    },

    "2s_to_5s": {

        "future_raw":
            150,

        "future_10hz":
            50
    }
}


# ============================================================
# CONSTANT VELOCITY
# ============================================================

def estimate_velocity_linear_regression(
    history,
    fps=10
):

    history = np.asarray(
        history,
        dtype=np.float64
    )

    T = len(
        history
    )

    times = np.arange(
        T,
        dtype=np.float64
    ) / float(
        fps
    )

    times_centered = (
        times -
        times.mean()
    )

    denom = np.sum(
        times_centered ** 2
    )

    if denom <= 1e-12:

        return np.zeros(
            2,
            dtype=np.float64
        )


    positions_centered = (

        history

        -

        history.mean(
            axis=0,
            keepdims=True
        )
    )


    velocity = (

        times_centered[:, None]

        *

        positions_centered

    ).sum(
        axis=0
    ) / denom


    return velocity


def predict_constant_velocity(
    history,
    future_length,
    fps=10
):

    velocity = (
        estimate_velocity_linear_regression(
            history,
            fps
        )
    )


    future_times = (

        np.arange(
            1,
            future_length + 1,
            dtype=np.float64
        )

        /

        float(
            fps
        )
    )


    prediction = (

        history[-1][None, :]

        +

        future_times[:, None]
        *
        velocity[None, :]
    )


    return prediction


# ============================================================
# STATIONARY BASELINE
# ============================================================

def predict_stationary(
    history,
    future_length
):

    return np.repeat(

        history[-1:, :],

        future_length,

        axis=0
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    prediction,
    truth
):

    errors = np.linalg.norm(

        np.asarray(
            prediction
        )

        -

        np.asarray(
            truth
        ),

        axis=1
    )


    ade = float(
        errors.mean()
    )

    fde = float(
        errors[-1]
    )


    return (
        ade,
        fde
    )


# ============================================================
# IMPROVEMENT
# ============================================================

def safe_improvement(
    raw_value,
    corrected_value
):

    if (
        not np.isfinite(
            raw_value
        )
        or
        abs(
            raw_value
        ) <= 1e-12
    ):

        return np.nan


    return (

        100.0

        *

        (
            raw_value
            -
            corrected_value
        )

        /

        raw_value
    )


# ============================================================
# MACRO AVERAGE
# ============================================================

def macro_average(
    df,
    group_col,
    metric_col
):

    return float(

        df.groupby(
            group_col
        )[metric_col]
        .mean()
        .mean()
    )


# ============================================================
# BUILD TRACK LOOKUP
# ============================================================

def build_track_lookup(
    cleaned
):

    lookup = {}


    for (
        split,
        sequence,
        segment_id
    ), group in cleaned.groupby(

        [
            "split",
            "sequence",
            "segment_id"
        ],

        sort=False
    ):


        group = (

            group
            .sort_values(
                "frame"
            )
            .copy()
        )


        group[
            "frame"
        ] = (

            group[
                "frame"
            ]
            .astype(
                int
            )
        )


        key = (

            str(
                split
            ),

            str(
                sequence
            ),

            str(
                segment_id
            )
        )


        lookup[
            key
        ] = (

            group
            .set_index(
                "frame"
            )
        )


    return lookup


# ============================================================
# EXTRACT RAW WINDOW
# ============================================================

def extract_raw_window(
    track_lookup,
    row,
    future_raw
):

    key = (

        str(
            row[
                "split"
            ]
        ),

        str(
            row[
                "sequence"
            ]
        ),

        str(
            row[
                "segment_id"
            ]
        )
    )


    if key not in track_lookup:

        raise KeyError(
            f"Track not found: {key}"
        )


    track = (
        track_lookup[
            key
        ]
    )


    start_frame = int(
        row[
            "obs_start_frame"
        ]
    )


    future_end = int(
        row[
            "future_end_frame"
        ]
    )


    needed_frames = list(

        range(
            start_frame,
            future_end + 1
        )
    )


    missing = [

        frame

        for frame
        in needed_frames

        if frame
        not in track.index
    ]


    if missing:

        raise ValueError(

            f"Missing raw frames for {key}: "
            f"first missing={missing[0]}"
        )


    raw = (

        track.loc[

            needed_frames,

            [
                "cx_n",
                "cy_n"
            ]
        ]

        .to_numpy(
            dtype=np.float64
        )
    )


    expected_length = (

        OBS_RAW

        +

        future_raw
    )


    if len(
        raw
    ) != expected_length:

        raise ValueError(

            f"Unexpected raw length "
            f"{len(raw)} != "
            f"{expected_length} "
            f"for {key}"
        )


    history_raw = (
        raw[
            :OBS_RAW
        ]
    )


    future_raw_array = (
        raw[
            OBS_RAW:
        ]
    )


    # --------------------------------------------
    # 30 Hz -> 10 Hz
    # every third frame
    # --------------------------------------------

    history_10hz = (

        history_raw[
            ::DOWNSAMPLE
        ]
    )


    future_10hz = (

        future_raw_array[
            ::DOWNSAMPLE
        ]
    )


    return (
        history_10hz,
        future_10hz
    )


# ============================================================
# EVALUATE ONE REPRESENTATION
# ============================================================

def evaluate_representation(
    history,
    future,
    model_name
):

    if model_name == "Stationary":

        prediction = (
            predict_stationary(
                history,
                len(
                    future
                )
            )
        )


    elif model_name == "ConstantVelocity":

        prediction = (
            predict_constant_velocity(

                history,

                len(
                    future
                ),

                TARGET_FPS
            )
        )


    else:

        raise ValueError(
            model_name
        )


    return metrics(
        prediction,
        future
    )


# ============================================================
# MAIN
# ============================================================

def main():


    # ========================================================
    # LOAD RAW CLEANED TRACK POINTS
    # ========================================================

    cleaned_path = (

        TRAJ_DIR /
        "cleaned_track_points.csv"
    )


    if not cleaned_path.exists():

        raise FileNotFoundError(

            f"\nMissing file:\n"
            f"{cleaned_path}\n\n"
            "This file is required for "
            "raw uncorrected trajectories."
        )


    print(
        "Loading raw cleaned tracks..."
    )


    cleaned = pd.read_csv(

        cleaned_path,

        dtype={

            "split":
                str,

            "sequence":
                str,

            "segment_id":
                str
        }
    )


    track_lookup = (
        build_track_lookup(
            cleaned
        )
    )


    print(
        "Track lookup created:",
        len(
            track_lookup
        )
    )


    # ========================================================
    # STORAGE
    # ========================================================

    all_sample_rows = []

    all_summary_rows = []

    all_sequence_rows = []


    # ========================================================
    # PROCESS EACH TASK
    # ========================================================

    for (
        task,
        config
    ) in TASKS.items():


        print(
            "\n"
            +
            "=" * 72
        )

        print(
            "TASK:",
            task
        )

        print(
            "=" * 72
        )


        metadata_path = (

            CORR_DIR /
            f"corrected_windows_{task}.csv"
        )


        corrected_npz_path = (

            CORR_DIR /
            f"corrected_{task}.npz"
        )


        if not metadata_path.exists():

            raise FileNotFoundError(
                metadata_path
            )


        if not corrected_npz_path.exists():

            raise FileNotFoundError(
                corrected_npz_path
            )


        # ----------------------------------------------------
        # LOAD CORRECTED METADATA
        # ----------------------------------------------------

        metadata = pd.read_csv(

            metadata_path,

            dtype={

                "split":
                    str,

                "sequence":
                    str,

                "segment_id":
                    str
            }
        )


        # ----------------------------------------------------
        # LOAD CORRECTED TRAJECTORY ARRAYS
        # ----------------------------------------------------

        arrays = np.load(
            corrected_npz_path
        )


        corrected_history_all = (

            arrays[
                "history"
            ]
        )


        corrected_future_all = (

            arrays[
                "future"
            ]
        )


        # ----------------------------------------------------
        # ONLY FINAL CORE TESTREAL SAMPLES
        # ----------------------------------------------------

        core = metadata[

            (
                metadata[
                    "accepted"
                ]
                ==
                1
            )

            &

            (
                metadata[
                    "quality_tier"
                ]
                .astype(
                    str
                )
                .str
                .lower()
                ==
                "core"
            )

            &

            (
                metadata[
                    "split"
                ]
                ==
                "test_real"
            )

        ].copy()


        core = (

            core
            .sort_values(
                "sample_id"
            )
            .reset_index(
                drop=True
            )
        )


        print(
            "Core TestReal samples:",
            len(
                core
            )
        )


        expected_n = (

            1053

            if task
            ==
            "2s_to_2s"

            else

            826
        )


        if len(
            core
        ) != expected_n:

            print(

                "WARNING: expected",
                expected_n,
                "samples but found",
                len(
                    core
                )
            )


        # ====================================================
        # SAMPLE LOOP
        # ====================================================

        for _, row in core.iterrows():


            sample_id = int(
                row[
                    "sample_id"
                ]
            )


            if (
                sample_id < 0
                or
                sample_id
                >=
                len(
                    corrected_history_all
                )
            ):

                raise IndexError(

                    f"Invalid sample_id "
                    f"{sample_id} "
                    f"for {task}"
                )


            # ------------------------------------------------
            # RAW TRAJECTORY
            # ------------------------------------------------

            (
                raw_history,
                raw_future
            ) = extract_raw_window(

                track_lookup,

                row,

                config[
                    "future_raw"
                ]
            )


            # ------------------------------------------------
            # CORRECTED TRAJECTORY
            # ------------------------------------------------

            corrected_history = (

                np.asarray(

                    corrected_history_all[
                        sample_id,
                        :,
                        :2
                    ],

                    dtype=np.float64
                )

                [
                    ::DOWNSAMPLE
                ]
            )


            corrected_future = (

                np.asarray(

                    corrected_future_all[
                        sample_id,
                        :,
                        :2
                    ],

                    dtype=np.float64
                )

                [
                    ::DOWNSAMPLE
                ]
            )


            # ------------------------------------------------
            # SHAPE CHECKS
            # ------------------------------------------------

            if raw_history.shape != (
                20,
                2
            ):

                raise ValueError(

                    f"Raw history shape "
                    f"{raw_history.shape} "
                    f"for {task}"
                )


            if corrected_history.shape != (
                20,
                2
            ):

                raise ValueError(

                    f"Corrected history shape "
                    f"{corrected_history.shape}"
                )


            expected_future_shape = (

                config[
                    "future_10hz"
                ],

                2
            )


            if raw_future.shape != (
                expected_future_shape
            ):

                raise ValueError(

                    f"Raw future shape "
                    f"{raw_future.shape} "
                    f"expected "
                    f"{expected_future_shape}"
                )


            if corrected_future.shape != (
                expected_future_shape
            ):

                raise ValueError(

                    f"Corrected future shape "
                    f"{corrected_future.shape} "
                    f"expected "
                    f"{expected_future_shape}"
                )


            # ------------------------------------------------
            # BASE METADATA
            # ------------------------------------------------

            base_row = {

                "task":
                    task,

                "sample_id":
                    sample_id,

                "sequence":
                    str(
                        row[
                            "sequence"
                        ]
                    ),

                "track_key":
                    str(
                        row[
                            "segment_id"
                        ]
                    ),

                "object_id":
                    row.get(
                        "object_id",
                        np.nan
                    ),

                "obs_start_frame":
                    int(
                        row[
                            "obs_start_frame"
                        ]
                    ),

                "obs_end_frame":
                    int(
                        row[
                            "obs_end_frame"
                        ]
                    ),

                "future_end_frame":
                    int(
                        row[
                            "future_end_frame"
                        ]
                    ),

                "repaired_motion_pct":
                    float(
                        row[
                            "repaired_motion_pct"
                        ]
                    )
            }


            # ------------------------------------------------
            # RUN BOTH BASELINES
            # ------------------------------------------------

            for model_name in [

                "Stationary",
                "ConstantVelocity"

            ]:


                (
                    raw_ADE,
                    raw_FDE
                ) = evaluate_representation(

                    raw_history,

                    raw_future,

                    model_name
                )


                (
                    corrected_ADE,
                    corrected_FDE
                ) = evaluate_representation(

                    corrected_history,

                    corrected_future,

                    model_name
                )


                result_row = (
                    dict(
                        base_row
                    )
                )


                result_row.update({

                    "model":
                        model_name,

                    "raw_ADE":
                        raw_ADE,

                    "raw_FDE":
                        raw_FDE,

                    "corrected_ADE":
                        corrected_ADE,

                    "corrected_FDE":
                        corrected_FDE,

                    "ADE_improvement_pct":
                        safe_improvement(

                            raw_ADE,

                            corrected_ADE
                        ),

                    "FDE_improvement_pct":
                        safe_improvement(

                            raw_FDE,

                            corrected_FDE
                        )
                })


                all_sample_rows.append(
                    result_row
                )


    # ========================================================
    # SAMPLE-LEVEL RESULTS
    # ========================================================

    samples = pd.DataFrame(
        all_sample_rows
    )


    samples.to_csv(

        OUT /
        "raw_vs_corrected_sample_metrics.csv",

        index=False
    )


    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    for (
        task,
        model
    ), group in samples.groupby(

        [
            "task",
            "model"
        ],

        sort=False
    ):


        summary_row = {

            "task":
                task,

            "model":
                model,

            "samples":
                len(
                    group
                ),

            "tracks":
                group[
                    "track_key"
                ]
                .nunique(),

            "sequences":
                group[
                    "sequence"
                ]
                .nunique(),


            # --------------------------------------------
            # WINDOW MEAN
            # --------------------------------------------

            "raw_mean_ADE":
                float(
                    group[
                        "raw_ADE"
                    ]
                    .mean()
                ),

            "corrected_mean_ADE":
                float(
                    group[
                        "corrected_ADE"
                    ]
                    .mean()
                ),

            "raw_mean_FDE":
                float(
                    group[
                        "raw_FDE"
                    ]
                    .mean()
                ),

            "corrected_mean_FDE":
                float(
                    group[
                        "corrected_FDE"
                    ]
                    .mean()
                ),


            # --------------------------------------------
            # TRACK MACRO
            # --------------------------------------------

            "raw_track_macro_ADE":
                macro_average(

                    group,

                    "track_key",

                    "raw_ADE"
                ),

            "corrected_track_macro_ADE":
                macro_average(

                    group,

                    "track_key",

                    "corrected_ADE"
                ),

            "raw_track_macro_FDE":
                macro_average(

                    group,

                    "track_key",

                    "raw_FDE"
                ),

            "corrected_track_macro_FDE":
                macro_average(

                    group,

                    "track_key",

                    "corrected_FDE"
                ),


            # --------------------------------------------
            # SOURCE MACRO
            # --------------------------------------------

            "raw_source_macro_ADE":
                macro_average(

                    group,

                    "sequence",

                    "raw_ADE"
                ),

            "corrected_source_macro_ADE":
                macro_average(

                    group,

                    "sequence",

                    "corrected_ADE"
                ),

            "raw_source_macro_FDE":
                macro_average(

                    group,

                    "sequence",

                    "raw_FDE"
                ),

            "corrected_source_macro_FDE":
                macro_average(

                    group,

                    "sequence",

                    "corrected_FDE"
                ),


            # --------------------------------------------
            # PAIRED SAMPLE MEDIAN
            # --------------------------------------------

            "median_sample_ADE_improvement_pct":
                float(
                    group[
                        "ADE_improvement_pct"
                    ]
                    .median()
                ),

            "median_sample_FDE_improvement_pct":
                float(
                    group[
                        "FDE_improvement_pct"
                    ]
                    .median()
                )
        }


        # --------------------------------------------
        # IMPROVEMENT %
        # --------------------------------------------

        summary_row[
            "ADE_improvement_pct"
        ] = safe_improvement(

            summary_row[
                "raw_mean_ADE"
            ],

            summary_row[
                "corrected_mean_ADE"
            ]
        )


        summary_row[
            "FDE_improvement_pct"
        ] = safe_improvement(

            summary_row[
                "raw_mean_FDE"
            ],

            summary_row[
                "corrected_mean_FDE"
            ]
        )


        summary_row[
            "track_macro_ADE_improvement_pct"
        ] = safe_improvement(

            summary_row[
                "raw_track_macro_ADE"
            ],

            summary_row[
                "corrected_track_macro_ADE"
            ]
        )


        summary_row[
            "track_macro_FDE_improvement_pct"
        ] = safe_improvement(

            summary_row[
                "raw_track_macro_FDE"
            ],

            summary_row[
                "corrected_track_macro_FDE"
            ]
        )


        summary_row[
            "source_macro_ADE_improvement_pct"
        ] = safe_improvement(

            summary_row[
                "raw_source_macro_ADE"
            ],

            summary_row[
                "corrected_source_macro_ADE"
            ]
        )


        summary_row[
            "source_macro_FDE_improvement_pct"
        ] = safe_improvement(

            summary_row[
                "raw_source_macro_FDE"
            ],

            summary_row[
                "corrected_source_macro_FDE"
            ]
        )


        all_summary_rows.append(
            summary_row
        )


        # ====================================================
        # PER-SEQUENCE RESULTS
        # ====================================================

        for (
            sequence,
            seq_group
        ) in group.groupby(

            "sequence",

            sort=False
        ):


            seq_row = {

                "task":
                    task,

                "model":
                    model,

                "sequence":
                    sequence,

                "samples":
                    len(
                        seq_group
                    ),

                "tracks":
                    seq_group[
                        "track_key"
                    ]
                    .nunique(),

                "raw_ADE":
                    float(
                        seq_group[
                            "raw_ADE"
                        ]
                        .mean()
                    ),

                "corrected_ADE":
                    float(
                        seq_group[
                            "corrected_ADE"
                        ]
                        .mean()
                    ),

                "raw_FDE":
                    float(
                        seq_group[
                            "raw_FDE"
                        ]
                        .mean()
                    ),

                "corrected_FDE":
                    float(
                        seq_group[
                            "corrected_FDE"
                        ]
                        .mean()
                    )
            }


            seq_row[
                "ADE_improvement_pct"
            ] = safe_improvement(

                seq_row[
                    "raw_ADE"
                ],

                seq_row[
                    "corrected_ADE"
                ]
            )


            seq_row[
                "FDE_improvement_pct"
            ] = safe_improvement(

                seq_row[
                    "raw_FDE"
                ],

                seq_row[
                    "corrected_FDE"
                ]
            )


            all_sequence_rows.append(
                seq_row
            )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        all_summary_rows
    )


    per_sequence = pd.DataFrame(
        all_sequence_rows
    )


    summary.to_csv(

        OUT /
        "raw_vs_corrected_summary.csv",

        index=False
    )


    per_sequence.to_csv(

        OUT /
        "raw_vs_corrected_per_sequence.csv",

        index=False
    )


    # ========================================================
    # HUMAN-READABLE SUMMARY
    # ========================================================

    with open(

        OUT /
        "SUMMARY.txt",

        "w",

        encoding="utf-8"

    ) as file:


        file.write(

            "STEP 2B: RAW VS EGO-MOTION-CORRECTED "
            "FORECASTING ABLATION\n"
        )


        file.write(
            "=" * 78
            +
            "\n\n"
        )


        file.write(
            "Fair comparison rules:\n"
        )


        file.write(

            "- Same BIRDSAI TestReal core windows "
            "in raw and corrected representations.\n"
        )


        file.write(

            "- Raw tracks come from "
            "cleaned_track_points.csv before "
            "ego-motion compensation.\n"
        )


        file.write(

            "- Corrected tracks come from "
            "corrected_*.npz after "
            "ego-motion compensation.\n"
        )


        file.write(

            "- Both are converted from 30 Hz "
            "to 10 Hz using every third frame.\n"
        )


        file.write(

            "- Stationary and Constant-Velocity "
            "require no retraining, so this is a "
            "clean representation-level ablation.\n"
        )


        file.write(

            "- Positive improvement means "
            "corrected trajectory has lower "
            "forecasting error.\n\n"
        )


        file.write(

            summary.to_string(
                index=False
            )
        )


        file.write(

            "\n\nIMPORTANT:\n"
        )


        file.write(

            "This experiment tests whether "
            "ego-motion compensation improves "
            "short-horizon trajectory predictability. "
            "It does not establish independent "
            "metric world-coordinate ground truth.\n"
        )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        "\nDONE. Outputs written to:"
    )

    print(
        OUT
    )


    print(
        "\nMAIN SUMMARY:\n"
    )


    show_columns = [

        "task",

        "model",

        "samples",

        "tracks",

        "sequences",

        "raw_mean_ADE",

        "corrected_mean_ADE",

        "ADE_improvement_pct",

        "raw_mean_FDE",

        "corrected_mean_FDE",

        "FDE_improvement_pct",

        "raw_track_macro_ADE",

        "corrected_track_macro_ADE",

        "track_macro_ADE_improvement_pct"
    ]


    print(

        summary[
            show_columns
        ]

        .to_string(
            index=False
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()