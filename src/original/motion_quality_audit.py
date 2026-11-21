from pathlib import Path
import pandas as pd
import numpy as np

# =====================================================
# PATHS
# =====================================================

MOTION_FILE = Path(
    r"C:\Users\HP\Downloads\ego_motion_full\all_ego_motion.csv"
)

WINDOW_2_FILE = Path(
    r"C:\Users\HP\Downloads\trajectory_dataset\windows_2s_to_2s.csv"
)

WINDOW_5_FILE = Path(
    r"C:\Users\HP\Downloads\trajectory_dataset\windows_2s_to_5s.csv"
)

OUT = Path(
    r"C:\Users\HP\Downloads\motion_quality_audit"
)

OUT.mkdir(exist_ok=True)


# =====================================================
# LOAD
# =====================================================

motion = pd.read_csv(MOTION_FILE)

windows2 = pd.read_csv(WINDOW_2_FILE)

windows5 = pd.read_csv(WINDOW_5_FILE)


# Convert naming
motion["split"] = motion["split"].replace({
    "train": "train_real",
    "test": "test_real"
})


# =====================================================
# UNRELIABLE RUN ANALYSIS
# =====================================================

run_results = []


for (split, sequence), group in motion.groupby(
    ["split", "sequence"]
):

    group = group.sort_values("pair").reset_index(drop=True)

    bad = (
        group["reliable"]
        .fillna(0)
        .astype(int)
        == 0
    ).astype(int)


    runs = []

    current = 0

    for value in bad:

        if value == 1:

            current += 1

        else:

            if current > 0:

                runs.append(current)

            current = 0


    if current > 0:

        runs.append(current)


    if len(runs) == 0:

        max_run = 0
        median_run = 0

    else:

        max_run = max(runs)
        median_run = np.median(runs)


    run_results.append({

        "split":
            split,

        "sequence":
            sequence,

        "total_pairs":
            len(group),

        "reliable_pairs":
            int(
                (group["reliable"] == 1).sum()
            ),

        "unreliable_pairs":
            int(
                (group["reliable"] != 1).sum()
            ),

        "reliable_pct":
            100 * (
                group["reliable"] == 1
            ).mean(),

        "number_bad_runs":
            len(runs),

        "max_bad_run":
            max_run,

        "median_bad_run":
            median_run,

        "bad_runs_1_frame":
            sum(
                r == 1
                for r in runs
            ),

        "bad_runs_2_to_3":
            sum(
                2 <= r <= 3
                for r in runs
            ),

        "bad_runs_4_to_5":
            sum(
                4 <= r <= 5
                for r in runs
            ),

        "bad_runs_gt_5":
            sum(
                r > 5
                for r in runs
            )
    })


run_df = pd.DataFrame(
    run_results
)

run_df.to_csv(
    OUT / "unreliable_run_statistics.csv",
    index=False
)


# =====================================================
# CHECK MOTION QUALITY OF A FORECAST WINDOW
# =====================================================

def audit_windows(window_df, horizon_name):

    result = []

    # Create lookup once
    motion_groups = {}

    for key, group in motion.groupby(
        ["split", "sequence"]
    ):

        motion_groups[key] = (
            group
            .set_index("pair")
            .sort_index()
        )


    for _, row in window_df.iterrows():

        split = row["split"]

        sequence = str(
            row["sequence"]
        )


        # pandas can sometimes remove leading zeroes
        # Recover expected 20-char sequence string if needed
        matches = [

            key

            for key
            in motion_groups.keys()

            if (
                key[0] == split
                and
                str(key[1]) == sequence
            )
        ]


        if len(matches) == 0:

            # try integer/string-safe comparison

            matches = [

                key

                for key
                in motion_groups.keys()

                if (
                    key[0] == split
                    and
                    str(key[1]).lstrip("0")
                    ==
                    sequence.lstrip("0")
                )
            ]


        if len(matches) == 0:

            continue


        key = matches[0]

        mg = motion_groups[key]


        start_frame = int(
            row["obs_start_frame"]
        )

        future_end = int(
            row["future_end_frame"]
        )


        # Need transforms:
        # start -> start+1 ... future_end
        #
        # pair n represents frame n -> n+1
        interval = mg.loc[
            (
                mg.index >= start_frame
            )
            &
            (
                mg.index < future_end
            )
        ].copy()


        expected_pairs = (
            future_end
            -
            start_frame
        )


        if expected_pairs <= 0:

            continue


        reliable_values = (

            interval["reliable"]
            .fillna(0)
            .astype(int)
            .tolist()
        )


        # If pair rows are missing, count those
        # missing transforms as unreliable.
        missing_pairs = max(
            0,
            expected_pairs
            -
            len(reliable_values)
        )


        reliable_count = sum(
            reliable_values
        )


        reliable_pct = (

            100
            *
            reliable_count
            /
            expected_pairs
        )


        # --------------------------
        # Longest bad run
        # --------------------------

        max_bad_run = 0
        current_bad = 0


        for value in reliable_values:

            if value == 0:

                current_bad += 1

                max_bad_run = max(
                    max_bad_run,
                    current_bad
                )

            else:

                current_bad = 0


        # missing pairs are automatically unsafe

        if missing_pairs > 0:

            max_bad_run = max(
                max_bad_run,
                missing_pairs
            )


        # --------------------------
        # Quality Levels
        # --------------------------

        # STRICT:
        # almost all motion estimates reliable,
        # no continuous bad period >3 frames

        strict_ok = int(

            reliable_pct >= 95

            and

            max_bad_run <= 3

            and

            missing_pairs == 0
        )


        # MODERATE:
        # potentially recoverable with fallback

        moderate_ok = int(

            reliable_pct >= 90

            and

            max_bad_run <= 5

            and

            missing_pairs == 0
        )


        result.append({

            "split":
                split,

            "sequence":
                key[1],

            "object_id":
                row["object_id"],

            "segment_id":
                row["segment_id"],

            "obs_start_frame":
                start_frame,

            "obs_end_frame":
                int(
                    row["obs_end_frame"]
                ),

            "future_end_frame":
                future_end,

            "horizon":
                horizon_name,

            "expected_motion_pairs":
                expected_pairs,

            "reliable_motion_pairs":
                reliable_count,

            "reliable_motion_pct":
                reliable_pct,

            "missing_motion_pairs":
                missing_pairs,

            "max_bad_run":
                max_bad_run,

            "strict_ok":
                strict_ok,

            "moderate_ok":
                moderate_ok
        })


    return pd.DataFrame(result)


# =====================================================
# PROCESS WINDOWS
# =====================================================

quality2 = audit_windows(
    windows2,
    "2s_to_2s"
)

quality5 = audit_windows(
    windows5,
    "2s_to_5s"
)


quality2.to_csv(
    OUT / "quality_windows_2s_to_2s.csv",
    index=False
)


quality5.to_csv(
    OUT / "quality_windows_2s_to_5s.csv",
    index=False
)


# =====================================================
# SUMMARY TABLE
# =====================================================

summary_rows = []


for name, df in [

    ("2s_to_2s", quality2),

    ("2s_to_5s", quality5)

]:

    for split in [
        "train_real",
        "test_real"
    ]:

        temp = df[
            df["split"] == split
        ]


        summary_rows.append({

            "forecast":
                name,

            "split":
                split,

            "total_windows":
                len(temp),

            "strict_windows":
                int(
                    temp[
                        "strict_ok"
                    ].sum()
                ),

            "strict_pct":
                (
                    100
                    *
                    temp[
                        "strict_ok"
                    ].mean()
                )
                if len(temp)
                else 0,

            "moderate_windows":
                int(
                    temp[
                        "moderate_ok"
                    ].sum()
                ),

            "moderate_pct":
                (
                    100
                    *
                    temp[
                        "moderate_ok"
                    ].mean()
                )
                if len(temp)
                else 0
        })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    OUT / "window_quality_summary.csv",
    index=False
)


# =====================================================
# PER-SEQUENCE WINDOW SUMMARY
# =====================================================

sequence_summary = []


for name, df in [

    ("2s_to_2s", quality2),

    ("2s_to_5s", quality5)

]:

    for (
        split,
        sequence
    ), group in df.groupby(
        [
            "split",
            "sequence"
        ]
    ):


        sequence_summary.append({

            "forecast":
                name,

            "split":
                split,

            "sequence":
                sequence,

            "windows":
                len(group),

            "strict":
                int(
                    group[
                        "strict_ok"
                    ].sum()
                ),

            "strict_pct":
                100
                *
                group[
                    "strict_ok"
                ].mean(),

            "moderate":
                int(
                    group[
                        "moderate_ok"
                    ].sum()
                ),

            "moderate_pct":
                100
                *
                group[
                    "moderate_ok"
                ].mean(),

            "median_motion_reliable_pct":
                group[
                    "reliable_motion_pct"
                ].median(),

            "max_bad_run_seen":
                group[
                    "max_bad_run"
                ].max()
        })


sequence_summary_df = pd.DataFrame(
    sequence_summary
)


sequence_summary_df.to_csv(

    OUT
    /
    "per_sequence_window_quality.csv",

    index=False
)


# =====================================================
# TEXT SUMMARY
# =====================================================

with open(

    OUT / "SUMMARY.txt",

    "w"

) as f:


    f.write(
        "MOTION QUALITY AUDIT\n"
    )

    f.write(
        "=" * 65
        +
        "\n\n"
    )


    f.write(
        "UNRELIABLE MOTION RUNS\n\n"
    )


    for _, row in run_df.iterrows():

        f.write(

            f"{row['split']} "
            f"{row['sequence']}\n"

        )

        f.write(

            f"Reliable: "
            f"{row['reliable_pct']:.2f}%\n"

        )

        f.write(

            f"Bad runs: "
            f"{int(row['number_bad_runs'])}\n"

        )

        f.write(

            f"Maximum bad run: "
            f"{int(row['max_bad_run'])} frames\n"

        )

        f.write(

            f"Bad runs >5 frames: "
            f"{int(row['bad_runs_gt_5'])}\n\n"

        )


    f.write(
        "\nFORECAST WINDOW QUALITY\n\n"
    )


    for _, row in summary_df.iterrows():

        f.write(

            f"{row['forecast']} "
            f"{row['split']}\n"

        )

        f.write(

            f"Total windows: "
            f"{int(row['total_windows'])}\n"

        )

        f.write(

            f"Strict usable: "
            f"{int(row['strict_windows'])} "
            f"({row['strict_pct']:.2f}%)\n"

        )

        f.write(

            f"Moderate usable: "
            f"{int(row['moderate_windows'])} "
            f"({row['moderate_pct']:.2f}%)\n\n"

        )


print(
    "\nMOTION QUALITY AUDIT COMPLETE"
)

print(summary_df)

print(
    "\nResults saved:"
)

print(OUT)