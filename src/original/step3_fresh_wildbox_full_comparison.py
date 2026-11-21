from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# PATHS
# Existing project = READ ONLY
# All NEW outputs = Downloads
# ============================================================

ROOT = Path(
    r"E:\AIUB\Capstone Project\Journal Based Results\Main Result"
)

FRESH_DIR = (
    ROOT /
    "wildbox_fresh_confirmatory_results"
)

LSTM_GRU_DIR = (
    ROOT /
    "lstm_gru_results"
)

OUT = Path(
    r"C:\Users\HP\Downloads\step3_fresh_wildbox_full_comparison"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

FPS = 10.0
DT = 1.0 / FPS

SEEDS = [
    42,
    52,
    62
]

TASKS = {

    "2s_to_2s": {
        "future_length": 20,
        "q": 0.1,
        "r": 0.0001,
        "expected_samples": 865
    },

    "2s_to_5s": {
        "future_length": 50,
        "q": 0.1,
        "r": 0.01,
        "expected_samples": 637
    }
}


# ============================================================
# RECURRENT MODEL
# Must match original training architecture
# ============================================================

class RecurrentForecaster(nn.Module):

    def __init__(
        self,
        model_type,
        future_length,
        hidden_size=64,
        num_layers=2,
        dropout=0.10
    ):

        super().__init__()

        self.model_type = model_type
        self.future_length = future_length

        if model_type == "LSTM":

            self.encoder = nn.LSTM(
                input_size=2,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )

        elif model_type == "GRU":

            self.encoder = nn.GRU(
                input_size=2,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )

        else:

            raise ValueError(
                model_type
            )

        self.head = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                future_length * 2
            )
        )


    def forward(self, x):

        if self.model_type == "LSTM":

            _, (
                hidden,
                _
            ) = self.encoder(x)

        else:

            _, hidden = self.encoder(x)

        latent = hidden[-1]

        output = self.head(
            latent
        )

        return output.view(
            -1,
            self.future_length,
            2
        )


# ============================================================
# LOAD FROZEN GRU / LSTM
# ============================================================

def load_recurrent(
    task,
    model_type,
    seed,
    future_length
):

    checkpoint_file = (

        LSTM_GRU_DIR /
        f"{task}_{model_type}_seed{seed}.pt"
    )

    if not checkpoint_file.exists():

        raise FileNotFoundError(
            f"\nMissing frozen checkpoint:\n"
            f"{checkpoint_file}"
        )

    checkpoint = torch.load(
        checkpoint_file,
        map_location=DEVICE
    )

    hidden_size = int(
        checkpoint.get(
            "hidden_size",
            64
        )
    )

    num_layers = int(
        checkpoint.get(
            "num_layers",
            2
        )
    )

    dropout = float(
        checkpoint.get(
            "dropout",
            0.10
        )
    )

    model = RecurrentForecaster(

        model_type=model_type,

        future_length=
            future_length,

        hidden_size=
            hidden_size,

        num_layers=
            num_layers,

        dropout=
            dropout

    ).to(
        DEVICE
    )

    state_dict = (
        checkpoint["state_dict"]
        if "state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# NEURAL PREDICTION
# ============================================================

def predict_neural(
    model,
    history,
    batch_size=256
):

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(history),
            batch_size
        ):

            x = torch.tensor(

                history[
                    start:
                    start + batch_size
                ],

                dtype=torch.float32,

                device=DEVICE
            )

            pred = model(x)

            predictions.append(
                pred.cpu()
                .numpy()
            )

    return np.concatenate(
        predictions,
        axis=0
    )


# ============================================================
# INVERSE CANONICAL TRANSFORM
# ============================================================

def inverse_transform_batch(
    canonical,
    origin,
    angle,
    scale
):

    output = []

    for i in range(
        len(canonical)
    ):

        a = float(
            angle[i]
        )

        c = np.cos(a)
        s = np.sin(a)

        R = np.asarray(

            [
                [c, s],
                [-s, c]
            ],

            dtype=np.float64
        )

        local = (

            canonical[i]

            *
            float(
                scale[i]
            )

        ) @ R

        absolute = (

            local
            +
            origin[i]
        )

        output.append(
            absolute
        )

    return np.asarray(
        output,
        dtype=np.float64
    )


# ============================================================
# KALMAN FILTER
# State = [x, y, vx, vy]
# Same formulation as original experiment
# ============================================================

F = np.asarray(

    [
        [1, 0, DT, 0],
        [0, 1, 0, DT],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ],

    dtype=np.float64
)

H = np.asarray(

    [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ],

    dtype=np.float64
)

I4 = np.eye(
    4,
    dtype=np.float64
)


def make_Q(q):

    dt = DT

    return q * np.asarray(

        [
            [
                dt**4 / 4,
                0,
                dt**3 / 2,
                0
            ],

            [
                0,
                dt**4 / 4,
                0,
                dt**3 / 2
            ],

            [
                dt**3 / 2,
                0,
                dt**2,
                0
            ],

            [
                0,
                dt**3 / 2,
                0,
                dt**2
            ]
        ],

        dtype=np.float64
    )


def initial_velocity(
    history,
    n_frames=5
):

    n = min(
        n_frames,
        len(history)
    )

    points = history[:n]

    times = (
        np.arange(
            n,
            dtype=np.float64
        )
        *
        DT
    )

    tc = (
        times
        -
        times.mean()
    )

    denominator = np.sum(
        tc ** 2
    )

    if denominator <= 1e-12:

        return np.zeros(
            2,
            dtype=np.float64
        )

    pc = (

        points
        -
        points.mean(
            axis=0,
            keepdims=True
        )
    )

    velocity = (

        (
            tc[:, None]
            *
            pc
        ).sum(
            axis=0
        )

        /
        denominator
    )

    return velocity


def kalman_forecast(
    history,
    future_length,
    q,
    r
):

    Q = make_Q(q)

    R = (
        np.eye(
            2,
            dtype=np.float64
        )
        *
        r
    )

    v0 = initial_velocity(
        history
    )

    x = np.asarray(

        [
            history[0, 0],
            history[0, 1],
            v0[0],
            v0[1]
        ],

        dtype=np.float64
    )

    P = np.eye(
        4,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # FILTER OBSERVATION HISTORY
    # --------------------------------------------------------

    for z in history:

        x = F @ x

        P = (
            F
            @ P
            @ F.T
            +
            Q
        )

        innovation = (
            z
            -
            H @ x
        )

        S = (
            H
            @ P
            @ H.T
            +
            R
        )

        K = (
            P
            @ H.T
            @ np.linalg.inv(S)
        )

        x = (
            x
            +
            K @ innovation
        )

        KH = K @ H

        P = (

            (I4 - KH)
            @ P
            @ (I4 - KH).T

            +

            K
            @ R
            @ K.T
        )

    # --------------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------------

    predictions = []

    for _ in range(
        future_length
    ):

        x = F @ x

        P = (
            F
            @ P
            @ F.T
            +
            Q
        )

        predictions.append(
            x[:2].copy()
        )

    return np.asarray(
        predictions,
        dtype=np.float64
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    truth,
    metadata
):

    errors = np.linalg.norm(

        prediction
        -
        truth,

        axis=2
    )

    ADE = errors.mean(
        axis=1
    )

    FDE = errors[:, -1]

    window_ADE = float(
        ADE.mean()
    )

    window_FDE = float(
        FDE.mean()
    )

    temp = metadata[
        [
            "track_key",
            "source_group"
        ]
    ].copy()

    temp["ADE"] = ADE
    temp["FDE"] = FDE

    track_table = (

        temp.groupby(
            "track_key",
            as_index=False
        )

        .agg(
            ADE=("ADE", "mean"),
            FDE=("FDE", "mean")
        )
    )

    source_table = (

        temp.groupby(
            "source_group",
            as_index=False
        )

        .agg(
            ADE=("ADE", "mean"),
            FDE=("FDE", "mean")
        )
    )

    return {

        "ADE":
            window_ADE,

        "FDE":
            window_FDE,

        "track_macro_ADE":
            float(
                track_table[
                    "ADE"
                ].mean()
            ),

        "track_macro_FDE":
            float(
                track_table[
                    "FDE"
                ].mean()
            ),

        "source_macro_ADE":
            float(
                source_table[
                    "ADE"
                ].mean()
            ),

        "source_macro_FDE":
            float(
                source_table[
                    "FDE"
                ].mean()
            ),

        "ADE_array":
            ADE,

        "FDE_array":
            FDE
    }


# ============================================================
# SAVE PER-SAMPLE RESULTS
# ============================================================

def build_sample_table(
    task,
    model_name,
    seed,
    metadata,
    metrics
):

    table = metadata[
        [
            "sample_id",
            "source_group",
            "track_key"
        ]
    ].copy()

    table["task"] = task
    table["model"] = model_name
    table["seed"] = seed

    table["ADE"] = (
        metrics[
            "ADE_array"
        ]
    )

    table["FDE"] = (
        metrics[
            "FDE_array"
        ]
    )

    return table


# ============================================================
# MAIN
# ============================================================

def main():

    new_metric_rows = []
    sample_tables = []

    # ========================================================
    # PROCESS TASKS
    # ========================================================

    for task, config in TASKS.items():

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

        dataset_file = (

            FRESH_DIR /
            f"{task}_fresh_dataset.npz"
        )

        metadata_file = (

            FRESH_DIR /
            f"{task}_fresh_metadata.csv"
        )

        if not dataset_file.exists():

            raise FileNotFoundError(
                f"\nMissing:\n{dataset_file}"
            )

        if not metadata_file.exists():

            raise FileNotFoundError(
                f"\nMissing:\n{metadata_file}"
            )

        arrays = np.load(
            dataset_file
        )

        metadata = pd.read_csv(

            metadata_file,

            dtype={
                "source_group": str,
                "track_key": str
            }
        )

        history_abs = np.asarray(
            arrays[
                "history_abs"
            ],
            dtype=np.float64
        )

        truth_abs = np.asarray(
            arrays[
                "future_abs"
            ],
            dtype=np.float64
        )

        history_canonical = np.asarray(
            arrays[
                "history_canonical"
            ],
            dtype=np.float32
        )

        origin = np.asarray(
            arrays["origin"],
            dtype=np.float64
        )

        angle = np.asarray(
            arrays[
                "heading_angle"
            ],
            dtype=np.float64
        )

        scale = np.asarray(
            arrays["scale"],
            dtype=np.float64
        )

        print(
            "Fresh samples:",
            len(metadata)
        )

        if len(metadata) != config[
            "expected_samples"
        ]:

            print(
                "WARNING: expected",
                config[
                    "expected_samples"
                ],
                "but found",
                len(metadata)
            )

        # ====================================================
        # KALMAN
        # Fixed Q/R selected earlier using TrainReal only
        # ====================================================

        print(
            "Evaluating frozen Kalman..."
        )

        kalman_predictions = []

        for history in history_abs:

            kalman_predictions.append(

                kalman_forecast(

                    history,

                    config[
                        "future_length"
                    ],

                    q=
                        config["q"],

                    r=
                        config["r"]
                )
            )

        kalman_predictions = (
            np.stack(
                kalman_predictions
            )
        )

        kalman_metrics = (
            calculate_metrics(

                kalman_predictions,

                truth_abs,

                metadata
            )
        )

        new_metric_rows.append({

            "task":
                task,

            "model":
                "KalmanCV",

            "ADE":
                kalman_metrics["ADE"],

            "FDE":
                kalman_metrics["FDE"],

            "track_macro_ADE":
                kalman_metrics[
                    "track_macro_ADE"
                ],

            "track_macro_FDE":
                kalman_metrics[
                    "track_macro_FDE"
                ],

            "source_macro_ADE":
                kalman_metrics[
                    "source_macro_ADE"
                ],

            "source_macro_FDE":
                kalman_metrics[
                    "source_macro_FDE"
                ]
        })

        sample_tables.append(

            build_sample_table(

                task,

                "KalmanCV",

                "fixed",

                metadata,

                kalman_metrics
            )
        )

        # ====================================================
        # GRU + LSTM
        # ====================================================

        for model_type in [
            "GRU",
            "LSTM"
        ]:

            for seed in SEEDS:

                print(
                    f"Evaluating "
                    f"{model_type} "
                    f"seed {seed}..."
                )

                model = load_recurrent(

                    task,

                    model_type,

                    seed,

                    config[
                        "future_length"
                    ]
                )

                prediction_canonical = (
                    predict_neural(

                        model,

                        history_canonical
                    )
                )

                prediction_abs = (
                    inverse_transform_batch(

                        prediction_canonical,

                        origin,

                        angle,

                        scale
                    )
                )

                metrics = calculate_metrics(

                    prediction_abs,

                    truth_abs,

                    metadata
                )

                model_name = (
                    f"{model_type}_seed{seed}"
                )

                new_metric_rows.append({

                    "task":
                        task,

                    "model":
                        model_name,

                    "ADE":
                        metrics["ADE"],

                    "FDE":
                        metrics["FDE"],

                    "track_macro_ADE":
                        metrics[
                            "track_macro_ADE"
                        ],

                    "track_macro_FDE":
                        metrics[
                            "track_macro_FDE"
                        ],

                    "source_macro_ADE":
                        metrics[
                            "source_macro_ADE"
                        ],

                    "source_macro_FDE":
                        metrics[
                            "source_macro_FDE"
                        ]
                })

                sample_tables.append(

                    build_sample_table(

                        task,

                        model_type,

                        seed,

                        metadata,

                        metrics
                    )
                )

    # ========================================================
    # SAVE NEW MODEL METRICS
    # ========================================================

    new_metrics = pd.DataFrame(
        new_metric_rows
    )

    new_metrics.to_csv(

        OUT /
        "step3_new_model_metrics.csv",

        index=False
    )

    all_samples = pd.concat(
        sample_tables,
        ignore_index=True
    )

    all_samples.to_csv(

        OUT /
        "step3_new_model_sample_metrics.csv",

        index=False
    )

    # ========================================================
    # LOAD EXISTING FRESH RESULTS
    # Stationary / CV / Transformers
    # ========================================================

    existing_file = (

        FRESH_DIR /
        "fresh_holdout_model_metrics.csv"
    )

    if not existing_file.exists():

        raise FileNotFoundError(
            existing_file
        )

    existing = pd.read_csv(
        existing_file
    )

    # Remove duplicates if rerun
    existing = existing[
        ~existing["model"].isin(
            ["KalmanCV"]
        )
    ].copy()

    combined = pd.concat(

        [
            existing,
            new_metrics
        ],

        ignore_index=True
    )

    combined.to_csv(

        OUT /
        "step3_all_seed_and_model_metrics.csv",

        index=False
    )

    # ========================================================
    # FINAL COMPARISON TABLE
    # ========================================================

    final_rows = []

    metric_columns = [

        "ADE",
        "FDE",
        "track_macro_ADE",
        "track_macro_FDE",
        "source_macro_ADE",
        "source_macro_FDE"
    ]

    for task in TASKS.keys():

        task_df = combined[
            combined["task"]
            ==
            task
        ]

        # ----------------------------------------------------
        # Stationary / CV / Kalman
        # ----------------------------------------------------

        for model in [
            "Stationary",
            "ConstantVelocity",
            "KalmanCV"
        ]:

            row = task_df[
                task_df["model"]
                ==
                model
            ]

            if len(row) == 0:
                continue

            r = row.iloc[0]

            out = {
                "task": task,
                "model": model,
                "seeds": 1
            }

            for metric in metric_columns:

                out[metric] = float(
                    r[metric]
                )

                out[
                    f"{metric}_SD"
                ] = 0.0

            final_rows.append(
                out
            )

        # ----------------------------------------------------
        # GRU / LSTM three-seed MEAN METRICS
        # ----------------------------------------------------

        for base_model in [
            "GRU",
            "LSTM"
        ]:

            seed_rows = task_df[

                task_df["model"]
                .str.startswith(
                    base_model
                    +
                    "_seed"
                )
            ]

            if len(seed_rows) == 0:
                continue

            out = {

                "task":
                    task,

                "model":
                    f"{base_model}_3Seed_Mean",

                "seeds":
                    len(seed_rows)
            }

            for metric in metric_columns:

                out[metric] = float(
                    seed_rows[
                        metric
                    ].mean()
                )

                out[
                    f"{metric}_SD"
                ] = float(
                    seed_rows[
                        metric
                    ].std(
                        ddof=0
                    )
                )

            final_rows.append(
                out
            )

        # ----------------------------------------------------
        # SINGLE TRANSFORMER = mean across frozen seeds
        # ----------------------------------------------------

        transformer_seed_rows = task_df[

            task_df["model"]
            .str.match(
                r"^Transformer_seed\d+$"
            )
        ]

        if len(
            transformer_seed_rows
        ) > 0:

            out = {

                "task":
                    task,

                "model":
                    "Transformer_SingleSeed_Mean",

                "seeds":
                    len(
                        transformer_seed_rows
                    )
            }

            for metric in metric_columns:

                out[metric] = float(

                    transformer_seed_rows[
                        metric
                    ].mean()
                )

                out[
                    f"{metric}_SD"
                ] = float(

                    transformer_seed_rows[
                        metric
                    ].std(
                        ddof=0
                    )
                )

            final_rows.append(
                out
            )

        # ----------------------------------------------------
        # FINAL TRANSFORMER ENSEMBLE
        # ----------------------------------------------------

        ensemble_row = task_df[

            task_df["model"]
            ==
            "Transformer_3Seed_Ensemble"
        ]

        if len(
            ensemble_row
        ) > 0:

            r = ensemble_row.iloc[0]

            out = {

                "task":
                    task,

                "model":
                    "Transformer_3Seed_Ensemble",

                "seeds":
                    3
            }

            for metric in metric_columns:

                out[metric] = float(
                    r[metric]
                )

                out[
                    f"{metric}_SD"
                ] = 0.0

            final_rows.append(
                out
            )

    final_table = pd.DataFrame(
        final_rows
    )

    # ========================================================
    # RANK MODELS BY ADE
    # ========================================================

    final_table[
        "ADE_rank"
    ] = (

        final_table.groupby(
            "task"
        )["ADE"]
        .rank(
            method="min",
            ascending=True
        )
        .astype(int)
    )

    final_table[
        "track_macro_ADE_rank"
    ] = (

        final_table.groupby(
            "task"
        )[
            "track_macro_ADE"
        ]
        .rank(
            method="min",
            ascending=True
        )
        .astype(int)
    )

    final_table = (

        final_table
        .sort_values(
            [
                "task",
                "ADE_rank"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    final_table.to_csv(

        OUT /
        "FINAL_FRESH_WILDBOX_MODEL_COMPARISON.csv",

        index=False
    )

    # ========================================================
    # HUMAN READABLE SUMMARY
    # ========================================================

    with open(

        OUT /
        "SUMMARY.txt",

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "STEP 3: FULL FRESH WILDBOX "
            "FROZEN MODEL COMPARISON\n"
        )

        f.write(
            "=" * 78
            +
            "\n\n"
        )

        f.write(
            "All models are evaluated on the SAME "
            "locked Fresh WildBox holdout.\n"
        )

        f.write(
            "No Fresh WildBox result is used for "
            "training, parameter selection, or "
            "checkpoint selection.\n"
        )

        f.write(
            "Kalman Q/R were previously selected "
            "using BIRDSAI TrainReal only.\n"
        )

        f.write(
            "GRU/LSTM/Transformer checkpoints were "
            "already frozen before Fresh WildBox "
            "evaluation.\n\n"
        )

        f.write(
            final_table.to_string(
                index=False
            )
        )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\nSTEP 3 COMPLETE"
    )

    print(
        "\nOutputs:"
    )

    print(
        OUT
    )

    print(
        "\nFINAL FRESH WILDBOX COMPARISON:\n"
    )

    display_cols = [

        "task",
        "model",
        "ADE",
        "ADE_SD",
        "FDE",
        "track_macro_ADE",
        "source_macro_ADE",
        "ADE_rank",
        "track_macro_ADE_rank"
    ]

    print(

        final_table[
            display_cols
        ]

        .to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()