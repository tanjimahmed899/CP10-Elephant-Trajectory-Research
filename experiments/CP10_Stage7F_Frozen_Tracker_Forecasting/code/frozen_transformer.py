"""Frozen Transformer architecture and canonical transforms copied verbatim from
CP10 Stage 5's previously verified evaluation code. No training implementation.
"""
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
EPS=1e-10
HEADING_FRAMES=10
DEVICE=torch.device('cpu')
ROOT=Path(os.environ.get('CP10_JOURNAL_ROOT','/mnt/e/AIUB/Capstone Project/Journal Based Results'))
MODEL_ROOT=ROOT/'Main Result'/'transformer_results'

def heading_angle_from_history(
    history
):

    recent = history[
        -min(
            HEADING_FRAMES,
            len(
                history
            )
        ):
    ]


    t = np.arange(
        len(
            recent
        ),
        dtype=np.float64
    )


    tc = (
        t
        -
        t.mean()
    )


    denominator = np.sum(
        tc ** 2
    )


    if denominator <= EPS:

        return 0.0


    x_centered = (
        recent[
            :,
            0
        ]
        -
        recent[
            :,
            0
        ].mean()
    )


    y_centered = (
        recent[
            :,
            1
        ]
        -
        recent[
            :,
            1
        ].mean()
    )


    vx = (
        np.sum(
            tc
            *
            x_centered
        )
        /
        denominator
    )


    vy = (
        np.sum(
            tc
            *
            y_centered
        )
        /
        denominator
    )


    if np.hypot(
        vx,
        vy
    ) <= EPS:

        return 0.0


    return float(
        np.arctan2(
            vy,
            vx
        )
    )


def canonicalize_window(
    history_abs,
    future_abs,
    scale_floor
):

    origin = history_abs[
        -1
    ].copy()


    angle = heading_angle_from_history(
        history_abs
    )


    history_steps = np.linalg.norm(

        np.diff(
            history_abs,
            axis=0
        ),

        axis=1

    )


    path_length = float(
        history_steps.sum()
    )


    scale = max(
        path_length,
        scale_floor
    )


    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )


    # inverse rotation used during reconstruction:
    #
    # canonical @ R -> absolute-local

    R = np.asarray(

        [
            [c, s],
            [-s, c]
        ],

        dtype=np.float64

    )


    history_can = (

        (
            history_abs
            -
            origin
        )

        @
        R.T

        /
        scale

    )


    future_can = (

        (
            future_abs
            -
            origin
        )

        @
        R.T

        /
        scale

    )


    return {
        "history_canonical":
            history_can.astype(
                np.float32
            ),

        "future_canonical":
            future_can.astype(
                np.float32
            ),

        "origin":
            origin.astype(
                np.float64
            ),

        "heading_angle":
            float(
                angle
            ),

        "scale":
            float(
                scale
            ),

        "history_path":
            path_length,
    }


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=32
    ):

        super().__init__()


        pe = torch.zeros(
            max_len,
            d_model
        )


        position = torch.arange(

            max_len,

            dtype=torch.float32

        ).unsqueeze(
            1
        )


        div_term = torch.exp(

            torch.arange(

                0,
                d_model,
                2,

                dtype=torch.float32

            )

            *
            (
                -math.log(
                    10000.0
                )
                /
                d_model
            )

        )


        pe[
            :,
            0::2
        ] = torch.sin(

            position
            *
            div_term

        )


        pe[
            :,
            1::2
        ] = torch.cos(

            position
            *
            div_term

        )


        self.register_buffer(

            "pe",

            pe.unsqueeze(
                0
            )

        )


    def forward(
        self,
        x
    ):

        return (

            x

            +
            self.pe[
                :,
                :x.size(
                    1
                )
            ]

        )


class TrajectoryTransformer(nn.Module):

    def __init__(
        self,
        future_length,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.10
    ):

        super().__init__()


        self.future_length = (
            future_length
        )


        self.input_projection = nn.Linear(
            2,
            d_model
        )


        self.position = PositionalEncoding(
            d_model,
            max_len=32
        )


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=dim_feedforward,

            dropout=dropout,

            activation="gelu",

            batch_first=True,

            norm_first=True

        )


        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )


        self.norm = nn.LayerNorm(
            d_model
        )


        self.head = nn.Sequential(

            nn.Linear(
                d_model,
                d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model,
                future_length * 2
            )

        )


    def forward(
        self,
        history
    ):

        x = self.input_projection(
            history
        )


        x = self.position(
            x
        )


        x = self.encoder(
            x
        )


        x = x.mean(
            dim=1
        )


        x = self.norm(
            x
        )


        return (

            self.head(
                x
            )

            .view(
                -1,
                self.future_length,
                2
            )

        )


def load_transformer(
    task,
    seed,
    future_length
):

    checkpoint_file = (

        MODEL_ROOT
        /
        f"{task}_Transformer_seed{seed}.pt"

    )


    if not checkpoint_file.exists():

        raise FileNotFoundError(
            f"Missing frozen checkpoint:\n"
            f"{checkpoint_file}"
        )


    checkpoint = torch.load(

        checkpoint_file,

        map_location=DEVICE

    )


    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint


    model = TrajectoryTransformer(

        future_length=
            future_length,

        d_model=
            int(
                checkpoint.get(
                    "d_model",
                    64
                )
            ),

        nhead=
            int(
                checkpoint.get(
                    "nhead",
                    4
                )
            ),

        num_layers=
            int(
                checkpoint.get(
                    "num_layers",
                    2
                )
            ),

        dim_feedforward=
            int(
                checkpoint.get(
                    "dim_feedforward",
                    128
                )
            ),

        dropout=
            float(
                checkpoint.get(
                    "dropout",
                    0.10
                )
            ),

    ).to(
        DEVICE
    )


    # Positional encoding is deterministic.
    state_dict = {

        k: v

        for k, v
        in state_dict.items()

        if k != "position.pe"

    }


    result = model.load_state_dict(

        state_dict,

        strict=False

    )


    unexpected_missing = [

        k
        for k in result.missing_keys
        if k != "position.pe"

    ]


    if (
        unexpected_missing
        or
        result.unexpected_keys
    ):

        raise RuntimeError(
            "Checkpoint architecture mismatch.\n"
            f"Missing: {unexpected_missing}\n"
            f"Unexpected: {result.unexpected_keys}"
        )


    model.eval()


    return model


def predict(
    model,
    histories,
    batch_size=256
):

    predictions = []


    with torch.no_grad():

        for start in range(

            0,
            len(
                histories
            ),
            batch_size

        ):


            x = torch.tensor(

                histories[
                    start:
                    start
                    +
                    batch_size
                ],

                dtype=torch.float32,

                device=DEVICE

            )


            y = model(
                x
            )


            predictions.append(

                y.cpu().numpy()

            )


    return np.concatenate(
        predictions,
        axis=0
    )


def inverse_transform_batch(
    canonical,
    origin,
    angle,
    scale
):

    output = []


    for i in range(
        len(
            canonical
        )
    ):

        a = float(
            angle[
                i
            ]
        )


        c = np.cos(
            a
        )

        s = np.sin(
            a
        )


        R = np.asarray(

            [
                [c, s],
                [-s, c]
            ],

            dtype=np.float64

        )


        local = (

            canonical[
                i
            ]

            *
            float(
                scale[
                    i
                ]
            )

        ) @ R


        output.append(

            local
            +
            origin[
                i
            ]

        )


    return np.asarray(
        output,
        dtype=np.float64
    )
