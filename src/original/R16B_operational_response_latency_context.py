
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

REVISION = DOWNLOADS / "paper_strengthening_revision"

R16A_DIR = (
    REVISION
    / "R16_latency"
    / "R16A_computational_latency_audit"
)

R16A_MANIFEST = R16A_DIR / "R16A_MANIFEST.json"
R16A_AGG = R16A_DIR / "R16A_AGGREGATE_WALLCLOCK_LATENCY.csv"

R13FC_MANIFEST = (
    REVISION
    / "R13_end_to_end_pipeline"
    / "R13F_C_end_to_end_feasibility_and_claim_lock"
    / "R13F_C_MANIFEST.json"
)

OUT = (
    REVISION
    / "R16_latency"
    / "R16B_operational_response_latency_context"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

WARNING_HORIZONS_S = [
    2.0,
    5.0,
]


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def require(path):
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            p
        )


def sha256_file(
    path,
    chunk=1024 * 1024,
):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            b = f.read(
                chunk
            )

            if not b:
                break

            h.update(
                b
            )

    return h.hexdigest()


def verify_upstream():
    for p in [
        R16A_MANIFEST,
        R16A_AGG,
        R13FC_MANIFEST,
    ]:
        require(
            p
        )

    r16a = json.loads(
        R16A_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    r13 = json.loads(
        R13FC_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_r16a = (
        "R16A_COMPUTATIONAL_LATENCY_AUDIT_COMPLETE_CPU_TIMING_LOCKED_READY_FOR_R16B_OPERATIONAL_RESPONSE_LATENCY_CONTEXT"
    )

    expected_r13 = (
        "R13_END_TO_END_FEASIBILITY_FULLY_ASSESSED_DETECTOR_BOTTLENECK_LOCKED_REPRESENTATIVE_FORECAST_PROPAGATION_NOT_SUPPORTED_READY_FOR_NEXT_STRENGTHENING_STEP"
    )

    if (
        r16a.get(
            "status"
        )
        !=
        expected_r16a
    ):
        raise RuntimeError(
            "Unexpected R16-A status: "
            + str(
                r16a.get(
                    "status"
                )
            )
        )

    if (
        r13.get(
            "status"
        )
        !=
        expected_r13
    ):
        raise RuntimeError(
            "Unexpected R13-F-C status: "
            + str(
                r13.get(
                    "status"
                )
            )
        )

    return (
        r16a,
        r13,
    )


def build_response_budget(
    agg,
):
    rows = []

    # Only detector+tracker pipelines matter for operational response context.
    use = agg.loc[
        agg[
            "component"
        ].isin(
            [
                "bytetrack",
                "botsort",
            ]
        )
    ].copy()

    for _, r in use.iterrows():
        compute_ms = float(
            r[
                "weighted_wall_ms_per_frame"
            ]
        )

        effective_fps = float(
            r[
                "effective_fps"
            ]
        )

        for horizon_s in WARNING_HORIZONS_S:
            horizon_ms = (
                horizon_s
                *
                1000.0
            )

            remaining_ms = (
                horizon_ms
                -
                compute_ms
            )

            rows.append(
                {
                    "component":
                        r[
                            "component"
                        ],

                    "component_label":
                        r[
                            "component_label"
                        ],

                    "effective_fps_cpu":
                        effective_fps,

                    "warning_horizon_seconds":
                        horizon_s,

                    "warning_horizon_ms":
                        horizon_ms,

                    "measured_compute_ms_per_frame":
                        compute_ms,

                    "compute_fraction_of_horizon":
                        compute_ms
                        /
                        horizon_ms,

                    "remaining_noncompute_budget_ms":
                        remaining_ms,

                    "remaining_noncompute_budget_seconds":
                        remaining_ms
                        /
                        1000.0,

                    "operational_feasibility_condition":
                        (
                            "L_comm + L_decision + L_intervention + "
                            "L_human <= remaining_noncompute_budget"
                        ),

                    "operational_sufficiency_measured":
                        False,

                    "reason":
                        (
                            "Communication, decision, intervention/drone-response, "
                            "and human-response latencies were not measured."
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_claim_matrix(
    response_df,
):
    byte2 = response_df.loc[
        (
            response_df[
                "component"
            ]
            ==
            "bytetrack"
        )
        &
        (
            response_df[
                "warning_horizon_seconds"
            ]
            ==
            2.0
        )
    ].iloc[
        0
    ]

    byte5 = response_df.loc[
        (
            response_df[
                "component"
            ]
            ==
            "bytetrack"
        )
        &
        (
            response_df[
                "warning_horizon_seconds"
            ]
            ==
            5.0
        )
    ].iloc[
        0
    ]

    bot2 = response_df.loc[
        (
            response_df[
                "component"
            ]
            ==
            "botsort"
        )
        &
        (
            response_df[
                "warning_horizon_seconds"
            ]
            ==
            2.0
        )
    ].iloc[
        0
    ]

    bot5 = response_df.loc[
        (
            response_df[
                "component"
            ]
            ==
            "botsort"
        )
        &
        (
            response_df[
                "warning_horizon_seconds"
            ]
            ==
            5.0
        )
    ].iloc[
        0
    ]

    claims = pd.DataFrame(
        [
            {
                "claim":
                    "The 2-s forecast horizon is equivalent to a 2-s operational response time.",

                "status":
                    "REJECTED",

                "basis":
                    (
                        "Prediction horizon and total operational response latency are different quantities."
                    ),
            },
            {
                "claim":
                    "The 5-s forecast horizon is equivalent to a 5-s operational response time.",

                "status":
                    "REJECTED",

                "basis":
                    (
                        "Prediction horizon and total operational response latency are different quantities."
                    ),
            },
            {
                "claim":
                    "ByteTrack compute alone fits inside a 2-s horizon.",

                "status":
                    "SUPPORTED",

                "basis":
                    (
                        f"Measured CPU compute={float(byte2['measured_compute_ms_per_frame']):.3f} ms; "
                        f"remaining non-compute budget={float(byte2['remaining_noncompute_budget_ms']):.3f} ms."
                    ),
            },
            {
                "claim":
                    "ByteTrack compute alone fits inside a 5-s horizon.",

                "status":
                    "SUPPORTED",

                "basis":
                    (
                        f"Measured CPU compute={float(byte5['measured_compute_ms_per_frame']):.3f} ms; "
                        f"remaining non-compute budget={float(byte5['remaining_noncompute_budget_ms']):.3f} ms."
                    ),
            },
            {
                "claim":
                    "BoT-SORT compute alone fits inside a 2-s horizon.",

                "status":
                    "SUPPORTED",

                "basis":
                    (
                        f"Measured CPU compute={float(bot2['measured_compute_ms_per_frame']):.3f} ms; "
                        f"remaining non-compute budget={float(bot2['remaining_noncompute_budget_ms']):.3f} ms."
                    ),
            },
            {
                "claim":
                    "BoT-SORT compute alone fits inside a 5-s horizon.",

                "status":
                    "SUPPORTED",

                "basis":
                    (
                        f"Measured CPU compute={float(bot5['measured_compute_ms_per_frame']):.3f} ms; "
                        f"remaining non-compute budget={float(bot5['remaining_noncompute_budget_ms']):.3f} ms."
                    ),
            },
            {
                "claim":
                    "The current experiment proves that a human team can respond within the 2-s horizon.",

                "status":
                    "REJECTED",

                "basis":
                    (
                        "Human-response latency was not measured."
                    ),
            },
            {
                "claim":
                    "The current experiment proves that a human team can respond within the 5-s horizon.",

                "status":
                    "REJECTED",

                "basis":
                    (
                        "Human-response latency was not measured."
                    ),
            },
            {
                "claim":
                    "The current experiment proves that a drone can be dispatched and intervene within 2 s or 5 s.",

                "status":
                    "REJECTED",

                "basis":
                    (
                        "Drone availability, dispatch, travel, intervention, and communication latencies were not measured."
                    ),
            },
            {
                "claim":
                    "The current CPU implementation is real-time at the standardized 10-Hz trajectory rate.",

                "status":
                    "REJECTED",

                "basis":
                    (
                        "R16-A measured 8.735 FPS for detector+ByteTrack and 7.764 FPS for detector+BoT-SORT, "
                        "both below 10 FPS."
                    ),
            },
        ]
    )

    return claims


def build_latency_equation():
    rows = [
        {
            "term":
                "L_total",

            "meaning":
                "Total alert-to-action operational latency",

            "measured_here":
                False,

            "definition":
                (
                    "L_compute + L_comm + L_decision + L_intervention + L_human"
                ),
        },
        {
            "term":
                "L_compute",

            "meaning":
                "Measured detector+tracker computational latency",

            "measured_here":
                True,

            "definition":
                (
                    "Frozen R13-D CPU wall-clock latency per processed frame"
                ),
        },
        {
            "term":
                "L_comm",

            "meaning":
                "Communication / networking delay",

            "measured_here":
                False,

            "definition":
                "Not measured in current experiment",
        },
        {
            "term":
                "L_decision",

            "meaning":
                "Decision / alarm logic latency beyond measured detector+tracker processing",

            "measured_here":
                False,

            "definition":
                "Not separately measured in current experiment",
        },
        {
            "term":
                "L_intervention",

            "meaning":
                "Drone availability, dispatch, travel, or automated intervention delay",

            "measured_here":
                False,

            "definition":
                "Not measured in current experiment",
        },
        {
            "term":
                "L_human",

            "meaning":
                "Human acknowledgement / response delay",

            "measured_here":
                False,

            "definition":
                "Not measured in current experiment",
        },
    ]

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR16-B — OPERATIONAL RESPONSE LATENCY CONTEXT / CLAIM LOCK"
    )

    print(
        "NO NEW INFERENCE. NO RETRAINING. NO TUNING. "
        "This step converts the frozen R16-A computational timing into a reviewer-safe latency budget. "
        "It does not invent communication, drone-dispatch, intervention, or human-response times. "
        "Instead, it reports the maximum remaining non-compute latency that would have to contain all of those components "
        "for a 2-s or 5-s forecast horizon to be operationally actionable.\n"
    )

    r16a, r13 = verify_upstream()

    agg = pd.read_csv(
        R16A_AGG,
        low_memory=False,
    )

    response_df = build_response_budget(
        agg
    )

    claims = build_claim_matrix(
        response_df
    )

    equation_df = build_latency_equation()

    response_path = (
        OUT
        /
        "R16B_OPERATIONAL_RESPONSE_BUDGET.csv"
    )

    claims_path = (
        OUT
        /
        "R16B_OPERATIONAL_LATENCY_CLAIM_MATRIX.csv"
    )

    equation_path = (
        OUT
        /
        "R16B_LATENCY_TERM_DEFINITION.csv"
    )

    response_df.to_csv(
        response_path,
        index=False,
    )

    claims.to_csv(
        claims_path,
        index=False,
    )

    equation_df.to_csv(
        equation_path,
        index=False,
    )

    byte2 = response_df.loc[
        (
            response_df[
                "component"
            ]
            ==
            "bytetrack"
        )
        &
        (
            response_df[
                "warning_horizon_seconds"
            ]
            ==
            2.0
        )
    ].iloc[
        0
    ]

    byte5 = response_df.loc[
        (
            response_df[
                "component"
            ]
            ==
            "bytetrack"
        )
        &
        (
            response_df[
                "warning_horizon_seconds"
            ]
            ==
            5.0
        )
    ].iloc[
        0
    ]

    bot2 = response_df.loc[
        (
            response_df[
                "component"
            ]
            ==
            "botsort"
        )
        &
        (
            response_df[
                "warning_horizon_seconds"
            ]
            ==
            2.0
        )
    ].iloc[
        0
    ]

    bot5 = response_df.loc[
        (
            response_df[
                "component"
            ]
            ==
            "botsort"
        )
        &
        (
            response_df[
                "warning_horizon_seconds"
            ]
            ==
            5.0
        )
    ].iloc[
        0
    ]

    manuscript = f"""R16-B operational response latency context — locked wording

The 2-s and 5-s quantities in this study are forecasting horizons, not measured human or field-response times. Computational timing was measured separately from the frozen CPU detector/tracker experiment. The detector+ByteTrack path required {float(byte2['measured_compute_ms_per_frame']):.3f} ms per processed frame, leaving at most {float(byte2['remaining_noncompute_budget_seconds']):.3f} s within a nominal 2-s horizon and {float(byte5['remaining_noncompute_budget_seconds']):.3f} s within a nominal 5-s horizon for all non-computational delays combined. The detector+BoT-SORT path required {float(bot2['measured_compute_ms_per_frame']):.3f} ms per processed frame, leaving at most {float(bot2['remaining_noncompute_budget_seconds']):.3f} s and {float(bot5['remaining_noncompute_budget_seconds']):.3f} s, respectively.

Operational feasibility would require:

L_total = L_compute + L_comm + L_decision + L_intervention + L_human

and, for a given warning horizon H:

L_total <= H.

Only L_compute was measured in the present experiment. Communication delay, alarm/decision overhead, drone availability or intervention latency, and human response time were not measured. Therefore, the current results do not establish that either the 2-s or 5-s forecasting horizon is sufficient for real-world human or drone intervention.

The CPU implementation also did not sustain the standardized 10-Hz processing rate: detector+ByteTrack achieved 8.735 FPS and detector+BoT-SORT achieved 7.764 FPS. Thus, the current CPU benchmark should be treated as an offline/research timing result rather than a demonstrated real-time deployment configuration.
"""

    manuscript_path = (
        OUT
        /
        "R16B_MANUSCRIPT_READY_OPERATIONAL_LATENCY_CONTEXT.txt"
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8",
    )

    status = (
        "R16_OPERATIONAL_LATENCY_CONTEXT_FULLY_LOCKED_COMPUTE_MEASURED_NONCOMPUTE_UNMEASURED_NO_2S_5S_RESPONSE_SUFFICIENCY_CLAIM_READY_FOR_R17_MIXED_SPECIES_ROBUSTNESS"
    )

    manifest = {
        "step":
            "R16B_OPERATIONAL_RESPONSE_LATENCY_CONTEXT",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "new_inference":
            False,

        "tuning":
            False,

        "prediction_horizon_equals_response_time":
            False,

        "human_response_measured":
            False,

        "communication_latency_measured":
            False,

        "drone_dispatch_or_intervention_latency_measured":
            False,

        "operational_sufficiency_claim_allowed":
            False,

        "cpu_10hz_realtime_demonstrated":
            False,

        "outputs": {
            "response_budget":
                str(
                    response_path
                ),

            "claim_matrix":
                str(
                    claims_path
                ),

            "latency_terms":
                str(
                    equation_path
                ),

            "manuscript_context":
                str(
                    manuscript_path
                ),
        },
    }

    manifest_path = (
        OUT
        /
        "R16B_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "OPERATIONAL RESPONSE BUDGET:"
    )

    print(
        response_df.to_string(
            index=False
        )
    )

    print(
        "\nLATENCY TERM DEFINITION:"
    )

    print(
        equation_df.to_string(
            index=False
        )
    )

    print(
        "\nCLAIM MATRIX:"
    )

    print(
        claims.to_string(
            index=False
        )
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR16-B COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
