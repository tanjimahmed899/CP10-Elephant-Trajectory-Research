
from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")
REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R11_ROOT = REVISION_ROOT / "R11_noise_floor_analysis"

R11B_DIR = R11_ROOT / "R11B_empirical_geometric_residual_floor_protocol_lock"
R11B_MANIFEST = R11B_DIR / "R11B_MANIFEST.json"

R11C_DIR = R11_ROOT / "R11C_testreal_forecast_error_vs_geometric_residual_floor"
R11C_MANIFEST = R11C_DIR / "R11C_MANIFEST.json"
R11C_SUMMARY = R11C_DIR / "R11C_TESTREAL_NOISE_FLOOR_SUMMARY.csv"
R11C_BOOT = R11C_DIR / "R11C_SEGMENT_CLUSTER_BOOTSTRAP.csv"

R11D_DIR = R11_ROOT / "R11D_ensemble_gain_vs_residual_scale"
R11D_MANIFEST = R11D_DIR / "R11D_MANIFEST.json"
R11D_SUMMARY = R11D_DIR / "R11D_ENSEMBLE_GAIN_VS_RESIDUAL_SCALE_SUMMARY.csv"
R11D_BOOT = R11D_DIR / "R11D_SEGMENT_CLUSTER_BOOTSTRAP_GAIN_SCALE.csv"
R11D_LOO = R11D_DIR / "R11D_LEAVE_ONE_SEQUENCE_OUT_GAIN_SCALE.csv"

OUT = R11_ROOT / "R11E_final_noise_floor_conclusion"
OUT.mkdir(parents=True, exist_ok=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def verify_upstream():
    for p in [
        R11B_MANIFEST,
        R11C_MANIFEST,
        R11C_SUMMARY,
        R11C_BOOT,
        R11D_MANIFEST,
        R11D_SUMMARY,
        R11D_BOOT,
        R11D_LOO,
    ]:
        require(p)

    b = json.loads(
        R11B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    c = json.loads(
        R11C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    d = json.loads(
        R11D_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_b = (
        "R11B_TESTREAL_EMPIRICAL_GEOMETRIC_RESIDUAL_F50_F95_LOCKED_READY_FOR_R11C_FORECAST_ERROR_ABOVE_FLOOR_ANALYSIS"
    )

    expected_c = (
        "R11C_TESTREAL_FORECAST_ERROR_RELATIVE_TO_EMPIRICAL_RESIDUAL_FLOOR_QUANTIFIED_READY_FOR_R11D_NOISE_FLOOR_CONCLUSION"
    )

    expected_d = (
        "R11D_ENSEMBLE_GAIN_CONTEXTUALIZED_AGAINST_F50_F95_READY_FOR_R11E_FINAL_NOISE_FLOOR_CONCLUSION"
    )

    if b.get("status") != expected_b:
        raise RuntimeError(
            f"Unexpected R11-B status: {b.get('status')}"
        )

    if c.get("status") != expected_c:
        raise RuntimeError(
            f"Unexpected R11-C status: {c.get('status')}"
        )

    if d.get("status") != expected_d:
        raise RuntimeError(
            f"Unexpected R11-D status: {d.get('status')}"
        )

    return b, c, d


def lookup(df, **filters):
    work = df.copy()

    for key, value in filters.items():
        work = work[
            work[key].astype(str)
            ==
            str(value)
        ]

    if len(work) != 1:
        raise RuntimeError(
            f"Expected exactly one row for {filters}, found {len(work)}"
        )

    return work.iloc[0]


def main():
    print(
        "\nR11-E — FINAL EMPIRICAL NOISE-FLOOR / RESIDUAL-SCALE CONCLUSION"
    )

    print(
        "CONCLUSION ONLY. NO training, NO inference, NO tuning. "
        "This step locks a reviewer-safe interpretation of the BIRDSAI TestReal "
        "post-correction geometric residual references and their relationship to "
        "absolute forecasting error and Transformer ensemble gains.\n"
    )

    b, _, _ = verify_upstream()

    f50 = float(
        b["F50_px"]
    )

    f95 = float(
        b["F95_px"]
    )

    c_summary = pd.read_csv(
        R11C_SUMMARY
    )

    c_boot = pd.read_csv(
        R11C_BOOT
    )

    d_summary = pd.read_csv(
        R11D_SUMMARY
    )

    d_boot = pd.read_csv(
        R11D_BOOT
    )

    d_loo = pd.read_csv(
        R11D_LOO
    )

    # Absolute forecast-error evidence.
    abs_rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        row = lookup(
            c_summary,
            task=task,
        )

        abs_rows.append(
            {
                "task":
                    task,

                "F50_px":
                    f50,

                "F95_px":
                    f95,

                "ensemble_ADE_px":
                    float(
                        row[
                            "ensemble_ADE_px_window_mean"
                        ]
                    ),

                "ensemble_FDE_px":
                    float(
                        row[
                            "ensemble_FDE_px_window_mean"
                        ]
                    ),

                "ADE_over_F95":
                    float(
                        row[
                            "window_mean_ADE_over_F95"
                        ]
                    ),

                "FDE_over_F95":
                    float(
                        row[
                            "window_mean_FDE_over_F95"
                        ]
                    ),

                "future_points_above_F95_pct":
                    float(
                        row[
                            "all_future_points_above_F95_pct"
                        ]
                    ),

                "samples_ADE_above_F95_pct":
                    float(
                        row[
                            "samples_ADE_above_F95_pct"
                        ]
                    ),

                "samples_FDE_above_F95_pct":
                    float(
                        row[
                            "samples_FDE_above_F95_pct"
                        ]
                    ),

                "interpretation":
                    (
                        "ABSOLUTE_FORECAST_ERROR_SUBSTANTIALLY_EXCEEDS_EMPIRICAL_RESIDUAL_ENVELOPE"
                    ),
            }
        )

    absolute_df = pd.DataFrame(
        abs_rows
    )

    # Ensemble-gain scale evidence.
    gain_rows = []

    for task in [
        "2s_to_2s",
        "2s_to_5s",
    ]:
        for comparator in [
            "mean_individual_seed",
            "validation_selected_seed",
        ]:
            for metric in [
                "ADE",
                "FDE",
            ]:
                row = lookup(
                    d_boot,
                    task=task,
                    comparator=comparator,
                    metric=metric,
                )

                gain = float(
                    row[
                        "gain_px"
                    ]
                )

                low = float(
                    row[
                        "ci95_low"
                    ]
                )

                high = float(
                    row[
                        "ci95_high"
                    ]
                )

                if gain > f95:
                    scale_class = (
                        "GAIN_EXCEEDS_F95_ENVELOPE"
                    )
                elif gain > f50:
                    scale_class = (
                        "GAIN_EXCEEDS_F50_BUT_BELOW_F95"
                    )
                else:
                    scale_class = (
                        "GAIN_BELOW_F50"
                    )

                gain_rows.append(
                    {
                        "task":
                            task,

                        "comparator":
                            comparator,

                        "metric":
                            metric,

                        "gain_px":
                            gain,

                        "ci95_low":
                            low,

                        "ci95_high":
                            high,

                        "gain_over_F50":
                            float(
                                row[
                                    "gain_over_F50"
                                ]
                            ),

                        "gain_over_F95":
                            float(
                                row[
                                    "gain_over_F95"
                                ]
                            ),

                        "statistically_supported":
                            bool(
                                low > 0
                            ),

                        "residual_scale_class":
                            scale_class,
                    }
                )

    gain_df = pd.DataFrame(
        gain_rows
    )

    primary_gain = gain_df[
        gain_df[
            "comparator"
        ]
        ==
        "mean_individual_seed"
    ].copy()

    primary_all_significant = bool(
        primary_gain[
            "statistically_supported"
        ].all()
    )

    primary_all_below_f95 = bool(
        (
            primary_gain[
                "gain_px"
            ]
            <
            f95
        ).all()
    )

    primary_all_above_f50 = bool(
        (
            primary_gain[
                "gain_px"
            ]
            >
            f50
        ).all()
    )

    val_5s = gain_df[
        (
            gain_df[
                "comparator"
            ]
            ==
            "validation_selected_seed"
        )
        &
        (
            gain_df[
                "task"
            ]
            ==
            "2s_to_5s"
        )
    ].copy()

    validation_5s_both_above_f95 = bool(
        (
            val_5s[
                "gain_px"
            ]
            >
            f95
        ).all()
    )

    # Leave-one-sequence-out directional stability.
    loo_grouped = (
        d_loo.groupby(
            [
                "task",
                "comparator",
                "metric",
            ],
            as_index=False
        )
        .agg(
            min_gain_px=(
                "gain_px_segment_median_macro",
                "min"
            ),

            max_gain_px=(
                "gain_px_segment_median_macro",
                "max"
            ),

            all_positive=(
                "gain_px_segment_median_macro",
                lambda s:
                    bool(
                        (
                            pd.to_numeric(
                                s,
                                errors="coerce"
                            )
                            >
                            0
                        ).all()
                    ),
            ),

            all_above_F50=(
                "gain_px_segment_median_macro",
                lambda s:
                    bool(
                        (
                            pd.to_numeric(
                                s,
                                errors="coerce"
                            )
                            >
                            f50
                        ).all()
                    ),
            ),

            all_above_F95=(
                "gain_px_segment_median_macro",
                lambda s:
                    bool(
                        (
                            pd.to_numeric(
                                s,
                                errors="coerce"
                            )
                            >
                            f95
                        ).all()
                    ),
            ),
        )
    )

    loo_primary = loo_grouped[
        loo_grouped[
            "comparator"
        ]
        ==
        "mean_individual_seed"
    ]

    primary_loo_all_positive = bool(
        loo_primary[
            "all_positive"
        ].all()
    )

    primary_loo_all_above_f50 = bool(
        loo_primary[
            "all_above_F50"
        ].all()
    )

    # Primary conclusion table.
    conclusion_df = pd.DataFrame(
        [
            {
                "question":
                    "Are absolute TestReal forecast errors dominated by the measured post-correction residual scale?",

                "answer":
                    "No",

                "evidence":
                    (
                        "Mean ensemble ADE/FDE are multiple times larger than F95, "
                        "and most future-point/sample errors exceed F95."
                    ),

                "claim_strength":
                    "SUPPORTED",
            },
            {
                "question":
                    "Are ensemble gains over mean individual-seed error statistically reproducible?",

                "answer":
                    "Yes",

                "evidence":
                    (
                        "All 2s/5s ADE/FDE segment-cluster 95% CIs for comparator-minus-ensemble gain exclude zero."
                    ),

                "claim_strength":
                    (
                        "SUPPORTED"
                        if primary_all_significant
                        else
                        "NOT_SUPPORTED"
                    ),
            },
            {
                "question":
                    "Are the primary ensemble gains over mean seed larger than the conservative F95 residual envelope?",

                "answer":
                    "No",

                "evidence":
                    (
                        "All primary mean-seed ADE/FDE gains are below F95."
                    ),

                "claim_strength":
                    (
                        "SUPPORTED"
                        if primary_all_below_f95
                        else
                        "NOT_SUPPORTED"
                    ),
            },
            {
                "question":
                    "Are the primary ensemble gains over mean seed larger than the central F50 residual reference?",

                "answer":
                    "Yes at the overall segment-macro estimate, but not under every leave-one-sequence-out deletion.",

                "evidence":
                    (
                        "All overall primary gains exceed F50, but some leave-one-sequence-out gains fall below F50."
                    ),

                "claim_strength":
                    (
                        "SUPPORTED_WITH_SENSITIVITY_CAVEAT"
                        if (
                            primary_all_above_f50
                            and
                            primary_loo_all_positive
                            and
                            not primary_loo_all_above_f50
                        )
                        else
                        "REVIEW"
                    ),
            },
            {
                "question":
                    "Does the 5s validation-selected-seed comparison exceed the conservative F95 envelope?",

                "answer":
                    "Yes for both ADE and FDE",

                "evidence":
                    (
                        "The 5s ensemble gains versus the validation-selected seed exceed F95 for both metrics."
                    ),

                "claim_strength":
                    (
                        "SUPPORTED"
                        if validation_5s_both_above_f95
                        else
                        "NOT_SUPPORTED"
                    ),
            },
        ]
    )

    absolute_path = (
        OUT
        /
        "R11E_ABSOLUTE_ERROR_VS_RESIDUAL_ENVELOPE.csv"
    )

    gain_path = (
        OUT
        /
        "R11E_ENSEMBLE_GAIN_RESIDUAL_SCALE_CLASSIFICATION.csv"
    )

    loo_path = (
        OUT
        /
        "R11E_LEAVE_ONE_SEQUENCE_OUT_ROBUSTNESS.csv"
    )

    conclusion_path = (
        OUT
        /
        "R11E_FINAL_CLAIM_MATRIX.csv"
    )

    absolute_df.to_csv(
        absolute_path,
        index=False
    )

    gain_df.to_csv(
        gain_path,
        index=False
    )

    loo_grouped.to_csv(
        loo_path,
        index=False
    )

    conclusion_df.to_csv(
        conclusion_path,
        index=False
    )

    manuscript = f"""R11 — Empirical geometric residual / noise-floor analysis

Locked residual references
---------------------------
Using held-out BIRDSAI TestReal stationary-landmark residual motion after ego-motion correction:

F50 = {f50:.6f} px
F95 = {f95:.6f} px

These are empirical post-correction geometric residual references in the BIRDSAI TestReal camera setting. They are not claimed to be universal annotation noise, an irreducible forecasting-error bound, or a transferable threshold for Fresh WildBox or BAMBI.

Absolute forecast error
-----------------------
The frozen three-seed Transformer ensemble operates well above the measured residual envelope. At 2 s, mean pixel-space ADE is {absolute_df.loc[absolute_df['task']=='2s_to_2s','ensemble_ADE_px'].iloc[0]:.6f} px ({absolute_df.loc[absolute_df['task']=='2s_to_2s','ADE_over_F95'].iloc[0]:.2f}× F95) and mean FDE is {absolute_df.loc[absolute_df['task']=='2s_to_2s','ensemble_FDE_px'].iloc[0]:.6f} px ({absolute_df.loc[absolute_df['task']=='2s_to_2s','FDE_over_F95'].iloc[0]:.2f}× F95). At 5 s, mean ADE is {absolute_df.loc[absolute_df['task']=='2s_to_5s','ensemble_ADE_px'].iloc[0]:.6f} px ({absolute_df.loc[absolute_df['task']=='2s_to_5s','ADE_over_F95'].iloc[0]:.2f}× F95) and mean FDE is {absolute_df.loc[absolute_df['task']=='2s_to_5s','ensemble_FDE_px'].iloc[0]:.6f} px ({absolute_df.loc[absolute_df['task']=='2s_to_5s','FDE_over_F95'].iloc[0]:.2f}× F95).

Thus, the dominant forecasting errors are much larger than the measured residual geometric motion remaining after correction. This supports the interpretation that the reported forecasting error cannot be explained by ego-motion residual alone.

Magnitude of ensemble gains
---------------------------
The statistically supported ensemble gains over the mean individual-seed error are smaller than the conservative F95 residual envelope:

2 s ADE gain = {lookup(gain_df, task='2s_to_2s', comparator='mean_individual_seed', metric='ADE')['gain_px']:.6f} px
2 s FDE gain = {lookup(gain_df, task='2s_to_2s', comparator='mean_individual_seed', metric='FDE')['gain_px']:.6f} px
5 s ADE gain = {lookup(gain_df, task='2s_to_5s', comparator='mean_individual_seed', metric='ADE')['gain_px']:.6f} px
5 s FDE gain = {lookup(gain_df, task='2s_to_5s', comparator='mean_individual_seed', metric='FDE')['gain_px']:.6f} px

All four overall gains exceed the central F50 reference, but leave-one-sequence-out sensitivity shows that this F50 magnitude relation is not invariant to every sequence deletion. The correct interpretation is therefore that the ensemble benefit is statistically consistent, while its absolute magnitude is modest relative to the conservative residual envelope.

Validation-selected seed comparison
-----------------------------------
The comparison against the seed selected only by validation performance shows larger gains. In the 5 s task, both ADE and FDE gains exceed F95. This comparison is valid because seed selection was based on validation performance rather than TestReal performance, but the mean-seed comparison remains the cleaner primary ensemble-value reference.

Manuscript-safe conclusion
--------------------------
Post-correction geometric residual motion is substantially smaller than the absolute BIRDSAI TestReal forecasting error, so residual camera-motion noise does not account for the overall error magnitude. The frozen three-seed Transformer ensemble provides statistically reproducible improvements over individual-seed performance, but its primary gain over the mean seed remains below the conservative F95 residual envelope. Accordingly, the ensemble should be described as providing a small but consistent improvement rather than a large error reduction beyond all measurement-scale effects.

Prohibited claims
-----------------
- Do not call F50 or F95 universal irreducible annotation-noise bounds.
- Do not subtract F50/F95 from forecast error and label the remainder 'true model error'.
- Do not claim that every ensemble gain exceeds the residual envelope.
- Do not transfer BIRDSAI pixel residual references directly to Fresh WildBox or BAMBI.
- Do not infer end-to-end field effectiveness from this analysis.
"""

    manuscript_path = (
        OUT
        /
        "R11E_MANUSCRIPT_READY_NOISE_FLOOR_CONCLUSION.txt"
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8"
    )

    status = (
        "R11E_EMPIRICAL_NOISE_FLOOR_CONCLUSION_LOCKED_R11_COMPLETE_READY_FOR_R12_TRACK_ACCOUNTING"
    )

    manifest = {
        "step":
            "R11E_FINAL_EMPIRICAL_NOISE_FLOOR_CONCLUSION",

        "created_utc":
            utc_now(),

        "status":
            status,

        "training":
            False,

        "inference":
            False,

        "F50_px":
            f50,

        "F95_px":
            f95,

        "absolute_error_conclusion":
            (
                "BIRDSAI TestReal forecast error is substantially above the measured "
                "post-correction geometric residual envelope."
            ),

        "ensemble_gain_conclusion":
            (
                "Primary ensemble gains versus mean individual-seed error are statistically "
                "supported but remain below F95; they are small-to-moderate relative to the "
                "empirical residual scale."
            ),

        "primary_mean_seed_gain_all_significant":
            primary_all_significant,

        "primary_mean_seed_gain_all_below_F95":
            primary_all_below_f95,

        "primary_mean_seed_gain_all_above_F50_overall":
            primary_all_above_f50,

        "primary_mean_seed_gain_leave_one_sequence_out_all_positive":
            primary_loo_all_positive,

        "primary_mean_seed_gain_leave_one_sequence_out_all_above_F50":
            primary_loo_all_above_f50,

        "validation_selected_seed_5s_ADE_FDE_both_above_F95":
            validation_5s_both_above_f95,

        "claim_guardrails": [
            "F50/F95 are empirical BIRDSAI TestReal post-correction geometric residual references.",
            "Do not call them universal irreducible annotation-noise floors.",
            "Do not subtract them from forecast errors as if additive error decomposition were established.",
            "Do not transfer them directly to Fresh WildBox or BAMBI.",
            "Describe ensemble gain over mean seed as statistically consistent but below the conservative F95 envelope.",
        ],

        "outputs": {
            "absolute_error_vs_residual":
                str(
                    absolute_path
                ),

            "gain_scale_classification":
                str(
                    gain_path
                ),

            "leave_one_sequence_out":
                str(
                    loo_path
                ),

            "claim_matrix":
                str(
                    conclusion_path
                ),

            "manuscript_ready_conclusion":
                str(
                    manuscript_path
                ),
        },
    }

    (
        OUT
        /
        "R11E_MANIFEST.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "FINAL R11 CLAIM MATRIX:"
    )

    print(
        conclusion_df.to_string(
            index=False
        )
    )

    print(
        "\nPRIMARY MEAN-SEED GAIN SCALE:"
    )

    print(
        primary_gain[
            [
                "task",
                "metric",
                "gain_px",
                "ci95_low",
                "ci95_high",
                "gain_over_F50",
                "gain_over_F95",
                "residual_scale_class",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nLEAVE-ONE-SEQUENCE-OUT PRIMARY ROBUSTNESS:"
    )

    print(
        loo_primary.to_string(
            index=False
        )
    )

    print(
        "\nFINAL R11 CONCLUSION:"
    )

    print(
        "  Absolute TestReal forecast error is far above F95, so residual ego-motion alone cannot explain the forecasting error magnitude."
    )

    print(
        "  Ensemble gains over the mean seed are statistically supported but all remain below F95."
    )

    print(
        "  Overall mean-seed gains exceed F50, but this magnitude relation is not preserved under every leave-one-sequence-out deletion."
    )

    print(
        "  Therefore describe the ensemble improvement as small but consistent, not as a large reduction beyond all measurement-scale effects."
    )

    print(
        "\nSTATUS:"
    )

    print(
        status
    )

    print(
        "\nR11-E COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


def lookup(df, **filters):
    work = df.copy()

    for key, value in filters.items():
        work = work[
            work[key].astype(str)
            ==
            str(value)
        ]

    if len(work) != 1:
        raise RuntimeError(
            f"Expected one row for {filters}, found {len(work)}"
        )

    return work.iloc[0]


if __name__ == "__main__":
    main()
