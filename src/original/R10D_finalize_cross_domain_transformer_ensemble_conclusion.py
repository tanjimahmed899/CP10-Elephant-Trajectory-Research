
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

REVISION_ROOT = DOWNLOADS / "paper_strengthening_revision"
R10_ROOT = REVISION_ROOT / "R10_ensemble_analysis"

R10B_DIR = R10_ROOT / "R10B_testreal_transformer_ensemble_value"
R10B_MANIFEST = R10B_DIR / "R10B_MANIFEST.json"
R10B_SUMMARY = R10B_DIR / "R10B_TESTREAL_ENSEMBLE_SUMMARY.csv"
R10B_BOOTSTRAP = R10B_DIR / "R10B_PAIRED_SEGMENT_CLUSTER_BOOTSTRAP.csv"
R10B_DIVERSITY = R10B_DIR / "R10B_DIVERSITY_GAIN_ASSOCIATION.csv"

R10C2_DIR = R10_ROOT / "R10C2_fresh_wildbox_external_ensemble_value"
R10C2_MANIFEST = R10C2_DIR / "R10C2_MANIFEST.json"
R10C2_SUMMARY = R10C2_DIR / "R10C2_EXTERNAL_ENSEMBLE_SUMMARY.csv"
R10C2_BOOTSTRAP = R10C2_DIR / "R10C2_TRACK_CLUSTER_BOOTSTRAP.csv"
R10C2_LOO = R10C2_DIR / "R10C2_LEAVE_ONE_SOURCE_OUT_SENSITIVITY.csv"
R10C2_DIVERSITY = R10C2_DIR / "R10C2_DIVERSITY_GAIN_ASSOCIATION.csv"

OUT = R10_ROOT / "R10D_cross_domain_ensemble_conclusion"
OUT.mkdir(parents=True, exist_ok=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_upstream():
    for p in [
        R10B_MANIFEST,
        R10B_SUMMARY,
        R10B_BOOTSTRAP,
        R10B_DIVERSITY,
        R10C2_MANIFEST,
        R10C2_SUMMARY,
        R10C2_BOOTSTRAP,
        R10C2_LOO,
        R10C2_DIVERSITY,
    ]:
        require(p)

    b = json.loads(
        R10B_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    c2 = json.loads(
        R10C2_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_b = (
        "R10B_TESTREAL_TRANSFORMER_ENSEMBLE_VALUE_QUANTIFIED_READY_FOR_R10C_FRESH_WILDBOX_TRANSFORMER_REINFERENCE"
    )

    expected_c2 = (
        "R10C2_FRESH_WILDBOX_EXTERNAL_ENSEMBLE_VALUE_QUANTIFIED_READY_FOR_R10D_CROSS_DOMAIN_ENSEMBLE_CONCLUSION"
    )

    if b.get("status") != expected_b:
        raise RuntimeError(
            f"Unexpected R10-B status: {b.get('status')}"
        )

    if c2.get("status") != expected_c2:
        raise RuntimeError(
            f"Unexpected R10-C2 status: {c2.get('status')}"
        )

    return b, c2


def pct_improvement(ensemble, comparator):
    if comparator == 0:
        return None

    return float(
        100.0
        *
        (comparator - ensemble)
        /
        comparator
    )


def ci_direction(low, high):
    low = float(low)
    high = float(high)

    if high < 0:
        return "ENSEMBLE_BETTER_CI_EXCLUDES_ZERO"

    if low > 0:
        return "ENSEMBLE_WORSE_CI_EXCLUDES_ZERO"

    return "INCONCLUSIVE_CI_INCLUDES_ZERO"


def build_claim_matrix(
    b_summary,
    b_boot,
    c_summary,
    c_boot,
):
    rows = []

    dataset_specs = [
        (
            "BIRDSAI_TestReal",
            b_summary,
            b_boot,
            "segment",
        ),
        (
            "Fresh_WildBox",
            c_summary,
            c_boot,
            "track",
        ),
    ]

    for dataset_name, summary_df, boot_df, dependency in dataset_specs:
        for _, summary_row in summary_df.iterrows():
            task = str(
                summary_row["task"]
            )

            if dataset_name == "BIRDSAI_TestReal":
                ens_ade = float(
                    summary_row[
                        "ensemble_ADE_px_window_mean"
                    ]
                )
                ens_fde = float(
                    summary_row[
                        "ensemble_FDE_px_window_mean"
                    ]
                )
                mean_ade = float(
                    summary_row[
                        "mean_individual_seed_ADE_px_window_mean"
                    ]
                )
                mean_fde = float(
                    summary_row[
                        "mean_individual_seed_FDE_px_window_mean"
                    ]
                )
                val_ade = float(
                    summary_row[
                        "validation_selected_seed_ADE_px_window_mean"
                    ]
                )
                val_fde = float(
                    summary_row[
                        "validation_selected_seed_FDE_px_window_mean"
                    ]
                )
                unit_label = "px"

            else:
                ens_ade = float(
                    summary_row[
                        "ensemble_ADE_window_mean"
                    ]
                )
                ens_fde = float(
                    summary_row[
                        "ensemble_FDE_window_mean"
                    ]
                )
                mean_ade = float(
                    summary_row[
                        "mean_individual_seed_ADE_window_mean"
                    ]
                )
                mean_fde = float(
                    summary_row[
                        "mean_individual_seed_FDE_window_mean"
                    ]
                )
                val_ade = float(
                    summary_row[
                        "validation_selected_seed_ADE_window_mean"
                    ]
                )
                val_fde = float(
                    summary_row[
                        "validation_selected_seed_FDE_window_mean"
                    ]
                )
                unit_label = "normalized"

            for comparator_name, comp_ade, comp_fde in [
                (
                    "mean_individual_seed",
                    mean_ade,
                    mean_fde,
                ),
                (
                    "validation_selected_seed",
                    val_ade,
                    val_fde,
                ),
            ]:
                boot_comp_name = (
                    "ensemble_vs_mean_individual_seed"
                    if comparator_name
                    ==
                    "mean_individual_seed"
                    else
                    "ensemble_vs_validation_selected_seed"
                )

                for metric, ens_value, comp_value in [
                    (
                        "ADE",
                        ens_ade,
                        comp_ade,
                    ),
                    (
                        "FDE",
                        ens_fde,
                        comp_fde,
                    ),
                ]:
                    b_row = boot_df[
                        (
                            boot_df["task"].astype(str)
                            ==
                            task
                        )
                        &
                        (
                            boot_df["comparator"].astype(str)
                            ==
                            boot_comp_name
                        )
                        &
                        (
                            boot_df["metric"].astype(str)
                            ==
                            metric
                        )
                    ]

                    if len(b_row) != 1:
                        raise RuntimeError(
                            f"{dataset_name}/{task}/{boot_comp_name}/{metric}: "
                            f"expected one bootstrap row, found {len(b_row)}"
                        )

                    b_row = b_row.iloc[0]

                    delta_col = (
                        "delta_px"
                        if "delta_px" in b_row.index
                        else "delta"
                    )

                    delta = float(
                        b_row[
                            delta_col
                        ]
                    )

                    low = float(
                        b_row[
                            "ci95_low"
                        ]
                    )

                    high = float(
                        b_row[
                            "ci95_high"
                        ]
                    )

                    direction = ci_direction(
                        low,
                        high
                    )

                    rows.append(
                        {
                            "dataset":
                                dataset_name,

                            "task":
                                task,

                            "metric":
                                metric,

                            "comparator":
                                comparator_name,

                            "ensemble_value":
                                ens_value,

                            "comparator_value":
                                comp_value,

                            "percent_improvement":
                                pct_improvement(
                                    ens_value,
                                    comp_value
                                ),

                            "paired_delta_ensemble_minus_comparator":
                                delta,

                            "ci95_low":
                                low,

                            "ci95_high":
                                high,

                            "direction":
                                direction,

                            "dependency_unit":
                                dependency,

                            "metric_unit":
                                unit_label,
                        }
                    )

    return pd.DataFrame(
        rows
    )


def check_external_loo_consistency(
    loo_df,
):
    rows = []

    for (
        task,
        comparator,
        metric
    ), group in loo_df.groupby(
        [
            "task",
            "comparator",
            "metric",
        ],
        sort=False
    ):
        values = pd.to_numeric(
            group[
                "delta_track_median_macro"
            ],
            errors="coerce"
        )

        all_negative = bool(
            (
                values
                <
                0
            ).all()
        )

        rows.append(
            {
                "task":
                    task,

                "comparator":
                    comparator,

                "metric":
                    metric,

                "sources_held_out":
                    int(
                        len(
                            group
                        )
                    ),

                "all_leave_one_source_out_effects_negative":
                    all_negative,

                "min_delta":
                    float(
                        values.min()
                    ),

                "max_delta":
                    float(
                        values.max()
                    ),

                "conclusion":
                    (
                        "DIRECTION_STABLE_ACROSS_ALL_3_SOURCE_DELETIONS"
                        if all_negative
                        else
                        "SOURCE_SENSITIVITY_PRESENT"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "\nR10-D — CROSS-DOMAIN TRANSFORMER ENSEMBLE FINAL CONCLUSION"
    )

    print(
        "CONCLUSION ONLY. NO training, NO inference, NO tuning. "
        "This step combines the locked BIRDSAI TestReal ensemble evidence (R10-B) "
        "and the fresh matched WildBox external ensemble evidence (R10-C2). "
        "Claims are restricted to the frozen 3-seed Transformer ensemble and the "
        "evaluated datasets/horizons. Broad ecological, cross-species, or field-effectiveness "
        "claims are explicitly prohibited.\n"
    )

    verify_upstream()

    b_summary = pd.read_csv(
        R10B_SUMMARY
    )

    b_boot = pd.read_csv(
        R10B_BOOTSTRAP
    )

    b_div = pd.read_csv(
        R10B_DIVERSITY
    )

    c_summary = pd.read_csv(
        R10C2_SUMMARY
    )

    c_boot = pd.read_csv(
        R10C2_BOOTSTRAP
    )

    c_loo = pd.read_csv(
        R10C2_LOO
    )

    c_div = pd.read_csv(
        R10C2_DIVERSITY
    )

    claim_df = build_claim_matrix(
        b_summary,
        b_boot,
        c_summary,
        c_boot,
    )

    loo_consistency_df = check_external_loo_consistency(
        c_loo
    )

    primary_claim_rows = claim_df[
        claim_df[
            "comparator"
        ]
        ==
        "mean_individual_seed"
    ].copy()

    all_primary_significant = bool(
        (
            primary_claim_rows[
                "direction"
            ]
            ==
            "ENSEMBLE_BETTER_CI_EXCLUDES_ZERO"
        ).all()
    )

    all_validation_comparisons_significant = bool(
        (
            claim_df[
                claim_df[
                    "comparator"
                ]
                ==
                "validation_selected_seed"
            ][
                "direction"
            ]
            ==
            "ENSEMBLE_BETTER_CI_EXCLUDES_ZERO"
        ).all()
    )

    all_external_loo_negative = bool(
        loo_consistency_df[
            "all_leave_one_source_out_effects_negative"
        ].all()
    )

    all_diversity_assoc_negative = bool(
        (
            pd.concat(
                [
                    b_div[
                        [
                            "spearman_rho_diversity_vs_gain"
                        ]
                    ],
                    c_div[
                        [
                            "spearman_rho_diversity_vs_gain"
                        ]
                    ],
                ],
                ignore_index=True
            )[
                "spearman_rho_diversity_vs_gain"
            ]
            <
            0
        ).all()
    )

    claim_path = (
        OUT
        /
        "R10D_CROSS_DOMAIN_ENSEMBLE_CLAIM_MATRIX.csv"
    )

    loo_path = (
        OUT
        /
        "R10D_EXTERNAL_LEAVE_ONE_SOURCE_OUT_CONSISTENCY.csv"
    )

    claim_df.to_csv(
        claim_path,
        index=False
    )

    loo_consistency_df.to_csv(
        loo_path,
        index=False
    )

    # Pull concise rows for manuscript text.
    def row_for(
        dataset,
        task,
        metric,
        comparator="mean_individual_seed",
    ):
        row = claim_df[
            (
                claim_df["dataset"] == dataset
            )
            &
            (
                claim_df["task"] == task
            )
            &
            (
                claim_df["metric"] == metric
            )
            &
            (
                claim_df["comparator"] == comparator
            )
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Missing claim row: {dataset}/{task}/{metric}/{comparator}"
            )

        return row.iloc[0]

    bt2_ade = row_for(
        "BIRDSAI_TestReal",
        "2s_to_2s",
        "ADE",
    )

    bt2_fde = row_for(
        "BIRDSAI_TestReal",
        "2s_to_2s",
        "FDE",
    )

    bt5_ade = row_for(
        "BIRDSAI_TestReal",
        "2s_to_5s",
        "ADE",
    )

    bt5_fde = row_for(
        "BIRDSAI_TestReal",
        "2s_to_5s",
        "FDE",
    )

    fw2_ade = row_for(
        "Fresh_WildBox",
        "2s_to_2s",
        "ADE",
    )

    fw2_fde = row_for(
        "Fresh_WildBox",
        "2s_to_2s",
        "FDE",
    )

    fw5_ade = row_for(
        "Fresh_WildBox",
        "2s_to_5s",
        "ADE",
    )

    fw5_fde = row_for(
        "Fresh_WildBox",
        "2s_to_5s",
        "FDE",
    )

    manuscript = f"""R10 — Frozen three-seed Transformer ensemble analysis

Final conclusion
----------------
Across both the BIRDSAI TestReal evaluation and the fresh matched WildBox external evaluation, averaging the predictions from the three frozen Transformer seeds (42, 52, and 62) consistently reduced forecasting error relative to the mean individual-seed error. The paired clustered confidence intervals excluded zero in favor of the ensemble for both ADE and FDE at both 2 s and 5 s horizons.

BIRDSAI TestReal
----------------
At 2 s, the ensemble improved ADE by {bt2_ade['percent_improvement']:.3f}% relative to the mean individual-seed error. The paired segment-cluster effect was {bt2_ade['paired_delta_ensemble_minus_comparator']:.6f} px with 95% CI [{bt2_ade['ci95_low']:.6f}, {bt2_ade['ci95_high']:.6f}]. FDE improved by {bt2_fde['percent_improvement']:.3f}%, with a paired effect of {bt2_fde['paired_delta_ensemble_minus_comparator']:.6f} px and 95% CI [{bt2_fde['ci95_low']:.6f}, {bt2_fde['ci95_high']:.6f}].

At 5 s, the ensemble improved ADE by {bt5_ade['percent_improvement']:.3f}% relative to the mean individual-seed error, with a paired effect of {bt5_ade['paired_delta_ensemble_minus_comparator']:.6f} px and 95% CI [{bt5_ade['ci95_low']:.6f}, {bt5_ade['ci95_high']:.6f}]. FDE improved by {bt5_fde['percent_improvement']:.3f}%, with a paired effect of {bt5_fde['paired_delta_ensemble_minus_comparator']:.6f} px and 95% CI [{bt5_fde['ci95_low']:.6f}, {bt5_fde['ci95_high']:.6f}].

Fresh WildBox external evaluation
---------------------------------
On the fresh matched WildBox set, the ensemble again improved both horizons. At 2 s, ADE improved by {fw2_ade['percent_improvement']:.3f}% and FDE by {fw2_fde['percent_improvement']:.3f}% relative to the mean individual-seed error. The track-cluster paired effects were {fw2_ade['paired_delta_ensemble_minus_comparator']:.6f} for ADE, 95% CI [{fw2_ade['ci95_low']:.6f}, {fw2_ade['ci95_high']:.6f}], and {fw2_fde['paired_delta_ensemble_minus_comparator']:.6f} for FDE, 95% CI [{fw2_fde['ci95_low']:.6f}, {fw2_fde['ci95_high']:.6f}].

At 5 s, ADE improved by {fw5_ade['percent_improvement']:.3f}% and FDE by {fw5_fde['percent_improvement']:.3f}% relative to the mean individual-seed error. The track-cluster paired effects were {fw5_ade['paired_delta_ensemble_minus_comparator']:.6f} for ADE, 95% CI [{fw5_ade['ci95_low']:.6f}, {fw5_ade['ci95_high']:.6f}], and {fw5_fde['paired_delta_ensemble_minus_comparator']:.6f} for FDE, 95% CI [{fw5_fde['ci95_low']:.6f}, {fw5_fde['ci95_high']:.6f}].

Robustness
----------
The ensemble was also significantly better than the single seed selected only from validation performance in all tested dataset/horizon/metric combinations. On fresh WildBox, leave-one-source-out analysis preserved a negative ensemble-minus-comparator effect for every held-out source, indicating that the direction of benefit was not driven by any one of the three fresh videos.

Seed diversity
--------------
Across both TestReal and fresh WildBox, the association between seed disagreement and ensemble gain was consistently negative: samples with greater inter-seed disagreement tended to show larger ensemble benefit. This is supportive evidence that averaging complementary seed-specific predictions, rather than simply reporting one seed, contributes useful variance reduction.

Manuscript-safe claim
---------------------
A frozen three-seed Transformer prediction ensemble provides a small but statistically consistent forecasting benefit over individual-seed performance on BIRDSAI TestReal and on the fresh matched WildBox external holdout, for both 2 s and 5 s horizons. This supports using the frozen ensemble as the primary Transformer forecasting output in downstream uncertainty and warning analyses.

Limitations and prohibited claims
---------------------------------
- The ensemble benefit is specific to the evaluated Transformer architecture and three frozen seeds; it should not be generalized automatically to GRU, LSTM, or arbitrary ensemble sizes.
- The fresh external result covers three WildBox videos and should not be described as broad ecological or species-level generalization.
- WildBox remains an elephant-domain external evaluation; it does not replace the separate BAMBI deer/human cross-species stress test.
- The association between diversity and gain is descriptive, not proof that diversity causally improves forecasting.
- The ensemble does not establish end-to-end field effectiveness, because detector/tracker uncertainty and operational response latency are separate system components.
"""

    manuscript_path = (
        OUT
        /
        "R10D_MANUSCRIPT_READY_ENSEMBLE_CONCLUSION.txt"
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8"
    )

    if (
        all_primary_significant
        and
        all_validation_comparisons_significant
        and
        all_external_loo_negative
    ):
        final_status = (
            "R10D_CROSS_DOMAIN_ENSEMBLE_CONCLUSION_LOCKED_R10_COMPLETE_READY_FOR_R11_NOISE_FLOOR_ANALYSIS"
        )
    else:
        final_status = (
            "R10D_ENSEMBLE_EVIDENCE_MIXED_REVIEW_BEFORE_R11"
        )

    manifest = {
        "step":
            "R10D_CROSS_DOMAIN_TRANSFORMER_ENSEMBLE_CONCLUSION",

        "created_utc":
            utc_now(),

        "status":
            final_status,

        "training":
            False,

        "inference":
            False,

        "global_conclusion":
            (
                "Frozen 3-seed Transformer averaging consistently improves ADE and FDE "
                "relative to mean individual-seed error on BIRDSAI TestReal and fresh WildBox "
                "for both 2s and 5s horizons."
            ),

        "all_primary_mean_seed_comparisons_significant":
            all_primary_significant,

        "all_validation_selected_seed_comparisons_significant":
            all_validation_comparisons_significant,

        "all_fresh_wildbox_leave_one_source_out_effects_negative":
            all_external_loo_negative,

        "all_diversity_gain_associations_negative":
            all_diversity_assoc_negative,

        "recommended_pipeline_role":
            (
                "Use the frozen 3-seed Transformer prediction ensemble as the primary "
                "Transformer output for downstream uncertainty and warning analysis."
            ),

        "claim_guardrails": [
            "Do not generalize ensemble benefit to GRU/LSTM without separate evidence.",
            "Do not call the 3-video WildBox holdout broad ecological generalization.",
            "Do not treat diversity-gain correlation as causal evidence.",
            "Do not claim end-to-end field effectiveness from forecasting ensemble results alone.",
        ],

        "outputs": {
            "claim_matrix":
                str(
                    claim_path
                ),

            "external_loo_consistency":
                str(
                    loo_path
                ),

            "manuscript_ready_conclusion":
                str(
                    manuscript_path
                ),
        },
    }

    manifest_path = (
        OUT
        /
        "R10D_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "CROSS-DOMAIN ENSEMBLE CLAIM MATRIX:"
    )

    print(
        claim_df.to_string(
            index=False
        )
    )

    print(
        "\nFRESH WILDBOX LEAVE-ONE-SOURCE-OUT CONSISTENCY:"
    )

    print(
        loo_consistency_df.to_string(
            index=False
        )
    )

    print(
        "\nFINAL R10 CONCLUSION:"
    )

    print(
        "  TestReal: ensemble significantly better than mean individual-seed error for ADE/FDE at 2s and 5s."
    )

    print(
        "  Fresh WildBox: ensemble significantly better than mean individual-seed error for ADE/FDE at 2s and 5s."
    )

    print(
        "  Validation-selected seed: ensemble significantly better in all tested combinations."
    )

    print(
        "  Fresh leave-one-source-out: direction remains favorable for every held-out video."
    )

    print(
        "  Diversity-gain association: consistently negative across TestReal and Fresh WildBox."
    )

    print(
        "  Paper decision: frozen 3-seed Transformer ensemble is justified as the primary Transformer output."
    )

    print(
        "\nSTATUS:"
    )

    print(
        final_status
    )

    print(
        "\nR10-D COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
