
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd


DOWNLOADS = Path(r"C:\Users\HP\Downloads")

BASE = (
    DOWNLOADS
    / "paper_strengthening_revision"
    / "R9_canonicalization_isotropic_ablation"
)

E2C_DIR = (
    BASE
    / "R9I_E2C_paired_segment_cluster_bootstrap_anisotropic_vs_isotropic"
)

E2C_MANIFEST = E2C_DIR / "R9I_E2C_MANIFEST.json"
BOOTSTRAP_CSV = (
    E2C_DIR
    / "R9I_E2C_PRIMARY_SEGMENT_CLUSTER_BOOTSTRAP_RESULTS.csv"
)
DECISION_CSV = (
    E2C_DIR
    / "R9I_E2C_REPRESENTATION_DECISION_TABLE.csv"
)
ENSEMBLE_CSV = (
    E2C_DIR
    / "R9I_E2C_TRANSFORMER_3SEED_PREDICTION_ENSEMBLE_COMPARISON.csv"
)
SEED_SENS_CSV = (
    E2C_DIR
    / "R9I_E2C_SEED_SENSITIVITY.csv"
)
LOO_CSV = (
    E2C_DIR
    / "R9I_E2C_LEAVE_ONE_SEQUENCE_OUT_SENSITIVITY.csv"
)

OUT = (
    BASE
    / "R9I_F_canonicalization_ablation_conclusion"
)

OUT.mkdir(parents=True, exist_ok=True)


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


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def verify_upstream():
    for p in [
        E2C_MANIFEST,
        BOOTSTRAP_CSV,
        DECISION_CSV,
        ENSEMBLE_CSV,
        SEED_SENS_CSV,
        LOO_CSV,
    ]:
        require(p)

    m = json.loads(
        E2C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected = (
        "R9I_E2C_PAIRED_SEGMENT_CLUSTER_BOOTSTRAP_COMPLETE_READY_FOR_R9I_F_CANONICALIZATION_ABLATION_CONCLUSION"
    )

    if m.get("status") != expected:
        raise RuntimeError(
            f"Unexpected R9-I-E2C status: {m.get('status')}"
        )

    return m


def classify_ci(low, high):
    low = float(low)
    high = float(high)

    if high < 0:
        return "isotropic_better"

    if low > 0:
        return "isotropic_worse"

    return "inconclusive"


def main():
    print(
        "\nR9-I-F — CANONICALIZATION / ISOTROPIC ABLATION FINAL CONCLUSION"
    )

    print(
        "CONCLUSION ONLY. NO training, NO inference, NO tuning. "
        "This step converts the completed matched anisotropic-vs-isotropic "
        "paired analysis into a conservative claim matrix and manuscript-ready "
        "conclusion. Negative delta means isotropic is better.\n"
    )

    verify_upstream()

    boot = pd.read_csv(
        BOOTSTRAP_CSV
    )

    decision = pd.read_csv(
        DECISION_CSV
    )

    ensemble = pd.read_csv(
        ENSEMBLE_CSV
    )

    seed_sens = pd.read_csv(
        SEED_SENS_CSV
    )

    loo = pd.read_csv(
        LOO_CSV
    )

    expected_rows = {
        ("2s_to_2s", "Transformer"),
        ("2s_to_2s", "LSTM"),
        ("2s_to_2s", "GRU"),
        ("2s_to_5s", "Transformer"),
        ("2s_to_5s", "LSTM"),
        ("2s_to_5s", "GRU"),
    }

    actual_rows = {
        (
            str(row["task"]),
            str(row["model"]),
        )
        for _, row in boot.iterrows()
    }

    if actual_rows != expected_rows:
        raise RuntimeError(
            f"Unexpected primary comparison rows: {sorted(actual_rows)}"
        )

    claim_rows = []

    for _, row in boot.iterrows():
        ade_class = classify_ci(
            row["ADE_delta_px_ci95_low"],
            row["ADE_delta_px_ci95_high"],
        )

        fde_class = classify_ci(
            row["FDE_delta_px_ci95_low"],
            row["FDE_delta_px_ci95_high"],
        )

        if ade_class == "isotropic_better" and fde_class == "isotropic_better":
            overall = "supports_isotropic_better_for_both_metrics"

        elif ade_class == "isotropic_worse" and fde_class == "isotropic_worse":
            overall = "supports_isotropic_worse_for_both_metrics"

        else:
            overall = "mixed_or_inconclusive"

        claim_rows.append(
            {
                "analysis_scope":
                    "single_model_3seed_mean_effect",

                "task":
                    row["task"],

                "model":
                    row["model"],

                "ADE_delta_px":
                    row["ADE_delta_px_primary"],

                "ADE_CI95":
                    (
                        f"[{row['ADE_delta_px_ci95_low']:.6f}, "
                        f"{row['ADE_delta_px_ci95_high']:.6f}]"
                    ),

                "ADE_interpretation":
                    ade_class,

                "FDE_delta_px":
                    row["FDE_delta_px_primary"],

                "FDE_CI95":
                    (
                        f"[{row['FDE_delta_px_ci95_low']:.6f}, "
                        f"{row['FDE_delta_px_ci95_high']:.6f}]"
                    ),

                "FDE_interpretation":
                    fde_class,

                "overall_interpretation":
                    overall,
            }
        )

    for _, row in ensemble.iterrows():
        ade_class = classify_ci(
            row["ADE_delta_px_ci95_low"],
            row["ADE_delta_px_ci95_high"],
        )

        fde_class = classify_ci(
            row["FDE_delta_px_ci95_low"],
            row["FDE_delta_px_ci95_high"],
        )

        if ade_class == "isotropic_better" and fde_class == "isotropic_better":
            overall = "supports_isotropic_better_for_both_metrics"

        elif ade_class == "isotropic_worse" and fde_class == "isotropic_worse":
            overall = "supports_isotropic_worse_for_both_metrics"

        else:
            overall = "mixed_or_inconclusive"

        claim_rows.append(
            {
                "analysis_scope":
                    "transformer_3seed_prediction_ensemble",

                "task":
                    row["task"],

                "model":
                    row["model"],

                "ADE_delta_px":
                    row["ADE_delta_px_primary"],

                "ADE_CI95":
                    (
                        f"[{row['ADE_delta_px_ci95_low']:.6f}, "
                        f"{row['ADE_delta_px_ci95_high']:.6f}]"
                    ),

                "ADE_interpretation":
                    ade_class,

                "FDE_delta_px":
                    row["FDE_delta_px_primary"],

                "FDE_CI95":
                    (
                        f"[{row['FDE_delta_px_ci95_low']:.6f}, "
                        f"{row['FDE_delta_px_ci95_high']:.6f}]"
                    ),

                "FDE_interpretation":
                    fde_class,

                "overall_interpretation":
                    overall,
            }
        )

    claim_df = pd.DataFrame(
        claim_rows
    )

    claim_path = (
        OUT
        / "R9I_F_CANONICALIZATION_CLAIM_MATRIX.csv"
    )

    claim_df.to_csv(
        claim_path,
        index=False
    )

    # Pull exact focal rows.
    ens_2 = ensemble[
        ensemble["task"] == "2s_to_2s"
    ].iloc[0]

    ens_5 = ensemble[
        ensemble["task"] == "2s_to_5s"
    ].iloc[0]

    gru_5 = boot[
        (boot["task"] == "2s_to_5s")
        &
        (boot["model"] == "GRU")
    ].iloc[0]

    lstm_5 = boot[
        (boot["task"] == "2s_to_5s")
        &
        (boot["model"] == "LSTM")
    ].iloc[0]

    tf_5 = boot[
        (boot["task"] == "2s_to_5s")
        &
        (boot["model"] == "Transformer")
    ].iloc[0]

    manuscript = f"""R9 canonicalization / isotropic-coordinate ablation — final interpretation

Primary conclusion
------------------
The aspect-ratio audit showed that the original x/W, y/H normalization materially changes heading, path length, scale-floor activation, and canonical tensors when image width and height differ. A matched isotropic representation x/S, y/S with S=max(W,H) was therefore retrained and evaluated using the same TrainReal/TestReal sample identities, the same train-validation split, the same sample weights, the same model architectures, the same hyperparameters, and the same three seeds. Cross-representation performance was compared only in pixel space.

The matched paired results do NOT support a universal forecasting-superiority claim for isotropic normalization. Effects are model- and horizon-dependent.

Strong positive result
----------------------
For the frozen 3-seed Transformer prediction ensemble at 2 s, isotropic normalization improved both pixel-space metrics:
- ADE difference (isotropic - anisotropic) = {ens_2['ADE_delta_px_primary']:.6f} px, 95% cluster-bootstrap CI [{ens_2['ADE_delta_px_ci95_low']:.6f}, {ens_2['ADE_delta_px_ci95_high']:.6f}].
- FDE difference (isotropic - anisotropic) = {ens_2['FDE_delta_px_primary']:.6f} px, 95% cluster-bootstrap CI [{ens_2['FDE_delta_px_ci95_low']:.6f}, {ens_2['FDE_delta_px_ci95_high']:.6f}].
Both confidence intervals exclude zero in the direction favoring isotropic normalization.

Five-second horizon
-------------------
At 5 s, the evidence is mixed:
- GRU FDE difference = {gru_5['FDE_delta_px_primary']:.6f} px, 95% CI [{gru_5['FDE_delta_px_ci95_low']:.6f}, {gru_5['FDE_delta_px_ci95_high']:.6f}], indicating significantly worse isotropic FDE.
- LSTM FDE difference = {lstm_5['FDE_delta_px_primary']:.6f} px, 95% CI [{lstm_5['FDE_delta_px_ci95_low']:.6f}, {lstm_5['FDE_delta_px_ci95_high']:.6f}], also indicating significantly worse isotropic FDE.
- Transformer ADE difference = {tf_5['ADE_delta_px_primary']:.6f} px, 95% CI [{tf_5['ADE_delta_px_ci95_low']:.6f}, {tf_5['ADE_delta_px_ci95_high']:.6f}], favoring isotropic normalization, but Transformer FDE remained inconclusive.
- The 5 s Transformer 3-seed prediction ensemble was inconclusive for both ADE and FDE: ADE difference = {ens_5['ADE_delta_px_primary']:.6f} px with CI [{ens_5['ADE_delta_px_ci95_low']:.6f}, {ens_5['ADE_delta_px_ci95_high']:.6f}], and FDE difference = {ens_5['FDE_delta_px_primary']:.6f} px with CI [{ens_5['FDE_delta_px_ci95_low']:.6f}, {ens_5['FDE_delta_px_ci95_high']:.6f}].

Two-second single-model results
-------------------------------
For GRU, LSTM, and Transformer considered individually at 2 s, the paired segment-cluster confidence intervals included zero for both ADE and FDE. Therefore, the 2 s ensemble improvement must not be generalized to all individual models or seeds.

Seed sensitivity
----------------
Per-seed effects change direction in several model/horizon combinations. This reinforces the need to avoid claiming universal isotropic superiority from one seed or one model.

Manuscript-safe claim
---------------------
Aspect-preserving isotropic normalization removes a real geometric distortion introduced by separate x/W and y/H scaling, but its forecasting effect is not uniformly beneficial. The matched ablation therefore supports isotropic normalization as a geometrically principled alternative whose performance impact is model- and horizon-dependent. The clearest benefit was observed for the 2 s frozen three-seed Transformer prediction ensemble, whereas 5 s recurrent-model endpoint error worsened and several other comparisons remained statistically inconclusive.

Protocol decision
-----------------
Keep the original anisotropic pipeline/results as the frozen primary historical experiment. Report isotropic normalization as a matched canonicalization ablation/sensitivity analysis. Do not replace all frozen primary results with isotropic results and do not claim general superiority. If an operational 2 s Transformer ensemble is highlighted, the isotropic representation can be reported as beneficial for that specific configuration, with the paired clustered confidence intervals stated explicitly.

Important preprocessing limitation
----------------------------------
For strict methodological fidelity, the isotropic retraining preserved the original historical scale-floor derivation protocol: the 5th-percentile path-length scale floor was estimated from the full BIRDSAI TrainReal pool before the track-disjoint neural train/validation split. Validation rows therefore contributed to this preprocessing statistic. TestReal and WildBox were not used. This should be disclosed as a minor preprocessing-protocol limitation; a strict-train-only sensitivity analysis may be added later if desired, but it should not replace the primary matched comparison.

Do not claim
------------
- Do not claim isotropic normalization universally improves trajectory forecasting.
- Do not compare anisotropic and isotropic native normalized ADE/FDE as if they were directly commensurate.
- Do not claim the 2 s ensemble benefit automatically extends to 5 s.
- Do not claim all three neural architectures improve under isotropic normalization.
- Do not hide the 5 s GRU/LSTM FDE degradation.
"""

    manuscript_path = (
        OUT
        / "R9I_F_MANUSCRIPT_READY_CANONICALIZATION_CONCLUSION.txt"
    )

    manuscript_path.write_text(
        manuscript,
        encoding="utf-8"
    )

    summary = {
        "step":
            "R9I_F_CANONICALIZATION_ABLATION_CONCLUSION",

        "created_utc":
            utc_now(),

        "status":
            "R9I_F_CANONICALIZATION_ABLATION_CONCLUSION_LOCKED_R9_COMPLETE_READY_FOR_R10_ENSEMBLE_ANALYSIS",

        "global_conclusion":
            "No universal isotropic forecasting superiority; effect is model- and horizon-dependent.",

        "geometry_conclusion":
            "Isotropic normalization is geometrically principled because it avoids aspect-ratio distortion from separate x/W and y/H scaling.",

        "strongest_positive_result":
            {
                "configuration":
                    "2s_to_2s Transformer 3-seed prediction ensemble",

                "ADE_delta_px":
                    float(
                        ens_2[
                            "ADE_delta_px_primary"
                        ]
                    ),

                "ADE_CI95":
                    [
                        float(
                            ens_2[
                                "ADE_delta_px_ci95_low"
                            ]
                        ),
                        float(
                            ens_2[
                                "ADE_delta_px_ci95_high"
                            ]
                        ),
                    ],

                "FDE_delta_px":
                    float(
                        ens_2[
                            "FDE_delta_px_primary"
                        ]
                    ),

                "FDE_CI95":
                    [
                        float(
                            ens_2[
                                "FDE_delta_px_ci95_low"
                            ]
                        ),
                        float(
                            ens_2[
                                "FDE_delta_px_ci95_high"
                            ]
                        ),
                    ],
            },

        "important_negative_results":
            [
                {
                    "configuration":
                        "2s_to_5s GRU",

                    "metric":
                        "FDE",

                    "delta_px":
                        float(
                            gru_5[
                                "FDE_delta_px_primary"
                            ]
                        ),

                    "CI95":
                        [
                            float(
                                gru_5[
                                    "FDE_delta_px_ci95_low"
                                ]
                            ),
                            float(
                                gru_5[
                                    "FDE_delta_px_ci95_high"
                                ]
                            ),
                        ],
                },
                {
                    "configuration":
                        "2s_to_5s LSTM",

                    "metric":
                        "FDE",

                    "delta_px":
                        float(
                            lstm_5[
                                "FDE_delta_px_primary"
                            ]
                        ),

                    "CI95":
                        [
                            float(
                                lstm_5[
                                    "FDE_delta_px_ci95_low"
                                ]
                            ),
                            float(
                                lstm_5[
                                    "FDE_delta_px_ci95_high"
                                ]
                            ),
                        ],
                },
            ],

        "paper_protocol_decision":
            (
                "Keep frozen anisotropic primary results; report isotropic as a matched "
                "canonicalization ablation/sensitivity analysis. Specific 2s Transformer "
                "ensemble benefit may be reported with paired clustered CIs."
            ),

        "outputs":
            {
                "claim_matrix":
                    str(
                        claim_path
                    ),

                "manuscript_ready_conclusion":
                    str(
                        manuscript_path
                    ),
            },
    }

    manifest_path = (
        OUT
        / "R9I_F_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "FINAL CLAIM MATRIX:"
    )

    print(
        claim_df.to_string(
            index=False
        )
    )

    print(
        "\nFINAL R9 CONCLUSION:"
    )

    print(
        "  Geometry: isotropic normalization is justified as an aspect-preserving alternative."
    )

    print(
        "  Forecasting: NO universal isotropic-superiority claim."
    )

    print(
        "  Strongest benefit: 2s Transformer 3-seed prediction ensemble improves both ADE and FDE with 95% clustered CIs excluding zero."
    )

    print(
        "  Important counter-result: 5s GRU/LSTM FDE is significantly worse under isotropic normalization."
    )

    print(
        "  Paper decision: retain original anisotropic primary results; report isotropic as matched ablation/sensitivity analysis."
    )

    print(
        "\nSTATUS:"
    )

    print(
        summary[
            "status"
        ]
    )

    print(
        "\nR9-I-F COMPLETE"
    )

    print(
        "Outputs:",
        OUT
    )


if __name__ == "__main__":
    main()
