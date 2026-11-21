# Stage 7F — Independent post-processing audit and one-line WSL report fix

- `STAGE7F_RESULTS_AND_LIMITATIONS.md`: precise results and admissible journal manuscript wording.
- `Stage7F_Independent_Ensemble_Summary.csv`, `Stage7F_Independent_Sample_Level.csv`, `Stage7F_Independent_GT_Macro_Results.csv`, `Stage7F_Independent_Per_GT_Identity.csv`, `Stage7F_Independent_Paired_Window_Differences.csv`: reproducible metrics.
- `Stage7F_Independent_Audit_Manifest.json`: SHA and observed execution-log checks.
- `Figure_Stage7F_Ensemble_vs_Stationary.png` and `Figure_Stage7F_Paired_Improvement_Distribution.png`: diagnostic figures.
- `analyze_stage7f.py`: independently recompute ensemble/stationary ADE/FDE directly from the provided NPZ arrays.
- `fix_stage7f_reporting_wsl.py`: optional guarded one-line patch to the ORIGINAL Stage7F Python runner. Does not retrain model or change predictions; rerunning will reconstruct metrics including per-seed results and provenance.
- `Original_Stage7F_Tracker_Forecast_Review.zip`: source of saved predictions and logs; archive lacked final official summary due reporting error.

To run optional patch after copying this audit ZIP into Windows Downloads and extracting under `~/cp10_stage7f_audit`:

```bash
mkdir -p ~/cp10_stage7f_audit
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage7F_Independent_Audit.zip" ~/cp10_stage7f_audit
python3 ~/cp10_stage7f_audit/fix_stage7f_reporting_wsl.py
bash ~/cp10_stage7f_tools/CP10_Stage7F_Frozen_Tracker_Forecasting/run_stage7f_wsl.sh
```

The patch makes a timestamp-preserving copy of the unpatched source as `.py.before_stage7f_reporting_fix` and refuses unknown code. Previous Stage7D/E successful review ZIPs are needed for the rerun, as before.
