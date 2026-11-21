# Stage 7 verified runner (WSL)

The previous `CP10_Stage7_Dev_Review.zip` contained five preflight files only. This utility creates a NEW output directory and captures exact import errors and run logs even when inference fails.

Download the ZIP into C:\Users\HP\Downloads and run from the Python environment intended for Stage 7:

```bash
mkdir -p ~/cp10_stage7_recovery
python3 -m zipfile -e "/mnt/c/Users/HP/Downloads/CP10_Stage7_Recovery_Utility.zip" ~/cp10_stage7_recovery
python3 -u ~/cp10_stage7_recovery/stage7_verified_runner.py
```

If dependencies are missing: the runner stops WITHOUT modifying your Python environment. It puts `CP10_Stage7_Run_Diagnostics.zip` in Windows Downloads. Upload that file. Fix the exact dependency issue before inference.

If complete: it puts `CP10_Stage7_Dev_RealInference_Review.zip` AND `CP10_Stage7_Run_Diagnostics.zip` in Windows Downloads. Upload the real-inference ZIP.

The utility makes NO model checkpoint changes, does NO TestReal evaluation or tuning, and installs NO packages. It uses the Python interpreter that runs the utility, as shown in diagnostic output.
