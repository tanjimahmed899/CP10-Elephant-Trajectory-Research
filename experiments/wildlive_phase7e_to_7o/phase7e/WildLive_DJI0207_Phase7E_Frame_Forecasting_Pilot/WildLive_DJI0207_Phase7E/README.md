# DJI_0207 Phase 7E: frame-only forecasting (private research pilot)

Required: Python 3.10+, NumPy, Matplotlib. The source `WildLive_DJI_0207_boxid.zip` is provided by the researcher **separately**, not redistributed here.

Run from Ubuntu/WSL after extracting this archive:

```bash
python3 -m venv ~/cp10-pilot-venv
source ~/cp10-pilot-venv/bin/activate
pip install numpy matplotlib
python src/test_phase7e.py
python src/run_phase7e.py \
  --zip ~/WildLive_DJI_0207_boxid.zip \
  --output ./local_recomputed_outputs
```

The included `outputs/` folder contains the actual metrics produced in this session, along with original source-derived centre coordinates and arrays. Keep derived annotations/private research results outside any **public GitHub** repository until rights are confirmed. No video file or source JPEG is included in this package.

Read `PHASE7E_FRAME_BASED_FORECASTING_REPORT.md` before interpreting the numbers.
