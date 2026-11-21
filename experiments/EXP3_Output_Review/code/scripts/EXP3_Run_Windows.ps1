# Run in PowerShell after extracting this package; Python 3.10+ recommended.
# Set up a fresh environment as needed: py -m pip install huggingface_hub hf_xet pandas numpy scipy torch
# Video archives are large; Internet/disk required. Nothing here invents EXP3 accuracy.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
py scripts/verify_checkpoint_upload_hashes.py
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint input hash mismatch' }
py scripts/exp3_download_revision_locked.py --download --full-crc
if ($LASTEXITCODE -ne 0) { throw 'Download/integrity failure' }
py scripts/exp3_validate_against_publisher.py
if ($LASTEXITCODE -ne 0) { throw 'Publisher checksum mismatch or selected archive missing' }
py scripts/exp3_preflight.py --lock data/EXP3_new_video_holdout_LOCKED.csv --data-root new_data --model-root checkpoints --out EXP3_local_preflight.json
if ($LASTEXITCODE -ne 0) { throw 'Required inputs missing' }
$env:EXP3_LOCK_CSV = (Resolve-Path data/EXP3_new_video_holdout_LOCKED.csv).Path
$env:EXP3_DATA_ROOT = (Resolve-Path new_data).Path
$env:EXP3_MODEL_ROOT = (Resolve-Path checkpoints).Path
$env:EXP3_OUT_ROOT = Join-Path (Get-Location).Path 'results'
py scripts/exp3_frozen_forecaster_adapter.py
if ($LASTEXITCODE -ne 0) { throw 'Forecast evaluator failed' }
py scripts/exp3_postprocess.py --results results --lock data/EXP3_new_video_holdout_LOCKED.csv
if ($LASTEXITCODE -ne 0) { throw 'Postprocess failed' }
Write-Host 'Check results. Original Step7H warning evaluation on these new predictions still requires exact event-instance construction/replay; do not call the experiment finished.'
