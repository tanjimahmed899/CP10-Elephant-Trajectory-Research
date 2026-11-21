import importlib.util,json,csv,sys,os
from pathlib import Path
r=Path(os.environ.get("CP10_JOURNAL_ROOT", "/mnt/e/AIUB/Capstone Project/Journal Based Results"))
needed=["numpy","pandas","torch"]
missing=[m for m in needed if importlib.util.find_spec(m) is None]
print("Python:",sys.version.split()[0], "missing modules:",missing)
for name in ["DJI_20240119152552_0006_V.zip","DJI_20240626143832_0004_V.zip","DJI_20260227084907_0004_V.zip"]:
    print(name, any(p.exists() for p in [r/"Main Result"/name,r/"ZIP 4.0"/name,r/"Q1_Research_Completion_Sprint"/"new_data"/("elep1" if "202401" in name else ("elep2" if "202406" in name else "elep3"))/name]))
for h in (2,5):
    for seed in (42,52,62):
        p=r/"Main Result"/"transformer_results"/f"2s_to_{h}s_Transformer_seed{seed}.pt"
        print("CHECKPOINT",p.name,p.exists())
missing_files=[]
for name in ["DJI_20240119152552_0006_V.zip","DJI_20240626143832_0004_V.zip","DJI_20260227084907_0004_V.zip"]:
 group = "elep1" if "202401" in name else ("elep2" if "202406" in name else "elep3")
 if not any(p.exists() for p in [r/"Main Result"/name,r/"ZIP 4.0"/name,r/"Q1_Research_Completion_Sprint"/"new_data"/group/name]):missing_files.append(name)
for h in (2,5):
 for seed in (42,52,62):
  path=r/"Main Result"/"transformer_results"/f"2s_to_{h}s_Transformer_seed{seed}.pt"
  if not path.is_file():missing_files.append(str(path))
if missing_files:print("Missing required source files:",*missing_files,sep="\n - ")
if missing or missing_files:sys.exit(2)
