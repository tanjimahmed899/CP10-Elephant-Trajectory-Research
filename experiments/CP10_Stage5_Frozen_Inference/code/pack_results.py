from pathlib import Path
import zipfile
root=Path.home()/"CP10_Stage5_Frozen_Inference_Results"
required=[root/f"2s_to_{h}s_FROZEN_PREDICTIONS_FOR_CALIBRATION.npz" for h in (2,5)]
required.extend([root/f"2s_to_{h}s_fresh_metadata.csv" for h in (2,5)])
missing=[str(x) for x in required if not x.is_file()]
if missing:
 raise SystemExit("Cannot pack incomplete inference. Missing:\n" + "\n".join(missing))
out=Path.home()/"CP10_Stage5_Frozen_Results_Review.zip"
files=[p for p in root.glob("*") if p.is_file() and p.suffix in (".npz",".csv",".txt",".json")]
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.name)
print("Created",out,"files",len(files),"bytes",out.stat().st_size)
