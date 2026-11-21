#!/usr/bin/env python3
"""Package all Stage7F output files including failure logs (no loss of diagnostics)."""
from pathlib import Path
import argparse,zipfile,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--zip',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);a.zip.parent.mkdir(parents=True,exist_ok=True)
files=[f for f in sorted(a.out.rglob('*')) if f.is_file() and f!=a.zip]
with zipfile.ZipFile(a.zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in files:z.write(f,f.relative_to(a.out).as_posix())
with zipfile.ZipFile(a.zip) as z:
    bad=z.testzip()
    if bad:raise RuntimeError(f'ZIP CRC error {bad}')
print(f'PACKAGED: {a.zip} files={len(files)} size_bytes={a.zip.stat().st_size}')
