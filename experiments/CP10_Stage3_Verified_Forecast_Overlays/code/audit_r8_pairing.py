#!/usr/bin/env python3
"""Audit exact sample/data lineage of corrected 30Hz trajectories vs canonical 10Hz data."""
from pathlib import Path
import io,zipfile
import numpy as np,pandas as pd
root=Path('/mnt/data');out=root/'CP10_Stage3_Verified_Forecast_Overlays'
Z=zipfile.ZipFile(root/'CP10_Stage3_R8_Matched_Data.zip');prefix='R8_inputs/Main Result/'
rows=[]
for task in ['2s_to_2s','2s_to_5s']:
 corrprefix=prefix+'corrected_trajectory_dataset/'
 ca=np.load(io.BytesIO(Z.read(corrprefix+f'corrected_{task}.npz')))
 cm=pd.read_csv(io.BytesIO(Z.read(corrprefix+f'corrected_windows_{task}.csv')))
 accepted=cm[cm.accepted==1].reset_index(drop=True)
 idx=pd.Series(accepted.index.values,index=accepted.source_index.astype(int).values)
 for split,name in [('train_real','birdsai_trainval'),('test_real','birdsai_test')]:
  rp=prefix+'final_neural_trajectory_dataset/'
  ref=np.load(io.BytesIO(Z.read(rp+f'{task}_{name}_canonical.npz')))
  meta=pd.read_csv(io.BytesIO(Z.read(rp+f'{task}_{name}_metadata.csv')))
  matches=idx.reindex(meta.source_index.astype(int).values)
  valid=matches.notna().values
  sub=matches[valid].astype(int).values
  h=np.abs(ca['history'][sub,::3,:2]-ref['history_abs'][valid])
  f=np.abs(ca['future'][sub,::3,:2]-ref['future_abs'][valid])
  rows.append(dict(task=task,split=split,canonical_samples=len(meta),corrected_accepted_samples=len(accepted[accepted.split==split]),
    matched_samples=int(valid.sum()),unmatched_samples=int((~valid).sum()),
    max_abs_history_coordinate_difference=float(h.max()) if len(h) else None,
    max_abs_future_coordinate_difference=float(f.max()) if len(f) else None,
    outcome='EXACT_MATCH' if valid.all() and np.max(h)==0 and np.max(f)==0 else 'NEEDS_REVIEW'))
pd.DataFrame(rows).to_csv(out/'R8_Corrected_to_Canonical_Exact_Pairing_Audit.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
