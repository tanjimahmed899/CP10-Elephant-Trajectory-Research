"""Independent structural assertions on saved Phase 7O analysis outputs, without original images."""
from pathlib import Path
import sys, json
import pandas as pd
import numpy as np
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'results'
r=pd.read_csv(p/'heldout_registration_models.csv')
a=pd.read_csv(p/'per_animal_registration_model_sensitivity.csv')
e=pd.read_csv(p/'high_residual_QA_events_NOT_confirmed_errors.csv')
w=pd.read_csv(p/'per_window_QA_exposure.csv')
g=pd.read_csv(p/'QA_exposure_ADE_sensitivity.csv')
assert len(r)==24 and len(a)==78 and len(e)==5
assert len(w)==7308
assert r.groupby(['sequence','frame_source','frame_target']).size().eq(3).all()
assert a.groupby(['sequence','frame_source','frame_target','track_id']).size().eq(3).all()
rank=r.pivot_table(index=['sequence','frame_source','frame_target'],columns='method',values='median_heldout_error_resized_px')
assert (rank.homography<rank.translation).all() and (rank.homography<rank.similarity).all()
assert e.registered_residual_px.ge(30.).all()
assert len(e[e.sequence=='DJI_0207'])==5
for seq,h,ntrue in [('DJI_0207',20,59),('DJI_0207',50,89),('DJI_0204_video2',20,0),('DJI_0204_video2',50,0)]:
    sub=w[(w.sequence==seq)&(w.horizon_frames==h)]; assert int(sub.any_event.sum())==ntrue
    agg=g[(g.sequence==seq)&(g.horizon_frames==h)]
    assert agg.n_windows.sum()==len(sub)
print('PASS: 8 pairs x 3 methods, 26 animal-pairs x 3, 5 unconfirmed QA events, 7,308 unique diagnostic windows.')
