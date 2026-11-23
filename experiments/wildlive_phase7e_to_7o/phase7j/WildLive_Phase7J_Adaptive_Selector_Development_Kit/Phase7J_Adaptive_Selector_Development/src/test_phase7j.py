#!/usr/bin/env python3
import unittest, tempfile, sys, json
from pathlib import Path
from unittest.mock import patch
import pandas as pd, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase7j as mod


def toy(seq='A',h=20,n=140):
    rows=[]
    for i in range(n):
        row=dict(sequence=seq,horizon_frames=h,track_id=i%3,first_observed_frame=i,
                 mean_step_norm_per_frame=.001+i*.000001,straightness=.3,
                 ols_line_residual_norm=.0004,acceleration_rms_norm_per_frame2=.0005,
                 mean_abs_turn_rad=np.nan if i%7==0 else 1.,ols_velocity_norm_per_frame=.001,
                 stationary_ade_norm=.002,stationary_fde_norm=.004,cv_ols_ade_norm=.001,
                 cv_ols_fde_norm=.002,transformer_ensemble_ade_norm=.003,
                 transformer_ensemble_fde_norm=.006)
        rows.append(row)
    return pd.DataFrame(rows)

class TestGate(unittest.TestCase):
    def setUp(self):
        self.A=toy('A');self.B=toy('B')
    def test_valid_schema(self):
        mod.validate(pd.concat([self.A,self.B]))
    def test_duplicate_rejected(self):
        df=pd.concat([self.A,self.B,self.A.iloc[[0]]],ignore_index=True)
        with self.assertRaises(ValueError):mod.validate(df)
    def test_missing_turn_not_zero(self):
        X,fill=mod.build_raw_features(self.A)
        self.assertGreater(fill['mean_abs_turn_rad'],0)
        self.assertEqual(X[0,len(mod.FEATURES)+4],1.)
    def test_observation_only_prediction_invariant_to_future_error(self):
        model=mod.fit_selector(self.A,20,['A'])
        before=mod.select(self.B,model)
        changed=self.B.copy()
        for expert in mod.EXPERTS:
            changed[f'{expert}_ade_norm']=np.linspace(0.9,2.,len(changed))
            changed[f'{expert}_fde_norm']=100.
        np.testing.assert_array_equal(before,mod.select(changed,model))
    def test_no_sequence_used_as_feature(self):
        model=mod.fit_selector(self.A,20,['A'])
        x=self.B.copy();x['sequence']='UNKNOWN';x['track_id']=999
        np.testing.assert_array_equal(mod.select(self.B,model),mod.select(x,model))
    def test_horizon_is_separate(self):
        model=mod.fit_selector(self.A,20,['A'])
        b=self.B.copy();b['horizon_frames']=50
        with self.assertRaises(ValueError):mod.select(b,model)
    def test_nonoverlap_starts(self):
        sub=mod.nonoverlap_train_subset(self.A,20)
        self.assertTrue((sub.first_observed_frame%40==0).all())
    def test_group_disjoint(self):
        m=mod.fit_selector(self.A,20,['A'])
        self.assertNotIn('B',m['train_sequences'])
    def test_eval_argmin_and_metric(self):
        m=mod.fit_selector(self.A,20,['A'])
        s=mod.score_eval(self.B,m)
        self.assertEqual(len(s),len(self.B))
        self.assertTrue(np.isfinite(s.adaptive_ade_norm).all())
        self.assertTrue((s.hindsight_oracle_ade_norm<=s.adaptive_ade_norm+1e-15).all())
    def test_serialization_and_deployment(self):
        m=mod.fit_selector(self.A,20,['A'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.json';p.write_text(json.dumps(m))
            df=toy('C',20,20)[list(mod.FEATURES)+['horizon_frames']]
            inp=Path(d)/'f.csv';df.to_csv(inp,index=False)
            out=Path(d)/'out.csv'
            mod.run_predict(p,inp,out)
            self.assertEqual(len(pd.read_csv(out)),20)
    def test_raw_data_7i_counts(self):
        path=Path('/mnt/data/WildLive_Phase7I_Reproduced_Results.zip')
        if path.exists():
            df=mod.load_phase7i(path)
            self.assertEqual(len(df),7308)
            self.assertEqual(df.sequence.nunique(),2)

if __name__=='__main__':unittest.main()
