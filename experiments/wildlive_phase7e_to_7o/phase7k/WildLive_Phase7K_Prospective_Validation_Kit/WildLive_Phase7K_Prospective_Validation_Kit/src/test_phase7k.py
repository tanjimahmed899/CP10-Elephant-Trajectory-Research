#!/usr/bin/env python3
"""Regression tests for Phase7K. Archived DJI_0204 is DEVELOPMENT only."""
import io, json, sys, unittest, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase7k import (ROOT, DEVELOPMENT, verify_assets, digest, build_observation_features,
   predict_gate_from_features, score_with_frozen_choices, summarize_scored,run_eval)
from phase7h import inventory
from phase7j import FEATURES,EXPERTS

def locate(name):
    for root in (Path('/mnt/data'),Path('/mnt/c/Users/HP/Downloads')):
        p=root/name
        if p.is_file():return p
    return Path('/NOT_AVAILABLE')

class Phase7KTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ann=locate('DJI_0204_video2_boxid.zip')
        cls.metr=locate('Phase7H_DJI0204_Frozen_Evaluation.zip')
        cls.p7i=locate('WildLive_Phase7I_Reproduced_Results.zip')
        cls.p7j=locate('WildLive_Phase7J_Reproduced_Results.zip')
        if not all(x.is_file() for x in (cls.ann,cls.metr,cls.p7i,cls.p7j)):
            raise unittest.SkipTest('Original development fixture archives not available; portable structural tests must be run separately.')
        cls.inv,_,segs=inventory(cls.ann,'DJI_0204_video2')
        cls.features=build_observation_features(segs,'DJI_0204_video2')
        cls.choices=predict_gate_from_features(cls.features)
        with zipfile.ZipFile(cls.metr) as z:
            name=next(n for n in z.namelist() if n.endswith('locked_model_per_window.csv'))
            cls.orig=pd.read_csv(io.BytesIO(z.read(name)))
        cls.scores=score_with_frozen_choices(cls.features,cls.choices,cls.orig.to_dict('records'))
    def test_locked_hashes(self):
        self.assertEqual(len(verify_assets()['model_checkpoint_sha256']),6)
    def test_exact_development_gate_bytes(self):
        with zipfile.ZipFile(self.p7j) as z:
            for h in (20,50):
                name=f'phase7j_dev_gate_h{h}.json'
                a=z.read(next(n for n in z.namelist() if n.endswith(name)))
                # Cross-platform BLAS roundoff makes byte hashes different by ~1e-15.
                # Phase7K locks the ORIGINAL published model version, not the re-fit bytes.
                local=(ROOT/'locked_models'/name).read_bytes()
                self.assertTrue(np.allclose(
                    np.array(json.loads(a)['weights'],float),
                    np.array(json.loads(local)['weights'],float),atol=1e-12,rtol=1e-12))
                self.assertEqual(json.loads(a)['train_sequences'],json.loads(local)['train_sequences'])
    def test_original_counts(self):
        self.assertEqual(self.inv['n_annotation_frames'],357)
        self.assertEqual(len(self.features),2424)
        self.assertEqual(self.inv['n_20_frame_windows'],1272)
        self.assertEqual(self.inv['n_50_frame_windows'],1152)
    def test_unique_window_identity(self):
        keys=['sequence','horizon_frames','track_id','segment_id','first_observed_frame']
        self.assertFalse(self.features.duplicated(keys).any())
        self.assertEqual(len(self.scores),len(self.features))
    def test_no_outcome_in_gate_input(self):
        self.assertFalse(any(s in self.features.columns for s in ('_ade_norm','_fde_norm')))
        self.assertTrue(self.choices.selected_expert.isin(EXPERTS).all())
    def test_locked_feature_reproduction(self):
        with zipfile.ZipFile(self.p7i) as z:
            name=next(n for n in z.namelist() if n.endswith('per_window_observation_motion_and_model_errors.csv'))
            archived=pd.read_csv(io.BytesIO(z.read(name)))
        archived=archived[archived.sequence.eq('DJI_0204_video2')].copy()
        archived['track_id']=archived['track_id'].astype(str)
        expected=self.features.merge(archived,on=['sequence','horizon_frames','track_id','first_observed_frame'],validate='one_to_one',suffixes=('_new','_old'))
        self.assertEqual(len(expected),2424)
        for feat in FEATURES:
            a=expected[f'{feat}_new'].to_numpy(float);b=expected[f'{feat}_old'].to_numpy(float)
            self.assertTrue(np.allclose(a,b,atol=1e-10,rtol=1e-7,equal_nan=True),feat)
    def test_scored_same_expert_ground_truth(self):
        s=self.scores
        for expert in EXPERTS:
            mask=s.selected_expert.eq(expert)
            self.assertTrue(np.allclose(s.loc[mask,'adaptive_ade_norm'],s.loc[mask,expert+'_ade_norm']))
            self.assertTrue(np.allclose(s.loc[mask,'adaptive_fde_norm'],s.loc[mask,expert+'_fde_norm']))
    def test_correct_score_distribution(self):
        summ=summarize_scored(self.scores)
        self.assertEqual(len([r for r in summ if r['track_id']=='pooled']),2)
        for r in summ:
            self.assertAlmostEqual(sum(r[f'selected_{n}_fraction'] for n in EXPERTS),1.,places=12)
    def test_reject_duplicate_future_scores(self):
        rows=self.orig.to_dict('records'); rows.append(rows[0].copy())
        with self.assertRaises(ValueError):score_with_frozen_choices(self.features,self.choices,rows)
    def test_reject_unmatched_future_scores(self):
        rows=self.orig.to_dict('records')[:-1]
        with self.assertRaises(ValueError):score_with_frozen_choices(self.features,self.choices,rows)
    def test_reject_development_promotion(self):
        with self.assertRaisesRegex(ValueError,'Development recording'):
            run_eval(self.ann,'DJI_0204_video2','/does/not/exist','/tmp/never-create')

class PortableLockAndLeakTests(unittest.TestCase):
    """These tests run on Ubuntu even when archived fixtures are not mounted."""
    def test_portable_release_lock(self):
        self.assertEqual(len(verify_assets()['release_files_sha256']),8)
    def test_six_frozen_checkpoint_entries(self):
        lock=verify_assets()
        self.assertEqual(len(lock['model_checkpoint_sha256']),6)
        self.assertEqual(lock['primary_comparator'],'cv_ols')
    def test_no_test_recordings_in_gate_fit(self):
        for h in [20,50]:
            gate=json.loads((ROOT/'locked_models'/f'phase7j_dev_gate_h{h}.json').read_text())
            self.assertEqual(set(gate['train_sequences']),DEVELOPMENT)
    def test_synthetic_observation_only_selection(self):
        from phase7h import make_windows
        from phase7k import build_observation_features,predict_gate_from_features
        sample=np.stack([np.linspace(0.10,0.12,100),np.linspace(0.32,0.33,100)],axis=-1)
        seg=[{'coords':sample,'track_id':'0','segment_id':0,'start_frame':0,'end_frame':99}]
        f=build_observation_features(seg,'SYNTHETIC_DO_NOT_REPORT')
        self.assertEqual(len(f), (100-20-20+1)+(100-20-50+1))
        self.assertFalse(any(str(col).endswith('_ade_norm') for col in f.columns))
        selections=predict_gate_from_features(f)
        self.assertEqual(len(selections),len(f))
        self.assertTrue(selections.selected_expert.isin(EXPERTS).all())
    def test_duplicate_sequence_not_new_confirmation(self):
        self.assertIn('DJI_0207',DEVELOPMENT)
        self.assertIn('DJI_0204_video2',DEVELOPMENT)

if __name__=='__main__':unittest.main()
