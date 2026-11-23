#!/usr/bin/env python3
"""Unit tests for Phase 7I; run inside same directory as run_phase7i.py."""
import json
import unittest
import numpy as np
import pandas as pd
from run_phase7i import motion_features, quantile_bins, compute_common_mode

class MotionFeatureTests(unittest.TestCase):
    def test_straight_motion(self):
        t=np.arange(20)[:,None]
        x=np.tile(np.array([[0.1,0.2]],float),(20,1))+t*np.array([[0.002,0.001]])
        f=motion_features(x[None])
        self.assertAlmostEqual(f['straightness'][0],1.,places=12)
        self.assertAlmostEqual(f['mean_step_norm_per_frame'][0],np.hypot(.002,.001),places=12)
        self.assertLess(f['ols_line_residual_norm'][0],1e-12)
        self.assertLess(f['acceleration_rms_norm_per_frame2'][0],1e-12)
    def test_stationary(self):
        f=motion_features(np.full((1,20,2),0.4))
        self.assertEqual(f['path_norm'][0],0)
        self.assertTrue(np.isnan(f['straightness'][0]))
        self.assertTrue(np.isnan(f['mean_abs_turn_rad'][0]))
        self.assertEqual(f['turn_pairs_used'][0],0)
    def test_zigzag(self):
        x=np.stack([np.arange(20)*.01,np.sin(np.arange(20)*np.pi/2)*.01],axis=1)
        f=motion_features(x[None]);self.assertLess(f['straightness'][0],.99)
        self.assertGreater(f['ols_line_residual_norm'][0],0)
    def test_history_only_shape(self):
        with self.assertRaises(ValueError):motion_features(np.zeros((10,50,2)))
    def test_temporal_offset_invariance(self):
        x=np.random.default_rng(4).normal(size=(2,20,2))*.001
        a=motion_features(x);b=motion_features(x+np.array([.25,.36]))
        for k in ['path_norm','net_displacement_norm','straightness','acceleration_rms_norm_per_frame2']:
            np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-12)
    def test_zero_disp_but_nonzero_path(self):
        x=np.zeros((1,20,2));x[0,1,0]=.1
        f=motion_features(x);self.assertAlmostEqual(f['straightness'][0],0)
    def test_within_group_speed_bins(self):
        a=pd.DataFrame({'sequence':['A']*6+['B']*6,'horizon_frames':[20]*12,
            'mean_step_norm_per_frame':list(range(1,7))+list(range(101,107))})
        b=quantile_bins(a,'mean_step_norm_per_frame')
        self.assertEqual(list(b.iloc[:6]),['low','low','mid','mid','high','high'])
        self.assertEqual(list(b.iloc[6:]),['low','low','mid','mid','high','high'])
    def test_animal_common_motion_proxy(self):
        a={0:{t:np.array([.1+.001*t,.1]) for t in range(8)},
           1:{t:np.array([.2+.001*t,.2]) for t in range(8)}}
        m=compute_common_mode(a)
        self.assertAlmostEqual(m['mean_residual_step_norm'],0,places=12)
        self.assertEqual(m['adjacent_frame_pairs_used'],7)
    def test_motion_proxy_not_camera_label(self):
        m=compute_common_mode({0:{0:np.array([.0,.0]),1:np.array([.1,.0])},
                            1:{0:np.array([.2,.2]),1:np.array([.3,.2])}})
        self.assertEqual(m['n_elephants'],2)
        self.assertNotIn('camera_velocity',m)

if __name__=='__main__': unittest.main()
