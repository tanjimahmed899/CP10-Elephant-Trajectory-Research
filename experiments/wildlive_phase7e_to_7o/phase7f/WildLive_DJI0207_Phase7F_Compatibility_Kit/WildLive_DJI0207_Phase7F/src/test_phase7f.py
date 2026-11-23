import os,sys,unittest,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from phase7f_preflight_and_inference import (canonicalize,reconstruct,heading_from_history,stationary,
cv_ols,model_factory,read_data,TASKS)

class Phase7FTests(unittest.TestCase):
 def test_roundtrip(self):
  rng=np.random.default_rng(7);x=rng.normal(.4,.01,(21,20,2))
  m=canonicalize(x,.0682)
  np.testing.assert_allclose(reconstruct(m['history_canonical'],m),x,atol=1e-8)
 def test_invariance_to_shift(self):
  x=np.linspace(0,.2,20)[None,:,None]*np.array([1,2])[None,None,:]
  m1=canonicalize(x,.05);m2=canonicalize(x+.3,.05)
  np.testing.assert_allclose(m1['history_canonical'],m2['history_canonical'],atol=1e-6)
 def test_stationary(self):
  x=np.zeros((2,20,2));x[0,-1]=[.2,.3]
  p=stationary(x,50);self.assertEqual(p.shape,(2,50,2));np.testing.assert_allclose(p[0],np.tile([.2,.3],(50,1)))
 def test_heading(self):
  t=np.arange(20,dtype=float)
  x=np.stack((t*2,t),axis=-1)[None,:,:]
  self.assertAlmostEqual(float(heading_from_history(x)[0]),np.arctan2(1,2),places=12)
 def test_cv_ols(self):
  t=np.arange(20,dtype=float);x=np.stack((t,t*2),axis=1)[None,:,:]
  pred=cv_ols(x,3)
  np.testing.assert_allclose(pred[0],[[20,40],[21,42],[22,44]],atol=1e-12)
 def test_matches_scalar_exp3_canonicalization(self):
  # Independent scalar restatement of the original EXP3 canonicalize_window.
  rng=np.random.default_rng(123); xs=rng.uniform(.25,.75,(8,20,2))
  actual=canonicalize(xs,.06820264416877736)
  for j,x in enumerate(xs):
   recent=x[-10:]; tt=np.arange(10,dtype=float); tc=tt-tt.mean(); denom=(tc**2).sum()
   vx=np.sum(tc*(recent[:,0]-recent[:,0].mean()))/denom
   vy=np.sum(tc*(recent[:,1]-recent[:,1].mean()))/denom
   a=float(np.arctan2(vy,vx)) if np.hypot(vx,vy)>1e-10 else 0.
   path=np.linalg.norm(np.diff(x,axis=0),axis=1).sum();sc=max(path,.06820264416877736)
   R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
   should=(x-x[-1])@R.T/sc
   np.testing.assert_allclose(actual['history_canonical'][j],should,atol=1e-7)
 def test_scale_floor(self):
  x=np.zeros((1,20,2)); meta=canonicalize(x,.0682)
  self.assertAlmostEqual(float(meta['scale'][0]),.0682)
 def test_real_npz(self):
  p=Path(os.environ.get('PHASE7F_TEST_NPZ','/mnt/data/WildLive_DJI0207_Phase7E/outputs/frame_based_windows_and_baselines.npz'))
  if not p.exists():self.skipTest('local test fixture missing')
  ds=read_data(p);self.assertEqual(sum(len(v['history']) for v in ds.values()),4884)
 def test_model_architecture_synthetic(self):
  try:import torch
  except ImportError:self.skipTest('torch not installed')
  m=model_factory(20,{})
  out=m(torch.zeros((2,20,2),dtype=torch.float32))
  self.assertEqual(tuple(out.shape),(2,20,2))
  m=model_factory(50,{})
  self.assertEqual(tuple(m(torch.zeros((1,20,2))).shape),(1,50,2))

if __name__=='__main__':unittest.main()
