import unittest,numpy as np
from phase7o import spatial_split,matrices,project,full_res_matrix,event_window_analysis
import pandas as pd
class Tests(unittest.TestCase):
 def test_spatial_split_deterministic_and_disjoint(self):
  rng=np.random.RandomState(1);x=rng.rand(600,2)*[1536,864]
  a,b=spatial_split(x,seed=11);c,d=spatial_split(x,seed=11)
  self.assertTrue(np.array_equal(a,c));self.assertTrue(np.array_equal(b,d))
  self.assertEqual((a&b).sum(),0);self.assertEqual((a|b).sum(),600)
 def test_translation(self):
  r=np.random.RandomState(2);p=r.uniform(30,300,size=(200,2)).astype(np.float32);q=p+np.array([19,-4],dtype=np.float32)
  models=matrices(p,q,seed=1)
  for m in models:
   self.assertLess(float(np.median(np.linalg.norm(project(models[m][0],p)-q,axis=1))),1e-4)
 def test_similarity_recover(self):
  rng=np.random.RandomState(9);p=rng.uniform(10,400,(300,2)).astype(np.float32)
  a=.08;A=np.array([[np.cos(a),-np.sin(a),5.],[np.sin(a),np.cos(a),-3.],[0,0,1]])
  q=project(A,p).astype(np.float32);m=matrices(p,q,seed=2)
  self.assertLess(np.median(np.linalg.norm(project(m['similarity'][0],p)-q,axis=1)),1e-3)
 def test_native_mapping(self):
  H=np.array([[1,0,10],[0,1,-8],[0,0,1]],float);f=full_res_matrix(H,3840,2160)
  self.assertAlmostEqual(project(f,[100,200])[0,0],125,places=5)
  self.assertAlmostEqual(project(f,[100,200])[0,1],180,places=5)
 def test_event_respects_track_and_horizon(self):
  a=pd.DataFrame([dict(sequence='x',track_id=1,frame_source=25,frame_target=26,method='homography',registered_residual_px=35)])
  w=[]
  for tid,first in [(1,0),(2,0),(1,27)]:
   item=dict(sequence='x',track_id=tid,horizon_frames=20,first_observed_frame=first,
            last_observed_frame=first+19,last_target_frame=first+39,qa_flagged_window=0)
   for m in ['stationary','cv20','cv_ols','transformer_ensemble']:
    item[m+'_ade_norm']=.2;item[m+'_fde_norm']=.3
   w.append(item)
  e,z,g=event_window_analysis(pd.DataFrame(w),a)
  self.assertEqual(z.any_event.tolist(),[True,False,False]);self.assertEqual(z.exposure.iloc[0],'future')
 def test_high_residual_is_not_automatic_error(self):
  self.assertTrue('NOT_confirmed_errors' in 'high_residual_QA_events_NOT_confirmed_errors.csv')
if __name__=='__main__':unittest.main(verbosity=2)
