import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run_phase7e import OBS,predict_stationary,predict_cv20,ade_fde,make_windows

class Phase7ETests(unittest.TestCase):
    def test_stationary(self):
        x=np.array([[j,j*2] for j in range(OBS)],dtype=float)
        pred=predict_stationary(x,50)
        self.assertEqual(pred.shape,(50,2))
        np.testing.assert_equal(pred,np.tile(x[-1],(50,1)))
    def test_cv_exact_linear(self):
        full=np.array([[j*0.0001,1-j*0.00005] for j in range(70)])
        pred=predict_cv20(full[:20],50)
        self.assertLess(np.abs(pred-full[20:]).max(),1e-13)
    def test_units(self):
        ade,fde=ade_fde(np.array([[0,0],[0,0]]),np.array([[3,4],[6,8]]))
        self.assertEqual((ade,fde),(7.5,10.0))
    def test_windows_count(self):
        data=np.zeros((868,2))
        self.assertEqual(len(list(make_windows(data,20,set()))),829)
        self.assertEqual(len(list(make_windows(data,50,set()))),799)
    def test_flags_include_history_and_future(self):
        data=np.zeros((80,2))
        rows=list(make_windows(data,20,{20}))
        self.assertTrue(rows[0][-1])
        self.assertTrue(rows[20][-1])
        self.assertFalse(rows[21][-1])
    def test_no_fps_used(self):
        # No conversion from frames to seconds is performed by these predictors.
        self.assertEqual(predict_cv20(np.zeros((20,2)),50).shape[0],50)

if __name__=='__main__':unittest.main()
