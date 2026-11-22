import math
import unittest
import numpy as np
from run_audit import q_conformal

class QuantileUnitTests(unittest.TestCase):
    def test_finite_sample_rank(self):
        self.assertEqual(q_conformal(np.arange(1,20,dtype=float),0.90),18)
    def test_infinite_when_insufficient(self):
        self.assertTrue(math.isinf(q_conformal([0.1,0.2,0.3],0.90)))
    def test_reject_nan(self):
        with self.assertRaises(ValueError):q_conformal([1.0,np.nan])
    def test_reject_negative(self):
        with self.assertRaises(ValueError):q_conformal([1.0,-1.0])

if __name__=='__main__':unittest.main()
