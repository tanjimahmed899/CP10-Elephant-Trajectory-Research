import unittest, importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit_dji0204.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class QA(unittest.TestCase):
    def test_iou_identical(self):self.assertAlmostEqual(audit.bbox_iou((0,0,10,10),(0,0,10,10)),1)
    def test_iou_nonoverlap(self):self.assertEqual(audit.bbox_iou((0,0,1,1),(2,2,3,3)),0)
    def test_iou_partial(self):self.assertAlmostEqual(audit.bbox_iou((0,0,2,2),(1,1,3,3)),1/7)
if __name__=='__main__':unittest.main()
