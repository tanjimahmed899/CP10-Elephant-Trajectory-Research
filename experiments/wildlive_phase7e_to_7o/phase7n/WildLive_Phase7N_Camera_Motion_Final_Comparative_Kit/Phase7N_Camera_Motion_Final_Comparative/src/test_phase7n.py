import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from phase7n import (
    DATA_FILES, apply_h, cam_pair_rows, exclusion_mask, frame_registration,
    get_annotation_boxes, score_tables, sha256
)

class TestPhase7N(unittest.TestCase):
    def test_nondestructive_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'x';f.write_bytes(b'abc')
            self.assertEqual(sha256(f),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
    def test_four_data_sources(self):
        self.assertEqual(set(DATA_FILES),{'DJI_0207','DJI_0204_video2','DJI_0204_video1','DJI_0161_video1'})
    def test_group_risk_metadata(self):
        self.assertEqual(DATA_FILES['DJI_0204_video1'][-1],DATA_FILES['DJI_0204_video2'][-1])
        self.assertNotEqual(DATA_FILES['DJI_0207'][-1],DATA_FILES['DJI_0204_video2'][-1])
    def test_homography_identity(self):
        H=np.eye(3);p=np.array([100.0,200.0]);np.testing.assert_allclose(apply_h(H,p),p)
    def test_homography_translation(self):
        H=np.array([[1,0,5],[0,1,-3],[0,0,1]],float)
        np.testing.assert_allclose(apply_h(H,[100,200]),[105,197])
    def test_exclusion_mask(self):
        M=exclusion_mask((100,100),{7:[40,40,60,60]},100,100)
        self.assertEqual(int(M[50,50]),0)
        self.assertEqual(int(M[2,2]),255)
    def test_four_and_two_point_annotation(self):
        ann={'shapes':[{'label':'elephant','group_id':5,'points':[[10,20],[20,30]]},
                       {'label':'elephant','group_id':3,'points':[[40,50],[50,50],[50,60],[40,60]]},
                       {'label':'zebra','group_id':8,'points':[[1,1],[3,3]]}]}
        with tempfile.TemporaryDirectory() as d:
            zfile=Path(d)/'a.zip'
            with zipfile.ZipFile(zfile,'w') as z:z.writestr('boxid/frame_4.json',json.dumps(ann))
            with zipfile.ZipFile(zfile) as z:
                b=get_annotation_boxes(z,4)
        self.assertEqual(b,{5:[10,20,20,30],3:[40,50,50,60]})
    def test_duplicate_track_rejected(self):
        s={'label':'elephant','group_id':0,'points':[[10,10],[15,15]]}
        with tempfile.TemporaryDirectory() as d:
            zfile=Path(d)/'a.zip'
            with zipfile.ZipFile(zfile,'w') as z:z.writestr('frame_0.json',json.dumps({'shapes':[s,s]}))
            with zipfile.ZipFile(zfile) as z:
                with self.assertRaises(ValueError):get_annotation_boxes(z,0)
    def test_no_fake_adaptive_group_comparison(self):
        rec=[]
        for sequence,group,has_adapt in [('a','SCENE','no'),('b','SCENE','yes')]:
            for model,v in [('cv_ols',2 if sequence=='a' else 4),('adaptive',1)]:
                if model=='adaptive' and has_adapt=='no':continue
                rec.append(dict(sequence=sequence,horizon_frames=20,model=model,n_windows=1,n_tracks=1,
                                role='D',scene_component=group,ADE=v,FDE=v))
        df=pd.DataFrame(rec)
        # grouping intentionally excludes adaptive when coverage differs across scene member recordings
        groups=df[df.model!='adaptive'].groupby(['scene_component','horizon_frames','model']).ADE.mean()
        self.assertEqual(float(groups.iloc[0]),3.)
        self.assertNotIn('adaptive',set(groups.index.get_level_values('model')))
    def test_registration_synthetic_translation(self):
        rng=np.random.default_rng(44)
        gray=rng.integers(0,256,(864,1536),dtype=np.uint8)
        gray=cv2.GaussianBlur(gray,(3,3),0)
        image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        M=np.float32([[1,0,6],[0,1,-4]])
        warped=cv2.warpAffine(image,M,(1536,864))
        H,info=frame_registration(image,warped,{}, {})
        self.assertTrue(info['bg_registration_qc_pass'])
        self.assertGreater(info['inliers'],100)
        np.testing.assert_allclose(apply_h(H,[500,400]),[506,396],atol=1)
    def test_per_track_motion_identity(self):
        # Testing numerical meaning: if the animal is stationary in the scene while image shifts,
        # compensated residual should be approximately zero.
        rng=np.random.default_rng(12)
        img=np.stack([rng.integers(0,256,(864,1536),dtype=np.uint8)]*3,-1)
        warped=cv2.warpAffine(img,np.array([[1,0,8],[0,1,2]],float),(1536,864))
        one={0:[490,390,510,410]}; two={0:[498,392,518,412]}
        records=cam_pair_rows('synthetic',0,1,img,warped,one,two)
        self.assertLess(records[0]['residual_after_BG_alignment_px'],1.5)

if __name__=='__main__':
    unittest.main(verbosity=2)
