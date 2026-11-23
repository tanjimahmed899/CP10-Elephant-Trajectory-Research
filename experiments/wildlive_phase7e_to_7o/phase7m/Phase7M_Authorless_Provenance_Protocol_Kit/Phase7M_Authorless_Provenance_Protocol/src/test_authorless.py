import importlib.util, pathlib, unittest, tempfile, json, zipfile
p=pathlib.Path(__file__).with_name('authorless_provenance.py')
spec=importlib.util.spec_from_file_location('audit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestAudit(unittest.TestCase):
 def test_source_names(self):self.assertEqual(len(m.SOURCE_IMAGES),4)
 def test_norelabel_test_as_new(self):
  self.assertIn('PREVIOUSLY_EVALUATED',m.ROLES['DJI_0161_video1'])
  self.assertEqual(m.ROLES['DJI_0204_video2'],'DEVELOPMENT')
 def test_sha256(self):self.assertEqual(m.sha256_bytes(b'abc'),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
 def test_census(self):
  with tempfile.TemporaryDirectory() as t:
   z=pathlib.Path(t)/'trial_boxid.zip'
   with zipfile.ZipFile(z,'w') as f:
    f.writestr('boxid/frame_0.json',json.dumps({'shapes':[{'label':'elephant'},{'label':'zebra'}]}))
    f.writestr('boxid/frame_2.json',json.dumps({'shapes':[{'label':'zebra'}]}))
   r=m.census_zip(z)
   self.assertEqual(r['elephant_annotations'],1)
   self.assertEqual(r['frames_with_json'],2)
   self.assertEqual(r['max_frame'],2)
 def test_compare_identical_and_different(self):
  import numpy as np, cv2
  rng=np.random.default_rng(19)
  img=rng.integers(0,255,(576,1024),dtype=np.uint8)
  for x in range(20,1000,75):cv2.circle(img,(x,280),12,255,2)
  img2=np.zeros_like(img)
  for a,b in [(img,img),(img,img2)]:
   feat1=m.detect_features(cv2.imencode('.jpg',a)[1].tobytes())
   feat2=m.detect_features(cv2.imencode('.jpg',b)[1].tobytes())
   o=m.compare_features(feat1,feat2)
   self.assertEqual(o['independent_flight_proven'],'NO')
  self.assertEqual(m.compare_features(feat1,feat2)['evidence'],'INCONCLUSIVE')
if __name__=='__main__':unittest.main(verbosity=2)
