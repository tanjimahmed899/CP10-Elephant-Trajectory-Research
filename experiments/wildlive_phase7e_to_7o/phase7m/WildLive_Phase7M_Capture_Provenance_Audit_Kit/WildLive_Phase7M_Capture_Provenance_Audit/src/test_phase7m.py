#!/usr/bin/env python3
import unittest, tempfile, json, zipfile, sys, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase7m as p

class TestProvenance(unittest.TestCase):
  def test_original_lock_integrity(self):
    self.assertEqual(p.sha256(p.LOCK_PATH),p.EXPECTED_LOCK_SHA256)
  def test_frozen_primary_comparator(self):
    self.assertEqual(p.check_lock()['primary_comparator'],'cv_ols')
  def test_development_not_test(self):
    self.assertEqual(p.decide_groups({'sequence_id':'DJI_0207'})['acceptable_independent_external_confirmation'],'NO')
  def test_0204_conservative_grouping(self):
    x=p.decide_groups({'sequence_id':'DJI_0204_video1'})
    y=p.decide_groups({'sequence_id':'DJI_0204_video2'})
    self.assertEqual(x['provisional_group'],y['provisional_group'])
  def test_0161_independence_unverified(self):
    x=p.decide_groups({'sequence_id':'DJI_0161_video1'})
    self.assertEqual(x['independent_of_development_capture_event'],'NOT_ESTABLISHED')
  def test_all_four_not_confirmed(self):
    for seq in p.SOURCE_MAP:
      with self.subTest(seq=seq):
        self.assertEqual(p.decide_groups({'sequence_id':seq})['acceptable_independent_external_confirmation'],'NO')
  def test_known_files_are_disjoint(self):
    self.assertEqual(len(set(p.SOURCE_MAP)),4)
    self.assertEqual(set(p.EVAL_MAP),{'DJI_0204_video1','DJI_0161_video1'})
  def test_small_synthetic_annotation_track(self):
    with tempfile.TemporaryDirectory() as t:
      f=Path(t)/'a.zip'
      with zipfile.ZipFile(f,'w') as z:
        for i in range(71):
          o={'imageWidth':3840,'imageHeight':2160,'imagePath':f'frame_{i}.jpg','shapes':[{'group_id':0,'label':'elephant','points':[[10,10],[20,20]]}]}
          z.writestr(f'boxid/frame_{i}.json',json.dumps(o))
      r=p.audit_annotations(f,'SYNTH')
      self.assertEqual(r['elephant_boxes'],71)
      self.assertEqual(r['eligible_h20_windows'],32)
      self.assertEqual(r['eligible_h50_windows'],2)
      self.assertEqual(r['authoritative_capture_fields_populated'],'NONE')
      self.assertEqual(r['missing_elephant_scores'],71)
  def test_synthetic_frame_gap_prevents_spanning_windows(self):
    with tempfile.TemporaryDirectory() as t:
      f=Path(t)/'a.zip'
      with zipfile.ZipFile(f,'w') as z:
        for i in list(range(30))+list(range(60,90)):
          o={'imageWidth':3840,'imageHeight':2160,'shapes':[{'group_id':0,'label':'elephant','points':[[2,2],[3,3]]}]}
          z.writestr(f'boxid/frame_{i}.json',json.dumps(o))
      r=p.audit_annotations(f,'SYNTH')
      self.assertEqual(r['eligible_h20_windows'],0)
      self.assertEqual(r['eligible_h50_windows'],0)
  def test_bad_checkpoint_hash_rejected(self):
    lock=p.check_lock()
    with tempfile.TemporaryDirectory() as t:
      f=Path(t)/'a.zip'
      wrong={'sequence_id':'DJI_0161_video1','file_sha256':'abc','gate_sha256':{},'checkpoint_hashes_verified':{}}
      with zipfile.ZipFile(f,'w') as z:z.writestr('evaluation_manifest.json',json.dumps(wrong))
      with self.assertRaisesRegex(ValueError,'Gate SHA mismatch'):
        p.verify_evaluation(f,'DJI_0161_video1','abc',lock)
  def test_wrong_annotation_sha_rejected(self):
    lock=p.check_lock()
    with tempfile.TemporaryDirectory() as t:
      f=Path(t)/'a.zip'
      wrong={'sequence_id':'DJI_0161_video1','file_sha256':'BAD','gate_sha256':{},'checkpoint_hashes_verified':{}}
      with zipfile.ZipFile(f,'w') as z:z.writestr('evaluation_manifest.json',json.dumps(wrong))
      with self.assertRaisesRegex(ValueError,'annotation sha mismatch'):
        p.verify_evaluation(f,'DJI_0161_video1','abc',lock)
  def test_no_exif_does_not_become_verified(self):
    from PIL import Image
    import io
    with tempfile.TemporaryDirectory() as t:
      f=Path(t)/'x.zip';out=io.BytesIO();Image.new('RGB',(32,32)).save(out,format='JPEG')
      with zipfile.ZipFile(f,'w') as z:z.writestr('frame_0.jpg',out.getvalue())
      a=p.sample_image_meta(f,'SYNTH')
      self.assertEqual(a[0]['exif_capture_datetime_present'],False)
      self.assertEqual(a[0]['usable_authoritative_flight_id_present'],False)
  def test_false_positive_avoidance(self):
    import numpy as np,cv2
    a=np.zeros((720,1280,3),np.uint8)
    b=np.zeros((720,1280,3),np.uint8)
    _,a0=cv2.imencode('.jpg',a);_,b0=cv2.imencode('.jpg',b)
    r=p.scene_compare(a0.tobytes(),b0.tobytes())
    self.assertEqual(r['scene_overlap_evidence'],'INCONCLUSIVE_SAMPLE')
    self.assertEqual(r['independence_verified'],'NO')
  def test_all_source_images_omitted_from_package_by_design(self):
    self.assertFalse(any('jpg' in x for x in p.SOURCE_MAP))

if __name__=='__main__':unittest.main(verbosity=2)
