#!/usr/bin/env python3
"""Offline tamper/regression checks. Uses public-facing Phase7K archived output when available."""
from __future__ import annotations
import io,json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
import pandas as pd
import phase7l as s

FIX=Path(os.environ.get('PHASE7L_FIXTURE','/mnt/data/Phase7K_DJI0161_Locked_Evaluation.zip'))
ANN=Path(os.environ.get('PHASE7L_ANN','/mnt/data/DJI_0161_video1_boxid.zip'))

class Phase7LTests(unittest.TestCase):
 def test_lock(self):
  self.assertEqual(s.lock()['primary_comparator'],'cv_ols')
 def test_register_count(self):
  with tempfile.TemporaryDirectory() as t:
   a=s.make_register(Path(t)/'reg.csv');self.assertEqual(len(a),22)
   self.assertEqual(len(set(x['sequence_id'] for x in a)),22)
 def test_0204_group_related(self):
  self.assertEqual(s.family('DJI_0204_video1'),s.family('DJI_0204_video2'))
 def test_0117_group_related(self):
  self.assertEqual(s.family('DJI_0117_video1'),s.family('DJI_0117_video4'))
 def test_2023_group_related(self):
  self.assertEqual(s.family('DJI_20230719145427_0002_V_video2'),s.family('DJI_20230719145427_0002_V_video5'))
 def test_0161_seen(self):self.assertIn('DJI_0161_video1',s.ALREADY_SEEN)
 def test_stem_does_not_prove_independence(self):
  with tempfile.TemporaryDirectory() as t:
   a=s.make_register(Path(t)/'reg.csv')
   self.assertTrue(all(x['independent_capture_event_verified']=='NO' for x in a))
 def test_provenance_pending_not_accepted(self):
  with tempfile.TemporaryDirectory() as t:
   src=Path(t)/'e.csv';src.write_text('sequence_id,original_video_or_flight_log,capture_datetime_utc_with_source,camera_device_or_flight_id,overlap_review_against_development,evidence_location,reviewer,review_date,reviewer_assessment\nDJI_0142_video1,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,independent\n')
   rows=s.provenance_review(src,Path(t)/'status.csv')
   self.assertFalse(rows[0]['independent_capture_event_verified_by_software'])
 def test_provenance_filled_not_automatically_accepted(self):
  with tempfile.TemporaryDirectory() as t:
   src=Path(t)/'e.csv';src.write_text('sequence_id,original_video_or_flight_log,capture_datetime_utc_with_source,camera_device_or_flight_id,overlap_review_against_development,evidence_location,reviewer,review_date,reviewer_assessment\nDJI_0142_video1,video_1,2025-01-01T12:00Z,FlightA,visual_review,private_video,Supervisor,2026-11-22,independent\n')
   rows=s.provenance_review(src,Path(t)/'status.csv')
   self.assertFalse(rows[0]['independent_capture_event_verified_by_software'])
   self.assertEqual(rows[0]['software_status'],'REVIEW_REQUIRES_HUMAN_SOURCE_VERIFICATION')
 @unittest.skipUnless(FIX.is_file(),'Prior Phase7K fixture not mounted')
 def test_actual_archive(self):
  a=s.load_archive(FIX,ANN if ANN.is_file() else None)
  self.assertEqual(a['n_windows'],2892)
  self.assertEqual(len(a['rows']),4)
 @unittest.skipUnless(FIX.is_file(),'Prior Phase7K fixture not mounted')
 def test_duplicate_archive_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaisesRegex(ValueError,'Repeated sequence'):
    s.audit_eval([FIX,FIX],Path(t))
 @unittest.skipUnless(FIX.is_file(),'Prior Phase7K fixture not mounted')
 def test_mismatched_annotation_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   bogus=Path(t)/'bad.zip';bogus.write_bytes(b'fake annotations')
   with self.assertRaisesRegex(ValueError,'Annotation source hash mismatch'):
    s.load_archive(FIX,bogus)
 def _mutate_archive(self,edit):
  # only for local fixture tests; original archive untouched
  z=zipfile.ZipFile(FIX)
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'mutant.zip'
   with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as zout:
    for fn in z.namelist():
     if fn.endswith('/'):
      zout.writestr(fn,b'');continue
     data=z.read(fn)
     if fn.endswith('all_experts_and_gate_per_window.csv'):
      df=pd.read_csv(io.BytesIO(data));df=edit(df)
      data=df.to_csv(index=False).encode()
     zout.writestr(fn,data)
   return s.load_archive(path)
 @unittest.skipUnless(FIX.is_file(),'Prior Phase7K fixture not mounted')
 def test_hindsight_or_adaptive_tamper_rejected(self):
  def change(d):d.loc[0,'adaptive_ade_norm']=float(d.loc[0,'adaptive_ade_norm'])+1;return d
  with self.assertRaisesRegex(ValueError,'adaptive-choice mismatch'):
   self._mutate_archive(change)
 @unittest.skipUnless(FIX.is_file(),'Prior Phase7K fixture not mounted')
 def test_duplicate_window_rejected(self):
  def change(d):d.loc[1,list(s.KEYS)]=d.loc[0,list(s.KEYS)].to_numpy();return d
  with self.assertRaisesRegex(ValueError,'duplicated prediction window'):
   self._mutate_archive(change)
 @unittest.skipUnless(FIX.is_file(),'Prior Phase7K fixture not mounted')
 def test_malformed_provenance_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'a.csv';p.write_text('sequence_id,claim\na,b\n')
   with self.assertRaisesRegex(ValueError,'mandatory provenance'):
    s.provenance_review(p,Path(t)/'o.csv')

if __name__=='__main__':unittest.main(verbosity=2)
