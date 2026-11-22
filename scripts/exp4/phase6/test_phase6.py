"""Unit tests use generated toy rows in temporary directories ONLY.
They are never evidence of wildlife model performance.
"""
import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from phase6_preflight import audit
from phase6_eval import analyze,score


def row(rec,grp,task='2s_to_2s',index=0,truth=False,z=-0.8,ts='2026-12-01T10:00:00Z'):
    return dict(recording_id=rec,collection_group_id=grp,task=task,sample_index=str(index),
                forecast_timestamp_utc=ts,track_key='trackA',axis='x',boundary_id='line50',
                direction='positive',true_crossing_bool=str(truth).lower(),
                predicted_crossing_bool='false',uncertainty_intersects_geofence='true',
                oriented_risk_score=str(z),absolute_conformal_radius='1.0',
                reference_geometry='moving_anchor',uncertainty_transport_method='native_corrected',uncertainty_transport_validated='true',ground_truth_label_source='manual_annotation_v1',
                eligible='true')

def manifest(rec,grp,split='confirmation',event='event42',inspected='false'):
    return dict(recording_id=rec,collection_group_id=grp,split=split,capture_event_id=event,
                recording_date_utc='2026-12-01',site_code='masked_A',sensor_modality='RGB',
                camera_id='cam1',subject_group_id='',raw_file_sha256=hashlib.sha256(rec.encode()).hexdigest(),
                annotation_file_sha256=hashlib.sha256((rec+'a').encode()).hexdigest(),
                previously_inspected=inspected,original_file_name=rec+'.mp4',annotation_complete='true',
                geometry_type='moving_anchor',geometry_validated='true',
                source_independence_evidence='distinct session under supervisor documentation',
                nominal_fps='30',monitoring_duration_s='60',wildlife_clearance='not_applicable_public_source',
                airspace_clearance='not_applicable_public_source',notes='toy software test')

def save(path,rows):
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class Phase6SemanticsTest(unittest.TestCase):
    def test_fpr_recall_and_undefined(self):
        z=score([(False,True),(False,False),(True,True),(True,False)])
        self.assertEqual(z['tp'],1)
        self.assertEqual(z['fp'],1)
        self.assertAlmostEqual(z['false_alert_rate'],.5)
        self.assertAlmostEqual(z['recall'],.5)
        self.assertIsNone(score([(False,False)])['recall'])

    def test_clearance_gate_rejects_far_uncertainty(self):
        rows=[row('new1','newA',index=0,z=-.8,ts='2026-12-01T10:00:00Z'),
              row('new1','newA',index=1,z=-.8,ts='2026-12-01T10:00:01Z')]
        r=analyze(rows)
        x={t['policy']:t for t in r['pooled']}
        self.assertEqual(x['baseline_N2_available']['fp'],1)
        self.assertEqual(x['phase5_locked_N2_available']['fp'],0)

    def test_observed_clock_gap_vs_available(self):
        rows=[row('new1','newA',index=0,z=-.2,ts='2026-12-01T10:00:00Z'),
              row('new1','newA',index=4,z=-.2,ts='2026-12-01T10:00:04Z')]
        r=analyze(rows)
        x={t['policy']:t for t in r['pooled']}
        self.assertEqual(x['baseline_N2_available']['fp'],1)
        self.assertEqual(x['baseline_N2_clock_1p1s']['fp'],0)
        self.assertEqual(r['skipped_sample_index_intervals'],1)

    def test_manifest_pass_and_leak_fail(self):
        with tempfile.TemporaryDirectory() as d:
            m=Path(d)/'m.csv'; t=Path(d)/'i.csv'
            save(m,[manifest('new1','newA')])
            save(t,[row('new1','newA')])
            self.assertEqual(audit(m,t)['status'],'SCHEMA_PASS_NOT_SCIENTIFIC_CONFIRMATION')
            save(m,[manifest('new1','newA'),manifest('cal1','newA',split='calibration',event='another')])
            self.assertEqual(audit(m,t)['status'],'FAIL')

    def test_past_data_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            m=Path(d)/'m.csv'
            save(m,[manifest('new1','elep1',inspected='true')])
            self.assertEqual(audit(m)['status'],'FAIL')

if __name__=='__main__':unittest.main()
