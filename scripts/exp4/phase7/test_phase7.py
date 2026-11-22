import csv,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase7_intake_gate import intake,REQ

class Phase7Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        p=Path(self.tmp.name)
        self.media=p/'clip.mp4';self.media.write_bytes(b'FAKE_MEDIA_FOR_UNIT_TEST_ONLY')
        self.ann=p/'labels.csv';self.ann.write_text('synthetic,unit,test\n')
        import hashlib
        self.row={k:'' for k in REQ}
        self.row.update(candidate_id='test_source',recording_id='test_clip_01',collection_group_id='new_group_01',capture_event_id='new_event_01',subject_group_id='new_herd_01',sensor_modality='RGB',species='elephant',raw_video_path=str(self.media),raw_file_sha256=hashlib.sha256(self.media.read_bytes()).hexdigest(),annotations_path=str(self.ann),annotation_sha256=hashlib.sha256(self.ann.read_bytes()).hexdigest(),track_ids_present='yes',camera_pose_present='no',temporal_timestamps_present='yes',source_rights_review='approved_for_this_research',overlap_review='verified_no_overlap',overlap_evidence_path='test_review.md',preregistration_reference='test_prereg_not_real',split='confirmation',source_date_utc='2026-01-01')
    def file(self,rows):
        f=Path(self.tmp.name)/'intake.csv'
        columns=list(self.row)
        with f.open('w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=columns);w.writeheader();w.writerows(rows)
        return f
    def test_empty_template_is_not_data(self):
        a=intake(self.file([]));self.assertEqual(a['status'],'NO_INPUT_RECORDINGS')
    def test_metadata_only_pass_has_disclaimer(self):
        a=intake(self.file([self.row]));self.assertEqual(a['status'],'METADATA_CHECKS_PASSED_HUMAN_REVIEW_REQUIRED');self.assertTrue(a['not_proof_of_independence'])
    def test_historical_group_blocked(self):
        r=dict(self.row,collection_group_id='elep2');a=intake(self.file([r]));self.assertEqual(a['status'],'BLOCKED')
    def test_unapproved_overlap_blocked(self):
        r=dict(self.row,overlap_review='unknown');a=intake(self.file([r]));self.assertEqual(a['status'],'BLOCKED')
    def test_bad_raw_hash_blocked(self):
        r=dict(self.row,raw_file_sha256='0'*64);a=intake(self.file([r]));self.assertTrue(any('SHA256 mismatch' in z for z in a['errors']))
    def test_same_capture_event_cannot_be_two_groups(self):
        r=dict(self.row,recording_id='test_clip_02',collection_group_id='new_group_02')
        a=intake(self.file([self.row,r]));self.assertTrue(any('split into multiple group' in z for z in a['errors']))
    def test_same_subject_cannot_be_two_groups(self):
        r=dict(self.row,recording_id='test_clip_02',capture_event_id='new_event_02',collection_group_id='new_group_02')
        a=intake(self.file([self.row,r]));self.assertTrue(any('subject group' in z for z in a['errors']))
    def test_no_pose_blocks_fixed_reference_only(self):
        a=intake(self.file([self.row]));self.assertTrue(any('fixed-reference arm NOT eligible' in z for z in a['warnings']))
if __name__=='__main__':unittest.main()
