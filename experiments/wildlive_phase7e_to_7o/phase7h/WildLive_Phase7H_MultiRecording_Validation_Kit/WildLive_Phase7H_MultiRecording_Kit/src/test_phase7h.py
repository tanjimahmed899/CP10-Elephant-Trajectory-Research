import io,sys,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase7h import inventory,make_windows,parse_archive,contiguous_runs,KNOWN_WILDBOX_VIDEOS
from phase7f_locked_reference import TASKS,canonicalize,reconstruct


def make_zip(path,frames=80,missing=(),kind='elephant',dupe=False):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for t in range(frames):
            if t in missing:continue
            j={'imageWidth':3840,'imageHeight':2160,'shapes':[{'label':kind,'group_id':2,
               'points':[[1500+t,600],[1620+t,800]],'shape_type':'rectangle'}]}
            z.writestr(f'boxid/frame_{t}.json',json.dumps(j))
        if dupe:z.writestr('backup/frame_0.json',json.dumps(j))

class TestPhase7H(unittest.TestCase):
    def test_continuous_count(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip';make_zip(p)
            meta,rows,segs=inventory(p,'candidate1')
            self.assertEqual((meta['n_20_frame_windows'],meta['n_50_frame_windows']),(41,11))
            self.assertEqual((len(segs),len(segs[0]['coords'])),(1,80))
            self.assertFalse(meta['new_independent_confirmation'])
    def test_missing_frame_splits_runs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip';make_zip(p,missing=[35])
            meta,rows,segs=inventory(p,'candidate2')
            self.assertEqual(len(segs),2)
            self.assertEqual(meta['n_50_frame_windows'],0)
            self.assertEqual(meta['n_20_frame_windows'],5)
    def test_no_elephant(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip';make_zip(p,kind='zebra')
            meta,rows,segs=inventory(p,'zebra')
            self.assertEqual(meta['elephant_track_ids'],0)
            self.assertFalse(segs)
    def test_duplicate_indices_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip';make_zip(p,dupe=True)
            with self.assertRaisesRegex(ValueError,'duplicate frame'):parse_archive(p)
    def test_window_no_future_leak(self):
        seg={'coords':np.arange(160,dtype=float).reshape(80,2)/1000.,'start_frame':11}
        x,y,s=make_windows(seg,50)
        self.assertEqual(x.shape,(11,20,2))
        self.assertEqual(y.shape,(11,50,2))
        self.assertEqual(int(s[-1]),21)
        self.assertTrue(np.array_equal(x[0,-1],seg['coords'][19]))
        self.assertTrue(np.array_equal(y[0,0],seg['coords'][20]))
    def test_canonical_roundtrip_reference(self):
        x=np.arange(80,dtype=float).reshape(2,20,2)/1000.
        c=canonicalize(x,TASKS[20]['scale_floor'])
        self.assertLess(np.max(np.abs(reconstruct(c['history_canonical'],c)-x)),1e-6)
    def test_known_wildbox_roster(self):
        self.assertEqual(len(KNOWN_WILDBOX_VIDEOS),6)
        self.assertIn('DJI_20260227082707_0003_V',KNOWN_WILDBOX_VIDEOS)
    def test_require_verified_timing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip';make_zip(p)
            meta,_,_=inventory(p,'candidate3')
            self.assertFalse(meta['released_frame_timestamps_verified'])
            self.assertFalse(meta['native_fps_verified'])
    def test_prior_recording_regression(self):
        p=Path('/mnt/data/WildLive_DJI_0207_boxid.zip')
        if not p.exists():self.skipTest('Only run when prior original user archive mounted')
        meta,rows,segs=inventory(p,'DJI_0207')
        self.assertEqual((meta['n_20_frame_windows'],meta['n_50_frame_windows']),(2487,2397))
        self.assertEqual(meta['labels'],{'elephant':2604})
        self.assertEqual(meta['missing_elephant_scores'],101)

if __name__=='__main__':unittest.main(verbosity=2)
