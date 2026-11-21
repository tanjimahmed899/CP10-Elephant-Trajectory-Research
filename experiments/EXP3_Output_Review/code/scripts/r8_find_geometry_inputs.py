#!/usr/bin/env python3
"""Locate files needed for strict R8 per-frame geometry on the original PC.
Outputs metadata only (paths/sizes); doesn't modify source material.
"""
import argparse,csv,pathlib
TERMS=('ego','homograph','affine','motion','camera','pose','transform','step2')
EXT={'.csv','.json','.npz','.npy','.txt','.mat','.pkl','.yaml','.yml'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=pathlib.Path);ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('R8_geometry_file_candidates.csv'))
    a=ap.parse_args()
    if not a.root.exists():raise SystemExit('Root does not exist')
    output=[]
    for p in a.root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in EXT:continue
        if not any(t in str(p).lower() for t in TERMS):continue
        output.append({'file':str(p.resolve()),'size_bytes':p.stat().st_size,'extension':p.suffix.lower()})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['file','size_bytes','extension']);writer.writeheader();writer.writerows(sorted(output,key=lambda d:d['file']))
    print(f'{len(output)} candidate files -> {a.output}')
    print('Relevant candidates may include ego_motion_full/all_ego_motion.csv, step2_ego_motion_validation outputs, per-frame affine matrices, or original frame image folders.')
    print('Names are insufficient; matrix conventions, coordinates and frame indices must be inspected before inference.')
if __name__=='__main__':main()
