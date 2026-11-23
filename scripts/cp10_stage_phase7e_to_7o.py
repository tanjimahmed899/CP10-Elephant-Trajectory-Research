#!/usr/bin/env python3
"""Stage a PUBLIC-SAFE subset of CP10 WildLive Phase 7E–7O evidence.

Read-only preview is the default. --apply writes to the chosen Git checkout.
Never pushes or commits; user must inspect the diff and push explicitly.
Python standard library only.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter
import zipfile

RELEASES = {
    "phase7e": [
        "WildLive_DJI0207_Phase7E_Frame_Forecasting_Pilot.zip",
        "WildLive_Phase7E_Reproduced_Results.zip",
    ],
    "phase7f": [
        "WildLive_DJI0207_Phase7F_Compatibility_Kit.zip",
        "Phase7F_Frozen_Results.zip",
    ],
    "phase7g": ["WildLive_DJI0207_Phase7G_Robustness_Kit.zip"],
    "phase7h": [
        "WildLive_Phase7H_MultiRecording_Validation_Kit.zip",
        "WildLive_DJI0204_Phase7H_Intake_and_Visual_QA_Tools.zip",
        "Phase7H_DJI0204_Frozen_Evaluation.zip",
        "WildLive_Phase7H_CrossSequence_Results_Audit.zip",
    ],
    "phase7i": [
        "WildLive_Phase7I_Motion_Regime_Diagnostic_Kit.zip",
        "WildLive_Phase7I_Reproduced_Results.zip",
    ],
    "phase7j": [
        "WildLive_Phase7J_Adaptive_Selector_Development_Kit.zip",
        "WildLive_Phase7J_Reproduced_Results.zip",
    ],
    "phase7k": [
        "WildLive_Phase7K_Prospective_Validation_Kit.zip",
        "Phase7K_DJI0161_Locked_Evaluation.zip",
        "Phase7K_DJI0161_Independent_Verification_Package.zip",
    ],
    "phase7l": [
        "WildLive_Phase7L_Capture_Group_Expansion_Kit.zip",
        "Phase7L_DJI0204_video1_Exploratory_Results.zip",
        "Phase7L_DJI0204_video1_Independent_Numerical_Verification.zip",
        "Phase7L_SixCandidate_Screening_Review_and_NextSteps.zip",
    ],
    "phase7m": [
        "WildLive_Phase7M_Capture_Provenance_Audit_Kit.zip",
        "Phase7M_Authorless_Provenance_Protocol_Kit.zip",
    ],
    "phase7n": ["WildLive_Phase7N_Camera_Motion_Final_Comparative_Kit.zip"],
    "phase7o": ["WildLive_Phase7O_Targeted_Robustness_and_Manuscript_Kit.zip"],
}
OPTIONAL_CENSUS = [
    "download_status.csv", "new_recording_species_census.csv",
    "selected_candidate_species_samples.csv", "remaining_six_candidate_species_samples.csv",
]
BASE = Path("experiments/wildlive_phase7e_to_7o")
TEXT_EXTS = {".py", ".sh", ".md", ".txt", ".csv", ".json", ".yaml", ".yml"}
MAX_FILE = 8 * 1024 * 1024
MAX_UNCOMPRESSED_ZIP = 200 * 1024 * 1024
MAX_ALL = 120 * 1024 * 1024
SECRET_RULES = [
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{24,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA|OPENSSH|EC|DSA|ENCRYPTED )?PRIVATE KEY-----"),
]
BLOCK_NAME = (
    "boxid", "bbox_cent", "bbox-cent", "bounding_box", "annotation_cent",
    "original_bbox", "raw_rgb", "original_rgb", "gps_coordinates",
    "gps_metadata", "raw_frame", "video_frame", "frame_overlay", "contact_sheet",
    "visual_qa_frames", "priority_frames", "track_centres", "track_centers",
)
BLOCK_COORD_HEADER = (
    "x_center_normalized", "y_center_normalized", "bbox_x1", "bbox_y1",
    "bbox_x2", "bbox_y2", "predicted_b_x", "predicted_b_y",
    "latitude", "longitude", "gps_lat", "gps_lon",
    "last_obs_x", "last_obs_y", "raw_dx_px", "raw_dy_px",
    "residual_dx_px", "residual_dy_px",
)


def sha256_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def allowed(path, data):
    name = path.name.lower()
    tokens = str(path).lower()
    ext = path.suffix.lower()
    if any(x in tokens for x in BLOCK_NAME):
        return 'restricted-source-name'
    if ext in {'.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.mp4', '.mov', '.avi', '.pt', '.pth', '.ckpt', '.onnx', '.zip', '.npz', '.npy', '.pkl', '.pickle', '.db', '.sqlite', '.pdf', '.docx'}:
        return 'excluded-binary-or-unreviewed-rights'
    if ext == '.png':
        # Only plots/charts published inside package results/figures; never actual RGB frames
        if '/figures/' not in ('/' + tokens) or any(x in name for x in ('frame', 'overlay', 'contact', 'rgb', 'image_crop', 'thumbnail', 'screenshot')):
            return 'not-allowlisted-plot'
    elif ext not in TEXT_EXTS:
        return 'not-allowlisted-extension'
    if len(data) > MAX_FILE:
        return 'too-large'
    if any(r.search(data) for r in SECRET_RULES):
        return 'possible-secret'
    if ext == '.csv':
        head = data.splitlines()[0].decode('utf-8-sig', 'replace').lower() if data else ''
        if any(w in head for w in BLOCK_COORD_HEADER):
            return 'position-derived-source-columns'
    if ext == '.json' and name.startswith(('frame_', 'box_')):
        return 'original-dataset-json'
    return None


def clean_member(info):
    p = PurePosixPath(info.filename)
    if info.is_dir() or p.is_absolute() or '..' in p.parts or not p.parts:
        return None
    if any(part.startswith('.') for part in p.parts):
        return None
    if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
        return None
    return Path(*p.parts)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--downloads', type=Path, default=Path('/mnt/c/Users/HP/Downloads'))
    p.add_argument('--repo', type=Path, required=True, help='Existing Git checkout root')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', help='Default: no writes in repository')
    g.add_argument('--apply', action='store_true', help='Write reviewed allowable artifacts to repo')
    p.add_argument('--include-manuscript', action='store_true', help='Opt-in only; requires reviewing manuscript illustration rights')
    return p.parse_args()


def main():
    a = parse_args()
    downloads = a.downloads.expanduser().resolve()
    repo = a.repo.expanduser().resolve()
    if not downloads.is_dir():
        sys.exit(f'ERROR: Downloads folder not found: {downloads}')
    if not (repo / '.git').exists():
        sys.exit(f'ERROR: Not an existing Git checkout: {repo}. Clone first.')
    out_dest = repo / BASE
    if out_dest.exists() and a.apply:
        sys.exit(f'ERROR: {out_dest} already exists. Review existing content; no automatic overwrite.')
    records = []
    skipped = Counter()
    missing = []
    staged = {}
    source_rows = []
    total_size = 0
    for phase, filenames in RELEASES.items():
        for filename in filenames:
            source = downloads / filename
            if not source.is_file():
                missing.append((phase, filename))
                continue
            try:
                with zipfile.ZipFile(source) as z:
                    if z.testzip() is not None:
                        skipped['damaged-zip'] += 1
                        continue
                    infos = z.infolist()
                    if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED_ZIP:
                        skipped['archive-too-large-uncompressed'] += 1
                        continue
                    source_rows.append((phase, filename, sha256_file(source), source.stat().st_size))
                    for info in infos:
                        p = clean_member(info)
                        if p is None:
                            skipped['directory-or-unsafe-path'] += 1
                            continue
                        if info.file_size > MAX_FILE:
                            skipped['too-large'] += 1
                            continue
                        with z.open(info) as h:
                            data = h.read(MAX_FILE + 1)
                        if len(data) > MAX_FILE:
                            skipped['too-large'] += 1
                            continue
                        reason = allowed(p, data)
                        if reason:
                            skipped[reason] += 1
                            continue
                        target = Path(phase) / Path(source.stem) / p
                        if target in staged:
                            sys.exit(f'ERROR: Duplicate target path {target}')
                        staged[target] = data
                        total_size += len(data)
            except (OSError, zipfile.BadZipFile) as e:
                print(f'WARNING: Could not inspect {source}: {e}', file=sys.stderr)
                skipped['broken-archive'] += 1
    for name in OPTIONAL_CENSUS:
        source = downloads / name
        if not source.is_file():
            continue
        data = source.read_bytes()
        reason = allowed(Path(name), data)
        if reason:
            skipped[reason] += 1
            continue
        target = Path('phase7m') / 'additional_species_census' / name
        staged[target] = data
        total_size += len(data)
        source_rows.append(('phase7m_census', name, sha256_file(source), source.stat().st_size))
    if a.include_manuscript:
        doczip = downloads / 'WildLive_Phase7O_Targeted_Robustness_and_Manuscript_Kit.zip'
        if doczip.is_file():
            with zipfile.ZipFile(doczip) as z:
                for item in z.namelist():
                    if item.endswith('/Final_Demo_paper_Phase7O_integrated.docx'):
                        data = z.read(item)
                        if len(data) > 10 * 1024 * 1024:
                            sys.exit('Manuscript exceeds 10 MiB; stage manually after rights check.')
                        staged[Path('phase7o') / 'reviewed_manuscript' / 'Final_Demo_paper_Phase7O_integrated.docx'] = data
                        total_size += len(data)
                        break
    if total_size > MAX_ALL:
        sys.exit(f'ERROR: selected {total_size / 1024**2:.1f} MiB exceeds safe size budget.')
    phase_counts = Counter(p.parts[0] for p in staged)
    report = [
        '# CP10 WildLive Phase 7E–7O: Public-safe research evidence', '',
        'This directory contains a **curated and non-exhaustive** subset of the original research archives.',
        'Individual phase folders preserve the source ZIP name to prevent provenance ambiguity.',
        'Some data files are intentionally absent because the dataset is licensed or because original',
        'frames, animal location data, binaries, and prediction/trajectory reconstruction may have redistribution restrictions.',
        '', '## Evidence limits', '',
        '- 20/50-**frame** prediction tasks are not automatically 2/5-second tasks.',
        '- DJI_0204_video1 and DJI_0204_video2 show shared-scene evidence and are not independent scene groups.',
        '- Adaptive selector superiority and flight-level independence have **not** been demonstrated.',
        '- Camera correction checks on sampled RGB image pairs are not full camera-corrected forecasting.',
        '- Per-window overlapping windows must not be treated as independent evaluation replicates.',
        '', '## Source datasets', '',
        'WildLive: https://doi.org/10.5523/bris.ewnuoroebuae20vei0cu6zu69',
        'This repository does not rehost the original WildLive imagery or bounding-box annotations.',
        '', '## Included package hashes (SHA-256)', '',
        '| Phase | Archive | SHA-256 | Source bytes |', '|---|---|---|---:|'
    ]
    for phase, filename, digest, size in source_rows:
        report.append(f'| {phase} | `{filename}` | `{digest}` | {size} |')
    report += ['', '## Public-stage contents', '', '| Phase | Included files |', '|---|---:|']
    for phase in RELEASES:
        report.append(f'| {phase} | {phase_counts[phase]} |')
    report += ['', '## Exclusions', '', 'The inclusion filter excludes original images, videos, boxes/centres,',
               'checkpoints, archives and large or unsafe files. Exclusion counts by rule:', '',
               '| Reason | Count |', '|---|---:|']
    for reason, count in sorted(skipped.items()):
        report.append(f'| {reason} | {count} |')
    report += ['', '## Reproduction', '',
               'For full reproduction, download the dataset from its cited source and use the',
               'frozen original checkpoint hashes and stage-specific protocols.',
               'The public code/results archive alone does not contain the original dataset or checkpoints.', '']
    staged[Path('README.md')] = ('\n'.join(report) + '\n').encode()
    manifest = [dict(phase=ph, archive=fn, sha256=d, bytes=b) for ph,fn,d,b in source_rows]
    staged[Path('source_archive_sha256.json')] = (json.dumps(manifest, indent=2) + '\n').encode()
    print(f'Mode: {"APPLY" if a.apply else "DRY RUN"}; repository={repo}')
    print(f'Archive/census files found: {len(source_rows)} | missing: {len(missing)}')
    for ph, fn in missing:
        print(f'  MISSING (skipped) [{ph}] {fn}')
    print(f'Included: {len(staged)} files ({total_size / 1024**2:.2f} MiB); excluded counts: {dict(skipped)}')
    print('Per-phase included:', dict(phase_counts))
    if not a.apply:
        print('\nNo files were written. Review the missing list and rerun with --apply.')
        return
    out_dest.mkdir(parents=True, exist_ok=False)
    try:
        for rel, data in staged.items():
            dest = out_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    except Exception:
        shutil.rmtree(out_dest, ignore_errors=True)
        raise
    print(f'\nSUCCESS: staged {len(staged)} files in {out_dest}')
    print('NOT COMMITTED; NOT PUSHED. Review: git status --short && git diff --check')

if __name__ == '__main__':
    main()
