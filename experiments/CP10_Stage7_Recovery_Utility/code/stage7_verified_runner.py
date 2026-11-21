#!/usr/bin/env python3
"""Run Stage 7 with reliable diagnostics; never mistake preflight-only for completed inference.
Does not install dependencies or modify original dataset/checkpoints.
"""
import json, os, pathlib, shutil, subprocess, sys, traceback, zipfile, platform
from datetime import datetime, timezone

HOME=pathlib.Path.home()
PILOT=HOME/'cp10_stage7_tools'/'CP10_Stage7_Detector_Pilot'
OUT=HOME/'CP10_Stage7_Dev_Verified_Run'
LOG=HOME/'CP10_Stage7_Verified_Run.log'
DIAG=HOME/'CP10_Stage7_Run_Diagnostics.zip'
SUCCESS=HOME/'CP10_Stage7_Dev_RealInference_Review.zip'

OUT.mkdir(parents=True,exist_ok=True)
status={'run_utc':datetime.now(timezone.utc).isoformat(), 'python':sys.executable,
        'python_version':sys.version, 'platform':platform.platform(),
        'pilot_dir':str(PILOT),'out_dir':str(OUT),'steps':[],'result':'NOT_STARTED',
        'checkpoint_untouched':True,'testreal_used_for_tuning':False}

def run(title,cmd):
    with LOG.open('a',encoding='utf-8') as f:
        f.write('\n=== '+title+' ===\n$ '+' '.join(map(str,cmd))+'\n')
        f.flush()
        try:
            result=subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, check=False)
            code=result.returncode
        except Exception as e:
            code=127
            f.write(repr(e)+'\n')
    status['steps'].append({'step':title,'returncode':code})
    print(f'{title}: exit={code}',flush=True)
    return code==0

def package():
    (OUT/'Stage7_Recovery_Status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    files=[LOG,OUT/'Stage7_Recovery_Status.json']+sorted(OUT.glob('Stage7_*'))
    with zipfile.ZipFile(DIAG,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        seen=set()
        for f in files:
            if f.is_file() and f not in seen:
                z.write(f,arcname=f.name);seen.add(f)
    with zipfile.ZipFile(DIAG) as z: assert z.testzip() is None
    print(f'Diagnostic ZIP: {DIAG}',flush=True)

LOG.write_text('Stage 7 verified run. This log reports the exact Python interpreter and process failures.\n',encoding='utf-8')
try:
    print('Python interpreter:',sys.executable,flush=True)
    if not PILOT.is_dir():
        status['result']='PILOT_SCRIPT_DIR_MISSING'
    elif not all((PILOT/x).is_file() for x in ('stage7_preflight_wsl.py','stage7_run_dev_inference.py','stage7_score_dev.py','stage7_pack_results.py')):
        status['result']='PILOT_SCRIPTS_MISSING'
    elif not run('DEPENDENCIES', [sys.executable,'-c',
            'import sys; print("Python:",sys.executable); import torch; print("torch:",torch.__version__); import numpy; print("numpy:",numpy.__version__); import ultralytics; print("ultralytics:",ultralytics.__version__)']):
        status['result']='DEPENDENCY_IMPORT_FAILED'
        run('PIP_PACKAGE_DETAILS', [sys.executable,'-m','pip','show','torch','torchvision','ultralytics','numpy'])
    elif not run('TRAINREAL_PREFLIGHT',[sys.executable,str(PILOT/'stage7_preflight_wsl.py'),'--root',os.getenv('CP10_ROOT','/mnt/e/AIUB/Capstone Project/Journal Based Results'),'--out',str(OUT)]):
        status['result']='PREFLIGHT_COMMAND_FAILED'
    else:
        mpath=OUT/'Stage7_Preflight_Manifest.json'
        manifest=json.loads(mpath.read_text(encoding='utf-8')) if mpath.is_file() else {}
        if manifest.get('status')!='READY_TRAINREAL_DEV_ONLY':
            status['result']='PREFLIGHT_NOT_READY'
        elif not run('TRAINREAL_FROZEN_INFERENCE',[sys.executable,'-u',str(PILOT/'stage7_run_dev_inference.py'),'--out',str(OUT),'--device',os.getenv('CP10_STAGE7_DEVICE','cpu')]):
            status['result']='INFERENCE_FAILED'
        elif not run('TRAINREAL_IOU_SCORING',[sys.executable,str(PILOT/'stage7_score_dev.py'),'--out',str(OUT)]):
            status['result']='SCORING_FAILED'
        elif not run('PACKAGE_REAL_INFERENCE',[sys.executable,str(PILOT/'stage7_pack_results.py'),'--out',str(OUT),'--zip',str(SUCCESS)]):
            status['result']='RESULT_PACKAGING_FAILED'
        elif not all((OUT/x).is_file() for x in ('Stage7_Dev_Detections.csv','Stage7_Dev_Overall_IoU50.csv','Stage7_Dev_Selected_Configuration_PRETEST.json','Stage7_Dev_Run_Provenance.json')):
            status['result']='INFERENCE_EVIDENCE_MISSING'
        else:
            prov=json.loads((OUT/'Stage7_Dev_Run_Provenance.json').read_text(encoding='utf-8'))
            status['result']='COMPLETE_REAL_DEVELOPMENT_INFERENCE' if prov.get('pilot_complete') else 'INFERENCE_INCOMPLETE'
    if status['result']=='COMPLETE_REAL_DEVELOPMENT_INFERENCE':
        print('SUCCESS: real TrainReal detector comparisons completed.',flush=True)
        print('Upload:',SUCCESS,flush=True)
    else:
        print('STOPPED:',status['result'],flush=True)
except Exception:
    with LOG.open('a',encoding='utf-8') as f:f.write(traceback.format_exc())
    status['result']='RUNNER_EXCEPTION'
finally:
    package()
    win_download=pathlib.Path('/mnt/c/Users/HP/Downloads')
    if win_download.is_dir():
        for p in (DIAG,SUCCESS if status['result']=='COMPLETE_REAL_DEVELOPMENT_INFERENCE' else None):
            if p and p.is_file():
                try:
                    shutil.copy2(p,win_download/p.name)
                    print('Copied to Windows Downloads:',p.name,flush=True)
                except OSError as exc: print('Windows copy failed:',str(exc),flush=True)
    if status['result']!='COMPLETE_REAL_DEVELOPMENT_INFERENCE':
        print('Upload the diagnostic ZIP. Do not reuse the old Stage7_Dev_Review.zip.',flush=True)
        sys.exit(2)
