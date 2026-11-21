# Python environment notes

Different historical experiments used different environments; do not assert a single frozen requirements.lock without checking run logs. Typical package families are numpy, pandas, scipy, PyYAML, Pillow, opencv-python, torch, torchvision, ultralytics, matplotlib and MOT metrics/TrackEval support.

The final exact dependency versions for every historical script have not all been recovered. Use per-experiment logs and preserve a new environment lock when rerunning.
