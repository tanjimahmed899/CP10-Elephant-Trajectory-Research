# CP10 Stage 6 Fixed Reference — Reproducible Analysis Package

This package takes the user-uploaded camera and tracking metadata plus EXP3 and Stage5 saved predictions. It performs geometric replay, static virtual-line retrospective decisions, stable annotated crossing event matching and a **non-deployable** future-depth sensitivity audit.

**No new model trained. No true geographic survey. No detector-driven measurements.**

- Read `STAGE6_FIXED_REFERENCE_RESULTS.md` first.
- To reproduce: `python -m pip install numpy pandas matplotlib`, then `python stage6_fixed_reference_study.py`, `python stage6_episode_sensitivity.py`, `python stage6_fixed_overlap_verify.py`, `python create_figures.py`.
- Inputs and their unmodified SHA-256 hashes are in `inputs/` and `INTEGRITY_AND_METHOD_MANIFEST.json`.
- All data are retrospective, including Stage5's same-recording-group candidates. Images are visualization, not deployment evidence.
- Treat **future depth** in the overlap-consistency and oracle-depth controls as **ground-truth-only diagnostic**, never as inputs to a putative forecast system.
