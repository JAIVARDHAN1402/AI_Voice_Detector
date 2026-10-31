#!/usr/bin/env bash
# Reproduce the conformal study end to end from the cached WavLM features (data/features/*.npz).
#   stage 1: nested leave-one-domain-out scores (45 fits per head; ~1-2 h on 16 cores)
#   stage 2: calibration, evaluation, figures, splice study -> reports/conformal_<head>.json, reports/figures/
#   export : website data (web/research/) and the per-tier limits used by VoiceLock
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
for head in mlp linear; do
  [ -f data/conformal/lodo_scores_$head.pkl ] || PYTHONUTF8=1 $PY -u scripts/conformal_lodo.py --head $head --jobs 4
  PYTHONUTF8=1 $PY scripts/conformal_study.py --head $head
done
PYTHONUTF8=1 $PY scripts/export_research_web.py
PYTHONUTF8=1 $PY -m pytest tests -q
