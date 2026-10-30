#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY -u scripts/extract_features.py mlaad --threads 12 > data/extract_mlaad.log 2>&1
echo "features done"
$PY -u scripts/extract_replay.py mlaad --threads 12 > data/extract_mlaad_replay.log 2>&1
echo "replay done"
