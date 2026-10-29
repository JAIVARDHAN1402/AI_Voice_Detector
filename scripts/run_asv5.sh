#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY -u scripts/extract_features.py asv5 --threads 12 --max-real 500 --max-fake 960 > data/extract_asv5.log 2>&1
echo "features done"
$PY -u scripts/extract_replay.py asv5 --threads 12 > data/extract_asv5_replay.log 2>&1
echo "replay done"
