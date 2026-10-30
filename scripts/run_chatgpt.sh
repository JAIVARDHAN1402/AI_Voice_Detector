#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY -u scripts/extract_features.py chatgpt --threads 12 > data/extract_chatgpt.log 2>&1
echo "features done"
$PY -u scripts/extract_replay.py chatgpt --threads 12 > data/extract_chatgpt_replay.log 2>&1
echo "replay done"
