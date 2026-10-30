#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY -u scripts/extract_features.py chatgpt2 --threads 12 > data/extract_chatgpt2.log 2>&1
echo "features done"
$PY -u scripts/extract_replay.py chatgpt2 --threads 12 > data/extract_chatgpt2_replay.log 2>&1
echo "replay done"
