#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
for step in speaker asr forensics attacks; do
  .venv/Scripts/python.exe -u scripts/voicelock_eval.py $step > data/vl_eval_$step.log 2>&1
  echo "$step done (exit $?)"
done
