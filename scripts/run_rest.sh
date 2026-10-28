#!/bin/bash
# Queue: prepare ASVspoof test slice, then embed the remaining datasets once the gary extraction ends.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
until [ "$(grep -c '^  row group' data/prep_asv_train.log)" -ge 20 ]; do sleep 5; done
echo "asv_train prep done"
$PY scripts/prepare_data.py asv_test --bonafide 400 --spoof 1300 > data/prep_asv_test.log 2>&1
echo "asv_test prep done"
until grep -q "^\[gary\] saved" data/extract_gary.log; do sleep 10; done
echo "gary extraction done"
$PY scripts/extract_features.py asv_train --threads 12 --max-real 600 --max-fake 800 > data/extract_asv_train.log 2>&1
echo "asv_train extraction done"
$PY scripts/extract_features.py asv_test --threads 12 --max-real 300 --max-fake 450 > data/extract_asv_test.log 2>&1
echo "asv_test extraction done"
