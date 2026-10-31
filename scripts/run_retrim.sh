#!/bin/bash
# Re-extract the three silence-confounded corpora with trimming, then retrain. Old features are kept in data/features_pre_trim/.
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe
$PY -u scripts/voicelock_eval.py attacks > data/vl_eval_attacks.log 2>&1
echo "attacks done (exit $?)"
mkdir -p data/features_pre_trim
mv data/features/asv_train* data/features/asv_test* data/features/unidata* data/features_pre_trim/
$PY -u scripts/extract_features.py asv_train --threads 12 --max-real 600 --max-fake 800 > data/extract_asv_train.log 2>&1
$PY -u scripts/extract_features.py asv_test --threads 12 --max-real 300 --max-fake 450 > data/extract_asv_test.log 2>&1
$PY -u scripts/extract_features.py unidata --threads 12 > data/extract_unidata.log 2>&1
echo "features done"
$PY -u scripts/extract_replay.py asv_train asv_test unidata --threads 12 > data/extract_retrim_replay.log 2>&1
echo "replay done"
cp models/detector.joblib models/detector_pre_trim.joblib; cp reports/evaluation.json reports/evaluation_pre_trim.json; cp web/model/classifier.json data/classifier_pre_trim.json
$PY -u scripts/train.py final --layers 2 3 5 7 > data/train_trim.log 2>&1
echo "train done (exit $?)"
