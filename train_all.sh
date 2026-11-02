#!/usr/bin/env bash
# Train all seven models, then compare them. Logs go to logs/<model>.log.
# Usage: ./train_all.sh            (full run)
#        ./train_all.sh --quick    (smoke test, ~5 minutes)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs
for m in 01_ridge 02_elasticnet 03_random_forest 04_xgboost 05_rnn 06_lstm 07_arima; do
  echo "== $m  $(date '+%H:%M:%S')"
  $PY "$m/train.py" "$@" 2>&1 | tee "logs/$m.log" | grep -E "val IC .* test IC|results ->" || true
done
$PY compare_models.py
