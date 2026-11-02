#!/usr/bin/env bash
# Rebuild every model-ready dataset from features_by_stock.xlsx (about 5 minutes).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY 00_data_audit/run_audit.py
for m in 01_ridge 02_elasticnet 03_random_forest 04_xgboost 05_rnn 06_lstm 07_arima; do
  echo "== $m"; $PY "$m/prepare_data.py"
done
