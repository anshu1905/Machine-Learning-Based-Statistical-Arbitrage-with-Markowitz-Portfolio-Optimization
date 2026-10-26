#!/usr/bin/env bash
# All four portfolio variants (long-only / long-short x 252-day / 20-day lookback),
# the 20-start-day robustness check for each, then every figure. Uses the models'
# existing predictions: no retraining needed. Takes a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
for side in long longshort; do
  for lb in 252 20; do
    echo "== $side, lookback $lb"
    $PY 08_markowitz/backtest.py --side $side --lookback $lb > /dev/null
    $PY 08_markowitz/robustness.py --side $side --lookback $lb > /dev/null
  done
done
$PY 08_markowitz/plot_backtest.py
