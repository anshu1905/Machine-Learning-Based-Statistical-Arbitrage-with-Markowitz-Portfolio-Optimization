"""LSTM — EDA, feature selection and model-ready sequence data.

    .venv/bin/python 06_lstm/prepare_data.py

Gated memory lets an LSTM use longer windows than a vanilla RNN, so it gets a
60-day lookback (one quarter: covers the 20/60-day features' own horizons) and
a richer input set.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.sequence import run  # noqa: E402

run(
    HERE, "06 — LSTM",
    lookback=60,
    n_core=14,
    corr_threshold=0.80,
    max_fast=5,
    market=["r_b", "bench_ret_20d", "bench_vol_20d", "bench_mom_60d", "bench_drawdown_252d"],
    probe_lags=[5, 10, 20, 40, 60],
    rationale=[
        "## Why the LSTM is prepared this way",
        "- Same leakage-free input scaling as the RNN (per-date Gaussian rank; train-fitted z-scores for market columns).",
        "- Gates (input/forget/output) keep gradients alive over long windows → **lookback 60 days** and a **larger "
        "input set** (14 core + up to 5 fast + 5 market regime channels).",
        "- More capacity means more overfitting risk on a weak signal: compare against the RNN and the linear probe below, "
        "and use dropout / early stopping on `val` when training.",
    ],
)
