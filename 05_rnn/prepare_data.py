"""RNN (vanilla / Elman) — EDA, feature selection and model-ready sequence data.

    .venv/bin/python 05_rnn/prepare_data.py

A vanilla RNN suffers from vanishing gradients beyond a few tens of steps, so
it gets a SHORT lookback (20 days = one target horizon) and a compact input set.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.sequence import run  # noqa: E402

run(
    HERE, "05 — RNN",
    lookback=20,
    n_core=8,
    corr_threshold=0.80,
    max_fast=4,
    market=["r_b", "bench_vol_20d", "bench_mom_60d"],
    probe_lags=[5, 10, 20],
    rationale=[
        "## Why the RNN is prepared this way",
        "- Recurrent nets need **bounded, comparable inputs** → per-date Gaussian rank for stock features "
        "(leakage-free, no fitting) and train-fitted z-scores for the market columns.",
        "- A vanilla RNN forgets quickly (vanishing gradients) → **lookback 20 days** and a **small input set** "
        "(8 core + up to 4 fast + 3 market) to keep the hidden state's job simple.",
        "- The input is a window, not a snapshot → the EDA below asks which features change over the window "
        "(persistence), how far back information reaches (lagged IC), and whether history adds anything (probe).",
    ],
)
