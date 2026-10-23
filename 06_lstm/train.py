"""Train the LSTM on 60-day windows. GPU recommended.

    .venv/bin/python 06_lstm/train.py                     # auto device, 12 configs, 3 seeds, quarterly walk-forward
    .venv/bin/python 06_lstm/train.py --device cuda:0 --n-iter 48 --seeds 5
    .venv/bin/python 06_lstm/train.py --quick             # smoke test (2 epochs)

Loss options (tuned):
  mse  mean squared error on y_win
  ic   minus the per-date correlation between score and y_rank (optimises ranking directly)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.seq_train import base_args, run  # noqa: E402

GRID = {
    "hidden": [32, 64, 128],
    "layers": [1, 2],
    "dropout": [0.1, 0.3],
    "lr": [1e-3, 3e-4],
    "loss": ["mse", "ic"],
}

if __name__ == "__main__":
    run(HERE, "LSTM", GRID, base_args(__doc__, cell="lstm").parse_args())
