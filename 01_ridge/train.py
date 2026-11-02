"""Train Ridge Regression.

    .venv/bin/python 01_ridge/train.py                 # tune + quarterly walk-forward over test
    .venv/bin/python 01_ridge/train.py --mode fixed    # tune + one refit on train+val
    .venv/bin/python 01_ridge/train.py --quick         # smoke test

CPU only; ~1 minute.
"""
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sklearn.linear_model import Ridge  # noqa: E402

from common.evaluate import finish  # noqa: E402
from common.training import (TabularTask, base_args, collect_predictions, expand_grid,  # noqa: E402
                             tune, walk_forward)

GRID = {
    "alpha": [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5],   # L2 penalty
    "target": ["y_win", "y_rank"],                     # winsorized return vs per-date Gaussian rank
}


def fit(params, train, val=None):
    m = Ridge(alpha=params["alpha"]).fit(train.X, train.y)
    return m.predict, m, {}


def main():
    args = base_args(__doc__).parse_args()
    grid = {"alpha": [1, 1e4], "target": ["y_win"]} if args.quick else GRID
    task = TabularTask(HERE, "ridge_dataset", thin=args.thin)

    tuning, best = tune(task, expand_grid(grid, args.n_iter), fit)
    test_pred, model, wf_log = walk_forward(task, best["params"], fit, args)

    preds = collect_predictions(task, best, test_pred, args.mode)
    finish(HERE, "Ridge", preds, tuning, best["params"],
           extra={"mode": args.mode, "retrain_every": args.retrain_every, "window_years": args.window_years,
                  "coefficients": dict(zip(task.features, np.round(model.coef_, 6).tolist())),
                  "walk_forward": wf_log})
    joblib.dump(model, HERE / "results" / "model_final.joblib")


if __name__ == "__main__":
    main()
