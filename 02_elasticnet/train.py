"""Train ElasticNet.

    .venv/bin/python 02_elasticnet/train.py            # tune + quarterly walk-forward over test
    .venv/bin/python 02_elasticnet/train.py --quick    # smoke test

CPU only; ~2-5 minutes. The target is divided by its train std before fitting
so the same alpha grid means the same thing for y_win and y_rank.
"""
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sklearn.linear_model import ElasticNet  # noqa: E402

from common.evaluate import finish  # noqa: E402
from common.training import (TabularTask, base_args, collect_predictions, expand_grid,  # noqa: E402
                             tune, walk_forward)

GRID = {
    "alpha": np.round(np.logspace(-4, -1, 7), 6).tolist(),  # overall penalty (on standardised y)
    "l1_ratio": [0.1, 0.5, 0.9],                             # 0 = ridge-like, 1 = lasso
    "target": ["y_win", "y_rank"],
}


def fit(params, train, val=None):
    sd = float(train.y.std())
    m = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=10_000)
    m.fit(train.X, train.y / sd)
    return (lambda X: m.predict(X) * sd), m, {"n_nonzero": int((m.coef_ != 0).sum())}


def main():
    args = base_args(__doc__).parse_args()
    grid = {"alpha": [1e-3, 1e-2], "l1_ratio": [0.5], "target": ["y_win"]} if args.quick else GRID
    task = TabularTask(HERE, "elasticnet_dataset", thin=args.thin)

    tuning, best = tune(task, expand_grid(grid, args.n_iter), fit)
    test_pred, model, wf_log = walk_forward(task, best["params"], fit, args)

    preds = collect_predictions(task, best, test_pred, args.mode)
    coef = {f: c for f, c in zip(task.features, np.round(model.coef_, 6).tolist()) if c != 0}
    finish(HERE, "ElasticNet", preds, tuning, best["params"],
           extra={"mode": args.mode, "retrain_every": args.retrain_every, "window_years": args.window_years,
                  "nonzero_coefficients_last_refit": coef, "walk_forward": wf_log})
    joblib.dump(model, HERE / "results" / "model_final.joblib")


if __name__ == "__main__":
    main()
