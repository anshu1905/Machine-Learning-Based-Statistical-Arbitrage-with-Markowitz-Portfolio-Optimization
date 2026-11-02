"""Train Random Forest.

    .venv/bin/python 03_random_forest/train.py               # tune + quarterly walk-forward
    .venv/bin/python 03_random_forest/train.py --n-jobs 32   # use 32 CPU cores
    .venv/bin/python 03_random_forest/train.py --quick       # smoke test

CPU only (scikit-learn has no GPU forest); scales with cores. Trains on every
5th date by default: consecutive 20d targets overlap 95%, so extra rows add
little information but make leaves over-confident.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from common import config as C  # noqa: E402
from common.evaluate import finish  # noqa: E402
from common.training import (TabularTask, base_args, collect_predictions, expand_grid,  # noqa: E402
                             tune, walk_forward)

GRID = {
    "max_depth": [6, 10, 14],
    "min_samples_leaf": [100, 500, 2000],   # large leaves = averaging over many stock-days
    "max_features": [0.33, 0.6],
}
N_TREES_TUNE, N_TREES_FINAL = 300, 500
N_JOBS = -1


def fit(params, train, val=None):
    m = RandomForestRegressor(n_estimators=params.get("n_estimators", N_TREES_TUNE),
                              max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
                              max_features=params["max_features"], n_jobs=N_JOBS, random_state=C.SEED)
    m.fit(train.X, train.y)
    return m.predict, m, {}


def main():
    global N_JOBS
    args = base_args(__doc__, thin=5).parse_args()
    N_JOBS = args.n_jobs
    grid = {"max_depth": [6], "min_samples_leaf": [500], "max_features": [0.33]} if args.quick else GRID
    task = TabularTask(HERE, "rf_dataset", thin=args.thin)

    tuning, best = tune(task, expand_grid(grid, args.n_iter), fit)
    final = {**best["params"], "n_estimators": 100 if args.quick else N_TREES_FINAL}
    test_pred, model, wf_log = walk_forward(task, final, fit, args)

    preds = collect_predictions(task, best, test_pred, args.mode)
    imp = pd.Series(model.feature_importances_, index=task.features).sort_values(ascending=False)
    finish(HERE, "Random Forest", preds, tuning, final,
           extra={"mode": args.mode, "retrain_every": args.retrain_every, "window_years": args.window_years,
                  "thin": args.thin, "impurity_importance_last_refit": imp.round(4).to_dict(),
                  "walk_forward": wf_log})
    joblib.dump(model, HERE / "results" / "model_final.joblib", compress=3)


if __name__ == "__main__":
    main()
