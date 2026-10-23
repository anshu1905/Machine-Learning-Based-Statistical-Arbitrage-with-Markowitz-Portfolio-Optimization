"""Train XGBoost (GPU if available).

    .venv/bin/python 04_xgboost/train.py                  # auto-detect GPU, 24 random configs
    .venv/bin/python 04_xgboost/train.py --device cuda    # force GPU
    .venv/bin/python 04_xgboost/train.py --n-iter 96      # full grid
    .venv/bin/python 04_xgboost/train.py --quick          # smoke test

Two objectives are tuned side by side:
  reg   regression on y_win (squared error)
  rank  learning-to-rank (rank:pairwise) within each date on per-date decile
        labels of y_win: optimises the ordering, which is what the strategy uses.
Early stopping on validation during tuning; walk-forward refits reuse the
best number of trees.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xgboost as xgb  # noqa: E402

from common import config as C  # noqa: E402
from common.evaluate import finish  # noqa: E402
from common.training import (TabularTask, base_args, collect_predictions, expand_grid,  # noqa: E402
                             tune, walk_forward)

GRID = {
    "objective": ["reg", "rank"],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.01, 0.03],
    "min_child_weight": [100, 500],
    "colsample_bytree": [0.5, 0.8],
    "reg_lambda": [1.0, 10.0],
}
FIXED = dict(subsample=0.7, tree_method="hist", random_state=C.SEED)
MAX_TREES, EARLY_STOP = 3000, 100
DEVICE, N_JOBS = "cpu", -1


def pick_device(arg):
    if arg != "auto":
        return arg
    has_cuda_build = bool(xgb.build_info().get("USE_CUDA"))
    return "cuda" if has_cuda_build and shutil.which("nvidia-smi") else "cpu"


def decile_labels(y, dates):
    r = pd.Series(y.to_numpy()).groupby(dates).rank(pct=True).to_numpy()
    return np.minimum((r * 10).astype(int), 9)


def qid(dates):
    return pd.factorize(dates)[0]          # frames are Date-sorted, so qid is non-decreasing


def fit(params, train, val=None):
    p = {k: v for k, v in params.items() if k not in ("objective", "n_estimators")}
    common = dict(n_estimators=params.get("n_estimators", MAX_TREES), device=DEVICE, n_jobs=N_JOBS, **FIXED, **p)
    if val is not None:
        common["early_stopping_rounds"] = EARLY_STOP

    if params["objective"] == "reg":
        m = xgb.XGBRegressor(objective="reg:squarederror", **common)
        m.fit(train.X, train.y, eval_set=[(val.X, val.y)] if val is not None else None, verbose=False)
    else:
        m = xgb.XGBRanker(objective="rank:pairwise", eval_metric="ndcg@20", **common)
        kw = {}
        if val is not None:
            kw = dict(eval_set=[(val.X, decile_labels(val.y, val.dates))], eval_qid=[qid(val.dates)])
        m.fit(train.X, decile_labels(train.y, train.dates), qid=qid(train.dates), verbose=False, **kw)

    info = {"n_estimators": int(m.best_iteration) + 1} if val is not None else {}
    return m.predict, m, info


def main():
    global DEVICE, N_JOBS
    ap = base_args(__doc__, n_iter=24)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    DEVICE, N_JOBS = pick_device(args.device), args.n_jobs
    print(f"XGBoost {xgb.__version__} on {DEVICE}")

    grid = GRID
    if args.quick:
        grid = {"objective": ["reg", "rank"], "max_depth": [4], "learning_rate": [0.1], "min_child_weight": [200],
                "colsample_bytree": [0.8], "reg_lambda": [1.0]}
        args.n_iter = None
    task = TabularTask(HERE, "xgb_dataset", thin=args.thin)

    tuning, best = tune(task, expand_grid(grid, args.n_iter), fit)
    final = {**best["params"], "n_estimators": best["info"]["n_estimators"]}
    test_pred, model, wf_log = walk_forward(task, final, fit, args)

    preds = collect_predictions(task, best, test_pred, args.mode)
    gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    finish(HERE, "XGBoost", preds, tuning, final,
           extra={"mode": args.mode, "device": DEVICE, "retrain_every": args.retrain_every,
                  "window_years": args.window_years, "thin": args.thin,
                  "gain_importance_last_refit": (gain / gain.sum()).sort_values(ascending=False).round(4).to_dict(),
                  "walk_forward": wf_log})
    model.save_model(HERE / "results" / "model_final.json")


if __name__ == "__main__":
    main()
