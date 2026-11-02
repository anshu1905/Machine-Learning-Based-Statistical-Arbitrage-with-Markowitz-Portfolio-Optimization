"""Train / forecast ARIMA per stock (CPU, parallel over stocks).

    .venv/bin/python 07_arima/train.py                    # tune on val (every 20th day), test every 5th day
    .venv/bin/python 07_arima/train.py --test-step 1      # daily test forecasts (slowest, most comparable)
    .venv/bin/python 07_arima/train.py --n-jobs 32
    .venv/bin/python 07_arima/train.py --quick            # smoke test (40 stocks)

At each rebalance date t and for each stock:
  1. take the last `window` daily log excess returns up to and including t,
  2. fit ARIMA(p, 0, q) with a constant (orders from data/orders.csv),
  3. forecast 20 steps and sum -> predicted 20d excess return from t.
ARIMA(0,0,0) has a closed form (20 x window mean), so it skips the optimiser.
Successive fits of a stock warm-start from the previous parameters.

Tuned on validation: order rule (bic / aic / fixed ARIMA(1,0,1)) x window
(250 / 500 / 1000 trading days).
"""
import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from statsmodels.tsa.arima.model import ARIMA  # noqa: E402

from common import config as C  # noqa: E402
from common.evaluate import finish  # noqa: E402
from common.prep import ic_with_t  # noqa: E402

DATA = HERE / "data"
ORDER_RULES = ["bic", "aic", "fixed101"]
WINDOWS = [250, 500, 1000]
MIN_OBS = 120


def stock_forecasts(e, idx, order, window):
    """e: full daily series (np array on the trading calendar); idx: rebalance positions."""
    warnings.simplefilter("ignore")
    p, q = order
    out, start = np.full(len(idx), np.nan), None
    for k, i in enumerate(idx):
        y = e[max(0, i - window + 1): i + 1]
        y = y[~np.isnan(y)]
        if len(y) < MIN_OBS:
            continue
        if p == 0 and q == 0:
            out[k] = C.HORIZON * y.mean()
            continue
        try:
            r = ARIMA(y * 100, order=(p, 0, q), trend="c").fit(start_params=start, method_kwargs={"maxiter": 50})
            out[k] = r.forecast(C.HORIZON).sum() / 100
            start = r.params
        except Exception:
            start = None
    return out


def order_of(orders, sym, rule):
    if rule == "fixed101":
        return (1, 1)
    return int(orders.loc[sym, f"{rule}_p"]), int(orders.loc[sym, f"{rule}_q"])


def forecast_panel(E, orders, idx, rule, window, n_jobs):
    cols = E.columns
    res = Parallel(n_jobs=n_jobs)(delayed(stock_forecasts)(E[s].to_numpy(), idx, order_of(orders, s, rule), window)
                                  for s in cols)
    return pd.DataFrame(np.column_stack(res), index=E.index[idx], columns=cols)


def to_long(F, Y, tradable, split, source):
    long = F.stack(future_stack=True).rename("pred").reset_index()
    long.columns = ["Date", "Symbol", "pred"]
    long["y_raw"] = Y.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(long[["Date", "Symbol"]])).to_numpy()
    long["tradable"] = tradable.stack(future_stack=True).reindex(
        pd.MultiIndex.from_frame(long[["Date", "Symbol"]])).to_numpy()
    long["split"] = long.Date.map(split)
    long["source"] = source
    return long.dropna(subset=["tradable"])          # stock not listed on that date


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune-step", type=int, default=20, help="rebalance every k days on validation")
    ap.add_argument("--test-step", type=int, default=5, help="rebalance every k days on test")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    E = pd.read_parquet(DATA / "log_excess_ret_wide.parquet")
    Y = pd.read_parquet(DATA / "y_raw_wide.parquet").reindex_like(E)
    tradable = pd.read_parquet(DATA / "tradable_wide.parquet").reindex_like(E)
    split = pd.read_parquet(DATA / "split_by_date.parquet").split.reindex(E.index)
    orders = pd.read_csv(DATA / "orders.csv", index_col=0)

    rules, windows = ORDER_RULES, WINDOWS
    if args.quick:
        E = E[E.columns[:40]]
        rules, windows, args.tune_step, args.test_step = ["bic", "fixed101"], [500], 60, 60

    pos = np.arange(len(E))
    val_idx = pos[split.eq("val").to_numpy()][::args.tune_step]
    test_idx = pos[split.eq("test").to_numpy()][::args.test_step]
    val_frame = lambda F: to_long(F, Y, tradable, split, "rolling_refit")  # noqa: E731

    # ---- 1. tune order rule x window on validation
    rows, best = [], None
    for rule in rules:
        for w in windows:
            t0 = time.time()
            F = forecast_panel(E, orders, val_idx, rule, w, args.n_jobs)
            lv = val_frame(F)
            ic, t = ic_with_t(lv.pred.to_numpy(), lv)
            rows.append({"order_rule": rule, "window": w, "val_ic": ic, "val_ic_t": t,
                         "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
            if best is None or (np.isfinite(ic) and ic > best["val_ic"]):
                best = {"order_rule": rule, "window": w, "val_ic": ic, "val_long": lv}
    tuning = pd.DataFrame(rows).sort_values("val_ic", ascending=False)

    # ---- 2. test forecasts with the best variant
    t0 = time.time()
    F_test = forecast_panel(E, orders, test_idx, best["order_rule"], best["window"], args.n_jobs)
    print(f"test forecasts: {len(test_idx)} dates x {E.shape[1]} stocks in {time.time() - t0:.0f}s")

    preds = pd.concat([best["val_long"].assign(source=f"rolling_refit_every_{args.tune_step}d"),
                       to_long(F_test, Y, tradable, split, f"rolling_refit_every_{args.test_step}d")],
                      ignore_index=True)
    params = {"order_rule": best["order_rule"], "window": best["window"]}
    # IC t-stats: forecasts every k days overlap for 20/k observations
    finish(HERE, "ARIMA", preds, tuning, params, nw_lags=math.ceil(C.HORIZON / args.test_step),
           extra={"tune_step": args.tune_step, "test_step": args.test_step,
                  "note": "val metrics use tune_step spacing; NW lags set for test_step"})
    (HERE / "results" / "best_config.json").write_text(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
