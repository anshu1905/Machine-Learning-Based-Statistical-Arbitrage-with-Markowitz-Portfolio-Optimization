"""Training harness shared by all models.

Protocol (same for every model):
  1. TUNE   fit each hyper-parameter config on TRAIN (2019 - Dec 2022),
            score by mean daily rank IC on VALIDATION (2023 - May 2024).
  2. TEST   with the best config, produce out-of-sample predictions for the
            TEST period (Jul 2024 onward):
              --mode wf     walk-forward: refit every --retrain-every trading days
                            on all data whose 20d target is already known
                            (expanding window, or rolling with --window-years)
              --mode fixed  one refit on train + val, predict all of test
  3. SAVE   results/predictions.parquet, metrics.json, tuning.csv, model file.

A refit at date T only uses rows dated <= T - 21 trading days, so every
training target (t -> t+20) has ended before T: no look-ahead.
"""
import argparse
import itertools
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .prep import ic_with_t

EVAL_COLS = ["Date", "Symbol", "split", "tradable", "y_raw"]


# --------------------------------------------------------------------------- CLI
def base_args(description, thin=1, retrain_every=63, n_iter=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mode", choices=["wf", "fixed"], default="wf",
                   help="wf = walk-forward refits over test (default); fixed = single refit on train+val")
    p.add_argument("--retrain-every", type=int, default=retrain_every,
                   help="trading days between walk-forward refits (21 = monthly, 63 = quarterly)")
    p.add_argument("--window-years", type=float, default=None,
                   help="rolling training window in years; default: expanding from 2019")
    p.add_argument("--thin", type=int, default=thin,
                   help="train on every k-th date (20d targets overlap 95%% day to day)")
    p.add_argument("--n-iter", type=int, default=n_iter,
                   help="evaluate a random subset of the grid of this size")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--quick", action="store_true",
                   help="smoke test: tiny grid and only two walk-forward blocks")
    return p


def expand_grid(grid, n_iter=None, seed=C.SEED):
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    if n_iter and n_iter < len(combos):
        pick = np.random.default_rng(seed).choice(len(combos), n_iter, replace=False)
        combos = [combos[i] for i in sorted(pick)]
    return combos


# ------------------------------------------------------------- walk-forward core
def wf_blocks(t_idx_test, retrain_every, mode, max_blocks=None):
    """[(i0, i1)] calendar-index ranges; refit at i0, predict dates i0..i1-1."""
    start, end = int(t_idx_test.min()), int(t_idx_test.max())
    step = end - start + 1 if mode == "fixed" else retrain_every
    blocks = [(i, min(i + step, end + 1)) for i in range(start, end + 1, step)]
    return blocks[:max_blocks] if max_blocks else blocks


def train_mask_for(t_idx, i0, window_years=None):
    """Rows whose 20d target has fully realised before date index i0."""
    cut = i0 - C.HORIZON - 1
    m = t_idx <= cut
    if window_years:
        m &= t_idx > cut - int(round(window_years * 252))
    return m


def safe_score(x):
    return x if np.isfinite(x) else -np.inf


# ------------------------------------------------------------------ tabular task
@dataclass
class Frame:
    X: pd.DataFrame
    y: pd.Series
    dates: np.ndarray
    rows: pd.DataFrame


class TabularTask:
    """Loads <folder>/data/<dataset>.parquet + selected_features.json.
    Rows are sorted Date-major (required by ranking objectives)."""

    def __init__(self, folder, dataset, thin=1):
        self.folder = folder
        self.features = json.loads((folder / "selected_features.json").read_text())["features"]
        df = pd.read_parquet(folder / "data" / f"{dataset}.parquet")
        df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
        self.calendar = np.sort(df.Date.unique())
        df["t_idx"] = np.searchsorted(self.calendar, df.Date.to_numpy())
        self.df = df
        self.thinned = (df.t_idx % thin == 0) if thin > 1 else pd.Series(True, index=df.index)
        self.train = df.split.eq("train") & df.has_target
        self.val = df.split.eq("val") & df.has_target
        self.test = df.split.eq("test")                         # incl. live rows without target
        self.eligible = df.split.ne("burn_in") & df.has_target  # usable for walk-forward refits

    def frame(self, mask, target):
        r = self.df[mask]
        return Frame(r[self.features], r[target], r.Date.to_numpy(), r)


def tune(task, combos, fit_fn, default_target="y_win"):
    """fit_fn(params, train: Frame, val: Frame|None) -> (predict, model, info)."""
    rows, best = [], None
    for k, params in enumerate(combos, 1):
        target = params.get("target", default_target)
        tr = task.frame(task.train & task.thinned, target)
        va = task.frame(task.val, target)
        t0 = time.time()
        predict, model, info = fit_fn(params, tr, va)
        pred = predict(va.X)
        ic, t = (np.nan, np.nan) if np.nanstd(pred) == 0 else ic_with_t(pred, va.rows)
        rows.append({**params, **info, "val_ic": ic, "val_ic_t": t, "seconds": round(time.time() - t0, 1)})
        print(f"[{k}/{len(combos)}] {params} {info} -> val IC {ic:.4f} (t {t:.2f})", flush=True)
        if best is None or safe_score(ic) > safe_score(best["val_ic"]):
            best = {"params": params, "info": info, "val_ic": ic, "val_pred": pred, "model": model}
    return pd.DataFrame(rows).sort_values("val_ic", ascending=False), best


def walk_forward(task, params, fit_fn, args, default_target="y_win"):
    df = task.df
    target = params.get("target", default_target)
    blocks = wf_blocks(df.loc[task.test, "t_idx"], args.retrain_every, args.mode,
                       max_blocks=2 if args.quick else None)
    pred = pd.Series(np.nan, index=df.index)
    log, model = [], None
    for k, (i0, i1) in enumerate(blocks, 1):
        m = task.eligible & task.thinned & train_mask_for(df.t_idx, i0, args.window_years)
        pm = task.test & (df.t_idx >= i0) & (df.t_idx < i1)
        t0 = time.time()
        predict, model, _ = fit_fn(params, task.frame(m, target), None)
        pred[pm] = predict(df.loc[pm, task.features])
        tr_dates = df.loc[m, "Date"]
        log.append({"refit_date": str(task.calendar[i0])[:10], "predict_to": str(task.calendar[i1 - 1])[:10],
                    "train_from": str(tr_dates.min())[:10], "train_to": str(tr_dates.max())[:10],
                    "train_rows": int(m.sum()), "seconds": round(time.time() - t0, 1)})
        print(f"walk-forward {k}/{len(blocks)}: {log[-1]}", flush=True)
    return pred, model, log


def collect_predictions(task, best, test_pred, mode):
    val = task.df.loc[task.val, EVAL_COLS].assign(pred=best["val_pred"], source="fit_on_train")
    test = task.df.loc[task.test, EVAL_COLS].assign(pred=test_pred[task.test].to_numpy(), source=mode)
    if test.pred.isna().all():
        test = test.iloc[0:0]
    return pd.concat([val, test], ignore_index=True)
