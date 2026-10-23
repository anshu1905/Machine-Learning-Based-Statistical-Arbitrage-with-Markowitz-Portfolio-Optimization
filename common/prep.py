"""Helpers shared by the per-model preparation scripts."""
import json

import numpy as np
import pandas as pd

from . import config as C
from .data import cs_gauss_rank, cs_winsorize
from .selection import daily_rank_ic, newey_west_tstat

META = ["Date", "Symbol", "split", "has_target", "tradable"]


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Target variants (the test plan lists target definition as a critical variable).

    y_raw    20d excess return as delivered (use for evaluation / backtest P&L)
    y_win    y_raw winsorized 1/99% per date (default training target for regressors)
    y_rank   per-date Gaussian rank of y_raw (scale-free, outlier-proof)
    y_class  1{y_raw > 0}
    """
    out = df.copy()
    out["y_raw"] = out[C.TARGET]
    out["y_win"] = cs_winsorize(out, [C.TARGET])[C.TARGET].where(out[C.TARGET].notna())
    out["y_rank"] = cs_gauss_rank(out, [C.TARGET])[C.TARGET]
    out["y_class"] = out[C.TARGET_CLASS]
    return out


TARGET_COLS = ["y_raw", "y_win", "y_rank", "y_class"]


def ic_series(pred: np.ndarray, frame: pd.DataFrame, target="y_raw") -> pd.Series:
    """Daily rank IC of a prediction vector."""
    tmp = pd.DataFrame({"Date": frame["Date"].to_numpy(), "p": pred,
                        target: frame[target].to_numpy()})
    return daily_rank_ic(tmp, ["p"], target)["p"]


def ic_of(pred: np.ndarray, frame: pd.DataFrame, target="y_raw") -> float:
    """Mean daily rank IC of a prediction vector."""
    return float(ic_series(pred, frame, target).mean())


def ic_with_t(pred, frame, target="y_raw"):
    """(mean daily IC, Newey-West t-stat) — overlapping 20d targets need NW errors."""
    s = ic_series(pred, frame, target)
    return float(s.mean()), float(newey_west_tstat(s))


def permutation_ic(predict, X: pd.DataFrame, frame: pd.DataFrame, repeats=3, seed=C.SEED):
    """Drop in validation rank IC when one feature is shuffled *within each date*
    (keeps the feature's distribution and the date structure intact)."""
    rng = np.random.default_rng(seed)
    base = ic_of(predict(X), frame)
    dates = frame["Date"].to_numpy()
    groups = pd.Series(np.arange(len(X))).groupby(dates).indices.values()
    rows = []
    for f in X.columns:
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            col = Xp[f].to_numpy().copy()
            for idx in groups:
                col[idx] = col[rng.permutation(idx)]
            Xp[f] = col
            drops.append(base - ic_of(predict(Xp), frame))
        rows.append({"feature": f, "ic_drop": np.mean(drops), "ic_drop_sd": np.std(drops)})
    return base, pd.DataFrame(rows).set_index("feature").sort_values("ic_drop", ascending=False)


def every_kth_date(frame: pd.DataFrame, k: int) -> pd.Series:
    """Boolean mask keeping every k-th trading date (thins overlapping targets)."""
    d = np.sort(frame["Date"].unique())[::k]
    return frame["Date"].isin(d)


def write_dataset(df, features, path, extra_cols=()):
    cols = META + list(extra_cols) + list(features) + TARGET_COLS
    out = df[cols].copy()
    out[list(features)] = out[list(features)].astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def write_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=str))


def split_table(df):
    return df.groupby("split").agg(rows=("Date", "size"), with_target=("has_target", "sum"),
                                   first=("Date", "min"), last=("Date", "max"))
