"""Loading, split assignment and leakage-free cross-sectional transforms."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C


def load_clean(columns=None) -> pd.DataFrame:
    """The audited panel written by 00_data_audit/run_audit.py."""
    if not C.PANEL_CLEAN.exists():
        raise FileNotFoundError("Run 00_data_audit/run_audit.py first.")
    return pd.read_parquet(C.PANEL_CLEAN, columns=columns)


def assign_splits(dates: pd.Series, horizon: int = C.HORIZON) -> pd.Series:
    """burn_in / train / purge / val / purge / test, by trading-day calendar."""
    cal = np.sort(dates.unique())
    val0 = np.searchsorted(cal, np.datetime64(C.VAL_START))
    test0 = np.searchsorted(cal, np.datetime64(C.TEST_START))
    purge = set(cal[val0 - horizon:val0]) | set(cal[test0 - horizon:test0])

    s = pd.Series("test", index=dates.index, dtype=object)
    s[dates <= C.VAL_END] = "val"
    s[dates <= C.TRAIN_END] = "train"
    s[dates <= C.BURN_IN_END] = "burn_in"
    s[dates.isin(purge)] = "purge"
    return s


# ---------------------------------------------------------------------------
# Cross-sectional (per-date) transforms. They use only same-day information, so
# they need no fitting and are valid unchanged inside a walk-forward backtest.
# ---------------------------------------------------------------------------
def _per_date(df, cols, stat):
    return df.groupby("Date")[cols].transform(stat)


def cs_winsorize(df, cols, lo=0.01, hi=0.99):
    g = df.groupby("Date")[cols]
    ql = g.quantile(lo).reindex(df["Date"]).to_numpy()
    qh = g.quantile(hi).reindex(df["Date"]).to_numpy()
    return pd.DataFrame(np.clip(df[cols].to_numpy(), ql, qh), index=df.index, columns=cols)


def cs_zscore(df, cols):
    x = df[cols]
    mu, sd = _per_date(df, cols, "mean"), _per_date(df, cols, "std")
    return (x - mu) / sd.replace(0, np.nan)


def cs_winsor_zscore(df, cols, lo=0.01, hi=0.99):
    w = cs_winsorize(df, cols, lo, hi)
    w["Date"] = df["Date"]
    return cs_zscore(w, cols)


def cs_rank(df, cols):
    """Percentile rank per date, centred to [-0.5, 0.5]."""
    return df.groupby("Date")[cols].rank(pct=True) - 0.5


def cs_gauss_rank(df, cols):
    """Rank -> standard normal quantile per date (robust to outliers, Gaussian-shaped)."""
    g = df.groupby("Date")[cols]
    r, n = g.rank(), g.transform("count")
    return pd.DataFrame(norm.ppf((r - 0.5) / n), index=df.index, columns=cols)


def ts_zscore_fit(df, cols, mask):
    """Time-series standardisation for date-constant (market) features.
    Statistics come from the rows in `mask` only (train)."""
    daily = df.loc[mask].drop_duplicates("Date")[cols]
    return daily.mean(), daily.std()
