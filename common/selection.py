"""Feature-selection tools: information coefficients, redundancy clustering, VIF."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import config as C


def daily_rank_ic(df: pd.DataFrame, features, target=C.TARGET) -> pd.DataFrame:
    """Spearman IC per date for each feature (dates x features).
    Pairwise-complete: stocks with a missing feature are dropped for that feature only."""
    out = {}
    base = df[["Date", target]].copy()
    for f in features:
        sub = base.assign(x=df[f]).dropna()
        g = sub.groupby("Date")
        rx = g["x"].rank()
        ry = g[target].rank()
        t = pd.DataFrame({"Date": sub["Date"], "x": rx, "y": ry, "xx": rx * rx,
                          "yy": ry * ry, "xy": rx * ry})
        s = t.groupby("Date").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
                                  sxx=("xx", "sum"), syy=("yy", "sum"), sxy=("xy", "sum"))
        cov = s.sxy - s.sx * s.sy / s.n
        var_x = s.sxx - s.sx ** 2 / s.n
        var_y = s.syy - s.sy ** 2 / s.n
        ic = cov / np.sqrt(var_x * var_y)
        out[f] = ic.where(s.n >= 30)
    return pd.DataFrame(out)


def newey_west_tstat(x: pd.Series, lags: int = C.HORIZON) -> float:
    """t-stat of the mean with Newey-West (Bartlett) errors; daily ICs on a 20-day
    target overlap, so the naive t-stat is inflated roughly sqrt(20)x."""
    x = x.dropna().to_numpy()
    T = len(x)
    if T < lags + 2:
        return np.nan
    e = x - x.mean()
    lrv = e @ e / T
    for l in range(1, lags + 1):
        lrv += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / T
    return x.mean() / np.sqrt(lrv / T)


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    yearly = ic.groupby(ic.index.year).mean()
    s = pd.DataFrame({
        "ic_mean": ic.mean(),
        "ic_std": ic.std(),
        "icir_ann": ic.mean() / ic.std() * np.sqrt(252 / C.HORIZON),
        "ic_tstat_nw": {f: newey_west_tstat(ic[f]) for f in ic},
        "hit_rate": (np.sign(ic).eq(np.sign(ic.mean()))).mean(),
        "year_sign_stability": yearly.apply(lambda c: (np.sign(c) == np.sign(c.mean())).mean()),
    })
    s["abs_t"] = s.ic_tstat_nw.abs()
    return s.sort_values("abs_t", ascending=False)


def cs_rank_corr(df: pd.DataFrame, features, every: int = 5) -> pd.DataFrame:
    """Correlation of per-date ranks, pooled over a subsample of dates. This is the
    redundancy that matters for a cross-sectional model (e.g. mom_20d and
    rel_mom_20d differ only by a date constant, so this gives exactly 1)."""
    dates = np.sort(df["Date"].unique())[::every]
    sub = df[df["Date"].isin(dates)]
    r = sub.groupby("Date")[features].rank(pct=True)
    return r.corr()


def redundancy_clusters(corr: pd.DataFrame, threshold: float) -> pd.Series:
    """Average-linkage clusters such that |rho| within a cluster is ~>= threshold."""
    dist = 1 - corr.abs().fillna(0).to_numpy()
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    return pd.Series(fcluster(Z, t=1 - threshold, criterion="distance"), index=corr.index)


def prune_redundant(corr, score: pd.Series, threshold: float, candidates=None):
    """Keep the feature with the highest `score` in each redundancy cluster."""
    feats = list(candidates if candidates is not None else corr.index)
    cl = redundancy_clusters(corr.loc[feats, feats], threshold)
    keep = score.reindex(feats).groupby(cl).idxmax().tolist()
    table = pd.DataFrame({"cluster": cl, "score": score.reindex(feats)})
    table["kept"] = table.index.isin(keep)
    return [f for f in feats if f in keep], table.sort_values(["cluster", "score"],
                                                               ascending=[True, False])


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor via the inverse correlation matrix."""
    Rinv = np.linalg.pinv(X.corr().to_numpy())
    return pd.Series(np.diag(Rinv), index=X.columns).sort_values(ascending=False)


def quantile_spread(df, feature, target=C.TARGET, q=5):
    """Mean target by per-date quantile of a feature (averaged over dates)."""
    sub = df[["Date", feature, target]].dropna()
    sub = sub[sub.groupby("Date")[feature].transform("size") >= 3 * q]
    b = sub.groupby("Date")[feature].transform(
        lambda x: pd.qcut(x.rank(method="first"), q, labels=False))
    return sub.groupby([sub["Date"], b])[target].mean().groupby(level=1).mean()
