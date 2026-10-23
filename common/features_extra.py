"""Extra OHLCV features from statarb_features_ranked.pdf that are not in the xlsx.

All are causal: the value on date t uses bars up to and including t, the same
convention as the features already in features_by_stock.xlsx.
"""
import numpy as np
import pandas as pd


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _one_stock(g: pd.DataFrame) -> pd.DataFrame:
    O, H, L, C, V = g.Open, g.High, g.Low, g.Close, g.Volume
    C1 = C.shift(1)
    r = C / C1 - 1
    out = pd.DataFrame(index=g.index)

    # Amihud illiquidity: mean |r| / traded value, log so it is not 10 orders wide
    illiq = (r.abs() / (C * V).replace(0, np.nan)).rolling(20, min_periods=15).mean()
    out["amihud_log_20d"] = np.log(illiq.replace(0, np.nan))

    # Yang-Zhang volatility (n = 20)
    n = 20
    o = np.log(O / C1)
    c = np.log(C / O)
    rs = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz_var = (o.rolling(n, min_periods=15).var() + k * c.rolling(n, min_periods=15).var()
              + (1 - k) * rs.rolling(n, min_periods=15).mean())
    out["yz_vol_20d"] = np.sqrt(yz_var.clip(lower=0))

    # Variance ratio VR(5) over a 60-day window: < 1 => mean reversion
    lr = np.log(C / C1)
    lr5 = lr.rolling(5).sum()
    out["variance_ratio_5_60"] = lr5.rolling(60, min_periods=45).var() / (
        5 * lr.rolling(60, min_periods=45).var())

    out["ret_autocorr_60d"] = r.rolling(60, min_periods=45).corr(r.shift(1))
    out["overnight_ret"] = o
    out["vol_ret_corr_20d"] = r.abs().rolling(20, min_periods=15).corr(V)

    # RSI (Wilder, 14)
    d = C.diff()
    gain, loss = _wilder(d.clip(lower=0), 14), _wilder((-d).clip(lower=0), 14)
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ADX (14)
    up, dn = H.diff(), -L.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([H - L, (H - C1).abs(), (L - C1).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14).replace(0, np.nan)
    pdi, mdi = 100 * _wilder(pdm, 14) / atr, 100 * _wilder(mdm, 14) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out["adx_14"] = _wilder(dx, 14)

    # MFI (14)
    tp = (H + L + C) / 3
    mf = tp * V
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    out["mfi_14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))

    # Chaikin money flow (20)
    clv = ((C - L) - (H - C)) / (H - L).replace(0, np.nan)
    out["cmf_20"] = (clv.fillna(0) * V).rolling(20, min_periods=15).sum() / \
        V.rolling(20, min_periods=15).sum().replace(0, np.nan)
    return out


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must be sorted by Symbol, Date."""
    extra = pd.concat([_one_stock(g) for _, g in df.groupby("Symbol", sort=False)])
    extra = extra.replace([np.inf, -np.inf], np.nan)
    return df.join(extra)
