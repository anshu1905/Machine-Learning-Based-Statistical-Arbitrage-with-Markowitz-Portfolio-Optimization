"""ARIMA — EDA, order identification and model-ready series.

    .venv/bin/python 07_arima/prepare_data.py

ARIMA is univariate: it uses no engineered features, only each stock's own
past. "Feature selection" for ARIMA is therefore (1) choosing the series to
model, (2) the differencing order d (stationarity tests) and (3) the AR/MA
orders p, q (ACF/PACF, information criteria) — all on TRAIN data only.

Series modelled: daily log EXCESS return  e_t = ln(1+r_t) - ln(1+r_b,t),
so a 20-step forecast  sum_{h=1..20} E[e_{t+h}]  targets the same quantity as
the ML models (20d excess return).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf  # noqa: E402

from common import config as C  # noqa: E402
from common.data import load_clean  # noqa: E402
from common.plotting import BLUE, INK_2, ORANGE, SEQUENTIAL, plt, save  # noqa: E402
from common.prep import add_targets  # noqa: E402

FIG = HERE / "figures"
OUT = HERE / "data"
MAX_P = MAX_Q = 3
NLAGS = 20


def stock_diagnostics(sym, logp, e):
    warnings.simplefilter("ignore")
    lp, ex = logp.dropna(), e.dropna()
    out = {"Symbol": sym, "n_train": len(ex)}
    out["adf_p_logprice"] = adfuller(lp, autolag="AIC")[1]
    out["kpss_p_logprice"] = kpss(lp, regression="c", nlags="auto")[1]
    out["adf_p_excess"] = adfuller(ex, autolag="AIC")[1]
    out["kpss_p_excess"] = kpss(ex, regression="c", nlags="auto")[1]
    out["ljungbox_p_10"] = float(acorr_ljungbox(ex, lags=[10])["lb_pvalue"].iloc[0])
    out["ljungbox_p_20"] = float(acorr_ljungbox(ex, lags=[20])["lb_pvalue"].iloc[0])
    out["arch_lm_p"] = het_arch(ex - ex.mean(), nlags=10)[1]
    out["skew"], out["kurtosis"] = ex.skew(), ex.kurt()
    a = acf(ex, nlags=NLAGS, fft=True)[1:]
    pa = pacf(ex, nlags=NLAGS, method="ywm")[1:]
    return out, a, pa


def order_search(sym, e):
    warnings.simplefilter("ignore")
    y = e.dropna().to_numpy() * 100            # percent units: better-conditioned MLE
    rows = []
    for p in range(MAX_P + 1):
        for q in range(MAX_Q + 1):
            try:
                r = ARIMA(y, order=(p, 0, q), trend="c").fit()
                rows.append({"Symbol": sym, "p": p, "q": q, "aic": r.aic, "bic": r.bic})
            except Exception:
                rows.append({"Symbol": sym, "p": p, "q": q, "aic": np.nan, "bic": np.nan})
    return rows


def main():
    df = add_targets(load_clean(["Date", "Symbol", "split", "has_target", "tradable", "Close", "ret_1d", "r_b",
                                 C.TARGET, C.TARGET_CLASS]))
    df["log_price"] = np.log(df.Close)
    df["log_ret"] = np.log1p(df.ret_1d)
    df["log_excess_ret"] = df.log_ret - np.log1p(df.r_b)

    wide = {c: df.pivot(index="Date", columns="Symbol", values=c)
            for c in ["log_price", "log_ret", "log_excess_ret", "y_raw", "y_win"]}
    split_by_date = df.drop_duplicates("Date").set_index("Date").split
    tradable = df.pivot(index="Date", columns="Symbol", values="tradable")

    # check: sum of next-20 daily log excess returns vs the 20d simple excess target
    fwd_sum = wide["log_excess_ret"][::-1].rolling(C.HORIZON).sum()[::-1].shift(-1)
    both = pd.concat([fwd_sum.stack().rename("s"), wide["y_raw"].stack().rename("y")], axis=1).dropna()
    target_link = both.corr(method="spearman").iloc[0, 1]

    # ---- diagnostics on the TRAIN window
    tr_dates = split_by_date.index[split_by_date.eq("train")]
    res = Parallel(n_jobs=-1)(delayed(stock_diagnostics)(s, wide["log_price"].loc[tr_dates, s],
                                                         wide["log_excess_ret"].loc[tr_dates, s])
                              for s in wide["log_price"].columns)
    diag = pd.DataFrame([r[0] for r in res]).set_index("Symbol")
    A = np.vstack([r[1] for r in res])
    PA = np.vstack([r[2] for r in res])
    band = 1.96 / np.sqrt(diag.n_train.median())

    # ---- order identification (AIC / BIC grid, train)
    grid = Parallel(n_jobs=-1)(delayed(order_search)(s, wide["log_excess_ret"].loc[tr_dates, s])
                               for s in wide["log_excess_ret"].columns)
    grid = pd.DataFrame([r for g in grid for r in g])
    best_aic = grid.loc[grid.groupby("Symbol").aic.idxmin()].set_index("Symbol")[["p", "q", "aic"]]
    best_bic = grid.loc[grid.groupby("Symbol").bic.idxmin()].set_index("Symbol")[["p", "q", "bic"]]
    orders = best_aic.add_prefix("aic_").join(best_bic.add_prefix("bic_"))
    orders.columns = ["aic_p", "aic_q", "aic", "bic_p", "bic_q", "bic"]
    orders["recommended_p"], orders["recommended_q"] = orders.bic_p, orders.bic_q   # BIC: parsimonious

    # ---- export
    OUT.mkdir(parents=True, exist_ok=True)
    for name, w in wide.items():
        w.to_parquet(OUT / f"{name}_wide.parquet")
    tradable.to_parquet(OUT / "tradable_wide.parquet")
    split_by_date.to_frame("split").to_parquet(OUT / "split_by_date.parquet")
    orders.to_csv(OUT / "orders.csv")
    diag.to_csv(OUT / "diagnostics_train.csv")

    # ---- figures
    lags = np.arange(1, NLAGS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, M, title in [(axes[0], A, "ACF"), (axes[1], PA, "PACF")]:
        q25, med, q75 = np.percentile(M, [25, 50, 75], axis=0)
        ax.fill_between(lags, q25, q75, color=BLUE, alpha=0.25, lw=0, label="IQR across stocks")
        ax.plot(lags, med, color=BLUE, marker="o", ms=4, label="median")
        ax.axhspan(-band, band, color=INK_2, alpha=0.12, lw=0, label="±1.96/√T")
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set(title=f"{title} of daily log excess returns (train)", xlabel="lag (days)")
        ax.set_xticks(lags[::2])
    axes[0].legend(fontsize=8)
    save(fig, FIG / "01_acf_pacf.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, col, title in [(axes[0], "ljungbox_p_10", "Ljung–Box Q(10) p-value"),
                           (axes[1], "arch_lm_p", "ARCH-LM(10) p-value"),
                           (axes[2], "adf_p_excess", "ADF p-value (excess returns)")]:
        ax.hist(diag[col], bins=20, range=(0, 1), color=BLUE)
        ax.axvline(0.05, color=ORANGE, ls="--", lw=1)
        ax.set(title=title, xlabel="p-value", ylabel="stocks")
    save(fig, FIG / "02_test_pvalues.png")

    def heat(ax, pcol, qcol, title):
        H = pd.crosstab(orders[pcol], orders[qcol]).reindex(index=range(MAX_P + 1), columns=range(MAX_Q + 1),
                                                            fill_value=0)
        ax.imshow(H.values, cmap=SEQUENTIAL)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                ax.text(j, i, H.iat[i, j], ha="center", va="center", fontsize=9)
        ax.set_xticks(range(MAX_Q + 1))
        ax.set_yticks(range(MAX_P + 1))
        ax.set(xlabel="q", ylabel="p", title=title)
        ax.grid(False)
        return H
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    H_aic = heat(axes[0], "aic_p", "aic_q", "Best order by AIC (stocks)")
    H_bic = heat(axes[1], "bic_p", "bic_q", "Best order by BIC (stocks)")
    save(fig, FIG / "03_order_selection.png")

    ex = "RIL IS Equity" if "RIL IS Equity" in wide["log_price"] else wide["log_price"].columns[0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(wide["log_price"][ex], color=BLUE, lw=1.2)
    axes[0].set(title=f"{ex}: log price (non-stationary, d = 1)")
    axes[1].plot(wide["log_excess_ret"][ex], color=BLUE, lw=0.6)
    axes[1].set(title="daily log excess return (stationary, d = 0)")
    save(fig, FIG / "04_example_series.png")

    # ---- report
    share = lambda m: f"{m.mean():.0%}"  # noqa: E731
    stat = pd.DataFrame({
        "log price": [share(diag.adf_p_logprice < .05), share(diag.kpss_p_logprice < .05)],
        "log excess return": [share(diag.adf_p_excess < .05), share(diag.kpss_p_excess < .05)],
    }, index=["ADF rejects unit root (p<5%)", "KPSS rejects stationarity (p<5%)"])
    sig_acf = (np.abs(A) > band).mean(axis=0)
    top_orders = (orders.groupby(["bic_p", "bic_q"]).size().sort_values(ascending=False).head(6)
                  .rename("stocks").reset_index())
    md = [
        "# 07 — ARIMA: EDA & order identification",
        "",
        "Prerequisite: `00_data_audit/REPORT.md` (shared splits).",
        "",
        "## Why ARIMA is prepared differently",
        "ARIMA is univariate and linear in its own past, so the 50 engineered features do not apply. Its \"feature "
        "selection\" is the choice of series and of (p, d, q):",
        "",
        "- **Series**: daily log excess return `e_t = ln(1+r_t) − ln(1+r_bench,t)`. The sum of the next 20 values "
        f"matches the ML target (Spearman correlation with `target_excess_ret_20d` = **{target_link:.3f}**), so ARIMA "
        "forecasts can be ranked and fed to Markowitz exactly like the ML forecasts.",
        "- Everything below uses the **train window (2019–2022)** per stock.",
        "",
        "## d — stationarity",
        "",
        stat.to_markdown(),
        "",
        "Log prices have a unit root, while excess returns are stationary → model returns with **d = 0** "
        "(equivalently, ARIMA(p,1,q) on log excess price).",
        "",
        "![series](figures/04_example_series.png)",
        "",
        "## p, q — autocorrelation structure",
        "",
        "![acf](figures/01_acf_pacf.png)",
        "",
        f"- Share of stocks whose ACF is outside ±1.96/√T at lags 1–5: "
        + ", ".join(f"lag {i + 1}: {sig_acf[i]:.0%}" for i in range(5)) + ".",
        f"- Ljung–Box Q(10) rejects 'no autocorrelation' for **{share(diag.ljungbox_p_10 < .05)}** of stocks "
        f"(Q(20): {share(diag.ljungbox_p_20 < .05)}).",
        f"- ARCH-LM finds volatility clustering in **{share(diag.arch_lm_p < .05)}** of stocks → ARIMA's constant-variance "
        "assumption is violated. Use robust standard errors, and consider ARIMA-GARCH as an extension.",
        f"- Median excess kurtosis of daily excess returns: {diag['kurtosis'].median():.1f} (fat tails).",
        "",
        "![tests](figures/02_test_pvalues.png)",
        "",
        f"### Information-criterion grid p, q ∈ 0…{MAX_P} (constant included), per stock",
        "",
        "![orders](figures/03_order_selection.png)",
        "",
        "Most common BIC orders:",
        "",
        top_orders.to_markdown(index=False),
        "",
        "BIC is the recommended criterion (consistent, parsimonious; AIC over-fits with ~1,000 noisy observations). "
        "Per-stock orders are in `data/orders.csv` (`recommended_p`, `recommended_q`).",
        "",
        f"**Key finding:** BIC chooses ARIMA(0,0,0) — white noise plus a constant — for "
        f"**{((orders.bic_p == 0) & (orders.bic_q == 0)).mean():.0%}** of stocks (AIC: "
        f"{((orders.aic_p == 0) & (orders.aic_q == 0)).mean():.0%}). For those stocks the 20-day ARIMA forecast is just "
        "20 × the window's mean excess return, i.e. a trailing-drift (momentum) signal. ARIMA is still the right "
        "classical benchmark, but when modelling, also compare (a) the BIC orders, (b) the AIC orders and (c) a fixed "
        "ARIMA(1,0,1) for every stock, so the benchmark is not reduced to a moving average by construction.",
        "",
        "## Model-ready data (`data/`)",
        "",
        "- `log_excess_ret_wide.parquet` — the series to model, [date × stock].",
        "- `log_ret_wide.parquet`, `log_price_wide.parquet` — alternatives (raw returns; prices for d = 1).",
        "- `y_raw_wide.parquet` / `y_win_wide.parquet` — the 20d excess-return target from each date, for scoring forecasts.",
        "- `tradable_wide.parquet`, `split_by_date.parquet` — universe flags and the shared splits.",
        "- `orders.csv` — AIC/BIC-best (p, q) per stock; `diagnostics_train.csv` — all test statistics per stock.",
        "",
        "Walk-forward use: at each rebalance date t, fit ARIMA(p,0,q) with the stock's recommended order on the last N "
        "days of `log_excess_ret` up to t (e.g. N = 500), forecast 20 steps, sum → predicted 20d excess return.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(md))
    print(stat)
    print(top_orders)
    print("ljung-box sig", (diag.ljungbox_p_10 < .05).mean(), "arch", (diag.arch_lm_p < .05).mean(),
          "target link", target_link)


if __name__ == "__main__":
    main()
