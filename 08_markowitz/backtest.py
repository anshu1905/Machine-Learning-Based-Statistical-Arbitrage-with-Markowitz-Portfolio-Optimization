"""Top-10 + Markowitz backtest: ML expected returns vs historical-mean expected returns.

Every 20 trading days over the test period (Jul 2024 -> Aug 2026), for each source of
expected returns mu:

  historical  mu_i = mean daily return over the last 252 days x 20   (classic Markowitz input)
  <ML model>  mu_i = mu_bar + IC x sigma_i x z_i                      (Grinold-Kahn alpha)
              mu_bar  = universe average of the historical mu (same market drift for both)
              sigma_i = 20d volatility of the stock's excess return (last 252 days)
              z_i     = the model's per-date Gaussian rank score (pred_z)
              IC      = the model's mean daily IC over the last 252 prediction dates whose
                        20d target had fully realised by the rebalance date (>= 0)

  1. Select the 10 stocks with the highest forecast (historical: mu, ML: the model's score);
     --side longshort also shorts the 10 lowest.
  2. Weight them  a) equally   b) Markowitz max-Sharpe with |w_i| <= cap and a Ledoit-Wolf
     covariance of the last --lookback daily returns (same for every source).
       long       w_i >= 0, sum w = 1                                  (fully invested)
       longshort  longs >= 0 summing to +1, shorts <= 0 summing to -1  (dollar-neutral)
  3. Hold 20 trading days (weights drift), pay cost_bps on one-way turnover, rebalance.

--lookback (default 252) is the window for the historical mean, the volatilities and the
covariance. The model IC used in mu always comes from the last 252 prediction dates.

No look-ahead: every input at rebalance date T uses data up to T's close; holding
returns start on T+1.

    python 08_markowitz/backtest.py                       # long-only, 252-day lookback
    python 08_markowitz/backtest.py --side longshort --lookback 20
    08_markowitz/run_all.sh                               # all four variants + robustness + figures
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, ttest_rel
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.selection import daily_rank_ic, newey_west_tstat  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = {"01_ridge": "Ridge", "02_elasticnet": "ElasticNet", "03_random_forest": "Random Forest",
          "04_xgboost": "XGBoost", "05_rnn": "RNN", "06_lstm": "LSTM", "07_arima": "ARIMA"}
IC_WINDOW = 252  # prediction dates used to estimate each model's IC

p = argparse.ArgumentParser()
p.add_argument("--top", type=int, default=10)
p.add_argument("--cap", type=float, default=0.25, help="max weight per stock in Markowitz")
p.add_argument("--cost-bps", type=float, default=10.0, help="cost per unit of one-way turnover")
p.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate for the Sharpe ratio")
p.add_argument("--side", choices=["long", "longshort"], default="long")
p.add_argument("--lookback", type=int, default=252, help="days for historical mean, volatility, covariance")
p.add_argument("--offset", type=int, default=0, help="start the 20-day cycle this many days into the test")
p.add_argument("--out", help="output folder (default results/<side>_lb<lookback>)")
args = p.parse_args()
LOOKBACK = args.lookback
OUT = Path(args.out or HERE / "results" / f"{args.side}_lb{LOOKBACK}")

# --------------------------------------------------------------------------- data
panel = pd.read_parquet(C.PANEL_CLEAN, columns=["Date", "Symbol", "ret_1d", "r_b", "tradable", "big_jump", C.TARGET])
# |daily return| > 35% are unadjusted corporate actions (demergers), not real P&L
panel.loc[panel.big_jump, "ret_1d"] = 0.0
R = panel.pivot(index="Date", columns="Symbol", values="ret_1d").sort_index()
TRAD = panel.pivot(index="Date", columns="Symbol", values="tradable").reindex_like(R).eq(True)
RB = panel.groupby("Date").r_b.first().reindex(R.index).fillna(0.0)
dates = R.index
pos = {d: i for i, d in enumerate(dates)}

test_idx = np.flatnonzero(dates >= pd.Timestamp(C.TEST_START))
rebal = [i for i in test_idx[args.offset::C.HORIZON] if i < len(dates) - 1]  # every 20th test day, needs a day to hold

preds = {}
for mdir, name in MODELS.items():
    f = C.ROOT / mdir / "results/predictions.parquet"
    if f.exists():
        d = pd.read_parquet(f, columns=["Date", "Symbol", "y_raw", "pred", "pred_z"])
        ic = daily_rank_ic(d.dropna(subset=["y_raw", "pred"]), ["pred"], target="y_raw")["pred"].dropna()
        preds[name] = (d.set_index(["Date", "Symbol"]).pred_z, ic.sort_index())
# Daily test IC of every expected-return input, on the same target and dates the models were scored on
panel = panel.sort_values(["Symbol", "Date"])
panel["mu_hist"] = panel.groupby("Symbol").ret_1d.transform(
    lambda s: s.rolling(LOOKBACK, min_periods=int(0.9 * LOOKBACK)).mean())
t_rows = panel[panel.Date >= C.TEST_START].dropna(subset=["mu_hist", C.TARGET])
daily_ic = {"Historical": daily_rank_ic(t_rows, ["mu_hist"], target=C.TARGET)["mu_hist"].dropna()}
daily_ic.update({n: ic[ic.index >= C.TEST_START] for n, (_, ic) in preds.items()})
print(f"models: {list(preds)} | rebalances: {len(rebal)} | {dates[rebal[0]].date()} -> {dates[-1].date()}")


def max_sharpe(mu, cov, cap, n_long):
    """Max-Sharpe weights. The first n_long assets are longs (sum +1); any others are shorts (sum -1)."""
    n = len(mu)
    sign = np.r_[np.ones(n_long), -np.ones(n - n_long)]
    w0 = sign / np.where(sign > 0, n_long, max(n - n_long, 1))
    if w0 @ mu <= 0:  # no positive expected return to exploit: fall back to minimum variance
        obj = lambda w: w @ cov @ w  # noqa: E731
    else:
        obj = lambda w: -(w @ mu) / np.sqrt(w @ cov @ w)  # noqa: E731
    bounds = [(0, cap) if s > 0 else (-cap, 0) for s in sign]
    cons = [{"type": "eq", "fun": lambda w: w[:n_long].sum() - 1}]
    if n > n_long:
        cons.append({"type": "eq", "fun": lambda w: w[n_long:].sum() + 1})
    w = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons).x
    w = np.where(sign > 0, np.clip(w, 0, None), np.clip(w, None, 0))
    w[:n_long] /= w[:n_long].sum()
    if n > n_long:
        w[n_long:] /= -w[n_long:].sum()
    return w, bool(w0 @ mu <= 0)


def score_ml(name, T, sigma, mu_bar, univ):
    """Model score z and expected return mu for the universe at date T."""
    pz, ic = preds[name]
    avail = pz.index.get_level_values(0).unique()
    d = avail[avail <= T].max()  # ARIMA predicts every 5th day; others every day
    z = pz.loc[d].reindex(univ)
    realised = ic[ic.index <= dates[pos[T] - C.HORIZON - 1]].iloc[-IC_WINDOW:]  # targets known by T
    ic_hat = max(float(realised.mean()), 0.0) if len(realised) else 0.0
    return z, mu_bar + ic_hat * sigma * z.fillna(0.0), ic_hat


# --------------------------------------------------------------------------- backtest
sources = ["Historical"] + list(preds)
strategies = [(s, wt) for s in sources for wt in ("EW", "MV")]
daily = {k: [] for k in strategies}
legs = {k: [] for k in strategies}  # daily return contribution of the long and the short leg
held = {k: pd.Series(dtype=float) for k in strategies}  # drifted weights from last period
log, picks, fc = [], [], []

for k, t in enumerate(rebal):
    T = dates[t]
    end = rebal[k + 1] if k + 1 < len(rebal) else len(dates) - 1
    win = R.iloc[t - LOOKBACK + 1: t + 1]
    univ = [s for s in R.columns if TRAD.at[T, s] and win[s].notna().sum() >= 0.9 * LOOKBACK]
    W = win[univ].fillna(0.0)
    mu_hist = W.mean() * C.HORIZON
    mu_bar = mu_hist.mean()
    sigma = W.sub(RB.loc[W.index], axis=0).std() * np.sqrt(C.HORIZON)
    fwd = (1 + R.iloc[t + 1: end + 1][univ].fillna(0.0)).prod() - 1  # realised holding return

    for src in sources:
        if src == "Historical":
            score, mu, ic_hat = mu_hist, mu_hist, np.nan
        else:
            score, mu, ic_hat = score_ml(src, T, sigma, mu_bar, univ)
        # forecast quality of the return input itself over the whole universe
        ok = score.notna() & fwd.notna()
        fc.append({"Date": T, "source": src, "rank_ic": spearmanr(score[ok], fwd[ok])[0], "ic_used": ic_hat})
        ranked = score.dropna().sort_values(ascending=False)
        top = ranked.index[:args.top]
        bottom = ranked.index[-args.top:] if args.side == "longshort" else ranked.index[:0]
        names = top.append(bottom)
        with np.errstate(all="ignore"):  # spurious BLAS warnings from the unused precision matrix on macOS
            cov = LedoitWolf().fit(W[names].values).covariance_ * C.HORIZON
        w_mv, fallback = max_sharpe(mu[names].values, cov, args.cap, len(top))
        w_ew = np.r_[np.full(len(top), 1 / len(top)), np.full(len(bottom), -1 / max(len(bottom), 1))]
        for wt, w in (("EW", w_ew), ("MV", w_mv)):
            key = (src, wt)
            w = pd.Series(w, index=names)
            turnover = w.sub(held[key], fill_value=0.0).abs().sum()
            path = (1 + R.iloc[t + 1: end + 1][names].fillna(0.0)).cumprod()
            gain = (path - 1) * w                       # P&L of each position per unit of capital
            value = 1 + gain.sum(axis=1)
            prev = value.shift(1).fillna(1.0)
            r = value.diff().fillna(value.iloc[0] - 1) / prev
            r.iloc[0] -= turnover * args.cost_bps / 1e4
            daily[key].append(r)
            lg_long = gain[top].sum(axis=1)
            lg_short = gain[bottom].sum(axis=1)
            legs[key].append(pd.DataFrame({"long": lg_long.diff().fillna(lg_long.iloc[0]) / prev,
                                           "short": lg_short.diff().fillna(lg_short.iloc[0]) / prev}))
            held[key] = (path.iloc[-1] * w) / value.iloc[-1]
            log.append({"Date": T, "source": src, "weighting": wt, "period_return": value.iloc[-1] - 1,
                        "expected_return": float(w @ mu[names]), "turnover": turnover,
                        "max_weight": w.abs().max(), "eff_n": 1 / (w ** 2).sum(),
                        "min_var_fallback": fallback and wt == "MV"})
            picks += [{"Date": T, "source": src, "weighting": wt, "side": "long" if s in top else "short",
                       "Symbol": s, "weight": w[s], "mu_20d": mu[s], "score": score[s]} for s in names]

bench = RB.iloc[rebal[0] + 1:]
log, picks, fc = pd.DataFrame(log), pd.DataFrame(picks), pd.DataFrame(fc)


# --------------------------------------------------------------------------- metrics
def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    ex = r - args.rf / 252
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "ann. vol": r.std() * np.sqrt(252),
            "Sharpe": ex.mean() / r.std() * np.sqrt(252), "max drawdown": (eq / eq.cummax() - 1).min(),
            "total return": eq.iloc[-1] - 1}


rets = {f"{s} {wt}": pd.concat(daily[(s, wt)]) for s, wt in strategies}
rets["Benchmark"] = bench
rows = []
hist_period = {wt: log[(log.source == "Historical") & (log.weighting == wt)].period_return.values for wt in ("EW", "MV")}
for s, wt in strategies:
    lg = log[(log.source == s) & (log.weighting == wt)]
    row = {"source": s, "weighting": wt, **stats(rets[f"{s} {wt}"]),
           "avg turnover": lg.turnover.iloc[1:].mean(), "daily test IC": daily_ic[s].mean(),
           "IC t": newey_west_tstat(daily_ic[s], C.HORIZON)}
    if s != "Historical":
        diff = lg.period_return.values - hist_period[wt]
        row["beats Hist. (periods)"] = (diff > 0).mean()
        row["t vs Hist."] = ttest_rel(lg.period_return.values, hist_period[wt]).statistic
    rows.append(row)
rows.append({"source": "Benchmark", "weighting": "-", **stats(bench)})
summary = pd.DataFrame(rows)

OUT.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT / "summary.csv", index=False)
log.to_csv(OUT / "periods.csv", index=False)
picks.to_csv(OUT / "holdings.csv", index=False)
fc.to_csv(OUT / "forecast_ic.csv", index=False)
pd.DataFrame(rets).to_parquet(OUT / "daily_returns.parquet")
pd.concat({f"{s} {wt}": pd.concat(legs[(s, wt)]) for s, wt in strategies}, axis=1).to_parquet(OUT / "leg_returns.parquet")

pct = ["CAGR", "ann. vol", "max drawdown", "total return", "avg turnover", "beats Hist. (periods)"]
show = summary.copy()
for c in pct:
    show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
for c in ["Sharpe", "daily test IC", "IC t", "t vs Hist."]:
    show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}" if c == "daily test IC" else f"{x:.2f}")
side = "long-only" if args.side == "long" else f"long top-{args.top} / short bottom-{args.top}, dollar-neutral"
md = [f"# Top-{args.top} + Markowitz backtest ({side}, {LOOKBACK}-day lookback): ML vs historical expected returns", "",
      f"Test period {dates[rebal[0]].date()} -> {dates[-1].date()}, {len(rebal)} rebalances every {C.HORIZON} "
      f"trading days, cost {args.cost_bps:g} bps per unit of one-way turnover, Markowitz cap {args.cap:.0%}, "
      f"Sharpe with rf = {args.rf:.1%}. EW = equal weight, MV = Markowitz max-Sharpe. "
      "`daily test IC` = mean daily Spearman correlation between the expected-return input (historical "
      f"{LOOKBACK}-day mean, or the model's prediction) and the realised 20-day excess return, over every test day; "
      "`IC t` = its Newey-West t-stat. `t vs Hist.` = paired t-stat of the "
      "per-period return difference against the historical-mean portfolio with the same weighting.", "",
      show.to_markdown(index=False), "",
      f"Markowitz fell back to minimum variance (no positive mu among the picks) in "
      f"{int(log.min_var_fallback.sum())} of {int((log.weighting == 'MV').sum())} portfolio-periods."]
(OUT / "summary.md").write_text("\n".join(md))
print(show.to_string(index=False))

print(f"results -> {OUT}")
