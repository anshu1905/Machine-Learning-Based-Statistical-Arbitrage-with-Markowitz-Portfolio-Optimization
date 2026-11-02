"""Copy of allocate.py fixed to one set-up: ₹10,00,000, Random Forest, 5 long + 5 short.

Same recipe as allocate.py (and backtest.py), applied once on the last date with predictions:
  mu_i  = mu_bar + IC x sigma_i x z_i   (Random Forest's ML expected 20-day return)
  Sigma = Ledoit-Wolf covariance of the last 252 daily returns
  max-Sharpe with longs summing to +100% and shorts to -100%, each |w| <= 25%

Logic:

Picks: Random Forest's top 5 stocks to buy and bottom 5 to short, on the latest date.
Weights: Markowitz max-Sharpe; longs between 0% and 25%, shorts between -25% and 0%.
Total: ₹10 lakh bought across the top 5 and ₹10 lakh sold short across the bottom 5 (dollar-neutral).
Lookback: fixed at 252 days.

With 5 stocks per side and a 25% cap, every stock must get at least 0% and the five must reach
100%, so each side holds at least 4 stocks.

Output: results/allocation_5long_5short.csv (shorts have negative weight and shares).

    python 08_markowitz/allocate_copy.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.selection import daily_rank_ic  # noqa: E402
from optimizer import max_sharpe  # noqa: E402  (08_markowitz/optimizer.py, shared with backtest.py)

LOOKBACK = 252  # days of history for sigma, the covariance and the market average; also the IC window
# Fixed set-up for this copy (allocate.py takes these as command-line options)
args = argparse.Namespace(capital=1_000_000,         # ₹10,00,000
                          model="03_random_forest",  # Random Forest predictions
                          top=5,                     # 5 long and 5 short
                          cap=0.25,                  # max |weight| per stock
                          side="longshort")

# --------------------------------------------------------------------------- data
panel = pd.read_parquet(C.PANEL_CLEAN, columns=["Date", "Symbol", "Close", "ret_1d", "r_b", "tradable", "big_jump"])
panel.loc[panel.big_jump, "ret_1d"] = 0.0  # |daily return| > 35% = unadjusted demerger, not a real move
R = panel.pivot(index="Date", columns="Symbol", values="ret_1d").sort_index()   # daily returns, dates x stocks
RB = panel.groupby("Date").r_b.first().reindex(R.index).fillna(0.0)             # benchmark daily return
close = panel.pivot(index="Date", columns="Symbol", values="Close").sort_index()  # prices, for share counts

# The model's predictions on the most recent date are today's forecasts
pr = pd.read_parquet(C.ROOT / args.model / "results/predictions.parquet")
T = pr.Date.max()
t = R.index.get_loc(T)
today = pr[(pr.Date == T) & pr.tradable].set_index("Symbol")

# --------------------------------------------------------------------------- expected returns
# IC: mean daily IC over the last 252 dates whose 20-day outcome is known by T.
# It measures how accurate the model has actually been and scales how far mu moves from the average.
ic = daily_rank_ic(pr.dropna(subset=["y_raw", "pred"]), ["pred"], target="y_raw")["pred"].dropna().sort_index()
ic_hat = max(float(ic[ic.index <= R.index[t - C.HORIZON - 1]].iloc[-LOOKBACK:].mean()), 0.0)

W = R.iloc[t - LOOKBACK + 1: t + 1]                                       # last 252 days up to T
univ = [s for s in today.index if W[s].notna().sum() >= 0.9 * LOOKBACK]   # enough history to estimate risk
W = W[univ].fillna(0.0)
mu_bar = (W.mean() * C.HORIZON).mean()                                    # market average 20-day return
sigma = W.sub(RB.loc[W.index], axis=0).std() * np.sqrt(C.HORIZON)         # 20-day vol of excess return
mu = mu_bar + ic_hat * sigma * today.pred_z.reindex(univ).fillna(0.0)     # Grinold-Kahn: mu_bar + IC*sigma*z

# --------------------------------------------------------------------------- Markowitz weights
ranked = today.pred.reindex(univ).dropna().sort_values(ascending=False)
top = ranked.index[:args.top]                                                        # the model's best: buy
bottom = ranked.index[-args.top:] if args.side == "longshort" else ranked.index[:0]  # its worst: short
names = top.append(bottom)
with np.errstate(all="ignore"):  # spurious BLAS warnings from the unused precision matrix on macOS
    cov = LedoitWolf().fit(W[names].values).covariance_ * C.HORIZON  # shrunk covariance, in 20-day units
# Same solver as the backtest: maximise Sharpe = w'mu / sqrt(w'Cov w) with longs summing to +1 and
# (long-short) shorts to -1, |w_i| <= cap; minimum variance if the picks offer no positive mu
w, fallback = max_sharpe(mu[names].values, cov, args.cap, len(top))
w = pd.Series(w, index=names)

# --------------------------------------------------------------------------- money -> shares
price = close.loc[T, names]
# Whole shares only, rounded towards zero: shorts get negative share counts; the remainder stays as cash
shares = np.trunc(args.capital * w / price).astype(int)
out = pd.DataFrame({"side": np.where(w > 0, "long", "short"), "action": np.where(w > 0, "BUY", "SHORT"),
                    "weight": w, "amount": args.capital * w, "price": price, "shares": shares,
                    "value": shares * price, "exp. 20d return": mu[names]})
out = out[out.weight.abs() > 1e-4]  # drop stocks Markowitz gave (practically) zero weight
out = out.sort_values(["side", "weight"], key=lambda c: c.abs() if c.name == "weight" else c,
                      ascending=[True, False])
port_mu, port_vol = float(w @ mu[names]), float(np.sqrt(w @ cov @ w))  # expected return and risk per unit capital
long_val, short_val = out.value[out.weight > 0].sum(), -out.value[out.weight < 0].sum()

print(f"Model {args.model}, date {T.date()}, {args.side}, IC used {ic_hat:.4f}, capital {args.capital:,.0f}"
      + (" (minimum-variance fallback)" if fallback else "") + "\n")
print(out.to_string(formatters={"weight": "{:+.1%}".format, "amount": "{:+,.0f}".format, "price": "{:,.2f}".format,
                                "value": "{:+,.0f}".format, "exp. 20d return": "{:.2%}".format}))
if args.side == "long":
    print(f"\ninvested {long_val:,.0f} | cash left {args.capital - long_val:,.0f}")
else:
    print(f"\nbought {long_val:,.0f} (long) | sold short {short_val:,.0f} | net exposure {long_val - short_val:+,.0f}"
          f" | gross {long_val + short_val:,.0f}")
print(f"portfolio: expected 20d return {port_mu:.2%}, 20d volatility {port_vol:.2%}, "
      f"expected 20d P&L {args.capital * port_mu:,.0f}")
out.to_csv(Path(__file__).resolve().parent / "results" / "allocation_5long_5short.csv")
