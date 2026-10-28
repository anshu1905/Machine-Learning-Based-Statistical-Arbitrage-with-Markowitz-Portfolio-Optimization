"""Split a given amount of money across the model's top stocks with Markowitz, on the latest date.

Same recipe as backtest.py, applied once on the last date with predictions:
  mu_i  = mu_bar + IC x sigma_i x z_i   (model's ML expected 20-day return)
  Sigma = Ledoit-Wolf covariance of the last 252 daily returns
  max-Sharpe, long-only, w_i <= cap, sum w = 1
then converts weights into whole shares at the latest close.


Logic:

Picks: the model’s top 10 stocks on the latest date (nlargest).
Weights: Markowitz max-Sharpe, each between 0% and 25% (bounds (0, cap)), so no negative weights and no shorting.
Total: the weights sum to 100% of your capital, e.g. ₹10 lakh all spread across the top 10.
Lookback: fixed at 252 days.

--side longshort (same as the backtest's long-short variant): also short the model's bottom 10.
The capital is then used twice over, dollar-neutral: buy capital's worth of the top 10 and short
capital's worth of the bottom 10 (longs sum to +100%, shorts to -100%, each |w| <= cap).
In India a short held for 20 days needs stock futures or securities lending.

backtest.py tests this recipe on 27 past rebalances; this script applies it once, to real money,
on the last date in the data. The allocation is valid for the next 20 trading days only.
Output: results/allocation_<side>.csv (one row per stock; shorts have negative weight and shares).

    python 08_markowitz/allocate.py --capital 1000000
    python 08_markowitz/allocate.py --capital 1000000 --side longshort
    python 08_markowitz/allocate.py --capital 500000 --model 04_xgboost --top 10 --cap 0.2
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
p = argparse.ArgumentParser()
p.add_argument("--capital", type=float, required=True, help="amount to invest (same currency as prices, INR)")
p.add_argument("--model", default="03_random_forest", help="model folder whose predictions to use")
p.add_argument("--top", type=int, default=10)
p.add_argument("--cap", type=float, default=0.25, help="max weight per stock")
p.add_argument("--side", choices=["long", "longshort"], default="long",
               help="long = buy the top stocks; longshort = also short the bottom stocks (dollar-neutral)")
args = p.parse_args()

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
out.to_csv(Path(__file__).resolve().parent / "results" / f"allocation_{args.side}.csv")
