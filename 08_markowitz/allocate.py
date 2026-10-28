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

backtest.py tests this recipe on 27 past rebalances; this script applies it once, to real money,
on the last date in the data. The allocation is valid for the next 20 trading days only.

    python 08_markowitz/allocate.py --capital 1000000
    python 08_markowitz/allocate.py --capital 500000 --model 04_xgboost --top 10 --cap 0.2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.selection import daily_rank_ic  # noqa: E402

LOOKBACK = 252  # days of history for sigma, the covariance and the market average; also the IC window
p = argparse.ArgumentParser()
p.add_argument("--capital", type=float, required=True, help="amount to invest (same currency as prices, INR)")
p.add_argument("--model", default="03_random_forest", help="model folder whose predictions to use")
p.add_argument("--top", type=int, default=10)
p.add_argument("--cap", type=float, default=0.25, help="max weight per stock")
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
top = today.pred.reindex(univ).nlargest(args.top).index  # the model's highest-ranked stocks
with np.errstate(all="ignore"):  # spurious BLAS warnings from the unused precision matrix on macOS
    cov = LedoitWolf().fit(W[top].values).covariance_ * C.HORIZON  # shrunk covariance, in 20-day units
m = mu[top].values
# Maximise Sharpe = w'mu / sqrt(w'Cov w) (minimise its negative); if no pick has a positive
# expected return, minimise variance instead
obj = (lambda w: w @ cov @ w) if (m <= 0).all() else (lambda w: -(w @ m) / np.sqrt(w @ cov @ w))
res = minimize(obj, np.full(len(top), 1 / len(top)), method="SLSQP", bounds=[(0, args.cap)] * len(top),
               constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])  # long-only, fully invested
w = pd.Series(np.clip(res.x, 0, None), index=top)
w /= w.sum()  # remove tiny solver negatives and make the weights sum to exactly 1

# --------------------------------------------------------------------------- money -> shares
price = close.loc[T, top]
shares = np.floor(args.capital * w / price).astype(int)  # whole shares only; the remainder stays as cash
out = pd.DataFrame({"weight": w, "amount": args.capital * w, "price": price, "shares": shares,
                    "invested": shares * price, "exp. 20d return": mu[top]}).sort_values("weight", ascending=False)
out = out[out.weight > 1e-4]  # drop stocks Markowitz gave (practically) zero weight
port_mu, port_vol = float(w @ mu[top]), float(np.sqrt(w @ cov @ w))  # portfolio expected return and risk

print(f"Model {args.model}, date {T.date()}, IC used {ic_hat:.4f}, capital {args.capital:,.0f}\n")
print(out.to_string(formatters={"weight": "{:.1%}".format, "amount": "{:,.0f}".format, "price": "{:,.2f}".format,
                                "invested": "{:,.0f}".format, "exp. 20d return": "{:.2%}".format}))
print(f"\ninvested {out.invested.sum():,.0f} | cash left {args.capital - out.invested.sum():,.0f}")
print(f"portfolio: expected 20d return {port_mu:.2%}, 20d volatility {port_vol:.2%}, "
      f"expected 20d P&L {args.capital * port_mu:,.0f}")
out.to_csv(Path(__file__).resolve().parent / "results" / "allocation.csv")
