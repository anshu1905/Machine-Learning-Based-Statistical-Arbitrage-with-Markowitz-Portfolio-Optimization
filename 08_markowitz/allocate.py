"""Split a given amount of money across the model's top stocks with Markowitz, on the latest date.

Same recipe as backtest.py, applied once on the last date with predictions:
  mu_i  = mu_bar + IC x sigma_i x z_i   (model's ML expected 20-day return)
  Sigma = Ledoit-Wolf covariance of the last 252 daily returns
  max-Sharpe, long-only, w_i <= cap, sum w = 1
then converts weights into whole shares at the latest close.

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

LOOKBACK = 252
p = argparse.ArgumentParser()
p.add_argument("--capital", type=float, required=True, help="amount to invest (same currency as prices, INR)")
p.add_argument("--model", default="03_random_forest", help="model folder whose predictions to use")
p.add_argument("--top", type=int, default=10)
p.add_argument("--cap", type=float, default=0.25, help="max weight per stock")
args = p.parse_args()

panel = pd.read_parquet(C.PANEL_CLEAN, columns=["Date", "Symbol", "Close", "ret_1d", "r_b", "tradable", "big_jump"])
panel.loc[panel.big_jump, "ret_1d"] = 0.0
R = panel.pivot(index="Date", columns="Symbol", values="ret_1d").sort_index()
RB = panel.groupby("Date").r_b.first().reindex(R.index).fillna(0.0)
close = panel.pivot(index="Date", columns="Symbol", values="Close").sort_index()

pr = pd.read_parquet(C.ROOT / args.model / "results/predictions.parquet")
T = pr.Date.max()
t = R.index.get_loc(T)
today = pr[(pr.Date == T) & pr.tradable].set_index("Symbol")

# IC: mean daily IC over the last 252 dates whose 20-day outcome is known by T
ic = daily_rank_ic(pr.dropna(subset=["y_raw", "pred"]), ["pred"], target="y_raw")["pred"].dropna().sort_index()
ic_hat = max(float(ic[ic.index <= R.index[t - C.HORIZON - 1]].iloc[-LOOKBACK:].mean()), 0.0)

W = R.iloc[t - LOOKBACK + 1: t + 1]
univ = [s for s in today.index if W[s].notna().sum() >= 0.9 * LOOKBACK]
W = W[univ].fillna(0.0)
mu_bar = (W.mean() * C.HORIZON).mean()
sigma = W.sub(RB.loc[W.index], axis=0).std() * np.sqrt(C.HORIZON)
mu = mu_bar + ic_hat * sigma * today.pred_z.reindex(univ).fillna(0.0)

top = today.pred.reindex(univ).nlargest(args.top).index
with np.errstate(all="ignore"):  # spurious BLAS warnings from the unused precision matrix on macOS
    cov = LedoitWolf().fit(W[top].values).covariance_ * C.HORIZON
m = mu[top].values
obj = (lambda w: w @ cov @ w) if (m <= 0).all() else (lambda w: -(w @ m) / np.sqrt(w @ cov @ w))
res = minimize(obj, np.full(len(top), 1 / len(top)), method="SLSQP", bounds=[(0, args.cap)] * len(top),
               constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
w = pd.Series(np.clip(res.x, 0, None), index=top)
w /= w.sum()

price = close.loc[T, top]
shares = np.floor(args.capital * w / price).astype(int)
out = pd.DataFrame({"weight": w, "amount": args.capital * w, "price": price, "shares": shares,
                    "invested": shares * price, "exp. 20d return": mu[top]}).sort_values("weight", ascending=False)
out = out[out.weight > 1e-4]
port_mu, port_vol = float(w @ mu[top]), float(np.sqrt(w @ cov @ w))

print(f"Model {args.model}, date {T.date()}, IC used {ic_hat:.4f}, capital {args.capital:,.0f}\n")
print(out.to_string(formatters={"weight": "{:.1%}".format, "amount": "{:,.0f}".format, "price": "{:,.2f}".format,
                                "invested": "{:,.0f}".format, "exp. 20d return": "{:.2%}".format}))
print(f"\ninvested {out.invested.sum():,.0f} | cash left {args.capital - out.invested.sum():,.0f}")
print(f"portfolio: expected 20d return {port_mu:.2%}, 20d volatility {port_vol:.2%}, "
      f"expected 20d P&L {args.capital * port_mu:,.0f}")
out.to_csv(Path(__file__).resolve().parent / "results" / "allocation.csv")
