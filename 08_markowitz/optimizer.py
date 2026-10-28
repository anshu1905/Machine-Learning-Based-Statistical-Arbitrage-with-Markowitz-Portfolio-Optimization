"""Markowitz max-Sharpe solver shared by backtest.py and allocate.py."""
import numpy as np
from scipy.optimize import minimize


def max_sharpe(mu, cov, cap, n_long):
    """Max-Sharpe weights. The first n_long assets are longs (sum +1); any others are shorts (sum -1).

    Solves  max  w'mu / sqrt(w'Cov w)  with SLSQP, subject to
      longs  0 <= w_i <= cap, summing to +1   (fully invested in the long leg)
      shorts -cap <= w_i <= 0, summing to -1  (long-short only: as much short as long)
    Returns the weights and whether it had to fall back to minimum variance.
    """
    n = len(mu)
    sign = np.r_[np.ones(n_long), -np.ones(n - n_long)]           # +1 for longs, -1 for shorts
    w0 = sign / np.where(sign > 0, n_long, max(n - n_long, 1))    # start from equal weight on each leg
    if w0 @ mu <= 0:  # no positive expected return to exploit: fall back to minimum variance
        obj = lambda w: w @ cov @ w  # noqa: E731
    else:
        obj = lambda w: -(w @ mu) / np.sqrt(w @ cov @ w)  # noqa: E731  (minimise minus Sharpe)
    bounds = [(0, cap) if s > 0 else (-cap, 0) for s in sign]
    cons = [{"type": "eq", "fun": lambda w: w[:n_long].sum() - 1}]
    if n > n_long:
        cons.append({"type": "eq", "fun": lambda w: w[n_long:].sum() + 1})
    w = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons).x
    # The solver can land a hair outside the bounds: clip each leg to its sign and rescale it
    w = np.where(sign > 0, np.clip(w, 0, None), np.clip(w, None, 0))
    w[:n_long] /= w[:n_long].sum()
    if n > n_long:
        w[n_long:] /= -w[n_long:].sum()
    return w, bool(w0 @ mu <= 0)
