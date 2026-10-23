"""Out-of-sample evaluation shared by every model.

All metrics compare predictions with y_raw (the unwinsorized 20d excess return)
and are computed per date, because the strategy ranks stocks within a date.
"""
import json

import numpy as np
import pandas as pd

from . import config as C
from .data import cs_gauss_rank
from .plotting import BLUE, INK_2, ORANGE, plt, save
from .selection import daily_rank_ic, newey_west_tstat

PRED_COLS = ["Date", "Symbol", "split", "tradable", "y_raw", "pred", "source"]


def quintile_spread(df, q=5):
    """Per date: mean y_raw of the top 1/q of predictions minus the bottom 1/q."""
    sub = df[["Date", "pred", "y_raw"]].dropna()
    r = sub.groupby("Date")["pred"].rank(pct=True)
    top = sub[r > 1 - 1 / q].groupby("Date").y_raw.mean()
    bot = sub[r <= 1 / q].groupby("Date").y_raw.mean()
    return (top - bot).dropna()


def evaluate(df, nw_lags=C.HORIZON):
    """Metrics for one block of predictions (rows without a target are ignored).

    nw_lags: Newey-West lags in *observations*. Daily predictions of a 20-day
    target overlap for 20 observations; predictions every k days overlap for 20/k.
    """
    d = df[df.y_raw.notna() & df.pred.notna()]
    if d.empty:
        return {}
    ic = daily_rank_ic(d, ["pred"], target="y_raw")["pred"].dropna()
    sp = quintile_spread(d)
    return {
        "n_dates": int(ic.size),
        "n_rows": int(len(d)),
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_tstat_nw": float(newey_west_tstat(ic, nw_lags)),
        "icir_ann": float(ic.mean() / ic.std() * np.sqrt(252 / C.HORIZON)),
        "ic_hit_rate": float((ic > 0).mean()),
        "q5_minus_q1_20d": float(sp.mean()),
        "q5_minus_q1_tstat_nw": float(newey_west_tstat(sp, nw_lags)),
        "q5_minus_q1_ann_gross": float(sp.mean() * 252 / C.HORIZON),
        "ic_by_year": {int(k): round(float(v), 4) for k, v in ic.groupby(ic.index.year).mean().items()},
    }


def plot_cumulative_ic(preds, path, title):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for part, color in [("val", BLUE), ("test", ORANGE)]:
        d = preds[(preds.split == part) & preds.y_raw.notna() & preds.pred.notna()]
        if d.empty:
            continue
        ic = daily_rank_ic(d, ["pred"], target="y_raw")["pred"].dropna()
        ax.plot(ic.index, ic.cumsum(), color=color, label=f"{part} (mean IC {ic.mean():.3f})")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(title=title, ylabel="cumulative daily rank IC")
    ax.legend(loc="upper left")
    save(fig, path)


def finish(folder, name, preds, tuning=None, best=None, extra=None, nw_lags=C.HORIZON):
    """Write results/: predictions, metrics, tuning table, best config, IC plot."""
    out = folder / "results"
    out.mkdir(exist_ok=True)
    preds = preds[PRED_COLS].sort_values(["Date", "Symbol"]).reset_index(drop=True)
    # per-date Gaussian rank of the prediction: the z_i used for mu_i = IC * sigma_i * z_i
    preds["pred_z"] = cs_gauss_rank(preds, ["pred"])["pred"]
    preds.to_parquet(out / "predictions.parquet", index=False)

    metrics = {"model": name,
               "val": evaluate(preds[preds.split == "val"], nw_lags),
               "test": evaluate(preds[preds.split == "test"], nw_lags),
               "best_params": best or {}, **(extra or {})}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    if tuning is not None:
        tuning.to_csv(out / "tuning.csv", index=False)
    plot_cumulative_ic(preds, out / "cumulative_ic.png", f"{name}: out-of-sample rank IC")

    v, t = metrics["val"], metrics["test"]
    print(f"\n{name}  val IC {v.get('ic_mean', np.nan):.4f} (t {v.get('ic_tstat_nw', np.nan):.2f})  "
          f"test IC {t.get('ic_mean', np.nan):.4f} (t {t.get('ic_tstat_nw', np.nan):.2f})  "
          f"test Q5-Q1 {t.get('q5_minus_q1_20d', np.nan):.4f}/20d")
    print(f"results -> {out}")
    return metrics
