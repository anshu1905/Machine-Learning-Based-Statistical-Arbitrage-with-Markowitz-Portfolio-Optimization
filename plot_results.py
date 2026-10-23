"""Comparison figures for all trained models -> results_figures/.

Reads <model>/results/{metrics.json, predictions.parquet, tuning.csv}.
Run after compare_models.py:  python plot_results.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.plotting import GRID, INK_2, SERIES, plt, save
from common.selection import daily_rank_ic

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_figures"
MODELS = {"01_ridge": "Ridge", "02_elasticnet": "ElasticNet", "03_random_forest": "Random Forest",
          "04_xgboost": "XGBoost", "05_rnn": "RNN", "06_lstm": "LSTM", "07_arima": "ARIMA"}
COLOR = dict(zip(MODELS.values(), SERIES))  # fixed per model, never by rank

metrics = {name: json.loads((ROOT / m / "results/metrics.json").read_text())
           for m, name in MODELS.items() if (ROOT / m / "results/metrics.json").exists()}
names = list(metrics)


def bars(ax, values, title, fmt="{:.3f}", ref=None):
    y = np.arange(len(values))
    ax.barh(y, values.values, color=[COLOR[n] for n in values.index], height=0.6)
    ax.set_yticks(y, values.index)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, lw=0.8)
    if ref is not None:
        ax.axvline(ref, color=INK_2, lw=1, ls="--")
    for i, v in enumerate(values.values):
        ax.text(v, i, " " + fmt.format(v), va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK_2)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)


# 1. Validation vs test IC and t-stat (overfitting gap)
m = pd.DataFrame({n: {"val IC": metrics[n]["val"]["ic_mean"], "test IC": metrics[n]["test"]["ic_mean"],
                      "test t": metrics[n]["test"]["ic_tstat_nw"],
                      "test Q5-Q1 ann.": metrics[n]["test"]["q5_minus_q1_ann_gross"],
                      "Q5-Q1 t": metrics[n]["test"]["q5_minus_q1_tstat_nw"],
                      "ICIR": metrics[n]["test"]["icir_ann"],
                      "IC>0 days": metrics[n]["test"]["ic_hit_rate"]} for n in names}).T
m = m.sort_values("test IC", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
y = np.arange(len(m))
for ax, col, title in [(axes[0], "val IC", "Validation IC (used for tuning)"),
                       (axes[1], "test IC", "Test IC (out-of-sample, honest)")]:
    bars(ax, m[col], title)
axes[1].set_xlim(axes[0].get_xlim())
fig.suptitle("Mean daily rank IC: validation vs test, same scale", x=0.01, ha="left", fontweight="bold")
save(fig, OUT / "01_val_vs_test_ic.png")

# 2. Test significance: IC t-stat and Q5-Q1 t-stat, 1.96 line
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
bars(axes[0], m["test t"], "Test IC Newey-West t-stat", "{:.2f}", ref=1.96)
bars(axes[1], m["Q5-Q1 t"], "Test Q5-Q1 spread Newey-West t-stat", "{:.2f}", ref=1.96)
fig.suptitle("Significance on test (dashed = 1.96, 5% two-sided)", x=0.01, ha="left", fontweight="bold")
save(fig, OUT / "02_test_tstats.png")

# 3. Economic size: annualised gross Q5-Q1, ICIR, hit rate
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
bars(axes[0], m["test Q5-Q1 ann."] * 100, "Test Q5-Q1, annualised gross (%)", "{:.1f}")
bars(axes[1], m["ICIR"], "Test ICIR (annualised)", "{:.2f}")
bars(axes[2], m["IC>0 days"] * 100, "Test days with IC > 0 (%)", "{:.1f}", ref=50)
axes[2].set_xlim(40, 62)
save(fig, OUT / "03_test_economics.png")

# 4. Cumulative daily IC over validation + test, all models on one axis
fig, ax = plt.subplots(figsize=(12, 4.5))
for mdir, name in MODELS.items():
    p = ROOT / mdir / "results/predictions.parquet"
    if name not in metrics or not p.exists():
        continue
    d = pd.read_parquet(p, columns=["Date", "split", "y_raw", "pred"]).dropna(subset=["y_raw", "pred"])
    ic = daily_rank_ic(d, ["pred"], target="y_raw")["pred"].dropna().sort_index()
    label = name + (" (sparse rebalance dates)" if name == "ARIMA" else "")
    ax.plot(ic.index, ic.cumsum(), color=COLOR[name], lw=1.6, label=label)
test_start = pd.Timestamp("2024-07-01")
ax.axvline(test_start, color=INK_2, lw=1, ls="--")
ax.text(test_start, ax.get_ylim()[1], "  test starts", va="top", fontsize=8, color=INK_2)
ax.set_title("Cumulative daily rank IC (validation then walk-forward test)", loc="left")
ax.set_ylabel("sum of daily IC")
ax.legend(ncol=7, loc="upper left", bbox_to_anchor=(0, -0.08))
save(fig, OUT / "04_cumulative_ic.png")

# 5. IC by year (val years + test years)
rows = {}
for n in names:
    yrs = {}
    for part in ("val", "test"):
        for yr, v in metrics[n][part].get("ic_by_year", {}).items():
            yrs[f"{yr} {part}"] = v
    rows[n] = yrs
by_year = pd.DataFrame(rows).T.loc[m.index]
by_year = by_year[sorted(by_year.columns, key=lambda c: (c.split()[0], c.split()[1] == "test"))]
fig, ax = plt.subplots(figsize=(12, 4))
w = 0.8 / len(by_year)
x = np.arange(len(by_year.columns))
for i, n in enumerate(by_year.index):
    ax.bar(x + (i - len(by_year) / 2 + 0.5) * w, by_year.loc[n].values, w * 0.9, color=COLOR[n], label=n)
ax.set_xticks(x, by_year.columns)
ax.axhline(0, color=INK_2, lw=0.8)
ax.set_title("Mean daily IC by year and split", loc="left")
ax.legend(ncol=7, loc="upper left", bbox_to_anchor=(0, -0.08))
ax.grid(axis="x", visible=False)
save(fig, OUT / "05_ic_by_year.png")

# 6. Tuning: spread of validation IC across all configs tried (the "loss landscape" of the search)
fig, ax = plt.subplots(figsize=(12, 4))
order = list(m.index)
for i, n in enumerate(order):
    mdir = next(k for k, v in MODELS.items() if v == n)
    t = pd.read_csv(ROOT / mdir / "results/tuning.csv")
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(t))
    ax.scatter(np.full(len(t), i) + jit, t.val_ic, s=30, color=COLOR[n], edgecolor="white", linewidth=1, zorder=3)
    ax.scatter([i], [metrics[n]["test"]["ic_mean"]], marker="D", s=60, color=INK_2, zorder=4,
               label="test IC of chosen config" if i == 0 else None)
ax.set_xticks(range(len(order)), order)
ax.set_ylabel("validation IC")
ax.set_title("Hyper-parameter search: validation IC of every config (dots) vs test IC of the winner", loc="left")
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, OUT / "06_tuning_spread.png")

print(m.round(4).to_string())
print(f"figures -> {OUT}")
